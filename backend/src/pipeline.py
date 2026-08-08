"""
LoongGuard 核心管道编排器

设计动机：
    串联"采集 -> 运动检测 -> ROI 调度 -> 姿态估计 -> 告警触发 -> 加密存储"
    的完整推理链路，是系统的主循环入口。
    所有模块通过此编排器协调，不相互直接调用，保持模块间低耦合。
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

from config import AppConfig, load_config
from src.camera.v4l2_capture import V4L2Capture, Frame
from src.camera.frame_buffer import FrameBuffer
from src.detection.yolo26_nano import YOLO26Nano
from src.motion.frame_diff import FrameDiffDetector
from src.roi.roi_scheduler import ROIScheduler
from src.pose.movenet import MoveNetLightning
from src.alarm.gpio_trigger import GPIOAlarmTrigger
from src.crypto.sm4_logger import SM4Logger
from src.api.server import AlertAPIServer
from src.api.metrics import (
    FRAME_COUNT, DETECTION_COUNT, ALERT_COUNT, ALERT_DEDUP_SUPPRESSED,
    INFERENCE_LATENCY, PIPELINE_FPS, DB_OPERATIONS, MetricsTimer,
)
from src.db.database import AlertDatabase
from src.alerts.dedup import AlertDeduplicator
from src.face import create_face_detector
from src.media import create_audio_backend, create_stream_publisher
from src.utils.schema import AlertLog, AlertSeverity, AlertType

logger = logging.getLogger(__name__)


class Pipeline:
    """
    LoongGuard 主推理管道

    生命周期：
        pipeline = Pipeline(config)
        await pipeline.start()    # 启动所有子模块
        await pipeline.run()      # 主循环（阻塞直到收到停止信号）
        await pipeline.stop()     # 清理所有资源

    或使用 async context manager：
        async with Pipeline(config) as pipeline:
            await pipeline.run()
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._running = False
        self._stopping = False  # 防止并发 stop 调用
        self._last_det_boxes: list = []  # 最新检测框（供 VideoHub 使用）
        self._last_motion_regions: list = []  # 最新运动区域
        self._show_debug_window = os.environ.get("LG_DEBUG_WINDOW", "").lower() in ("1", "true", "yes")
        self._frame_skip_counter: int = 0  # 自适应跳帧计数器
        self._frame_count: int = 0  # 总帧计数（用于周期性任务）
        self._fps_window: list[float] = []  # FPS 滑动窗口

        # ── 推理与显示解耦（独立推理线程）───────────────────────
        # 主循环（采集线程）只负责把帧推送到 VideoHub，保证显示帧率；
        # 独立推理线程负责"运动检测 + 周期全帧扫描（静止检测）+ ROI + 姿态"，
        # 通过 video_hub.set_overlay 更新叠加数据，二者在不同线程互不阻塞。
        # 设计动机（静止检测）：运动触发的 ROI 检测永远捕捉不到静止危险品
        # （如放在桌面的剪刀），因此推理线程还需按 full_scan_interval_sec
        # 周期性对整帧执行一次 YOLO 检测，作为"静止检测"兜底。
        self._latest_frame_lock = threading.Lock()
        self._latest_frame: Optional[Frame] = None  # 最新帧槽（采集线程写入，推理线程读取）
        self._infer_thread: Optional[threading.Thread] = None
        self._alert_queue: Optional[asyncio.Queue] = None  # 推理线程 -> 事件循环的告警队列
        self._alert_consumer_task: Optional[asyncio.Task] = None

        # ── 子模块实例化 ──────────────────────────────────────
        self._camera = V4L2Capture(config.camera)
        self._frame_buffer = FrameBuffer(
            max_debug_frames=10 if config.debug else 0
        )
        self._detector = YOLO26Nano(config.detection)
        self._motion = FrameDiffDetector(config.motion)
        self._roi_scheduler = ROIScheduler(config.roi, config.detection, self._detector)
        self._pose = MoveNetLightning(config.pose)
        self._alarm = GPIOAlarmTrigger(config.alarm)
        self._crypto = SM4Logger(config.crypto)
        self._api = AlertAPIServer(config.api)

        # ── 跨平台预留接口实例化 ────────────────────────────────
        # 人脸检测（Person+Face 组合）：无模型时用 DummyFaceDetector 桩
        self._face = create_face_detector(config.face)
        # 音频后端（语音唤醒/ASR/通话预留）
        self._audio = create_audio_backend(config.media.audio_backend)
        # 流媒体发布器（MJPEG 桩 / 板端 WebRTC 预留），start() 中注入 video_hub
        self._stream_publisher = None

        # ── 新增：持久化 / 去重 / 保留策略 ────────────────────
        self._db = AlertDatabase(config.database.db_path)
        self._dedup = AlertDeduplicator(config.dedup)
        self._retention_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """
        启动所有子模块

        启动顺序：数据库 -> 加密 -> 摄像头 -> 检测模型 -> 告警 -> API
        （数据库先行确保告警持久化通道可用）
        """
        logger.info("Starting LoongGuard pipeline...")

        # 初始化数据库（失败不阻塞启动，降级为纯内存模式）
        try:
            self._db.initialize()
            self._api.set_database(self._db)
            logger.info("Database initialized: %s", self._config.database.db_path)
        except Exception:
            logger.exception("Database init failed, running without persistence")

        self._crypto.load_key()
        self._camera.open()
        self._detector.load_model()
        # pose 加载失败时仅降级自身，不阻塞主流程
        self._pose.load_model()
        # 注入模型健康快照提供者（绑定当前实例方法，避免循环依赖）
        self._api.set_pose_health_provider(self._pose.get_health_snapshot)
        self._api.set_detection_health_provider(self._detector.get_health_snapshot)
        self._alarm.setup()
        await self._api.start()

        # 注入检测模块到视频分析器（用于上传视频分析功能）
        self._api.video_analyzer.set_modules(
            detector=self._detector,
            motion=self._motion,
            roi_scheduler=self._roi_scheduler,
        )

        # 创建流媒体发布器（MJPEG 桩 / 板端 WebRTC 预留）
        self._stream_publisher = create_stream_publisher(
            kind=self._config.media.stream_publisher,
            video_hub=self._api.video_hub,
            signal_port=self._config.media.webrtc_signal_port,
        )
        logger.info(
            "Stream publisher ready: %s", self._config.media.stream_publisher
        )

        # 启动周期性清理任务
        self._retention_task = asyncio.create_task(self._retention_loop())
        self._cleanup_task = asyncio.create_task(self._db_cleanup_loop())

        # 启动独立推理线程 + 告警消费者（推理与显示解耦）
        # 设计动机：推理线程在后台以自身节奏处理最新帧，采集线程可达到
        # 摄像头原生帧率推送显示，互不阻塞。告警通过队列跨线程投递到
        # 事件循环侧消费（API 推送等异步操作仍在事件循环执行）。
        self._alert_queue = asyncio.Queue()
        self._alert_consumer_task = asyncio.create_task(self._consume_alerts())
        self._infer_thread = threading.Thread(
            target=self._infer_loop,
            name="loongguard-inference",
            daemon=True,
        )
        self._infer_thread.start()
        logger.info("Inference thread started")

        self._running = True
        logger.info("Pipeline started")

    async def stop(self) -> None:
        """停止所有子模块，释放资源（幂等：多次调用安全）"""
        if self._stopping:
            return
        self._stopping = True
        logger.info("Stopping pipeline...")
        self._running = False

        # 取消周期性任务
        for task in (self._retention_task, self._cleanup_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 停止独立推理线程（先置 running=False 使循环退出，再 join 等待）
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=5)
            self._infer_thread = None
        # 取消告警消费者
        if self._alert_consumer_task is not None:
            self._alert_consumer_task.cancel()
            try:
                await self._alert_consumer_task
            except asyncio.CancelledError:
                pass
            self._alert_consumer_task = None

        self._camera.close()
        self._frame_buffer.clear()
        self._alarm.cleanup()
        self._db.close()
        await self._api.stop()
        logger.info("Pipeline stopped")

    async def __aenter__(self) -> Pipeline:
        """async context manager 入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """async context manager 退出，确保资源释放"""
        await self.stop()

    async def run(self) -> None:
        """
        采集主循环（与推理解耦）

        单帧流程：
            1. 采集一帧
            2. 将最新帧写入共享帧槽（推理线程异步消费）
            3. 推送该帧到 VideoHub（单参数，显示帧率不被推理阻塞）
            4. 帧率控制，避免忙轮询

        设计动机（修复接口不一致 + 帧率解耦）：
            旧版在此循环内同步执行推理，显示帧率受推理耗时限制，
            且 push_frame 使用已被废弃的多参数签名导致接口报错。
            新版采集线程只负责推帧，推理在独立线程（_infer_loop）执行，
            通过 video_hub.set_overlay 更新叠加数据，显示可达摄像头原生帧率。
        """
        logger.info("Entering main capture loop")
        frame_interval = 1.0 / max(self._config.camera.fps, 1)

        while self._running:
            loop_start = time.monotonic()

            frame = self._camera.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # 将最新帧写入共享槽（加锁保护并释放旧帧）
            # 推理线程在锁内 .copy() 取走数据副本，二者互斥，无竞态。
            with self._latest_frame_lock:
                old = self._latest_frame
                self._latest_frame = frame
                if old is not None:
                    old.release()

            # 推送帧到视频流（单参数，仅负责显示，不阻塞推理）
            if self._api.video_hub is not None:
                self._api.video_hub.push_frame(frame.data)

            # 帧率控制：保证不超频，同时 yield 控制权给事件循环
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def _infer_loop(self) -> None:
        """
        独立推理线程主循环（推理与显示解耦）

        每个循环从最新帧槽取一帧，执行：
            1. 帧差分运动检测 -> 运动区域
            2. 周期全帧扫描（静止检测）：按 full_scan_interval_sec 对整帧
               执行一次 YOLO 检测，捕捉静止危险品（放在桌面的剪刀等）
            3. 有运动 -> Dynamic ROI 二级检测 + 姿态估计
            4. 通过 video_hub.set_overlay 更新叠加数据（显示线程读取）
            5. 告警放入队列，由事件循环侧 _consume_alerts 处理

        设计动机：
            - 推理线程以自身节奏处理"最新帧"，采集线程不受限，显示帧率
              可达摄像头原生帧率；推理耗时只影响检测结果的刷新频率。
            - 周期全帧扫描是"静止检测"的核心：运动触发无法捕捉静止物品。
        """
        import numpy as np

        prev_gray: Optional[np.ndarray] = None
        last_full_scan: float = 0.0
        full_scan_interval = max(
            self._config.detection.full_scan_interval_sec, 0.25
        )

        while self._running:
            try:
                # 从共享槽取最新帧数据副本（加锁，避免与采集线程竞态）
                with self._latest_frame_lock:
                    cur = self._latest_frame
                    data = cur.data.copy() if cur is not None else None
                if data is None:
                    time.sleep(0.01)
                    continue

                now = time.monotonic()

                # 帧差运动检测（prev_gray 由本线程自维护）
                current_gray = np.dot(
                    data[..., :3], [0.299, 0.587, 0.114]
                ).astype(np.uint8)
                motion_regions = self._motion.detect(current_gray, prev_gray)
                prev_gray = current_gray
                self._last_motion_regions = motion_regions

                # 指标：帧计数
                FRAME_COUNT.inc()
                self._frame_count += 1

                det_boxes: list = []
                alerts: list[AlertLog] = []

                # 周期全帧扫描（静止检测）
                # 设计动机：运动触发的 ROI 检测无法捕捉静止危险品。
                # 每间隔 full_scan_interval 秒对整帧执行一次 YOLO 检测，
                # 作为"静止检测"兜底，无论是否有运动都会执行。
                if now - last_full_scan >= full_scan_interval:
                    last_full_scan = now
                    try:
                        full_boxes = self._detector.infer(data)
                        det_boxes.extend(full_boxes)
                    except Exception:
                        logger.exception("Full-frame scan inference failed")

                # 运动触发 Dynamic ROI + 姿态
                if motion_regions:
                    # 自适应跳帧：大面积运动（场景切换/抖动）时跳过 ROI 推理
                    max_ratio = max(r.motion_ratio for r in motion_regions)
                    if max_ratio <= 0.80:
                        try:
                            roi_alerts = self._roi_scheduler.process(
                                data, motion_regions
                            )
                        except Exception:
                            logger.exception("ROI scheduling failed")
                            roi_alerts = []
                        for alert in roi_alerts:
                            alerts.append(alert)
                            det_boxes.extend(alert.detections)

                        # 姿态估计：有运动主体且到达推理节流间隔时执行
                        # 设计动机：俯卧检测与危险物品检测是两个独立安全功能，
                        # 不应相互依赖。有运动即可能有人活动，触发俯卧检测。
                        if (
                            self._pose.is_available()
                            and self._pose.should_run(self._frame_count)
                        ):
                            try:
                                pose_alerts = self._pose.detect_prone(data)
                                alerts.extend(pose_alerts)
                            except Exception:
                                logger.exception("Pose detection raised unexpectedly")
                            finally:
                                self._pose.mark_ran(self._frame_count)

                        # 睡姿（人脸可见性）监测：仅当显式启用时
                        if (
                            self._config.face.enabled
                            and self._face.is_available()
                        ):
                            sleep_alerts = self._check_sleep_posture(data)
                            alerts.extend(sleep_alerts)

                self._last_det_boxes = det_boxes

                # 更新视频叠加数据（显示线程 push_frame 读取，线程安全）
                if self._api.video_hub is not None:
                    self._api.video_hub.set_overlay(
                        boxes=det_boxes,
                        motion_regions=motion_regions,
                    )

                # 指标：检测结果分类计数
                for box in det_boxes:
                    DETECTION_COUNT.labels(class_name=box.class_name).inc()

                # 指标：FPS 滑动窗口（每 30 帧更新一次）
                self._fps_window.append(now)
                self._fps_window = [
                    t for t in self._fps_window if now - t < 1.0
                ]
                if self._frame_count % 30 == 0:
                    PIPELINE_FPS.set(len(self._fps_window))

                # 周期性去重清理（每 100 帧）
                if self._frame_count % 100 == 0:
                    self._dedup.cleanup_expired()

                # 告警投递到事件循环侧队列（API 推送等异步工作由消费者处理）
                for alert in alerts:
                    if self._alert_queue is not None:
                        self._alert_queue.put_nowait(alert)

            except Exception:
                logger.exception("Inference thread loop error")
                time.sleep(0.01)

    async def _consume_alerts(self) -> None:
        """
        消费推理线程投递的告警（事件循环侧）

        设计动机：API WebSocket 推送、加密存储、数据库写入等均为异步操作，
        必须在事件循环上下文中执行。推理线程只负责把告警放入队列，
        本协程在此消费并调用 _handle_alert 完成完整处理链路。
        """
        while True:
            alert = await self._alert_queue.get()
            try:
                await self._handle_alert(alert, None)
            except Exception:
                logger.exception("Error handling alert from queue")

    async def _handle_alert(self, alert: AlertLog, frame: Frame) -> None:
        """
        处理单条告警

        处理链路：去重检查 -> 日志 -> 声光告警 -> SM4加密 -> 数据库持久化 -> 指标 -> API推送

        设计动机：
            去重检查在最前面，避免重复告警触发后续所有开销。
            数据库写入失败不阻塞主循环（降级为内存模式）。
            alarm.trigger 使用 asyncio.create_task 避免阻塞帧处理。
        """
        # 去重检查：抑制重复告警
        if not self._dedup.should_alert(alert):
            ALERT_DEDUP_SUPPRESSED.inc()
            return

        logger.warning(
            "ALERT [%s] severity=%s: %s",
            alert.alert_type.value,
            alert.severity.value,
            alert.description,
        )

        # 声光告警（后台任务，不阻塞主循环的帧处理）
        asyncio.create_task(self._alarm.trigger(alert.severity))

        # SM4 加密存储
        self._crypto.encrypt_and_store(alert)

        # SQLite 持久化（失败不阻塞）
        try:
            self._db.insert_alert(alert)
            DB_OPERATIONS.labels(operation="insert").inc()
        except Exception:
            logger.exception("Failed to persist alert %s to database", alert.alert_id)

        # 指标：告警分类计数
        ALERT_COUNT.labels(
            severity=alert.severity.value,
            alert_type=alert.alert_type.value,
        ).inc()

        # API 实时推送
        await self._api.push_alert(alert)

    def _check_sleep_posture(self, rgb: np.ndarray) -> list[AlertLog]:
        """
        Person + Face 组合逻辑：画面有运动主体但未检测到人脸 -> 异常睡姿

        算法：
            1. 对当前帧执行人脸检测
            2. 检测到人脸 -> 安全（脸可见，非趴睡）
            3. 未检测到人脸 -> 判定异常睡姿（俯卧/遮挡/趴睡）

        预留设计（跨平台）：
            - 此方法仅依赖 FaceDetector 接口，与具体实现解耦
            - DummyFaceDetector 恒返回人脸，故默认不触发告警
            - 板端接入真实人脸模型后，可在 detect 结果上进一步限定
              "Person bbox 内"的人脸，无需改动本方法调用方

        Args:
            rgb: RGB 图像 (H, W, 3)

        Returns:
            告警列表；人脸可见时为空列表
        """
        faces = self._face.detect(rgb)
        if faces:
            # 检测到人脸 -> 安全，不告警
            return []

        # 人脸不可见 -> 异常睡姿（严重等级最高，窒息风险）
        logger.warning(
            "PRONE_SLEEP trigger: 画面存在运动主体但未检测到人脸"
        )
        return [
            AlertLog(
                alert_type=AlertType.PRONE_SLEEP,
                severity=AlertSeverity.CRITICAL,
                description="人脸不可见，儿童可能存在趴睡/遮挡风险",
            )
        ]

    async def _retention_loop(self) -> None:
        """
        周期性视频文件清理

        设计动机：
            上传的视频文件（data/uploads/）会持续占用磁盘空间。
            每小时检查一次，删除超龄文件，不删除正在分析的文件。
        """
        from src.utils.retention import VideoRetentionManager

        interval = self._config.retention.cleanup_interval_sec
        manager = VideoRetentionManager(self._config.retention)

        while self._running:
            await asyncio.sleep(interval)
            try:
                active = {
                    t.file_path for t in self._api.video_analyzer.tasks.values()
                    if t.status.value == "processing"
                }
                result = await asyncio.to_thread(manager.cleanup, active_files=active)
                if result.get("deleted_count", 0) > 0:
                    logger.info(
                        "Retention cleanup: deleted %d files, freed %.1f MB",
                        result["deleted_count"],
                        result.get("freed_bytes", 0) / 1024 / 1024,
                    )
            except Exception:
                logger.exception("Retention cleanup failed")

    async def _db_cleanup_loop(self) -> None:
        """
        周期性数据库清理

        每 6 小时清理超出保留期的告警记录。
        """
        interval = 6 * 3600  # 6 小时
        while self._running:
            await asyncio.sleep(interval)
            try:
                deleted = await asyncio.to_thread(
                    self._db.cleanup_old_alerts,
                    self._config.database.retention_days,
                )
                if deleted > 0:
                    logger.info("DB cleanup: removed %d expired alerts", deleted)
                    DB_OPERATIONS.labels(operation="cleanup").inc()
            except Exception:
                logger.exception("Database cleanup failed")

    def _draw_debug_window(self, frame: Frame) -> None:
        """
        在本地 OpenCV 窗口显示摄像头画面及检测结果标注

        仅通过环境变量 LG_DEBUG_WINDOW=1 开启，
        不影响生产部署行为。
        """
        try:
            import cv2

            display = cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)

            # 绘制运动区域（蓝色半透明矩形）
            for r in self._last_motion_regions:
                overlay = display.copy()
                cv2.rectangle(
                    overlay,
                    (r.x, r.y), (r.x + r.w, r.y + r.h),
                    (255, 128, 0), 2,
                )
                cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

            # 绘制检测框
            for box in self._last_det_boxes:
                color = (0, 0, 255)  # 红色：危险物品
                cv2.rectangle(
                    display,
                    (box.x1, box.y1), (box.x2, box.y2),
                    color, 2,
                )
                label = f"{box.class_name} {box.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
                )
                cv2.rectangle(
                    display,
                    (box.x1, box.y1 - th - 6), (box.x1 + tw, box.y1),
                    color, -1,
                )
                cv2.putText(
                    display, label,
                    (box.x1, box.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

            # HUD 信息
            h, w = display.shape[:2]
            hud_lines = [
                f"Frame:{frame.frame_id}",
                f"Motion:{len(self._last_motion_regions)}",
                f"Det:{len(self._last_det_boxes)}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(
                    display, line, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                )

            cv2.imshow("LoongGuard Debug", display)
            # 1ms 非阻塞刷新，允许窗口事件处理
            cv2.waitKey(1)
        except Exception:
            logger.debug("Debug window render failed", exc_info=True)


async def main(config_path: Optional[str] = None) -> None:
    """
    应用入口

    Args:
        config_path: 配置文件路径，None 使用默认配置
    """
    config = load_config(config_path)

    # 向后兼容：旧 JSON 配置只有 log_level 而无 log 段时，用 log_level 回填
    if config.log.level == "INFO" and config.log_level != "INFO":
        config.log.level = config.log_level

    from src.utils.logging_config import setup_logging

    setup_logging(
        level=config.log.level,
        fmt=config.log.format,
        log_dir=config.log.log_dir,
        max_bytes=config.log.max_bytes,
        backup_count=config.log.backup_count,
        console=config.log.console,
    )

    # 端口占用自动重试：最多尝试 10 个连续端口
    original_port = config.api.port
    pipeline = None
    for port_offset in range(10):
        config.api.port = original_port + port_offset
        pipeline = Pipeline(config)
        try:
            await pipeline.start()
            break
        except OSError as e:
            if "10048" in str(e) or "address already in use" in str(e).lower():
                logger.warning(
                    "Port %d occupied, trying %d...",
                    config.api.port, config.api.port + 1,
                )
                await pipeline.stop()
                pipeline = None
                continue
            raise

    if pipeline is None:
        logger.error(
            "Cannot start: ports %d-%d all occupied. "
            "Kill other processes or set LG_API_PORT to a free port.",
            original_port, original_port + 9,
        )
        return

    logger.info("Dashboard: http://localhost:%d", config.api.port)

    # 信号处理：Windows 仅支持 SIGINT（Ctrl+C），不支持 SIGTERM
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(pipeline.stop()))

    try:
        await pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.json"
    asyncio.run(main(config_path))
