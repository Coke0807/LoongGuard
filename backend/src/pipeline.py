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
import time
from datetime import datetime, timedelta
from typing import Optional

from config import AppConfig, load_config, validate_config
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
from src.alerts.notifier import AlertNotifier
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

        # ── 新增：外部告警通知（webhook，无人值守防漏报）──────
        self._notifier = AlertNotifier(config.notify)

        # ── 新增：数据库定期备份任务 ─────────────────────────
        self._backup_task: Optional[asyncio.Task] = None

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

        # ── 新增：外部告警通知器启动 ─────────────────────────
        await self._notifier.start()

        # 启动周期性清理任务
        self._retention_task = asyncio.create_task(self._retention_loop())
        self._cleanup_task = asyncio.create_task(self._db_cleanup_loop())
        self._backup_task = asyncio.create_task(self._db_backup_loop())

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
        for task in (self._retention_task, self._cleanup_task, self._backup_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._camera.close()
        self._frame_buffer.clear()
        self._alarm.cleanup()
        self._db.close()
        await self._notifier.stop()
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
        主推理循环

        单帧处理流程：
            1. 采集一帧
            2. 帧差分运动检测 -> 运动区域
            3. 有运动 -> Dynamic ROI 调度（二级检测）+ 姿态估计
            4. 无运动 -> 仅推送原始帧到视频流（跳过推理节省算力）
            5. 告警触发 -> 声光 + 加密日志 + API 推送

        性能优化：
            - 帧率控制：空闲时 sleep 避免忙轮询浪费 CPU
            - 灰度缓存：当前帧灰度仅计算一次，避免 motion + buffer 重复计算
            - 无运动时跳过推理：节省 GPU/CPU 算力
        """
        logger.info("Entering main inference loop")
        frame_interval = 1.0 / max(self._config.camera.fps, 1)

        while self._running:
            frame = self._camera.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            loop_start = time.monotonic()

            try:
                await self._process_frame(frame)
            except Exception:
                logger.exception("Error processing frame %d", frame.frame_id)
            finally:
                self._frame_buffer.push(frame)

            # 帧率控制：保证不超频，同时 yield 控制权给事件循环
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _process_frame(self, frame: Frame) -> None:
        """
        处理单帧

        性能优化策略：
            - 灰度转换仅计算一次（当前帧），避免帧缓冲模块重复计算
            - 帧推送到 VideoHub 仅在此处执行一次，移除 run() 中的重复调用
            - 无运动时仅推送原始帧，跳过所有推理
            - 姿态估计仅在有目标检测结果时执行（避免无目标时的无效推理）

        关键约束（修复 #1）：无论处理过程中是否发生异常，都必须在 finally
        块中调用 push_frame，确保前端 MJPEG 视频流不中断。
        """
        import numpy as np

        # 计算当前帧灰度（仅一次，供运动检测使用）
        current_gray = np.dot(
            frame.data[..., :3], [0.299, 0.587, 0.114]
        ).astype(np.uint8)
        prev_gray = self._frame_buffer.get_prev_gray()

        # 帧差分运动检测
        motion_regions: list = []
        alerts: list[AlertLog] = []
        det_boxes: list = []

        try:
            motion_regions = self._motion.detect(current_gray, prev_gray)
            self._last_motion_regions = motion_regions

            # 指标：帧计数
            FRAME_COUNT.inc()
            self._frame_count += 1

            if motion_regions:
                # 自适应跳帧：大面积运动（如场景切换）时跳过推理
                # 设计动机：当运动覆盖超过 80% 画面时，通常是场景切换或摄像头抖动，
                # 此时 YOLO 推理效果差且耗时。跳过推理帧，仅在低运动帧上执行检测。
                max_ratio = max(r.motion_ratio for r in motion_regions)
                if max_ratio > 0.80:
                    self._frame_skip_counter += 1
                    if self._frame_skip_counter <= 2:
                        # 跳过推理，仅推送原始帧
                        self._last_det_boxes = []
                        return
                else:
                    self._frame_skip_counter = 0

                # Dynamic ROI 调度（含二级检测）
                alerts = self._roi_scheduler.process(
                    frame.data, motion_regions
                )
                for alert in alerts:
                    det_boxes.extend(alert.detections)

                # 姿态估计：基于"运动区域 + 帧节流"独立触发
                # 设计动机（修复 #bug-decouple-pose）：
                #   旧实现 if det_boxes: 将俯卧检测耦合到危险物品检测结果，
                #   导致"画面中只有小孩趴睡、无危险物品"时俯卧检测完全失效。
                #   新策略：只要画面有运动（间接说明有人活动）且到达推理节流
                #   间隔，就执行俯卧检测。危险物品检测和俯卧检测是两个独立
                #   的安全功能，不应相互依赖。
                # 降级策略：pose 模型未加载成功或单次推理失败时静默跳过。
                if (
                    motion_regions
                    and self._pose.is_available()
                    and self._pose.should_run(self._frame_count)
                ):
                    try:
                        pose_alerts = self._pose.detect_prone(frame.data)
                        alerts.extend(pose_alerts)
                    except Exception:
                        # 防御性兜底：pose 模块已自带 try/except，此处仅
                        # 防止未来重构时未捕获的异常影响主链路
                        logger.exception("Pose detection raised unexpectedly")
                    finally:
                        self._pose.mark_ran(self._frame_count)

                # 睡姿（人脸可见性）监测：仅当显式启用时执行
                # 设计动机（Person+Face 组合，跨平台预留）：
                #   异常睡姿 = Person bbox 内无 Face。板端接入真实人脸
                #   模型后，可将 faces 限定为 Person bbox 内再做判定。
                #   当前 DummyFaceDetector 恒返回人脸，故默认不触发告警。
                if (
                    self._config.face.enabled
                    and self._face.is_available()
                    and motion_regions
                ):
                    sleep_alerts = self._check_sleep_posture(frame.data)
                    alerts.extend(sleep_alerts)

                # 告警处理
                for alert in alerts:
                    await self._handle_alert(alert, frame)

            self._last_det_boxes = det_boxes

            # 指标：检测结果分类计数
            for box in det_boxes:
                DETECTION_COUNT.labels(class_name=box.class_name).inc()

            # 指标：FPS 滑动窗口（每 30 帧更新一次 Gauge）
            now = time.monotonic()
            self._fps_window.append(now)
            # 保留最近 1 秒的帧时间戳
            self._fps_window = [t for t in self._fps_window if now - t < 1.0]
            if self._frame_count % 30 == 0:
                PIPELINE_FPS.set(len(self._fps_window))

            # 周期性去重清理（每 100 帧）
            if self._frame_count % 100 == 0:
                self._dedup.cleanup_expired()

        except Exception:
            logger.exception("Frame processing error (frame %d), pushing raw frame", frame.frame_id)
        finally:
            # 统一推送帧到 VideoHub（始终执行，确保视频流不中断）
            # 设计动机（修复 #1）：即使帧处理失败，也要推送原始帧到前端，
            # 否则 MJPEG 流会卡住，前端显示为无响应。
            if self._api.video_hub is not None:
                self._api.video_hub.push_frame(
                    frame.data,
                    boxes=det_boxes,
                    motion_regions=motion_regions,
                )

        # 本地调试窗口
        if self._show_debug_window:
            self._draw_debug_window(frame)

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

        # 外部通知（webhook，含重试；后台任务，不阻塞帧处理）
        asyncio.create_task(self._notifier.notify(alert))

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

    async def _db_backup_loop(self) -> None:
        """
        周期性数据库热备份

        设计动机：
            SQLite 单机无并发保护与备份策略，设备故障/误删会导致告警记录
            丢失。按配置间隔生成一致性快照，并清理超期备份。
        """
        interval = self._config.database.backup_interval_hours * 3600
        while self._running:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(self._backup_db_once, datetime.now())
            except Exception:
                logger.exception("Database backup task failed")

    def _backup_db_once(self, now) -> bool:
        """执行一次备份并清理超期备份（在线程池中运行）"""
        cfg = self._config.database
        backup_dir = Path(cfg.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        ts = now.strftime("%Y%m%d_%H%M%S")
        dest = backup_dir / f"loongguard_{ts}.db"
        ok = self._db.backup(str(dest))

        # 清理超期备份（保留 cfg.backup_retention_days 天）
        try:
            cutoff = now - timedelta(days=cfg.backup_retention_days)
            for old in backup_dir.glob("loongguard_*.db"):
                try:
                    if datetime.fromtimestamp(old.stat().st_mtime) < cutoff:
                        old.unlink()
                        logger.info("清理过期备份: %s", old.name)
                except OSError:
                    pass
        except Exception:
            logger.exception("Backup retention cleanup failed")

        return ok

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

    # 配置校验：启动早期一次性暴露配置错误，阻断启动而非带病运行
    config_errors = validate_config(config)
    if config_errors:
        for err in config_errors:
            logger.error("配置校验失败: %s", err)
        raise SystemExit(
            "配置校验失败，已阻止启动。请修正上述配置项后重试。"
        )

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

    # 端口占用即失败退出，交由进程守护（systemd/start.ps1）重启。
    # 设计动机（修复隐患）：旧实现端口被占时自动 +1 递增，会导致端口漂移，
    # 与写死 8081 的前端失联；且漂移后无人值守会静默运行在错误端口。
    # 现在改为"失败即退出 + 守护进程重启"，保证端口始终与配置一致。
    pipeline = Pipeline(config)
    try:
        await pipeline.start()
    except OSError as e:
        if "10048" in str(e) or "address already in use" in str(e).lower():
            logger.error(
                "端口 %d 已被占用，启动失败退出。请先释放端口或修改 "
                "LG_API_PORT，然后由守护进程自动重启。",
                config.api.port,
            )
            raise SystemExit(1)
        raise

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
