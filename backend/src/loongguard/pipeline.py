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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from config import AppConfig, load_config, validate_config
from loongguard.alarm.gpio_trigger import GPIOAlarmTrigger
from loongguard.alerts.dedup import AlertDeduplicator
from loongguard.alerts.notifier import AlertNotifier
from loongguard.alerts.persistence import ObjectPersistenceTracker
from loongguard.api.metrics import (
    ALERT_COUNT,
    ALERT_DEDUP_SUPPRESSED,
    DB_OPERATIONS,
    DETECTION_COUNT,
    FRAME_COUNT,
    PIPELINE_FPS,
)
from loongguard.api.server import AlertAPIServer
from loongguard.camera.frame_buffer import FrameBuffer
from loongguard.camera.v4l2_capture import Frame, V4L2Capture
from loongguard.core.config_watcher import DynamicConfig
from loongguard.core.event_bus import Event, EventPriority, get_event_bus
from loongguard.crypto.sm4_logger import SM4Logger
from loongguard.db.database import AlertDatabase
from loongguard.detection.yolo26_nano import YOLO26Nano
from loongguard.face import SleepPostureClassifier, create_face_detector
from loongguard.media import create_audio_backend, create_stream_publisher
from loongguard.motion.frame_diff import FrameDiffDetector
from loongguard.pose.movenet import MoveNetLightning
from loongguard.roi.roi_scheduler import ROIScheduler
from loongguard.utils.schema import AlertLog, AlertType, BoundingBox
from loongguard.voice import VoiceService

logger = logging.getLogger(__name__)


class _FrameRecorder:
    """
    摄像头帧录制器（语音"开始录制"指令的后端实现）

    Pipeline 主循环每帧调用 write()，RGB 帧转 BGR 后写入 mp4。
    单次录制时长受 max_sec 限制，防止无人值守时写满磁盘。
    """

    def __init__(self, path: str, width: int, height: int, fps: float,
                 max_sec: float = 60.0) -> None:
        import cv2  # opencv 为硬依赖，此处延迟导入避免顶层变更

        self._path = path
        self._max_sec = max_sec
        self._start = time.monotonic()
        self._writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        self.frame_count = 0
        logger.info("Recording started: %s (max %.0fs)", path, max_sec)

    @property
    def expired(self) -> bool:
        """是否超过最大录制时长（由主循环检查并自动停止）"""
        return (time.monotonic() - self._start) >= self._max_sec

    def write(self, frame: Frame) -> None:
        import cv2

        if frame.data.size == 0:
            return
        self._writer.write(cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def stop(self) -> str | None:
        """结束录制并释放文件句柄，返回录制文件路径"""
        if self._writer is None:
            return None
        path = self._path
        count = self.frame_count
        self._writer.release()
        self._writer = None
        logger.info("Recording stopped: %s (%d frames)", path, count)
        return path


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
        self._last_inference_time: float = 0.0  # 上一轮推理耗时（用于自适应节流）
        self._inference_time_budget: float = 1.0 / max(config.camera.fps, 1)  # 单帧推理时间预算

        # ── 事件总线（用于解耦告警处理逻辑）─────────────────────
        self._event_bus = get_event_bus()

        # ── 配置热更新（支持运行时动态调整参数）─────────────────
        # 设计动机：无需重启即可调整检测阈值、告警策略等参数
        self._dynamic_config: DynamicConfig | None = None

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
        # 睡姿分类器：基于人脸状态 + 头部姿态做多分类，
        # 并对"人脸不可见"做连续帧持续性过滤（防单帧漏检误报）
        self._posture_classifier = SleepPostureClassifier(config.face)
        # 音频后端（语音唤醒/ASR/通话预留）
        self._audio = create_audio_backend(config.media.audio_backend)
        # 流媒体发布器（MJPEG 桩 / 板端 WebRTC 预留），start() 中注入 video_hub
        self._stream_publisher = None

        # ── 新增：持久化 / 去重 / 保留策略 ────────────────────
        self._db = AlertDatabase(config.database.db_path)
        self._dedup = AlertDeduplicator(config.dedup)
        # 物体持续出现跟踪器：检测结果必须持续出现超过指定时间才允许报警
        # 设计动机：单帧误检闪烁不触发告警，避免瞬时误报
        # 可通过环境变量 LG_PERSISTENCE_PERSISTENCE_SEC 调整持续阈值
        self._persistence_tracker = ObjectPersistenceTracker(
            persistence_sec=config.persistence.persistence_sec,
            max_missed_sec=config.persistence.max_missed_sec,
            position_grid_size=config.persistence.position_grid_size,
        )
        self._retention_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # ── 新增：外部告警通知（webhook，无人值守防漏报）──────
        self._notifier = AlertNotifier(config.notify)

        # ── 新增：HLS 直播发布器（家长端远程观看）────────────
        # ffmpeg 缺失/密钥缺失时 HLSPublisher 内部 enabled=False 并安静停用，
        # 不阻塞主链路；对应端点返回 404，MJPEG 本地观看不受影响。
        self._hls = None

        # ── 新增：数据库定期备份任务 ─────────────────────────
        self._backup_task: asyncio.Task | None = None

        # ── 新增：语音服务（openWakeWord 唤醒 + PocketSphinx 指令）──
        # 缺依赖/模型时 VoiceService 内部安静降级，不阻塞主链路
        self._current_mode = "normal"  # normal / nap / public
        self._recorder: _FrameRecorder | None = None
        self._voice = VoiceService(config.voice, callbacks={
            "set_mode": self.set_mode,
            "ack_all": self.ack_all_alerts,
            "get_sensor_data": self.get_sensor_data,
            "start_record": self.start_recording,
            "get_recent_logs": self.get_recent_logs,
            "shutdown": self._voice_shutdown,
        })

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
        # 人脸检测（ONNX 模型加载失败时回退到 DummyFaceDetector 桩）
        # 设计动机：人脸检测是非核心功能，失败不应阻塞主链路
        if hasattr(self._face, "load_model"):
            self._face.load_model()
        # 注入模型健康快照提供者（绑定当前实例方法，避免循环依赖）
        self._api.set_pose_health_provider(self._pose.get_health_snapshot)
        self._api.set_detection_health_provider(self._detector.get_health_snapshot)
        self._api.set_face_health_provider(
            lambda: {
                "available": self._face.is_available(),
                "posture": self._posture_classifier.get_snapshot(),
            }
        )
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

        # ── HLS 直播发布器（家长端远程观看）──────────────────
        # 必须在 API 服务启动之后拉起：loopback 输入源是本服务 MJPEG 端点
        if self._config.media.hls_enabled:
            from loongguard.media.hls_publisher import HLSPublisher

            self._hls = HLSPublisher(
                self._config.media,
                output_dir=Path(__file__).resolve().parents[2] / "data" / "hls",
                stream_base_url=f"http://127.0.0.1:{self._config.api.port}/stream",
            )
            self._hls.start()
        self._api.set_hls_publisher(self._hls)
        if self._hls is not None and self._hls.enabled:
            logger.info("HLS live stream ready: /stream/live.m3u8 (token-protected)")

        # ── 新增：外部告警通知器启动 ─────────────────────────
        await self._notifier.start()

        # ── 新增：语音服务启动（唤醒词/模型缺失时安静降级）────
        await self._voice.start()

        # ── 新增：配置热更新启动 ─────────────────────────────
        # 设计动机：支持运行时动态调整检测参数，无需重启
        self._setup_dynamic_config()

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

        # 语音服务先停（含 TTS 播报收尾），再释放音视频资源
        await self._voice.stop()
        self.stop_recording()

        # 停止配置热更新
        if self._dynamic_config is not None:
            self._dynamic_config.stop()
            self._dynamic_config = None

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
        # HLS 编码进程先停，再释放 API 服务
        if self._hls is not None:
            self._hls.stop()
            self._hls = None
        # 流媒体发布器（WebRTC 信令端口等）随 API 服务一并释放
        if self._stream_publisher is not None:
            try:
                self._stream_publisher.stop()
            except Exception:
                logger.exception("Stream publisher stop failed")
            self._stream_publisher = None
        await self._api.stop()
        logger.info("Pipeline stopped")

    async def __aenter__(self) -> Pipeline:
        """async context manager 入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """async context manager 退出，确保资源释放"""
        await self.stop()

    # ── 语音指令回调（供 VoiceService 注入，均为同步可调用）──

    def set_mode(self, mode: str) -> None:
        """
        切换检测模式（语音"切换到XX模式"指令）

        模式语义与桌面端模式切换一致：当前为运行状态标记 + 日志，
        检测行为差异化调度为后续迭代项（前端模式切换亦未改变检测参数）。
        """
        if mode not in ("normal", "nap", "public"):
            logger.warning("未知检测模式: %s", mode)
            return
        self._current_mode = mode
        logger.info("Detection mode changed to: %s", mode)

    @property
    def current_mode(self) -> str:
        """当前检测模式（normal / nap / public）"""
        return self._current_mode

    def ack_all_alerts(self) -> int:
        """批量确认所有未确认告警（语音"关闭告警"指令）"""
        acked = 0
        if self._db is not None:
            try:
                result = self._db.query_alerts(page=1, page_size=10000, acknowledged=False)
                for item in result.get("data", []):
                    if self._db.ack_alert(item["alert_id"]):
                        acked += 1
                logger.info("Voice ack-all: %d alerts acknowledged", acked)
            except Exception:
                logger.exception("Voice ack-all failed")
        return acked

    def get_sensor_data(self) -> tuple[float, float] | None:
        """
        读取 RS485 温湿度（语音"查询温湿度"指令）

        复用 frontend/ui/mgt_rs485.py 的 Modbus 读取实现（按文件路径
        延迟加载，避免 backend 依赖 PyQt 前端包）。读取失败或依赖
        缺失返回 None，由语音模块播报"传感器异常"。
        """
        try:
            reader = self._load_sensor_reader()
        except Exception:
            logger.warning("RS485 传感器读取模块不可用（minimalmodbus/pyserial 未安装？）")
            return None
        if reader is None:
            return None
        try:
            temp, hum = reader()
            if temp is None or hum is None:
                return None
            return temp, hum
        except Exception:
            logger.exception("RS485 传感器读取失败")
            return None

    def _load_sensor_reader(self):
        """按文件路径加载 frontend/ui/mgt_rs485.py 的 read_temp_hum（带缓存）"""
        cached = getattr(self, "_sensor_reader", None)
        if cached is not None:
            return cached
        import importlib.util

        module_path = (Path(__file__).resolve().parents[3]
                       / "frontend" / "ui" / "mgt_rs485.py")
        if not module_path.exists():
            logger.warning("传感器模块不存在: %s", module_path)
            return None
        spec = importlib.util.spec_from_file_location("lg_mgt_rs485", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._sensor_reader = module.read_temp_hum
        return self._sensor_reader

    def start_recording(self) -> str:
        """开始视频录制（语音"开始录制"指令），返回 TTS 提示文本"""
        if self._recorder is not None:
            return "录制已在进行中"
        record_dir = Path(__file__).resolve().parents[2] / "data" / "recordings"
        record_dir.mkdir(parents=True, exist_ok=True)
        path = str(record_dir / f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        try:
            self._recorder = _FrameRecorder(
                path,
                width=self._config.camera.width,
                height=self._config.camera.height,
                fps=self._config.camera.fps,
                max_sec=60.0,
            )
        except Exception:
            logger.exception("录制器初始化失败")
            self._recorder = None
            return "录制开启失败"
        return "视频录制已开启"

    def stop_recording(self) -> None:
        """停止视频录制（幂等）"""
        if self._recorder is None:
            return
        recorder, self._recorder = self._recorder, None
        try:
            recorder.stop()
        except Exception:
            logger.exception("录制器关闭异常")

    def get_recent_logs(self) -> str:
        """最近告警摘要（语音"查看日志"指令），返回 TTS 播报文本"""
        if self._db is None:
            return "日志服务未启动"
        try:
            result = self._db.query_alerts(page=1, page_size=5)
            total = result.get("total", 0)
            data = result.get("data", [])
            if not data:
                return f"最近没有告警记录，历史累计{total}条"
            latest = data[0]
            desc = latest.get("description") or latest.get("alert_type", "")
            return f"最近共{total}条告警，最新一条：{desc}"
        except Exception:
            logger.exception("查询最近日志失败")
            return "日志读取失败"

    def _voice_shutdown(self) -> None:
        """
        语音"退出程序"指令：置停运行标志。

        主循环 while self._running 随即退出，进入 main() finally 的
        统一 pipeline.stop() 清理路径（与 Ctrl+C 行为一致）。
        """
        logger.info("Voice shutdown requested")
        self._running = False

    def _setup_dynamic_config(self) -> None:
        """
        设置配置热更新

        设计动机：支持运行时动态调整检测参数，无需重启。
        监控配置文件变化，自动重新加载并应用新配置。
        """
        try:
            # 获取配置文件路径
            config_path = Path(__file__).parent.parent.parent / "config" / "default.json"
            if not config_path.exists():
                logger.warning("Config file not found: %s", config_path)
                return

            # 创建动态配置管理器
            self._dynamic_config = DynamicConfig(config_path)

            # 注册配置变更回调
            self._dynamic_config.on_change(self._on_config_change)

            # 启动配置监控
            self._dynamic_config.start()
            logger.info("Dynamic config started, watching: %s", config_path)
        except Exception as e:
            logger.warning("Failed to setup dynamic config: %s", e)

    def _on_config_change(self, new_config: dict) -> None:
        """
        配置变更回调

        Args:
            new_config: 新的配置字典
        """
        logger.info("Config changed, applying new settings...")

        try:
            # 更新检测阈值
            if "detection" in new_config:
                det_config = new_config["detection"]
                if "conf_threshold" in det_config:
                    self._config.detection.conf_threshold = det_config["conf_threshold"]
                    logger.info("Updated detection conf_threshold: %s", det_config["conf_threshold"])

            # 更新姿态检测参数
            if "pose" in new_config:
                pose_config = new_config["pose"]
                if "prone_angle_threshold" in pose_config:
                    self._config.pose.prone_angle_threshold = pose_config["prone_angle_threshold"]
                    logger.info("Updated pose prone_angle_threshold: %s", pose_config["prone_angle_threshold"])
                if "prone_frame_threshold" in pose_config:
                    self._config.pose.prone_frame_threshold = pose_config["prone_frame_threshold"]
                    logger.info("Updated pose prone_frame_threshold: %s", pose_config["prone_frame_threshold"])

            # 更新告警去重配置
            if "dedup" in new_config:
                dedup_config = new_config["dedup"]
                if "time_window_sec" in dedup_config:
                    self._config.dedup.time_window_sec = dedup_config["time_window_sec"]
                    logger.info("Updated dedup time_window_sec: %s", dedup_config["time_window_sec"])

            logger.info("Config changes applied successfully")
        except Exception as e:
            logger.error("Failed to apply config changes: %s", e)

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
                # 语音"开始录制"触发的录制钩子（原始帧，不受推理异常影响）
                if self._recorder is not None:
                    self._recorder.write(frame)
                    if self._recorder.expired:
                        self.stop_recording()

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
            - 首帧（无上一帧）时执行全帧推理，确保首次检测即可产出结果
            - ROI 推理与姿态估计并行执行，缩短单帧处理耗时
            - 自适应推理跳帧：推理耗时超过 2 倍帧间隔时，中间帧跳过推理

        关键约束（修复 #1）：无论处理过程中是否发生异常，都必须在 finally
        块中调用 push_frame，确保前端 MJPEG 视频流不中断。
        """
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

            # 首帧兜底：无上一帧时帧差分无法检测运动，但应执行一次全帧推理
            # 设计动机：确保首帧即可产出检测结果（特别是静态场景中已有目标物体），
            # 而非等待第二帧产生运动差分后才开始检测。
            is_first_frame = prev_gray is None

            if motion_regions or is_first_frame:
                # 自适应跳帧：大面积运动（如场景切换）时跳过推理
                # 设计动机：当运动覆盖超过 80% 画面时，通常是场景切换或摄像头抖动，
                # 此时 YOLO 推理效果差且耗时。跳过推理帧，仅在低运动帧上执行检测。
                if motion_regions:
                    max_ratio = max(r.motion_ratio for r in motion_regions)
                    if max_ratio > 0.80:
                        self._frame_skip_counter += 1
                        if self._frame_skip_counter <= 2:
                            # 跳过推理，仅推送原始帧
                            self._last_det_boxes = []
                            return
                    else:
                        self._frame_skip_counter = 0

                # 自适应推理节流：当上一轮推理耗时超过 2 倍帧间隔时，
                # 跳过本次推理，仅推送上一帧检测结果，让视频流保持流畅
                if (
                    self._last_inference_time > 0
                    and self._frame_count % 2 == 0
                    and self._last_inference_time > self._inference_time_budget * 2
                ):
                    det_boxes = self._last_det_boxes
                    return

                # ROI 推理与姿态估计并行执行
                # 设计动机：YOLO ROI 推理和 MoveNet 姿态估计是两个独立的
                # CPU 密集型操作，在多核环境下并行执行可显著缩短单帧处理耗时。
                # 两个推理结果互不依赖（dangerous_object vs prone_sleep），
                # 合并到 alerts 后统一走后续过滤和告警链路。
                inference_start = time.monotonic()

                run_pose = (
                    self._pose.is_available()
                    and self._pose.should_run(self._frame_count)
                )

                if run_pose:
                    # 并行：ROI 检测 + 姿态估计同时执行
                    roi_task = asyncio.to_thread(
                        self._roi_scheduler.process, frame.data, motion_regions
                    )
                    pose_task = asyncio.to_thread(self._pose.detect_prone, frame.data)
                    try:
                        roi_alerts, pose_alerts = await asyncio.gather(
                            roi_task, pose_task, return_exceptions=True
                        )
                    except Exception:
                        roi_alerts, pose_alerts = [], []

                    # 处理 ROI 结果（可能是异常对象）
                    if isinstance(roi_alerts, Exception):
                        logger.exception("ROI inference failed")
                        roi_alerts = []
                    alerts.extend(roi_alerts)

                    # 处理姿态结果
                    if isinstance(pose_alerts, Exception):
                        logger.exception("Pose detection raised unexpectedly")
                    elif isinstance(pose_alerts, list):
                        alerts.extend(pose_alerts)
                    self._pose.mark_ran(self._frame_count)
                else:
                    # 仅 ROI 推理（姿态估计未就绪或未到节流间隔）
                    alerts = await asyncio.to_thread(
                        self._roi_scheduler.process, frame.data, motion_regions
                    )

                self._last_inference_time = time.monotonic() - inference_start
                for alert in alerts:
                    det_boxes.extend(alert.detections)

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
                    sleep_alerts = await self._check_sleep_posture(frame.data)
                    alerts.extend(sleep_alerts)

                # ── 物体持续出现过滤 ──────────────────────────────────
                # 设计动机：只有置信度 > 0.8（已由 conf_threshold 保证）
                # 且目标物体持续出现超过 1 秒的检测结果才允许触发告警，
                # 避免单帧误检闪烁导致的瞬时误报。
                persistent_alerts: list[AlertLog] = []
                if alerts:
                    now = time.monotonic()
                    # 收集所有危险物品告警中的检测框，统一更新跟踪器
                    all_dets: list[BoundingBox] = []
                    for alert in alerts:
                        if alert.alert_type == AlertType.DANGEROUS_OBJECT:
                            all_dets.extend(alert.detections)
                    # 更新跟踪状态，获取已持续的 key 集合
                    persistent_keys = self._persistence_tracker.update(
                        all_dets, now
                    )
                    # 过滤告警
                    for alert in alerts:
                        if alert.alert_type == AlertType.DANGEROUS_OBJECT:
                            # 检查该告警的检测结果是否已持续
                            should_alert = any(
                                self._persistence_tracker.make_key(det)
                                in persistent_keys
                                for det in alert.detections
                            )
                            if should_alert:
                                persistent_alerts.append(alert)
                        else:
                            # 非危险物品告警（如俯卧睡姿）直接通过
                            persistent_alerts.append(alert)
                    alerts = persistent_alerts

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

        处理链路：去重检查 -> 日志 -> 事件发布 -> 声光告警 -> SM4加密 -> 数据库持久化 -> 指标 -> API推送

        设计动机：
            去重检查在最前面，避免重复告警触发后续所有开销。
            数据库写入失败不阻塞主循环（降级为内存模式）。
            alarm.trigger 使用 asyncio.create_task 避免阻塞帧处理。
            通过事件总线发布告警事件，支持灵活的扩展和订阅。
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

        # 发布告警事件到事件总线（支持异步处理）
        # 设计动机：通过事件总线解耦告警处理逻辑，其他模块可以订阅和处理告警事件
        # frame 可能为 None（如数据库失败降级路径直接构造告警），需空值保护
        alert_event = Event(
            name="alert",
            data={
                "alert": alert.to_dict(),
                "frame_id": frame.frame_id if frame is not None else None,
                "timestamp": frame.timestamp if frame is not None else None,
            },
            priority=EventPriority.HIGH if alert.severity.value in ("high", "critical") else EventPriority.NORMAL,
            source="pipeline",
        )
        await self._event_bus.emit_async(alert_event)

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

    async def _check_sleep_posture(self, rgb: np.ndarray) -> list[AlertLog]:
        """
        人脸状态 + 头部姿态组合逻辑：多分类睡姿判定

        算法：
            1. 对当前帧执行人脸检测（含 5 点关键点）
            2. 分类器对每张人脸做睡姿分类（FACE_UP/SIDE/TILTED）
            3. 无人脸时累计"不可见"帧，连续 N 帧触发 PRONE_SLEEP

        预留设计（跨平台）：
            - 此方法仅依赖 FaceDetector 接口与分类器，与具体实现解耦
            - DummyFaceDetector 恒返回人脸（无关键点），分类为 FACE_UP，
              故默认不触发告警

        Args:
            rgb: RGB 图像 (H, W, 3)

        Returns:
            告警列表；人脸可见时为空列表
        """
        # 人脸 ONNX 推理为阻塞调用，卸载到工作线程
        faces = await asyncio.to_thread(self._face.detect_with_landmarks, rgb)
        return self._posture_classifier.evaluate(faces)

    async def _retention_loop(self) -> None:
        """
        周期性视频文件清理

        设计动机：
            上传的视频文件（data/uploads/）会持续占用磁盘空间。
            每小时检查一次，删除超龄文件，不删除正在分析的文件。
        """
        from loongguard.utils.retention import VideoRetentionManager

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
                # 使用 UTC 时间，与 _db_cleanup_loop 保持一致
                await asyncio.to_thread(self._backup_db_once, datetime.now(UTC))
            except Exception:
                logger.exception("Database backup task failed")

    def _backup_db_once(self, now: datetime) -> bool:
        """
        执行一次备份并清理超期备份（在线程池中运行）

        Args:
            now: 当前 UTC 时间戳

        注意：使用 UTC 时间进行所有时间比较，与 _db_cleanup_loop 保持一致。
        """
        cfg = self._config.database
        # 统一为 UTC aware 时间：生产调用传 datetime.now(UTC)，测试可能传 naive
        # 的 datetime.now()，两者与 mtime_utc 比较需一致时区，否则 TypeError
        if now.tzinfo is None:
            now = now.astimezone(UTC)
        backup_dir = Path(cfg.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        ts = now.strftime("%Y%m%d_%H%M%S")
        dest = backup_dir / f"loongguard_{ts}.db"
        ok = self._db.backup(str(dest))

        # 清理超期备份（保留 cfg.backup_retention_days 天）
        # 使用 UTC 时间进行比较，与 _db_cleanup_loop 保持一致
        try:
            cutoff = now - timedelta(days=cfg.backup_retention_days)
            for old in backup_dir.glob("loongguard_*.db"):
                try:
                    # 将文件修改时间转换为 UTC 进行比较
                    mtime_utc = datetime.fromtimestamp(old.stat().st_mtime, tz=UTC)
                    if mtime_utc < cutoff:
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


async def main(config_path: str | None = None) -> None:
    """
    应用入口

    Args:
        config_path: 配置文件路径，None 使用默认配置
    """
    # 加载 .env 文件到环境变量（在 load_config 之前执行，确保环境变量可被读取）
    from config.settings import _load_dotenv
    _load_dotenv()

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

    from loongguard.utils.logging_config import setup_logging

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
            raise SystemExit(1) from e
        raise

    logger.info("API: http://localhost:%d", config.api.port)

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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.json"
    asyncio.run(main(config_path))
