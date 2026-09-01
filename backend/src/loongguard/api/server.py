"""
后端 API 模块

设计动机：
    为局域网 Web 管理端提供 REST + WebSocket + MJPEG 视频流接口。
    管理端可通过浏览器实时查看摄像头画面、检测框、告警信息。
    所有接口仅在局域网内暴露，绝不外连互联网。

    扩展功能：
    - 视频上传（POST /api/v1/upload）：开发测试环境下通过上传视频文件
      验证 AI 模型检测效果，替代摄像头实时采集。
    - 视频分析（POST /api/v1/analyze/{task_id}）：触发上传视频的帧级检测，
      通过 SSE 实时推送分析进度和检测结果。
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from aiohttp import web

from config import APIConfig
from loongguard.api.metrics import RATE_LIMITED
from loongguard.db.database import AlertDatabase
from loongguard.utils.schema import AlertLog, BoundingBox

# 类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from loongguard.api.parent_routes import ParentAPIRoutes
    from loongguard.media.hls_publisher import HLSPublisher

logger = logging.getLogger(__name__)

# 临时上传目录（前端正式 UI 通过 API 交互，后端不再内嵌页面）
_UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"

# 上传限制：500MB
_MAX_UPLOAD_SIZE = 500 * 1024 * 1024
# 支持的视频格式
_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

# SSE 队列大小限制（每个客户端独立队列）
_SSE_QUEUE_MAXSIZE = 256

# JPEG 编码质量
_JPEG_QUALITY = 60
_WEB_JPEG_QUALITY = 50


# ── 模块接口协议（结构化鸭子类型约束）─────────────────────────
# 设计动机：VideoAnalyzer / AlertAPIServer 通过注入方式持有检测、运动、
# ROI 等子模块引用，旧实现使用 Any 类型，调用方传错对象只能运行期暴露。
# 此处用 Protocol 形式化各模块的最小接口，静态检查器（mypy/pyright）
# 可在注入点校验类型，同时不引入运行时继承耦合。


@runtime_checkable
class DetectorLike(Protocol):
    """目标检测器最小接口（YOLO26Nano 满足）"""

    def infer(self, image: np.ndarray) -> list[BoundingBox]: ...


@runtime_checkable
class MotionDetectorLike(Protocol):
    """运动检测器最小接口（FrameDiffDetector 满足）"""

    def detect(
        self, current_gray: np.ndarray, prev_gray: np.ndarray | None
    ) -> list: ...


@runtime_checkable
class ROISchedulerLike(Protocol):
    """ROI 调度器最小接口（ROIScheduler 满足）"""

    def process(
        self, frame: np.ndarray, motion_regions: list
    ) -> list[AlertLog]: ...


@runtime_checkable
class HealthProviderLike(Protocol):
    """模型健康快照提供者（无参可调用，返回 dict）"""

    def __call__(self) -> dict: ...


class VideoHub:
    """
    视频帧缓冲 + 标注渲染中心

    职责：
        - 缓存最新标注帧（含检测框、运动区域、关键点）
        - 提供 MJPEG 编码输出
        - 维护实时统计（FPS、检测数、帧计数）

    Pipeline 每处理完一帧，调用 push_frame() 推送数据。
    浏览器通过 /stream 端点拉取 MJPEG 视频流。
    """

    def __init__(self, max_fps: int = 25) -> None:
        self._max_fps = max_fps
        self._latest_jpeg: bytes | None = None
        self._latest_web_jpeg: bytes | None = None
        self._lock = threading.Lock()

        # 实时统计
        self._frame_count: int = 0
        self._alert_count: int = 0
        self._motion_count: int = 0
        self._start_time: float = time.time()
        self._frame_times: deque[float] = deque(maxlen=60)

    def push_frame(
        self,
        frame_rgb: np.ndarray,
        boxes: list[BoundingBox] | None = None,
        motion_regions: list | None = None,
        keypoints: np.ndarray | None = None,
    ) -> None:
        """
        推送一帧到视频流缓冲

        Pipeline 主循环每处理完一帧调用一次。
        在帧上绘制检测框、运动区域后编码为 JPEG。

        Args:
            frame_rgb: RGB 格式帧数据 (H, W, 3)
            boxes: 检测框列表
            motion_regions: 运动区域列表
            keypoints: MoveNet 关键点 (17, 3)
        """
        try:
            import cv2

            display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 绘制运动区域
            if motion_regions:
                overlay = display.copy()
                for r in motion_regions:
                    cv2.rectangle(
                        overlay,
                        (r.x, r.y), (r.x + r.w, r.y + r.h),
                        (255, 128, 0), 2,
                    )
                cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
                self._motion_count += len(motion_regions)

            # 绘制检测框
            if boxes:
                for box in boxes:
                    color = _BOX_COLORS.get(box.class_name, (255, 255, 255))
                    cv2.rectangle(
                        display,
                        (box.x1, box.y1), (box.x2, box.y2),
                        color, 2,
                    )
                    label = f"{box.class_name} {box.confidence:.0%}"
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
                self._alert_count += len(boxes)
                # 红色告警边框
                h, w = display.shape[:2]
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

            # 绘制关键点
            if keypoints is not None:
                _draw_keypoints(display, keypoints)

            # HUD 信息
            self._frame_count += 1
            now = time.time()
            self._frame_times.append(now)
            fps = len(self._frame_times) / max(
                now - (self._frame_times[0] if self._frame_times else now), 0.001,
            )
            hud = (
                f"FPS:{fps:.0f} "
                f"Frame:{self._frame_count} "
                f"Det:{len(boxes) if boxes else 0} "
                f"TotalAlerts:{self._alert_count}"
            )
            cv2.putText(
                display, hud, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
            )

            # JPEG 编码（单次编码，quality=60 兼顾画质与带宽）
            # 设计动机：原实现每帧编码两次（quality 75 + 50），在嵌入式平台上
            # 编码耗时 ~20-40ms/帧，是帧率瓶颈之一。改为单次编码后本地和 Web
            # 端共用同一帧数据，省掉 ~50% 编码开销。
            ret, buf = cv2.imencode(
                ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
            )
            if ret:
                jpeg_bytes = buf.tobytes()
                with self._lock:
                    self._latest_jpeg = jpeg_bytes
                    self._latest_web_jpeg = jpeg_bytes
            else:
                # JPEG 编码失败降级：生成纯色带文字的信息帧
                # 设计动机（修复 #2）：在 LoongArch 等平台若 OpenCV 缺少 JPEG 编码器，
                # 不能静默丢弃帧导致前端流无响应，必须生成兜底帧。
                logger.warning("JPEG encoding failed, generating fallback frame")
                fallback = self._make_fallback_jpeg(display.shape[1], display.shape[0])
                with self._lock:
                    self._latest_jpeg = fallback
                    self._latest_web_jpeg = fallback

        except Exception:
            logger.exception("VideoHub push_frame failed")

    def get_jpeg(self) -> bytes | None:
        """获取最新一帧的 JPEG 字节（线程安全，本地高质量）"""
        with self._lock:
            return self._latest_jpeg

    def get_web_jpeg(self) -> bytes | None:
        """获取最新一帧的 Web 优化 JPEG（低质量、小体积）"""
        with self._lock:
            return self._latest_web_jpeg or self._latest_jpeg

    @staticmethod
    def _make_fallback_jpeg(width: int, height: int) -> bytes:
        """
        生成 JPEG 编码失败的兜底画面

        当 cv2.imencode 无法编码 JPEG 时，使用 numpy 直接构造
        一个带说明文字的纯色图像作为兜底帧，确保前端 MJPEG 流不断。

        Returns:
            JPEG 字节串
        """
        import cv2 as _cv2
        import numpy as _np

        fb = _np.full((max(height, 100), max(width, 200), 3), [30, 30, 60], dtype=_np.uint8)
        _cv2.putText(fb, "JPEG Encoder Unavailable", (20, fb.shape[0] // 2 - 10),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        _cv2.putText(fb, "Check OpenCV imgcodecs build", (20, fb.shape[0] // 2 + 20),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        _ret, _buf = _cv2.imencode(".jpg", fb, [_cv2.IMWRITE_JPEG_QUALITY, _WEB_JPEG_QUALITY])
        if _ret:
            return _buf.tobytes()
        # 极端情况：连纯色帧都无法编码，返回硬编码的最小 JPEG
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa7\x93"
            b"\xff\xd9"
        )

    def get_stats(self) -> dict:
        """获取实时统计"""
        elapsed = time.time() - self._start_time
        return {
            "frame_count": self._frame_count,
            "alert_count": self._alert_count,
            "motion_count": self._motion_count,
            "uptime_sec": round(elapsed, 1),
            "avg_fps": round(self._frame_count / max(elapsed, 0.001), 1),
        }


# 检测框颜色映射
_BOX_COLORS: dict[str, tuple[int, int, int]] = {
    "magnetic_bead":  (0, 0, 255),
    "button_battery": (0, 0, 255),
    "scissors":       (0, 165, 255),
    "utility_knife":  (0, 165, 255),
    "needle":         (0, 255, 255),
    "glass_shard":    (255, 0, 0),
    "wire":           (255, 255, 0),
    "small_toy_part": (255, 0, 255),
}

# MoveNet 骨骼连线
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _draw_keypoints(frame_bgr: np.ndarray, kps: np.ndarray) -> None:
    """在 BGR 帧上绘制 MoveNet 关键点和骨骼连线"""
    import cv2

    h, w = frame_bgr.shape[:2]
    min_score = 0.3

    for i, j in _SKELETON:
        if kps[i, 2] >= min_score and kps[j, 2] >= min_score:
            pt1 = (int(kps[i, 1] * w), int(kps[i, 0] * h))
            pt2 = (int(kps[j, 1] * w), int(kps[j, 0] * h))
            cv2.line(frame_bgr, pt1, pt2, (0, 255, 0), 2)

    for i in range(17):
        if kps[i, 2] >= min_score:
            cx = int(kps[i, 1] * w)
            cy = int(kps[i, 0] * h)
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 255), -1)


# ── 视频分析任务管理 ─────────────────────────────────────────


class AnalysisStatus(str, Enum):
    """分析任务状态"""

    PENDING = "pending"       # 排队等待
    PROCESSING = "processing" # 分析中
    COMPLETED = "completed"   # 完成
    FAILED = "failed"         # 失败


@dataclass
class AnalysisTask:
    """视频分析任务"""

    task_id: str
    filename: str
    file_path: str
    status: AnalysisStatus = AnalysisStatus.PENDING
    # 进度追踪
    total_frames: int = 0
    processed_frames: int = 0
    current_frame_jpeg: bytes | None = None
    # 检测结果
    alerts: list[dict] = field(default_factory=list)
    detections_timeline: list[dict] = field(default_factory=list)
    # 统计
    class_counts: dict = field(default_factory=dict)
    error_message: str = ""
    # 视频元信息（用于 SSE 重连时补发 meta 事件）
    video_meta: dict | None = field(default=None, repr=False)
    # 事件循环引用（create_task 时捕获），供工作线程跨线程唤醒 SSE 等待者
    loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)
    # SSE 事件队列（每个连接的客户端独立队列）
    _sse_queues: list = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def push_event(self, event_type: str, data: dict) -> None:
        """向所有 SSE 客户端推送事件"""
        payload = {"type": event_type, "data": data}
        dead_queues: list = []
        with self._lock:
            for q in self._sse_queues:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead_queues.append(q)
            for q in dead_queues:
                self._sse_queues.remove(q)
        # 跨线程唤醒事件循环：本方法运行于视频分析工作线程，而 asyncio.Queue
        # 并非线程安全——put_nowait 内部的 waiter.set_result 走的是 loop.call_soon，
        # 不会写 self-pipe，事件循环可能阻塞在 selector 上迟迟取不到事件
        #（表现为 SSE 流最长停顿 ~30s）。call_soon_threadsafe 强制写 self-pipe
        # 立即唤醒循环，属于官方指定的线程间调度入口。
        loop = self.loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass  # 事件循环已关闭（服务停止中），无需唤醒

    def register_sse(self) -> asyncio.Queue:
        """注册一个 SSE 客户端队列"""
        q: asyncio.Queue = asyncio.Queue(maxsize=_SSE_QUEUE_MAXSIZE)
        with self._lock:
            self._sse_queues.append(q)
        return q

    def unregister_sse(self, q: asyncio.Queue) -> None:
        """注销 SSE 客户端"""
        with self._lock:
            if q in self._sse_queues:
                self._sse_queues.remove(q)

    def to_status_dict(self) -> dict:
        """序列化为状态查询响应"""
        return {
            "task_id": self.task_id,
            "filename": self.filename,
            "status": self.status.value,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "progress": round(
                self.processed_frames / max(self.total_frames, 1) * 100, 1
            ),
            "alert_count": len(self.alerts),
            "class_counts": self.class_counts,
            "error_message": self.error_message,
        }


class VideoAnalyzer:
    """
    视频文件分析器

    在后台线程中逐帧处理上传的视频文件，复用项目已有的
    运动检测 -> Dynamic ROI -> YOLO26 检测管道，
    通过 SSE 向前端实时推送每帧的分析结果。

    设计动机：
        此模块仅用于开发测试阶段，验证 AI 模型在录制视频上的检测效果。
        生产环境以摄像头实时检测为主，此模块不参与。
    """

    def __init__(self) -> None:
        self._tasks: dict[str, AnalysisTask] = {}
        # 分析器引用（由 Pipeline 初始化后注入，Protocol 约束接口形状）
        self._detector: DetectorLike | None = None
        self._motion: MotionDetectorLike | None = None
        self._roi_scheduler: ROISchedulerLike | None = None

    def set_modules(
        self,
        detector: DetectorLike,
        motion: MotionDetectorLike,
        roi_scheduler: ROISchedulerLike,
    ) -> None:
        """
        注入 Pipeline 的检测子模块引用

        运行期校验 Protocol 形状，注入错误对象时立即报错而非延迟到分析阶段。
        """
        if not isinstance(detector, DetectorLike):
            raise TypeError(f"detector 不满足 DetectorLike 协议: {type(detector)}")
        if not isinstance(motion, MotionDetectorLike):
            raise TypeError(f"motion 不满足 MotionDetectorLike 协议: {type(motion)}")
        if not isinstance(roi_scheduler, ROISchedulerLike):
            raise TypeError(
                f"roi_scheduler 不满足 ROISchedulerLike 协议: {type(roi_scheduler)}"
            )
        self._detector = detector
        self._motion = motion
        self._roi_scheduler = roi_scheduler

    @property
    def tasks(self) -> dict[str, AnalysisTask]:
        return self._tasks

    def get_task(self, task_id: str) -> AnalysisTask | None:
        return self._tasks.get(task_id)

    def create_task(self, filename: str, file_path: str) -> AnalysisTask:
        """创建分析任务"""
        task_id = uuid.uuid4().hex[:12]
        task = AnalysisTask(
            task_id=task_id,
            filename=filename,
            file_path=file_path,
            loop=asyncio.get_running_loop(),
        )
        self._tasks[task_id] = task
        return task

    async def run_analysis(self, task_id: str) -> None:
        """
        执行视频分析（后台协程）

        主协程仅负责打开视频读取元信息；CPU 密集的逐帧分析
        （解码 + 灰度 + 运动检测 + ONNX 推理）通过 asyncio.to_thread
        移入线程池，避免阻塞事件循环（旧实现每帧阻塞事件循环数十
        毫秒，导致 MJPEG 流与 WebSocket 推送卡顿）。
        """
        task = self._tasks.get(task_id)
        if not task:
            return

        if self._detector is None or self._motion is None or self._roi_scheduler is None:
            task.status = AnalysisStatus.FAILED
            task.error_message = "Detection modules not initialized"
            task.push_event("error", {"message": task.error_message})
            return

        try:
            import cv2

            task.status = AnalysisStatus.PROCESSING
            task.push_event("status", {"status": "processing"})

            cap = cv2.VideoCapture(task.file_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {task.file_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            meta_data = {
                "total_frames": total_frames,
                "fps": round(fps, 1),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            cap.release()

            task.total_frames = total_frames
            task.video_meta = meta_data
            task.push_event("meta", meta_data)

            # 逐帧分析在线程池中执行（纯同步逻辑）
            await asyncio.to_thread(self._analyze_frames, task, fps)

        except Exception as e:
            logger.exception("Video analysis failed for task %s", task_id)
            task.status = AnalysisStatus.FAILED
            task.error_message = str(e)
            task.push_event("error", {"message": str(e)})

    def _analyze_frames(self, task: AnalysisTask, fps: float) -> None:
        """
        逐帧视频分析（在线程池中运行，不占用事件循环）

        线程安全说明：
            - push_event 内部由 threading.Lock 保护；
            - asyncio.Queue.put_nowait 是线程安全的（官方文档保证），
              事件循环侧的 get() 会在下一次调度时收到事件。
        """
        import cv2

        motion = self._motion
        roi_scheduler = self._roi_scheduler
        if motion is None or roi_scheduler is None:
            raise RuntimeError("Detection modules not initialized")

        cap = cv2.VideoCapture(task.file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {task.file_path}")

        total_frames = task.total_frames
        frame_idx = 0
        prev_gray: np.ndarray | None = None
        yield_interval = max(1, int(fps / 5))

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_idx += 1

                if frame_idx % yield_interval != 0 and frame_idx != 1:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                current_gray = np.dot(
                    frame_rgb[..., :3], [0.299, 0.587, 0.114]
                ).astype(np.uint8)

                motion_regions = motion.detect(current_gray, prev_gray)
                prev_gray = current_gray

                det_boxes: list[BoundingBox] = []
                frame_alerts: list[dict] = []

                if motion_regions:
                    alerts = roi_scheduler.process(frame_rgb, motion_regions)
                    for alert in alerts:
                        det_boxes.extend(alert.detections)
                        alert_dict = alert.to_dict()
                        alert_dict["frame_index"] = frame_idx
                        alert_dict["video_time"] = round(frame_idx / fps, 2)
                        frame_alerts.append(alert_dict)
                        task.alerts.append(alert_dict)

                for box in det_boxes:
                    name = box.class_name
                    task.class_counts[name] = task.class_counts.get(name, 0) + 1

                if det_boxes:
                    task.detections_timeline.append({
                        "frame": frame_idx,
                        "time": round(frame_idx / fps, 2),
                        "count": len(det_boxes),
                        "classes": [b.class_name for b in det_boxes],
                    })

                annotated = frame_bgr.copy()
                for box in det_boxes:
                    cv2.rectangle(
                        annotated,
                        (box.x1, box.y1), (box.x2, box.y2),
                        (0, 0, 255), 2,
                    )
                    label = f"{box.class_name} {box.confidence:.0%}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
                    )
                    cv2.rectangle(
                        annotated,
                        (box.x1, box.y1 - th - 6), (box.x1 + tw, box.y1),
                        (0, 0, 255), -1,
                    )
                    cv2.putText(
                        annotated, label,
                        (box.x1, box.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    )

                _, jpeg_buf = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
                )
                task.current_frame_jpeg = jpeg_buf.tobytes()

                task.processed_frames = frame_idx

                task.push_event("frame", {
                    "frame_index": frame_idx,
                    "total_frames": total_frames,
                    "progress": round(frame_idx / max(total_frames, 1) * 100, 1),
                    "detections": [
                        {
                            "class_name": b.class_name,
                            "confidence": round(b.confidence, 3),
                            "x1": b.x1, "y1": b.y1,
                            "x2": b.x2, "y2": b.y2,
                        }
                        for b in det_boxes
                    ],
                    "motion_count": len(motion_regions),
                    "video_time": round(frame_idx / fps, 2),
                    "alerts": frame_alerts,
                })
        finally:
            cap.release()

        task.status = AnalysisStatus.COMPLETED
        task.processed_frames = task.total_frames
        task.push_event("complete", {
            "total_alerts": len(task.alerts),
            "total_detections": sum(task.class_counts.values()),
            "class_counts": task.class_counts,
            "timeline": task.detections_timeline,
        })


class _TokenBucketRateLimiter:
    """
    每客户端 IP 的令牌桶限流器

    设计动机：
        局域网部署无 WAF/反向代理，应用层需防暴力请求与端口扫描。
        令牌桶允许短时突发（burst），长期速率受 rate 约束。
        线程安全：中间件在事件循环单线程内调用，但 to_thread 场景
        下也可能并发，统一加锁成本极低。
    """

    _MAX_TRACKED_IPS = 10000  # 跟踪表上限，防止恶意源耗尽内存

    def __init__(self, rate: float, burst: float) -> None:
        self._rate = rate
        self._burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """rate <= 0 视为禁用限流"""
        return self._rate > 0

    def allow(self, key: str) -> bool:
        """消耗 1 个令牌；桶空则拒绝"""
        now = time.monotonic()
        with self._lock:
            if len(self._buckets) > self._MAX_TRACKED_IPS:
                # 惰性清理：丢弃 60 秒未活跃的桶，控制内存
                cutoff = now - 60.0
                self._buckets = {
                    k: v for k, v in self._buckets.items() if v[1] >= cutoff
                }
            tokens, last = self._buckets.get(key, (self._burst, now))
            tokens = min(self._burst, tokens + (now - last) * self._rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True
            self._buckets[key] = (tokens, now)
            return False


class AlertAPIServer:
    """
    告警 API 服务（基于 aiohttp）

    接口列表：
        GET  /stream                  - MJPEG 实时视频流
        GET  /api/v1/status           - 设备状态
        GET  /api/v1/stats            - 实时统计
        GET  /api/v1/alerts           - 告警历史查询
        POST /api/v1/alerts/:id/ack   - 确认/消警
        WS   /ws/alerts               - 告警实时推送
        GET  /health/pose             - MoveNet 姿态模型健康检查（可观测性）
        GET  /health/detection        - YOLO26 检测模型健康检查（可观测性）
        GET  /health                  - 总体健康检查（含 YOLO + MoveNet）

    说明：
        前端正式 UI 为独立项目（frontend/，PyQt6 桌面端），后端不内嵌
        任何页面，仅暴露 API / 视频流 / WebSocket 供前端或调试工具消费。
    """

    _MAX_ALERT_HISTORY: int = 10000

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        self._app: web.Application | None = None
        self._ws_clients: list[web.WebSocketResponse] = []
        self._alert_history: deque[AlertLog] = deque(maxlen=self._MAX_ALERT_HISTORY)
        self._runner: web.AppRunner | None = None
        self.video_hub = VideoHub()
        self.video_analyzer = VideoAnalyzer()
        self._db: AlertDatabase | None = None
        # 家长端路由（start() 时构造并注册）
        self._parent_routes: ParentAPIRoutes | None = None
        # HLS 直播发布器（家长端远程观看，pipeline.start() 时注入；
        # None 或 enabled=False 时对应端点返回 404）
        self._hls_publisher: HLSPublisher | None = None
        # 认证凭据（在 .env 加载后构造，因此此处读取是安全的）
        # 设计动机：旧实现模块导入时读取 os.environ，导致 .env 未注入时
        # 凭据为空、认证静默失效。改为实例级读取，保证拿到真实配置。
        self._auth_user = os.environ.get("LG_AUTH_USER", "")
        self._auth_pass = os.environ.get("LG_AUTH_PASS", "")
        # 注入：模型健康快照提供者（None 表示未注入，对应端点返回 503）
        self._pose_health_provider: HealthProviderLike | None = None
        self._detection_health_provider: HealthProviderLike | None = None
        self._face_health_provider: HealthProviderLike | None = None
        # 每 IP 令牌桶限流（rate<=0 时禁用）
        self._rate_limiter = _TokenBucketRateLimiter(
            config.rate_limit_rps, config.rate_limit_burst
        )

    async def start(self) -> None:
        """启动 API 服务"""
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for API server: pip install aiohttp"
            ) from None

        self._app = web.Application(
            client_max_size=_MAX_UPLOAD_SIZE,
            # 始终挂认证+审计中间件（内部按 auth_enabled 决定是否强校验）
            middlewares=[_make_server_middleware(self)],
        )

        # 路由注册
        self._app.router.add_get("/stream", self._handle_stream)
        # HLS 直播（家长端远程观看；静态路由须注册在 /stream/{segment} 之前）
        self._app.router.add_get("/stream/live.m3u8", self._handle_hls_playlist)
        self._app.router.add_get("/stream/{segment}", self._handle_hls_segment)
        self._app.router.add_get(self._config.status_path, self._handle_status)
        self._app.router.add_get("/api/v1/stats", self._handle_stats)
        self._app.router.add_get(self._config.alerts_path, self._handle_alerts)
        self._app.router.add_post(
            "/api/v1/alerts/{alert_id}/ack", self._handle_ack,
        )
        self._app.router.add_post("/api/v1/alerts/ack-all", self._handle_ack_all)
        self._app.router.add_get(self._config.ws_alerts_path, self._handle_ws)

        # 视频上传 & 分析路由
        self._app.router.add_post("/api/v1/upload", self._handle_upload)
        self._app.router.add_post(
            "/api/v1/analyze/{task_id}", self._handle_analyze,
        )
        self._app.router.add_get(
            "/api/v1/analyze/{task_id}/status", self._handle_analyze_status,
        )
        self._app.router.add_get(
            "/api/v1/analyze/{task_id}/stream", self._handle_analyze_stream,
        )
        self._app.router.add_get(
            "/api/v1/analyze/{task_id}/frame", self._handle_analyze_frame,
        )

        # Prometheus 指标端点
        self._app.router.add_get("/metrics", self._handle_metrics)

        # 健康检查端点（运维可观测性：分别报告 YOLO/MoveNet 状态）
        self._app.router.add_get("/health/pose", self._handle_health_pose)
        self._app.router.add_get("/health/detection", self._handle_health_detection)
        self._app.router.add_get("/health/face", self._handle_health_face)
        self._app.router.add_get("/health", self._handle_health)

        # 审计日志查询端点（合规：查看敏感接口访问记录）
        self._app.router.add_get("/api/v1/audit", self._handle_audit)

        # 家长端路由（微信小程序 + 管理后台，自带 JWT 鉴权，
        # 由中间件按路径前缀豁免 Basic Auth）
        from loongguard.api.parent_routes import ParentAPIRoutes

        self._parent_routes = ParentAPIRoutes()
        self._parent_routes.register(self._app)

        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self._config.host, self._config.port)
        await site.start()
        self._runner = runner

        # 获取实际绑定端口
        actual_port = (
            self._runner.addresses[0][1] if self._runner.addresses else self._config.port
        )
        logger.info(
            "API server started on http://%s:%d", self._config.host, actual_port,
        )
        logger.info(
            "Video stream: http://localhost:%d/stream", actual_port,
        )

    async def stop(self) -> None:
        """停止 API 服务"""
        for ws in self._ws_clients:
            await ws.close()
        self._ws_clients.clear()

        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

        logger.info("API server stopped")

    # ── 告警推送（供 Pipeline 调用）───────────────────────────

    async def push_alert(self, alert: AlertLog) -> None:
        """推送新告警到所有 WebSocket 客户端"""
        self._alert_history.append(alert)
        payload = json.dumps(alert.to_dict(), ensure_ascii=False)

        disconnected: list[web.WebSocketResponse] = []
        for ws in self._ws_clients:
            try:
                await ws.send_str(payload)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self._ws_clients.remove(ws)

    # ── HTTP handlers ─────────────────────────────────────────

    async def _handle_stream(self, request: web.Request) -> web.StreamResponse:
        """
        MJPEG 实时视频流

        浏览器通过 <img src="/stream"> 接收连续 JPEG 帧。
        Content-Type: multipart/x-mixed-replace 实现自动刷新。

        性能优化（解决 Web 端帧率卡顿）：
            - 帧跳过：若客户端 write 未完成而新帧已就绪，跳过旧帧
            - 自适应间隔：根据实际传输耗时动态调整 sleep
            - 降质传输：Web 端 JPEG quality=50（比本地 75 更小）
        """

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache, no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        last_sent_count = -1

        try:
            while True:
                t0 = time.monotonic()

                current_count = self.video_hub._frame_count
                jpeg = self.video_hub.get_web_jpeg()
                if jpeg is None:
                    await _send_placeholder(response)
                    await _async_sleep(0.5)
                    continue

                if current_count == last_sent_count:
                    # 无新帧（摄像头暂时停流/文件源播完）：短暂退避后重发
                    # 最新帧而非空转——浏览器画面不变，而以本流为输入的
                    # ffmpeg/HLS 等消费方不会因长期无数据卡死在探测阶段。
                    await _async_sleep(0.5)
                    jpeg = self.video_hub.get_web_jpeg()
                    current_count = self.video_hub._frame_count

                try:
                    await asyncio.wait_for(
                        response.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + jpeg
                            + b"\r\n",
                        ),
                        timeout=2.0,
                    )
                    last_sent_count = current_count
                except (TimeoutError, ConnectionResetError):
                    break

                # 自适应间隔：保证总间隔不低于 40ms，但不叠加传输耗时
                elapsed = time.monotonic() - t0
                await _async_sleep(max(0.04 - elapsed, 0.005))

        except (ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception:
            logger.debug("Stream client disconnected")

        return response

    async def _handle_status(self, request: web.Request) -> web.Response:
        """设备状态接口"""

        return web.json_response({
            "status": "running",
            "model": "YOLO26-Nano + MoveNet-Lightning",
            "total_alerts": len(self._alert_history),
        })

    async def _handle_stats(self, request: web.Request) -> web.Response:
        """实时统计接口"""

        stats = self.video_hub.get_stats()
        stats["total_alerts_history"] = len(self._alert_history)
        return web.json_response(stats)

    async def _handle_alerts(self, request: web.Request) -> web.Response:
        """告警历史查询接口（支持分页、过滤、时间范围）"""

        try:
            page = max(1, int(request.query.get("page", "1")))
        except (ValueError, TypeError):
            page = 1
        try:
            page_size = min(100, max(1, int(request.query.get("page_size", "20"))))
        except (ValueError, TypeError):
            page_size = 20

        severity = request.query.get("severity") or None
        alert_type = request.query.get("type") or None
        ack_str = request.query.get("acknowledged")
        acknowledged = None
        if ack_str is not None:
            acknowledged = ack_str.lower() in ("true", "1")
        start_time = request.query.get("start_time")
        end_time = request.query.get("end_time")

        # 优先从数据库查询（持久化数据更完整）
        if self._db is not None:
            try:
                result = self._db.query_alerts(
                    page=page,
                    page_size=page_size,
                    severity=severity,
                    alert_type=alert_type,
                    acknowledged=acknowledged,
                    start_time=start_time,
                    end_time=end_time,
                )
                return web.json_response(result)
            except Exception:
                logger.exception("DB query failed, falling back to memory")

        # 回退：从内存 deque 查询
        filtered = list(self._alert_history)
        if start_time:
            filtered = [a for a in filtered if a.timestamp >= start_time]
        if end_time:
            filtered = [a for a in filtered if a.timestamp <= end_time]

        total = len(filtered)
        start_idx = (page - 1) * page_size
        page_data = [
            a.to_dict() for a in filtered[start_idx:start_idx + page_size]
        ]

        return web.json_response({
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": page_data,
        })

    async def _handle_ack(self, request: web.Request) -> web.Response:
        """确认/消警接口"""

        alert_id = request.match_info["alert_id"]

        # 写入数据库
        if self._db is not None:
            try:
                if self._db.ack_alert(alert_id):
                    # 同步更新内存副本
                    for alert in self._alert_history:
                        if alert.alert_id == alert_id:
                            alert.acknowledged = True
                    return web.json_response({"ok": True})
                return web.json_response({"error": "not found"}, status=404)
            except Exception:
                logger.exception("DB ack failed, falling back to memory")

        # 回退：仅内存
        for alert in self._alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return web.json_response({"ok": True})

        return web.json_response({"error": "not found"}, status=404)

    async def _handle_ack_all(self, request: web.Request) -> web.Response:
        """批量确认所有未确认的告警"""

        acked = 0

        if self._db is not None:
            try:
                result = self._db.query_alerts(
                    page=1, page_size=10000, acknowledged=False,
                )
                for item in result.get("data", []):
                    if self._db.ack_alert(item["alert_id"]):
                        acked += 1
            except Exception:
                logger.exception("DB ack-all failed")

        # 同步更新内存副本（不重复计数，DB 侧已计）
        for alert in self._alert_history:
            if not alert.acknowledged:
                alert.acknowledged = True

        return web.json_response({"ok": True, "acked": acked})

    async def _handle_ws(self, request: web.Request) -> web.StreamResponse:
        """WebSocket 告警推送端点"""

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.append(ws)
        logger.info(
            "WebSocket client connected (total: %d)", len(self._ws_clients),
        )

        # 空转等待断开：async for 自然结束（连接关闭）后清理客户端
        async for _ in ws:
            pass

        self._ws_clients.remove(ws)
        logger.info("WebSocket client disconnected")
        return ws

    # ── HLS 直播（家长端远程观看，token 保护）────────────────

    def set_hls_publisher(self, publisher: HLSPublisher) -> None:
        """注入 HLS 发布器（pipeline.start() 时构造；None 表示未启用）"""
        self._hls_publisher = publisher

    async def _handle_hls_playlist(self, request: web.Request) -> web.Response:
        """
        HLS 播放列表（/stream/live.m3u8?token=...）

        鉴权：带过期 HMAC token（家长端经 /api/wx/stream/url 获取）。
        空闲停机后的首个请求会重新拉起 ffmpeg，并短暂等待首个分片产出。
        """
        pub = self._hls_publisher
        if pub is None or not getattr(pub, "enabled", False):
            return web.json_response(
                {"error": "HLS 直播未启用（ffmpeg 缺失或未配置）"}, status=404)
        if not pub.verify_token(request.query.get("token", "")):
            return web.json_response(
                {"error": "无效或已过期的观看凭证"}, status=401)

        pub.ensure_started()

        playlist_text = pub.read_playlist()
        if playlist_text is None:
            # 首次启动 ffmpeg 需要完成输入探测并产出首个分片（约 5~15s），
            # 轮询等待；超时则让播放器稍后重试（小程序端有刷新入口）
            deadline = time.monotonic() + 20.0
            while playlist_text is None and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                playlist_text = pub.read_playlist()
        if playlist_text is None:
            return web.json_response(
                {"error": "直播流尚未就绪，请稍后重试"}, status=503)

        pub.touch()
        return web.Response(
            text=playlist_text,
            content_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "no-cache, no-store"},
        )

    async def _handle_hls_segment(self, request: web.Request) -> web.Response:
        """HLS TS 分片（/stream/live_XXXX.ts?token=...），同样要求 token"""
        pub = self._hls_publisher
        if pub is None or not getattr(pub, "enabled", False):
            return web.json_response({"error": "HLS 直播未启用"}, status=404)
        if not pub.verify_token(request.query.get("token", "")):
            return web.json_response(
                {"error": "无效或已过期的观看凭证"}, status=401)

        seg_bytes = pub.read_segment(request.match_info["segment"])
        if seg_bytes is None:
            return web.json_response({"error": "分片不存在或已过期"}, status=404)

        pub.touch()
        return web.Response(
            body=seg_bytes,
            content_type="video/mp2t",
            headers={"Cache-Control": "no-cache"},
        )

    # ── 视频上传 & 分析 handlers ─────────────────────────────

    async def _handle_upload(self, request: web.Request) -> web.Response:
        """
        视频文件上传接口

        支持 multipart/form-data 格式，字段名 "video"。
        文件大小限制 500MB，仅接受主流视频格式。
        """

        reader = await request.multipart()
        video_field = await reader.next()

        if video_field is None or video_field.name != "video":
            return web.json_response(
                {"error": "请上传视频文件（字段名: video）"},
                status=400,
            )

        filename = video_field.filename or "unknown.mp4"
        ext = Path(filename).suffix.lower()

        if ext not in _ALLOWED_VIDEO_EXTENSIONS:
            return web.json_response(
                {
                    "error": f"不支持的视频格式: {ext}",
                    "allowed": sorted(_ALLOWED_VIDEO_EXTENSIONS),
                },
                status=400,
            )

        # 流式写入，实时检查文件大小
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        save_name = f"{uuid.uuid4().hex[:12]}_{Path(filename).name}"
        save_path = _UPLOAD_DIR / save_name
        size = 0

        with open(save_path, "wb") as f:
            while True:
                chunk = await video_field.read_chunk(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_UPLOAD_SIZE:
                    f.close()
                    save_path.unlink(missing_ok=True)
                    return web.json_response(
                        {
                            "error": f"文件大小超过限制（最大 {_MAX_UPLOAD_SIZE // 1024 // 1024}MB）",
                        },
                        status=413,
                    )
                f.write(chunk)

        # 创建分析任务
        task = self.video_analyzer.create_task(filename, str(save_path))

        logger.info("Video uploaded: %s (%.1f MB), task_id=%s", filename, size / 1024 / 1024, task.task_id)

        return web.json_response({
            "task_id": task.task_id,
            "filename": filename,
            "size_bytes": size,
            "message": "上传成功，请调用 /api/v1/analyze/{task_id} 开始分析",
        })

    async def _handle_analyze(self, request: web.Request) -> web.Response:
        """触发视频分析任务"""

        task_id = request.match_info["task_id"]
        task = self.video_analyzer.get_task(task_id)
        if not task:
            return web.json_response({"error": "任务不存在"}, status=404)

        if task.status not in (AnalysisStatus.PENDING, AnalysisStatus.FAILED):
            return web.json_response(
                {"error": f"任务状态为 {task.status.value}，无法重新启动"},
                status=409,
            )

        # 在后台启动分析
        asyncio.create_task(self.video_analyzer.run_analysis(task_id))

        return web.json_response({
            "task_id": task_id,
            "status": "processing",
            "message": "分析已启动",
        })

    async def _handle_analyze_status(self, request: web.Request) -> web.Response:
        """查询分析任务状态"""

        task_id = request.match_info["task_id"]
        task = self.video_analyzer.get_task(task_id)
        if not task:
            return web.json_response({"error": "任务不存在"}, status=404)

        return web.json_response(task.to_status_dict())

    async def _handle_analyze_stream(self, request: web.Request) -> web.StreamResponse:
        """
        SSE 实时分析结果流

        前端通过 EventSource 连接此端点，实时接收逐帧分析结果。
        事件类型：meta / frame / complete / error / status
        """

        task_id = request.match_info["task_id"]
        task = self.video_analyzer.get_task(task_id)
        if not task:
            return web.json_response({"error": "任务不存在"}, status=404)

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        q = task.register_sse()

        try:
            # 如果任务已经开始处理，补发 meta 事件给新连接的 SSE 客户端
            if task.video_meta is not None and task.status == AnalysisStatus.PROCESSING:
                meta_json = json.dumps(task.video_meta, ensure_ascii=False)
                await response.write(f"event: meta\ndata: {meta_json}\n\n".encode())

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except TimeoutError:
                    # 心跳：防止连接超时
                    await response.write(b": heartbeat\n\n")
                    continue

                event_type = event.get("type", "message")
                event_data = json.dumps(event.get("data", {}), ensure_ascii=False)
                await response.write(
                    f"event: {event_type}\ndata: {event_data}\n\n".encode()
                )

                if event_type in ("complete", "error"):
                    break

        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            task.unregister_sse(q)

        return response

    async def _handle_analyze_frame(self, request: web.Request) -> web.Response:
        """获取当前分析帧的标注图片"""

        task_id = request.match_info["task_id"]
        task = self.video_analyzer.get_task(task_id)
        if not task:
            return web.json_response({"error": "任务不存在"}, status=404)

        if task.current_frame_jpeg is None:
            return web.Response(status=204)

        return web.Response(
            body=task.current_frame_jpeg,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Prometheus 指标端点"""

        from loongguard.api.metrics import get_metrics_bytes, get_metrics_content_type

        resp = web.Response(
            body=get_metrics_bytes(),
            content_type="text/plain",
        )
        # CONTENT_TYPE_LATEST 形如 "text/plain; version=0.0.4; charset=utf-8"，
        # aiohttp 的 content_type 参数不允许携带 charset，故直接覆写完整 Header，
        # 以忠实保留 Prometheus 文本格式的 version 参数。
        resp.headers["Content-Type"] = get_metrics_content_type()
        return resp

    async def _handle_health_pose(self, request: web.Request) -> web.Response:
        """
        MoveNet 姿态模型健康检查端点

        设计动机：
            俯卧检测是核心安全功能，必须能独立于 YOLO 状态被监控。
            当 MoveNet 加载失败/推理异常时，本端点立即返回 503，
            运维可据此触发告警 + 派单，而无需检查 Prometheus 指标。

        Returns:
            200: 正常快照
            503: 模型未注入或不可用（pose 模块降级关闭）
        """

        if self._pose_health_provider is None:
            return web.json_response(
                {"available": False, "reason": "pose module not injected"},
                status=503,
            )
        try:
            snapshot = self._pose_health_provider()
        except Exception as exc:
            logger.exception("pose health provider failed")
            return web.json_response(
                {"available": False, "reason": f"health error: {exc}"},
                status=503,
            )
        status = 200 if snapshot.get("available") else 503
        return web.json_response(snapshot, status=status)

    async def _handle_health_detection(self, request: web.Request) -> web.Response:
        """
        YOLO26 检测模型健康检查端点

        与 /health/pose 对称。YOLO 是危险物品检测主链路，监控其健康度
        比 pose 更关键：当 YOLO 挂掉时，整个系统失去最核心的告警能力。
        """

        if self._detection_health_provider is None:
            return web.json_response(
                {"available": False, "reason": "detection module not injected"},
                status=503,
            )
        try:
            snapshot = self._detection_health_provider()
        except Exception as exc:
            logger.exception("detection health provider failed")
            return web.json_response(
                {"available": False, "reason": f"health error: {exc}"},
                status=503,
            )
        status = 200 if snapshot.get("available") else 503
        return web.json_response(snapshot, status=status)

    async def _handle_health_face(self, request: web.Request) -> web.Response:
        """
        人脸检测模型健康检查端点

        与 /health/pose 对称。人脸检测是睡姿监测辅助功能，
        其 health 状态用于运维确认模型是否正常加载。
        """

        if self._face_health_provider is None:
            return web.json_response(
                {"available": False, "reason": "face module not injected"},
                status=503,
            )
        try:
            snapshot = self._face_health_provider()
        except Exception as exc:
            logger.exception("face health provider failed")
            return web.json_response(
                {"available": False, "reason": f"health error: {exc}"},
                status=503,
            )
        status = 200 if snapshot.get("available") else 503
        return web.json_response(snapshot, status=status)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """
        总体健康检查

        设计动机：负载均衡器 / K8s liveness probe 探活使用，
        不应细粒度到具体模型，避免一个模型挂掉就重启整个进程。
        进程存活即视为健康；模型级健康请查询 /health/pose、/health/detection、/health/face。
        """

        # 聚合所有子端点的状态：任一不可用时 overall=false
        components = {
            "pose": "missing",
            "detection": "missing",
            "face": "missing",
        }
        if self._pose_health_provider is not None:
            try:
                snap = self._pose_health_provider()
                components["pose"] = "ok" if snap.get("available") else "degraded"
            except Exception:
                components["pose"] = "error"
        if self._detection_health_provider is not None:
            try:
                snap = self._detection_health_provider()
                components["detection"] = "ok" if snap.get("available") else "degraded"
            except Exception:
                components["detection"] = "error"
        if self._face_health_provider is not None:
            try:
                snap = self._face_health_provider()
                components["face"] = "ok" if snap.get("available") else "degraded"
            except Exception:
                components["face"] = "error"

        overall_ok = all(v in ("ok", "missing") for v in components.values())

        # 暴露实际生效的 ONNX 执行提供程序（运维确认 CUDA/OpenCL 是否真生效）
        try:
            from loongguard.utils.onnx_session import select_providers

            providers = select_providers()
        except Exception:
            providers = []

        return web.json_response({
            "status": "running" if overall_ok else "degraded",
            "components": components,
            "onnx_providers": providers,
        }, status=200 if overall_ok else 503)

    def set_database(self, db: AlertDatabase) -> None:
        """注入数据库引用，供告警查询和确认接口使用"""
        self._db = db

    # ── 认证与审计（供中间件调用）─────────────────────────

    def auth_ok(self, request: web.Request) -> bool:
        """
        Basic Auth 校验。

        未启用认证（auth_enabled=False）时直接放行；
        启用后凭据缺失视为未授权（fail-closed，安全优先）。
        """
        if not self._config.auth_enabled:
            return True
        if not self._auth_user or not self._auth_pass:
            return False
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            user, _, passwd = decoded.partition(":")
        except (ValueError, UnicodeDecodeError):
            return False
        return hmac.compare_digest(user, self._auth_user) and hmac.compare_digest(
            passwd, self._auth_pass
        )

    def enqueue_audit(self, request: web.Request) -> None:
        """
        异步记录敏感接口访问审计（不阻塞请求）。

        sqlite 写入放到线程池执行，避免阻塞事件循环。
        """
        if self._db is None:
            return
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "method": request.method,
            "path": request.path,
            "remote": request.remote or "",
            "user_agent": (request.headers.get("User-Agent", "") or "")[:200],
        }
        asyncio.create_task(asyncio.to_thread(self._db.insert_audit, entry))

    async def _handle_audit(self, request: web.Request) -> web.Response:
        """审计日志查询（分页），用于合规审查"""

        if self._db is None:
            return web.json_response({"error": "database not initialized"}, status=503)

        try:
            page = max(1, int(request.query.get("page", "1")))
        except (ValueError, TypeError):
            page = 1
        try:
            page_size = min(100, max(1, int(request.query.get("page_size", "20"))))
        except (ValueError, TypeError):
            page_size = 20

        return web.json_response(self._db.query_audit(page=page, page_size=page_size))

    def set_pose_health_provider(self, provider: HealthProviderLike) -> None:
        """
        注入姿态模型健康快照提供者

        Args:
            provider: 无参可调用对象（如 MoveNetLightning.get_health_snapshot 绑定的 lambda）

        设计动机：API 层不应直接持有 MoveNetLightning 引用以避免循环依赖。
        Pipeline 在 start() 时注入一个 bound method 或 lambda，
        server 仅负责序列化和 HTTP 状态码映射。
        """
        self._pose_health_provider = provider

    def set_detection_health_provider(self, provider: HealthProviderLike) -> None:
        """
        注入 YOLO26 检测模型健康快照提供者（与 pose provider 对称）
        """
        self._detection_health_provider = provider

    def set_face_health_provider(self, provider: HealthProviderLike) -> None:
        """
        注入人脸检测模型健康快照提供者

        Args:
            provider: 无参可调用对象（如 ONNXFaceDetector 的 is_available 封装）

        设计动机：与 pose/detection provider 对称，统一健康检查模式。
        """
        self._face_health_provider = provider


# ── 辅助函数 ──────────────────────────────────────────────────


async def _async_sleep(seconds: float) -> None:
    """非阻塞等待"""
    await asyncio.sleep(seconds)


async def _send_placeholder(response: web.StreamResponse) -> None:
    """发送 1x1 黑色占位 JPEG（无帧数据时）"""
    # 最小合法 JPEG: 1x1 黑色像素
    placeholder = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa7\x93"
        b"\xff\xd9"
    )
    await response.write(
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + placeholder
        + b"\r\n",
    )


# ── Basic Auth 中间件 + 访问审计 ────────────────────────────

# 需要审计记录的高敏感路径（涉及未成年人隐私）
_AUDIT_PATHS = {
    "/stream",
    "/stream/live.m3u8",
    "/ws/alerts",
    "/api/v1/alerts",
    "/api/v1/stats",
    "/api/v1/status",
}

# 家长端路由前缀（微信小程序 /api/wx/*、管理后台 /api/admin/* 与 /admin 页面、
# 静态资源与上传文件）。这些路径自带独立 JWT 鉴权（parent_routes.py），
# 豁免主仓库 Basic Auth，否则小程序与浏览器后台无法携带 Basic 凭据访问。
_PARENT_PATH_PREFIXES = (
    "/api/wx/",
    "/api/admin/",
    "/static/",
    "/uploads/",
)


def _is_parent_path(path: str) -> bool:
    """
    判断请求路径是否属于家长端（豁免 Basic Auth，限流与审计不受影响）。

    - /admin 采用"精确匹配 + /admin/ 子路径"两种形态，避免 /administrator
      等无关路径被误豁免主仓库 Basic Auth。
    - /stream/<子路径> 为 HLS 直播（mjpeg 的 /stream 无尾斜杠、不受影响），
      由带过期 HMAC token 自行鉴权（parent_routes 签发），故豁免 Basic Auth。
    """
    if path.startswith(_PARENT_PATH_PREFIXES):
        return True
    if path == "/stream/live.m3u8" or path.startswith("/stream/"):
        return True
    return path == "/admin" or path.startswith("/admin/")


def _make_server_middleware(server: AlertAPIServer) -> web.middleware:
    """
    生成后端中间件（限流 + 认证 + 访问审计）闭包工厂。

    设计动机：
        1. 认证凭据从 server 实例读取（.env 加载后构造），修复旧实现
           "模块导入时读 os.environ 导致凭据为空"的时序 bug。
        2. 敏感路径访问（含失败尝试）写入审计日志，满足未成年人隐私合规。
        3. 使用 hmac.compare_digest 防止时序攻击。
        4. 每 IP 令牌桶限流置于最前：先挡暴力流量，再做认证与审计，
           避免扫描请求消耗认证计算与审计写入。/metrics 豁免
           （Prometheus 抓取频率固定且为运维必需）。
    """

    @web.middleware
    async def _middleware(request: web.Request, handler: web.RequestHandler) -> web.StreamResponse:
        # 限流：每客户端 IP 令牌桶（/metrics 豁免）
        limiter = server._rate_limiter
        if limiter.enabled and request.path != "/metrics":
            client_ip = request.remote or "unknown"
            if not limiter.allow(client_ip):
                RATE_LIMITED.inc()
                raise web.HTTPTooManyRequests(
                    headers={"Retry-After": "1"},
                    text="Rate limit exceeded",
                )

        # 审计：高敏感路径记录访问（认证前记录，失败尝试同样留痕）
        if request.path in _AUDIT_PATHS:
            server.enqueue_audit(request)

        # 认证：/metrics 放行（Prometheus 抓取），家长端路径自带 JWT 鉴权放行，
        # 其余按配置校验
        if (
            request.path != "/metrics"
            and not _is_parent_path(request.path)
            and not server.auth_ok(request)
        ):
            raise web.HTTPUnauthorized(
                headers={"WWW-Authenticate": 'Basic realm="LoongGuard"'},
            )

        return await handler(request)

    return _middleware
