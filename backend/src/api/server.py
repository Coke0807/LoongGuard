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
import io
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
from aiohttp import web

from config import APIConfig
from src.utils.schema import AlertLog, BoundingBox

logger = logging.getLogger(__name__)

# 前端页面目录
_WEB_DIR = Path(__file__).parent / "web"
# 临时上传目录
_UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"

# 上传限制：500MB
_MAX_UPLOAD_SIZE = 500 * 1024 * 1024
# 支持的视频格式
_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


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
        self._latest_jpeg: Optional[bytes] = None
        self._latest_web_jpeg: Optional[bytes] = None
        self._lock = __import__("threading").Lock()

        # 最新推理叠加数据（由独立推理线程 set_overlay 更新，显示线程读取）
        # 设计动机（帧率解耦）：显示帧率不应受推理耗时限制。
        # 推理线程持续更新最新检测框/运动区域/关键点，push_frame 只负责
        # 把"最新结果"叠加到当前待显示帧上并编码，从而显示可达摄像头原生帧率。
        self._latest_boxes: list[BoundingBox] = []
        self._latest_motion: list = []
        self._latest_keypoints: Optional[np.ndarray] = None

        # 实时统计
        self._frame_count: int = 0
        self._alert_count: int = 0
        self._motion_count: int = 0
        self._start_time: float = time.time()
        self._frame_times: deque[float] = deque(maxlen=60)

    def set_overlay(
        self,
        boxes: Optional[list[BoundingBox]] = None,
        motion_regions: Optional[list] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> None:
        """
        更新最新推理叠加数据（推理线程调用，线程安全）

        与 push_frame 解耦：推理线程在后台持续更新这些结果，
        显示线程 push_frame 时直接读取最新值叠加，互不阻塞。
        """
        with self._lock:
            self._latest_boxes = list(boxes) if boxes else []
            self._latest_motion = list(motion_regions) if motion_regions else []
            self._latest_keypoints = keypoints

    def push_frame(
        self,
        frame_rgb: np.ndarray,
    ) -> None:
        """
        推送一帧到视频流缓冲

        采集线程高频调用（可达摄像头原生帧率），仅负责把推理线程
        更新的最新叠加数据（set_overlay 写入）绘制到当前帧并编码。
        推理与显示解耦，帧率不受推理耗时限制。

        Args:
            frame_rgb: RGB 格式帧数据 (H, W, 3)
        """
        try:
            import cv2

            display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 读取最新推理叠加数据（线程安全快照）
            with self._lock:
                boxes = list(self._latest_boxes)
                motion_regions = list(self._latest_motion)
                keypoints = self._latest_keypoints

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

            # JPEG 编码
            ret, buf = cv2.imencode(
                ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            if ret:
                # Web 端用更低质量，减小带宽（约 20-40KB vs 50-100KB）
                ret_web, buf_web = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 50],
                )
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    if ret_web:
                        self._latest_web_jpeg = buf_web.tobytes()
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

    def get_jpeg(self) -> Optional[bytes]:
        """获取最新一帧的 JPEG 字节（线程安全，本地高质量）"""
        with self._lock:
            return self._latest_jpeg

    def get_web_jpeg(self) -> Optional[bytes]:
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
        _ret, _buf = _cv2.imencode(".jpg", fb, [_cv2.IMWRITE_JPEG_QUALITY, 50])
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
    current_frame_jpeg: Optional[bytes] = None
    # 检测结果
    alerts: list[dict] = field(default_factory=list)
    detections_timeline: list[dict] = field(default_factory=list)
    # 统计
    class_counts: dict = field(default_factory=dict)
    error_message: str = ""
    # 视频元信息（用于 SSE 重连时补发 meta 事件）
    video_meta: Optional[dict] = field(default=None, repr=False)
    # SSE 事件队列（每个连接的客户端独立队列）
    _sse_queues: list = field(default_factory=list, repr=False)
    _lock: Any = field(default_factory=lambda: __import__("threading").Lock(), repr=False)

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

    def register_sse(self) -> asyncio.Queue:
        """注册一个 SSE 客户端队列"""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
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
        # 分析器引用（由 Pipeline 初始化后注入）
        self._detector: Any = None
        self._motion: Any = None
        self._roi_scheduler: Any = None

    def set_modules(self, detector: Any, motion: Any, roi_scheduler: Any) -> None:
        """注入 Pipeline 的检测子模块引用"""
        self._detector = detector
        self._motion = motion
        self._roi_scheduler = roi_scheduler

    @property
    def tasks(self) -> dict[str, AnalysisTask]:
        return self._tasks

    def get_task(self, task_id: str) -> Optional[AnalysisTask]:
        return self._tasks.get(task_id)

    def create_task(self, filename: str, file_path: str) -> AnalysisTask:
        """创建分析任务"""
        task_id = uuid.uuid4().hex[:12]
        task = AnalysisTask(
            task_id=task_id,
            filename=filename,
            file_path=file_path,
        )
        self._tasks[task_id] = task
        return task

    async def run_analysis(self, task_id: str) -> None:
        """
        执行视频分析（后台协程）

        逐帧读取视频 -> 运动检测 -> ROI 检测 -> 推送结果
        """
        task = self._tasks.get(task_id)
        if not task:
            return

        if self._detector is None or self._motion is None:
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
            task.total_frames = total_frames
            meta_data = {
                "total_frames": total_frames,
                "fps": round(fps, 1),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            task.video_meta = meta_data
            task.push_event("meta", meta_data)

            frame_idx = 0
            prev_gray: Optional[np.ndarray] = None

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

                    motion_regions = self._motion.detect(current_gray, prev_gray)
                    prev_gray = current_gray

                    det_boxes: list[BoundingBox] = []
                    frame_alerts: list[dict] = []

                    if motion_regions:
                        alerts = self._roi_scheduler.process(
                            frame_rgb, motion_regions,
                        )
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
                        import cv2 as _cv2
                        _cv2.rectangle(
                            annotated,
                            (box.x1, box.y1), (box.x2, box.y2),
                            (0, 0, 255), 2,
                        )
                        label = f"{box.class_name} {box.confidence:.0%}"
                        (tw, th), _ = _cv2.getTextSize(
                            label, _cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
                        )
                        _cv2.rectangle(
                            annotated,
                            (box.x1, box.y1 - th - 6), (box.x1 + tw, box.y1),
                            (0, 0, 255), -1,
                        )
                        _cv2.putText(
                            annotated, label,
                            (box.x1, box.y1 - 4),
                            _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        )

                    _, jpeg_buf = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60],
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

                    await asyncio.sleep(0)
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

        except Exception as e:
            logger.exception("Video analysis failed for task %s", task_id)
            task.status = AnalysisStatus.FAILED
            task.error_message = str(e)
            task.push_event("error", {"message": str(e)})


class AlertAPIServer:
    """
    告警 API 服务（基于 aiohttp）

    接口列表：
        GET  /                        - Web 可视化面板
        GET  /stream                  - MJPEG 实时视频流
        GET  /api/v1/status           - 设备状态
        GET  /api/v1/stats            - 实时统计
        GET  /api/v1/alerts           - 告警历史查询
        POST /api/v1/alerts/:id/ack   - 确认/消警
        WS   /ws/alerts               - 告警实时推送
        GET  /health/pose             - MoveNet 姿态模型健康检查（可观测性）
        GET  /health/detection        - YOLO26 检测模型健康检查（可观测性）
        GET  /health                  - 总体健康检查（含 YOLO + MoveNet）
    """

    _MAX_ALERT_HISTORY: int = 10000

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        self._app: Any = None
        self._ws_clients: list[Any] = []
        self._alert_history: deque[AlertLog] = deque(maxlen=self._MAX_ALERT_HISTORY)
        self._runner: Any = None
        self.video_hub = VideoHub()
        self.video_analyzer = VideoAnalyzer()
        self._db: Any = None
        # 注入：模型健康快照提供者（None 表示未注入，对应端点返回 503）
        self._pose_health_provider: Optional[Any] = None
        self._detection_health_provider: Optional[Any] = None

    async def start(self) -> None:
        """启动 API 服务"""
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for API server: pip install aiohttp"
            )

        self._app = web.Application(
            client_max_size=_MAX_UPLOAD_SIZE,
            middlewares=[_auth_middleware] if self._config.auth_enabled else [],
        )

        # 路由注册
        self._app.router.add_get("/", self._handle_dashboard)
        self._app.router.add_get("/stream", self._handle_stream)
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
        self._app.router.add_get("/health", self._handle_health)

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
            "Dashboard: http://localhost:%d  |  Video stream: http://localhost:%d/stream",
            actual_port, actual_port,
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

        disconnected: list[Any] = []
        for ws in self._ws_clients:
            try:
                await ws.send_str(payload)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self._ws_clients.remove(ws)

    # ── HTTP handlers ─────────────────────────────────────────

    async def _handle_dashboard(self, request: Any) -> Any:
        """Web 可视化面板"""
        from aiohttp import web

        html_path = _WEB_DIR / "index.html"
        if not html_path.exists():
            return web.Response(
                text="Dashboard not found. Create src/api/web/index.html",
                status=404,
            )
        return web.FileResponse(html_path)

    async def _handle_stream(self, request: Any) -> Any:
        """
        MJPEG 实时视频流

        浏览器通过 <img src="/stream"> 接收连续 JPEG 帧。
        Content-Type: multipart/x-mixed-replace 实现自动刷新。

        性能优化（解决 Web 端帧率卡顿）：
            - 帧跳过：若客户端 write 未完成而新帧已就绪，跳过旧帧
            - 自适应间隔：根据实际传输耗时动态调整 sleep
            - 降质传输：Web 端 JPEG quality=50（比本地 75 更小）
        """
        from aiohttp import web

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

                # 帧跳过：如果 VideoHub 已经更新到新帧，直接发最新帧
                current_count = self.video_hub._frame_count
                if current_count == last_sent_count:
                    # 无新帧，短暂等待
                    await _async_sleep(0.01)
                    continue

                jpeg = self.video_hub.get_web_jpeg()
                if jpeg is None:
                    await _send_placeholder(response)
                else:
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
                    except (asyncio.TimeoutError, ConnectionResetError):
                        break

                # 自适应间隔：保证总间隔不低于 40ms，但不叠加传输耗时
                elapsed = time.monotonic() - t0
                await _async_sleep(max(0.04 - elapsed, 0.005))

        except (ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception:
            logger.debug("Stream client disconnected")

        return response

    async def _handle_status(self, request: Any) -> Any:
        """设备状态接口"""
        from aiohttp import web

        return web.json_response({
            "status": "running",
            "model": "YOLO26-Nano + MoveNet-Lightning",
            "total_alerts": len(self._alert_history),
        })

    async def _handle_stats(self, request: Any) -> Any:
        """实时统计接口"""
        from aiohttp import web

        stats = self.video_hub.get_stats()
        stats["total_alerts_history"] = len(self._alert_history)
        return web.json_response(stats)

    async def _handle_alerts(self, request: Any) -> Any:
        """告警历史查询接口（支持分页、过滤、时间范围）"""
        from aiohttp import web

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

    async def _handle_ack(self, request: Any) -> Any:
        """确认/消警接口"""
        from aiohttp import web

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

    async def _handle_ack_all(self, request: Any) -> Any:
        """批量确认所有未确认的告警"""
        from aiohttp import web

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

    async def _handle_ws(self, request: Any) -> Any:
        """WebSocket 告警推送端点"""
        from aiohttp import web

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.append(ws)
        logger.info(
            "WebSocket client connected (total: %d)", len(self._ws_clients),
        )

        async for msg in ws:
            pass

        self._ws_clients.remove(ws)
        logger.info("WebSocket client disconnected")
        return ws

    # ── 视频上传 & 分析 handlers ─────────────────────────────

    async def _handle_upload(self, request: Any) -> Any:
        """
        视频文件上传接口

        支持 multipart/form-data 格式，字段名 "video"。
        文件大小限制 500MB，仅接受主流视频格式。
        """
        from aiohttp import web

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

    async def _handle_analyze(self, request: Any) -> Any:
        """触发视频分析任务"""
        from aiohttp import web

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

    async def _handle_analyze_status(self, request: Any) -> Any:
        """查询分析任务状态"""
        from aiohttp import web

        task_id = request.match_info["task_id"]
        task = self.video_analyzer.get_task(task_id)
        if not task:
            return web.json_response({"error": "任务不存在"}, status=404)

        return web.json_response(task.to_status_dict())

    async def _handle_analyze_stream(self, request: Any) -> Any:
        """
        SSE 实时分析结果流

        前端通过 EventSource 连接此端点，实时接收逐帧分析结果。
        事件类型：meta / frame / complete / error / status
        """
        from aiohttp import web

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
                await response.write(f"event: meta\ndata: {meta_json}\n\n".encode("utf-8"))

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 心跳：防止连接超时
                    await response.write(b": heartbeat\n\n")
                    continue

                event_type = event.get("type", "message")
                event_data = json.dumps(event.get("data", {}), ensure_ascii=False)
                await response.write(
                    f"event: {event_type}\ndata: {event_data}\n\n".encode("utf-8")
                )

                if event_type in ("complete", "error"):
                    break

        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            task.unregister_sse(q)

        return response

    async def _handle_analyze_frame(self, request: Any) -> Any:
        """获取当前分析帧的标注图片"""
        from aiohttp import web

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

    async def _handle_metrics(self, request: Any) -> Any:
        """Prometheus 指标端点"""
        from aiohttp import web

        from src.api.metrics import get_metrics_bytes, get_metrics_content_type

        return web.Response(
            body=get_metrics_bytes(),
            content_type=get_metrics_content_type(),
        )

    async def _handle_health_pose(self, request: Any) -> Any:
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
        from aiohttp import web

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

    async def _handle_health_detection(self, request: Any) -> Any:
        """
        YOLO26 检测模型健康检查端点

        与 /health/pose 对称。YOLO 是危险物品检测主链路，监控其健康度
        比 pose 更关键：当 YOLO 挂掉时，整个系统失去最核心的告警能力。
        """
        from aiohttp import web

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

    async def _handle_health(self, request: Any) -> Any:
        """
        总体健康检查

        设计动机：负载均衡器 / K8s liveness probe 探活使用，
        不应细粒度到具体模型，避免一个模型挂掉就重启整个进程。
        进程存活即视为健康；模型级健康请查询 /health/pose、/health/detection。
        """
        from aiohttp import web

        # 聚合两个子端点的状态：任一不可用时 overall=false
        components = {
            "pose": "missing",
            "detection": "missing",
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

        overall_ok = all(v in ("ok", "missing") for v in components.values())
        return web.json_response({
            "status": "running" if overall_ok else "degraded",
            "components": components,
        }, status=200 if overall_ok else 503)

    def set_database(self, db: Any) -> None:
        """注入数据库引用，供告警查询和确认接口使用"""
        self._db = db

    def set_pose_health_provider(self, provider: Any) -> None:
        """
        注入姿态模型健康快照提供者

        Args:
            provider: 无参可调用对象（如 MoveNetLightning.get_health_snapshot 绑定的 lambda）

        设计动机：API 层不应直接持有 MoveNetLightning 引用以避免循环依赖。
        Pipeline 在 start() 时注入一个 bound method 或 lambda，
        server 仅负责序列化和 HTTP 状态码映射。
        """
        self._pose_health_provider = provider

    def set_detection_health_provider(self, provider: Any) -> None:
        """
        注入 YOLO26 检测模型健康快照提供者（与 pose provider 对称）
        """
        self._detection_health_provider = provider


# ── 辅助函数 ──────────────────────────────────────────────────


async def _async_sleep(seconds: float) -> None:
    """非阻塞等待"""
    await asyncio.sleep(seconds)


async def _send_placeholder(response: Any) -> None:
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


# ── Basic Auth 中间件 ──────────────────────────────────────────

_AUTH_USER = os.environ.get("LG_AUTH_USER", "")
_AUTH_PASS = os.environ.get("LG_AUTH_PASS", "")


@web.middleware
async def _auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    """
    Basic Auth 验证中间件

    设计动机：
        内网 API 需要轻量级访问控制，避免未授权访问管理面板和视频流。
        Basic Auth 足够用于局域网场景，无需引入 JWT 等第三方依赖。

    仅当 LG_AUTH_USER 和 LG_AUTH_PASS 同时设置时启用验证。
    使用 hmac.compare_digest 防止时序攻击。
    """
    if request.path in ("/metrics",):
        return await handler(request)

    if not _AUTH_USER or not _AUTH_PASS:
        return await handler(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        raise web.HTTPUnauthorized(
            headers={"WWW-Authenticate": 'Basic realm="LoongGuard"'},
        )

    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
        user, _, passwd = decoded.partition(":")
    except (ValueError, UnicodeDecodeError):
        raise web.HTTPUnauthorized(
            headers={"WWW-Authenticate": 'Basic realm="LoongGuard"'},
        )

    if not (hmac.compare_digest(user, _AUTH_USER) and hmac.compare_digest(passwd, _AUTH_PASS)):
        raise web.HTTPUnauthorized(
            headers={"WWW-Authenticate": 'Basic realm="LoongGuard"'},
        )

    return await handler(request)
