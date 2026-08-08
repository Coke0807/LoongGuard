"""
Prometheus 性能指标模块

设计动机：
    暴露系统关键性能指标，供 Prometheus 采集。
    指标命名遵循 Prometheus 最佳实践：{namespace}_{subsystem}_{name}_{unit}

    当 prometheus_client 未安装时（如 LoongArch 最小部署），
    自动降级为 Mock 实现，确保核心功能不受监控组件缺失影响。

暴露端点：GET /metrics

指标列表：
    - loongguard_frames_total              (Counter)   已处理帧总数
    - loongguard_detections_total          (Counter)   检测结果总数（按类别标签）
    - loongguard_alerts_total              (Counter)   告警总数（按严重级别和类型标签）
    - loongguard_inference_latency_seconds (Histogram) 推理延迟分布
    - loongguard_pipeline_fps              (Gauge)     当前管道帧率
    - loongguard_ws_clients                (Gauge)     活跃 WebSocket 连接数
    - loongguard_alert_dedup_suppressed_total (Counter) 被去重抑制的告警数
    - loongguard_uploads_total             (Counter)   视频上传总数
    - loongguard_db_operations_total       (Counter)   数据库操作总数（按 operation 标签）
    - loongguard_uptime_seconds            (Gauge)     系统运行时长
"""

from __future__ import annotations

import logging
import time
from typing import Union

logger = logging.getLogger(__name__)

# ── 尝试导入 prometheus_client，失败则使用 Mock ──────────────────────
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    logger.warning(
        "prometheus_client 未安装，监控指标将以 Mock 模式运行。"
        "如需 Prometheus 监控，请安装: pip install prometheus_client"
    )

    # Mock 实现：所有指标操作为空操作，不产生任何开销
    class _MockMetric:
        """Mock 指标基类，所有操作为空操作"""

        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, amount=1):
            pass

        def dec(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

    Counter = _MockMetric
    Gauge = _MockMetric
    Histogram = _MockMetric

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def generate_latest() -> bytes:
        """Mock：返回空指标"""
        return b"# prometheus_client not installed\n"

# ── 命名空间：所有指标统一前缀 ──────────────────────────────

FRAME_COUNT = Counter(
    "loongguard_frames_total",
    "Total frames processed by the pipeline",
)

DETECTION_COUNT = Counter(
    "loongguard_detections_total",
    "Total object detections",
    ["class_name"],  # 按检测类别统计
)

ALERT_COUNT = Counter(
    "loongguard_alerts_total",
    "Total alerts generated",
    ["severity", "alert_type"],  # 按严重级别和类型统计
)

INFERENCE_LATENCY = Histogram(
    "loongguard_inference_latency_seconds",
    "Detection model inference latency in seconds",
    buckets=[0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
)

PIPELINE_FPS = Gauge(
    "loongguard_pipeline_fps",
    "Current pipeline processing FPS",
)

ACTIVE_WS_CLIENTS = Gauge(
    "loongguard_ws_clients",
    "Number of active WebSocket connections",
)

ALERT_DEDUP_SUPPRESSED = Counter(
    "loongguard_alert_dedup_suppressed_total",
    "Alerts suppressed by deduplication",
)

UPLOAD_COUNT = Counter(
    "loongguard_uploads_total",
    "Total video file uploads",
)

DB_OPERATIONS = Counter(
    "loongguard_db_operations_total",
    "Database operations count",
    ["operation"],  # insert, query, ack, cleanup
)

START_TIME = Gauge(
    "loongguard_uptime_seconds",
    "System uptime in seconds",
)

# ── 姿态估计 (MoveNet) 专用指标 ────────────────────────────────
# 设计动机：YOLO 与 MoveNet 是两个独立模型，应能独立观测健康状态。
# 当 POSE_AVAILABLE=0 时俯卧检测已降级关闭，运维通过该指标可立即识别。

POSE_AVAILABLE = Gauge(
    "loongguard_pose_available",
    "1 if MoveNet model loaded successfully, 0 if degraded/disabled",
)

POSE_INFERENCE_COUNT = Counter(
    "loongguard_pose_inference_total",
    "Total successful MoveNet pose inferences",
)

POSE_ERROR_COUNT = Counter(
    "loongguard_pose_errors_total",
    "Total MoveNet pose inference failures (caught exceptions)",
)

POSE_INFERENCE_LATENCY = Histogram(
    "loongguard_pose_inference_latency_seconds",
    "MoveNet pose inference latency in seconds",
    buckets=[0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
)

# ── 目标检测 (YOLO26) 专用指标 ────────────────────────────────
# 设计动机：YOLO26 是危险物品检测主链路，应能独立观测健康状态。
# 与 pose 指标对称：available / count / error / latency。

DETECTION_AVAILABLE = Gauge(
    "loongguard_detection_available",
    "1 if YOLO26 model loaded successfully, 0 if load failed",
)

DETECTION_INFERENCE_COUNT = Counter(
    "loongguard_detection_inference_total",
    "Total successful YOLO26 inferences",
)

DETECTION_ERROR_COUNT = Counter(
    "loongguard_detection_errors_total",
    "Total YOLO26 inference failures (caught exceptions)",
)


def get_metrics_bytes() -> bytes:
    """生成 Prometheus 文本格式的指标快照"""
    return generate_latest()


def get_metrics_content_type() -> str:
    """返回 Prometheus 指标的 Content-Type"""
    return CONTENT_TYPE_LATEST


class MetricsTimer:
    """
    上下文管理器：自动记录代码块执行耗时到 Histogram

    用法：
        with MetricsTimer(INFERENCE_LATENCY):
            result = model.predict(image)
    """

    def __init__(self, histogram: Union[Histogram, _MockMetric]):
        self._histogram = histogram
        self._start: float = 0

    def __enter__(self) -> MetricsTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        elapsed = time.monotonic() - self._start
        self._histogram.observe(elapsed)
