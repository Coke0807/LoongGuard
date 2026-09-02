"""
性能优化工具模块

提供性能监控和优化工具。
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    性能监控器

    用于监控和记录关键操作的性能指标。
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._metrics: dict[str, list[float]] = {}
        self._counters: dict[str, int] = {}

    def record(self, metric_name: str, value: float) -> None:
        """
        记录性能指标

        Args:
            metric_name: 指标名称
            value: 指标值（通常是耗时秒数）
        """
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append(value)

    def increment(self, counter_name: str, amount: int = 1) -> None:
        """
        增加计数器

        Args:
            counter_name: 计数器名称
            amount: 增加数量
        """
        self._counters[counter_name] = self._counters.get(counter_name, 0) + amount

    def get_stats(self, metric_name: str) -> dict[str, float]:
        """
        获取指标统计信息

        Args:
            metric_name: 指标名称

        Returns:
            包含 min, max, mean, median, p95, p99 的字典
        """
        if metric_name not in self._metrics:
            return {}

        values = self._metrics[metric_name]
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / n,
            "median": sorted_values[n // 2],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
            "count": n,
        }

    def get_counter(self, counter_name: str) -> int:
        """
        获取计数器值

        Args:
            counter_name: 计数器名称

        Returns:
            计数器值
        """
        return self._counters.get(counter_name, 0)

    def reset(self) -> None:
        """重置所有指标和计数器"""
        self._metrics.clear()
        self._counters.clear()

    def summary(self) -> dict[str, Any]:
        """
        获取性能摘要

        Returns:
            包含所有指标和计数器的字典
        """
        result = {
            "counters": self._counters.copy(),
            "metrics": {},
        }

        for metric_name in self._metrics:
            result["metrics"][metric_name] = self.get_stats(metric_name)

        return result


@contextmanager
def measure_time(
    monitor: PerformanceMonitor | None = None,
    metric_name: str = "operation",
) -> Generator[dict[str, float], None, None]:
    """
    测量代码块执行时间的上下文管理器

    Args:
        monitor: 性能监控器（可选）
        metric_name: 指标名称

    Yields:
        包含 elapsed 键的字典，用于在代码块中获取耗时

    Example:
        with measure_time(monitor, "inference") as ctx:
            result = model.predict(image)
            print(f"Inference took {ctx['elapsed']:.3f}s")
    """
    result = {"elapsed": 0.0}
    start = time.perf_counter()

    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed

        if monitor is not None:
            monitor.record(metric_name, elapsed)


def optimize_numpy_memory() -> None:
    """
    优化 NumPy 内存使用

    调用垃圾回收并清理 NumPy 内部缓存。
    """
    gc.collect()


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的 IoU (Intersection over Union)

    Args:
        box1: 第一个边界框 [x1, y1, x2, y2]
        box2: 第二个边界框 [x1, y1, x2, y2]

    Returns:
        IoU 值（0.0 到 1.0）
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def non_max_suppression(
    detections: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    非极大值抑制 (NMS)

    Args:
        detections: 检测结果 (N, 6): [x1, y1, x2, y2, confidence, class_id]
        iou_threshold: IoU 阈值

    Returns:
        过滤后的检测结果
    """
    if len(detections) == 0:
        return detections

    # 按置信度降序排序
    sorted_indices = np.argsort(detections[:, 4])[::-1]

    keep = []
    while len(sorted_indices) > 0:
        # 选择置信度最高的检测框
        current = sorted_indices[0]
        keep.append(current)

        # 计算当前框与剩余框的 IoU
        remaining = sorted_indices[1:]
        if len(remaining) == 0:
            break

        ious = np.array([
            calculate_iou(detections[current], detections[idx])
            for idx in remaining
        ])

        # 保留 IoU 低于阈值的框
        mask = ious < iou_threshold
        sorted_indices = remaining[mask]

    return detections[keep]


class FrameBuffer:
    """
    帧缓冲区

    用于缓存最近 N 帧，支持异步读取和写入。
    """

    def __init__(self, maxsize: int = 10) -> None:
        self.maxsize = maxsize
        self._buffer: list[np.ndarray] = []
        self._timestamps: list[float] = []

    def push(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        """
        推送一帧到缓冲区

        Args:
            frame: 帧数据
            timestamp: 时间戳（可选）
        """
        if timestamp is None:
            timestamp = time.time()

        self._buffer.append(frame)
        self._timestamps.append(timestamp)

        # 保持缓冲区大小
        while len(self._buffer) > self.maxsize:
            self._buffer.pop(0)
            self._timestamps.pop(0)

    def get_latest(self) -> tuple[np.ndarray, float] | None:
        """
        获取最新一帧

        Returns:
            (frame, timestamp) 或 None（缓冲区为空）
        """
        if not self._buffer:
            return None
        return self._buffer[-1], self._timestamps[-1]

    def get_frame(self, index: int = -1) -> tuple[np.ndarray, float] | None:
        """
        获取指定索引的帧

        Args:
            index: 帧索引（支持负索引）

        Returns:
            (frame, timestamp) 或 None（索引越界）
        """
        if not self._buffer:
            return None
        try:
            return self._buffer[index], self._timestamps[index]
        except IndexError:
            return None

    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer.clear()
        self._timestamps.clear()

    @property
    def size(self) -> int:
        """当前缓冲区大小"""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """缓冲区是否已满"""
        return len(self._buffer) >= self.maxsize
