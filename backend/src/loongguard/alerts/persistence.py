"""
目标物体持续出现跟踪器

设计动机：
    单帧检测结果不可靠，可能出现误检闪烁（如光线变化导致模型短暂误判）。
    只有目标物体在连续帧中持续出现超过指定时间（默认 1 秒）才允许触发告警，
    避免瞬时误检导致的误报。

与 AlertDeduplicator 的区别：
    - AlertDeduplicator：抑制"已触发过的告警"在时间窗口内重复触发
    - ObjectPersistenceTracker：确保"首次触发告警前"物体已稳定出现足够长时间

使用流程：
    tracker = ObjectPersistenceTracker(persistence_sec=1.0)
    ...
    # 每帧检测到物体后调用
    if tracker.is_persistent(detections, time.monotonic()):
        # 物体已持续出现超过 persistence_sec，允许报警
        ...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from loongguard.utils.schema import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class _TrackedObject:
    """单个物体的跟踪状态"""

    # 跟踪 key: "{class_name}:{grid_x}:{grid_y}"
    key: str
    # 首次连续出现的时间戳（time.monotonic）
    first_seen: float
    # 最后一次出现的时间戳
    last_seen: float
    # 历史最高置信度
    peak_confidence: float
    # 最新的检测框
    bbox: BoundingBox


class ObjectPersistenceTracker:
    """
    目标物体持续出现跟踪器

    跟踪策略：
        1. 以 (class_name, grid_x, grid_y) 为 key 跟踪每个物体
        2. 每帧将当前检测到的物体与跟踪表对比：
           - 已存在：更新 last_seen 和峰值置信度
           - 不存在：新增跟踪记录，记录 first_seen = now
        3. 物体持续出现时间 >= persistence_sec 时，标记为"已持续"
        4. 物体在 max_missed_sec 内未出现，移除跟踪记录

    Thread-safety:
        当前仅在 Pipeline 主循环中单线程调用，无需加锁。
    """

    def __init__(
        self,
        persistence_sec: float = 1.0,
        max_missed_sec: float = 0.5,
        position_grid_size: int = 50,
    ) -> None:
        """
        Args:
            persistence_sec: 物体需持续出现的最小秒数，达到此值才允许报警
            max_missed_sec: 物体消失超过此秒数则移除跟踪记录
            position_grid_size: 位置网格量化粒度（像素），用于生成跟踪 key
        """
        self._persistence_sec = persistence_sec
        self._max_missed_sec = max_missed_sec
        self._position_grid_size = position_grid_size
        self._tracked: dict[str, _TrackedObject] = {}

    def update(
        self, detections: list[BoundingBox], now: float | None = None
    ) -> set[str]:
        """
        更新跟踪状态，返回已持续超过 persistence_sec 的物体 key 集合。

        Args:
            detections: 当前帧检测到的物体列表
            now: 当前时间戳（time.monotonic），None 时自动获取

        Returns:
            已持续超过 persistence_sec 的物体 key 集合（set[str]）。
            空集合表示所有检测均未达到持续时间阈值。
        """
        if now is None:
            now = time.monotonic()

        if not detections:
            # 当前帧无检测：清理超时未出现的跟踪记录
            self._cleanup_expired(now)
            return set()

        # 当前帧检测到的 key 集合
        current_keys: set[str] = set()
        persistent_keys: set[str] = set()

        for det in detections:
            key = self.make_key(det)
            current_keys.add(key)

            if key in self._tracked:
                # 已有跟踪记录：更新
                obj = self._tracked[key]
                obj.last_seen = now
                if det.confidence > obj.peak_confidence:
                    obj.peak_confidence = det.confidence
                    obj.bbox = det

                # 检查是否已持续足够长时间
                elapsed = now - obj.first_seen
                if elapsed >= self._persistence_sec:
                    logger.debug(
                        "持续跟踪: key=%s, 已持续 %.2fs (阈值 %.1fs), 放行",
                        key, elapsed, self._persistence_sec,
                    )
                    persistent_keys.add(key)
                else:
                    logger.debug(
                        "持续跟踪: key=%s, 已持续 %.2fs (阈值 %.1fs), 等待中",
                        key, elapsed, self._persistence_sec,
                    )
            else:
                # 新物体：开始跟踪
                self._tracked[key] = _TrackedObject(
                    key=key,
                    first_seen=now,
                    last_seen=now,
                    peak_confidence=det.confidence,
                    bbox=det,
                )
                logger.debug(
                    "持续跟踪: 新物体 key=%s, 开始计时 (%.1fs)",
                    key, self._persistence_sec,
                )

        # 清理当前帧未出现的过期跟踪记录
        expired_keys = [
            key
            for key, obj in self._tracked.items()
            if key not in current_keys and now - obj.last_seen > self._max_missed_sec
        ]
        for key in expired_keys:
            obj = self._tracked.pop(key)
            elapsed = now - obj.first_seen
            logger.debug(
                "持续跟踪: key=%s 已消失(持续%.2fs), 移除跟踪",
                key, elapsed,
            )

        return persistent_keys

    def reset(self) -> None:
        """清空所有跟踪状态（用于重置场景）"""
        self._tracked.clear()
        logger.info("持续跟踪器: 已重置")

    @property
    def tracked_count(self) -> int:
        """当前跟踪的物体数量"""
        return len(self._tracked)

    def _cleanup_expired(self, now: float) -> None:
        """清理超时未出现的跟踪记录"""
        expired = [
            key
            for key, obj in self._tracked.items()
            if now - obj.last_seen > self._max_missed_sec
        ]
        for key in expired:
            obj = self._tracked.pop(key)
            elapsed = now - obj.first_seen
            logger.debug(
                "持续跟踪: key=%s 超时(持续%.2fs), 移除跟踪",
                key, elapsed,
            )

    def make_key(self, bbox: BoundingBox) -> str:
        """
        生成跟踪 key，与 AlertDeduplicator._make_key 保持一致。

        公开方法，供 Pipeline 在过滤告警时调用以判断检测结果是否已持续。
        """
        cx = (bbox.x1 + bbox.x2) // 2
        cy = (bbox.y1 + bbox.y2) // 2
        gx = cx // self._position_grid_size
        gy = cy // self._position_grid_size
        return f"{bbox.class_name}:{gx}:{gy}"

