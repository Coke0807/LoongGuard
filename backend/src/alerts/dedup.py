"""
告警去重模块

设计动机：
    同一危险物品在摄像头画面中持续存在时，每帧都会触发检测。
    如果不去重，一把剪刀在画面中停留 10 秒（300 帧）会产生 300 条告警，
    导致告警风暴，管理端无法处理。

去重策略：
    以 (物体类别, 位置网格坐标) 作为去重 key。
    在 time_window_sec 内相同 key 的告警只触发一次。
    网格量化容忍位置抖动（物体轻微移动不视为新告警）。

环境变量覆盖（通过 config/settings.py DedupConfig）：
    LG_DEDUP_ENABLED=true
    LG_DEDUP_TIME_WINDOW_SEC=30
    LG_DEDUP_POSITION_GRID_SIZE=50
"""

import time
import logging

from src.utils.schema import AlertLog

logger = logging.getLogger(__name__)


class AlertDeduplicator:
    """基于 (类别, 网格位置) 的告警去重器"""

    def __init__(self, config):
        """
        Args:
            config: DedupConfig 实例，需具备以下字段：
                - enabled (bool): 是否启用去重
                - time_window_sec (float): 去重时间窗口（秒）
                - position_grid_size (int): 位置网格量化粒度（像素）
                - max_tracked_objects (int): 跟踪表容量上限
        """
        self._config = config
        self._recent: dict[str, float] = {}  # dedup_key -> last_trigger_timestamp

    def should_alert(self, alert: AlertLog) -> bool:
        """
        判断是否应触发此告警。

        Returns:
            True: 新告警，应触发（同时注册到跟踪表）
            False: 重复告警，应抑制
        """
        if not self._config.enabled:
            return True

        key = self._make_key(alert)
        now = time.time()

        if key not in self._recent:
            # 首次出现，注册并放行
            self._recent[key] = now
            logger.debug("去重: 新告警 key=%s, 放行", key)
            return True

        elapsed = now - self._recent[key]
        if elapsed > self._config.time_window_sec:
            # 超过时间窗口，重新触发
            self._recent[key] = now
            logger.debug("去重: key=%s 超过窗口(%.1fs), 重新触发", key, elapsed)
            return True

        # 时间窗口内重复，抑制
        logger.debug("去重: key=%s 在窗口内(%.1fs), 抑制", key, elapsed)
        return False

    def cleanup_expired(self) -> int:
        """清理过期的跟踪条目。返回清理数量。"""
        now = time.time()
        expired = [
            key for key, ts in self._recent.items()
            if now - ts > self._config.time_window_sec
        ]
        for key in expired:
            del self._recent[key]

        # 如果跟踪表过大，清理最旧的条目
        if len(self._recent) > self._config.max_tracked_objects:
            sorted_keys = sorted(self._recent, key=self._recent.get)
            to_remove = sorted_keys[:len(self._recent) - self._config.max_tracked_objects]
            for key in to_remove:
                del self._recent[key]
                expired.append(key)

        if expired:
            logger.info("去重: 清理 %d 条过期/溢出条目", len(expired))
        return len(expired)

    def reset(self) -> None:
        """清空所有跟踪状态"""
        self._recent.clear()

    def _make_key(self, alert: AlertLog) -> str:
        """
        生成去重 key。

        规则：
            1. 取 alert.detections 中置信度最高的检测框
            2. 计算中心点 (cx, cy)
            3. 量化为网格坐标 (gx, gy) = (cx // grid_size, cy // grid_size)
            4. key = "{class_name}:{gx}:{gy}"
            5. 对于无检测框的告警（如 PRONE_SLEEP），key = "{alert_type}:global"
        """
        if not alert.detections:
            return f"{alert.alert_type.value}:global"

        best = max(alert.detections, key=lambda d: d.confidence)
        cx = (best.x1 + best.x2) // 2
        cy = (best.y1 + best.y2) // 2
        gx = cx // self._config.position_grid_size
        gy = cy // self._config.position_grid_size
        return f"{best.class_name}:{gx}:{gy}"

    @property
    def tracked_count(self) -> int:
        """当前跟踪的去重 key 数量"""
        return len(self._recent)
