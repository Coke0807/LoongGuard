"""告警去重模块单元测试"""

import time
from dataclasses import dataclass

import pytest

from src.alerts.dedup import AlertDeduplicator
from src.utils.schema import AlertLog, AlertType, AlertSeverity, BoundingBox


@dataclass
class MockDedupConfig:
    enabled: bool = True
    time_window_sec: float = 5.0
    position_grid_size: int = 50
    max_tracked_objects: int = 1000


def _make_alert(
    class_name="scissors",
    x1=100,
    y1=100,
    x2=200,
    y2=200,
    alert_type=AlertType.DANGEROUS_OBJECT,
) -> AlertLog:
    """构造测试用告警"""
    detections = []
    if class_name:
        detections.append(
            BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=0.9,
                class_id=0,
                class_name=class_name,
            )
        )
    return AlertLog(
        alert_id="test123",
        timestamp="2026-06-25T10:00:00Z",
        alert_type=alert_type,
        severity=AlertSeverity.HIGH,
        detections=detections,
        description="test",
    )


class TestAlertDeduplicator:

    def test_first_alert_passes(self):
        """首次告警应通过"""
        config = MockDedupConfig()
        dedup = AlertDeduplicator(config)
        alert = _make_alert()
        assert dedup.should_alert(alert) is True

    def test_duplicate_within_window_suppressed(self):
        """时间窗口内的重复告警应被抑制"""
        config = MockDedupConfig(time_window_sec=5.0)
        dedup = AlertDeduplicator(config)
        alert = _make_alert()

        assert dedup.should_alert(alert) is True
        # 立即再次触发，应在窗口内被抑制
        assert dedup.should_alert(alert) is False
        assert dedup.should_alert(alert) is False

    def test_same_key_outside_window_passes(self):
        """超过时间窗口的相同 key 应重新触发"""
        config = MockDedupConfig(time_window_sec=0.1)
        dedup = AlertDeduplicator(config)
        alert = _make_alert()

        assert dedup.should_alert(alert) is True
        assert dedup.should_alert(alert) is False

        # 等待超过时间窗口
        time.sleep(0.15)
        assert dedup.should_alert(alert) is True

    def test_different_class_not_deduped(self):
        """不同类别的物体不应去重"""
        config = MockDedupConfig()
        dedup = AlertDeduplicator(config)

        alert_scissors = _make_alert(class_name="scissors")
        alert_knife = _make_alert(class_name="knife")

        assert dedup.should_alert(alert_scissors) is True
        assert dedup.should_alert(alert_knife) is True
        # 重复的各自仍应被抑制
        assert dedup.should_alert(alert_scissors) is False
        assert dedup.should_alert(alert_knife) is False

    def test_different_position_not_deduped(self):
        """相同类别但不同网格位置不应去重"""
        config = MockDedupConfig(position_grid_size=50)
        dedup = AlertDeduplicator(config)

        # center=(150,150) -> grid=(3,3)
        alert1 = _make_alert(x1=100, y1=100, x2=200, y2=200)
        # center=(350,350) -> grid=(7,7)
        alert2 = _make_alert(x1=300, y1=300, x2=400, y2=400)

        assert dedup.should_alert(alert1) is True
        assert dedup.should_alert(alert2) is True

    def test_disabled_passes_all(self):
        """去重禁用时所有告警应通过"""
        config = MockDedupConfig(enabled=False)
        dedup = AlertDeduplicator(config)
        alert = _make_alert()

        # 即使完全相同的告警也应全部通过
        assert dedup.should_alert(alert) is True
        assert dedup.should_alert(alert) is True
        assert dedup.should_alert(alert) is True

    def test_no_detections_uses_global_key(self):
        """无检测框的告警（如 PRONE_SLEEP）使用 global key"""
        config = MockDedupConfig()
        dedup = AlertDeduplicator(config)

        alert = _make_alert(class_name=None, alert_type=AlertType.PRONE_SLEEP)
        # 不带 class_name 时 _make_alert 生成空 detections
        assert dedup.should_alert(alert) is True
        assert dedup.should_alert(alert) is False

        # 不同 alert_type 的无检测框告警使用不同 key
        alert_env = _make_alert(class_name=None, alert_type=AlertType.ENVIRONMENT)
        assert dedup.should_alert(alert_env) is True

    def test_cleanup_expired_removes_stale(self):
        """过期条目应被清理"""
        config = MockDedupConfig(time_window_sec=0.1)
        dedup = AlertDeduplicator(config)

        alert = _make_alert()
        dedup.should_alert(alert)
        assert dedup.tracked_count == 1

        time.sleep(0.15)
        removed = dedup.cleanup_expired()
        assert removed == 1
        assert dedup.tracked_count == 0

    def test_cleanup_respects_max_tracked(self):
        """跟踪表超过 max_tracked_objects 时应清理最旧的"""
        config = MockDedupConfig(
            time_window_sec=999, max_tracked_objects=3
        )
        dedup = AlertDeduplicator(config)

        # 填入 3 条不同位置的告警
        for i in range(3):
            alert = _make_alert(x1=i * 200, y1=i * 200, x2=i * 200 + 50, y2=i * 200 + 50)
            dedup.should_alert(alert)

        assert dedup.tracked_count == 3

        # 再添加 2 条，使总数达到 5，超出 max=3
        time.sleep(0.01)  # 确保时间戳有差异
        for i in range(3, 5):
            alert = _make_alert(x1=i * 200, y1=i * 200, x2=i * 200 + 50, y2=i * 200 + 50)
            dedup.should_alert(alert)

        assert dedup.tracked_count == 5

        # cleanup 应移除最旧的 2 条以回到 max_tracked_objects
        removed = dedup.cleanup_expired()
        assert removed == 2
        assert dedup.tracked_count == 3

    def test_reset_clears_all(self):
        """reset() 应清空所有跟踪状态"""
        config = MockDedupConfig()
        dedup = AlertDeduplicator(config)

        for i in range(5):
            alert = _make_alert(x1=i * 100, y1=i * 100, x2=i * 100 + 50, y2=i * 100 + 50)
            dedup.should_alert(alert)

        assert dedup.tracked_count == 5
        dedup.reset()
        assert dedup.tracked_count == 0

        # reset 后相同告警应重新放行
        alert = _make_alert()
        assert dedup.should_alert(alert) is True

    def test_tracked_count(self):
        """tracked_count 属性应正确反映跟踪数量"""
        config = MockDedupConfig()
        dedup = AlertDeduplicator(config)

        assert dedup.tracked_count == 0

        alert1 = _make_alert(class_name="scissors")
        dedup.should_alert(alert1)
        assert dedup.tracked_count == 1

        alert2 = _make_alert(class_name="knife")
        dedup.should_alert(alert2)
        assert dedup.tracked_count == 2

        # 重复告警不增加跟踪数
        dedup.should_alert(alert1)
        assert dedup.tracked_count == 2

    def test_nearby_positions_same_grid(self):
        """同一网格内的不同位置应被视为相同"""
        config = MockDedupConfig(position_grid_size=100)
        dedup = AlertDeduplicator(config)
        # 两个告警在同一个 100x100 网格内
        alert1 = _make_alert(x1=10, y1=10, x2=50, y2=50)   # center=(30,30), grid=(0,0)
        alert2 = _make_alert(x1=60, y1=60, x2=90, y2=90)   # center=(75,75), grid=(0,0)
        assert dedup.should_alert(alert1) is True
        assert dedup.should_alert(alert2) is False  # same grid cell

    def test_multiple_detections_uses_highest_confidence(self):
        """多个检测框时应使用置信度最高的框生成 key"""
        config = MockDedupConfig(position_grid_size=50)
        dedup = AlertDeduplicator(config)

        # 构造含多个检测框的告警
        detections = [
            BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.5, class_id=0, class_name="scissors"),
            BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.95, class_id=0, class_name="scissors"),
        ]
        alert = AlertLog(
            alert_id="multi",
            timestamp="2026-06-25T10:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT,
            severity=AlertSeverity.HIGH,
            detections=detections,
            description="multi test",
        )

        assert dedup.should_alert(alert) is True

        # 另一个告警，仅包含高置信度框附近的位置
        alert2 = _make_alert(x1=110, y1=110, x2=190, y2=190)  # center=(150,150)
        # 两个都应产生 key "scissors:3:3" (grid_size=50, center 150//50=3)
        assert dedup.should_alert(alert2) is False

    def test_cleanup_no_expired_returns_zero(self):
        """无过期条目时 cleanup 应返回 0"""
        config = MockDedupConfig(time_window_sec=999)
        dedup = AlertDeduplicator(config)

        dedup.should_alert(_make_alert())
        assert dedup.cleanup_expired() == 0
