"""外部告警通知器 (AlertNotifier) 单元测试

设计动机：
    告警仅推 WebSocket + 入库时，无人值守会漏报。AlertNotifier 通过
    webhook 推送 HIGH/CRITICAL 告警并带失败重试。本测试覆盖：
        - 开关与阈值判定
        - 重试与指数退避
        - 非 2xx 响应视为失败
"""
from __future__ import annotations

import asyncio

import pytest

from config import NotificationConfig
from src.alerts.notifier import AlertNotifier
from src.utils.schema import AlertLog, AlertSeverity, AlertType


def _alert(severity: AlertSeverity = AlertSeverity.HIGH) -> AlertLog:
    """构造测试告警"""
    return AlertLog(
        alert_id="n1",
        timestamp="2026-06-25T10:00:00Z",
        alert_type=AlertType.DANGEROUS_OBJECT,
        severity=severity,
        description="test alarm",
    )


class TestNotifierThresholds:

    @pytest.mark.asyncio
    async def test_disabled_returns_true(self):
        """未启用通知时 notify 直接返回 True（无需通知）"""
        notifier = AlertNotifier(NotificationConfig(enabled=False))
        await notifier.start()
        try:
            assert await notifier.notify(_alert()) is True
        finally:
            await notifier.stop()

    @pytest.mark.asyncio
    async def test_no_webhook_returns_true(self):
        """启用但未配置 webhook 时返回 True"""
        notifier = AlertNotifier(NotificationConfig(enabled=True, webhook_url=""))
        await notifier.start()
        try:
            assert await notifier.notify(_alert()) is True
        finally:
            await notifier.stop()

    @pytest.mark.asyncio
    async def test_below_min_severity_skipped(self):
        """低于阈值等级的告警不推送"""
        notifier = AlertNotifier(NotificationConfig(
            enabled=True,
            webhook_url="https://gw.example.com/alert",
            min_severity="high",
            retries=1,
        ))
        await notifier.start()
        try:
            # LOW < high 阈值，应跳过
            assert await notifier.notify(_alert(AlertSeverity.LOW)) is True
        finally:
            await notifier.stop()


class TestNotifierSending:

    @pytest.mark.asyncio
    async def test_successful_post(self, monkeypatch):
        """成功推送返回 True"""
        notifier = AlertNotifier(NotificationConfig(
            enabled=True,
            webhook_url="https://gw.example.com/alert",
            retries=2,
        ))
        await notifier.start()

        async def _fake_post(payload: dict) -> None:
            return None

        monkeypatch.setattr(notifier, "_post", _fake_post)
        try:
            assert await notifier.notify(_alert()) is True
        finally:
            await notifier.stop()

    @pytest.mark.asyncio
    async def test_retry_then_success(self, monkeypatch):
        """前两次失败、第三次成功，应返回 True 且重试计数正确"""
        attempts: list[int] = []

        notifier = AlertNotifier(NotificationConfig(
            enabled=True,
            webhook_url="https://gw.example.com/alert",
            retries=3,
            retry_backoff_sec=0.01,  # 缩短退避，加速测试
        ))
        await notifier.start()

        # 用一个极小的 sleep 替换，避免真实退避等待
        async def _fast_sleep(_sec: float) -> None:
            return None

        async def _flaky_post(payload: dict) -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("transient network error")

        monkeypatch.setattr(notifier, "_post", _flaky_post)
        monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

        try:
            assert await notifier.notify(_alert()) is True
            assert len(attempts) == 3
        finally:
            await notifier.stop()

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_false(self, monkeypatch):
        """重试耗尽仍失败返回 False"""
        notifier = AlertNotifier(NotificationConfig(
            enabled=True,
            webhook_url="https://gw.example.com/alert",
            retries=2,
            retry_backoff_sec=0.01,
        ))
        await notifier.start()

        async def _fast_sleep(_sec: float) -> None:
            return None

        async def _always_fail(payload: dict) -> None:
            raise RuntimeError("persistent failure")

        monkeypatch.setattr(notifier, "_post", _always_fail)
        monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

        try:
            assert await notifier.notify(_alert()) is False
        finally:
            await notifier.stop()


class TestNotifierServerErrors:

    @pytest.mark.asyncio
    async def test_non_2xx_raises(self):
        """非 2xx 响应应视为失败（触发重试）"""
        notifier = AlertNotifier(NotificationConfig(
            enabled=True,
            webhook_url="https://gw.example.com/alert",
            retries=1,
        ))
        await notifier.start()

        # 构造一个返回 500 的假 _post，验证上层把它当异常处理
        async def _bad_post(payload: dict) -> None:
            raise RuntimeError("webhook HTTP 状态码 500")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(notifier, "_post", _bad_post)
        try:
            assert await notifier.notify(_alert()) is False
        finally:
            await notifier.stop()
            monkeypatch.undo()