"""
GPIO 告警模块单元测试

覆盖：
    - 告警触发基本功能（异步）
    - 冷却机制（防止重复触发）
    - 不同严重等级的蜂鸣模式选择
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from config.settings import AlarmConfig
from loongguard.alarm.gpio_trigger import _BUZZER_PATTERNS, GPIOAlarmTrigger
from loongguard.utils.schema import AlertSeverity


class TestGPIOAlarmTrigger:
    """GPIO 告警触控层测试套件"""

    @pytest.fixture
    def alarm(self) -> GPIOAlarmTrigger:
        config = AlarmConfig(cooldown_sec=1.0)
        trigger = GPIOAlarmTrigger(config)
        # 跳过真实 GPIO 初始化
        return trigger

    @pytest.mark.asyncio
    async def test_trigger_high_severity(self, alarm: GPIOAlarmTrigger) -> None:
        """HIGH 严重等级告警应使用急促三响模式"""
        with patch.object(alarm, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            await alarm.trigger(AlertSeverity.HIGH)
            mock_exec.assert_called_once()
            pattern = mock_exec.call_args[0][0]
            assert pattern == _BUZZER_PATTERNS[AlertSeverity.HIGH]

    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicate(self, alarm: GPIOAlarmTrigger) -> None:
        """冷却期内同一等级的告警不应重复触发"""
        with patch.object(alarm, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            await alarm.trigger(AlertSeverity.MEDIUM)
            assert mock_exec.call_count == 1

            # 冷却期内再次触发
            await alarm.trigger(AlertSeverity.MEDIUM)
            assert mock_exec.call_count == 1  # 不应增加

    @pytest.mark.asyncio
    async def test_different_severity_bypasses_cooldown(self, alarm: GPIOAlarmTrigger) -> None:
        """不同严重等级的告警不受彼此冷却期影响"""
        with patch.object(alarm, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            await alarm.trigger(AlertSeverity.LOW)
            assert mock_exec.call_count == 1

            # 不同等级应能触发
            await alarm.trigger(AlertSeverity.HIGH)
            assert mock_exec.call_count == 2

    def test_cleanup_resets_state(self, alarm: GPIOAlarmTrigger) -> None:
        """cleanup 应关闭所有 GPIO 输出"""
        with patch.object(alarm, "_buzzer_off") as mock_buzz, \
             patch.object(alarm, "_led_off") as mock_led:
            alarm.cleanup()
            mock_buzz.assert_called()
            mock_led.assert_called()
