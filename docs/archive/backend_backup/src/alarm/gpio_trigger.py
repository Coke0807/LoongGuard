"""
GPIO 声光告警触控层

设计动机：
    通过 GPIO 控制蜂鸣器和 LED 灯组实现硬件级声光告警。
    支持不同严重等级对应不同告警模式（频率、闪烁模式）。
    内置冷却机制，防止同一类型告警短时间内重复触发。

平台适配：
    - LoongArch：优先使用 gpiod（libgpiod v2 Python 绑定），
      降级到 /sys/class/gpio sysfs 接口
    - Windows / 其他平台：降级为终端 Mock 日志输出
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

from config import AlarmConfig
from src.utils.schema import AlertSeverity

logger = logging.getLogger(__name__)


# 不同严重等级对应的蜂鸣模式
_BUZZER_PATTERNS: dict[AlertSeverity, list[float]] = {
    AlertSeverity.LOW:      [0.1, 0.9],       # 短响一次
    AlertSeverity.MEDIUM:   [0.2, 0.3, 0.2, 0.3],  # 两短响
    AlertSeverity.HIGH:     [0.3, 0.2] * 3,   # 急促三响
    AlertSeverity.CRITICAL: [0.5, 0.1] * 5,   # 连续急促五响
}

# ── 平台检测 ──────────────────────────────────────────────


def _detect_loongarch() -> bool:
    """检测当前 CPU 架构是否为 LoongArch

    检测优先级：
        1. platform.machine() 中包含 "loongarch"（不区分大小写）
        2. /proc/cpuinfo 中包含 "LoongArch"
    """
    machine = platform.machine().lower()
    if "loongarch" in machine:
        return True
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        if "loongarch" in cpuinfo.lower():
            return True
    except OSError:
        pass
    return False


def _detect_gpiod() -> bool:
    """检测系统是否安装了 gpiod Python 绑定（libgpiod v2）"""
    try:
        import gpiod  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_sysfs_gpio() -> bool:
    """检测 sysfs GPIO 接口是否可用（旧内核兼容）"""
    return Path("/sys/class/gpio").exists()


# ── Sysfs GPIO 操作 ───────────────────────────────────────


def _sysfs_export(pin: int) -> bool:
    """导出 GPIO 引脚到 sysfs"""
    try:
        gpio_path = Path(f"/sys/class/gpio/gpio{pin}")
        if gpio_path.exists():
            return True
        Path("/sys/class/gpio/export").write_text(str(pin))
        return True
    except (OSError, IOError):
        return False


def _sysfs_set_dir(pin: int, direction: str = "out") -> bool:
    """设置 GPIO 引脚方向"""
    try:
        Path(f"/sys/class/gpio/gpio{pin}/direction").write_text(direction)
        return True
    except (OSError, IOError):
        return False


def _sysfs_write(pin: int, value: int) -> bool:
    """写入 GPIO 引脚电平"""
    try:
        Path(f"/sys/class/gpio/gpio{pin}/value").write_text(str(value))
        return True
    except (OSError, IOError):
        return False


def _sysfs_unexport(pin: int) -> None:
    """取消导出 GPIO 引脚"""
    try:
        Path("/sys/class/gpio/unexport").write_text(str(pin))
    except (OSError, IOError):
        pass


# ── GPIO 后端封装 ─────────────────────────────────────────


class _GpiodBackend:
    """基于 gpiod（libgpiod v2）的 GPIO 后端"""

    def __init__(self) -> None:
        import gpiod
        self._gpiod = gpiod
        self._chips: dict[int, object] = {}

    def setup(self, chip_id: int, pins: list[int]) -> bool:
        try:
            chip = self._gpiod.Chip(f"/dev/gpiochip{chip_id}")
            self._chips[chip_id] = chip
            for pin in pins:
                chip.get_line(pin).request(consumer="loongguard", type=self._gpiod.LINE_REQ_DIR_OUT)
            return True
        except Exception:
            logger.warning("gpiod backend setup failed for chip %d", chip_id)
            return False

    def write(self, chip_id: int, pin: int, value: int) -> None:
        try:
            chip = self._chips.get(chip_id)
            if chip is not None:
                chip.get_line(pin).set_value(value)
        except Exception:
            pass

    def cleanup(self, pins: list[int]) -> None:
        pass


class _SysfsBackend:
    """基于 sysfs (/sys/class/gpio) 的 GPIO 后端"""

    def __init__(self) -> None:
        self._exported_pins: list[int] = []

    def setup(self, chip_id: int, pins: list[int]) -> bool:
        for pin in pins:
            if not _sysfs_export(pin):
                return False
            if not _sysfs_set_dir(pin, "out"):
                return False
            self._exported_pins.append(pin)
        return True

    def write(self, chip_id: int, pin: int, value: int) -> None:
        _sysfs_write(pin, value)

    def cleanup(self, pins: list[int]) -> None:
        for pin in self._exported_pins:
            _sysfs_write(pin, 0)
            _sysfs_unexport(pin)
        self._exported_pins.clear()


# ── 主类 ──────────────────────────────────────────────────


class GPIOAlarmTrigger:
    """
    GPIO 声光告警触发器

    设计约束：
        - 告警操作同步执行（GPIO 切换是微秒级，不阻塞主循环）
        - 冷却机制：同一 severity 等级在 cooldown_sec 内不重复触发
        - LoongArch：gpiod > sysfs > mock 三级降级
        - Windows / 其他：直接降级为终端日志
    """

    def __init__(self, config: AlarmConfig) -> None:
        self._config = config
        self._last_trigger: dict[str, float] = {}
        self._backend: Optional[object] = None
        self._is_mock: bool = False

    def setup(self) -> None:
        """
        初始化 GPIO 后端

        平台检测链：Windows(mock) <- 其他Linux(mock) <- sysfs <- gpiod <- LoongArch+gpiod
        """
        logger.info("Setting up GPIO alarm (buzzer=%d, led=%d)",
                     self._config.buzzer_pin,
                     self._config.led_pin)

        is_loongarch = _detect_loongarch()

        # 非 Linux 平台直接降级为 Mock
        if sys.platform != "linux":
            self._is_mock = True
            logger.info("GPIO alarm ready (mock=True, non-Linux platform)")
            return

        # LoongArch 优先使用 gpiod 后端
        if is_loongarch and _detect_gpiod():
            backend = _GpiodBackend()
            if backend.setup(self._config.gpio_chip,
                             [self._config.buzzer_pin, self._config.led_pin]):
                self._backend = backend
                logger.info("GPIO alarm ready (backend=gpiod, arch=LoongArch)")
                return

        # 尝试 sysfs GPIO 后端（通用 Linux）
        if _detect_sysfs_gpio():
            backend = _SysfsBackend()
            if backend.setup(self._config.gpio_chip,
                             [self._config.buzzer_pin, self._config.led_pin]):
                self._backend = backend
                logger.info("GPIO alarm ready (backend=sysfs, arch=%s)",
                            "LoongArch" if is_loongarch else platform.machine())
                return

        # 无可用的硬件后端，降级为 Mock
        self._is_mock = True
        logger.warning("GPIO alarm ready (mock=True, no hardware backend available, "
                       "arch=%s)", platform.machine())

    async def trigger(self, severity: AlertSeverity) -> None:
        """
        触发声光告警（异步，不阻塞事件循环）

        Args:
            severity: 告警严重等级，决定蜂鸣模式和 LED 闪烁频率
        """
        key = severity.value
        now = time.monotonic()

        last = self._last_trigger.get(key, 0.0)
        if now - last < self._config.cooldown_sec:
            logger.debug("Alarm cooldown active for %s, skipping", key)
            return

        self._last_trigger[key] = now

        pattern = _BUZZER_PATTERNS.get(severity, [0.2, 0.3])
        logger.warning(
            "[ALARM] severity=%s pattern=%s", severity.value, pattern
        )

        await self._execute_pattern(pattern)

    async def _execute_pattern(self, pattern: list[float]) -> None:
        """执行蜂鸣模式（异步）"""
        for i, duration in enumerate(pattern):
            if i % 2 == 0:
                self._buzzer_on()
                self._led_on()
            else:
                self._buzzer_off()
                self._led_off()
            await asyncio.sleep(duration)

        self._buzzer_off()
        self._led_off()

    # ── GPIO 写入 ──────────────────────────────────────

    def _gpio_write(self, pin: int, value: int) -> None:
        """通过当前后端写入 GPIO 引脚"""
        if self._backend is not None:
            self._backend.write(self._config.gpio_chip, pin, value)

    def _buzzer_on(self) -> None:
        """开启蜂鸣器"""
        if self._is_mock:
            logger.debug("[GPIO MOCK] Buzzer ON  <<BEEP>>")
            return
        self._gpio_write(self._config.buzzer_pin, 1)
        logger.debug("[GPIO] Buzzer ON (pin=%d)", self._config.buzzer_pin)

    def _buzzer_off(self) -> None:
        """关闭蜂鸣器"""
        if self._is_mock:
            logger.debug("[GPIO MOCK] Buzzer OFF")
            return
        self._gpio_write(self._config.buzzer_pin, 0)
        logger.debug("[GPIO] Buzzer OFF (pin=%d)", self._config.buzzer_pin)

    def _led_on(self) -> None:
        """开启告警 LED"""
        if self._is_mock:
            logger.debug("[GPIO MOCK] LED ON  **FLASH**")
            return
        self._gpio_write(self._config.led_pin, 1)
        logger.debug("[GPIO] LED ON (pin=%d)", self._config.led_pin)

    def _led_off(self) -> None:
        """关闭告警 LED"""
        if self._is_mock:
            logger.debug("[GPIO MOCK] LED OFF")
            return
        self._gpio_write(self._config.led_pin, 0)
        logger.debug("[GPIO] LED OFF (pin=%d)", self._config.led_pin)

    def cleanup(self) -> None:
        """释放 GPIO 资源"""
        self._buzzer_off()
        self._led_off()
        if self._backend is not None:
            self._backend.cleanup([self._config.buzzer_pin, self._config.led_pin])
        self._backend = None
        self._is_mock = True
        logger.info("GPIO alarm cleaned up")
