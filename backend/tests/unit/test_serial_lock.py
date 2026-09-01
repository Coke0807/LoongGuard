"""
跨进程串口互斥锁单元测试

验证 frontend/ui/mgt_rs485.py 的 _serial_lock 文件锁语义：

    设计动机：
        backend（语音"查询温湿度"）与 frontend（桌面端传感器线程）是两个独立
        进程，都可能打开同一 RS485 半双工总线。RS485 同一时刻只允许一个主站
        访问，否则帧数据互相破坏。本测试验证锁的进程间互斥语义（临界区互斥 +
        持锁结束后自动释放），不依赖真实串口硬件。

运行：
    cd backend && python -m pytest tests/unit/test_serial_lock.py -q
"""

import threading
import time
from pathlib import Path

import pytest

# 与 backend 生产代码 _load_sensor_reader 一致：按文件路径加载前端模块
MODULE_PATH = Path(__file__).resolve().parents[3] / "frontend" / "ui" / "mgt_rs485.py"


@pytest.fixture(scope="module")
def rs485_module():
    """动态加载 frontend/ui/mgt_rs485.py（backend venv 需已装 minimalmodbus/pyserial）"""
    import importlib.util

    assert MODULE_PATH.exists(), f"传感器模块不存在: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("lg_test_mgt_rs485", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_serial_lock_mutual_exclusion(rs485_module) -> None:
    """持锁期间其他线程应被阻塞，直到持有者释放"""
    lock = rs485_module._serial_lock
    entered: list[str] = []
    release = threading.Event()

    def holder() -> None:
        with lock():
            entered.append("holder")
            release.wait(timeout=5)

    def contender() -> None:
        with lock():
            entered.append("contender")

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=contender)
    t1.start()

    # 等待 holder 进入临界区
    deadline = time.time() + 3
    while not entered and time.time() < deadline:
        time.sleep(0.01)
    assert entered == ["holder"], "holder 应已进入临界区"

    # contender 启动后应被锁阻塞（holder 释放前不得进入）
    t2.start()
    time.sleep(0.3)
    assert entered == ["holder"], "contender 在锁释放前不应进入临界区"

    # 释放锁，contender 应随后进入
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive(), "两个线程都应正常结束"
    assert entered == ["holder", "contender"], "锁释放后 contender 应能进入"


def test_serial_lock_auto_release(rs485_module) -> None:
    """持锁线程结束后锁自动释放，后续获取不应阻塞"""
    lock = rs485_module._serial_lock
    with lock():
        pass
    start = time.time()
    with lock():
        elapsed = time.time() - start
    assert elapsed < 1.0, "持锁结束后锁应立即可重新获取"


def test_read_temp_hum_without_hardware(rs485_module) -> None:
    """无串口硬件/权限时 read_temp_hum 走锁路径并安全降级为 (None, None)"""
    temp, hum = rs485_module.read_temp_hum()
    assert temp is None and hum is None
