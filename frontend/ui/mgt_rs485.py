import contextlib
import logging
import os
import tempfile
import time

import minimalmodbus
import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)

# ===================== Loongnix 配置区（按实际修改） =====================
# Linux串口设备，常见：/dev/ttyUSB0(CH340/RS485)、/dev/ttyACM0
COM_NAME = "/dev/ttyUSB0"
SLAVE_ADDR = 1           # 传感器从站地址
BAUD = 9600
REGISTER_START = 0       # 温湿度起始寄存器
REGISTER_NUM = 2         # 读取2个寄存器：温度、湿度
# =================================================================

def get_all_serial_ports():
    """获取Loongnix系统所有可用串口列表"""
    ports_list = serial.tools.list_ports.comports()
    device_names = [port.device for port in ports_list]
    return device_names

def check_serial_permission(port_path):
    """检测当前用户是否拥有串口读写权限"""
    try:
        # 尝试只读打开判断权限
        ser = serial.Serial(port_path, timeout=0.1)
        ser.close()
        return True
    except PermissionError:
        logger.error("【权限错误】无串口 %s 访问权限！", port_path)
        logger.error("  执行命令授权：sudo usermod -aG dialout $USER，重新登录后生效")
        return False
    except Exception:
        return True

def check_serial_exists(target_com):
    """判断目标串口是否存在"""
    all_ports = get_all_serial_ports()
    if len(all_ports) == 0:
        logger.error("【错误】本机未检测到任何串口设备，请插入USB转RS485模块！")
        logger.error("当前扫描串口列表：%s", all_ports)
        return False
    if target_com not in all_ports:
        logger.error("【错误】未找到串口 %s，当前可用串口：%s", target_com, all_ports)
        return False
    # 检查串口权限
    if not check_serial_permission(target_com):
        return False
    return True

@contextlib.contextmanager
def _serial_lock():
    """
    跨进程串口互斥锁（文件锁实现）

    设计动机：
        backend（语音"查询温湿度"）与 frontend（桌面端传感器线程）是两个独立
        进程，都可能打开同一 RS485 半双工总线。RS485 同一时刻只允许一个主站
        访问，否则帧数据互相破坏（NoResponse/乱码）。本锁用文件锁实现进程间
        互斥：Linux 用 fcntl.flock，Windows 用 msvcrt.locking（开发机兼容）。
        锁随文件句柄关闭自动释放，进程异常退出不会死锁。
    """
    lock_name = "lg_rs485_{}.lock".format(os.path.basename(COM_NAME))
    lock_path = os.path.join(tempfile.gettempdir(), lock_name)
    lock_file = open(lock_path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            # 确保文件至少 1 字节，否则 LockFile 无法锁定区域
            # 用 fstat 判断而非读取内容：Windows LockFile 排斥其他句柄读取，
            # 持锁方未释放时 read 会抛 PermissionError
            if os.fstat(lock_file.fileno()).st_size == 0:
                lock_file.write("\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            # 释放失败不掩盖业务异常；文件锁随句柄关闭自动清理
            pass
        lock_file.close()


def read_temp_hum():
    # 跨进程互斥：backend 与 frontend 两个进程可能同时打开同一 RS485 串口，
    # 用文件锁保证同一时刻只有一个读取者（见 _serial_lock 设计动机）
    with _serial_lock():
        # 第一步：检测串口硬件是否存在+权限校验
        if not check_serial_exists(COM_NAME):
            return None, None

        # 第二步：初始化Modbus RTU设备
        try:
            instrument = minimalmodbus.Instrument(COM_NAME, SLAVE_ADDR)
            instrument.serial.baudrate = BAUD
            instrument.serial.bytesize = 8
            instrument.serial.parity = serial.PARITY_NONE
            instrument.serial.stopbits = 1
            instrument.serial.timeout = 0.8
            instrument.mode = minimalmodbus.MODE_RTU
        except Exception as e:
            logger.error("【错误】串口打开失败，RS485转换器离线/被占用：%s", e)
            return None, None

        # 第三步：读取传感器寄存器
        try:
            regs = instrument.read_registers(REGISTER_START, REGISTER_NUM)
            temp = round(regs[0] / 10.0, 1)
            hum = round(regs[1] / 10.0, 1)
            return temp, hum
        except minimalmodbus.NoResponseError:
            logger.error("【错误】串口已识别，但温湿度传感器无应答：")
            logger.error("      1. A/B信号线接反；2. 传感器未外接供电；3. 从站地址/波特率不匹配")
            return None, None
        except Exception as e:
            logger.error("【读取异常】%s", e)
            return None, None

if __name__ == "__main__":
    # 独立运行脚本：控制台 INFO 级别输出
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=== LoongArch Loongnix25 RS485温湿度采集程序 ===")
    logger.info("目标串口：%s  从站地址：%s 波特率：%s\n", COM_NAME, SLAVE_ADDR, BAUD)

    while True:
        temperature, humidity = read_temp_hum()
        if temperature is not None and humidity is not None:
            logger.info("正常采集 | 温度：%s ℃  湿度：%s %%RH", temperature, humidity)
        logger.info("-" * 50)
        time.sleep(1)