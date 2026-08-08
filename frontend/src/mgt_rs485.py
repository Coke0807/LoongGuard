import serial
import serial.tools.list_ports
import minimalmodbus
import time
import os

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
        print(f"【权限错误】无串口 {port_path} 访问权限！")
        print("  执行命令授权：sudo usermod -aG dialout $USER，重新登录后生效")
        return False
    except Exception:
        return True

def check_serial_exists(target_com):
    """判断目标串口是否存在"""
    all_ports = get_all_serial_ports()
    if len(all_ports) == 0:
        print("【错误】本机未检测到任何串口设备，请插入USB转RS485模块！")
        print(f"当前扫描串口列表：{all_ports}")
        return False
    if target_com not in all_ports:
        print(f"【错误】未找到串口 {target_com}，当前可用串口：{all_ports}")
        return False
    # 检查串口权限
    if not check_serial_permission(target_com):
        return False
    return True

def read_temp_hum():
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
        print(f"【错误】串口打开失败，RS485转换器离线/被占用：{str(e)}")
        return None, None

    # 第三步：读取传感器寄存器
    try:
        regs = instrument.read_registers(REGISTER_START, REGISTER_NUM)
        temp = round(regs[0] / 10.0, 1)
        hum = round(regs[1] / 10.0, 1)
        return temp, hum
    except minimalmodbus.NoResponseError:
        print("【错误】串口已识别，但温湿度传感器无应答：")
        print("      1. A/B信号线接反；2. 传感器未外接供电；3. 从站地址/波特率不匹配")
        return None, None
    except Exception as e:
        print(f"【读取异常】{str(e)}")
        return None, None

if __name__ == "__main__":
    print("=== LoongArch Loongnix25 RS485温湿度采集程序 ===")
    print(f"目标串口：{COM_NAME}  从站地址：{SLAVE_ADDR} 波特率：{BAUD}\n")

    while True:
        temperature, humidity = read_temp_hum()
        if temperature is not None and humidity is not None:
            print(f"正常采集 | 温度：{temperature} ℃  湿度：{humidity} %RH")
        print("-" * 50)
        time.sleep(1)