import serial
import minimalmodbus

# 修改为你的实际设备地址
sensor = minimalmodbus.Instrument('/dev/ttyUSB0', 2)
sensor.serial.baudrate = 9600
sensor.serial.bytesize = 8
sensor.serial.parity = serial.PARITY_NONE
sensor.serial.stopbits = 1
sensor.debug = True  # 打印Modbus原始收发报文

try:
    temp = sensor.read_register(0, 1, signed=True)
    humi = sensor.read_register(1, 1, signed=True)
    print(f"温度：{temp/10} ℃，湿度：{humi/10} %RH")
except Exception as err:
    print("通讯失败：", err)
