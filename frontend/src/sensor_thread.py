from PyQt6.QtCore import QThread, pyqtSignal
import time
# 导入串口读取函数与串口名称
from src.mgt_rs485 import read_temp_hum, COM_NAME as SENSOR_COM


class SensorReadThread(QThread):
    # 信号：温度、湿度、传感器在线状态
    sensor_data_signal = pyqtSignal(float, float, bool)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            temp, hum = read_temp_hum()
            sensor_online = False
            if temp is not None and hum is not None:
                sensor_online = True
            # 发送数据到UI主线程
            self.sensor_data_signal.emit(temp if temp is not None else 0.0,
                                         hum if hum is not None else 0.0,
                                         sensor_online)
            time.sleep(1.0)

    def stop(self):
        self.running = False
        self.wait()