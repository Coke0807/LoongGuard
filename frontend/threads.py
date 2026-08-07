import cv2
import time
import random
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from utils import write_log

from voiceplayer import VoicePlayer

# ===================== 视频录制线程 =====================
class VideoRecordThread(QThread):
    def __init__(self, save_path, width, height, fps=20):
        super().__init__()
        self.save_path = save_path
        self.w = width
        self.h = height
        self.fps = fps
        self.running = False
        self.writer = None
        self.frame_queue = []
        self.write_count = 0
        self.audio_proc = None
        self.audio_tmp = ""
        self.record_origin_path = ""

    def draw_frame_time_text(self, frame):
        from datetime import datetime
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, now_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, now_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
        return frame

    def run(self):
        self.running = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, (self.w, self.h))
        if not self.writer.isOpened():
            write_log("ERROR", f"编码器打开失败 {self.save_path}")
            self.running = False
            return
        write_log("RECORD", f"录制启动成功 {self.save_path} 分辨率{self.w}*{self.h}")

        while self.running:
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
                drawed = self.draw_frame_time_text(frame)
                self.writer.write(drawed)
                self.write_count += 1
                # 每50帧打印一次日志，确认真的写入画面
                if self.write_count % 50 == 0:
                    write_log("RECORD", f"已写入{self.write_count}帧画面")
            time.sleep(0.001)

        # 退出前把剩余全部帧写完
        while self.frame_queue:
            frame = self.frame_queue.pop(0)
            drawed = self.draw_frame_time_text(frame)
            self.writer.write(drawed)
            self.write_count += 1

        if self.writer:
            self.writer.release()
            self.writer = None
        write_log("RECORD", f"录制结束，总共写入{self.write_count}帧")

    def push_frame(self, bgr_frame):
        if len(self.frame_queue) < 120:
            self.frame_queue.append(bgr_frame)

    def stop_record(self):
        self.running = False

# ===================== AI后台模拟线程 =====================
class AiMonitorThread(QThread):
    signal_warn = pyqtSignal(str, str)
    signal_temp_humi_gas = pyqtSignal(float, float, float)
    signal_device_online = pyqtSignal(int)
    signal_sensor_status = pyqtSignal(bool)

    def run(self):
        write_log("INFO", "AI环境模拟线程启动")
        while True:
            temp = 0
            humi = 0
            #gas = round(random.uniform(0, 50), 1)
            gas = 0
            self.signal_temp_humi_gas.emit(temp, humi, gas)
            self.signal_device_online.emit(random.randint(4, 7))
            sensor_ok = random.random() > 0.08
            self.signal_sensor_status.emit(sensor_ok)
            warn_prob = random.random()
            if warn_prob > 0.93:
                self.signal_warn.emit("危险品预警", "幼儿手持别针行走过道")
            elif warn_prob > 0.86:
                self.signal_warn.emit("奔跑预警", "幼儿在过道快速追逐打闹")
            elif warn_prob > 0.80:
                self.signal_warn.emit("环境异常", "过道湿滑，存在滑倒风险")
            time.sleep(3)

# ===================== 摄像头读取线程 =====================
# CameraThread 增加获取真实宽高、真实fps
class CameraThread(QThread):
    ui_frame_signal = pyqtSignal(QImage)
    yolo_frame_signal = pyqtSignal(np.ndarray)
    cam_status_signal = pyqtSignal(bool)
    raw_frame_signal = pyqtSignal(np.ndarray)
    # 新增信号：输出真实分辨率、真实帧率给主窗口
    cam_info_signal = pyqtSignal(int, int, float)

    def __init__(self, dev_id=0):
        super().__init__()
        self.dev_id = dev_id
        self.running = False
        self.latest_bgr = None
        self.cap = None
        self.retry_delay = 0.5
        self.fps = 20.0
        self.width = 1280
        self.height = 800

        self.voice_player = VoicePlayer()

    def _release_cap(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            write_log("CAM", "摄像头捕获器已释放")

    def draw_qimg_time(self, rgb_arr):
        from datetime import datetime
        h, w, c = rgb_arr.shape
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temp_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        cv2.putText(temp_bgr, now_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(temp_bgr, now_text, (12,32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
        return cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)

    def run(self):
        self.running = True
        write_log("INFO", "摄像头采集线程启动")
        flag=True
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._release_cap()
                # 优先请求MJPG压缩流，解决龙芯YUYV帧率低
                self.cap = cv2.VideoCapture(self.dev_id, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                if not self.cap.isOpened():
                    write_log("CAM", "摄像头未检测到，等待重连...")
                    if flag:
                        flag = False
                        # 播报摄像头异常
                        time.sleep(2)
                        self.voice_player.voice_camera_error()

                    self.cam_status_signal.emit(False)
                    time.sleep(self.retry_delay)
                    continue

                # 设置状态
                if flag==False:
                    self.voice_player.voice_camera_ok()

                flag=True

                # 读取硬件真实生效的分辨率、帧率
                real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                real_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if real_fps <= 0:
                    real_fps = 20.0
                self.width = real_w
                self.height = real_h
                self.fps = real_fps
                # 发送给主窗口，录制时使用真实fps
                self.cam_info_signal.emit(real_w, real_h, real_fps)
                write_log("CAM", f"摄像头已连接 分辨率{real_w}*{real_h} 真实帧率{real_fps}")
                self.cam_status_signal.emit(True)

            ret, bgr_frame = self.cap.read()
            if not ret:
                write_log("ERROR", "摄像头读取帧失败，断开重连")
                self.cam_status_signal.emit(False)
                self._release_cap()
                time.sleep(self.retry_delay)
                continue

            self.cam_status_signal.emit(True)
            self.latest_bgr = bgr_frame.copy()
            self.raw_frame_signal.emit(bgr_frame)
            self.yolo_frame_signal.emit(bgr_frame)

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rgb_with_time = self.draw_qimg_time(rgb_frame)
            h, w, c = rgb_with_time.shape
            qimg = QImage(rgb_with_time.data, w, h, w * c, QImage.Format.Format_RGB888)
            self.ui_frame_signal.emit(qimg)

        self._release_cap()
        self.cam_status_signal.emit(False)
        write_log("INFO", "摄像头采集线程正常退出")

    def stop(self):
        self.running = False

    def get_latest_frame(self):
        return self.latest_bgr
