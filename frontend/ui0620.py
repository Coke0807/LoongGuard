import sys
import math
import time
import random
import os
import subprocess
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QFrame, QListWidget, QListWidgetItem,
                             QDialog, QLCDNumber, QMessageBox)
from PyQt6.QtGui import QFont, QColor, QPixmap, QPainter, QPolygon, QPen, QBrush, QImage
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QPoint, QRect
import cv2

# ===================== 日志全局工具 =====================
LOG_DIR = "./log"
# 初始化日志目录
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def get_log_file_path():
    """获取当天日志文件路径"""
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(LOG_DIR, f"app_{today}.log")

def write_log(msg_type: str, msg: str):
    """
    写入日志，同时打印控制台
    :param msg_type: INFO/WARN/ERROR/CAM/NET/SENSOR
    :param msg: 日志内容
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{now_str}] [{msg_type}] {msg}"
    # 控制台输出
    print(log_line)
    # 追加写入日志文件
    log_path = get_log_file_path()
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[LOG_ERROR] 写入日志失败: {str(e)}")

# ===================== AI后台模拟线程 =====================
class AiMonitorThread(QThread):
    signal_warn = pyqtSignal(str, str)
    signal_temp_humi_gas = pyqtSignal(float, float, float)
    signal_device_online = pyqtSignal(int)
    signal_sensor_status = pyqtSignal(bool)

    def run(self):
        write_log("INFO", "AI环境模拟线程启动")
        while True:
            # 模拟温湿度气体
            temp = round(22 + random.uniform(-3, 6), 1)
            humi = round(45 + random.uniform(-10, 15), 1)
            gas = round(random.uniform(0, 50), 1)
            self.signal_temp_humi_gas.emit(temp, humi, gas)
            self.signal_device_online.emit(random.randint(4, 7))
            sensor_ok = random.random() > 0.08
            self.signal_sensor_status.emit(sensor_ok)
            # 修正原代码告警概率逻辑
            warn_prob = random.random()
            if warn_prob == 0.93:
                self.signal_warn.emit("危险品预警", "幼儿手持别针行走过道")
            elif warn_prob == 0.86:
                self.signal_warn.emit("奔跑预警", "幼儿在过道快速追逐打闹")
            elif warn_prob == 0.80:
                self.signal_warn.emit("环境异常", "过道湿滑，存在滑倒风险")
            time.sleep(3)

# ===================== 摄像头读取线程（修复拔插卡死 + 日志） =====================
class CameraThread(QThread):
    ui_frame_signal = pyqtSignal(QImage)
    yolo_frame_signal = pyqtSignal(np.ndarray)
    cam_status_signal = pyqtSignal(bool)

    def __init__(self, dev_id=0):
        super().__init__()
        self.dev_id = dev_id
        self.running = False
        self.latest_bgr = None
        self.cap = None
        self.retry_delay = 0.5

    def _release_cap(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            write_log("CAM", "摄像头捕获器已释放")

    def run(self):
        self.running = True
        write_log("INFO", "摄像头采集线程启动")
        while self.running:
            # 无设备则重建捕获器
            if self.cap is None or not self.cap.isOpened():
                self._release_cap()
                self.cap = cv2.VideoCapture(self.dev_id, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 20)

                if not self.cap.isOpened():
                    write_log("CAM", "摄像头未检测到，等待重连...")
                    self.cam_status_signal.emit(False)
                    time.sleep(self.retry_delay)
                    continue
                write_log("CAM", "摄像头已成功连接")
                self.cam_status_signal.emit(True)

            # 读取帧
            ret, bgr_frame = self.cap.read()
            if not ret:
                write_log("ERROR", "摄像头读取帧失败，断开重连")
                self.cam_status_signal.emit(False)
                self._release_cap()
                time.sleep(self.retry_delay)
                continue

            self.cam_status_signal.emit(True)
            self.latest_bgr = bgr_frame.copy()
            self.yolo_frame_signal.emit(bgr_frame)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, w * c, QImage.Format.Format_RGB888)
            self.ui_frame_signal.emit(qimg)

        # 线程退出释放资源
        self._release_cap()
        self.cam_status_signal.emit(False)
        write_log("INFO", "摄像头采集线程正常退出")

    def stop(self):
        self.running = False

    def get_latest_frame(self):
        return self.latest_bgr

# ===================== 告警弹窗 =====================
class WarnDialog(QDialog):
    def __init__(self, warn_type, warn_msg):
        super().__init__()
        self.setWindowTitle("⚠️ 园区安全预警")
        self.setFixedSize(440, 240)
        self.setStyleSheet("background-color: #fff3f3; border-radius:12px;")
        layout = QVBoxLayout()
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(12)
        title = QLabel(f"【{warn_type}】")
        title.setFont(QFont("Noto Sans SC", 15, QFont.Weight.Bold))
        title.setStyleSheet("color:#d92121;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg = QLabel(warn_msg)
        msg.setFont(QFont("Noto Sans SC",12))
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_ok = QPushButton("确认处理")
        btn_ok.setFixedHeight(40)
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.setStyleSheet("""
            QPushButton{background:#3488d8;color:white;border-radius:8px;font-size:11pt;}
            QPushButton:hover{background:#2770b8;}
        """)
        btn_ok.clicked.connect(self.close)
        layout.addWidget(title)
        layout.addWidget(msg)
        layout.addWidget(btn_ok)
        self.setLayout(layout)

# ===================== 主窗口 =====================
class KindergartenGuardMain(QMainWindow):
    def __init__(self):
        super().__init__()
        write_log("INFO", "主程序窗口初始化完成")
        self.setWindowTitle("幼儿园AI安全卫士 - 10.1寸工控屏")
        self.resize(1280, 800)
        self.setStyleSheet("QMainWindow{background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #fff1d8, stop:1 #ffe6c2);}")
        self.ai_thread = AiMonitorThread()
        self.cam_thread = None
        self.btn_capture = None

        # 设备状态变量
        self.camera_ok = False
        self.sensor_ok = False
        self.network_ok = False

        self.save_dir = "./capture"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            write_log("INFO", "抓拍保存目录不存在，已创建 ./capture")

        self.mode_list = [
            {
                "text": "公共巡查模式",
                "bg_color": "#4096ee",
                "icon_func": self.draw_patrol_icon,
                "scene_title": "公共过道"
            },
            {
                "text": "教室模式",
                "bg_color": "#32a852",
                "icon_func": self.draw_class_icon,
                "scene_title": "中四班"
            },
            {
                "text": "睡眠室模式",
                "bg_color": "#9966cc",
                "icon_func": self.draw_sleep_icon,
                "scene_title": "一楼左侧睡眠室"
            }
        ]
        self.cur_mode_idx = 0
        self.top_frame = None
        self.label_mode = None
        self.mode_icon_label = None
        self.label_time = None
        self.label_scene = None
        self.label_cam_img = None

        self.lbl_cam_status = None
        self.lbl_sensor_status = None
        self.lbl_net_status = None

        self.init_ui()
        self.bind_signal()
        self.ai_thread.start()
        self.start_camera()
        # 系统定时器
        self.timer_clock = QTimer()
        self.timer_clock.timeout.connect(self.refresh_time)
        self.timer_clock.start(100)
        self.refresh_time()
        # 网络检测定时器 2秒ping一次
        self.timer_net_check = QTimer()
        self.timer_net_check.timeout.connect(self.check_network)
        self.timer_net_check.start(2000)

    # 12*12 对勾/红叉矢量图标
    def draw_check_icon(self, size:QSize, is_ok:bool):
        pix = QPixmap(size)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if is_ok:
            pen = QPen(QColor("#00aa22"), 2)
            pts = [QPoint(1, size.height()//2),
                   QPoint(size.width()//3, size.height()-2),
                   QPoint(size.width()-1, 1)]
            p.setPen(pen)
            p.drawPolyline(pts)
        else:
            pen = QPen(QColor("#dd2222"), 2)
            p.setPen(pen)
            p.drawLine(1,1, size.width()-1, size.height()-1)
            p.drawLine(size.width()-1, 1, 1, size.height()-1)
        p.end()
        return pix

    # 真实网络ping检测
    def check_network(self):
        try:
            res = subprocess.run(
                ["ping", "-c", "1", "-W", "1", "8.8.8.8"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            new_net_state = (res.returncode == 0)
            if new_net_state != self.network_ok:
                self.network_ok = new_net_state
                if self.network_ok:
                    write_log("NET", "网络已恢复连通")
                else:
                    write_log("ERROR", "网络断开，无法访问外网")
            self.refresh_device_status_ui()
        except Exception as e:
            write_log("ERROR", f"网络检测异常: {str(e)}")
            self.network_ok = False
            self.refresh_device_status_ui()

    # 刷新三个设备状态图标
    def refresh_device_status_ui(self):
        icon_size = QSize(12, 12)
        self.lbl_cam_status.setPixmap(self.draw_check_icon(icon_size, self.camera_ok))
        self.lbl_sensor_status.setPixmap(self.draw_check_icon(icon_size, self.sensor_ok))
        self.lbl_net_status.setPixmap(self.draw_check_icon(icon_size, self.network_ok))

    def start_camera(self):
        self.cam_thread = CameraThread(dev_id=0)
        self.cam_thread.ui_frame_signal.connect(self.update_cam_ui)
        self.cam_thread.yolo_frame_signal.connect(self.yolo_infer_frame)
        self.cam_thread.cam_status_signal.connect(self.set_camera_status)
        self.cam_thread.start()

    def set_camera_status(self, ok:bool):
        if ok != self.camera_ok:
            self.camera_ok = ok
            if ok:
                write_log("CAM", "摄像头设备状态变为正常")
            else:
                write_log("CAM", "摄像头设备断开")
        self.refresh_device_status_ui()
        if not ok:
            self.label_cam_img.setText("摄像头已断开，请检查设备")
            self.label_cam_img.setPixmap(QPixmap())

    def set_sensor_status(self, ok:bool):
        if ok != self.sensor_ok:
            self.sensor_ok = ok
            if ok:
                write_log("SENSOR", "传感器采集正常")
            else:
                write_log("SENSOR", "传感器数据异常")
        self.refresh_device_status_ui()

    def update_cam_ui(self, qimg):
        pix = QPixmap.fromImage(qimg)
        scaled_pix = pix.scaled(self.label_cam_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label_cam_img.setPixmap(scaled_pix)

    def yolo_infer_frame(self, bgr_mat):
        pass

    def manual_capture(self):
        if not self.cam_thread or not self.camera_ok:
            QMessageBox.warning(self, "提示", "摄像头未启动！")
            write_log("WARN", "手动抓拍失败：摄像头未就绪")
            return
        frame = self.cam_thread.get_latest_frame()
        if frame is None:
            QMessageBox.warning(self, "提示", "暂无摄像头画面！")
            write_log("WARN", "手动抓拍失败：无有效画面帧")
            return
        now = datetime.now()
        file_name = now.strftime("cap_%Y%m%d_%H%M%S.jpg")
        save_path = os.path.join(self.save_dir, file_name)
        cv2.imwrite(save_path, frame)
        QMessageBox.information(self, "抓拍成功", f"图片已保存")
        write_log("INFO", f"手动抓拍完成，保存路径: {save_path}")

    def switch_mode(self):
        self.cur_mode_idx = (self.cur_mode_idx + 1) % len(self.mode_list)
        mode_info = self.mode_list[self.cur_mode_idx]
        self.top_frame.setStyleSheet(f"QFrame{{background:{mode_info['bg_color']};border-radius:16px;}}")
        self.label_mode.setText(mode_info["text"])
        self.label_scene.setText(mode_info["scene_title"])
        pix = mode_info["icon_func"](QSize(32,32))
        self.mode_icon_label.setPixmap(pix)
        write_log("INFO", f"切换运行模式: {mode_info['text']} 场景:{mode_info['scene_title']}")

    def draw_patrol_icon(self, size: QSize, color=QColor("#ffffff")):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy = size.width()//2, size.height()//2
        painter.drawEllipse(QPoint(cx, int(cy - 10)), 6, 6)
        painter.drawLine(cx, int(cy - 4), cx, int(cy + 8))
        painter.drawLine(cx, int(cy), int(cx - 8), int(cy + 4))
        painter.drawLine(cx, int(cy), int(cx + 8), int(cy + 4))
        painter.drawLine(cx, int(cy + 8), int(cx - 6), int(cy + 14))
        painter.drawLine(cx, int(cy + 8), int(cx + 6), int(cy + 14))
        painter.end()
        return pix

    def draw_class_icon(self, size: QSize, color=QColor("#ffffff")):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 3)
        painter.setPen(pen)
        w, h = size.width(), size.height()
        painter.drawRect(4, 4, w-8, h-8)
        painter.drawLine(w//2, 4, w//2, h-4)
        painter.end()
        return pix

    def draw_sleep_icon(self, size: QSize, color=QColor("#ffffff")):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        w = size.width()
        h = size.height()
        painter.drawRect(3, h-12, w-6, 8)
        painter.drawRect(3, h-18, 6, 6)
        painter.drawRect(w-9, h-18, 6, 6)
        painter.drawRoundedRect(6, h-16, w-12, 4, 2, 2)
        painter.end()
        return pix

    def draw_play_icon(self, size: QSize, color=QColor("#ff7850")):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 3)
        brush = QBrush(color)
        painter.setPen(pen)
        painter.setBrush(brush)
        pts = [QPoint(8, 4), QPoint(size.width()-4, size.height()//2), QPoint(8, size.height()-4)]
        painter.drawPolygon(QPolygon(pts))
        painter.end()
        return pix

    def draw_bear_icon(self, size: QSize, color=QColor("#ff88aa")):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 3)
        brush = QBrush(color)
        painter.setPen(pen)
        painter.setBrush(brush)
        cx, cy = size.width()//2, size.height()//2
        r = min(cx, cy)-6
        painter.drawEllipse(QPoint(cx, cy), r, r)
        painter.drawEllipse(QPoint(cx-r, cy-r), r//3, r//3)
        painter.drawEllipse(QPoint(cx+r, cy-r), r//3, r//3)
        painter.end()
        return pix

    def draw_capture_icon(self, size: QSize, color=QColor("#f2b848")):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 4)
        painter.setPen(pen)
        w, h = size.width()-8, size.height()-8
        painter.drawRect(4, 4, w, h)
        painter.end()
        return pix

    def draw_trash_icon(self, size: QSize, color=QColor("#666666")):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        w = size.width()
        h = size.height()
        bx = int(w * 0.25)
        by = int(h * 0.3)
        bw = int(w * 0.5)
        bh = int(h * 0.6)
        painter.drawRect(bx, by, bw, bh)
        gx = int(w * 0.15)
        gy = int(h * 0.2)
        gw = int(w * 0.7)
        gh = int(h * 0.1)
        painter.drawRect(gx, gy, gw, gh)
        arc_x = int(w * 0.35)
        arc_y = int(h * 0.05)
        arc_w = int(w * 0.3)
        arc_h = int(h * 0.2)
        painter.drawArc(arc_x, arc_y, arc_w, arc_h, 0, 180 * 16)
        painter.end()
        return pix

    def draw_flower_icon(self, size: QSize, color=QColor("#ff4444")):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 2)
        brush = QBrush(color)
        painter.setPen(pen)
        painter.setBrush(brush)
        cx, cy = size.width()//2, size.height()//2
        r = min(cx, cy)//3
        for i in range(5):
            angle = i * 72
            dx = r * math.cos(math.radians(angle))
            dy = r * math.sin(math.radians(angle))
            px = int(cx + dx)
            py = int(cy + dy)
            painter.drawEllipse(QPoint(px, py), r, r)
        painter.drawEllipse(QPoint(cx, cy), r//2, r//2)
        painter.end()
        return pix

    def draw_hum_temp_icon(self, size=QSize(32,32)):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#33bb33"), 2)
        painter.setPen(pen)
        cx, cy = size.width()//2, size.height()//2
        painter.drawRoundedRect(cx-3, 2, 6, size.height()-6, 3,3)
        painter.drawEllipse(QPoint(cx, size.height()-4), 4,4)
        painter.end()
        return pix

    def draw_warn_tri_icon(self, size=QSize(32,32)):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#ffaa00"), 2)
        brush = QBrush(QColor("#ffdd77"))
        painter.setPen(pen)
        painter.setBrush(brush)
        pts = [QPoint(size.width()//2, 2), QPoint(2, size.height()-2), QPoint(size.width()-2, size.height()-2)]
        painter.drawPolygon(QPolygon(pts))
        painter.drawLine(size.width()//2, 5, size.width()//2, size.height()-6)
        painter.drawEllipse(QPoint(size.width()//2, size.height()-4), 1,1)
        painter.end()
        return pix

    def draw_cam_icon(self, size=QSize(32,32)):
        pix = QPixmap(size.width(), size.height())
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#3388dd"), 2)
        painter.setPen(pen)
        w, h = size.width(), size.height()
        painter.drawRoundedRect(2, 4, w-4, h-8, 2,2)
        painter.drawEllipse(QPoint(w-4, h//2), 3,3)
        painter.end()
        return pix

    def create_painter_icon_label(self, draw_func, size=QSize(44,44)):
        label = QLabel()
        pix = draw_func(size)
        label.setPixmap(pix)
        label.setStyleSheet("background: transparent;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def create_title_icon(self, draw_func):
        lbl = QLabel()
        lbl.setPixmap(draw_func())
        lbl.setStyleSheet("background: transparent;")
        return lbl

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6,6,6,6)
        root_layout.setSpacing(6)

        self.top_frame = QFrame()
        init_mode = self.mode_list[self.cur_mode_idx]
        self.top_frame.setStyleSheet(f"QFrame{{background:{init_mode['bg_color']};border-radius:16px;}}")
        self.top_frame.setFixedHeight(65)
        top_layout = QHBoxLayout(self.top_frame)
        top_layout.setContentsMargins(12,0,12,0)
        top_layout.setSpacing(6)

        self.label_time = QLabel("")
        self.label_time.setFont(QFont("Noto Sans SC", 14, QFont.Weight.Bold))
        self.label_time.setStyleSheet("color:white;")
        self.label_scene = QLabel(init_mode["scene_title"])
        self.label_scene.setFont(QFont("Noto Sans SC", 16, QFont.Weight.Bold))
        self.label_scene.setStyleSheet("color:white;")
        self.label_scene.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_mode_box = QWidget()
        left_mode_layout = QHBoxLayout(left_mode_box)
        left_mode_layout.setContentsMargins(0,0,0,0)
        left_mode_layout.setSpacing(6)
        self.mode_icon_label = QLabel()
        self.mode_icon_label.setStyleSheet("background: transparent;")
        init_pix = init_mode["icon_func"](QSize(32,32))
        self.mode_icon_label.setPixmap(init_pix)
        self.label_mode = QLabel(init_mode["text"])
        self.label_mode.setFont(QFont("Noto Sans SC",14, QFont.Weight.Bold))
        self.label_mode.setStyleSheet("color:white;")
        left_mode_layout.addWidget(self.mode_icon_label)
        left_mode_layout.addWidget(self.label_mode)

        top_layout.addWidget(left_mode_box)
        top_layout.addStretch()
        top_layout.addWidget(self.label_scene)
        top_layout.addStretch()
        top_layout.addWidget(self.label_time)
        root_layout.addWidget(self.top_frame)

        mid_box = QWidget()
        mid_layout = QHBoxLayout(mid_box)
        mid_layout.setSpacing(6)
        cam_frame = QFrame()
        cam_frame.setStyleSheet("QFrame{background:#ffffff;border-radius:16px;}")
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.setContentsMargins(6,6,6,6)
        cam_layout.setSpacing(6)

        self.label_cam_img = QLabel("摄像头未开启...")
        self.label_cam_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_cam_img.setScaledContents(False)
        self.label_cam_img.setFont(QFont("Noto Sans SC",13))
        self.label_cam_img.setStyleSheet("color:#666666;")

        cam_tip = QLabel("【实时监控画面】")
        cam_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam_tip.setFont(QFont("Noto Sans SC",13))
        cam_tip.setStyleSheet("color:#666666;")
        cam_layout.addWidget(cam_tip)
        cam_layout.addWidget(self.label_cam_img, stretch=1)
        mid_layout.addWidget(cam_frame, stretch=3)

        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setSpacing(6)

        env_card = QFrame()
        env_card.setStyleSheet("QFrame{background:#e7f9e7;border-radius:14px;padding:8px;}")
        env_layout = QVBoxLayout(env_card)
        env_layout.setContentsMargins(0,0,0,0)
        env_layout.setSpacing(4)
        env_title_box = QWidget()
        env_title_layout = QHBoxLayout(env_title_box)
        env_title_layout.setContentsMargins(0,0,0,0)
        env_title_layout.setSpacing(4)
        env_title_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        env_title_layout.addWidget(self.create_title_icon(self.draw_hum_temp_icon))
        env_title_text = QLabel("环境感知数据")
        env_title_text.setFont(QFont("Noto Sans SC",12, QFont.Weight.Bold))
        env_title_layout.addWidget(env_title_text)
        env_layout.addWidget(env_title_box)

        row_temp = QHBoxLayout()
        row_temp.setContentsMargins(0,0,0,0)
        row_temp.setSpacing(0)
        lab_temp_text = QLabel("温度(℃)")
        lab_temp_text.setFont(QFont("Noto Sans SC",13))
        self.lcd_temp = QLCDNumber()
        self.lcd_temp.setDigitCount(5)
        self.lcd_temp.setStyleSheet("color:#d82626;")
        self.lcd_temp.setFont(QFont("Noto Sans SC", 16, QFont.Weight.Bold))
        row_temp.addWidget(lab_temp_text)
        row_temp.addStretch()
        row_temp.addWidget(self.lcd_temp)
        env_layout.addLayout(row_temp)

        row_humi = QHBoxLayout()
        row_humi.setContentsMargins(0,0,0,0)
        row_humi.setSpacing(0)
        lab_humi_text = QLabel("湿度(%)")
        lab_humi_text.setFont(QFont("Noto Sans SC",13))
        self.lcd_humi = QLCDNumber()
        self.lcd_humi.setDigitCount(5)
        self.lcd_humi.setStyleSheet("color:#2670d8;")
        self.lcd_humi.setFont(QFont("Noto Sans SC", 16, QFont.Weight.Bold))
        row_humi.addWidget(lab_humi_text)
        row_humi.addStretch()
        row_humi.addWidget(self.lcd_humi)
        env_layout.addLayout(row_humi)

        row_gas = QHBoxLayout()
        row_gas.setContentsMargins(0,0,0,0)
        row_gas.setSpacing(0)
        lab_gas_text = QLabel("气体浓度(ppm)")
        lab_gas_text.setFont(QFont("Noto Sans SC",13))
        self.lcd_gas = QLCDNumber()
        self.lcd_gas.setDigitCount(5)
        self.lcd_gas.setStyleSheet("color:#26a846;")
        self.lcd_gas.setFont(QFont("Noto Sans SC", 16, QFont.Weight.Bold))
        row_gas.addWidget(lab_gas_text)
        row_gas.addStretch()
        row_gas.addWidget(self.lcd_gas)
        env_layout.addLayout(row_gas)

        warn_card = QFrame()
        warn_card.setStyleSheet("QFrame{background:#fff6e0;border-radius:14px;padding:8px;}")
        warn_layout = QVBoxLayout(warn_card)
        warn_layout.setContentsMargins(0,0,0,0)
        warn_layout.setSpacing(4)
        warn_title_box = QWidget()
        warn_title_layout = QHBoxLayout(warn_title_box)
        warn_title_layout.setContentsMargins(0,0,0,0)
        warn_title_layout.setSpacing(4)
        warn_title_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        warn_title_layout.addWidget(self.create_title_icon(self.draw_warn_tri_icon))
        warn_title_text = QLabel("AI预警列表")
        warn_title_text.setFont(QFont("Noto Sans SC",12, QFont.Weight.Bold))
        warn_title_layout.addWidget(warn_title_text)
        warn_layout.addWidget(warn_title_box)
        self.warn_list = QListWidget()
        self.warn_list.setMaximumHeight(160)
        warn_layout.addWidget(self.warn_list)

        # 设备状态：摄像头、传感器、网络 同一行
        dev_card = QFrame()
        dev_card.setStyleSheet("QFrame{background:#e6f2ff;border-radius:14px;padding:6px;}")
        dev_layout = QVBoxLayout(dev_card)
        dev_layout.setContentsMargins(0,0,0,0)
        dev_layout.setSpacing(3)
        dev_title_box = QWidget()
        dev_title_layout = QHBoxLayout(dev_title_box)
        dev_title_layout.setContentsMargins(0,0,0,0)
        dev_title_layout.setSpacing(4)
        dev_title_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        dev_title_layout.addWidget(self.create_title_icon(self.draw_cam_icon))
        dev_title_text = QLabel("设备状态")
        dev_title_text.setFont(QFont("Noto Sans SC",12, QFont.Weight.Bold))
        dev_title_layout.addWidget(dev_title_text)
        dev_layout.addWidget(dev_title_box)

        row_dev_all = QHBoxLayout()
        row_dev_all.setSpacing(16)
        # 摄像头
        cam_wrap = QHBoxLayout()
        cam_wrap.setSpacing(4)
        self.lbl_cam_status = QLabel()
        cam_wrap.addWidget(self.lbl_cam_status)
        cam_wrap.addWidget(QLabel("摄像头"))
        row_dev_all.addLayout(cam_wrap)
        # 传感器
        sen_wrap = QHBoxLayout()
        sen_wrap.setSpacing(4)
        self.lbl_sensor_status = QLabel()
        sen_wrap.addWidget(self.lbl_sensor_status)
        sen_wrap.addWidget(QLabel("传感器"))
        row_dev_all.addLayout(sen_wrap)
        # 网络
        net_wrap = QHBoxLayout()
        net_wrap.setSpacing(4)
        self.lbl_net_status = QLabel()
        net_wrap.addWidget(self.lbl_net_status)
        net_wrap.addWidget(QLabel("网络"))
        row_dev_all.addLayout(net_wrap)

        row_dev_all.addStretch()
        dev_layout.addLayout(row_dev_all)
        self.refresh_device_status_ui()

        right_panel_layout.addWidget(env_card)
        right_panel_layout.addWidget(warn_card)
        right_panel_layout.addWidget(dev_card)
        mid_layout.addWidget(right_panel, stretch=1)
        root_layout.addWidget(mid_box, stretch=4)

        # 底部按钮
        bottom_box = QWidget()
        bottom_layout = QHBoxLayout(bottom_box)
        bottom_layout.setSpacing(10)
        btn_config = [
            {"draw_func": self.draw_play_icon, "text":"模式切换", "bg":"#ffe0b2"},
            {"draw_func": self.draw_bear_icon, "text":"语音唤醒", "bg":"#ffd6e0"},
            {"draw_func": self.draw_capture_icon, "text":"手动抓拍", "bg":"#fff2cc"},
            {"draw_func": self.draw_trash_icon, "text":"预警清零", "bg":"#fff0d6"},
            {"draw_func": self.draw_flower_icon, "text":"紧急预警", "bg":"#ff7878"}
        ]
        self.btn_clear_warn = None
        self.btn_capture = None
        for cfg in btn_config:
            btn_container = QVBoxLayout()
            btn_container.setSpacing(4)
            btn_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label = self.create_painter_icon_label(cfg["draw_func"], QSize(44,44))
            text_label = QLabel(cfg["text"])
            text_label.setFont(QFont("Noto Sans SC",12, QFont.Weight.Bold))
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text_label.setStyleSheet("background: transparent;")
            btn = QPushButton()
            btn.setLayout(btn_container)
            btn_container.addWidget(icon_label)
            btn_container.addWidget(text_label)
            btn.setFixedHeight(110)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton{{background:{cfg['bg']};border-radius:22px;border:none;}}
                QPushButton:hover{{background:#ffc896;}}
            """)
            if cfg["text"] == "预警清零":
                self.btn_clear_warn = btn
                self.btn_clear_warn.clicked.connect(self.clear_all_warn)
            if cfg["text"] == "模式切换":
                btn.clicked.connect(self.switch_mode)
            if cfg["text"] == "手动抓拍":
                self.btn_capture = btn
                self.btn_capture.clicked.connect(self.manual_capture)
            bottom_layout.addWidget(btn)
        root_layout.addWidget(bottom_box)

    def bind_signal(self):
        self.ai_thread.signal_temp_humi_gas.connect(self.update_env_data)
        self.ai_thread.signal_device_online.connect(self.update_device_count)
        self.ai_thread.signal_warn.connect(self.show_warn_dialog)
        self.ai_thread.signal_sensor_status.connect(self.set_sensor_status)

    def refresh_time(self):
        week_map = {0:"星期一",1:"星期二",2:"星期三",3:"星期四",4:"星期五",5:"星期六",6:"星期日"}
        now = datetime.now()
        weekday = week_map[now.weekday()]
        time_str = now.strftime(f"%Y-%m-%d %H:%M:%S {weekday}")
        self.label_time.setText(time_str)

    def update_env_data(self, temp, humi, gas):
        self.lcd_temp.display(temp)
        self.lcd_humi.display(humi)
        self.lcd_gas.display(gas)

    def update_device_count(self, num):
        pass

    def show_warn_dialog(self, warn_type, warn_msg):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item = QListWidgetItem(f"[{now_str}] {warn_type}：{warn_msg}")
        item.setForeground(QColor("#d62020"))
        self.warn_list.addItem(item)
        write_log("WARN", f"触发弹窗告警：{warn_type} 内容:{warn_msg}")
        dialog = WarnDialog(warn_type, warn_msg)
        dialog.exec()

    def clear_all_warn(self):
        self.warn_list.clear()
        write_log("INFO", "手动清空告警列表")

    def closeEvent(self, event):
        write_log("INFO", "程序收到关闭信号，开始安全退出流程")
        # 安全停止摄像头线程
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread.wait(1000)
        if self.ai_thread:
            self.ai_thread.terminate()
            self.ai_thread.wait(1000)
        write_log("INFO", "程序正常退出")
        event.accept()

if __name__ == "__main__":
    write_log("INFO", "========== 幼儿园AI安全卫士程序启动 ==========")
    app = QApplication(sys.argv)
    global_font = QFont("Noto Sans SC", 10)
    app.setFont(global_font)
    win = KindergartenGuardMain()
    win.show()
    exit_code = app.exec()
    write_log("INFO", f"程序结束，退出码:{exit_code}")
    sys.exit(exit_code)
