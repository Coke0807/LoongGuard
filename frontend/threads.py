import cv2
import os
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

# ===================== AI 告警订阅线程 =====================
# 设计动机（跨平台集成）：
#   前端不再模拟随机告警，而是订阅后端 AlertAPIServer 的
#   WebSocket /ws/alerts 实时推送，收到 AlertLog 后转成中文文案显示。
#   断线自动重连，连接状态通过 signal_sensor_status 上抛。
class AiMonitorThread(QThread):
    signal_warn = pyqtSignal(str, str)
    signal_temp_humi_gas = pyqtSignal(float, float, float)
    signal_device_online = pyqtSignal(int)
    signal_sensor_status = pyqtSignal(bool)

    # 后端告警 WebSocket 地址，可用环境变量覆盖（默认本机 8080）
    _WS_URL = os.environ.get("LG_WS_URL", "ws://127.0.0.1:8080/ws/alerts")

    def __init__(self, ws_url=None):
        super().__init__()
        self.ws_url = ws_url or self._WS_URL
        self.running = False
        self.retry_delay = 2.0

    def run(self):
        self.running = True
        write_log("INFO", f"AI 告警订阅线程启动，连接: {self.ws_url}")
        import asyncio
        asyncio.run(self._monitor_loop())

    async def _monitor_loop(self):
        """订阅后端 /ws/alerts，断线自动重连"""
        import asyncio
        import aiohttp

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        self.ws_url, heartbeat=15,
                    ) as ws:
                        write_log("INFO", "已连接后端告警 WebSocket")
                        self.signal_sensor_status.emit(True)
                        while self.running:
                            try:
                                msg = await asyncio.wait_for(
                                    ws.receive(), timeout=1.0,
                                )
                            except asyncio.TimeoutError:
                                continue
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._handle_alert(msg.data)
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break
            except Exception as exc:
                write_log("ERROR", f"告警流中断: {exc}")
                self.signal_sensor_status.emit(False)
            if not self.running:
                break
            await asyncio.sleep(self.retry_delay)

    def _handle_alert(self, data: str):
        """解析后端 AlertLog JSON 并转中文后上抛 signal_warn"""
        import json

        try:
            alert = json.loads(data)
        except Exception:
            return

        alert_type = alert.get("alert_type", "")
        severity = alert.get("severity", "")
        description = alert.get("description", "")

        type_zh = _ALERT_TYPE_ZH.get(alert_type, alert_type)
        sev_zh = _SEVERITY_ZH.get(severity, severity)
        msg = f"[{sev_zh}] {description}" if description else f"[{sev_zh}] {type_zh}"
        write_log("WARN", f"收到后端告警: {type_zh} - {msg}")
        self.signal_warn.emit(type_zh, msg)

    def stop(self):
        self.running = False


# 后端告警类型 / 严重等级 -> 前端中文文案映射
_ALERT_TYPE_ZH = {
    "dangerous_object": "危险物品预警",
    "prone_sleep": "俯卧趴睡预警",
    "environment": "环境异常",
    "motion_abnormal": "异常运动预警",
}
_SEVERITY_ZH = {
    "low": "提醒",
    "medium": "警告",
    "high": "严重",
    "critical": "紧急",
}

# ===================== 摄像头读取线程 =====================
# 设计动机（跨平台集成）：
#   前端不再直接打开本地摄像头（避免与后端推理抢占 /dev/video0），
#   而是拉取后端 AlertAPIServer 暴露的 MJPEG 流 /stream 显示画面。
#   画面由后端 VideoHub 已绘制好检测框/运动区域，前端仅消费解码显示。
class CameraThread(QThread):
    ui_frame_signal = pyqtSignal(QImage)
    yolo_frame_signal = pyqtSignal(np.ndarray)
    cam_status_signal = pyqtSignal(bool)
    raw_frame_signal = pyqtSignal(np.ndarray)
    # 输出真实分辨率、估算帧率给主窗口
    cam_info_signal = pyqtSignal(int, int, float)

    # 后端 MJPEG 流地址，可用环境变量覆盖（默认本机 8080）
    _STREAM_URL = os.environ.get("LG_STREAM_URL", "http://127.0.0.1:8080/stream")

    def __init__(self, dev_id=0, stream_url=None):
        """
        Args:
            dev_id: 兼容旧调用保留（不再直接使用），实际从后端拉流
            stream_url: 后端 /stream 地址，None 时用默认值
        """
        super().__init__()
        self.dev_id = dev_id
        self.stream_url = stream_url or self._STREAM_URL
        self.running = False
        self.latest_bgr = None
        self.retry_delay = 0.5
        self.fps = 20.0        # 估算值，由解码帧间隔得出
        self.width = 1280
        self.height = 800
        self.voice_player = VoicePlayer()

    def draw_qimg_time(self, rgb_arr):
        """在 RGB 帧上叠加时间戳，返回带时间戳的 RGB 数组"""
        from datetime import datetime
        h, w, c = rgb_arr.shape
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temp_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        cv2.putText(temp_bgr, now_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(temp_bgr, now_text, (12,32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
        return cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)

    def run(self):
        self.running = True
        write_log("INFO", f"摄像头线程启动，拉取后端流: {self.stream_url}")
        while self.running:
            try:
                self._stream_loop()
            except Exception as exc:
                write_log("ERROR", f"MJPEG 流中断: {exc}")
                self.cam_status_signal.emit(False)
                time.sleep(self.retry_delay)

        self.cam_status_signal.emit(False)
        write_log("INFO", "摄像头线程正常退出")

    def _stream_loop(self):
        """
        连接后端 /stream 并解析 multipart/x-mixed-replace MJPEG 帧

        帧格式（见 backend server.py _handle_stream）：
            --frame\r\nContent-Type: image/jpeg\r\n\r\n<jpeg>\r\n
        """
        import urllib.request

        req = urllib.request.Request(self.stream_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            write_log("CAM", f"MJPEG 流已连接: {self.stream_url}")
            self.cam_status_signal.emit(True)

            buf = b""
            boundary = b"--frame"
            while self.running:
                chunk = resp.read(65536)
                if not chunk:
                    break
                buf += chunk

                # 从缓冲区逐个提取完整帧
                while True:
                    start = buf.find(boundary)
                    if start == -1:
                        break
                    # 定位 JPEG 数据起始（跳过响应头两行）
                    hdr_end = buf.find(b"\r\n\r\n", start)
                    if hdr_end == -1:
                        break
                    jpeg_start = hdr_end + 4
                    # 定位到下一个边界，即本帧结束
                    next_b = buf.find(boundary, jpeg_start)
                    if next_b == -1:
                        break
                    jpeg_bytes = buf[jpeg_start:next_b].rstrip(b"\r\n")
                    buf = buf[next_b:]

                    if jpeg_bytes:
                        self._handle_jpeg(jpeg_bytes)

    def _handle_jpeg(self, jpeg_bytes: bytes):
        """解码一帧 JPEG，更新最新帧并发出各信号"""
        img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            write_log("WARN", "JPEG 解码失败，跳过该帧")
            return

        h, w = img.shape[:2]
        self.width = w
        self.height = h
        self.latest_bgr = img.copy()

        # 估算帧率：依据上一次到本次的时间间隔
        now = time.monotonic()
        if hasattr(self, "_last_ts"):
            dt = now - self._last_ts
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self._last_ts = now

        # 供录制使用原始帧（含后端标注，但无前端时间戳）
        self.raw_frame_signal.emit(img)
        self.yolo_frame_signal.emit(img)
        self.cam_info_signal.emit(w, h, self.fps)

        # 叠加时间戳后显示
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_with_time = self.draw_qimg_time(rgb)
        # 用 .copy() 让 QImage 持有数据副本（不引用 numpy 缓冲区）。
        # 否则跨线程队列投递到 GUI 线程时，numpy 数组可能已被释放，
        # Qt 再读取该内存会触发 access violation 崩溃。
        qimg = QImage(
            rgb_with_time.data, w, h, w * 3, QImage.Format.Format_RGB888
        ).copy()
        self.ui_frame_signal.emit(qimg)

    def stop(self):
        self.running = False

    def get_latest_frame(self):
        return self.latest_bgr
