import threading
import subprocess
import os
from ui.utils import write_log

class VoicePlayer:
    # 音频存放目录
    VOICE_DIR = "./voice"
    # 超时时间
    PLAY_TIMEOUT = 8

    def play_audio(self, audio_filename: str):
        """兼容 mp3 / m4a 通用异步播放"""
        def play_task():
            audio_path = os.path.join(self.VOICE_DIR, audio_filename)
            if not os.path.exists(audio_path):
                write_log("VOICE", f"音频文件不存在：{audio_filename}")
                return

            cmd = [
                "ffplay",
                "-autoexit",
                "-nodisp",
                "-hide_banner",
                audio_path
            ]
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.PLAY_TIMEOUT
                )
            except Exception as e:
                write_log("VOICE", f"音频播放失败：{str(e)}")

        # 开启守护子线程异步播放
        thread = threading.Thread(target=play_task, daemon=True)
        thread.start()

    # 场景快捷调用方法（后缀改成mp3即可）
    def voice_started(self):
        self.play_audio("started.mp3")

    def voice_start_ok(self):
        self.play_audio("start_ok.mp3")

    # 模式切换
    def voice_class(self):
        self.play_audio("class_mode.mp3")

    def voice_sleep(self):
        self.play_audio("sleep_mode.mp3")

    def voice_patrol(self):
        self.play_audio("patrol.mp3")

    # 设备检查
    def voice_sensor_error(self):
        self.play_audio("sensor_error.mp3")

    def voice_sensor_ok(self):
        self.play_audio("sensor_ok.mp3")

    def voice_camera_error(self):
        self.play_audio("camera_error.mp3")

    def voice_camera_ok(self):
        self.play_audio("camera_ok.mp3")

    def voice_network_error(self):
        self.play_audio("network_error.mp3")

    def voice_network_ok(self):
        self.play_audio("network_ok.mp3")

    def voice_mic_error(self):
        self.play_audio("mic_error.mp3")

    def voice_mic_ok(self):
        self.play_audio("mic_ok.mp3")

    # 系统设置页面
    def voice_sys_settings(self):
        self.play_audio("sys_settings.mp3")

    def voice_save_ok(self):
        self.play_audio("save_ok.mp3")

    def voice_close_sys_settings(self):
        self.play_audio("close_sys_settings.mp3")

    def voice_word_settings(self):
        self.play_audio("word_settings.mp3")

    def voice_alarm_policy(self):
        self.play_audio("alarm_policy.mp3")

    def voice_cartoon_theme(self):
        self.play_audio("cartoon_theme.mp3")

    def voice_ntp_valid(self):
        self.play_audio("ntp_valid.mp3")

    def voice_online_upgrade(self):
        self.play_audio("onlice_upgrade.mp3")

    def voice_video_record(self):
        self.play_audio("video_record.mp3")