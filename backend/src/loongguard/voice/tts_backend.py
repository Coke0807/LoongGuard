"""
TTS 语音播报后端（跨平台）

设计动机：
    板端（Loongnix）用 espeak 系统命令合成中文，Windows 开发机用
    pyttsx3（SAPI5）。业务层只依赖 TTSBackend 接口，按平台自动选择，
    两者均不可用时降级为 DummyTTS（仅日志），保证主链路不中断。

来源：重构自 VoiceDetection/long3.py / long4.py 的 speak() 函数。
"""

from __future__ import annotations

import logging
import subprocess
import sys
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    """TTS 语音播报抽象接口"""

    @abstractmethod
    def speak(self, text: str) -> None:
        """同步播报一段中文文本（阻塞至播报完成或放弃）"""

    def stop(self) -> None:
        """释放后端资源（默认无操作）"""


class DummyTTS(TTSBackend):
    """桩实现：仅写日志，不发声（无音频设备/测试环境）"""

    def speak(self, text: str) -> None:
        logger.info("[TTS-dummy] %s", text)


class Pyttsx3TTS(TTSBackend):
    """Windows TTS 后端（pyttsx3 / SAPI5，需要系统安装中文语音包）"""

    def __init__(self) -> None:
        import pyttsx3  # noqa: PLC0415 延迟导入：可选依赖

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 185)
        self._engine.setProperty("volume", 0.95)
        # 尝试加载中文语音
        for voice in self._engine.getProperty("voices"):
            vid = (voice.id or "").lower()
            vname = (voice.name or "").lower()
            if "zh" in vid or "chinese" in vname or "huihui" in vname:
                self._engine.setProperty("voice", voice.id)
                break

    def speak(self, text: str) -> None:
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            logger.exception("pyttsx3 播报失败: %s", text)

    def stop(self) -> None:
        try:
            self._engine.stop()
        except Exception:
            pass


class EspeakTTS(TTSBackend):
    """Loongnix 板端 TTS 后端（espeak 系统命令）"""

    def __init__(self, rate: int = 185, amplitude: int = 95) -> None:
        self._rate = rate
        self._amplitude = amplitude

    def speak(self, text: str) -> None:
        try:
            subprocess.run(
                ["espeak", "-v", "zh", "-s", str(self._rate),
                 "-a", str(self._amplitude), text],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=10,
            )
        except FileNotFoundError:
            logger.warning("espeak 未安装，跳过播报: %s", text)
        except Exception as exc:
            logger.warning("espeak 播报失败: %s (%s)", text, exc)


def create_tts_backend(platform: str = "auto") -> TTSBackend:
    """
    工厂函数：按平台/配置选择 TTS 后端

    Args:
        platform: "auto"（按 sys.platform 自动选择）/
                  "pyttsx3" / "espeak" / "dummy"
    """
    if platform == "auto":
        platform = sys.platform
    if platform in ("win32", "pyttsx3"):
        try:
            return Pyttsx3TTS()
        except Exception:
            logger.warning("pyttsx3 初始化失败，降级 DummyTTS")
            return DummyTTS()
    if platform in ("espeak", "linux"):
        return EspeakTTS()
    return DummyTTS()
