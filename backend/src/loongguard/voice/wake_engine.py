"""
语音唤醒引擎（openWakeWord）+ 麦克风采集

设计动机：
    双引擎分离架构的第一级——专职唤醒词检测，不做指令识别。
    openWakeWord 的 ONNX 模型仅 ~200KB，CPU 实时推理无压力，
    ONNX Runtime 有龙芯 loongarch64 官方 wheel。

音频采集：
    - Linux/板端：arecord 管道（16kHz/16bit/单声道 PCM），无 PyAudio 依赖
    - Windows 开发机：PyAudio（可选安装）
    - 均不可用：DummyCapture（静音流，链路可跑通但永远不唤醒）

降级策略：
    openwakeword 依赖未安装或唤醒模型文件缺失时，is_available=False，
    VoiceService 据此决定是否启用语音功能，不阻塞 Pipeline。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# PCM 格式常量：16kHz / 16bit / 单声道
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes
CHANNELS = 1
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS


# ── 麦克风采集 ────────────────────────────────────────────────


class AudioCapture(ABC):
    """麦克风 PCM 采集抽象接口（阻塞式读取）"""

    @abstractmethod
    def read(self, nbytes: int) -> bytes:
        """读取指定字节数的 PCM 数据（阻塞直至读满或流结束）"""

    def close(self) -> None:  # noqa: B027 可选覆写钩子：子类按需实现，默认无操作
        """释放采集资源（默认无操作）"""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """采集流是否仍然健康"""


class ArecordCapture(AudioCapture):
    """板端采集：arecord 原始 PCM 管道（Loongnix ALSA）"""

    def __init__(self, device: str | None = None) -> None:
        cmd = ["arecord", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
               "-c", str(CHANNELS), "-t", "raw", "-q"]
        if device:
            cmd.extend(["-D", device])
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        logger.info("ArecordCapture started (pid=%s)", self._proc.pid)

    def read(self, nbytes: int) -> bytes:
        assert self._proc.stdout is not None
        data = self._proc.stdout.read(nbytes)
        return data or b""

    @property
    def is_running(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        logger.info("ArecordCapture stopped")


class PyAudioCapture(AudioCapture):
    """Windows 开发机采集：PyAudio 流式读取（需 pip install pyaudio）"""

    def __init__(self, device_index: int | None = None) -> None:
        import pyaudio  # noqa: PLC0415 延迟导入：可选依赖

        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            rate=SAMPLE_RATE,
            channels=CHANNELS,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1600,  # 100ms
        )
        logger.info("PyAudioCapture started")

    def read(self, nbytes: int) -> bytes:
        chunks: list[bytes] = []
        remaining = nbytes
        while remaining > 0:
            data = self._stream.read(
                min(remaining // SAMPLE_WIDTH, 1600), exception_on_overflow=False
            )
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    @property
    def is_running(self) -> bool:
        return self._stream is not None and not self._stream.is_stopped()

    def close(self) -> None:
        try:
            self._stream.stop_stream()
            self._stream.close()
            self._pyaudio.terminate()
        except Exception:
            pass
        logger.info("PyAudioCapture stopped")


class DummyCapture(AudioCapture):
    """桩实现：持续产出静音帧（无麦克风/CI 环境，链路可跑通）"""

    def __init__(self) -> None:
        self._running = True
        logger.warning("DummyCapture: 无真实麦克风，唤醒词检测不会触发")

    def read(self, nbytes: int) -> bytes:
        time.sleep(nbytes / BYTES_PER_SEC)  # 模拟实时速率
        return b"\x00" * nbytes

    @property
    def is_running(self) -> bool:
        return self._running

    def close(self) -> None:
        self._running = False


def create_capture(platform: str = "auto") -> AudioCapture:
    """按平台与可用性选择采集后端（永不抛异常，最差降级 Dummy）"""
    if platform == "auto":
        platform = sys_platform()
    try:
        if platform in ("linux", "loongarch") and shutil.which("arecord"):
            return ArecordCapture()
        if platform == "win32":
            try:
                return PyAudioCapture()
            except Exception:
                logger.warning("PyAudio 不可用（未安装或无麦克风），降级 DummyCapture")
                return DummyCapture()
    except Exception:
        logger.exception("音频采集初始化失败，降级 DummyCapture")
    return DummyCapture()


def sys_platform() -> str:
    import sys

    return sys.platform


# ── 唤醒词检测引擎 ────────────────────────────────────────────


class WakeEngine:
    """
    openWakeWord 唤醒词检测引擎

    职责：
        - 加载自定义唤醒词 ONNX 模型（"小龙小龙" / "晓珑监护"）
        - 逐帧推理并按阈值判定唤醒命中
        - 冷却窗口防止一次唤醒被连续判定多次
    """

    def __init__(self, model_path: str, threshold: float = 0.5,
                 cooldown_sec: float = 3.0) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._cooldown_sec = cooldown_sec
        self._last_wake_time = 0.0
        self._model = None
        self._available = False
        self._wake_name = Path(model_path).stem or "wakeword"
        self._load()

    def _load(self) -> None:
        if not Path(self._model_path).exists():
            logger.warning(
                "唤醒词模型不存在: %s（参考 backend/models/wakeword/README.md "
                "训练并放置模型后重启）", self._model_path,
            )
            return
        try:
            from openwakeword.model import Model  # noqa: PLC0415 延迟导入：可选依赖
        except ImportError:
            logger.warning(
                "openwakeword 未安装（pip install openwakeword），唤醒功能停用"
            )
            return
        try:
            self._model = Model(wakeword_models=[self._model_path])
            self._available = True
            logger.info(
                "WakeEngine ready: %s (threshold=%.2f, cooldown=%.1fs)",
                self._model_path, self._threshold, self._cooldown_sec,
            )
        except Exception:
            logger.exception("唤醒词模型加载失败: %s", self._model_path)

    @property
    def is_available(self) -> bool:
        return self._available

    def predict(self, pcm_chunk: bytes) -> bool:
        """
        检测一段 PCM 是否包含唤醒词。

        Args:
            pcm_chunk: 16kHz/16bit/单声道 PCM 字节流

        Returns:
            True 表示唤醒命中（且已过冷却窗口）
        """
        if not self._available or not pcm_chunk:
            return False
        try:
            prediction = self._model.predict(pcm_chunk)
        except Exception:
            logger.exception("openWakeWord 推理异常")
            return False
        score = max(prediction.values()) if prediction else 0.0
        now = time.monotonic()
        if (score >= self._threshold
                and now - self._last_wake_time >= self._cooldown_sec):
            self._last_wake_time = now
            logger.info("唤醒命中: score=%.3f", score)
            return True
        return False

    def reset(self) -> None:
        """重置模型内部音频缓冲状态"""
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass
