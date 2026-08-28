"""
音频后端抽象接口（语音功能跨平台预留）

设计动机：
    语音唤醒 / ASR / 双向通话等能力未来接入，但底层音频设备与命令
    平台差异大（Windows 用 ffplay/ffmpeg，板端用 aplay/arecord 或
    PulseAudio）。本模块定义统一 AudioBackend 接口，业务层只依赖
    接口，不绑定具体平台命令，替换实现无需改动调用方。

扩展方式：
    1. 实现 AudioBackend 的子类（如接入 FunASR 的 ASR 后端）
    2. 在 create_audio_backend() 中按配置返回对应实现
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AudioBackend(ABC):
    """
    音频后端协议

    方法：
        play(path):      播放音频文件（语音播报）
        record(path):    录制麦克风音频（ASR / 通话上行）
    """

    @abstractmethod
    def play(self, path: str) -> None:
        """播放音频文件，非阻塞"""

    @abstractmethod
    def record(self, path: str, duration: float = 5.0) -> None:
        """录制麦克风音频到指定路径"""


class DummyAudioBackend(AudioBackend):
    """
    桩实现：不执行真实音频操作

    用于 Windows 开发期 / 无音频设备环境，保证链路可跑通。
    """

    def play(self, path: str) -> None:
        # 桩实现：仅记录，不实际播放
        pass

    def record(self, path: str, duration: float = 5.0) -> None:
        # 桩实现：仅记录，不实际录制
        pass


class CommandAudioBackend(AudioBackend):
    """
    基于系统命令的音频后端（跨平台）

    通过外部命令播放 / 录制，平台差异由命令本身屏蔽：
        - 播放：优先 ffplay，缺失时回退
        - 录制：Linux 用 arecord，Windows 用 ffmpeg(需 dshow 设备)

    说明：仅提供跨平台封装骨架，具体命令在当前 MVP 阶段不强制可用，
    替换为真实后端（如 FFmpeg API / sounddevice）时无需改调用方。
    """

    def __init__(self, play_cmd: str = "ffplay", record_cmd: str = "") -> None:
        self._play_cmd = play_cmd
        self._record_cmd = record_cmd

    def play(self, path: str) -> None:
        if not Path(path).exists():
            return
        # TODO: 接入跨平台播放命令（ffplay aplay 等），MVP 阶段由桩替代
        import subprocess

        try:
            subprocess.Popen(
                [self._play_cmd, "-autoexit", "-nodisp", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            # 命令缺失时静默降级，避免打断主链路
            pass

    def record(self, path: str, duration: float = 5.0) -> None:
        # TODO: 接入跨平台录音命令，MVP 阶段由桩替代
        pass


def create_audio_backend(backend: str = "command") -> AudioBackend:
    """
    依据配置创建音频后端

    Args:
        backend: "command"（系统命令）/ "dummy"（桩）

    Returns:
        AudioBackend 实现实例
    """
    if backend == "command":
        return CommandAudioBackend()
    return DummyAudioBackend()
