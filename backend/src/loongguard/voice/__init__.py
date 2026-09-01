"""
LoongGuard 语音模块（openWakeWord 唤醒 + PocketSphinx 指令）

架构：
    麦克风 PCM → WakeEngine（唤醒词检测）
                    ↓ 命中
                SphinxCommandRecognizer（JSGF 指令识别）
                    ↓
                TextCommandDispatcher（业务分发，回调注入 Pipeline）
                    ↓
                TTSBackend（语音播报）

用法：
    from loongguard.voice import VoiceService
    service = VoiceService(config.voice, callbacks={...})
    await service.start()
    ...
    await service.stop()

模型文件（部署时放置，见 backend/models/{wakeword,pocketsphinx}/README.md）：
    - wakeword/my_wakeword.onnx    自训练唤醒词模型
    - pocketsphinx/zh-cn           中文声学模型
    - pocketsphinx/cmudict-cn.dict 中文词典
    - pocketsphinx/commands.jsgf   指令语法
"""

from loongguard.voice.command_handler import (
    SphinxCommandRecognizer,
    TextCommandDispatcher,
    match_command_key,
)
from loongguard.voice.tts_backend import (
    DummyTTS,
    EspeakTTS,
    Pyttsx3TTS,
    TTSBackend,
    create_tts_backend,
)
from loongguard.voice.voice_service import VoiceService
from loongguard.voice.wake_engine import (
    ArecordCapture,
    AudioCapture,
    DummyCapture,
    PyAudioCapture,
    WakeEngine,
    create_capture,
)

__all__ = [
    "VoiceService",
    "WakeEngine",
    "SphinxCommandRecognizer",
    "TextCommandDispatcher",
    "match_command_key",
    "AudioCapture",
    "ArecordCapture",
    "PyAudioCapture",
    "DummyCapture",
    "create_capture",
    "TTSBackend",
    "Pyttsx3TTS",
    "EspeakTTS",
    "DummyTTS",
    "create_tts_backend",
]
