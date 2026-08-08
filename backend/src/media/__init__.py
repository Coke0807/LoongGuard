"""媒体能力模块（语音 + 流媒体，跨平台预留扩展）"""

from __future__ import annotations

from src.media.audio import (
    AudioBackend,
    CommandAudioBackend,
    DummyAudioBackend,
    create_audio_backend,
)
from src.media.stream_publisher import (
    MJPEGStreamPublisher,
    StreamPublisher,
    WebRTCStreamPublisher,
    create_stream_publisher,
)

__all__ = [
    "AudioBackend",
    "CommandAudioBackend",
    "DummyAudioBackend",
    "create_audio_backend",
    "StreamPublisher",
    "MJPEGStreamPublisher",
    "WebRTCStreamPublisher",
    "create_stream_publisher",
]