"""人脸检测模块（跨平台预留扩展）"""

from __future__ import annotations

from config import FaceConfig
from src.face.base import FaceDetector
from src.face.dummy import DummyFaceDetector
from src.face.factory import create_face_detector

__all__ = [
    "FaceConfig",
    "FaceDetector",
    "DummyFaceDetector",
    "create_face_detector",
]