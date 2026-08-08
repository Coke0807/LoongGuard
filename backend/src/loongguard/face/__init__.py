"""人脸检测模块（跨平台预留扩展）"""

from __future__ import annotations

from config import FaceConfig
from loongguard.face.base import FaceDetector
from loongguard.face.dummy import DummyFaceDetector
from loongguard.face.factory import create_face_detector

__all__ = [
    "FaceConfig",
    "FaceDetector",
    "DummyFaceDetector",
    "create_face_detector",
]