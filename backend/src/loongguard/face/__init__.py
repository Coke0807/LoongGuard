"""人脸检测模块（跨平台预留扩展）"""

from __future__ import annotations

from config import FaceConfig
from loongguard.face.base import FaceDetection, FaceDetector
from loongguard.face.dummy import DummyFaceDetector
from loongguard.face.factory import create_face_detector
from loongguard.face.head_pose import HeadPose, HeadState, estimate_head_pose
from loongguard.face.posture import (
    PostureResult,
    SleepPosture,
    SleepPostureClassifier,
)

__all__ = [
    "FaceConfig",
    "FaceDetection",
    "FaceDetector",
    "DummyFaceDetector",
    "create_face_detector",
    "HeadPose",
    "HeadState",
    "estimate_head_pose",
    "PostureResult",
    "SleepPosture",
    "SleepPostureClassifier",
]
