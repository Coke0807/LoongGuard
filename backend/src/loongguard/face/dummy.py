"""
人脸检测桩实现（开发期 / 无模型兜底）

设计动机：
    板端人脸检测权重尚未补充，为避免阻塞 Windows 开发与 MVP 演示，
    提供 DummyFaceDetector 桩：默认返回一整帧"人脸框"，即认为画面中
    始终检测到人脸。对睡姿判定而言，其效果是"不触发异常告警"，
    保证链路可跑通且不产生误报。
"""

from __future__ import annotations

import logging

import numpy as np

from loongguard.face.base import FaceDetector
from loongguard.utils.schema import BoundingBox

logger = logging.getLogger(__name__)


class DummyFaceDetector(FaceDetector):
    """
    桩实现：始终认为检测到人脸

    使用场景：
        - Windows 开发期验证链路
        - 板端尚未部署真实人脸模型时的安全兜底

    替换为真实实现时，仅需在 factory.create_face_detector() 中
    返回基于 ONNX 的实现，对外行为一致。
    """

    def is_available(self) -> bool:
        # 桩实现恒可用
        return True

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        h, w = image.shape[:2]
        # 返回一整帧人脸框（置信度 1.0），表示"检测到人脸，安全"
        logger.debug(
            "DummyFaceDetector 返回整帧人脸框 (%dx%d)", w, h
        )
        return [
            BoundingBox(
                x1=0,
                y1=0,
                x2=w,
                y2=h,
                confidence=1.0,
                class_id=0,
                class_name="face",
            )
        ]