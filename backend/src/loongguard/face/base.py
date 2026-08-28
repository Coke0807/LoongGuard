"""
人脸检测抽象接口（跨平台预留）

设计动机：
    儿童睡姿监测依赖"Person bbox 内无 Face -> 异常睡姿"的判定逻辑。
    实际的人脸检测模型权重尚未提供（板端后续补充），本模块定义统一
    FaceDetector 接口，Windows 开发期用 DummyFaceDetector 桩，板端
    切换真实 ONNX 实现，业务代码（pipeline）不感知底层差异。

扩展方式：
    1. 新增 ONNX 实现类，实现 FaceDetector.detect()
    2. 在 factory.create_face_detector() 中按配置返回对应实现
    （模型文件放入 backend/models/ 即可，无需改动 pipeline 调用方）

头部姿态增强（landmarks）：
    det_10g.onnx (SCRFD) 除人脸框外还输出 5 点关键点（双眼、鼻、双嘴角）。
    detect_with_landmarks() 返回带关键点的 FaceDetection，供头部姿态估计
    （head_pose.py）与睡姿分类（posture.py）使用。默认实现将 landmarks
    置为 None，桩实现无需感知关键点。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from loongguard.utils.schema import BoundingBox


@dataclass
class FaceDetection:
    """
    单张人脸的检测结果（框 + 可选关键点）

    Attributes:
        bbox: 人脸检测框（原图像素坐标）
        landmarks: 5 点关键点 (5, 2) [x, y]，顺序为
            [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]；
            模型/实现不支持时为 None
    """

    bbox: BoundingBox
    landmarks: np.ndarray | None = None


class FaceDetector(ABC):
    """
    人脸检测器协议

    所有实现需满足：
        - is_available(): 是否已加载模型可执行推理
        - detect(image): 返回人脸检测框列表（BoundingBox）

    可选能力：
        - detect_with_landmarks(image): 返回带 5 点关键点的 FaceDetection 列表，
          用于头部姿态估计与睡姿分类。默认实现基于 detect() 退化。
    """

    @abstractmethod
    def is_available(self) -> bool:
        """是否可用（模型加载成功）"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """
        检测图像中的人脸

        Args:
            image: RGB 图像 (H, W, 3)，uint8

        Returns:
            人脸检测框列表；无可检测到的人脸时返回空列表
        """

    def detect_with_landmarks(self, image: np.ndarray) -> list[FaceDetection]:
        """
        检测人脸并返回 5 点关键点（可选能力）

        默认实现：调用 detect()，landmarks 置为 None。
        支持关键点的实现（如 ONNXFaceDetector）应重写此方法。
        """
        return [FaceDetection(bbox=b, landmarks=None) for b in self.detect(image)]

    def has_face(self, image: np.ndarray) -> bool:
        """便捷方法：图像中是否至少存在一张人脸"""
        return len(self.detect(image)) > 0
