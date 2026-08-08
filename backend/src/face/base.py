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
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.utils.schema import BoundingBox


class FaceDetector(ABC):
    """
    人脸检测器协议

    所有实现需满足：
        - is_available(): 是否已加载模型可执行推理
        - detect(image): 返回人脸检测框列表（BoundingBox）
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

    def has_face(self, image: np.ndarray) -> bool:
        """便捷方法：图像中是否至少存在一张人脸"""
        return len(self.detect(image)) > 0