"""
人脸检测器工厂（跨平台选择）

设计动机：
    根据配置返回对应的人脸检测实现，Windows 开发与板端运行共用同一入口：

        - backend="dummy" 或模型文件缺失 -> DummyFaceDetector（桩，不阻塞链路）
        - backend="onnx"   且模型存在   -> 真实 ONNX 实现（板端后续接入）

    板端接入真实模型时，在此工厂中新增 ONNX 实现分支即可，
    无需改动 pipeline 等调用方。
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import FaceConfig
from loongguard.face.base import FaceDetector
from loongguard.face.dummy import DummyFaceDetector

logger = logging.getLogger(__name__)


def create_face_detector(config: FaceConfig) -> FaceDetector:
    """
    依据配置创建人脸检测器

    Args:
        config: 人脸检测配置（FaceConfig）

    Returns:
        FaceDetector 实现实例
    """
    # 显式指定 ONNX 后端但模型文件存在时，接入真实实现。
    # 当前真实实现未提供，先回退到桩并告警，避免静默失败。
    if config.backend == "onnx":
        model_path = Path(config.model_path)
        if model_path.exists():
            try:
                from loongguard.face.onnx_detector import ONNXFaceDetector

                logger.info("创建 ONNX 人脸检测器: %s", config.model_path)
                return ONNXFaceDetector(config)
            except Exception as exc:  # pragma: no cover - 防御性兜底
                logger.warning(
                    "ONNX 人脸检测器创建失败，回退到桩实现: %s", exc
                )
        else:
            logger.warning(
                "人脸模型不存在(%s)，使用 DummyFaceDetector 桩",
                config.model_path,
            )
    else:
        logger.info("人脸检测后端=%s，使用 DummyFaceDetector 桩", config.backend)

    return DummyFaceDetector()
