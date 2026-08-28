"""
测试配置与公共 fixtures

设计动机：
    集中管理测试所需的 mock 对象和公共配置，
    避免各测试文件重复创建相同的 fixture。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 将项目根目录与 src 目录加入 sys.path（src-layout：包位于 backend/src/loongguard）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def sample_frame_rgb() -> np.ndarray:
    """生成一个 640x480 RGB 测试帧（随机噪声）"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_gray() -> np.ndarray:
    """生成一个 640x480 灰度测试帧"""
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)


@pytest.fixture
def detection_config():
    """返回测试用的检测配置（路径指向 models/ 目录下模型）"""
    from config.settings import DetectionConfig
    return DetectionConfig(
        model_path=str(PROJECT_ROOT / "models" / "best.onnx"),
        input_size=640,
        conf_threshold=0.5,
        quantized=False,
    )


@pytest.fixture
def motion_config():
    """返回测试用的运动检测配置"""
    from config.settings import MotionConfig
    return MotionConfig()


@pytest.fixture
def app_config():
    """返回完整的测试用配置（使用真实 dummy 模型路径）"""
    from config.settings import AppConfig
    config = AppConfig()
    config.detection.model_path = str(PROJECT_ROOT / "models" / "best.onnx")
    config.detection.input_size = 640
    config.debug = True
    return config
