"""
测试配置与公共 fixtures

设计动机：
    集中管理测试所需的 mock 对象和公共配置，
    避免各测试文件重复创建相同的 fixture。

依赖标准 pip install -e 安装方式，不再使用 sys.path.insert hack。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# 项目根目录（用于构建模型文件路径）
PROJECT_ROOT = Path(__file__).parent.parent


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


@pytest.fixture
def mock_detector():
    """
    Mock 目标检测器

    返回一个模拟的 YOLO26Nano 检测器，用于测试不依赖真实模型。
    """
    from loongguard.utils.schema import BoundingBox

    detector = MagicMock()
    detector.is_available.return_value = True
    detector.infer.return_value = [
        BoundingBox(
            x1=100, y1=100, x2=200, y2=200,
            confidence=0.95,
            class_id=0,
            class_name="scissor",
        )
    ]
    detector.infer_stage1.return_value = detector.infer.return_value
    detector.infer_stage2.return_value = detector.infer.return_value
    detector.get_health_snapshot.return_value = {
        "available": True,
        "model_path": "mock_model.onnx",
        "input_size": 640,
        "conf_threshold": 0.5,
        "last_inference_ts": 0.0,
        "success_count": 0,
        "error_count": 0,
        "last_error": "",
    }
    return detector


@pytest.fixture
def mock_motion_detector():
    """
    Mock 运动检测器

    返回一个模拟的 FrameDiffDetector，用于测试不依赖真实算法。
    """
    from loongguard.motion.frame_diff import MotionRegion

    detector = MagicMock()
    detector.detect.return_value = [
        MotionRegion(x=100, y=100, w=100, h=100, motion_ratio=0.1)
    ]
    return detector


@pytest.fixture
def mock_camera():
    """
    Mock 摄像头

    返回一个模拟的 V4L2Capture，用于测试不依赖真实硬件。
    """
    from loongguard.camera.v4l2_capture import Frame

    camera = MagicMock()
    camera.is_opened = True
    camera.frame_count = 0

    def read_side_effect():
        camera.frame_count += 1
        return Frame(
            data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            timestamp=camera.frame_count / 30.0,
            frame_id=camera.frame_count,
        )

    camera.read.side_effect = read_side_effect
    return camera


@pytest.fixture
def mock_database():
    """
    Mock 数据库

    返回一个模拟的 AlertDatabase，用于测试不依赖真实数据库。
    """
    db = MagicMock()
    db.initialize.return_value = None
    db.close.return_value = None
    db.insert_alert.return_value = None
    db.query_alerts.return_value = {
        "total": 0,
        "page": 1,
        "page_size": 20,
        "data": [],
    }
    db.ack_alert.return_value = True
    db.cleanup_old_alerts.return_value = 0
    db.backup.return_value = True
    db.get_stats.return_value = {
        "total": 0,
        "by_severity": {},
        "by_type": {},
        "acknowledged": 0,
        "unacknowledged": 0,
    }
    return db


@pytest.fixture
def mock_alarm():
    """
    Mock 告警触发器

    返回一个模拟的 GPIOAlarmTrigger，用于测试不依赖真实 GPIO。
    """
    alarm = MagicMock()
    alarm.setup.return_value = None
    alarm.cleanup.return_value = None
    alarm.trigger.return_value = None
    return alarm


@pytest.fixture
def mock_crypto():
    """
    Mock 加密日志管理器

    返回一个模拟的 SM4Logger，用于测试不依赖真实加密。
    """
    crypto = MagicMock()
    crypto.load_key.return_value = None
    crypto.encrypt_and_store.return_value = "/tmp/mock_encrypted.enc"
    crypto.encrypt_slice.return_value = "/tmp/mock_slice.enc"
    return crypto


@pytest.fixture
def mock_notifier():
    """
    Mock 告警通知器

    返回一个模拟的 AlertNotifier，用于测试不依赖真实通知服务。
    """
    notifier = MagicMock()
    notifier.start.return_value = None
    notifier.stop.return_value = None
    notifier.notify.return_value = None
    return notifier
