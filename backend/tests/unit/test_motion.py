"""
帧差分运动检测模块单元测试

覆盖：
    - prev_gray=None 首帧返回空列表
    - 相同帧返回空列表
    - 显著差异帧检测到运动区域
    - 小面积轮廓 (< 100 像素) 被过滤
    - motion_ratio_threshold 过滤逻辑
    - MotionRegion dataclass 字段验证
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MotionConfig
from loongguard.motion.frame_diff import FrameDiffDetector, MotionRegion


class TestFrameDiffDetector:
    """帧差分运动检测器测试套件"""

    @pytest.fixture
    def config(self) -> MotionConfig:
        """默认运动检测配置"""
        return MotionConfig(
            diff_threshold=25,
            motion_ratio_threshold=0.01,
            blur_kernel_size=5,
            dilate_kernel_size=7,
        )

    @pytest.fixture
    def detector(self, config: MotionConfig) -> FrameDiffDetector:
        """使用默认配置的检测器实例"""
        return FrameDiffDetector(config)

    def test_prev_gray_none_returns_empty(
        self, detector: FrameDiffDetector
    ) -> None:
        """首帧（prev_gray=None）应返回空列表"""
        frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        result = detector.detect(frame, None)
        assert result == []

    def test_identical_frames_return_empty(
        self, detector: FrameDiffDetector
    ) -> None:
        """完全相同的两帧应返回空列表（无运动）"""
        frame = np.full((480, 640), 128, dtype=np.uint8)
        result = detector.detect(frame, frame.copy())
        assert result == []

    def test_significantly_different_frames_detect_motion(
        self, detector: FrameDiffDetector
    ) -> None:
        """显著不同的两帧应检测到运动区域"""
        # 创建一个大面积白色方块差异（确保面积 > 100 且 motion_ratio > 阈值）
        frame1 = np.full((480, 640), 50, dtype=np.uint8)
        frame2 = np.full((480, 640), 50, dtype=np.uint8)
        # 在 frame2 中画一个 200x200 的白色区域（灰度值 200）
        frame2[100:300, 200:400] = 200

        result = detector.detect(frame2, frame1)
        assert len(result) > 0
        for region in result:
            assert isinstance(region, MotionRegion)
            assert region.w > 0
            assert region.h > 0

    def test_small_area_contours_filtered(
        self, detector: FrameDiffDetector
    ) -> None:
        """面积 < 100 像素的轮廓应被过滤"""
        # 创建只有极小差异的两帧（单像素差异会被滤波消除或面积 < 100）
        frame1 = np.full((480, 640), 100, dtype=np.uint8)
        frame2 = frame1.copy()
        # 仅修改几个相邻像素，面积远 < 100
        frame2[240, 320] = 255
        frame2[241, 320] = 255
        frame2[240, 321] = 255

        result = detector.detect(frame2, frame1)
        # 微小差异经高斯模糊和面积过滤后应全部被移除
        assert result == []

    def test_motion_ratio_threshold_filtering(
        self, detector: FrameDiffDetector
    ) -> None:
        """运动区域的 motion_ratio 必须 >= motion_ratio_threshold"""
        frame1 = np.full((480, 640), 50, dtype=np.uint8)
        frame2 = np.full((480, 640), 50, dtype=np.uint8)
        # 画一个大面积差异区域
        frame2[50:250, 100:500] = 200

        result = detector.detect(frame2, frame1)
        for region in result:
            assert region.motion_ratio >= detector._config.motion_ratio_threshold

    def test_motion_region_dataclass_fields(self) -> None:
        """MotionRegion dataclass 应包含正确的字段"""
        region = MotionRegion(x=10, y=20, w=100, h=80, motion_ratio=0.05)
        assert region.x == 10
        assert region.y == 20
        assert region.w == 100
        assert region.h == 80
        assert region.motion_ratio == 0.05

    def test_high_threshold_filters_most_motion(
        self, config: MotionConfig
    ) -> None:
        """极高的 diff_threshold 应过滤掉大部分运动"""
        strict_config = MotionConfig(
            diff_threshold=200,
            motion_ratio_threshold=0.01,
            blur_kernel_size=config.blur_kernel_size,
            dilate_kernel_size=config.dilate_kernel_size,
        )
        detector = FrameDiffDetector(strict_config)

        # 灰度差异为 100，低于阈值 200
        frame1 = np.full((480, 640), 50, dtype=np.uint8)
        frame2 = np.full((480, 640), 150, dtype=np.uint8)  # diff = 100 < 200

        result = detector.detect(frame2, frame1)
        assert result == []

    def test_multiple_motion_regions_detected(
        self, detector: FrameDiffDetector
    ) -> None:
        """画面中多个不相邻的大面积变化区域应分别检测"""
        frame1 = np.full((480, 640), 50, dtype=np.uint8)
        frame2 = frame1.copy()
        # 左上角一个大区域
        frame2[20:120, 20:220] = 200
        # 右下角另一个大区域
        frame2[300:450, 400:600] = 200

        result = detector.detect(frame2, frame1)
        # 应至少检测到 2 个区域（形态学操作可能合并相邻区域，
        # 但这两个区域相距很远不应合并）
        assert len(result) >= 2

    def test_detect_with_zero_config_thresholds(
        self,
    ) -> None:
        """零阈值配置下任何灰度差异都应触发检测"""
        config = MotionConfig(
            diff_threshold=1,
            motion_ratio_threshold=0.0,
            blur_kernel_size=3,
            dilate_kernel_size=3,
        )
        detector = FrameDiffDetector(config)

        # 极小的均匀灰度差异
        frame1 = np.full((200, 200), 100, dtype=np.uint8)
        frame2 = np.full((200, 200), 110, dtype=np.uint8)

        result = detector.detect(frame2, frame1)
        # diff=10 > threshold=1，整幅画面都是运动区域
        assert len(result) > 0
