"""
ROIScheduler 单元测试

设计动机：
    验证 Dynamic ROI 二级检测调度器的核心逻辑：
    运动区域过滤、ROI 裁剪边界处理、告警严重等级映射、
    以及 max_rois_per_frame 限制等。
    通过 mock detector 隔离 YOLO26 推理依赖。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from config import DetectionConfig, ROIConfig
from loongguard.motion.frame_diff import MotionRegion
from loongguard.roi.roi_scheduler import ROI, ROIScheduler
from loongguard.utils.schema import AlertSeverity, AlertType, BoundingBox


# ── 工厂函数 ──────────────────────────────────────────────────


def _make_bbox(class_name: str = "scissors", confidence: float = 0.9) -> BoundingBox:
    """快速构造一个测试用 BoundingBox"""
    return BoundingBox(
        x1=10, y1=20, x2=50, y2=60,
        confidence=confidence,
        class_id=0,
        class_name=class_name,
    )


def _make_motion_region(
    x: int = 100, y: int = 100, w: int = 50, h: int = 50, ratio: float = 0.05
) -> MotionRegion:
    """快速构造一个 MotionRegion"""
    return MotionRegion(x=x, y=y, w=w, h=h, motion_ratio=ratio)


def _make_scheduler(
    roi_overrides: dict | None = None,
    stage1_results: list | None = None,
    stage2_results: list | None = None,
) -> tuple[ROIScheduler, MagicMock]:
    """
    构造 ROIScheduler 实例和对应的 mock detector。

    Returns:
        (scheduler, mock_detector) 元组
    """
    roi_cfg = ROIConfig(**(roi_overrides or {}))
    det_cfg = DetectionConfig()

    mock_detector = MagicMock()
    mock_detector.infer_stage1.return_value = stage1_results or []
    mock_detector.infer_stage2.return_value = stage2_results or []

    scheduler = ROIScheduler(roi_cfg, det_cfg, mock_detector)
    return scheduler, mock_detector


# ── ROIScheduler.process() 测试 ───────────────────────────────


class TestROISchedulerProcess:

    def test_empty_motion_regions_returns_empty_alerts(self):
        """无运动区域时应返回空告警列表"""
        scheduler, mock_det = _make_scheduler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        alerts = scheduler.process(frame, motion_regions=[])
        assert alerts == []
        mock_det.infer_stage1.assert_not_called()

    def test_small_motion_regions_filtered_by_min_area(self):
        """面积低于 min_roi_area 的运动区域应被过滤"""
        scheduler, mock_det = _make_scheduler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        small_region = _make_motion_region(x=10, y=10, w=10, h=10)

        alerts = scheduler.process(frame, [small_region])
        assert alerts == []
        mock_det.infer_stage1.assert_not_called()

    def test_stage1_empty_skips_stage2(self):
        """第一级粗筛无结果时，不应调用第二级"""
        scheduler, mock_det = _make_scheduler(stage1_results=[])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        region = _make_motion_region(x=100, y=100, w=50, h=50)

        alerts = scheduler.process(frame, [region])
        assert alerts == []
        mock_det.infer_stage1.assert_called_once()
        mock_det.infer_stage2.assert_not_called()

    def test_stage1_and_stage2_produce_alerts(self):
        """两级检测均有结果时，应生成告警（stage2 在 stage1 检测框扩展区域上执行）"""
        stage1_bbox = _make_bbox(class_name="scissors", confidence=0.7)
        stage2_bbox = _make_bbox(class_name="scissors", confidence=0.95)
        scheduler, mock_det = _make_scheduler(
            stage1_results=[stage1_bbox], stage2_results=[stage2_bbox]
        )
        # 需要足够大的帧让 stage1 bbox 扩展后的裁剪区域有效
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        region = _make_motion_region(x=100, y=100, w=200, h=200)

        alerts = scheduler.process(frame, [region])
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.DANGEROUS_OBJECT
        assert alerts[0].detections[0].class_name == "scissors"

    def test_max_rois_per_frame_limits_processing(self):
        """超过 max_rois_per_frame 的 ROI 应被截断"""
        scheduler, mock_det = _make_scheduler(
            roi_overrides={"max_rois_per_frame": 2},
            stage1_results=[_make_bbox()],
            stage2_results=[_make_bbox()],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = [_make_motion_region(x=i * 100, y=100, w=50, h=50) for i in range(4)]

        scheduler.process(frame, regions)
        assert mock_det.infer_stage1.call_count == 2

    def test_multiple_regions_multiple_alerts(self):
        """多个运动区域各自独立执行两级检测"""
        bbox1 = _make_bbox(class_name="needle")
        bbox2 = _make_bbox(class_name="glass_shard")
        # 每次 stage1 返回一个 bbox，stage2 也返回一个
        scheduler, mock_det = _make_scheduler(stage1_results=[bbox1], stage2_results=[bbox2])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        r1 = _make_motion_region(x=100, y=100, w=200, h=200)
        r2 = _make_motion_region(x=400, y=200, w=200, h=200)

        alerts = scheduler.process(frame, [r1, r2])
        assert len(alerts) == 2


# ── _crop_roi 边界测试 ────────────────────────────────────────


class TestCropROI:

    def test_normal_crop(self):
        """正常区域内裁剪，输出尺寸应与 ROI 匹配"""
        scheduler, _ = _make_scheduler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = ROI(x=100, y=100, w=50, h=30, source="motion")

        crop = scheduler._crop_roi(frame, roi)
        assert crop is not None
        assert crop.shape == (30, 50, 3)

    def test_roi_extends_beyond_frame_right_bottom(self):
        """ROI 超出帧右下边界时，应被裁剪到帧边缘"""
        scheduler, _ = _make_scheduler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = ROI(x=600, y=450, w=200, h=100, source="motion")

        crop = scheduler._crop_roi(frame, roi)
        assert crop is not None
        assert crop.shape == (30, 40, 3)

    def test_roi_extends_beyond_frame_left_top(self):
        """ROI 坐标为负数时，应从 0 开始裁剪"""
        scheduler, _ = _make_scheduler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = ROI(x=-20, y=-10, w=100, h=80, source="motion")

        crop = scheduler._crop_roi(frame, roi)
        assert crop is not None
        assert crop.shape == (70, 80, 3)

    def test_zero_area_roi_returns_none(self):
        """裁剪后宽或高为 0 的 ROI 应返回 None"""
        scheduler, _ = _make_scheduler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = ROI(x=700, y=100, w=50, h=50, source="motion")

        crop = scheduler._crop_roi(frame, roi)
        assert crop is None

    def test_crop_returns_copy(self):
        """裁剪结果应是副本，修改不影响原帧"""
        scheduler, _ = _make_scheduler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = ROI(x=0, y=0, w=10, h=10, source="motion")

        crop = scheduler._crop_roi(frame, roi)
        crop[:] = 255
        assert frame[0, 0, 0] == 0


# ── _bbox_to_alert 严重等级测试 ───────────────────────────────


class TestBboxToAlert:

    @pytest.mark.parametrize(
        "class_name,expected_severity",
        [
            ("magnetic_bead", AlertSeverity.CRITICAL),
            ("button_battery", AlertSeverity.CRITICAL),
        ],
    )
    def test_critical_severity_objects(self, class_name: str, expected_severity: AlertSeverity):
        """磁力珠和纽扣电池应为 CRITICAL 级别"""
        scheduler, _ = _make_scheduler()
        bbox = _make_bbox(class_name=class_name)

        alert = scheduler._bbox_to_alert(bbox)
        assert alert.severity == expected_severity
        assert alert.alert_type == AlertType.DANGEROUS_OBJECT

    @pytest.mark.parametrize(
        "class_name",
        ["scissors", "utility_knife", "needle", "glass_shard", "wire", "small_toy_part"],
    )
    def test_high_severity_objects(self, class_name: str):
        """其他危险物品应为 HIGH 级别"""
        scheduler, _ = _make_scheduler()
        bbox = _make_bbox(class_name=class_name)

        alert = scheduler._bbox_to_alert(bbox)
        assert alert.severity == AlertSeverity.HIGH

    def test_alert_description_contains_class_name(self):
        """告警描述中应包含检测到的物品名称"""
        scheduler, _ = _make_scheduler()
        bbox = _make_bbox(class_name="needle", confidence=0.87)

        alert = scheduler._bbox_to_alert(bbox)
        assert "needle" in alert.description
        assert "0.87" in alert.description

    def test_alert_preserves_detections(self):
        """告警应携带原始 bbox 检测列表"""
        scheduler, _ = _make_scheduler()
        bbox = _make_bbox()

        alert = scheduler._bbox_to_alert(bbox)
        assert len(alert.detections) == 1
        assert alert.detections[0] is bbox


# ── _motion_to_rois 扩展逻辑测试 ─────────────────────────────


class TestMotionToRois:

    def test_roi_expand_pixels_applied(self):
        """运动区域应向四周扩展 roi_expand_pixels 像素"""
        scheduler, _ = _make_scheduler(roi_overrides={
            "roi_expand_pixels": 20, "min_roi_area": 1,
        })
        region = _make_motion_region(x=100, y=200, w=50, h=40)

        rois = scheduler._motion_to_rois([region])
        assert len(rois) == 1
        roi = rois[0]
        assert roi.x == 80
        assert roi.y == 180
        assert roi.w == 90
        assert roi.h == 80
        assert roi.source == "motion"

    def test_roi_expand_clamped_to_zero(self):
        """靠近帧左上角的区域，扩展后坐标不应为负数"""
        scheduler, _ = _make_scheduler()
        region = _make_motion_region(x=5, y=5, w=50, h=50)

        rois = scheduler._motion_to_rois([region])
        roi = rois[0]
        assert roi.x == 0
        assert roi.y == 0

    def test_small_regions_filtered(self):
        """面积小于 min_roi_area 的运动区域应被过滤"""
        scheduler, _ = _make_scheduler(roi_overrides={
            "roi_expand_pixels": 20, "min_roi_area": 400,
        })
        small = _make_motion_region(x=10, y=10, w=10, h=10)
        large = _make_motion_region(x=100, y=100, w=100, h=100)

        rois = scheduler._motion_to_rois([small, large])
        assert len(rois) == 1
        assert rois[0].x == 80

    def test_empty_regions_returns_empty(self):
        """空列表输入应返回空 ROI 列表"""
        scheduler, _ = _make_scheduler()
        rois = scheduler._motion_to_rois([])
        assert rois == []
