"""
Dynamic ROI 二级检测调度器

设计动机：
    系统的核心调度逻辑。接收运动检测区域，采用两级递进式检测策略：
    第一级（低分辨率 320）快速粗筛锁定可疑区域，
    第二级（高分辨率 640）在放大后的 ROI 子区域上精准分类。
    用算法策略弥补端侧算力与分辨率的矛盾。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import DetectionConfig, ROIConfig
from loongguard.detection.yolo26_nano import YOLO26Nano
from loongguard.motion.frame_diff import MotionRegion
from loongguard.utils.schema import AlertLog, AlertSeverity, AlertType, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class ROI:
    """待检测的感兴趣区域"""

    x: int
    y: int
    w: int
    h: int
    source: str  # "motion" / "stage1"


class ROIScheduler:
    """
    Dynamic ROI 二级检测调度器

    处理流程（修正后的两级递进策略）：
        1. 接收运动检测区域（MotionRegion 列表）
        2. 对运动区域裁剪，在 stage1_input_size（320）下快速粗筛
        3. 若粗筛有结果，将每个检测框扩展后裁剪为高分辨率 ROI
        4. 对 ROI 在 stage2_input_size（640）下精准检测
        5. stage2 结果作为最终告警依据

    设计动机：
        原始实现 stage1 和 stage2 对同一裁剪区域做两次相同推理，
        完全重复计算。修正后 stage2 在 stage1 检测框的扩展子区域上
        运行更高分辨率推理，实现真正的"粗筛 -> 精定位"两级递进。
    """

    def __init__(
        self, roi_config: ROIConfig, det_config: DetectionConfig, detector: YOLO26Nano
    ) -> None:
        self._roi_config = roi_config
        self._det_config = det_config
        self._detector = detector

    def process(
        self, frame: np.ndarray, motion_regions: list[MotionRegion]
    ) -> list[AlertLog]:
        """
        处理一帧中的所有运动区域

        Args:
            frame: 原始 RGB 帧 (H, W, 3)
            motion_regions: 帧差分检测到的运动区域

        Returns:
            告警列表
        """
        if not motion_regions:
            return []

        frame_h, frame_w = frame.shape[:2]
        alerts: list[AlertLog] = []
        rois = self._motion_to_rois(motion_regions)
        rois = rois[: self._roi_config.max_rois_per_frame]

        for roi in rois:
            # 裁剪运动区域
            roi_crop = self._crop_roi(frame, roi)
            if roi_crop is None:
                logger.debug("ROI crop 无效: roi=(%d,%d,%d,%d)", roi.x, roi.y, roi.w, roi.h)
                continue

            crop_h, crop_w = roi_crop.shape[:2]

            # 第一级：在运动区域裁剪上快速粗筛
            stage1_results = self._detector.infer_stage1(roi_crop)
            if stage1_results:
                logger.debug(
                    "stage1 检出 %d 个目标, ROI=%dx%d, classes=%s",
                    len(stage1_results), crop_w, crop_h,
                    [b.class_name for b in stage1_results],
                )

                # 第二级：对 stage1 检测到的每个目标区域做高分辨率精准检测
                for bbox in stage1_results:
                    stage2_bbox = self._run_stage2_on_detection(
                        frame, frame_w, frame_h, bbox, roi,
                    )
                    if stage2_bbox is not None:
                        logger.debug(
                            "stage2 确认: %s %.2f (%d,%d,%d,%d)",
                            stage2_bbox.class_name, stage2_bbox.confidence,
                            stage2_bbox.x1, stage2_bbox.y1, stage2_bbox.x2, stage2_bbox.y2,
                        )
                        alert = self._bbox_to_alert(stage2_bbox)
                        alerts.append(alert)
            else:
                # stage1 无结果：直接在 ROI 裁剪上运行 stage2 全分辨率兜底
                # 设计动机：stage1 分辨率（320px）过低或裁剪区域过小时，
                # 小物体容易漏检。stage2 以更高分辨率（640px）重新检测，
                # 作为召回率兜底策略。
                logger.debug(
                    "stage1 未检出, 尝试 stage2 兜底, ROI=%dx%d, pos=(%d,%d)",
                    crop_w, crop_h, roi.x, roi.y,
                )
                fallback_results = self._detector.infer_stage2(roi_crop)
                if fallback_results:
                    logger.info(
                        "stage2 兜底检出 %d 个目标, ROI=%dx%d, classes=%s",
                        len(fallback_results), crop_w, crop_h,
                        [b.class_name for b in fallback_results],
                    )
                    # 映射回原始帧坐标（ROI 裁剪起点为全局偏移）
                    best = max(fallback_results, key=lambda b: b.confidence)
                    mapped = BoundingBox(
                        x1=max(0, min(frame_w, roi.x + best.x1)),
                        y1=max(0, min(frame_h, roi.y + best.y1)),
                        x2=max(0, min(frame_w, roi.x + best.x2)),
                        y2=max(0, min(frame_h, roi.y + best.y2)),
                        confidence=best.confidence,
                        class_id=best.class_id,
                        class_name=best.class_name,
                    )
                    alerts.append(self._bbox_to_alert(mapped))

        return alerts

    def _run_stage2_on_detection(
        self,
        frame: np.ndarray,
        frame_w: int,
        frame_h: int,
        stage1_bbox: BoundingBox,
        roi: ROI,
    ) -> BoundingBox | None:
        """
        对 stage1 检测框进行高分辨率精准验证

        将 stage1 检测框坐标映射回原始帧空间，扩展边距后裁剪出
        高分辨率 ROI 子区域，送入 stage2 进行精确分类和定位。

        Args:
            frame: 原始 RGB 帧
            frame_w, frame_h: 帧尺寸
            stage1_bbox: stage1 在 roi_crop 上的检测结果
            roi: 当前运动区域对应的 ROI（提供全局偏移量）

        Returns:
            stage2 精准检测结果，映射到原始帧坐标。None 表示区域无效或未检出。
        """
        expand = self._roi_config.roi_expand_pixels

        # stage1 检测框坐标 -> 原始帧全局坐标
        global_x1 = roi.x + stage1_bbox.x1 - expand
        global_y1 = roi.y + stage1_bbox.y1 - expand
        global_x2 = roi.x + stage1_bbox.x2 + expand
        global_y2 = roi.y + stage1_bbox.y2 + expand

        # 边界裁剪
        global_x1 = max(0, global_x1)
        global_y1 = max(0, global_y1)
        global_x2 = min(frame_w, global_x2)
        global_y2 = min(frame_h, global_y2)

        if global_x2 <= global_x1 or global_y2 <= global_y1:
            return None

        # 高分辨率 ROI 裁剪
        stage2_crop = frame[global_y1:global_y2, global_x1:global_x2].copy()
        crop_h, crop_w = stage2_crop.shape[:2]
        if crop_w < 10 or crop_h < 10:
            return None

        # 第二级精准检测（在 stage1 检测框扩展区域上运行）
        stage2_results = self._detector.infer_stage2(stage2_crop)
        if not stage2_results:
            return None

        # 取置信度最高的结果，映射回原始帧坐标并 clamp 到帧边界
        best = max(stage2_results, key=lambda b: b.confidence)
        return BoundingBox(
            x1=max(0, min(frame_w, global_x1 + best.x1)),
            y1=max(0, min(frame_h, global_y1 + best.y1)),
            x2=max(0, min(frame_w, global_x1 + best.x2)),
            y2=max(0, min(frame_h, global_y1 + best.y2)),
            confidence=best.confidence,
            class_id=best.class_id,
            class_name=best.class_name,
        )

    def _motion_to_rois(self, regions: list[MotionRegion]) -> list[ROI]:
        """将运动检测区域转换为 ROI，并扩展边距"""
        rois: list[ROI] = []
        expand = self._roi_config.roi_expand_pixels

        for r in regions:
            area = r.w * r.h
            if area < self._roi_config.min_roi_area:
                continue

            rois.append(
                ROI(
                    x=max(0, r.x - expand),
                    y=max(0, r.y - expand),
                    w=r.w + 2 * expand,
                    h=r.h + 2 * expand,
                    source="motion",
                )
            )
        return rois

    def _crop_roi(self, frame: np.ndarray, roi: ROI) -> np.ndarray | None:
        """从帧中裁剪 ROI 区域"""
        h, w = frame.shape[:2]
        x1 = max(0, roi.x)
        y1 = max(0, roi.y)
        x2 = min(w, roi.x + roi.w)
        y2 = min(h, roi.y + roi.h)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def _bbox_to_alert(self, bbox: BoundingBox) -> AlertLog:
        """将检测框转换为告警日志"""
        severity = AlertSeverity.HIGH
        if bbox.class_name in ("magnetic_bead", "button_battery"):
            severity = AlertSeverity.CRITICAL

        return AlertLog(
            alert_type=AlertType.DANGEROUS_OBJECT,
            severity=severity,
            detections=[bbox],
            description=f"检测到危险物品: {bbox.class_name} (置信度: {bbox.confidence:.2f})",
        )
