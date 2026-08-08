"""
帧差分运动检测

设计动机：
    在 YOLO26 全局推理之前，先用轻量帧差分算法快速判断
    "画面是否有变化"，只在检测到运动的区域触发 Dynamic ROI 截取。
    避免对每一帧都进行全图高分辨率推理，大幅节省 LG200 算力。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

# cv2 在 LoongArch 定制版中可能路径非标准，延迟导入便于测试时 mock。
# 注意：import 必须在 class 定义之前完成，否则运行时 detect() 内的 cv2 调用
# 会引用到未赋值的模块级变量。
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from config import MotionConfig

logger = logging.getLogger(__name__)


@dataclass
class MotionRegion:
    """运动区域"""

    x: int
    y: int
    w: int
    h: int
    # 运动像素占比
    motion_ratio: float


class FrameDiffDetector:
    """
    帧差分运动检测器

    算法流程：
        1. 当前帧灰度化
        2. 与上一帧做绝对差分
        3. 阈值二值化
        4. 高斯模糊降噪 + 膨胀填补空洞
        5. 寻找轮廓，返回运动区域
    """

    def __init__(self, config: MotionConfig) -> None:
        self._config = config

    def detect(
        self, current_gray: np.ndarray, prev_gray: Optional[np.ndarray]
    ) -> list[MotionRegion]:
        """
        检测当前帧与上一帧之间的运动区域

        Args:
            current_gray: 当前帧灰度图 (H, W), dtype=uint8
            prev_gray: 上一帧灰度图 (H, W), dtype=uint8，None 表示首帧

        Returns:
            运动区域列表。空列表表示无显著运动。
        """
        if prev_gray is None:
            return []

        if cv2 is None:
            logger.error("cv2 unavailable, cannot run motion detection")
            return []

        # 帧差分
        diff = cv2.absdiff(current_gray, prev_gray)

        # 二值化
        _, thresh = cv2.threshold(
            diff, self._config.diff_threshold, 255, cv2.THRESH_BINARY
        )

        # 高斯模糊降噪
        k = self._config.blur_kernel_size
        thresh = cv2.GaussianBlur(thresh, (k, k), 0)

        # 膨胀填补运动区域空洞
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._config.dilate_kernel_size, self._config.dilate_kernel_size),
        )
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # 寻找轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        total_pixels = current_gray.shape[0] * current_gray.shape[1]
        regions: list[MotionRegion] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 过小区域忽略
                continue

            x, y, w, h = cv2.boundingRect(contour)
            motion_ratio = area / total_pixels

            if motion_ratio >= self._config.motion_ratio_threshold:
                regions.append(
                    MotionRegion(x=x, y=y, w=w, h=h, motion_ratio=motion_ratio)
                )

        logger.debug("Motion detection: %d regions found", len(regions))
        return regions
