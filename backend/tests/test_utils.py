"""
测试工具模块

提供测试所需的数据生成和断言辅助函数。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from loongguard.utils.schema import AlertLog, AlertSeverity, AlertType, BoundingBox


def generate_random_frame(
    width: int = 640,
    height: int = 480,
    channels: int = 3,
) -> np.ndarray:
    """
    生成随机测试帧

    Args:
        width: 帧宽度
        height: 帧高度
        channels: 通道数（3=RGB, 1=灰度）

    Returns:
        随机噪声帧
    """
    if channels == 1:
        return np.random.randint(0, 255, (height, width), dtype=np.uint8)
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)


def generate_bounding_box(
    x1: int = 100,
    y1: int = 100,
    x2: int = 200,
    y2: int = 200,
    confidence: float = 0.9,
    class_id: int = 0,
    class_name: str = "scissor",
) -> BoundingBox:
    """
    生成测试用的边界框

    Args:
        x1, y1, x2, y2: 边界框坐标
        confidence: 置信度
        class_id: 类别 ID
        class_name: 类别名称

    Returns:
        BoundingBox 实例
    """
    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
    )


def generate_alert_log(
    alert_type: AlertType = AlertType.DANGEROUS_OBJECT,
    severity: AlertSeverity = AlertSeverity.HIGH,
    detections: list[BoundingBox] | None = None,
    description: str = "Test alert",
) -> AlertLog:
    """
    生成测试用的告警日志

    Args:
        alert_type: 告警类型
        severity: 告警严重等级
        detections: 检测框列表
        description: 告警描述

    Returns:
        AlertLog 实例
    """
    if detections is None:
        detections = [generate_bounding_box()]

    return AlertLog(
        alert_type=alert_type,
        severity=severity,
        detections=detections,
        description=description,
    )


def assert_bounding_box_equal(
    box1: BoundingBox,
    box2: BoundingBox,
    tolerance: float = 0.01,
) -> bool:
    """
    断言两个边界框相等

    Args:
        box1: 第一个边界框
        box2: 第二个边界框
        tolerance: 坐标容差（像素）

    Returns:
        True if equal, False otherwise
    """
    return (
        abs(box1.x1 - box2.x1) <= tolerance
        and abs(box1.y1 - box2.y1) <= tolerance
        and abs(box1.x2 - box2.x2) <= tolerance
        and abs(box1.y2 - box2.y2) <= tolerance
        and abs(box1.confidence - box2.confidence) <= 0.001
        and box1.class_id == box2.class_id
        and box1.class_name == box2.class_name
    )


def assert_alert_log_equal(
    alert1: AlertLog,
    alert2: AlertLog,
    ignore_timestamp: bool = True,
    ignore_alert_id: bool = True,
) -> bool:
    """
    断言两个告警日志相等

    Args:
        alert1: 第一个告警日志
        alert2: 第二个告警日志
        ignore_timestamp: 是否忽略时间戳
        ignore_alert_id: 是否忽略告警 ID

    Returns:
        True if equal, False otherwise
    """
    if not ignore_alert_id and alert1.alert_id != alert2.alert_id:
        return False

    if alert1.alert_type != alert2.alert_type:
        return False

    if alert1.severity != alert2.severity:
        return False

    if alert1.description != alert2.description:
        return False

    if len(alert1.detections) != len(alert2.detections):
        return False

    for det1, det2 in zip(alert1.detections, alert2.detections):
        if not assert_bounding_box_equal(det1, det2):
            return False

    return True


def create_mock_detection_result(
    num_detections: int = 1,
    class_names: list[str] | None = None,
) -> list[BoundingBox]:
    """
    创建模拟的检测结果

    Args:
        num_detections: 检测数量
        class_names: 类别名称列表（循环使用）

    Returns:
        BoundingBox 列表
    """
    if class_names is None:
        class_names = ["scissor", "utility_knife", "needle"]

    detections = []
    for i in range(num_detections):
        class_name = class_names[i % len(class_names)]
        detections.append(
            generate_bounding_box(
                x1=100 + i * 50,
                y1=100 + i * 50,
                x2=200 + i * 50,
                y2=200 + i * 50,
                confidence=0.8 + i * 0.05,
                class_id=i % len(class_names),
                class_name=class_name,
            )
        )

    return detections
