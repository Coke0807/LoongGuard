"""
LoongGuard 告警日志 Schema

设计动机：
    定义统一的告警数据结构，所有检测模块产出的告警均封装为 AlertLog，
    供加密存储、API 推送、声光告警层消费。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class AlertSeverity(str, Enum):
    """告警严重等级"""

    LOW = "low"           # 提醒：如环境轻微异常
    MEDIUM = "medium"     # 警告：如非致命危险物品
    HIGH = "high"         # 严重：如磁力珠、纽扣电池
    CRITICAL = "critical" # 紧急：如窒息风险（俯卧睡姿持续）


class AlertType(str, Enum):
    """告警类型"""

    DANGEROUS_OBJECT = "dangerous_object"       # 危险物品手持
    PRONE_SLEEP = "prone_sleep"                 # 俯卧趴睡
    ENVIRONMENT = "environment"                 # 环境合规风险
    MOTION_ABNORMAL = "motion_abnormal"         # 异常运动


@dataclass
class BoundingBox:
    """检测框"""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


@dataclass
class AlertLog:
    """
    告警日志条目

    所有检测模块统一输出此结构，下游消费者（加密存储、API、声光告警）
    仅依赖此数据类，不直接接触原始检测结果。
    """

    # 唯一标识
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # UTC 时间戳
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # 告警类型
    alert_type: AlertType = AlertType.DANGEROUS_OBJECT
    # 严重等级
    severity: AlertSeverity = AlertSeverity.MEDIUM
    # 检测到的框（可能多个）
    detections: list[BoundingBox] = field(default_factory=list)
    # 风险切片图路径（SM4 加密后），None 表示无切片
    slice_path: Optional[str] = None
    # 人类可读描述
    description: str = ""
    # 是否已确认/消警
    acknowledged: bool = False

    def to_dict(self) -> dict:
        """序列化为 dict，用于 JSON 编码"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AlertLog:
        """从 dict 反序列化"""
        detections = [
            BoundingBox(**d) for d in data.get("detections", [])
        ]
        return cls(
            alert_id=data.get("alert_id", uuid.uuid4().hex[:12]),
            timestamp=data.get("timestamp", ""),
            alert_type=AlertType(data.get("alert_type", "dangerous_object")),
            severity=AlertSeverity(data.get("severity", "medium")),
            detections=detections,
            slice_path=data.get("slice_path"),
            description=data.get("description", ""),
            acknowledged=data.get("acknowledged", False),
        )
