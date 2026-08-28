"""
睡姿分类器（人脸状态 -> 睡姿判定）

设计动机：
    旧版睡姿判定为二值逻辑："画面有运动但未检测到人脸 -> 异常睡姿"，
    且单帧即触发告警，易受瞬时漏检干扰。

    本模块在人脸检测基础上引入头部姿态（head_pose.py），将睡姿从
    二值升级为多分类，并对"人脸不可见"增加连续帧持续性过滤：

        - FACE_UP     正脸 / 脸朝上 -> 安全（仰卧）
        - SIDE        侧脸         -> 侧卧，低风险，不告警
        - TILTED      头倾斜       -> 正常睡姿变体，不告警
        - NOT_VISIBLE 人脸不可见   -> 疑似俯卧/遮挡，持续 N 帧触发 CRITICAL

    告警策略：
        仅当连续 not_visible_frames_threshold 帧未检测到任何人脸时
        触发 PRONE_SLEEP 告警，避免单帧漏检误报。触发后清空缓冲
        （下游 dedup 另有 30s 时间窗抑制重复告警）。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

from config import FaceConfig
from loongguard.face.base import FaceDetection
from loongguard.face.head_pose import HeadPose, HeadState, estimate_head_pose
from loongguard.utils.schema import AlertLog, AlertSeverity, AlertType, BoundingBox

logger = logging.getLogger(__name__)


class SleepPosture(str, Enum):
    """睡姿分类"""

    FACE_UP = "face_up"        # 正脸 / 脸朝上（安全）
    SIDE = "side"              # 侧脸（侧卧，低风险）
    TILTED = "tilted"          # 头倾斜（正常睡姿变体）
    NOT_VISIBLE = "not_visible"  # 人脸不可见（疑似俯卧/遮挡）


@dataclass
class PostureResult:
    """
    单张人脸的睡姿判定结果

    Attributes:
        posture: 睡姿分类
        bbox: 人脸框；NOT_VISIBLE 时为 None
        head_pose: 头部姿态估计；无关键点或 NOT_VISIBLE 时为 None
    """

    posture: SleepPosture
    bbox: BoundingBox | None = None
    head_pose: HeadPose | None = None


class SleepPostureClassifier:
    """
    睡姿分类器

    使用流程：
        clf = SleepPostureClassifier(config.face)
        alerts = clf.evaluate(faces)   # 每帧调用，内部维护持续性缓冲
    """

    def __init__(self, config: FaceConfig) -> None:
        self._side_yaw_threshold = config.side_yaw_threshold
        self._roll_threshold = config.roll_threshold
        self._not_visible_threshold = config.not_visible_frames_threshold

        # 连续"人脸不可见"帧缓冲（达到阈值才告警）
        self._not_visible_buffer: deque[bool] = deque(
            maxlen=max(1, self._not_visible_threshold)
        )
        # 最近一次分类结果（供 health 端点 / 调试）
        self._last_results: list[PostureResult] = []

    def classify(self, faces: list[FaceDetection]) -> list[PostureResult]:
        """
        对每张人脸做睡姿分类

        Args:
            faces: 人脸检测结果列表（含可选关键点）

        Returns:
            PostureResult 列表；faces 为空时返回空列表
            （"无人脸"的整体判定由 evaluate() 负责）
        """
        results: list[PostureResult] = []
        for fd in faces:
            if fd.landmarks is None:
                # 无关键点（桩实现/旧模型）：仅以"脸可见"判定为安全
                results.append(PostureResult(
                    posture=SleepPosture.FACE_UP,
                    bbox=fd.bbox,
                    head_pose=None,
                ))
                continue

            pose = estimate_head_pose(
                fd.landmarks,
                side_yaw_threshold=self._side_yaw_threshold,
                roll_threshold=self._roll_threshold,
            )
            posture = {
                HeadState.UP: SleepPosture.FACE_UP,
                HeadState.SIDE: SleepPosture.SIDE,
                HeadState.TILTED: SleepPosture.TILTED,
            }[pose.state]
            results.append(PostureResult(
                posture=posture,
                bbox=fd.bbox,
                head_pose=pose,
            ))

        return results

    def evaluate(self, faces: list[FaceDetection]) -> list[AlertLog]:
        """
        每帧调用：分类 + 持续性过滤，返回需触发的告警

        Args:
            faces: 当前帧人脸检测结果

        Returns:
            告警列表；仅在连续 N 帧无人脸时返回 PRONE_SLEEP 告警
        """
        self._last_results = self.classify(faces)
        not_visible = len(faces) == 0

        if not_visible:
            self._not_visible_buffer.append(True)
        else:
            # 检测到人脸即视为安全，重置持续性缓冲
            self._not_visible_buffer.clear()

        # 连续 N 帧无人脸 -> 触发 CRITICAL 告警
        if (
            len(self._not_visible_buffer) >= self._not_visible_threshold
            and all(self._not_visible_buffer)
        ):
            self._not_visible_buffer.clear()
            logger.warning(
                "PRONE_SLEEP trigger: 连续 %d 帧未检测到人脸，"
                "儿童可能存在趴睡/遮挡风险",
                self._not_visible_threshold,
            )
            return [
                AlertLog(
                    alert_type=AlertType.PRONE_SLEEP,
                    severity=AlertSeverity.CRITICAL,
                    description="人脸不可见，儿童可能存在趴睡/遮挡风险",
                )
            ]

        return []

    def get_snapshot(self) -> dict:
        """
        最近一次分类快照（供 /health/face 端点）

        Returns:
            {"faces": N, "postures": [...], "not_visible_frames": M}
        """
        return {
            "faces": len(self._last_results),
            "postures": [r.posture.value for r in self._last_results],
            "not_visible_frames": len(self._not_visible_buffer),
        }
