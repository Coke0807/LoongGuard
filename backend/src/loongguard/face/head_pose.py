"""
头部姿态估计（基于 SCRFD 5 点关键点）

设计动机：
    det_10g.onnx 除人脸框外还输出 5 点关键点（双眼、鼻尖、双嘴角）。
    本模块用纯几何方法从关键点估计头部姿态，无需额外模型，
    为睡姿分类（posture.py）提供"脸朝向"信号：

        - roll（面内旋转）: 双眼连线相对水平面的夹角。
          正脸平躺时 roll 接近 0；头歪向一侧时 |roll| 增大。
        - yaw_ratio（水平压缩比）: 双眼距 / 脸高（眼线中点到嘴线中点）。
          正脸时双眼横向展开，比值接近 1；脸转向侧面时双眼在
          水平方向被压缩，比值显著下降。

    经验阈值（scripts/_measure_kps_geometry.py 实测校准）：
        正脸 eye/height ≈ 0.72~1.07，侧脸 ≈ 0.04~0.59，
        故 side_yaw_threshold 默认 0.6 可稳定区分正/侧脸。

几何示意（正脸）：
        le ●────● re        <- 双眼连线（水平）
             ●  nose
        lm ●────● rm        <- 双嘴角连线
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

# 关键点索引（SCRFD 5 点顺序）
_LEFT_EYE = 0
_RIGHT_EYE = 1
_NOSE = 2
_LEFT_MOUTH = 3
_RIGHT_MOUTH = 4


class HeadState(str, Enum):
    """头部朝向状态"""

    UP = "up"          # 正脸 / 脸朝上（仰卧时脸可见）
    SIDE = "side"      # 水平旋转 / 侧脸（侧卧时脸转向侧面）
    TILTED = "tilted"  # 面内倾斜（头歪向一侧）


@dataclass
class HeadPose:
    """
    单张人脸的头部姿态估计结果

    Attributes:
        roll_deg: 面内旋转角（度），双眼连线相对水平面，正脸接近 0
        yaw_ratio: 水平压缩比 = 双眼距 / 脸高，正脸接近 1，侧脸趋近 0
        state: 头部朝向状态
    """

    roll_deg: float
    yaw_ratio: float
    state: HeadState


def estimate_head_pose(
    landmarks: np.ndarray,
    side_yaw_threshold: float = 0.6,
    roll_threshold: float = 45.0,
) -> HeadPose:
    """
    从 5 点关键点估计头部姿态

    Args:
        landmarks: (5, 2) 关键点 [x, y]，顺序
            [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
        side_yaw_threshold: eye/height 低于此值判定为侧脸
        roll_threshold: |roll| 高于此值判定为头倾斜

    Returns:
        HeadPose（roll_deg, yaw_ratio, state）

    判定优先级：
        先判 SIDE（水平压缩），再判 TILTED（面内旋转），否则 UP。
        退化关键点（脸高过小）时 yaw_ratio 置 0 并归为 SIDE。
    """
    le = landmarks[_LEFT_EYE].astype(np.float64)
    re = landmarks[_RIGHT_EYE].astype(np.float64)
    lm = landmarks[_LEFT_MOUTH].astype(np.float64)
    rm = landmarks[_RIGHT_MOUTH].astype(np.float64)

    # 面内旋转角：双眼连线相对水平面
    roll_deg = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))

    # 水平压缩比：双眼距 / 脸高
    eye_dist = float(np.linalg.norm(re - le))
    eye_mid = (le + re) / 2.0
    mouth_mid = (lm + rm) / 2.0
    face_height = float(np.linalg.norm(mouth_mid - eye_mid))

    if face_height < 1e-6:
        # 退化关键点（脸高过小），无法判定为正脸，归为 SIDE
        return HeadPose(roll_deg=roll_deg, yaw_ratio=0.0, state=HeadState.SIDE)

    yaw_ratio = eye_dist / face_height

    if yaw_ratio < side_yaw_threshold:
        state = HeadState.SIDE
    elif abs(roll_deg) > roll_threshold:
        state = HeadState.TILTED
    else:
        state = HeadState.UP

    return HeadPose(roll_deg=roll_deg, yaw_ratio=yaw_ratio, state=state)
