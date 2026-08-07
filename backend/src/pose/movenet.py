"""
MoveNet-Lightning 姿态估计模块

设计动机：
    检测幼儿午睡时的骨骼关键点，判断是否处于高危俯卧趴睡姿态。
    俯卧趴睡是幼儿园窒息事故的主要原因之一，持续检测到俯卧
    超过阈值帧数后触发 CRITICAL 级告警。
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config import PoseConfig
from src.api.metrics import (
    POSE_AVAILABLE,
    POSE_ERROR_COUNT,
    POSE_INFERENCE_COUNT,
    POSE_INFERENCE_LATENCY,
)
from src.utils.onnx_session import create_session
from src.utils.schema import AlertLog, AlertSeverity, AlertType

logger = logging.getLogger(__name__)

# MoveNet 17 个关键点索引
_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class MoveNetLightning:
    """
    MoveNet-Lightning 姿态估计器

    使用流程：
        pose = MoveNetLightning(config)
        pose.load_model()
        alerts = pose.detect_prone(frame_rgb)

    俯卧判定逻辑：
        计算肩-髋连线与水平面的夹角，低于阈值视为俯卧。
        连续 N 帧检测到俯卧则触发告警（避免瞬时误判）。
    """

    def __init__(self, config: PoseConfig) -> None:
        self._config = config
        self._session = None
        self._input_name: str = ""
        self._output_names: list[str] = []
        self._prone_frame_buffer: deque[bool] = deque(maxlen=self._config.prone_frame_threshold * 2)

        # ── 健康状态（供 /health/pose 端点查询）──────────────
        # _available: 模型是否已成功加载（false 时整模块降级跳过）
        self._available: bool = False
        # 最近一次推理的时间戳（秒，单调时间）
        self._last_inference_ts: float = 0.0
        # 累计推理成功次数（用于 /health/pose 端点）
        self._success_count: int = 0
        # 累计推理失败次数
        self._error_count: int = 0
        # 最近一次错误信息（截断 200 字符）
        self._last_error: str = ""
        # 上次执行推理的帧序号（节流用，None 表示尚未执行）
        self._last_run_frame: Optional[int] = None
        # 模型路径（脱敏后展示用）
        self._model_path: str = self._config.model_path

    def is_available(self) -> bool:
        """是否已成功加载模型并可执行推理（Pipeline 据此降级）"""
        return self._available

    def get_health_snapshot(self) -> dict:
        """
        返回模型健康状态快照，供 /health/pose 端点序列化
        设计动机：把指标从 metrics 模块独立出来，便于按需查询而非依赖 Prometheus 抓取
        """
        return {
            "available": self._available,
            "model_path": self._model_path,
            "last_inference_ts": self._last_inference_ts,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "last_run_frame": self._last_run_frame,
        }

    def should_run(self, frame_count: int) -> bool:
        """
        是否应在当前帧执行姿态推理

        Args:
            frame_count: 全局帧序号（Pipeline._frame_count）

        Returns:
            True 表示应执行；False 表示本帧跳过

        设计动机：
            MoveNet 推理 20-30ms，在 25 FPS 目标下不可逐帧执行。
            通过 inference_interval（默认 3 帧）做时间维度节流，
            避免俯卧检测抢占主链路算力。
            首次调用（_last_run_frame=None）总是返回 True，
            保证俯卧检测不会因节流逻辑导致首帧漏检。
        """
        if not self._available:
            return False
        if self._last_run_frame is None:
            return True
        interval = max(1, getattr(self._config, "inference_interval", 3))
        return (frame_count - self._last_run_frame) >= interval

    def mark_ran(self, frame_count: int) -> None:
        """记录本帧已执行推理（供下一帧节流判断）"""
        self._last_run_frame = frame_count

    def load_model(self) -> None:
        """
        加载 MoveNet-Lightning ONNX 模型

        自动检测可用的执行提供程序（CUDA > OpenCL > CPU），
        确保在 x86 开发环境和 LoongArch 生产环境均可运行。

        降级策略：
            加载失败时设置 _available=False，Pipeline 自动跳过俯卧检测，
            不阻塞危险物品检测主链路。
        """
        model_path = self._config.model_path
        logger.info("Loading MoveNet-Lightning model: %s", model_path)

        try:
            # 通过共享工厂创建 Session，配置与 YOLO26Nano 保持一致
            self._session = create_session(model_path)
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
        except FileNotFoundError:
            self._available = False
            self._last_error = f"Model file not found: {model_path}"
            logger.error(
                "MoveNet model not found, pose detection disabled: %s", model_path,
            )
            POSE_AVAILABLE.set(0)
            return
        except Exception as exc:
            self._available = False
            self._last_error = str(exc)[:200]
            logger.exception("MoveNet model load failed, pose detection disabled")
            POSE_AVAILABLE.set(0)
            return

        self._available = True
        self._model_path = model_path
        POSE_AVAILABLE.set(1)
        logger.info("MoveNet-Lightning model loaded")

    def detect_prone(self, frame_rgb: np.ndarray) -> list[AlertLog]:
        """
        检测帧中是否存在俯卧趴睡幼儿

        Args:
            frame_rgb: RGB 图像 (H, W, 3)

        Returns:
            告警列表（空表示无俯卧风险）

        降级策略：
            模型未加载成功时直接返回空列表，调用方不应重复触发。
            单次推理失败仅记 metric，不中断后续帧。
        """
        if not self._available or self._session is None:
            self._prone_frame_buffer.append(False)
            return []

        try:
            keypoints = self._infer_keypoints(frame_rgb)
        except Exception as exc:
            self._error_count += 1
            self._last_error = str(exc)[:200]
            POSE_ERROR_COUNT.inc()
            logger.exception("Pose inference failed")
            self._prone_frame_buffer.append(False)
            return []

        # 仅在成功拿到关键点时累加成功计数（推理成功 + 解析成功）
        self._success_count += 1
        self._last_inference_ts = time.monotonic()
        POSE_INFERENCE_COUNT.inc()

        if keypoints is None:
            self._prone_frame_buffer.append(False)
            return []

        is_prone = self._is_prone(keypoints)
        self._prone_frame_buffer.append(is_prone)
        logger.debug("Prone detection buffer: %s", list(self._prone_frame_buffer))

        if is_prone and len(self._prone_frame_buffer) == self._prone_frame_buffer.maxlen:
            prone_ratio = sum(self._prone_frame_buffer) / len(self._prone_frame_buffer)
            if prone_ratio >= 0.5:
                self._prone_frame_buffer.clear()
                return [self._build_prone_alert(keypoints)]

        return []

    def _infer_keypoints(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        执行 MoveNet ONNX 推理

        前处理：resize 到 192x192，转为 NHWC float32
        推理：session.run()
        后处理：解析 (1,1,17,3) 输出为 (17,3) 数组

        Args:
            frame_rgb: RGB 图像 (H, W, 3)，uint8

        Returns:
            (17, 3) 数组，每行为 [y, x, score]，归一化坐标 [0, 1]
            推理失败或所有关键点不可见时返回 None
        """
        if self._session is None:
            raise RuntimeError("Model not loaded, call load_model() first")

        input_size = self._config.input_size  # 192

        # 前处理：resize（MoveNet-Lightning INT8 输入为 NHWC uint8）
        # 模型内部已完成归一化，外部仅需保持 uint8 像素值直传
        resized = cv2.resize(frame_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

        # 检测模型输入数据类型要求（uint8 vs float32）
        input_meta = self._session.get_inputs()[0]
        if "uint8" in input_meta.type:
            input_tensor = resized.astype(np.uint8)[np.newaxis, ...]  # (1, 192, 192, 3) uint8
        else:
            # 兼容 float32 输入的旧版 MoveNet 模型
            input_tensor = resized.astype(np.float32)[np.newaxis, ...]  # (1, 192, 192, 3)

        # 推理（带延迟监控；session.run 失败时返回 None 由上层重试）
        with POSE_INFERENCE_LATENCY.time():
            outputs = self._session.run(self._output_names, {self._input_name: input_tensor})

        # 后处理：输出形状 (1, 1, 17, 3) -> (17, 3)
        keypoints = outputs[0]
        if keypoints.ndim == 4:
            keypoints = keypoints[0, 0]  # (17, 3)
        elif keypoints.ndim == 3:
            keypoints = keypoints[0]  # (17, 3)

        # 归一化坐标到 [0, 1]（MoveNet 输出可能需要根据模型归一化方式调整）
        if keypoints.max() > 1.0:
            keypoints = keypoints / np.array([input_size, input_size, 1.0], dtype=np.float32)

        # 检查是否有任何关键点有足够置信度
        if np.all(keypoints[:, 2] < self._config.min_keypoint_score):
            return None

        return keypoints

    def _is_prone(self, keypoints: np.ndarray) -> bool:
        """
        判断骨骼关键点是否呈俯卧姿态

        判定依据：
            - 肩膀和髋部关键点可见性足够高
            - 肩-髋连线与水平面夹角 < prone_angle_threshold
            - 鼻子关键点位置低于肩膀（面朝下）
        """
        l_shoulder = keypoints[5]  # [y, x, score]
        r_shoulder = keypoints[6]
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        nose = keypoints[0]

        min_score = self._config.min_keypoint_score
        if any(k[2] < min_score for k in [l_shoulder, r_shoulder, l_hip, r_hip, nose]):
            return False

        # 肩髋中点
        shoulder_mid_y = (l_shoulder[0] + r_shoulder[0]) / 2
        hip_mid_y = (l_hip[0] + r_hip[0]) / 2
        shoulder_mid_x = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_mid_x = (l_hip[1] + r_hip[1]) / 2

        # 肩-髋连线角度（与垂直方向的夹角）
        dy = abs(hip_mid_y - shoulder_mid_y)
        dx = abs(hip_mid_x - shoulder_mid_x)
        angle = 90.0 if dy < 1e-6 else math.degrees(math.atan(dx / dy))

        # 俯卧判定：肩髋几乎水平（角度小）且鼻子低于肩膀
        is_flat = angle < self._config.prone_angle_threshold
        face_down = nose[0] > shoulder_mid_y  # y 坐标越大越靠下方

        return is_flat and face_down

    def _build_prone_alert(self, keypoints: np.ndarray) -> AlertLog:
        """构建俯卧睡姿告警日志"""
        return AlertLog(
            alert_type=AlertType.PRONE_SLEEP,
            severity=AlertSeverity.CRITICAL,
            description=(
                f"检测到幼儿俯卧趴睡（持续超过"
                f"{self._config.prone_frame_threshold}帧），存在窒息风险！"
            ),
        )
