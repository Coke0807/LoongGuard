"""
YOLO26-Nano 目标检测推理管道

设计动机：
    封装 YOLO26-Nano INT8 量化模型的加载、前处理、推理、后处理全流程。
    YOLO26 采用 NMS-Free 端到端设计，不需要传统的 NMS 后处理，
    端侧 CPU 后处理延迟显著降低。

    本模块作为 inference_onnx.LoongONNXPredictor 的薄封装层，
    将 numpy 检测结果转换为结构化的 BoundingBox 列表，
    供 pipeline / roi_scheduler 等上层模块消费。
"""

from __future__ import annotations

import logging
import time

import numpy as np

from config import DetectionConfig
from loongguard.api.metrics import (
    DETECTION_AVAILABLE,
    DETECTION_ERROR_COUNT,
    DETECTION_INFERENCE_COUNT,
    INFERENCE_LATENCY,
)
from loongguard.detection.inference_onnx import LoongONNXPredictor
from loongguard.utils.schema import BoundingBox

logger = logging.getLogger(__name__)


class YOLO26Nano:
    """
    YOLO26-Nano 推理器

    委托 LoongONNXPredictor 执行实际的 ONNX 推理，
    本类负责将 numpy 检测结果转换为 BoundingBox 结构体列表。

    使用流程：
        detector = YOLO26Nano(config)
        detector.load_model()
        boxes = detector.infer(raw_rgb_frame)
    """

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._predictor: LoongONNXPredictor | None = None

        # ── 健康状态（供 /health/detection 端点查询）───────────
        self._available: bool = False
        self._last_inference_ts: float = 0.0
        self._success_count: int = 0
        self._error_count: int = 0
        self._last_error: str = ""
        self._model_path: str = self._config.model_path

    def is_available(self) -> bool:
        """是否已成功加载模型并可执行推理（Pipeline 据此降级）"""
        return self._available

    def get_health_snapshot(self) -> dict:
        """返回模型健康状态快照，供 /health/detection 端点序列化"""
        return {
            "available": self._available,
            "model_path": self._model_path,
            "input_size": self._config.input_size,
            "conf_threshold": self._config.conf_threshold,
            "last_inference_ts": self._last_inference_ts,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
        }

    def load_model(self) -> None:
        """
        加载 ONNX 模型

        通过 LoongONNXPredictor 初始化 ONNX Runtime 会话，
        自动检测可用的执行提供程序（CUDA > OpenCL > CPU）。
        INT8 量化模型自动跳过 ImageNet 归一化预处理。

        降级策略：
            加载失败时设置 _available=False，Pipeline 抛出异常由上层
            决定是否降级（YOLO 是主链路，失败时通常选择让进程退出）。
        """
        self._model_path = self._config.model_path
        logger.info(
            "Loading YOLO26-Nano model: %s (quantized=%s)",
            self._config.model_path,
            self._config.quantized,
        )

        try:
            self._predictor = LoongONNXPredictor(
                model_path=self._config.model_path,
                input_size=self._config.input_size,
                conf_threshold=self._config.conf_threshold,
                classes=self._config.classes,
                quantized=self._config.quantized,
            )
        except FileNotFoundError as exc:
            self._available = False
            self._last_error = str(exc)[:200]
            DETECTION_AVAILABLE.set(0)
            logger.error("YOLO26 model not found: %s", self._config.model_path)
            raise
        except Exception as exc:
            self._available = False
            self._last_error = str(exc)[:200]
            DETECTION_AVAILABLE.set(0)
            logger.exception("YOLO26 model load failed")
            raise

        self._available = True
        DETECTION_AVAILABLE.set(1)
        logger.info("YOLO26-Nano model loaded")

    def infer(self, image: np.ndarray) -> list[BoundingBox]:
        """
        执行推理

        Args:
            image: 原始 RGB 图像 (H, W, 3)，uint8

        Returns:
            检测框列表（NMS-Free 直接输出，无需 NMS 后处理）

        设计约束：
            - 输出直接来自模型端到端头，不经过 NMS
            - 置信度低于 conf_threshold 的结果已过滤

        错误处理：
            推理失败（predictor.predict 抛异常）时记 metric + 错误计数，
            返回空列表而非抛出，让 pipeline 可以降级运行。
        """
        if self._predictor is None:
            raise RuntimeError("Model not loaded, call load_model() first")

        try:
            with INFERENCE_LATENCY.time():
                result = self._predictor.predict(image)
        except Exception as exc:
            self._error_count += 1
            self._last_error = str(exc)[:200]
            DETECTION_ERROR_COUNT.inc()
            logger.exception("YOLO26 inference failed")
            return []

        self._success_count += 1
        self._last_inference_ts = time.monotonic()
        DETECTION_INFERENCE_COUNT.inc()
        return self._detections_to_bboxes(result.detections)

    def infer_stage1(self, image: np.ndarray) -> list[BoundingBox]:
        """
        第一级粗筛（在运动区域裁剪上运行，大区域、小目标）

        Args:
            image: 裁剪出的运动区域 RGB 图像

        Returns:
            粗筛结果（用于定位可疑目标区域）
        """
        if self._predictor is None:
            raise RuntimeError("Model not loaded, call load_model() first")

        with INFERENCE_LATENCY.time():
            result = self._predictor.infer_at_size(image)
        return self._detections_to_bboxes(result)

    def infer_stage2(self, roi_tensor: np.ndarray) -> list[BoundingBox]:
        """
        第二级精准检测（在 stage1 检测框的扩展裁剪上运行，小区域、大目标）

        Args:
            roi_tensor: 从 stage1 检测框扩展裁剪出的高分辨率 ROI 区域

        Returns:
            精确检测结果，用于最终告警决策
        """
        if self._predictor is None:
            raise RuntimeError("Model not loaded, call load_model() first")

        with INFERENCE_LATENCY.time():
            result = self._predictor.infer_at_size(roi_tensor)
        return self._detections_to_bboxes(result)

    def _detections_to_bboxes(self, detections: np.ndarray) -> list[BoundingBox]:
        """
        将 numpy 检测结果转换为 BoundingBox 列表

        Args:
            detections: 形状 (N, 6)，每行为 [x1, y1, x2, y2, confidence, class_id]
        """
        if len(detections) == 0:
            return []

        bboxes: list[BoundingBox] = []
        classes = self._config.classes

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id_int = int(cls_id)
            class_name = classes[cls_id_int] if 0 <= cls_id_int < len(classes) else f"class_{cls_id_int}"

            bboxes.append(BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=float(conf),
                class_id=cls_id_int,
                class_name=class_name,
            ))

        return bboxes
