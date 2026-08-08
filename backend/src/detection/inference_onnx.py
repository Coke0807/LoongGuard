"""
ONNX Runtime 推理引擎

设计动机：
    LoongArch 端侧推理的核心执行器。负责 ONNX 模型的加载、图像前处理、
    模型推理、后处理（置信度过滤 + 坐标反变换）全流程。

    YOLO26-Nano 采用 NMS-Free 端到端设计，本模块直接输出最终检测结果，
    不经过传统 NMS 后处理，端侧 CPU 延迟显著降低。

    推理后端优先级：CUDA > OpenCL > CPU（LoongArch 上通过 OpenCL 调度 LG200 GPU）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src.utils.onnx_session import create_session, select_providers

logger = logging.getLogger(__name__)

# 可视化调色板：8 种高对比度颜色，循环分配给检测类别
_COLORS = [
    (255, 56, 56),    # 红
    (255, 157, 151),   # 浅红
    (255, 112, 31),    # 橙
    (255, 178, 29),    # 金
    (207, 210, 49),    # 黄
    (72, 249, 10),     # 绿
    (146, 204, 23),    # 青绿
    (61, 219, 134),    # 翠绿
]


@dataclass
class InferenceResult:
    """单次推理结果封装"""

    # 原始图像尺寸 (H, W)
    original_shape: tuple[int, int]
    # 模型输入尺寸 (H, W)
    input_shape: tuple[int, int]
    # 检测结果 (N, 6): [x1, y1, x2, y2, confidence, class_id]
    detections: np.ndarray
    # 前处理耗时 (秒)
    preprocess_time: float
    # 推理耗时 (秒)
    inference_time: float
    # 后处理耗时 (秒)
    postprocess_time: float
    # 端到端总耗时 (秒)
    total_time: float


class LoongONNXPredictor:
    """
    LoongArch ONNX 推理器

    使用 ONNX Runtime 加载量化模型，在端侧执行前处理 -> 推理 -> 后处理。
    支持 INT8 量化模型（跳过 ImageNet 归一化）和 FP32 模型。

    使用流程：
        predictor = LoongONNXPredictor(model_path="model.onnx", ...)
        result = predictor.predict(image_bgr)
        # result.detections -> (N, 6) numpy array
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.45,
        classes: Optional[list[str]] = None,
        quantized: bool = False,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.classes = classes or []
        self.quantized = quantized

        # 校验模型文件存在性
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        # 初始化 ONNX Runtime 推理会话
        # 设计动机：LoongArch 上通过 OpenCL 调度 LG200 GPU，
        # x86 开发机 fallback 到 CPU。Provider 优先级由共享工厂统一管理。
        providers = select_providers()
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

        logger.info(
            "LoongONNXPredictor initialized: model=%s, input_size=%d, "
            "conf=%.2f, quantized=%s, providers=%s",
            model_path, input_size, conf_threshold, quantized, providers,
        )

    def preprocess(
        self, image: Union[np.ndarray, None]
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        图像前处理：缩放 + 填充 + 归一化

        将任意尺寸的 BGR 图像缩放并居中填充到 (input_size x input_size) 的
        正方形画布上，返回模型可接受的 (1, 3, H, W) float32 张量。

        Args:
            image: BGR 格式图像 (H, W, 3)，uint8。None 或空数组抛出 ValueError。

        Returns:
            (batch_tensor, scale_ratio, (pad_h, pad_w))
            - batch_tensor: (1, 3, input_size, input_size) float32
            - scale_ratio: 缩放比例，用于后处理坐标反变换
            - (pad_h, pad_w): 填充偏移，用于后处理坐标反变换

        设计动机：
            INT8 量化模型已在量化时内置归一化，故 quantized=True 时
            跳过 ImageNet 均值/标准差归一化，直接 /255.0。
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        h, w = image.shape[:2]
        target = self.input_size

        # 等比缩放：以长边为基准
        scale = min(target / w, target / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # 居中填充到 target x target 画布
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        pad_h = (target - new_h) // 2
        pad_w = (target - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR -> RGB, HWC -> CHW, 归一化
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32)

        if not self.quantized:
            # FP32 模型：ImageNet 归一化
            tensor = tensor / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = (tensor - mean) / std
        else:
            # INT8 量化模型：仅 /255（量化时已内置归一化参数）
            tensor = tensor / 255.0

        # HWC -> CHW, 添加 batch 维度
        batch = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        return batch, scale, (pad_h, pad_w)

    def predict(self, image: Union[np.ndarray, str]) -> InferenceResult:
        """
        完整推理流程：前处理 -> ONNX 推理 -> 后处理

        Args:
            image: BGR 图像 (np.ndarray) 或图像文件路径 (str)

        Returns:
            InferenceResult 包含检测结果和各阶段耗时

        Raises:
            FileNotFoundError: 图像文件路径不存在
            ValueError: 图像为空
        """
        # 用 perf_counter 而非 monotonic：dummy/量化模型各阶段耗时常在 1ms 以下，
        # monotonic 在部分平台分辨率约 1ms，会把 sub-ms 耗时截断为 0，导致计时失真
        t_total_start = time.perf_counter()

        # 支持文件路径输入
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(
                    f"Image file not found: {image}"
                )
            image = cv2.imread(image)

        original_shape = image.shape[:2]

        # 前处理
        t0 = time.perf_counter()
        batch, scale, (pad_h, pad_w) = self.preprocess(image)
        preprocess_time = time.perf_counter() - t0

        # ONNX 推理
        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: batch})
        inference_time = time.perf_counter() - t0

        # 后处理：置信度过滤 + 坐标反变换
        t0 = time.perf_counter()
        detections = self.postprocess(
            outputs, original_shape, scale, (pad_h, pad_w)
        )
        postprocess_time = time.perf_counter() - t0

        total_time = time.perf_counter() - t_total_start

        return InferenceResult(
            original_shape=original_shape,
            input_shape=(self.input_size, self.input_size),
            detections=detections,
            preprocess_time=preprocess_time,
            inference_time=inference_time,
            postprocess_time=postprocess_time,
            total_time=total_time,
        )

    def infer_at_size(
        self, image: np.ndarray, size: Optional[int] = None
    ) -> np.ndarray:
        """
        在指定分辨率下推理（用于 Dynamic ROI 二级检测）

        Args:
            image: BGR 图像
            size: 推理分辨率，默认使用 self.input_size

        Returns:
            检测结果 (N, 6) numpy array
        """
        original_size = self.input_size
        if size is not None:
            self.input_size = size

        try:
            result = self.predict(image)
            return result.detections
        finally:
            self.input_size = original_size

    def postprocess(
        self,
        outputs: list[np.ndarray],
        original_shape: tuple[int, int],
        scale_ratio: float,
        padding: tuple[int, int],
    ) -> np.ndarray:
        """
        后处理：置信度过滤 + 坐标反变换 + 边界裁剪

        YOLO26-Nano NMS-Free 输出格式为角点坐标 [x1, y1, x2, y2, conf, cls]，
        无需 NMS，直接按置信度阈值过滤。

        坐标反变换公式：
            x_orig = (x_model - pad_w) / scale
            y_orig = (y_model - pad_h) / scale

        Args:
            outputs: ONNX 模型原始输出列表
            original_shape: 原始图像尺寸 (H, W)
            scale_ratio: 前处理缩放比例
            padding: 前处理填充偏移 (pad_h, pad_w)

        Returns:
            检测结果 (N, 6): [x1, y1, x2, y2, confidence, class_id]
            当无有效检测时返回 (0, 6) 的空数组。
        """
        orig_h, orig_w = original_shape
        pad_h, pad_w = padding

        # 提取检测数据，兼容带/不带 batch 维度
        raw = outputs[0] if isinstance(outputs, list) else outputs
        if isinstance(raw, np.ndarray):
            preds = raw
        else:
            preds = np.array(raw)

        # 去除 batch 维度：(1, N, 6) -> (N, 6)
        if preds.ndim == 3:
            preds = preds[0]

        if preds.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        # 按置信度过滤
        valid_mask = preds[:, 4] >= self.conf_threshold
        filtered = preds[valid_mask]

        if filtered.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        # 坐标反变换：去除 padding 和缩放
        result = filtered.copy()
        result[:, 0] = (filtered[:, 0] - pad_w) / scale_ratio  # x1
        result[:, 1] = (filtered[:, 1] - pad_h) / scale_ratio  # y1
        result[:, 2] = (filtered[:, 2] - pad_w) / scale_ratio  # x2
        result[:, 3] = (filtered[:, 3] - pad_h) / scale_ratio  # y2

        # 边界裁剪
        result[:, 0] = np.clip(result[:, 0], 0, orig_w)
        result[:, 1] = np.clip(result[:, 1], 0, orig_h)
        result[:, 2] = np.clip(result[:, 2], 0, orig_w)
        result[:, 3] = np.clip(result[:, 3], 0, orig_h)

        return result.astype(np.float32)

    def draw_detections(
        self,
        image: np.ndarray,
        detections: np.ndarray,
    ) -> np.ndarray:
        """
        在图像上绘制检测框和标签

        Args:
            image: BGR 图像
            detections: (N, 6) 检测结果

        Returns:
            标注后的图像副本
        """
        result = image.copy()

        if detections.size == 0:
            return result

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id_int = int(cls_id)
            color = self._get_color(cls_id_int)
            label = self._get_label(cls_id_int, float(conf))

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # 标签背景
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
            )
            cv2.rectangle(
                result,
                (x1, y1 - th - 6), (x1 + tw + 4, y1),
                color, -1,
            )
            cv2.putText(
                result, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        return result

    def _get_color(self, class_id: int) -> tuple[int, int, int]:
        """获取类别对应的绘制颜色，循环使用调色板"""
        return _COLORS[class_id % len(_COLORS)]

    def _get_label(self, class_id: int, confidence: float) -> str:
        """获取类别标签文本"""
        if 0 <= class_id < len(self.classes):
            name = self.classes[class_id]
        else:
            name = f"class_{class_id}"
        return f"{name} {confidence:.2f}"
