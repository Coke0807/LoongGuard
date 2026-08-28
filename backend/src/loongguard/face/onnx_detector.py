"""
ONNX 人脸检测器（基于 InsightFace SCRFD / det_10g.onnx）

设计动机：
    使用 buffalo_l/det_10g.onnx 模型在端侧执行实时人脸检测。
    检测结果用于睡姿监测（Person bbox 内无 Face -> 异常睡姿），
    5 点关键点用于头部姿态估计（head_pose.py）与睡姿分类（posture.py）。
    当 face.enabled=false 或模型文件不存在时，由工厂回退到 DummyFaceDetector。

模型来源：
    det_10g.onnx 来自 InsightFace buffalo_l 预训练套件，
    基于 SCRFD 架构（Sample and Computation Redistribution for
    Efficient Face Detection），3 级 FPN 输出，支持动态输入尺寸。

输入输出格式：
    - 输入: (1, 3, H, W) float32，归一化到 [0, 1]（推荐 640×640）
    - 输出: 9 个张量，按 stride 升序排列：
        [0] scores_stride8  (12800, 1)   已为概率（无需 sigmoid）
        [1] scores_stride16 (3200, 1)
        [2] scores_stride32 (800, 1)
        [3] bboxes_stride8  (12800, 4)   distance [l, t, r, b]
        [4] bboxes_stride16 (3200, 4)
        [5] bboxes_stride32 (800, 4)
        [6] landmarks_stride8  (12800, 10)  5 点 (x, y) distance
        [7] landmarks_stride16 (3200, 10)
        [8] landmarks_stride32 (800, 10)

Anchor 生成策略（与 insightface 官方 scrfd.py 一致）：
    - 3 级 FPN: strides = [8, 16, 32]
    - 每级 2 个 anchor: num_anchors = 2
    - 中心点 = (col * stride, row * stride)，无 +stride/2 偏移
    - 排列顺序：位置外循环、anchor 内循环（交替）
    - 总计: 80×80×2 + 40×40×2 + 20×20×2 = 16800 个 anchor（640×640 输入）

解码公式（SCRFD distance-based）：
    bbox:  x1 = cx - l*stride, y1 = cy - t*stride,
           x2 = cx + r*stride, y2 = cy + b*stride
    kps:   px = cx + dx*stride, py = cy + dy*stride
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from config import FaceConfig
from loongguard.face.base import FaceDetection, FaceDetector
from loongguard.utils.schema import BoundingBox

logger = logging.getLogger(__name__)

# ── SCRFD det_10g Anchor 配置 ──────────────────────────────
_FPN_STRIDES = [8, 16, 32]
_FPN_NUM_ANCHORS = 2  # 每级 2 个 anchor（交替排列）
# 输入尺寸（模型为动态输入，此处固定为 640×640 保持一致性）
_INPUT_SIZE = 640
# NMS 参数
_NMS_THRESHOLD = 0.4
# NMS 后最多保留的人脸数
_MAX_FACES = 20


def _generate_anchor_centers(
    input_size: int,
    strides: list[int],
    num_anchors: int,
) -> np.ndarray:
    """
    生成 SCRFD 所有 FPN 级别的 anchor 中心点（与官方 scrfd.py 一致）

    官方实现：
        anchor_centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1)  # (H, W, 2)
        anchor_centers = np.stack([anchor_centers] * num_anchors, axis=2)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
    即位置外循环、anchor 内循环（交替），中心 = col*stride（无 +stride/2）。

    Returns:
        (N, 2) ndarray，每行为 [cx, cy]（输入图像素坐标）
    """
    all_centers: list[np.ndarray] = []

    for stride in strides:
        feature_h = math.ceil(input_size / stride)
        feature_w = math.ceil(input_size / stride)

        # mgrid[:h, :w][::-1] -> [x_grid, y_grid]，各 (H, W)
        gy, gx = np.mgrid[:feature_h, :feature_w]
        centers = np.stack([gx, gy], axis=-1).astype(np.float32)  # (H, W, 2)
        centers = np.stack([centers] * num_anchors, axis=2)  # (H, W, 2, 2)
        centers = (centers * stride).reshape((-1, 2))  # 位置外、anchor 内

        all_centers.append(centers)

    return np.concatenate(all_centers, axis=0)  # (N, 2)


def _generate_strides(
    input_size: int,
    strides: list[int],
    num_anchors: int,
) -> np.ndarray:
    """生成与 anchor 中心一一对应的 stride 数组 (N,)"""
    arr: list[float] = []
    for stride in strides:
        feature_h = math.ceil(input_size / stride)
        feature_w = math.ceil(input_size / stride)
        arr.extend([stride] * (feature_h * feature_w * num_anchors))
    return np.array(arr, dtype=np.float32)


def _decode_bboxes(
    raw_bboxes: np.ndarray,
    centers: np.ndarray,
    strides: np.ndarray,
) -> np.ndarray:
    """
    解码 SCRFD raw bbox 预测为实际坐标（distance-based）

    SCRFD bbox 解码公式：
        x1 = cx - l * stride
        y1 = cy - t * stride
        x2 = cx + r * stride
        y2 = cy + b * stride

    Args:
        raw_bboxes: (N, 4) 原始预测 [l, t, r, b]（以 stride 为单位的距离）
        centers: (N, 2) anchor 中心 [cx, cy]
        strides: (N,) 每个 anchor 对应的 stride

    Returns:
        (N, 4) 解码后的坐标 [x1, y1, x2, y2]（输入图像素坐标）
    """
    left = raw_bboxes[:, 0]
    top = raw_bboxes[:, 1]
    right = raw_bboxes[:, 2]
    bottom = raw_bboxes[:, 3]

    x1 = centers[:, 0] - left * strides
    y1 = centers[:, 1] - top * strides
    x2 = centers[:, 0] + right * strides
    y2 = centers[:, 1] + bottom * strides

    return np.stack([x1, y1, x2, y2], axis=1)


def _decode_landmarks(
    raw_kps: np.ndarray,
    centers: np.ndarray,
    strides: np.ndarray,
) -> np.ndarray:
    """
    解码 SCRFD raw 关键点预测为实际坐标（distance-based）

    Args:
        raw_kps: (N, 10) 原始预测，5 个 (x, y) 距离
        centers: (N, 2) anchor 中心 [cx, cy]
        strides: (N,) 每个 anchor 对应的 stride

    Returns:
        (N, 5, 2) 解码后的关键点坐标（输入图像素坐标），
        顺序 [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
    """
    kps = raw_kps.reshape(-1, 5, 2)
    kps[:, :, 0] = centers[:, None, 0] + kps[:, :, 0] * strides[:, None]
    kps[:, :, 1] = centers[:, None, 1] + kps[:, :, 1] * strides[:, None]
    return kps


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    非极大值抑制（NMS）

    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,) 置信度分数
        iou_threshold: IoU 阈值

    Returns:
        keep_indices: 保留的索引数组
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


class ONNXFaceDetector(FaceDetector):
    """
    基于 ONNX 的 SCRFD 人脸检测器

    使用 InsightFace buffalo_l 的 det_10g.onnx 模型，
    在端侧执行实时人脸检测，返回所有人脸框及 5 点关键点。

    使用流程：
        detector = ONNXFaceDetector(config)
        detector.load_model()
        faces = detector.detect_with_landmarks(rgb_frame)
    """

    def __init__(self, config: FaceConfig) -> None:
        self._config = config
        self._session = None
        self._input_name: str = ""
        self._available: bool = False

        # 预计算 anchor 中心与 stride（输入尺寸固定，仅生成一次）
        self._centers = _generate_anchor_centers(
            _INPUT_SIZE, _FPN_STRIDES, _FPN_NUM_ANCHORS,
        )
        self._strides = _generate_strides(
            _INPUT_SIZE, _FPN_STRIDES, _FPN_NUM_ANCHORS,
        )
        self._input_size = _INPUT_SIZE
        self._conf_threshold = config.conf_threshold
        logger.debug(
            "ONNXFaceDetector 初始化: anchors=%d, input_size=%d",
            len(self._centers), self._input_size,
        )

    def is_available(self) -> bool:
        """是否已成功加载模型"""
        return self._available

    def load_model(self) -> None:
        """
        加载 det_10g.onnx 模型

        降级策略：
            加载失败时设置 _available=False，由工厂回退到 DummyFaceDetector。
        """
        model_path = self._config.model_path
        logger.info("加载人脸检测模型: %s", model_path)

        try:
            from loongguard.utils.onnx_session import create_session

            self._session = create_session(model_path)
            self._input_name = self._session.get_inputs()[0].name
            self._available = True
            logger.info("人脸检测模型加载成功")
        except Exception as exc:
            self._available = False
            logger.warning(
                "人脸检测模型加载失败，人脸检测不可用: %s", exc,
            )

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """
        检测图像中的人脸（仅返回框）

        Args:
            image: RGB 图像 (H, W, 3)，uint8

        Returns:
            人脸检测框列表；无检测结果时返回空列表
        """
        return [fd.bbox for fd in self.detect_with_landmarks(image)]

    def detect_with_landmarks(self, image: np.ndarray) -> list[FaceDetection]:
        """
        检测图像中的人脸并返回 5 点关键点

        Args:
            image: RGB 图像 (H, W, 3)，uint8

        Returns:
            FaceDetection 列表（bbox + landmarks）；无检测结果时返回空列表
        """
        if not self._available or self._session is None:
            return []

        try:
            return self._infer(image)
        except Exception as exc:
            logger.exception("人脸检测推理失败: %s", exc)
            return []

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        前处理：缩放 + 归一化 + 填充到正方形

        Args:
            image: RGB 图像 (H, W, 3)，uint8

        Returns:
            (tensor, scale, (pad_w, pad_h))
            - tensor: (1, 3, input_size, input_size) float32，归一化到 [0, 1]
            - scale: 原始图像缩放比例
            - (pad_w, pad_h): 居中填充偏移量
        """
        h, w = image.shape[:2]
        target = self._input_size

        # 保持宽高比缩放，填充到正方形
        scale = min(target / w, target / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 居中填充到 target x target
        canvas = np.full((target, target, 3), 0, dtype=np.uint8)
        pad_h = (target - new_h) // 2
        pad_w = (target - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # HWC -> CHW, 归一化到 [0, 1], 添加 batch 维度
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # (3, H, W)
        tensor = tensor[np.newaxis, ...]  # (1, 3, H, W)

        return tensor, scale, (pad_w, pad_h)

    def _infer(self, image: np.ndarray) -> list[FaceDetection]:
        """
        完整推理流程：前处理 -> ONNX 推理 -> 后处理 -> NMS

        Args:
            image: RGB 图像 (H, W, 3)，uint8

        Returns:
            FaceDetection 列表（bbox + 5 点 landmarks）
        """
        tensor, scale, (pad_w, pad_h) = self._preprocess(image)

        # ONNX 推理
        outputs = self._session.run(None, {self._input_name: tensor})

        # 解析输出（9 个输出按 stride 升序排列）
        raw_scores = np.concatenate([outputs[0], outputs[1], outputs[2]], axis=0).ravel()
        raw_bboxes = np.concatenate([outputs[3], outputs[4], outputs[5]], axis=0)
        raw_kps = np.concatenate([outputs[6], outputs[7], outputs[8]], axis=0)

        # 分数处理：det_10g 的 ONNX 导出已包含 Sigmoid，输出为概率。
        # 防御性检测：若范围超出 [0, 1] 则视为 logits 并补一次 sigmoid。
        if raw_scores.min() >= 0.0 and raw_scores.max() <= 1.0:
            scores = raw_scores
        else:
            scores = 1.0 / (1.0 + np.exp(-raw_scores))

        valid_indices = np.where(scores >= self._conf_threshold)[0]

        if len(valid_indices) == 0:
            return []

        # 解码 bbox 与关键点坐标（distance-based）
        decoded_bboxes = _decode_bboxes(raw_bboxes, self._centers, self._strides)
        decoded_kps = _decode_landmarks(raw_kps, self._centers, self._strides)

        # 只保留有效检测
        valid_scores = scores[valid_indices]
        valid_bboxes = decoded_bboxes[valid_indices]
        valid_kps = decoded_kps[valid_indices]

        # 坐标反变换（去除 padding 和缩放）
        valid_bboxes[:, 0] = (valid_bboxes[:, 0] - pad_w) / scale
        valid_bboxes[:, 1] = (valid_bboxes[:, 1] - pad_h) / scale
        valid_bboxes[:, 2] = (valid_bboxes[:, 2] - pad_w) / scale
        valid_bboxes[:, 3] = (valid_bboxes[:, 3] - pad_h) / scale
        valid_kps[:, :, 0] = (valid_kps[:, :, 0] - pad_w) / scale
        valid_kps[:, :, 1] = (valid_kps[:, :, 1] - pad_h) / scale

        # NMS 去除重叠框
        keep = _nms(valid_bboxes, valid_scores, _NMS_THRESHOLD)
        keep = keep[:_MAX_FACES]  # 限制最大人脸数

        # 转换为 FaceDetection 列表
        h, w = image.shape[:2]
        results: list[FaceDetection] = []
        for idx in keep:
            x1, y1, x2, y2 = valid_bboxes[idx]
            # 边界裁剪
            x1 = max(0, min(int(x1), w))
            y1 = max(0, min(int(y1), h))
            x2 = max(0, min(int(x2), w))
            y2 = max(0, min(int(y2), h))
            score = float(valid_scores[idx])

            bbox = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=score,
                class_id=0,
                class_name="face",
            )
            results.append(FaceDetection(
                bbox=bbox,
                landmarks=valid_kps[idx].astype(np.float32),
            ))

        return results
