"""
YOLO26-Nano 目标检测模块单元测试

覆盖：
    - YOLO26Nano: load_model, infer, _detections_to_bboxes, model-not-loaded 错误
    - LoongONNXPredictor: preprocess 输出形状, postprocess 置信度过滤与坐标变换,
      draw_detections, 空检测处理
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent

from config.settings import DetectionConfig
from loongguard.detection.inference_onnx import InferenceResult, LoongONNXPredictor
from loongguard.detection.yolo26_nano import YOLO26Nano
from loongguard.utils.schema import BoundingBox

# 模型路径
YOLO_MODEL_PATH = str(PROJECT_ROOT / "models" / "best.onnx")


# ═══════════════════════════════════════════════════════════════
#  YOLO26Nano 测试套件
# ═══════════════════════════════════════════════════════════════


class TestYOLO26Nano:
    """YOLO26Nano 推理封装层测试"""

    @pytest.fixture
    def config(self) -> DetectionConfig:
        """使用 dummy 模型的检测配置"""
        return DetectionConfig(
            model_path=YOLO_MODEL_PATH,
            input_size=640,
            conf_threshold=0.5,
            quantized=False,
            classes=["magnetic_bead", "button_battery", "scissors"],
        )

    @pytest.fixture
    def detector(self, config: DetectionConfig) -> YOLO26Nano:
        """已加载模型的 YOLO26Nano 实例"""
        det = YOLO26Nano(config)
        det.load_model()
        return det

    def test_load_model_initializes_predictor(self, config: DetectionConfig) -> None:
        """load_model() 应成功创建 LoongONNXPredictor 实例"""
        det = YOLO26Nano(config)
        assert det._predictor is None
        det.load_model()
        assert det._predictor is not None
        assert isinstance(det._predictor, LoongONNXPredictor)

    def test_infer_without_load_raises(self, config: DetectionConfig) -> None:
        """未调用 load_model() 时 infer() 应抛出 RuntimeError"""
        det = YOLO26Nano(config)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            det.infer(dummy_frame)

    def test_infer_returns_list_of_bounding_box(
        self, detector: YOLO26Nano
    ) -> None:
        """infer() 应返回 list[BoundingBox]（通过 mock 确保结果可控）"""
        # dummy 模型的权重随机初始化，输出的 confidence / class_id 可能越界，
        # 因此用 mock 控制 predict 返回值来验证类型转换逻辑
        known_result = InferenceResult(
            original_shape=(480, 640),
            input_shape=(640, 640),
            detections=np.array([
                [100, 200, 300, 400, 0.85, 0],
            ], dtype=np.float32),
            preprocess_time=0.0,
            inference_time=0.0,
            postprocess_time=0.0,
            total_time=0.0,
        )
        with patch.object(detector._predictor, "predict", return_value=known_result):
            result = detector.infer(np.zeros((480, 640, 3), dtype=np.uint8))

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], BoundingBox)
        assert 0.0 <= result[0].confidence <= 1.0

    def test_infer_returns_bounding_box_with_valid_class_name(
        self, detector: YOLO26Nano
    ) -> None:
        """通过 mock 使 infer() 产生已知检测结果，验证 BoundingBox 字段"""
        # dummy 模型的 class_id 可能为负值或超出范围，直接 mock predict 返回值
        known_result = InferenceResult(
            original_shape=(480, 640),
            input_shape=(640, 640),
            detections=np.array([
                [100, 200, 300, 400, 0.9, 0],
                [50, 60, 150, 250, 0.7, 2],
            ], dtype=np.float32),
            preprocess_time=0.0,
            inference_time=0.0,
            postprocess_time=0.0,
            total_time=0.0,
        )
        with patch.object(detector._predictor, "predict", return_value=known_result):
            result = detector.infer(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(result) == 2
        assert result[0].class_name == "magnetic_bead"
        assert result[1].class_name == "scissors"

    def test_detections_to_bboxes_known_array(self, detector: YOLO26Nano) -> None:
        """_detections_to_bboxes() 对已知 numpy 数组应正确转换为 BoundingBox"""
        # 模拟检测结果：2 个检测 [x1, y1, x2, y2, confidence, class_id]
        detections = np.array([
            [10, 20, 100, 200, 0.85, 0],
            [50, 60, 150, 250, 0.92, 1],
        ], dtype=np.float32)

        bboxes = detector._detections_to_bboxes(detections)
        assert len(bboxes) == 2

        # 验证第一个检测框
        assert bboxes[0].x1 == 10
        assert bboxes[0].y1 == 20
        assert bboxes[0].x2 == 100
        assert bboxes[0].y2 == 200
        assert abs(bboxes[0].confidence - 0.85) < 1e-5
        assert bboxes[0].class_id == 0
        assert bboxes[0].class_name == "magnetic_bead"

        # 验证第二个检测框
        assert bboxes[1].class_id == 1
        assert bboxes[1].class_name == "button_battery"

    def test_detections_to_bboxes_empty_array(self, detector: YOLO26Nano) -> None:
        """_detections_to_bboxes() 对空数组应返回空列表"""
        empty = np.empty((0, 6), dtype=np.float32)
        assert detector._detections_to_bboxes(empty) == []

    def test_detections_to_bboxes_unknown_class_id(self, detector: YOLO26Nano) -> None:
        """class_id 超出 classes 列表范围时应使用 fallback 名称 class_N"""
        detections = np.array([[10, 20, 100, 200, 0.7, 99]], dtype=np.float32)
        bboxes = detector._detections_to_bboxes(detections)
        assert len(bboxes) == 1
        assert bboxes[0].class_name == "class_99"

    def test_infer_stage1_uses_predictor(
        self, detector: YOLO26Nano
    ) -> None:
        """infer_stage1() 应调用 predictor.infer_at_size()"""
        fake_dets = np.array([[10, 20, 50, 60, 0.8, 0]], dtype=np.float32)
        with patch.object(detector._predictor, "infer_at_size", return_value=fake_dets) as mock_infer:
            result = detector.infer_stage1(np.zeros((100, 100, 3), dtype=np.uint8))
            mock_infer.assert_called_once()
            assert len(result) == 1

    def test_infer_stage2_uses_predictor(
        self, detector: YOLO26Nano
    ) -> None:
        """infer_stage2() 应调用 predictor.infer_at_size()"""
        fake_dets = np.array([[10, 20, 50, 60, 0.9, 1]], dtype=np.float32)
        with patch.object(detector._predictor, "infer_at_size", return_value=fake_dets) as mock_infer:
            result = detector.infer_stage2(np.zeros((100, 100, 3), dtype=np.uint8))
            mock_infer.assert_called_once()
            assert len(result) == 1


# ═══════════════════════════════════════════════════════════════
#  LoongONNXPredictor 测试套件
# ═══════════════════════════════════════════════════════════════


class TestLoongONNXPredictor:
    """LoongONNXPredictor ONNX 推理管道测试"""

    @pytest.fixture
    def predictor(self) -> LoongONNXPredictor:
        """使用 dummy 模型的推理器实例"""
        return LoongONNXPredictor(
            model_path=YOLO_MODEL_PATH,
            input_size=640,
            conf_threshold=0.45,
            classes=["magnetic_bead", "button_battery", "scissors"],
        )

    def test_init_model_not_found_raises(self) -> None:
        """模型文件不存在时应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            LoongONNXPredictor(model_path="/nonexistent/path/model.onnx")

    def test_preprocess_output_shape(self, predictor: LoongONNXPredictor) -> None:
        """preprocess() 输出 tensor 应为 (1, 3, H, W) float32 格式"""
        # 创建一个 BGR 格式的测试图像
        image_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        batch, scale, (pad_h, pad_w) = predictor.preprocess(image_bgr)

        assert batch.shape == (1, 3, 640, 640)
        assert batch.dtype == np.float32
        assert scale > 0
        assert pad_h >= 0
        assert pad_w >= 0

    def test_preprocess_square_image_no_padding(self, predictor: LoongONNXPredictor) -> None:
        """640x640 方形图像预处理后 scale 为 1.0，无填充"""
        image_bgr = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        batch, scale, (pad_h, pad_w) = predictor.preprocess(image_bgr)

        assert abs(scale - 1.0) < 1e-5
        assert pad_h == 0
        assert pad_w == 0
        assert batch.shape == (1, 3, 640, 640)

    def test_preprocess_empty_image_raises(self, predictor: LoongONNXPredictor) -> None:
        """空图像或 None 输入应抛出 ValueError"""
        with pytest.raises(ValueError, match="empty or None"):
            predictor.preprocess(None)
        with pytest.raises(ValueError, match="empty or None"):
            predictor.preprocess(np.array([]))

    def test_postprocess_conf_threshold_filtering(
        self, predictor: LoongONNXPredictor
    ) -> None:
        """postprocess() 应过滤掉低于 conf_threshold 的检测结果"""
        # 构造模拟输出：(1, 3, 6)，3 个检测结果
        # 格式为 [x1, y1, x2, y2, confidence, class_id]（角点坐标）
        mock_output = np.array([[
            [270, 200, 370, 280, 0.8, 0],   # 高置信度 -> 保留
            [75, 75, 125, 125, 0.2, 1],      # 低置信度 -> 过滤
            [170, 170, 230, 230, 0.6, 2],    # 中等置信度 -> 保留
        ]], dtype=np.float32)

        # 阈值为 0.45，预期保留 conf=0.8 和 conf=0.6 两个
        original_shape = (480, 640)
        detections = predictor.postprocess(mock_output, original_shape, scale_ratio=1.0, padding=(0, 0))

        assert len(detections) == 2
        assert all(det[4] >= predictor.conf_threshold for det in detections)

    def test_postprocess_empty_when_all_below_threshold(
        self, predictor: LoongONNXPredictor
    ) -> None:
        """所有检测结果低于阈值时应返回空数组"""
        mock_output = np.array([[
            [270, 200, 370, 280, 0.1, 0],
            [75, 75, 125, 125, 0.05, 1],
        ]], dtype=np.float32)

        detections = predictor.postprocess(mock_output, (480, 640), 1.0, (0, 0))
        assert len(detections) == 0
        assert detections.shape == (0, 6)

    def test_postprocess_coordinate_transform(self, predictor: LoongONNXPredictor) -> None:
        """postprocess() 坐标反变换应正确移除 padding 和缩放"""
        # 预设参数：scale=0.5，pad=(200, 160)
        # 输入图像 640x480，缩放后 320x240，居中放置在 640x640 canvas
        # pad_h = (640 - 240) / 2 = 200
        # pad_w = (640 - 320) / 2 = 160
        scale_ratio = 0.5
        pad_h, pad_w = 200, 160

        # 模型输出角点坐标 [x1, y1, x2, y2] = [270, 270, 370, 370]
        # 反变换：x1 = (270 - 160) / 0.5 = 220, y1 = (270 - 200) / 0.5 = 140
        #          x2 = (370 - 160) / 0.5 = 420, y2 = (370 - 200) / 0.5 = 340
        mock_output = np.array([[
            [270, 270, 370, 370, 0.9, 0],
        ]], dtype=np.float32)

        detections = predictor.postprocess(mock_output, (480, 640), scale_ratio, (pad_h, pad_w))
        assert len(detections) == 1

        det = detections[0]
        assert abs(det[0] - 220.0) < 1.0  # x1
        assert abs(det[1] - 140.0) < 1.0  # y1
        assert abs(det[2] - 420.0) < 1.0  # x2
        assert abs(det[3] - 340.0) < 1.0  # y2

    def test_postprocess_clips_to_image_bounds(
        self, predictor: LoongONNXPredictor
    ) -> None:
        """postprocess() 应将坐标裁剪到原始图像范围内"""
        # 一个超出图像边界的检测框（角点坐标格式）
        mock_output = np.array([[
            [-5, -5, 15, 15, 0.9, 0],  # 左上角超出边界
        ]], dtype=np.float32)

        detections = predictor.postprocess(mock_output, (100, 100), 1.0, (0, 0))
        assert len(detections) == 1
        det = detections[0]
        # 所有坐标应 >= 0
        assert det[0] >= 0.0
        assert det[1] >= 0.0
        assert det[2] >= 0.0
        assert det[3] >= 0.0

    def test_postprocess_removes_batch_dimension(
        self, predictor: LoongONNXPredictor
    ) -> None:
        """postprocess() 应正确处理带和不带 batch 维度的输入"""
        # 角点坐标格式 [x1, y1, x2, y2, conf, cls]
        with_batch = [np.array([[
            [270, 200, 370, 280, 0.9, 0],
            [75, 75, 125, 125, 0.8, 1],
        ]], dtype=np.float32)]

        without_batch = [np.array([
            [270, 200, 370, 280, 0.9, 0],
            [75, 75, 125, 125, 0.8, 1],
        ], dtype=np.float32)]

        det1 = predictor.postprocess(with_batch, (480, 640), 1.0, (0, 0))
        det2 = predictor.postprocess(without_batch, (480, 640), 1.0, (0, 0))

        # 两者应产生相同结果
        np.testing.assert_allclose(det1, det2)

    def test_draw_detections_returns_image(self, predictor: LoongONNXPredictor) -> None:
        """draw_detections() 应返回与输入相同尺寸的图像"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = np.array([
            [10, 20, 100, 200, 0.85, 0],
            [50, 60, 150, 250, 0.92, 1],
        ], dtype=np.float32)

        result = predictor.draw_detections(image, detections)
        assert result.shape == image.shape
        # draw_detections 使用 image.copy()，返回的不应是同一对象
        assert result is not image

    def test_draw_detections_empty_detections(self, predictor: LoongONNXPredictor) -> None:
        """draw_detections() 空检测结果应返回原图的副本"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        empty = np.empty((0, 6), dtype=np.float32)

        result = predictor.draw_detections(image, empty)
        np.testing.assert_array_equal(result, image)

    def test_predict_full_pipeline(
        self, predictor: LoongONNXPredictor
    ) -> None:
        """predict() 完整流程应返回 InferenceResult 且各字段合理"""
        image_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = predictor.predict(image_bgr)

        assert isinstance(result, InferenceResult)
        assert result.original_shape == (480, 640)
        assert result.input_shape == (640, 640)
        assert isinstance(result.detections, np.ndarray)
        assert result.total_time > 0
        assert result.preprocess_time >= 0
        assert result.inference_time >= 0
        assert result.postprocess_time >= 0


# ── YOLO26Nano 健康状态与降级测试 ──────────────────────────
# 设计动机：放在文件末尾而非插入 TestLoongONNXPredictor 中间，
# 避免破坏现有类的缩进/边界。


    def test_predict_with_file_path_raises_not_found(
        self, predictor: LoongONNXPredictor
    ) -> None:
        """predict() 传入不存在的文件路径应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            predictor.predict("/nonexistent/image.jpg")

    def test_get_color_cycles_through_palette(self, predictor: LoongONNXPredictor) -> None:
        """_get_color() 对不同 class_id 返回不同颜色，循环使用"""
        colors = [predictor._get_color(i) for i in range(10)]
        assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)
        # 前 8 个应互不相同
        assert len(set(colors[:8])) == 8

    def test_get_label_known_class(self, predictor: LoongONNXPredictor) -> None:
        """_get_label() 对已知类别返回类别名称"""
        label = predictor._get_label(0, 0.95)
        assert "0.95" in label
        assert predictor.classes[0] in label

    def test_get_label_unknown_class(self, predictor: LoongONNXPredictor) -> None:
        """_get_label() 对未知类别返回 class_N"""
        label = predictor._get_label(999, 0.50)
        assert "class_999" in label

    def test_postprocess_with_high_confidence(self, predictor: LoongONNXPredictor) -> None:
        """postprocess() 高置信度检测应全部保留"""
        # 角点坐标格式 [x1, y1, x2, y2, confidence, class_id]
        preds = np.array([
            [270, 270, 370, 370, 0.9, 0],
            [75, 75, 125, 125, 0.8, 1],
            [160, 160, 240, 240, 0.7, 2],
        ], dtype=np.float32)
        outputs = [preds[np.newaxis, ...]]
        dets = predictor.postprocess(outputs, (640, 640), 1.0, (0, 0))
        assert len(dets) == 3


# ── YOLO26Nano 健康状态与降级测试 ──────────────────────────


class TestYOLO26NanoHealth:
    """
    YOLO26Nano 健康快照与降级行为

    覆盖：
        - is_available() 反映加载状态
        - get_health_snapshot() 字段完整
        - 模型文件不存在时 load_model 抛 FileNotFoundError 且 _available=False
        - 推理失败时 infer 返回空列表,error_count 增加
    """

    def test_load_success_marks_available(self) -> None:
        """正常加载后 is_available=True"""
        cfg = DetectionConfig(model_path=YOLO_MODEL_PATH)
        d = YOLO26Nano(cfg)
        d.load_model()
        assert d.is_available() is True

    def test_load_failure_raises_and_marks_unavailable(self) -> None:
        """模型文件不存在时 load_model 抛 FileNotFoundError"""
        cfg = DetectionConfig(model_path="models/nonexistent.onnx")
        d = YOLO26Nano(cfg)
        with pytest.raises((FileNotFoundError, Exception)):
            d.load_model()
        assert d.is_available() is False
        assert d._last_error  # 错误信息已记录

    def test_health_snapshot_fields(self) -> None:
        """get_health_snapshot() 包含运维所需的全部字段"""
        cfg = DetectionConfig(model_path=YOLO_MODEL_PATH)
        d = YOLO26Nano(cfg)
        d.load_model()
        snap = d.get_health_snapshot()
        assert snap["available"] is True
        assert snap["model_path"].endswith(".onnx")
        assert "input_size" in snap
        assert "conf_threshold" in snap
        assert snap["last_inference_ts"] == 0.0  # 未推理
        assert snap["success_count"] == 0
        assert snap["error_count"] == 0
        assert snap["last_error"] == ""

    def test_inference_failure_returns_empty_and_increments_error(self) -> None:
        """推理抛异常时 infer 返回空列表,error_count +1,不向上抛"""
        cfg = DetectionConfig(model_path=YOLO_MODEL_PATH)
        d = YOLO26Nano(cfg)
        d.load_model()

        with patch.object(
            d._predictor, "predict", side_effect=RuntimeError("infer boom")
        ):
            result = d.infer(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []
        assert d._error_count == 1
        assert "infer boom" in d._last_error

    def test_successful_inference_increments_success_counter(self) -> None:
        """成功推理时 success_count +1, last_inference_ts 更新"""
        cfg = DetectionConfig(model_path=YOLO_MODEL_PATH)
        d = YOLO26Nano(cfg)
        d.load_model()

        before = d._last_inference_ts
        d.infer(np.zeros((480, 640, 3), dtype=np.uint8))
        assert d._success_count == 1
        assert d._last_inference_ts >= before
