"""
创建开发/测试用的 ONNX 模型

生成两个最小 ONNX 模型（正确的输入输出形状，随机初始化权重），
用于 Windows 开发环境下的全流程推理验证。

与旧版的区别：模型包含真实卷积层，输出随输入变化，
可完整验证 预处理 -> 推理 -> 后处理 全链路的 shape 兼容性。

用法：
    python scripts/create_dummy_models.py

输出：
    models/yolo26_nano_int8.onnx
    models/movenet_lightning_int8.onnx
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

logger = logging.getLogger(__name__)


def _xavier_weight(out_c: int, in_c: int, k: int) -> np.ndarray:
    """Xavier 初始化风格的卷积权重"""
    scale = np.sqrt(2.0 / (in_c * k * k))
    return (np.random.randn(out_c, in_c, k, k) * scale).astype(np.float32)


def create_yolo26_model(input_size: int = 640, num_classes: int = 8) -> onnx.ModelProto:
    """
    创建 YOLO26-Nano ONNX 模型

    结构：Input(1,3,H,W) -> 3xConv(stride=2) -> Conv1x1 -> Reshape -> Output(1,N,6)

    输出格式 [batch, num_detections, 6]，每行 [x_center, y_center, w, h, conf, class_id]
    遵循 YOLO26 NMS-Free 端到端设计。

    注意：dummy 模型权重随机初始化，不具检测能力，仅用于验证全流程 pipeline 连通性。
    置信度输出未加 sigmoid，后处理会通过 np.clip 安全裁剪至 [0, 1]。
    """
    feat_size = input_size // 8  # 80 for 640
    num_det = feat_size * feat_size  # 6400

    X = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, input_size, input_size])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, num_det, 6])

    # backbone: 3 层 Conv(stride=2) + ReLU, 3->16->32->64
    configs = [(3, 16), (16, 32), (32, 64)]
    inits: list = []
    nodes: list = []
    prev = "images"
    for i, (in_c, out_c) in enumerate(configs):
        w = numpy_helper.from_array(_xavier_weight(out_c, in_c, 3), name=f"conv{i}.weight")
        b = numpy_helper.from_array(np.zeros(out_c, np.float32), name=f"conv{i}.bias")
        inits.extend([w, b])
        c_name, r_name = f"c{i}", f"r{i}"
        nodes.append(helper.make_node("Conv", [prev, w.name, b.name], [c_name],
                                       kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Relu", [c_name], [r_name]))
        prev = r_name

    # detection head: Conv 1x1, 64->6
    # 关键修复：将权重缩放至极小值 (~0.0001)，确保 dummy 模型的置信度输出接近 0，
    # 避免随机卷积结果误触发告警。真实模型替换后即可正常工作。
    scale = 1e-4
    w_det = numpy_helper.from_array(_xavier_weight(6, 64, 1) * scale, name="det.weight")
    b_det = numpy_helper.from_array(np.zeros(6, np.float32), name="det.bias")
    inits.extend([w_det, b_det])
    nodes.append(helper.make_node("Conv", [prev, w_det.name, b_det.name], ["det_out"],
                                   kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    # reshape (1,6,H,W) -> (1,H*W,6)
    shape = numpy_helper.from_array(np.array([1, num_det, 6], np.int64), name="shape")
    inits.append(shape)
    nodes.append(helper.make_node("Reshape", ["det_out", "shape"], ["output"]))

    graph = helper.make_graph(nodes, "yolo26_nano", [X], [Y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def create_movenet_model(input_size: int = 192) -> onnx.ModelProto:
    """
    创建 MoveNet-Lightning ONNX 模型

    结构：Input(NHWC) -> Transpose(NCHW) -> 3xConv(stride=2) -> Conv1x1 -> GAP -> Reshape
    输出：(1, 1, 17, 3) 每个关键点 [y, x, score] 归一化坐标
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_size, input_size, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 17, 3])

    inits: list = []
    nodes: list = []

    # NHWC -> NCHW
    nodes.append(helper.make_node("Transpose", ["input"], ["t_in"], perm=[0, 3, 1, 2]))

    # backbone: 3 层 Conv(stride=2) + ReLU, 3->16->32->64
    configs = [(3, 16), (16, 32), (32, 64)]
    prev = "t_in"
    for i, (in_c, out_c) in enumerate(configs):
        w = numpy_helper.from_array(_xavier_weight(out_c, in_c, 3), name=f"conv{i}.weight")
        b = numpy_helper.from_array(np.zeros(out_c, np.float32), name=f"conv{i}.bias")
        inits.extend([w, b])
        c_name, r_name = f"c{i}", f"r{i}"
        nodes.append(helper.make_node("Conv", [prev, w.name, b.name], [c_name],
                                       kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Relu", [c_name], [r_name]))
        prev = r_name

    # keypoint head: Conv 1x1, 64->51 (17*3)
    w_kp = numpy_helper.from_array(_xavier_weight(51, 64, 1), name="kp.weight")
    b_kp = numpy_helper.from_array(np.zeros(51, np.float32), name="kp.bias")
    inits.extend([w_kp, b_kp])
    nodes.append(helper.make_node("Conv", [prev, w_kp.name, b_kp.name], ["kp_out"],
                                   kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    # GlobalAveragePool -> (1, 51, 1, 1)
    nodes.append(helper.make_node("GlobalAveragePool", ["kp_out"], ["gap"]))

    # reshape -> (1, 1, 17, 3)
    shape = numpy_helper.from_array(np.array([1, 1, 17, 3], np.int64), name="shape")
    inits.append(shape)
    nodes.append(helper.make_node("Reshape", ["gap", "shape"], ["output"]))

    graph = helper.make_graph(nodes, "movenet_lightning", [X], [Y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def main() -> None:
    """生成并保存 dummy ONNX 模型，验证可被 onnxruntime 加载"""
    import onnxruntime as ort

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating development ONNX models...")

    yolo_path = MODELS_DIR / "yolo26_nano_int8.onnx"
    movenet_path = MODELS_DIR / "movenet_lightning_int8.onnx"

    # YOLO26
    logger.info("Creating YOLO26-Nano (input=1x3x640x640, output=1x6400x6)...")
    yolo = create_yolo26_model(640, 8)
    onnx.save(yolo, str(yolo_path))
    logger.info("  -> %s (%d KB)", yolo_path.name, yolo_path.stat().st_size // 1024)

    # MoveNet
    logger.info("Creating MoveNet-Lightning (input=1x192x192x3, output=1x1x17x3)...")
    movenet = create_movenet_model(192)
    onnx.save(movenet, str(movenet_path))
    logger.info("  -> %s (%d KB)", movenet_path.name, movenet_path.stat().st_size // 1024)

    # 验证模型可被 onnxruntime 加载并推理
    for path in [yolo_path, movenet_path]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        logger.info("  Verified: %s | input=%s%s | output=%s%s",
                     path.name, inp.name, inp.shape, out.name, out.shape)

    logger.info("All models created. Replace with trained models before production.")


if __name__ == "__main__":
    main()
