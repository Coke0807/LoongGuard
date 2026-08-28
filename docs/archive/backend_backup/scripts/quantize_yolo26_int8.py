"""
YOLO26-Nano ONNX 模型 INT8 静态量化脚本 (PTQ)

技术方案:
    使用 onnxruntime.quantization.quantize_static 做 QDQ 格式的 INT8 PTQ。
    QDQ 格式在模型图中插入 QuantizeLinear / DequantizeLinear 算子对，
    在 LoongArch 端侧可被 LG200 GPU 通过 OpenCL 加速执行。

量化策略:
    - 权重: per-channel symmetric INT8 (Conv/MatMul 层)
    - 激活: per-tensor asymmetric UINT8
    - 格式: QDQ (QuantizeLinear + DequantizeLinear 节点)
    - 排除: TopK, GatherElements, Shape 等非算术节点（不适合量化）

数据流变化:
    量化前后，模型的输入输出类型不变 (float32)。
    QuantizeLinear 节点插入在输入之后（模型内部第一层），
    DequantizeLinear 节点插入在输出之前（模型内部最后一层）。
    外部推理代码的 pre/post-process 无需任何修改。

校准数据:
    当前使用随机噪声校准（验证流程可用性）。
    正式部署前应替换为真实教室场景图片以获得最佳精度。

用法:
    python scripts/quantize_yolo26_int8.py [--calib-dir <图片目录>]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

logger = logging.getLogger(__name__)


# ── 校准数据读取器 ────────────────────────────────────────────

class YOLOCalibrationDataReader(CalibrationDataReader):
    """
    YOLO26 模型校准数据读取器

    支持两种模式:
    1. 随机噪声模式 (默认): 用于验证量化流程，精度非最优
    2. 图片目录模式: 加载真实图片做校准，精度最佳

    输入格式与推理代码 preprocess(non-quantized) 一致:
        float32, NCHW, ImageNet 归一化, 640x640
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        input_size: int = 640,
        num_samples: int = 50,
    ) -> None:
        self.input_size = input_size
        self.num_samples = num_samples

        if image_dir and Path(image_dir).is_dir():
            self._images = self._load_images(image_dir)
            print(f"   校准图片数: {len(self._images)}")
        else:
            self._images = None
            print(f"   校准模式: 随机噪声 ({num_samples} 样本)")

        self._iter_count = 0

    def _load_images(self, image_dir: str) -> List[np.ndarray]:
        """加载并预处理校准图片"""
        img_dir = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        img_paths = sorted(
            p for p in img_dir.rglob("*")
            if p.suffix.lower() in extensions
        )

        images = []
        for p in img_paths[: self.num_samples]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            # 与 inference_onnx.py preprocess(non-quantized) 一致
            h, w = img.shape[:2]
            scale = min(self.input_size / w, self.input_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
            pad_h = (self.input_size - new_h) // 2
            pad_w = (self.input_size - new_w) // 2
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
            chw = np.transpose(normalized, (2, 0, 1))
            images.append(chw)

        return images

    def get_next(self) -> Optional[dict]:
        """返回下一批校准数据"""
        if self._iter_count >= self.num_samples:
            return None

        if self._images:
            idx = self._iter_count % len(self._images)
            batch = self._images[idx][np.newaxis, ...]
        else:
            # 随机噪声: 模拟 ImageNet 归一化后的数据分布
            batch = np.random.randn(1, 3, self.input_size, self.input_size).astype(np.float32) * 0.5

        self._iter_count += 1
        return {"images": batch}

    def rewind(self) -> None:
        self._iter_count = 0


# ── 量化执行 ──────────────────────────────────────────────────

def run_quantization(
    model_path: str,
    output_path: str,
    calib_dir: Optional[str] = None,
    num_calib_samples: int = 50,
) -> None:
    """执行 INT8 静态量化并验证"""

    src = Path(model_path)
    dst = Path(output_path)

    print("=" * 60)
    print("  YOLO26-Nano INT8 静态量化 (PTQ)")
    print("=" * 60)

    # 1. 检查源模型
    print(f"\n[1/5] 加载源模型: {src}")
    model = onnx.load(str(src))
    onnx.checker.check_model(model)
    graph = model.graph
    print(f"   节点数: {len(graph.node)}")
    print(f"   输出数: {len(graph.output)}")
    src_size_mb = src.stat().st_size / (1024 * 1024)
    print(f"   文件大小: {src_size_mb:.2f} MB")

    # 2. 准备校准数据
    print(f"\n[2/5] 准备校准数据...")
    calib_reader = YOLOCalibrationDataReader(
        image_dir=calib_dir,
        input_size=640,
        num_samples=num_calib_samples,
    )

    # 3. 执行量化
    print(f"\n[3/5] 执行 INT8 静态量化...")
    print(f"   量化格式: QDQ (QuantizeLinear + DequantizeLinear)")
    print(f"   权重类型: QInt8 (per-channel symmetric)")
    print(f"   激活类型: QUInt8 (per-tensor asymmetric)")

    # 排除不适合量化的算子:
    # - TopK: 排序操作，无算术运算
    # - GatherElements: 索引操作
    # - Shape: 形状提取，纯元数据
    # - Gather: 索引操作
    # - Reshape: 形状变换，无算术运算
    nodes_to_exclude = []
    for node in graph.node:
        if node.op_type in ("TopK", "GatherElements", "Shape", "Gather",
                            "Reshape", "Squeeze", "Unsqueeze", "Concat",
                            "Split", "Slice", "Flatten", "Cast",
                            "ConstantOfShape", "Range", "Expand", "Tile"):
            nodes_to_exclude.append(node.name)

    print(f"   排除节点数: {len(nodes_to_exclude)} (非算术算子)")

    t0 = time.perf_counter()
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=CalibrationMethod.Entropy,
    )
    quant_time = time.perf_counter() - t0
    print(f"   量化耗时: {quant_time:.1f} 秒")

    # 4. 验证量化模型
    print(f"\n[4/5] 验证量化模型...")

    # 4a. ONNX checker
    q_model = onnx.load(str(dst))
    onnx.checker.check_model(q_model)
    print("   ONNX checker 验证通过")

    # 4b. 检查量化节点
    q_graph = q_model.graph
    op_types = {}
    for node in q_graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    ql_count = op_types.get("QuantizeLinear", 0)
    dql_count = op_types.get("DequantizeLinear", 0)
    print(f"   QuantizeLinear 节点:   {ql_count}")
    print(f"   DequantizeLinear 节点: {dql_count}")
    print(f"   总节点数: {len(q_graph.node)}")

    # 4c. 推理一致性验证
    import onnxruntime as ort
    print("\n   推理一致性验证...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    session_orig = ort.InferenceSession(str(src), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
    session_quant = ort.InferenceSession(str(dst), sess_options=opts,
                                          providers=["CPUExecutionProvider"])

    # 使用随机输入验证（与校准数据独立）
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 640, 640).astype(np.float32) * 0.3

    out_orig = session_orig.run(["output"], {"images": test_input})[0]
    out_quant = session_quant.run(["output"], {"images": test_input})[0]

    # 比较: 由于 INT8 量化有精度损失，允许一定误差
    abs_diff = np.abs(out_orig - out_quant)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    # 对于 detection output (1,300,6)，比较前 4 列 (坐标) 和第 5 列 (置信度)
    coord_diff = abs_diff[0, :, :4].mean()
    conf_diff = abs_diff[0, :, 4].mean()

    print(f"   最大绝对误差:  {max_diff:.6f}")
    print(f"   平均绝对误差:  {mean_diff:.6f}")
    print(f"   坐标平均误差:  {coord_diff:.6f}")
    print(f"   置信度平均误差: {conf_diff:.6f}")

    # 精度损失判定
    if conf_diff < 0.05:
        print("   精度评估: 优秀 (置信度误差 < 0.05)")
    elif conf_diff < 0.15:
        print("   精度评估: 良好 (置信度误差 < 0.15)")
    elif conf_diff < 0.30:
        print("   精度评估: 可接受 (置信度误差 < 0.30)")
        print("   建议: 使用真实校准图片替换随机噪声以提高精度")
    else:
        print("   精度评估: 较差 — 建议使用真实校准图片重新量化")

    # 5. 性能对比
    print(f"\n[5/5] 性能对比...")

    # warm-up
    for _ in range(3):
        session_orig.run(None, {"images": test_input})
        session_quant.run(None, {"images": test_input})

    n = 20
    # 原始模型
    t0 = time.perf_counter()
    for _ in range(n):
        session_orig.run(None, {"images": test_input})
    lat_orig = (time.perf_counter() - t0) / n * 1000

    # 量化模型
    t0 = time.perf_counter()
    for _ in range(n):
        session_quant.run(None, {"images": test_input})
    lat_quant = (time.perf_counter() - t0) / n * 1000

    dst_size_mb = dst.stat().st_size / (1024 * 1024)
    speedup = lat_orig / lat_quant if lat_quant > 0 else 0

    print(f"\n   {'指标':<20} {'float32 原始':<20} {'INT8 量化':<20}")
    print(f"   {'─'*58}")
    print(f"   {'文件大小':<20} {src_size_mb:.2f} MB{'':<13} {dst_size_mb:.2f} MB")
    print(f"   {'节点数':<20} {len(graph.node):<20} {len(q_graph.node):<20}")
    print(f"   {'QDQ 节点数':<20} {'0':<20} {ql_count + dql_count:<20}")
    print(f"   {'推理延迟 (CPU)':<20} {lat_orig:.1f} ms{'':<14} {lat_quant:.1f} ms")
    print(f"   {'吞吐量':<20} {1000/lat_orig:.1f} FPS{'':<14} {1000/lat_quant:.1f} FPS")
    print(f"   {'加速比':<20} {'1.00x':<20} {speedup:.2f}x")

    # 总结
    print(f"\n{'='*60}")
    print(f"  量化完成")
    print(f"{'='*60}")
    print(f"  原始模型:  {src} ({src_size_mb:.2f} MB, float32)")
    print(f"  量化模型:  {dst} ({dst_size_mb:.2f} MB, INT8)")
    print(f"  模型压缩:  {src_size_mb/dst_size_mb:.1f}x")
    print(f"  推理加速:  {speedup:.2f}x")
    print(f"  坐标误差:  {coord_diff:.6f}")
    print(f"  置信度误差: {conf_diff:.6f}")

    if calib_dir is None:
        print(f"\n  [!] 当前使用随机噪声校准，精度非最优。")
        print(f"      正式部署前，请用真实教室场景图片重新量化:")
        print(f"      python scripts/quantize_yolo26_int8.py --calib-dir <图片目录>")
        print(f"      建议校准图片数量: 50-200 张，覆盖不同光照/角度/目标物。")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26-Nano INT8 静态量化")
    parser.add_argument(
        "--model",
        default="models/yolo26_nano_int8.onnx",
        help="输入 float32 ONNX 模型路径",
    )
    parser.add_argument(
        "--output",
        default="models/yolo26_nano_int8_quantized.onnx",
        help="输出 INT8 ONNX 模型路径",
    )
    parser.add_argument(
        "--calib-dir",
        default=None,
        help="校准图片目录 (None=随机噪声校准)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="校准样本数量 (默认 50)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_quantization(args.model, args.output, args.calib_dir, args.num_samples)


if __name__ == "__main__":
    main()
