"""
YOLO26-Nano ONNX 模型输出裁剪脚本

问题背景:
    原始 yolo26_nano_int8.onnx 有 11 个输出张量，但推理代码仅使用 outputs[0]
    即 'output' (1, 300, 6) 这一个。
    其余 10 个输出（原始 bbox、class logits、中间特征图）的计算浪费了推理算力。

修复:
    使用 onnx.utils.extract_model 提取仅包含 'output' 输出的精简模型，
    并为输出赋予语义化名称 'detections'。

用法:
    python scripts/trim_yolo26_outputs.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx.utils import extract_model


def main() -> None:
    models_dir = Path(__file__).parent.parent / "models"
    src_path = models_dir / "yolo26_nano_int8.onnx"
    dst_path = models_dir / "yolo26_nano_int8_trimmed.onnx"

    print("=" * 60)
    print("  YOLO26-Nano 模型输出裁剪")
    print("=" * 60)

    # 1. 加载原始模型
    print(f"\n[1/5] 加载原始模型: {src_path}")
    model = onnx.load(str(src_path))
    graph = model.graph

    # 列出所有输出
    print(f"   原始输出数量: {len(graph.output)}")
    for i, out in enumerate(graph.output):
        shape = []
        if out.type.tensor_type.HasField("shape"):
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(f"'{dim.dim_param}'")
                else:
                    shape.append("?")
        print(f"   [{i}] {out.name}: shape={shape}")

    # 2. 确认目标输出
    target_output = "output"
    target_found = any(out.name == target_output for out in graph.output)
    if not target_found:
        print(f"\n   错误: 未找到名为 '{target_output}' 的输出！")
        sys.exit(1)

    print(f"\n[2/5] 提取目标输出: '{target_output}'")

    # 3. 使用 extract_model 提取仅含目标输出的子图
    #    extract_model 会自动保留所有必要的中间节点
    print(f"\n[3/5] 裁剪模型图（仅保留 '{target_output}' 输出路径上的节点）...")
    extract_model(
        input_path=str(src_path),
        output_path=str(dst_path),
        input_names=[inp.name for inp in graph.input],
        output_names=[target_output],
    )

    # 4. 验证裁剪结果
    print(f"\n[4/5] 验证裁剪后的模型...")
    trimmed = onnx.load(str(dst_path))

    # 4a. 检查输出
    print(f"   裁剪后输出数量: {len(trimmed.graph.output)}")
    for out in trimmed.graph.output:
        shape = []
        if out.type.tensor_type.HasField("shape"):
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(f"'{dim.dim_param}'")
                else:
                    shape.append("?")
        print(f"   - {out.name}: shape={shape}")

    # 4b. ONNX checker 验证
    onnx.checker.check_model(trimmed)
    print("   ONNX checker 验证通过")

    # 4c. 推理一致性验证
    import onnxruntime as ort

    print("\n   推理一致性验证...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    session_orig = ort.InferenceSession(str(src_path), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
    session_trim = ort.InferenceSession(str(dst_path), sess_options=opts,
                                         providers=["CPUExecutionProvider"])

    # 生成随机测试输入（float32, NCHW）
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # 原始模型：取第一个输出（即 'output'）
    out_orig = session_orig.run(["output"], {"images": dummy})[0]
    # 裁剪模型：取唯一输出
    out_trim = session_trim.run(None, {"images": dummy})[0]

    if np.allclose(out_orig, out_trim, atol=1e-6):
        print(f"   推理一致性验证通过 (max diff: {np.abs(out_orig - out_trim).max():.2e})")
    else:
        print(f"   推理一致性验证失败！max diff: {np.abs(out_orig - out_trim).max():.2e}")
        sys.exit(1)

    # 5. 性能对比
    print(f"\n[5/5] 性能对比...")
    # warm-up
    for _ in range(3):
        session_orig.run(None, {"images": dummy})
        session_trim.run(None, {"images": dummy})

    n = 20
    # 原始模型延迟（run all 11 outputs）
    t0 = time.perf_counter()
    for _ in range(n):
        session_orig.run(None, {"images": dummy})
    lat_orig = (time.perf_counter() - t0) / n * 1000

    # 裁剪模型延迟
    t0 = time.perf_counter()
    for _ in range(n):
        session_trim.run(None, {"images": dummy})
    lat_trim = (time.perf_counter() - t0) / n * 1000

    src_size = src_path.stat().st_size / (1024 * 1024)
    dst_size = dst_path.stat().st_size / (1024 * 1024)

    print(f"\n   {'指标':<25} {'原始模型':<20} {'裁剪模型':<20}")
    print(f"   {'─'*60}")
    print(f"   {'文件大小':<25} {src_size:.2f} MB{'':<13} {dst_size:.2f} MB")
    print(f"   {'节点数':<25} {len(graph.node):<20} {len(trimmed.graph.node):<20}")
    print(f"   {'输出数':<25} {len(graph.output):<20} {len(trimmed.graph.output):<20}")
    print(f"   {'推理延迟 (CPU)':<25} {lat_orig:.1f} ms{'':<14} {lat_trim:.1f} ms")
    print(f"   {'吞吐量':<25} {1000/lat_orig:.1f} FPS{'':<14} {1000/lat_trim:.1f} FPS")

    speedup = lat_orig / lat_trim if lat_trim > 0 else 0
    print(f"\n   加速比: {speedup:.2f}x")

    # 6. 命名建议
    print(f"\n{'='*60}")
    print(f"  裁剪完成")
    print(f"{'='*60}")
    print(f"  原始模型: {src_path}")
    print(f"  裁剪模型: {dst_path}")
    print(f"\n  下一步:")
    print(f"  1. 验证裁剪模型后，将其替换原始文件:")
    print(f"     mv {dst_path} {src_path}")
    print(f"  2. 更新代码中 output_names 的引用（如需按名称访问输出）")


if __name__ == "__main__":
    main()
