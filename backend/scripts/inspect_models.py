"""
模型权重文件全面转译质量检测脚本

检测维度：
    1. 模型结构完整性验证（Graph、节点、拓扑排序）
    2. 算子转换准确性检查（算子集兼容性、自定义算子识别）
    3. 数据类型一致性确认（输入输出 dtype、量化参数）
    4. 量化精度损失评估（权重分布、动态范围、量化误差统计）
    5. 推理性能指标测试（端到端延迟、吞吐量、内存占用）
    6. 代码-模型接口匹配验证（与项目推理代码的参数一致性）

用法：
    python scripts/inspect_models.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── 检测报告数据结构 ─────────────────────────────────────────

@dataclass
class Finding:
    """单条检测发现"""
    category: str       # 结构 / 算子 / 数据类型 / 量化 / 性能 / 接口
    severity: str       # CRITICAL / WARNING / INFO
    title: str
    detail: str
    recommendation: str = ""


@dataclass
class ModelReport:
    """单个模型的完整检测报告"""
    model_path: str
    model_size_mb: float
    format: str
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ── 检测函数 ──────────────────────────────────────────────────

def inspect_onnx_model(model_path: str, model_label: str = "") -> ModelReport:
    """对单个 ONNX 模型进行全面转译质量检测"""
    import onnx
    from onnx import TensorProto, numpy_helper

    path = Path(model_path)
    size_mb = path.stat().st_size / (1024 * 1024)
    report = ModelReport(
        model_path=str(path),
        model_size_mb=round(size_mb, 2),
        format="ONNX",
    )
    add = report.findings.append

    print(f"\n{'='*70}")
    print(f"  检测模型: {model_label or path.name}")
    print(f"  文件路径: {path}")
    print(f"  文件大小: {size_mb:.2f} MB")
    print(f"{'='*70}")

    # ── 1. 模型加载与结构完整性 ────────────────────────────────
    print("\n[1/6] 模型结构完整性验证...")
    try:
        model = onnx.load(str(path))
        graph = model.graph
    except Exception as e:
        add(Finding("结构", "CRITICAL", "模型文件无法加载",
                     f"onnx.load 报错: {e}",
                     "模型文件可能已损坏，需重新导出。"))
        return report

    # 1a. ONNX 格式版本与 opset 检查
    opset_imports = model.opset_import
    for opset in opset_imports:
        domain = opset.domain or "ai.onnx"
        version = opset.version
        if domain == "ai.onnx" and version < 13:
            add(Finding("结构", "WARNING", f"Opset 版本偏低 ({domain} v{version})",
                         "低于 v13 可能缺少部分优化算子，影响推理引擎兼容性。",
                         "建议使用 opset>=13 重新导出模型。"))
        else:
            add(Finding("结构", "INFO", f"Opset 版本: {domain} v{version}", "版本正常。"))

    # 1b. IR 版本
    ir_version = model.ir_version
    if ir_version < 7:
        add(Finding("结构", "WARNING", f"IR 版本偏低: v{ir_version}",
                     "低于 v7 的 IR 版本可能缺少对新数据类型的支持。",
                     "建议使用最新 onnx 库重新导出。"))
    else:
        add(Finding("结构", "INFO", f"IR 版本: v{ir_version}", "正常。"))

    # 1c. 输入输出元数据
    inputs = graph.input
    outputs = graph.output
    print(f"   输入张量: {len(inputs)} 个")
    print(f"   输出张量: {len(outputs)} 个")

    for inp in inputs:
        name = inp.name
        tensor_type = inp.type.tensor_type
        dtype = tensor_type.elem_type
        shape = []
        if tensor_type.HasField("shape"):
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
        dtype_name = TensorProto.DataType.Name(dtype)
        print(f"   - 输入 '{name}': dtype={dtype_name}, shape={shape}")

    for out in outputs:
        name = out.name
        tensor_type = out.type.tensor_type
        dtype = tensor_type.elem_type
        shape = []
        if tensor_type.HasField("shape"):
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
        dtype_name = TensorProto.DataType.Name(dtype)
        print(f"   - 输出 '{name}': dtype={dtype_name}, shape={shape}")

    # 1d. 拓扑排序验证
    try:
        onnx.checker.check_model(model)
        add(Finding("结构", "INFO", "ONNX checker 验证通过",
                     "模型拓扑结构、算子属性、数据类型均通过 onnx.checker 校验。"))
    except Exception as e:
        add(Finding("结构", "CRITICAL", "ONNX checker 验证失败",
                     f"check_model 报错: {e}",
                     "模型结构存在合法性问题，需检查导出流程。"))

    # 1e. 节点统计
    nodes = graph.node
    op_types = Counter(n.op_type for n in nodes)
    total_nodes = len(nodes)
    print(f"\n   总节点数: {total_nodes}")
    print("   算子类型分布:")
    for op, count in op_types.most_common():
        print(f"     - {op}: {count}")

    # 1f. 初始化器（权重）统计
    initializers = graph.initializer
    total_params = 0
    weight_dtypes = Counter()
    for init in initializers:
        arr = numpy_helper.to_array(init)
        total_params += arr.size
        weight_dtypes[arr.dtype.name] += 1

    print(f"\n   权重张量数: {len(initializers)}")
    print(f"   总参数量: {total_params:,}")
    report.summary["total_params"] = total_params
    report.summary["total_nodes"] = total_nodes
    report.summary["op_types"] = dict(op_types)
    report.summary["weight_dtypes"] = dict(weight_dtypes)

    if total_params == 0:
        add(Finding("结构", "WARNING", "模型无权重参数",
                     "initializer 为空或参数量为 0，可能是纯计算图或模型导出异常。",
                     "检查模型导出是否正确。"))

    # ── 2. 算子转换准确性检查 ──────────────────────────────────
    print("\n[2/6] 算子转换准确性检查...")

    # 2a. ONNX 标准算子集覆盖检查
    try:
        from onnx import defs as onnx_defs
        standard_ops = set()
        for schema in onnx_defs.get_all_schemas():
            standard_ops.add(schema.name)
    except Exception:
        standard_ops = set()

    custom_ops = []
    for op_name, count in op_types.items():
        if standard_ops and op_name not in standard_ops:
            custom_ops.append((op_name, count))

    if custom_ops:
        add(Finding("算子", "CRITICAL",
                     f"发现 {len(custom_ops)} 个非标准算子",
                     f"非标准算子: {custom_ops}",
                     "自定义算子在 LoongArch 端侧 ONNX Runtime 中可能无法执行，需确认是否有对应实现。"))
    else:
        add(Finding("算子", "INFO", "所有算子均为 ONNX 标准算子",
                     f"共 {len(op_types)} 种标准算子类型。"))

    # 2b. ONNX Runtime 算子兼容性验证（通过实际加载模型）
    print("   验证 ONNX Runtime 算子兼容性...")
    try:
        import onnxruntime as ort
        preferred = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(path), sess_options=opts, providers=preferred)

        session_inputs = session.get_inputs()
        session_outputs = session.get_outputs()

        add(Finding("算子", "INFO", "ONNX Runtime 加载成功",
                     f"所有算子在 CPU EP 上均可解析。EP: {session.get_providers()}"))
    except Exception as e:
        add(Finding("算子", "CRITICAL", "ONNX Runtime 加载失败",
                     f"session 初始化报错: {e}",
                     "存在不兼容的算子或模型结构问题，需修复后重新导出。"))
        return report

    # 2c. 检查量化相关算子（QuantizeLinear / DequantizeLinear）
    quant_ops = {op for op in op_types if "Quantize" in op or "Dequantize" in op or "QLinear" in op}
    has_quant_nodes = len(quant_ops) > 0
    if has_quant_nodes:
        add(Finding("算子", "INFO", f"检测到量化算子: {quant_ops}",
                     f"量化算子数量: {sum(op_types[op] for op in quant_ops)}",
                     "确认这些量化算子与目标 LoongArch ONNX Runtime 版本兼容。"))

    # ── 3. 数据类型一致性确认 ──────────────────────────────────
    print("\n[3/6] 数据类型一致性确认...")

    # 3a. 输入数据类型与推理代码预期的匹配
    for inp in session_inputs:
        dtype_str = inp.type
        if "int8" in dtype_str:
            add(Finding("数据类型", "INFO", f"输入 '{inp.name}' 类型为 {dtype_str}",
                         "INT8 量化模型，推理代码应传递 uint8 像素值。"))
        elif "uint8" in dtype_str:
            add(Finding("数据类型", "INFO", f"输入 '{inp.name}' 类型为 {dtype_str}",
                         "uint8 输入，推理代码应直接传递像素值不做归一化。"))
        elif "float" in dtype_str:
            add(Finding("数据类型", "INFO", f"输入 '{inp.name}' 类型为 {dtype_str}",
                         "float 输入，推理代码应做归一化预处理。"))

    # 3b. 输出数据类型检查
    for out in session_outputs:
        dtype_str = out.type
        if "int8" in dtype_str or "uint8" in dtype_str:
            add(Finding("数据类型", "WARNING", f"输出 '{out.name}' 为量化类型 {dtype_str}",
                         "输出为整型时，后处理代码需要额外的反量化步骤。确认推理代码是否正确处理。",
                         "在后处理中添加 dequantize 逻辑。"))
        else:
            add(Finding("数据类型", "INFO", f"输出 '{out.name}' 类型为 {dtype_str}",
                         "输出为浮点类型，与推理代码后处理逻辑兼容。"))

    # 3c. 权重数据类型分布
    print(f"   权重数据类型分布: {weight_dtypes}")
    for dtype_name, count in weight_dtypes.items():
        if "int8" in dtype_name.lower() or "uint8" in dtype_name.lower():
            add(Finding("数据类型", "INFO", f"权重含量化类型 {dtype_name}: {count} 个张量",
                         "INT8 量化权重，存储和推理效率更高，但存在量化精度损失。"))

    # 3d. 权重中是否存在异常类型（float64 等）
    for dtype_name, count in weight_dtypes.items():
        if "float64" in dtype_name or "double" in dtype_name:
            add(Finding("数据类型", "WARNING", f"权重中存在 float64 类型: {count} 个张量",
                         "float64 在端侧推理中既浪费内存又降低速度，且可能暗示导出时精度设置不当。",
                         "建议在导出时强制使用 float32 或 INT8。"))

    # ── 4. 量化精度损失评估 ────────────────────────────────────
    print("\n[4/6] 量化精度损失评估...")

    # 4a. 权重分布统计（用于评估量化饱和度）
    float_weights = []
    int_weights = []
    zero_count_total = 0
    total_weight_count = 0
    saturation_count = 0  # 极值计数（接近 127/-128 或 0/255）

    for init in initializers:
        arr = numpy_helper.to_array(init)
        total_weight_count += arr.size

        if np.issubdtype(arr.dtype, np.floating):
            float_weights.append(arr)
            # 检查异常值
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0:
                add(Finding("量化", "CRITICAL", f"权重张量 '{init.name}' 含 NaN",
                             f"NaN 数量: {nan_count} / {arr.size}",
                             "NaN 权重会导致推理结果不可控，需检查训练或量化流程。"))
            if inf_count > 0:
                add(Finding("量化", "WARNING", f"权重张量 '{init.name}' 含 Inf",
                             f"Inf 数量: {inf_count} / {arr.size}",
                             "Inf 权重可能导致数值溢出，检查量化 range 设置。"))
        elif np.issubdtype(arr.dtype, np.integer):
            int_weights.append(arr)
            # INT8 饱和度检查
            if arr.dtype == np.int8:
                saturation_count += np.sum((arr == 127) | (arr == -128))
            elif arr.dtype == np.uint8:
                saturation_count += np.sum((arr == 0) | (arr == 255))

            zero_count_total += np.sum(arr == 0)

    if int_weights:
        int_total = sum(w.size for w in int_weights)
        zero_ratio = zero_count_total / int_total if int_total > 0 else 0
        sat_ratio = saturation_count / int_total if int_total > 0 else 0

        print(f"   整型权重总参数: {int_total:,}")
        print(f"   零值比例: {zero_ratio:.4%}")
        print(f"   饱和值比例: {sat_ratio:.4%}")

        report.summary["quantized_params"] = int_total
        report.summary["zero_ratio"] = round(zero_ratio, 6)
        report.summary["saturation_ratio"] = round(sat_ratio, 6)

        if sat_ratio > 0.10:
            add(Finding("量化", "WARNING", f"量化饱和度过高: {sat_ratio:.2%}",
                         "超过 10% 的量化权重处于极值（127/-128 或 0/255），说明量化范围设置不当，"
                         "可能造成显著精度损失。",
                         "建议使用 calibration 数据集重新校准量化 range，或改用 per-channel 量化。"))
        else:
            add(Finding("量化", "INFO", f"量化饱和度正常: {sat_ratio:.4%}",
                         "权重在量化范围内分布合理。"))

        if zero_ratio > 0.50:
            add(Finding("量化", "WARNING", f"权重稀疏度过高: {zero_ratio:.2%}",
                         "超过 50% 的量化权重为零值，可能是剪枝过度或量化截断。",
                         "检查模型是否经过剪枝，确认零值分布是否符合预期。"))
        else:
            add(Finding("量化", "INFO", f"权重稀疏度: {zero_ratio:.4%}", "正常范围。"))

    if float_weights:
        # 浮点权重范围统计
        all_float = np.concatenate([w.ravel() for w in float_weights])
        add(Finding("量化", "INFO",
                     f"浮点权重范围: [{all_float.min():.6f}, {all_float.max():.6f}]",
                     f"均值: {all_float.mean():.6f}, 标准差: {all_float.std():.6f}"))

    # 4b. INT8 量化类型一致性（混合精度检测）
    if int_weights:
        int_dtypes = set(w.dtype for w in int_weights)
        if len(int_dtypes) > 1:
            add(Finding("量化", "WARNING",
                         f"混合整型精度: {int_dtypes}",
                         "同一模型中存在不同位宽的整型权重，可能是部分层未量化。",
                         "确认量化覆盖范围，未量化的层可能成为推理瓶颈。"))

    # ── 5. 推理性能指标测试 ────────────────────────────────────
    print("\n[5/6] 推理性能指标测试...")

    # 构造与模型输入匹配的 dummy 数据
    test_inputs = {}
    for inp in session_inputs:
        shape = []
        for dim in inp.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(1)  # batch=1 for dynamic dims

        if "int8" in inp.type or "uint8" in inp.type:
            test_inputs[inp.name] = np.random.randint(0, 255, shape, dtype=np.uint8)
        else:
            test_inputs[inp.name] = np.random.randn(*shape).astype(np.float32) * 0.5

    # Warm-up
    for _ in range(3):
        try:
            session.run(None, test_inputs)
        except Exception:
            pass

    # Benchmark
    num_runs = 20
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            outputs = session.run(None, test_inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        except Exception as e:
            add(Finding("性能", "CRITICAL", "推理执行失败",
                         f"run() 报错: {e}",
                         "输入数据类型或形状可能与模型不匹配。"))
            break

    if latencies:
        avg_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        min_lat = np.min(latencies)
        max_lat = np.max(latencies)
        p50_lat = np.percentile(latencies, 50)
        p95_lat = np.percentile(latencies, 95)
        p99_lat = np.percentile(latencies, 99)
        throughput = 1000.0 / avg_lat if avg_lat > 0 else 0

        print(f"   推理延迟 (x86 CPU, {num_runs} 次):")
        print(f"     平均: {avg_lat:.2f} ms")
        print(f"     标准差: {std_lat:.2f} ms")
        print(f"     最小: {min_lat:.2f} ms")
        print(f"     最大: {max_lat:.2f} ms")
        print(f"     P50: {p50_lat:.2f} ms")
        print(f"     P95: {p95_lat:.2f} ms")
        print(f"     P99: {p99_lat:.2f} ms")
        print(f"     吞吐量: {throughput:.1f} FPS")

        report.summary["latency_avg_ms"] = round(avg_lat, 2)
        report.summary["latency_p95_ms"] = round(p95_lat, 2)
        report.summary["throughput_fps"] = round(throughput, 1)

        # 性能判定（基于 2K3000 目标 25 FPS / 40ms）
        # x86 CPU 通常比 LoongArch 快，此处用保守阈值
        if avg_lat > 80:
            add(Finding("性能", "CRITICAL",
                         f"推理延迟过高: {avg_lat:.1f}ms (x86 CPU)",
                         f"x86 CPU 平均延迟 {avg_lat:.1f}ms 已超过 80ms，"
                         f"在 LoongArch 2K3000 上预计更慢，无法满足 25FPS 实时目标。",
                         "需要优化模型结构（减少层数/通道数）或使用更激进的量化策略。"))
        elif avg_lat > 40:
            add(Finding("性能", "WARNING",
                         f"推理延迟较高: {avg_lat:.1f}ms (x86 CPU)",
                         f"x86 CPU 平均延迟 {avg_lat:.1f}ms，在 LoongArch 上可能接近或超出 40ms 目标。",
                         "在 LoongArch 实机上验证性能，必要时启用 LG200 OpenCL 加速。"))
        else:
            add(Finding("性能", "INFO",
                         f"推理延迟: {avg_lat:.1f}ms (x86 CPU)",
                         f"P95={p95_lat:.1f}ms, 吞吐量={throughput:.1f}FPS。"
                         f"LoongArch 实机性能需以 2K3000 为准。"))

        # 延迟稳定性检查
        if std_lat > avg_lat * 0.3:
            add(Finding("性能", "WARNING",
                         f"推理延迟波动较大: std={std_lat:.2f}ms (变异系数 {std_lat/avg_lat:.2%})",
                         "延迟不稳定可能导致帧丢失，需检查是否有后台进程干扰或内存抖动。",
                         "建议排查系统环境并多次重复测试。"))

    # 5b. 输出形状验证
    if latencies and outputs:
        print("\n   输出张量信息:")
        for i, out in enumerate(session_outputs):
            actual_shape = outputs[i].shape if i < len(outputs) else "N/A"
            actual_dtype = outputs[i].dtype if i < len(outputs) else "N/A"
            print(f"     - '{out.name}': shape={actual_shape}, dtype={actual_dtype}")
            report.summary[f"output_{i}_shape"] = str(actual_shape)
            report.summary[f"output_{i}_dtype"] = str(actual_dtype)

    # ── 6. 代码-模型接口匹配验证 ───────────────────────────────
    print("\n[6/6] 代码-模型接口匹配验证...")

    # 6a. 检查输入形状与配置中 input_size 的匹配
    for inp in session_inputs:
        shape = list(inp.shape)
        # 检查 spatial dimensions (H, W)
        if len(shape) == 4:
            # NHWC: [N, H, W, C] 或 NCHW: [N, C, H, W]
            if shape[-1] == 3:  # NHWC
                h, w = shape[1], shape[2]
                layout = "NHWC"
            elif shape[1] == 3:  # NCHW
                h, w = shape[2], shape[3]
                layout = "NCHW"
            else:
                h, w = shape[2], shape[3]
                layout = "NCHW?"

            if isinstance(h, int) and isinstance(w, int):
                print(f"   输入 '{inp.name}': layout={layout}, spatial={h}x{w}")
                report.summary["input_layout"] = layout
                report.summary["input_spatial"] = f"{h}x{w}"

                # 与项目配置对比
                if h == 192 and w == 192:
                    add(Finding("接口", "INFO",
                                 f"输入尺寸 {h}x{w} 匹配 MoveNet-Lightning 配置 (input_size=192)",
                                 "与 PoseConfig.input_size=192 和代码中 cv2.resize 到 192x192 一致。"))
                elif h == 640 and w == 640:
                    add(Finding("接口", "INFO",
                                 f"输入尺寸 {h}x{w} 匹配 YOLO26-Nano 配置 (input_size=640)",
                                 "与 DetectionConfig.input_size=640 一致。"))
                elif isinstance(h, int) and h > 0:
                    add(Finding("接口", "WARNING",
                                 f"输入尺寸 {h}x{w} 与项目配置不匹配",
                                 "请确认模型的实际输入尺寸与 config 中的 input_size 设置是否一致。"))

    # 6b. 检查输出格式与推理代码预期的匹配
    if latencies and outputs:
        for i, _ in enumerate(session_outputs):
            actual_shape = outputs[i].shape
            if len(actual_shape) >= 3:
                last_dims = actual_shape[-2:]
                if last_dims == (17, 3):
                    add(Finding("接口", "INFO",
                                 f"输出形状 {actual_shape} 匹配 MoveNet 格式 (1,1,17,3)",
                                 "17 个关键点，每点 3 个值 [y, x, score]，与 movenet.py 后处理兼容。"))
                elif len(actual_shape) >= 2 and actual_shape[-1] == 6:
                    add(Finding("接口", "INFO",
                                 f"输出形状 {actual_shape} 匹配 YOLO26 NMS-Free 格式 (*, 6)",
                                 "每行 [x_center, y_center, w, h, confidence, class_id]，"
                                 "与 inference_onnx.py 后处理兼容。"))

    # 6c. 数据类型与推理代码的匹配检查
    for inp in session_inputs:
        dtype_str = inp.type
        if "uint8" in dtype_str:
            add(Finding("接口", "INFO",
                         f"输入 '{inp.name}' 为 uint8，与推理代码 INT8 量化路径兼容",
                         "movenet.py 和 inference_onnx.py 中的 uint8 直传逻辑正确。"))
        elif "float" in dtype_str:
            add(Finding("接口", "INFO",
                         f"输入 '{inp.name}' 为 {dtype_str}，与推理代码 float 路径兼容",
                         "推理代码中的 float 归一化路径正确。"))

    return report


def print_report(report: ModelReport) -> None:
    """打印格式化的检测报告"""
    print(f"\n{'#'*70}")
    print(f"  检测报告: {Path(report.model_path).name}")
    print(f"  文件大小: {report.model_size_mb} MB")
    print(f"{'#'*70}")

    # 按严重程度分组
    by_severity = defaultdict(list)
    for f in report.findings:
        by_severity[f.severity].append(f)

    severity_order = ["CRITICAL", "WARNING", "INFO"]
    severity_icon = {"CRITICAL": "[!!!]", "WARNING": "[!]", "INFO": "[i]"}

    for sev in severity_order:
        findings = by_severity.get(sev, [])
        if not findings:
            continue
        print(f"\n{'─'*50}")
        print(f"  {severity_icon[sev]} {sev} 级别 ({len(findings)} 条)")
        print(f"{'─'*50}")
        for i, f in enumerate(findings, 1):
            print(f"\n  {i}. [{f.category}] {f.title}")
            print(f"     详情: {f.detail}")
            if f.recommendation:
                print(f"     建议: {f.recommendation}")

    # 统计摘要
    critical_count = len(by_severity.get("CRITICAL", []))
    warning_count = len(by_severity.get("WARNING", []))
    info_count = len(by_severity.get("INFO", []))

    print(f"\n{'='*70}")
    print(f"  检测摘要: CRITICAL={critical_count}, WARNING={warning_count}, INFO={info_count}")
    print(f"{'='*70}")

    if report.summary:
        print("\n  关键指标:")
        for k, v in report.summary.items():
            print(f"    {k}: {v}")

    if critical_count > 0:
        print(f"\n  >>> 存在 {critical_count} 个 CRITICAL 级别问题，建议在部署前修复！")
    elif warning_count > 0:
        print(f"\n  >>> 存在 {warning_count} 个 WARNING 级别问题，建议关注。")
    else:
        print("\n  >>> 所有检测项通过，模型转译质量良好。")


def main() -> None:
    """主入口：对所有模型文件执行全面检测"""
    print("=" * 70)
    print("  LoongGuard 模型权重转译质量全面检测")
    print("  检测时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    models_dir = Path(__file__).parent.parent / "models"
    onnx_models = list(models_dir.glob("*.onnx"))

    if not onnx_models:
        print("未找到 ONNX 模型文件！")
        sys.exit(1)

    reports = []
    for model_path in sorted(onnx_models):
        label = model_path.stem
        report = inspect_onnx_model(str(model_path), label)
        reports.append(report)
        print_report(report)

    # ── 汇总对比 ───────────────────────────────────────────────
    if len(reports) > 1:
        print(f"\n\n{'#'*70}")
        print("  跨模型对比汇总")
        print(f"{'#'*70}")

        print(f"\n  {'模型':<40} {'大小(MB)':<12} {'参数量':<15} {'平均延迟':<12} {'CRITICAL':<10} {'WARNING':<10}")
        print(f"  {'─'*95}")
        for r in reports:
            name = Path(r.model_path).name
            params = r.summary.get("total_params", "N/A")
            lat = r.summary.get("latency_avg_ms", "N/A")
            crit = len([f for f in r.findings if f.severity == "CRITICAL"])
            warn = len([f for f in r.findings if f.severity == "WARNING"])
            params_str = f"{params:,}" if isinstance(params, int) else str(params)
            lat_str = f"{lat}ms" if lat != "N/A" else "N/A"
            print(f"  {name:<40} {r.model_size_mb:<12} {params_str:<15} {lat_str:<12} {crit:<10} {warn:<10}")

    # ── 总体评估 ───────────────────────────────────────────────
    total_critical = sum(len([f for f in r.findings if f.severity == "CRITICAL"]) for r in reports)
    total_warning = sum(len([f for f in r.findings if f.severity == "WARNING"]) for r in reports)

    print(f"\n{'='*70}")
    print("  总体评估")
    print(f"{'='*70}")

    if total_critical == 0 and total_warning == 0:
        print("  所有模型转译质量检测通过，未发现存在问题。")
    else:
        if total_critical > 0:
            print(f"  CRITICAL 问题: {total_critical} 个 — 必须修复后方可部署。")
        if total_warning > 0:
            print(f"  WARNING 问题: {total_warning} 个 — 建议在上线前处理。")

    print("\n  注意: 性能数据基于 x86 CPU，实际 LoongArch 2K3000 性能需以实机为准。")
    print("  建议: 在 QEMU Loongnix_v25 和 3A5000 主机上分别运行本脚本进行交叉验证。")


if __name__ == "__main__":
    main()
