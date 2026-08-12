"""
将自定义训练的 YOLO 权重 best.pt 转换为 ONNX 格式

- best.pt -> best.onnx
- 输入: (1, 3, 640, 640) float32
- 输出: (1, 300, 6) NMS-Free 格式 [x1, y1, x2, y2, conf, cls]
- 与 inference_onnx.py 后处理逻辑完全兼容

转换日志输出到控制台和 convert_best_log.txt
"""
import os
import sys
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('convert_best_log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

MODELS_DIR = 'models'
INPUT_SIZE = 640


def convert():
    """执行 best.pt -> best.onnx 转换"""
    pt_path = os.path.join(MODELS_DIR, 'best.pt')
    onnx_path = os.path.join(MODELS_DIR, 'best.onnx')

    logger.info('=' * 60)
    logger.info('开始转换: best.pt -> best.onnx')
    logger.info('=' * 60)

    if not os.path.exists(pt_path):
        logger.error(f'源文件不存在: {pt_path}')
        return False

    logger.info(f'源文件大小: {os.path.getsize(pt_path) / (1024 * 1024):.2f} MB')

    try:
        import torch
        import onnx
        import onnxruntime as ort
        import numpy as np

        logger.info(f'PyTorch: {torch.__version__} | ONNX: {onnx.__version__} | ORT: {ort.__version__}')

        # ── 加载 PT 模型 ──────────────────────────────────────
        # ultralytics 的 torch.load 直接返回 nn.Module，方便手动导出
        logger.info('加载 PT 模型...')
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

        if not (isinstance(checkpoint, dict) and 'model' in checkpoint):
            logger.error('无法识别的模型格式（缺少 model 键）')
            return False

        model = checkpoint['model']
        logger.info(f'模型类型: {type(model).__name__}')

        # 类别名（写入 ONNX 元数据，便于下游解析）
        names = getattr(model, 'names', None)
        if names:
            logger.info(f'类别: {names}')

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f'参数数量: {total_params:,}')

        # ── FP16 -> FP32 修复 ─────────────────────────────────
        # 部分训练在混合精度下保存的权重为 FP16，ONNX 导出前需统一为 FP32
        weight_dtypes = set(p.dtype for p in model.parameters())
        logger.info(f'权重数据类型: {weight_dtypes}')
        if torch.float16 in weight_dtypes:
            logger.info('检测到 FP16 权重，转换为 FP32...')
            model = model.float()

        model.eval()

        # ── 测试 PyTorch 前向 ─────────────────────────────────
        dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
        logger.info(f'虚拟输入: {dummy_input.shape}')

        logger.info('测试 PyTorch 前向...')
        with torch.no_grad():
            torch_output = model(dummy_input)

        # 兼容 tuple 输出：取第一个张量作为导出目标
        if isinstance(torch_output, (list, tuple)):
            torch_out_tensor = torch_output[0]
            logger.info(f'PyTorch 输出为 tuple (len={len(torch_output)})，取 out[0]: {torch_out_tensor.shape}')
        else:
            torch_out_tensor = torch_output
            logger.info(f'PyTorch 输出: {torch_out_tensor.shape}')

        # ── 导出 ONNX ────────────────────────────────────────
        logger.info('开始 ONNX 导出...')
        start_time = time.time()

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=13,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            do_constant_folding=True,
            export_params=True,
        )

        export_time = time.time() - start_time
        logger.info(f'ONNX 导出完成，耗时: {export_time:.2f} 秒')

        # ── 写入类别名元数据 ──────────────────────────────────
        if names:
            onnx_model = onnx.load(onnx_path)
            meta = onnx_model.metadata_props.add()
            meta.key = 'names'
            meta.value = str(names)
            onnx.save_model(onnx_model, onnx_path)
            logger.info(f'已写入 ONNX 元数据: names={names}')

        # ── 验证 ONNX 模型 ────────────────────────────────────
        logger.info('验证 ONNX 模型...')
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info('ONNX checker: PASSED')

        for inp in onnx_model.graph.input:
            shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
            logger.info(f'  输入: {inp.name}, shape={shape}')
        for out in onnx_model.graph.output:
            shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
            logger.info(f'  输出: {out.name}, shape={shape}')

        output_size = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f'输出文件大小: {output_size:.2f} MB')

        # ── ONNX Runtime 推理测试 ─────────────────────────────
        logger.info('ONNX Runtime 推理测试...')
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_meta = sess.get_inputs()[0]
        logger.info(f'ORT 输入: {input_meta.name}, shape={input_meta.shape}, type={input_meta.type}')

        test_input = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

        # warm-up
        sess.run(None, {input_meta.name: test_input})

        start_time = time.time()
        for _ in range(20):
            ort_output = sess.run(None, {input_meta.name: test_input})[0]
        avg_inference = (time.time() - start_time) / 20 * 1000

        logger.info(f'ORT 推理耗时(平均): {avg_inference:.1f} ms')
        logger.info(f'ORT 输出 shape: {ort_output.shape}, dtype: {ort_output.dtype}')
        logger.info(f'ORT 输出范围: [{ort_output.min():.4f}, {ort_output.max():.4f}]')

        # ── PyTorch vs ONNX 数值一致性校验 ────────────────────
        logger.info('数值一致性校验 (PyTorch vs ONNX)...')
        # 使用相同的固定输入做对比
        fixed_input = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.from_numpy(fixed_input))
            pt_out = pt_out[0] if isinstance(pt_out, (list, tuple)) else pt_out
            pt_out_np = pt_out.cpu().numpy()

        ort_out = sess.run(None, {input_meta.name: fixed_input})[0]

        max_diff = np.abs(pt_out_np - ort_out).max()
        mean_diff = np.abs(pt_out_np - ort_out).mean()
        logger.info(f'最大绝对误差: {max_diff:.6e}')
        logger.info(f'平均绝对误差: {mean_diff:.6e}')

        if max_diff < 1e-3:
            logger.info('数值一致性: PASSED (误差 < 1e-3)')
        else:
            logger.warning(f'数值一致性: 误差较大 (max={max_diff:.6e})，端侧推理可能存在精度偏差')

        logger.info('=' * 60)
        logger.info('转换成功!')
        logger.info(f'  输出: {onnx_path}')
        logger.info(f'  输入: (batch, 3, {INPUT_SIZE}, {INPUT_SIZE}) float32')
        logger.info(f'  输出: (batch, 300, 6) [x1, y1, x2, y2, conf, cls]')
        if names:
            logger.info(f'  类别: {names}')
        logger.info('=' * 60)
        return True

    except Exception as e:
        logger.error(f'转换失败: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    logger.info(f'转换开始: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'工作目录: {os.getcwd()}')
    success = convert()
    logger.info(f'转换结束: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('详细日志已保存到: convert_best_log.txt')
    sys.exit(0 if success else 1)
