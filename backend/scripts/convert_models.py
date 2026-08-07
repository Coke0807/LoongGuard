"""
模型转换脚本：将原始模型转换为 ONNX 格式
- yolo26n.pt -> yolo26_nano_int8.onnx
- 4.tflite -> movenet_lightning_int8.onnx

转换日志将输出到控制台和 convert_log.txt
"""
import os
import sys
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('convert_log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

MODELS_DIR = 'models'


def convert_yolo_pt_to_onnx():
    """将 YOLO PT 模型转换为 ONNX（处理 FP16 权重问题）"""
    pt_path = os.path.join(MODELS_DIR, 'yolo26n.pt')
    onnx_path = os.path.join(MODELS_DIR, 'yolo26_nano_int8.onnx')
    
    logger.info('=' * 60)
    logger.info('开始转换 YOLO 模型: yolo26n.pt -> yolo26_nano_int8.onnx')
    logger.info('=' * 60)
    
    if not os.path.exists(pt_path):
        logger.error(f'源文件不存在: {pt_path}')
        return False
    
    logger.info(f'源文件大小: {os.path.getsize(pt_path) / (1024*1024):.2f} MB')
    
    try:
        import torch
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        logger.info(f'PyTorch 版本: {torch.__version__}')
        logger.info(f'ONNX 版本: {onnx.__version__}')
        logger.info(f'ONNX Runtime 版本: {ort.__version__}')
        
        # 加载模型
        logger.info('加载 PT 模型...')
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
        
        if not (isinstance(checkpoint, dict) and 'model' in checkpoint):
            logger.error('无法识别的模型格式')
            return False
        
        model = checkpoint['model']
        logger.info(f'模型类型: {type(model).__name__}')
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f'模型参数数量: {total_params:,}')
        
        # 检查权重数据类型
        weight_dtypes = set()
        for name, param in model.named_parameters():
            weight_dtypes.add(param.dtype)
        logger.info(f'权重数据类型: {weight_dtypes}')
        
        # 关键修复：将 FP16 权重转换为 FP32
        if torch.float16 in weight_dtypes:
            logger.info('检测到 FP16 权重，正在转换为 FP32...')
            model = model.float()  # 转换所有参数为 FP32
            logger.info('FP16 -> FP32 转换完成')
        
        # 设置为评估模式
        model.eval()
        
        # 创建 dummy 输入 (NCHW, 640x640, FP32)
        dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)
        logger.info(f'虚拟输入形状: {dummy_input.shape}, 类型: {dummy_input.dtype}')
        
        # 测试 PyTorch 模型推理
        logger.info('测试 PyTorch 模型推理...')
        with torch.no_grad():
            torch_output = model(dummy_input)
        logger.info(f'PyTorch 输出形状: {torch_output[0].shape if isinstance(torch_output, tuple) else torch_output.shape}')
        
        # 导出为 ONNX
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
            export_params=True
        )
        
        export_time = time.time() - start_time
        logger.info(f'ONNX 导出完成，耗时: {export_time:.2f} 秒')
        
        # 验证导出的 ONNX 模型
        logger.info('验证 ONNX 模型...')
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info('ONNX 模型验证: PASSED')
        
        # 获取输入输出信息
        inputs = onnx_model.graph.input
        outputs = onnx_model.graph.output
        for inp in inputs:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            logger.info(f'输入: {inp.name}, 形状: {shape}')
        for out in outputs:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            logger.info(f'输出: {out.name}, 形状: {shape}')
        
        # 检查输出文件
        output_size = os.path.getsize(onnx_path) / 1024
        logger.info(f'输出文件大小: {output_size:.1f} KB')
        
        # 统计 ONNX 模型参数
        total_onnx_params = 0
        for init in onnx_model.graph.initializer:
            if init.raw_data:
                arr = np.frombuffer(init.raw_data, dtype=np.float32)
                total_onnx_params += len(arr)
        logger.info(f'ONNX 模型参数数量: {total_onnx_params:,}')
        
        # 使用 ONNX Runtime 测试推理
        logger.info('使用 ONNX Runtime 测试推理...')
        sess = ort.InferenceSession(onnx_path)
        input_meta = sess.get_inputs()[0]
        logger.info(f'ONNX Runtime 输入: {input_meta.name}, 形状: {input_meta.shape}, 类型: {input_meta.type}')
        
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        start_time = time.time()
        outputs = sess.run(None, {'images': test_input})
        inference_time = (time.time() - start_time) * 1000
        
        logger.info(f'ONNX Runtime 推理测试: PASSED')
        logger.info(f'推理耗时: {inference_time:.1f} ms')
        logger.info(f'输出形状: {outputs[0].shape}')
        logger.info(f'输出范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]')
        logger.info(f'输出均值: {outputs[0].mean():.6f}')
        logger.info(f'输出标准差: {outputs[0].std():.6f}')
        
        logger.info('YOLO 模型转换成功!')
        return True
            
    except Exception as e:
        logger.error(f'转换失败: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return False


def convert_tflite_to_onnx():
    """将 TFLite 模型转换为 ONNX（使用 tf2onnx）"""
    tflite_path = os.path.join(MODELS_DIR, '4.tflite')
    onnx_path = os.path.join(MODELS_DIR, 'movenet_lightning_int8.onnx')
    
    logger.info('=' * 60)
    logger.info('开始转换 MoveNet 模型: 4.tflite -> movenet_lightning_int8.onnx')
    logger.info('=' * 60)
    
    if not os.path.exists(tflite_path):
        logger.error(f'源文件不存在: {tflite_path}')
        return False
    
    logger.info(f'源文件大小: {os.path.getsize(tflite_path) / (1024*1024):.2f} MB')
    
    try:
        import subprocess
        
        # 检查 tf2onnx 是否可用
        try:
            import tf2onnx
            logger.info(f'tf2onnx 版本: {tf2onnx.__version__}')
        except ImportError:
            logger.error('tf2onnx 未安装，请执行: pip install tf2onnx')
            logger.info('尝试安装 tf2onnx...')
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'tf2onnx'], 
                         capture_output=True, text=True)
            import tf2onnx
            logger.info(f'tf2onnx 安装成功，版本: {tf2onnx.__version__}')
        
        # 使用 tf2onnx 转换
        logger.info('使用 tf2onnx 转换 TFLite 模型...')
        start_time = time.time()
        
        cmd = [
            sys.executable, '-m', 'tf2onnx.convert',
            '--tflite', tflite_path,
            '--output', onnx_path,
            '--opset', '13'
        ]
        
        logger.info(f'执行命令: {" ".join(cmd)}')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f'tf2onnx 转换失败:\nstdout: {result.stdout}\nstderr: {result.stderr}')
            return False
        
        export_time = time.time() - start_time
        logger.info(f'tf2onnx 转换完成，耗时: {export_time:.2f} 秒')
        if result.stdout:
            logger.info(f'tf2onnx 输出:\n{result.stdout}')
        
        # 验证 ONNX 模型
        if not os.path.exists(onnx_path):
            logger.error('输出文件未生成')
            return False
        
        import onnx
        import numpy as np
        import onnxruntime as ort
        
        logger.info('验证 ONNX 模型...')
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info('ONNX 模型验证: PASSED')
        
        # 获取输入输出信息
        inputs = onnx_model.graph.input
        outputs = onnx_model.graph.output
        for inp in inputs:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            logger.info(f'输入: {inp.name}, 形状: {shape}')
        for out in outputs:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            logger.info(f'输出: {out.name}, 形状: {shape}')
        
        # 检查输出文件
        output_size = os.path.getsize(onnx_path) / 1024
        logger.info(f'输出文件大小: {output_size:.1f} KB')
        
        # 统计参数
        total_onnx_params = 0
        for init in onnx_model.graph.initializer:
            if init.raw_data:
                arr = np.frombuffer(init.raw_data, dtype=np.float32)
                total_onnx_params += len(arr)
        logger.info(f'ONNX 模型参数数量: {total_onnx_params:,}')
        
        # 使用 ONNX Runtime 测试推理
        logger.info('使用 ONNX Runtime 测试推理...')
        sess = ort.InferenceSession(onnx_path)
        input_meta = sess.get_inputs()[0]
        logger.info(f'ONNX Runtime 输入: {input_meta.name}, 形状: {input_meta.shape}, 类型: {input_meta.type}')
        
        # 创建测试输入
        input_shape = [d if isinstance(d, int) else 1 for d in input_meta.shape]
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 如果是量化模型，可能需要 uint8 输入
        if 'uint8' in str(input_meta.type):
            test_input = (test_input * 127 + 128).clip(0, 255).astype(np.uint8)
        
        start_time = time.time()
        outputs = sess.run(None, {input_meta.name: test_input})
        inference_time = (time.time() - start_time) * 1000
        
        logger.info(f'ONNX Runtime 推理测试: PASSED')
        logger.info(f'推理耗时: {inference_time:.1f} ms')
        logger.info(f'输出形状: {outputs[0].shape}')
        logger.info(f'输出范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]')
        
        logger.info('MoveNet 模型转换成功!')
        return True
            
    except Exception as e:
        logger.error(f'转换失败: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info(f'模型转换开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'工作目录: {os.getcwd()}')
    logger.info('')
    
    # 检查依赖
    logger.info('检查依赖库...')
    deps = ['torch', 'onnx', 'onnxruntime', 'numpy', 'tf2onnx']
    
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, '__version__', 'unknown')
            logger.info(f'  {dep}: {version}')
        except ImportError:
            logger.warning(f'  {dep}: 未安装')
    
    logger.info('')
    
    # 执行转换
    results = {}
    
    # 转换 YOLO 模型
    results['yolo'] = convert_yolo_pt_to_onnx()
    logger.info('')
    
    # 转换 MoveNet 模型
    results['movenet'] = convert_tflite_to_onnx()
    logger.info('')
    
    # 汇总结果
    logger.info('=' * 60)
    logger.info('转换结果汇总')
    logger.info('=' * 60)
    logger.info(f'YOLO (yolo26n.pt -> yolo26_nano_int8.onnx): {"成功" if results["yolo"] else "失败"}')
    logger.info(f'MoveNet (4.tflite -> movenet_lightning_int8.onnx): {"成功" if results["movenet"] else "失败"}')
    logger.info('')
    logger.info(f'转换结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('详细日志已保存到: convert_log.txt')
    
    return all(results.values())


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
