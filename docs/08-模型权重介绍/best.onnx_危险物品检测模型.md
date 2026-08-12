# best.onnx - YOLO26 危险物品检测模型

## 基本信息

| 属性 | 值 |
|------|-----|
| 文件名 | `best.onnx` |
| 文件路径 | `backend/models/best.onnx` |
| 文件大小 | 9.80 MB |
| ONNX 版本 | Opset 13 |
| 节点数量 | 769 |
| 模型类型 | YOLO26-Nano 目标检测 |
| 量化状态 | FP32（未量化） |

## 模型用途

**核心功能**：危险物品检测（剪刀、美工刀）

这是 LoongGuard 系统的**主检测模型**，用于实时检测画面中的危险物品，保障幼儿安全。

检测类别：
- `scissor` - 剪刀
- `utility_knife` - 美工刀

## 输入输出格式

### 输入

| 输入名称 | 形状 | 数据类型 | 说明 |
|----------|------|----------|------|
| `images` | (1, 3, 640, 640) | float32 | RGB 图像，已归一化 |

预处理流程：
1. BGR → RGB 转换
2. 缩放至 640×640（保持宽高比，灰色填充）
3. 归一化到 [0, 1] 范围
4. ImageNet 均值/标准差归一化（可选）

### 输出

| 输出名称 | 形状 | 说明 |
|----------|------|------|
| `output` | (1, 300, 6) | NMS-Free 检测结果 |
| 其他 | ... | 内部特征图 |

输出格式 `(*, 6)` 每行：`[x1, y1, x2, y2, confidence, class_id]`

## 在项目中的应用

### 配置引用

```json
// config/default.json
{
  "detection": {
    "model_path": "models/best.onnx",
    "input_size": 640,
    "conf_threshold": 0.80,
    "quantized": false,
    "classes": ["scissor", "utility_knife"]
  }
}
```

### 代码路径

- 推理类：`loongguard/detection/yolo26_nano.py` → `YOLO26Nano`
- 底层引擎：`loongguard/detection/inference_onnx.py` → `LoongONNXPredictor`
- 环境变量覆盖：`LG_DETECTION_CONF_THRESHOLD`

### 使用方式

```python
from config import DetectionConfig
from loongguard.detection.yolo26_nano import YOLO26Nano

config = DetectionConfig()
detector = YOLO26Nano(config)
detector.load_model()

# 推理
bboxes = detector.infer(rgb_frame)
for box in bboxes:
    print(f"{box.class_name}: {box.confidence:.2f}")
```

## 性能指标

| 指标 | x86 CPU | LoongArch 2K3000 (预估) |
|------|---------|------------------------|
| 推理延迟 | ~30-50ms | ~60-100ms |
| 吞吐量 | 20-30 FPS | 10-15 FPS |
| 内存占用 | ~100MB | ~100MB |

## 注意事项

1. **输入尺寸固定**：模型要求 640×640 输入，不同尺寸会导致错误
2. **置信度阈值**：默认 0.80（较高），减少误报
3. **NMS-Free 设计**：YOLO26 端到端输出，无需后处理 NMS
4. **类别映射**：`class_id=0` 对应 `scissor`，`class_id=1` 对应 `utility_knife`

## 模型来源

基于自定义训练的 YOLO26-Nano 模型，通过 `scripts/convert_best_to_onnx.py` 从 PyTorch 格式（`.pt`）转换而来。

## 相关脚本

- 转换脚本：`backend/scripts/convert_best_to_onnx.py`
- 模型检测脚本：`backend/scripts/inspect_models.py`
