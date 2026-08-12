# det_10g.onnx - 人脸检测模型（RetinaFace）

## 基本信息

| 属性 | 值 |
|------|-----|
| 文件名 | `det_10g.onnx` |
| 文件路径 | `backend/models/buffalo_l/det_10g.onnx` |
| 文件大小 | 16.14 MB |
| ONNX 版本 | Opset 11 |
| 节点数量 | 158 |
| 模型类型 | RetinaFace 人脸检测 |
| 所属套件 | InsightFace Buffalo L |

## 模型用途

**核心功能**：人脸检测（睡姿监测辅助）

检测画面中的人脸，用于睡姿监测逻辑：画面有运动主体但未检测到人脸 → 判定为异常睡姿（俯卧/遮挡/趴睡）。

**应用场景**：
- 幼儿午睡时检测人脸是否可见
- 无人脸 → 可能趴睡/遮挡 → 触发告警

## 输入输出格式

### 输入

| 输入名称 | 形状 | 数据类型 | 说明 |
|----------|------|----------|------|
| `input.1` | (1, 3, 640, 640) | float32 | RGB 图像，归一化到 [0, 1] |

预处理流程：
1. RGB 图像缩放至 640×640（保持宽高比，黑色填充）
2. 归一化到 [0, 1] 范围
3. HWC → CHW 转换

### 输出

模型输出 9 个张量，3 级 FPN（Feature Pyramid Network）：

| 输出索引 | 名称 | 形状 | 说明 |
|----------|------|------|------|
| 0 | scores_stride8 | (12800, 1) | 8x 下采样层置信度 |
| 1 | scores_stride16 | (3200, 1) | 16x 下采样层置信度 |
| 2 | scores_stride32 | (800, 1) | 32x 下采样层置信度 |
| 3 | bboxes_stride8 | (12800, 4) | 8x 下采样层边界框 |
| 4 | bboxes_stride16 | (3200, 4) | 16x 下采样层边界框 |
| 5 | bboxes_stride32 | (800, 4) | 32x 下采样层边界框 |
| 6 | landmarks_stride8 | (12800, 10) | 8x 下采样层关键点 |
| 7 | landmarks_stride16 | (3200, 10) | 16x 下采样层关键点 |
| 8 | landmarks_stride32 | (800, 10) | 32x 下采样层关键点 |

输出后处理：
- 置信度过滤（sigmoid + 阈值）
- Anchor 解码（中心点 + 宽高）
- 坐标反变换（去除填充和缩放）
- NMS 去除重叠框

## 在项目中的应用

### 配置引用

```json
// config/default.json
{
  "face": {
    "backend": "onnx",
    "model_path": "models/buffalo_l/det_10g.onnx",
    "conf_threshold": 0.5,
    "enabled": false,
    "person_class": "person"
  }
}
```

### 代码路径

- 推理类：`loongguard/face/onnx_detector.py` → `ONNXFaceDetector`
- 工厂模式：`loongguard/face/factory.py` → `create_face_detector()`
- 环境变量覆盖：`LG_FACE_ENABLED`、`LG_FACE_BACKEND`、`LG_FACE_CONF_THRESHOLD`

### 使用方式

```python
from config import FaceConfig
from loongguard.face.factory import create_face_detector

config = FaceConfig()
config.enabled = True
detector = create_face_detector(config)

# 检测人脸
faces = detector.detect(rgb_frame)
for face in faces:
    print(f"人脸: ({face.x1}, {face.y1}) - ({face.x2}, {face.y2})")
```

## 性能指标

| 指标 | x86 CPU | LoongArch 2K3000 (预估) |
|------|---------|------------------------|
| 推理延迟 | ~30-50ms | ~60-100ms |
| 内存占用 | ~150MB | ~150MB |
| 检测精度 | mAP ~95% | - |

## 注意事项

1. **默认关闭**：需设置 `LG_FACE_ENABLED=true` 启用
2. **降级策略**：模型加载失败时自动回退到 DummyFaceDetector（桩实现）
3. **Dynamic Input**：模型支持动态输入尺寸，但项目固定使用 640×640
4. **Anchor 配置**：3 级 FPN，strides = [8, 16, 32]，scales = [1.0, 2.0]
5. **NMS 阈值**：IoU = 0.4，最多保留 20 个人脸

## RetinaFace 架构说明

RetinaFace 是一种单阶段人脸检测器，具有以下特点：
- **多尺度检测**：通过 FPN 在不同分辨率层检测不同大小的人脸
- **Anchor-Based**：使用预定义 anchor 框检测人脸
- **端到端训练**：同时优化分类、回归和关键点检测任务

## 模型来源

InsightFace Buffalo L 预训练套件，专为人脸分析任务设计的高质量模型。

## 相关文档

- InsightFace 官方：https://github.com/deepinsight/insightface
- Buffalo L 模型说明：https://github.com/deepinsight/insightface/tree/master/python-package
