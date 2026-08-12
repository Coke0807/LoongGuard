# movenet_lightning_int8.onnx - MoveNet 姿态估计模型

## 基本信息

| 属性 | 值 |
|------|-----|
| 文件名 | `movenet_lightning_int8.onnx` |
| 文件路径 | `backend/models/movenet_lightning_int8.onnx` |
| 文件大小 | 2.80 MB |
| ONNX 版本 | Opset 13 |
| 节点数量 | 595 |
| 模型类型 | MoveNet-Lightning 姿态估计 |
| 量化状态 | INT8 量化 |

## 模型用途

**核心功能**：幼儿俯卧趴睡检测

通过检测幼儿的骨骼关键点，判断是否存在俯卧趴睡的危险姿态。俯卧趴睡是幼儿园窒息事故的主要原因之一。

检测逻辑：
1. 检测 17 个人体关键点（鼻子、眼睛、肩膀、肘部等）
2. 计算肩-髋连线与水平面的夹角
3. 夹角低于阈值且鼻子低于肩膀 → 判定为俯卧
4. 连续多帧检测到俯卧 → 触发 CRITICAL 级告警

## 输入输出格式

### 输入

| 输入名称 | 形状 | 数据类型 | 说明 |
|----------|------|----------|------|
| `serving_default_input:0` | (1, 192, 192, 3) | uint8 | RGB 图像，保持原始像素值 |

预处理流程：
1. RGB 图像缩放至 192×192
2. INT8 量化模型，直接传入 uint8 像素值（无需归一化）

### 输出

| 输出名称 | 形状 | 说明 |
|----------|------|------|
| `StatefulPartitionedCall:0` | (1, 1, 17, 3) | 关键点坐标和置信度 |

输出格式：每个关键点为 `[y, x, score]`，坐标归一化到 [0, 1]

### 17 个关键点索引

| 索引 | 关键点名称 |
|------|-----------|
| 0 | nose（鼻子） |
| 1 | left_eye（左眼） |
| 2 | right_eye（右眼） |
| 3 | left_ear（左耳） |
| 4 | right_ear（右耳） |
| 5 | left_shoulder（左肩） |
| 6 | right_shoulder（右肩） |
| 7 | left_elbow（左肘） |
| 8 | right_elbow（右肘） |
| 9 | left_wrist（左腕） |
| 10 | right_wrist（右腕） |
| 11 | left_hip（左髋） |
| 12 | right_hip（右髋） |
| 13 | left_knee（左膝） |
| 14 | right_knee（右膝） |
| 15 | left_ankle（左踝） |
| 16 | right_ankle（右踝） |

## 在项目中的应用

### 配置引用

```json
// config/default.json
{
  "pose": {
    "input_size": 192,
    "min_keypoint_score": 0.3,
    "prone_angle_threshold": 30.0,
    "prone_frame_threshold": 30,
    "inference_interval": 3
  }
}
```

### 代码路径

- 推理类：`loongguard/pose/movenet.py` → `MoveNetLightning`
- 环境变量覆盖：`LG_POSE_*` 系列

### 使用方式

```python
from config import PoseConfig
from loongguard.pose.movenet import MoveNetLightning

config = PoseConfig()
pose = MoveNetLightning(config)
pose.load_model()

# 检测俯卧
alerts = pose.detect_prone(rgb_frame)
for alert in alerts:
    print(f"告警: {alert.description}")
```

## 俯卧判定算法

```python
# 1. 提取肩膀和髋部关键点
shoulder_mid = (left_shoulder + right_shoulder) / 2
hip_mid = (left_hip + right_hip) / 2

# 2. 计算肩-髋连线与垂直方向的夹角
angle = atan(dx / dy)  # dx = 水平距离, dy = 垂直距离

# 3. 俯卧判定条件
is_prone = (angle < prone_angle_threshold) and (nose.y > shoulder_mid.y)
```

## 性能指标

| 指标 | x86 CPU | LoongArch 2K3000 (预估) |
|------|---------|------------------------|
| 推理延迟 | ~20-30ms | ~40-60ms |
| 内存占用 | ~50MB | ~50MB |

## 注意事项

1. **INT8 量化**：输入为 uint8 像素值，无需归一化
2. **帧节流**：默认每 3 帧执行一次推理，避免抢占主链路算力
3. **降级策略**：模型加载失败时自动禁用，不阻塞主链路
4. **关键点置信度**：低于 `min_keypoint_score`（默认 0.3）的关键点被忽略
5. **持续检测**：需连续多帧检测到俯卧才触发告警，避免瞬时误判

## 模型来源

Google MoveNet-Lightning 预训练模型，经过 INT8 量化优化，适用于端侧部署。

## 相关文档

- MoveNet 官方文档：https://www.tensorflow.org/hub/tutorials/movenet
- 关键点可视化：https://www.tensorflow.org/hub/tutorials/pose_classification
