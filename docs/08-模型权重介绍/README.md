# LoongGuard 模型权重文件说明

## 概述

LoongGuard 项目使用了多个 ONNX 模型权重文件，用于实现危险物品检测、姿态估计、人脸检测等功能。本文档对所有模型进行汇总说明。

## 模型文件列表

### 核心模型（在项目中被引用）

| 模型文件 | 功能 | 大小 | 量化状态 | 状态 |
|----------|------|------|----------|------|
| `best.onnx` | 危险物品检测（剪刀/美工刀） | 9.80 MB | FP32 | ✅ 主模型 |
| `movenet_lightning_int8.onnx` | 姿态估计（俯卧检测） | 2.80 MB | INT8 | ✅ 启用 |
| `buffalo_l/det_10g.onnx` | 人脸检测（RetinaFace） | 16.14 MB | FP32 | ⚠️ 默认关闭 |

### 可选模型（可用于替代）

| 模型文件 | 功能 | 大小 | 量化状态 | 状态 |
|----------|------|------|----------|------|
| `yolo26_nano_int8.onnx` | 危险物品检测（INT8 量化版） | 9.49 MB | INT8 | ⚡ 可选 |
| `det_500m.onnx` | 人脸检测（轻量版） | 2.41 MB | FP32 | 📦 预留 |
| `w600k_mbf.onnx` | 人脸识别（MobileFaceNet） | 12.99 MB | FP32 | 📦 预留 |

### 预留模型（Buffalo L 套件）

| 模型文件 | 功能 | 大小 | 量化状态 | 状态 |
|----------|------|------|----------|------|
| `buffalo_l/w600k_r50.onnx` | 人脸识别（ResNet-50） | 166.31 MB | FP32 | 📦 预留 |
| `buffalo_l/1k3d68.onnx` | 3D 人脸关键点检测 | 136.95 MB | FP32 | 📦 预留 |
| `buffalo_l/2d106det.onnx` | 2D 人脸关键点检测 | 4.80 MB | FP32 | 📦 预留 |
| `buffalo_l/genderage.onnx` | 性别年龄识别 | 1.26 MB | FP32 | 📦 预留 |

## 模型用途详解

### 1. 危险物品检测

**主模型**：`best.onnx`（YOLO26-Nano）

- 检测剪刀和美工刀两种危险物品
- 输入：640×640 RGB 图像
- 输出：检测框坐标、置信度、类别
- 配置：`config.detection`

**替代模型**：`yolo26_nano_int8.onnx`

- INT8 量化版本，推理更快但精度略低
- 启用方式：设置 `LG_DETECTION_QUANTIZED=true`

### 2. 姿态估计（俯卧检测）

**模型**：`movenet_lightning_int8.onnx`（MoveNet-Lightning）

- 检测 17 个人体关键点
- 判断幼儿是否处于俯卧趴睡姿态
- 输入：192×192 RGB 图像（uint8）
- 输出：关键点坐标和置信度
- 配置：`config.pose`

### 3. 人脸检测（睡姿监测）

**模型**：`buffalo_l/det_10g.onnx`（RetinaFace）

- 检测画面中的人脸
- 用于睡姿监测：有运动但无人脸 → 异常睡姿
- 输入：640×640 RGB 图像
- 输出：人脸边界框
- 配置：`config.face`

**启用方式**：
```bash
# .env 文件
LG_FACE_ENABLED=true
LG_FACE_BACKEND=onnx
LG_FACE_CONF_THRESHOLD=0.5
```

## 配置引用

### 配置文件路径

- 主配置：`backend/config/default.json`
- 环境变量：`backend/.env`（覆盖默认值）

### 环境变量覆盖

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `LG_DETECTION_MODEL_PATH` | 检测模型路径 | `models/best.onnx` |
| `LG_DETECTION_QUANTIZED` | 是否使用量化模型 | `false` |
| `LG_DETECTION_CONF_THRESHOLD` | 检测置信度阈值 | `0.80` |
| `LG_POSE_MODEL_PATH` | 姿态模型路径 | `models/movenet_lightning_int8.onnx` |
| `LG_FACE_ENABLED` | 是否启用人脸检测 | `false` |
| `LG_FACE_BACKEND` | 人脸检测后端 | `onnx` |
| `LG_FACE_CONF_THRESHOLD` | 人脸置信度阈值 | `0.5` |

## 性能对比

### x86 CPU 性能

| 模型 | 推理延迟 | 吞吐量 | 内存占用 |
|------|---------|--------|----------|
| best.onnx (FP32) | ~30-50ms | 20-30 FPS | ~100MB |
| yolo26_nano_int8.onnx (INT8) | ~20-35ms | 30-40 FPS | ~80MB |
| movenet_lightning_int8.onnx | ~20-30ms | 30-40 FPS | ~50MB |
| det_10g.onnx | ~30-50ms | 20-30 FPS | ~150MB |

### LoongArch 2K3000 预估性能

| 模型 | 推理延迟 | 吞吐量 |
|------|---------|--------|
| best.onnx (FP32) | ~60-100ms | 10-15 FPS |
| yolo26_nano_int8.onnx (INT8) | ~40-70ms | 15-25 FPS |
| movenet_lightning_int8.onnx | ~40-60ms | 15-25 FPS |
| det_10g.onnx | ~60-100ms | 10-15 FPS |

## 模型文件存放路径

```
backend/models/
├── best.onnx                    # 危险物品检测主模型
├── yolo26_nano_int8.onnx        # INT8 量化版检测模型
├── movenet_lightning_int8.onnx  # 姿态估计模型
├── det_500m.onnx                # 早期人脸检测模型
├── w600k_mbf.onnx               # 早期人脸识别模型
└── buffalo_l/                   # InsightFace Buffalo L 套件
    ├── det_10g.onnx             # 人脸检测
    ├── w600k_r50.onnx           # 人脸识别
    ├── 1k3d68.onnx              # 3D 关键点
    ├── 2d106det.onnx            # 2D 关键点
    └── genderage.onnx           # 性别年龄
```

## 相关文档

- [模型权重文件更换指南](../02-部署与容器/模型权重文件更换指南.md)
- [配置说明](../01-环境配置/配置说明.md)
- [部署文档](../02-部署与容器/部署文档.md)

## 注意事项

1. **模型文件大小**：总大小约 350MB，部署时需确保存储空间充足
2. **ONNX 版本兼容**：确保 ONNX Runtime 版本支持模型使用的 opset
3. **量化模型**：INT8 量化模型推理更快但精度略低
4. **默认配置**：人脸检测默认关闭，需手动启用
5. **降级策略**：模型加载失败时自动回退到桩实现，不阻塞主链路
