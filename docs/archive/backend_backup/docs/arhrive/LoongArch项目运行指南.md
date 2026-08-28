# LoongArch 部署启动指南

本文档总结在 LoongArch（Loongnix\_v25）机器上启动 LoongGuard 服务的完整流程、常见问题及解决方法。

***

## 一、环境准备

### 1.1 激活 Python 虚拟环境

```bash
# 指定 Python 版本为3.12.13（需要系统已安装该版本）
python3 -m venv venv
cd ~/project/loong-guard # (请将项目放于此位置)
source venv/bin/activate
```

### 1.2 安装依赖

```bash
pip install -r requirements_loongarch.txt
```

***

## 二、首次启动配置

### 2.1 设置 SM4 加密密钥

系统启动时需要加载 SM4 密钥用于加密告警日志。密钥有三种提供方式：

**方式一：环境变量（推荐）**

```bash
# 生成随机密钥并设置
export LG_SM4_KEY=$(openssl rand -hex 16)
```

**方式二：写入** **`.env.local`** **文件**

```bash
# 生成密钥文件
echo "LG_SM4_KEY=$(openssl rand -hex 16)" > .env.local

# 加载环境变量（注意：必须用 set -a 使其导出到子进程）
set -a; source .env.local; set +a
```

**方式三：创建密钥文件**

```bash
mkdir -p config
openssl rand 16 > config/.sm4_key
```

> **密钥格式说明**
>
> | 方式                | 格式                 | 示例                                 |
> | ----------------- | ------------------ | ---------------------------------- |
> | 环境变量 `LG_SM4_KEY` | 32 个 hex 字符（16 字节） | `a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6` |
> | 密钥文件 `.sm4_key`   | 16 字节二进制           | 任意 16 字节随机数据                       |

### 2.2 配置视频源

QEMU 或无摄像头环境下，需指定 mock 视频文件：

```bash
export LG_CAMERA_DEVICE=tests/mock_classroom.mp4
```

### 2.3 检查模型文件

确保 `models/` 目录下存在正确的模型文件：

```bash
ls -la models/
```

需要的文件：

- `yolo26_nano_int8.onnx` — YOLO26-Nano 目标检测模型
- `movenet_lightning_int8.onnx` — MoveNet 姿态估计模型

如果模型文件名与 `config/default.json` 中配置不一致，需要修改配置或重命名文件：

```bash
# 查看当前配置的模型路径
grep model_path config/default.json

# 方式一：修改配置文件
sed -i 's/yolo26_nano_int8_quantized.onnx/yolo26_nano_int8.onnx/' config/default.json

# 方式二：重命名模型文件
mv models/yolo26_nano_int8.onnx models/yolo26_nano_int8_quantized.onnx

# 方式三：环境变量覆盖
export LG_DETECTION_MODEL_PATH=models/yolo26_nano_int8.onnx
```

***

## 三、启动服务

### 3.1 一键启动（推荐）

```bash
export LG_SM4_KEY=eed9d822b75912f72fc3e587a77ed503 && \
export LG_CAMERA_DEVICE=tests/mock_classroom.mp4 && \
python -m src.pipeline
```

### 3.2 使用配置文件启动

```bash
python -m src.pipeline config/default.json
```

### 3.3 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_detection.py -v

# 运行单个测试用例
python -m pytest tests/test_detection.py::TestYOLO26::test_inference_output_shape -v
```

***

## 四、常见问题与解决方法

### 4.1 SM4 密钥缺失

**错误信息：**

```
FileNotFoundError: SM4 key not found: no LG_SM4_KEY env var, and key file missing: /home/cc/project/loong-guard/config/.sm4_key. The key must be injected by the administrator.
```

**原因：** 未设置 `LG_SM4_KEY` 环境变量，且密钥文件不存在。

**解决：**

```bash
export LG_SM4_KEY=$(openssl rand -hex 16)
```

***

### 4.2 环境变量未生效

**错误信息：** 设置了 `.env` 文件但环境变量未被 Python 进程读取。

**原因：** `source .env` 不会自动导出变量到子进程。`KEY=VALUE` 语法在 bash 中只设置局部变量。

**错误做法：**

```bash
source .env        # 不会导出到子进程
python -m src.pipeline
```

**正确做法：**

```bash
# 方式一：直接 export
export LG_SM4_KEY=eed9d822b75912f72fc3e587a77ed503
python -m src.pipeline

# 方式二：使用 set -a 自动导出
set -a; source .env; set +a
python -m src.pipeline
```

***

### 4.3 摄像头设备不存在

**错误信息：**

```
RuntimeError: Cannot open camera: 0. On Linux, use /dev/videoN or provide a video file path.
```

**原因：** QEMU 或无摄像头的 Linux 环境中 `/dev/video0` 不存在，且代码未自动降级到 mock 视频。

**解决：** 通过环境变量指定 mock 视频文件：

```bash
export LG_CAMERA_DEVICE=tests/mock_classroom.mp4
```

***

### 4.4 环境变量命名错误

**现象：** 设置了环境变量但不生效。

**原因：** 环境变量命名不符合 `LG_<SECTION>_<KEY>` 规则。

**命名规则：**

| 环境变量                          | 配置路径                       | 说明         |
| ----------------------------- | -------------------------- | ---------- |
| `LG_SM4_KEY`                  | SM4 密钥                     | 特殊变量，非配置覆盖 |
| `LG_CAMERA_DEVICE`            | `camera.device`            | 摄像头设备路径    |
| `LG_DETECTION_MODEL_PATH`     | `detection.model_path`     | 检测模型路径     |
| `LG_DETECTION_CONF_THRESHOLD` | `detection.conf_threshold` | 检测置信度阈值    |
| `LG_API_PORT`                 | `api.port`                 | API 服务端口   |
| `LG_POSE_MODEL_PATH`          | `pose.model_path`          | 姿态估计模型路径   |

**错误示例：**

```bash
# 错误：不符合命名规则
export LOONGGUARD_CAMERA_DEVICE=tests/mock_classroom.mp4

# 正确：
export LG_CAMERA_DEVICE=tests/mock_classroom.mp4
```

***

### 4.5 模型文件缺失或名称不匹配

**错误信息：**

```
FileNotFoundError: Model file not found: models/yolo26_nano_int8_quantized.onnx
```

**原因：** `config/default.json` 中配置的模型文件名与实际文件名不一致。

**解决：**

```bash
# 检查实际模型文件
ls -la models/

# 修改配置文件中的模型路径
sed -i 's/yolo26_nano_int8_quantized.onnx/yolo26_nano_int8.onnx/' config/default.json
```

***

### 4.6 服务启动后无法远程访问

**现象：** 服务正常启动，本地可访问，但 Windows 主机无法访问 `http://<ip>:8080/`。

**排查步骤：**

**步骤 1：检查服务监听地址**

```bash
ss -tlnp | grep 8080
# 或
netstat -tlnp | grep 8080
```

应显示 `0.0.0.0:8080`（监听所有接口）。如果是 `127.0.0.1:8080`，说明只监听本地。

**步骤 2：检查 Linux 防火墙**

```bash
# firewalld
sudo firewall-cmd --add-port=8080/tcp --permanent
sudo firewall-cmd --reload

# 或 iptables
sudo iptables -L -n | grep 8080
```

**步骤 3：QEMU 端口转发**

如果使用 QEMU 虚拟机，需要配置端口转发。启动 QEMU 时添加：

```bash
-netdev user,id=net0,hostfwd=tcp::8080-:8080
```

**步骤 4：本地验证**

在 LoongArch 机器上先确认服务正常：

```bash
curl http://localhost:8080/api/v1/status
```

***

### 4.7 测试被跳过（SKIPPED）

**现象：** `pytest` 输出中有 8 个测试显示 `SKIPPED`。

**原因：** `test_camera.py` 中部分测试需要实际摄像头硬件，在无摄像头环境下自动跳过。

**影响：** 无影响，这是设计预期行为。使用 mock 视频文件的测试会正常通过。

***

## 五、完整启动示例

```bash
# 1. 进入项目目录
cd ~/project/loong-guard

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 设置所有必要环境变量
export LG_SM4_KEY=eed9d822b75912f72fc3e587a77ed503
export LG_CAMERA_DEVICE=tests/mock_classroom.mp4

# 4. 启动服务
python -m src.pipeline

# 5. 验证服务（另开终端）
curl http://localhost:8080/api/v1/status

# 6. 同一局域网的另一台设备
浏览器打开 http://<龙芯机器局域网IP>:8080
# 或使用本机浏览器打开浏览
http://localhost:8080
```

启动成功日志示例：

```
2026-06-26 14:04:13 [INFO] __main__: Starting LoongGuard pipeline...
2026-06-26 14:04:13 [INFO] src.db.database: 数据库初始化完成: /home/cc/project/loong-guard/data/loongguard.db
2026-06-26 14:04:13 [INFO] src.crypto.sm4_logger: SM4 key loaded (16 bytes, mode=cbc)
2026-06-26 14:04:13 [INFO] src.camera.v4l2_capture: Camera opened via OpenCV: tests/mock_classroom.mp4
2026-06-26 14:04:13 [INFO] __main__: Pipeline started
2026-06-26 14:04:13 [INFO] __main__: Dashboard: http://localhost:8080
2026-06-26 14:04:13 [INFO] __main__: Entering main inference loop
```

