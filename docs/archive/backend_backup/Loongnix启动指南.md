# LoongArch 平台项目启动流程指南

> **适用对象**：需要在龙芯 LoongArch 平台上从零开始部署 LoongGuard 项目的开发者  
> **目标硬件**：龙芯 2K3000 开发板 (LG200 GPGPU) / 3A5000 主机  
> **文档版本**：v1.0 | 更新日期：2026-07-03

---

## 📋 目录

- [一、硬件环境准备](#一硬件环境准备)
- [二、系统环境配置](#二系统环境配置)
- [三、项目依赖安装](#三项目依赖安装)
- [四、配置文件设置](#四配置文件设置)
- [五、模型文件准备](#五模型文件准备)
- [六、启动与验证](#六启动与验证)
- [七、常见问题排查](#七常见问题排查)
- [八、性能优化建议](#八性能优化建议)

---

## 一、硬件环境准备

### 1.1 目标硬件规格

| 硬件组件 | 最低规格 | 推荐规格 | 说明 |
|---------|---------|---------|------|
| **处理器** | 龙芯 3A5000 | 龙芯 2K3000 | 8x LA364E 核心 |
| **GPU** | - | LG200 GPGPU | 第二代自研通用 GPU，图形性能提升 2 倍 |
| **内存** | 8GB DDR4 | 16GB DDR4 | 模型推理需要足够内存 |
| **存储** | 32GB SSD | 64GB SSD | 存储模型、日志、截图 |
| **摄像头** | USB 2.0 | USB 3.0 | 支持 V4L2 协议 |
| **网络** | 千兆以太网 | 千兆以太网 | 局域网 API 访问 |

### 1.2 外设连接检查

```bash
# 1. 检查摄像头设备
ls -l /dev/video*
# 预期输出：/dev/video0 (或 video1、video2...)

# 2. 检查 USB 设备
lsusb
# 确认摄像头已被识别

# 3. 测试摄像头采集（可选）
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### 1.3 GPIO 引脚连接（可选）

如需使用声光告警功能，需连接以下 GPIO 引脚：

| 功能 | GPIO 引脚 | 说明 |
|-----|----------|------|
| 蜂鸣器 | GPIO 18 | 高电平触发 |
| LED 指示灯 | GPIO 24 | 高电平触发 |

---

## 二、系统环境配置

### 2.1 操作系统要求

- **推荐系统**：Loongnix_v25 (龙芯官方发行版)
- **内核版本**：Linux 5.10+ (支持 V4L2、OpenCL)
- **架构**：LoongArch (mips64el 兼容模式)

### 2.2 Python 环境安装

```bash
# 1. 检查系统 Python 版本
python3 --version
# 预期输出：Python 3.12.x (或 3.11.x)

# 2. 安装 Python 虚拟环境工具（如未安装）
sudo apt-get update
sudo apt-get install python3-venv python3-pip

# 3. 创建项目目录
mkdir -p ~/loongguard && cd ~/loongguard
```

### 2.3 系统依赖安装

```bash
# 安装编译依赖（OpenCV、ONNX Runtime 需要）
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran

# 安装 OpenCL 运行时（LG200 GPU 加速）
sudo apt-get install -y \
    ocl-icd-opencl-dev \
    clinfo

# 验证 OpenCL 设备
clinfo | grep "Device Name"
# 预期输出：Loongson LG200 (或类似)
```

---

## 三、项目依赖安装

### 3.1 获取项目代码

```bash
# 方式一：从 Git 仓库克隆（推荐）
git clone <repository-url> ~/loongguard
cd ~/loongguard

# 方式二：从 Windows 主机同步（开发环境）
# 在 Windows 主机上执行：
python scripts/remote_exec.py sync
```

### 3.2 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 验证环境
which python
# 预期输出：/home/<user>/loongguard/.venv/bin/python
```

### 3.3 安装依赖包

```bash
# 升级 pip 到最新版本
pip install --upgrade pip setuptools wheel

# 安装最小运行时依赖（推荐，体积小）
pip install -r requirements_loongarch.txt

# 或安装完整开发依赖（包含训练、测试工具）
pip install -r requirements.txt
```

### 3.4 验证依赖安装

```bash
# 验证核心依赖
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import gmssl; print('GMSSL: OK')"
python -c "import aiohttp; print(f'aiohttp: {aiohttp.__version__}')"

# 验证 OpenCL 支持（可选）
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 预期输出包含：['OpenCLExecutionProvider', 'CPUExecutionProvider']
```

---

## 四、配置文件设置

### 4.1 创建环境变量文件

```bash
# 从模板创建配置文件
cp .env.example .env

# 编辑配置文件
nano .env
```

### 4.2 必填配置项

```bash
# .env 文件内容示例

# ── 运行环境 ──────────────────────────────────────────────
LG_ENV=production

# ── 摄像头配置 ────────────────────────────────────────────
LG_CAMERA_DEVICE=/dev/video0
LG_CAMERA_WIDTH=1280
LG_CAMERA_HEIGHT=720
LG_CAMERA_FPS=30

# ── 检测参数 ──────────────────────────────────────────────
LG_DETECTION_CONF_THRESHOLD=0.45
LG_DETECTION_QUANTIZED=true

# ── GPIO 告警引脚（如未连接可注释） ──────────────────────
LG_GPIO_BUZZER_PIN=18
LG_GPIO_LED_PIN=24

# ── SM4 国密加密（必须配置） ─────────────────────────────
# 生成密钥命令：openssl rand -hex 16
LG_SM4_KEY=<your-32-char-hex-key>
LG_SM4_MODE=cbc

# ── API 服务配置 ─────────────────────────────────────────
LG_API_HOST=0.0.0.0
LG_API_PORT=8080

# ── 日志级别 ──────────────────────────────────────────────
LG_LOG_LEVEL=INFO
```

### 4.3 生成 SM4 密钥

```bash
# 生成 32 位 hex 密钥（16 字节）
openssl rand -hex 16

# 将输出复制到 .env 文件的 LG_SM4_KEY 变量
# 示例输出：a1b2c3d4e5f6789012345678abcdefab
```

### 4.4 验证配置文件

```bash
# 检查配置是否正确加载
python -c "
from config.settings import Settings
config = Settings.from_env()
print(f'摄像头设备: {config.camera.device}')
print(f'检测阈值: {config.detection.conf_threshold}')
print(f'API 端口: {config.api.port}')
print(f'SM4 密钥已配置: {bool(config.sm4.key)}')
"
```

---

## 五、模型文件准备

### 5.1 检查模型文件

```bash
# 检查模型目录
ls -lh models/

# 预期输出：
# yolo26_nano_int8.onnx        (~5MB)
# movenet_lightning_int8.onnx  (~10MB)
```

### 5.2 模型文件说明

| 模型文件 | 用途 | 输入尺寸 | 量化 | 说明 |
|---------|------|---------|------|------|
| `yolo26_nano_int8.onnx` | 危险物品检测 | 640x640 | INT8 | NMS-Free 架构，CPU 推理优化 |
| `movenet_lightning_int8.onnx` | 幼儿姿态估计 | 192x192 | INT8 | 俯卧睡姿检测 |

### 5.3 模型文件缺失处理

```bash
# 如果模型文件缺失，可从项目仓库下载
# 或使用占位模型（仅用于测试）
python scripts/create_dummy_models.py

# 注意：占位模型无实际推理能力，仅用于环境验证
```

---

## 六、启动与验证

### 6.1 快速启动

```bash
# 确保在项目根目录
cd ~/loongguard

# 激活虚拟环境
source .venv/bin/activate

# 启动主程序
python -m src.pipeline

# 或使用 CLI 入口（如已安装）
loongguard
```

### 6.2 启动成功标志

```
[2026-07-03 10:30:45] INFO     | pipeline:__init__:42 - LoongGuard 初始化完成
[2026-07-03 10:30:45] INFO     | camera:open:87 - 摄像头已打开: /dev/video0 @ 1280x720
[2026-07-03 10:30:45] INFO     | detection:load:56 - YOLO26-Nano 模型加载完成 (INT8)
[2026-07-03 10:30:45] INFO     | pose:load:48 - MoveNet-Lightning 模型加载完成 (INT8)
[2026-07-03 10:30:45] INFO     | api:start:123 - API 服务已启动: http://0.0.0.0:8080
[2026-07-03 10:30:45] INFO     | pipeline:run:158 - 主循环启动，帧率目标: 30 FPS
```

### 6.3 功能验证

#### 6.3.1 API 健康检查

```bash
# 检查 API 服务状态
curl http://localhost:8080/api/health

# 预期返回：
# {"status": "ok", "timestamp": "2026-07-03T10:30:45Z"}
```

#### 6.3.2 实时视频流

```bash
# 在局域网内其他设备上访问
# 浏览器打开：http://<设备IP>:8080/

# 或使用 MJPEG 流地址
# http://<设备IP>:8080/api/video/stream
```

#### 6.3.3 Prometheus 指标

```bash
# 访问监控指标
curl http://localhost:8080/metrics

# 关键指标：
# - loongguard_fps_current: 当前帧率
# - loongguard_detection_latency_seconds: 检测延迟
# - loongguard_alert_total: 告警总数
```

### 6.4 停止服务

```bash
# 按 Ctrl+C 停止主程序

# 或使用 systemd 管理（推荐生产环境）
sudo systemctl stop loongguard
```

---

## 七、常见问题排查

### 7.1 摄像头无法打开

**症状**：`CameraError: 无法打开设备 /dev/video0`

**排查步骤**：

```bash
# 1. 检查设备是否存在
ls -l /dev/video0

# 2. 检查设备权限
sudo chmod 666 /dev/video0

# 3. 检查设备是否被占用
fuser /dev/video0
# 如有进程占用，kill <PID>

# 4. 测试摄像头采集
ffplay -f video4linux2 -i /dev/video0
```

### 7.2 模型加载失败

**症状**：`RuntimeError: Failed to load ONNX model`

**排查步骤**：

```bash
# 1. 检查模型文件完整性
md5sum models/yolo26_nano_int8.onnx

# 2. 检查 ONNX Runtime 后端
python -c "
import onnxruntime as ort
print('可用后端:', ort.get_available_providers())
"

# 3. 尝试 CPU 后端
export ONNXRUNTIME_EXECUTION_PROVIDER=CPU
python -m src.pipeline
```

### 7.3 SM4 加密失败

**症状**：`ValueError: SM4 密钥格式错误`

**解决方案**：

```bash
# 检查密钥格式（必须为 32 位 hex 字符串）
echo $LG_SM4_KEY | wc -c
# 预期输出：33 (包含换行符)

# 重新生成密钥
openssl rand -hex 16
```

### 7.4 API 无法访问

**症状**：浏览器无法打开 `http://<IP>:8080`

**排查步骤**：

```bash
# 1. 检查服务是否启动
netstat -tunlp | grep 8080

# 2. 检查防火墙
sudo ufw status
sudo ufw allow 8080/tcp

# 3. 检查监听地址
# 确保 .env 中 LG_API_HOST=0.0.0.0
# 而非 127.0.0.1
```

### 7.5 性能不达标

**症状**：帧率低于 15 FPS，延迟超过 100ms

**优化建议**：

```bash
# 1. 降低采集分辨率
# 编辑 .env：
LG_CAMERA_WIDTH=640
LG_CAMERA_HEIGHT=480

# 2. 降低检测阈值
LG_DETECTION_CONF_THRESHOLD=0.30

# 3. 启用 INT8 量化
LG_DETECTION_QUANTIZED=true

# 4. 检查 CPU 占用
top -p $(pgrep -f src.pipeline)

# 5. 检查内存占用
free -h
```

---

## 八、性能优化建议

### 8.1 硬件加速配置

```bash
# 启用 OpenCL GPU 加速（LG200）
export ONNXRUNTIME_EXECUTION_PROVIDER=OpenCL

# 验证 GPU 使用
clinfo | grep "Device Name"
```

### 8.2 系统级优化

```bash
# 1. 设置 CPU 性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 2. 关闭不必要的服务
sudo systemctl disable bluetooth
sudo systemctl disable cups

# 3. 增加文件描述符限制
ulimit -n 65535
```

### 8.3 应用级优化

```bash
# 1. 调整检测参数
# 编辑 config/default.json：
{
  "detection": {
    "conf_threshold": 0.35,
    "input_size": 416
  },
  "roi": {
    "stage1_input_size": 256,
    "stage2_input_size": 512
  }
}

# 2. 启用日志级别为 WARNING（减少 I/O）
LG_LOG_LEVEL=WARNING

# 3. 禁用调试窗口
unset LG_DEBUG_WINDOW
```

### 8.4 性能基准参考

| 硬件平台 | 分辨率 | 帧率 | 延迟 | 说明 |
|---------|-------|------|------|------|
| 2K3000 (LG200) | 1280x720 | 25-30 FPS | 35-45ms | 推荐配置 |
| 2K3000 (LG200) | 640x480 | 30+ FPS | 25-35ms | 性能优先 |
| 3A5000 (CPU) | 1280x720 | 15-20 FPS | 60-80ms | 仅 CPU 推理 |
| 3A5000 (CPU) | 640x480 | 20-25 FPS | 45-60ms | 降低分辨率 |

---

## 九、生产环境部署

### 9.1 创建 systemd 服务

```bash
# 创建服务文件
sudo nano /etc/systemd/system/loongguard.service
```

```ini
[Unit]
Description=LoongGuard 幼儿园安全监护系统
After=network.target

[Service]
Type=simple
User=loongson
WorkingDirectory=/home/loongson/loongguard
Environment="PATH=/home/loongson/loongguard/.venv/bin"
ExecStart=/home/loongson/loongguard/.venv/bin/python -m src.pipeline
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable loongguard
sudo systemctl start loongguard

# 查看服务状态
sudo systemctl status loongguard

# 查看日志
sudo journalctl -u loongguard -f
```

### 9.2 日志轮转配置

```bash
# 创建日志轮转配置
sudo nano /etc/logrotate.d/loongguard
```

```
/home/loongson/loongguard/logs/**/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 loongson loongson
}
```

---

## 十、安全注意事项

### 10.1 密钥管理

- **[严重]** 禁止在代码中硬编码 SM4 密钥
- **[严重]** `.env` 文件权限应设为 `600`：`chmod 600 .env`
- **[严重]** 定期轮换 SM4 密钥（建议每 90 天）

### 10.2 网络安全

- **[高危]** API 端点无身份验证，**仅限内网部署**
- **[高危]** 禁止将设备暴露到公网
- **[中危]** 使用防火墙限制访问来源 IP

```bash
# 限制仅特定 IP 访问
sudo ufw allow from 192.168.1.0/24 to any port 8080
```

### 10.3 数据安全

- **[中危]** 告警日志使用 SM4 加密存储
- **[中危]** 视频帧数据仅在内存中处理，断电即失
- **[低危]** 定期清理过期日志和截图

---

## 十一、技术支持

### 11.1 日志收集

```bash
# 收集系统信息
uname -a > debug_info.txt
cat /etc/os-release >> debug_info.txt

# 收集应用日志
tar -czf logs_$(date +%Y%m%d).tar.gz logs/

# 收集配置信息（脱敏）
env | grep LG_ | sed 's/=.*/=***/ >> debug_info.txt
```

### 11.2 问题反馈

提交 Issue 时请包含以下信息：

- [ ] 硬件型号（2K3000 / 3A5000）
- [ ] 操作系统版本（`cat /etc/os-release`）
- [ ] Python 版本（`python3 --version`）
- [ ] 依赖版本（`pip freeze`）
- [ ] 错误日志（`logs/` 目录）
- [ ] 复现步骤

---

## 附录A：环境变量完整清单

| 变量名 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `LG_ENV` | 运行环境 | `development` | 否 |
| `LG_CAMERA_DEVICE` | 摄像头设备路径 | `/dev/video0` | 否 |
| `LG_CAMERA_WIDTH` | 采集宽度 | `1280` | 否 |
| `LG_CAMERA_HEIGHT` | 采集高度 | `720` | 否 |
| `LG_CAMERA_FPS` | 目标帧率 | `30` | 否 |
| `LG_DETECTION_CONF_THRESHOLD` | 检测置信度阈值 | `0.45` | 否 |
| `LG_DETECTION_QUANTIZED` | 是否使用量化模型 | `true` | 否 |
| `LG_GPIO_BUZZER_PIN` | 蜂鸣器 GPIO 引脚 | `18` | 否 |
| `LG_GPIO_LED_PIN` | LED GPIO 引脚 | `24` | 否 |
| `LG_SM4_KEY` | SM4 加密密钥（32 位 hex） | - | **是** |
| `LG_SM4_MODE` | SM4 加密模式 | `cbc` | 否 |
| `LG_API_HOST` | API 监听地址 | `0.0.0.0` | 否 |
| `LG_API_PORT` | API 端口 | `8080` | 否 |
| `LG_LOG_LEVEL` | 日志级别 | `INFO` | 否 |

---

## 附录B：快速命令参考

```bash
# ── 环境管理 ──────────────────────────────────────────────
python3 -m venv .venv                    # 创建虚拟环境
source .venv/bin/activate                # 激活环境
deactivate                               # 退出环境

# ── 依赖管理 ──────────────────────────────────────────────
pip install -r requirements_loongarch.txt  # 安装最小依赖
pip install -r requirements.txt            # 安装完整依赖
pip freeze > requirements.lock             # 锁定依赖版本

# ── 配置管理 ──────────────────────────────────────────────
cp .env.example .env                      # 创建配置文件
openssl rand -hex 16                      # 生成 SM4 密钥

# ── 启动服务 ──────────────────────────────────────────────
python -m src.pipeline                    # 直接启动
loongguard                                # CLI 入口启动
sudo systemctl start loongguard           # systemd 启动

# ── 测试验证 ──────────────────────────────────────────────
pytest                                    # 运行所有测试
pytest tests/unit/                        # 运行单元测试
curl http://localhost:8080/api/health     # API 健康检查

# ── 日志查看 ──────────────────────────────────────────────
tail -f logs/app.log                      # 实时查看日志
sudo journalctl -u loongguard -f          # systemd 日志

# ── 性能监控 ──────────────────────────────────────────────
top -p $(pgrep -f src.pipeline)           # CPU/内存监控
curl http://localhost:8080/metrics        # Prometheus 指标
```

---

**文档结束** | 如有问题请参考 [CLAUDE.md](./CLAUDE.md) 或提交 Issue
