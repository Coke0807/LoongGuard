# LoongGuard

## 1. 项目概述
- **项目定位与核心功能**: 基于 LoongArch 架构的幼儿园端侧 AI 安全监护系统。产品定位：不替代老师，而是通过边缘端侧计算，成为老师的 **"超级辅助眼"** 与 **"全能班务管家"**。核心功能包括：实时视频监控、危险物品检测（磁力珠、纽扣电池、剪刀等）、幼儿俯卧睡姿检测、声光告警、SM4 国密加密存储、Web 管理面板。
- **业务背景**: 近5年全国发生数千起学龄前儿童因微小异物误吞、手持受伤的案例。92.9% 的误吞伤害受害者集中在 0-6 岁幼儿园学龄前儿童。39.3% 因微小危险品导致中重度伤害，其中 25% 需住院手术治疗。真实案例包括磁力珠误吞、纽扣电池灼伤、细小尖锐物扎伤等。
- **目标用户与使用场景**: 幼儿园管理人员、安防运维人员。应用场景：幼儿园教室实时监护、幼儿俯卧/跌倒检测、危险物品识别、安防记录留存。部署环境：龙芯 LoongArch 单板计算机（LG200）(2k3000开发板)，教室局域网。
- **技术栈与依赖环境**: 
  - 核心语言/版本: Python 3.11+
  - 运行时环境: 龙芯 LG200 (LoongArch) / Windows x86 (开发环境，全 Mock 降级支持)
  - 核心框架: aiohttp 3.9+ (异步 Web 服务器)、ONNX Runtime 1.16+ (推理引擎)、OpenCV 4.8+ (计算机视觉)
  - 关键依赖: ultralytics 8.4+ (YOLO 训练)、PyTorch 2.0+ (训练)、gmssl 3.2+ (国密 SM4)、prometheus-client 0.19+ (监控)、paramiko 3.3+ (SSH 远程同步)
- **快速开始指南**: 
  ```bash
  # 1. 创建虚拟环境
  python -m venv .venv
  .venv\Scripts\activate  # Windows
  source .venv/bin/activate  # Linux
  
  # 2. 安装依赖
  pip install -r requirements.txt
  
  # 3. 配置环境变量
  cp .env.example .env
  # 编辑 .env 填写必要配置（SM4 密钥、SSH 配置等）
  
  # 4. 运行主程序
  python -m src.pipeline
  
  # 或使用 CLI 入口
  loongguard
  ```

## 2. 目录结构
- **关键目录说明**:
  ```
  LoongGuard/
  ├── src/                    # 源代码主目录
  │   ├── pipeline.py         # 核心管道编排器（主入口）
  │   ├── alarm/              # GPIO 声光告警触发
  │   ├── alerts/             # 告警去重逻辑
  │   ├── api/                # Web API 服务器（REST + WebSocket + MJPEG）
  │   │   ├── server.py       # aiohttp 服务器
  │   │   ├── metrics.py      # Prometheus 指标
  │   │   └── web/            # 前端静态文件
  │   ├── camera/             # 摄像头采集（V4L2 + OpenCV 降级）
  │   ├── crypto/             # SM4 国密加密存储
  │   ├── db/                 # SQLite 数据库持久化
  │   ├── detection/          # YOLO26-Nano 目标检测
  │   ├── motion/             # 帧差分运动检测
  │   ├── pose/               # MoveNet-Lightning 姿态估计
  │   ├── roi/                # 动态 ROI 二级检测调度
  │   └── utils/              # 工具类（Schema、日志配置、保留策略）
  ├── config/                 # 配置文件目录
  │   ├── settings.py         # dataclass 配置定义
  │   └── default.json        # 默认配置
  ├── models/                 # 模型文件目录（已提交）
  │   ├── yolo26_nano_int8.onnx    # YOLO 检测模型
  │   └── movenet_lightning_int8.onnx  # 姿态估计模型
  ├── data/                   # 数据目录
  │   ├── raw/                # 原始训练数据
  │   ├── processed/          # 处理后数据
  │   └── screenshots/        # 截图保存
  ├── logs/                   # 日志目录
  │   └── encrypted/          # SM4 加密告警日志
  ├── tests/                  # 测试文件目录
  │   ├── unit/               # 单元测试
  │   ├── integration/        # 集成测试
  │   └── fixtures/           # 测试固件
  ├── scripts/                # 工具脚本
  │   ├── convert_models.py   # 模型转换
  │   ├── remote_exec.py      # 远程执行
  │   ├── visual_test.py      # 可视化测试
  │   └── sync.py             # 代码同步
  ├── pyproject.toml          # 项目元数据（PEP 621）
  ├── requirements.txt        # 依赖清单
  └── .env.example            # 环境变量模板
  ```
- **文件组织方式**: 按功能模块划分，每个子目录对应一个独立功能域。主入口为 `src/pipeline.py`，编排所有子模块的生命周期。
- **配置文件位置**: 
  - 主配置：`config/settings.py`（dataclass 定义）
  - 默认值：`config/default.json`
  - 环境变量：`.env`（从 `.env.example` 复制）
- **资源文件位置**: 
  - 模型文件：`models/`（.onnx、.tflite、.pt）
  - 测试视频：`tests/mock_classroom.mp4`
  - 日志输出：`logs/encrypted/`

## 3. 开发规范
- **代码风格与格式要求**: 
  - Python 3.11+ 现代语法，强制类型注解（禁止 `any`）
  - Docstring 采用 Google 风格，必须说明设计动机（Why）
  - 模块级注释说明职责边界，避免冗余翻译
  - 单行长度 100 字符，使用 UTF-8 编码
- **命名规范与约定**: 
  - 类名：PascalCase（如 `AlertLog`、`YOLO26Nano`）
  - 函数/方法：snake_case（如 `detect_objects`、`trigger_alarm`）
  - 常量：UPPER_SNAKE_CASE（如 `DEFAULT_CONFIG_DIR`）
  - 私有属性：单下划线前缀（如 `_running`、`_config`）
  - 环境变量：`LG_` 前缀（如 `LG_SM4_KEY`、`LG_SSH_HOST`）
- **Git 提交规范**: 
  - 提交信息格式：`<type>(<scope>): <subject>`
  - 类型：feat、fix、docs、style、refactor、test、chore
  - 示例：`feat(detection): 添加 INT8 量化支持`
- **安全注意事项**: 
  - **[严重]** 禁止在代码中硬编码密钥、密码、凭证
  - **[严重]** 所有敏感配置必须通过环境变量注入（`.env` 不提交）
  - **[严重]** SM4 密钥必须使用 `openssl rand -hex 16` 生成，禁止使用示例密钥
  - **[高危]** API 端点无身份验证，仅限内网部署，禁止暴露到公网
  - **[中危]** 文件上传需验证文件名，防止路径遍历攻击
  - 详细安全审计见 `安全审计报告.md`

## 4. 常用命令
- **安装和启动命令**: 
  ```bash
  # 安装依赖
  pip install -r requirements.txt
  
  # 开发模式启动（带调试窗口）
  set LG_DEBUG_WINDOW=1 && python -m src.pipeline
  
  # 生产模式启动
  python -m src.pipeline
  
  # 或使用 CLI 入口
  loongguard
  ```
- **测试和检查命令**: 
  ```bash
  # 运行所有测试
  pytest
  
  # 运行单元测试
  pytest tests/unit/
  
  # 运行集成测试
  pytest tests/integration/
  
  # 带覆盖率报告
  pytest --cov=src --cov-report=html
  
  # 可视化测试（Windows 本地调试）
  python scripts/visual_test.py
  python scripts/visual_test.py --camera 0
  ```
- **构建和部署命令**: 
  ```bash
  # 远程同步到龙芯设备
  python scripts/remote_exec.py sync
  
  # 远程运行测试
  python scripts/remote_exec.py test
  
  # 远程启动服务
  python scripts/remote_exec.py run
  
  # 模型转换
  python scripts/convert_models.py
  ```
- **环境变量配置**: 
  | 变量名 | 说明 | 示例值 |
  |--------|------|--------|
  | `LG_ENV` | 运行环境 | `development` / `testing` / `production` |
  | `LG_SSH_HOST` | 龙芯设备 IP 地址 | `192.168.1.100` |
  | `LG_SSH_PORT` | SSH 端口 | `22` |
  | `LG_SSH_USER` | SSH 用户名 | `loongson` |
  | `LG_SSH_PASS` | SSH 密码 | （留空使用密钥） |
  | `LG_SSH_KEY` | SSH 私钥路径 | `~/.ssh/id_rsa` |
  | `LG_CAMERA_DEVICE` | 摄像头设备路径 | `0` 或 `/dev/video0` |
  | `LG_CAMERA_WIDTH` | 采集宽度 | `640` |
  | `LG_CAMERA_HEIGHT` | 采集高度 | `480` |
  | `LG_CAMERA_FPS` | 目标帧率 | `30` |
  | `LG_DETECTION_CONF_THRESHOLD` | 检测置信度阈值 | `0.30` |
  | `LG_DETECTION_QUANTIZED` | 是否使用量化模型 | `true` / `false` |
  | `LG_GPIO_BUZZER_PIN` | 蜂鸣器 GPIO 引脚 | `18` |
  | `LG_GPIO_LED_PIN` | LED GPIO 引脚 | `24` |
  | `LG_SM4_KEY` | SM4 加密密钥（32 位 hex） | `openssl rand -hex 16` 生成 |
  | `LG_SM4_MODE` | SM4 加密模式 | `cbc` / `ecb` |
  | `LG_LOG_LEVEL` | 日志级别 | `INFO` / `DEBUG` / `WARNING` |
  | `LG_API_HOST` | API 监听地址 | `0.0.0.0` |
  | `LG_API_PORT` | API 端口 | `8080` |

## 5. 技术决策
- **架构设计原因**: 
  - **端侧推理**: 数据不出园，满足幼儿园数据合规要求，无云依赖降低成本
  - **局域网本地闭环**: 所有数据采集、AI 边缘推理、加密存储全流程在教室内终端设备本地闭环完成，**坚决不连外网、不上公有云**
  - **两级检测**: Stage1 粗筛（320px）+ Stage2 精检（640px），平衡精度与算力
  - **动态 ROI**: 基于运动区域裁剪，减少无效推理，提升实时性
  - **异步架构**: asyncio 主循环，充分利用 I/O 等待时间
  - **模块化设计**: Pipeline 编排器协调各模块，低耦合易扩展
- **技术选型理由**: 
  - **LoongArch**: 国产化硬件平台，满足信创要求。硬件规格：8x LA364E 核心 + LG200 GPGPU (第二代自研通用 GPU，图形性能相比第一代提升 2 倍) + 16GB RAM。应用领域：工业控制、边缘计算
  - **YOLO26-Nano**: 边缘优先设计的轻量级检测模型，专为实时部署优化。采用 **NMS-Free 端到端架构**，移除 DFL 模块，消除传统 NMS 后处理，大幅减少边缘 CPU 延迟。INT8 量化提供约 **30% 的延迟改进**，支持 **43% 更快的 CPU 推理**（无精度损失），量化稳定性优秀，适合资源受限的边缘设备
  - **MoveNet-Lightning**: TensorFlow Hub 超快速姿态估计模型，**30+ FPS 实时性能**，比 OpenPose 更快。单人场景优化，准确率 75.1%（Thunder 版本 80.6%）。适用场景：健身应用、实时监控、幼儿俯卧睡姿检测
  - **SM4 国密**: 符合国密标准，满足国内安全合规要求。依托龙芯 2K3000 芯片内置的 SM2/3/4 国密硬件加速模块
  - **aiohttp**: 异步 Web 框架，支持 REST + WebSocket + MJPEG 多协议
  - **SQLite**: 轻量级数据库，无需额外服务，适合嵌入式场景
  - **包管理**: pip + requirements.txt（不使用 uv/conda）
- **重要设计模式**: 
  - **编排器模式**: `Pipeline` 类协调所有子模块生命周期
  - **策略模式**: 运动检测、目标检测、姿态估计可替换实现
  - **降级模式**: 所有硬件相关模块支持 Mock 降级（Windows 开发环境）
  - **观察者模式**: WebSocket 实时推送告警事件
- **关键技术约束**: 
  - **算力预算**: LG200 仅 8 TOPS INT8，模型必须 INT8 量化，推理管道需严格控制内存占用
  - **NMS-Free 架构**: YOLO26 采用 NMS-Free 端到端设计，代码中**禁止引入传统 NMS 逻辑**
  - **内存安全**: 视频帧数据**仅在内存 Tensor 中参与推理，断电即失，绝不落盘**
  - **实时性**: 端到端推理延迟需控制在 40ms 以内（25 FPS），含 ROI 裁剪和二级推理
  - **跨平台 Mock**: Windows 环境下 `camera` 模块自动降级为 `cv2.VideoCapture(0)` 或读取 `tests/mock_classroom.mp4`，`alarm` 模块降级为终端日志输出
- **历史包袱说明**: 
  - **安全债务**: 存在多个高危漏洞（密钥明文存储、API 无认证、XSS、路径遍历），需优先修复
  - **测试覆盖**: 集成测试覆盖不足，关键路径需补充端到端测试
  - **文档缺失**: 缺少 README.md，新成员上手成本高
  - **环境隔离**: `.env` 文件被提交到仓库，需立即移除并轮换密钥
- **模块边界**: 
  - **本仓库负责**: V4L2 摄像头采集、YOLO26-Nano 目标检测、MoveNet-Lightning 姿态估计、Dynamic ROI 二级检测调度、帧差分/运动检测、GPIO 告警触控层、SM4 加密日志、摄像头和算法相关的后端 API 接口
  - **不属于本仓库**: Web 管理端前端页面、语音唤醒/语音交互模块、温湿度/紫外线传感器模块、系统级服务管理（systemd 等）

## 6. 工作流程
- **部署流水线**: 
  ```
  Windows 主机（日常编码）
    └─> 
          └─> 3A5000 主机 (Loongnix_v25)（集成测试 / 性能验证）
                └─> 龙芯 2K3000 原型机（最终上线前测试）
  ```
  跨平台开发时注意 LoongArch 架构的字节序和对齐差异，所有推理性能基准以 2K3000 实机为准。
- **开发流程步骤**: 
  1. **本地开发**: Windows 环境编写代码，所有硬件模块自动 Mock 降级
  2. **本地测试**: `pytest` 运行单元测试，`visual_test.py` 可视化验证
  3. **远程同步**: `python scripts/remote_exec.py sync` 同步到龙芯设备
  4. **远程测试**: `python scripts/remote_exec.py test` 在目标设备运行测试
  5. **远程部署**: `python scripts/remote_exec.py run` 启动服务
- **PR 审核标准**: 
  - [ ] 代码通过所有测试（`pytest`）
  - [ ] 无硬编码密钥/密码/凭证
  - [ ] 类型注解完整，无 `any` 使用
  - [ ] 关键逻辑有设计动机注释
  - [ ] 新增功能有对应测试用例
  - [ ] 安全审计通过（无新增高危漏洞）
- **发布流程说明**: 
  - 当前为 alpha 版本（0.1.0），无正式发布流程
  - 版本号遵循语义化版本规范（SemVer）
  - 发布前需完成安全漏洞修复和测试覆盖
- **问题排查指南**: 
  - **摄像头无法打开**: 检查 `LG_CAMERA_DEVICE` 是否正确，Linux 下确认 `/dev/videoN` 权限
  - **模型加载失败**: 确认 `models/` 目录下模型文件存在，路径配置正确
  - **SM4 加密失败**: 检查 `LG_SM4_KEY` 是否为 32 位 hex 字符串
  - **API 无法访问**: 确认 `LG_API_HOST=0.0.0.0`，防火墙开放端口
  - **远程同步失败**: 检查 SSH 配置（`LG_SSH_*` 环境变量），确认网络连通性
  - **性能问题**: 调整检测阈值、ROI 参数，启用 INT8 量化，降低输入分辨率
