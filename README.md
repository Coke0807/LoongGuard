# LoongGuard — 幼儿园安全守卫监控与管理系统

基于龙芯 2K3000 开发板（Loongnix 25 桌面版 / LoongArch64）的幼儿园安全监控系统：
AI 视觉检测危险物品与异常睡姿，语音唤醒交互，声光告警，家长通过微信小程序远程查看
幼儿动态。**最终部署目标为板侧单机运行**，开发机为 Windows。

## 系统组成（本仓库 = 主开发仓库）

```
LoongGuard/
├── backend/          # 核心：aiohttp 单端口后端（8080）
│   ├── src/loongguard/   # 检测/姿态/人脸/语音/告警/API/DB 各模块
│   ├── config/           # settings.py 全局配置 + default.json
│   ├── models/           # ONNX 模型权重（Git LFS）
│   ├── templates/ static/# 管理后台页面（Jinja2）
│   └── tests/            # 单元 + 集成 + 真实素材端到端
├── frontend/         # PyQt6 桌面管理端（板端触摸屏）
├── mini-program/     # 微信小程序（家长端）
└── docs/             # 项目文档（部署/模型/语音/集成，见下文导航）
```

- 后端在 8080 端口同时提供：REST API、MJPEG 实时流 `/stream`、告警 WebSocket
  `/ws/alerts`、管理后台 `/admin`、小程序 API `/api/wx/*`。
- `D:\User\Desktop\uiprogram`（分仓库1）与 `uiprogram\VoiceDetection`（分仓库2）
  仅为集成参考的历史仓库，不再开发；现行代码以本仓库为准。

## 核心功能与模型

| 能力 | 实现 | 模型/依赖 |
|---|---|---|
| 危险物品检测（剪刀/美工刀等） | 帧差分运动检测 → 动态 ROI 二级检测 | `models/best.onnx`（YOLO26-Nano） |
| 俯卧趴睡检测 | MoveNet 关键点 + 肩髋角度判定 | `models/movenet_lightning_int8.onnx` |
| 睡姿监测（可选） | 人脸可见性 + 头部姿态多分类 | `models/buffalo_l/det_10g.onnx` |
| 语音唤醒 | openWakeWord（ONNX，~200KB 自训模型） | `models/wakeword/my_wakeword.onnx` |
| 语音指令 | PocketSphinx + JSGF 中文语法 | `models/pocketsphinx/`（zh-cn 声学模型） |
| 告警链路 | 去重/持续过滤 → GPIO 声光 → SM4 加密日志 → SQLite → WebSocket/webhook | — |
| 家长端 | 幼儿绑定/公告/成长记录/远程观看授权 | `data/class_monitor.db` |
| 远程观看 | 小程序 `<video>` 播放 HLS 直播（ffmpeg 切片 + 过期 token 保护，无人观看自动暂停编码） | 板端 `apt install ffmpeg` |

各模型详细说明见 [docs/08-模型权重介绍/README.md](docs/08-模型权重介绍/README.md)，
权重替换方式见 [docs/02-部署与容器/模型权重文件更换指南.md](docs/02-部署与容器/模型权重文件更换指南.md)。

## 快速开始（Windows 开发机）

```bash
# 0. 前置：本仓库模型权重走 Git LFS，clone 后必须先拉取真实权重，
#    否则 backend/models/*.onnx 只有 ~130 字节的指针文件，测试与启动全部失败
git lfs install
git lfs pull          # 首次 clone 后执行（约 580MB）

# 1. 创建虚拟环境（Python 3.13.x）并安装依赖
cd backend
python -m venv ../.venv
../.venv/Scripts/pip install -r requirements.txt

# 2. 配置环境变量：复制 .env.example 为项目根目录 .env 并填入
#    LG_SM4_KEY（openssl rand -hex 16）等必填项
cp ../.env.example ../.env

# 3. 启动后端（无摄像头时自动降级 OpenCV 读测试视频）
python run.py

# 4. 验证
#    浏览器打开 http://127.0.0.1:8080/stream 与 http://127.0.0.1:8080/admin/login
#    健康检查 http://127.0.0.1:8080/health

# 5. 运行测试
../.venv/Scripts/python -m pytest tests/ -q
```

环境变量总表见根目录 `.env.example`（`LG_` 前缀，前后端共享同一份根 `.env`）。

## 板端部署（龙芯 2K3000）

部署唯一权威入口：
[docs/02-部署与容器/龙芯2K3000手动部署文档.md](docs/02-部署与容器/龙芯2K3000手动部署文档.md)。
2026-08-29 三模块集成（语音 + 家长端 + 小程序）之后的增补事项见文档开头的
「集成增补」章节；语音模块部署细节见
[docs/03-语音与音频/语音唤醒与指令实现.md](docs/03-语音与音频/语音唤醒与指令实现.md)。

## 文档导航

| 目录 | 内容 |
|---|---|
| [docs/00-项目规划与架构](docs/00-项目规划与架构) | 集成策略评估、架构记录 |
| [docs/01-环境配置](docs/01-环境配置) | 开发环境搭建记录 |
| [docs/02-部署与容器](docs/02-部署与容器) | **板端部署权威文档**、模型权重更换指南 |
| [docs/03-语音与音频](docs/03-语音与音频) | **语音唤醒与指令实现（现行方案）**、声音配置 |
| [docs/04-远程访问与视频流](docs/04-远程访问与视频流) | **家长端与小程序接入**、端口映射/HTTPS |
| [docs/05-系统功能与需求](docs/05-系统功能与需求) | 功能列表、系统设置需求 |
| [docs/06-硬件平台](docs/06-硬件平台) | 2K3000 板资料、RS485 传感器调试 |
| [docs/07-账号与密钥](docs/07-账号与密钥) | 板端系统账号 |
| [docs/08-模型权重介绍](docs/08-模型权重介绍/README.md) | **每个模型的作用与参数** |
| [docs/09-项目集成](docs/09-项目集成) | 2026-08 三仓库集成方案与完成报告 |
| docs/archive/ | 历史归档（旧版二进制资料），不承担现行文档职责 |

各文档时效状态详见 [docs/README.md](docs/README.md)。
