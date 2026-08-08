# 跨平台 MVP 集成策略评估报告

**项目**：幼儿园安全守护卫士
**评估目标**：Windows 开发环境的 MVP 集成可行性 + 跨平台扩展架构
**目标运行环境**：龙芯 2K3000 + Loongnix V25 桌面版（LoongArch）
**评估日期**：2026-08-07

---

## 一、前端代码 Windows 端探查结果

### 1.1 资产盘点

| 文件 | 角色 | 状态 |
| :--- | :--- | :--- |
| `frontend/main.py` | PyQt6 主窗口（1280×720，含模式/录制/温湿度/语音） | 完整，但绑定板端硬件 |
| `frontend/threads.py` | 摄像头采集 + 录制 + AiMonitor 模拟线程 | ⚠️ 关键：AI 是假随机，未接后端推理 |
| `frontend/utils.py` | 日志/视频清理/音视频合并 | ⚠️ 依赖 `arecord`/`ffmpeg`（Linux） |
| `frontend/voiceplayer.py` | 语音播报 | ⚠️ 依赖 `ffplay`（Linux） |
| `frontend/sensor_thread.py` / `mgtRs485.py` | RS485 温湿度采集 | 硬件绑定（板端 `/dev/ttyS*`） |
| `cature.py` / `main1.py` / `ui.py` / `ui0620.py` / `ui_可以录制画面.py` | 历史/开发中间版本 | 冗余，MVP 不纳入 |

### 1.2 前端运行阻塞点（静态判定）

1. **摄像头**：`threads.py` 使用 `cv2.CAP_V4L2` —— 仅 Linux 存在。
2. **音频**：`utils.py` 用 `arecord` 录麦、`voiceplayer.py` 用 `ffplay` 播 —— 均为 Linux 命令。
3. **RS485 温湿度**：需真实串口 + 传感器硬件。
4. **AI 集成缺口（最关键）**：前端 `AiMonitorThread` 是随机告警模拟，与后端 ONNX 推理无调用关系；前端无 HTTP/WebSocket 客户端消费后端 `/stream`、`/api/v1/alerts`、`/ws/alerts`。

> **核心结论**：前端与后端是两个独立工程，未做对接。前端本质是"板端带屏展示 + 硬件控制"客户端。

---

## 二、MVP 阶段性集成可行性结论

**决策：采用「B 为主的混合方案」—— A/B 组合。**

| 方案 | 是否采用 | 理由 |
| :--- | :--- | :--- |
| A) 修复前端跑 Windows | 部分 | 仅做跨平台封装，前端最终在板端运行 |
| **B) 临时替代（推荐主路径）** | ✅ | 后端已内置 Web 管理面板（`backend/src/api/web/index.html`），自带 MJPEG `/stream` + WebSocket 告警，可在 Windows 零改动验证 YOLO→告警→直播 |
| C) 等待完整前端 | 否 | 前端已存在，无阻塞 |

**最终目标形态（板端）**：
```
后端 pipeline（YOLO26 推理 + 运动检测 + ROI 告警）
        │
        ├─ /stream  MJPEG 直播
        ├─ /ws/alerts  WebSocket 告警推送
        └─ /api/v1/status|stats|alerts  REST
前端 PyQt6（板端显示屏展示 + 硬件控制）
```

**Windows 开发期验证**：后端 + 上传视频跑通 YOLO→告警→直播 全链路；前端跨平台封装静态验证，最终在板端运行。

---

## 三、跨平台扩展接口设计方案摘要

### 3.1 人脸检测联动（Person + Face 组合）

- **算法逻辑**：`Person bbox` 内无 `Face` → 判定异常睡姿（脸被遮挡/趴睡）。
- **抽象接口**：新增 `FaceDetector` 协议（`detect(image) -> [BoundingBox]`），与现有 `inference_onnx.py` 解耦。
- **跨平台切换**：复用 `onnx_session.py` 的 provider 选择（CUDA > OpenCL > CPU），无需改业务代码，仅提供模型文件。
- **Windows/板端桩实现**：提供 `DummyFaceDetector`（无模型时默认"检测到人脸"），保证开发期不阻塞。

### 3.2 语音功能预留

- **标准化事件总线**：定义 `VoiceEvent` 消息队列接口（`wakeup / asr_result / call_push / call_answer / call_hangup`），以 `asyncio.Queue` 实现，业务层不绑定具体 ASR/通话 SDK。
- **跨平台后端**：板端可接 LoongArch 适配的 ASR（如 FunASR）与音频设备；开发期提供桩实现。
- **约定**：所有语音能力以"接口 + 桩实现"方式提供，替换实现不改调用方。

### 3.3 小程序集成评估（家长端远程客户端）

| 维度 | 评估 |
| :--- | :--- |
| 后端是否需加 WebSocket/WebRTC | **需要**：`/ws/alerts` 已够告警；家长端视频需 WebRTC 信令或复用 MJPEG。建议优先 WebRTC |
| 流媒体选型 | **WebRTC**（推荐）：小程序 `live-player` 原生支持，低延迟；MJPEG/HTTP-FLV 仅适合局域网管理端；RTSP 小程序不支持，弃用 |
| 技术路线 | 后端暴露 WebSocket 信令 + 视频帧（复用 VideoHub）→ 板端/开发机做 WebRTC 转发 → 小程序 `live-player` 拉流；语音通话用 `live-pusher`/`live-player` 双通道 |

**迁移兼容性**：WebRTC 需板端 loongarch 库，建议抽象为 `StreamPublisher` 接口，Windows 用 MJPEG 桩、板端用 WebRTC 实现，避免现在绑定。

---

## 四、Windows → Loongnix V25 迁移风险点

| 风险项 | 影响 | 对策 |
| :--- | :--- | :--- |
| ONNX Runtime LoongArch 版本 | 官方 pip 无 loongarch 轮子 | 从源码/龙芯仓库编译，或使用龙芯 AI 适配版；`requirements_loongarch.txt` 已隔离部署依赖 |
| `opencv-python` 轮子 | pip 无 loongarch | 板端用系统包管理器装 `python3-opencv`；`CAP_V4L2` 保留板端、Windows 走 `CAP_DSHOW` |
| 音频/视频命令 | `arecord`/`ffplay`/`ffmpeg` 路径差异 | 抽取 `AudioBackend`/`MediaBackend` 抽象，按平台注入 |
| RS485 串口 | 板端 `/dev/ttyS*` vs Windows COM | 串口名走配置项，不硬编码 |
| 性能 | LG200 8 核 CPU 推理 | 量化 INT8（`scripts/quantize_yolo26_int8.py` 已存在）、线程绑定（`onnx_session.py` 支持 `LG_ONNX_INTRA_OP_THREADS`）、MJPG 压缩流 |
| 依赖双版本 | 开发 torch/py 3.11 vs 板端 3.12 | 已用两套 requirements 隔离，运行时仅装 `requirements_loongarch` |

所有预留接口均提供 Windows 桩实现，保证开发阶段不被平台阻塞。

---

## 五、待确认事项清单（已确认）

| # | 事项 | 确认结果 |
| :--- | :--- | :--- |
| 1 | MVP 主路径 | 最终运行于 2K3000 板端 Loongnix V25；Windows 仅作开发环境 |
| 2 | 人脸模型 | 无 ONNX 权重，接受 `DummyFaceDetector` 桩 + 预留接口 |
| 3 | 前端改造范围 | 跨平台封装，最终于板端运行 |
| 4 | 小程序路线 | 按 "WebRTC 信令 + 抽象 `StreamPublisher` 接口" 预留，暂不实现具体转发 |
| 5 | 报告归档 | 本文件（`.md`）作为归档 |

---

## 六、后续实施计划（待确认后执行）

1. 后端新增跨平台预留接口：`FaceDetector` 协议 + `DummyFaceDetector` 桩、`AudioBackend`/`MediaBackend` 抽象、`StreamPublisher` 接口。
2. 前端跨平台封装：修正 `CAP_V4L2`/`arecord`/`ffplay`；`AiMonitorThread` 增加对接后端告警的"真实模式"。
3. Windows 验证：后端 + 上传视频跑通 YOLO→告警→直播 全链路。