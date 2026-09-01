# backend — LoongGuard 后端（aiohttp 单端口 8080）

启动入口统一为 `run.py`（在 `backend/` 目录下执行 `python run.py`，可选参数为
JSON 配置路径，默认 `config/default.json`）。`.env` 由 `pipeline.main()` 在启动时
加载（优先 `backend/.env`，其次仓库根 `.env`）。

## 目录结构

```
backend/
├── run.py                 # 唯一启动入口（配置校验失败/端口占用即退出，交由守护重启）
├── config/
│   ├── settings.py        # AppConfig 全部 dataclass + LG_* 环境变量覆盖机制
│   └── default.json       # JSON 配置（优先级低于环境变量）
├── src/loongguard/
│   ├── pipeline.py        # 主编排器：采集→运动→ROI→姿态→告警 全链路主循环
│   ├── camera/            # V4L2 C 扩展采集（板端）+ OpenCV 降级（Windows/CI）
│   ├── motion/            # 帧差分运动检测
│   ├── roi/               # 动态 ROI 二级检测调度
│   ├── detection/         # YOLO26-Nano ONNX 推理（危险物品）
│   ├── pose/              # MoveNet-Lightning（俯卧检测）
│   ├── face/              # 人脸检测 + 睡姿多分类（默认关闭，LG_FACE_ENABLED=true）
│   ├── voice/             # 语音模块：openWakeWord 唤醒 + PocketSphinx 指令 + TTS
│   ├── alarm/             # GPIO 声光告警（非 Linux 平台自动 mock）
│   ├── alerts/            # 告警去重/持续性过滤/webhook 通知
│   ├── crypto/            # SM4 加密日志存储
│   ├── db/                # database.py 告警库 + parent_db.py 家长端业务库
│   ├── api/               # server.py REST/MJPEG/WS + parent_routes.py 家长端 + metrics
│   ├── media/             # 音频后端 + 流媒体发布器（mjpeg/webrtc 预留）
│   └── utils/             # schema/logging/onnx_session/retention/parent_utils
├── models/                # ONNX 权重（Git LFS；见 docs/08-模型权重介绍/）
├── templates/ static/     # 管理后台 Jinja2 页面与静态资源（/admin 路由）
├── data/                  # 运行时数据（db/、uploads/、recordings/、class_monitor.db）
├── scripts/               # 训练/转换/同步/端到端脚本（train_wakeword.py 等）
├── deploy/                # systemd 服务单元 + Caddyfile + start.ps1
└── tests/                 # unit / integration / fixtures
```

## 两套数据库（物理隔离）

| 库 | 路径 | 内容 |
|---|---|---|
| 告警库 | `data/db/loongguard.db` | 告警记录、审计日志（`LG_DATABASE_DB_PATH`） |
| 家长端业务库 | `data/class_monitor.db` | 管理员/班级/幼儿/家长/公告/成长记录（`LG_PARENT_DB_PATH`） |

家长端业务库为空库时自动建表并生成**随机初始口令**的 `admin` 账号（口令仅在
首次创建时的日志中输出一次，请立即登录 `/admin/login` 创建自有账号）。

## 测试

```bash
python -m pytest tests/ -q                 # 全量
python -m pytest tests/unit -q             # 单元（CI 必跑）
python -m pytest tests/integration -q      # 冒烟 + 真实素材端到端
```

前置条件：模型权重为真实文件（clone 后先 `git lfs pull`）。`tests/fixtures/`
下的幼儿园实拍素材同样走 LFS。

## 端口与路径约定

后端只在 `LG_API_PORT`（默认 8080）一个端口上同时提供 REST、`/stream` MJPEG、
`/ws/alerts` WebSocket、`/admin` 管理后台、`/api/wx/*` 小程序接口。端口被占用时
启动失败退出（不自动换端口），由 systemd/start.ps1 守护重启。
