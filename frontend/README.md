# frontend — LoongGuard 桌面管理端（PyQt6）

运行于龙芯 2K3000 板端触摸屏的 PyQt6 桌面应用，作为后端（aiohttp 8080）的客户端：
连接后端 MJPEG 视频流与告警 WebSocket，并提供本地系统设置、RS485 温湿度传感、
语音提示音播放等板端能力。

## 目录结构

```
frontend/
├── run.py             # 启动入口（python run.py）
├── ui/
│   ├── main.py        # 主窗口；模块导入期先加载项目根 .env（须早于 threads）
│   ├── threads.py     # 后端连接线程：读取 LG_STREAM_URL / LG_WS_URL
│   ├── sensor_thread.py / mgt_rs485.py  # RS485 Modbus 温湿度读取
│   │                  # （backend 语音"查询温湿度"指令亦按路径加载本文件）
│   ├── settings_db.py / sys_settings.py # 系统设置持久化（SQLite）
│   ├── voiceplayer.py # 语音提示音播放
│   ├── widgets.py / utils.py
│   └── voice/         # 26 个 mp3 提示音素材
├── capture/ video_record/ icons/ log/
└── tests/             # test_smoke.py
```

## 运行要点

- **`.env` 加载时机**：`ui/main.py` 在导入 `ui.threads` 之前加载仓库根 `.env`
  （threads 在类体里读取 `LG_WS_URL`/`LG_STREAM_URL`），不要调整 main.py 的
  import 顺序。
- 板端依赖：PyQt6（apt）、ffplay/alsa-utils（音频）、minimalmodbus + pyserial
  （RS485）；Windows 开发机仅作 UI 调试。
- 后端地址：与 `LG_API_PORT` 对应，`LG_STREAM_URL=http://<host>:8080/stream`、
  `LG_WS_URL=ws://<host>:8080/ws/alerts`。
