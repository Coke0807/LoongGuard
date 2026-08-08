"""
LoongGuard 后端统一启动入口

设计动机：
    消除多入口（python -m src.pipeline / loongguard 控制台脚本）导致的维护歧义，
    统一以 run.py 作为唯一执行入口，内部委托 src.pipeline.main 启动应用。

使用方式：
    cd backend
    python run.py [config/default.json]

说明：
    - 默认加载 config/default.json（与旧入口行为一致）
    - 端口被占用等致命错误会以非零退出码结束，交由进程守护（start.ps1 / systemd）重启
"""
from __future__ import annotations

import asyncio
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# backend/ 根目录使 config/ 可导入；backend/src 使 loongguard 顶级包可导入（标准 src-layout）
sys.path.insert(0, os.path.join(BACKEND_DIR, "src"))
sys.path.insert(0, BACKEND_DIR)

from loongguard.pipeline import main

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.json"
    try:
        asyncio.run(main(config_path))
    except KeyboardInterrupt:
        # Ctrl+C 优雅退出（Windows 下 SIGINT 唯一支持的信号）
        pass