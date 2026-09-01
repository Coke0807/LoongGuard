"""
LoongGuard 后端统一启动入口

设计动机：
    消除多入口（python -m src.pipeline / loongguard 控制台脚本）导致的维护歧义，
    统一以 run.py 作为唯一执行入口，内部委托 src.pipeline.main 启动应用。

使用方式：
    cd backend
    # 先安装为开发模式（仅首次）
    pip install -e ".[dev]"
    # 然后运行
    python run.py [config/default.json]

说明：
    - 默认加载 config/default.json（与旧入口行为一致）
    - 端口被占用等致命错误会以非零退出码结束，交由进程守护（start.ps1 / systemd）重启
    - 依赖标准 pip install -e 安装方式，不再使用 sys.path.insert hack
"""
from __future__ import annotations

import asyncio
import sys

# 标准包导入：依赖 pip install -e 安装后，loongguard 和 config 包可直接导入
from loongguard.pipeline import main

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.json"
    try:
        asyncio.run(main(config_path))
    except KeyboardInterrupt:
        # Ctrl+C 优雅退出（Windows 下 SIGINT 唯一支持的信号）
        pass
