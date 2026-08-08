"""
LoongGuard 前端启动入口

使用方式：
    cd frontend
    python run.py

依赖：PyQt6, opencv-python, numpy, aiohttp, pyserial
"""
import sys
import os

# 确保项目根目录在 Python 路径中，使 ui/ 包可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main import main

if __name__ == "__main__":
    main()