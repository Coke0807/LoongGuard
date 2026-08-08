"""前端 ui 包冒烟测试

用途：
    验证核心 UI 模块可正常导入，防止 src->ui 目录重命名或导入路径回归。

运行：
    cd frontend
    python -m pytest tests/
"""

import sys
from pathlib import Path

FRONTEND_ROOT = Path(__file__).parent.parent
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))


def test_ui_package_importable():
    """ui 顶级包应可导入"""
    import ui
    assert ui.__name__ == "ui"


def test_ui_core_modules_importable():
    """核心模块应可导入（覆盖 src->ui 改名后的导入路径）"""
    import ui.main
    import ui.threads
    import ui.sys_settings
    import ui.voiceplayer
    import ui.sensor_thread
    import ui.mgt_rs485
    import ui.settings_db
    import ui.widgets
    import ui.utils