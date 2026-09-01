"""
LoongGuard 共享的 .env 文件加载模块

设计动机：
    消除 backend/config/settings.py 和 frontend/ui/main.py 中
    重复的 .env 加载逻辑，提供统一的环境变量解析语义。

使用方式：
    from loongguard.utils.dotenv import load_dotenv, strip_env_value

    # 加载 .env 文件到 os.environ
    load_dotenv()

    # 解析单个环境变量值（处理引号和注释）
    value = strip_env_value("some_value  # comment")
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def strip_env_value(value: str) -> str:
    """
    解析 .env 值（统一语义）：

        - 引号包裹（' 或 "）的值原样保留，内部 # 不截断：
            LG_AUTH_PASS="p@ss#word" -> p@ss#word
          引号闭合后的剩余内容视为行内注释丢弃。
        - 未加引号时，仅当 # 前有空白才视为行内注释（与既有 .env.example
          的 "值  # 说明" 写法兼容）；紧跟值的 #（如 abc#def）属于值本身。
          设计动机（修复）：旧实现一律从首个 # 截断，含 # 的密码/密钥会被
          静默破坏且无任何提示。

    Args:
        value: 原始环境变量值字符串

    Returns:
        解析后的干净值字符串
    """
    value = value.strip()
    if value[:1] in ("'", '"'):
        end = value.find(value[0], 1)
        if end > 0:
            return value[1:end]
    match = re.search(r"\s#", value)
    if match:
        value = value[: match.start()].rstrip()
    return value


def load_dotenv(
    dotenv_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> None:
    """
    极简 .env 加载器（零第三方依赖）。

    设计动机：
        避免为加载 .env 引入 python-dotenv 依赖（LoongArch 环境包管理
        不便）。已存在于 os.environ 的变量优先，不覆盖，保证 shell 显式
        注入的配置具备最高优先级。

    查找顺序（dotenv_path 为 None 时）：
        1. backend/.env   —— 与 deploy/loongguard.service 的 EnvironmentFile
           及手动部署文档口径一致
        2. 仓库根 .env    —— 开发机"前后端共享单一配置源"口径
        取第一个存在的文件。

    同时检测文件内重复 KEY 并告警，避免"后值覆盖前值"的隐性 bug。

    Args:
        dotenv_path: 显式指定的 .env 文件路径，None 时自动查找
        project_root: 项目根目录，None 时自动推断（仅用于自动查找模式）
    """
    if dotenv_path:
        path = Path(dotenv_path)
        if not path.exists():
            return
    else:
        # 自动查找 .env 文件
        if project_root is not None:
            candidates = [Path(project_root) / ".env"]
        else:
            # 从当前文件位置推断：loongguard/utils/dotenv.py -> backend/src/loongguard/utils/
            # parents[3] = backend/, parents[4] = 仓库根
            here = Path(__file__).resolve()
            candidates = [here.parents[3] / ".env", here.parents[4] / ".env"]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            return

    seen: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("无法读取 .env 文件: %s", path)
        return

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = strip_env_value(value)
        if not key:
            continue

        # 重复 KEY 检测
        if key in seen:
            logger.warning(
                ".env 中 KEY 重复定义: %s (第 %d 行 与 第 %d 行)，后值将覆盖前值",
                key, seen[key], line_no,
            )
        else:
            seen[key] = line_no

        # 已存在的环境变量优先，不覆盖
        if key not in os.environ:
            os.environ[key] = value


# 便捷函数：供 frontend/ui/main.py 使用的简化版本
def load_project_dotenv() -> None:
    """
    加载项目根目录的 .env 文件（便捷函数）。

    此函数会自动查找项目根目录的 .env 文件并加载到 os.environ。
    已存在的环境变量优先，不覆盖。
    """
    load_dotenv()
