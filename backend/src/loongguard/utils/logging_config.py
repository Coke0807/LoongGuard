"""
结构化日志配置模块

设计动机（12-Factor App: Logs）：
    将日志视为事件流，输出到 stdout + 文件。
    开发环境使用人类可读格式，生产环境使用 JSON 结构化格式，
    便于日志采集器（Filebeat / Fluentd）解析。

环境变量：
    LG_LOG_LEVEL   - 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    LG_LOG_FORMAT  - 输出格式 (text/json)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """将每条日志记录输出为单行 JSON 对象"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno,
        }

        # 合并调用方传入的 extra 字段
        if hasattr(record, "__dict__"):
            standard_attrs = logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__.keys()
            for key, value in record.__dict__.items():
                if key not in standard_attrs and not key.startswith("_"):
                    log_entry[key] = value

        # 异常信息单独提取为 exception 字段
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    fmt: str = "text",
    log_dir: str = "logs",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    console: bool = True,
) -> None:
    """
    初始化全局日志系统

    Args:
        level:          日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        fmt:            输出格式 —— "text" 人类可读 / "json" 结构化
        log_dir:        日志文件存放目录
        max_bytes:      单个日志文件大小上限（字节）
        backup_count:   RotatingFileHandler 保留的轮转文件数
        console:        是否同时输出到 stdout
    """
    root = logging.getLogger()
    # 清除已有的 handler，避免重复输出
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Windows 终端 UTF-8 修复：确保中文日志正确显示
    # 设计动机：中文 Windows 默认 GBK 编码，无法显示中文 Unicode 字符。
    # 强制 stdout 使用 UTF-8 编码，日志文件始终使用 UTF-8。
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 选择格式化器
    if fmt == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # 文件 handler：轮转写入，每文件最大 10 MB，保留 5 个备份
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "loongguard.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # 控制台 handler：输出到 stdout（12-Factor: 日志即事件流）
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # 抑制第三方库的冗余日志
    for noisy_logger_name in ("aiohttp", "paramiko", "urllib3"):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)
