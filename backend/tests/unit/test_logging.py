"""
结构化日志配置模块单元测试

覆盖：
    - JsonFormatter JSON 输出格式正确性
    - setup_logging 基本功能（控制台/文件 handler 创建）
    - 参数自定义（级别、格式、目录）
    - 异常信息序列化
    - extra 字段合并
    - 第三方库日志抑制
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

from loongguard.utils.logging_config import JsonFormatter, setup_logging


class TestJsonFormatter:
    """JsonFormatter 单行 JSON 输出测试"""

    def _make_record(self, **kwargs) -> logging.LogRecord:
        """构造测试用 LogRecord"""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        for k, v in kwargs.items():
            setattr(record, k, v)
        return record

    def test_format_returns_valid_json(self) -> None:
        """format() 输出必须是合法 JSON"""
        fmt = JsonFormatter()
        record = self._make_record()
        result = fmt.format(record)

        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_format_contains_required_fields(self) -> None:
        """JSON 输出包含 timestamp/level/logger/message/module/lineno"""
        fmt = JsonFormatter()
        record = self._make_record()
        result = json.loads(fmt.format(record))

        assert "timestamp" in result
        assert result["level"] == "INFO"
        assert result["logger"] == "test.logger"
        assert result["message"] == "test message"
        assert result["module"] == "test"
        assert result["lineno"] == 42

    def test_format_includes_extra_fields(self) -> None:
        """extra 字段应合并到 JSON 输出中"""
        fmt = JsonFormatter()
        record = self._make_record()
        record.custom_key = "custom_value"
        record.alert_id = "abc-123"
        result = json.loads(fmt.format(record))

        assert result["custom_key"] == "custom_value"
        assert result["alert_id"] == "abc-123"

    def test_format_excludes_internal_fields(self) -> None:
        """以 _ 开头的字段不应出现在 JSON 输出中"""
        fmt = JsonFormatter()
        record = self._make_record()
        record._internal = "should_not_appear"
        result = json.loads(fmt.format(record))

        assert "_internal" not in result

    def test_format_with_exception_info(self) -> None:
        """异常信息应序列化为 exception 字段"""
        fmt = JsonFormatter()

        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = self._make_record(exc_info=exc_info)
        result = json.loads(fmt.format(record))

        assert "exception" in result
        assert "ValueError" in result["exception"]
        assert "test error" in result["exception"]

    def test_format_handles_non_serializable_args(self) -> None:
        """message 中包含非 JSON 序列化对象时不崩溃"""
        fmt = JsonFormatter()
        record = self._make_record()
        record.msg = "object: %s"
        record.args = (object(),)
        # 应不抛出异常
        result = fmt.format(record)
        parsed = json.loads(result)
        assert "message" in parsed

    def test_format_timestamp_iso8601(self) -> None:
        """timestamp 字段应为 ISO 8601 UTC 格式"""
        fmt = JsonFormatter()
        record = self._make_record()
        result = json.loads(fmt.format(record))

        ts = result["timestamp"]
        # ISO 8601 格式: YYYY-MM-DDTHH:MM:SS
        assert "T" in ts
        assert len(ts) == 19  # "2026-06-26T12:00:00"


class TestSetupLogging:
    """setup_logging 函数测试"""

    def test_setup_logging_text_format(self, tmp_path) -> None:
        """text 格式创建控制台 + 文件 handler"""
        setup_logging(
            level="DEBUG",
            fmt="text",
            log_dir=str(tmp_path / "logs"),
            console=True,
        )

        root = logging.getLogger()
        # 至少有 1 个 handler
        assert len(root.handlers) >= 1
        # 级别应为 DEBUG
        assert root.level == logging.DEBUG

    def test_setup_logging_json_format(self, tmp_path) -> None:
        """json 格式使用 JsonFormatter"""
        setup_logging(
            level="INFO",
            fmt="json",
            log_dir=str(tmp_path / "logs"),
            console=True,
        )

        root = logging.getLogger()
        # 验证至少有一个 handler 使用 JsonFormatter
        has_json = any(
            isinstance(h.formatter, JsonFormatter)
            for h in root.handlers
            if h.formatter is not None
        )
        assert has_json, "Expected at least one handler with JsonFormatter"

    def test_setup_logging_creates_log_dir(self, tmp_path) -> None:
        """setup_logging 应自动创建日志目录"""
        log_dir = tmp_path / "new_logs"
        assert not log_dir.exists()

        setup_logging(level="INFO", log_dir=str(log_dir), console=False)
        assert log_dir.exists()

    def test_setup_logging_no_console(self, tmp_path) -> None:
        """console=False 时不应有 StreamHandler"""
        setup_logging(
            level="INFO",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )

        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0

    def test_setup_logging_creates_log_file(self, tmp_path) -> None:
        """日志文件应被创建"""
        log_dir = tmp_path / "logs"
        setup_logging(level="INFO", log_dir=str(log_dir), console=False)

        # 写一条日志触发文件创建
        test_logger = logging.getLogger("test.file_creation")
        test_logger.info("hello")

        log_file = log_dir / "loongguard.log"
        assert log_file.exists()

    def test_setup_logging_custom_level(self, tmp_path) -> None:
        """自定义日志级别生效"""
        setup_logging(
            level="WARNING",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )

        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_setup_logging_suppresses_noisy_loggers(self, tmp_path) -> None:
        """第三方库日志应被抑制到 WARNING"""
        setup_logging(level="INFO", log_dir=str(tmp_path / "logs"), console=False)

        for name in ("aiohttp", "paramiko", "urllib3"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_setup_logging_idempotent(self, tmp_path) -> None:
        """多次调用 setup_logging 不应累积 handler"""
        log_dir = str(tmp_path / "logs")
        setup_logging(level="INFO", log_dir=log_dir, console=True)
        count1 = len(logging.getLogger().handlers)

        setup_logging(level="DEBUG", log_dir=log_dir, console=True)
        count2 = len(logging.getLogger().handlers)

        # handler 数量不应翻倍
        assert count2 <= count1 + 1

    def test_json_formatter_record_with_extra_dict(self, tmp_path) -> None:
        """JSON 格式下通过 extra 传入的字段能正确输出到日志文件"""
        setup_logging(level="DEBUG", fmt="json", log_dir=str(tmp_path / "logs"), console=False)

        test_logger = logging.getLogger("test.extra")
        test_logger.info("extra test", extra={"request_id": "req-42", "user": "admin"})

        log_file = tmp_path / "logs" / "loongguard.log"
        content = log_file.read_text(encoding="utf-8").strip()

        parsed = json.loads(content)
        assert parsed["request_id"] == "req-42"
        assert parsed["user"] == "admin"
