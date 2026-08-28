"""Prometheus 指标模块单元测试"""

from __future__ import annotations

import time

import pytest

from src.api.metrics import (
    ALERT_COUNT,
    ALERT_DEDUP_SUPPRESSED,
    DB_OPERATIONS,
    DETECTION_COUNT,
    FRAME_COUNT,
    UPLOAD_COUNT,
    MetricsTimer,
    get_metrics_bytes,
    get_metrics_content_type,
    _HAS_PROMETHEUS,
)
import src.api.metrics as metrics_module

# 根据 prometheus_client 是否安装决定跳过测试
requires_prometheus = pytest.mark.skipif(
    not _HAS_PROMETHEUS,
    reason="prometheus_client 未安装，跳过真实指标测试"
)


class TestMetricsModule:
    """指标定义和基本操作测试"""

    def test_get_metrics_bytes_returns_bytes(self):
        """get_metrics_bytes() 应返回 bytes"""
        result = get_metrics_bytes()
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_metrics_content_type(self):
        """Content-Type 应为 prometheus 文本格式"""
        ct = get_metrics_content_type()
        assert "text/plain" in ct

    @requires_prometheus
    def test_frame_count_increments(self):
        """FRAME_COUNT 递增后应体现在输出中"""
        before = FRAME_COUNT._value.get()
        FRAME_COUNT.inc()
        after = FRAME_COUNT._value.get()
        assert after == before + 1

    @requires_prometheus
    def test_detection_count_with_labels(self):
        """DETECTION_COUNT 应支持按类别标签统计"""
        DETECTION_COUNT.labels(class_name="scissors").inc()
        DETECTION_COUNT.labels(class_name="needle").inc(3)
        output = metrics_module.generate_latest().decode()
        assert 'class_name="scissors"' in output
        assert 'class_name="needle"' in output

    @requires_prometheus
    def test_alert_count_with_labels(self):
        """ALERT_COUNT 应支持严重级别和类型标签"""
        ALERT_COUNT.labels(severity="high", alert_type="dangerous_object").inc()
        output = metrics_module.generate_latest().decode()
        assert 'severity="high"' in output

    @requires_prometheus
    def test_metrics_timer_records_latency(self):
        """MetricsTimer 上下文管理器应记录执行耗时"""
        histogram_before = metrics_module.INFERENCE_LATENCY._sum.get()
        with metrics_module.INFERENCE_LATENCY.time():
            time.sleep(0.01)  # ~10ms
        histogram_after = metrics_module.INFERENCE_LATENCY._sum.get()
        assert histogram_after > histogram_before

    @requires_prometheus
    def test_dedup_suppressed_counter(self):
        """ALERT_DEDUP_SUPPRESSED 递增"""
        before = ALERT_DEDUP_SUPPRESSED._value.get()
        ALERT_DEDUP_SUPPRESSED.inc(5)
        assert ALERT_DEDUP_SUPPRESSED._value.get() == before + 5

    @requires_prometheus
    def test_upload_count(self):
        """UPLOAD_COUNT 递增"""
        before = UPLOAD_COUNT._value.get()
        UPLOAD_COUNT.inc()
        assert UPLOAD_COUNT._value.get() == before + 1

    @requires_prometheus
    def test_db_operations_with_labels(self):
        """DB_OPERATIONS 应支持按操作类型统计"""
        DB_OPERATIONS.labels(operation="insert").inc()
        DB_OPERATIONS.labels(operation="query").inc(10)
        output = metrics_module.generate_latest().decode()
        assert 'operation="insert"' in output


class TestMetricsMockMode:
    """Mock 模式基本功能测试（无需 prometheus_client）"""

    def test_mock_metrics_are_callable(self):
        """Mock 指标应支持基本操作而不报错"""
        # 这些操作在 Mock 模式下应为空操作
        FRAME_COUNT.inc()
        DETECTION_COUNT.labels(class_name="test").inc(5)
        ALERT_COUNT.labels(severity="high", alert_type="test").inc()
        ALERT_DEDUP_SUPPRESSED.inc()
        # Gauge 支持 inc/dec/set
        from src.api.metrics import PIPELINE_FPS
        PIPELINE_FPS.set(25.0)
        PIPELINE_FPS.inc()
        PIPELINE_FPS.dec()

    def test_mock_generate_latest(self):
        """Mock generate_latest() 应返回 bytes"""
        result = metrics_module.generate_latest()
        assert isinstance(result, bytes)
