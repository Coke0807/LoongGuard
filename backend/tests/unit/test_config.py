"""
AppConfig 和 load_config 单元测试

设计动机：
    验证配置模块的默认值正确性、JSON 加载合并逻辑、
    边界错误处理等，确保配置系统在各环境下行为一致。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from config.settings import (
    AppConfig,
    CameraConfig,
    CryptoConfig,
    DetectionConfig,
    NotificationConfig,
    load_config,
    validate_config,
)


# ── AppConfig 默认值测试 ──────────────────────────────────────


class TestAppConfigDefaults:

    def test_default_app_config_types(self):
        """AppConfig 各子配置字段应为正确的 dataclass 类型"""
        cfg = AppConfig()
        assert isinstance(cfg.camera, CameraConfig)
        assert isinstance(cfg.detection, DetectionConfig)
        assert isinstance(cfg.crypto, CryptoConfig)

    def test_default_log_level(self):
        """默认日志级别应为 INFO"""
        cfg = AppConfig()
        assert cfg.log_level == "INFO"

    def test_default_debug_is_false(self):
        """默认 debug 应为 False"""
        cfg = AppConfig()
        assert cfg.debug is False

    def test_detection_config_classes_has_8_items(self):
        """DetectionConfig.classes 默认应包含 8 个检测类别"""
        cfg = DetectionConfig()
        assert len(cfg.classes) == 8
        # 验证所有关键类别都在
        expected = {
            "magnetic_bead", "button_battery", "scissors", "utility_knife",
            "needle", "glass_shard", "wire", "small_toy_part",
        }
        assert set(cfg.classes) == expected

    def test_detection_config_default_values(self):
        """DetectionConfig 默认量化标志和阈值应与 default.json 一致"""
        cfg = DetectionConfig()
        assert cfg.input_size == 640
        assert cfg.conf_threshold == 0.30
        assert cfg.iou_threshold == 0.50
        assert cfg.quantized is False
        assert cfg.backend == "onnxruntime"

    def test_camera_config_default_values(self):
        """CameraConfig 默认应与 default.json 保持一致（640x480@30fps MJPG）"""
        cfg = CameraConfig()
        assert cfg.device == "0"
        assert cfg.width == 640
        assert cfg.height == 480
        assert cfg.fps == 30
        assert cfg.pixel_format == "MJPG"
        assert cfg.buffer_count == 4

    def test_crypto_config_default_paths(self):
        """CryptoConfig 默认路径应指向合理的目录结构"""
        cfg = CryptoConfig()
        assert cfg.key_file.endswith(".sm4_key")
        assert "encrypted" in cfg.log_dir
        assert "slices" in cfg.slice_dir
        assert cfg.max_log_size_mb == 50
        assert cfg.retention_days == 90


# ── load_config 测试 ──────────────────────────────────────────


class TestLoadConfig:
    """load_config 测试套件

    设计动机：
        所有测试需要隔离外部 LG_* 环境变量（可能来自 .env 文件或 shell 环境），
        否则 _apply_env_overrides 会覆盖预期默认值。
    """

    @pytest.fixture(autouse=True)
    def _clean_lg_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """自动清理所有 LG_* 环境变量，确保测试不受外部环境干扰"""
        for key in list(os.environ):
            if key.startswith("LG_"):
                monkeypatch.delenv(key, raising=False)

    def test_load_config_none_returns_defaults(self):
        """source=None 应返回完全使用默认值的 AppConfig"""
        cfg = load_config(None)
        assert isinstance(cfg, AppConfig)
        assert cfg.camera.width == 640
        assert cfg.detection.conf_threshold == 0.30
        assert cfg.log_level == "INFO"

    def test_load_config_from_real_json(self):
        """加载 config/default.json 应正确合并配置"""
        default_json = Path(__file__).parent.parent.parent / "config" / "default.json"
        assert default_json.exists(), f"default.json not found at {default_json}"

        cfg = load_config(str(default_json))
        assert isinstance(cfg, AppConfig)
        # default.json 中指定了 api.port=8080
        assert cfg.api.port == 8080
        # camera.device 为 "0"（Windows 兼容，自动探测 USB 摄像头索引）
        assert cfg.camera.device == "0"
        # roi.max_rois_per_frame 为 5
        assert cfg.roi.max_rois_per_frame == 5

    def test_load_config_from_json_overrides_only_specified(self):
        """JSON 中仅覆盖指定字段，未指定字段应保留默认值"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"camera": {"fps": 15}, "log_level": "DEBUG"}, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            # 覆盖的字段
            assert cfg.camera.fps == 15
            assert cfg.log_level == "DEBUG"
            # 未覆盖的字段应保持默认值
            assert cfg.camera.width == 640
            assert cfg.camera.height == 480
            assert cfg.camera.device == "0"
            assert cfg.detection.conf_threshold == 0.30
            assert cfg.debug is False
        finally:
            Path(tmp_path).unlink()

    def test_load_config_from_json_top_level_override(self):
        """JSON 顶层非 dict 字段（如 log_level, debug）应直接覆盖"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"log_level": "WARNING", "debug": True}, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.log_level == "WARNING"
            assert cfg.debug is True
        finally:
            Path(tmp_path).unlink()

    def test_load_config_invalid_path_raises(self):
        """不存在的 JSON 文件应抛出异常"""
        with pytest.raises((FileNotFoundError, OSError)):
            load_config("/nonexistent/path/to/config.json")

    def test_load_config_unsupported_source_raises(self):
        """不支持的配置源类型应抛出 ValueError"""
        with pytest.raises(ValueError, match="Unsupported config source"):
            load_config("some_config.yaml")

    def test_load_config_partial_json_unknown_keys_ignored(self):
        """JSON 中包含未知 section 名时，不影响已有字段"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"unknown_section": {"foo": "bar"}, "camera": {"fps": 25}}, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.camera.fps == 25
            # 未知 section 不会破坏 AppConfig
            assert not hasattr(cfg, "unknown_section")
        finally:
            Path(tmp_path).unlink()

    def test_load_config_json_unknown_field_key_ignored(self):
        """JSON section 内包含未知 key 时应被忽略，不影响已有字段"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"camera": {"fps": 25, "unknown_key": "value"}}, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.camera.fps == 25
            assert not hasattr(cfg.camera, "unknown_key")
        finally:
            Path(tmp_path).unlink()


# ── 环境变量覆盖测试 (新加:覆盖本次优化项) ─────────────────


class TestEnvOverride:
    """
    LG_<SECTION>_<KEY> 格式的环境变量自动覆盖 dataclass 字段

    覆盖本次新加的可调参数：
        - LG_POSE_INFERENCE_INTERVAL
        - LG_POSE_PRONE_ANGLE_THRESHOLD
        - LG_POSE_PRONE_FRAME_THRESHOLD
        - LG_DETECTION_CONF_THRESHOLD
    """

    @pytest.fixture(autouse=True)
    def _clean_lg_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in list(os.environ):
            if key.startswith("LG_"):
                monkeypatch.delenv(key, raising=False)

    def test_lg_pose_inference_interval_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LG_POSE_INFERENCE_INTERVAL=5 应覆盖 pose.inference_interval=3 默认值"""
        monkeypatch.setenv("LG_POSE_INFERENCE_INTERVAL", "5")
        cfg = load_config(None)
        assert cfg.pose.inference_interval == 5

    def test_lg_pose_prone_angle_threshold_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LG_POSE_PRONE_ANGLE_THRESHOLD=45.0 应覆盖默认值 30.0"""
        monkeypatch.setenv("LG_POSE_PRONE_ANGLE_THRESHOLD", "45.0")
        cfg = load_config(None)
        assert cfg.pose.prone_angle_threshold == 45.0

    def test_lg_pose_prone_frame_threshold_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LG_POSE_PRONE_FRAME_THRESHOLD=60 应覆盖默认值 30"""
        monkeypatch.setenv("LG_POSE_PRONE_FRAME_THRESHOLD", "60")
        cfg = load_config(None)
        assert cfg.pose.prone_frame_threshold == 60

    def test_lg_detection_conf_threshold_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LG_DETECTION_CONF_THRESHOLD=0.5 应覆盖默认值 0.30"""
        monkeypatch.setenv("LG_DETECTION_CONF_THRESHOLD", "0.5")
        cfg = load_config(None)
        assert cfg.detection.conf_threshold == 0.5

    def test_env_var_overrides_json_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """环境变量优先级高于 JSON 文件(12-Factor 原则)"""
        monkeypatch.setenv("LG_POSE_INFERENCE_INTERVAL", "7")
        cfg = load_config("config/default.json")
        # JSON 默认 inference_interval=3,env 覆盖为 7
        assert cfg.pose.inference_interval == 7


# ── validate_config 测试（覆盖本次安全加固项）────────────────


class TestValidateConfig:
    """配置校验逻辑测试

    设计动机：
        校验在启动早期一次性暴露配置错误，避免生产环境带病/不安全运行。
        每项校验规则独立测试，确保安全红线不被静默绕过。
    """

    @pytest.fixture(autouse=True)
    def _clean_lg_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """清理所有 LG_* 环境变量，隔离外部配置干扰"""
        for key in list(os.environ):
            if key.startswith("LG_"):
                monkeypatch.delenv(key, raising=False)

    def test_missing_sm4_key_reports_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LG_SM4_KEY 缺失应阻断启动"""
        monkeypatch.delenv("LG_SM4_KEY", raising=False)
        errors = validate_config(AppConfig())
        assert any("LG_SM4_KEY" in e for e in errors)

    def test_invalid_sm4_key_length_reports_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LG_SM4_KEY 长度错误（非 32 hex 字符）应报错"""
        monkeypatch.setenv("LG_SM4_KEY", "abcd")  # 长度不足
        errors = validate_config(AppConfig())
        assert any("长度错误" in e for e in errors)

    def test_invalid_sm4_key_hex_reports_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LG_SM4_KEY 非 hex 字符串应报错"""
        monkeypatch.setenv("LG_SM4_KEY", "z" * 32)
        errors = validate_config(AppConfig())
        assert any("不是有效的 hex" in e for e in errors)

    def test_valid_sm4_key_passes_nonprod(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """合法 32 hex 密钥 + 非生产环境应通过校验"""
        monkeypatch.setenv("LG_SM4_KEY", "a" * 32)
        errors = validate_config(AppConfig())
        assert errors == []

    def test_prod_requires_basic_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """生产环境必须启用 Basic Auth，未启用则报错"""
        monkeypatch.setenv("LG_SM4_KEY", "a" * 32)
        cfg = AppConfig()
        cfg.env = "production"
        cfg.api.auth_enabled = False
        errors = validate_config(cfg)
        assert any("Basic Auth" in e for e in errors)

    def test_prod_with_auth_enabled_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """生产环境启用 Basic Auth 且 host 非 0.0.0.0 应通过"""
        monkeypatch.setenv("LG_SM4_KEY", "a" * 32)
        cfg = AppConfig()
        cfg.env = "production"
        cfg.api.auth_enabled = True
        cfg.api.host = "127.0.0.1"
        errors = validate_config(cfg)
        assert errors == []

    def test_prod_host_not_wildcard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """生产环境 host=0.0.0.0 应报错（应由 HTTPS 反向代理转发）"""
        monkeypatch.setenv("LG_SM4_KEY", "a" * 32)
        cfg = AppConfig()
        cfg.env = "production"
        cfg.api.auth_enabled = True
        cfg.api.host = "0.0.0.0"
        errors = validate_config(cfg)
        assert any("0.0.0.0" in e for e in errors)

    def test_notify_enabled_without_webhook_reports_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """外部通知启用但未配置 webhook_url 应报错"""
        monkeypatch.setenv("LG_SM4_KEY", "a" * 32)
        cfg = AppConfig()
        cfg.notify = NotificationConfig(enabled=True, webhook_url="")
        errors = validate_config(cfg)
        assert any("LG_NOTIFY_WEBHOOK_URL" in e for e in errors)

    def test_notify_enabled_with_webhook_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """外部通知启用且配置 webhook_url 应通过"""
        monkeypatch.setenv("LG_SM4_KEY", "a" * 32)
        cfg = AppConfig()
        cfg.notify = NotificationConfig(enabled=True, webhook_url="https://gw.example.com/alert")
        errors = validate_config(cfg)
        assert errors == []
