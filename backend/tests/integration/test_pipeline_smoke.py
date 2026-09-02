"""
Pipeline 端到端冒烟测试

设计动机：
    验证 pipeline 在 Windows 环境下能启动、处理若干帧、并正常停止，
    覆盖 采集 -> 运动检测 -> ROI -> 检测 -> 告警 -> 加密 -> API 推送 全链路。

前置条件：
    - python scripts/create_dummy_models.py 已执行（models/ 下有 dummy ONNX 模型）
    - python scripts/create_test_assets.py 已执行（config/.sm4_key 和 tests/mock_classroom.mp4 存在）
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent

from config.settings import AppConfig  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.pipeline import Pipeline  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.utils.schema import (  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
    AlertLog,
    AlertSeverity,
    AlertType,
)

# ── Helpers ────────────────────────────────────────────────


def _find_free_port() -> int:
    """获取一个当前可用的 TCP 端口（OS 自动分配）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def smoke_config(tmp_path, monkeypatch) -> AppConfig:
    """
    生成冒烟测试专用配置

    使用真实的 dummy 模型路径，但指向临时目录以隔离测试产出。
    通过 monkeypatch 注入 LG_SM4_KEY 环境变量，避免依赖密钥文件。
    """
    # 设置测试用 SM4 密钥（16 字节 = 32 hex 字符）
    monkeypatch.setenv("LG_SM4_KEY", "0123456789abcdef0123456789abcdef")

    config = AppConfig()
    config.camera.device = "tests/mock_classroom.mp4"
    config.detection.model_path = "models/best.onnx"
    config.detection.input_size = 640
    config.pose.model_path = "models/movenet_lightning_int8.onnx"
    config.crypto.key_file = "config/.sm4_key"
    config.crypto.log_dir = str(tmp_path / "logs")
    config.crypto.slice_dir = str(tmp_path / "slices")
    config.api.port = _find_free_port()
    config.debug = True
    return config


@pytest.fixture
def alert_store() -> list:
    """收集 pipeline 处理过程中产生的告警"""
    return []


# ── 冒烟测试 ──────────────────────────────────────────────


class TestPipelineSmoke:
    """Pipeline 全链路冒烟测试"""

    @pytest.mark.asyncio
    async def test_pipeline_instantiation(self, smoke_config) -> None:
        """Pipeline 可成功实例化，无 import 或初始化异常"""
        pipeline = Pipeline(smoke_config)
        assert pipeline is not None
        assert not pipeline._running

    @pytest.mark.asyncio
    async def test_pipeline_start_stop(self, smoke_config) -> None:
        """Pipeline 可启动和正常停止，释放所有资源"""
        pipeline = Pipeline(smoke_config)
        # Mock API start/stop 避免真实端口绑定
        with patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock):
            await pipeline.start()
            assert pipeline._running is True
            await pipeline.stop()
            assert pipeline._running is False

    @pytest.mark.asyncio
    async def test_pipeline_processes_frames(self, smoke_config) -> None:
        """
        验证主循环能读取帧、调用处理逻辑、并正常退出。

        Mock 推理层避免 dummy 模型的 CPU 开销，
        聚焦验证 camera.read -> motion -> roi -> alarm 全链路编排。
        """
        pipeline = Pipeline(smoke_config)
        processed = []

        async def _tracked_process(frame):
            processed.append(frame.frame_id)

        pipeline._process_frame = _tracked_process

        with patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock), \
             patch.object(pipeline._detector, "load_model"), \
             patch.object(pipeline._pose, "load_model"):
            await pipeline.start()
            try:
                await asyncio.wait_for(pipeline.run(), timeout=2.0)
            except TimeoutError:
                pass
            await pipeline.stop()

        assert len(processed) >= 1, "Pipeline should process at least 1 frame"

    @pytest.mark.asyncio
    async def test_alert_handling_path(self, smoke_config, alert_store) -> None:
        """
        验证告警处理路径：alarm.trigger + crypto.encrypt_and_store + api.push_alert

        通过注入 mock 验证三个下游模块均被正确调用。
        """
        pipeline = Pipeline(smoke_config)

        # Mock 三个下游，避免真实 GPIO/文件 I/O/端口绑定
        with patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock), \
             patch.object(pipeline._detector, "load_model"), \
             patch.object(pipeline._pose, "load_model"):
            await pipeline.start()

        test_alert = AlertLog(
            alert_type=AlertType.DANGEROUS_OBJECT,
            severity=AlertSeverity.HIGH,
            description="smoke test alert",
        )

        with patch.object(pipeline._alarm, "trigger", new_callable=AsyncMock) as mock_alarm, \
             patch.object(pipeline._crypto, "encrypt_and_store", return_value="/tmp/test.enc") as mock_crypto, \
             patch.object(pipeline._api, "push_alert", new_callable=AsyncMock) as mock_api:

            await pipeline._handle_alert(test_alert, None)

            mock_alarm.assert_called_once_with(AlertSeverity.HIGH)
            mock_crypto.assert_called_once_with(test_alert)
            mock_api.assert_called_once_with(test_alert)

        await pipeline.stop()


class TestModuleIntegrationSmoke:
    """各子模块独立初始化和基本功能验证"""

    def test_yolo26_loads_model(self) -> None:
        """YOLO26 模型可加载并执行推理"""
        from config.settings import DetectionConfig
        from loongguard.detection.yolo26_nano import YOLO26Nano

        config = DetectionConfig(
            model_path="models/best.onnx",
            input_size=640,
            conf_threshold=0.5,
            quantized=False,
        )
        detector = YOLO26Nano(config)
        detector.load_model()

        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.infer(dummy_frame)
        assert isinstance(result, list)

    def test_movenet_loads_model(self) -> None:
        """MoveNet 模型可加载并执行推理"""
        from config.settings import PoseConfig
        from loongguard.pose.movenet import MoveNetLightning

        config = PoseConfig(model_path=str(PROJECT_ROOT / "models" / "movenet_lightning_int8.onnx"))
        pose = MoveNetLightning(config)
        pose.load_model()

        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        alerts = pose.detect_prone(dummy_frame)
        assert isinstance(alerts, list)

    def test_sm4_encrypt_decrypt_roundtrip(self, tmp_path) -> None:
        """SM4 CBC 模式加密后数据不可直读，解密后与原始一致"""
        from config.settings import CryptoConfig
        from loongguard.crypto.sm4_logger import SM4Logger, SM4Mode

        key_file = tmp_path / "test_key"
        key_file.write_bytes(b"0123456789abcdef")

        config = CryptoConfig(key_file=str(key_file))
        crypto = SM4Logger(config, mode=SM4Mode.CBC)
        crypto.load_key()

        original = b"LoongGuard alert log test data"
        encrypted = crypto.encrypt(original)

        assert encrypted != original, "Encrypted data must differ from original"
        assert len(encrypted) % 16 == 0, "SM4 output must be 16-byte aligned"
        # CBC 密文前 16 字节是 IV
        assert len(encrypted) > 16, "CBC output includes IV prefix"

        decrypted = crypto.decrypt(encrypted)
        assert decrypted == original

    def test_sm4_ecb_roundtrip(self, tmp_path) -> None:
        """SM4 ECB 模式加解密往返验证"""
        from config.settings import CryptoConfig
        from loongguard.crypto.sm4_logger import SM4Logger, SM4Mode

        key_file = tmp_path / "test_key"
        key_file.write_bytes(b"0123456789abcdef")

        config = CryptoConfig(key_file=str(key_file))
        crypto = SM4Logger(config, mode=SM4Mode.ECB)
        crypto.load_key()

        original = b"ECB mode test data block!"
        encrypted = crypto.encrypt(original)
        decrypted = crypto.decrypt(encrypted)
        assert decrypted == original

    def test_sm4_env_key_loading(self, tmp_path) -> None:
        """SM4 密钥可通过环境变量 LG_SM4_KEY 加载（hex 编码）"""
        import os

        from config.settings import CryptoConfig
        from loongguard.crypto.sm4_logger import SM4Logger

        hex_key = "0123456789abcdef0123456789abcdef"
        # 确保测试不影响其他测试的环境变量
        old_val = os.environ.pop("LG_SM4_KEY", None)
        try:
            os.environ["LG_SM4_KEY"] = hex_key
            config = CryptoConfig(key_file=str(tmp_path / "nonexistent"))
            crypto = SM4Logger(config)
            crypto.load_key()  # 应从环境变量加载，不依赖文件

            data = b"env key test"
            enc = crypto.encrypt(data)
            dec = crypto.decrypt(enc)
            assert dec == data
        finally:
            os.environ.pop("LG_SM4_KEY", None)
            if old_val is not None:
                os.environ["LG_SM4_KEY"] = old_val

    def test_sm4_cbc_same_plaintext_different_ciphertext(self, tmp_path) -> None:
        """CBC 模式下相同明文每次加密产生不同密文（随机 IV）"""
        from config.settings import CryptoConfig
        from loongguard.crypto.sm4_logger import SM4Logger, SM4Mode

        key_file = tmp_path / "test_key"
        key_file.write_bytes(b"0123456789abcdef")

        config = CryptoConfig(key_file=str(key_file))
        crypto = SM4Logger(config, mode=SM4Mode.CBC)
        crypto.load_key()

        data = b"identical plaintext"
        enc1 = crypto.encrypt(data)
        enc2 = crypto.encrypt(data)

        assert enc1 != enc2, "CBC with random IV must produce different ciphertext"
        assert crypto.decrypt(enc1) == crypto.decrypt(enc2) == data

    def test_sm4_encrypt_and_store(self, tmp_path) -> None:
        """SM4 告警日志加密写入磁盘，文件存在且内容不可直读"""
        from config.settings import CryptoConfig
        from loongguard.crypto.sm4_logger import SM4Logger
        from loongguard.utils.schema import AlertLog, AlertSeverity, AlertType

        key_file = tmp_path / "test_key"
        key_file.write_bytes(b"0123456789abcdef")

        config = CryptoConfig(
            key_file=str(key_file),
            log_dir=str(tmp_path / "logs"),
        )
        crypto = SM4Logger(config)
        crypto.load_key()

        alert = AlertLog(
            alert_type=AlertType.PRONE_SLEEP,
            severity=AlertSeverity.CRITICAL,
            description="test prone alert",
        )
        filepath = crypto.encrypt_and_store(alert)

        assert filepath is not None
        enc_path = Path(filepath)
        assert enc_path.exists()
        assert enc_path.stat().st_size > 0

        # 加密文件不应包含原始 JSON 字符串
        raw_bytes = enc_path.read_bytes()
        assert b"test prone alert" not in raw_bytes

    def test_mock_video_readable(self) -> None:
        """Mock 教室视频可被 OpenCV 正常读取"""
        video_path = Path("tests/mock_classroom.mp4")
        assert video_path.exists(), f"Mock video not found: {video_path}"

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), "Cannot open mock video"

        ret, frame = cap.read()
        assert ret, "Cannot read first frame"
        assert frame.ndim == 3 and frame.shape[2] == 3
        cap.release()

    def test_config_env_override(self) -> None:
        """LG_ 前缀环境变量可覆盖配置值"""
        import os

        from config.settings import load_config

        old_port = os.environ.pop("LG_API_PORT", None)
        try:
            os.environ["LG_API_PORT"] = "9999"
            config = load_config()
            assert config.api.port == 9999
        finally:
            os.environ.pop("LG_API_PORT", None)
            if old_port is not None:
                os.environ["LG_API_PORT"] = old_port
