"""
SM4 加密日志模块单元测试

覆盖：
    - 密钥加载（正常 / 缺失 / 长度不足）
    - SM4 加密输出正确性（16 字节对齐、密文不等于明文）
    - encrypt -> decrypt 往返一致性
    - encrypt_and_store 文件写入与内容验证
    - encrypt_slice 图片加密写入
    - SM4 未初始化时的降级行为
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import CryptoConfig
from loongguard.crypto.sm4_logger import SM4Logger, _pkcs7_pad, _pkcs7_unpad
from loongguard.utils.schema import AlertLog, AlertSeverity, AlertType

# ── 常量 ──────────────────────────────────────────────────────

_SM4_KEY_LEN = 16


# ── Fixtures ──────────────────────────────────────────────────


def _make_key_file(tmp_path: Path, length: int = _SM4_KEY_LEN) -> Path:
    """在 tmp_path 下创建指定长度的密钥文件并返回路径"""
    key_path = tmp_path / ".sm4_key"
    key_path.write_bytes(b"\x01\x02\x03\x04\x05\x06\x07\x08" * (length // 8 or 1))
    return key_path


@pytest.fixture
def crypto_config(tmp_path: Path) -> CryptoConfig:
    """返回指向临时目录的 CryptoConfig，密钥文件在 tmp_path 中"""
    key_path = _make_key_file(tmp_path)
    return CryptoConfig(
        key_file=str(key_path),
        log_dir=str(tmp_path / "logs" / "encrypted"),
        slice_dir=str(tmp_path / "logs" / "slices"),
    )


@pytest.fixture
def sm4_instance(crypto_config: CryptoConfig) -> SM4Logger:
    """返回已完成 load_key 的 SM4Logger 实例"""
    instance = SM4Logger(crypto_config)
    instance.load_key()
    return instance


def _make_alert(alert_id: str = "test_001") -> AlertLog:
    """构造一个最小化的测试告警"""
    return AlertLog(
        alert_id=alert_id,
        alert_type=AlertType.DANGEROUS_OBJECT,
        severity=AlertSeverity.HIGH,
        description="测试告警：磁力珠检测",
    )


# ── 密钥加载测试 ─────────────────────────────────────────────


class TestLoadKey:
    """load_key() 密钥文件读取测试"""

    @pytest.fixture(autouse=True)
    def _clean_lg_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """自动清理 LG_SM4_KEY 环境变量，避免环境中的密钥干扰文件路径测试"""
        monkeypatch.delenv("LG_SM4_KEY", raising=False)

    def test_load_key_reads_16_bytes(self, tmp_path: Path) -> None:
        """load_key 应成功读取 16 字节密钥"""
        key_path = _make_key_file(tmp_path)
        config = CryptoConfig(
            key_file=str(key_path),
            log_dir=str(tmp_path / "logs"),
            slice_dir=str(tmp_path / "slices"),
        )
        instance = SM4Logger(config)
        instance.load_key()

        assert instance._key == key_path.read_bytes()[:16]
        assert len(instance._key) == _SM4_KEY_LEN
        assert instance._sm4 is not None

    def test_load_key_raises_on_missing_file(self, tmp_path: Path) -> None:
        """密钥文件不存在且无环境变量时应抛出 FileNotFoundError"""
        config = CryptoConfig(
            key_file=str(tmp_path / "nonexistent_key"),
            log_dir=str(tmp_path / "logs"),
            slice_dir=str(tmp_path / "slices"),
        )
        instance = SM4Logger(config)
        with pytest.raises(FileNotFoundError, match="SM4 key not found"):
            instance.load_key()

    def test_load_key_raises_on_short_key(self, tmp_path: Path) -> None:
        """密钥文件不足 16 字节时应抛出 ValueError"""
        short_key_path = tmp_path / ".sm4_key_short"
        short_key_path.write_bytes(b"\xAA\xBB\xCC")  # 仅 3 字节

        config = CryptoConfig(
            key_file=str(short_key_path),
            log_dir=str(tmp_path / "logs"),
            slice_dir=str(tmp_path / "slices"),
        )
        instance = SM4Logger(config)
        with pytest.raises(ValueError, match="SM4 key must be 16 bytes"):
            instance.load_key()


# ── 加密输出特性测试 ─────────────────────────────────────────


class TestSM4Encrypt:
    """_sm4_encrypt() 加密输出测试"""

    def test_encrypted_differs_from_plaintext(
        self, sm4_instance: SM4Logger
    ) -> None:
        """密文不应与明文相同"""
        plaintext = b"Hello LoongGuard SM4 test data!"
        encrypted = sm4_instance._sm4_encrypt(plaintext)
        assert encrypted != plaintext

    def test_encrypted_length_is_block_aligned(
        self, sm4_instance: SM4Logger
    ) -> None:
        """密文长度必须是 16 字节的整数倍（SM4 分组大小）"""
        for size in (1, 15, 16, 17, 31, 32, 100):
            data = b"\xAA" * size
            encrypted = sm4_instance._sm4_encrypt(data)
            assert len(encrypted) % 16 == 0, (
                f"输入 {size} 字节，密文 {len(encrypted)} 字节未对齐到 16"
            )

    def test_different_inputs_produce_different_outputs(
        self, sm4_instance: SM4Logger
    ) -> None:
        """不同明文应产生不同密文"""
        enc_a = sm4_instance._sm4_encrypt(b"message_A")
        enc_b = sm4_instance._sm4_encrypt(b"message_B")
        assert enc_a != enc_b


# ── 往返加解密测试 ───────────────────────────────────────────


class TestRoundtrip:
    """encrypt -> decrypt 往返一致性"""

    def test_decrypt_after_encrypt_returns_original(
        self, sm4_instance: SM4Logger
    ) -> None:
        """加密再解密应得到原始数据"""
        original = b"LoongGuard roundtrip test 12345"
        encrypted = sm4_instance._sm4_encrypt(original)
        decrypted = sm4_instance.decrypt(encrypted)
        assert decrypted == original

    def test_roundtrip_with_exact_block_size(
        self, sm4_instance: SM4Logger
    ) -> None:
        """恰好 16 字节（一个 SM4 分组）的往返测试"""
        original = b"A" * 16
        encrypted = sm4_instance._sm4_encrypt(original)
        decrypted = sm4_instance.decrypt(encrypted)
        assert decrypted == original

    def test_roundtrip_with_utf8_json(self, sm4_instance: SM4Logger) -> None:
        """JSON 中文内容的往返测试"""
        payload = json.dumps(
            {"desc": "磁力珠检测告警", "severity": "high"},
            ensure_ascii=False,
        ).encode("utf-8")
        encrypted = sm4_instance._sm4_encrypt(payload)
        decrypted = sm4_instance.decrypt(encrypted)
        assert decrypted == payload


# ── PKCS#7 填充工具函数测试 ──────────────────────────────────


class TestPKCS7:
    """PKCS#7 填充/去填充正确性"""

    def test_pad_unpad_roundtrip(self) -> None:
        """填充再去填充应得到原始数据"""
        for size in (0, 1, 15, 16, 17, 100):
            data = b"\x42" * size
            padded = _pkcs7_pad(data)
            assert len(padded) % 16 == 0
            unpadded = _pkcs7_unpad(padded)
            assert unpadded == data

    def test_unpad_raises_on_empty(self) -> None:
        """空数据去填充应抛出 ValueError"""
        with pytest.raises(ValueError, match="Cannot unpad empty"):
            _pkcs7_unpad(b"")

    def test_unpad_raises_on_corrupted_padding(self) -> None:
        """损坏的填充字节应抛出 ValueError"""
        # 末尾字节为 0x04，但倒数 4 字节不全是 0x04
        corrupted = b"\x00" * 15 + b"\x04"
        with pytest.raises(ValueError, match="Corrupted PKCS7 padding"):
            _pkcs7_unpad(corrupted)


# ── encrypt_and_store 文件写入测试 ───────────────────────────


class TestEncryptAndStore:
    """encrypt_and_store 磁盘写入测试"""

    def test_creates_enc_file(self, sm4_instance: SM4Logger) -> None:
        """应创建 .enc 后缀的加密文件"""
        alert = _make_alert("file_test_001")
        filepath = sm4_instance.encrypt_and_store(alert)

        assert filepath is not None
        assert filepath.endswith(".enc")
        assert Path(filepath).exists()

    def test_enc_file_contains_encrypted_not_plaintext(
        self, sm4_instance: SM4Logger
    ) -> None:
        """文件内容应为密文，不应包含明文 JSON 字符串"""
        alert = _make_alert("content_test_002")
        filepath = sm4_instance.encrypt_and_store(alert)

        assert filepath is not None
        raw = Path(filepath).read_bytes()
        # 明文 JSON 包含 alert_id，密文中不应出现
        plaintext_json = json.dumps(alert.to_dict(), ensure_ascii=False).encode("utf-8")
        assert raw != plaintext_json
        # 密文中不应直接包含描述文本
        assert b"\xe7\xa3\x81\xe5\x8a\x9b\xe7\x8f\xa0" not in raw  # "磁力珠" UTF-8

    def test_returns_none_when_sm4_not_initialized(
        self, tmp_path: Path
    ) -> None:
        """SM4 未初始化（未调用 load_key）时应返回 None"""
        config = CryptoConfig(
            key_file=str(tmp_path / "missing_key"),
            log_dir=str(tmp_path / "logs"),
            slice_dir=str(tmp_path / "slices"),
        )
        instance = SM4Logger(config)
        # 故意不调用 load_key
        alert = _make_alert("no_key_003")
        result = instance.encrypt_and_store(alert)
        assert result is None

    def test_decrypt_stored_file_recovers_alert(
        self, sm4_instance: SM4Logger
    ) -> None:
        """从磁盘读取加密文件并解密，应能还原原始告警 JSON"""
        alert = _make_alert("roundtrip_disk_004")
        filepath = sm4_instance.encrypt_and_store(alert)

        assert filepath is not None
        encrypted_bytes = Path(filepath).read_bytes()
        decrypted_bytes = sm4_instance.decrypt(encrypted_bytes)
        recovered = json.loads(decrypted_bytes.decode("utf-8"))

        assert recovered["alert_id"] == "roundtrip_disk_004"
        assert recovered["description"] == "测试告警：磁力珠检测"


# ── encrypt_slice 图片加密测试 ───────────────────────────────


class TestEncryptSlice:
    """encrypt_slice 风险切片图加密测试"""

    def test_creates_enc_file_for_image(
        self, sm4_instance: SM4Logger
    ) -> None:
        """应为图片字节创建 .enc 文件"""
        fake_jpeg = b"\xFF\xD8\xFF\xE0" + b"\x00" * 500  # 最小 JPEG 头 + 填充
        filepath = sm4_instance.encrypt_slice(fake_jpeg, "slice_001")

        assert filepath is not None
        assert "slice_001" in filepath
        assert filepath.endswith(".enc")
        assert Path(filepath).exists()

    def test_encrypted_slice_differs_from_input(
        self, sm4_instance: SM4Logger
    ) -> None:
        """加密后的切片文件不应与原始字节相同"""
        image_data = bytes(range(256))  # 256 字节测试数据
        filepath = sm4_instance.encrypt_slice(image_data, "slice_002")

        assert filepath is not None
        stored = Path(filepath).read_bytes()
        assert stored != image_data

    def test_slice_returns_none_when_not_initialized(
        self, tmp_path: Path
    ) -> None:
        """SM4 未初始化时 encrypt_slice 应返回 None"""
        config = CryptoConfig(
            key_file=str(tmp_path / "missing_key"),
            log_dir=str(tmp_path / "logs"),
            slice_dir=str(tmp_path / "slices"),
        )
        instance = SM4Logger(config)
        result = instance.encrypt_slice(b"\x00" * 50, "slice_nokey")
        assert result is None
