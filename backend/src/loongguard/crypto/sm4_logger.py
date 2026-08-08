"""
SM4 加密日志存储

设计动机：
    告警日志和风险切片图通过龙芯硬件 SM4 加密后才能存储到磁盘。
    视频帧数据本身绝不落盘，仅告警切片（静态图片）经加密后持久化。
    无管理员硬件密钥，任何人拆解闪存均无法读取内容。

    开发环境使用 gmssl 库的软件 SM4 实现，
    生产环境切换到龙芯硬件 SM4 加速（通过 /dev/crypto 或 C 扩展）。

加密模式：
    - SM4-CBC（默认，更安全）：密文格式 [IV(16B) || ciphertext]
    - SM4-ECB（兼容模式）：密文格式 [ciphertext]
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from config import CryptoConfig
from loongguard.utils.schema import AlertLog

logger = logging.getLogger(__name__)

# SM4 分组长度（128-bit）
_SM4_BLOCK_SIZE = 16


class SM4Mode(str, Enum):
    """SM4 加密模式"""
    ECB = "ecb"
    CBC = "cbc"


def _pkcs7_pad(data: bytes, block_size: int = _SM4_BLOCK_SIZE) -> bytes:
    """PKCS#7 填充，使数据长度为 block_size 的整数倍"""
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def _pkcs7_unpad(data: bytes) -> bytes:
    """移除 PKCS#7 填充"""
    if not data:
        raise ValueError("Cannot unpad empty data")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > _SM4_BLOCK_SIZE:
        raise ValueError(f"Invalid PKCS7 padding byte: {pad_len}")
    if data[-pad_len:] != bytes([pad_len] * pad_len):
        raise ValueError("Corrupted PKCS7 padding")
    return data[:-pad_len]


class SM4Logger:
    """
    SM4 加密日志管理器

    使用流程：
        crypto = SM4Logger(config)
        crypto.load_key()  # 自动按 环境变量 > HSM > 文件 优先级加载
        crypto.encrypt_and_store(alert_log)

    设计约束：
        - 默认使用 SM4-CBC 模式（更安全，IV 随机生成）
        - PKCS#7 填充处理非 16 字节对齐的数据
        - 龙芯平台自动检测硬件加速（TODO: 阶段四实现）
        - 密钥管理支持环境变量 / HSM / 文件三种来源
    """

    def __init__(self, config: CryptoConfig, mode: SM4Mode = SM4Mode.CBC) -> None:
        self._config = config
        self._key: bytes = b""
        self._sm4 = None  # 延迟初始化，避免 import 失败影响其他模块
        self._mode = mode

    def load_key(self) -> None:
        """
        加载 SM4 密钥（按优先级：环境变量 > HSM > 文件）

        密钥来源优先级：
            1. 环境变量 LG_SM4_KEY（hex 编码的 16 字节密钥）
            2. 硬件安全模块（HSM）/dev/crypto（LoongArch 生产环境）
            3. 密钥文件（开发/测试环境）

        Raises:
            FileNotFoundError: 所有密钥来源均不可用
            ValueError: 密钥长度不合法
        """
        self._key = self._resolve_key()
        self._init_sm4()
        logger.info("SM4 key loaded (%d bytes, mode=%s)", len(self._key), self._mode.value)

    def _resolve_key(self) -> bytes:
        """
        按优先级解析密钥来源

        设计动机：
            生产环境通过环境变量注入密钥（Factor III: Config），
            避免密钥文件泄露风险。开发环境 fallback 到文件。
        """
        # 来源 1：环境变量（hex 编码，32 个 hex 字符 = 16 字节）
        env_key = os.environ.get("LG_SM4_KEY")
        if env_key:
            try:
                key = bytes.fromhex(env_key)
                if len(key) == 16:
                    logger.debug("SM4 key loaded from environment variable LG_SM4_KEY")
                    return key
                logger.warning("LG_SM4_KEY env var is %d bytes, expected 16", len(key))
            except ValueError:
                logger.warning("LG_SM4_KEY env var is not valid hex, ignoring")

        # 来源 2：HSM / /dev/crypto（LoongArch 生产环境）
        # TODO: 阶段四实现 -- 检测 /dev/crypto 并读取硬件密钥

        # 来源 3：密钥文件（开发/测试环境 fallback）
        key_path = Path(self._config.key_file)
        if not key_path.exists():
            raise FileNotFoundError(
                f"SM4 key not found: no LG_SM4_KEY env var, "
                f"and key file missing: {key_path}. "
                "The key must be injected by the administrator."
            )
        key = key_path.read_bytes()[:16]
        if len(key) < 16:
            raise ValueError(f"SM4 key must be 16 bytes, got {len(key)}")
        logger.debug("SM4 key loaded from file: %s", key_path)
        return key

    def _init_sm4(self) -> None:
        """
        初始化 SM4 加密/解密器

        设计动机：
            优先调用龙芯硬件 SM4 加速指令（/dev/crypto），
            fallback 到 gmssl 软件实现。
        """
        # TODO: 检测是否在龙芯平台，调用硬件 SM4（阶段四实现）
        from gmssl import sm4 as sm4_mod
        self._sm4 = sm4_mod
        logger.debug("Using gmssl software SM4 implementation")

    @property
    def mode(self) -> SM4Mode:
        """当前加密模式"""
        return self._mode

    def encrypt_and_store(self, alert: AlertLog) -> Optional[str]:
        """
        加密告警日志并写入磁盘

        Args:
            alert: 告警日志条目

        Returns:
            加密后的日志文件路径，失败返回 None
        """
        try:
            if self._sm4 is None:
                logger.error("SM4 not initialized, call load_key() first")
                return None

            log_dir = Path(self._config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_data = json.dumps(alert.to_dict(), ensure_ascii=False).encode("utf-8")
            encrypted = self.encrypt(log_data)

            filename = f"alert_{alert.alert_id}.enc"
            filepath = log_dir / filename
            filepath.write_bytes(encrypted)

            logger.info("Alert encrypted and stored: %s", filepath)
            return str(filepath)
        except Exception:
            logger.exception("Failed to encrypt and store alert")
            return None

    def encrypt_slice(self, image_bytes: bytes, alert_id: str) -> Optional[str]:
        """
        加密风险切片图并写入磁盘

        Args:
            image_bytes: JPEG/PNG 编码后的图片字节
            alert_id: 关联的告警 ID

        Returns:
            加密后的切片文件路径
        """
        try:
            if self._sm4 is None:
                logger.error("SM4 not initialized, call load_key() first")
                return None

            slice_dir = Path(self._config.slice_dir)
            slice_dir.mkdir(parents=True, exist_ok=True)

            encrypted = self.encrypt(image_bytes)

            filename = f"slice_{alert_id}.enc"
            filepath = slice_dir / filename
            filepath.write_bytes(encrypted)

            logger.info("Alert slice encrypted: %s", filepath)
            return str(filepath)
        except Exception:
            logger.exception("Failed to encrypt alert slice")
            return None

    def encrypt(self, data: bytes) -> bytes:
        """
        SM4 加密（根据模式选择 ECB 或 CBC）

        CBC 模式密文格式：[IV(16字节) || 密文]
        ECB 模式密文格式：[密文]

        Args:
            data: 待加密的原始数据

        Returns:
            加密后的字节数据
        """
        if self._sm4 is None:
            raise RuntimeError("SM4 not initialized, call load_key() first")

        if self._mode == SM4Mode.CBC:
            return self._sm4_cbc_encrypt(data)
        return self._sm4_ecb_encrypt(data)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        SM4 解密（自动根据模式选择 ECB 或 CBC）

        Args:
            encrypted_data: 加密的数据

        Returns:
            解密后的原始数据

        Raises:
            ValueError: 填充校验失败
        """
        if self._sm4 is None:
            raise RuntimeError("SM4 not initialized, call load_key() first")

        if self._mode == SM4Mode.CBC:
            return self._sm4_cbc_decrypt(encrypted_data)
        return self._sm4_ecb_decrypt(encrypted_data)

    # ── ECB 模式 ──────────────────────────────────────────

    def _sm4_ecb_encrypt(self, data: bytes) -> bytes:
        """SM4-ECB 加密（含 PKCS#7 填充）"""
        padded = _pkcs7_pad(data)
        crypt = self._sm4.CryptSM4()
        crypt.set_key(self._key, self._sm4.SM4_ENCRYPT)
        crypt.padding_mode = 0  # 无内置填充，已手动 PKCS7
        return crypt.crypt_ecb(padded)

    def _sm4_ecb_decrypt(self, encrypted_data: bytes) -> bytes:
        """SM4-ECB 解密"""
        crypt = self._sm4.CryptSM4()
        crypt.set_key(self._key, self._sm4.SM4_DECRYPT)
        crypt.padding_mode = 0
        decrypted = crypt.crypt_ecb(encrypted_data)
        return _pkcs7_unpad(decrypted)

    # ── CBC 模式 ──────────────────────────────────────────

    def _sm4_cbc_encrypt(self, data: bytes) -> bytes:
        """
        SM4-CBC 加密（含 PKCS#7 填充）

        密文格式：[IV(16字节随机) || SM4-CBC密文]

        设计动机：
            CBC 模式使用随机 IV，相同的明文每次加密产生不同的密文，
            防止通过密文模式推断告警类型。
        """
        iv = os.urandom(_SM4_BLOCK_SIZE)
        padded = _pkcs7_pad(data)
        crypt = self._sm4.CryptSM4()
        crypt.set_key(self._key, self._sm4.SM4_ENCRYPT)
        crypt.padding_mode = 0
        ciphertext = crypt.crypt_cbc(iv, padded)
        return iv + ciphertext

    def _sm4_cbc_decrypt(self, encrypted_data: bytes) -> bytes:
        """SM4-CBC 解密，密文格式：[IV(16字节) || SM4-CBC密文]"""
        if len(encrypted_data) < _SM4_BLOCK_SIZE * 2:
            raise ValueError("CBC encrypted data too short (need IV + at least 1 block)")
        iv = encrypted_data[:_SM4_BLOCK_SIZE]
        ciphertext = encrypted_data[_SM4_BLOCK_SIZE:]
        crypt = self._sm4.CryptSM4()
        crypt.set_key(self._key, self._sm4.SM4_DECRYPT)
        crypt.padding_mode = 0
        decrypted = crypt.crypt_cbc(iv, ciphertext)
        return _pkcs7_unpad(decrypted)

    # ── 向后兼容 ──────────────────────────────────────────

    def _sm4_encrypt(self, data: bytes) -> bytes:
        """向后兼容接口，委托给 encrypt()"""
        return self.encrypt(data)
