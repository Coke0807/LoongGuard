"""
SM4 硬件加速模块

设计动机：
    在龙芯平台上利用硬件 SM4 加速指令（/dev/crypto）提升加密性能。
    当硬件不可用时，自动降级到 gmssl 软件实现。

Linux Crypto API 使用方式：
    1. 打开 /dev/crypto 设备
    2. 使用 ioctl 创建加密会话（CIOCGSESSION）
    3. 使用 ioctl 执行加密操作（CIOCCRYPT）
    4. 使用 ioctl 关闭会话（CIOCFSESSION）
    5. 关闭设备文件描述符
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

# 龙芯平台标识
_LOONGARCH_PLATFORMS = {"loongarch64", "loongarch32"}

# 硬件加密设备路径
_CRYPTO_DEVICE_PATHS = [
    "/dev/crypto",
    "/dev/loongarch_crypto",
]

# Linux Crypto API ioctl 命令号
# 参考：include/linux/crypto.h
_CIOCGSESSION = 0xC14063C0  # 创建加密会话
_CIOCFSESSION = 0xC00463D1  # 关闭加密会话
_CIOCCRYPT = 0xC14863C2     # 执行加密操作

# SM4 算法名称（Linux Crypto API）
_SM4_ALGORITHM_NAME = b"sm4"


def is_loongarch_platform() -> bool:
    """
    检测是否在龙芯平台

    Returns:
        True if running on LoongArch platform
    """
    machine = platform.machine().lower()
    return machine in _LOONGARCH_PLATFORMS


def detect_hardware_sm4() -> bool:
    """
    检测硬件 SM4 支持

    Returns:
        True if hardware SM4 is available
    """
    if not is_loongarch_platform():
        logger.debug("Not on LoongArch platform, hardware SM4 unavailable")
        return False

    # 检查加密设备文件
    for device_path in _CRYPTO_DEVICE_PATHS:
        if Path(device_path).exists():
            logger.info("Found crypto device: %s", device_path)
            return True

    logger.debug("No crypto device found on LoongArch platform")
    return False


# ── C 结构体定义 ──────────────────────────────────────────────

class CryptoSession(ctypes.Structure):
    """Linux Crypto API 会话结构体"""
    _fields_ = [
        ("cipher", ctypes.c_char * 64),  # 算法名称
        ("key", ctypes.c_char_p),        # 密钥指针
        ("keylen", ctypes.c_uint32),     # 密钥长度
        ("flags", ctypes.c_uint32),      # 标志
    ]


class CryptoOperation(ctypes.Structure):
    """Linux Crypto API 操作结构体"""
    _fields_ = [
        ("src", ctypes.c_char_p),        # 源数据指针
        ("dst", ctypes.c_char_p),        # 目标数据指针
        ("len", ctypes.c_uint32),        # 数据长度
        ("iv", ctypes.c_char * 16),      # 初始化向量（CBC 模式）
        ("op", ctypes.c_uint32),         # 操作类型（0=加密，1=解密）
        ("flags", ctypes.c_uint32),      # 标志
    ]


class HardwareSM4:
    """
    硬件 SM4 加密器

    使用龙芯平台的 /dev/crypto 设备进行 SM4 加密/解密。
    当硬件不可用时，抛出异常由调用方降级到软件实现。

    使用方式：
        try:
            encryptor = HardwareSM4(key)
            encrypted = encryptor.encrypt(data)
            decrypted = encryptor.decrypt(encrypted)
        except RuntimeError:
            # 降级到软件实现
            pass
    """

    def __init__(self, key: bytes) -> None:
        """
        初始化硬件 SM4 加密器

        Args:
            key: 16 字节 SM4 密钥

        Raises:
            RuntimeError: 硬件 SM4 不可用
            ValueError: 密钥长度不合法
        """
        if len(key) != 16:
            raise ValueError(f"SM4 key must be 16 bytes, got {len(key)}")

        self._key = key
        self._device_path: str | None = None
        self._fd: int = -1
        self._session_id: int = -1

        # 检测硬件支持
        if not detect_hardware_sm4():
            raise RuntimeError("Hardware SM4 not available on this platform")

        # 尝试打开加密设备
        for device_path in _CRYPTO_DEVICE_PATHS:
            try:
                self._fd = os.open(device_path, os.O_RDWR)
                self._device_path = device_path
                logger.info("Opened crypto device: %s", device_path)
                break
            except OSError as e:
                logger.debug("Failed to open %s: %s", device_path, e)
                continue

        if self._fd < 0:
            raise RuntimeError("Failed to open any crypto device")

        # 创建加密会话
        self._session_id = self._create_session()

    def __del__(self) -> None:
        """析构函数：关闭设备文件描述符"""
        self.close()

    def close(self) -> None:
        """关闭加密设备"""
        if self._session_id >= 0:
            self._close_session()
            self._session_id = -1

        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1

    def _create_session(self) -> int:
        """
        创建加密会话

        Returns:
            会话 ID

        Raises:
            RuntimeError: 创建会话失败
        """
        # 准备会话结构体
        session = CryptoSession()
        session.cipher = _SM4_ALGORITHM_NAME
        session.key = ctypes.cast(self._key, ctypes.c_char_p)
        session.keylen = len(self._key)
        session.flags = 0

        # 调用 ioctl 创建会话
        try:
            import fcntl
            session_id = fcntl.ioctl(self._fd, _CIOCGSESSION, session)
            logger.debug("Created crypto session: %d", session_id)
            return session_id
        except Exception as e:
            raise RuntimeError(f"Failed to create crypto session: {e}") from e

    def _close_session(self) -> None:
        """关闭加密会话"""
        try:
            import fcntl
            fcntl.ioctl(self._fd, _CIOCFSESSION, self._session_id)
            logger.debug("Closed crypto session: %d", self._session_id)
        except Exception as e:
            logger.warning("Failed to close crypto session: %s", e)

    def _crypt(self, data: bytes, encrypt: bool = True) -> bytes:
        """
        执行加密/解密操作

        Args:
            data: 输入数据
            encrypt: True=加密，False=解密

        Returns:
            输出数据

        Raises:
            RuntimeError: 操作失败
        """
        if self._fd < 0:
            raise RuntimeError("Crypto device not opened")

        # 准备操作结构体
        operation = CryptoOperation()
        operation.src = ctypes.cast(data, ctypes.c_char_p)
        operation.dst = ctypes.create_string_buffer(len(data))
        operation.len = len(data)
        operation.iv = b'\x00' * 16  # ECB 模式不需要 IV
        operation.op = 0 if encrypt else 1
        operation.flags = 0

        # 调用 ioctl 执行加密
        try:
            import fcntl
            fcntl.ioctl(self._fd, _CIOCCRYPT, operation)
            return bytes(operation.dst)
        except Exception as e:
            raise RuntimeError(f"Crypto operation failed: {e}") from e

    def encrypt(self, data: bytes) -> bytes:
        """
        SM4 加密（ECB 模式）

        Args:
            data: 待加密数据（必须是 16 字节的倍数）

        Returns:
            加密后的数据

        Raises:
            RuntimeError: 加密失败
            ValueError: 数据长度不合法
        """
        if len(data) % 16 != 0:
            raise ValueError(f"Data length must be multiple of 16, got {len(data)}")

        return self._crypt(data, encrypt=True)

    def decrypt(self, data: bytes) -> bytes:
        """
        SM4 解密（ECB 模式）

        Args:
            data: 待解密数据（必须是 16 字节的倍数）

        Returns:
            解密后的数据

        Raises:
            RuntimeError: 解密失败
            ValueError: 数据长度不合法
        """
        if len(data) % 16 != 0:
            raise ValueError(f"Data length must be multiple of 16, got {len(data)}")

        return self._crypt(data, encrypt=False)


def create_sm4_encryptor(
    key: bytes,
    prefer_hardware: bool = True,
) -> object:
    """
    创建 SM4 加密器工厂函数

    Args:
        key: 16 字节 SM4 密钥
        prefer_hardware: 是否优先使用硬件加速

    Returns:
        SM4 加密器实例（硬件或软件）

    Raises:
        ValueError: 密钥长度不合法
    """
    if len(key) != 16:
        raise ValueError(f"SM4 key must be 16 bytes, got {len(key)}")

    # 尝试硬件加速
    if prefer_hardware and detect_hardware_sm4():
        try:
            return HardwareSM4(key)
        except Exception as e:
            logger.warning("Hardware SM4 initialization failed: %s", e)

    # 降级到软件实现
    logger.info("Using software SM4 implementation (gmssl)")
    from gmssl import sm4 as sm4_mod

    crypt = sm4_mod.CryptSM4()
    crypt.set_key(key, sm4_mod.SM4_ENCRYPT)
    return crypt
