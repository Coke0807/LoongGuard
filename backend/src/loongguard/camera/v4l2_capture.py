"""
V4L2 摄像头采集模块

设计动机：
    封装 Video4Linux2 接口，提供面向 Python 的帧采集抽象。
    底层使用 C 扩展 (v4l2_ext.so) 调用 ioctl，Python 侧仅负责
    缓冲区管理和帧生命周期管控，确保视频帧数据仅驻留内存、断电即失。

    跨平台降级策略（对应 CLAUDE.md 约束 #8）：
    - Linux + /dev/video* 设备路径：优先使用 V4L2 C 扩展 (v4l2_ext.so)，
      通过 ctypes 桥接调用，自动编译（需 gcc）；加载失败则降级到 OpenCV。
    - Linux + 视频文件路径：使用 OpenCV VideoCapture
    - Windows / QEMU / CI 环境：自动降级到 OpenCV，优先读取 mock 视频文件，
      回退到 USB 摄像头，无需编译 C/C++ 即可跑通算法逻辑。
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

# OpenCV 作为跨平台 fallback 依赖
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from config import CameraConfig

logger = logging.getLogger(__name__)

# 默认 mock 视频路径（相对于项目根目录，开发时由 create_test_assets.py 生成）
_DEFAULT_MOCK_VIDEO = Path("tests") / "mock_classroom.mp4"

# ── V4L2 C 扩展加载（ctypes 桥接）──────────────────────────────
# 设计动机：
#   v4l2_ext.c 编译为 .so 共享库后，通过 ctypes 在 Python 中调用。
#   在非 Linux 环境或 .so 不存在时，_v4l2_lib 为 None，
#   V4L2 功能自动降级为 OpenCV。
# ────────────────────────────────────────────────────────────────

_V4L2_LIB_PATH = Path(__file__).parent / "v4l2_ext.so"
_v4l2_lib: ctypes.CDLL | None = None


def _try_compile_v4l2_ext() -> bool:
    """
    尝试自动编译 v4l2_ext.c -> v4l2_ext.so

    仅在 Linux 上执行，需要 gcc 和 linux-headers。
    编译失败不阻塞程序，仅记录警告后 fallback 到 OpenCV。
    """
    if sys.platform != "linux":
        return False

    src = Path(__file__).parent / "v4l2_ext.c"
    if not src.exists():
        return False

    logger.info("正在编译 V4L2 C 扩展: %s -> .so", src.name)
    try:
        result = subprocess.run(
            ["gcc", "-Wall", "-O2", "-fPIC", "-shared",
             "-o", str(_V4L2_LIB_PATH), str(src)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logger.info("V4L2 C 扩展编译成功")
            return True
        else:
            logger.warning("V4L2 C 扩展编译失败:\n%s", result.stderr)
            return False
    except FileNotFoundError:
        logger.warning("gcc 未找到，跳过 V4L2 C 扩展编译")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("V4L2 C 扩展编译超时")
        return False


def _load_v4l2_lib() -> ctypes.CDLL | None:
    """
    加载 v4l2_ext.so 并声明 C 函数签名。

    返回 None 表示加载失败，调用方应 fallback 到 OpenCV。
    """
    if sys.platform != "linux":
        return None

    # .so 不存在时尝试自动编译
    if not _V4L2_LIB_PATH.exists():
        if not _try_compile_v4l2_ext():
            return None

    try:
        lib = ctypes.CDLL(str(_V4L2_LIB_PATH))
    except OSError as e:
        logger.warning("无法加载 v4l2_ext.so: %s", e)
        return None

    # 声明函数签名，确保 ctypes 正确处理参数类型转换
    lib.v4l2_open.argtypes = [
        ctypes.c_char_p,      # device_path
        ctypes.c_int,         # width
        ctypes.c_int,         # height
        ctypes.c_int,         # pixel_format
    ]
    lib.v4l2_open.restype = ctypes.c_int

    lib.v4l2_read_frame.argtypes = [
        ctypes.c_int,                             # fd
        ctypes.POINTER(ctypes.c_uint8),           # buffer
        ctypes.c_size_t,                          # buffer_size
        ctypes.POINTER(ctypes.c_int),             # out_width
        ctypes.POINTER(ctypes.c_int),             # out_height
    ]
    lib.v4l2_read_frame.restype = ctypes.c_int

    lib.v4l2_close.argtypes = [ctypes.c_int]
    lib.v4l2_close.restype = None

    lib.v4l2_get_resolution.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.v4l2_get_resolution.restype = ctypes.c_int

    logger.info("V4L2 C 扩展加载成功: %s", _V4L2_LIB_PATH)
    return lib


class PixelFormat(IntEnum):
    """V4L2 像素格式常量"""

    YUYV = 0x56595559
    MJPG = 0x47504A4D
    NV12 = 0x3231564E


@dataclass
class Frame:
    """
    单帧数据

    Attributes:
        data: RGB numpy 数组 (H, W, 3)，dtype=uint8
        timestamp: 帧采集时间戳（单调时钟，秒）
        frame_id: 帧序号
    """

    data: np.ndarray
    timestamp: float
    frame_id: int

    def release(self) -> None:
        """释放帧数据内存（显式置空，帮助 GC 及时回收大数组）"""
        self.data = np.empty(0)


class V4L2Capture:
    """
    跨平台摄像头采集器

    使用流程：
        cap = V4L2Capture(config)
        cap.open()             # 自动选择 V4L2 或 OpenCV 后端
        frame = cap.read()     # 返回 Frame 对象（RGB 格式）
        frame.release()        # 用完立即释放
        cap.close()

    后端选择逻辑：
        1. config.device 为 /dev/video* 且在 Linux 上 -> V4L2 C 扩展
        2. config.device 为视频文件路径 -> OpenCV
        3. Windows 环境 -> OpenCV（自动搜索 mock 视频 / USB 摄像头）

    设计约束：
        - 视频帧数据仅在内存中，绝不落盘
        - 环形缓冲区循环使用，减少内存分配开销（V4L2 模式）
        - 视频文件自动循环播放（开发/测试便利）
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._device_fd: int = -1
        self._frame_count: int = 0
        self._opened: bool = False
        self._lib: ctypes.CDLL | None = None

        # OpenCV fallback 后端状态
        self._use_opencv: bool = False
        self._cap: object | None = None  # cv2.VideoCapture 实例
        self._source_is_file: bool = False
        self._file_total_frames: int = 0  # 视频文件总帧数，用于循环控制
        self._file_loops: int = 0         # 已循环播放次数

    def open(self, source: str | None = None) -> None:
        """
        打开摄像头设备并初始化采集流程

        Args:
            source: 视频源路径。None 时使用 config.device，
                    Windows 上自动尝试 mock 视频和 USB 摄像头。

        Raises:
            RuntimeError: 无法打开任何视频源
        """
        if self._opened:
            return

        device = source or self._config.device

        logger.info(
            "Opening camera device=%s resolution=%dx%d fps=%d format=%s",
            device,
            self._config.width,
            self._config.height,
            self._config.fps,
            self._config.pixel_format,
        )

        # 路径 1：Linux V4L2 设备
        if _is_v4l2_device(device):
            self._open_v4l2(device)
            return

        # 路径 2：指定的视频文件或摄像头索引，使用 OpenCV
        if device and self._try_open_opencv(device):
            return

        # 显式传入的 source 打开失败，不应 fallback 到其他设备
        if source is not None:
            raise RuntimeError(
                f"无法打开指定视频源: {source}"
            )

        # 路径 3：自动降级（mock 视频 -> USB 摄像头）
        # 设计动机：在无硬件摄像头环境（Windows / QEMU / CI）下
        # 自动降级到 mock 视频文件，确保算法流程可验证
        self._open_fallback()
        return

    def read(self) -> Frame | None:
        """
        读取一帧

        Returns:
            Frame 对象（data 为 RGB 格式），或 None（读取失败/超时）

        设计动机：
            返回 None 而非抛异常，因为摄像头帧丢失在边缘设备上是
            常见场景，调用方可选择跳过或重试。
        """
        if not self._opened:
            raise RuntimeError("Camera not opened, call open() first")

        if self._use_opencv:
            return self._read_opencv()

        return self._read_v4l2()

    def close(self) -> None:
        """关闭摄像头，释放所有缓冲区"""
        if not self._opened:
            return

        logger.info("Closing camera (frames captured: %d)", self._frame_count)

        if self._use_opencv:
            if self._cap is not None:
                try:
                    self._cap.release()  # type: ignore[union-attr]
                except Exception:
                    pass
                self._cap = None
        else:
            # 释放 V4L2 设备文件描述符
            if self._lib is not None and self._device_fd >= 0:
                try:
                    self._lib.v4l2_close(self._device_fd)
                except Exception:
                    pass
                self._device_fd = -1

        self._opened = False
        self._frame_count = 0
        self._file_loops = 0
        self._file_total_frames = 0

    @property
    def is_opened(self) -> bool:
        return self._opened

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __enter__(self) -> V4L2Capture:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── OpenCV fallback 私有方法 ────────────────────────────────

    def _try_open_opencv(self, source: str) -> bool:
        """尝试用 OpenCV 打开指定源，成功返回 True"""
        if cv2 is None:
            logger.warning("cv2 not available, cannot use OpenCV fallback")
            return False

        try:
            logger.debug("Trying to open video source: %s", source)

            # 尝试将数字字符串转换为整数（OpenCV 需要整数索引）
            try:
                source_int = int(source)
                logger.debug("Converting source to integer: %d", source_int)
                cap = cv2.VideoCapture(source_int)
            except ValueError:
                # 非数字字符串，作为文件路径处理
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                logger.debug("Failed to open source: %s (isOpened=False)", source)
                cap.release()
                return False

            # 测试读取一帧
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                logger.debug("Failed to read frame from source: %s", source)
                cap.release()
                return False

            # 回退到开始位置（仅对文件有效）
            if Path(source).is_file():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self._cap = cap
            self._use_opencv = True
            self._source_is_file = Path(source).is_file()
            self._opened = True
            self._file_loops = 0

            # 记录视频文件总帧数，用于控制循环播放上限
            if self._source_is_file:
                self._file_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self._file_total_frames = 0

            logger.info("Camera opened via OpenCV: %s (file=%s, resolution=%dx%d, total_frames=%d)",
                       source, self._source_is_file,
                       test_frame.shape[1], test_frame.shape[0],
                       self._file_total_frames)
            return True
        except Exception as e:
            logger.debug("Failed to open %s via OpenCV: %s", source, str(e))
            return False

    def _open_fallback(self) -> None:
        """
        自动降级探测视频源（跨平台）

        设计动机：在无硬件摄像头环境（Windows / QEMU / CI）下
        自动降级到 mock 视频文件，确保算法流程可验证。

        优先级：mock 视频文件 -> USB 摄像头
        """
        # 尝试 mock 视频
        if _DEFAULT_MOCK_VIDEO.exists():
            if self._try_open_opencv(str(_DEFAULT_MOCK_VIDEO)):
                return
            logger.warning("Mock video exists but cannot open: %s", _DEFAULT_MOCK_VIDEO)
        else:
            logger.info("Mock video not found: %s", _DEFAULT_MOCK_VIDEO)

        # 尝试 USB 摄像头（索引 0-3）
        for idx in range(4):
            if self._try_open_opencv(str(idx)):
                return

        raise RuntimeError(
            "无可用视频源：mock 视频文件不存在，且未检测到 USB 摄像头。"
            "请将测试视频放置到 tests/mock_classroom.mp4"
        )

    def _read_opencv(self) -> Frame | None:
        """通过 OpenCV 读取一帧，处理循环播放和颜色空间转换"""
        if self._cap is None:
            return None

        try:
            ret, bgr_frame = self._cap.read()  # type: ignore[union-attr]
        except Exception:
            logger.exception("OpenCV read failed")
            return None

        if not ret or bgr_frame is None:
            # 视频文件结束时自动循环播放（仅允许循环一次，避免测试/消费时死循环）
            if self._source_is_file and self._file_loops < 1:
                try:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore[union-attr]
                    ret, bgr_frame = self._cap.read()  # type: ignore[union-attr]
                    self._file_loops += 1
                except Exception:
                    logger.exception("OpenCV loop restart failed")
                    return None
                if not ret or bgr_frame is None:
                    return None
            else:
                return None

        self._frame_count += 1

        # OpenCV 输出 BGR，下游模块（YOLO26、MoveNet、ROI）均期望 RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        return Frame(
            data=rgb_frame,
            timestamp=self._frame_count / max(self._config.fps, 1),
            frame_id=self._frame_count,
        )

    # ── V4L2 C 扩展私有方法 ────────────────────────────────────

    def _open_v4l2(self, device: str) -> None:
        """
        打开 V4L2 设备

        优先尝试加载 C 扩展 v4l2_ext.so（零拷贝 DMA，性能最优），
        若 .so 不存在或加载失败则降级到 OpenCV VideoCapture 直接读取 /dev/videoN。
        降级后 self._use_opencv=True，read() 走 OpenCV 路径。
        """
        # 优先：加载 V4L2 C 扩展
        global _v4l2_lib
        if _v4l2_lib is None:
            _v4l2_lib = _load_v4l2_lib()

        if _v4l2_lib is not None:
            # 像素格式：v4l2_ext.c 内部统一输出 BGR24 (0x3352424)
            V4L2_PIX_FMT_BGR24 = 0x3352424
            fd = _v4l2_lib.v4l2_open(
                device.encode(),
                self._config.width,
                self._config.height,
                V4L2_PIX_FMT_BGR24,
            )
            if fd < 0:
                logger.warning(
                    "v4l2_open 失败 (fd=%d), 降级到 OpenCV", fd,
                )
            else:
                self._lib = _v4l2_lib
                self._device_fd = fd
                self._use_opencv = False
                self._opened = True

                # 获取驱动协商后的实际分辨率
                w = ctypes.c_int()
                h = ctypes.c_int()
                if _v4l2_lib.v4l2_get_resolution(fd, w, h) == 0:
                    logger.info(
                        "V4L2 设备打开成功: %s (实际分辨率: %dx%d, fd=%d)",
                        device, w.value, h.value, fd,
                    )
                else:
                    logger.info(
                        "V4L2 设备打开成功: %s (fd=%d)", device, fd,
                    )
                return

        # 降级：使用 OpenCV 直接读取 V4L2 设备
        # OpenCV 内部通过 V4L2 后端访问 /dev/videoN，支持设置分辨率和帧率
        if cv2 is None:
            raise RuntimeError(
                "Neither v4l2_ext.so nor OpenCV is available. "
                "Install opencv-python or compile v4l2_ext.so."
            )

        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open V4L2 device: {device}")

        # 尝试设置分辨率和帧率（V4L2 驱动可能不支持所有组合）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        # 读取实际生效的分辨率（驱动可能协商为不同值）
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "V4L2 设备通过 OpenCV 打开: %s (实际: %dx%d@%.1ffps, 降级模式)",
            device, actual_w, actual_h, actual_fps,
        )

        self._cap = cap
        self._use_opencv = True
        self._source_is_file = False
        self._opened = True

    def _read_v4l2(self) -> Frame | None:
        """通过 V4L2 C 扩展读取一帧，失败时自动降级到 OpenCV"""
        if self._lib is None or self._device_fd < 0:
            return None

        # 分配 BGR24 接收缓冲区
        buf_size = self._config.width * self._config.height * 3
        buf = np.empty(buf_size, dtype=np.uint8)

        # 获取 numpy 缓冲区的原始指针，传给 C 函数直接写入
        buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        out_w = ctypes.c_int()
        out_h = ctypes.c_int()

        ret = self._lib.v4l2_read_frame(
            self._device_fd, buf_ptr, buf_size, out_w, out_h,
        )

        if ret != 0:
            # C 扩展读帧失败，降级到 OpenCV（如果可用）
            if self._cap is None and cv2 is not None:
                logger.warning("V4L2 read_frame 失败，降级到 OpenCV")
                self._lib.v4l2_close(self._device_fd)
                self._device_fd = -1
                fallback_cap = cv2.VideoCapture(self._config.device)
                if fallback_cap.isOpened():
                    self._cap = fallback_cap
                    self._use_opencv = True
                    return self._read_opencv()
            return None

        self._frame_count += 1

        # 将 BGR 缓冲区 reshape 为 (H, W, 3)，再转换为 RGB
        actual_w = out_w.value
        actual_h = out_h.value
        frame_bgr = buf[:actual_w * actual_h * 3].reshape(actual_h, actual_w, 3)
        frame_rgb = frame_bgr[..., ::-1].copy()  # BGR -> RGB

        return Frame(
            data=frame_rgb,
            timestamp=self._frame_count / max(self._config.fps, 1),
            frame_id=self._frame_count,
        )


def _is_v4l2_device(path: str) -> bool:
    """判断路径是否为 Linux V4L2 设备（/dev/video*）"""
    return sys.platform == "linux" and path.startswith("/dev/video")
