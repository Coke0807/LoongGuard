"""
帧缓冲管理

设计动机：
    管理帧的环形缓冲，供运动检测模块取"当前帧 vs 上一帧"的差分比较。
    同时在 debug 模式下保留最近 N 帧用于问题排查。
    帧数据仅驻留内存，绝不落盘。
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from .v4l2_capture import Frame


class FrameBuffer:
    """
    线程安全的帧缓冲区

    - prev_frame: 运动检测需要的上一帧引用
    - debug_frames: debug 模式下保留最近 N 帧
    """

    def __init__(self, max_debug_frames: int = 0) -> None:
        self._lock = threading.Lock()
        self._prev_frame: Frame | None = None
        self._debug_mode = max_debug_frames > 0
        self._debug_frames: deque[Frame] = deque(
            maxlen=max_debug_frames if max_debug_frames > 0 else 1
        )

    def push(self, frame: Frame) -> None:
        """
        推入新帧，更新 prev_frame 引用

        线程安全：检测线程和采集线程可并发调用。
        """
        with self._lock:
            old = self._prev_frame
            self._prev_frame = frame
            if self._debug_mode:
                self._debug_frames.append(frame)
            # 非 debug 模式下立即释放旧帧，防止内存泄漏
            if old is not None and not self._debug_mode:
                old.release()

    @property
    def prev_frame(self) -> Frame | None:
        """获取上一帧（运动检测用）"""
        with self._lock:
            return self._prev_frame

    def get_prev_gray(self) -> np.ndarray | None:
        """
        获取上一帧的灰度版本（用于帧差分）

        Returns:
            灰度 numpy 数组 (H, W)，或 None
        """
        with self._lock:
            if self._prev_frame is None:
                return None
            # RGB -> Gray: 使用加权平均（标准 ITU-R BT.601）
            return np.dot(
                self._prev_frame.data[..., :3], [0.299, 0.587, 0.114]
            ).astype(np.uint8)

    def clear(self) -> None:
        """清空所有缓冲帧，释放内存"""
        with self._lock:
            if self._prev_frame is not None:
                self._prev_frame.release()
                self._prev_frame = None
            while self._debug_frames:
                self._debug_frames.pop().release()
