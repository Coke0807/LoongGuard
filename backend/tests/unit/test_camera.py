"""
V4L2Capture 单元测试

设计动机：
    验证摄像头采集模块的生命周期管理、帧数据正确性和跨平台降级逻辑。
    在 Windows 环境下通过 mock 视频文件隔离硬件依赖，确保纯算法逻辑可验证。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import CameraConfig
from loongguard.camera.frame_buffer import FrameBuffer
from loongguard.camera.v4l2_capture import Frame, V4L2Capture

# 项目根目录下的 mock 视频路径
MOCK_VIDEO = str(Path(__file__).parent.parent / "mock_classroom.mp4")


# ── V4L2Capture 测试 ──────────────────────────────────────────


class TestV4L2Capture:
    """V4L2 采集器核心功能测试"""

    def _make_capture(self, **overrides) -> V4L2Capture:
        """构造一个使用默认配置的 V4L2Capture 实例"""
        cfg = CameraConfig(**overrides)
        return V4L2Capture(cfg)

    def test_open_close_lifecycle(self):
        """open() 后 is_opened 为 True，close() 后恢复为 False"""
        cap = self._make_capture()
        assert cap.is_opened is False
        try:
            cap.open()
        except RuntimeError:
            # Windows 无摄像头环境：open() 按预期抛出，无资源需要释放
            pytest.skip("No video source available in current environment")
        assert cap.is_opened is True

        cap.close()
        assert cap.is_opened is False

    def test_double_open_is_safe(self):
        """连续调用两次 open() 不应抛异常，且仍处于 opened 状态"""
        cap = self._make_capture()
        try:
            cap.open()
        except RuntimeError:
            pytest.skip("No video source available in current environment")
        cap.open()  # 第二次应被忽略
        assert cap.is_opened is True
        cap.close()

    def test_close_without_open_is_safe(self):
        """未打开状态调用 close() 不应抛异常"""
        cap = self._make_capture()
        cap.close()  # 应直接返回
        assert cap.is_opened is False

    def test_read_before_open_raises(self):
        """未 open 时调用 read() 应抛出 RuntimeError"""
        cap = self._make_capture()
        with pytest.raises(RuntimeError, match="Camera not opened"):
            cap.read()

    def test_read_returns_frame_with_correct_shape(self):
        """read() 返回的 Frame.data 应为 (H, W, 3) 格式"""
        cap = self._make_capture(width=320, height=240)
        try:
            cap.open()
        except RuntimeError:
            pytest.skip("No video source available in current environment")

        frame = cap.read()
        assert frame is not None
        assert isinstance(frame, Frame)
        # numpy shape 为 (H, W, 3)，H 和 W 取决于实际视频源分辨率
        # （视频文件源返回文件原生分辨率，非 config 配置值）
        assert frame.data.ndim == 3
        assert frame.data.shape[2] == 3
        assert frame.data.dtype == np.uint8
        cap.close()

    def test_frame_id_increments(self):
        """每次 read() 后 frame_id 应递增"""
        cap = self._make_capture()
        try:
            cap.open()
        except RuntimeError:
            pytest.skip("No video source available in current environment")

        f1 = cap.read()
        f2 = cap.read()
        f3 = cap.read()

        assert f1.frame_id == 1
        assert f2.frame_id == 2
        assert f3.frame_id == 3
        cap.close()

    def test_frame_count_property(self):
        """frame_count 应与实际读取帧数一致"""
        cap = self._make_capture()
        try:
            cap.open()
        except RuntimeError:
            pytest.skip("No video source available in current environment")

        for _ in range(5):
            cap.read()

        assert cap.frame_count == 5
        cap.close()

    def test_frame_release(self):
        """Frame.release() 应将 data 替换为空数组"""
        cap = self._make_capture()
        try:
            cap.open()
        except RuntimeError:
            pytest.skip("No video source available in current environment")
        frame = cap.read()

        assert frame.data.size > 0
        frame.release()
        assert frame.data.size == 0
        cap.close()

    def test_context_manager(self):
        """with 语句自动管理 open/close 生命周期"""
        cap = self._make_capture()
        try:
            with cap:
                assert cap.is_opened is True
                frame = cap.read()
                assert frame is not None
        except RuntimeError:
            pytest.skip("No video source available in current environment")
        assert cap.is_opened is False


# ── V4L2Capture 视频源管理测试 ──────────────────────────────────


class TestVideoSourceManagement:
    """
    视频流源管理测试

    覆盖原 StreamHandler 的核心验证点：
    - mock 视频文件能正常读取帧
    - 不存在的文件正确抛异常
    - USB 摄像头在无硬件环境下的降级行为
    """

    def test_read_from_mock_video_file(self):
        """从 mock 视频文件读取帧，返回 BGR uint8 格式"""
        cap = V4L2Capture(CameraConfig())
        try:
            cap.open(source=MOCK_VIDEO)
        except RuntimeError as e:
            pytest.skip(f"Cannot open mock video: {e}")

        frame = cap.read()
        assert frame is not None
        assert isinstance(frame, Frame)
        assert frame.data.ndim == 3
        assert frame.data.shape[2] == 3
        assert frame.data.dtype == np.uint8
        cap.close()

    def test_full_consumption_of_mock_video(self):
        """完整消费 mock 视频文件，确保不抛异常"""
        cap = V4L2Capture(CameraConfig())
        try:
            cap.open(source=MOCK_VIDEO)
        except RuntimeError as e:
            pytest.skip(f"Cannot open mock video: {e}")

        total = 0
        while True:
            frame = cap.read()
            if frame is None:
                break
            total += 1
            assert frame.data.size > 0

        assert total > 0
        cap.close()

    def test_open_with_missing_file_raises(self):
        """指定不存在的视频文件应抛出 RuntimeError"""
        cap = V4L2Capture(CameraConfig())
        with pytest.raises((RuntimeError, FileNotFoundError)):
            cap.open(source="non_existent_video_xyz.mp4")

    def test_open_with_valid_video_file(self):
        """指定有效的 mock 视频文件应成功打开"""
        cap = V4L2Capture(CameraConfig())
        try:
            cap.open(source=MOCK_VIDEO)
        except RuntimeError as e:
            pytest.skip(f"Cannot open mock video: {e}")

        assert cap.is_opened is True
        frame = cap.read()
        assert frame is not None
        cap.close()

    def test_auto_fallback_to_mock_video(self):
        """source=None 时应自动尝试 USB 摄像头或 mock 视频"""
        cap = V4L2Capture(CameraConfig())
        try:
            cap.open()
        except RuntimeError:
            # 无摄像头环境，预期行为
            pytest.skip("No video source available in current environment")

        frame = cap.read()
        assert frame is not None
        cap.close()

    def test_multi_frame_read_consistency(self):
        """连续多帧读取，frame_id 单调递增"""
        cap = V4L2Capture(CameraConfig())
        try:
            cap.open(source=MOCK_VIDEO)
        except RuntimeError as e:
            pytest.skip(f"Cannot open mock video: {e}")

        prev_id = 0
        for _ in range(10):
            frame = cap.read()
            if frame is None:
                break
            assert frame.frame_id > prev_id
            prev_id = frame.frame_id

        cap.close()


# ── FrameBuffer 测试 ───────────────────────────────────────


class TestFrameBuffer:
    """帧缓冲区功能测试"""

    @staticmethod
    def _make_frame(frame_id: int = 1, h: int = 480, w: int = 640) -> Frame:
        """构造测试帧"""
        return Frame(
            data=np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            timestamp=0.0,
            frame_id=frame_id,
        )

    def test_initial_state_has_no_prev_frame(self):
        """初始状态 prev_frame 为 None"""
        buf = FrameBuffer()
        assert buf.prev_frame is None

    def test_push_makes_frame_the_prev(self):
        """push() 后 prev_frame 应为最新帧"""
        buf = FrameBuffer()
        f1 = self._make_frame(1)
        buf.push(f1)
        assert buf.prev_frame is f1

    def test_push_releases_old_frame_in_non_debug(self):
        """非 debug 模式下 push 新帧后旧帧应通过 clear() 释放"""
        buf = FrameBuffer(max_debug_frames=0)
        f1 = self._make_frame(1)
        f2 = self._make_frame(2)
        buf.push(f1)
        buf.push(f2)
        assert buf.prev_frame is f2
        # clear 后 f2 也被释放
        buf.clear()
        assert f2.data.size == 0

    def test_get_prev_gray_returns_none_initially(self):
        """初始状态 get_prev_gray() 返回 None"""
        buf = FrameBuffer()
        assert buf.get_prev_gray() is None

    def test_get_prev_gray_returns_grayscale(self):
        """get_prev_gray() 应返回灰度数组 (H, W)"""
        buf = FrameBuffer()
        f = self._make_frame(1, h=240, w=320)
        buf.push(f)
        gray = buf.get_prev_gray()
        assert gray is not None
        assert gray.shape == (240, 320)
        assert gray.dtype == np.uint8

    def test_clear_releases_all_frames(self):
        """clear() 应释放所有帧"""
        buf = FrameBuffer(max_debug_frames=3)
        frames = [self._make_frame(i) for i in range(3)]
        for f in frames:
            buf.push(f)
        buf.clear()
        assert buf.prev_frame is None
        # 所有帧应该被释放
        for f in frames:
            assert f.data.size == 0
