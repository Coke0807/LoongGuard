"""
Pipeline 核心编排器单元测试

覆盖：
    - Pipeline 实例化与子模块正确绑定
    - start() 子模块启动顺序
    - stop() 资源释放与幂等性
    - _process_frame 运动检测分支
    - _handle_alert 告警链路（去重 -> 告警 -> 加密 -> DB -> API）
    - _handle_alert 去重抑制路径
    - 自适应跳帧逻辑
    - async context manager 生命周期
"""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from config.settings import AppConfig
from loongguard.pipeline import Pipeline
from loongguard.utils.schema import AlertLog, AlertSeverity, AlertType, BoundingBox
from loongguard.motion.frame_diff import MotionRegion
from loongguard.camera.v4l2_capture import Frame


# ── Helpers ────────────────────────────────────────────────


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_frame(frame_id: int = 1) -> Frame:
    """构造测试用 Frame"""
    return Frame(
        data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        timestamp=0.0,
        frame_id=frame_id,
    )


def _make_alert(
    severity: AlertSeverity = AlertSeverity.HIGH,
    alert_type: AlertType = AlertType.DANGEROUS_OBJECT,
    detections: list | None = None,
) -> AlertLog:
    """构造测试用 AlertLog"""
    return AlertLog(
        alert_type=alert_type,
        severity=severity,
        description="test alert",
        detections=detections or [],
    )


def _make_motion_region(ratio: float = 0.05) -> MotionRegion:
    """构造测试用 MotionRegion"""
    return MotionRegion(x=100, y=100, w=50, h=50, motion_ratio=ratio)


# ── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def app_config(monkeypatch) -> AppConfig:
    """测试专用配置，隔离端口和密钥"""
    monkeypatch.setenv("LG_SM4_KEY", "0123456789abcdef0123456789abcdef")
    config = AppConfig()
    config.camera.device = "tests/mock_classroom.mp4"
    config.detection.model_path = "models/best.onnx"
    config.detection.input_size = 640
    config.pose.model_path = "models/movenet_lightning_int8.onnx"
    config.api.port = _find_free_port()
    config.debug = True
    return config


@pytest.fixture
def pipeline(app_config) -> Pipeline:
    """Pipeline 实例，未 start"""
    return Pipeline(app_config)


# ── 测试类 ─────────────────────────────────────────────────


class TestPipelineInstantiation:
    """Pipeline 实例化测试"""

    def test_submodules_created(self, pipeline) -> None:
        """所有子模块在 __init__ 中正确实例化"""
        assert pipeline._camera is not None
        assert pipeline._detector is not None
        assert pipeline._motion is not None
        assert pipeline._roi_scheduler is not None
        assert pipeline._pose is not None
        assert pipeline._alarm is not None
        assert pipeline._crypto is not None
        assert pipeline._api is not None
        assert pipeline._db is not None
        assert pipeline._dedup is not None
        assert pipeline._frame_buffer is not None

    def test_initial_state(self, pipeline) -> None:
        """初始状态：未运行、无检测框、帧计数为 0"""
        assert pipeline._running is False
        assert pipeline._stopping is False
        assert pipeline._last_det_boxes == []
        assert pipeline._last_motion_regions == []
        assert pipeline._frame_count == 0

    def test_config_stored(self, pipeline, app_config) -> None:
        """配置对象被正确存储"""
        assert pipeline._config is app_config


class TestPipelineStartStop:
    """Pipeline 生命周期测试"""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, pipeline) -> None:
        """start() 后 _running 为 True"""
        with patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock):
            await pipeline.start()
            assert pipeline._running is True
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, pipeline) -> None:
        """stop() 后 _running 为 False"""
        with patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock):
            await pipeline.start()
            await pipeline.stop()
            assert pipeline._running is False
            assert pipeline._stopping is True

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, pipeline) -> None:
        """多次调用 stop() 不应抛出异常"""
        with patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock):
            await pipeline.start()
            await pipeline.stop()
            await pipeline.stop()  # 第二次调用
            await pipeline.stop()  # 第三次调用

    @pytest.mark.asyncio
    async def test_start_calls_module_methods(self, pipeline) -> None:
        """start() 依次调用各子模块的初始化方法"""
        with patch.object(pipeline._api, "start", new_callable=AsyncMock) as mock_api_start, \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock), \
             patch.object(pipeline._crypto, "load_key") as mock_load_key, \
             patch.object(pipeline._camera, "open") as mock_cam_open, \
             patch.object(pipeline._detector, "load_model") as mock_det_load, \
             patch.object(pipeline._pose, "load_model") as mock_pose_load, \
             patch.object(pipeline._alarm, "setup") as mock_alarm_setup, \
             patch.object(pipeline._db, "initialize") as mock_db_init:
            await pipeline.start()

            mock_db_init.assert_called_once()
            mock_load_key.assert_called_once()
            mock_cam_open.assert_called_once()
            mock_det_load.assert_called_once()
            mock_pose_load.assert_called_once()
            mock_alarm_setup.assert_called_once()
            mock_api_start.assert_called_once()

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_start_degrades_on_db_failure(self, pipeline) -> None:
        """数据库初始化失败时 pipeline 仍应正常启动"""
        with patch.object(pipeline._db, "initialize", side_effect=Exception("db error")), \
             patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock), \
             patch.object(pipeline._camera, "open"), \
             patch.object(pipeline._detector, "load_model"), \
             patch.object(pipeline._pose, "load_model"), \
             patch.object(pipeline._alarm, "setup"):
            # 不应抛出异常
            await pipeline.start()
            assert pipeline._running is True
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self, pipeline) -> None:
        """async with Pipeline() 自动 start/stop"""
        with patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock), \
             patch.object(pipeline._camera, "open"), \
             patch.object(pipeline._detector, "load_model"), \
             patch.object(pipeline._pose, "load_model"), \
             patch.object(pipeline._alarm, "setup"):
            async with pipeline as p:
                assert p._running is True
            assert pipeline._running is False


class TestPipelineProcessFrame:
    """单帧处理逻辑测试"""

    @pytest.mark.asyncio
    async def test_process_frame_no_motion(self, pipeline) -> None:
        """无运动区域时：帧计数增加，无告警，帧推送 VideoHub"""
        frame = _make_frame()

        with patch.object(pipeline._motion, "detect", return_value=[]), \
             patch.object(pipeline._api, "video_hub") as mock_hub:
            mock_hub.push_frame = MagicMock()
            await pipeline._process_frame(frame)

        assert pipeline._frame_count == 1
        mock_hub.push_frame.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_frame_with_motion(self, pipeline) -> None:
        """有运动区域时：调用 roi_scheduler.process()"""
        frame = _make_frame()
        region = _make_motion_region()

        with patch.object(pipeline._motion, "detect", return_value=[region]), \
             patch.object(pipeline._roi_scheduler, "process", return_value=[]), \
             patch.object(pipeline._api, "video_hub") as mock_hub:
            mock_hub.push_frame = MagicMock()
            await pipeline._process_frame(frame)

        assert pipeline._frame_count == 1

    @pytest.mark.asyncio
    async def test_process_frame_skip_on_large_motion(self, pipeline) -> None:
        """大面积运动 (>80%) 时自适应跳帧，不调用 roi_scheduler"""
        frame = _make_frame()
        region = _make_motion_region(ratio=0.90)
        pipeline._frame_skip_counter = 0

        with patch.object(pipeline._motion, "detect", return_value=[region]), \
             patch.object(pipeline._roi_scheduler, "process") as mock_roi, \
             patch.object(pipeline._api, "video_hub") as mock_hub:
            mock_hub.push_frame = MagicMock()
            await pipeline._process_frame(frame)

        # 跳帧期间 roi_scheduler 不应被调用
        mock_roi.assert_not_called()
        assert pipeline._frame_skip_counter == 1

    @pytest.mark.asyncio
    async def test_process_frame_resets_skip_counter_on_normal_motion(self, pipeline) -> None:
        """正常运动幅度时跳帧计数器归零"""
        frame = _make_frame()
        region = _make_motion_region(ratio=0.05)
        pipeline._frame_skip_counter = 5

        with patch.object(pipeline._motion, "detect", return_value=[region]), \
             patch.object(pipeline._roi_scheduler, "process", return_value=[]), \
             patch.object(pipeline._api, "video_hub") as mock_hub:
            mock_hub.push_frame = MagicMock()
            await pipeline._process_frame(frame)

        assert pipeline._frame_skip_counter == 0

    @pytest.mark.asyncio
    async def test_process_frame_pose_runs_on_motion_only(self, pipeline) -> None:
        """
        姿态估计应在"有运动 + pose 可用 + 到达节流间隔"时执行，
        即使画面中没有危险物品（修复 pose 触发条件耦合 bug）
        """
        frame = _make_frame()
        region = _make_motion_region()
        # 模拟 pose 状态：已加载、未到节流间隔（不执行）
        pipeline._pose.is_available = MagicMock(return_value=True)  # type: ignore[method-assign]
        pipeline._pose.should_run = MagicMock(return_value=True)  # type: ignore[method-assign]
        pipeline._pose.mark_ran = MagicMock()  # type: ignore[method-assign]

        with patch.object(pipeline._motion, "detect", return_value=[region]), \
             patch.object(pipeline._roi_scheduler, "process", return_value=[]), \
             patch.object(pipeline._pose, "detect_prone", return_value=[]) as mock_pose, \
             patch.object(pipeline, "_handle_alert", new_callable=AsyncMock), \
             patch.object(pipeline._api, "video_hub") as mock_hub:
            mock_hub.push_frame = MagicMock()
            await pipeline._process_frame(frame)

        mock_pose.assert_called_once()
        # mark_ran 必须在 finally 中被调用（保证节流计数器推进）
        pipeline._pose.mark_ran.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_frame_pose_skipped_when_pose_unavailable(self, pipeline) -> None:
        """
        pose 模型加载失败（is_available=False）时应跳过俯卧检测，
        不影响危险物品检测主链路
        """
        frame = _make_frame()
        region = _make_motion_region()
        pipeline._pose.is_available = MagicMock(return_value=False)  # type: ignore[method-assign]

        with patch.object(pipeline._motion, "detect", return_value=[region]), \
             patch.object(pipeline._roi_scheduler, "process", return_value=[]), \
             patch.object(pipeline._pose, "detect_prone") as mock_pose, \
             patch.object(pipeline, "_handle_alert", new_callable=AsyncMock), \
             patch.object(pipeline._api, "video_hub") as mock_hub:
            mock_hub.push_frame = MagicMock()
            await pipeline._process_frame(frame)

        # 不应调用 detect_prone（模型已降级）
        mock_pose.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_frame_pose_throttled_by_interval(self, pipeline) -> None:
        """帧节流：should_run 返回 False 时跳过本次推理"""
        frame = _make_frame()
        region = _make_motion_region()
        pipeline._pose.is_available = MagicMock(return_value=True)  # type: ignore[method-assign]
        pipeline._pose.should_run = MagicMock(return_value=False)  # type: ignore[method-assign]

        with patch.object(pipeline._motion, "detect", return_value=[region]), \
             patch.object(pipeline._roi_scheduler, "process", return_value=[]), \
             patch.object(pipeline._pose, "detect_prone") as mock_pose, \
             patch.object(pipeline, "_handle_alert", new_callable=AsyncMock), \
             patch.object(pipeline._api, "video_hub") as mock_hub:
            mock_hub.push_frame = MagicMock()
            await pipeline._process_frame(frame)

        mock_pose.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_frame_pose_failure_does_not_break_pipeline(
        self, pipeline
    ) -> None:
        """
        pose 推理抛出未预期异常时，主链路（危险物品告警）继续执行
        """
        frame = _make_frame()
        region = _make_motion_region()
        bbox = BoundingBox(100, 100, 150, 150, 0.9, 0, "scissors")
        alert = _make_alert(detections=[bbox])
        pipeline._pose.is_available = MagicMock(return_value=True)  # type: ignore[method-assign]
        pipeline._pose.should_run = MagicMock(return_value=True)  # type: ignore[method-assign]
        pipeline._pose.mark_ran = MagicMock()  # type: ignore[method-assign]

        with patch.object(pipeline._motion, "detect", return_value=[region]), \
             patch.object(pipeline._roi_scheduler, "process", return_value=[alert]), \
             patch.object(pipeline._pose, "detect_prone", side_effect=RuntimeError("boom")), \
             patch.object(pipeline, "_handle_alert", new_callable=AsyncMock) as mock_handle, \
             patch.object(pipeline._api, "video_hub") as mock_hub:
            mock_hub.push_frame = MagicMock()
            # 不应抛异常
            await pipeline._process_frame(frame)

        # 危险物品告警仍应被处理（pose 异常不影响主链路）
        assert mock_handle.await_count >= 1


class TestPipelineHandleAlert:
    """告警处理链路测试"""

    @pytest.mark.asyncio
    async def test_handle_alert_full_chain(self, pipeline) -> None:
        """正常告警应经过：去重 -> 告警 -> 加密 -> DB -> API 全链路"""
        alert = _make_alert()
        frame = _make_frame()

        with patch.object(pipeline._dedup, "should_alert", return_value=True), \
             patch.object(pipeline._alarm, "trigger", new_callable=AsyncMock) as mock_alarm, \
             patch.object(pipeline._crypto, "encrypt_and_store") as mock_crypto, \
             patch.object(pipeline._db, "insert_alert") as mock_db, \
             patch.object(pipeline._api, "push_alert", new_callable=AsyncMock) as mock_api:
            await pipeline._handle_alert(alert, frame)

        mock_alarm.assert_called_once_with(AlertSeverity.HIGH)
        mock_crypto.assert_called_once_with(alert)
        mock_db.assert_called_once_with(alert)
        mock_api.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_handle_alert_dedup_suppressed(self, pipeline) -> None:
        """去重抑制的告警不触发任何下游操作"""
        alert = _make_alert()

        with patch.object(pipeline._dedup, "should_alert", return_value=False), \
             patch.object(pipeline._alarm, "trigger", new_callable=AsyncMock) as mock_alarm, \
             patch.object(pipeline._crypto, "encrypt_and_store") as mock_crypto, \
             patch.object(pipeline._db, "insert_alert") as mock_db, \
             patch.object(pipeline._api, "push_alert", new_callable=AsyncMock) as mock_api:
            await pipeline._handle_alert(alert, None)

        mock_alarm.assert_not_called()
        mock_crypto.assert_not_called()
        mock_db.assert_not_called()
        mock_api.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_alert_db_failure_does_not_block(self, pipeline) -> None:
        """数据库写入失败不阻塞 API 推送"""
        alert = _make_alert()

        with patch.object(pipeline._dedup, "should_alert", return_value=True), \
             patch.object(pipeline._alarm, "trigger", new_callable=AsyncMock), \
             patch.object(pipeline._crypto, "encrypt_and_store"), \
             patch.object(pipeline._db, "insert_alert", side_effect=Exception("db error")), \
             patch.object(pipeline._api, "push_alert", new_callable=AsyncMock) as mock_api:
            # 不应抛出异常
            await pipeline._handle_alert(alert, None)

        # API 推送仍应执行
        mock_api.assert_called_once()


class TestPipelineDedupCleanup:
    """周期性去重清理测试"""

    @pytest.mark.asyncio
    async def test_dedup_cleanup_every_100_frames(self, pipeline) -> None:
        """每 100 帧执行一次去重清理"""
        frame = _make_frame()

        with patch.object(pipeline._motion, "detect", return_value=[]), \
             patch.object(pipeline._api, "video_hub") as mock_hub, \
             patch.object(pipeline._dedup, "cleanup_expired") as mock_cleanup:
            mock_hub.push_frame = MagicMock()

            # 处理 100 帧
            pipeline._frame_count = 99  # 下一帧是第 100 帧
            await pipeline._process_frame(frame)

        mock_cleanup.assert_called_once()
