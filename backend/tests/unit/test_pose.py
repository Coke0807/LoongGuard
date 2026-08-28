"""
MoveNet-Lightning 姿态估计模块单元测试

覆盖：
    - load_model() 正常加载
    - detect_prone() 返回值类型
    - _is_prone() 姿态判定逻辑（站立、俯卧、低置信度）
    - 俯卧帧计数器：未达阈值不告警，达到阈值触发告警
    - _build_prone_alert() 返回正确的 AlertType / AlertSeverity
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PoseConfig
from loongguard.pose.movenet import MoveNetLightning
from loongguard.utils.schema import AlertLog, AlertSeverity, AlertType

# 模型路径
MOVENET_MODEL_PATH = str(PROJECT_ROOT / "models" / "movenet_lightning_int8.onnx")


def _make_keypoints(
    *,
    nose_y: float = 0.3,
    nose_x: float = 0.5,
    nose_score: float = 0.8,
    l_shoulder_y: float = 0.4,
    r_shoulder_y: float = 0.4,
    l_shoulder_x: float = 0.45,
    r_shoulder_x: float = 0.55,
    l_shoulder_score: float = 0.8,
    r_shoulder_score: float = 0.8,
    l_hip_y: float = 0.6,
    r_hip_y: float = 0.6,
    l_hip_x: float = 0.45,
    r_hip_x: float = 0.55,
    l_hip_score: float = 0.8,
    r_hip_score: float = 0.8,
) -> np.ndarray:
    """
    构造 (17, 3) 关键点数组 [y, x, score]

    MoveNet 17 个关键点索引:
        0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
        5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
        9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
        13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
    """
    kp = np.zeros((17, 3), dtype=np.float32)

    # nose
    kp[0] = [nose_y, nose_x, nose_score]
    # eyes
    kp[1] = [nose_y - 0.03, nose_x - 0.02, 0.7]
    kp[2] = [nose_y - 0.03, nose_x + 0.02, 0.7]
    # ears
    kp[3] = [nose_y, nose_x - 0.04, 0.6]
    kp[4] = [nose_y, nose_x + 0.04, 0.6]
    # shoulders
    kp[5] = [l_shoulder_y, l_shoulder_x, l_shoulder_score]
    kp[6] = [r_shoulder_y, r_shoulder_x, r_shoulder_score]
    # elbows
    kp[7] = [0.5, 0.40, 0.6]
    kp[8] = [0.5, 0.60, 0.6]
    # wrists
    kp[9] = [0.55, 0.35, 0.5]
    kp[10] = [0.55, 0.65, 0.5]
    # hips
    kp[11] = [l_hip_y, l_hip_x, l_hip_score]
    kp[12] = [r_hip_y, r_hip_x, r_hip_score]
    # knees
    kp[13] = [0.75, 0.45, 0.6]
    kp[14] = [0.75, 0.55, 0.6]
    # ankles
    kp[15] = [0.90, 0.45, 0.5]
    kp[16] = [0.90, 0.55, 0.5]

    return kp


class TestMoveNetLightning:
    """MoveNet-Lightning 姿态估计器测试套件"""

    @pytest.fixture
    def config(self) -> PoseConfig:
        """使用 dummy 模型的 PoseConfig，阈值设低便于测试"""
        return PoseConfig(
            model_path=str(PROJECT_ROOT / "models" / "movenet_lightning_int8.onnx"),
            input_size=192,
            min_keypoint_score=0.3,
            prone_angle_threshold=30.0,
            prone_frame_threshold=5,  # 降低阈值便于测试帧计数
        )

    @pytest.fixture
    def pose(self, config: PoseConfig) -> MoveNetLightning:
        """已加载模型的 MoveNetLightning 实例"""
        p = MoveNetLightning(config)
        p.load_model()
        return p

    # ── 模型加载 ──────────────────────────────────────────────

    def test_load_model_initializes_session(self, config: PoseConfig) -> None:
        """load_model() 应成功创建 ONNX Runtime 会话"""
        p = MoveNetLightning(config)
        assert p._session is None
        p.load_model()
        assert p._session is not None

    def test_infer_without_load_returns_empty(self, config: PoseConfig) -> None:
        """
        未加载模型时调用 detect_prone() 应降级返回空列表（不抛异常）

        设计动机：Pipeline 主链路需要"pose 挂掉不影响危险物品检测"。
        此前的 RuntimeError 行为已重构为软降级，测试同步更新。
        """
        p = MoveNetLightning(config)
        dummy = np.zeros((192, 192, 3), dtype=np.uint8)
        alerts = p.detect_prone(dummy)
        assert alerts == []
        assert p.is_available() is False

    # ── _is_prone 姿态判定 ─────────────────────────────────────

    def test_is_prone_standing_pose_returns_false(
        self, pose: MoveNetLightning
    ) -> None:
        """正常站立姿态：肩膀竖直、鼻子在肩膀上方 -> 非俯卧"""
        # 站立：肩膀 y=0.3/0.31，髋 y=0.6/0.61，鼻子 y=0.15
        # 肩髋角度接近 90 度（垂直），鼻子在肩膀上方
        kp = _make_keypoints(
            nose_y=0.15,
            l_shoulder_y=0.30,
            r_shoulder_y=0.31,
            l_hip_y=0.60,
            r_hip_y=0.61,
        )
        # _is_prone() 返回 numpy.bool_，用 not 而非 is False
        assert not pose._is_prone(kp)

    def test_is_prone_flat_prone_pose_returns_true(
        self, pose: MoveNetLightning
    ) -> None:
        """俯卧姿态：肩膀水平、髋部水平、鼻子低于肩膀 -> 俯卧"""
        # 俯卧趴着：肩膀和髋几乎在同一水平线上（y 相近），鼻子 y 更大（更靠下）
        # 角度 ≈ atan(dx/dy)，dy 很小 -> 角度接近 0 -> < 30 度阈值
        kp = _make_keypoints(
            nose_y=0.55,         # 鼻子低于肩膀（y 更大 = 更靠下）
            l_shoulder_y=0.45,
            r_shoulder_y=0.45,   # 肩膀水平
            l_hip_y=0.46,
            r_hip_y=0.46,        # 髋部与肩膀几乎同高
            l_shoulder_x=0.40,
            r_shoulder_x=0.60,
            l_hip_x=0.40,
            r_hip_x=0.60,
        )
        assert pose._is_prone(kp)

    def test_is_prone_low_confidence_returns_false(
        self, pose: MoveNetLightning
    ) -> None:
        """关键点置信度过低时应返回 False（即使几何形状像俯卧）"""
        kp = _make_keypoints(
            nose_y=0.55,
            nose_score=0.1,       # 低置信度
            l_shoulder_y=0.45,
            r_shoulder_y=0.45,
            l_shoulder_score=0.1,
            r_shoulder_score=0.1,
            l_hip_y=0.46,
            r_hip_y=0.46,
            l_hip_score=0.1,
            r_hip_score=0.1,
        )
        assert pose._is_prone(kp) is False

    def test_is_prone_face_up_returns_false(
        self, pose: MoveNetLightning
    ) -> None:
        """仰卧姿态：肩膀水平但鼻子高于肩膀 -> 非俯卧（面朝上）"""
        kp = _make_keypoints(
            nose_y=0.30,          # 鼻子高于肩膀（y 更小 = 更靠上）
            l_shoulder_y=0.45,
            r_shoulder_y=0.45,
            l_hip_y=0.46,
            r_hip_y=0.46,
        )
        assert not pose._is_prone(kp)

    # ── 帧计数器与告警 ─────────────────────────────────────────

    def test_prone_counter_below_threshold_no_alert(
        self, pose: MoveNetLightning
    ) -> None:
        """连续俯卧帧数未达到阈值时不应产生告警"""
        prone_kp = _make_keypoints(
            nose_y=0.55,
            l_shoulder_y=0.45,
            r_shoulder_y=0.45,
            l_hip_y=0.46,
            r_hip_y=0.46,
        )

        # mock _infer_keypoints 返回俯卧关键点
        with patch.object(pose, "_infer_keypoints", return_value=prone_kp):
            # 阈值为 5，只触发 3 次
            for _ in range(3):
                alerts = pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))
                assert alerts == []

    def test_prone_counter_at_threshold_triggers_alert(
        self, pose: MoveNetLightning
    ) -> None:
        """连续俯卧帧数达到阈值时应触发 CRITICAL 告警"""
        prone_kp = _make_keypoints(
            nose_y=0.55,
            l_shoulder_y=0.45,
            r_shoulder_y=0.45,
            l_hip_y=0.46,
            r_hip_y=0.46,
        )

        with patch.object(pose, "_infer_keypoints", return_value=prone_kp):
            alerts = []
            # 滑动窗口需要 2 * threshold 次才会触发（窗口满且比例 >= 50%）
            for _ in range(pose._config.prone_frame_threshold * 2):
                alerts = pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))

            assert len(alerts) == 1
            assert isinstance(alerts[0], AlertLog)

    def test_prone_counter_resets_on_non_prone(
        self, pose: MoveNetLightning
    ) -> None:
        """非俯卧帧应重置计数器，防止误报"""
        prone_kp = _make_keypoints(
            nose_y=0.55,
            l_shoulder_y=0.45,
            r_shoulder_y=0.45,
            l_hip_y=0.46,
            r_hip_y=0.46,
        )
        standing_kp = _make_keypoints(
            nose_y=0.15,
            l_shoulder_y=0.30,
            r_shoulder_y=0.31,
            l_hip_y=0.60,
            r_hip_y=0.61,
        )

        with patch.object(pose, "_infer_keypoints") as mock_kp:
            # 先触发 3 次俯卧
            mock_kp.return_value = prone_kp
            for _ in range(3):
                pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))

            # 插入 1 次站立 -> 计数器归零
            mock_kp.return_value = standing_kp
            pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))

            # 再触发 3 次俯卧（计数器从 1 重新开始），不应达到阈值 5
            mock_kp.return_value = prone_kp
            for _ in range(3):
                alerts = pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))

            assert alerts == []

    def test_detect_prone_inference_failure_returns_empty(
        self, pose: MoveNetLightning
    ) -> None:
        """推理失败（返回 None）时应重置计数器并返回空列表"""
        with patch.object(pose, "_infer_keypoints", return_value=None):
            alerts = pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))
            assert alerts == []
            # 推理失败时 buffer 记录 False
            assert list(pose._prone_frame_buffer) == [False]

    # ── _build_prone_alert ─────────────────────────────────────

    def test_build_prone_alert_fields(self, pose: MoveNetLightning) -> None:
        """_build_prone_alert() 应返回正确的 AlertType 和 AlertSeverity"""
        kp = np.zeros((17, 3), dtype=np.float32)
        alert = pose._build_prone_alert(kp)

        assert isinstance(alert, AlertLog)
        assert alert.alert_type == AlertType.PRONE_SLEEP
        assert alert.severity == AlertSeverity.CRITICAL
        assert "俯卧" in alert.description
        assert str(pose._config.prone_frame_threshold) in alert.description

    def test_detect_prone_full_pipeline_alert_type(
        self, pose: MoveNetLightning
    ) -> None:
        """完整 detect_prone() 流程触发的告警应包含 PRONE_SLEEP 类型"""
        prone_kp = _make_keypoints(
            nose_y=0.55,
            l_shoulder_y=0.45,
            r_shoulder_y=0.45,
            l_hip_y=0.46,
            r_hip_y=0.46,
        )

        with patch.object(pose, "_infer_keypoints", return_value=prone_kp):
            alerts = []
            for _ in range(pose._config.prone_frame_threshold * 2):
                alerts = pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))

            assert len(alerts) == 1
            assert alerts[0].alert_type == AlertType.PRONE_SLEEP
            assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_detect_prone_alert_resets_counter_after_trigger(
        self, pose: MoveNetLightning
    ) -> None:
        """触发告警后计数器应归零，避免连续重复告警"""
        prone_kp = _make_keypoints(
            nose_y=0.55,
            l_shoulder_y=0.45,
            r_shoulder_y=0.45,
            l_hip_y=0.46,
            r_hip_y=0.46,
        )

        with patch.object(pose, "_infer_keypoints", return_value=prone_kp):
            # 触发第一次告警
            for _ in range(pose._config.prone_frame_threshold * 2):
                pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))

            assert len(pose._prone_frame_buffer) == 0  # 触发后清空缓冲区

            # 再来一次，不应立即触发（缓冲区为空，需重新累积）
            alerts = pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))
            assert alerts == []


class TestOpenCLBackendVerification:
    """
    OpenCL 后端可用性验证

    MoveNet 模型支持 CUDA > OpenCL > CPU 三级后端选择。
    在 LoongArch 2K3000 上，LG200 GPU 通过 OpenCL 调度算力。
    此测试类验证 ONNX Runtime 的 OpenCL 执行提供程序是否可用。
    """

    def test_onnxruntime_available_providers(self) -> None:
        """ONNX Runtime 可用提供程序列表非空"""
        import onnxruntime as ort
        available = ort.get_available_providers()
        assert isinstance(available, list)
        assert len(available) > 0

    def test_movenet_model_loads_on_cpu(self) -> None:
        """MoveNet 模型在 CPU 后端下可正常加载并推理"""
        from config.settings import PoseConfig

        config = PoseConfig(model_path="models/movenet_lightning_int8.onnx")
        pose = MoveNetLightning(config)
        pose.load_model()

        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        alerts = pose.detect_prone(dummy)
        assert isinstance(alerts, list)

    def test_movenet_session_provider_priority(self) -> None:
        """MoveNet 加载后，session 的实际 provider 符合优先级预期"""
        import onnxruntime as ort

        from config.settings import PoseConfig

        config = PoseConfig(model_path="models/movenet_lightning_int8.onnx")
        pose = MoveNetLightning(config)
        pose.load_model()

        session = pose._session
        actual = session.get_providers()

        available = ort.get_available_providers()
        # CUDA 优先于 OpenCL 优先于 CPU
        if "CUDAExecutionProvider" in available:
            assert "CUDAExecutionProvider" in actual
        elif "OpenCLExecutionProvider" in available:
            assert "OpenCLExecutionProvider" in actual
        else:
            assert "CPUExecutionProvider" in actual


class TestMoveNetAvailability:
    """
    MoveNet 降级与节流测试

    覆盖：
        - load_model 失败标记 _available=False 且不抛异常
        - is_available() 反映加载状态
        - should_run() 基于帧序号做节流，未到间隔返回 False
        - mark_ran() 推进 _last_run_frame
        - get_health_snapshot() 字段完整
        - 推理失败（_infer_keypoints 抛异常）不中断、降级
    """

    def _make_config(self) -> PoseConfig:
        return PoseConfig(
            model_path=str(PROJECT_ROOT / "models" / "movenet_lightning_int8.onnx"),
            inference_interval=3,
        )

    @pytest.fixture
    def pose(self) -> MoveNetLightning:
        """已加载模型的 MoveNetLightning 实例（与 TestMoveNetLightning 共用 fixture 逻辑）"""
        p = MoveNetLightning(self._make_config())
        p.load_model()
        return p

    def test_load_failure_marks_unavailable(self) -> None:
        """模型文件不存在时 _available=False,且不抛异常"""
        bad_config = PoseConfig(model_path="models/nonexistent_model.onnx")
        p = MoveNetLightning(bad_config)
        # 不应抛异常
        p.load_model()
        assert p.is_available() is False
        # ONNX Runtime 把缺失文件包装为 NO_SUCHFILE 异常
        assert "no_suchfile" in p._last_error.lower() or "not found" in p._last_error.lower()

    def test_load_success_marks_available(self) -> None:
        """正常加载时 _available=True"""
        p = MoveNetLightning(self._make_config())
        p.load_model()
        assert p.is_available() is True

    def test_should_run_throttles_by_frame_count(self) -> None:
        """should_run() 按 inference_interval 节流,首次调用总返回 True"""
        p = MoveNetLightning(self._make_config())
        # 模拟已加载（不实际加载模型文件）
        p._available = True
        # _last_run_frame 初始为 None,首次调用应执行
        assert p._last_run_frame is None
        assert p.should_run(0) is True
        p.mark_ran(0)

        # 帧 1、2：距上次 1、2 帧，< interval=3，不应执行
        assert p.should_run(1) is False
        assert p.should_run(2) is False

        # 帧 3：距上次 3 帧，== interval，应执行
        assert p.should_run(3) is True

    def test_should_run_returns_false_when_unavailable(self) -> None:
        """模型未加载时 should_run 永远返回 False"""
        p = MoveNetLightning(self._make_config())
        # _available 默认 False
        assert p.should_run(0) is False
        assert p.should_run(100) is False

    def test_health_snapshot_fields(self) -> None:
        """get_health_snapshot() 返回所有运维所需的字段"""
        p = MoveNetLightning(self._make_config())
        snap = p.get_health_snapshot()
        assert "available" in snap
        assert "model_path" in snap
        assert "last_inference_ts" in snap
        assert "success_count" in snap
        assert "error_count" in snap
        assert "last_error" in snap
        assert "last_run_frame" in snap
        assert snap["available"] is False  # 未加载

    def test_inference_exception_increments_error_counter(
        self, pose: MoveNetLightning
    ) -> None:
        """_infer_keypoints 抛异常时，错误计数 +1，pipeline 不中断"""
        with patch.object(
            pose, "_infer_keypoints", side_effect=RuntimeError("infer boom")
        ):
            alerts = pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))
        assert alerts == []
        assert pose._error_count == 1
        assert "infer boom" in pose._last_error

    def test_successful_inference_increments_success_counter(
        self, pose: MoveNetLightning
    ) -> None:
        """成功推理时 success_count +1, last_inference_ts 更新"""
        standing_kp = _make_keypoints()  # 默认站立姿态
        with patch.object(pose, "_infer_keypoints", return_value=standing_kp):
            before_ts = pose._last_inference_ts
            pose.detect_prone(np.zeros((192, 192, 3), dtype=np.uint8))
        assert pose._success_count == 1
        assert pose._last_inference_ts > before_ts
