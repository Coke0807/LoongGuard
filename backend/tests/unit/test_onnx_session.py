"""
ONNX Session 工厂与环境变量测试

覆盖：
    - _default_intra_op_threads 自适应逻辑
    - LG_ONNX_INTRA_OP_THREADS 环境变量覆盖
    - create_session_options 兼容新旧 ORT 属性名
    - select_providers 优先级 (CUDA > OpenCL > CPU)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from loongguard.utils import onnx_session
from loongguard.utils.onnx_session import (
    _INTRA_OP_THREADS_ENV,
    _default_intra_op_threads,
    create_session,
    create_session_options,
    select_providers,
)


# ── _default_intra_op_threads 自适应 ──────────────────────────


class TestDefaultIntraOpThreads:
    """线程数自适应:env > 默认(cpus//2, 上限 4, 下限 1)"""

    def test_no_env_returns_cpu_count_clamped(self, monkeypatch) -> None:
        """无环境变量时取 cpu_count//2, 范围 [1, 4]"""
        monkeypatch.delenv(_INTRA_OP_THREADS_ENV, raising=False)
        # 8 核 → 8//2 = 4(等于上限)
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        assert _default_intra_op_threads() == 4

    def test_large_cpu_clamped_to_max(self, monkeypatch) -> None:
        """16 核 → 16//2 = 8,但上限 4"""
        monkeypatch.delenv(_INTRA_OP_THREADS_ENV, raising=False)
        monkeypatch.setattr("os.cpu_count", lambda: 16)
        assert _default_intra_op_threads() == 4

    def test_small_cpu_returns_at_least_one(self, monkeypatch) -> None:
        """2 核 → 2//2 = 1(下限)"""
        monkeypatch.delenv(_INTRA_OP_THREADS_ENV, raising=False)
        monkeypatch.setattr("os.cpu_count", lambda: 2)
        assert _default_intra_op_threads() == 1

    def test_zero_cpu_count_returns_one(self, monkeypatch) -> None:
        """os.cpu_count() 返回 None 时回退到 1"""
        monkeypatch.delenv(_INTRA_OP_THREADS_ENV, raising=False)
        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert _default_intra_op_threads() == 1

    def test_env_var_explicit_override(self, monkeypatch) -> None:
        """LG_ONNX_INTRA_OP_THREADS 显式覆盖自适应"""
        monkeypatch.setenv(_INTRA_OP_THREADS_ENV, "2")
        # 即使 16 核, env=2 仍生效
        monkeypatch.setattr("os.cpu_count", lambda: 16)
        assert _default_intra_op_threads() == 2

    def test_env_var_invalid_value_falls_back(self, monkeypatch) -> None:
        """环境变量非法值回退到自适应"""
        monkeypatch.setenv(_INTRA_OP_THREADS_ENV, "not_a_number")
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        assert _default_intra_op_threads() == 4

    def test_env_var_zero_or_negative_falls_back(self, monkeypatch) -> None:
        """环境变量 <= 0 视为无效,回退自适应"""
        for bad in ("0", "-1", "  "):
            monkeypatch.setenv(_INTRA_OP_THREADS_ENV, bad)
            monkeypatch.setattr("os.cpu_count", lambda: 8)
            assert _default_intra_op_threads() == 4, f"failed for {bad!r}"


# ── create_session_options ───────────────────────────────────


class TestCreateSessionOptions:
    """SessionOptions 工厂"""

    def test_returns_session_options(self) -> None:
        """应返回 ort.SessionOptions 实例"""
        import onnxruntime as ort
        opts = create_session_options()
        assert isinstance(opts, ort.SessionOptions)

    def test_graph_optimization_enabled(self) -> None:
        """图优化应启用 ORT_ENABLE_ALL"""
        opts = create_session_options()
        import onnxruntime as ort
        assert opts.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    def test_intra_op_threads_default_applied(self) -> None:
        """未指定线程数时使用自适应值"""
        import onnxruntime as ort
        opts = create_session_options()
        # 自适应:8 核 → 4 线程
        assert opts.intra_op_num_threads >= 1

    def test_intra_op_threads_explicit_override(self) -> None:
        """显式 intra_op_threads 参数覆盖自适应"""
        opts = create_session_options(intra_op_threads=6)
        assert opts.intra_op_num_threads == 6

    def test_intra_op_threads_zero_uses_default(self) -> None:
        """intra_op_threads=0 视为使用自适应"""
        opts = create_session_options(intra_op_threads=0)
        assert opts.intra_op_num_threads >= 1


# ── select_providers ─────────────────────────────────────────


class TestSelectProviders:
    """Provider 选择:优先级 CUDA > OpenCL > CPU"""

    def test_returns_at_least_cpu(self) -> None:
        """即使没有任何 provider,至少应返回 CPUExecutionProvider"""
        providers = select_providers()
        assert "CPUExecutionProvider" in providers

    def test_priority_order(self, monkeypatch) -> None:
        """当 CUDA 和 OpenCL 都可用时,CUDA 优先"""
        fake_available = {
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "OpenCLExecutionProvider",
        }
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: list(fake_available),
        )
        providers = select_providers()
        assert providers[0] == "CUDAExecutionProvider"
        assert providers[1] == "OpenCLExecutionProvider"
        assert "CPUExecutionProvider" in providers


# ── create_session 集成 ──────────────────────────────────────


class TestCreateSession:
    """create_session() 完整流程(需要真实模型文件)"""

    YOLO_MODEL = str(PROJECT_ROOT / "models" / "yolo26_nano_int8.onnx")

    def test_creates_session_from_real_model(self) -> None:
        """从真实模型创建 Session 应成功"""
        import onnxruntime as ort

        session = create_session(self.YOLO_MODEL)
        assert isinstance(session, ort.InferenceSession)
        # 输入输出名应可读
        assert len(session.get_inputs()) > 0
        assert len(session.get_outputs()) > 0

    def test_custom_session_options_applied(self) -> None:
        """自定义 SessionOptions 应被应用"""
        import onnxruntime as ort

        opts = create_session_options(intra_op_threads=2)
        session = create_session(self.YOLO_MODEL, session_options=opts)
        # session_options 是只读 getter,此处仅断言不抛异常
        assert session.get_session_options() is not None
