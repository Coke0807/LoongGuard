"""
ONNX Runtime 共享 Session 工厂

设计动机：
    YOLO26-Nano 和 MoveNet-Lightning 各自创建 ort.InferenceSession 时
    会重复构造 SessionOptions（线程数、图优化级别、内存 arena 配置），
    且执行提供程序（Execution Provider）选择逻辑在两处重复。

    统一抽出本工厂实现：
    1. 单点维护 SessionOptions 配置（线程数、图优化级别等）
    2. 统一执行提供程序选择策略（CUDA > OpenCL > CPU）
    3. 便于未来切换底层推理后端而不修改业务模块

注意：
    SessionOptions 本身在两个 Session 间无法"共享"——ONNX Runtime
    每个 InferenceSession 必须持有一份独立 SessionOptions。本工厂
    提供的是"统一创建入口"，确保两处配置一致，避免漂移。
    内存 arena 共享需借助 IOBinding + ArenaAllocator 工厂，
    当前 LG200 8GB RAM 下两份独立 arena 完全可承受。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# 优先级：CUDA(x86 训练机) > OpenCL(龙芯 LG200 GPU) > CPU
_PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "OpenCLExecutionProvider",
    "CPUExecutionProvider",
)

# 环境变量：显式覆盖 intra_op_num_threads（留空则自适应）
# 优先级：环境变量 > 默认自适应
_INTRA_OP_THREADS_ENV = "LG_ONNX_INTRA_OP_THREADS"

# 默认线程数（覆盖式自适应，详见 _default_intra_op_threads 注释）
_DEFAULT_INTRA_OP_THREADS = 4


def _default_intra_op_threads() -> int:
    """
    计算默认的 intra_op_num_threads

    策略：
        1. 若设置了环境变量 LG_ONNX_INTRA_OP_THREADS 且 >0，使用之
        2. 否则取 os.cpu_count() 的一半，下限 1，上限 _DEFAULT_INTRA_OP_THREADS

    设计动机：
        LG200 8 核 CPU 下，单模型推理最优线程数通常为 4 核左右。
        过多线程因锁竞争和上下文切换反而降低吞吐；过少则不能
        充分利用多核。此处采用"一半核心"经验值，并允许环境变量覆盖。
    """
    env_val = os.environ.get(_INTRA_OP_THREADS_ENV, "").strip()
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            logger.warning(
                "%s=%r 无法解析为正整数,回退到自适应",
                _INTRA_OP_THREADS_ENV, env_val,
            )

    cpu = os.cpu_count() or 1
    # 经验值：单模型推理用一半核心，最少 1 核，最多 4 核
    return max(1, min(cpu // 2, _DEFAULT_INTRA_OP_THREADS))


def select_providers() -> list[str]:
    """
    按优先级选择可用的 ONNX Runtime 执行提供程序

    Returns:
        可用 provider 列表，至少包含 CPUExecutionProvider
    """
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        logger.warning("onnxruntime 未安装，回退到 CPUExecutionProvider")
        return ["CPUExecutionProvider"]

    selected = [p for p in _PROVIDER_PRIORITY if p in available]
    if not selected:
        selected = ["CPUExecutionProvider"]
    return selected


def create_session_options(
    *,
    intra_op_threads: Optional[int] = None,
) -> "ort.SessionOptions":  # type: ignore[name-defined]
    """
    创建统一的 ONNX Runtime SessionOptions

    Args:
        intra_op_threads: 算子内并行线程数；None 时由 _default_intra_op_threads() 自适应

    设计动机：
        LG200 8 核 CPU 下，单模型推理线程数过多反而会因线程切换损耗
        性能。默认交由自适应函数计算，应用层不强行指定。

    注意：
        ORT 1.16+ 中 mem arena 启用属性名是 enable_cpu_mem_arena，
        早期版本叫 enable_mem_arena。本工厂使用 setattr + hasattr
        兼容两种命名，避免硬编码在升级时炸掉。
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime 未安装，无法创建 SessionOptions"
        ) from exc

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 兼容新旧 ORT 版本的 mem arena 属性名
    for attr in ("enable_cpu_mem_arena", "enable_mem_arena"):
        if hasattr(opts, attr):
            setattr(opts, attr, True)
    # 线程数：显式参数 > 环境变量 > 自适应
    threads = intra_op_threads if intra_op_threads and intra_op_threads > 0 \
        else _default_intra_op_threads()
    opts.intra_op_num_threads = threads
    return opts


def create_session(
    model_path: str,
    *,
    session_options: Optional["ort.SessionOptions"] = None,  # type: ignore[name-defined]
) -> "ort.InferenceSession":  # type: ignore[name-defined]
    """
    加载 ONNX 模型并创建推理 Session

    Args:
        model_path: ONNX 模型文件路径
        session_options: 自定义 SessionOptions；None 时调用 create_session_options()

    Returns:
        初始化完成的 InferenceSession
    """
    import onnxruntime as ort

    opts = session_options or create_session_options()
    providers = select_providers()
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

    logger.info(
        "ONNX session created: model=%s, providers=%s",
        model_path, providers,
    )
    return session
