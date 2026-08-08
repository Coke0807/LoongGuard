"""
LoongGuard 全局配置模块

设计动机：
    使用 dataclass 定义配置，零第三方依赖，支持从 dict/YAML/env 加载。
    所有推理阈值、硬件参数、加密路径集中管控，避免硬编码散布各模块。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── 默认值常量 ──────────────────────────────────────────────

_DEFAULT_CONFIG_DIR = Path(__file__).parent
_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
_DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"


@dataclass
class CameraConfig:
    """V4L2 摄像头采集参数"""

    # 设备路径，Linux 下为 /dev/videoN，Windows 下为摄像头索引
    device: str = "0"
    # 采集分辨率（与 config/default.json 保持一致）
    width: int = 640
    height: int = 480
    # 目标帧率
    fps: int = 30
    # 像素格式：YUYV / MJPG / NV12
    pixel_format: str = "MJPG"
    # 环形缓冲区帧数
    buffer_count: int = 4


@dataclass
class DetectionConfig:
    """YOLO26-Nano 目标检测参数"""

    # ONNX 模型路径
    model_path: str = str(_DEFAULT_MODEL_DIR / "yolo26_nano_int8.onnx")
    # 输入尺寸（正方形）
    input_size: int = 640
    # 置信度阈值（与 config/default.json 保持一致）
    conf_threshold: float = 0.30
    # NMS-Free 设计下不使用 IoU 阈值，此处保留用于未来扩展
    iou_threshold: float = 0.50
    # INT8 量化标志（与 config/default.json 保持一致）
    quantized: bool = False
    # 推理后端：onnxruntime / tflite
    backend: str = "onnxruntime"
    # OpenCL 设备索引（LG200）
    opencl_device_id: int = 0
    # 静止危险品全帧扫描间隔（秒）
    # 设计动机：运动触发的 ROI 检测无法捕捉静止危险品（如放在桌面的剪刀）。
    # 每间隔该时长对整帧执行一次 YOLO 检测，作为"静止检测"兜底。
    # 在独立推理线程中执行，不阻塞视频显示帧率。
    full_scan_interval_sec: float = 1.5
    # 检测类别（危险物品）
    classes: list[str] = field(default_factory=lambda: [
        "magnetic_bead",      # 磁力珠
        "button_battery",     # 纽扣电池
        "scissors",           # 剪刀
        "utility_knife",      # 美工刀
        "needle",             # 别针/针
        "glass_shard",        # 碎玻璃
        "wire",               # 铁丝/细金属
        "small_toy_part",     # 微小玩具零件
    ])


@dataclass
class PoseConfig:
    """MoveNet-Lightning 姿态估计参数"""

    # ONNX 模型路径
    model_path: str = str(_DEFAULT_MODEL_DIR / "movenet_lightning_int8.onnx")
    # 输入尺寸
    input_size: int = 192
    # 关键点置信度阈值
    min_keypoint_score: float = 0.3
    # 俯卧判定：肩-髋角度阈值（度），低于此值视为俯卧
    prone_angle_threshold: float = 30.0
    # 俯卧持续帧数，超过此值触发告警
    prone_frame_threshold: int = 30
    # 推理帧节流：每 N 帧执行一次姿态推理（与帧差分运动区域共同决定）
    # 设计动机：MoveNet 推理 20-30ms，逐帧执行会抢占主链路算力。
    # 设为 1 表示每帧都跑，3 表示每 3 帧跑一次，权衡实时性与算力。
    inference_interval: int = 3


@dataclass
class ROIConfig:
    """Dynamic ROI 二级检测参数"""

    # 第一级全局检测分辨率（缩放后）
    stage1_input_size: int = 320
    # 第二级 ROI 裁剪后放大尺寸
    stage2_input_size: int = 640
    # 运动检测激活后 ROI 区域扩展像素边距（与 config/default.json 保持一致）
    roi_expand_pixels: int = 30
    # 最小 ROI 面积（像素），过小的区域跳过（与 config/default.json 保持一致）
    min_roi_area: int = 2500
    # 最大同时处理的 ROI 数量
    max_rois_per_frame: int = 5


@dataclass
class MotionConfig:
    """帧差分运动检测参数"""

    # 差分阈值（像素灰度差），超过此值视为运动像素
    diff_threshold: int = 25
    # 运动像素占比超过此值触发 ROI 更新（与 config/default.json 保持一致）
    motion_ratio_threshold: float = 0.005
    # 高斯模糊核大小
    blur_kernel_size: int = 5
    # 形态学膨胀核大小（填补空洞）
    dilate_kernel_size: int = 7


@dataclass
class FaceConfig:
    """
    人脸检测配置（跨平台预留）

    睡姿监测依赖"Person bbox 内无 Face -> 异常睡姿"判定。
    板端人脸模型权重尚未补充，当前默认 backend="dummy" 使用桩实现，
    保证 Windows 开发与 MVP 链路不阻塞。接入真实模型时：
        1. 将 ONNX 权重放入 backend/models/
        2. 设置 backend="onnx" 与 model_path
    """

    # 检测后端：dummy（桩，默认）/ onnx（真实模型，板端接入）
    backend: str = "dummy"
    # ONNX 人脸模型路径
    model_path: str = str(_DEFAULT_MODEL_DIR / "face_detector.onnx")
    # 置信度阈值
    conf_threshold: float = 0.5
    # 是否启用睡姿（人脸可见性）监测；关闭时 pipeline 不执行该逻辑
    enabled: bool = False
    # 定义 YOLO 检测结果中"人"类别名（用于 Person+Face 组合判定）
    person_class: str = "person"


@dataclass
class MediaConfig:
    """
    媒体能力配置（语音 + 流媒体，跨平台预留）

    - audio_backend: command（系统命令）/ dummy（桩）
    - stream_publisher: mjpeg（开发/局域网桩）/ webrtc（板端预留）
    - webrtc_signal_port: 小程序 WebRTC 信令端口
    """

    # 音频后端
    audio_backend: str = "command"
    # 流媒体发布器类型
    stream_publisher: str = "mjpeg"
    # WebRTC 信令端口（stream_publisher="webrtc" 时生效）
    webrtc_signal_port: int = 8888


@dataclass
class AlarmConfig:
    """GPIO 声光告警参数"""

    # GPIO 芯片编号
    gpio_chip: int = 0
    # 蜂鸣器 GPIO 引脚号
    buzzer_pin: int = 18
    # LED 告警灯 GPIO 引脚号
    led_pin: int = 23
    # LED 状态灯 GPIO 引脚号
    status_led_pin: int = 24
    # 告警持续时间（秒）
    alarm_duration_sec: float = 5.0
    # 同一类型告警冷却时间（秒），防止重复触发
    cooldown_sec: float = 30.0
    # PWM 蜂鸣频率（Hz）
    buzzer_freq_hz: int = 2700


@dataclass
class CryptoConfig:
    """SM4 加密存储参数"""

    # SM4 密钥文件路径（运行时由管理员通过硬件密钥注入）
    key_file: str = str(_DEFAULT_CONFIG_DIR / ".sm4_key")
    # 加密日志存储目录
    log_dir: str = str(_DEFAULT_LOG_DIR / "encrypted")
    # 风险切片图存储目录
    slice_dir: str = str(_DEFAULT_LOG_DIR / "slices")
    # 单个日志文件最大大小（MB）
    max_log_size_mb: int = 50
    # 保留最近 N 天的日志
    retention_days: int = 90


@dataclass
class APIConfig:
    """后端 API 服务参数"""

    host: str = "0.0.0.0"
    port: int = 8080
    # 告警事件 WebSocket 推送路径
    ws_alerts_path: str = "/ws/alerts"
    # 设备状态 REST 路径
    status_path: str = "/api/v1/status"
    # 告警历史 REST 路径
    alerts_path: str = "/api/v1/alerts"
    # Basic Auth 认证（通过 LG_API_AUTH_ENABLED=true 启用）
    auth_enabled: bool = False


@dataclass
class LogConfig:
    """日志系统配置"""
    # 日志级别
    level: str = "INFO"
    # 输出格式：text（人类可读）/ json（结构化）
    format: str = "text"
    # 日志文件目录
    log_dir: str = str(_DEFAULT_LOG_DIR)
    # 单个日志文件最大大小（字节）
    max_bytes: int = 10_485_760  # 10 MB
    # 保留的轮转文件数
    backup_count: int = 5
    # 是否输出到控制台
    console: bool = True


@dataclass
class DatabaseConfig:
    """SQLite 数据库配置"""

    # 数据库文件路径
    db_path: str = str(Path(__file__).parent.parent / "data" / "loongguard.db")
    # 告警保留天数
    retention_days: int = 90
    # 启用 WAL 模式（提升并发读写性能）
    wal_mode: bool = True


@dataclass
class DedupConfig:
    """告警去重配置"""

    # 启用去重
    enabled: bool = True
    # 去重时间窗口（秒）：同一物体在此时间内不重复告警
    time_window_sec: float = 30.0
    # 位置网格大小（像素）：将坐标量化为网格以容忍位置抖动
    position_grid_size: int = 50
    # 最大跟踪对象数（防止内存泄漏）
    max_tracked_objects: int = 1000


@dataclass
class RetentionConfig:
    """视频文件保留策略配置"""

    # 上传视频目录
    upload_dir: str = str(Path(__file__).parent.parent / "data" / "uploads")
    # 文件最大保留时间（小时）
    max_age_hours: float = 24.0
    # 上传目录总大小上限（MB）
    max_total_size_mb: float = 1024.0
    # 清理检查间隔（秒）
    cleanup_interval_sec: float = 3600.0


@dataclass
class AppConfig:
    """应用总配置"""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log: LogConfig = field(default_factory=LogConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    # 已废弃：请使用 log.level，保留仅为向后兼容旧 JSON 配置
    log_level: str = "INFO"
    # 调试模式：开启后保存最近 N 帧到内存用于排查
    debug: bool = False


def load_config(source: Optional[str] = None) -> AppConfig:
    """
    加载配置。

    Args:
        source: 配置来源，支持：
            - None: 使用默认值
            - JSON 文件路径

    Returns:
        AppConfig 实例

    环境变量覆盖规则（始终生效）：
        - 前缀 LG_ + SECTION_KEY（全大写，下划线分隔）
        - 例如 LG_DETECTION_CONF_THRESHOLD=0.5 覆盖 detection.conf_threshold
        - 布尔值：true/false/1/0
        - 列表值：JSON 数组字符串
    """
    if source is None:
        config = AppConfig()
    else:
        source_path = Path(source)
        if source_path.suffix == ".json":
            config = _load_from_json(source_path)
        else:
            raise ValueError(f"Unsupported config source: {source}")

    # 环境变量覆盖（LG_ 前缀），始终生效
    _apply_env_overrides(config)
    return config


def _load_from_json(path: Path) -> AppConfig:
    """从 JSON 文件加载配置，合并到默认值上"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    config = AppConfig()
    for section_name, section_data in data.items():
        section = getattr(config, section_name, None)
        if section is not None and isinstance(section_data, dict):
            for key, value in section_data.items():
                if hasattr(section, key):
                    setattr(section, key, value)
        elif hasattr(config, section_name):
            setattr(config, section_name, section_data)

    return config


def _apply_env_overrides(config: AppConfig) -> None:
    """
    从环境变量覆盖配置值

    设计动机（Factor III: Config）：
        生产环境通过环境变量注入敏感配置（如 API 端口、密钥路径），
        避免将配置硬编码在 JSON 文件中。开发环境使用 .env.local。

    命名规则：
        LG_<SECTION>_<KEY> = value
        例如：LG_API_PORT=9090 覆盖 config.api.port
    """
    prefix = "LG_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].split("_", 1)
        if len(parts) != 2:
            continue
        section_name, key_name = parts[0].lower(), parts[1].lower()

        section = getattr(config, section_name, None)
        if section is None or not hasattr(section, "__dict__"):
            continue

        # 匹配 dataclass 字段名（不区分大小写）
        matched_attr = None
        for attr_name in vars(section):
            if attr_name.startswith("_"):
                continue
            if attr_name.lower() == key_name:
                matched_attr = attr_name
                break
        if matched_attr is None:
            continue

        # 类型转换
        current_val = getattr(section, matched_attr)
        try:
            new_val = _coerce_env_value(env_val, current_val)
            setattr(section, matched_attr, new_val)
            logger.debug("Config override: %s.%s = %r (from %s)",
                         section_name, matched_attr, new_val, env_key)
        except (ValueError, TypeError):
            logger.warning("Cannot coerce env %s=%s to match %s.%s type %s",
                           env_key, env_val, section_name, matched_attr,
                           type(current_val).__name__)


def _coerce_env_value(env_val: str, current_val: object) -> object:
    """将环境变量字符串转换为目标字段的类型"""
    if isinstance(current_val, bool):
        return env_val.lower() in ("true", "1", "yes")
    if isinstance(current_val, int):
        return int(env_val)
    if isinstance(current_val, float):
        return float(env_val)
    if isinstance(current_val, list):
        import json as _json
        return _json.loads(env_val)
    return env_val
