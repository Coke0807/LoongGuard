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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

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

    # ONNX 模型路径（与 config/default.json 保持一致）
    model_path: str = str(_DEFAULT_MODEL_DIR / "best.onnx")
    # 输入尺寸（正方形）
    input_size: int = 640
    # 置信度阈值（与 config/default.json 保持一致）
    conf_threshold: float = 0.80
    # NMS-Free 设计下不使用 IoU 阈值，此处保留用于未来扩展
    iou_threshold: float = 0.50
    # INT8 量化标志（与 config/default.json 保持一致）
    quantized: bool = False
    # 推理后端：onnxruntime / tflite
    backend: str = "onnxruntime"
    # OpenCL 设备索引（LG200）
    opencl_device_id: int = 0
    # 检测类别（危险物品），后续可能新增 person 类
    classes: list[str] = field(default_factory=lambda: [
        "scissor",            # 剪刀
        "utility_knife",      # 美工刀
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
    人脸检测配置（ONNX 实现已就绪）

    睡姿监测依赖"画面有运动主体但未检测到人脸 -> 异常睡姿"判定。
    默认使用 buffalo_l/det_10g.onnx（InsightFace RetinaFace）模型。

    默认关闭（enabled=false）：
        通过环境变量 LG_FACE_ENABLED=true 启用。
        启用后 pipeline 对每帧执行人脸检测，结果用于睡姿监测。
    """

    # 检测后端：dummy（桩）/ onnx（真实模型，默认启用）
    backend: str = "onnx"
    # ONNX 人脸模型路径（默认使用 buffalo_l/det_10g.onnx）
    model_path: str = str(_DEFAULT_MODEL_DIR / "buffalo_l" / "det_10g.onnx")
    # 置信度阈值
    conf_threshold: float = 0.5
    # 是否启用睡姿（人脸可见性）监测；关闭时 pipeline 不执行该逻辑
    enabled: bool = False
    # 定义 YOLO 检测结果中"人"类别名（用于 Person+Face 组合判定）
    person_class: str = "person"
    # 头部姿态：侧脸判定阈值（双眼距/脸高 低于此值视为侧脸）
    # 经验值 0.6（实测正脸 0.72~1.07，侧脸 0.04~0.59）
    side_yaw_threshold: float = 0.6
    # 头部姿态：面内倾斜判定阈值（|roll| 度）
    roll_threshold: float = 45.0
    # 睡姿告警：连续 N 帧未检测到人脸才触发（防单帧漏检误报）
    not_visible_frames_threshold: int = 30


@dataclass
class MediaConfig:
    """
    媒体能力配置（语音 + 流媒体，跨平台预留）

    - audio_backend: command（系统命令）/ dummy（桩）
    - stream_publisher: mjpeg（开发/局域网桩）/ webrtc（板端预留）
    - webrtc_signal_port: 小程序 WebRTC 信令端口
    - hls_*: 家长端远程观看的 HLS 直播（ffmpeg 切片，见 media/hls_publisher.py）

    HLS 工作方式：ffmpeg 消费回环 MJPEG 源（或板端 v4l2 设备）切片输出
    data/hls/live.m3u8，服务端经 /stream/live.m3u8 + /stream/<seg>.ts 提供，
    由带过期时间的 HMAC token 保护（家长端经 /api/wx/stream/url 获取带
    token 的地址）。无观看者超过 idle_timeout 自动停止编码省算力，下次
    请求播放列表时自动重启。ffmpeg 缺失或密钥未配置时整体降级停用。
    """

    # 音频后端
    audio_backend: str = "command"
    # 流媒体发布器类型
    stream_publisher: str = "mjpeg"
    # WebRTC 信令端口（stream_publisher="webrtc" 时生效）
    webrtc_signal_port: int = 8888

    # ── HLS 远程直播（家长端）─────────────────────────────
    # 是否启用；ffmpeg 缺失/密钥缺失时即便 true 也会自动停用
    hls_enabled: bool = True
    # ffmpeg 输入源：loopback=回环消费本服务 MJPEG /stream；
    # 板端可填 v4l2 设备路径（如 /dev/video0）降低一层转码开销
    hls_input: str = "loopback"
    # 强制输入 demuxer（如 mpjpeg）；留空由 ffmpeg 自动探测
    hls_input_format: str = ""
    # 输出视频宽度上限（等比缩放，高度自动取偶）
    hls_video_width: int = 640
    # 输出帧率
    hls_fps: int = 15
    # 输出码率（ffmpeg -b:v）
    hls_video_bitrate: str = "800k"
    # 单个 TS 分片时长（秒），越小延迟越低
    hls_segment_duration_sec: float = 2.0
    # 播列窗口内的分片数（回看窗口 ≈ list_size * segment_duration）
    hls_list_size: int = 6
    # 无观看者超过该秒数自动停止 ffmpeg；有请求时自动重启
    hls_idle_timeout_sec: float = 180.0
    # 观看 token 有效期（秒）
    hls_token_ttl_sec: int = 21600
    # token 签名密钥；留空回退 LG_SM4_KEY，两者都空则 HLS 停用（fail-closed）
    hls_stream_secret: str = ""


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
    # 速率限制：每客户端 IP 每秒允许的请求数（令牌桶填充速率）
    # 设计动机：局域网部署无 WAF/反向代理，需在应用层防暴力请求。
    # 默认 20 rps 远高于正常管理端使用强度，仅抑制扫描/暴力破解。
    rate_limit_rps: float = 20.0
    # 令牌桶容量（允许的突发请求数）
    rate_limit_burst: float = 60.0


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

    # 数据库文件路径（运行时数据，位于 data/db/）
    db_path: str = str(Path(__file__).parent.parent / "data" / "db" / "loongguard.db")
    # 告警保留天数
    retention_days: int = 90
    # 启用 WAL 模式（提升并发读写性能）
    wal_mode: bool = True
    # 定期备份目录（运行时数据，与主库同目录便于统一清理）
    backup_dir: str = str(Path(__file__).parent.parent / "data" / "db" / "backup")
    # 备份间隔（小时）
    backup_interval_hours: int = 24
    # 备份保留天数
    backup_retention_days: int = 7


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
class PersistenceConfig:
    """物体持续出现跟踪配置

    设计动机：
        与 DedupConfig 互补——DedupConfig 控制"已触发告警的重复抑制"，
        而 PersistenceConfig 控制"首次触发告警前物体需稳定出现的时间"。
        两者共同构成完整的告警防误报策略。

    环境变量覆盖：
        LG_PERSISTENCE_PERSISTENCE_SEC=1.0    # 物体需持续出现的最小秒数
        LG_PERSISTENCE_MAX_MISSED_SEC=0.5     # 物体消失超过此秒数则移除跟踪
        LG_PERSISTENCE_POSITION_GRID_SIZE=50  # 位置网格量化粒度（像素）
    """

    # 物体需持续出现的最小秒数，达到此值才允许报警
    persistence_sec: float = 1.0
    # 物体消失超过此秒数则移除跟踪记录
    max_missed_sec: float = 0.5
    # 位置网格量化粒度（像素），用于生成跟踪 key
    position_grid_size: int = 50


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
class NotificationConfig:
    """
    外部告警通知配置

    设计动机：
        告警仅推送到 WebSocket + 入库，无人值守时依赖 WS 客户端在线，
        漏报风险高。此配置允许将 HIGH/CRITICAL 级告警通过 webhook 推送
        到外部网关（微信小程序 / 短信 / 电话），并带失败重试与升级降级。
    """

    # 是否启用外部通知
    enabled: bool = False
    # webhook 回调地址（POST JSON），对接微信小程序/短信/电话网关
    webhook_url: str = ""
    # 触发外部通知的最低严重等级（low/medium/high/critical）
    min_severity: str = "high"
    # 每次告警最大重试次数
    retries: int = 3
    # 重试退避基数（秒），第 n 次重试等待 retry_backoff_sec * 2^(n-1)
    retry_backoff_sec: float = 2.0
    # 请求超时（秒）
    timeout_sec: float = 5.0


@dataclass
class VoiceConfig:
    """
    语音唤醒/指令模块配置（openWakeWord + PocketSphinx 双引擎）

    环境变量覆盖（LG_VOICE_ 前缀）：
        LG_VOICE_ENABLED=false          # 关闭语音模块
        LG_VOICE_WAKE_THRESHOLD=0.5     # 唤醒判定阈值
        LG_VOICE_WAKE_TIMEOUT_SEC=8.0   # 唤醒后等待指令超时
        LG_VOICE_TTS_BACKEND=auto       # auto/pyttsx3/espeak/dummy

    降级策略：唤醒模型或 PocketSphinx 模型缺失/依赖不可用时，
    VoiceService 记录警告并保持待机，不阻塞 Pipeline 主链路。
    """

    # 是否启用语音模块
    enabled: bool = True
    # 采样率（openWakeWord / PocketSphinx 均要求 16kHz）
    sample_rate: int = 16000
    # openWakeWord 自定义唤醒词模型（训练产出，~200KB）
    wake_model_path: str = str(_DEFAULT_MODEL_DIR / "wakeword" / "my_wakeword.onnx")
    # 唤醒判定阈值（0~1，越高越保守）
    wake_threshold: float = 0.5
    # 唤醒冷却秒数（防止连续重复唤醒）
    wake_cooldown_sec: float = 3.0
    # 唤醒后等待指令的超时秒数，超时回到待机
    wake_timeout_sec: float = 8.0
    # 唤醒后单次指令录音时长（秒）
    command_duration_sec: float = 3.0
    # PocketSphinx 中文声学模型目录
    sphinx_model_dir: str = str(_DEFAULT_MODEL_DIR / "pocketsphinx" / "zh-cn")
    # PocketSphinx 中文词典
    sphinx_dict_path: str = str(_DEFAULT_MODEL_DIR / "pocketsphinx" / "cmudict-cn.dict")
    # PocketSphinx 指令 JSGF 语法
    sphinx_jsgf_path: str = str(_DEFAULT_MODEL_DIR / "pocketsphinx" / "commands.jsgf")
    # TTS 后端：auto（按平台）/ pyttsx3 / espeak / dummy
    tts_backend: str = "auto"


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
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log: LogConfig = field(default_factory=LogConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    notify: NotificationConfig = field(default_factory=NotificationConfig)
    # 运行环境：development / testing / production（由 LG_ENV 注入）
    env: str = "development"
    # 已废弃：请使用 log.level，保留仅为向后兼容旧 JSON 配置
    log_level: str = "INFO"
    # 调试模式：开启后保存最近 N 帧到内存用于排查
    debug: bool = False


def load_config(source: str | None = None) -> AppConfig:
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

    注意：.env 的加载不在此处进行（保持配置模块纯净、可测试），
    由应用入口（pipeline.main / 启动脚本）调用 load_dotenv() 完成。
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

    # LG_ENV 为单 token 变量（无 SECTION_KEY 结构），通用覆盖机制无法匹配，
    # 必须显式读取。这是 LG_ENV=production 触发 validate_config 生产红线
    # （强制 Basic Auth、禁止 0.0.0.0）的唯一通道。
    env_name = os.environ.get("LG_ENV", "").strip()
    if env_name:
        config.env = env_name

    # 平台默认摄像头设备：Windows 用索引，Linux 用 /dev/videoN
    # 设计动机：避免同一份 .env 在双平台语义不一致导致打不开摄像头。
    if os.getenv("LG_CAMERA_DEVICE") is None:
        config.camera.device = "0" if sys.platform == "win32" else "/dev/video0"

    return config


def validate_config(config: AppConfig) -> list[str]:
    """
    校验配置合法性，返回错误信息列表（空列表表示通过）。

    设计动机：
        将"配置错误"在启动早期一次性暴露，而不是等到运行到某个模块才炸，
        或更糟——在生产环境静默以不安全配置裸奔。

    当前校验项：
        - SM4 密钥必须为 32 位 hex 字符（16 字节），缺失即阻断启动
        - 生产环境（LG_ENV=production）必须启用 Basic Auth
        - 生产环境 LG_API_HOST 不应为 0.0.0.0（应由 HTTPS 反向代理转发）
        - 外部通知启用时必须配置 webhook_url
    """
    errors: list[str] = []
    is_prod = config.env == "production"

    # SM4 密钥：缺失即阻断（加密存储是核心合规能力）
    raw_key = os.environ.get("LG_SM4_KEY", "")
    if not raw_key:
        errors.append(
            "LG_SM4_KEY 未设置：请填入 32 位 hex 字符(16字节) 密钥，"
            "或注入 config/.sm4_key 文件"
        )
    else:
        try:
            key = bytes.fromhex(raw_key)
            if len(key) != 16:
                errors.append(
                    f"LG_SM4_KEY 长度错误：期望 32 个 hex 字符(16字节)，"
                    f"当前 {len(raw_key)} 个字符"
                )
        except ValueError:
            errors.append("LG_SM4_KEY 不是有效的 hex 字符串")

    # 生产环境安全红线
    if is_prod:
        if not config.api.auth_enabled:
            errors.append(
                "生产环境必须启用 Basic Auth：设置 LG_API_AUTH_ENABLED=true "
                "及 LG_AUTH_USER / LG_AUTH_PASS"
            )
        if config.api.host in ("0.0.0.0", "::"):
            errors.append(
                "生产环境 LG_API_HOST 不应为 0.0.0.0：应由 HTTPS 反向代理"
                "（Caddy/nginx）转发，Python 侧仅监听 127.0.0.1"
            )

    # 外部通知：启用但未配置 webhook
    if config.notify.enabled and not config.notify.webhook_url.strip():
        errors.append("LG_NOTIFY_ENABLED=true 但 LG_NOTIFY_WEBHOOK_URL 未配置")

    return errors


# 导入共享的 .env 加载逻辑（消除代码重复）
# 设计动机：backend/config/settings.py 和 frontend/ui/main.py 共用同一套解析语义
try:
    from loongguard.utils.dotenv import load_dotenv as _load_dotenv
    from loongguard.utils.dotenv import strip_env_value as _strip_env_value
except ImportError:
    # 降级：如果 loongguard 包未安装（如仅导入 config 模块），使用本地实现
    # 这种情况仅在开发初期或配置测试时出现
    import re as _re

    def _strip_env_value(value: str) -> str:
        """本地实现：解析 .env 值（与 loongguard.utils.dotenv 保持同一语义）"""
        value = value.strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            if end > 0:
                return value[1:end]
        match = _re.search(r"\s#", value)
        if match:
            value = value[: match.start()].rstrip()
        return value

    def _load_dotenv(dotenv_path: Path | str | None = None) -> None:
        """本地实现：极简 .env 加载器"""
        if dotenv_path:
            path = Path(dotenv_path)
            if not path.exists():
                return
        else:
            here = Path(__file__).resolve()
            candidates = [here.parents[1] / ".env", here.parents[2] / ".env"]
            path = next((c for c in candidates if c.exists()), None)
            if path is None:
                return

        seen: dict[str, int] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.warning("无法读取 .env 文件: %s", path)
            return

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = _strip_env_value(value)
            if not key:
                continue
            if key in seen:
                logger.warning(
                    ".env 中 KEY 重复定义: %s (第 %d 行 与 第 %d 行)，后值将覆盖前值",
                    key, seen[key], line_no,
                )
            else:
                seen[key] = line_no
            if key not in os.environ:
                os.environ[key] = value


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


# 合法的"非 AppConfig 配置节"LG_ 变量首段（由各模块显式 os.environ 读取，
# 不经通用覆盖机制）。这些变量即使未命中任何配置节也不应告警；
# 不在白名单且未命中的 LG_ 变量（如旧名 LG_GPIO_*）一律 warning，
# 避免"配置写了却静默失效"的隐性 bug 再次发生。
_KNOWN_NON_CONFIG_SECTIONS = frozenset({
    "env",      # LG_ENV（单 token，显式处理）
    "sm4",      # LG_SM4_KEY
    "auth",     # LG_AUTH_USER / LG_AUTH_PASS
    "ssh",      # LG_SSH_*（scripts/sync.py、remote_exec.py）
    "parent",   # LG_PARENT_*（家长端，parent_routes/parent_db/parent_utils）
    "stream",   # LG_STREAM_URL（前端）
    "ws",       # LG_WS_URL（前端）
    "onnx",     # LG_ONNX_INTRA_OP_THREADS（utils/onnx_session.py）
    "debug",    # LG_DEBUG_WINDOW（pipeline 显式读取）
})


def _warn_unmatched_env(section_name: str, env_key: str) -> None:
    if section_name not in _KNOWN_NON_CONFIG_SECTIONS:
        logger.warning(
            "环境变量 %s 未命中任何配置字段（LG_<SECTION>_<KEY> 需与 "
            "config/settings.py 的配置节和字段名对应），已忽略",
            env_key,
        )


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
            _warn_unmatched_env(section_name, env_key)
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
            _warn_unmatched_env(section_name, env_key)
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
