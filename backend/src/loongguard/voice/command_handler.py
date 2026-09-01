"""
语音指令识别（PocketSphinx）+ 业务分发

设计动机：
    双引擎分离架构的第二级——唤醒后用 PocketSphinx（JSGF 语法约束）
    识别固定指令集，比自由 ASR 误识别率低、资源占用小。
    识别产出指令文本后，由 TextCommandDispatcher 做业务分发，
    分发逻辑同时兼容中文文本（Whisper 等通用 ASR）与
    拼音文本（CMU Mandarin 词典的 JSGF 识别输出）两种来源。

指令到真实功能的对接（回调注入，由 Pipeline 提供）：
    - set_mode(mode)     切换检测模式（normal/nap/public）
    - ack_all()          批量确认未确认告警
    - get_sensor_data()  RS485 温湿度读取（无传感器时返回 None）
    - start_record()     开始视频录制
    - get_recent_logs()  最近告警摘要
    - shutdown()         安全退出系统
    - stop_listen        内部状态（回到待机），不注入回调

来源：重构自 VoiceDetection/long3.py / long4.py 的 handle_command()。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── 指令别名表 ────────────────────────────────────────────────
# 中文别名：匹配通用 ASR 文本输出（与 long3/long4 原始指令集一致）
# 拼音别名：匹配 CMU Mandarin 模型 + 拼音词典的 PocketSphinx 输出
# （匹配前会去掉声调数字与空白）

COMMAND_ALIASES: dict[str, dict[str, list[str]]] = {
    "set_mode_normal": {
        "zh": ["切换到正常课堂模式", "切换到课堂模式", "课堂"],
        "pinyin": ["qiehuandaozhengchangketangmoshi", "qiehuandaoketangmoshi"],
    },
    "set_mode_nap": {
        "zh": ["切换到午睡模式", "午睡"],
        "pinyin": ["qiehuandaowushuimoshi"],
    },
    "set_mode_public": {
        "zh": ["切换到公共区域模式", "切换到公共巡查模式", "巡查"],
        "pinyin": ["qiehuandaogonggongquyumoshi", "qiehuandaogonggongxunchamoshi"],
    },
    "prompt_mode": {
        "zh": ["切换模式"],
        "pinyin": ["qiehuanmoshi"],
    },
    "ack_all": {
        "zh": ["关闭告警", "关闭当前告警", "告警"],
        "pinyin": ["guanbigaojing", "guanbidangqiangaojing"],
    },
    "get_temp": {
        "zh": ["查询温湿度", "温湿度", "温度"],
        "pinyin": ["chaxunwenshidu", "wenshidu"],
    },
    "start_record": {
        "zh": ["开始录制", "录制"],
        "pinyin": ["kaishiluzhi"],
    },
    "get_logs": {
        "zh": ["查看日志", "调取日志", "日志"],
        "pinyin": ["chakanrizhi", "diaojurizhi"],
    },
    "stop_listen": {
        "zh": ["关闭识别"],
        "pinyin": ["guanbishibie"],
    },
    "shutdown": {
        "zh": ["退出程序", "关闭程序", "退出"],
        "pinyin": ["tuichuchengxu", "guanbichengxu", "tuichu"],
    },
}


def _normalize(text: str) -> str:
    """去掉空白与声调数字，统一小写，供别名匹配"""
    return re.sub(r"[\s\d]+", "", text or "").lower()


def match_command_key(text: str) -> str | None:
    """把识别文本映射到指令 key（中文/拼音别名均可）"""
    normalized = _normalize(text)
    if not normalized:
        return None
    for key, alias in COMMAND_ALIASES.items():
        for a in alias.get("zh", []) + alias.get("pinyin", []):
            if _normalize(a) in normalized:
                return key
    return None


# ── PocketSphinx 识别器 ───────────────────────────────────────


class SphinxCommandRecognizer:
    """
    PocketSphinx 指令识别器（JSGF 语法约束）

    依赖（板端）：
        - apt install pocketsphinx python3-pocketsphinx
        - backend/models/pocketsphinx/zh-cn 声学模型
        - backend/models/pocketsphinx/cmudict-cn.dict 词典
        - backend/models/pocketsphinx/commands.jsgf 指令语法
    """

    def __init__(self, model_dir: str, dict_path: str, jsgf_path: str) -> None:
        self._model_dir = model_dir
        self._dict_path = dict_path
        self._jsgf_path = jsgf_path
        self._decoder = None
        self._available = False
        self._load()

    def _load(self) -> None:
        for path in (self._model_dir, self._dict_path, self._jsgf_path):
            if not Path(path).exists():
                logger.warning(
                    "PocketSphinx 模型文件缺失: %s（参考 "
                    "backend/models/pocketsphinx/README.md 下载放置）", path,
                )
                return
        try:
            import pocketsphinx  # noqa: PLC0415 延迟导入：可选依赖
        except ImportError:
            logger.warning(
                "pocketsphinx 未安装（apt install python3-pocketsphinx），"
                "指令识别停用"
            )
            return
        try:
            config = pocketsphinx.Decoder.default_config()
            # 关键：default_config 预置了英文 LM，与 -jsgf 互斥
            #（"Only one of lm, jsgf, ... can be enabled"），必须显式清空
            config.set_string("-lm", None)
            config.set_string("-hmm", self._model_dir)
            config.set_string("-dict", self._dict_path)
            config.set_string("-jsgf", self._jsgf_path)
            self._decoder = pocketsphinx.Decoder(config)
            self._available = True
            logger.info("SphinxCommandRecognizer ready (jsgf=%s)", self._jsgf_path)
        except Exception:
            logger.exception("PocketSphinx 初始化失败，指令识别停用")

    @property
    def is_available(self) -> bool:
        return self._available

    def recognize(self, pcm: bytes) -> str | None:
        """
        识别一段完整指令音频（16kHz/16bit/单声道 PCM）。

        Returns:
            识别出的指令文本（词典词条序列），无结果返回 None
        """
        if not self._available or not pcm:
            return None
        try:
            self._decoder.start_utt()
            # full_utt=True：整段按一次完整语句解码
            self._decoder.process_raw(pcm, False, True)
            self._decoder.end_utt()
            hyp = self._decoder.hyp()
            if hyp and hyp.hypstr:
                text = hyp.hypstr.strip()
                logger.info("指令识别结果: %s", text)
                return text
            return None
        except Exception:
            logger.exception("PocketSphinx 解码异常")
            return None


# ── 业务分发器 ────────────────────────────────────────────────


class TextCommandDispatcher:
    """
    指令 key → 业务回调分发（纯同步，在语音线程内执行）。

    回调约定（由 Pipeline 注入，均为可选，全部为同步可调用对象；
    阻塞型回调如串口读取只阻塞语音线程，不影响 Pipeline 主循环）：
        - set_mode(mode: str)：normal / nap / public
        - ack_all()：批量确认告警
        - get_sensor_data()：返回 (temp, hum) 或 None
        - start_record()：开始录制，返回提示文本（可选）
        - get_recent_logs()：返回日志摘要文本（可选）
        - shutdown()：安全退出（内部自行调度到主事件循环）

    handle() 返回 TTS 播报文本；返回 None 表示未知指令。
    """

    def __init__(self, callbacks: dict[str, Callable[..., Any]] | None = None) -> None:
        self._callbacks = callbacks or {}

    def _call(self, name: str, *args, default=None):
        cb = self._callbacks.get(name)
        if cb is None:
            return default
        try:
            return cb(*args)
        except Exception:
            logger.exception("指令回调执行失败: %s", name)
            return default

    def handle(self, command_key: str) -> str | None:
        """执行指令 key 对应的业务，返回 TTS 播报文本（同步，语音线程内执行）"""
        if command_key == "set_mode_normal":
            self._call("set_mode", "normal")
            return "已切换到正常课堂模式"

        if command_key == "set_mode_nap":
            self._call("set_mode", "nap")
            return "已切换到午睡模式"

        if command_key == "set_mode_public":
            self._call("set_mode", "public")
            return "已切换到公共区域模式"

        if command_key == "prompt_mode":
            return "请说出你需要切换的模式"

        if command_key == "ack_all":
            self._call("ack_all")
            return "已关闭当前告警"

        if command_key == "get_temp":
            data = self._call("get_sensor_data")
            if data is None:
                return "传感器异常，无法查询温湿度"
            temp, hum = data
            return f"当前环境温度{temp}摄氏度，湿度{hum}%"

        if command_key == "start_record":
            hint = self._call("start_record")
            return hint or "视频录制已开启"

        if command_key == "get_logs":
            summary = self._call("get_recent_logs")
            return summary or "正在读取设备运行日志"

        if command_key == "stop_listen":
            # 由 VoiceService 捕获后回到待机
            return "__STOP_LISTEN__"

        if command_key == "shutdown":
            self._call("shutdown")
            return "AI监护已关闭"

        return None
