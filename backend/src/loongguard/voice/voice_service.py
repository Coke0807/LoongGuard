"""
语音服务主控（VoiceService）

职责：
    整合 WakeEngine（唤醒）+ SphinxCommandRecognizer（指令）
    + TextCommandDispatcher（业务分发）+ TTSBackend（播报），
    作为独立后台线程运行，不阻塞 Pipeline 主循环。

状态机：
    standby（待机，检测唤醒词）
      └─ 唤醒命中 → active（播报"我在"，循环听指令）
            ├─ 识别到指令 → 分发 + TTS 播报
            ├─ "关闭识别" / 超时 → standby
            └─ "退出程序" → 触发 shutdown 回调

线程模型：
    - 语音线程内做音频采集/推理/TTS/业务回调（均可阻塞，
      不影响 Pipeline 主事件循环）
    - 仅 shutdown（安全退出）由 Pipeline 回调内部通过
      run_coroutine_threadsafe 调度回主事件循环
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable

from loongguard.voice.command_handler import (
    SphinxCommandRecognizer,
    TextCommandDispatcher,
    match_command_key,
)
from loongguard.voice.tts_backend import TTSBackend, create_tts_backend
from loongguard.voice.wake_engine import (
    BYTES_PER_SEC,
    AudioCapture,
    WakeEngine,
    create_capture,
)

logger = logging.getLogger(__name__)

# "关闭识别"指令的哨兵返回值（见 TextCommandDispatcher.handle）
_STOP_LISTEN = "__STOP_LISTEN__"


class VoiceService:
    """
    语音服务生命周期管理。

    Args:
        config: VoiceConfig（wake/sphinx 模型路径、阈值、超时等）
        callbacks: 业务回调字典（见 TextCommandDispatcher 文档）

    降级行为：
        - config.enabled=False → start() 直接跳过
        - 唤醒引擎不可用（缺依赖/模型）→ 记录警告，服务保持待机
        - 指令识别器不可用 → 唤醒后播报提示但无法解析指令
    """

    def __init__(self, config, callbacks: dict[str, Callable[..., Any]] | None = None,
                 tts: TTSBackend | None = None) -> None:
        self._config = config
        self._tts = tts or create_tts_backend(getattr(config, "tts_backend", "auto"))
        self._dispatcher = TextCommandDispatcher(callbacks)
        self._thread: threading.Thread | None = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # 引擎延迟到 start() 构造，避免 Pipeline 启动早期被阻塞
        self._wake_engine: WakeEngine | None = None
        self._recognizer: SphinxCommandRecognizer | None = None
        self._capture: AudioCapture | None = None

    # ── 生命周期 ────────────────────────────────────────────

    async def start(self) -> None:
        """启动语音服务（不可用时安静降级，不抛异常）"""
        if not getattr(self._config, "enabled", True):
            logger.info("Voice module disabled by config")
            return

        self._loop = asyncio.get_running_loop()
        self._wake_engine = WakeEngine(
            model_path=self._config.wake_model_path,
            threshold=self._config.wake_threshold,
            cooldown_sec=self._config.wake_cooldown_sec,
        )
        self._recognizer = SphinxCommandRecognizer(
            model_dir=self._config.sphinx_model_dir,
            dict_path=self._config.sphinx_dict_path,
            jsgf_path=self._config.sphinx_jsgf_path,
        )

        if not self._wake_engine.is_available:
            logger.warning(
                "唤醒引擎不可用，语音服务保持停用（不影响主链路）"
            )
            return

        self._capture = create_capture()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="voice-service", daemon=True
        )
        self._thread.start()
        logger.info("Voice service started (wake=%s, sphinx=%s)",
                    self._wake_engine.is_available, self._recognizer.is_available)
        self._speak("AI监护已启动")

    async def stop(self) -> None:
        """停止语音服务（幂等）"""
        if not self._running and self._thread is None:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._capture is not None:
            self._capture.close()
            self._capture = None
        self._speak("AI监护已关闭")
        logger.info("Voice service stopped")

    # ── 状态 ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        """语音模块状态快照（供健康检查/调试）"""
        return {
            "enabled": bool(getattr(self._config, "enabled", True)),
            "running": self.is_running,
            "wake_engine": bool(self._wake_engine and self._wake_engine.is_available),
            "recognizer": bool(self._recognizer and self._recognizer.is_available),
            "tts": type(self._tts).__name__,
        }

    # ── 主循环（语音线程内执行）─────────────────────────────

    def _run_loop(self) -> None:
        assert self._capture is not None and self._wake_engine is not None
        # 唤醒检测每次消费 0.5 秒音频（openWakeWord 内部按 80ms 帧缓冲）
        wake_chunk = BYTES_PER_SEC // 2
        command_bytes = int(BYTES_PER_SEC * self._config.command_duration_sec)
        timeout_sec = self._config.wake_timeout_sec

        while self._running and self._capture.is_running:
            pcm = self._capture.read(wake_chunk)
            if not self._wake_engine.predict(pcm):
                continue

            # ── 唤醒命中，进入指令阶段 ──
            self._speak("我在")
            self._wake_engine.reset()
            active_until = time.monotonic() + timeout_sec
            while self._running and time.monotonic() < active_until:
                cmd_pcm = self._capture.read(command_bytes)
                text = (self._recognizer.recognize(cmd_pcm)
                        if self._recognizer.is_available else None)
                if not text:
                    continue
                key = match_command_key(text)
                if key is None:
                    self._speak("我没有听清，请再说一遍")
                    continue
                reply = self._dispatch(key)
                if reply == _STOP_LISTEN:
                    self._speak("已回到待机，唤醒请说小龙小龙")
                    break
                if reply:
                    self._speak(reply)
                if key == "shutdown":
                    return
                # 指令处理后刷新超时窗口，允许连续下指令
                active_until = time.monotonic() + timeout_sec
            else:
                logger.info("唤醒超时，自动返回待机")

        logger.info("Voice loop exited")

    # ── 辅助 ────────────────────────────────────────────────

    def _speak(self, text: str) -> None:
        try:
            self._tts.speak(text)
        except Exception:
            logger.exception("TTS 播报异常: %s", text)

    def _dispatch(self, key: str) -> str | None:
        """指令分发（同步执行于语音线程，阻塞不影响主循环）"""
        return self._dispatcher.handle(key)
