"""
语音模块单元测试（voice/）

覆盖：
    - match_command_key：中文/拼音/声调归一化/拒识
    - TextCommandDispatcher：全部指令的回调分发与 TTS 播报文本
    - TTS 工厂与 Dummy 后端
    - WakeEngine / SphinxCommandRecognizer：模型缺失时安静降级
    - VoiceService：禁用/降级状态机
    - _FrameRecorder：录制产出 mp4 文件
    - Pipeline 语音回调方法：模式切换/告警确认/日志摘要/录制/关机标志
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import AppConfig
from loongguard.pipeline import Pipeline, _FrameRecorder
from loongguard.voice import (
    DummyTTS,
    SphinxCommandRecognizer,
    TextCommandDispatcher,
    VoiceService,
    WakeEngine,
    create_tts_backend,
    match_command_key,
)

# ── match_command_key ────────────────────────────────────────


class TestMatchCommandKey:
    def test_chinese_full_commands(self) -> None:
        assert match_command_key("切换到正常课堂模式") == "set_mode_normal"
        assert match_command_key("切换到午睡模式") == "set_mode_nap"
        assert match_command_key("切换到公共区域模式") == "set_mode_public"
        assert match_command_key("关闭告警") == "ack_all"
        assert match_command_key("查询温湿度") == "get_temp"
        assert match_command_key("开始录制") == "start_record"
        assert match_command_key("查看日志") == "get_logs"
        assert match_command_key("关闭识别") == "stop_listen"
        assert match_command_key("退出程序") == "shutdown"

    def test_chinese_with_noise(self) -> None:
        """识别文本常带空白与标点，须能容错"""
        assert match_command_key("  请 关闭告警 。 ") == "ack_all"
        assert match_command_key("嗯，切换到午睡模式吧") == "set_mode_nap"

    def test_pinyin_with_tones(self) -> None:
        """CMU 拼音词典输出带声调数字，须归一化后匹配"""
        assert match_command_key("guan1 bi4 gao4 jing3") == "ack_all"
        assert match_command_key("qie1 huan4 dao4 wu3 shui4 mo2 shi4") == "set_mode_nap"

    def test_pinyin_without_tones(self) -> None:
        assert match_command_key("guanbigaojing") == "ack_all"
        assert match_command_key("kaishiluzhi") == "start_record"

    def test_unknown_text(self) -> None:
        assert match_command_key("今天天气怎么样") is None
        assert match_command_key("") is None
        assert match_command_key(None) is None


# ── TextCommandDispatcher ────────────────────────────────────


class TestTextCommandDispatcher:
    def _make(self, callbacks=None):
        return TextCommandDispatcher(callbacks or {})

    def test_mode_switch_calls_callback(self) -> None:
        called = []
        d = self._make({"set_mode": called.append})
        assert d.handle("set_mode_nap") == "已切换到午睡模式"
        assert called == ["nap"]

    def test_prompt_mode(self) -> None:
        assert self._make().handle("prompt_mode") == "请说出你需要切换的模式"

    def test_get_temp_with_data(self) -> None:
        d = self._make({"get_sensor_data": lambda: (26.5, 58.0)})
        reply = d.handle("get_temp")
        assert "26.5" in reply and "58.0" in reply

    def test_get_temp_sensor_missing(self) -> None:
        """传感器不可用必须播报异常（需求：无传感器时提示）"""
        d = self._make({"get_sensor_data": lambda: None})
        assert "传感器异常" in d.handle("get_temp")

    def test_get_temp_callback_absent(self) -> None:
        assert "传感器异常" in self._make().handle("get_temp")

    def test_start_record_hint(self) -> None:
        d = self._make({"start_record": lambda: "视频录制已开启"})
        assert d.handle("start_record") == "视频录制已开启"

    def test_get_logs_summary(self) -> None:
        d = self._make({"get_recent_logs": lambda: "最近共3条告警"})
        assert "最近共3条告警" in d.handle("get_logs")

    def test_stop_listen_sentinel(self) -> None:
        assert d_handle_stop() == "__STOP_LISTEN__"

    def test_shutdown_callback(self) -> None:
        called = []
        d = self._make({"shutdown": lambda: called.append(True)})
        assert d.handle("shutdown") == "AI监护已关闭"
        assert called == [True]

    def test_unknown_key(self) -> None:
        assert self._make().handle("nonexistent") is None

    def test_callback_exception_does_not_propagate(self) -> None:
        """回调抛异常时返回默认文本，不能炸掉语音线程"""
        def boom():
            raise RuntimeError("sensor exploded")

        d = self._make({"get_sensor_data": boom})
        assert "传感器异常" in d.handle("get_temp")


def d_handle_stop() -> str:
    return TextCommandDispatcher().handle("stop_listen")


# ── TTS 工厂 ────────────────────────────────────────────────


class TestTTSBackend:
    def test_dummy_factory(self) -> None:
        tts = create_tts_backend("dummy")
        assert isinstance(tts, DummyTTS)
        tts.speak("测试")  # 不抛异常

    def test_unknown_platform_falls_back_dummy(self) -> None:
        assert isinstance(create_tts_backend("nonexistent-platform"), DummyTTS)


# ── 引擎降级 ────────────────────────────────────────────────


class TestEngineGracefulDegradation:
    """缺依赖/缺模型时必须安静降级（不抛异常、不阻塞 Pipeline）"""

    def test_wake_engine_missing_model(self, tmp_path: Path) -> None:
        engine = WakeEngine(str(tmp_path / "no_such.onnx"), threshold=0.5)
        assert engine.is_available is False
        assert engine.predict(b"\x00" * 3200) is False

    def test_sphinx_missing_models(self, tmp_path: Path) -> None:
        recognizer = SphinxCommandRecognizer(
            model_dir=str(tmp_path / "no_zh_cn"),
            dict_path=str(tmp_path / "no.dict"),
            jsgf_path=str(tmp_path / "no.jsgf"),
        )
        assert recognizer.is_available is False
        assert recognizer.recognize(b"\x00" * 3200) is None


# ── VoiceService 生命周期 ───────────────────────────────────


class TestVoiceServiceLifecycle:
    def _config(self, **overrides):
        from config.settings import VoiceConfig

        return VoiceConfig(**overrides)

    @pytest.mark.asyncio
    async def test_disabled_by_config(self) -> None:
        """enabled=False 时 start() 直接跳过，不启动线程"""
        service = VoiceService(self._config(enabled=False), tts=DummyTTS())
        await service.start()
        assert service.is_running is False
        await service.stop()

    @pytest.mark.asyncio
    async def test_missing_wake_model_degrades(self, tmp_path: Path) -> None:
        """唤醒模型缺失时不启动语音线程，但 start/stop 不抛异常"""
        service = VoiceService(
            self._config(wake_model_path=str(tmp_path / "no.onnx")), tts=DummyTTS()
        )
        await service.start()
        assert service.is_running is False
        status = service.status()
        assert status["wake_engine"] is False
        await service.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        service = VoiceService(self._config(enabled=False), tts=DummyTTS())
        await service.start()
        await service.stop()
        await service.stop()  # 幂等


# ── 帧录制器 ────────────────────────────────────────────────


class TestFrameRecorder:
    def test_records_to_mp4(self, tmp_path: Path) -> None:
        """录制 10 帧后产出非空 mp4 文件"""
        import numpy as np

        from loongguard.camera.v4l2_capture import Frame

        path = tmp_path / "rec_test.mp4"
        recorder = _FrameRecorder(str(path), width=64, height=48, fps=10, max_sec=60)
        for i in range(10):
            frame = Frame(
                data=np.full((48, 64, 3), i * 5 % 255, dtype=np.uint8),
                timestamp=float(i), frame_id=i,
            )
            recorder.write(frame)
        recorder.stop()
        assert path.exists()
        assert path.stat().st_size > 0
        assert recorder.frame_count == 10

    def test_expired_flag(self, tmp_path: Path) -> None:
        recorder = _FrameRecorder(
            str(tmp_path / "r.mp4"), width=64, height=48, fps=10, max_sec=0.0)
        assert recorder.expired is True
        recorder.stop()


# ── Pipeline 语音回调 ───────────────────────────────────────


@pytest.fixture
def app_config(monkeypatch) -> AppConfig:
    monkeypatch.setenv("LG_SM4_KEY", "0123456789abcdef0123456789abcdef")
    config = AppConfig()
    config.voice.enabled = False  # 测试中不真正启动语音线程
    return config


@pytest.fixture
def pipeline(app_config) -> Pipeline:
    return Pipeline(app_config)


class TestPipelineVoiceCallbacks:
    def test_pipeline_holds_voice_service(self, pipeline) -> None:
        assert isinstance(pipeline._voice, VoiceService)

    def test_set_mode(self, pipeline) -> None:
        pipeline.set_mode("nap")
        assert pipeline.current_mode == "nap"
        pipeline.set_mode("public")
        assert pipeline.current_mode == "public"
        pipeline.set_mode("invalid-mode")  # 未知模式忽略
        assert pipeline.current_mode == "public"

    def test_voice_shutdown_stops_running(self, pipeline) -> None:
        pipeline._running = True
        pipeline._voice_shutdown()
        assert pipeline._running is False

    def test_get_recent_logs_without_db(self, pipeline) -> None:
        pipeline._db = None
        assert "日志服务未启动" in pipeline.get_recent_logs()

    def test_ack_all_without_db(self, pipeline) -> None:
        pipeline._db = None
        assert pipeline.ack_all_alerts() == 0

    def test_get_sensor_data_unavailable(self, pipeline) -> None:
        """minimalmodbus 未安装（后端 venv）时返回 None → 播报传感器异常"""
        assert pipeline.get_sensor_data() is None

    def test_recording_roundtrip(self, pipeline, tmp_path: Path) -> None:
        """start_recording 返回提示文本并创建录制器，stop 幂等"""
        hint = pipeline.start_recording()
        assert "录制" in hint
        assert pipeline._recorder is not None
        pipeline.stop_recording()
        assert pipeline._recorder is None
        pipeline.stop_recording()  # 幂等

    def test_recording_twice_returns_hint(self, pipeline) -> None:
        pipeline.start_recording()
        assert "已在进行中" in pipeline.start_recording()
        pipeline.stop_recording()
