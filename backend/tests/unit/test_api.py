"""
后端 API 模块单元测试

覆盖：
    - AlertAPIServer 启动与停止
    - GET /api/v1/status 返回设备状态
    - GET /api/v1/alerts 返回告警历史
    - POST /api/v1/alerts/:id/ack 消警功能
    - WebSocket /ws/alerts 连接与推送
    - push_alert 广播到多个 WebSocket 客户端
    - 断开的客户端自动清理
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

# aiohttp 可选依赖检测
aiohttp = pytest.importorskip("aiohttp", reason="aiohttp is required for API tests")

from config.settings import APIConfig  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.api.server import (  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
    AlertAPIServer,
    AnalysisStatus,
    VideoAnalyzer,
    _TokenBucketRateLimiter,
)
from loongguard.utils.schema import (  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
    AlertLog,
    AlertSeverity,
    AlertType,
)

# ── 辅助函数 ─────────────────────────────────────────────────


def _make_alert(
    alert_id: str = "api_test_001",
    severity: AlertSeverity = AlertSeverity.HIGH,
    description: str = "测试：磁力珠检测",
) -> AlertLog:
    """构造测试告警"""
    return AlertLog(
        alert_id=alert_id,
        alert_type=AlertType.DANGEROUS_OBJECT,
        severity=severity,
        description=description,
    )


@pytest.fixture
def api_config() -> APIConfig:
    """返回使用随机端口的 API 配置（port=0 让 OS 分配空闲端口）"""
    return APIConfig(host="127.0.0.1", port=0)


@pytest_asyncio.fixture
async def server(api_config: APIConfig):
    """
    启动 AlertAPIServer 并在测试结束后清理。

    使用 port=0 绑定随机端口避免冲突，
    返回 (server_instance, base_url) 元组。
    """
    srv = AlertAPIServer(api_config)
    await srv.start()

    # 从 runner 中获取实际绑定的端口
    # addresses 返回 (host, port) 元组列表，port=0 时 OS 自动分配空闲端口
    actual_port = srv._runner.addresses[0][1] if srv._runner.addresses else 0
    base_url = f"http://127.0.0.1:{actual_port}"

    yield srv, base_url

    await srv.stop()


# ── 服务启停测试 ─────────────────────────────────────────────


class TestServerLifecycle:
    """服务生命周期测试"""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, api_config: APIConfig) -> None:
        """服务应能正常启动和停止"""
        srv = AlertAPIServer(api_config)
        await srv.start()

        assert srv._app is not None
        assert srv._runner is not None

        await srv.stop()
        assert srv._runner is None

    @pytest.mark.asyncio
    async def test_stop_clears_ws_clients(self, api_config: APIConfig) -> None:
        """停止时应清理所有 WebSocket 客户端引用"""
        srv = AlertAPIServer(api_config)
        # 模拟已连接的客户端
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()
        srv._ws_clients.append(mock_ws)

        await srv.start()
        await srv.stop()

        assert len(srv._ws_clients) == 0
        mock_ws.close.assert_awaited_once()


# ── REST 端点测试 ────────────────────────────────────────────


class TestRESTEndpoints:
    """REST API 端点测试"""

    @pytest.mark.asyncio
    async def test_get_status_returns_running(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """GET /api/v1/status 应返回设备状态 JSON"""
        _, base_url = server
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/v1/status") as resp:
                assert resp.status == 200
                data = await resp.json()

                assert data["status"] == "running"
                assert "YOLO26-Nano" in data["model"]
                assert "total_alerts" in data

    @pytest.mark.asyncio
    async def test_get_status_shows_alert_count(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """推送告警后，status 中的 total_alerts 应递增"""
        srv, base_url = server

        # 推送两条告警
        await srv.push_alert(_make_alert("cnt_001"))
        await srv.push_alert(_make_alert("cnt_002"))

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/v1/status") as resp:
                data = await resp.json()
                assert data["total_alerts"] == 2

    @pytest.mark.asyncio
    async def test_get_alerts_returns_empty_list_initially(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """初始状态下 GET /api/v1/alerts 应返回空数据"""
        _, base_url = server
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/v1/alerts") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["total"] == 0
                assert data["page"] == 1
                assert isinstance(data["data"], list)
                assert len(data["data"]) == 0

    @pytest.mark.asyncio
    async def test_get_alerts_returns_pushed_alerts(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """推送告警后 GET /api/v1/alerts 应返回对应数据"""
        srv, base_url = server

        alert = _make_alert("hist_001", description="纽扣电池检测")
        await srv.push_alert(alert)

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/v1/alerts") as resp:
                data = await resp.json()
                assert data["total"] == 1
                assert len(data["data"]) == 1
                assert data["data"][0]["alert_id"] == "hist_001"
                assert data["data"][0]["description"] == "纽扣电池检测"

    @pytest.mark.asyncio
    async def test_ack_alert_marks_as_acknowledged(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """POST /api/v1/alerts/:id/ack 应将告警标记为已确认"""
        srv, base_url = server

        alert = _make_alert("ack_001")
        await srv.push_alert(alert)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/v1/alerts/ack_001/ack"
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["ok"] is True

        # 验证内部状态已更新
        assert srv._alert_history[0].acknowledged is True

    @pytest.mark.asyncio
    async def test_ack_nonexistent_alert_returns_404(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """对不存在的告警 ID 消警应返回 404"""
        _, base_url = server
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/v1/alerts/nonexistent_id/ack"
            ) as resp:
                assert resp.status == 404
                data = await resp.json()
                assert "error" in data


# ── 健康检查端点测试 ─────────────────────────────────────────


class TestHealthEndpoints:
    """可观测性：/health 与 /health/pose 端点"""

    @pytest.mark.asyncio
    async def test_health_overview_returns_running(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """GET /health 应返回 running 状态 + 组件注入情况"""
        _, base_url = server
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "running"
                assert "components" in data
                assert "pose" in data["components"]

    @pytest.mark.asyncio
    async def test_health_pose_not_injected_returns_503(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """未注入 pose provider 时 /health/pose 返回 503"""
        _, base_url = server
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health/pose") as resp:
                assert resp.status == 503
                data = await resp.json()
                assert data["available"] is False

    @pytest.mark.asyncio
    async def test_health_pose_injected_available(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """注入可用 pose provider 时 /health/pose 返回 200 + 完整快照"""
        srv, base_url = server
        # 注入一个 mock provider
        srv.set_pose_health_provider(lambda: {
            "available": True,
            "model_path": "models/movenet_lightning_int8.onnx",
            "last_inference_ts": 12345.6,
            "success_count": 42,
            "error_count": 0,
            "last_error": "",
            "last_run_frame": 120,
        })

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health/pose") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["available"] is True
                assert data["success_count"] == 42
                assert data["model_path"].endswith(".onnx")

    @pytest.mark.asyncio
    async def test_health_pose_injected_unavailable_returns_503(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """注入的 provider 报告 available=False 时也应返回 503"""
        srv, base_url = server
        srv.set_pose_health_provider(lambda: {
            "available": False,
            "model_path": "missing.onnx",
            "last_inference_ts": 0.0,
            "success_count": 0,
            "error_count": 0,
            "last_error": "Model file not found",
            "last_run_frame": -1,
        })

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health/pose") as resp:
                assert resp.status == 503
                data = await resp.json()
                assert data["available"] is False
                assert "not found" in data["last_error"]

    @pytest.mark.asyncio
    async def test_health_pose_provider_exception_returns_503(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """provider 抛异常时返回 503 而非 500,保证监控可探测"""
        srv, base_url = server
        def bad_provider():
            raise RuntimeError("health boom")
        srv.set_pose_health_provider(bad_provider)

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health/pose") as resp:
                assert resp.status == 503
                data = await resp.json()
                assert "health error" in data["reason"]

    @pytest.mark.asyncio
    async def test_health_detection_not_injected_returns_503(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """未注入 detection provider 时 /health/detection 返回 503"""
        _, base_url = server
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health/detection") as resp:
                assert resp.status == 503
                data = await resp.json()
                assert data["available"] is False
                assert "detection module" in data["reason"]

    @pytest.mark.asyncio
    async def test_health_detection_injected_available(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """注入可用 detection provider 时 /health/detection 返回 200 + 快照"""
        srv, base_url = server
        srv.set_detection_health_provider(lambda: {
            "available": True,
            "model_path": "models/best.onnx",
            "input_size": 640,
            "conf_threshold": 0.3,
            "last_inference_ts": 99.9,
            "success_count": 100,
            "error_count": 0,
            "last_error": "",
        })

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health/detection") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["available"] is True
                assert data["input_size"] == 640
                assert data["success_count"] == 100

    @pytest.mark.asyncio
    async def test_health_detection_injected_unavailable_returns_503(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """detection provider 报告 available=False 时返回 503"""
        srv, base_url = server
        srv.set_detection_health_provider(lambda: {
            "available": False,
            "model_path": "missing.onnx",
            "input_size": 640,
            "conf_threshold": 0.3,
            "last_inference_ts": 0.0,
            "success_count": 0,
            "error_count": 1,
            "last_error": "FileNotFound",
        })

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health/detection") as resp:
                assert resp.status == 503
                data = await resp.json()
                assert data["available"] is False

    @pytest.mark.asyncio
    async def test_health_overview_aggregates_components(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """GET /health 应聚合 pose + detection 两个子端点状态"""
        srv, base_url = server
        srv.set_pose_health_provider(lambda: {
            "available": True, "model_path": "p.onnx",
            "last_inference_ts": 0.0, "success_count": 0,
            "error_count": 0, "last_error": "", "last_run_frame": None,
        })
        srv.set_detection_health_provider(lambda: {
            "available": True, "model_path": "d.onnx",
            "input_size": 640, "conf_threshold": 0.3,
            "last_inference_ts": 0.0, "success_count": 0,
            "error_count": 0, "last_error": "",
        })

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "running"
                assert data["components"]["pose"] == "ok"
                assert data["components"]["detection"] == "ok"

    @pytest.mark.asyncio
    async def test_health_overview_degraded_when_one_component_down(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """任一子组件不可用时,/health 返回 503 + status=degraded"""
        srv, base_url = server
        # pose OK
        srv.set_pose_health_provider(lambda: {
            "available": True, "model_path": "p.onnx",
            "last_inference_ts": 0.0, "success_count": 0,
            "error_count": 0, "last_error": "", "last_run_frame": None,
        })
        # detection DOWN
        srv.set_detection_health_provider(lambda: {
            "available": False, "model_path": "d.onnx",
            "input_size": 640, "conf_threshold": 0.3,
            "last_inference_ts": 0.0, "success_count": 0,
            "error_count": 1, "last_error": "load failed",
        })

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                assert resp.status == 503
                data = await resp.json()
                assert data["status"] == "degraded"
                assert data["components"]["pose"] == "ok"
                assert data["components"]["detection"] == "degraded"


# ── WebSocket 推送测试 ───────────────────────────────────────


class TestWebSocketPush:
    """WebSocket 告警推送测试"""

    @pytest.mark.asyncio
    async def test_ws_client_receives_pushed_alert(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """已连接的 WebSocket 客户端应收到推送的告警"""
        srv, base_url = server
        ws_url = base_url.replace("http://", "ws://") + "/ws/alerts"

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # 等待连接建立
                assert len(srv._ws_clients) == 1

                alert = _make_alert("ws_push_001", description="碎玻璃检测")
                await srv.push_alert(alert)

                msg = await asyncio.wait_for(ws.receive(), timeout=3.0)
                assert msg.type == aiohttp.WSMsgType.TEXT

                received = json.loads(msg.data)
                assert received["alert_id"] == "ws_push_001"
                assert received["description"] == "碎玻璃检测"

    @pytest.mark.asyncio
    async def test_ws_multiple_clients_receive_alert(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """多个 WebSocket 客户端都应收到同一告警"""
        srv, base_url = server
        ws_url = base_url.replace("http://", "ws://") + "/ws/alerts"

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws1:
                async with session.ws_connect(ws_url) as ws2:
                    assert len(srv._ws_clients) == 2

                    alert = _make_alert("multi_ws_001")
                    await srv.push_alert(alert)

                    msg1 = await asyncio.wait_for(ws1.receive(), timeout=3.0)
                    msg2 = await asyncio.wait_for(ws2.receive(), timeout=3.0)

                    data1 = json.loads(msg1.data)
                    data2 = json.loads(msg2.data)
                    assert data1["alert_id"] == "multi_ws_001"
                    assert data2["alert_id"] == "multi_ws_001"

    @pytest.mark.asyncio
    async def test_push_alert_with_no_clients_does_not_raise(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """无 WebSocket 客户端时推送告警不应抛出异常"""
        srv, _ = server
        # 确保无客户端
        assert len(srv._ws_clients) == 0

        # 不应抛出异常
        alert = _make_alert("no_client_001")
        await srv.push_alert(alert)
        assert len(srv._alert_history) == 1

    @pytest.mark.asyncio
    async def test_push_alert_appends_to_history(
        self, server: tuple[AlertAPIServer, str]
    ) -> None:
        """每次 push_alert 都应将告警追加到历史记录"""
        srv, _ = server

        for i in range(5):
            await srv.push_alert(_make_alert(f"hist_{i:03d}"))

        assert len(srv._alert_history) == 5
        assert srv._alert_history[-1].alert_id == "hist_004"


# ── 令牌桶限流器测试 ────────────────────────────────────────


class TestTokenBucketRateLimiter:
    """_TokenBucketRateLimiter 单元测试"""

    def test_allows_within_burst(self) -> None:
        """突发容量内的请求应全部放行"""
        limiter = _TokenBucketRateLimiter(rate=10.0, burst=5.0)
        results = [limiter.allow("1.2.3.4") for _ in range(5)]
        assert all(results)

    def test_rejects_after_burst_exhausted(self) -> None:
        """耗尽突发容量后应立即拒绝"""
        limiter = _TokenBucketRateLimiter(rate=10.0, burst=3.0)
        for _ in range(3):
            assert limiter.allow("1.2.3.4")
        assert not limiter.allow("1.2.3.4")

    def test_refills_over_time(self) -> None:
        """令牌应按速率随时间补充"""
        limiter = _TokenBucketRateLimiter(rate=100.0, burst=1.0)
        assert limiter.allow("1.2.3.4")
        assert not limiter.allow("1.2.3.4")
        # 100 rps 下 20ms 补充 2 个令牌
        import time as _time
        _time.sleep(0.03)
        assert limiter.allow("1.2.3.4")

    def test_independent_buckets_per_ip(self) -> None:
        """不同客户端 IP 的桶相互独立"""
        limiter = _TokenBucketRateLimiter(rate=1.0, burst=1.0)
        assert limiter.allow("1.1.1.1")
        assert not limiter.allow("1.1.1.1")
        # 另一个 IP 不受影响
        assert limiter.allow("2.2.2.2")

    def test_disabled_when_rate_zero(self) -> None:
        """rate<=0 时 enabled 为 False（限流关闭）"""
        assert not _TokenBucketRateLimiter(rate=0.0, burst=10.0).enabled
        assert _TokenBucketRateLimiter(rate=5.0, burst=10.0).enabled


# ── 速率限制中间件集成测试 ──────────────────────────────────


class TestRateLimitMiddleware:
    """中间件层限流行为测试（真实 HTTP 请求）"""

    @pytest.mark.asyncio
    async def test_exceeding_rate_limit_returns_429(self) -> None:
        """超出突发容量的连续请求应返回 429"""
        cfg = APIConfig(host="127.0.0.1", port=0)
        cfg.rate_limit_rps = 1.0
        cfg.rate_limit_burst = 3.0
        srv = AlertAPIServer(cfg)
        await srv.start()
        try:
            port = srv._runner.addresses[0][1]
            statuses = []
            async with aiohttp.ClientSession() as session:
                for _ in range(6):
                    async with session.get(
                        f"http://127.0.0.1:{port}/api/v1/status"
                    ) as resp:
                        statuses.append(resp.status)
            assert 429 in statuses
            assert statuses[:3] == [200, 200, 200]
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_metrics_exempt_from_rate_limit(self) -> None:
        """Prometheus 抓取端点 /metrics 不受限流影响"""
        cfg = APIConfig(host="127.0.0.1", port=0)
        cfg.rate_limit_rps = 1.0
        cfg.rate_limit_burst = 1.0
        srv = AlertAPIServer(cfg)
        await srv.start()
        try:
            port = srv._runner.addresses[0][1]
            async with aiohttp.ClientSession() as session:
                for _ in range(5):
                    async with session.get(
                        f"http://127.0.0.1:{port}/metrics"
                    ) as resp:
                        assert resp.status == 200
        finally:
            await srv.stop()


# ── VideoAnalyzer 注入与线程池分析测试 ──────────────────────


class _FakeDetector:
    """满足 DetectorLike 协议的假检测器"""

    def infer(self, image: np.ndarray) -> list:
        return []


class _FakeMotion:
    """满足 MotionDetectorLike 协议的假运动检测器（恒无运动）"""

    def detect(self, current_gray: np.ndarray, prev_gray):
        return []


class _FakeROIScheduler:
    """满足 ROISchedulerLike 协议的假 ROI 调度器"""

    def process(self, frame: np.ndarray, motion_regions: list) -> list:
        return []


class TestVideoAnalyzerInjection:
    """set_modules Protocol 运行期校验测试"""

    def test_valid_modules_accepted(self) -> None:
        """满足协议的模块应成功注入"""
        analyzer = VideoAnalyzer()
        analyzer.set_modules(
            detector=_FakeDetector(),
            motion=_FakeMotion(),
            roi_scheduler=_FakeROIScheduler(),
        )
        assert analyzer._detector is not None
        assert analyzer._motion is not None
        assert analyzer._roi_scheduler is not None

    def test_invalid_detector_rejected(self) -> None:
        """不满足 DetectorLike 的对象应在注入时立即报 TypeError"""
        analyzer = VideoAnalyzer()
        with pytest.raises(TypeError):
            analyzer.set_modules(
                detector=object(),
                motion=_FakeMotion(),
                roi_scheduler=_FakeROIScheduler(),
            )

    def test_invalid_motion_rejected(self) -> None:
        """不满足 MotionDetectorLike 的对象应报 TypeError"""
        analyzer = VideoAnalyzer()
        with pytest.raises(TypeError):
            analyzer.set_modules(
                detector=_FakeDetector(),
                motion=object(),
                roi_scheduler=_FakeROIScheduler(),
            )


class TestVideoAnalyzerThreading:
    """run_analysis 线程池执行行为测试"""

    @pytest.mark.asyncio
    async def test_analysis_runs_off_event_loop(self, tmp_path) -> None:
        """
        逐帧分析应在非事件循环线程中执行。

        用假模块记录执行线程标识：若与事件循环线程不同，
        证明 CPU 密集逻辑已移入线程池（不阻塞事件循环）。
        """
        import threading

        loop_thread = threading.get_ident()
        observed: list[int] = []

        class _ThreadSpyMotion(_FakeMotion):
            def detect(self, current_gray, prev_gray):
                observed.append(threading.get_ident())
                return []

        # 生成一个最小可解码视频（2 帧）
        cv2 = pytest.importorskip("cv2", reason="cv2 required for video tests")
        video_path = tmp_path / "tiny.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (64, 64),
        )
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.write(np.full((64, 64, 3), 128, dtype=np.uint8))
        writer.release()
        assert video_path.exists()

        analyzer = VideoAnalyzer()
        analyzer.set_modules(
            detector=_FakeDetector(),
            motion=_ThreadSpyMotion(),
            roi_scheduler=_FakeROIScheduler(),
        )
        task = analyzer.create_task("tiny.mp4", str(video_path))
        await analyzer.run_analysis(task.task_id)

        assert task.status == AnalysisStatus.COMPLETED
        # 运动检测在线程池线程中执行，且与事件循环线程不同
        assert observed, "detect 应至少被调用一次"
        assert all(t != loop_thread for t in observed)

    @pytest.mark.asyncio
    async def test_missing_modules_fails_task(self, tmp_path) -> None:
        """未注入模块时任务应标记 FAILED 并推送 error 事件"""
        analyzer = VideoAnalyzer()
        task = analyzer.create_task("x.mp4", str(tmp_path / "x.mp4"))
        await analyzer.run_analysis(task.task_id)
        assert task.status == AnalysisStatus.FAILED
        assert "not initialized" in task.error_message
