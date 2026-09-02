"""
HLS 直播发布器单元测试（家长端远程观看）

覆盖：
    - 观看 token：签发/校验/过期/篡改/空密钥 fail-closed
    - 播放列表重写：分片 URI 注入 token，注释行原样保留
    - 路由鉴权与产出：/stream/live.m3u8 与 /stream/<seg>.ts
      （有效 token 200 / 无效 401 / 未启用 404 / 分片内容与防穿越）

ffmpeg 进程本身不在此测试（用 fake bin + no-op _spawn），
板端真实编码验证依赖 ffmpeg 在场，见集成测试与手动验证清单。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import pytest_asyncio

aiohttp = pytest.importorskip("aiohttp", reason="aiohttp is required for API tests")

from config.settings import APIConfig, MediaConfig  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.api.server import AlertAPIServer  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.media.hls_publisher import (  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
    HLSPublisher,
    rewrite_playlist,
    sign_stream_token,
    verify_stream_token,
)

_SECRET = "unit-test-hls-secret"


class TestStreamToken:
    def test_roundtrip(self) -> None:
        token = sign_stream_token(_SECRET, ttl_sec=100, now=1000.0)
        assert verify_stream_token(_SECRET, token, now=1000.0)
        assert verify_stream_token(_SECRET, token, now=1099.0)

    def test_expired_rejected(self) -> None:
        token = sign_stream_token(_SECRET, ttl_sec=100, now=1000.0)
        assert not verify_stream_token(_SECRET, token, now=1101.0)

    def test_wrong_secret_rejected(self) -> None:
        token = sign_stream_token(_SECRET, ttl_sec=100, now=1000.0)
        assert not verify_stream_token("other-secret", token, now=1000.0)

    def test_tampered_mac_rejected(self) -> None:
        token = sign_stream_token(_SECRET, ttl_sec=100, now=1000.0)
        bad = token[:-1] + ("0" if token[-1] != "0" else "1")
        assert not verify_stream_token(_SECRET, bad, now=1000.0)

    def test_malformed_tokens_rejected(self) -> None:
        for bad in ("", "garbage", "123-not-hex", f"{int(time.time()) + 9999}-zz"):
            assert not verify_stream_token(_SECRET, bad, now=1000.0)

    def test_empty_secret_fails_closed(self) -> None:
        token = sign_stream_token("any", ttl_sec=100, now=1000.0)
        assert not verify_stream_token("", token, now=1000.0)

    def test_resolve_secret_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from loongguard.media.hls_publisher import resolve_stream_secret

        monkeypatch.delenv("LG_HLS_STREAM_SECRET", raising=False)
        monkeypatch.delenv("LG_SM4_KEY", raising=False)
        assert resolve_stream_secret() == ""
        monkeypatch.setenv("LG_SM4_KEY", "a" * 32)
        assert resolve_stream_secret() == "a" * 32
        monkeypatch.setenv("LG_HLS_STREAM_SECRET", "custom")
        assert resolve_stream_secret() == "custom"


class TestPlaylistRewrite:
    PLAYLIST = (
        "#EXTM3U\n"
        "#EXT-X-VERSION:3\n"
        "#EXT-X-TARGETDURATION:2\n"
        "#EXTINF:2.000000,\n"
        "live_0001.ts\n"
        "#EXTINF:2.000000,\n"
        "live_0002.ts\n"
        "#EXT-X-ENDLIST\n"
    )

    def test_segments_get_token(self) -> None:
        out = rewrite_playlist(self.PLAYLIST, "tok123")
        assert "live_0001.ts?token=tok123" in out
        assert "live_0002.ts?token=tok123" in out
        assert "#EXT-X-ENDLIST" in out
        assert "#EXTINF:2.000000," in out

    def test_existing_query_uses_amp(self) -> None:
        assert rewrite_playlist("a.ts?x=1\n", "tok") == "a.ts?x=1&token=tok\n"

    def test_empty_token_is_noop(self) -> None:
        assert rewrite_playlist(self.PLAYLIST, "") == self.PLAYLIST


@pytest.fixture
def _hls_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """HLS 输出目录（含预置播放列表与分片）+ 签名密钥环境"""
    monkeypatch.setenv("LG_HLS_STREAM_SECRET", _SECRET)
    out_dir = tmp_path / "hls"
    out_dir.mkdir()
    (out_dir / "live.m3u8").write_text(TestPlaylistRewrite.PLAYLIST, encoding="utf-8")
    (out_dir / "live_0001.ts").write_bytes(b"\x47\x40\x00\x10" * 16)
    return out_dir


@pytest.fixture
def hls_publisher(_hls_env: Path, monkeypatch: pytest.MonkeyPatch) -> HLSPublisher:
    """enabled=True 但不真实拉起 ffmpeg（_spawn 置空）的发布器"""
    monkeypatch.setattr(HLSPublisher, "_spawn", lambda self: None)
    pub = HLSPublisher(
        MediaConfig(),
        output_dir=_hls_env,
        stream_base_url="http://127.0.0.1:8080/stream",
        ffmpeg_bin="/usr/bin/ffmpeg-fake",  # 只为 enabled 判定为真
    )
    assert pub.enabled
    return pub


@pytest_asyncio.fixture
async def hls_server(hls_publisher: HLSPublisher):
    srv = AlertAPIServer(APIConfig(host="127.0.0.1", port=0))
    srv.set_hls_publisher(hls_publisher)
    await srv.start()
    base_url = f"http://127.0.0.1:{srv._runner.addresses[0][1]}"
    yield srv, base_url
    await srv.stop()


class TestHLSRoutes:
    @pytest.mark.asyncio
    async def test_playlist_served_with_rewritten_token(
        self, hls_server, hls_publisher: HLSPublisher
    ) -> None:
        _, base_url = hls_server
        token = hls_publisher.issue_token()
        async with aiohttp.ClientSession() as http:
            async with http.get(
                f"{base_url}/stream/live.m3u8?token={token}") as resp:
                assert resp.status == 200
                assert resp.content_type == "application/vnd.apple.mpegurl"
                text = await resp.text()
        assert "live_0001.ts?token=" in text
        assert "#EXT-X-ENDLIST" in text

    @pytest.mark.asyncio
    async def test_playlist_rejects_missing_or_bad_token(
        self, hls_server, hls_publisher: HLSPublisher
    ) -> None:
        _, base_url = hls_server
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{base_url}/stream/live.m3u8") as resp:
                assert resp.status == 401
            bad = sign_stream_token("wrong", ttl_sec=60)
            async with http.get(
                f"{base_url}/stream/live.m3u8?token={bad}") as resp:
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_segment_served_and_authed(
        self, hls_server, hls_publisher: HLSPublisher
    ) -> None:
        _, base_url = hls_server
        token = hls_publisher.issue_token()
        async with aiohttp.ClientSession() as http:
            async with http.get(
                f"{base_url}/stream/live_0001.ts?token={token}") as resp:
                assert resp.status == 200
                assert resp.content_type == "video/mp2t"
                body = await resp.read()
            assert body.startswith(b"\x47")  # MPEG-TS sync byte
            async with http.get(
                f"{base_url}/stream/live_0001.ts") as resp:
                assert resp.status == 401
            async with http.get(
                f"{base_url}/stream/live_9999.ts?token={token}") as resp:
                assert resp.status == 404

    @pytest.mark.asyncio
    async def test_segment_path_traversal_blocked(
        self, hls_server, hls_publisher: HLSPublisher
    ) -> None:
        _, base_url = hls_server
        token = hls_publisher.issue_token()
        async with aiohttp.ClientSession() as http:
            async with http.get(
                f"{base_url}/stream/..%2F..%2F.env?token={token}") as resp:
                assert resp.status in (401, 404)

    @pytest.mark.asyncio
    async def test_disabled_publisher_returns_404(
        self, _hls_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(HLSPublisher, "_spawn", lambda self: None)
        pub = HLSPublisher(
            MediaConfig(),
            output_dir=_hls_env,
            stream_base_url="http://127.0.0.1:8080/stream",
            ffmpeg_bin="",  # 模拟 ffmpeg 缺失
        )
        assert not pub.enabled
        srv = AlertAPIServer(APIConfig(host="127.0.0.1", port=0))
        srv.set_hls_publisher(pub)
        await srv.start()
        try:
            base_url = f"http://127.0.0.1:{srv._runner.addresses[0][1]}"
            async with aiohttp.ClientSession() as http:
                async with http.get(f"{base_url}/stream/live.m3u8") as resp:
                    assert resp.status == 404
        finally:
            await srv.stop()


class TestHLSPublisherEnabledGate:
    def test_disabled_without_ffmpeg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LG_HLS_STREAM_SECRET", _SECRET)
        # 仅验证 ffmpeg_bin=None 时构造不抛异常（None 会触发自动探测，结果不确定故不断言）
        HLSPublisher(
            MediaConfig(), output_dir=tmp_path, stream_base_url="http://x/stream",
            ffmpeg_bin=None,
        )
        # shutil.which 在测试环境可能真找到 ffmpeg，与断言无关——
        # 这里显式验证 ffmpeg_bin 传入 falsy 时停用
        pub2 = HLSPublisher(
            MediaConfig(), output_dir=tmp_path, stream_base_url="http://x/stream",
            ffmpeg_bin="",
        )
        assert not pub2.enabled

    def test_disabled_without_secret(self, tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LG_HLS_STREAM_SECRET", raising=False)
        monkeypatch.delenv("LG_SM4_KEY", raising=False)
        pub = HLSPublisher(
            MediaConfig(), output_dir=tmp_path, stream_base_url="http://x/stream",
            ffmpeg_bin="/usr/bin/ffmpeg-fake",
        )
        assert not pub.enabled

    def test_disabled_by_config(self, tmp_path: Path,
                                monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LG_HLS_STREAM_SECRET", _SECRET)
        pub = HLSPublisher(
            MediaConfig(hls_enabled=False), output_dir=tmp_path,
            stream_base_url="http://x/stream", ffmpeg_bin="/usr/bin/ffmpeg-fake",
        )
        assert not pub.enabled
