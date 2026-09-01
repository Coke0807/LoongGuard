"""
HLS 直播发布器（家长端远程观看，方案 B：ffmpeg 切片 + 小程序 video 组件）

设计动机：
    家长端远程观看需要小程序 <video> 可直接消费的直播流。板端产出 HLS
    （m3u8 + TS 分片）是兼容性最好、无需特殊类目资质的方案。ffmpeg 消费
    回环 MJPEG 源（或板端 v4l2 设备）切片写入 data/hls/，aiohttp 服务
    经 /stream/live.m3u8 与 /stream/<seg>.ts 对外提供。

安全模型：
    视频内容涉及未成年人隐私，播放列表与分片均要求带过期 HMAC token
    （家长端经 /api/wx/stream/url 权限校验后签发）。分片 URL 是播放列表
    的相对路径、播放器不会自动带上 query，因此服务端下发 m3u8 时把
    token 重写注入每个分片 URI。密钥缺失时整体 fail-closed 停用。

生命周期：
    - start()：有 ffmpeg 且密钥就绪时立即起编码进程
    - 空闲自动停：无观看者超过 idle_timeout_sec 后由看护线程停止 ffmpeg
    - 按需自启：播放列表请求（ensure_started）再次拉起编码
    - 崩溃自愈：看护线程发现进程退出且非人为停止时带退避重启
    - ffmpeg 缺失/密钥缺失：enabled=False，接口返回 404，MJPEG 不受影响
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# 分片文件名白名单（防路径穿越），与 ffmpeg hls_segment_filename 保持一致
_SEGMENT_RE = re.compile(r"^[\w.-]+\.ts$")

# token 格式：<过期时间戳>-<hmac 前 32 hex>
_TOKEN_RE = re.compile(r"^(\d+)-([0-9a-f]{32})$")


def resolve_stream_secret() -> str:
    """HLS token 签名密钥：LG_HLS_STREAM_SECRET 优先，回退 LG_SM4_KEY。

    SM4 密钥是启动必填项（validate_config 阻断），因此生产环境必然有值；
    两者都为空（如部分测试环境）时返回空串，调用方据此停用 HLS。
    """
    return (
        os.environ.get("LG_HLS_STREAM_SECRET", "").strip()
        or os.environ.get("LG_SM4_KEY", "").strip()
    )


def token_ttl_from_env() -> int:
    """观看 token 有效期（秒），与 MediaConfig.hls_token_ttl_sec 同源同默认"""
    raw = os.environ.get("LG_MEDIA_HLS_TOKEN_TTL_SEC", "").strip()
    try:
        return int(raw) if raw else 21600
    except ValueError:
        return 21600


def sign_stream_token(secret: str, ttl_sec: int = 21600,
                      now: float | None = None) -> str:
    """签发带过期时间的流观看 token：<exp>-<hmac("hls:<exp>")[:32]>"""
    exp = int(now if now is not None else time.time()) + int(ttl_sec)
    mac = hmac.new(secret.encode("utf-8"), f"hls:{exp}".encode("utf-8"),
                   hashlib.sha256).hexdigest()[:32]
    return f"{exp}-{mac}"


def verify_stream_token(secret: str, token: str,
                        now: float | None = None) -> bool:
    """校验观看 token：格式、未过期、HMAC 一致（恒时比较防时序攻击）"""
    if not secret or not token:
        return False
    match = _TOKEN_RE.match(token)
    if not match:
        return False
    exp_str, mac = match.groups()
    exp = int(exp_str)
    now_ts = now if now is not None else time.time()
    if now_ts > exp:
        return False
    expected = hmac.new(secret.encode("utf-8"), f"hls:{exp}".encode("utf-8"),
                        hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(mac, expected)


def rewrite_playlist(m3u8_text: str, token: str) -> str:
    """把 token 注入播放列表的每个分片 URI。

    ffmpeg 生成的分片 URI 是相对路径（如 live_0001.ts），播放器请求分片时
    不会携带播放列表 URL 上的 query，故必须在下发前重写。注释行
    （#EXTINF/#EXT-X-* 等）原样保留。
    """
    if not token:
        return m3u8_text
    out_lines: list[str] = []
    for line in m3u8_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            joiner = "&" if "?" in stripped else "?"
            stripped = f"{stripped}{joiner}token={token}"
        out_lines.append(stripped)
    return "\n".join(out_lines) + ("\n" if m3u8_text.endswith("\n") else "")


class HLSPublisher:
    """
    ffmpeg HLS 切片进程管理器（线程安全）。

    Args:
        media: MediaConfig（hls_* 字段）
        output_dir: m3u8 与 TS 分片输出目录（backend/data/hls）
        stream_base_url: 回环输入源地址（仅 hls_input="loopback" 时使用）
        ffmpeg_bin: ffmpeg 可执行文件；None 时按 PATH 探测，找不到则停用
    """

    def __init__(self, media, output_dir: Path | str, stream_base_url: str,
                 ffmpeg_bin: str | None = None) -> None:
        self._cfg = media
        self._output_dir = Path(output_dir)
        self._stream_base_url = stream_base_url
        self._ffmpeg_bin = ffmpeg_bin if ffmpeg_bin is not None \
            else shutil.which("ffmpeg")
        self._secret = resolve_stream_secret()

        self._proc: subprocess.Popen | None = None
        self._stopped_by_us = False
        self._last_touch = 0.0
        self._spawn_attempts = 0
        self._lock = threading.Lock()
        self._watcher: threading.Thread | None = None

        self.enabled = bool(
            media.hls_enabled
            and self._ffmpeg_bin
            and self._secret
        )
        if media.hls_enabled and not self.enabled:
            reason = "ffmpeg 未安装" if not self._ffmpeg_bin else \
                "签名密钥为空（LG_HLS_STREAM_SECRET / LG_SM4_KEY）"
            logger.warning("HLS 直播停用：%s（远程观看不可用，MJPEG 不受影响）",
                           reason)

    # ── 路径与鉴权 ──────────────────────────────────────────

    @property
    def playlist_path(self) -> Path:
        return self._output_dir / "live.m3u8"

    def verify_token(self, token: str) -> bool:
        return verify_stream_token(self._secret, token)

    def issue_token(self) -> str:
        return sign_stream_token(self._secret, self._cfg.hls_token_ttl_sec)

    def touch(self) -> None:
        """记录最近一次观看请求时间（供空闲看护判断）"""
        self._last_touch = time.monotonic()

    # ── 生命周期 ────────────────────────────────────────────

    def start(self) -> None:
        """启用即拉起编码进程；失败不抛异常（HLS 降级，主链路无关）"""
        if not self.enabled:
            return
        self.touch()
        self._spawn()
        self._watcher = threading.Thread(
            target=self._watch_loop, name="hls-watcher", daemon=True
        )
        self._watcher.start()
        logger.info(
            "HLS publisher started (input=%s, out=%s, seg=%.1fs, idle=%.0fs)",
            self._cfg.hls_input, self._output_dir,
            self._cfg.hls_segment_duration_sec, self._cfg.hls_idle_timeout_sec,
        )

    def stop(self) -> None:
        """停止编码进程与看护线程（幂等）"""
        self._stopped_by_us = True
        with self._lock:
            proc, self._proc = self._proc, None
        self._terminate(proc)
        logger.info("HLS publisher stopped")

    def ensure_started(self) -> None:
        """观看请求到来时按需拉起（空闲已停/崩溃未愈时）"""
        if not self.enabled:
            return
        self.touch()
        with self._lock:
            alive = self._proc is not None and self._proc.poll() is None
        if not alive:
            self._spawn()

    # ── 内部 ────────────────────────────────────────────────

    def _spawn(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._stopped_by_us = False
            self._output_dir.mkdir(parents=True, exist_ok=True)

            if self._cfg.hls_input == "loopback":
                source = self._stream_base_url
            else:
                source = self._cfg.hls_input

            seg_pattern = str(self._output_dir / "live_%04d.ts")
            cmd = [
                self._ffmpeg_bin, "-hide_banner", "-loglevel", "warning",
                "-nostdin",
            ]
            if self._cfg.hls_input_format:
                cmd += ["-f", self._cfg.hls_input_format]
            cmd += [
                # 限制探测缓冲量：直播源帧率低/网络慢时，默认 5MB 探测会让
                # 首个分片产出延迟十几秒，压到 2MB 让播放列表尽快就绪
                "-probesize", "2M",
                "-analyzeduration", "2M",
                "-i", source,
                "-vf", f"scale={int(self._cfg.hls_video_width)}:-2",
                "-r", str(int(self._cfg.hls_fps)),
                "-c:v", "libx264", "-preset", "veryfast",
                "-tune", "zerolatency", "-pix_fmt", "yuv420p",
                "-b:v", self._cfg.hls_video_bitrate,
                "-g", str(int(self._cfg.hls_fps * 2)),  # 2 秒一个关键帧
                "-sc_threshold", "0",
                "-f", "hls",
                "-hls_time", f"{self._cfg.hls_segment_duration_sec:.2f}",
                "-hls_list_size", str(int(self._cfg.hls_list_size)),
                "-hls_flags", "delete_segments+independent_segments",
                "-hls_segment_filename", seg_pattern,
                str(self.playlist_path),
            ]

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                self._spawn_attempts += 1
                logger.info("HLS ffmpeg spawned (pid=%s, attempt=%d, src=%s)",
                            self._proc.pid, self._spawn_attempts, source)
            except OSError:
                logger.exception("HLS ffmpeg 启动失败，本次保持停用")
                self._proc = None

    @staticmethod
    def _terminate(proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass

    def _watch_loop(self) -> None:
        """看护线程：空闲自动停 + 崩溃退避重启"""
        idle_sec = max(float(self._cfg.hls_idle_timeout_sec), 30.0)
        backoff = 5.0
        while True:
            time.sleep(5.0)
            with self._lock:
                proc = self._proc
                stopped = self._stopped_by_us
            if stopped and proc is None:
                return  # stop() 已调用，看护线程退出
            if proc is not None and proc.poll() is not None:
                # 编码进程异常退出：退避重启（人为 stop 置空了 _proc，不会进来）
                with self._lock:
                    self._proc = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                logger.warning("HLS ffmpeg 退出（code=%s），%.0fs 后重启",
                               proc.returncode, backoff)
                self._spawn()
                continue
            backoff = 5.0
            if self._last_touch and time.monotonic() - self._last_touch > idle_sec:
                with self._lock:
                    proc, self._proc = self._proc, None
                if proc is not None:
                    self._terminate(proc)
                    logger.info("HLS 空闲超过 %.0fs，已暂停编码（有观看请求时自动恢复）",
                                idle_sec)

    # ── 供路由层使用 ────────────────────────────────────────

    def read_playlist(self) -> str | None:
        """读取并重写播放列表；不存在返回 None"""
        try:
            return rewrite_playlist(
                self.playlist_path.read_text(encoding="utf-8"),
                self.issue_token(),
            )
        except OSError:
            return None

    def read_segment(self, name: str) -> bytes | None:
        """读取分片；名称不合法或文件缺失返回 None"""
        if not _SEGMENT_RE.match(name):
            return None
        try:
            return (self._output_dir / name).read_bytes()
        except OSError:
            return None
