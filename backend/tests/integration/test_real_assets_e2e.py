"""
真实素材端到端连通性测试

与 test_pipeline_smoke.py 互补：
    - smoke 测试：tests/mock_classroom.mp4，验证"接得通"，每次 CI 必跑
    - 本文件：tests/fixtures/ 下的真实幼儿园素材，验证"跑得好"，可按需运行

前置：
    - 真实素材已放入 tests/fixtures/（参见 README 或脚本）
    - python scripts/create_dummy_models.py 已执行
"""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig
from loongguard.pipeline import Pipeline

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# 素材注册表：名称 → 文件路径
# 实际文件名优先，保持简洁
REAL_ASSETS = {
    "12s":         FIXTURES_DIR / "12s.mp4",
    "60s":         FIXTURES_DIR / "60s.mp4",
    "youeryuan":   FIXTURES_DIR / "youeryuan120s.mp4",
    "sucai2":      FIXTURES_DIR / "sucai2.mp4",
}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestRealAssetPipeline:
    """每个真实素材都能启动 Pipeline → 读帧 → 处理 → 正常停止"""

    @pytest.mark.parametrize("name,video_path", list(REAL_ASSETS.items()))
    @pytest.mark.asyncio
    async def test_pipeline_runs_with_real_asset(
        self, tmp_path, monkeypatch, name: str, video_path: Path,
    ) -> None:
        if not video_path.exists():
            pytest.skip(f"素材不存在: {video_path}（需手动放入 tests/fixtures/）")

        monkeypatch.setenv("LG_SM4_KEY", "0" * 32)
        monkeypatch.setenv("LG_CAMERA_DEVICE", str(video_path))

        config = AppConfig()
        config.camera.device = str(video_path)
        config.detection.model_path = str(PROJECT_ROOT / "models" / "yolo26_nano_int8.onnx")
        config.detection.input_size = 640
        config.pose.model_path = str(PROJECT_ROOT / "models" / "movenet_lightning_int8.onnx")
        config.crypto.log_dir = str(tmp_path / "logs")
        config.crypto.slice_dir = str(tmp_path / "slices")
        config.api.port = _find_free_port()
        config.debug = False

        pipeline = Pipeline(config)
        processed = []

        original_process = pipeline._process_frame

        async def _track(frame):
            processed.append(frame.frame_id)
            # 只跑 3 帧验证链路即可，不做完整推理
            if len(processed) >= 3:
                pipeline._running = False
            await original_process(frame)

        pipeline._process_frame = _track

        with patch.object(pipeline._api, "start", new_callable=AsyncMock), \
             patch.object(pipeline._api, "stop", new_callable=AsyncMock), \
             patch.object(pipeline._detector, "load_model"), \
             patch.object(pipeline._pose, "load_model"):
            await pipeline.start()
            try:
                await asyncio.wait_for(pipeline.run(), timeout=10.0)
            except asyncio.TimeoutError:
                pass
            await pipeline.stop()

        assert len(processed) >= 1, f"至少应处理 1 帧，实际处理 {len(processed)} 帧"

    @pytest.mark.parametrize("name,video_path", list(REAL_ASSETS.items()))
    def test_asset_readable(self, name: str, video_path: Path) -> None:
        """素材可被 OpenCV 正常打开并读取"""
        if not video_path.exists():
            pytest.skip(f"素材不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"无法打开: {video_path}"
        ret, frame = cap.read()
        assert ret, f"无法读取首帧: {video_path}"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"帧格式异常: {frame.shape}"
        cap.release()
