"""视频文件保留策略单元测试"""
import os
import time
from dataclasses import dataclass
from pathlib import Path
import pytest
from loongguard.utils.retention import VideoRetentionManager


@dataclass
class MockRetentionConfig:
    upload_dir: str = ""
    max_age_hours: float = 1.0
    max_total_size_mb: float = 1.0
    cleanup_interval_sec: float = 3600.0


class TestVideoRetentionManager:
    @pytest.fixture
    def config(self, tmp_path):
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        c = MockRetentionConfig()
        c.upload_dir = str(upload_dir)
        return c

    @pytest.fixture
    def manager(self, config):
        return VideoRetentionManager(config)

    def test_cleanup_empty_dir(self, manager):
        """空目录清理应返回零结果"""
        result = manager.cleanup()
        assert result["deleted_count"] == 0

    def test_cleanup_deletes_old_files(self, manager, config):
        """超过 max_age_hours 的文件应被删除"""
        upload_dir = Path(config.upload_dir)
        # 创建一个"旧"文件（通过设置 mtime）
        old_file = upload_dir / "old_video.mp4"
        old_file.write_bytes(b"x" * 1024)
        # 设置修改时间为 2 小时前（超过 max_age_hours=1）
        old_time = time.time() - 7200
        os.utime(old_file, (old_time, old_time))

        result = manager.cleanup()
        assert result["deleted_count"] == 1
        assert not old_file.exists()

    def test_cleanup_preserves_new_files(self, manager, config):
        """未超龄的文件不应被删除"""
        upload_dir = Path(config.upload_dir)
        new_file = upload_dir / "new_video.mp4"
        new_file.write_bytes(b"x" * 1024)

        result = manager.cleanup()
        assert result["deleted_count"] == 0
        assert new_file.exists()

    def test_cleanup_preserves_active_files(self, manager, config):
        """正在处理的文件即使超龄也不应被删除"""
        upload_dir = Path(config.upload_dir)
        active_file = upload_dir / "processing.mp4"
        active_file.write_bytes(b"x" * 1024)
        old_time = time.time() - 7200
        os.utime(active_file, (old_time, old_time))

        result = manager.cleanup(active_files={str(active_file)})
        assert result["deleted_count"] == 0
        assert active_file.exists()

    def test_cleanup_size_limit(self, manager, config):
        """超过总大小上限时应删除最旧的文件"""
        config.max_total_size_mb = 0.002  # 2KB 上限
        config.max_age_hours = 9999  # 禁用时间清理
        manager = VideoRetentionManager(config)

        upload_dir = Path(config.upload_dir)
        for i in range(5):
            f = upload_dir / f"video_{i}.mp4"
            f.write_bytes(b"x" * 1024)  # 每个 1KB

        result = manager.cleanup()
        # 5KB 总量超过 2KB 上限，应删除一些文件
        assert result["deleted_count"] > 0
        remaining = list(upload_dir.iterdir())
        total_remaining = sum(f.stat().st_size for f in remaining)
        assert total_remaining <= 2048

    def test_cleanup_nonexistent_dir(self, tmp_path):
        """不存在的目录应安全返回零结果"""
        config = MockRetentionConfig(upload_dir=str(tmp_path / "nonexistent"))
        manager = VideoRetentionManager(config)
        result = manager.cleanup()
        assert result["deleted_count"] == 0

    def test_cleanup_returns_freed_bytes(self, manager, config):
        """返回值应包含释放的字节数"""
        upload_dir = Path(config.upload_dir)
        old_file = upload_dir / "old.mp4"
        old_file.write_bytes(b"x" * 5000)
        old_time = time.time() - 7200
        os.utime(old_file, (old_time, old_time))

        result = manager.cleanup()
        assert result["freed_bytes"] == 5000
