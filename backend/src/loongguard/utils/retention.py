"""
视频文件保留策略模块

设计动机：
    上传的视频文件（data/uploads/）持续占用磁盘空间。
    定期清理超龄文件，同时保护正在分析的文件不被误删。
    采用两级策略：时间过期 + 空间上限。

环境变量覆盖（通过 config/settings.py RetentionConfig）：
    LG_RETENTION_MAX_AGE_HOURS=24
    LG_RETENTION_MAX_TOTAL_SIZE_MB=1024
    LG_RETENTION_CLEANUP_INTERVAL_SEC=3600
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoRetentionManager:
    """
    上传视频文件生命周期管理器

    清理策略：
        1. 删除超过 max_age_hours 的文件（跳过 active_files）
        2. 若目录总大小仍超过 max_total_size_mb，按文件年龄从旧到新删除
        3. 永不删除 active_files 中的文件（正在被分析的视频）

    用法：
        manager = VideoRetentionManager(config.retention)
        result = manager.cleanup(active_files={"/path/to/processing.mp4"})
        # result = {"deleted_count": 3, "freed_bytes": 52428800, "errors": []}
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: RetentionConfig with fields:
                - upload_dir: str
                - max_age_hours: float
                - max_total_size_mb: float
                - cleanup_interval_sec: float
        """
        self._upload_dir = Path(config.upload_dir)
        self._max_age_hours = config.max_age_hours
        self._max_total_size_bytes = int(config.max_total_size_mb * 1024 * 1024)

    def cleanup(self, active_files: set[str] | None = None) -> dict:
        """
        执行一次清理。

        Args:
            active_files: 正在处理中的文件路径集合，不会被删除

        Returns:
            {"deleted_count": int, "freed_bytes": int, "errors": list[str]}
        """
        if not self._upload_dir.exists():
            return {"deleted_count": 0, "freed_bytes": 0, "errors": []}

        active = active_files or set()
        deleted_count = 0
        freed_bytes = 0
        errors: list[str] = []

        # 第一轮：按时间过期删除
        now = time.time()
        max_age_sec = self._max_age_hours * 3600

        files = self._get_upload_files()
        for file_path, mtime, size in files:
            if str(file_path) in active:
                continue
            age_sec = now - mtime
            if age_sec > max_age_sec:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    freed_bytes += size
                    logger.debug("Retention: deleted %s (age=%.1fh)", file_path.name, age_sec / 3600)
                except OSError as e:
                    errors.append(f"{file_path}: {e}")

        # 第二轮：空间上限检查（删除最旧的文件直到低于阈值）
        current_size = self._get_dir_size()
        if current_size > self._max_total_size_bytes:
            remaining_files = self._get_upload_files()
            for file_path, _, size in remaining_files:
                if current_size <= self._max_total_size_bytes:
                    break
                if str(file_path) in active:
                    continue
                try:
                    file_path.unlink()
                    deleted_count += 1
                    freed_bytes += size
                    current_size -= size
                    logger.debug("Retention: deleted %s (size limit)", file_path.name)
                except OSError as e:
                    errors.append(f"{file_path}: {e}")

        return {"deleted_count": deleted_count, "freed_bytes": freed_bytes, "errors": errors}

    def _get_upload_files(self) -> list[tuple[Path, float, int]]:
        """获取上传目录中的所有文件，按修改时间从旧到新排序"""
        files = []
        for f in self._upload_dir.iterdir():
            if f.is_file():
                try:
                    stat = f.stat()
                    files.append((f, stat.st_mtime, stat.st_size))
                except OSError:
                    pass
        files.sort(key=lambda x: x[1])  # 按 mtime 排序（最旧的在前）
        return files

    def _get_dir_size(self) -> int:
        """计算上传目录总大小"""
        total = 0
        for f in self._upload_dir.iterdir():
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total
