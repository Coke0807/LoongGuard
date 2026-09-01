"""
配置热更新模块

设计动机：
    支持运行时动态调整检测参数，无需重启即可切换检测模式。
    通过文件系统监控配置文件变化，自动重新加载配置。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """
    配置文件监控器

    监控配置文件变化，自动重新加载并通知订阅者。

    使用方式：
        watcher = ConfigWatcher("config/default.json")
        watcher.on_change(callback)
        watcher.start()
    """

    def __init__(
        self,
        config_path: str | Path,
        check_interval: float = 1.0,
    ) -> None:
        """
        初始化配置监控器

        Args:
            config_path: 配置文件路径
            check_interval: 检查间隔（秒）
        """
        self._config_path = Path(config_path)
        self._check_interval = check_interval
        self._last_mtime: float = 0
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._running = False
        self._task: asyncio.Task | None = None

    def on_change(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """
        注册配置变更回调

        Args:
            callback: 回调函数，接收新配置字典
        """
        self._callbacks.append(callback)

    def _load_config(self) -> dict[str, Any] | None:
        """
        加载配置文件

        Returns:
            配置字典或 None（加载失败）
        """
        try:
            if not self._config_path.exists():
                logger.warning("Config file not found: %s", self._config_path)
                return None

            with open(self._config_path, encoding="utf-8") as f:
                config = json.load(f)

            return config
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return None

    def _check_for_updates(self) -> None:
        """检查配置文件更新"""
        try:
            mtime = self._config_path.stat().st_mtime
            if mtime > self._last_mtime:
                self._last_mtime = mtime
                config = self._load_config()
                if config is not None:
                    self._notify_callbacks(config)
        except Exception as e:
            logger.error("Error checking config updates: %s", e)

    def _notify_callbacks(self, config: dict[str, Any]) -> None:
        """通知所有回调函数"""
        for callback in self._callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error("Error in config change callback: %s", e)

    async def _watch_loop(self) -> None:
        """监控循环"""
        while self._running:
            self._check_for_updates()
            await asyncio.sleep(self._check_interval)

    def start(self) -> None:
        """启动配置监控"""
        if self._running:
            return

        self._running = True
        self._last_mtime = self._config_path.stat().st_mtime if self._config_path.exists() else 0
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Config watcher started for: %s", self._config_path)

    def stop(self) -> None:
        """停止配置监控"""
        if not self._running:
            return

        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        logger.info("Config watcher stopped")

    def reload(self) -> dict[str, Any] | None:
        """
        手动重新加载配置

        Returns:
            新配置字典或 None（加载失败）
        """
        config = self._load_config()
        if config is not None:
            self._notify_callbacks(config)
        return config


class DynamicConfig:
    """
    动态配置管理器

    支持运行时动态调整配置参数，无需重启应用。

    使用方式：
        config = DynamicConfig("config/default.json")
        config.start()

        # 获取配置
        threshold = config.get("detection.conf_threshold", 0.5)

        # 更新配置
        config.set("detection.conf_threshold", 0.6)
    """

    def __init__(self, config_path: str | Path) -> None:
        """
        初始化动态配置管理器

        Args:
            config_path: 配置文件路径
        """
        self._config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._watcher = ConfigWatcher(config_path)
        self._watcher.on_change(self._on_config_change)

    def _on_config_change(self, new_config: dict[str, Any]) -> None:
        """配置变更回调"""
        self._config = new_config
        logger.info("Config reloaded from: %s", self._config_path)

    def start(self) -> None:
        """启动配置管理器"""
        # 初始加载
        self._config = self._watcher._load_config() or {}
        # 启动监控
        self._watcher.start()

    def stop(self) -> None:
        """停止配置管理器"""
        self._watcher.stop()

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键（支持点号分隔的嵌套键，如 "detection.conf_threshold"）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        设置配置值（运行时）

        Args:
            key: 配置键（支持点号分隔的嵌套键）
            value: 配置值
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logger.info("Config updated: %s = %s", key, value)

    def get_all(self) -> dict[str, Any]:
        """
        获取所有配置

        Returns:
            配置字典
        """
        return self._config.copy()

    def reload(self) -> None:
        """手动重新加载配置"""
        self._watcher.reload()
