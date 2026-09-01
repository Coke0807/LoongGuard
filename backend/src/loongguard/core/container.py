"""
轻量级依赖注入容器

设计动机：
    替代手动注入，提升模块解耦程度。
    支持单例、工厂、延迟初始化等模式。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DIContainer:
    """
    依赖注入容器

    使用方式：
        container = DIContainer()
        container.register("database", lambda: AlertDatabase("data/db.sqlite"))
        container.register("detector", lambda: YOLO26Nano(config))

        db = container.resolve("database")
        detector = container.resolve("detector")
    """

    def __init__(self) -> None:
        self._services: dict[str, Callable[[], Any]] = {}
        self._singletons: dict[str, Any] = {}
        self._singleton_flags: dict[str, bool] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], T],
        singleton: bool = False,
    ) -> None:
        """
        注册服务

        Args:
            name: 服务名称
            factory: 工厂函数（无参数，返回服务实例）
            singleton: 是否为单例模式
        """
        self._services[name] = factory
        self._singleton_flags[name] = singleton
        logger.debug("Registered service: %s (singleton=%s)", name, singleton)

    def register_instance(self, name: str, instance: Any) -> None:
        """
        注册现有实例（自动设为单例）

        Args:
            name: 服务名称
            instance: 服务实例
        """
        self._singletons[name] = instance
        self._singleton_flags[name] = True
        logger.debug("Registered instance: %s", name)

    def resolve(self, name: str) -> Any:
        """
        解析服务

        Args:
            name: 服务名称

        Returns:
            服务实例

        Raises:
            KeyError: 服务未注册
        """
        # 检查单例缓存
        if name in self._singletons:
            return self._singletons[name]

        # 检查服务是否注册
        if name not in self._services:
            raise KeyError(f"Service not registered: {name}")

        # 创建实例
        factory = self._services[name]
        instance = factory()

        # 缓存单例
        if self._singleton_flags.get(name, False):
            self._singletons[name] = instance

        return instance

    def has(self, name: str) -> bool:
        """
        检查服务是否已注册

        Args:
            name: 服务名称

        Returns:
            True if registered
        """
        return name in self._services or name in self._singletons

    def reset(self) -> None:
        """重置容器（清除所有注册和缓存）"""
        self._services.clear()
        self._singletons.clear()
        self._singleton_flags.clear()
        logger.debug("Container reset")

    def list_services(self) -> list[str]:
        """
        列出所有已注册的服务

        Returns:
            服务名称列表
        """
        return list(set(self._services.keys()) | set(self._singletons.keys()))


# 全局容器实例（单例模式）
_global_container: DIContainer | None = None


def get_container() -> DIContainer:
    """
    获取全局依赖注入容器

    Returns:
        全局容器实例
    """
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container


def reset_container() -> None:
    """重置全局容器"""
    global _global_container
    if _global_container is not None:
        _global_container.reset()
    _global_container = None
