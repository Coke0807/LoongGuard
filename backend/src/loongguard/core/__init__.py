"""
LoongGuard 核心架构模块

提供依赖注入、事件驱动、配置热更新等架构基础设施。
"""

from .config_watcher import ConfigWatcher, DynamicConfig
from .container import DIContainer, get_container, reset_container
from .event_bus import Event, EventBus, EventPriority, get_event_bus, reset_event_bus

__all__ = [
    # 依赖注入
    "DIContainer",
    "get_container",
    "reset_container",
    # 事件驱动
    "Event",
    "EventBus",
    "EventPriority",
    "get_event_bus",
    "reset_event_bus",
    # 配置热更新
    "ConfigWatcher",
    "DynamicConfig",
]
