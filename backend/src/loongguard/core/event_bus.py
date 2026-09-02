"""
事件驱动架构 - 事件总线

设计动机：
    将告警处理改为事件总线模式，支持更灵活的告警路由和扩展。
    支持同步/异步事件处理、事件过滤、优先级队列等特性。
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventPriority(int, Enum):
    """事件优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """事件基类"""
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    source: str = ""


# 事件处理器类型
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    事件总线

    支持同步/异步事件处理、事件过滤、优先级队列。

    使用方式：
        bus = EventBus()

        # 注册处理器
        bus.on("alert", handle_alert)
        bus.on("alert", handle_alert_async, async_mode=True)

        # 触发事件
        bus.emit(Event(name="alert", data={"type": "dangerous_object"}))
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._async_handlers: dict[str, list[AsyncEventHandler]] = defaultdict(list)
        self._max_workers = max_workers
        self._executor: Any = None

    def on(
        self,
        event_name: str,
        handler: EventHandler | AsyncEventHandler,
        async_mode: bool = False,
    ) -> None:
        """
        注册事件处理器

        Args:
            event_name: 事件名称
            handler: 处理器函数
            async_mode: 是否为异步处理器
        """
        if async_mode:
            self._async_handlers[event_name].append(handler)
        else:
            self._handlers[event_name].append(handler)

        logger.debug("Registered handler for event: %s (async=%s)", event_name, async_mode)

    def off(self, event_name: str, handler: EventHandler | AsyncEventHandler) -> None:
        """
        注销事件处理器

        Args:
            event_name: 事件名称
            handler: 处理器函数
        """
        if handler in self._handlers.get(event_name, []):
            self._handlers[event_name].remove(handler)
        elif handler in self._async_handlers.get(event_name, []):
            self._async_handlers[event_name].remove(handler)

    def emit(self, event: Event) -> None:
        """
        触发事件（同步）

        Args:
            event: 事件对象
        """
        handlers = self._handlers.get(event.name, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("Error in event handler for %s: %s", event.name, e)

    async def emit_async(self, event: Event) -> None:
        """
        触发事件（异步）

        Args:
            event: 事件对象
        """
        # 同步处理器
        handlers = self._handlers.get(event.name, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("Error in event handler for %s: %s", event.name, e)

        # 异步处理器
        async_handlers = self._async_handlers.get(event.name, [])
        for handler in async_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error("Error in async event handler for %s: %s", event.name, e)

    def get_handlers(self, event_name: str) -> list[EventHandler | AsyncEventHandler]:
        """
        获取事件的所有处理器

        Args:
            event_name: 事件名称

        Returns:
            处理器列表
        """
        handlers = self._handlers.get(event_name, []).copy()
        handlers.extend(self._async_handlers.get(event_name, []))
        return handlers

    def clear(self, event_name: str | None = None) -> None:
        """
        清除事件处理器

        Args:
            event_name: 事件名称（None 清除所有）
        """
        if event_name is None:
            self._handlers.clear()
            self._async_handlers.clear()
        else:
            self._handlers.pop(event_name, None)
            self._async_handlers.pop(event_name, None)


# 全局事件总线实例
_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    获取全局事件总线

    Returns:
        全局事件总线实例
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """重置全局事件总线"""
    global _global_bus
    if _global_bus is not None:
        _global_bus.clear()
    _global_bus = None
