"""
LoongGuard 持久化数据库模块

使用 stdlib sqlite3 实现告警数据的持久化存储，替代运行时内存 deque。
SQLite WAL 模式确保在 asyncio 单线程场景下的读写性能。
"""

from loongguard.db.database import AlertDatabase

__all__ = ["AlertDatabase"]
