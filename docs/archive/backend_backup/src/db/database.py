"""
LoongGuard 告警数据库持久层

设计动机：
    将告警数据从运行时内存 deque 持久化至 SQLite 文件，确保重启后数据不丢失。
    使用 stdlib sqlite3，零外部依赖，WAL 模式适配 asyncio 单线程读写场景。
    所有 SQL 均采用参数化查询，杜绝注入风险。

约束：
    - 视频帧数据绝不落盘，仅存储结构化告警元数据
    - 告警切片图路径仅为引用，加密存储由 crypto 模块负责
"""

from __future__ import annotations

import sqlite3
import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.utils.schema import AlertLog, BoundingBox

logger = logging.getLogger(__name__)

# ── Schema 定义 ────────────────────────────────────────────────
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT DEFAULT '',
    slice_path TEXT DEFAULT '',
    acknowledged INTEGER DEFAULT 0,
    acknowledged_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL REFERENCES alerts(alert_id) ON DELETE CASCADE,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    confidence REAL NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_ack ON alerts(acknowledged);
CREATE INDEX IF NOT EXISTS idx_detections_alert ON detections(alert_id);
"""


class AlertDatabase:
    """
    SQLite 告警数据库

    每个实例绑定一个 SQLite 文件，生命周期由调用方管理。
    推荐通过上下文管理器使用：
        with AlertDatabase("data/alerts.db") as db:
            db.insert_alert(alert)
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ── 上下文管理器 ───────────────────────────────────────────
    def __enter__(self) -> AlertDatabase:
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── 生命周期 ───────────────────────────────────────────────
    def initialize(self) -> None:
        """创建数据库文件、表结构，设置 PRAGMA 优化参数。"""
        db_file = Path(self._db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row

        # WAL 模式：读写并发、写入性能提升
        self._conn.execute("PRAGMA journal_mode=WAL")
        # 外键约束：确保 detections 引用有效性
        self._conn.execute("PRAGMA foreign_keys=ON")
        # 忙等待超时 5s：避免并发写入时立即报错
        self._conn.execute("PRAGMA busy_timeout=5000")

        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("数据库初始化完成: %s", self._db_path)

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 写入 ──────────────────────────────────────────────────
    def insert_alert(self, alert: AlertLog) -> None:
        """
        插入一条告警及其检测框，原子事务保证一致性。
        使用 INSERT OR REPLACE 处理重复 alert_id 的幂等写入。
        """
        # 先删除旧记录的 detections（如果 alert_id 已存在）
        # INSERT OR REPLACE 会替换 alerts 行，但 detections 需要手动清理
        self._conn.execute(
            "DELETE FROM detections WHERE alert_id = ?", (alert.alert_id,)
        )
        self._conn.execute(
            """INSERT OR REPLACE INTO alerts
               (alert_id, timestamp, alert_type, severity,
                description, slice_path, acknowledged)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                alert.alert_id,
                alert.timestamp,
                alert.alert_type.value,
                alert.severity.value,
                alert.description,
                alert.slice_path or "",
                int(alert.acknowledged),
            ),
        )
        # 批量插入检测框
        if alert.detections:
            self._conn.executemany(
                """INSERT INTO detections
                   (alert_id, x1, y1, x2, y2, confidence, class_id, class_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        alert.alert_id,
                        d.x1, d.y1, d.x2, d.y2,
                        d.confidence,
                        d.class_id,
                        d.class_name,
                    )
                    for d in alert.detections
                ],
            )
        self._conn.commit()

    # ── 查询 ──────────────────────────────────────────────────
    def query_alerts(
        self,
        page: int = 1,
        page_size: int = 20,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> dict:
        """
        分页查询告警列表，支持多维度过滤。

        返回结构: {"total": int, "page": int, "page_size": int, "data": list[dict]}
        每条告警 dict 包含 alerts 字段 + "detections" 列表。
        """
        conditions: list[str] = []
        params: list = []

        if severity is not None:
            conditions.append("severity = ?")
            params.append(severity)
        if alert_type is not None:
            conditions.append("alert_type = ?")
            params.append(alert_type)
        if acknowledged is not None:
            conditions.append("acknowledged = ?")
            params.append(int(acknowledged))
        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # 计算总数
        total = self._conn.execute(
            f"SELECT COUNT(*) FROM alerts {where_clause}", params
        ).fetchone()[0]

        # 分页查询
        offset = (page - 1) * page_size
        rows = self._conn.execute(
            f"""SELECT * FROM alerts {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?""",
            params + [page_size, offset],
        ).fetchall()

        data = [self._row_to_alert_dict(row) for row in rows]
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": data,
        }

    def get_alert_by_id(self, alert_id: str) -> Optional[dict]:
        """根据 alert_id 查询单条告警（含 detections），未找到返回 None。"""
        row = self._conn.execute(
            "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_alert_dict(row)

    # ── 更新 ──────────────────────────────────────────────────
    def ack_alert(self, alert_id: str) -> bool:
        """
        确认告警：标记 acknowledged=1 并记录确认时间。
        返回 True 表示成功更新，False 表示 alert_id 不存在。
        """
        cursor = self._conn.execute(
            """UPDATE alerts
               SET acknowledged = 1, acknowledged_at = datetime('now')
               WHERE alert_id = ? AND acknowledged = 0""",
            (alert_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ── 清理 ──────────────────────────────────────────────────
    def cleanup_old_alerts(self, retention_days: int = 90) -> int:
        """
        清理超过保留天数的旧告警。
        外键 CASCADE 自动级联删除关联的 detections。
        返回删除的告警条数。
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM alerts WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info("清理旧告警 %d 条 (截止 %s)", deleted, cutoff)
        return deleted

    # ── 统计 ──────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """
        返回告警统计摘要：
        {"total": int, "by_severity": dict, "by_type": dict,
         "acknowledged": int, "unacknowledged": int}
        """
        if self._conn is None:
            return {"total": 0, "by_severity": {}, "by_type": {},
                    "acknowledged": 0, "unacknowledged": 0}

        total = self._conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

        by_severity: dict[str, int] = {}
        for row in self._conn.execute(
            "SELECT severity, COUNT(*) as cnt FROM alerts GROUP BY severity"
        ):
            by_severity[row["severity"]] = row["cnt"]

        by_type: dict[str, int] = {}
        for row in self._conn.execute(
            "SELECT alert_type, COUNT(*) as cnt FROM alerts GROUP BY alert_type"
        ):
            by_type[row["alert_type"]] = row["cnt"]

        acked = self._conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE acknowledged = 1"
        ).fetchone()[0]

        return {
            "total": total,
            "by_severity": by_severity,
            "by_type": by_type,
            "acknowledged": acked,
            "unacknowledged": total - acked,
        }

    # ── 内部辅助 ──────────────────────────────────────────────
    def _row_to_alert_dict(self, row: sqlite3.Row) -> dict:
        """将 alerts 行 + 关联 detections 转换为完整的告警 dict。"""
        alert_id = row["alert_id"]
        det_rows = self._conn.execute(
            "SELECT * FROM detections WHERE alert_id = ?", (alert_id,)
        ).fetchall()

        detections = [
            {
                "x1": d["x1"],
                "y1": d["y1"],
                "x2": d["x2"],
                "y2": d["y2"],
                "confidence": d["confidence"],
                "class_id": d["class_id"],
                "class_name": d["class_name"],
            }
            for d in det_rows
        ]

        return {
            "alert_id": alert_id,
            "timestamp": row["timestamp"],
            "alert_type": row["alert_type"],
            "severity": row["severity"],
            "description": row["description"],
            "slice_path": row["slice_path"],
            "acknowledged": bool(row["acknowledged"]),
            "acknowledged_at": row["acknowledged_at"],
            "created_at": row["created_at"],
            "detections": detections,
        }
