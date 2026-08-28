"""SQLite 告警数据库单元测试

每个测试用例使用独立的 tmp_path 临时数据库，互不干扰。
"""

import pytest

from loongguard.db.database import AlertDatabase
from loongguard.utils.schema import AlertLog, AlertSeverity, AlertType, BoundingBox


@pytest.fixture
def db(tmp_path):
    """每个测试使用独立的临时数据库"""
    database = AlertDatabase(str(tmp_path / "test.db"))
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def sample_alert():
    """构造一个带检测框的示例告警"""
    return AlertLog(
        alert_id="abc123def456",
        timestamp="2026-06-25T10:00:00Z",
        alert_type=AlertType.DANGEROUS_OBJECT,
        severity=AlertSeverity.HIGH,
        detections=[
            BoundingBox(
                x1=100, y1=100, x2=200, y2=200,
                confidence=0.95, class_id=0, class_name="scissors",
            ),
            BoundingBox(
                x1=300, y1=300, x2=400, y2=400,
                confidence=0.80, class_id=2, class_name="needle",
            ),
        ],
        description="Found scissors on floor",
    )


class TestAlertDatabase:

    def test_initialize_creates_tables(self, db):
        """initialize() 应创建 alerts 和 detections 表"""
        # 通过成功插入和查询来验证表已存在
        alert = AlertLog(
            alert_id="init_test",
            timestamp="2026-06-25T00:00:00Z",
            alert_type=AlertType.ENVIRONMENT,
            severity=AlertSeverity.LOW,
            detections=[],
        )
        db.insert_alert(alert)
        result = db.get_alert_by_id("init_test")
        assert result is not None
        assert result["alert_id"] == "init_test"

    def test_insert_and_get_by_id(self, db, sample_alert):
        """insert_alert + get_alert_by_id 完整往返"""
        db.insert_alert(sample_alert)
        result = db.get_alert_by_id(sample_alert.alert_id)

        assert result is not None
        assert result["alert_id"] == sample_alert.alert_id
        assert result["timestamp"] == "2026-06-25T10:00:00Z"
        assert result["alert_type"] == "dangerous_object"
        assert result["severity"] == "high"
        assert result["description"] == "Found scissors on floor"
        assert result["acknowledged"] is False

    def test_insert_preserves_detections(self, db, sample_alert):
        """插入的告警应保留所有检测框"""
        db.insert_alert(sample_alert)
        result = db.get_alert_by_id(sample_alert.alert_id)

        detections = result["detections"]
        assert len(detections) == 2

        d0 = detections[0]
        assert d0["x1"] == 100
        assert d0["y1"] == 100
        assert d0["x2"] == 200
        assert d0["y2"] == 200
        assert d0["confidence"] == 0.95
        assert d0["class_id"] == 0
        assert d0["class_name"] == "scissors"

        d1 = detections[1]
        assert d1["class_name"] == "needle"
        assert d1["confidence"] == 0.80

    def test_query_pagination(self, db):
        """插入 35 条告警，验证分页返回数量正确"""
        for i in range(35):
            alert = AlertLog(
                alert_id=f"id{i:04d}",
                timestamp=f"2026-06-25T10:{i:02d}:00Z",
                alert_type=AlertType.DANGEROUS_OBJECT,
                severity=AlertSeverity.HIGH,
                detections=[],
                description=f"Alert {i}",
            )
            db.insert_alert(alert)

        result = db.query_alerts(page=1, page_size=20)
        assert result["total"] == 35
        assert result["page"] == 1
        assert result["page_size"] == 20
        assert len(result["data"]) == 20

        result2 = db.query_alerts(page=2, page_size=20)
        assert result2["total"] == 35
        assert result2["page"] == 2
        assert len(result2["data"]) == 15

    def test_query_filter_by_severity(self, db):
        """按严重级别过滤"""
        db.insert_alert(AlertLog(
            alert_id="high1", timestamp="2026-06-25T10:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.HIGH,
        ))
        db.insert_alert(AlertLog(
            alert_id="low1", timestamp="2026-06-25T10:01:00Z",
            alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
        ))
        db.insert_alert(AlertLog(
            alert_id="high2", timestamp="2026-06-25T10:02:00Z",
            alert_type=AlertType.PRONE_SLEEP, severity=AlertSeverity.HIGH,
        ))

        result = db.query_alerts(severity="high")
        assert result["total"] == 2
        assert all(r["severity"] == "high" for r in result["data"])

        result_low = db.query_alerts(severity="low")
        assert result_low["total"] == 1

    def test_query_filter_by_type(self, db):
        """按告警类型过滤"""
        db.insert_alert(AlertLog(
            alert_id="obj1", timestamp="2026-06-25T10:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.HIGH,
        ))
        db.insert_alert(AlertLog(
            alert_id="sleep1", timestamp="2026-06-25T10:01:00Z",
            alert_type=AlertType.PRONE_SLEEP, severity=AlertSeverity.CRITICAL,
        ))

        result = db.query_alerts(alert_type="dangerous_object")
        assert result["total"] == 1
        assert result["data"][0]["alert_id"] == "obj1"

    def test_query_filter_by_acknowledged(self, db):
        """按确认状态过滤"""
        db.insert_alert(AlertLog(
            alert_id="ack1", timestamp="2026-06-25T10:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.HIGH,
        ))
        db.insert_alert(AlertLog(
            alert_id="unack1", timestamp="2026-06-25T10:01:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.MEDIUM,
        ))
        db.ack_alert("ack1")

        acked = db.query_alerts(acknowledged=True)
        assert acked["total"] == 1
        assert acked["data"][0]["alert_id"] == "ack1"

        unacked = db.query_alerts(acknowledged=False)
        assert unacked["total"] == 1
        assert unacked["data"][0]["alert_id"] == "unack1"

    def test_query_filter_by_time_range(self, db):
        """按时间范围过滤"""
        db.insert_alert(AlertLog(
            alert_id="early", timestamp="2026-06-20T08:00:00Z",
            alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
        ))
        db.insert_alert(AlertLog(
            alert_id="mid", timestamp="2026-06-25T12:00:00Z",
            alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
        ))
        db.insert_alert(AlertLog(
            alert_id="late", timestamp="2026-06-30T18:00:00Z",
            alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
        ))

        result = db.query_alerts(
            start_time="2026-06-25T00:00:00Z",
            end_time="2026-06-26T00:00:00Z",
        )
        assert result["total"] == 1
        assert result["data"][0]["alert_id"] == "mid"

    def test_ack_alert(self, db, sample_alert):
        """确认告警：ack_alert 返回 True，再次查询 acknowledged=1"""
        db.insert_alert(sample_alert)

        assert db.ack_alert(sample_alert.alert_id) is True

        result = db.get_alert_by_id(sample_alert.alert_id)
        assert result["acknowledged"] is True
        assert result["acknowledged_at"] is not None

    def test_ack_nonexistent_returns_false(self, db):
        """确认不存在的告警返回 False"""
        assert db.ack_alert("nonexistent_id") is False

    def test_cleanup_old_alerts(self, db):
        """清理旧告警：插入新旧告警，cleanup 只删除旧的"""
        db.insert_alert(AlertLog(
            alert_id="old1", timestamp="2025-01-01T00:00:00Z",
            alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
        ))
        db.insert_alert(AlertLog(
            alert_id="old2", timestamp="2025-06-15T00:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.MEDIUM,
        ))
        # 使用一个保证在 90 天内的近期时间戳
        db.insert_alert(AlertLog(
            alert_id="recent1", timestamp="2026-06-25T00:00:00Z",
            alert_type=AlertType.PRONE_SLEEP, severity=AlertSeverity.CRITICAL,
        ))

        deleted = db.cleanup_old_alerts(retention_days=90)
        assert deleted == 2

        remaining = db.query_alerts(page_size=100)
        assert remaining["total"] == 1
        assert remaining["data"][0]["alert_id"] == "recent1"

    def test_cleanup_cascades_detections(self, db):
        """清理旧告警时，关联的 detections 应被级联删除"""
        alert = AlertLog(
            alert_id="cascade_old", timestamp="2025-01-01T00:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.HIGH,
            detections=[
                BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9,
                            class_id=0, class_name="scissors"),
            ],
        )
        db.insert_alert(alert)
        db.cleanup_old_alerts(retention_days=90)

        # 验证 detections 也被清理
        det_rows = db._conn.execute(
            "SELECT * FROM detections WHERE alert_id = 'cascade_old'"
        ).fetchall()
        assert len(det_rows) == 0

    def test_get_stats(self, db):
        """统计信息：各类型/级别计数、确认率"""
        db.insert_alert(AlertLog(
            alert_id="s1", timestamp="2026-06-25T10:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.HIGH,
        ))
        db.insert_alert(AlertLog(
            alert_id="s2", timestamp="2026-06-25T10:01:00Z",
            alert_type=AlertType.PRONE_SLEEP, severity=AlertSeverity.CRITICAL,
        ))
        db.insert_alert(AlertLog(
            alert_id="s3", timestamp="2026-06-25T10:02:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.MEDIUM,
        ))
        db.ack_alert("s1")

        stats = db.get_stats()
        assert stats["total"] == 3
        assert stats["by_severity"]["high"] == 1
        assert stats["by_severity"]["critical"] == 1
        assert stats["by_severity"]["medium"] == 1
        assert stats["by_type"]["dangerous_object"] == 2
        assert stats["by_type"]["prone_sleep"] == 1
        assert stats["acknowledged"] == 1
        assert stats["unacknowledged"] == 2

    def test_empty_database_query(self, db):
        """空数据库查询返回空结果"""
        result = db.query_alerts()
        assert result["total"] == 0
        assert result["data"] == []
        assert result["page"] == 1
        assert result["page_size"] == 20

    def test_insert_duplicate_id_replaces(self, db, sample_alert):
        """重复 alert_id 使用 INSERT OR REPLACE 幂等写入"""
        db.insert_alert(sample_alert)

        # 同 ID，修改描述
        updated = AlertLog(
            alert_id=sample_alert.alert_id,
            timestamp=sample_alert.timestamp,
            alert_type=sample_alert.alert_type,
            severity=sample_alert.severity,
            detections=[
                BoundingBox(x1=50, y1=50, x2=150, y2=150,
                            confidence=0.7, class_id=1, class_name="knife"),
            ],
            description="Updated description",
        )
        db.insert_alert(updated)

        result = db.get_alert_by_id(sample_alert.alert_id)
        assert result["description"] == "Updated description"
        # detections 应该被替换，不再有旧的
        assert len(result["detections"]) == 1
        assert result["detections"][0]["class_name"] == "knife"

    def test_query_result_order_desc(self, db):
        """查询结果应按 timestamp 降序排列"""
        db.insert_alert(AlertLog(
            alert_id="first", timestamp="2026-06-25T08:00:00Z",
            alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
        ))
        db.insert_alert(AlertLog(
            alert_id="third", timestamp="2026-06-25T12:00:00Z",
            alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
        ))
        db.insert_alert(AlertLog(
            alert_id="second", timestamp="2026-06-25T10:00:00Z",
            alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
        ))

        result = db.query_alerts()
        ids = [r["alert_id"] for r in result["data"]]
        assert ids == ["third", "second", "first"]

    def test_context_manager(self, tmp_path):
        """上下文管理器方式初始化和关闭"""
        with AlertDatabase(str(tmp_path / "ctx_test.db")) as db:
            db.insert_alert(AlertLog(
                alert_id="ctx1", timestamp="2026-06-25T10:00:00Z",
                alert_type=AlertType.ENVIRONMENT, severity=AlertSeverity.LOW,
            ))
            result = db.get_alert_by_id("ctx1")
            assert result is not None

    def test_multiple_filters_combined(self, db):
        """多个过滤条件同时生效"""
        db.insert_alert(AlertLog(
            alert_id="combo1", timestamp="2026-06-25T10:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.HIGH,
        ))
        db.insert_alert(AlertLog(
            alert_id="combo2", timestamp="2026-06-25T11:00:00Z",
            alert_type=AlertType.PRONE_SLEEP, severity=AlertSeverity.HIGH,
        ))
        db.insert_alert(AlertLog(
            alert_id="combo3", timestamp="2026-06-25T12:00:00Z",
            alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.LOW,
        ))

        result = db.query_alerts(
            severity="high",
            alert_type="dangerous_object",
        )
        assert result["total"] == 1
        assert result["data"][0]["alert_id"] == "combo1"


# ── 新增：热备份 & 审计日志测试（覆盖可靠性/合规项）─────────


class TestDatabaseBackup:

    def test_backup_creates_file(self, db, sample_alert, tmp_path):
        """backup() 应生成可读取的一致性快照文件"""
        db.insert_alert(sample_alert)
        backup_path = str(tmp_path / "backup" / "db_backup.db")

        assert db.backup(backup_path) is True

        # 独立连接打开备份文件，验证数据完整
        import sqlite3
        conn = sqlite3.connect(backup_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE alert_id=?",
                (sample_alert.alert_id,),
            ).fetchone()[0]
            assert count == 1
        finally:
            conn.close()

    def test_backup_preserves_all_alerts(self, db, tmp_path):
        """备份应包含全部告警数据"""
        for i in range(10):
            db.insert_alert(AlertLog(
                alert_id=f"bk{i}", timestamp=f"2026-06-25T1{i}:00:00Z",
                alert_type=AlertType.DANGEROUS_OBJECT, severity=AlertSeverity.HIGH,
            ))

        backup_path = str(tmp_path / "backup.db")
        assert db.backup(backup_path) is True

        import sqlite3
        conn = sqlite3.connect(backup_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            assert count == 10
        finally:
            conn.close()

    def test_backup_returns_false_when_closed(self, db, tmp_path):
        """数据库未初始化（连接关闭）时 backup 应返回 False"""
        db.close()
        assert db.backup(str(tmp_path / "closed.db")) is False


class TestAuditLog:

    def test_insert_and_query_audit(self, db):
        """insert_audit + query_audit 往返"""
        db.insert_audit({
            "timestamp": "2026-06-25T10:00:00+00:00",
            "method": "GET",
            "path": "/stream",
            "remote": "127.0.0.1",
            "user_agent": "pytest",
            "username": "admin",
            "status": 200,
        })

        result = db.query_audit(page=1, page_size=10)
        assert result["total"] == 1
        assert result["data"][0]["path"] == "/stream"
        assert result["data"][0]["method"] == "GET"
        assert result["data"][0]["username"] == "admin"

    def test_query_audit_empty(self, db):
        """无审计记录时返回空列表"""
        result = db.query_audit()
        assert result["total"] == 0
        assert result["data"] == []

    def test_query_audit_pagination(self, db):
        """审计日志分页"""
        for i in range(5):
            db.insert_audit({
                "timestamp": f"2026-06-25T10:0{i}:00+00:00",
                "method": "GET",
                "path": f"/api/v1/alerts?p={i}",
                "remote": "127.0.0.1",
                "user_agent": "pytest",
                "username": "",
                "status": 200,
            })

        result = db.query_audit(page=1, page_size=2)
        assert result["total"] == 5
        assert len(result["data"]) == 2
