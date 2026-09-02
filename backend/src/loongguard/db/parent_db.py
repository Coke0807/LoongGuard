"""
家长端业务数据库持久层（class_monitor.db）

设计动机：
    家长端（微信小程序 + 管理后台）自成一套业务库，与主仓库告警库
    （loongguard.db）物理隔离，避免跨业务耦合。本模块由分仓库
    ParentModel/website/db.py 与 ParentModel/db.py 合并而来：
        - 管理后台 CRUD（班级/幼儿/家长/公告/成长记录/管理员）
        - 小程序家长侧查询（绑定列表/公告已读/成长记录/手机号绑定）
    两侧共用同一 schema 与同一库文件。

约束：
    - 表结构与原 class_monitor.db 完全一致，可直接沿用既有数据文件
    - 所有 SQL 参数化查询，杜绝注入
    - init_db() 由 ParentAPIRoutes 构造时显式调用，模块导入零副作用
"""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# 数据库路径：默认 backend/data/class_monitor.db，可用 LG_PARENT_DB_PATH 覆盖
# 注意：.env 由 pipeline.main() 在运行时加载（晚于模块导入），
# 因此路径必须在每次取连接时解析，不能在模块级固化。
_BACKEND_DIR = Path(__file__).resolve().parents[3]
_DEFAULT_DB_PATH = _BACKEND_DIR / "data" / "class_monitor.db"


def db_path() -> str:
    """惰性解析数据库路径（尊重运行期注入的 LG_PARENT_DB_PATH）

    相对路径统一相对 backend/ 目录解析（而非进程 CWD），
    与 run.py"cd backend && python run.py"的启动约定解耦。
    """
    raw = os.environ.get("LG_PARENT_DB_PATH", "")
    if not raw:
        return str(_DEFAULT_DB_PATH)
    p = Path(raw)
    if not p.is_absolute():
        p = _BACKEND_DIR / p
    return str(p)


def init_db() -> None:
    """建表 + 空库时写入默认公告（幂等，可重复调用）"""
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()

    # 1. 微信家长表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS parent (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        openid TEXT NOT NULL UNIQUE,
        parent_name TEXT,
        phone TEXT,
        nickname TEXT,
        avatar TEXT,
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 2. 管理员账号
    cur.execute('''
    CREATE TABLE IF NOT EXISTS admin_account (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        real_name TEXT NOT NULL,
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 3. 班级表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS classinfo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_name TEXT UNIQUE NOT NULL,
        year INTEGER
    )
    ''')

    # 4. 幼儿学生表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS student (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        class_name TEXT NOT NULL,
        class_id INTEGER,
        parent_watch_switch INTEGER DEFAULT 1,
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (class_id) REFERENCES classinfo(id)
    )
    ''')

    # 5. 家长-幼儿绑定中间表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS parent_student_bind (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(parent_id) REFERENCES parent(id),
        FOREIGN KEY(student_id) REFERENCES student(id),
        UNIQUE(parent_id, student_id)
    )
    ''')

    # 6. 公告表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS notice (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        summary TEXT,
        notice_type TEXT DEFAULT 'notice',
        publish_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1
    )
    ''')

    # 7. 公告已读记录表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS notice_read (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id INTEGER NOT NULL,
        notice_id INTEGER NOT NULL,
        read_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(parent_id) REFERENCES parent(id),
        FOREIGN KEY(notice_id) REFERENCES notice(id),
        UNIQUE(parent_id, notice_id)
    )
    ''')

    # 8. 成长记录表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS growth_record (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        image_path TEXT NOT NULL,
        record_date TEXT NOT NULL,
        student_id INTEGER,
        is_active INTEGER DEFAULT 1,
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES student(id)
    )
    ''')

    # 空库时写入默认公告（首次部署体验数据）
    cur.execute("SELECT COUNT(*) FROM notice")
    if cur.fetchone()[0] == 0:
        cur.executemany('''
            INSERT INTO notice (title, content, summary, notice_type) VALUES (?, ?, ?, ?)
        ''', _DEFAULT_NOTICES)

    # 空库时创建引导管理员：口令随机生成，仅打印一次。
    # 设计动机（安全修复）：旧实现固定口令 admin/admin123，公网可达的
    # 管理后台等价于公开写入权限。随机口令无人可猜，且只在创建时输出。
    cur.execute("SELECT COUNT(*) FROM admin_account")
    if cur.fetchone()[0] == 0:
        import secrets

        from loongguard.utils.parent_utils import hash_pwd

        bootstrap_password = secrets.token_urlsafe(9)
        cur.execute(
            "INSERT INTO admin_account(username, password, real_name) VALUES (?,?,?)",
            ("admin", hash_pwd(bootstrap_password), "超级管理员"),
        )
        logger.warning(
            "首次部署已创建管理员账号 admin，随机初始口令（仅此一次明文输出，"
            "请立即登录 /admin/login 并创建自有账号后停用该账号）：%s",
            bootstrap_password,
        )

    conn.commit()
    conn.close()
    logger.info("Parent DB ready at %s", db_path())


# （默认管理员口令已于 2026-08-29 改为首次部署随机生成，见 init_db）


_DEFAULT_NOTICES = [
    ('【放假通知】暑假放假安排',
     '各位家长：\n\n本学期将于7月31日结束，暑假从8月1日正式开始，9月1日开学报到。\n\n'
     '假期注意事项：\n1. 注意孩子人身安全，不在危险水域游泳；\n2. 合理安排作息时间，'
     '避免长时间使用电子产品；\n3. 保持良好的饮食习惯，注意饮食卫生；\n4. 适当进行户外活动，增强体质。\n\n'
     '祝孩子们度过一个安全、快乐的暑假！',
     '本学期将于7月31日结束，暑假从8月1日开始，9月1日正式开学...',
     'important'),
    ('【安全提醒】夏季防溺水提示',
     '各位家长：\n\n夏季到来，溺水事故进入高发期。请家长们加强对孩子的防溺水安全教育，做到「六不」：\n\n'
     '1. 不私自下水游泳；\n2. 不擅自与他人结伴游泳；\n3. 不在无家长或教师带领的情况下游泳；\n'
     '4. 不到无安全设施、无救援人员的水域游泳；\n5. 不到不熟悉的水域游泳；\n6. 不熟悉水性的学生不擅自下水施救。\n\n'
     '安全无小事，防患于未然！',
     '夏季溺水事故高发，请家长加强防溺水安全教育，做到「六不」原则...',
     'safe'),
    ('【活动通知】亲子运动会报名',
     '各位家长：\n\n本园将于8月15日（周六）上午9:00举办「健康同行，快乐成长」亲子运动会。\n\n'
     '活动内容：\n- 亲子接力赛\n- 趣味拔河\n- 两人三足\n- 亲子跳绳\n\n'
     '报名方式：请在小程序内回复「报名+幼儿姓名+参加人数」\n截止时间：8月10日\n\n期待您的参与！',
     '本园将于8月15日举办亲子运动会，请家长踊跃报名，截止8月10日...',
     'activity'),
    ('【健康提示】夏季传染病预防',
     '各位家长：\n\n近期气温升高，手足口病、水痘等传染病进入活跃期。请家长注意：\n\n'
     '1. 保持孩子个人卫生，勤洗手；\n2. 居室常通风，保持空气流通；\n3. 少去人群聚集的公共场所；\n'
     '4. 如发现孩子有发热、皮疹等异常症状，请及时就医并告知园方；\n5. 保证充足睡眠，均衡营养，提高免疫力。\n\n'
     '家园携手，共筑健康防线！',
     '近期手足口病、水痘等传染病活跃，请注意个人卫生，勤洗手常通风...',
     'health'),
    ('【园所动态】本周食谱公示',
     '各位家长：\n\n本周（7月22日-7月26日）幼儿食谱如下：\n\n'
     '周一：西红柿炒蛋、紫菜蛋花汤、米饭\n周二：红烧鸡翅、冬瓜排骨汤、米饭\n周三：清蒸鱼、丝瓜蛋汤、米饭\n'
     '周四：糖醋排骨、菌菇汤、米饭\n周五：扬州炒饭、玉米排骨汤\n\n如孩子有食物过敏，请提前告知班主任。',
     '本周（7月22日-7月26日）幼儿食谱已更新，请家长查阅...',
     'notice'),
]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ==================== 通用工具 ====================
def check_watch_time(periods) -> bool:
    now = datetime.datetime.now().time()
    for start_str, end_str in periods:
        start = datetime.datetime.strptime(start_str, "%H:%M").time()
        end = datetime.datetime.strptime(end_str, "%H:%M").time()
        if start <= now <= end:
            return True
    return False


# ==================== 班级相关 ====================
def get_class_by_name_year(name: str, year: int) -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM classinfo WHERE class_name = ? AND year = ?", (name, year))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def create_class(name: str, year: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO classinfo (class_name, year) VALUES (?, ?)", (name, year))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def list_class(page: int, page_size: int, keyword: str = "") -> tuple[int, list[dict]]:
    conn = get_conn()
    cur = conn.cursor()
    offset = (page - 1) * page_size
    sql = "SELECT * FROM classinfo WHERE 1=1"
    params: list = []
    if keyword:
        sql += " AND class_name LIKE ?"
        params.append(f"%{keyword}%")
    count_sql = sql.replace("SELECT *", "SELECT COUNT(*)")
    cur.execute(count_sql, params)
    total = cur.fetchone()[0]
    sql += " ORDER BY id LIMIT ? OFFSET ?"
    params.extend([page_size, offset])
    cur.execute(sql, params)
    rows = cur.fetchall()
    data = [dict(r) for r in rows]
    conn.close()
    return total, data


def update_class(cid: int, name: str, year: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE classinfo SET class_name=?, year=? WHERE id=?", (name, year, cid))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return affected > 0


def delete_class(cid: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM student WHERE class_id = ?", (cid,))
    if cur.fetchone()[0] > 0:
        conn.close()
        return False
    cur.execute("DELETE FROM classinfo WHERE id = ?", (cid,))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return affected > 0


# ==================== 家长相关 ====================
def get_parent_by_openid(openid: str) -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM parent WHERE openid = ?", (openid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_parent_by_phone(phone: str) -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM parent WHERE phone = ?", (phone,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def create_parent(openid: str, parent_name: str = "", phone: str = "") -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO parent(openid, parent_name, phone) VALUES (?, ?, ?)",
                (openid, parent_name, phone))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid


def update_parent(pid: int, parent_name: str, phone: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE parent SET parent_name=?, phone=? WHERE id=?", (parent_name, phone, pid))
    conn.commit()
    conn.close()


def update_parent_wx_info(openid: str, nickname: str = None, avatar: str = None) -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        if nickname and avatar:
            cur.execute("UPDATE parent SET nickname=?, avatar=? WHERE openid=?",
                        (nickname, avatar, openid))
        elif nickname:
            cur.execute("UPDATE parent SET nickname=? WHERE openid=?", (nickname, openid))
        elif avatar:
            cur.execute("UPDATE parent SET avatar=? WHERE openid=?", (avatar, openid))
        conn.commit()
    finally:
        conn.close()


def bind_phone_to_openid(phone: str, real_openid: str) -> tuple[bool, str, dict | None]:
    """将微信真实 openid 合并到后台按手机号预录入的家长记录（import_ 占位）"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM parent WHERE phone = ?", (phone,))
        target = cur.fetchone()
        if not target:
            conn.close()
            return False, "请先联系后台工作人员注册", None
        target = dict(target)

        if target["openid"] == real_openid:
            conn.close()
            return True, "绑定成功", target

        if not target["openid"].startswith("import_"):
            conn.close()
            return False, "该手机号已被其他微信账号绑定，如需更换请联系管理员", None

        # 先清理临时家长记录的公告已读记录，避免外键约束失败
        cur.execute("""
            DELETE FROM notice_read
            WHERE parent_id = (SELECT id FROM parent WHERE openid = ? AND (phone IS NULL OR phone = ''))
        """, (real_openid,))
        # 删除登录时临时创建的无手机号家长记录
        cur.execute(
            "DELETE FROM parent WHERE openid = ? AND (phone IS NULL OR phone = '')",
            (real_openid,),
        )

        # 将后台预录入家长的 openid 从占位符更新为真实 openid
        # 绑定表用 parent_id 引用，openid 变了但 parent.id 不变，绑定关系自动保持
        cur.execute("UPDATE parent SET openid = ? WHERE id = ?", (real_openid, target["id"]))

        conn.commit()
        target["openid"] = real_openid
        return True, "绑定成功", target
    except Exception as e:
        conn.rollback()
        return False, f"绑定失败: {str(e)}", None
    finally:
        conn.close()


def get_parent_bind_students(openid: str) -> list[dict]:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute('''
        SELECT s.id, s.student_name, s.class_name, s.parent_watch_switch, p.phone, c.year
        FROM parent_student_bind pb
        JOIN student s ON pb.student_id = s.id
        JOIN parent p ON pb.parent_id = p.id
        LEFT JOIN classinfo c ON s.class_id = c.id
        WHERE p.openid = ?
        ''', (openid,))
        rows = cur.fetchall()
        return [
            {
                "student_id": r["id"],
                "name": r["student_name"],
                "class": r["class_name"],
                "grade": f"{r['year']}级" if r["year"] else "",
                "switch": r["parent_watch_switch"],
                "phone": r["phone"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def check_parent_permission(openid: str, student_id: int, time_periods) -> tuple[bool, str]:
    binds = get_parent_bind_students(openid)
    target = None
    for item in binds:
        if item["student_id"] == student_id:
            target = item
            break
    if not target:
        return False, "未绑定该学生"
    if target["switch"] != 1:
        return False, "该学生家长观看权限已关闭"
    if not check_watch_time(time_periods):
        return False, "不在允许观看时段内"
    return True, "允许观看"


def list_parent(page: int, page_size: int, keyword: str = "") -> tuple[int, list[dict]]:
    conn = get_conn()
    cur = conn.cursor()
    offset = (page - 1) * page_size
    sql = "SELECT * FROM parent WHERE 1=1"
    params: list = []
    if keyword:
        sql += " AND (parent_name LIKE ? OR phone LIKE ? OR openid LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
    count_sql = sql.replace("SELECT *", "SELECT COUNT(*)")
    cur.execute(count_sql, params)
    total = cur.fetchone()[0]
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([page_size, offset])
    cur.execute(sql, params)
    rows = cur.fetchall()
    data = []
    for r in rows:
        item = dict(r)
        cur.execute("""
            SELECT s.student_name FROM parent_student_bind pb
            JOIN student s ON pb.student_id = s.id
            WHERE pb.parent_id = ?
        """, (item["id"],))
        students = cur.fetchall()
        item["bound_students"] = ", ".join(
            [s["student_name"] for s in students]) if students else ""
        data.append(item)
    conn.close()
    return total, data


# ==================== 学生相关 ====================
def get_student_by_name_class(name: str, class_id: int) -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM student WHERE student_name = ? AND class_id = ?", (name, class_id))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_student(page: int, page_size: int, keyword: str = "") -> tuple[int, list[dict]]:
    conn = get_conn()
    cur = conn.cursor()
    offset = (page - 1) * page_size
    sql = "SELECT * FROM student WHERE 1=1"
    params: list = []
    if keyword:
        sql += " AND (student_name LIKE ? OR class_name LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    count_sql = sql.replace("SELECT *", "SELECT COUNT(*)")
    cur.execute(count_sql, params)
    total = cur.fetchone()[0]
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([page_size, offset])
    cur.execute(sql, params)
    rows = cur.fetchall()
    data = [dict(r) for r in rows]
    conn.close()
    return total, data


def create_student(student_name: str, class_name: str, class_id: int,
                   watch_switch: int = 1) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO student (student_name, class_name, class_id, parent_watch_switch) VALUES (?, ?, ?, ?)",
        (student_name, class_name, class_id, watch_switch))
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def update_student(sid: int, student_name: str, class_name: str, class_id: int,
                   watch_switch: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE student SET student_name=?, class_name=?, class_id=?, parent_watch_switch=? WHERE id=?",
        (student_name, class_name, class_id, watch_switch, sid))
    conn.commit()
    conn.close()


# ==================== 绑定关系 ====================
def bind_parent_student(parent_id: int, student_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO parent_student_bind (parent_id, student_id) VALUES (?, ?)",
        (parent_id, student_id))
    conn.commit()
    conn.close()


def unbind_parent_student(parent_id: int, student_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM parent_student_bind WHERE parent_id = ? AND student_id = ?",
                (parent_id, student_id))
    conn.commit()
    conn.close()


# ==================== 导出查询 ====================
def get_student_parent_bindings() -> list[dict]:
    """返回每个学生-家长的绑定关系（一行一个绑定），包含班级ID"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
        SELECT
            s.id AS student_id,
            s.student_name,
            s.class_name,
            s.class_id,
            c.year AS grade,
            s.parent_watch_switch,
            p.id AS parent_id,
            p.openid,
            p.parent_name,
            p.phone,
            p.nickname
        FROM student s
        LEFT JOIN classinfo c ON s.class_id = c.id
        LEFT JOIN parent_student_bind pb ON s.id = pb.student_id
        LEFT JOIN parent p ON pb.parent_id = p.id
        ORDER BY s.id, p.id
    ''')
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_parent_student_bindings() -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
        SELECT
            p.id AS parent_id,
            p.openid,
            p.parent_name,
            p.phone,
            p.nickname,
            s.id AS student_id,
            s.student_name,
            s.class_name,
            c.year AS grade
        FROM parent p
        LEFT JOIN parent_student_bind pb ON p.id = pb.parent_id
        LEFT JOIN student s ON pb.student_id = s.id
        LEFT JOIN classinfo c ON s.class_id = c.id
        ORDER BY p.id, s.id
    ''')
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ==================== 管理员账号 ====================
def get_admin_by_username(username: str) -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM admin_account WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_admin(page: int, page_size: int) -> tuple[int, list[dict]]:
    conn = get_conn()
    cur = conn.cursor()
    offset = (page - 1) * page_size
    cur.execute("SELECT COUNT(*) FROM admin_account")
    total = cur.fetchone()[0]
    cur.execute("SELECT * FROM admin_account ORDER BY id DESC LIMIT ? OFFSET ?",
                (page_size, offset))
    rows = cur.fetchall()
    conn.close()
    return total, [dict(r) for r in rows]


# ==================== 成长记录（管理后台） ====================
def list_growth(page: int, page_size: int, keyword: str = "") -> tuple[int, list[dict]]:
    conn = get_conn()
    cur = conn.cursor()
    offset = (page - 1) * page_size
    sql = ("SELECT g.*, s.student_name FROM growth_record g "
           "LEFT JOIN student s ON g.student_id = s.id WHERE g.is_active = 1")
    params: list = []
    if keyword:
        sql += " AND g.title LIKE ?"
        params.append(f"%{keyword}%")
    count_sql = sql.replace("SELECT g.*, s.student_name", "SELECT COUNT(*)")
    cur.execute(count_sql, params)
    total = cur.fetchone()[0]
    sql += " ORDER BY g.record_date DESC, g.id DESC LIMIT ? OFFSET ?"
    params.extend([page_size, offset])
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return total, [dict(r) for r in rows]


def create_growth(title: str, image_path: str, record_date: str,
                  student_id: int | None = None) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO growth_record (title, image_path, record_date, student_id) VALUES (?, ?, ?, ?)",
        (title, image_path, record_date, student_id))
    conn.commit()
    gid = cur.lastrowid
    conn.close()
    return gid


def update_growth(gid: int, title: str, image_path: str, record_date: str,
                  student_id: int | None = None) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE growth_record SET title=?, image_path=?, record_date=?, student_id=? WHERE id=?",
        (title, image_path, record_date, student_id, gid))
    conn.commit()
    conn.close()


def delete_growth(gid: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE growth_record SET is_active=0 WHERE id=?", (gid,))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return affected > 0


def get_growth_list(openid: str) -> list[dict]:
    """小程序侧：当前家长绑定幼儿的成长记录（含未关联学生的全员记录）"""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, title, image_path, record_date
            FROM growth_record
            WHERE is_active = 1
            AND (
                student_id IS NULL
                OR student_id IN (
                    SELECT pb.student_id
                    FROM parent_student_bind pb
                    JOIN parent p ON pb.parent_id = p.id
                    WHERE p.openid = ?
                )
            )
            ORDER BY record_date DESC, id DESC
        ''', (openid,))
        rows = cur.fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "image": r["image_path"],
                "date": r["record_date"],
            }
            for r in rows
        ]
    finally:
        conn.close()


# ==================== 公告管理 ====================
NOTICE_TYPE_MAP = {
    "important": {"text": "重要", "tag": "tag-red"},
    "safe":      {"text": "安全", "tag": "tag-orange"},
    "activity":  {"text": "活动", "tag": "tag-blue"},
    "health":    {"text": "健康", "tag": "tag-green"},
    "notice":    {"text": "通知", "tag": "tag-gray"},
}


def list_notice(page: int, page_size: int, keyword: str = "") -> tuple[int, list[dict]]:
    conn = get_conn()
    cur = conn.cursor()
    offset = (page - 1) * page_size
    sql = "SELECT * FROM notice WHERE is_active = 1"
    params: list = []
    if keyword:
        sql += " AND title LIKE ?"
        params.append(f"%{keyword}%")
    count_sql = sql.replace("SELECT *", "SELECT COUNT(*)")
    cur.execute(count_sql, params)
    total = cur.fetchone()[0]
    sql += " ORDER BY publish_time DESC, id DESC LIMIT ? OFFSET ?"
    params.extend([page_size, offset])
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return total, [dict(r) for r in rows]


def create_notice(title: str, content: str, summary: str, notice_type: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notice (title, content, summary, notice_type) VALUES (?, ?, ?, ?)",
        (title, content, summary, notice_type))
    conn.commit()
    nid = cur.lastrowid
    conn.close()
    return nid


def update_notice(nid: int, title: str, content: str, summary: str, notice_type: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE notice SET title=?, content=?, summary=?, notice_type=? WHERE id=?",
                (title, content, summary, notice_type, nid))
    conn.commit()
    conn.close()


def delete_notice(nid: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE notice SET is_active=0 WHERE id=?", (nid,))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return affected > 0


def get_notice_list(openid: str) -> list[dict]:
    """小程序侧：全部公告 + 当前家长已读状态"""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, title, content, summary, notice_type, publish_time
            FROM notice
            WHERE is_active = 1
            ORDER BY publish_time DESC
        ''')
        rows = cur.fetchall()

        cur.execute('''
            SELECT notice_id FROM notice_read
            WHERE parent_id = (SELECT id FROM parent WHERE openid = ?)
        ''', (openid,))
        read_ids = {r["notice_id"] for r in cur.fetchall()}

        result = []
        for r in rows:
            t = NOTICE_TYPE_MAP.get(r["notice_type"], NOTICE_TYPE_MAP["notice"])
            summary = (r["summary"] if r["summary"] else
                       (r["content"][:50] + "..." if len(r["content"]) > 50 else r["content"]))
            result.append({
                "id": r["id"],
                "title": r["title"],
                "content": r["content"],
                "summary": summary,
                "type": r["notice_type"],
                "typeText": t["text"],
                "tagClass": t["tag"],
                "date": str(r["publish_time"])[:10],
                "read": r["id"] in read_ids,
            })
        return result
    finally:
        conn.close()


def mark_notice_read(openid: str, notice_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM parent WHERE openid = ?", (openid,))
        parent = cur.fetchone()
        if not parent:
            conn.close()
            return False
        cur.execute('''
            INSERT OR IGNORE INTO notice_read (parent_id, notice_id)
            VALUES (?, ?)
        ''', (parent["id"], notice_id))
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.rollback()
        conn.close()
        return False
