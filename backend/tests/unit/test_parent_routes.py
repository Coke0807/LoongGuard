"""
家长端路由单元测试（微信小程序 API + 管理后台 API + 页面路由）

覆盖：
    - 服务启动时家长端路由自动注册
    - 管理员登录（错误口令 / 正确口令 / Cookie 签发）
    - 管理后台 API 鉴权（无 Cookie 401 / 有 Cookie 正常 CRUD）
    - 小程序 API 鉴权（无/伪造 Bearer token 401，合法 token 返回数据）
    - 根路径与登录页渲染
    - 主仓库既有端点不受影响（/api/v1/status 回归）

测试数据库使用临时文件，不触碰真实 class_monitor.db。
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest
import pytest_asyncio

aiohttp = pytest.importorskip("aiohttp", reason="aiohttp is required for API tests")

from config.settings import APIConfig
from loongguard.api.server import AlertAPIServer
from loongguard.utils import parent_utils

# 测试用 JWT 密钥（与生产 .env 隔离）
_TEST_ADMIN_SECRET = "test-admin-secret"
_TEST_WX_SECRET = "test-wx-secret"


@pytest.fixture(autouse=True)
def _temp_parent_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """重定向家长端数据库与 JWT 密钥到测试环境"""
    db_file = tmp_path / "class_monitor_test.db"
    monkeypatch.setenv("LG_PARENT_DB_PATH", str(db_file))
    monkeypatch.setenv("LG_PARENT_ADMIN_JWT_SECRET", _TEST_ADMIN_SECRET)
    monkeypatch.setenv("LG_PARENT_WX_JWT_SECRET", _TEST_WX_SECRET)
    monkeypatch.setenv("LG_PARENT_UPLOAD_DIR", str(tmp_path / "uploads"))
    return db_file


def _seed_admin(db_file: Path, username: str = "tester", password: str = "pass1234") -> None:
    conn = sqlite3.connect(db_file)
    conn.execute(
        "INSERT INTO admin_account(username, password, real_name) VALUES (?,?,?)",
        (username, parent_utils.hash_pwd(password), "测试管理员"),
    )
    conn.commit()
    conn.close()


def _seed_parent_with_student(db_file: Path, openid: str = "openid-ut-001") -> None:
    """预置家长 + 班级 + 学生 + 绑定关系"""
    conn = sqlite3.connect(db_file)
    conn.execute("INSERT INTO classinfo(class_name, year) VALUES ('大一班', 2026)")
    conn.execute("INSERT INTO student(student_name, class_name, class_id) VALUES ('小明', '大一班', 1)")
    conn.execute("INSERT INTO parent(openid, parent_name, phone) VALUES (?, ?, ?)",
                 (openid, "小明爸爸", "13800138000"))
    conn.execute("INSERT INTO parent_student_bind(parent_id, student_id) VALUES (1, 1)")
    conn.execute(
        "INSERT INTO notice(title, content, summary, notice_type) VALUES ('t1', 'c1', 's1', 'notice')")
    conn.commit()
    conn.close()


@pytest_asyncio.fixture
async def parent_server(_temp_parent_db: Path):
    """启动含家长端路由的 API 服务（随机端口）"""
    srv = AlertAPIServer(APIConfig(host="127.0.0.1", port=0))
    await srv.start()
    actual_port = srv._runner.addresses[0][1] if srv._runner.addresses else 0
    base_url = f"http://127.0.0.1:{actual_port}"
    yield srv, base_url
    await srv.stop()


# ── 页面与基础路由 ────────────────────────────────────────────


class TestParentPages:
    @pytest.mark.asyncio
    async def test_root_redirects_to_admin_login(self, parent_server) -> None:
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.get(base_url + "/", allow_redirects=False) as resp:
                assert resp.status in (302, 303)
                assert resp.headers["Location"] == "/admin/login"

    @pytest.mark.asyncio
    async def test_login_page_renders(self, parent_server) -> None:
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.get(base_url + "/admin/login") as resp:
                assert resp.status == 200
                body = await resp.text()
                assert "login" in body.lower()

    @pytest.mark.asyncio
    async def test_static_css_served(self, parent_server) -> None:
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.get(base_url + "/static/css/kindergarten.css") as resp:
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_admin_page_requires_login(self, parent_server) -> None:
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.get(base_url + "/admin/index", allow_redirects=False) as resp:
                assert resp.status in (302, 303)
                assert resp.headers["Location"] == "/admin/login"


# ── 管理员登录流程 ────────────────────────────────────────────


class TestAdminLogin:
    @pytest.mark.asyncio
    async def test_login_wrong_password(self, parent_server, _temp_parent_db) -> None:
        _seed_admin(_temp_parent_db)
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base_url + "/admin/login",
                data={"username": "tester", "password": "wrong"},
                allow_redirects=False,
            ) as resp:
                assert resp.status == 200
                body = await resp.text()
                assert "账号或密码错误" in body

    @pytest.mark.asyncio
    async def test_login_success_sets_cookie(self, parent_server, _temp_parent_db) -> None:
        _seed_admin(_temp_parent_db)
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base_url + "/admin/login",
                data={"username": "tester", "password": "pass1234"},
                allow_redirects=False,
            ) as resp:
                assert resp.status == 303
                assert resp.headers["Location"] == "/admin/index"
                assert "admin_token" in resp.cookies

    @pytest.mark.asyncio
    async def test_login_success_can_access_index(
        self, parent_server, _temp_parent_db
    ) -> None:
        _seed_admin(_temp_parent_db)
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base_url + "/admin/login",
                data={"username": "tester", "password": "pass1234"},
                allow_redirects=False,
            ) as resp:
                token = resp.cookies["admin_token"].value
            async with http.get(
                base_url + "/admin/index",
                cookies={"admin_token": token},
            ) as resp:
                assert resp.status == 200


# ── 管理后台 API ─────────────────────────────────────────────


class TestAdminAPI:
    async def _admin_cookie(self, http, base_url) -> str:
        async with http.post(
            base_url + "/admin/login",
            data={"username": "tester", "password": "pass1234"},
            allow_redirects=False,
        ) as resp:
            return resp.cookies["admin_token"].value

    @pytest.mark.asyncio
    async def test_admin_api_without_cookie_401(self, parent_server) -> None:
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.get(base_url + "/api/admin/class/list") as resp:
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_class_crud(self, parent_server, _temp_parent_db) -> None:
        _seed_admin(_temp_parent_db)
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            token = await self._admin_cookie(http, base_url)
            cookies = {"admin_token": token}

            # 新增
            async with http.post(
                base_url + "/api/admin/class/add?class_name=中一班&year=2026",
                cookies=cookies,
            ) as resp:
                data = await resp.json()
                assert data["code"] == 0

            # 列表
            async with http.get(
                base_url + "/api/admin/class/list?page=1&page_size=10",
                cookies=cookies,
            ) as resp:
                data = await resp.json()
                assert data["code"] == 0
                assert data["total"] == 1
                assert data["data"][0]["class_name"] == "中一班"

            # 修改
            async with http.put(
                base_url + "/api/admin/class/edit?id=1&class_name=中二班&year=2026",
                cookies=cookies,
            ) as resp:
                data = await resp.json()
                assert data["code"] == 0

            # 删除
            async with http.delete(
                base_url + "/api/admin/class/delete?id=1", cookies=cookies
            ) as resp:
                data = await resp.json()
                assert data["code"] == 0

    @pytest.mark.asyncio
    async def test_student_add_requires_existing_class(
        self, parent_server, _temp_parent_db
    ) -> None:
        _seed_admin(_temp_parent_db)
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            token = await self._admin_cookie(http, base_url)
            async with http.post(
                base_url + "/api/admin/student/add?name=小明&cls=不存在的班级",
                cookies={"admin_token": token},
            ) as resp:
                data = await resp.json()
                assert data["code"] == -1
                assert "不存在" in data["msg"]


# ── 小程序 API ───────────────────────────────────────────────


class TestWxAPI:
    @pytest.mark.asyncio
    async def test_wx_api_without_token_401(self, parent_server) -> None:
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.post(base_url + "/api/wx/get-student-list", json={}) as resp:
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_wx_api_with_forged_token_401(self, parent_server) -> None:
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base_url + "/api/wx/get-student-list",
                json={},
                headers={"Authorization": "Bearer forged-token"},
            ) as resp:
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_wx_get_student_list(self, parent_server, _temp_parent_db) -> None:
        _seed_parent_with_student(_temp_parent_db)
        _, base_url = parent_server
        token = parent_utils.generate_wx_token("openid-ut-001")
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base_url + "/api/wx/get-student-list",
                json={},
                headers={"Authorization": f"Bearer {token}"},
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["code"] == 0
                assert data["data"][0]["name"] == "小明"
                assert data["data"][0]["class"] == "大一班"

    @pytest.mark.asyncio
    async def test_wx_notice_list_and_read(self, parent_server, _temp_parent_db) -> None:
        _seed_parent_with_student(_temp_parent_db)
        _, base_url = parent_server
        token = parent_utils.generate_wx_token("openid-ut-001")
        headers = {"Authorization": f"Bearer {token}"}
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base_url + "/api/wx/get-notice-list", json={}, headers=headers
            ) as resp:
                data = await resp.json()
                assert data["code"] == 0
                # init_db 预置了默认公告，此处定位测试插入的 t1
                mine = [n for n in data["data"] if n["title"] == "t1"]
                assert len(mine) == 1
                assert mine[0]["read"] is False

            notice_id = mine[0]["id"]
            async with http.post(
                base_url + "/api/wx/read-notice",
                json={"notice_id": notice_id},
                headers=headers,
            ) as resp:
                data = await resp.json()
                assert data["code"] == 0

            async with http.post(
                base_url + "/api/wx/get-notice-list", json={}, headers=headers
            ) as resp:
                data = await resp.json()
                mine = [n for n in data["data"] if n["title"] == "t1"]
                assert mine[0]["read"] is True

    @pytest.mark.asyncio
    async def test_wx_growth_list(self, parent_server, _temp_parent_db) -> None:
        _seed_parent_with_student(_temp_parent_db)
        conn = sqlite3.connect(_temp_parent_db)
        conn.execute(
            "INSERT INTO growth_record(title, image_path, record_date, student_id) "
            "VALUES ('春游', '/uploads/growth/x.png', '2026-08-01', 1)")
        conn.commit()
        conn.close()

        _, base_url = parent_server
        token = parent_utils.generate_wx_token("openid-ut-001")
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base_url + "/api/wx/get-growth-list",
                json={},
                headers={"Authorization": f"Bearer {token}"},
            ) as resp:
                data = await resp.json()
                assert data["code"] == 0
                assert data["data"][0]["title"] == "春游"

    @pytest.mark.asyncio
    async def test_wx_student_list_get_compat(self, parent_server, _temp_parent_db) -> None:
        """website 兼容端点：GET + query token"""
        _seed_parent_with_student(_temp_parent_db)
        _, base_url = parent_server
        token = parent_utils.generate_wx_token("openid-ut-001")
        async with aiohttp.ClientSession() as http:
            async with http.get(
                base_url + f"/api/wx/student/list?token={token}"
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["code"] == 0
                assert data["data"][0]["name"] == "小明"


# ── 主仓库回归 ───────────────────────────────────────────────


class TestMainRepoRegression:
    @pytest.mark.asyncio
    async def test_status_endpoint_still_works(self, parent_server) -> None:
        """主仓库既有 /api/v1/status 不受家长端路由合并影响"""
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.get(base_url + "/api/v1/status") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "status" in data


# ── JWT 有效期与密钥校验 ─────────────────────────────────────


class TestJwtTokenExpiry:
    """回归：修复前签发的 JWT 无 exp 声明，泄露后永久有效"""

    def test_wx_token_contains_exp(self) -> None:
        import time

        import jwt as pyjwt

        token = parent_utils.generate_wx_token("openid-exp-check")
        payload = pyjwt.decode(token, _TEST_WX_SECRET, algorithms=["HS256"])
        assert "exp" in payload
        assert payload["exp"] > time.time()

    def test_expired_wx_token_rejected(self) -> None:
        import time

        import jwt as pyjwt

        now = int(time.time())
        expired = pyjwt.encode(
            {"openid": "openid-ut-001", "exp": now - 10},
            _TEST_WX_SECRET, algorithm="HS256",
        )
        assert parent_utils.decode_wx_token(expired) is None

    def test_expired_admin_token_rejected(self) -> None:
        import time

        import jwt as pyjwt

        now = int(time.time())
        expired = pyjwt.encode(
            {"username": "tester", "exp": now - 10},
            _TEST_ADMIN_SECRET, algorithm="HS256",
        )
        assert parent_utils.decode_admin_token(expired) is None


class TestValidateJwtSecrets:
    def test_dev_placeholder_secret_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LG_ENV", "development")
        monkeypatch.setenv("LG_PARENT_ADMIN_JWT_SECRET", "change-me-admin-jwt-secret")
        monkeypatch.setenv("LG_PARENT_WX_JWT_SECRET", "dev-ok-secret")
        parent_utils.validate_jwt_secrets()  # 开发环境仅记录 error，不阻断

    def test_prod_empty_secret_blocks_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LG_ENV", "production")
        monkeypatch.setenv("LG_PARENT_ADMIN_JWT_SECRET", "")
        monkeypatch.setenv("LG_PARENT_WX_JWT_SECRET", "prod-ok-secret")
        with pytest.raises(RuntimeError):
            parent_utils.validate_jwt_secrets()

    def test_prod_placeholder_secret_blocks_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LG_ENV", "production")
        monkeypatch.setenv("LG_PARENT_ADMIN_JWT_SECRET", "prod-ok-secret")
        monkeypatch.setenv("LG_PARENT_WX_JWT_SECRET", "change-me-wx-jwt-secret")
        with pytest.raises(RuntimeError):
            parent_utils.validate_jwt_secrets()


# ── 管理员 Cookie 安全属性 ───────────────────────────────────


class TestAdminCookieFlags:
    @pytest.mark.asyncio
    async def test_login_cookie_httponly_samesite(
        self, parent_server, _temp_parent_db
    ) -> None:
        """admin_token Cookie 必须携带 HttpOnly 与 SameSite=Lax（防 XSS 窃取）"""
        _seed_admin(_temp_parent_db)
        _, base_url = parent_server
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base_url + "/admin/login",
                data={"username": "tester", "password": "pass1234"},
                allow_redirects=False,
            ) as resp:
                assert resp.status == 303
                set_cookie = resp.headers.get("Set-Cookie", "")
        assert "admin_token=" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "SameSite=Lax" in set_cookie


# ── 上传扩展名白名单 ─────────────────────────────────────────


class TestSaveUploadWhitelist:
    class _FakeField:
        def __init__(self, filename: str, payload: bytes) -> None:
            self.filename = filename
            self.file = io.BytesIO(payload)

    def test_html_extension_neutralized(self, _temp_parent_db) -> None:
        """白名单外扩展名（如 .html，可构成同源存储型 XSS）落盘为 .png"""
        from loongguard.api.parent_routes import ParentAPIRoutes

        path = Path(ParentAPIRoutes._save_upload(
            self._FakeField("evil.html", b"<h1>xss</h1>")))
        assert path.suffix == ".png"
        assert path.exists()

    def test_jpg_extension_preserved(self, _temp_parent_db) -> None:
        from loongguard.api.parent_routes import ParentAPIRoutes

        path = Path(ParentAPIRoutes._save_upload(
            self._FakeField("photo.JPG", b"\xff\xd8\xff\xe0")))
        assert path.suffix == ".jpg"


# ── Excel 导入数值单元格兼容 ─────────────────────────────────


class TestParseDataImportExcel:
    def test_float_year_and_phone_cells(self, tmp_path: Path) -> None:
        """Excel 数值单元格经 openpyxl 常读到 float（2026.0 / 1.38e+10），
        不应被误判为'年级必须为数字'或生成科学计数法手机号"""
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["学生姓名", "班级名称", "年级", "家长姓名", "电话",
                   "远程观看权限", "家长OpenID（可选）"])
        ws.append(["张三", "中一班", 2026.0, "张三爸爸", 13800138001.0, 1.0, ""])
        path = tmp_path / "import.xlsx"
        wb.save(path)

        items = parent_utils.parse_data_import_excel(str(path))
        assert items[0]["year"] == 2026
        assert items[0]["phone"] == "13800138001"
        assert items[0]["errors"] == []


# ── 引导管理员与路径豁免精确匹配 ─────────────────────────────


class TestBootstrapAdmin:
    def test_bootstrap_admin_password_is_random(self, _temp_parent_db) -> None:
        """首次部署管理员口令必须随机（回归：旧实现固定 admin/admin123）"""
        from loongguard.db import parent_db

        parent_db.init_db()
        admin = parent_db.get_admin_by_username("admin")
        assert admin is not None
        assert not parent_utils.verify_pwd("admin123", admin["password"])
        assert parent_utils.hash_pwd("x")  # hash 格式由 passlib 保证


class TestParentPathMatching:
    def test_admin_prefix_precision(self) -> None:
        """/admin 豁免必须精确到 /admin 或 /admin/，不得误伤 /administrator"""
        from loongguard.api.server import _is_parent_path

        assert _is_parent_path("/admin")
        assert _is_parent_path("/admin/login")
        assert _is_parent_path("/admin/index")
        assert not _is_parent_path("/administrator")
        assert not _is_parent_path("/adminX")
        assert _is_parent_path("/api/wx/login")
        assert _is_parent_path("/api/admin/class/list")
        assert _is_parent_path("/static/css/app.css")
        assert not _is_parent_path("/api/v1/status")


# ── 流地址 token 签发（HLS 远程观看，方案 B）────────────────


class TestWxStreamUrl:
    @pytest.mark.asyncio
    async def test_stream_url_contains_signed_token(
        self, parent_server, _temp_parent_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """/api/wx/stream/url 必须下发带过期 HMAC token 的 m3u8 地址"""
        from loongguard.media.hls_publisher import verify_stream_token

        hls_secret = "wx-stream-test-secret"
        monkeypatch.setenv("LG_HLS_STREAM_SECRET", hls_secret)
        _seed_parent_with_student(_temp_parent_db)
        _, base_url = parent_server
        wx_token = parent_utils.generate_wx_token("openid-ut-001")
        async with aiohttp.ClientSession() as http:
            async with http.get(
                f"{base_url}/api/wx/stream/url?token={wx_token}&student_id=1",
            ) as resp:
                assert resp.status == 200
                body = await resp.json()
        assert body["code"] == 0
        stream_url = body["data"]["stream_url"]
        assert stream_url.startswith("http://")
        assert "/stream/live.m3u8?token=" in stream_url
        hls_token = stream_url.split("token=")[1]
        assert verify_stream_token(hls_secret, hls_token)
