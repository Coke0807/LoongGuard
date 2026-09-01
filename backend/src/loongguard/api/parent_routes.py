"""
家长端 API 路由（微信小程序 + 管理后台，迁移自 FastAPI）

设计动机：
    将分仓库 ParentModel 的家长端服务合并进主仓库 aiohttp 服务（8080 单端口），
    路由分三组：
        1. 小程序 API   POST /api/wx/*     —— Bearer JWT 鉴权，JSON body
        2. 管理后台 API  /api/admin/*      —— Cookie JWT 鉴权，参数走 query string
        3. 管理后台页面  /admin/*          —— Cookie JWT 鉴权，aiohttp_jinja2 渲染

    迁移原则：API 路径、参数位置、响应 JSON 结构与原 FastAPI 完全一致，
    前端（小程序页面 + 后台模板 JS）零改动。

鉴权说明：
    - 本组路由自带 JWT 鉴权，由 server.py 中间件按路径前缀豁免主仓库 Basic Auth
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from urllib.parse import quote

import aiohttp
import aiohttp_jinja2
import jinja2
from aiohttp import web

from loongguard.db import parent_db
from loongguard.utils import parent_utils

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[3]
_TEMPLATES_DIR = _BACKEND_DIR / "templates"
_STATIC_DIR = _BACKEND_DIR / "static"

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_PHONE_RE = re.compile(r"^1[3-9]\d{9}$")

# 家长端上传文件允许的扩展名（成长记录图片等）。
# 白名单外的扩展名一律落盘为 .png（内容不变，仅中和扩展名），
# 防止 .html/.svg 等可执行内容经 /uploads/ 静态路由形成同源存储型 XSS。
_UPLOAD_ALLOWED_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}


def _allow_time_periods() -> list:
    """允许远程观看时段，如 LG_PARENT_ALLOW_TIME_PERIODS=[["08:00","18:00"]]"""
    raw = os.environ.get("LG_PARENT_ALLOW_TIME_PERIODS", "")
    if not raw:
        return [["00:00", "23:59"]]
    try:
        import json
        periods = json.loads(raw)
        return [tuple(p) for p in periods]
    except (ValueError, TypeError):
        logger.warning("LG_PARENT_ALLOW_TIME_PERIODS 解析失败，回退为全天允许: %r", raw)
        return [["00:00", "23:59"]]


def _enroll_year() -> int:
    """幼儿入班默认年级（沿用原实现硬编码 2026，改为可配置）"""
    try:
        return int(os.environ.get("LG_PARENT_ENROLL_YEAR", "2026"))
    except ValueError:
        return 2026


class ParentAPIRoutes:
    """家长端路由集合：注册 + 全部 Handler"""

    def __init__(self) -> None:
        parent_db.init_db()
        # 启动期校验 JWT 密钥：空/占位密钥在生产环境直接阻断启动
        parent_utils.validate_jwt_secrets()
        # 运行期目录就绪
        os.makedirs(parent_utils.upload_dir(), exist_ok=True)

    # ── 注册 ────────────────────────────────────────────────

    def register(self, app: web.Application) -> None:
        """在主服务 app 上注册全部家长端路由 + Jinja2 环境 + 静态资源"""
        aiohttp_jinja2.setup(
            app, loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR))
        )

        # ── 小程序 API（POST JSON + Bearer JWT）──
        app.router.add_post("/api/wx/login", self._wx_login)
        app.router.add_post("/api/wx/bind-phone", self._wx_bind_phone)
        app.router.add_post("/api/wx/get-student-list", self._wx_student_list)
        app.router.add_post("/api/wx/update-info", self._wx_update_info)
        app.router.add_post("/api/wx/unbind-student", self._wx_unbind_student)
        app.router.add_post("/api/wx/get-notice-list", self._wx_notice_list)
        app.router.add_post("/api/wx/read-notice", self._wx_read_notice)
        app.router.add_post("/api/wx/get-growth-list", self._wx_growth_list)

        # ── 小程序兼容 API（query token，迁移自 website/wx_api.py）──
        app.router.add_get("/api/wx/student/list", self._wx_student_list_get)
        app.router.add_get("/api/wx/stream/url", self._wx_stream_url)

        # ── 管理后台 API（Cookie JWT + query 参数）──
        app.router.add_get("/api/admin/admin/list", self._admin_list)
        app.router.add_post("/api/admin/admin/create", self._admin_create)
        app.router.add_get("/api/admin/student/list", self._admin_student_list)
        app.router.add_post("/api/admin/student/add", self._admin_student_add)
        app.router.add_post("/api/admin/student/switch", self._admin_student_switch)
        app.router.add_get("/api/admin/student/export", self._admin_student_export)
        app.router.add_get("/api/admin/parent/list", self._admin_parent_list)
        app.router.add_post("/api/admin/parent/bind", self._admin_parent_bind)
        app.router.add_get("/api/admin/parent/export", self._admin_parent_export)
        app.router.add_get("/api/admin/class/list", self._admin_class_list)
        app.router.add_post("/api/admin/class/add", self._admin_class_add)
        app.router.add_put("/api/admin/class/edit", self._admin_class_edit)
        app.router.add_delete("/api/admin/class/delete", self._admin_class_delete)
        app.router.add_post("/api/admin/data/import", self._admin_data_import)
        app.router.add_get("/api/admin/class/export", self._admin_class_export)
        app.router.add_get("/api/admin/growth/list", self._admin_growth_list)
        app.router.add_post("/api/admin/growth/add", self._admin_growth_add)
        app.router.add_put("/api/admin/growth/edit", self._admin_growth_edit)
        app.router.add_delete("/api/admin/growth/delete", self._admin_growth_delete)
        app.router.add_get("/api/admin/notice/list", self._admin_notice_list)
        app.router.add_post("/api/admin/notice/add", self._admin_notice_add)
        app.router.add_put("/api/admin/notice/edit", self._admin_notice_edit)
        app.router.add_delete("/api/admin/notice/delete", self._admin_notice_delete)

        # ── 管理后台页面（Jinja2 渲染）──
        app.router.add_get("/admin/login", self._page_login)
        app.router.add_post("/admin/login", self._page_login_submit)
        app.router.add_get("/admin/index", self._page_index)
        app.router.add_get("/admin/parent", self._page_parent)
        app.router.add_get("/admin/student", self._page_student)
        app.router.add_get("/admin/account", self._page_account)
        app.router.add_get("/admin/import", self._page_import)
        app.router.add_get("/admin/growth", self._page_growth)
        app.router.add_get("/admin/notice", self._page_notice)
        app.router.add_get("/admin/class", self._page_class)
        app.router.add_get("/admin/template/data_import", self._page_download_template)
        app.router.add_get("/admin/logout", self._page_logout)

        # ── 静态资源与根路径 ──
        app.router.add_get("/", self._root_redirect)
        app.router.add_static("/static/", str(_STATIC_DIR))
        app.router.add_static("/uploads/", parent_utils.upload_dir())

        logger.info("Parent routes registered (mini-program + admin)")

    # ── 通用辅助 ────────────────────────────────────────────

    @staticmethod
    async def _json_body(request: web.Request) -> dict:
        """解析 JSON body（空 body / 非 JSON 一律返回空 dict）"""
        if not request.can_read_body:
            return {}
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    @staticmethod
    def _wx_openid_from_bearer(request: web.Request) -> str | None:
        """从 Authorization: Bearer <token> 解析家长 openid，失败返回 None"""
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return None
        payload = parent_utils.decode_wx_token(header[7:])
        if not payload or "openid" not in payload:
            return None
        return payload["openid"]

    @staticmethod
    def _wx_openid_from_query(request: web.Request) -> str | None:
        """从 ?token= 解析家长 openid（website 兼容接口用）"""
        token = request.query.get("token", "")
        if not token:
            return None
        payload = parent_utils.decode_wx_token(token)
        if not payload or "openid" not in payload:
            return None
        return payload["openid"]

    @staticmethod
    def _require_wx(request: web.Request) -> str:
        """小程序接口鉴权：失败直接抛 401（前端据此清除 token 重登录）"""
        openid = ParentAPIRoutes._wx_openid_from_bearer(request)
        if not openid:
            raise web.HTTPUnauthorized(
                text='{"code": -1, "msg": "登录失效，请重新登录", "data": null}',
                content_type="application/json",
            )
        return openid

    @staticmethod
    def _admin_user(request: web.Request) -> str | None:
        """从 admin_token Cookie 解析管理员用户名，失败返回 None"""
        token = request.cookies.get("admin_token")
        if not token:
            return None
        payload = parent_utils.decode_admin_token(token)
        if not payload or "username" not in payload:
            return None
        return payload["username"]

    @staticmethod
    def _require_admin_api(request: web.Request) -> str:
        """管理后台 API 鉴权：失败抛 401 JSON"""
        username = ParentAPIRoutes._admin_user(request)
        if not username:
            raise web.HTTPUnauthorized(
                text='{"code": -1, "msg": "管理员登录过期，请重新登录", "data": null}',
                content_type="application/json",
            )
        return username

    @staticmethod
    def _require_admin_page(request: web.Request) -> str:
        """后台页面鉴权：失败重定向登录页（优于原实现的裸 401）"""
        username = ParentAPIRoutes._admin_user(request)
        if not username:
            raise web.HTTPFound("/admin/login")
        return username

    @staticmethod
    def _page_params(request: web.Request) -> tuple[int, int, str]:
        page = int(request.query.get("page", "1") or "1")
        page_size = int(request.query.get("page_size", "10") or "10")
        keyword = request.query.get("keyword", "") or ""
        return page, page_size, keyword

    @staticmethod
    def _xlsx_response(data: bytes, filename: str) -> web.Response:
        encoded = f"attachment; filename*=UTF-8''{quote(filename)}"
        return web.Response(
            body=data,
            content_type=_XLSX_MIME,
            headers={"Content-Disposition": encoded},
        )

    @staticmethod
    def _save_upload(field, subdir: str = "") -> str:
        """把 multipart 文件字段写入上传目录，返回绝对路径"""
        ext = "png"
        if field.filename and "." in field.filename:
            candidate = field.filename.split(".")[-1].lower()
            if candidate in _UPLOAD_ALLOWED_EXTS:
                ext = candidate
        base = parent_utils.upload_dir()
        target_dir = os.path.join(base, subdir) if subdir else base
        os.makedirs(target_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(target_dir, filename)
        with open(save_path, "wb") as f:
            f.write(field.file.read())
        return save_path

    # ── 根路径 ──────────────────────────────────────────────

    @staticmethod
    async def _root_redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/admin/login")

    # ── 小程序 API：登录与家长信息 ──────────────────────────

    async def _wx_login(self, request: web.Request) -> web.Response:
        body = await self._json_body(request)
        code = str(body.get("code") or "")
        appid = os.environ.get("LG_PARENT_WX_APPID", "")
        appsecret = os.environ.get("LG_PARENT_WX_APPSECRET", "")
        if not appid or not appsecret:
            logger.warning("微信登录未配置：缺少 LG_PARENT_WX_APPID/LG_PARENT_WX_APPSECRET")
            return web.json_response({
                "code": -1, "msg": "微信登录未配置，请联系管理员", "data": None,
            })
        if not code:
            return web.json_response({"code": -1, "msg": "缺少登录code", "data": None})

        url = (
            "https://api.weixin.qq.com/sns/jscode2session"
            f"?appid={appid}&secret={appsecret}"
            f"&js_code={code}&grant_type=authorization_code"
        )
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.get(url) as resp:
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("微信 code2session 请求失败: %s", exc)
            return web.json_response({
                "code": -1, "msg": "微信授权服务不可达，请稍后重试", "data": None,
            })

        if "openid" not in data:
            return web.json_response({
                "code": -1, "msg": data.get("errmsg", "微信授权失败"), "data": None,
            })

        openid = data["openid"]
        parent = parent_db.get_parent_by_openid(openid)
        if not parent:
            parent_db.create_parent(openid)
            parent = parent_db.get_parent_by_openid(openid)

        token = parent_utils.generate_wx_token(openid)
        has_phone = bool(parent.get("phone"))
        bind_students = parent_db.get_parent_bind_students(openid) if has_phone else []
        return web.json_response({
            "code": 0,
            "msg": "success",
            "data": {"token": token, "has_phone": has_phone, "bind_students": bind_students},
        })

    async def _wx_bind_phone(self, request: web.Request) -> web.Response:
        openid = self._require_wx(request)
        body = await self._json_body(request)
        phone = str(body.get("phone") or "")
        nickname = body.get("nickname") or ""
        avatar = body.get("avatar") or ""

        if not _PHONE_RE.match(phone):
            return web.json_response({"code": -1, "msg": "手机号格式不正确", "data": None})

        success, msg, _parent = parent_db.bind_phone_to_openid(phone, openid)
        if not success:
            return web.json_response({"code": -1, "msg": msg, "data": None})

        if nickname or avatar:
            parent_db.update_parent_wx_info(openid, nickname or None, avatar or None)

        bind_students = parent_db.get_parent_bind_students(openid)
        return web.json_response({
            "code": 0, "msg": "绑定成功", "data": {"bind_students": bind_students},
        })

    async def _wx_student_list(self, request: web.Request) -> web.Response:
        openid = self._require_wx(request)
        students = parent_db.get_parent_bind_students(openid)
        return web.json_response({"code": 0, "data": students})

    async def _wx_update_info(self, request: web.Request) -> web.Response:
        openid = self._require_wx(request)
        body = await self._json_body(request)
        nickname = body.get("nickname") or ""
        avatar = body.get("avatar") or ""
        parent_db.update_parent_wx_info(openid, nickname or None, avatar or None)
        return web.json_response({"code": 0, "msg": "更新成功"})

    async def _wx_unbind_student(self, request: web.Request) -> web.Response:
        openid = self._require_wx(request)
        body = await self._json_body(request)
        try:
            student_id = int(body.get("student_id"))
        except (TypeError, ValueError):
            return web.json_response({"code": -1, "msg": "参数错误"})

        conn = parent_db.get_conn()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM parent WHERE openid = ?", (openid,))
            parent = cur.fetchone()
            if not parent:
                conn.close()
                return web.json_response({"code": -1, "msg": "家长信息不存在"})
            parent_id = parent["id"]

            cur.execute(
                "SELECT id FROM parent_student_bind WHERE parent_id = ? AND student_id = ?",
                (parent_id, student_id),
            )
            if not cur.fetchone():
                conn.close()
                return web.json_response({"code": -1, "msg": "未绑定该幼儿，无法解除"})

            cur.execute(
                "DELETE FROM parent_student_bind WHERE parent_id = ? AND student_id = ?",
                (parent_id, student_id),
            )
            conn.commit()
            conn.close()
            return web.json_response({"code": 0, "msg": "解除绑定成功"})
        except Exception as exc:
            conn.rollback()
            conn.close()
            logger.warning("解除绑定失败: %s", exc)
            return web.json_response({"code": -1, "msg": f"解除绑定失败: {exc}"})

    async def _wx_notice_list(self, request: web.Request) -> web.Response:
        openid = self._require_wx(request)
        notices = parent_db.get_notice_list(openid)
        return web.json_response({"code": 0, "data": notices})

    async def _wx_read_notice(self, request: web.Request) -> web.Response:
        openid = self._require_wx(request)
        body = await self._json_body(request)
        try:
            notice_id = int(body.get("notice_id"))
        except (TypeError, ValueError):
            return web.json_response({"code": -1, "msg": "参数错误"})
        if parent_db.mark_notice_read(openid, notice_id):
            return web.json_response({"code": 0, "msg": "已标记为已读"})
        return web.json_response({"code": -1, "msg": "标记失败"})

    async def _wx_growth_list(self, request: web.Request) -> web.Response:
        openid = self._require_wx(request)
        growth_list = parent_db.get_growth_list(openid)
        return web.json_response({"code": 0, "data": growth_list})

    # ── 小程序兼容 API（website/wx_api.py 迁移）────────────

    async def _wx_student_list_get(self, request: web.Request) -> web.Response:
        openid = self._wx_openid_from_query(request)
        if not openid:
            raise web.HTTPUnauthorized(
                text='{"code": -1, "msg": "登录失效，请重新微信授权"}',
                content_type="application/json",
            )
        data = parent_db.get_parent_bind_students(openid)
        return web.json_response({"code": 0, "data": data})

    async def _wx_stream_url(self, request: web.Request) -> web.Response:
        openid = self._wx_openid_from_query(request)
        if not openid:
            raise web.HTTPUnauthorized(
                text='{"code": -1, "msg": "登录失效，请重新微信授权"}',
                content_type="application/json",
            )
        try:
            student_id = int(request.query.get("student_id", "0"))
        except ValueError:
            student_id = 0

        # 流地址用请求自身的 Host 头拼装（合并后单端口，天然同源）。
        # 追加带过期时间的 HMAC 观看 token：/stream/live.m3u8 与分片端点
        # 均校验（家长端无 Basic Auth 凭据，token 是唯一的访问凭证）。
        stream_url = f"http://{request.host}/stream/live.m3u8"
        from loongguard.media.hls_publisher import (
            resolve_stream_secret,
            sign_stream_token,
            token_ttl_from_env,
        )

        secret = resolve_stream_secret()
        if secret:
            stream_url += f"?token={sign_stream_token(secret, token_ttl_from_env())}"
        else:
            logger.warning(
                "HLS 签名密钥未配置（LG_HLS_STREAM_SECRET/LG_SM4_KEY），"
                "下发的流地址将无法通过 token 校验"
            )
        client_ip = request.remote or ""
        if parent_utils.is_local_network(client_ip):
            return web.json_response(
                {"code": 0, "data": {"stream_url": stream_url, "allow_record": True}})

        ok, msg = parent_db.check_parent_permission(openid, student_id, _allow_time_periods())
        if not ok:
            return web.json_response({"code": -1, "msg": msg}, status=403)
        return web.json_response(
            {"code": 0, "data": {"stream_url": stream_url, "allow_record": False}})

    # ── 管理后台 API：管理员 ────────────────────────────────

    async def _admin_list(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        page, page_size, _ = self._page_params(request)
        total, data = parent_db.list_admin(page, page_size)
        return web.json_response({"code": 0, "total": total, "data": data})

    async def _admin_create(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        username = request.query.get("username", "")
        pwd = request.query.get("pwd", "")
        real_name = request.query.get("real_name", "")
        if not username or not pwd or not real_name:
            return web.json_response({"code": -1, "msg": "参数不完整"})

        if parent_db.get_admin_by_username(username):
            return web.json_response({"code": -1, "msg": "账号已存在"})

        conn = parent_db.get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO admin_account(username, password, real_name) VALUES (?,?,?)",
            (username, parent_utils.hash_pwd(pwd), real_name),
        )
        conn.commit()
        conn.close()
        return web.json_response({"code": 0, "msg": "管理员创建成功"})

    # ── 管理后台 API：幼儿 ──────────────────────────────────

    async def _admin_student_list(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        page, page_size, keyword = self._page_params(request)
        total, data = parent_db.list_student(page, page_size, keyword)
        return web.json_response({"code": 0, "total": total, "data": data})

    async def _admin_student_add(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        name = request.query.get("name", "")
        cls = request.query.get("cls", "")
        class_info = parent_db.get_class_by_name_year(cls, _enroll_year())
        if not class_info:
            return web.json_response({
                "code": -1, "msg": f"班级 '{cls}' 不存在，请先创建班级",
            })
        conn = parent_db.get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO student(student_name, class_name, class_id) VALUES (?,?,?)",
            (name, cls, class_info["id"]),
        )
        sid = cur.lastrowid
        conn.commit()
        conn.close()
        return web.json_response({"code": 0, "sid": sid, "msg": "添加成功"})

    async def _admin_student_switch(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        try:
            student_id = int(request.query.get("student_id", "0"))
            status = int(request.query.get("status", "1"))
        except ValueError:
            return web.json_response({"code": -1, "msg": "参数错误"})
        conn = parent_db.get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE student SET parent_watch_switch=? WHERE id=?", (status, student_id))
        conn.commit()
        conn.close()
        return web.json_response({"code": 0, "msg": "权限切换完成"})

    async def _admin_student_export(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        data = parent_utils.export_students_with_parents()
        return self._xlsx_response(data, "学生信息导出.xlsx")

    # ── 管理后台 API：家长 ──────────────────────────────────

    async def _admin_parent_list(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        page, page_size, keyword = self._page_params(request)
        total, data = parent_db.list_parent(page, page_size, keyword)
        return web.json_response({"code": 0, "total": total, "data": data})

    async def _admin_parent_bind(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        openid = request.query.get("openid", "")
        student_name = request.query.get("student_name", "")
        class_name = request.query.get("class_name", "")
        conn = parent_db.get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM parent WHERE openid = ?", (openid,))
            row = cur.fetchone()
            if not row:
                return web.json_response({"code": -1, "msg": "家长不存在"})
            parent_id = row[0]

            cur.execute(
                "SELECT id FROM student WHERE student_name = ? AND class_name = ?",
                (student_name, class_name),
            )
            student_row = cur.fetchone()
            if not student_row:
                return web.json_response({
                    "code": -1,
                    "msg": f"未找到学生：{student_name}（班级：{class_name}）",
                })
            student_id = student_row[0]

            cur.execute(
                "INSERT OR IGNORE INTO parent_student_bind (parent_id, student_id) VALUES (?,?)",
                (parent_id, student_id),
            )
            conn.commit()
            return web.json_response({"code": 0, "msg": "绑定成功"})
        finally:
            conn.close()

    async def _admin_parent_export(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        data = parent_utils.export_parents_with_students()
        return self._xlsx_response(data, "家长信息导出.xlsx")

    # ── 管理后台 API：班级 ──────────────────────────────────

    async def _admin_class_list(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        page, page_size, keyword = self._page_params(request)
        total, data = parent_db.list_class(page, page_size, keyword)
        return web.json_response({"code": 0, "total": total, "data": data})

    async def _admin_class_add(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        class_name = request.query.get("class_name", "")
        try:
            year = int(request.query.get("year", "0"))
        except ValueError:
            return web.json_response({"code": -1, "msg": "参数错误"})
        if parent_db.get_class_by_name_year(class_name, year):
            return web.json_response({"code": -1, "msg": "班级已存在"})
        cid = parent_db.create_class(class_name, year)
        return web.json_response({"code": 0, "id": cid, "msg": "添加成功"})

    async def _admin_class_edit(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        try:
            cid = int(request.query.get("id", "0"))
            year = int(request.query.get("year", "0"))
        except ValueError:
            return web.json_response({"code": -1, "msg": "参数错误"})
        class_name = request.query.get("class_name", "")
        if parent_db.update_class(cid, class_name, year):
            return web.json_response({"code": 0, "msg": "更新成功"})
        return web.json_response({"code": -1, "msg": "更新失败或班级不存在"})

    async def _admin_class_delete(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        try:
            cid = int(request.query.get("id", "0"))
        except ValueError:
            return web.json_response({"code": -1, "msg": "参数错误"})
        if parent_db.delete_class(cid):
            return web.json_response({"code": 0, "msg": "删除成功"})
        return web.json_response({"code": -1, "msg": "删除失败，该班级下存在学生"})

    async def _admin_class_export(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        data = parent_utils.export_class_with_students()
        return self._xlsx_response(data, "班级信息导出.xlsx")

    # ── 管理后台 API：Excel 批量导入 ────────────────────────

    async def _admin_data_import(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        form = await request.post()
        file_field = form.get("file")
        if file_field is None or not hasattr(file_field, "file"):
            return web.json_response({"code": -1, "msg": "仅支持Excel文件"})

        filename = file_field.filename or ""
        ext = filename.split(".")[-1] if "." in filename else ""
        if ext not in ("xlsx", "xls"):
            return web.json_response({"code": -1, "msg": "仅支持Excel文件"})

        save_path = os.path.join(
            parent_utils.upload_dir(), f"{uuid.uuid4()}.xlsx")
        with open(save_path, "wb") as f:
            f.write(file_field.file.read())

        items = parent_utils.parse_data_import_excel(save_path)
        if not items:
            return web.json_response({"code": -1, "msg": "表格无有效数据"})

        success = 0
        errors: list[str] = []
        for idx, item in enumerate(items, start=2):
            if item.get("errors"):
                errors.append(f"第{idx}行：{', '.join(item['errors'])}")
                continue
            try:
                class_info = parent_db.get_class_by_name_year(
                    item["class_name"], item["year"])
                if not class_info:
                    errors.append(
                        f"第{idx}行：班级 '{item['class_name']} ({item['year']}年)' 不存在")
                    continue
                class_id = class_info["id"]

                student = parent_db.get_student_by_name_class(
                    item["student_name"], class_id)
                if student:
                    parent_db.update_student(
                        student["id"], item["student_name"],
                        item["class_name"], class_id, item["watch_switch"])
                    student_id = student["id"]
                else:
                    student_id = parent_db.create_student(
                        item["student_name"], item["class_name"],
                        class_id, item["watch_switch"])

                parent = parent_db.get_parent_by_phone(item["phone"])
                if parent:
                    parent_id = parent["id"]
                    parent_db.update_parent(
                        parent_id, item["parent_name"], item["phone"])
                else:
                    openid = (item["openid"] if item["openid"]
                              else parent_utils.generate_openid(item["phone"]))
                    parent_id = parent_db.create_parent(
                        openid, item["parent_name"], item["phone"])

                parent_db.unbind_parent_student(parent_id, student_id)
                parent_db.bind_parent_student(parent_id, student_id)
                success += 1
            except Exception as exc:
                errors.append(f"第{idx}行：系统错误 {exc}")

        return web.json_response({
            "code": 0,
            "msg": f"成功导入 {success} 条，失败 {len(errors)} 条",
            "errors": errors,
        })

    # ── 管理后台 API：成长记录 ──────────────────────────────

    async def _admin_growth_list(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        page, page_size, keyword = self._page_params(request)
        total, data = parent_db.list_growth(page, page_size, keyword)
        return web.json_response({"code": 0, "total": total, "data": data})

    async def _admin_growth_add(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        title = request.query.get("title", "")
        record_date = request.query.get("record_date", "")
        sid_raw = request.query.get("student_id", "")
        try:
            student_id = int(sid_raw) if sid_raw not in ("", "None") else 0
        except ValueError:
            student_id = 0

        form = await request.post()
        image = form.get("image")
        if image is None or not hasattr(image, "file"):
            return web.json_response({"code": -1, "msg": "请上传图片"})

        save_path = self._save_upload(image, subdir="growth")
        image_path = f"/uploads/growth/{os.path.basename(save_path)}"
        sid = student_id if student_id and student_id > 0 else None
        gid = parent_db.create_growth(title, image_path, record_date, sid)
        return web.json_response({"code": 0, "msg": "添加成功", "id": gid})

    async def _admin_growth_edit(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        try:
            gid = int(request.query.get("id", "0"))
        except ValueError:
            return web.json_response({"code": -1, "msg": "参数错误"})
        title = request.query.get("title", "")
        record_date = request.query.get("record_date", "")
        sid_raw = request.query.get("student_id", "")
        try:
            student_id = int(sid_raw) if sid_raw not in ("", "None") else 0
        except ValueError:
            student_id = 0

        conn = parent_db.get_conn()
        cur = conn.cursor()
        cur.execute("SELECT image_path FROM growth_record WHERE id=?", (gid,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return web.json_response({"code": -1, "msg": "记录不存在"})
        image_path = row["image_path"]

        form = await request.post()
        image = form.get("image")
        if image is not None and hasattr(image, "file") and image.filename:
            save_path = self._save_upload(image, subdir="growth")
            image_path = f"/uploads/growth/{os.path.basename(save_path)}"

        sid = student_id if student_id and student_id > 0 else None
        parent_db.update_growth(gid, title, image_path, record_date, sid)
        return web.json_response({"code": 0, "msg": "更新成功"})

    async def _admin_growth_delete(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        try:
            gid = int(request.query.get("id", "0"))
        except ValueError:
            return web.json_response({"code": -1, "msg": "参数错误"})
        if parent_db.delete_growth(gid):
            return web.json_response({"code": 0, "msg": "删除成功"})
        return web.json_response({"code": -1, "msg": "删除失败"})

    # ── 管理后台 API：公告 ──────────────────────────────────

    async def _admin_notice_list(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        page, page_size, keyword = self._page_params(request)
        total, data = parent_db.list_notice(page, page_size, keyword)
        return web.json_response({"code": 0, "total": total, "data": data})

    async def _admin_notice_add(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        title = request.query.get("title", "")
        content = request.query.get("content", "")
        summary = request.query.get("summary", "")
        notice_type = request.query.get("notice_type", "notice")
        nid = parent_db.create_notice(title, content, summary, notice_type)
        return web.json_response({"code": 0, "msg": "添加成功", "id": nid})

    async def _admin_notice_edit(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        try:
            nid = int(request.query.get("id", "0"))
        except ValueError:
            return web.json_response({"code": -1, "msg": "参数错误"})
        title = request.query.get("title", "")
        content = request.query.get("content", "")
        summary = request.query.get("summary", "")
        notice_type = request.query.get("notice_type", "notice")
        parent_db.update_notice(nid, title, content, summary, notice_type)
        return web.json_response({"code": 0, "msg": "更新成功"})

    async def _admin_notice_delete(self, request: web.Request) -> web.Response:
        self._require_admin_api(request)
        try:
            nid = int(request.query.get("id", "0"))
        except ValueError:
            return web.json_response({"code": -1, "msg": "参数错误"})
        if parent_db.delete_notice(nid):
            return web.json_response({"code": 0, "msg": "删除成功"})
        return web.json_response({"code": -1, "msg": "删除失败"})

    # ── 管理后台页面 ────────────────────────────────────────

    async def _page_login(self, request: web.Request) -> web.Response:
        return aiohttp_jinja2.render_template("login.html", request, {})

    async def _page_login_submit(self, request: web.Request) -> web.Response:
        form = await request.post()
        username = form.get("username", "")
        password = form.get("password", "")

        admin = parent_db.get_admin_by_username(username)
        if not admin or not parent_utils.verify_pwd(password, admin["password"]):
            return aiohttp_jinja2.render_template(
                "login.html", request, {"err": "账号或密码错误"})

        token = parent_utils.generate_admin_token(username)
        resp = web.Response(status=303, headers={"Location": "/admin/index"})
        # httponly 防止 XSS 窃取 token；samesite=Lax 兼容登录后的 303 跳转
        resp.set_cookie(
            "admin_token", token, max_age=60 * 60 * 24 * 7,
            httponly=True, samesite="Lax",
        )
        return resp

    async def _page_index(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        return aiohttp_jinja2.render_template("index.html", request, {})

    async def _page_parent(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        return aiohttp_jinja2.render_template("parent_list.html", request, {})

    async def _page_student(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        return aiohttp_jinja2.render_template("student_list.html", request, {})

    async def _page_account(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        return aiohttp_jinja2.render_template("admin_list.html", request, {})

    async def _page_import(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        return aiohttp_jinja2.render_template("import_page.html", request, {})

    async def _page_growth(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        return aiohttp_jinja2.render_template("growth_list.html", request, {})

    async def _page_notice(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        return aiohttp_jinja2.render_template("notice_list.html", request, {})

    async def _page_class(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        return aiohttp_jinja2.render_template("class_list.html", request, {})

    async def _page_download_template(self, request: web.Request) -> web.Response:
        self._require_admin_page(request)
        path = parent_utils.create_data_import_template()
        with open(path, "rb") as f:
            data = f.read()
        return self._xlsx_response(data, "数据导入模板.xlsx")

    @staticmethod
    async def _page_logout(request: web.Request) -> web.Response:
        resp = web.Response(status=303, headers={"Location": "/admin/login"})
        resp.del_cookie("admin_token")
        return resp
