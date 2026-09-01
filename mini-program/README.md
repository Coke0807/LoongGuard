# mini-program — 微信小程序（家长端）

家长侧微信小程序：绑定幼儿、查看公告/成长记录、远程观看授权管理。

## 页面清单

| 页面 | 功能 | 依赖的后端接口 |
|---|---|---|
| `pages/index` | 首页/登录引导 | `POST /api/wx/login` |
| `pages/bindPhone` | 手机号绑定（匹配后台预录入家长） | `POST /api/wx/bind-phone` |
| `pages/monitor` | 远程观看（HLS 直播：绑定幼儿列表 → `/api/wx/stream/url` 取带 token 的 m3u8 → `<video>` 播放；全屏/声音/刷新） | `GET /api/wx/stream/url`、`POST /api/wx/get-student-list` |
| `pages/notices` | 公告列表/已读标记 | `POST /api/wx/get-notice-list`、`/read-notice` |
| `pages/growth` | 成长记录相册 | `POST /api/wx/get-growth-list` |

（完整页面树见 `app.json`；后端全部接口清单见
[docs/04-远程访问与视频流/家长端与小程序接入.md](../docs/04-远程访问与视频流/家长端与小程序接入.md)。）

## 配置

`config.js` 是唯一的后端地址配置：

```js
const config = {
  env: "development",
  apiBaseUrl: "http://127.0.0.1:8080/api/wx",   // 开发：微信开发者工具本地调试
  fileBaseUrl: "http://127.0.0.1"
}
```

- 局域网调试：把 `127.0.0.1` 换成后端所在机器 IP，并在开发者工具勾选
  「不校验合法域名」。
- 正式发布：必须替换为 HTTPS 域名（小程序不允许 HTTP 与 IP 直连），并在
  微信公众平台「开发管理 → 服务器域名」登记；同时后端须配套 HTTPS 反向代理
  （见 `backend/deploy/Caddyfile`）。
- appid：`project.config.json` 使用 `wxd0037d52450fefd0`，须与后端
  `LG_PARENT_WX_APPID/LG_PARENT_WX_APPSECRET` 成对配置。
