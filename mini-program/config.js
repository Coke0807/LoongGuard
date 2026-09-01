/*
 * 后端接口地址配置
 *
 * 【重要】当前为开发环境配置，适用于：
 *   1. 微信开发者工具本地调试
 *   2. LoongGuard 主后端（aiohttp，含家长端/管理后台 API）运行在本机 127.0.0.1:8080
 *
 * 部署到龙芯板/局域网时，将 127.0.0.1 替换为后端所在机器的局域网 IP，
 * 例如 apiBaseUrl: "http://192.168.1.100:8080/api/wx"。
 *
 * 正式发布前，必须修改 apiBaseUrl 为 HTTPS 域名，例如：
 *   apiBaseUrl: "https://your-domain.com/api/wx"
 *
 * 注意事项：
 *   - 小程序正式环境只允许 HTTPS 请求，不支持 HTTP 和 IP 地址
 *   - 不要把 127.0.0.1 或 localhost 当作正式发布地址
 *   - 正式域名需要在微信公众平台 → 开发管理 → 服务器域名中配置
 *   - 开发阶段可在微信开发者工具中勾选"不校验合法域名"进行调试
 */

const config = {
  env: "development",
  apiBaseUrl: "http://127.0.0.1:8080/api/wx",
  fileBaseUrl: "http://127.0.0.1:8080"
}

module.exports = config
