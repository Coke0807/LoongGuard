"""
外部告警通知分发器

设计动机：
    告警仅推送到 WebSocket + 入库，依赖 WS 客户端在线，无人值守时漏报风险高。
    本模块将达到阈值（默认 HIGH/CRITICAL）的告警通过 webhook POST 推送到
    外部网关（微信小程序 / 短信 / 电话），并带失败重试与指数退避。

扩展预留：
    后续可增加 SMSSender / WeChatSender 等适配器，均提供相同的
    notify(alert) -> bool 接口，Pipeline 只依赖本类，不接触具体通道。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from config import NotificationConfig
from loongguard.utils.schema import AlertLog, AlertSeverity

logger = logging.getLogger(__name__)

# 严重等级排序（用于阈值判定）
_SEVERITY_ORDER: dict[AlertSeverity, int] = {
    AlertSeverity.LOW: 0,
    AlertSeverity.MEDIUM: 1,
    AlertSeverity.HIGH: 2,
    AlertSeverity.CRITICAL: 3,
}


def _parse_severity(value: str) -> int:
    """将配置的严重等级字符串解析为排序值，非法时回退到 high。"""
    try:
        sev = AlertSeverity(value.strip().lower())
    except ValueError:
        logger.warning(
            "无效的 LG_NOTIFY_MIN_SEVERITY=%r，回退到 high", value
        )
        sev = AlertSeverity.HIGH
    return _SEVERITY_ORDER.get(sev, 2)


class AlertNotifier:
    """基于 webhook 的外部告警通知器（带失败重试与指数退避）"""

    def __init__(self, config: NotificationConfig) -> None:
        self._config = config
        self._session: Optional[Any] = None
        self._min_severity = _parse_severity(config.min_severity)

    async def start(self) -> None:
        """初始化 HTTP 会话（延迟创建，避免未启用时空占资源）"""
        if self._session is None and self._config.enabled:
            import aiohttp

            self._session = aiohttp.ClientSession()

    async def stop(self) -> None:
        """关闭 HTTP 会话"""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def notify(self, alert: AlertLog) -> bool:
        """
        推送告警通知。

        未启用 / 未配置 webhook / 未达阈值时直接返回 True（视为无需通知）；
        否则带重试推送，返回最终是否成功。
        """
        if not self._config.enabled or not self._config.webhook_url.strip():
            return True
        if _SEVERITY_ORDER.get(alert.severity, 0) < self._min_severity:
            return True
        return await self._send_with_retry(alert)

    async def _send_with_retry(self, alert: AlertLog) -> bool:
        """带指数退避的重试推送"""
        payload = {
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp,
            "severity": alert.severity.value,
            "alert_type": alert.alert_type.value,
            "description": alert.description,
        }

        for attempt in range(1, self._config.retries + 1):
            try:
                await self._post(payload)
                logger.info(
                    "webhook 通知成功: %s (尝试 %d)", alert.alert_id, attempt
                )
                return True
            except Exception as exc:
                wait = self._config.retry_backoff_sec * (2 ** (attempt - 1))
                logger.warning(
                    "webhook 通知失败(%d/%d): %s; %.1fs 后重试",
                    attempt, self._config.retries, exc, wait,
                )
                if attempt < self._config.retries:
                    await asyncio.sleep(wait)

        logger.error("webhook 通知最终失败: %s", alert.alert_id)
        return False

    async def _post(self, payload: dict) -> None:
        """发送单个 POST 请求，非 2xx 视为失败"""
        if self._session is None:
            raise RuntimeError("AlertNotifier 未启动，请先调用 start()")

        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_sec)
        async with self._session.post(
            self._config.webhook_url,
            json=payload,
            timeout=timeout,
        ) as resp:
            if resp.status >= 300:
                raise RuntimeError(f"webhook HTTP 状态码 {resp.status}")