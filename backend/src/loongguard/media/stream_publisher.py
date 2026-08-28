"""
流媒体发布器抽象接口（微信小程序集成预留）

设计动机：
    家长端小程序需远程查看摄像头画面（低延迟视频流）。流媒体协议
    选型为 WebRTC（小程序 live-player 原生支持），但 WebRTC 需板端
    loongarch 库支持，当前阶段不宜绑定。本模块定义 StreamPublisher
    接口：

        - MJPEGStreamPublisher：开发期 / 局域网 Web 管理端桩（复用 VideoHub）
        - WebRTCStreamPublisher：板端未来实现（当前为占位，抛 NotImplementedError）

    业务层（pipeline / api）只依赖 StreamPublisher 接口，
    板端接入 WebRTC 时替换实现即可，不影响调用方。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class StreamPublisher(ABC):
    """
    视频流发布器协议

    方法：
        start():        启动发布（建立传输通道）
        publish(frame):  发布一帧 RGB 图像
        stop():         停止并释放资源
    """

    @abstractmethod
    def start(self) -> None:
        """启动发布通道"""

    @abstractmethod
    def publish(self, frame: np.ndarray) -> None:
        """发布一帧 RGB 图像 (H, W, 3)"""

    @abstractmethod
    def stop(self) -> None:
        """停止并释放资源"""


class MJPEGStreamPublisher(StreamPublisher):
    """
    MJPEG 流发布器（开发期 / 局域网桩）

    复用后端 VideoHub 已实现的 MJPEG 编码与 /stream 端点，
    作为板端 WebRTC 前的过渡实现（Windows 开发与板端演示均可用）。
    """

    def __init__(self, video_hub) -> None:
        # video_hub: src.api.server.VideoHub 实例（延迟注入，避免循环依赖）
        self._video_hub = video_hub

    def start(self) -> None:
        # MJPEG 通道由 API 服务生命周期管理，此处无需额外启动
        pass

    def publish(self, frame: np.ndarray) -> None:
        if self._video_hub is not None:
            self._video_hub.push_frame(frame)

    def stop(self) -> None:
        pass


class WebRTCStreamPublisher(StreamPublisher):
    """
    WebRTC 流发布器（板端未来实现占位）

    面向小程序 live-player 的低延迟推流。
    板端需编译 loongarch 的 WebRTC 库（如 aiortc）后实现 publish 逻辑。
    当前仅预留接口，调用即抛 NotImplementedError，避免静默空跑。
    """

    def __init__(self, signal_port: int = 8888) -> None:
        self._signal_port = signal_port

    def start(self) -> None:
        raise NotImplementedError(
            "WebRTCStreamPublisher 尚未实现：板端需接入 loongarch WebRTC 库"
        )

    def publish(self, frame: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError(
            "WebRTCStreamPublisher 尚未实现：板端需接入 loongarch WebRTC 库"
        )

    def stop(self) -> None:  # pragma: no cover
        raise NotImplementedError(
            "WebRTCStreamPublisher 尚未实现：板端需接入 loongarch WebRTC 库"
        )


def create_stream_publisher(
    kind: str = "mjpeg", video_hub=None, signal_port: int = 8888
) -> StreamPublisher:
    """
    依据配置创建流媒体发布器

    Args:
        kind: "mjpeg"（桩）/ "webrtc"（板端预留）
        video_hub: MJPEG 发布器所需的 VideoHub 实例
        signal_port: WebRTC 信令端口

    Returns:
        StreamPublisher 实现实例
    """
    if kind == "webrtc":
        return WebRTCStreamPublisher(signal_port=signal_port)
    return MJPEGStreamPublisher(video_hub)
