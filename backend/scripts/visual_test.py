"""
Windows 可视化测试 -- 实时展示摄像头画面 + 检测框 + 告警信息

用法：
    python scripts/visual_test.py                       # 使用 mock 视频
    python scripts/visual_test.py --camera 0            # 使用 USB 摄像头
    python scripts/visual_test.py --source my_video.mp4 # 使用指定视频文件

按 Q 退出，按 S 截图保存到 data/screenshots/

功能：
    - 实时显示摄像头画面（RGB）
    - YOLO26 检测框叠加显示（类别 + 置信度）
    - MoveNet 关键点叠加显示
    - 帧率 / 检测数 / 运动状态 HUD 信息
    - 告警计数（模拟，有检测结果时递增）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config
from loongguard.camera.v4l2_capture import V4L2Capture
from loongguard.motion.frame_diff import FrameDiffDetector
from loongguard.detection.yolo26_nano import YOLO26Nano
from loongguard.pose.movenet import MoveNetLightning


# ── 颜色常量 ──────────────────────────────────────────────────

_COLORS = {
    "magnetic_bead":   (0, 0, 255),      # 红
    "button_battery":  (0, 0, 255),      # 红
    "scissors":        (0, 165, 255),    # 橙
    "utility_knife":   (0, 165, 255),    # 橙
    "needle":          (0, 255, 255),    # 黄
    "glass_shard":     (255, 0, 0),      # 蓝
    "wire":            (255, 255, 0),    # 青
    "small_toy_part":  (255, 0, 255),    # 紫
}

# MoveNet 17 关键点连线（骨骼图）
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # 头部
    (5, 6),                                  # 肩膀
    (5, 7), (7, 9),                          # 左臂
    (6, 8), (8, 10),                         # 右臂
    (5, 11), (6, 12),                        # 躯干
    (11, 12),                                # 髋部
    (11, 13), (13, 15),                      # 左腿
    (12, 14), (14, 16),                      # 右腿
]


def draw_hud(frame: np.ndarray, fps: float, n_boxes: int,
             n_motion: int, n_alerts: int, sm4_count: int) -> None:
    """在画面上绘制 HUD 信息"""
    h, w = frame.shape[:2]
    # 半透明背景条
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    info = (f"FPS: {fps:.1f}  |  "
            f"Detections: {n_boxes}  |  "
            f"Motion: {'YES' if n_motion > 0 else 'no'}  |  "
            f"Alerts: {n_alerts}  |  "
            f"SM4 encrypted: {sm4_count}  |  "
            f"Press Q to quit")
    cv2.putText(frame, info, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_boxes(frame: np.ndarray, boxes: list) -> None:
    """绘制检测框"""
    for box in boxes:
        color = _COLORS.get(box.class_name, (255, 255, 255))
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        label = f"{box.class_name}: {box.confidence:.2f}"
        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (box.x1, box.y1 - th - 6), (box.x1 + tw, box.y1), color, -1)
        cv2.putText(frame, label, (box.x1, box.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_keypoints(frame: np.ndarray, keypoints: np.ndarray,
                   min_score: float = 0.3) -> None:
    """绘制 MoveNet 关键点和骨骼连线"""
    h, w = frame.shape[:2]

    # 画连线
    for i, j in _SKELETON:
        if keypoints[i, 2] < min_score or keypoints[j, 2] < min_score:
            continue
        pt1 = (int(keypoints[i, 1] * w), int(keypoints[i, 0] * h))
        pt2 = (int(keypoints[j, 1] * w), int(keypoints[j, 0] * h))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # 画关键点
    for i in range(17):
        if keypoints[i, 2] < min_score:
            continue
        cx = int(keypoints[i, 1] * w)
        cy = int(keypoints[i, 0] * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)


def draw_motion_regions(frame: np.ndarray, regions: list) -> None:
    """绘制运动区域（半透明蓝色框）"""
    overlay = frame.copy()
    for r in regions:
        cv2.rectangle(overlay, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 100, 0), 2)
        label = f"motion {r.motion_ratio:.1%}"
        cv2.putText(overlay, label, (r.x, r.y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


def main():
    parser = argparse.ArgumentParser(description="LoongGuard 可视化测试")
    parser.add_argument("--source", default=None, help="视频源路径（默认 mock 视频）")
    parser.add_argument("--camera", type=int, default=None, help="USB 摄像头索引")
    parser.add_argument("--conf", type=float, default=0.45, help="检测置信度阈值")
    parser.add_argument("--no-pose", action="store_true", help="跳过姿态估计（加速）")
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────
    config = load_config()
    config.detection.conf_threshold = args.conf

    if args.camera is not None:
        config.camera.device = str(args.camera)
    elif args.source is not None:
        config.camera.device = args.source
    else:
        config.camera.device = "tests/mock_classroom.mp4"

    # ── 初始化各模块 ──────────────────────────────────────────
    print("[INIT] Loading models...")

    camera = V4L2Capture(config.camera)
    camera.open()
    print(f"  Camera: opened ({camera.frame_count} frames)")

    detector = YOLO26Nano(config.detection)
    detector.load_model()
    print("  YOLO26: loaded")

    motion = FrameDiffDetector(config.motion)

    pose = None
    if not args.no_pose:
        pose = MoveNetLightning(config.pose)
        pose.load_model()
        print("  MoveNet: loaded")

    print("[INIT] Ready. Starting visualization...")
    print()

    # ── 主循环 ──────────────────────────────────────────────
    prev_gray = None
    sm4_count = 0
    alert_count = 0
    frame_times: list[float] = []
    window_name = "LoongGuard - Detection Visualization"

    while True:
        t_start = time.perf_counter()

        frame_obj = camera.read()
        if frame_obj is None:
            print("[END] Video ended")
            break

        # BGR 用于 OpenCV 显示
        display = cv2.cvtColor(frame_obj.data, cv2.COLOR_RGB2BGR)

        # 灰度用于运动检测
        current_gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

        # 运动检测
        regions = motion.detect(current_gray, prev_gray)
        prev_gray = current_gray

        # 检测（有运动时才推理，节省算力）
        all_boxes = []
        if regions:
            all_boxes = detector.infer(frame_obj.data)

        # 姿态估计
        keypoints = None
        if pose is not None:
            # 直接调用 _infer_keypoints 获取原始关键点（不触发告警逻辑）
            keypoints = pose._infer_keypoints(frame_obj.data)

        # 模拟 SM4 加密计数（有检测结果时 +1）
        if all_boxes:
            sm4_count += 1
            alert_count += len(all_boxes)

        # ── 绘制 ──────────────────────────────────────────────
        draw_motion_regions(display, regions)
        draw_boxes(display, all_boxes)

        if keypoints is not None:
            draw_keypoints(display, keypoints, config.pose.min_keypoint_score)

        # FPS 计算
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times = frame_times[-30:]
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

        draw_hud(display, fps, len(all_boxes), len(regions), alert_count, sm4_count)

        # 告警高亮闪烁（有检测结果时边框变红）
        if all_boxes:
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
            # 顶部告警文字
            sev = max(all_boxes, key=lambda b: b.confidence)
            cv2.putText(display, f"ALERT: {sev.class_name} ({sev.confidence:.0%})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, display)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[USER] Quit")
            break
        elif key == ord('s') or key == ord('S'):
            save_dir = PROJECT_ROOT / "data" / "screenshots"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"frame_{camera.frame_count:06d}.jpg"
            cv2.imwrite(str(save_path), display)
            print(f"[SAVE] Screenshot: {save_path}")

    # ── 清理 ──────────────────────────────────────────────────
    camera.close()
    cv2.destroyAllWindows()

    print()
    print("=" * 50)
    print(f"  Total frames:     {camera.frame_count}")
    print(f"  Total alerts:     {alert_count}")
    print(f"  SM4 encrypted:    {sm4_count}")
    print(f"  Average FPS:      {fps:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
