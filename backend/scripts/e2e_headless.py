"""
LoongGuard 端到端 headless 验证脚本

设计动机：
    scripts/visual_test.py 依赖 cv2.imshow()，需要 GUI 显示器。
    在无显示器/CI/Windows 服务器环境下无法运行。
    本脚本复用 visual_test.py 的核心模块初始化逻辑，
    但绕过 GUI，直接统计运行结果 + 写截图到 data/screenshots/。

用法：
    python scripts/e2e_headless.py
    python scripts/e2e_headless.py --frames 100        # 只跑前 N 帧
    python scripts/e2e_headless.py --source video.mp4  # 指定视频
    python scripts/e2e_headless.py --no-pose           # 跳过姿态估计

输出：
    控制台报告：帧数、检测数、姿态数、平均 FPS、降级状态
    截图：data/screenshots/headless_*.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).parent.parent

from config import load_config  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.camera.v4l2_capture import V4L2Capture  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.detection.yolo26_nano import YOLO26Nano  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.motion.frame_diff import FrameDiffDetector  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后
from loongguard.pose.movenet import MoveNetLightning  # noqa: E402 本地包导入需在 PROJECT_ROOT 设置之后


def main() -> int:
    parser = argparse.ArgumentParser(description="LoongGuard headless E2E")
    parser.add_argument("--source", default="tests/mock_classroom.mp4")
    parser.add_argument("--frames", type=int, default=60, help="最多处理帧数")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--no-pose", action="store_true")
    parser.add_argument(
        "--save-screenshots", action="store_true",
        help="每隔 N 帧保存一张截图到 data/screenshots/",
    )
    parser.add_argument(
        "--screenshot-interval", type=int, default=20,
        help="截图间隔(每 N 帧保存一张)",
    )
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────
    config = load_config()
    config.camera.device = args.source
    config.detection.conf_threshold = args.conf

    # 关闭 debug window（headless 场景下不需要）
    config.debug = False

    print("=" * 60)
    print(" LoongGuard Headless E2E Verification")
    print("=" * 60)
    print(f"Source:        {args.source}")
    print(f"Max frames:    {args.frames}")
    print(f"Conf:          {args.conf}")
    print(f"Pose enabled:  {not args.no_pose}")
    print()

    # ── 初始化模块 ────────────────────────────────────────────
    t0 = time.perf_counter()

    print("[1/4] Camera init...")
    camera = V4L2Capture(config.camera)
    camera.open()
    print(f"      OK ({camera.frame_count} frames read so far)")

    print("[2/4] YOLO26-Nano init...")
    detector = YOLO26Nano(config.detection)
    detector.load_model()
    print(f"      OK (available={detector.is_available()})")
    det_snap = detector.get_health_snapshot()
    print(f"      model: {Path(det_snap['model_path']).name}")
    print(f"      input: {det_snap['input_size']}x{det_snap['input_size']}")

    print("[3/4] Motion detector init...")
    motion = FrameDiffDetector(config.motion)
    print("      OK")

    pose = None
    if not args.no_pose:
        print("[4/4] MoveNet init...")
        pose = MoveNetLightning(config.pose)
        pose.load_model()
        print(f"      OK (available={pose.is_available()})")
        if pose.is_available():
            print(f"      model: {Path(pose._model_path).name}")
            print(f"      input: {pose._config.input_size}x{pose._config.input_size}")
    else:
        print("[4/4] MoveNet: SKIPPED (--no-pose)")

    init_elapsed = time.perf_counter() - t0
    print(f"\nInit time: {init_elapsed:.2f}s")
    print("-" * 60)

    # ── 主循环 ────────────────────────────────────────────────
    prev_gray = None
    frame_times: list[float] = []
    total_detections = 0
    total_motion_regions = 0
    pose_inference_count = 0
    pose_throttled = 0
    first_detection_at: int | None = None
    first_pose_at: int | None = None

    screenshot_dir = PROJECT_ROOT / "data" / "screenshots"
    if args.save_screenshots:
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    print("Processing frames...")
    for frame_idx in range(args.frames):
        t_start = time.perf_counter()
        frame_obj = camera.read()
        if frame_obj is None:
            print(f"[END] Video ended at frame {frame_idx}")
            break

        # 灰度用于运动检测
        display = cv2.cvtColor(frame_obj.data, cv2.COLOR_RGB2BGR)
        current_gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

        # 运动检测
        regions = motion.detect(current_gray, prev_gray)
        prev_gray = current_gray
        total_motion_regions += len(regions)

        # 目标检测（解耦后仅当有运动 + 模型可用时跑）
        all_boxes = []
        if regions and detector.is_available():
            all_boxes = detector.infer(frame_obj.data)
            total_detections += len(all_boxes)
            if all_boxes and first_detection_at is None:
                first_detection_at = frame_idx

        # 姿态估计（节流 + 解耦）
        if pose is not None:
            if pose.is_available() and pose.should_run(frame_idx):
                keypoints = pose._infer_keypoints(frame_obj.data)
                pose_inference_count += 1
                if keypoints is not None and first_pose_at is None:
                    first_pose_at = frame_idx
                pose.mark_ran(frame_idx)
            else:
                pose_throttled += 1

        # 截图
        if args.save_screenshots and frame_idx % args.screenshot_interval == 0:
            # 标注
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, 0), (w, 36), (0, 0, 0), -1)
            cv2.putText(
                display,
                f"Frame {frame_idx} | D={len(all_boxes)} M={len(regions)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            for box in all_boxes:
                cv2.rectangle(
                    display,
                    (box.x1, box.y1), (box.x2, box.y2),
                    (0, 0, 255), 2,
                )
            path = screenshot_dir / f"headless_{frame_idx:06d}.jpg"
            cv2.imwrite(str(path), display)

        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times = frame_times[-30:]

    # ── 汇总 ────────────────────────────────────────────────
    camera.close()
    avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
    avg_latency_ms = (sum(frame_times) / len(frame_times)) * 1000 if frame_times else 0

    print("-" * 60)
    print(" RESULTS")
    print("-" * 60)
    print(f"  Total frames processed:  {len(frame_times)}")
    print(f"  Average latency:         {avg_latency_ms:.1f} ms/frame")
    print(f"  Average FPS:             {avg_fps:.1f}")
    print(f"  Motion regions total:    {total_motion_regions}")
    print(f"  YOLO detections total:   {total_detections}")
    print(f"  First detection at:      frame {first_detection_at}")
    print(f"  Pose inferences:         {pose_inference_count}")
    print(f"  Pose throttled:          {pose_throttled}")
    print(f"  First pose at:           frame {first_pose_at}")
    print()

    # 末态健康快照
    print(" HEALTH SNAPSHOT")
    print("-" * 60)
    ds = detector.get_health_snapshot()
    print(f"  YOLO26:    available={ds['available']}, "
          f"success={ds['success_count']}, error={ds['error_count']}")
    if pose is not None:
        ps = pose.get_health_snapshot()
        print(f"  MoveNet:   available={ps['available']}, "
              f"success={ps['success_count']}, error={ps['error_count']}")
    print()

    # 验证
    if avg_fps >= 5:
        print("  [PASS] FPS 达到 5+ (可交互响应)")
    else:
        print("  [WARN] FPS < 5,端到端体验可能卡顿")

    if detector.is_available():
        print("  [PASS] YOLO26 在线")
    else:
        print("  [FAIL] YOLO26 不可用")

    if pose is not None and not pose.is_available():
        print("  [WARN] MoveNet 不可用(俯卧检测已降级)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
