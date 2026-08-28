"""
LoongGuard 双模式 FPS 对比 benchmark

设计动机：
    用户问"去掉 MoveNet 能否提高帧数"。这是性能问题,必须实测。
    同一份代码、同一段视频,跑两次(有 MoveNet / 无 MoveNet),对比结果。

用法：
    python scripts/benchmark_pose_vs_no_pose.py
    python scripts/benchmark_pose_vs_no_pose.py --frames 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from src.camera.v4l2_capture import V4L2Capture
from src.detection.yolo26_nano import YOLO26Nano
from src.motion.frame_diff import FrameDiffDetector
from src.pose.movenet import MoveNetLightning


def run_one_mode(
    name: str,
    source: str,
    frames: int,
    conf: float,
    enable_pose: bool,
    pose_interval: int,
    warmup: int,
) -> dict:
    """跑一次端到端,返回结果字典"""
    print(f"\n{'=' * 60}")
    print(f" MODE: {name}")
    print(f"{'=' * 60}")

    config = load_config()
    config.camera.device = source
    config.detection.conf_threshold = conf
    config.pose.inference_interval = pose_interval

    # 初始化(冷启动耗时记入 init)
    t_init_start = time.perf_counter()
    camera = V4L2Capture(config.camera)
    camera.open()
    detector = YOLO26Nano(config.detection)
    detector.load_model()
    motion = FrameDiffDetector(config.motion)
    pose = None
    if enable_pose:
        pose = MoveNetLightning(config.pose)
        pose.load_model()
    t_init = time.perf_counter() - t_init_start
    print(f"  init: {t_init:.2f}s, pose={'ON' if enable_pose else 'OFF'}")

    prev_gray = None
    frame_times: list[float] = []

    # 预热(warmup):先跑 N 帧,不计入统计,避免冷启动影响
    for _ in range(warmup):
        frame_obj = camera.read()
        if frame_obj is None:
            break
        display = cv2.cvtColor(frame_obj.data, cv2.COLOR_RGB2BGR)
        prev_gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

    # 正式测试
    pose_calls = 0
    for i in range(frames):
        t0 = time.perf_counter()
        frame_obj = camera.read()
        if frame_obj is None:
            break
        display = cv2.cvtColor(frame_obj.data, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        regions = motion.detect(gray, prev_gray)
        prev_gray = gray
        if regions:
            detector.infer(frame_obj.data)
        if pose is not None and pose.is_available() and pose.should_run(i):
            pose._infer_keypoints(frame_obj.data)
            pose_calls += 1
            pose.mark_ran(i)
        frame_times.append(time.perf_counter() - t0)

    camera.close()

    # 统计
    arr = np.array(frame_times[5:])  # 丢弃前 5 帧避免离群值
    avg = float(arr.mean()) * 1000 if len(arr) > 0 else 0
    p50 = float(np.percentile(arr, 50)) * 1000 if len(arr) > 0 else 0
    p95 = float(np.percentile(arr, 95)) * 1000 if len(arr) > 0 else 0
    fps = 1000.0 / avg if avg > 0 else 0
    print(f"  samples:    {len(arr)} frames (after warmup discard)")
    print(f"  avg:        {avg:.1f} ms/frame  ->  {fps:.1f} FPS")
    print(f"  p50:        {p50:.1f} ms")
    print(f"  p95:        {p95:.1f} ms")
    print(f"  pose calls: {pose_calls}/{frames}")
    return {
        "name": name,
        "avg_ms": avg,
        "p50_ms": p50,
        "p95_ms": p95,
        "fps": fps,
        "pose_calls": pose_calls,
        "init_ms": t_init * 1000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="tests/mock_classroom.mp4")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--pose-interval", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print(" LoongGuard Pose Impact Benchmark")
    print(f" source: {args.source}")
    print(f" frames: {args.frames} (warmup {args.warmup} discarded)")
    print(f" pose_interval: {args.pose_interval}")
    print("=" * 60)

    # 跑两次(顺序对调避免单次噪声,实际取两次平均)
    # 第一轮:先 WITHOUT 再 WITH
    a = run_one_mode(
        "WITHOUT MoveNet (run 1)",
        args.source, args.frames, args.conf,
        enable_pose=False, pose_interval=args.pose_interval,
        warmup=args.warmup,
    )
    b = run_one_mode(
        "WITH MoveNet (run 1)",
        args.source, args.frames, args.conf,
        enable_pose=True, pose_interval=args.pose_interval,
        warmup=args.warmup,
    )

    # 第二轮:反过来,先 WITH 再 WITHOUT
    c = run_one_mode(
        "WITH MoveNet (run 2)",
        args.source, args.frames, args.conf,
        enable_pose=True, pose_interval=args.pose_interval,
        warmup=args.warmup,
    )
    d = run_one_mode(
        "WITHOUT MoveNet (run 2)",
        args.source, args.frames, args.conf,
        enable_pose=False, pose_interval=args.pose_interval,
        warmup=args.warmup,
    )

    # 对比报告(取 WITH 和 WITHOUT 的两次平均)
    with_avg_ms = (b["avg_ms"] + c["avg_ms"]) / 2
    with_p95_ms = (b["p95_ms"] + c["p95_ms"]) / 2
    with_fps = (b["fps"] + c["fps"]) / 2
    with_pose_calls = b["pose_calls"] + c["pose_calls"]

    without_avg_ms = (a["avg_ms"] + d["avg_ms"]) / 2
    without_p95_ms = (a["p95_ms"] + d["p95_ms"]) / 2
    without_fps = (a["fps"] + d["fps"]) / 2

    print("\n" + "=" * 60)
    print(" COMPARISON (averaged over 2 runs each, order-swapped)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'WITH MoveNet':>14} {'WITHOUT':>14} {'Diff':>14}")
    print(f"  {'-' * 64}")
    diff_avg = with_avg_ms - without_avg_ms
    diff_p95 = with_p95_ms - without_p95_ms
    diff_fps = with_fps - without_fps
    print(f"  {'avg latency':<20} {with_avg_ms:>12.1f} ms {without_avg_ms:>12.1f} ms {diff_avg:>+10.1f} ms")
    print(f"  {'p95 latency':<20} {with_p95_ms:>12.1f} ms {without_p95_ms:>12.1f} ms {diff_p95:>+10.1f} ms")
    print(f"  {'FPS':<20} {with_fps:>12.1f}     {without_fps:>12.1f}     {diff_fps:>+10.2f}")
    print(f"  {'pose calls':<20} {with_pose_calls:>13d}            0")
    print()
    speedup_pct = (diff_fps / without_fps * 100) if without_fps > 0 else 0
    print(f"  >>> 去掉 MoveNet 后 FPS 变化: {diff_fps:+.2f} ({speedup_pct:+.1f}%)")
    print(f"  >>> 去掉 MoveNet 节省延迟:    {diff_avg:+.1f} ms/帧")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
