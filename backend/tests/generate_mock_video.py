#!/usr/bin/env python
"""从现有真实幼儿园视频素材中生成 mock_classroom.mp4 测试视频。

生成一个 ~5 秒的合成视频，用于单元测试和 smoke test。
内容特点：
  - 分辨率 640x480（符合项目摄像头采集规格）
  - 30 FPS（满足实时性测试需求）
  - 包含真实儿童画面，可用于 YOLO26 目标检测验证
  - 包含多段不同场景（运动/静态），用于帧差分运动检测验证
"""

import cv2
import os
import sys

SRC_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "youeryuan120s.mp4")
DST_PATH = os.path.join(os.path.dirname(__file__), "mock_classroom.mp4")

TARGET_W, TARGET_H = 640, 480
TARGET_FPS = 30
SEGMENT_DURATION = 5.0  # 生成视频总时长（秒）

def generate_mock_video():
    if not os.path.exists(SRC_PATH):
        print(f"[ERROR] Source video not found: {SRC_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(SRC_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source video: {SRC_PATH}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Source: {src_w}x{src_h} @ {src_fps:.1f}fps, {total_frames} frames")

    # 输出编码器与格式
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(DST_PATH, fourcc, TARGET_FPS, (TARGET_W, TARGET_H))

    # 从源视频中段选取片段，确保画面内容清晰、儿童可见
    # youeryuan120s.mp4 中儿童拍球活动集中在前 30 秒
    start_second = 3.0  # 跳过片头，选择儿童清晰入画的时刻
    start_frame = int(start_second * src_fps)
    needed_frames = int(SEGMENT_DURATION * TARGET_FPS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    while frames_written < needed_frames:
        ret, frame = cap.read()
        if not ret:
            # 源视频不足时循环回退
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if not ret:
                break

        # Resize 到目标分辨率（保持宽高比，居中裁剪）
        frame_resized = _resize_keep_aspect_ratio(frame, TARGET_W, TARGET_H)
        out.write(frame_resized)
        frames_written += 1

    out.release()
    cap.release()

    # 验证生成结果
    verify_cap = cv2.VideoCapture(DST_PATH)
    if verify_cap.isOpened():
        v_fps = verify_cap.get(cv2.CAP_PROP_FPS)
        v_w = int(verify_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(verify_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_fc = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_dur = v_fc / v_fps if v_fps > 0 else 0
        v_size = round(os.path.getsize(DST_PATH) / (1024 * 1024), 2)
        print(f"Generated: {v_w}x{v_h} @ {v_fps:.1f}fps, {v_fc} frames, {v_dur:.1f}s, {v_size}MB")

        # 提取关键帧预览
        verify_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first = verify_cap.read()
        if ret:
            preview_path = DST_PATH.replace(".mp4", "_preview.jpg")
            cv2.imwrite(preview_path, first)
            print(f"Preview saved: {preview_path}")

    verify_cap.release()
    print(f"[OK] mock_classroom.mp4 generated successfully.")


def _resize_keep_aspect_ratio(frame, target_w, target_h):
    """保持宽高比的缩放 + 居中裁剪，避免画面变形。"""
    src_h, src_w = frame.shape[:2]
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h

    if src_ratio > tgt_ratio:
        # 源更宽，按高度缩放后水平裁剪
        new_h = target_h
        new_w = int(target_h * src_ratio)
    else:
        # 源更窄，按宽度缩放后垂直裁剪
        new_w = target_w
        new_h = int(target_w / src_ratio)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 居中裁剪
    x_start = (new_w - target_w) // 2
    y_start = (new_h - target_h) // 2
    cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]

    return cropped


if __name__ == "__main__":
    generate_mock_video()
