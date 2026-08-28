"""创建 16 字节 SM4 测试密钥和 Mock 教室视频"""
from pathlib import Path
import os

def create_sm4_key():
    key_path = Path("config/.sm4_key")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(16)
    key_path.write_bytes(key)
    print(f"SM4 key written: {key_path} ({len(key)} bytes)")

def create_mock_video():
    import cv2
    import numpy as np
    video_path = Path("tests/mock_classroom.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h, fps, duration = 640, 480, 25, 3
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    total_frames = fps * duration
    for i in range(total_frames):
        frame = np.random.randint(80, 180, (h, w, 3), dtype=np.uint8)
        # 移动色块模拟运动物体
        cx = int(100 + 400 * (i / total_frames))
        cy = 240
        cv2.rectangle(frame, (cx-30, cy-30), (cx+30, cy+30), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    print(f"Mock video created: {video_path} ({video_path.stat().st_size // 1024} KB, {total_frames} frames)")

if __name__ == "__main__":
    create_sm4_key()
    create_mock_video()
    print("Done.")
