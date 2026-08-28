#!/usr/bin/env python
"""
Generate a synthetic mock video for LoongGuard testing.
Creates simplified stick-figure scenes simulating kindergarten scenarios:
  1. Standing child walking across the room
  2. Child lying prone (for MoveNet pose estimation testing)
  3. Two children interacting (for multi-object YOLO detection testing)

Usage:
    python scripts/generate_mock_video.py
Output:
    tests/mock_classroom.mp4 (640x480, 25 FPS, ~12 seconds)
"""

import math
import os

import cv2
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 640, 480
FPS = 25
DURATION_SEC = 12
TOTAL_FRAMES = FPS * DURATION_SEC
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "tests", "mock_classroom.mp4")

# Scene transitions (frame indices)
SCENE1_END = FPS * 4   # 4s: walking
SCENE2_END = FPS * 8   # 8s: prone
# remaining: interaction

COLORS = {
    "bg": (180, 220, 180),          # light green classroom wall/floor feel
    "floor": (200, 200, 190),
    "skin": (240, 200, 160),
    "shirt_blue": (80, 140, 220),
    "shirt_red": (220, 80, 80),
    "pants": (60, 80, 120),
    "hair": (50, 30, 20),
    "mat": (180, 160, 130),
    "furniture": (140, 120, 100),
}

# Scene labels
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.7


def draw_stick_person(canvas, cx, cy, scale, shirt_color, facing_right=True):
    """
    Draw a simplified human figure composed of ellipses and rectangles.
    cx, cy: center position; scale: overall size multiplier.
    Returns list of bounding boxes for debugging.
    """
    s = scale
    bboxes = []

    # Head
    head_r = int(14 * s)
    head_center = (int(cx), int(cy - 38 * s))
    cv2.circle(canvas, head_center, head_r, COLORS["skin"], -1)
    cv2.circle(canvas, head_center, head_r, COLORS["hair"], 1)
    bboxes.append((int(cx - head_r), int(cy - 38 * s - head_r), int(head_r * 2), int(head_r * 2)))

    # Torso
    torso_h = int(30 * s)
    torso_w = int(16 * s)
    torso_x = int(cx - torso_w // 2)
    torso_y = int(cy - 24 * s)
    cv2.rectangle(canvas, (torso_x, torso_y), (torso_x + torso_w, torso_y + torso_h), shirt_color, -1)
    bboxes.append((torso_x, torso_y, torso_w, torso_h))

    # Arms (angled based on pose)
    arm_len = int(24 * s)
    arm_w = int(5 * s)
    angle = math.radians(30 if facing_right else -30)
    # Left arm
    la_x = int(cx - 2 * s)
    la_y = int(cy - 20 * s)
    la_end = (int(la_x - arm_len * math.cos(angle)), int(la_y + arm_len * math.sin(angle)))
    cv2.line(canvas, (la_x, la_y), la_end, COLORS["skin"], arm_w)
    # Right arm
    ra_x = int(cx + 2 * s)
    ra_y = int(cy - 20 * s)
    ra_end = (int(ra_x + arm_len * math.cos(angle)), int(ra_y + arm_len * math.sin(angle)))
    cv2.line(canvas, (ra_x, ra_y), ra_end, COLORS["skin"], arm_w)

    # Legs
    leg_len = int(28 * s)
    leg_w = int(5 * s)
    leg_angle = math.radians(15 if facing_right else -15)
    # Left leg
    ll_x = int(cx - 4 * s)
    ll_y = int(cy + 6 * s)
    ll_end = (int(ll_x - leg_len * math.sin(leg_angle)), int(ll_y + leg_len * math.cos(leg_angle)))
    cv2.line(canvas, (ll_x, ll_y), ll_end, COLORS["pants"], leg_w)
    # Right leg
    rl_x = int(cx + 4 * s)
    rl_y = int(cy + 6 * s)
    rl_end = (int(rl_x + leg_len * math.sin(leg_angle)), int(rl_y + leg_len * math.cos(leg_angle)))
    cv2.line(canvas, (rl_x, rl_y), rl_end, COLORS["pants"], leg_w)

    return bboxes


def draw_prone_person(canvas, cx, cy, scale, shirt_color, head_dir=1):
    """Draw a person lying prone (for MoveNet testing)."""
    s = scale
    bboxes = []

    # Body (horizontal)
    body_w = int(60 * s)
    body_h = int(18 * s)
    body_x = int(cx - body_w // 2)
    body_y = int(cy - body_h // 2)
    cv2.rectangle(canvas, (body_x, body_y), (body_x + body_w, body_y + body_h), shirt_color, -1)
    bboxes.append((body_x, body_y, body_w, body_h))

    # Head at one end
    head_r = int(12 * s)
    head_x = int(cx + head_dir * body_w // 2)
    head_y = int(cy - 2 * s)
    cv2.circle(canvas, (head_x, head_y), head_r, COLORS["skin"], -1)
    cv2.circle(canvas, (head_x, head_y), head_r, COLORS["hair"], 1)
    bboxes.append((int(head_x - head_r), int(head_y - head_r), head_r * 2, head_r * 2))

    # Arms along body
    arm_y = int(cy + 2 * s)
    cv2.line(canvas, (int(cx - 20 * s), arm_y), (int(cx - 30 * s), int(cy + 10 * s)), COLORS["skin"], int(4 * s))
    cv2.line(canvas, (int(cx + 20 * s), arm_y), (int(cx + 30 * s), int(cy + 10 * s)), COLORS["skin"], int(4 * s))

    # Legs
    leg_x = int(cx - head_dir * body_w // 2)
    cv2.line(canvas, (leg_x, int(cy - 3 * s)), (leg_x, int(cy + 12 * s)), COLORS["pants"], int(5 * s))

    return bboxes


def draw_background(canvas, frame_idx):
    """Draw classroom background elements."""
    # Floor (bottom 30%)
    floor_y = int(HEIGHT * 0.7)
    cv2.rectangle(canvas, (0, floor_y), (WIDTH, HEIGHT), COLORS["floor"], -1)

    # Wall
    cv2.rectangle(canvas, (0, 0), (WIDTH, floor_y), COLORS["bg"], -1)

    # Window (top-left)
    cv2.rectangle(canvas, (30, 40), (120, 120), (200, 200, 200), -1)
    cv2.rectangle(canvas, (30, 40), (120, 120), (120, 100, 80), 2)
    cv2.line(canvas, (75, 40), (75, 120), (120, 100, 80), 2)
    cv2.line(canvas, (30, 80), (120, 80), (120, 100, 80), 2)

    # Small table (right side)
    table_x, table_y = 480, floor_y - 60
    cv2.rectangle(canvas, (table_x, table_y), (table_x + 80, table_y + 60), COLORS["furniture"], -1)
    cv2.rectangle(canvas, (table_x, table_y + 60), (table_x + 80, table_y + 70), (100, 80, 60), -1)

    # Shelf (left side)
    shelf_x, shelf_y = 20, floor_y - 100
    cv2.rectangle(canvas, (shelf_x, shelf_y), (shelf_x + 50, shelf_y + 100), COLORS["furniture"], -1)
    for i in range(3):
        sy = shelf_y + 20 + i * 30
        cv2.rectangle(canvas, (shelf_x + 5, sy), (shelf_x + 15, sy + 15), (200, 100, 100), -1)
        cv2.rectangle(canvas, (shelf_x + 20, sy), (shelf_x + 30, sy + 20), (100, 150, 100), -1)

    # Floor line
    cv2.line(canvas, (0, floor_y), (WIDTH, floor_y), (160, 160, 150), 2)


def draw_scene_label(canvas, text):
    """Draw scene description label."""
    cv2.putText(canvas, text, (20, 30), LABEL_FONT, LABEL_SCALE, (255, 255, 255), 2)


def frame_countdown(canvas, total, current):
    """Draw a small frame counter."""
    cv2.putText(canvas, f"frame {current}/{total}", (WIDTH - 200, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)


# ── Main generation loop ────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))

if not writer.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {OUTPUT}")

print(f"Generating {DURATION_SEC}s video at {FPS} FPS, {WIDTH}x{HEIGHT} ...")

for frame in range(TOTAL_FRAMES):
    canvas = np.full((HEIGHT, WIDTH, 3), COLORS["bg"], dtype=np.uint8)
    draw_background(canvas, frame)

    if frame < SCENE1_END:
        # Scene 1: Child walking left-to-right
        progress = frame / SCENE1_END
        cx = int(50 + progress * (WIDTH - 100))
        cy = int(HEIGHT * 0.75)
        bob = int(math.sin(frame * 0.3) * 4)
        draw_scene_label(canvas, "Scene 1: Child walking (YOLO26 detection)")
        draw_stick_person(canvas, cx, cy + bob, 1.2, COLORS["shirt_blue"], facing_right=True)

    elif frame < SCENE2_END:
        # Scene 2: Child lying prone (prone sleep position for MoveNet)
        progress = (frame - SCENE1_END) / (SCENE2_END - SCENE1_END)
        cx = int(100 + progress * (WIDTH - 200))
        cy = int(HEIGHT * 0.78)
        # Gentle sway
        sway = math.sin(frame * 0.1) * 2

        # Draw a sleep mat
        mat_x = int(cx - 50)
        mat_y = int(HEIGHT * 0.75)
        cv2.ellipse(canvas, (mat_x, mat_y), (55, 12), 0, 0, 360, COLORS["mat"], -1)

        draw_scene_label(canvas, "Scene 2: Child prone sleeping (MoveNet pose estimation)")
        draw_prone_person(canvas, cx + int(sway), cy, 1.1, COLORS["shirt_red"], head_dir=1)

    else:
        # Scene 3: Two children interacting
        progress = (frame - SCENE2_END) / (TOTAL_FRAMES - SCENE2_END)

        draw_scene_label(canvas, "Scene 3: Two children interacting (multi-object detection)")

        # Child A: blue shirt, walking toward center
        cx_a = int(100 + progress * 200)
        cy_a = int(HEIGHT * 0.72)
        bob_a = int(math.sin(frame * 0.4) * 3)
        draw_stick_person(canvas, cx_a, cy_a + bob_a, 1.0, COLORS["shirt_blue"], facing_right=True)

        # Child B: red shirt, stationary
        cx_b = int(WIDTH - 150)
        cy_b = int(HEIGHT * 0.74)
        draw_stick_person(canvas, cx_b, cy_b, 1.0, COLORS["shirt_red"], facing_right=False)

    frame_countdown(canvas, TOTAL_FRAMES, frame)
    writer.write(canvas)

writer.release()

# Verify output
if os.path.exists(OUTPUT):
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\nDone! Output: {OUTPUT}")
    print(f"  Size: {size_mb:.2f} MB")

    # Quick validation
    cap = cv2.VideoCapture(OUTPUT)
    if cap.isOpened():
        print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"  Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        cap.release()
else:
    print("ERROR: Output file was not created!")
