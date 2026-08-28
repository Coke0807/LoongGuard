"""
人脸关键点 / 头部姿态 / 睡姿分类 单元测试

覆盖：
    - SCRFD anchor 生成与 bbox/landmarks 解码（纯函数）
    - 头部姿态估计（head_pose）
    - 睡姿分类器持续性过滤（posture）
    - 真实模型集成（det_10g.onnx 存在时）：landmarks 落在 bbox 内
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config.settings import FaceConfig
from loongguard.face.base import FaceDetection
from loongguard.face.dummy import DummyFaceDetector
from loongguard.face.head_pose import HeadState, estimate_head_pose
from loongguard.face.onnx_detector import (
    _decode_bboxes,
    _decode_landmarks,
    _generate_anchor_centers,
    _generate_strides,
)
from loongguard.face.posture import SleepPosture, SleepPostureClassifier
from loongguard.utils.schema import AlertType, BoundingBox

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "buffalo_l" / "det_10g.onnx"
VIDEO_PATH = PROJECT_ROOT / "tests" / "fixtures" / "youeryuan120s.mp4"


def _make_kps(eye_dist=20.0, face_h=20.0, roll=0.0):
    """构造正脸 5 点关键点（可控眼距/脸高/面内旋转）"""
    cx, cy = 100.0, 100.0
    r = np.radians(roll)
    # 双眼连线方向
    ex = np.cos(r) * eye_dist / 2
    ey = np.sin(r) * eye_dist / 2
    le = np.array([cx - ex, cy - ey])
    re = np.array([cx + ex, cy + ey])
    # 嘴线中点在眼线中点下方 face_h
    mouth_cy = cy + face_h
    mw = eye_dist * 0.7 / 2
    lm = np.array([cx - mw, mouth_cy])
    rm = np.array([cx + mw, mouth_cy])
    nose = np.array([cx, cy + face_h * 0.5])
    return np.array([le, re, nose, lm, rm], dtype=np.float32)


class TestAnchorGeneration:
    def test_anchor_count_640(self):
        centers = _generate_anchor_centers(640, [8, 16, 32], 2)
        # 80*80*2 + 40*40*2 + 20*20*2 = 16800
        assert len(centers) == 16800

    def test_strides_match_centers(self):
        centers = _generate_anchor_centers(640, [8, 16, 32], 2)
        strides = _generate_strides(640, [8, 16, 32], 2)
        assert len(strides) == len(centers)
        # 前 12800 个为 stride8
        assert strides[0] == 8
        assert strides[12799] == 8
        assert strides[12800] == 16
        assert strides[-1] == 32

    def test_anchor_order_position_outer(self):
        """位置外循环、anchor 内循环（交替）"""
        centers = _generate_anchor_centers(16, [8], 2)
        expected = np.array([
            [0, 0], [0, 0], [8, 0], [8, 0],
            [0, 8], [0, 8], [8, 8], [8, 8],
        ], dtype=np.float32)
        np.testing.assert_array_equal(centers, expected)


class TestDecode:
    def test_decode_bboxes_distance_based(self):
        centers = np.array([[100.0, 200.0]])
        strides = np.array([8.0])
        raw = np.array([[1.0, 2.0, 3.0, 4.0]])
        bboxes = _decode_bboxes(raw, centers, strides)
        # x1=100-8, y1=200-16, x2=100+24, y2=200+32
        np.testing.assert_allclose(bboxes[0], [92, 184, 124, 232])

    def test_decode_landmarks_distance_based(self):
        centers = np.array([[100.0, 200.0]])
        strides = np.array([8.0])
        raw = np.array([[1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.5, 2.0, -0.5, 2.0]])
        kps = _decode_landmarks(raw, centers, strides)
        assert kps.shape == (1, 5, 2)
        np.testing.assert_allclose(kps[0, 0], [108, 200])
        np.testing.assert_allclose(kps[0, 2], [100, 208])


class TestHeadPose:
    def test_frontal_face_up(self):
        pose = estimate_head_pose(_make_kps(eye_dist=20, face_h=20))
        assert pose.state == HeadState.UP
        assert abs(pose.roll_deg) < 1e-6
        assert pose.yaw_ratio == pytest.approx(1.0, abs=0.01)

    def test_side_face_low_yaw_ratio(self):
        # 眼距被压缩到脸高的 0.3 倍 -> SIDE
        pose = estimate_head_pose(_make_kps(eye_dist=6, face_h=20))
        assert pose.state == HeadState.SIDE

    def test_tilted_face_high_roll(self):
        # roll=60 度 -> TILTED
        pose = estimate_head_pose(_make_kps(eye_dist=20, face_h=20, roll=60))
        assert pose.state == HeadState.TILTED

    def test_degenerate_kps_is_side(self):
        # 脸高为 0 -> SIDE（无法判定正脸）
        kps = _make_kps(eye_dist=20, face_h=0)
        pose = estimate_head_pose(kps)
        assert pose.state == HeadState.SIDE


class TestPostureClassifier:
    def _config(self, threshold=3):
        return FaceConfig(not_visible_frames_threshold=threshold)

    def _face(self, kps=None):
        bbox = BoundingBox(
            x1=80, y1=80, x2=120, y2=130,
            confidence=0.9, class_id=0, class_name="face",
        )
        return FaceDetection(bbox=bbox, landmarks=kps)

    def test_classify_frontal(self):
        clf = SleepPostureClassifier(self._config())
        results = clf.classify([self._face(_make_kps())])
        assert results[0].posture == SleepPosture.FACE_UP

    def test_classify_side(self):
        clf = SleepPostureClassifier(self._config())
        results = clf.classify([self._face(_make_kps(eye_dist=6, face_h=20))])
        assert results[0].posture == SleepPosture.SIDE

    def test_classify_no_landmarks_is_face_up(self):
        """桩实现无关键点 -> 视为脸可见（FACE_UP），不告警"""
        clf = SleepPostureClassifier(self._config())
        results = clf.classify([self._face(None)])
        assert results[0].posture == SleepPosture.FACE_UP
        assert results[0].head_pose is None

    def test_evaluate_face_present_no_alert(self):
        clf = SleepPostureClassifier(self._config())
        assert clf.evaluate([self._face(_make_kps())]) == []

    def test_evaluate_not_visible_triggers_after_threshold(self):
        clf = SleepPostureClassifier(self._config(threshold=3))
        assert clf.evaluate([]) == []
        assert clf.evaluate([]) == []
        alerts = clf.evaluate([])
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.PRONE_SLEEP

    def test_evaluate_face_resets_buffer(self):
        clf = SleepPostureClassifier(self._config(threshold=3))
        clf.evaluate([])
        clf.evaluate([])
        # 中间出现人脸 -> 缓冲重置
        clf.evaluate([self._face(_make_kps())])
        assert clf.evaluate([]) == []
        assert clf.evaluate([]) == []
        alerts = clf.evaluate([])
        assert len(alerts) == 1

    def test_snapshot(self):
        clf = SleepPostureClassifier(self._config())
        clf.evaluate([self._face(_make_kps())])
        snap = clf.get_snapshot()
        assert snap["faces"] == 1
        assert snap["postures"] == ["face_up"]
        assert snap["not_visible_frames"] == 0


class TestDummyDetector:
    def test_dummy_landmarks_none(self):
        det = DummyFaceDetector()
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        faces = det.detect_with_landmarks(img)
        assert len(faces) == 1
        assert faces[0].landmarks is None


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="det_10g.onnx 不存在")
class TestONNXIntegration:
    def test_real_frame_landmarks_within_bbox(self):
        """真实帧检测：landmarks 应落在对应 bbox 附近"""
        import cv2

        from loongguard.face.onnx_detector import ONNXFaceDetector

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ret, bgr = cap.read()
        cap.release()
        assert ret

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = ONNXFaceDetector(FaceConfig(conf_threshold=0.7))
        det.load_model()
        assert det.is_available()

        faces = det.detect_with_landmarks(rgb)
        assert len(faces) > 0

        top = max(faces, key=lambda f: f.bbox.confidence)
        assert top.landmarks is not None
        assert top.landmarks.shape == (5, 2)

        # 关键点应落在 bbox 附近（允许少量越界）
        b = top.bbox
        margin = max(b.x2 - b.x1, b.y2 - b.y1) * 0.3
        for x, y in top.landmarks:
            assert b.x1 - margin <= x <= b.x2 + margin
            assert b.y1 - margin <= y <= b.y2 + margin

        # 正脸帧的头部姿态应可计算
        pose = estimate_head_pose(top.landmarks)
        assert pose.state in (HeadState.UP, HeadState.SIDE, HeadState.TILTED)
