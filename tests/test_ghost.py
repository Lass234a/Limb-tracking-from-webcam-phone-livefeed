from pathlib import Path

import numpy as np
import pytest

from limbtrack.angles import segment_pairs
from limbtrack.engine import Engine
from limbtrack.geometry import anchored_ghost
from limbtrack.protocol import load_protocol
from limbtrack.review import Trial
from limbtrack.synthetic import leg_points, render

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def test_anchored_ghost_maths():
    snap = {"a": (100.0, 100.0), "b": (200.0, 100.0)}
    live = {"a": (130.0, 90.0), "b": (999.0, 999.0)}
    g = anchored_ghost(snap, live, "a")
    assert g["a"] == (130.0, 90.0) and g["b"] == (230.0, 90.0)        # whole ghost shifted by the anchor's offset
    assert anchored_ghost(snap, live, None) == snap                    # not anchored: stays where it was locked
    assert anchored_ghost(snap, {}, "a") == snap                       # anchor dot not visible: no shift


def test_segment_pairs_are_unique_and_dot_to_dot_only():
    pairs = segment_pairs(TESTS["knee_extension"].angles)
    assert ("trochanter", "epicondyle") in pairs and ("fibular_head", "malleolus") in pairs
    assert len(pairs) == len(set(pairs))
    assert all(isinstance(p[0], str) and isinstance(p[1], str) for p in pairs)


def _engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    pts = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(pts, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(pts[n] + 2))
    eng.process(render(pts, seed=0), 0.01)
    return eng, pts


def test_ghost_only_after_lock_and_is_anchored_to_main_fulcrum(tmp_path):
    eng, pts = _engine(tmp_path)
    assert eng.process(render(pts, seed=1), 0.02).ghost is None       # not locked yet
    assert eng.lock_position()
    assert eng.process(render(pts, seed=2), 0.03).ghost is not None

    res = None
    for step in range(1, 9):                                            # whole body slides (40, 25) px over 8 frames
        shifted = {k: v + np.array([5.0, 3.1]) * step for k, v in pts.items()}
        res = eng.process(render(shifted, seed=3 + step), 0.04 + step / 30)
    anchor = eng.test.primary.fulcrum                                   # 'epicondyle' for the knee test
    live = res.dots[anchor]
    assert res.ghost[anchor] == pytest.approx((live.x, live.y), abs=1e-6)
    # a pure shift leaves no angular drift, so every ghost dot sits on its live dot
    for n, (gx, gy) in res.ghost.items():
        assert np.hypot(res.dots[n].x - gx, res.dots[n].y - gy) < 1.0

    eng.ghost_anchored = False
    res = eng.process(render(shifted, seed=4), 0.05)
    assert np.hypot(res.dots["mt5_head"].x - res.ghost["mt5_head"][0], res.dots["mt5_head"].y - res.ghost["mt5_head"][1]) > 30

    eng.ghost_enabled = False
    assert eng.process(render(shifted, seed=5), 0.06).ghost is None
    eng.ghost_enabled = True
    eng.unlock_position()
    assert eng.process(render(shifted, seed=6), 0.07).ghost is None


def test_ghost_shows_angular_drift_and_is_saved_for_review(tmp_path):
    eng, pts = _engine(tmp_path)
    assert eng.lock_position()
    eng.start_trial("P01")
    drifted = leg_points(hip_flex=70, knee_flex=66, ankle_df=5)          # knee drifts 6 deg
    res = None
    for i in range(10):
        res = eng.process(render(drifted, seed=i), 0.1 + i / 30)
    # fibular-head/malleolus/foot dots are away from the ghost, femur dots are not
    far = {n: np.hypot(res.dots[n].x - res.ghost[n][0], res.dots[n].y - res.ghost[n][1]) for n in res.dots}
    assert far["malleolus"] > 10 and far["trochanter"] < 1.0
    paths = eng.stop_trial()
    tr = Trial(paths["meta.json"])
    assert tr.ghost is not None and tr.ghost_anchor == "epicondyle"
    img = tr.drawn(5)
    assert img is not None and img.shape[:2] == (720, 1280)
