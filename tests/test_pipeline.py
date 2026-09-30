"""End-to-end checks with synthetic video whose angles are known exactly."""
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from limbtrack.engine import Engine
from limbtrack.protocol import load_protocol
from limbtrack.synthetic import leg_points, render
from limbtrack.tracker import DotTracker, to_gray

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def click_all(engine, pts, offset=(3, -2)):
    for name in engine.test.dots:
        x, y = pts[name]
        assert engine.mark_dot(name, x + offset[0], y + offset[1]), f"could not lock onto {name}"


@pytest.mark.parametrize("kind", ["white", "black"])
def test_dot_centre_is_subpixel_accurate(kind):
    pts = leg_points(hip_flex=30, knee_flex=60)
    img = to_gray(render(pts, kind=kind, seed=1))
    for name, (x, y) in pts.items():
        t = DotTracker(name, kind)
        assert t.init_at(img, x + 4, y + 3)
        assert math.hypot(t.x - x, t.y - y) < 0.4, (name, t.x - x, t.y - y)


def test_auto_kind_detects_white_and_black():
    for kind in ("white", "black"):
        pts = leg_points(knee_flex=40)
        img = to_gray(render(pts, kind=kind))
        t = DotTracker("knee", "auto")
        assert t.init_at(img, *(pts["knee"] + 3))
        assert t.kind == kind


@pytest.mark.parametrize("kind", ["white", "black"])
@pytest.mark.parametrize("knee", [20, 45, 65, 90, 110, 140])
def test_knee_angle_accuracy_static(kind, knee):
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("knee_extension")
    eng.set_dot_kind(kind)
    pts = leg_points(hip_flex=60, knee_flex=knee, ankle_df=10, trunk_lean=5)
    eng.process(render(pts, kind=kind, seed=3), 0.0)
    click_all(eng, pts)
    res = eng.process(render(pts, kind=kind, seed=4), 0.033)
    got = {a.name: a.value for a in res.angles}
    assert got["knee"] == pytest.approx(knee, abs=0.5)
    assert got["hip"] == pytest.approx(60, abs=0.5)
    assert got["ankle"] == pytest.approx(10, abs=0.5)
    assert got["trunk"] == pytest.approx(5, abs=0.5)


def _sequence(n, knee_of):
    return [leg_points(hip_flex=70, knee_flex=knee_of(i), ankle_df=5, trunk_lean=2) for i in range(n)]


def test_tracking_a_moving_knee_and_recording(tmp_path):
    n = 90
    truth = [40 + 25 * math.sin(2 * math.pi * i / n) for i in range(n)]
    seq = _sequence(n, lambda i: truth[i])
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 40.0
    eng.process(render(seq[0], seed=0), 0.0)
    click_all(eng, seq[0])
    eng.process(render(seq[0], seed=0), 0.0)
    assert eng.lock_position()
    eng.start_trial("P01", notes="unit test")
    got = []
    for i in range(n):
        res = eng.process(render(seq[i], seed=i), i / 30.0)
        got.append(next(a.value for a in res.angles if a.name == "knee"))
    paths = eng.stop_trial()

    err = np.abs(np.array(got) - np.array(truth))
    assert err.max() < 1.0, err.max()

    with open(paths["data.csv"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n
    t = [float(r["t_s"]) for r in rows]
    assert t[0] == 0 and all(b > a for a, b in zip(t, t[1:]))
    assert all(r["knee_lost" if "knee_lost" in r else "knee_x"] != "" for r in rows)
    # deviation flag: knee swings +/-25 deg around 40, tolerance 3 -> must be flagged when far off
    flagged = [r for r in rows if r["knee_ok"] == "0"]
    assert len(flagged) > n * 0.5
    ok_rows = [r for r in rows if r["knee_ok"] == "1"]
    assert all(abs(float(r["knee_dev"])) <= 3.0 for r in ok_rows)
    # neighbours were locked and stayed put -> never flagged
    assert all(r["hip_ok"] == "1" for r in rows)

    for kind in ("raw.mp4", "overlay.mp4"):
        cap = cv2.VideoCapture(str(paths[kind]))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        cap.release()
    meta = json.loads(Path(paths["meta.json"]).read_text())
    assert meta["frames"] == n and meta["participant"] == "P01" and meta["target_deg"] == 40.0


def test_neighbour_drift_is_flagged_after_lock(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    still = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(still), 0.0)
    click_all(eng, still)
    eng.process(render(still), 0.01)
    assert eng.lock_position()
    # trunk leans 9 deg further: hip angle (trunk-thigh) drifts ~9 deg, neighbour tolerance is 5
    drifted = dict(still)
    drifted["shoulder"] = leg_points(hip_flex=70, knee_flex=60, ankle_df=5, trunk_lean=9)["shoulder"]
    res = None
    for i in range(3):
        res = eng.process(render(drifted, seed=i), (i + 1) / 30)
    st = {a.name: a.status for a in res.angles}
    assert st["hip"] == "out"
    assert st["knee"] == "ok" and st["ankle"] == "ok"


def test_lost_dot_is_flagged_and_reacquired(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 50.0
    pts = leg_points(hip_flex=70, knee_flex=50)
    eng.process(render(pts), 0.0)
    click_all(eng, pts)
    statuses = []
    for i in range(30):
        img = render(pts, seed=i)
        if 10 <= i < 20:  # hide the knee dot behind a hand
            kx, ky = map(int, pts["knee"])
            cv2.rectangle(img, (kx - 25, ky - 25), (kx + 25, ky + 25), (95, 110, 140), -1)
        res = eng.process(img, i / 30)
        statuses.append(next(a.status for a in res.angles if a.name == "knee"))
    assert all(s == "lost" for s in statuses[11:20])
    assert statuses[0] != "lost" and statuses[-1] != "lost"
    res_knee = next(a for a in res.angles if a.name == "knee")
    assert res_knee.value == pytest.approx(50, abs=0.7)


def test_bright_distractor_is_not_mistaken_for_dot():
    pts = leg_points(knee_flex=50)
    img = to_gray(render(pts, distractors=True))
    t = DotTracker("toe", "white")
    assert t.init_at(img, *(pts["toe"] + 2))
    for i in range(5):
        st = t.update(to_gray(render(pts, distractors=True, seed=i)))
        assert not st.lost
        assert math.hypot(st.x - pts["toe"][0], st.y - pts["toe"][1]) < 0.5
