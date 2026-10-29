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
OUT = ROOT / "tests" / "_out"


def make_engine(tmp_path, test="knee_extension", kind="auto", facing="right"):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.set_dot_kind(kind)
    eng.facing_right = facing == "right"
    return eng


def click_all(engine, pts, offset=(3, -2)):
    for name in engine.test.dots:
        x, y = pts[name]
        assert engine.mark_dot(name, x + offset[0], y + offset[1]), f"could not lock onto {name}"


def values(res):
    return {a.name: a.value for a in res.angles}


def test_protocol_loads_with_goniometer_landmarks():
    t = TESTS["knee_extension"]
    assert t.dots == ["iliac_crest", "trochanter", "epicondyle", "fibular_head", "malleolus"]       # 0.24.0: no foot in the knee and hip tests
    assert TESTS["belt_squat"].dots == [*t.dots, "mt5_base", "mt5_head"]                          # ... the ankle tests and the belt squat keep it
    assert t.primary.name == "knee"
    assert TESTS["hip_extension_k90"].primary.name == "hip"
    assert TESTS["ankle_dorsiflexion"].primary.name == "ankle"
    assert set(TESTS) == {"knee_extension", "knee_flexion", "hip_extension_k0", "hip_extension_k90", "hip_flexion",
                          "ankle_plantarflexion", "ankle_dorsiflexion", "hip_abduction", "hip_adduction", "belt_squat"}


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
        assert t.init_at(img, *(pts["epicondyle"] + 3))
        assert t.kind == kind


@pytest.mark.parametrize("facing", ["right", "left"])
@pytest.mark.parametrize("kind", ["white", "black"])
@pytest.mark.parametrize("hip,knee,ankle,tilt", [
    (0, 0, 0, 0), (45, 20, 5, 4), (80, 65, -10, -6), (-20, 110, 10, 12), (90, 140, 0, 0), (10, 90, -20, 8)])
def test_all_angles_accurate_static(kind, facing, hip, knee, ankle, tilt):
    eng = make_engine(OUT, test="belt_squat", kind=kind, facing=facing)          # (knee, hip and ankle: the knee test has no foot since 0.24.0)
    pts = leg_points(hip, knee, ankle, tilt, facing)
    eng.process(render(pts, kind=kind, seed=3), 0.0)
    click_all(eng, pts)
    got = values(eng.process(render(pts, kind=kind, seed=4), 0.033))
    assert set(got) == {"knee", "hip", "ankle"}
    assert got["knee"] == pytest.approx(knee, abs=0.6)
    assert got["hip"] == pytest.approx(hip, abs=0.6)       # also with the pelvis tipped by `tilt`: the hip is measured against the pelvis
    assert got["ankle"] == pytest.approx(ankle, abs=0.6)


def test_facing_flag_mirrors_signs():
    """Hip extension reads negative only when the 'faces' setting matches the picture."""
    pts = leg_points(hip_flex=-20, knee_flex=30, facing="right")
    right = make_engine(OUT, facing="right")
    right.process(render(pts), 0)
    click_all(right, pts)
    assert values(right.process(render(pts), 0.03))["hip"] == pytest.approx(-20, abs=0.6)
    wrong = make_engine(OUT, facing="left")   # wrong setting for this picture
    wrong.process(render(pts), 0)
    click_all(wrong, pts)
    assert values(wrong.process(render(pts), 0.03))["hip"] != pytest.approx(-20, abs=5)


def _seq(n, knee_of, tilt=2.0):
    return [leg_points(hip_flex=90, knee_flex=knee_of(i), ankle_df=5, pelvic_tilt=tilt) for i in range(n)]       # hip 90: the required neighbour angle in the knee tests


def test_tracking_a_moving_knee_and_recording(tmp_path):
    n = 90
    truth = [40 + 25 * math.sin(2 * math.pi * i / n) for i in range(n)]
    seq = _seq(n, lambda i: truth[i])
    eng = make_engine(tmp_path)
    eng.target, eng.tolerance = 40.0, 3.0                  # (the protocol's default is 5 since 0.20.0)
    eng.process(render(seq[0], seed=0), 0.0)
    click_all(eng, seq[0])
    eng.process(render(seq[0], seed=0), 0.0)
    assert eng.lock_position()
    eng.start_trial("P01", notes="unit test")
    got = []
    for i in range(n):
        res = eng.process(render(seq[i], seed=i), i / 30.0)
        got.append(next(a.raw_value for a in res.angles if a.name == "knee"))   # recorded value, unsmoothed
    paths = eng.stop_trial()

    err = np.abs(np.array(got) - np.array(truth))
    assert err.max() < 1.0, err.max()

    with open(paths["data.csv"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n
    t = [float(r["t_s"]) for r in rows]
    assert t[0] == 0 and all(b > a for a, b in zip(t, t[1:]))
    flagged = [r for r in rows if r["knee_ok"] == "0"]
    assert len(flagged) > n * 0.5
    assert all(abs(float(r["knee_dev"])) <= 3.0 for r in rows if r["knee_ok"] == "1")
    assert all(r["hip_ok"] == "1" for r in rows)           # neighbours locked and still

    for kind in ("raw.mp4", "overlay.mp4"):
        cap = cv2.VideoCapture(str(paths[kind]))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        cap.release()
    meta = json.loads(Path(paths["meta.json"]).read_text())
    assert meta["frames"] == n and meta["participant"] == "P01" and meta["target_deg"] == 40.0
    assert meta["facing"] == "right" and meta["angles"][0]["name"] == "knee"


def _locked_engine(tmp_path, still):
    eng = make_engine(tmp_path, test="belt_squat")
    eng.target = 60.0
    eng.process(render(still), 0.0)
    click_all(eng, still)
    eng.process(render(still), 0.01)
    assert eng.lock_position()
    return eng


def _statuses(eng, pts):
    res = None
    for i in range(3):
        res = eng.process(render(pts, seed=i), (i + 1) / 30)
    return {a.name: a.status for a in res.angles}


def test_neighbour_drift_is_flagged_after_lock(tmp_path):
    still = leg_points(hip_flex=90, knee_flex=60, ankle_df=5, pelvic_tilt=0)       # hip 90 = the required neighbour angle in this test
    # the ankle moves 10 deg while everything else stays put: judged against its value at Lock
    eng = _locked_engine(tmp_path, still)
    foot_moved = leg_points(hip_flex=90, knee_flex=60, ankle_df=15, pelvic_tilt=0)
    st = _statuses(eng, foot_moved)
    assert st["ankle"] == "out" and st["knee"] == "ok" and st["hip"] == "ok"
    # the pelvis tips 9 deg forward while the leg stays put: the hip (measured against the pelvis midline) changes by 9 deg
    eng = _locked_engine(tmp_path, still)
    tipped = leg_points(hip_flex=90, knee_flex=60, ankle_df=5, pelvic_tilt=9)
    drifted = dict(still)
    drifted["iliac_crest"] = tipped["iliac_crest"]
    st = _statuses(eng, drifted)
    assert st["hip"] == "out" and st["knee"] == "ok" and st["ankle"] == "ok"


def test_lost_dot_is_flagged_and_reacquired(tmp_path):
    eng = make_engine(tmp_path, test="belt_squat")
    eng.target = 50.0
    pts = leg_points(hip_flex=70, knee_flex=50)
    eng.process(render(pts), 0.0)
    click_all(eng, pts)
    statuses = []
    for i in range(30):
        img = render(pts, seed=i)
        if 10 <= i < 20:  # hide the epicondyle dot behind a hand
            kx, ky = map(int, pts["epicondyle"])
            cv2.rectangle(img, (kx - 22, ky - 22), (kx + 22, ky + 22), (95, 110, 140), -1)
        res = eng.process(img, i / 30)
        statuses.append({a.name: a.status for a in res.angles})
    for s in statuses[11:20]:
        assert s["knee"] == "lost" and s["hip"] == "lost"       # both use the epicondyle
        assert s["ankle"] != "lost"
    assert statuses[0]["knee"] != "lost" and statuses[-1]["knee"] != "lost"
    assert values(res)["knee"] == pytest.approx(50, abs=0.7)


def test_bright_distractor_is_not_mistaken_for_dot():
    pts = leg_points(knee_flex=50)
    img = to_gray(render(pts, distractors=True))
    t = DotTracker("mt5_head", "white")
    assert t.init_at(img, *(pts["mt5_head"] + 2))
    for i in range(5):
        st = t.update(to_gray(render(pts, distractors=True, seed=i)))
        assert not st.lost
        assert math.hypot(st.x - pts["mt5_head"][0], st.y - pts["mt5_head"][1]) < 0.5


@pytest.mark.parametrize("kind", ["white", "black"])
def test_fading_dots_are_flagged_weak_before_they_are_lost(tmp_path, kind):
    eng = make_engine(tmp_path, kind=kind)
    eng.target = 50.0
    pts = leg_points(hip_flex=70, knee_flex=50)
    eng.process(render(pts, kind=kind), 0.0)
    click_all(eng, pts)
    good = eng.process(render(pts, kind=kind, seed=1), 0.03)
    assert all(st.quality > 0.9 and not st.weak for st in good.dots.values())

    faded = eng.process(render(pts, kind=kind, seed=2, dot_gain=0.45), 0.06)
    assert all(st.weak and not st.lost for st in faded.dots.values())
    assert values(faded)["knee"] == pytest.approx(50, abs=1.0)        # still tracked accurately while weak

    gone = eng.process(render(pts, kind=kind, seed=3, dot_gain=0.1), 0.09)
    assert all(st.lost for st in gone.dots.values())
