"""Three-point angle: any three landmarks, unsigned angle at the middle one."""
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.geometry import interior_angle  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402
from limbtrack.tracker import TrackerSet, to_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
THREE = ("trochanter", "epicondyle", "malleolus")          # the dots on your video
OTHERS = ["asis", "psis", "fibular_head", "mt5_base", "mt5_head"]


def engine(tmp_path, pts, only_three=True, facing="right"):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.facing_right = facing == "right"
    eng.target = 100.0
    if only_three:
        for d in OTHERS:
            eng.set_dot_enabled(d, False)
    eng.process(render(pts, seed=0), 0.0)
    return eng


def mark_all(eng, pts):
    while (n := eng.next_dot_to_mark()) is not None:
        assert eng.mark_dot(n, *(pts[n] + 2)), n
    eng.process(render(pts, seed=1), 0.03)


def test_with_only_three_landmarks_no_protocol_angle_exists_and_lock_explains_why(tmp_path):
    pts = leg_points(hip_flex=70, knee_flex=60)
    eng = engine(tmp_path, pts)
    assert eng.active_angles == [] and eng.primary_angle is None
    mark_all(eng, pts)
    assert eng.lock_position() is False
    assert "No angle can be computed" in eng.lock_problem()          # the old message blamed the dots


@pytest.mark.parametrize("facing", ["right", "left"])
@pytest.mark.parametrize("hip,knee", [(0, 0), (70, 20), (70, 65), (45, 110), (90, 140)])
def test_angle_at_the_middle_dot_is_exact_and_unsigned(tmp_path, facing, hip, knee):
    pts = leg_points(hip_flex=hip, knee_flex=knee, facing=facing)
    eng = engine(tmp_path, pts, facing=facing)
    eng.set_three_point(*THREE)
    mark_all(eng, pts)
    res = eng.process(render(pts, seed=2), 0.06)
    tp = next(a for a in res.angles if a.name == "three_point")
    expected = interior_angle(pts["trochanter"], pts["epicondyle"], pts["malleolus"])
    assert tp.raw_value == pytest.approx(expected, abs=0.6)
    assert 0 <= tp.raw_value <= 180
    assert tp.primary and eng.primary_angle.name == "three_point"
    assert tp.reference == 100.0                                      # judged against the typed target


def test_it_works_with_the_middle_dot_anywhere_and_rejects_bad_choices(tmp_path):
    pts = leg_points(hip_flex=70, knee_flex=60)
    eng = engine(tmp_path, pts, only_three=False)
    mark_all(eng, pts)
    eng.set_three_point("psis", "trochanter", "epicondyle")           # hip-like angle from other dots
    res = eng.process(render(pts, seed=2), 0.06)
    tp = next(a for a in res.angles if a.name == "three_point")
    assert tp.raw_value == pytest.approx(interior_angle(pts["psis"], pts["trochanter"], pts["epicondyle"]), abs=0.6)
    with pytest.raises(ValueError):
        eng.set_three_point("asis", "asis", "psis")
    with pytest.raises(ValueError):
        eng.set_three_point("asis", "nonsense", "psis")
    eng.clear_three_point()
    assert "three_point" not in [a.name for a in eng.active_angles]


def test_choosing_it_switches_the_three_landmarks_on_and_enables_lock(tmp_path):
    pts = leg_points(hip_flex=70, knee_flex=60)
    eng = engine(tmp_path, pts)
    eng.set_dot_enabled("malleolus", False)
    eng.set_three_point(*THREE)
    assert "malleolus" not in eng.disabled_dots
    mark_all(eng, pts)
    assert eng.lock_position() is True and eng.locked


def test_as_a_neighbour_it_does_not_take_over_the_main_joint(tmp_path):
    pts = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng = engine(tmp_path, pts, only_three=False)
    eng.target = 60.0
    eng.set_three_point(*THREE, main=False)
    mark_all(eng, pts)
    res = eng.process(render(pts, seed=2), 0.06)
    primaries = [a.name for a in res.angles if a.primary]
    assert primaries == ["knee"]                                      # the test's own main joint is untouched
    assert eng.lock_position()
    drift = dict(pts)
    shin = pts["malleolus"] - pts["epicondyle"]
    side = np.array([-shin[1], shin[0]]) / np.hypot(*shin) * 25.0     # 25 px sideways = about 8 deg at the knee
    drift["malleolus"] = pts["malleolus"] + side
    for i in range(10):                                               # long enough for the 0.2 s display average to settle
        res = eng.process(render(drift, seed=10 + i), 0.1 + i / 30)
        if i == 0:
            first = next(a for a in res.angles if a.name == "three_point")
    assert first.raw_status == "out"                                  # the recorded verdict reacts at once
    tp = next(a for a in res.angles if a.name == "three_point")
    assert tp.reference is not None and tp.raw_status == "out" and tp.status == "out"   # judged against the locked position


def test_as_main_it_demotes_the_tests_own_main_joint(tmp_path):
    pts = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng = engine(tmp_path, pts, only_three=False)
    eng.set_three_point(*THREE, main=True)
    assert [a.name for a in eng.active_angles if a.primary] == ["three_point"]
    assert next(a for a in eng.test.angles if a.name == "knee").primary        # the protocol itself is not modified


def test_cannot_change_while_recording_and_is_saved_with_the_trial(tmp_path):
    pts = leg_points(hip_flex=70, knee_flex=60)
    eng = engine(tmp_path, pts)
    eng.set_three_point(*THREE)
    mark_all(eng, pts)
    eng.start_trial("P01")
    with pytest.raises(RuntimeError):
        eng.set_three_point(*THREE)
    with pytest.raises(RuntimeError):
        eng.clear_three_point()
    for i in range(10):
        eng.process(render(pts, seed=i), 0.1 + i / 30)
    paths = eng.stop_trial()
    head = next(csv.reader(open(paths["data.csv"], newline="")))
    assert "three_point_deg" in head and "knee_deg" not in head
    meta = json.loads(Path(paths["meta.json"]).read_text())
    assert meta["three_point"] == list(THREE) and meta["angles"][0]["name"] == "three_point"
    tr = Trial(paths["meta.json"])                                            # review can draw it
    assert tr.drawn(4) is not None and np.isfinite(tr.deg["three_point"]).all()


def test_in_the_window_the_exact_situation_from_the_screenshot(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text, *a, **k: shown.append(text))
    pts = leg_points(hip_flex=70, knee_flex=60)
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live, eng = win.live, win.engine
    for d in OTHERS:
        live.lm_checks[d].setChecked(False)
    eng.process(render(pts, seed=0), 0.0)
    mark_all(eng, pts)
    live.btn_lock.setChecked(True)
    live.toggle_lock()
    assert "No angle can be computed" in shown[-1] and not live.btn_lock.isChecked()

    live.chk_tp.setChecked(True)                                              # defaults: trochanter / epicondyle / malleolus
    assert eng.three_point == THREE and eng.primary_angle.name == "three_point"
    assert not live.suggest.isEnabled() and "180" in live.tp_info.text()
    assert "✓ Angle at lateral femoral epicondyle" in live.lm_info.text()
    eng.process(render(pts, seed=3), 0.1)
    live.btn_lock.setChecked(True)
    live.toggle_lock()
    assert eng.locked and len(shown) == 1

    live.chk_tp.setChecked(False)
    assert eng.three_point is None and live.suggest.isEnabled()
    win.close()


# ------------------------------------------------------------------ real video (not in git: skipped if absent)
VIDEO = ROOT / "Bundepus.mp4"


@pytest.mark.skipif(not VIDEO.exists(), reason="Bundepus.mp4 (real test video) not present")
def test_real_video_three_dots_give_a_three_point_angle(tmp_path):
    cap = cv2.VideoCapture(str(VIDEO))
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.set_dot_kind("black")
    for d in OTHERS:
        eng.set_dot_enabled(d, False)
    eng.set_three_point(*THREE)
    clicks = {"trochanter": (1008, 128), "epicondyle": (622, 182), "malleolus": (566, 563)}
    vals = []
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok, fr = cap.read()
        res = eng.process(fr, i / 30.0)
        if i == 0:
            for n, (x, y) in clicks.items():
                assert eng.mark_dot(n, x + 2, y + 2)
            res = eng.process(fr, 0.0001 + i / 30.0)
        tp = next(a for a in res.angles if a.name == "three_point")
        vals.append(tp.raw_value)
    vals = np.array(vals)
    assert np.isfinite(vals[2:238]).all()                          # all three dots visible
    assert np.isnan(vals[241:256]).all()                           # ankle dot behind the table leg: blank, not guessed
    good = vals[np.isfinite(vals)]
    assert 60 < good.min() and good.max() < 180
    assert 0 <= vals[2] <= 180
