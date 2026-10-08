"""Dot displacement over a window (Review tab and export). Computed after the fact from the recorded positions."""
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.displacement import window_displacements  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


# ---- the maths
def test_a_dot_that_moves_3_right_and_4_down_has_a_shift_of_5():
    t = np.arange(40) / 10.0
    x = np.full(40, 100.0)
    y = np.full(40, 200.0)
    x[20:] += 3.0
    y[20:] += 4.0
    rows = window_displacements(t, {"a": (x, y)}, 0.0, 3.9, px_per_cm=5.0)
    r = rows[0]
    assert r["end_dx_px"] == pytest.approx(3.0) and r["end_dy_px"] == pytest.approx(4.0)
    assert r["end_shift_px"] == pytest.approx(5.0) and r["max_shift_px"] == pytest.approx(5.0)
    assert r["end_shift_cm"] == pytest.approx(1.0)
    assert r["frames"] == 40 and r["frames_missing"] == 0


def test_only_the_window_counts_and_the_start_is_the_window_start():
    t = np.arange(100) / 10.0
    x = np.linspace(0.0, 99.0, 100)                       # moves 1 px per frame
    rows = window_displacements(t, {"a": (x, np.zeros(100))}, 3.0, 5.0)
    r = rows[0]
    # frames 30..50 (21 frames): start = mean of frames 30-34 = 32, end = mean of 46-50 = 48
    assert r["end_shift_px"] == pytest.approx(16.0)
    assert r["max_shift_px"] == pytest.approx(18.0)                       # frame 50 (x = 50) is 18 from the start mean 32


def test_lost_frames_are_skipped_not_filled_and_a_fully_lost_dot_is_blank():
    t = np.arange(20) / 10.0
    x = np.full(20, 50.0)
    y = np.full(20, 60.0)
    x[5:12] = np.nan
    y[5:12] = np.nan
    r = window_displacements(t, {"a": (x, y), "b": (np.full(20, np.nan), np.full(20, np.nan))}, 0.0, 2.0)
    assert r[0]["frames_missing"] == 7 and r[0]["end_shift_px"] == pytest.approx(0.0)
    assert r[1]["frames_missing"] == 20 and math.isnan(r[1]["end_shift_px"]) and math.isnan(r[1]["max_shift_cm"])


def test_numbers_relative_to_the_anchor_remove_a_common_movement():
    t = np.arange(40) / 10.0
    shift = np.where(np.arange(40) >= 20, 8.0, 0.0)       # the whole picture moves 8 px (camera bumped)
    strap = (200.0 + shift, 300.0 + 0 * shift)
    anchor = (50.0 + shift, 400.0 + 0 * shift)
    rows = {r["dot"]: r for r in window_displacements(t, {"strap": strap, "anchor": anchor}, 0.0, 3.9, reference="anchor")}
    assert rows["strap"]["end_shift_px"] == pytest.approx(8.0)
    assert rows["strap"]["rel_end_shift_px"] == pytest.approx(0.0, abs=1e-9)
    assert "rel_end_shift_px" not in rows["anchor"]        # the reference has nothing relative to itself


# ---- a recorded trial opened in Review
def record(tmp_path, markers=True):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.set_dot_enabled("strap", True)
    eng.set_dot_enabled("anchor", True)
    p = dict(leg_points(hip_flex=70, knee_flex=60, ankle_df=5))
    mal, fib = np.array(p["malleolus"]), np.array(p["fibular_head"])
    p["strap"] = mal + 0.35 * (fib - mal)
    p["anchor"] = p["strap"] + np.array([260.0, -90.0])
    p["ruler_a"], p["ruler_b"] = np.array([120.0, 670.0]), np.array([240.0, 670.0])
    eng.scale.length_cm = 20.0
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    paths = eng.start_trial("P01")
    for i in range(90):
        q = dict(p)
        if i >= 45:                                         # in the second half the strap slides 12 px along the shank
            q["strap"] = p["strap"] + (fib - mal) / np.linalg.norm(fib - mal) * 12.0
        if markers and i == 20:
            eng.add_event("MVIC start")
        if markers and i == 80:
            eng.add_event("MVIC end")
        eng.process(render(q, seed=i), 0.3 + i / 30)
    eng.stop_trial()
    return paths, p


def test_review_reports_the_strap_slide_and_nothing_for_still_dots(tmp_path):
    paths, p = record(tmp_path)
    tr = Trial(paths["meta.json"])
    assert tr.px_per_cm == pytest.approx(6.0, rel=0.01)
    rows = {r["dot"]: r for r in tr.displacement(float(tr.t[0]), float(tr.t[-1]))}
    assert rows["strap"]["end_shift_px"] == pytest.approx(12.0, abs=1.0)
    assert rows["strap"]["end_shift_cm"] == pytest.approx(2.0, abs=0.2)             # 12 px at 6 px/cm
    assert rows["trochanter"]["end_shift_px"] < 1.0 and rows["anchor"]["end_shift_px"] < 1.0
    assert rows["strap"]["rel_end_shift_px"] == pytest.approx(12.0, abs=1.0)
    pull = {q["quantity"]: q for q in tr.pull_summary(float(tr.t[0]), float(tr.t[-1]))}
    assert set(pull) == {"pull_angle_deg", "moment_arm_cm", "strap_joint_cm"}
    assert abs(pull["strap_joint_cm"]["change_start_to_end"]) == pytest.approx(2.0, abs=0.3)   # the strap moved 2 cm from the joint


def test_the_window_defaults_to_the_hold_and_the_text_and_export_follow_it(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    paths, _ = record(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    t0, t1 = tab.window.getRegion()
    hold = tab.trial.hold_window()
    assert hold is not None and (t0, t1) == pytest.approx(hold)                      # MVIC start to end, not the whole trial
    text = tab.disp.text()
    assert "Dot movement" in text and "strap" in text and "end vs anchor" in text and "Pull (strap and anchor dots)" in text
    tab.export()
    out = tmp_path / "P01" / f"{tab.trial.base}_displacement.csv"
    rows = list(csv.DictReader(open(out, newline="")))
    by = {r["dot"]: r for r in rows}
    assert float(by["strap"]["end_shift_px"]) == pytest.approx(12.0, abs=1.5)
    assert float(by["strap"]["window_start_s"]) == pytest.approx(t0, abs=0.01)


def test_without_markers_the_window_is_the_whole_trial(tmp_path):
    app = QApplication.instance() or QApplication([])
    paths, _ = record(tmp_path, markers=False)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    t0, t1 = tab.window.getRegion()
    assert (t0, t1) == pytest.approx((tab.trial.t[0], tab.trial.t[-1]))


def test_a_trial_without_strap_dots_has_no_pull_section_and_no_anchor_columns(tmp_path):
    app = QApplication.instance() or QApplication([])
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    p = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(20):
        eng.process(render(p, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    assert "Pull (strap" not in tab.disp.text() and "vs anchor" not in tab.disp.text()
    assert "no ruler scale" in tab.disp.text()
