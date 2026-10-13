"""What is read out of a trial beyond the angle table: force timing, torque, the check list, one row per trial, and the Review layout."""
import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.force import ForceCalibration  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402

from tests import test_force as T  # noqa: E402  (its helpers build a recording and a known force signal)


def nanlike(n):
    return np.full(n, np.nan)


def torque_trial(tmp_path, strap_seen=True):
    """A trial with a calibrated force of 200 N during the hold and simple geometry: vertical shank, pull horizontal at the ankle."""
    tr, clean = T.humming_trial(tmp_path, hum=0.0)
    cal = ForceCalibration()
    cal.set_zero(0.1)
    cal.n_per_v = 200.0 / 0.9                                          # 0.9 V above rest = 200 N
    tr.apply_calibration(cal, "test calibration")
    n = len(tr.t)
    const = lambda x, y: (np.full(n, float(x)), np.full(n, float(y)))  # noqa: E731
    tr.tracks["epicondyle"] = const(100, 100)
    tr.tracks["fibular_head"] = const(100, 150)
    tr.tracks["malleolus"] = const(100, 250)
    tr.tracks["anchor"] = const(300, 250)
    if strap_seen:
        tr.tracks["strap"] = const(100, 250)
    else:
        tr.tracks.pop("strap", None)
    tr.px_per_cm = 5.0
    tr.meta["pull"] = {"segment_from_to": ["fibular_head", "malleolus"], "moment_arm_from_joint_dot": "epicondyle"}
    return tr


# ------------------------------------------------------------------------------------------ force timing
def test_force_timing_finds_the_onset_the_rise_and_the_steadiness(tmp_path):
    tr, clean = T.humming_trial(tmp_path)
    ft = analysis.force_timing(tr, 2.0, 15.0)
    assert ft["unit"] == "V" and ft["peak"] == pytest.approx(1.0, abs=0.02)
    assert ft["onset_s"] == pytest.approx(3.0, abs=0.3)                      # force last below 10 % of the rise
    assert ft["time_to_90pct_s"] == pytest.approx(5.5, abs=0.4)             # 10 % -> 90 % (slow because of the light push first)
    assert ft["hold_above_90pct_s"] == pytest.approx(4.8, abs=0.5) and ft["hold_cv_pct"] < 3.0
    assert ft["rise_rate_per_s"] == pytest.approx(0.7 / 0.6, rel=0.15)       # the steep part: 0.7 V in 0.6 s
    quiet, _ = T.humming_trial(tmp_path / "q", hum=0.0)
    quiet.force_v = np.full_like(quiet.force_v, 0.1)
    assert analysis.force_timing(quiet, 2.0, 15.0) is None


# ------------------------------------------------------------------------------------------ torque
def test_torque_is_force_times_the_moment_arm(tmp_path):
    tr = torque_trial(tmp_path)
    tq = analysis.torque_summary(tr, 9.0, 12.5)
    assert tq["arm_at_peak_force_cm"] == pytest.approx(30.0, abs=0.1)         # 150 px / 5 px per cm
    assert tq["peak_nm"] == pytest.approx(200.0 * 0.30, rel=0.03)
    assert tq["plateau_nm"] == pytest.approx(60.0, rel=0.03) and tq["pictures_with_stand_in"] == 0


def test_torque_uses_the_ankle_dot_when_the_strap_dot_is_not_seen_and_says_so(tmp_path):
    tr = torque_trial(tmp_path, strap_seen=False)
    tq = analysis.torque_summary(tr, 9.0, 12.5)
    assert tq["peak_nm"] == pytest.approx(60.0, rel=0.03) and tq["pictures_with_stand_in"] == tq["pictures"] > 0


def test_torque_needs_a_calibrated_force_a_scale_and_the_anchor(tmp_path):
    tr = torque_trial(tmp_path)
    tr.px_per_cm = None
    assert analysis.torque_summary(tr, 9.0, 12.5) is None
    tr = torque_trial(tmp_path / "b")
    del tr.tracks["anchor"]
    assert analysis.torque_summary(tr, 9.0, 12.5) is None
    uncal, _ = T.humming_trial(tmp_path / "c")
    uncal.px_per_cm = 5.0
    assert analysis.torque_summary(uncal, 9.0, 12.5) is None


def test_the_picture_delay_shifts_the_force_that_goes_with_each_picture(tmp_path):
    tr = torque_trial(tmp_path)
    a = analysis.torque_series(tr, 0.0)["force_n"]
    b = analysis.torque_series(tr, 0.5)["force_n"]
    i = int(np.argmin(np.abs(tr.t - 8.3)))                                   # during the rise: an earlier scene means less force
    assert b[i] <= a[i] + 1e-9 and np.nanmax(np.abs(a - b)) > 5.0


# ------------------------------------------------------------------------------------------ the check list
def levels(checks):
    return {text.split(":")[0][:40]: level for level, text in checks}


def test_checks_flag_what_went_wrong_in_the_first_force_trials(tmp_path):
    tr = torque_trial(tmp_path / "a", strap_seen=True)
    tr.meta.update(measured_fps=11.1, pipeline_timing={"process_share_of_interval": 0.93, "process_ms": {"mean": 84.0}, "frame_interval_ms": {"median": 68.0}},
                   dot_size_check={"below_minimum": ["malleolus"]}, partial_dot_frames={"mt5_head": 8}, gaps={"count": 0})
    tr.tracks["strap"] = (np.where(np.arange(len(tr.t)) < 5, 100.0, np.nan), np.where(np.arange(len(tr.t)) < 5, 250.0, np.nan))
    ax, ay = tr.tracks["anchor"]
    tr.tracks["anchor"] = (ax + np.sin(np.arange(len(ax)) / 3.0) * 9.0, ay)                       # an anchor that moves
    checks = analysis.trial_checks(tr, 8.5, 13.0)
    text = "\n".join(f"{lvl}: {t}" for lvl, t in checks)
    assert "check: Pictures: 11.1 per second" in text and "The computer was the limit" in text
    assert "Dots under the 2 cm minimum: malleolus" in text and "info: Dots seen only partly" in text
    assert "check: Strap dot seen in" in text and "check: Anchor dot wandered" in text
    assert "no calibration" not in text.lower()                                                   # this trial is calibrated


def test_checks_are_green_for_a_good_trial(tmp_path):
    tr = torque_trial(tmp_path)
    tr.meta.update(measured_fps=29.0, pipeline_timing={}, dot_size_check=None, partial_dot_frames={}, gaps={"count": 0}, target_deg=None)
    levels_ = [lvl for lvl, _ in analysis.trial_checks(tr, 8.5, 13.0)]
    assert "check" not in levels_ and "ok" in levels_


def test_a_marker_far_from_the_force_onset_is_called_out(tmp_path):
    tr = torque_trial(tmp_path)
    tr.events = [(0, 3.0, "MVIC start"), (1, 14.0, "MVIC end")]
    t = "\n".join(text for _, text in analysis.trial_checks(tr))
    assert "MVIC start marker was pressed" in t and "before the force rose" in t


# ------------------------------------------------------------------------------------------ one row per trial
def test_session_rows_and_the_ten_percent_repeatability_rule(tmp_path):
    rows = []
    for k, level in enumerate((1.0, 1.25, 0.95)):
        tr, _ = T.humming_trial(tmp_path / str(k))
        tr.force_v = 0.1 + (tr.force_v - 0.1) * 1.0
        tr.force_v = np.where(tr.force_v > 0.5, 0.1 + (tr.force_v - 0.1) * level, tr.force_v)
        rows.append(analysis.session_row(tr))
        rows[-1]["trial"] = f"t{k}"
    pair = analysis.repeatability(rows[:2])
    assert pair[0]["repeatability_pct_same_position"] > 10 and pair[0]["third_trial_needed"] is True
    close = analysis.repeatability([rows[0], rows[2]])
    assert close[0]["repeatability_pct_same_position"] < 10 and close[0]["third_trial_needed"] is False
    assert rows[0]["window"] == "force above half of peak" and rows[0]["force_peak"] > 0.8
    out = analysis.write_session_csv(rows, tmp_path / "session.csv")
    back = list(csv.DictReader(open(out, newline="")))
    assert len(back) == 3 and list(back[0]) == analysis.SESSION_COLUMNS and back[0]["force_unit"] == "V"


# ------------------------------------------------------------------------------------------ the Review window
def test_review_shows_force_in_its_own_strip_and_numbers_in_tabs(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a[2]))
    eng, paths = T.recording(tmp_path, seconds=2.0)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    assert tab.tabs.count() == 5 and [tab.tabs.tabText(i) for i in range(5)] == ["Check", "Angles", "Force", "Calibration", "Dots and pull"]
    assert not tab.force_plot.isHidden() and not tab.force_box.isHidden()
    assert "Pictures" in tab.check_text.text() or "Dot" in tab.check_text.text() or "calibration" in tab.check_text.text()
    assert "Force in the window" in tab.force_text.text() and "Knee" in tab.stats.text() or "mean" in tab.stats.text()
    tab.export_session()                                                      # one trial in the folder: one row, no repeatability
    out = next(tmp_path.rglob("*_session_summary.csv"))
    rows = list(csv.DictReader(open(out, newline="")))
    assert len(rows) == 1 and rows[0]["test_key"] == "knee_extension" and "Saved" in infos[-1]


def test_review_without_force_hides_the_force_strip(tmp_path):
    from limbtrack.engine import Engine
    from limbtrack.synthetic import render
    app = QApplication.instance() or QApplication([])
    eng = Engine(T.TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(T.PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(T.PTS[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(6):
        eng.process(render(T.PTS, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    assert tab.force_plot.isHidden() and tab.force_box.isHidden()
    assert "No force was recorded" in tab.force_text.text()
