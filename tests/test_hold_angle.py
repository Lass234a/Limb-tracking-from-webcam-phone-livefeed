"""VAL-01: Review and the export report the angle during the hold (plateau), the rest angle and the shift between them."""
import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def record(tmp_path, markers=True):
    """1.5 s of rest with the knee at 90, then a 5 s hold in which the knee has moved to 80 (a ramp in the first 0.5 s)."""
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 90.0
    p0 = leg_points(hip_flex=90, knee_flex=90, ankle_df=5)
    eng.process(render(p0, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p0[n] + 2))
    paths = eng.start_trial("P01")
    fps = 30
    for i in range(int(6.5 * fps)):
        t = i / fps
        knee = 90.0 if t < 1.5 else max(80.0, 90.0 - (t - 1.5) * 20.0)
        if markers and i == int(1.5 * fps):
            eng.add_event("MVIC start")
        if markers and i == int(6.5 * fps) - 1:
            eng.add_event("MVIC end")
        eng.process(render(leg_points(hip_flex=90, knee_flex=knee, ankle_df=5), seed=i), 0.1 + t)
    eng.stop_trial()
    return paths


def main_row(rows):
    return next(r for r in rows if r["role"] == "main")


def test_rest_plateau_and_shift_are_reported(tmp_path):
    tr = Trial(record(tmp_path)["meta.json"])
    t0, t1 = tr.hold_window()
    knee = main_row(tr.summary(t0, t1))
    assert knee["plateau_mean_deg"] == pytest.approx(80.0, abs=1.0)           # the angle during the hold
    assert knee["rest_mean_deg"] == pytest.approx(90.0, abs=1.0)              # the second before it
    assert knee["rest_to_hold_shift_deg"] == pytest.approx(-10.0, abs=1.5)
    assert knee["target_deg"] == 90.0 and knee["mean_minus_target_deg"] < -7.0   # the typed target is not what was reached
    assert knee["median_deg"] == pytest.approx(80.0, abs=1.5)


def test_the_plateau_leaves_out_the_ramps(tmp_path):
    tr = Trial(record(tmp_path)["meta.json"])
    t0, t1 = tr.hold_window()
    knee = main_row(tr.summary(t0, t1))
    assert knee["mean_deg"] > knee["plateau_mean_deg"]                        # the mean includes the ramp from 90 down to 80


def test_a_window_without_enough_rest_or_plateau_gives_nan_not_a_guess(tmp_path):
    tr = Trial(record(tmp_path)["meta.json"])
    start = float(tr.t[0])
    knee = main_row(tr.summary(start, start + 0.1))                           # 0.1 s window at the start: no rest before it, no plateau
    assert knee["rest_mean_deg"] != knee["rest_mean_deg"]                     # NaN
    assert knee["plateau_mean_deg"] != knee["plateau_mean_deg"]


def test_the_export_identifies_the_trial_and_says_where_the_window_came_from(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    paths = record(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    assert "Hold: the angle during the contraction" in tab.stats.text() and "plateau" in tab.stats.text()
    tab.export()
    rows = list(csv.DictReader(open(tmp_path / "P01" / f"{tab.trial.base}_summary.csv", newline="")))
    knee = next(r for r in rows if r["role"] == "main")
    assert knee["participant"] == "P01" and knee["test_key"] == "knee_extension" and knee["trial"] == tab.trial.base
    assert knee["window_source"] == "MVIC markers" and knee["software_version"]
    assert float(knee["plateau_mean_deg"]) == pytest.approx(80.0, abs=1.0) and float(knee["rest_to_hold_shift_deg"]) < -8.0
    tab.window.setRegion((float(tab.trial.t[0]), float(tab.trial.t[-1])))     # whole trial
    tab.export()
    rows = list(csv.DictReader(open(tmp_path / "P01" / f"{tab.trial.base}_summary.csv", newline="")))
    assert rows[0]["window_source"].startswith("window as set")


def test_without_markers_the_screen_says_so(tmp_path):
    app = QApplication.instance() or QApplication([])
    paths = record(tmp_path, markers=False)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    assert "no MVIC start/end markers" in tab.stats.text()
