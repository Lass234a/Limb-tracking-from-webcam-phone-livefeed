"""The force zero in Review (0.30.0): typed, taken from a stretch of the graph or from the rest before the push, saved in the review layer (never in
the trial's files), and followed by every number, the overview and the exports. And the zero-drift check."""
import csv
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack import review_layer as rl  # noqa: E402
from limbtrack.force import ForceCalibration  # noqa: E402
from limbtrack.loadcell import active_regression  # noqa: E402
from limbtrack.review import ReviewTab, Trial, trial_row  # noqa: E402

from tests import test_force as T  # noqa: E402


@pytest.fixture()
def quiet(monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(("info", a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warning", a[2])))
    return shown


def _app():
    return QApplication.instance() or QApplication([])


def _known_force(paths, rest_v=0.0200):
    """Replace the recorded force file with a known volts signal (rest at rest_v, a light push, a hold at 1.0 V, release), newtons blank:
    what the trial was recorded with is then exactly known. Done before anything opens the trial."""
    t = np.arange(-4.0, 10.0, 0.001)
    clean = np.interp(t, [-4, 3, 3.05, 5, 5.6, 8, 8.5, 10], [rest_v, rest_v, 0.3, 0.3, 1.0, 1.0, rest_v, rest_v])
    with open(paths["force.csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "clock_s", "volts", "newtons"])
        for ti, v in zip(t, clean):
            w.writerow([round(float(ti), 5), round(float(ti) + 100.0, 5), round(float(v), 6), ""])
    return t, clean


def _uncalibrated(tmp_path):
    eng, paths = T.uncalibrated_recording(tmp_path)
    t, clean = _known_force(paths)
    return paths["meta.json"], t, clean


def test_a_typed_zero_is_saved_in_the_review_layer_and_every_number_follows(tmp_path, quiet):
    _app()
    meta, t, clean = _uncalibrated(tmp_path)
    plain_bytes = {p.name: p.read_bytes() for p in meta.parent.iterdir()}
    tab = ReviewTab(tmp_path)
    tab.load(meta)
    assert tab.trial.has_force() and tab.trial.zero_source() == "resting level" and "volts only" in tab.force_tree.text(everything=True)
    tab.zero_spin.setValue(0.0200)
    tab.apply_force_calibration()
    reg = active_regression()
    tr = tab.trial
    assert tr.source == "reviewed" and tr.zero_source() == "typed" and tr.zero_edit["from"] == "active" and tr.zero_edit["regression_id"] == reg.id
    assert tr.zero_edit["n_per_v"] == pytest.approx(reg.n_per_v) and tr.force_cal["offset_v"] == pytest.approx(0.0200)
    assert np.nanmax(tr.force_n) == pytest.approx((1.0 - 0.0200) * reg.n_per_v, rel=1e-3)
    assert "applied later" in tab.force_tree.text(everything=True) and "zero set in Review (typed)" in tr.force_cal_note
    saved = json.loads(rl.review_json_path(meta).read_text())
    assert saved["force"]["zero"]["zero_v"] == 0.02 and saved["force"]["zero"]["source"] == "typed" and saved["original"]["files"]["force.csv"]["sha256"]
    assert {p.name: p.read_bytes() for p in meta.parent.iterdir()} == plain_bytes                  # the recorded files are exactly as they were
    # the zero follows into a fresh trial, the overview row and the exports; the original reading stays volts
    assert Trial(meta, open_video=False, source="reviewed").force_cal["offset_v"] == pytest.approx(0.0200)
    assert Trial(meta, open_video=False).has_force() and np.all(np.isnan(Trial(meta, open_video=False).force_n))
    row_r, _ = trial_row(meta, source="reviewed")
    row_o, _ = trial_row(meta, source="original")
    assert row_r["force_unit"] == "N" and row_r["force_zero_source"] == "typed" and row_r["data_source"] == "reviewed" and "force zero typed" in row_r["review_edits"]
    assert row_o["force_unit"] == "V" and row_o["data_source"] == "original" and row_o["review_edits"] == ""
    assert float(row_r["force_zero_v"]) == pytest.approx(0.02)
    tab.trial.close()


def test_a_zero_without_any_slope_is_explained_and_nothing_is_saved(tmp_path, quiet, monkeypatch):
    from limbtrack import review as review_mod
    _app()
    meta, _, _ = _uncalibrated(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(meta)
    monkeypatch.setattr(review_mod, "active_regression", lambda *a: None)
    tab.apply_force_calibration()
    assert quiet and quiet[-1][0] == "info" and "regression" in quiet[-1][1] and not rl.review_json_path(meta).exists()
    tab.trial.close()


def test_the_slope_recorded_with_the_trial_is_kept_when_a_zero_is_set(tmp_path, quiet):
    _app()
    eng, paths = T.recording(tmp_path)                                         # calibrated when recorded
    meta = paths["meta.json"]
    t, clean = _known_force(paths)
    rec = json.loads(meta.read_text())["force_setup"]["calibration"] if "force_setup" in json.loads(meta.read_text()) else json.loads(meta.read_text())["force"]["calibration"]
    tab = ReviewTab(tmp_path)
    tab.load(meta)
    tab.zero_spin.setValue(0.0500)
    tab.apply_force_calibration()
    z = tab.trial.zero_edit
    assert z["from"] == "recorded" and z["n_per_v"] == pytest.approx(rec["n_per_v"])
    want = (np.nanmax(tab.trial.force_v) - 0.0500) * rec["n_per_v"]
    assert np.nanmax(tab.trial.force_n) == pytest.approx(want, rel=1e-6)
    tab.trial.close()


def test_the_zero_can_be_taken_from_a_stretch_of_the_graph(tmp_path, quiet):
    _app()
    meta, t, clean = _uncalibrated(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(meta)
    tab.zero_region.setRegion((-3.0, -1.0))                                    # 2 s of rest: a multiple of 20 ms
    tab.zero_from_stretch()
    z = tab.trial.zero_edit
    assert z["source"] == "stretch" and z["zero_v"] == pytest.approx(0.0200, abs=1e-5) and z["region_s"] == [-3.0, -1.0]
    assert "mains hum does not cancel" not in tab.zero_label.text()
    tab.zero_region.setRegion((-3.0, -1.0 + 0.013))
    tab.zero_from_stretch()
    assert "not a multiple of 20 ms" in tab.zero_label.text()
    tab.zero_region.setRegion((50.0, 60.0))                                    # nowhere on the force graph
    tab.zero_from_stretch()
    assert quiet and "no force samples" in quiet[-1][1]
    tab.trial.close()


def test_the_zero_can_be_taken_from_the_rest_before_the_push_and_taken_back(tmp_path, quiet):
    _app()
    meta, t, clean = _uncalibrated(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(meta)
    ri = tab.trial.force_rest()
    assert ri is not None and not np.isnan(ri["level"])
    tab.zero_from_rest()
    z = tab.trial.zero_edit
    assert z["source"] == "rest" and z["zero_v"] == pytest.approx(0.0200, abs=1e-5) and z["region_s"] == pytest.approx([ri["t_from"], ri["t_to"]])
    assert tab.zero_spin.value() == pytest.approx(0.0200, abs=1e-4) and "set in Review (rest)" in tab.zero_label.text()
    tab.zero_reset()
    assert tab.trial.zero_edit is None and tab.trial.source == "original" and not json.loads(rl.review_json_path(meta).read_text())["force"]["zero"]
    assert tab.trial.zero_source() == "resting level" and np.all(np.isnan(tab.trial.force_n))
    hist = [h["action"] for h in json.loads(rl.review_json_path(meta).read_text())["history"]]
    assert hist == ["zero", "zero reset"]                                       # what was done is kept
    tab.trial.close()


def test_an_edit_made_while_looking_at_the_original_switches_to_the_reviewed_data(tmp_path, quiet):
    _app()
    meta, _, _ = _uncalibrated(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(meta)
    tab.set_source("original")
    assert tab.btn_data_original.isChecked() and tab.trial.source == "original"
    tab.zero_spin.setValue(0.0200)
    tab.apply_force_calibration()
    assert tab._source == "reviewed" and tab.btn_data_reviewed.isChecked() and tab.trial.zero_edit is not None
    tab.set_source("original")
    assert tab.trial.source == "original" and tab.trial.zero_edit is None and np.all(np.isnan(tab.trial.force_n))          # Original shows the trial as recorded
    assert tab._entries[tab._index].edited and "✎ edited" in tab.trial_combo.itemText(tab._index)
    tab.trial.close()


# ------------------------------------------------------------------------------------------------ does the zero hold?
def _calibrated_humming(tmp_path, after_v):
    tr, clean = T.humming_trial(tmp_path, hum=0.0)
    cal = ForceCalibration()
    cal.set_zero(0.1)
    cal.n_per_v = 200.0 / 0.9                                                   # 0.9 V above rest = 200 N
    tr.apply_calibration(cal, "test calibration")
    last = tr.force_t >= tr.force_t[-1] - 1.2
    tr.force_v = np.where(last, after_v, tr.force_v)
    tr.force_n = cal.to_newton(tr.force_v)
    return tr


def test_a_zero_that_moved_during_the_trial_is_called_out_in_newtons(tmp_path):
    tr = _calibrated_humming(tmp_path, after_v=0.1 + 0.06)                      # 0.06 V = 13 N higher at the end than before the push
    zd = analysis.zero_drift(tr, 2.0, 15.0)
    assert zd["unit"] == "N" and zd["drift"] == pytest.approx(0.06 * 200.0 / 0.9, abs=1.5)
    checks = " ".join(text for lvl, text in analysis.trial_checks(tr, 2.0, 15.0) if lvl == "check")
    assert "the zero moved during the trial" in checks
    assert analysis.session_row(tr)["force_zero_drift_n"] == pytest.approx(zd["drift"], abs=1.5)


def test_a_zero_that_held_raises_nothing(tmp_path):
    tr = _calibrated_humming(tmp_path, after_v=0.1 + 0.01)                      # 2 N
    zd = analysis.zero_drift(tr, 2.0, 15.0)
    assert abs(zd["drift"]) < analysis.ZERO_DRIFT_WARN_N
    assert "the zero moved" not in " ".join(text for lvl, text in analysis.trial_checks(tr, 2.0, 15.0))


def test_there_is_no_drift_to_report_when_the_recording_ends_with_the_strap_still_loaded(tmp_path):
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    cal = ForceCalibration()
    cal.set_zero(0.1)
    cal.n_per_v = 200.0 / 0.9
    tr.apply_calibration(cal, "test calibration")
    tr.force_v = np.where(tr.force_t >= 13.0, 1.0, tr.force_v)                  # never let go
    tr.force_n = cal.to_newton(tr.force_v)
    assert analysis.zero_drift(tr, 2.0, 15.0) is None
    quiet_trial, _ = T.humming_trial(tmp_path / "q", hum=0.0)
    quiet_trial.force_v = np.full_like(quiet_trial.force_v, 0.1)
    assert analysis.zero_drift(quiet_trial, 2.0, 15.0) is None                  # no push at all
