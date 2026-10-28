"""0.22.0: the second zero level. Zero (Z) is the strap with nothing on it; Zero 2 (X) is the participant relaxed in the strap, before
the push. It is kept beside the zero (the newtons stay the total force), shown live as "above relaxed", saved with every trial, and used
by Review and the session sheet only for the position it was taken for. Samples are made up: no DAQ is needed."""
import csv
import json
import os
import re
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import ForceCalibration, ForceSource, SimulatedForce  # noqa: E402
from limbtrack.review import ReviewTab  # noqa: E402
from limbtrack.synthetic import render  # noqa: E402

from tests import test_force as T  # noqa: E402
from tests.test_force_zero import RATE, TRUE_ZERO, signal, source_with  # noqa: E402

N_PER_V = 526.95
TENSION_V = 0.15                                           # the strap's tension with the leg relaxed: 79 N above the zero


def zeroed_source(extra_v=TENSION_V, **kw):
    """A source that has received 2.4 s of the relaxed leg: the strap's tension on top of the free-hanging level TRUE_ZERO. (The tests set
    the zero by hand: samples pushed in one go all carry 'now' as their time, so two phases cannot be told apart by time.)"""
    return source_with(signal(2.4, seed=2, **kw) + extra_v)


def test_the_relaxed_level_is_taken_on_top_of_the_zero_and_is_good_to_a_millivolt():
    src = zeroed_source(noise=0.01)
    src.calibration.set_zero(TRUE_ZERO)                    # (the zero was taken while the strap hung free)
    assert src.relax("P01 | knee | 70") is None
    cal = src.calibration
    assert cal.relaxed_v == pytest.approx(TRUE_ZERO + TENSION_V, abs=0.003)
    assert cal.relaxed_n == pytest.approx(TENSION_V * N_PER_V, abs=2.0)                    # newtons ABOVE the zero
    assert cal.relaxed_for == "P01 | knee | 70" and cal.relaxed_when
    assert cal.relaxed_se_v == pytest.approx(cal.relaxed_noise_v / np.sqrt(src.last_reading["samples"]))    # noise / sqrt(samples)
    assert cal.relaxed_se_v < 0.001                                                       # under a millivolt, under half a newton
    assert cal.offset_v == TRUE_ZERO                                                      # the first zero is untouched: newtons stay the total force


def test_the_relaxed_level_needs_a_zero_first_and_changes_nothing_without_one():
    src = zeroed_source()
    why = src.relax("x")
    assert why and "Zero (Z) first" in why and src.calibration.relaxed_v is None


def test_a_leg_still_relaxing_is_refused_with_a_hint_for_the_operator():
    """A strap or a muscle that is still settling drifts: the level would be where it was a second ago, not where it ends up."""
    src = source_with(signal(2.4, noise=0.002, ramp=0.05) + TENSION_V)                 # 50 mV over the window: its halves differ by 25 mV
    src.calibration.set_zero(TRUE_ZERO)
    why = src.relax("x")
    assert why and "not steady" in why and "relaxed and still" in why
    assert src.calibration.relaxed_v is None


def test_a_new_zero_clears_the_relaxed_level_because_the_set_up_changed():
    src = zeroed_source()
    src.calibration.set_zero(TRUE_ZERO)
    assert src.relax("x") is None and src.calibration.relaxed_v is not None
    src.calibration.set_zero(TRUE_ZERO + 0.001)
    cal = src.calibration
    assert cal.relaxed_v is None and cal.relaxed_n is None and cal.relaxed_for == "" and cal.relaxed_when == ""


def test_the_relaxed_level_is_refused_during_a_recording_and_old_files_still_load():
    src = zeroed_source()
    src.calibration.set_zero(TRUE_ZERO)
    src.frozen = True
    assert "Stop recording" in src.relax("x")
    cal = ForceCalibration(n_per_v=N_PER_V)
    cal.set_zero(0.1)
    cal.set_relaxed(0.25, noise_v=0.01, samples=2000, for_what="a")
    d = cal.to_dict()
    assert ForceCalibration.from_dict(d).to_dict() == d
    old = {k: v for k, v in d.items() if not k.startswith("relaxed_")}                  # a file from before 0.22.0
    back = ForceCalibration.from_dict(old)
    assert back.relaxed_v is None and back.relaxed_n is None and back.calibrated


def test_without_a_slope_the_relaxed_level_is_volts_only():
    cal = ForceCalibration()
    cal.set_zero(0.1)
    cal.set_relaxed(0.25)
    assert cal.relaxed_n is None and cal.relaxed_v == 0.25


# ------------------------------------------------------------------------------------------ the real recording path
def position_engine(tmp_path):
    eng = Engine(T.TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.participant_code = "P01"
    eng.process(render(T.PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(T.PTS[n] + 2))
    return eng


def record(eng, who="P01", frames=40):
    paths = eng.start_trial(who)
    for i in range(frames):
        eng.process(render(T.PTS, seed=i % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    return json.loads(paths["meta.json"].read_text())


def test_a_recording_saves_the_relaxed_level_and_only_uses_it_for_the_position_it_was_taken_for(tmp_path):
    eng = position_engine(tmp_path)
    force = SimulatedForce(rest_s=60.0, offset_v=0.0123, n_per_v=100.0)               # a long rest: the participant does not push
    force.start()
    eng.force = force
    try:
        time.sleep(2.4)
        assert force.zero(seconds=1.0) is None
        assert "above relaxed" not in eng._force_header()[0]                          # nothing taken yet
        assert force.relax(eng.relaxed_key()) is None
        assert eng.relaxed_here() is not None
        assert re.search(r"above relaxed -?\d N", eng._force_header()[0])                 # at rest: about nothing above it
        meta = record(eng)
        rel = meta["force"]["relaxed"]
        assert rel["this_trial"] is True and rel["taken_for"] == "P01 | knee_extension | 70"
        assert rel["volts"] == pytest.approx(0.0123, abs=0.002) and rel["newtons_above_zero"] == pytest.approx(0.0, abs=0.5)
        assert meta["force_setup"]["calibration"]["relaxed_v"] == rel["volts"]                # also in the set-up the trial started with

        eng.target = 60.0                                                                      # another position: the tension is not the same
        assert eng.relaxed_here() is None and "above relaxed" not in eng._force_header()[0]
        other = record(eng)["force"]["relaxed"]
        assert other["this_trial"] is False and other["taken_for"] == "P01 | knee_extension | 70"

        eng.target = 70.0
        eng.participant_code = "P02"                                                         # another person
        assert eng.relaxed_here() is None
    finally:
        force.stop()


def test_a_recording_without_a_relaxed_level_has_none_in_its_meta(tmp_path):
    eng, paths = T.recording(tmp_path)
    assert json.loads(paths["meta.json"].read_text())["force"]["relaxed"] is None


# ------------------------------------------------------------------------------------------ Review and the session sheet
def trial_with_relaxed(tmp_path, this_trial=True, relaxed_v=0.2):
    """humming_trial's force is volts only: rest 0.1 V, hold 1.0 V. Newtons are made from it with a slope of 500 N/V and a zero at 0.05 V."""
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    tr.force_cal = {"offset_v": 0.05, "n_per_v": 500.0, "zero_set": True, "zero_v": 0.05}
    tr.force_n = (tr.force_v - 0.05) * 500.0
    tr.meta.setdefault("force", {})["relaxed"] = {"volts": relaxed_v, "this_trial": this_trial, "taken_for": "P01 | knee_extension | 80"}
    return tr


def test_review_gives_the_peak_above_the_relaxed_level_beside_the_total_force(tmp_path):
    tr = trial_with_relaxed(tmp_path)
    fs = tr.force_summary(*tr.force_window())
    assert fs["unit"] == "N"
    assert fs["relaxed"] == pytest.approx((0.2 - 0.05) * 500.0)                          # 75 N: the tension that was there before the push
    assert fs["peak_minus_relaxed"] == pytest.approx(fs["peak"] - fs["relaxed"])
    assert fs["peak"] == pytest.approx((1.0 - 0.05) * 500.0, rel=0.02)                   # the total force from the zero is unchanged
    assert fs["rest_minus_relaxed"] == pytest.approx(fs["rest_mean"] - fs["relaxed"])    # the strap eased 25 N (rest 0.1 V) since it was taken
    assert fs["rest_minus_relaxed"] < 0
    QApplication.instance() or QApplication([])
    tab = ReviewTab(tmp_path)
    tab.trial = tr
    text = tab._force_text(*tr.force_window())
    assert "relaxed (Zero 2)" in text and "peak above relaxed" in text and "lower than that" in text


def test_review_follows_a_zero_recalculated_later_because_it_works_from_the_saved_volts(tmp_path):
    tr = trial_with_relaxed(tmp_path)
    tr.force_cal = {"offset_v": 0.10, "n_per_v": 500.0, "zero_set": True, "zero_v": 0.10}
    assert tr.relaxed_level("N")[0] == pytest.approx((0.2 - 0.10) * 500.0)
    assert tr.relaxed_level("V")[0] == pytest.approx(0.2)


def test_a_level_taken_for_another_position_is_not_used_and_review_says_so(tmp_path):
    tr = trial_with_relaxed(tmp_path, this_trial=False)
    fs = tr.force_summary(*tr.force_window())
    assert np.isnan(fs["relaxed"]) and np.isnan(fs["peak_minus_relaxed"])
    assert "another position (P01 | knee_extension | 80)" in fs["relaxed_note"]
    QApplication.instance() or QApplication([])
    tab = ReviewTab(tmp_path)
    tab.trial = tr
    assert "another position" in tab._force_text(*tr.force_window())


def test_a_trial_without_a_relaxed_level_is_as_before(tmp_path):
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    fs = tr.force_summary(*tr.force_window())
    assert np.isnan(fs["relaxed"]) and fs["relaxed_note"] == ""
    row = analysis.session_row(tr)
    assert row["force_relaxed"] == "" or row["force_relaxed"] != row["force_relaxed"]       # blank or NaN


def test_the_session_sheet_has_both_columns_and_the_values(tmp_path):
    tr = trial_with_relaxed(tmp_path)
    row = analysis.session_row(tr)
    assert row["force_relaxed"] == pytest.approx(75.0) and row["force_peak_above_relaxed"] == pytest.approx(row["force_peak"] - 75.0)
    assert row["force_peak_above_rest"] == pytest.approx(row["force_peak"] - row["force_rest"])      # the automatic rest is still there
    out = tmp_path / "sheet.csv"
    analysis.write_session_csv([row], out)
    back = list(csv.DictReader(open(out, newline="")))
    assert list(back[0]) == analysis.SESSION_COLUMNS and float(back[0]["force_relaxed"]) == pytest.approx(75.0, abs=0.01)


# ------------------------------------------------------------------------------------------ the window
def test_the_second_zero_button_and_its_key_in_the_window(tmp_path, monkeypatch):
    """Press the real buttons (and the X key's own shortcut) on a window with the simulated force; read what the operator would read."""
    from pathlib import Path

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWidgets import QMessageBox

    from limbtrack.gui import MainWindow
    from limbtrack.protocol import load_protocol

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    win = MainWindow(load_protocol(Path(__file__).resolve().parent.parent / "protocol.json"), tmp_path)
    live = win.live

    def pump(cond, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    try:
        assert not live.btn_force_relax.isEnabled()                                      # nothing connected
        live.force_kind.setCurrentIndex(live.force_kind.findData("simulated"))
        live.btn_force.click()
        live.force_source.rest_s = 60.0                                                  # the participant does not push
        assert live.btn_force_relax.isEnabled() and "(X)" in live.btn_force_relax.text()
        assert pump(lambda: live.force_source.average_volts(3.0)[1] >= 2400)
        shortcut = next(sc for sc in live.findChildren(QShortcut) if sc.key() == QKeySequence(Qt.Key_X))
        assert not shortcut.autoRepeat()                                                 # a held key must not fire again and again
        shortcut.activated.emit()                                                        # X before the zero: told what to do first
        assert "Zero (Z) first" in live.force_info.text()
        live.btn_force_zero.click()
        assert "Zero set" in live.force_info.text() and "above relaxed" not in live.force_label.text()
        shortcut.activated.emit()
        info = live.force_info.text()
        assert "Relaxed level set" in info and "for this position" in info, info
        assert "above relaxed" in live.force_label.text()
        win.engine.target = 90.0                                                         # another position: no longer shown, and said so
        live.update_force_box()
        assert "above relaxed" not in live.force_label.text() and "another position" in live.force_info.text()
        win.engine.target = 70.0
        live.btn_force_zero.click()                                                      # a new zero: the relaxed level goes
        assert "relaxed level was cleared" in live.force_info.text() and live.force_source.calibration.relaxed_v is None
        assert "above relaxed" not in live.force_label.text()
    finally:
        win.close()
