"""0.26.0, from the review of the pilot sessions P01 (2026-10-28) and P02 (2026-10-29): a trial whose pull segment dots were switched off
crashed Review and the 10 % check; the 10 % check called two trials without any force 'no third trial needed'; a trial in which the
automatic MVIC marker never fired got no warning; a peak cut off at the input limit or beyond the calibration was not said after the trial;
and, new after each trial, the main joint during the push (with the shift from rest), an angle that does not fit the position, and a Zero 2
that does not fit the trial."""
import json
import math
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import NO_FORCE_RISE_N, SimulatedForce  # noqa: E402
from limbtrack.loadcell import GRAVITY  # noqa: E402
from limbtrack.review import ReviewTab, Trial, after_trial_check  # noqa: E402
from limbtrack.synthetic import render  # noqa: E402

from tests import test_force as T  # noqa: E402
from tests.test_analysis import torque_trial  # noqa: E402
from tests.test_p20_checks import contraction, force_trial  # noqa: E402
from tests.test_rest_and_ten_percent import rows, with_force  # noqa: E402


# ------------------------------------------------------------------------------------------------ F2: segment dots switched off
def test_a_trial_whose_segment_dots_are_off_opens_and_has_no_torque(tmp_path):
    tr = torque_trial(tmp_path)
    tr.meta["pull"] = {"segment_from_to": ["mt5_base", "mt5_head"], "moment_arm_from_joint_dot": "malleolus"}   # P01 2026-10-28: foot dots off
    assert "mt5_base" not in tr.tracks
    assert analysis.torque_series(tr) is None and analysis.torque_summary(tr, 0.0, 15.0) is None
    row = analysis.session_row(tr)                                         # the 10 % check and the export: KeyError 'mt5_base' until 0.25.0
    assert row["torque_peak_nm"] == "" and row["force_peak"] != ""
    QApplication.instance() or QApplication([])
    tab = ReviewTab(tmp_path)
    tab.trial = tr
    assert "Torque" not in tab._force_text(*tr.force_window()).split("Timing")[0]          # Review's force text is made (it raised before)


def test_with_the_segment_dots_there_the_torque_is_unchanged(tmp_path):
    tr = torque_trial(tmp_path)
    tq = analysis.torque_summary(tr, 9.0, 12.0)
    assert tq is not None and tq["peak_nm"] > 0


# ------------------------------------------------------------------------------------------------ F3: the 10 % check without force
def test_two_trials_without_a_force_rise_are_not_called_fine():
    chk = analysis.position_check(rows(-1.0, -3.0, above=[3.0, 2.0]))     # P02 knee extension 70, T1 and T2 (strap not pulling)
    assert chk["state"] == "no_rise" and "No third trial needed" not in chk["text"]
    assert "No force rise in trial 1 and 2" in chk["text"] and f"{NO_FORCE_RISE_N:g} N" in chk["text"]
    r = [{"test_key": "k", "target_deg": 70, **x} for x in rows(-1.0, -3.0, above=[3.0, 2.0])]
    analysis.repeatability(r)
    assert "repeatability_pct_same_position" not in r[0] and "third_trial_needed" not in r[0]       # left alone (was -73.5 % and False)


def test_trials_without_a_rise_are_named_and_left_out_and_the_numbers_are_kept():
    one = analysis.position_check(rows(-1.0, -3.0, 653.0, above=[3.0, 2.0, 643.0]))
    assert one["state"] == "one" and "trial 3 of this position; the 10 % check follows after trial 4" in one["text"]
    assert "Not compared: trial 1 and 2" in one["text"]
    two = analysis.position_check(rows(-1.0, -3.0, 653.0, 720.0, above=[3.0, 2.0, 643.0, 710.0]))
    assert two["state"] == "ok" and "trial 3 653 N, trial 4 720 N" in two["text"] and two["pct"] == pytest.approx(9.76, abs=0.01)
    assert analysis.position_check(rows(400.0, 380.0))["text"].endswith("No third trial needed.")   # with force only: as before


def test_a_peak_cut_off_at_the_input_limit_is_a_lower_bound():
    r = rows(3264.0, 5231.0, above=[3051.0, 4936.0])
    r[1]["force_samples_at_limit"] = 57                                         # P02 belt squat 60, T2
    chk = analysis.position_check(r)
    assert chk["state"] == "third" and "trial 2 at least 5231 N" in chk["text"] and "lower bound" in chk["text"]


def test_the_export_row_says_how_many_samples_were_at_the_limit(tmp_path):
    tr = torque_trial(tmp_path)
    tr.meta.setdefault("force", {})["samples_at_range_limit"] = 12
    assert analysis.session_row(tr)["force_samples_at_limit"] == 12 and "force_samples_at_limit" in analysis.SESSION_COLUMNS


# ------------------------------------------------------------------------------------------------ F5: the rest note without a push
def test_no_steadiness_note_when_there_was_no_push(tmp_path):
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    t = np.arange(-1.0, 16.0, 0.001)
    flat = np.random.RandomState(0).normal(0.0, 0.5, t.size) + np.interp(t, [-1, 8, 8.5, 10, 10.5, 16], [0, 0, 4, 4, 0, 0])
    tr = with_force(tr, t, flat)                                               # a 4 N 'push': the window exists, the rise does not
    ri = tr.force_rest()
    assert ri is not None and ri["note"] == ""                                 # was 'not steady (it varied by 0 N, SD)'


def test_a_real_push_after_an_unsteady_rest_still_gets_the_note(tmp_path):
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    t = np.arange(-1.0, 16.0, 0.001)
    wobble = 80.0 * np.sin(2 * np.pi * 0.7 * t) * (t < 5.0)                  # a strap loaded and moving before the push (SD 57 N)
    tr = with_force(tr, t, 50.0 + wobble + np.interp(t, [-1, 5, 6, 11, 11.5, 16], [0, 0, 350, 350, 0, 0]))
    assert "not steady" in tr.force_rest()["note"]


# ------------------------------------------------------------------------------------------------ F4: no MVIC start, no force
def test_no_marker_and_no_force_is_called_out(tmp_path):
    eng, meta, paths, _ = force_trial(tmp_path, lambda s: 0.0 * s, markers=False)
    f = meta["force"]
    assert f["rise_basis"] == "whole recording" and f["no_rise_warning"] is True and f["rise_n"] < 5.0
    w = [x for x in eng.last_trial_warnings if "No force rise" in x]
    assert w and "no MVIC start was marked, and during the whole recording" in w[0]
    assert any("during the whole recording (no MVIC start was marked)" in text for _, text in analysis.trial_checks(Trial(paths["meta.json"])))


def test_the_automatic_marker_that_never_fired_is_named(tmp_path, monkeypatch):
    real_start = Engine.start_trial

    def start(self, *a, **k):
        self.auto_mvic = True                                                  # (armed by Record, as in P02 2026-10-29 11:53)
        return real_start(self, *a, **k)
    monkeypatch.setattr(Engine, "start_trial", start)
    eng, meta, _, _ = force_trial(tmp_path, lambda s: 0.0 * s, markers=False)
    assert meta["auto_mvic"]["state"] == "waiting"
    assert any("the automatic marker never fired" in x for x in eng.last_trial_warnings)


def test_a_push_without_markers_is_not_called_out(tmp_path):
    eng, meta, _, _ = force_trial(tmp_path, contraction(300.0), markers=False)
    f = meta["force"]
    assert f["rise_basis"] == "whole recording" and f["rise_n"] == pytest.approx(300.0, abs=10.0) and f["no_rise_warning"] is False
    assert not any("No force rise" in x for x in eng.last_trial_warnings)


def test_with_markers_the_rise_is_still_taken_between_them(tmp_path):
    _, meta, _, _ = force_trial(tmp_path, contraction(300.0))
    assert meta["force"]["rise_basis"] == "MVIC markers" and meta["force"]["no_rise_warning"] is False


# ------------------------------------------------------------------------------------------------ F1a: input limit and calibration range
def test_samples_at_the_input_limit_are_said_after_the_trial(tmp_path, monkeypatch):
    real_stop = Engine.stop_trial

    def stop(self, *a, **k):
        self.force.clipped_total = 57                                          # (the reader counts them as the chunks come in)
        return real_stop(self, *a, **k)
    monkeypatch.setattr(Engine, "stop_trial", stop)
    eng, meta, _, _ = force_trial(tmp_path, contraction(300.0))
    assert meta["force"]["samples_at_range_limit"] == 57
    w = [x for x in eng.last_trial_warnings if "input range" in x]
    assert w and "the peak is cut off" in w[0] and "Turn the amplifier gain down" in w[0]


def test_a_peak_beyond_the_calibration_is_said_and_saved(tmp_path, monkeypatch):
    real_start = Engine.start_trial

    def start(self, *a, **k):
        self.force.calibration.known_mass_kg = 250.0                           # calibrated with up to 250 kg, as the 2026-10-13 regression
        return real_start(self, *a, **k)
    monkeypatch.setattr(Engine, "start_trial", start)
    eng, meta, _, _ = force_trial(tmp_path, contraction(3000.0))              # 306 kg: the P01 standing trials went to 500 kg
    ac = meta["force"]["above_calibration"]
    assert ac["calibrated_to_kg"] == 250.0 and ac["peak_kg"] == pytest.approx(3000.0 / GRAVITY, abs=2.0)
    assert ac["seconds_above"] == pytest.approx(2.5 + 0.5 * (1 - 250.0 * GRAVITY / 3000.0), abs=0.05)
    assert any("above the heaviest load of the calibration (250 kg)" in x for x in eng.last_trial_warnings)


def test_a_peak_inside_the_calibration_says_nothing(tmp_path, monkeypatch):
    real_start = Engine.start_trial

    def start(self, *a, **k):
        self.force.calibration.known_mass_kg = 250.0
        return real_start(self, *a, **k)
    monkeypatch.setattr(Engine, "start_trial", start)
    eng, meta, _, _ = force_trial(tmp_path, contraction(1500.0))
    assert "above_calibration" not in meta["force"] and not any("calibration" in x for x in eng.last_trial_warnings)


def test_review_names_a_peak_beyond_the_regression(tmp_path):
    from limbtrack.loadcell import active_regression
    reg = active_regression()
    if reg is None:
        pytest.skip("no load cell regression stored")
    tr = torque_trial(tmp_path)
    tr.regression = reg
    tr.force_n = tr.force_n * (1.2 * reg.range_kg[1] * GRAVITY / np.nanmax(tr.force_n))
    assert any("above the heaviest calibration load" in text for lvl, text in analysis.trial_checks(tr) if lvl == "check")


# ------------------------------------------------------------------------------------------------ the three after-trial notes
def row(plateau, shift=None, target=70.0, window="force above half of peak", main="Knee flexion"):
    return {"main_angle": main, "plateau_angle_deg": plateau, "angle_rest_to_hold_deg": "" if shift is None else shift,
            "target_deg": target, "window": window}


def test_the_angle_during_the_push_with_the_shift_and_where_to_set_it():
    t = analysis.hold_angle_note(row(64.0, -24.0), 5.0)                     # P01 knee extension 70, T1
    assert t.startswith("Knee flexion during the push: 64 deg (target 70 +-5: 6 deg below).")
    assert "At rest before the push it was 88 deg, so it moved -24 deg under load." in t
    assert "set about 94 deg at rest" in t
    near = analysis.hold_angle_note(row(71.0, -3.0), 5.0)
    assert "(target 70 +-5: within)" in near and "set about" not in near   # a small shift needs no hint
    assert analysis.hold_angle_note(row(71.0, None, target=None), 5.0) == "Knee flexion during the push: 71 deg."
    assert analysis.hold_angle_note(row(71.0, -3.0, window="whole trial"), 5.0) == ""
    assert analysis.hold_angle_note(row(""), 5.0) == ""
    assert "set about 0 deg" in analysis.hold_angle_note(row(-21.0, -20.4, target=-20.0), 5.0)      # never '-0'


def test_an_angle_that_does_not_fit_the_position_is_called_out():
    t = analysis.position_angle_note("Hip flex (+) / ext (-)", 30.8, [math.nan, 70.6, 67.7, 68.1], 90.0, 5.0)   # P02 hip flexion 90, T5
    assert "30 deg" not in t and "31 deg; the earlier trials of this position: 68 to 71 deg (-37 deg)" in t and "right position" in t
    assert analysis.position_angle_note("Hip", 66.0, [70.6, 67.7], 90.0, 5.0) == ""                       # within 15 deg: fine
    assert analysis.position_angle_note("Hip", "", [70.6], 90.0, 5.0) == ""


def test_a_first_trial_that_reads_like_the_previous_position_is_called_out():
    t = analysis.position_angle_note("Knee flexion", 63.6, [], 20.0, 5.0, previous=(70.0, [64.0, 62.3, 73.1]))  # P01 knee ext 20, T1
    assert "the same as at the previous position of this test (target 70: 64 deg)" in t and "50 deg apart" in t
    assert analysis.position_angle_note("Knee flexion", 36.7, [], 20.0, 5.0, previous=(70.0, [64.0])) == ""      # moved: fine
    assert analysis.position_angle_note("Knee flexion", 64.0, [], 60.0, 5.0, previous=(70.0, [64.0])) == ""      # targets close
    assert analysis.position_angle_note("Knee flexion", 64.0, [], 20.0, 5.0, previous=None) == ""


def test_a_zero_2_that_does_not_fit_the_trial_is_called_out():
    t = analysis.relaxed_mismatch("N", 9.4, 423.0, 1715.0)                    # P02 belt squat 90, T1
    assert "Zero 2 (relaxed) was 9 N, but the force just before this push was 423 N (+414 N)" in t and "press X again" in t
    assert analysis.relaxed_mismatch("N", 7.0, 12.0, 400.0) == ""             # 5 N: fine
    assert analysis.relaxed_mismatch("N", 7.0, 38.0, 400.0) != ""             # 31 N > 20 N
    assert analysis.relaxed_mismatch("N", 0.0, 60.0, 2000.0) == ""            # 60 N < 5 % of a 2000 N rise
    assert analysis.relaxed_mismatch("V", 0.01, 0.8, 1.0) == "" and analysis.relaxed_mismatch("N", "", 423.0, 1715.0) == ""


def recording_at(root, target, seconds=3.0, **sim):
    """tests.test_force.recording, at another knee extension target (the same leg picture: the knee reads the same at every target)."""
    eng = Engine(T.TESTS, recordings_dir=root)
    eng.set_test("knee_extension")
    eng.target = target
    eng.process(render(T.PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(T.PTS[n] + 2))
    force = SimulatedForce(**{"rest_s": 0.3, "ramp_s": 0.6, "hold_s": 1.2, "peak_n": 400.0, **sim})
    force.start()
    eng.force = force
    time.sleep(0.3)
    paths = eng.start_trial("P01")
    for i in range(int(seconds * 30)):
        eng.process(render(T.PTS, seed=i % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    return paths


def test_after_a_trial_the_window_gets_the_angle_and_the_previous_position_check(tmp_path):
    root = tmp_path / "recordings"
    a = recording_at(root, 70.0)
    first = after_trial_check(a["meta.json"])
    assert first["angle_note"].startswith("Knee flexion during the push:") and first["position_note"] == "" and first["relaxed_note"] == ""
    time.sleep(1.1)
    b = recording_at(root, 70.0)
    assert after_trial_check(b["meta.json"])["position_note"] == ""          # the same angle as trial 1: fine
    time.sleep(1.1)
    c = recording_at(root, 20.0)                                              # 'moved to 20' in the program, but the leg did not move
    res = after_trial_check(c["meta.json"])
    assert "the same as at the previous position of this test (target 70" in res["position_note"]
