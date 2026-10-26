"""Force from a load cell through a DAQ: calibration, the simulated source, recording (force.csv, force_n, meta) and Review.
The real NI device is not needed: SimulatedForce uses the same buffer, clock alignment and recording path."""
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import (STANDARD_GRAVITY, ForceCalibration, ForceError, NiDaqForce,  # noqa: E402
                             SimulatedForce, make_source)
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


# ------------------------------------------------------------------------------------------ calibration
def test_calibration_is_zero_then_a_known_mass():
    c = ForceCalibration()
    assert not c.calibrated and np.isnan(c.to_newton(0.5)).all()
    c.set_zero(0.100)
    assert c.calibrate_with_mass(0.100 + 0.5, 5.0) is None
    assert c.n_per_v == pytest.approx(5.0 * STANDARD_GRAVITY / 0.5)
    assert c.to_newton(0.100) == pytest.approx(0.0, abs=1e-9)
    assert c.to_newton(0.600) == pytest.approx(5.0 * STANDARD_GRAVITY)
    assert c.to_newton(np.array([0.1, 0.6])).shape == (2,)


def test_the_load_cell_slope_gives_newtons_only_after_a_zero():
    c = ForceCalibration.load_cell()
    assert c.has_slope and not c.calibrated and np.isnan(c.to_newton(0.5)).all()      # an unzeroed offset would be a guess
    assert c.regression_id and c.method == f"load cell regression {c.regression_id}"
    c.set_zero(0.0870)
    assert c.calibrated and c.method.startswith("load cell regression")              # set_zero keeps the slope and its name
    assert c.to_newton(0.0870) == pytest.approx(0.0, abs=1e-9)
    assert c.to_newton(0.0870 + 0.0186102 * 10) == pytest.approx(10 * STANDARD_GRAVITY, rel=2e-4)       # 10 kg at 18.6102 mV/kg
    c.calibrate_with_mass(0.0870 + 0.5, 5.0)                                         # a known weight replaces the regression's slope
    assert c.regression_id == "" and c.n_per_v == pytest.approx(5.0 * STANDARD_GRAVITY / 0.5)


def test_a_calibration_that_cannot_be_right_is_refused_and_changes_nothing():
    c = ForceCalibration()
    c.set_zero(0.1)
    assert "moved only" in c.calibrate_with_mass(0.1005, 5.0)           # the load did not move the signal
    assert "above zero" in c.calibrate_with_mass(0.6, 0.0)
    assert not c.calibrated


def test_calibration_is_saved_and_read_back(tmp_path):
    c = ForceCalibration()
    c.set_zero(0.2)
    c.calibrate_with_mass(0.7, 10.0)
    path = tmp_path / "cal.json"
    c.save(path)
    back = ForceCalibration.load(path)
    assert back.to_dict() == c.to_dict() and back.calibrated
    assert not ForceCalibration.load(tmp_path / "missing.json").calibrated


# ------------------------------------------------------------------------------------------ the source
def test_the_simulated_source_delivers_the_asked_rate_with_timestamps_that_run_forward():
    s = SimulatedForce()
    t_start = time.perf_counter()
    s.start()
    time.sleep(1.6)
    t, v = s.samples_between(t_start - 1, time.perf_counter() + 1)
    s.stop()
    assert 1400 < t.size < 1800
    assert np.all(np.diff(t) >= 0)
    assert (t.size - 1) / (t[-1] - t[0]) == pytest.approx(1000.0, rel=0.03)
    assert 0 <= time.perf_counter() - t[-1] < 0.2                       # the newest sample is recent: the clock alignment works


def test_zero_and_calibration_from_the_live_signal():
    s = SimulatedForce(rest_s=30.0, offset_v=0.0123, n_per_v=100.0)     # a long rest: nothing is loaded
    s.calibration = ForceCalibration()
    s.start()
    time.sleep(2.3)
    assert s.zero(seconds=2.0) is None
    assert s.calibration.offset_v == pytest.approx(0.0123, abs=0.001)
    s.stop()


def test_zero_is_refused_while_the_signal_moves_and_during_a_recording():
    s = SimulatedForce(rest_s=0.0, ramp_s=0.5, hold_s=0.3)             # the force is rising while it is measured
    s.start()
    time.sleep(1.2)
    assert "not steady" in (s.zero(seconds=0.4) or "")
    s.frozen = True
    assert "Stop recording" in s.zero()
    s.stop()


def test_a_missing_or_busy_device_gives_advice_not_a_traceback():
    f = NiDaqForce(device="Dev9")
    assert "in use by another program" in f._explain_open(Exception("Resource reserved. Error code: -50103"))
    assert "No DAQ device called 'Dev9'" in f._explain_open(Exception("Device identifier is invalid. Error code: -200220"))
    assert "lost samples" in f._explain(Exception("Onboard device memory overflow. -200279"))
    assert isinstance(NiDaqForce.devices(), list)
    with pytest.raises(ValueError):
        make_source("nonsense")
    assert make_source("simulated").kind == "simulated"


# ------------------------------------------------------------------------------------------ recording
def recording(tmp_path, seconds=3.0, **sim):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    force = SimulatedForce(**{"rest_s": 0.3, "ramp_s": 0.6, "hold_s": 1.2, "peak_n": 400.0, **sim})
    force.start()
    eng.force = force
    time.sleep(0.3)
    paths = eng.start_trial("P01")
    n = int(seconds * 30)
    for i in range(n):
        eng.process(render(PTS, seed=i % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    return eng, paths


def test_a_recording_with_force_saves_every_sample_and_a_column_per_picture(tmp_path):
    eng, paths = recording(tmp_path)
    assert "force.csv" in paths
    meta = json.loads(paths["meta.json"].read_text())
    f = meta["force"]
    assert f["source"] == "simulated" and f["calibrated"] is True and f["sample_rate_hz"] == 1000.0
    assert f["samples"] > 2400 and f["measured_rate_hz"] == pytest.approx(1000.0, rel=0.05)           # (rendering the pictures is slow: the run lasts longer than 3 s)
    assert f["duration_s"] == pytest.approx(f["samples"] / 1000.0, rel=0.05)
    assert f["peak_n"] == pytest.approx(400.0, rel=0.05)
    assert 0.0 < f["peak_t_s"] < meta["duration_s"] + 0.5
    assert meta["force_setup"]["calibration"]["n_per_v"] == 100.0 and "clock" in f and f["frames_without_samples_over_0_5s"] == 0
    frows = list(csv.DictReader(open(paths["force.csv"], newline="")))
    assert list(frows[0]) == ["t_s", "clock_s", "volts", "newtons"] and len(frows) == f["samples"]
    t = np.array([float(r["t_s"]) for r in frows])
    assert np.all(np.diff(t) >= 0)
    peak_row = max(frows, key=lambda r: float(r["newtons"]))
    assert float(peak_row["newtons"]) == pytest.approx(f["peak_raw_n"], abs=0.01)     # the raw peak is the highest single sample
    assert f["peak_n"] <= f["peak_raw_n"] and f["peak_raw_n"] - f["peak_n"] < 5.0   # the 20 ms average: the noise of a sample or two less
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert "force_n" in rows[0]
    vals = [float(r["force_n"]) for r in rows if r["force_n"] != ""]
    assert len(vals) > 0.8 * len(rows) and max(vals) == pytest.approx(400.0, rel=0.1)


def test_a_recording_without_force_has_no_force_files_or_column(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(5):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    assert "force.csv" not in paths
    meta = json.loads(paths["meta.json"].read_text())
    assert "force" not in meta and meta["force_setup"] is None
    assert "force_n" not in next(csv.DictReader(open(paths["data.csv"], newline="")))


def test_recording_refuses_a_force_that_is_not_running_and_force_cannot_change_while_recording(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    dead = SimulatedForce()                                             # never started
    eng.force = dead
    with pytest.raises(RuntimeError, match="force reading is not running"):
        eng.start_trial("P01")
    live = SimulatedForce()
    live.start()
    eng.force = live
    time.sleep(0.2)
    eng.start_trial("P01")
    assert live.frozen is True
    with pytest.raises(RuntimeError, match="Stop recording"):
        eng.force = None
    eng.stop_trial()
    assert live.frozen is False
    live.stop()


def test_force_is_refused_with_a_video_file_because_the_clocks_do_not_match(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.set_source("clip.mp4", {"is_file": True})
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    live = SimulatedForce()
    live.start()
    eng.force = live
    with pytest.raises(RuntimeError, match="live camera"):
        eng.start_trial("P01")
    live.stop()


def test_the_picture_header_shows_the_force(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), time.perf_counter())
    live = SimulatedForce(rest_s=30.0)
    live.start()
    eng.force = live
    time.sleep(0.4)
    res = eng.process(render(PTS, seed=1), time.perf_counter())
    assert any(h.startswith("Force ") and " N" in h for h in res.header)
    live.calibration = ForceCalibration()                              # uncalibrated: volts, and it says so
    res = eng.process(render(PTS, seed=1), time.perf_counter())
    assert any("not calibrated" in h for h in res.header)
    live.stop()


# ------------------------------------------------------------------------------------------ review
def test_review_reports_the_peak_force_and_the_angle_at_the_peak(tmp_path):
    eng, paths = recording(tmp_path, seconds=3.0)
    tr = Trial(paths["meta.json"])
    assert tr.has_force()
    fs = tr.force_summary(-1.0, float(tr.t[-1]))
    assert fs["unit"] == "N" and fs["peak"] == pytest.approx(400.0, rel=0.05)
    assert 0 < fs["t_peak_s"] < float(tr.t[-1]) + 0.5
    assert fs["main_angle_at_peak_deg"] == pytest.approx(70.0, abs=2.0)             # the leg stayed at 70 deg
    high = tr.force_t[tr.force_n > 350.0]                                             # the hold: where the force is near its top
    first = high[: int(np.argmax(np.diff(high) > 0.5)) + 1] if np.any(np.diff(high) > 0.5) else high    # the first hold only (a slow run holds twice)
    hold = tr.force_summary(float(first.min()), float(first.max()))
    assert hold["plateau_mean"] == pytest.approx(400.0, rel=0.03) and hold["rest_mean"] < 300.0
    assert tr.force_summary(100.0, 101.0) is None                                    # a window with no samples


def test_review_without_force_has_no_force_numbers(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(5):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    tr = Trial(paths["meta.json"])
    assert not tr.has_force() and tr.force_summary(0, 1) is None


# ------------------------------------------------------------------------------------------ voltage channel, range limit, later calibration
def uncalibrated_recording(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    force = SimulatedForce(rest_s=0.3, ramp_s=0.6, hold_s=1.2)
    force.calibration = ForceCalibration()                              # nothing calibrated yet: only volts can be known
    force.start()
    eng.force = force
    time.sleep(0.3)
    paths = eng.start_trial("P01")
    for i in range(60):
        eng.process(render(PTS, seed=i % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    return eng, paths


def test_the_voltage_is_always_logged_and_newtons_stay_blank_until_a_calibration_exists(tmp_path):
    eng, paths = uncalibrated_recording(tmp_path)
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["force"]["calibrated"] is False and meta["force"]["peak_n"] is None
    frows = list(csv.DictReader(open(paths["force.csv"], newline="")))
    assert all(r["newtons"] == "" for r in frows) and all(r["volts"] != "" for r in frows)
    volts = np.array([float(r["volts"]) for r in frows])
    assert volts.max() > 3.5 and abs(np.percentile(volts, 2) - 0.0123) < 0.02     # the made-up contraction reaches ~4 V; at rest it reads its offset
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert "force_v" in rows[0] and "force_n" in rows[0]
    assert all(r["force_n"] == "" for r in rows) and sum(r["force_v"] != "" for r in rows) > 0.8 * len(rows)
    assert max(float(r["force_v"]) for r in rows if r["force_v"]) == pytest.approx(volts.max(), rel=0.2)


def test_a_signal_at_the_edge_of_the_input_range_is_flagged(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    force = SimulatedForce(rest_s=0.2, ramp_s=0.3, hold_s=1.5, peak_n=1500.0)       # 15 V: beyond the +-10 V input
    force.start()
    eng.force = force
    time.sleep(0.3)
    paths = eng.start_trial("P01")
    warned = False
    for i in range(45):
        res = eng.process(render(PTS, seed=i % 5), time.perf_counter())
        warned = warned or any("LIMIT" in h for h in res.header)
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    meta = json.loads(paths["meta.json"].read_text())["force"]
    assert warned and meta["samples_at_range_limit"] > 100 and meta["range_v"] == 10.0
    assert force.clipped_total >= meta["samples_at_range_limit"]


def test_a_calibration_made_later_turns_the_saved_volts_into_newtons(tmp_path):
    eng, paths = uncalibrated_recording(tmp_path)
    tr = Trial(paths["meta.json"])
    assert tr.force_cal_note == "no calibration yet: volts only"
    before = tr.force_summary(-1.0, float(tr.t[-1]))
    assert before["unit"] == "V" and before["peak"] > 3.5
    volts_before = tr.force_v.copy()
    cal = ForceCalibration()
    cal.set_zero(0.0123)
    cal.n_per_v = 100.0
    cal.method = "zero and 5 kg"
    tr.apply_calibration(cal, "applied later from force_calibration.json (test)")
    after = tr.force_summary(-1.0, float(tr.t[-1]))
    assert after["unit"] == "N" and after["peak"] == pytest.approx(400.0, rel=0.05)
    assert np.array_equal(tr.force_v, volts_before)                                  # the recorded volts are untouched
    assert "applied later" in tr.force_cal_note


def test_the_picture_delay_moves_where_the_angle_at_the_peak_is_read(tmp_path):
    eng, paths = recording(tmp_path)
    tr = Trial(paths["meta.json"])
    main = next(a for a in tr.angle_defs if a.primary)
    tr.deg[main.name] = np.linspace(0.0, 100.0, len(tr.t))                           # an angle that changes steadily with time
    at0 = tr.force_summary(-1.0, float(tr.t[-1]), 0.0)
    at_delay = tr.force_summary(-1.0, float(tr.t[-1]), 0.5)
    slope = 100.0 / float(tr.t[-1] - tr.t[0])
    assert at_delay["main_angle_at_peak_deg"] - at0["main_angle_at_peak_deg"] == pytest.approx(0.5 * slope, rel=0.2)
    assert at_delay["picture_delay_s"] == 0.5


def test_review_recalculates_newtons_with_the_active_regression_and_exports_what_was_used(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from limbtrack import review as review_mod
    from limbtrack.review import ReviewTab
    app = QApplication.instance() or QApplication([])
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a[2]))
    eng, paths = uncalibrated_recording(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    assert "volts only" in tab.force_text.text() and "not calibrated" in tab.force_text.text()
    assert "in volts" in tab.calib.text.text()                                  # the Calibration tab says there are no newtons yet
    assert tab.zero_spin.value() == pytest.approx(0.0123, abs=0.003)           # no zero was taken: the resting level
    real = review_mod.active_regression
    monkeypatch.setattr(review_mod, "active_regression", lambda *a: None)
    tab.apply_force_calibration()                                              # no regression: explained, nothing changes
    assert infos and "regression" in infos[-1]
    monkeypatch.setattr(review_mod, "active_regression", real)
    tab.zero_spin.setValue(0.0123)
    tab.apply_force_calibration()
    txt, ctxt = tab.force_text.text(), tab.calib.text.text()
    assert "applied later" in txt and "Force in the window (N" in txt
    reg = real()
    assert reg.id in ctxt and "peak" in ctxt and "about 95 %" in ctxt and "N per volt" in ctxt          # the regression and the precision of the peak
    tab.export()
    summary = list(csv.DictReader(open(next(tmp_path.rglob("*_summary.csv")), newline="")))
    assert summary[0]["force_unit"] == "N" and float(summary[0]["force_n_per_v"]) == pytest.approx(reg.n_per_v, abs=0.01)
    assert summary[0]["force_regression_id"] == reg.id and float(summary[0]["force_peak_95pct_n"]) > 1.0
    assert "applied later" in summary[0]["force_calibration_note"] and summary[0]["picture_delay_s"] == "0.0"
    saved_volts = [float(r["volts"]) for r in csv.DictReader(open(paths["force.csv"], newline=""))]
    assert max(saved_volts) > 3.5                                              # the file on disk still holds the volts, unchanged


def test_a_trial_remembers_the_regression_it_was_measured_with(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    force = SimulatedForce(rest_s=0.3, ramp_s=0.6, hold_s=1.2)
    force.calibration = ForceCalibration.load_cell()
    force.calibration.set_zero(0.0123)
    force.start()
    eng.force = force
    time.sleep(0.3)
    paths = eng.start_trial("P01")
    for i in range(45):
        eng.process(render(PTS, seed=i % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    meta = json.loads(paths["meta.json"].read_text())
    cal = meta["force_setup"]["calibration"]
    assert cal["regression_id"] and cal["zero_v"] == 0.0123 and meta["force"]["calibrated"] is True
    tr = Trial(paths["meta.json"])
    assert tr.regression is not None and tr.regression.id == cal["regression_id"]


# ------------------------------------------------------------------------------------------ smoothing, mains hum, window from the force
def humming_trial(tmp_path, hum=0.09):
    """A recording whose force file is replaced by a known signal: rest, a light push, a hold at 1.0 V, release; plus 50 Hz hum and noise."""
    eng, paths = recording(tmp_path)
    tr = Trial(paths["meta.json"])
    t = np.arange(-1.0, 16.0, 0.001)
    clean = np.interp(t, [-1, 3, 3.05, 8, 8.6, 13, 13.5, 16], [0.1, 0.1, 0.3, 0.3, 1.0, 1.0, 0.1, 0.1])   # rest, a light push, the hold at 1.0 V, release
    rng = np.random.RandomState(1)
    v = clean + hum * np.sin(2 * np.pi * 50.0 * t) + rng.normal(0, 0.01, t.size)
    tr.force_t, tr.force_v, tr.force_n = t, v, np.full(t.size, np.nan)
    # The pictures of the real (3 s of pictures) recording get a time axis that spans the injected signal. Before, they ended where the
    # real recording ended, which depends on how fast the computer renders the pictures (about 7.5 s on an idle machine): the torque
    # and picture-delay tests, which look at 8 to 13 s, passed only on a busy machine.
    tr.t = np.linspace(0.0, 15.0, len(tr.t))
    return tr, clean


def test_a_moving_average_that_is_a_multiple_of_20_ms_removes_50_hz_hum_exactly():
    from limbtrack.force import smooth_signal
    t = np.arange(0, 3, 0.001)
    x = 1.0 + 0.09 * np.sin(2 * np.pi * 50 * t + 0.7)
    for ms in (20, 40, 100, 200):
        y = smooth_signal(x, 1000.0, ms / 1000.0)
        assert np.abs(y[300:-300] - 1.0).max() < 1e-9, ms
    assert np.abs(smooth_signal(x, 1000.0, 0.03)[300:-300] - 1.0).max() > 0.01          # 30 ms is not a whole number of hum cycles
    assert np.array_equal(smooth_signal(x, 1000.0, 0.0), x) and smooth_signal(x, 1000.0, 0.1).shape == x.shape
    edge = smooth_signal(np.array([5.0, 5.0, 5.0, 5.0, 5.0]), 1000.0, 0.003)
    assert np.allclose(edge, 5.0)                                                      # the edges do not drift toward zero


def test_the_hum_amplitude_is_measured():
    from limbtrack.force import mains_amplitude
    t = np.arange(0, 4, 0.001)
    x = 0.3 + 0.2 * t + 0.09 * np.sin(2 * np.pi * 50 * t + 1.1) + 0.01 * np.random.RandomState(0).randn(t.size)
    assert mains_amplitude(x, 1000.0) == pytest.approx(0.09, rel=0.08)
    assert np.isnan(mains_amplitude(x[:50], 1000.0))


def test_smoothing_lowers_the_peak_of_a_noisy_signal_and_review_reports_the_hum(tmp_path):
    tr, clean = humming_trial(tmp_path)
    tr.force_smooth_s = 0.0
    raw = tr.force_summary(8.0, 13.0)
    tr.force_smooth_s = 0.1
    smooth = tr.force_summary(8.0, 13.0)
    assert raw["peak"] > smooth["peak"] + 0.05                                          # single noisy samples overstate the peak
    assert smooth["peak"] == pytest.approx(1.0, abs=0.01) and smooth["raw_peak"] == pytest.approx(raw["peak"])
    assert smooth["mains_hum_v"] == pytest.approx(0.09, rel=0.08) and 11 < smooth["mains_hum_pct_of_peak"] < 15        # 0.09 V of a 0.7 V rise above the rest before the window
    assert smooth["smoothing_s"] == 0.1


def test_the_window_from_the_force_ignores_a_light_first_push_and_early_markers(tmp_path):
    tr, clean = humming_trial(tmp_path)
    w = tr.force_window()
    assert w is not None and w[0] == pytest.approx(8.3, abs=0.2) and w[1] == pytest.approx(13.2, abs=0.2)     # the hold, not 3-6 s
    early = tr.force_window(0.10)
    assert early[0] < 3.5                                                               # a low threshold would include the first push
    quiet, _ = humming_trial(tmp_path / "q", hum=0.0)
    quiet.force_v = np.full_like(quiet.force_v, 0.1) + np.random.RandomState(2).normal(0, 0.002, quiet.force_v.size)
    assert quiet.force_window() is None                                                 # no contraction: no window


def test_the_live_read_out_is_smoothed_but_the_samples_stay_raw():
    from limbtrack.force import ForceSource
    src = ForceSource()
    src.calibration = ForceCalibration(offset_v=0.0, n_per_v=10.0)
    t = np.arange(0, 1.0, 0.001)
    v = 1.0 + 0.09 * np.sin(2 * np.pi * 50 * t)
    for k in range(0, 1000, 10):
        src._push(v[k:k + 10], time.perf_counter())
    src.smooth_s = 0.0
    raw_values = [src.latest_volts() for _ in range(1)]
    src.smooth_s = 0.1
    assert src.latest_volts() == pytest.approx(1.0, abs=1e-6) and src.latest_newtons() == pytest.approx(10.0, abs=1e-5)
    _, shown = src.recent(2.0)
    assert np.abs(shown[100:-100] - 1.0).max() < 1e-6                                   # the trace shows the smoothed signal
    _, stored = src.samples_between(0, float("inf"))
    assert stored.size == 1000 and np.abs(stored - v).max() < 1e-12                       # what is saved is the raw signal
    assert abs(raw_values[0] - 1.0) <= 0.0901
