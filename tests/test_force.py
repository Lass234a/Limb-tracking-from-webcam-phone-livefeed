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
from limbtrack.force import (CALIBRATION_FILE, STANDARD_GRAVITY, ForceCalibration, ForceError, NiDaqForce,  # noqa: E402
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
    path = tmp_path / CALIBRATION_FILE
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
    assert float(peak_row["newtons"]) == pytest.approx(f["peak_n"], abs=0.01)
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
    hold = tr.force_summary(float(high.min()), float(high.max()))
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


def test_review_applies_the_saved_calibration_on_request_and_exports_what_was_used(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from limbtrack.review import ReviewTab
    app = QApplication.instance() or QApplication([])
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a[2]))
    eng, paths = uncalibrated_recording(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.force_cal_path = tmp_path / CALIBRATION_FILE
    tab.load(str(paths["meta.json"]))
    assert "volts only" in tab.stats.text() and "not calibrated" in tab.stats.text()
    tab.apply_force_calibration()                                              # no calibration file yet: explained, nothing changes
    assert infos and "no calibration" in infos[-1].lower() or "There is no calibration" in infos[-1]
    cal = ForceCalibration()
    cal.set_zero(0.0123)
    cal.n_per_v = 100.0
    cal.method = "zero and 5 kg"
    cal.save(tab.force_cal_path)
    tab.apply_force_calibration()
    assert "applied later" in tab.stats.text() and " N)" in tab.stats.text() or "Force in the window (N" in tab.stats.text()
    tab.export()
    summary = list(csv.DictReader(open(next(tmp_path.rglob("*_summary.csv")), newline="")))
    assert summary[0]["force_unit"] == "N" and float(summary[0]["force_n_per_v"]) == 100.0
    assert "applied later" in summary[0]["force_calibration_note"] and summary[0]["picture_delay_s"] == "0.0"
    saved_volts = [float(r["volts"]) for r in csv.DictReader(open(paths["force.csv"], newline=""))]
    assert max(saved_volts) > 3.5                                              # the file on disk still holds the volts, unchanged
