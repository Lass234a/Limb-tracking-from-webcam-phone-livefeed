"""Automatic MVIC markers in a recording (engine, Review check lines, the Record box). The force is the simulated source: a made-up
contraction (rest, ramp, hold, release) on the same clock as the pictures, like the real DAQ."""
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.analysis import trial_checks  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import SimulatedForce  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


def make_engine(tmp_path, **sim):
    """Engine with the dots marked and a running simulated force: rest 4 s, ramp 0.6 s, hold 2.0 s, release 1 s from its start.
    Record is pressed 1.5 s after the force starts, so the ramp begins 1.5 to 2.5 s after Record (starting a recording takes a
    variable time): tests read the true onset from the saved force.csv instead of assuming it."""
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    force = SimulatedForce(**{"rest_s": 4.0, "ramp_s": 0.6, "hold_s": 2.0, "peak_n": 400.0, "period_s": 30.0, **sim})
    force.start()                                      # (after the dots are marked: the force clock starts here)
    eng.force = force
    time.sleep(1.5)                                    # 1.5 s of rest signal exists before Record is pressed
    return eng, force


def run_frames(eng, seconds, on_frame=None):
    n = int(seconds * 30)
    for i in range(n):
        eng.process(render(PTS, seed=i % 5), time.perf_counter())
        if on_frame:
            on_frame(i)
        time.sleep(1 / 30)


def true_onset(paths):
    """Seconds (after the first picture) at which the saved force first rose 10 N above rest: the real start of the ramp."""
    rows = list(csv.DictReader(open(paths["force.csv"], newline="")))
    t = [float(r["t_s"]) for r in rows]
    n = [float(r["newtons"]) for r in rows]
    return next(ti for ti, ni in zip(t, n) if ni > 10.0) - 10.0 / (400.0 / 0.6)      # (the ramp takes 15 ms to reach 10 N)


def events_of(paths):
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    return [(float(r["t_s"]), r["event"]) for r in rows if r["event"]]


def test_the_start_is_marked_when_the_ramp_begins_and_the_end_a_set_time_later(tmp_path):
    eng, force = make_engine(tmp_path)
    eng.auto_mvic, eng.auto_mvic_seconds = True, 2.5
    paths = eng.start_trial("P01")
    assert eng.auto_mvic_status()[0] == "waiting" and "waiting for the ramp" in eng.auto_mvic_status()[1]
    run_frames(eng, 6.5)
    eng.stop_trial()
    force.stop()
    ev = events_of(paths)
    onset = true_onset(paths)
    assert [e[1] for e in ev] == ["MVIC start", "MVIC end"]
    (t_start, _), (t_end, _) = ev
    assert 0.0 <= t_start - onset < 0.4                                # called within a fraction of a second of the real onset
    assert t_end - t_start == pytest.approx(2.5, abs=0.2)              # the end goes on the first picture after it is due
    meta = json.loads(paths["meta.json"].read_text())
    am = meta["auto_mvic"]
    assert am["state"] == "done" and am["unit"] == "N" and am["length_s"] == 2.5
    assert am["start_marker_t_s"] == pytest.approx(t_start, abs=0.05) and am["end_marker_t_s"] == pytest.approx(t_end, abs=0.05)
    assert am["force_onset_estimate_t_s"] == pytest.approx(t_start - am["start_marker_after_onset_s"], abs=0.01)
    assert am["force_onset_estimate_t_s"] == pytest.approx(onset, abs=0.06)       # its estimate of the onset agrees with the saved force
    assert 0.0 < am["start_marker_after_onset_s"] < 0.45
    assert meta["auto_mvic_setup"] == {"enabled": True, "threshold_n": 10.0, "length_s": 2.5}
    assert "changed_during_recording" not in meta
    # Review reads the markers like pressed ones, and says they were automatic
    tr = Trial(paths["meta.json"])
    hw = tr.hold_window()
    assert hw is not None and hw[0] == pytest.approx(t_start, abs=0.05) and hw[1] - hw[0] == pytest.approx(2.5, abs=0.2)
    checks = trial_checks(tr)
    assert any(lvl == "info" and "marked automatically" in text for lvl, text in checks), checks


def test_a_marker_pressed_by_hand_takes_over_from_the_automatic_one(tmp_path):
    eng, force = make_engine(tmp_path)
    eng.auto_mvic = True
    paths = eng.start_trial("P01")
    run_frames(eng, 0.4, on_frame=lambda i: eng.add_event("MVIC start") if i == 3 else None)     # pressed before the ramp
    assert eng.auto_mvic_status()[0] == "manual" and "by hand" in eng.auto_mvic_status()[1]
    run_frames(eng, 3.0)
    eng.add_event("MVIC end")
    run_frames(eng, 0.2)
    eng.stop_trial()
    force.stop()
    assert [e[1] for e in events_of(paths)] == ["MVIC start", "MVIC end"]          # exactly the two pressed ones, nothing automatic
    am = json.loads(paths["meta.json"].read_text())["auto_mvic"]
    assert am["state"] == "manual" and am["start_marker_t_s"] is None
    assert any(lvl == "info" and "pressed by hand" in text for lvl, text in trial_checks(Trial(paths["meta.json"])))


def test_when_the_force_never_rises_nothing_is_marked_and_the_check_list_says_so(tmp_path):
    eng, force = make_engine(tmp_path, rest_s=100.0)
    eng.auto_mvic = True
    paths = eng.start_trial("P01")
    run_frames(eng, 1.5)
    assert eng.auto_mvic_status()[0] == "waiting"
    eng.stop_trial()
    force.stop()
    assert events_of(paths) == []
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["auto_mvic"]["state"] == "waiting" and meta["auto_mvic"]["start_marker_t_s"] is None
    checks = trial_checks(Trial(paths["meta.json"]))
    assert any(lvl == "check" and "never rose" in text for lvl, text in checks), checks


def test_without_the_automatic_markers_nothing_changes(tmp_path):
    eng, force = make_engine(tmp_path)
    paths = eng.start_trial("P01")
    assert eng.auto_mvic_status() == ("off", "")
    run_frames(eng, 2.5)
    eng.stop_trial()
    force.stop()
    assert events_of(paths) == []
    meta = json.loads(paths["meta.json"].read_text())
    assert "auto_mvic" not in meta and meta["auto_mvic_setup"] == {"enabled": False}


def test_the_automatic_markers_need_the_force_and_cannot_change_during_a_recording(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    eng.auto_mvic = True
    with pytest.raises(RuntimeError, match="force"):
        eng.start_trial("P01")
    assert eng.rec is None
    eng.auto_mvic = False
    force = SimulatedForce(period_s=30.0)
    force.start()
    eng.force = force
    time.sleep(0.2)
    eng.auto_mvic = True
    eng.start_trial("P01")
    for attr, value in (("auto_mvic", False), ("auto_mvic_threshold", 50.0), ("auto_mvic_seconds", 3.0)):
        with pytest.raises(RuntimeError, match="Stop recording"):
            setattr(eng, attr, value)
    eng.stop_trial()
    force.stop()
    eng.auto_mvic_seconds = 3.0                          # allowed again afterwards
    assert eng.auto_mvic_seconds == 3.0


def test_the_picture_header_shows_what_the_automatic_markers_are_doing(tmp_path):
    eng, force = make_engine(tmp_path)
    eng.auto_mvic, eng.auto_mvic_seconds = True, 2.0
    eng.start_trial("P01")
    run_frames(eng, 0.3)
    res = eng.process(render(PTS, seed=1), time.perf_counter())
    assert any("AUTO MVIC: armed" in line for line in res.header), res.header           # before the ramp
    run_frames(eng, 4.8)
    res = eng.process(render(PTS, seed=1), time.perf_counter())
    eng.stop_trial()
    force.stop()
    assert any("AUTO MVIC: start and end marked" in line for line in res.header), res.header      # after the 2 s MVIC


# ------------------------------------------------------------------------------------------ the Record box
@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    yield app, win, win.live
    win.close()


def test_the_record_box_offers_the_automatic_markers_only_with_the_force_connected(window):
    app, win, live = window
    assert not live.chk_auto_mvic.isEnabled() and not live.auto_thr_spin.isEnabled()
    live.force_kind.setCurrentIndex(live.force_kind.findData("simulated"))
    live.btn_force.click()
    live.update_force_box()
    assert live.chk_auto_mvic.isEnabled() and live.auto_thr_spin.isEnabled() and live.auto_len_spin.isEnabled()
    live.chk_auto_mvic.setChecked(True)
    live.auto_thr_spin.setValue(20.0)
    live.auto_len_spin.setValue(6.5)
    assert win.engine.auto_mvic is True and win.engine.auto_mvic_threshold == 20.0 and win.engine.auto_mvic_seconds == 6.5
    live.btn_force.click()                                             # disconnecting the force switches the automatic markers off
    live.update_force_box()
    assert win.engine.auto_mvic is False and not live.chk_auto_mvic.isChecked() and not live.chk_auto_mvic.isEnabled()
