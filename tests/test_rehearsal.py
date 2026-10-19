"""Trying the automatic MVIC markers and the beeps WITHOUT a recording ("Try it without recording"): the force is read exactly as in a
recording, the start is called, the beeps follow, the end is announced; nothing is saved and no empty trial is made.
The force is the simulated source (rest, ramp, hold, release on the same clock as the pictures); the sound device is a list of names."""
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.cues import CuePlayer  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import SimulatedForce  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


def make_engine(tmp_path, **sim):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.auto_mvic_seconds = 3.5                                  # shorter than the protocol's 7 s (the test stays short), long enough for a hold cue
    heard = []
    eng._cues = CuePlayer(backend=heard.append)
    force = SimulatedForce(**{"rest_s": 4.0, "ramp_s": 0.6, "hold_s": 2.0, "peak_n": 400.0, "period_s": 30.0, **sim})
    force.start()
    eng.force = force
    time.sleep(1.5)                                              # some rest signal before the try-out begins
    return eng, force, heard


def run_until(eng, states, timeout=20.0):
    end, last = time.time() + timeout, ("off", "")
    while time.time() < end:
        last = eng.rehearsal_step()
        if last[0] in states:
            return last
        time.sleep(0.05)
    return last


def test_it_needs_the_force_and_refuses_during_a_recording(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    assert "force" in eng.rehearsal_start().lower()
    assert eng.rehearsal_step() == ("off", "")


def test_a_push_is_found_the_beeps_follow_and_nothing_is_saved(tmp_path):
    eng, force, heard = make_engine(tmp_path)
    try:
        assert eng.rehearsal_start() is None
        state, text = eng.rehearsal_step()
        assert state == "waiting" and ("push" in text.lower() or "ready" in text.lower())
        state, text = run_until(eng, ("done", "gave up"))
        assert state == "done", text
        assert "Start found" in text and "both markers would be written" in text and "Beeps played" in text
        time.sleep(0.6)                                          # (the last beep is played on its own thread)
        assert heard[0] == "ramp" and heard[-1] == "end" and "hold" in heard and "relax" in heard       # the whole 1-5-1 order
        assert eng.rec is None and list(tmp_path.iterdir()) == []   # no trial, no folder, no file
        assert eng.rehearsal_step() == ("off", "")                # it ended by itself
    finally:
        force.stop()
        eng._cues.close()


def test_with_the_beeps_off_it_still_finds_the_push_but_says_there_were_no_beeps(tmp_path):
    eng, force, heard = make_engine(tmp_path)
    eng.audio_cues = False
    try:
        assert eng.rehearsal_start() is None
        state, text = run_until(eng, ("done", "gave up"))
        assert state == "done" and "Beeps are switched off" in text and heard == []
    finally:
        force.stop()
        eng._cues.close()


def test_no_push_for_a_minute_gives_up_with_a_plain_message(tmp_path):
    eng, force, heard = make_engine(tmp_path, rest_s=600.0)
    eng.REHEARSAL_GIVE_UP_S = 2.0
    try:
        assert eng.rehearsal_start() is None
        state, text = run_until(eng, ("done", "gave up"), timeout=10.0)
        assert state == "gave up" and "No push" in text and heard == []
    finally:
        force.stop()
        eng._cues.close()


def test_cancelling_stops_it_and_silences_the_beeps(tmp_path):
    eng, force, heard = make_engine(tmp_path)
    try:
        assert eng.rehearsal_start() is None
        run_until(eng, ("running",))
        eng.rehearsal_stop()
        assert eng.rehearsal_step() == ("off", "")
        n = len(heard)
        time.sleep(1.5)
        assert len(heard) == n or heard[-1] != "end"             # the cues that were still waiting were dropped
    finally:
        force.stop()
        eng._cues.close()


def test_starting_a_recording_ends_the_try_out(tmp_path):
    eng, force, heard = make_engine(tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    try:
        assert eng.rehearsal_start() is None
        eng.start_trial("P01")
        assert eng.rehearsal_step() == ("off", "")
        assert "Stop recording" in eng.rehearsal_start()
        eng.stop_trial()
    finally:
        force.stop()
        eng._cues.close()


def test_the_button_in_the_window_runs_it_and_reports(tmp_path):
    QApplication.instance() or QApplication([])
    win = MainWindow(TESTS, tmp_path)
    live = win.live
    try:
        assert not live.btn_rehearse.isEnabled() and not live.rehearse_label.isVisibleTo(live)   # no force: nothing to try
        win.engine.auto_mvic_seconds = 3.5
        live._pick(live.force_kind, "simulated")
        live.btn_force.click()
        live.force_source.stop()                                  # replace the default simulated contraction with a known one
        force = SimulatedForce(rest_s=4.0, ramp_s=0.6, hold_s=2.0, peak_n=400.0, period_s=30.0)
        force.start()
        live.force_source = win.engine.force = force
        time.sleep(1.5)
        live.update_force_box()
        assert live.btn_rehearse.isEnabled()
        live.btn_rehearse.click()
        assert live.btn_rehearse.text() == "Cancel the try-out" and live.rehearse_label.text()
        end = time.time() + 20
        while time.time() < end and live._reh_timer.isActive():
            QApplication.processEvents()
            time.sleep(0.02)
        assert not live._reh_timer.isActive() and live.btn_rehearse.text() == "Try it without recording"
        assert "Start found" in live.rehearse_label.text()
        live.btn_rehearse.click()                                 # again, and cancel at once
        assert live._reh_timer.isActive()
        live.btn_rehearse.click()
        assert not live._reh_timer.isActive() and "cancelled" in live.rehearse_label.text().lower()
    finally:
        live.worker = None
        win.close()
