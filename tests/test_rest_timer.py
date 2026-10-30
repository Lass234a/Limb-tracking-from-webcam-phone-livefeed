"""0.27.0: the rest between contractions, timed by the program (SOP v0.18: 60 s between the trials of a position, 120 s before the next
position and between belt squat trials, 30 s after the submaximal contraction). Until now a stopwatch; in the pilot sessions 33 of 89 rests
between two trials of one position were shorter than 60 s."""
import json
import math
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.rest import (REST_AFTER_SUBMAX_S, REST_NEXT_POSITION_S, REST_SAME_POSITION_S, RestTimer,  # noqa: E402
                            clock_text)
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

from tests.test_reminders import P0, TESTS  # noqa: E402

A, B = ("P01", "knee_extension", 70.0), ("P01", "knee_extension", 20.0)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ------------------------------------------------------------------------------------------------ the rules
def test_the_sop_times():
    assert (REST_SAME_POSITION_S, REST_NEXT_POSITION_S, REST_AFTER_SUBMAX_S) == (60.0, 120.0, 30.0)
    assert [clock_text(s) for s in (0, 9.9, 61.9, 725, -3)] == ["0:00", "0:09", "1:01", "12:05", "0:00"]


def test_between_two_trials_of_one_position_60_s():
    c = Clock()
    r = RestTimer(clock=c)
    assert r.status(A) is None and r.meta(A) is None and r.reminder(A) == ""
    r.start("trial", A, "knee_extension")
    c.t += 42
    st = r.status(A)
    assert st["required"] == 60 and st["short"] and st["text"] == "REST 0:42 of 1:00 (before the next trial at this position)"
    assert r.reminder(A) == "Rest: only 0:42 since the last trial (the SOP asks for 1:00 before the next trial at this position)."
    m = r.meta(A)
    assert m["seconds"] == 42.0 and m["required_s"] == 60 and m["after"] == "trial" and m["short"] is True
    c.t += 18
    assert r.status(A)["done"] and r.status(A)["text"].startswith("Rest 1:00: done") and r.reminder(A) == ""


def test_before_the_next_position_and_in_the_belt_squat_120_s():
    c = Clock()
    r = RestTimer(clock=c)
    r.start("trial", A, "knee_extension")
    c.t += 90
    assert r.status(A)["done"]                                                 # the same position: 60 s are over ...
    assert not r.status(B)["done"] and r.status(B)["required"] == 120          # ... another one is chosen: 120 s
    assert r.status(B)["why"] == "before the next position"
    assert r.status(A, position_done=True)["required"] == 120                 # the 10 % check needs no further trial here
    r.start("trial", ("P01", "belt_squat", 90.0), "belt_squat")
    c.t += 90
    st = r.status(("P01", "belt_squat", 90.0))
    assert st["required"] == 120 and st["why"] == "belt squat" and not st["done"]


def test_after_the_submaximal_contraction_30_s():
    c = Clock()
    r = RestTimer(clock=c)
    r.start("submax")
    c.t += 12
    assert r.status(A)["text"] == "REST 0:12 of 0:30 (after the submaximal contraction)"
    assert "since the submaximal contraction" in r.reminder(A) and r.meta(A)["after"] == "submaximal contraction"
    c.t += 18
    assert r.status(B)["done"]                                                 # whatever position is chosen


def test_the_beep_comes_once_and_again_when_the_rest_became_longer():
    c = Clock()
    r = RestTimer(clock=c)
    r.start("trial", A, "knee_extension")
    c.t += 59
    assert not r.beep_due(A)
    c.t += 1
    assert r.beep_due(A) and not r.beep_due(A)                                 # once
    c.t += 10
    assert not r.beep_due(B)                                                   # the next position was chosen: 120 s, not over yet
    c.t += 50
    assert r.beep_due(B) and not r.beep_due(B)                                 # 120 s are over: once more
    r.stop()
    assert not r.running and r.status(A) is None and not r.beep_due(A)


# ------------------------------------------------------------------------------------------------ in the window
@pytest.fixture
def real(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(300):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 10 * math.sin(math.pi * i / 150), ankle_df=5), seed=i % 5))
    w.release()
    window = MainWindow(TESTS, tmp_path)

    def pump(cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    live = window.live
    live.start_source(str(video))
    assert pump(lambda: window.engine._last_gray is not None)
    for name in window.engine.test.dots:
        live.view.clicked.emit(float(P0[name][0]) + 2, float(P0[name][1]) + 2)
    live.participant.setText("P01")
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live.chk_priv_ok.setChecked(True)
    clock = Clock()
    live.rest = RestTimer(clock=clock)
    beeps = []
    monkeypatch.setattr(window.engine, "rest_beep", lambda: beeps.append(clock.t))
    yield window, live, pump, tmp_path, clock, beeps
    window.close()


def record(window, live, pump):
    live.btn_rec.click()
    assert window.engine.rec is not None
    assert pump(lambda: window.engine.rec.frames > 5)
    assert not live.rest_label.isVisible() or live.rest_label.isHidden()        # no rest line while recording
    live.btn_rec.click()
    assert window.engine.rec is None


def test_a_rest_starts_when_a_recording_stops_and_is_saved_with_the_next_trial(real):
    window, live, pump, tmp, clock, beeps = real
    assert not live.rest.running and not live.rest_label.isVisibleTo(live)    # nothing before the first trial
    record(window, live, pump)
    assert live.rest.running and live.rest.kind == "trial"
    clock.t += 20
    live._update_rest()
    assert live.rest_label.isVisibleTo(live) and live.rest_label.text() == "REST 0:20 of 1:00 (before the next trial at this position)"
    assert any("Rest: only 0:20 since the last trial" in t for t, _ in live._open_reminders())
    time.sleep(1.1)                                                              # (a new file name: they carry the second)
    record(window, live, pump)                                                   # the reminder never refuses Record
    metas = sorted((tmp / "recordings").rglob("*_meta.json"))
    first, second = (json.loads(p.read_text()) for p in metas)
    assert first["rest_before"] is None                                        # the first trial of the run: no rest was timed
    rb = second["rest_before"]
    assert rb["seconds"] == 20.0 and rb["required_s"] == 60 and rb["short"] is True and rb["after"] == "trial"
    assert any(t.startswith("Rest: only 0:20") for t in second["reminders_open_at_start"])
    assert any("Recorded while this was open: Rest: only 0:20" in w for w in live._last_trial_warnings)
    tr = Trial(metas[1])                                                         # Review and the export
    assert any("Only 0:20 of rest before this trial (the SOP asks for 1:00" in text for lvl, text in analysis.trial_checks(tr) if lvl == "check")
    assert analysis.session_row(tr)["rest_before_s"] == 20.0 and "rest_before_s" in analysis.SESSION_COLUMNS
    tr.cap.release()


def test_the_rest_line_turns_green_and_beeps_once(real):
    window, live, pump, tmp, clock, beeps = real
    record(window, live, pump)
    clock.t += 59.5
    live._update_rest()
    assert not beeps and "#b45309" in live.rest_label.styleSheet()
    clock.t += 1
    live._update_rest()
    live._update_rest()
    assert len(beeps) == 1 and "done" in live.rest_label.text() and "#16a34a" in live.rest_label.styleSheet()
    assert "rest over" in (tmp / "recordings" / "session_log.csv").read_text(encoding="utf-8")
    live.chk_rest_beep.setChecked(False)                                       # switched off: the line, no beep
    live.target_spin.setValue(live.target_spin.value() - 30)                   # another position: 120 s
    live._update_rest()
    assert "of 2:00 (before the next position)" in live.rest_label.text()
    clock.t += 61
    live._update_rest()
    assert len(beeps) == 1 and "done" in live.rest_label.text()


def test_the_submax_button_starts_30_s_and_t_is_its_key(real):
    window, live, pump, tmp, clock, beeps = real
    keys = {sc.key().toString(): sc for sc in live.findChildren(QShortcut)}
    assert QKeySequence("T").toString() in keys
    live.btn_rest_submax.click()
    clock.t += 5
    live._update_rest()
    assert live.rest.kind == "submax" and live.rest_label.text() == "REST 0:05 of 0:30 (after the submaximal contraction)"
    log = tmp / "recordings" / "session_log.csv"                               # (kept in memory until the first recording makes the folder)
    assert "after the submaximal contraction" in (log.read_text(encoding="utf-8") if log.exists() else repr(live.log._waiting))


def test_the_beep_switch_is_remembered(real):
    window, live, pump, tmp, clock, beeps = real
    assert live.chk_rest_beep.isChecked()                                      # on by default
    live.chk_rest_beep.setChecked(False)
    state = live.session_state()
    assert state["rest"] == {"beep": False}
    live.chk_rest_beep.setChecked(True)
    live.restore_session(state)
    assert not live.chk_rest_beep.isChecked()


def test_the_rest_beep_is_a_cue_and_never_plays_during_a_recording(tmp_path):
    from limbtrack.cues import CUES, CuePlayer, duration_s
    from limbtrack.engine import Engine
    assert "rest" in CUES and duration_s("rest") < 0.6
    eng = Engine(TESTS, recordings_dir=tmp_path)
    heard = []
    eng._cues = CuePlayer(backend=heard.append)
    eng.rest_beep()
    time.sleep(0.3)
    assert heard == ["rest"]
    eng.rec = object()
    eng.rest_beep()
    time.sleep(0.3)
    assert heard == ["rest"]
    eng.rec = None
