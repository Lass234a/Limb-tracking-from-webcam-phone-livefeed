"""From one measurement to the next (2026-10-19, from the P11 trials of 2026-10-15):

* the recording stops by itself a set time after the MVIC end marker (P11: recordings ran on 1.5 to 39 s after the end marker, one
  until the participant had walked away);
* Record asks before recording with a force that has not been zeroed (P11: 4 of 11 trials have volts but no newtons);
* the session log records what the operator did and when (so the next review can measure set-up time);
* the 'click dot' prompt on the picture is drawn in yellow on a darkened band (white text was hard to read on a pale wall)."""
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import overlay  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import SimulatedForce  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.session_log import SessionLog  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)


# ------------------------------------------------------------------------------------------ the engine's part of the stop
def marked_engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    return eng


def test_the_stop_is_due_a_set_time_after_the_end_marker_in_picture_time(tmp_path):
    eng = marked_engine(tmp_path)
    assert eng.auto_stop_s == 2.0 and eng.seconds_to_auto_stop() is None          # not recording
    paths = eng.start_trial("P01")
    for i in range(1, 10):
        eng.process(render(PTS, seed=i % 5), i / 30)
    assert eng.seconds_to_auto_stop() is None                                     # no end marker yet
    eng.add_event("MVIC end")
    eng.process(render(PTS, seed=1), 10 / 30)                                     # the marker is written on this picture
    assert eng.seconds_to_auto_stop() == pytest.approx(2.0)
    eng.process(render(PTS, seed=2), 10 / 30 + 1.5)
    assert eng.seconds_to_auto_stop() == pytest.approx(0.5)
    eng.process(render(PTS, seed=3), 10 / 30 + 2.0)
    assert eng.seconds_to_auto_stop() <= 0
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["stopped_automatically"]["after_mvic_end_s"] == 2.0
    assert "stopped_because" not in meta                                           # not a problem: Review shows no STOPPED line


def test_switched_off_or_stopped_by_hand_nothing_is_said_in_meta(tmp_path):
    eng = marked_engine(tmp_path)
    eng.auto_stop_s = 0.0
    paths = eng.start_trial("P01")
    eng.add_event("MVIC end")
    for i in range(1, 5):
        eng.process(render(PTS, seed=i), i * 1.0)
    assert eng.seconds_to_auto_stop() is None
    eng.stop_trial()
    assert "stopped_automatically" not in json.loads(paths["meta.json"].read_text())
    eng.auto_stop_s = 2.0
    paths = eng.start_trial("P01")                                                # a new recording forgets the last end marker
    eng.process(render(PTS, seed=1), 10.0)
    assert eng.seconds_to_auto_stop() is None
    eng.stop_trial()
    assert "stopped_automatically" not in json.loads(paths["meta.json"].read_text())


# ------------------------------------------------------------------------------------------ the window
@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    warnings, questions = [], []
    answer = {"yes": True}
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (questions.append(a[2]),
                                                                  QMessageBox.Yes if answer["yes"] else QMessageBox.No)[1])
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(180):                                                          # 6 s at 30 pictures per second
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 5 * math.sin(i / 10), ankle_df=5), seed=i % 7))
    w.release()
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)

    def pump(cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    def start_and_mark():
        live = win.live
        live.start_source(str(video))
        assert pump(lambda: win.engine._last_gray is not None)
        live.btn_pause.setChecked(True)                                           # hold the file on its first picture while marking
        live.toggle_pause()
        for name in win.engine.test.dots:
            live.view.clicked.emit(float(PTS[name][0]) + 2, float(PTS[name][1]) + 2)
        assert win.engine.next_dot_to_mark() is None
        live.participant.setText("P01")
        live.btn_lock.click()

    yield app, win, warnings, questions, answer, pump, start_and_mark, tmp_path
    win.close()


def log_rows(folder):
    with open(Path(folder) / "session_log.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_window_stops_the_recording_after_the_end_marker_and_says_so(setup):
    app, win, warnings, questions, answer, pump, start_and_mark, tmp = setup
    live = win.live
    start_and_mark()
    assert live.chk_auto_stop.isChecked() and live.auto_stop_spin.value() == 2.0
    live.btn_rec.click()                                                          # Record also starts the paused file
    assert win.engine.rec is not None
    assert pump(lambda: win.engine.frame_index > 20)
    paths = win.engine.rec.paths
    live.btn_end_mark.click()                                                     # E: the end marker
    assert pump(lambda: "Stopping by itself in" in live.rec_label.text(), timeout=5)
    assert pump(lambda: win.engine.rec is None, timeout=10)
    assert not live.btn_rec.isChecked() and live.btn_rec.text().startswith("Start recording")
    assert "Stopped by itself 2 s after the MVIC end" in live.instruction.text()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["status"] == "complete" and meta["stopped_automatically"]["after_mvic_end_s"] == 2.0
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    t_end = next(float(r["t_s"]) for r in rows if r["event"] == "MVIC end")
    assert 2.0 <= float(rows[-1]["t_s"]) - t_end < 2.6                            # 2 s of picture time, plus at most one status tick
    stops = [r for r in log_rows(tmp / "recordings") if r["event"] == "recording stopped"]
    assert stops and "by itself" in stops[-1]["detail"]


def test_unticked_the_recording_goes_on_and_the_setting_is_remembered(setup):
    app, win, warnings, questions, answer, pump, start_and_mark, tmp = setup
    live = win.live
    live.chk_auto_stop.setChecked(False)
    assert win.engine.auto_stop_s == 0 and not live.auto_stop_spin.isEnabled()
    start_and_mark()
    live.btn_rec.click()
    assert pump(lambda: win.engine.frame_index > 20)
    live.btn_end_mark.click()
    pump(lambda: False, timeout=3.0)
    assert win.engine.rec is not None and "Stopping by itself" not in live.rec_label.text()
    live.btn_rec.click()
    live.auto_stop_spin.setValue(4.5)
    assert live.session_state()["auto_stop"] == {"on": False, "after_s": 4.5}
    live.chk_auto_stop.setChecked(True)
    assert win.engine.auto_stop_s == 4.5
    win.live.save_session()
    again = MainWindow(load_protocol(ROOT / "protocol.json"), tmp)
    assert again.live.chk_auto_stop.isChecked() and again.live.auto_stop_spin.value() == 4.5 and again.engine.auto_stop_s == 4.5
    again.close()


def test_record_asks_before_recording_a_force_that_was_not_zeroed(setup):
    app, win, warnings, questions, answer, pump, start_and_mark, tmp = setup
    live = win.live
    start_and_mark()
    src = SimulatedForce()
    src.calibration.zero_set = False                                              # the load cell's slope, no zero yet (as after a start)
    src.start()
    try:
        win.engine.force = src
        live.force_source = src
        answer["yes"] = False
        live.btn_rec.click()
        assert win.engine.rec is None and not live.btn_rec.isChecked()
        assert questions and "not been zeroed" in questions[-1] and "no newtons" in questions[-1]
        assert live.accordion.current() == "force"                               # the Force step is opened, with the Zero button
        n = len(questions)
        src.calibration.zero_set = True
        live.btn_rec.click()                                                     # zeroed: no question about it (the file refuses the force)
        assert len(questions) == n and not any("zeroed" in q for q in questions[n:])
    finally:
        win.engine.force = None
        live.force_source = None
        src.stop()


def test_the_session_log_has_the_operators_actions_in_order(setup):
    app, win, warnings, questions, answer, pump, start_and_mark, tmp = setup
    live = win.live
    start_and_mark()
    live.view.clicked.emit(5.0, 5.0)                                              # far from any dot: nothing to re-mark
    live.btn_rec.click()
    assert pump(lambda: win.engine.frame_index > 10)
    live.btn_rec.click()
    events = [r["event"] for r in log_rows(tmp / "recordings")]
    assert events[0] == "video source started"
    assert events.count("dot marked") == len(win.engine.test.dots)
    assert events.index("position locked") < events.index("recording started") < events.index("recording stopped")
    rows = log_rows(tmp / "recordings")
    assert all(r["time"] and float(r["seconds_since_start"]) >= 0 for r in rows)
    started = next(r for r in rows if r["event"] == "recording started")
    assert started["detail"].startswith("P01 knee_extension")


def test_the_log_waits_for_the_folder_and_says_why_when_it_cannot_write(tmp_path):
    folder = tmp_path / "rec"
    folder.mkdir()
    (folder / "session_log.csv").mkdir()                                          # something that cannot be appended to
    log = SessionLog(folder)
    assert log.write("x") is False and "could not be written" in log.problem
    ok = SessionLog(tmp_path / "later")
    assert ok.write("a", "b") and ok.write("c") and ok.problem is None
    assert not (tmp_path / "later").exists()                                      # the log never makes the folder ...
    (tmp_path / "later").mkdir()                                                  # ... the first recording does
    assert ok.write("d")
    rows = list(csv.DictReader(open(ok.path, newline="", encoding="utf-8")))
    assert [r["event"] for r in rows] == ["a", "c", "d"] and rows[0]["detail"] == "b"


# ------------------------------------------------------------------------------------------ the prompt on the picture
def test_the_dot_prompt_is_yellow_on_a_darkened_band():
    frame = np.full((720, 1280, 3), 235, np.uint8)                                # a pale wall
    img = overlay.draw(frame, {}, [], [TESTS["knee_extension"].label, overlay.PROMPT_PREFIX + "greater trochanter"])
    band = img[40:80, 12:400]                                                     # the second line
    yellow = (band[..., 2] > 200) & (band[..., 1] > 170) & (band[..., 0] < 80)
    assert yellow.sum() > 200
    assert np.median(img[40:80, 12:400]) < 150                                    # the wall behind the text is darkened
    assert np.median(img[300:400, 600:800]) == 235                                # and nowhere else
    first = img[10:40, 12:300]
    assert ((first[..., 0] > 240) & (first[..., 1] > 240) & (first[..., 2] > 240)).sum() > 100     # other lines stay white
