"""0.20.0: the tested leg belongs to the participant code, is saved in every trial, and Record asks for it when it is not chosen.

Until 0.19.0 it was one setting per computer (frontal tests only) and carried over to the next participant: P20 (2026-10-22) had the other
leg set in all five dotted frontal trials."""
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import session_memory  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import LiveTab, MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

from tests.test_frontal_engine import engine as frontal_engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
REAL_ASK = LiveTab._ask_tested_leg               # taken before tests/conftest.py answers the question for the other window tests


# ------------------------------------------------------------------------------------------------ engine
def test_the_leg_belongs_to_the_participant_code(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.participant_code = "P01"
    assert eng.tested_leg is None                                    # nothing assumed
    eng.tested_leg = "left"
    eng.participant_code = "P02"
    assert eng.tested_leg is None                                    # the next participant does not get P01's leg
    eng.tested_leg = "right"
    eng.participant_code = "P0"                                      # a typing slip on the way back ...
    eng.participant_code = "P01"
    assert eng.tested_leg == "left"                                  # ... and P01's leg is there again
    assert eng.tested_legs == {"P01": "left", "P02": "right"}
    with pytest.raises(ValueError):
        eng.tested_leg = "both"


def test_every_trial_saves_the_leg_and_it_cannot_change_while_recording(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    q = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(q, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(q[n] + 2))
    eng.participant_code = "P03"
    eng.tested_leg = "left"
    paths = eng.start_trial("P03")
    eng.participant_code = "P04"                                     # the next code typed during the recording
    assert eng.tested_leg == "left"                                  # the trial keeps its own leg
    with pytest.raises(RuntimeError):
        eng.tested_leg = "right"
    for i in range(5):
        eng.process(render(q, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["view"] == "sagittal" and meta["tested_leg"] == "left"   # a side-view trial too (frontal only before 0.20.0)
    assert eng.tested_leg is None                                    # P04 has none yet


def test_a_frontal_test_without_a_leg_cannot_be_locked(tmp_path):
    eng, p = frontal_engine(tmp_path, "hip_abduction", 10.0)
    eng.participant_code = "P09"                                     # (the helper chose the leg before any code)
    from tests.test_frontal_engine import run
    run(eng, p)
    assert any("choose the tested leg" in t for t in eng.marking_problems())
    assert not eng.lock_position() and "choose the tested leg" in eng.lock_problem().lower()
    assert "leg_not_chosen" in eng.label_check_texts()
    eng.tested_leg = "right"
    run(eng, p)
    assert eng.marking_problems() == [] and eng.lock_position()


# ------------------------------------------------------------------------------------------------ window
@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(90):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 15 * math.sin(math.pi * i / 90), ankle_df=5), seed=i))
    w.release()
    win = MainWindow(TESTS, tmp_path)

    def pump(cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False
    yield app, win, video, pump
    win.close()


def answer_with(monkeypatch, label):
    """Make the real question answer itself: click the button with this text (or Cancel)."""
    def exec_(box):
        for b in box.buttons():
            if b.text() == label or (label == "Cancel" and box.buttonRole(b) == QMessageBox.RejectRole):
                b.click()
                return 0
        raise AssertionError(f"no button {label!r}: {[b.text() for b in box.buttons()]}")
    monkeypatch.setattr(QMessageBox, "exec", exec_)
    monkeypatch.setattr(LiveTab, "_ask_tested_leg", REAL_ASK)


def test_the_question_sets_the_leg_or_opens_the_participant_section(setup, monkeypatch):
    app, win, video, pump = setup
    live = win.live
    live.participant.setText("P01")
    answer_with(monkeypatch, "Left leg")
    assert live._ask_tested_leg() and win.engine.tested_leg == "left" and live.leg_combo.currentData() == "left"
    live.participant.setText("P02")
    live.update_steps()
    assert live.leg_combo.currentData() is None and live.strip.chips["participant"].text().endswith("P02: leg?")
    answer_with(monkeypatch, "Cancel")
    assert not live._ask_tested_leg() and win.engine.tested_leg is None and live.accordion.current() == "participant"


def test_record_asks_for_the_leg_and_the_trial_saves_it(setup, monkeypatch):
    app, win, video, pump = setup
    live = win.live
    live.start_source(str(video))
    assert pump(lambda: win.engine._last_gray is not None)
    p0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)
    for name in win.engine.test.dots:
        live.view.clicked.emit(float(p0[name][0]) + 2, float(p0[name][1]) + 2)
    assert pump(lambda: win.engine.frame_index > 0)
    live.participant.setText("P01")
    live.btn_lock.click()
    answer_with(monkeypatch, "Cancel")
    live.btn_rec.click()
    assert win.engine.rec is None and not live.btn_rec.isChecked()            # no leg, no recording
    answer_with(monkeypatch, "Right leg")
    live.btn_rec.click()
    assert win.engine.rec is not None
    assert pump(lambda: win.engine.rec is not None and win.engine.rec.frames > 5)
    live.btn_rec.click()
    metas = sorted((live.recordings_dir / "P01").glob("*_meta.json"))
    assert metas and json.loads(metas[-1].read_text())["tested_leg"] == "right"
    assert live.session_state()["tested_legs"] == {"P01": "right"}


def test_the_legs_are_remembered_with_a_recent_session_only(setup):
    app, win, video, pump = setup
    live = win.live
    recent = {"participant": "P07", "tested_legs": {"P07": "left"}, "saved_at": time.time() - 60, "format": session_memory.FORMAT}
    live.restore_session(recent)
    assert live.participant.text() == "P07" and win.engine.tested_leg == "left" and live.leg_combo.currentData() == "left"
    old = {"participant": "P08", "tested_legs": {"P08": "right"}, "saved_at": time.time() - 3 * 24 * 3600, "format": session_memory.FORMAT}
    win.engine.tested_legs = {}
    live.restore_session(old)
    assert win.engine.tested_legs == {}                                       # a new day starts blank, like the code
    legacy = {"participant": "P09", "tested_leg": "right", "saved_at": time.time() - 60, "format": session_memory.FORMAT}
    live.restore_session(legacy)
    assert win.engine.tested_leg is None                                      # one leg for the whole computer is not taken over
