"""Privacy at every change of position (0.23.0): the participant has moved, so the picture is looked at again before the next recording.
The 'I looked' tick is cleared, the step list asks again, recording is refused until it has been looked at, and every trial says whether
privacy was checked for ITS position (meta.json privacy.checked_for_this_position)."""
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.gui import STEP_HINT, MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)


class RunningCamera:
    is_file = False
    paused = False
    at_end = False
    source = 0
    camera = None
    status_note = ""

    def stop(self):
        pass

    def wait(self, ms=0):
        return True

    def isRunning(self):
        return True


@pytest.fixture
def win(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    w.live.worker = RunningCamera()
    w.engine.process(render(P0, seed=0), 0.0)
    w.live.update_steps()
    yield w
    w.live.worker = None
    w.close()


def state(live):
    return live.strip.states["privacy"][0]


def another_target(live):
    t = live.target_spin.value()
    live.target_spin.setValue(t + 10 if t < 100 else t - 10)


def other_test(live):
    return next(k for k in live.engine.tests if k != live.engine.test.key)


# ------------------------------------------------------------------------------------------------ the window
def test_a_checked_picture_stays_checked_while_the_position_stays(win):
    live = win.live
    live.chk_priv_ok.setChecked(True)
    live.target_spin.setValue(live.target_spin.value())                              # the same position again: nothing changes
    live.update_steps()
    assert not live._priv_recheck and state(live) == "done" and live.chk_priv_ok.isChecked()
    assert live.btn_priv_again.isHidden()


def test_a_new_target_clears_the_look_and_asks_again(win):
    live = win.live
    live.chk_priv_ok.setChecked(True)
    live.update_steps()
    assert state(live) == "done"
    another_target(live)
    live.update_steps()
    assert live._priv_recheck and not live.chk_priv_ok.isChecked()
    assert state(live) in ("todo", "next") and "recheck" in live.strip.chips["privacy"].text()
    assert "CHECK AGAIN" in live.accordion.sections["privacy"].summary and not live.btn_priv_again.isHidden()
    assert "position changed" in live._hint("privacy") and STEP_HINT["privacy"] not in live._hint("privacy")
    assert len(live.btn_priv_again.text()) <= 42                                      # (the panel's label rule)


def test_a_new_test_asks_again_too(win):
    live = win.live
    live.chk_priv_ok.setChecked(True)
    live._pick(live.test_combo, other_test(live))
    live.update_steps()
    assert live._priv_recheck and state(live) in ("todo", "next")


def test_a_crop_that_was_drawn_does_not_count_for_the_next_position(win):
    live = win.live
    ok, _ = win.engine.set_privacy_rect("crop", 100, 100, 900, 600)
    assert ok
    live.update_steps()
    assert state(live) == "done"
    another_target(live)
    live.update_steps()
    assert win.engine.privacy_crop and live._priv_recheck and state(live) != "done"        # the crop is still there; it has to be looked at


def test_the_button_confirms_with_a_crop_and_is_the_same_as_the_tick_without_one(win):
    live = win.live
    win.engine.set_privacy_rect("crop", 100, 100, 900, 600)
    another_target(live)
    live.btn_priv_again.click()
    live.update_steps()
    assert not live._priv_recheck and state(live) == "done" and live.btn_priv_again.isHidden()
    win.engine.clear_privacy()
    another_target(live)
    live.btn_priv_again.click()                                                         # no crop, no mask: it ticks 'I looked'
    live.update_steps()
    assert live.chk_priv_ok.isChecked() and not live._priv_recheck and state(live) == "done"


def test_ticking_or_drawing_also_confirms(win):
    live = win.live
    another_target(live)
    live.chk_priv_ok.setChecked(True)
    assert not live._priv_recheck
    another_target(live)
    assert live._priv_recheck
    live.start_privacy_drawing("mask")
    live.view.rect_drawn.emit(0, 0, 200, 200)
    assert not live._priv_recheck and win.engine.privacy_masks


def test_clearing_the_crop_does_not_count_as_looking(win):
    live = win.live
    win.engine.set_privacy_rect("crop", 100, 100, 900, 600)
    another_target(live)
    live.btn_priv_clear.click()
    assert live._priv_recheck


def test_a_position_change_that_the_engine_refuses_asks_nothing(win):
    live, e = win.live, win.engine
    live.chk_priv_ok.setChecked(True)
    e.rec = object()
    try:
        another_target(live)
    finally:
        e.rec = None
    live.update_steps()
    assert not live._priv_recheck and live.chk_priv_ok.isChecked()                       # (nothing moved: the look still stands)


def test_recording_is_refused_until_the_picture_has_been_looked_at(win, monkeypatch):
    live = win.live
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    live.participant.setText("P01")
    live.chk_priv_ok.setChecked(True)
    another_target(live)
    live.btn_rec.click()
    assert warnings and "position changed" in warnings[-1] and "Privacy checked at this position" in warnings[-1]
    assert win.engine.rec is None and not live.btn_rec.isChecked()
    assert live.accordion.current() == "privacy"                                         # the button to press is in front of you


def test_a_restored_session_asks_nothing(win):
    live = win.live
    live.chk_priv_ok.setChecked(True)
    live._restoring = True
    try:
        another_target(live)
    finally:
        live._restoring = False
    assert not live._priv_recheck


def test_the_session_log_says_when_it_was_asked_and_answered(win):
    live = win.live
    notes = []
    live.note = lambda event, detail="": notes.append(event)
    another_target(live)
    live.btn_priv_again.click()
    assert "privacy to be checked again" in notes and "privacy checked again" in notes


# ------------------------------------------------------------------------------------------------ the engine's record of it
def test_the_engine_says_for_which_position_privacy_was_looked_at(win):
    e = win.engine
    assert e._privacy_meta()["checked"] is None and e._privacy_meta()["checked_for_this_position"] is False
    e.note_privacy_check("looked: nothing identifies the participant")
    m = e._privacy_meta()
    assert m["checked_for_this_position"] is True and m["checked"]["test"] == e.test.key and m["checked"]["target"] == e.target
    assert m["checked"]["how"].startswith("looked") and m["checked"]["taken"]
    e.target = (e.target or 0) + 10
    assert e._privacy_meta()["checked_for_this_position"] is False                       # another position: the old look does not count


# ------------------------------------------------------------------------------------------------ a real recording
@pytest.fixture
def real(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(150):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 10 * math.sin(math.pi * i / 150), ankle_df=5), seed=i))
    w.release()
    window = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)

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
    assert window.engine.next_dot_to_mark() is None
    live.btn_lock.click()
    live.participant.setText("P01")
    yield window, live, warnings, pump, tmp_path
    window.close()


def test_a_trial_says_that_privacy_was_checked_for_its_position(real):
    window, live, warnings, pump, tmp = real
    live.chk_priv_ok.setChecked(True)
    live.btn_rec.click()
    assert window.engine.rec is not None, warnings
    assert pump(lambda: window.engine.rec.frames > 5)
    live.btn_rec.click()
    meta = json.loads(next((tmp / "recordings").rglob("*_meta.json")).read_text())
    assert meta["privacy"]["checked_for_this_position"] is True and meta["privacy"]["checked"]["how"].startswith("looked")


def test_a_trial_that_was_never_looked_at_says_so(real):
    window, live, warnings, pump, tmp = real
    live.btn_rec.click()                                                                  # (the first position: not blocked, but not looked at)
    assert window.engine.rec is not None, warnings
    assert pump(lambda: window.engine.rec.frames > 5)
    live.btn_rec.click()
    meta = json.loads(next((tmp / "recordings").rglob("*_meta.json")).read_text())
    assert meta["privacy"]["checked"] is None and meta["privacy"]["checked_for_this_position"] is False
