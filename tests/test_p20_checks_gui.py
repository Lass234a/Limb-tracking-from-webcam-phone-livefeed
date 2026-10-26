"""0.19.0, the window side of the P20 review: keys act once per press, Record asks when the main angle cannot be measured, and a trial with
a problem (here CHECK LABELS) is called out when it is saved, without a box that blocks the next trial."""
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    asked = []
    answer = {"value": QMessageBox.Yes}
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append((a[1], a[2])), answer["value"])[1])
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(150):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 3 * math.sin(math.pi * i / 45), ankle_df=5), seed=i % 7))
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

    def start_and_mark(swap=()):
        live = win.live
        live.start_source(str(video))
        assert pump(lambda: win.engine._last_gray is not None)
        p0 = dict(leg_points(hip_flex=80, knee_flex=20, ankle_df=5))
        for a, b in swap:
            p0[a], p0[b] = p0[b], p0[a]
        for name in win.engine.test.dots:
            live.view.clicked.emit(float(p0[name][0]) + 2, float(p0[name][1]) + 2)
        assert win.engine.next_dot_to_mark() is None
        assert pump(lambda: win.engine.frame_index > 0)
        live.participant.setText("P01")

    yield app, win, asked, answer, pump, start_and_mark
    win.close()


def test_only_left_and_right_repeat_while_held(setup):
    app, win, *_ = setup
    shortcuts = win.live.findChildren(QShortcut)
    keys = {sc.key().toString(): sc.autoRepeat() for sc in shortcuts}
    assert keys[QKeySequence(Qt.Key_Left).toString()] and keys[QKeySequence(Qt.Key_Right).toString()]
    for name in ("Z", "R", "Space", "D", "N", "B", "1", "9", "F5", "F9"):
        assert keys[QKeySequence(name).toString()] is False, name


def test_record_asks_when_the_main_angle_cannot_be_measured(setup, monkeypatch):
    app, win, asked, answer, pump, start_and_mark = setup
    start_and_mark()
    live = win.live
    live.btn_lock.click()
    monkeypatch.setattr(win.engine, "main_angle_problem", lambda: "The Knee flexion cannot be measured right now: the lateral malleolus dot is not visible.")
    answer["value"] = QMessageBox.No
    live.btn_rec.click()
    assert win.engine.rec is None and not live.btn_rec.isChecked()
    assert asked and asked[-1][0] == "Main angle" and "Record anyway?" in asked[-1][1]
    answer["value"] = QMessageBox.Yes
    live.btn_rec.click()
    assert win.engine.rec is not None
    live.btn_rec.click()


def test_no_question_when_the_main_angle_is_fine(setup):
    app, win, asked, answer, pump, start_and_mark = setup
    start_and_mark()
    live = win.live
    live.btn_lock.click()
    assert win.engine.locked and win.engine.main_angle_problem() is None
    live.btn_rec.click()
    assert win.engine.rec is not None and not any(title == "Main angle" for title, _ in asked)
    live.btn_rec.click()


def test_a_trial_with_swapped_knee_dots_is_called_out_when_saved(setup):
    app, win, asked, answer, pump, start_and_mark = setup
    start_and_mark(swap=(("epicondyle", "fibular_head"),))
    live = win.live
    assert pump(lambda: bool(win.engine.marking_problems()))
    live.btn_rec.click()                                               # not locked (Lock is refused): 'Record anyway?' -> yes
    assert win.engine.rec is not None
    assert pump(lambda: win.engine.rec is not None and win.engine.rec.frames > 10)
    live.btn_rec.click()
    assert win.engine.rec is None
    assert "CHECK THIS TRIAL" in live.rec_label.text() and "CHECK LABELS" in live.rec_label.text()
    box = live._trial_box
    assert box is not None and box.isVisible() and not box.isModal() and "CHECK LABELS" in box.text()
    log = (win.live.recordings_dir / "session_log.csv").read_text(encoding="utf-8")
    assert "trial warning" in log
    live.btn_rec.click()                                               # the box does not block the next trial
    assert win.engine.rec is not None
    live.btn_rec.click()
