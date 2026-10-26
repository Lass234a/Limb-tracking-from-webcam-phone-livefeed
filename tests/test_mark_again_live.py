"""The pinned bar and Redo all against a real video playing in the window: the picture is processed on the video thread while the operator
works the keys and clicks (the other tests call everything from the one test thread), and a real recording (not a stand-in) is running."""
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "demo.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(240):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 10 * math.sin(math.pi * i / 150), ankle_df=5), seed=i))
    w.release()
    return path


@pytest.fixture
def running(tmp_path, video, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    win.show()
    win.activateWindow()
    live = win.live

    def pump(cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    live.start_source(str(video))
    assert pump(lambda: win.engine._last_gray is not None)
    p0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)
    for name in win.engine.test.dots:
        live.view.clicked.emit(float(p0[name][0]) + 2, float(p0[name][1]) + 2)
    assert pump(lambda: win.engine.frame_index > 0)
    pump(lambda: False, 0.4)
    live.setFocus()
    yield app, win, live, pump, tmp_path
    if win.engine.rec is not None:
        live.btn_rec.click()
    win.close()


def where(win, name):
    t = win.engine.trackers.trackers[name]
    return float(t.x), float(t.y)


def test_a_pass_works_while_the_video_is_being_processed(running):
    """The operator clicks each dot where it is now (the old mark follows the sticker); the clicks arrive between pictures that the video
    thread is tracking. Afterwards every dot is marked, the video went on, and the position can be locked again."""
    app, win, live, pump, tmp = running
    e = win.engine
    live.btn_lock.click()
    assert pump(lambda: e.locked, 5) and live.btn_lock.isChecked()
    frames0 = e.frame_index
    QTest.keyClick(live, Qt.Key_D)
    assert live._pass and live.btn_redo_dots.text() == "Stop (D)"
    for name in list(e.test.dots):
        assert live._arm == name
        x, y = where(win, name)
        live.view.clicked.emit(x + 2, y + 2)
        pump(lambda: False, 0.05)                                                     # the video thread keeps tracking meanwhile
    assert not live._pass and live._arm is None
    assert not e.locked and not live.btn_lock.isChecked()                              # released by the first dot marked again
    assert pump(lambda: e.frame_index > frames0 + 5, 10)                               # and the video went on
    assert set(e.dot_status().values()) - {"off"} == {"ok"}
    live.btn_lock.click()
    assert pump(lambda: e.locked, 5) and live.btn_lock.isChecked()


def test_stopping_a_pass_half_way_while_playing_keeps_every_dot_tracking(running):
    app, win, live, pump, tmp = running
    e = win.engine
    names = list(e.test.dots)
    QTest.keyClick(live, Qt.Key_D)
    x, y = where(win, names[0])
    live.view.clicked.emit(x + 2, y + 2)                                               # one dot marked again ...
    QTest.keyClick(live, Qt.Key_F4)                                                    # ... then stop
    assert not live._pass and live._arm is None
    f0 = e.frame_index
    assert pump(lambda: e.frame_index > f0 + 10, 10)
    assert set(e.dot_status().values()) - {"off"} == {"ok"}                            # nothing was lost by stopping


def test_during_a_real_recording_the_bar_and_all_its_keys_do_nothing(running):
    app, win, live, pump, tmp = running
    e = win.engine
    live.btn_lock.click()
    live.participant.setText("P01")
    live.btn_rec.click()
    assert e.rec is not None
    pump(lambda: False, 0.5)                                                           # the 200 ms timer has switched the bar off
    names = list(e.test.dots)
    positions = {n: where(win, n) for n in names}
    test_before, target_before, plan_before = e.test.key, e.target, live._plan_current
    assert not live.btn_redo_dots.isEnabled() and not live.pos_combo.isEnabled()
    assert not live.btn_pos_next.isEnabled() and not live.btn_pos_prev.isEnabled()
    assert all(not live.chips[n].isEnabled() for n in names)
    for key in (Qt.Key_D, Qt.Key_N, Qt.Key_B, Qt.Key_3, Qt.Key_F2, Qt.Key_F3, Qt.Key_F4):
        QTest.keyClick(live, key)
    live.view.clicked.emit(*[v + 3 for v in positions[names[0]]])                      # and a click on the video is ignored as before
    pump(lambda: False, 0.3)
    assert not live._pass and live._arm is None and e.rec is not None and e.locked
    assert (e.test.key, e.target, live._plan_current) == (test_before, target_before, plan_before)
    assert set(e.dot_status().values()) - {"off"} == {"ok"}
    live.btn_rec.click()
    assert pump(lambda: e.rec is None, 20)
    meta = json.loads(next((tmp / "recordings" / "P01").glob("*_meta.json")).read_text())
    assert meta["test_key"] == test_before and meta["target_deg"] == target_before            # the trial's settings are what they were at the start
    assert "changed_during_recording" not in meta
    assert live.btn_redo_dots.isEnabled() or pump(lambda: live.btn_redo_dots.isEnabled(), 2)    # and the bar is back after the recording
