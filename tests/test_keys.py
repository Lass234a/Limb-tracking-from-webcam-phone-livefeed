"""GUI-13: the marker and record keys keep working while a text box has focus, and the operator can see that markers landed."""
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def recording(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(150):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 10 * math.sin(math.pi * i / 150), ankle_df=5), seed=i))
    w.release()
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
    live.btn_lock.click()
    live.participant.setText("P01")
    live.btn_rec.click()
    assert win.engine.rec is not None
    return app, win, live, pump, tmp_path


def events(win, live, pump, tmp_path):
    live.btn_rec.click() if win.engine.rec is not None else None
    meta = next((tmp_path / "recordings" / "P01").glob("*_meta.json"))
    return [e["label"] for e in json.loads(meta.read_text())["events"]]


def test_f_keys_add_markers_while_typing_in_the_notes_box(recording):
    app, win, live, pump, tmp = recording
    live.notes.setFocus()
    QTest.keyClicks(live.notes, "rest ok, see me")           # contains s, e, r: these must just be typed
    assert live.notes.text() == "rest ok, see me"
    assert win.engine.rec is not None and win.engine._pending_event is None
    QTest.keyClick(live.notes, Qt.Key_F6)
    pump(lambda: False, 0.15)
    QTest.keyClick(live.notes, Qt.Key_F7)
    pump(lambda: False, 0.15)
    assert live._marker_counts["start"] == 1 and live._marker_counts["end"] == 1
    assert events(win, live, pump, tmp) == ["MVIC start", "MVIC end"]
    win.close()


def test_enter_leaves_the_text_box_and_the_letter_keys_work_again(recording):
    app, win, live, pump, tmp = recording
    live.open_step("participant")                            # (the notes box is in the Participant section, which is not the first one open since 0.24.0)
    live.notes.setFocus()
    app.processEvents()
    assert live.key_hint.isVisible()                         # the operator is told the letters are off
    QTest.keyClick(live.notes, Qt.Key_Return)
    app.processEvents()
    assert not live.notes.hasFocus() and not live.key_hint.isVisible()
    QTest.keyClick(live, Qt.Key_S)
    pump(lambda: False, 0.15)
    assert live._marker_counts["start"] == 1
    live.participant.setFocus()
    QTest.keyClick(live.participant, Qt.Key_Escape)          # Esc also lets go
    app.processEvents()
    assert not live.participant.hasFocus()
    live.btn_rec.click()
    win.close()


def test_every_marker_is_confirmed_on_screen_with_a_running_count(recording):
    app, win, live, pump, tmp = recording
    live.btn_start_mark.click()
    live.btn_mark.click()
    live.refresh_status()
    text = live.rec_label.text()
    assert "MVIC start 1" in text and "other 1" in text and "Marker marked at" in text
    live.btn_rec.click()
    win.close()
