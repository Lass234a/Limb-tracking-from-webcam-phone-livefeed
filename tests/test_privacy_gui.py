"""REP-05 in the window: drawing a crop and masks, the participant-code and notes checks, the recordings folder."""
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(150):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 10 * math.sin(math.pi * i / 150), ankle_df=5), seed=i))
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

    live = win.live
    live.start_source(str(video))
    assert pump(lambda: win.engine._last_gray is not None)
    for name in win.engine.test.dots:
        live.view.clicked.emit(float(P0[name][0]) + 2, float(P0[name][1]) + 2)
    assert win.engine.next_dot_to_mark() is None
    live.btn_lock.click()
    return app, win, live, warnings, pump, tmp_path


def draw(live, kind, x0, y0, x1, y1):
    live.start_privacy_drawing(kind)
    live.view.rect_drawn.emit(x0, y0, x1, y1)


def around_leg(pad=70):
    xs = [p[0] for p in P0.values()]
    ys = [p[1] for p in P0.values()]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def test_drawn_crop_and_mask_reach_the_engine_and_the_label(setup):
    app, win, live, warnings, pump, tmp = setup
    assert "whole picture" in live.priv_label.text()
    draw(live, "crop", *around_leg())
    draw(live, "mask", 0, 600, 300, 700)
    app.processEvents()
    assert win.engine.privacy_crop is not None and len(win.engine.privacy_masks) == 1
    live.update_privacy_label()
    assert "cropped to" in live.priv_label.text() and "1 black mask" in live.priv_label.text()
    live.btn_priv_clear.click()
    assert win.engine.privacy_crop is None and win.engine.privacy_masks == []
    win.close()


def test_a_crop_that_cuts_off_a_dot_warns_and_refuses_to_record(setup):
    app, win, live, warnings, pump, tmp = setup
    x0, y0, x1, y1 = around_leg()
    draw(live, "crop", x0 + 150, y0, x1, y1)                       # leaves out the left-most dots
    live.update_privacy_label()
    assert "WARNING" in live.priv_label.text()
    live.participant.setText("P01")
    live.btn_rec.click()
    assert win.engine.rec is None and not live.btn_rec.isChecked()
    assert warnings and ("crop" in warnings[-1] or "mask" in warnings[-1])
    win.close()


def test_recording_with_a_crop_through_the_window(setup):
    app, win, live, warnings, pump, tmp = setup
    draw(live, "crop", *around_leg())
    live.participant.setText("P01")
    live.btn_rec.click()
    assert win.engine.rec is not None, warnings
    assert pump(lambda: not live.btn_crop.isEnabled() and not live.btn_folder.isEnabled())        # fixed while recording
    assert pump(lambda: win.engine.rec.frames > 10)
    live.btn_rec.click()
    meta = json.loads(next((tmp / "recordings").rglob("*_meta.json")).read_text())
    assert meta["privacy"]["crop_xywh"] and meta["status"] == "complete"
    win.close()


@pytest.mark.parametrize("bad", ["", "Anna", "p01", "P1", "P01 test"])
def test_only_a_participant_code_is_accepted(setup, bad):
    app, win, live, warnings, pump, tmp = setup
    live.participant.setText(bad)
    live.btn_rec.click()
    assert win.engine.rec is None and warnings and "participant code" in warnings[-1]
    win.close()


def test_notes_with_an_email_or_long_number_are_refused(setup):
    app, win, live, warnings, pump, tmp = setup
    live.participant.setText("P01")
    for text in ("mail anna@x.dk", "cpr 010203-1234"):
        live.notes.setText(text)
        live.btn_rec.click()
        assert win.engine.rec is None and "Remove it" in warnings[-1]
    live.notes.setText("strap loosened after trial 1")
    live.btn_rec.click()
    assert win.engine.rec is not None
    live.btn_rec.click()
    win.close()


def test_recordings_folder_can_be_changed_and_is_remembered(setup, monkeypatch):
    app, win, live, warnings, pump, tmp = setup
    new = tmp / "elsewhere"
    new.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(new))
    live.btn_folder.click()
    assert win.engine.recordings_dir == new and win.review.dir == new and str(new) in live.folder_label.text()
    live.participant.setText("P01")
    live.btn_rec.click()
    assert pump(lambda: win.engine.rec.frames > 3)
    live.btn_rec.click()
    assert list(new.rglob("*_meta.json")) and not (tmp / "recordings").exists()
    win.close()
    again = MainWindow(load_protocol(ROOT / "protocol.json"), tmp)
    assert again.engine.recordings_dir == new
    again.close()
