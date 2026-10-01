"""If the video thread hits an error it must say so, write last_error.log and keep any recording."""
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def pump(app, cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_video(path, n=60):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(n):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 20 * math.sin(math.pi * i / n), ankle_df=5), seed=i))
    w.release()


def test_an_error_in_the_video_thread_is_shown_logged_and_the_recording_is_kept(tmp_path):
    app = QApplication.instance() or QApplication([])
    video = tmp_path / "demo.mp4"
    make_video(video)
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    live.start_source(str(video))
    assert pump(app, lambda: win.engine._last_gray is not None)

    p0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)
    for name in win.engine.test.dots:
        live.view.clicked.emit(float(p0[name][0]) + 2, float(p0[name][1]) + 2)
    assert pump(app, lambda: win.engine.frame_index > 0)
    live.btn_lock.click()
    assert win.engine.locked
    live.participant.setText("P01")
    live.btn_rec.click()
    assert win.engine.rec is not None

    original = win.engine.process
    calls = {"n": 0}

    def failing(frame, t):
        calls["n"] += 1
        if calls["n"] > 8:
            raise RuntimeError("simulated tracking failure")
        return original(frame, t)

    win.engine.process = failing
    assert pump(app, lambda: live.worker is None, timeout=15)          # the thread ended instead of dying silently
    assert "simulated tracking failure" in live.instruction.text()
    assert "last_error.log" in live.instruction.text()

    log = tmp_path / "recordings" / "last_error.log"
    assert log.exists() and "RuntimeError" in log.read_text(encoding="utf-8")

    assert win.engine.rec is None                                      # recording closed properly
    metas = list((tmp_path / "recordings" / "P01").glob("*_meta.json"))
    assert len(metas) == 1
    win.review.load(str(metas[0]))
    assert win.review.trial.meta["frames"] >= 1
    win.close()
