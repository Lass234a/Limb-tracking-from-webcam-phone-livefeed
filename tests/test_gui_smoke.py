"""Drives the real window (offscreen) with a synthetic video file: mark, lock, record, review."""
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

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


def test_full_session(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2])))

    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    n = 75
    for i in range(n):
        knee = 20 + 30 * math.sin(math.pi * i / n)
        w.write(render(leg_points(hip_flex=80, knee_flex=knee, ankle_df=5, pelvic_tilt=3), seed=i))
    w.release()

    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    live.test_combo.setCurrentIndex(0)                 # knee extension
    live.target_spin.setValue(20.0)
    live.start_source(str(video))
    assert pump(app, lambda: win.engine._last_gray is not None)

    p0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5, pelvic_tilt=3)
    for name in win.engine.test.dots:
        assert win.engine.next_dot_to_mark() == name
        live.view.clicked.emit(float(p0[name][0]) + 2, float(p0[name][1]) + 2)
    assert win.engine.next_dot_to_mark() is None

    assert pump(app, lambda: win.engine.frame_index > 0)
    live.btn_lock.click()
    assert live.btn_lock.isChecked() and win.engine.locked

    live.participant.setText("P01")
    live.btn_rec.click()
    assert win.engine.rec is not None
    assert pump(app, lambda: win.engine.rec is None or live.worker is None, timeout=30)  # video ends -> stops
    assert win.engine.rec is None

    metas = list((tmp_path / "recordings" / "P01").glob("*_meta.json"))
    assert len(metas) == 1
    win.review.load(str(metas[0]))
    tr = win.review.trial
    assert tr.meta["frames"] > 20
    assert len(tr.rows) == tr.meta["frames"]
    win.review.show_frame(len(tr.rows) // 2)
    assert win.review.stats.text().count("\n") >= 4
    win.close()


def test_pause_and_frame_stepping(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    n = 40
    for i in range(n):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + i, ankle_df=5), seed=i))
    w.release()

    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    live.start_source(str(video))
    pos, imgs = [], []
    live.worker.position.connect(lambda c, t: pos.append((c, t)))
    live.worker.frame_ready.connect(lambda im: imgs.append(im))
    assert pump(app, lambda: len(pos) > 3)

    live.btn_pause.click()
    pump(app, lambda: False, 0.3)
    frozen = pos[-1][0]
    pump(app, lambda: False, 0.4)
    assert pos[-1][0] == frozen, "video kept playing while paused"
    assert "paused" in live.frame_label.text()

    live.btn_fwd.click()
    assert pump(app, lambda: pos[-1][0] == frozen + 1)
    live.btn_back.click()
    live.btn_back.click()
    assert pump(app, lambda: pos[-1][0] == frozen - 1)
    assert pos[-1][1] == n

    # marking while paused redraws the frame
    before = len(imgs)
    p = leg_points(hip_flex=80, knee_flex=20 + frozen - 2, ankle_df=5)
    live.view.clicked.emit(float(p["asis"][0]) + 1, float(p["asis"][1]) + 1)
    assert pump(app, lambda: len(imgs) > before)
    assert pos[-1][0] == frozen - 1

    # play resumes; stepping past the end just stays on the last frame
    live.btn_pause.click()
    assert pump(app, lambda: pos[-1][0] > frozen)
    assert pump(app, lambda: pos[-1][0] == n and live.btn_pause.isChecked(), timeout=10)
    live.btn_fwd.click()
    pump(app, lambda: False, 0.2)
    assert pos[-1][0] == n
    live.btn_back.click()
    assert pump(app, lambda: pos[-1][0] == n - 1)
    win.close()
