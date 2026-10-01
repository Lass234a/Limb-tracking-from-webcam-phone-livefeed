"""Mirror toggle: the picture is flipped before tracking and recording, and the flip is written to meta.json."""
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
W = 1280


def pump(app, cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_window(tmp_path, n=40):
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, 720))
    for i in range(n):
        w.write(render(leg_points(hip_flex=80, knee_flex=40, ankle_df=5), seed=i))   # the same pose in every frame
    w.release()
    return MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path), video


def test_the_engine_receives_the_flipped_picture_only_when_mirror_is_on(tmp_path):
    app = QApplication.instance() or QApplication([])
    win, video = make_window(tmp_path)
    seen = []
    original = win.engine.process
    win.engine.process = lambda frame, t: (seen.append(frame.copy()), original(frame, t))[1]
    live = win.live
    live.start_source(str(video))
    assert pump(app, lambda: len(seen) > 3)
    live.btn_pause.click()
    pump(app, lambda: False, 0.2)
    plain = seen[-1].copy()

    live.chk_mirror.setChecked(True)                                   # redraws the paused frame, flipped
    assert pump(app, lambda: np.array_equal(seen[-1], cv2.flip(plain, 1)), timeout=5)
    assert win.engine.mirror is True

    live.chk_mirror.setChecked(False)
    pump(app, lambda: False, 0.3)
    assert np.array_equal(seen[-1], plain)
    live.stop_camera()
    win.close()


def test_changing_mirror_clears_the_marked_dots_and_the_lock(tmp_path):
    app = QApplication.instance() or QApplication([])
    win, video = make_window(tmp_path)
    live = win.live
    live.start_source(str(video))
    assert pump(app, lambda: win.engine._last_gray is not None)
    p0 = leg_points(hip_flex=80, knee_flex=40, ankle_df=5)
    for name in win.engine.test.dots:
        live.view.clicked.emit(float(p0[name][0]) + 2, float(p0[name][1]) + 2)
    assert win.engine.next_dot_to_mark() is None
    live.chk_mirror.setChecked(True)
    assert win.engine.next_dot_to_mark() is not None and not win.engine.locked
    assert "mirrored" in live.instruction.text().lower()
    live.stop_camera()
    win.close()


def test_recording_with_mirror_stores_flipped_coordinates_and_says_so(tmp_path):
    app = QApplication.instance() or QApplication([])
    win, video = make_window(tmp_path, n=45)
    live = win.live
    live.chk_mirror.setChecked(True)
    live.start_source(str(video))
    assert pump(app, lambda: win.engine._last_gray is not None)
    live.btn_pause.click()
    pump(app, lambda: False, 0.3)
    p0 = leg_points(hip_flex=80, knee_flex=40, ankle_df=5)
    flipped = {n: (W - 1 - float(p[0]), float(p[1])) for n, p in p0.items()}
    for name in win.engine.test.dots:
        live.view.clicked.emit(flipped[name][0] + 2, flipped[name][1] + 2)      # the dots now sit on the flipped side
    assert win.engine.next_dot_to_mark() is None
    live.btn_lock.click()
    live.participant.setText("P01")
    live.btn_pause.click()
    live.btn_rec.click()
    assert not live.chk_mirror.isEnabled() or pump(app, lambda: not live.chk_mirror.isEnabled(), timeout=2)
    assert pump(app, lambda: win.engine.rec is None, timeout=30)
    meta_path = next((tmp_path / "recordings" / "P01").glob("*_meta.json"))
    meta = json.loads(meta_path.read_text())
    assert meta["mirrored_image"] is True
    rows = list(csv.DictReader(open(str(meta_path).replace("_meta.json", "_data.csv"), newline="")))
    xs = [float(r["trochanter_x"]) for r in rows if r["trochanter_x"]]
    assert xs and abs(np.mean(xs) - flipped["trochanter"][0]) < 3                # coordinates belong to the flipped picture
    win.close()
