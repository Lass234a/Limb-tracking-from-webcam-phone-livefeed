import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow, trace_y_range  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def test_trace_y_range_covers_band_and_data():
    assert trace_y_range([], 60, 6) == (54, 66)
    lo, hi = trace_y_range([np.array([50.0, 61.0, math.nan])], 60, 6)
    assert lo == pytest.approx(49.0) and hi == 66
    assert trace_y_range([np.array([math.nan])], 0, 4) == (-4, 4)


def _engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    pts = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(pts, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(pts[n] + 2))
    return eng, pts


def test_history_window_deviation_and_reset(tmp_path):
    eng, pts = _engine(tmp_path)
    assert eng.history() is not None                                  # the first frame is already in
    for i in range(1, 61):
        eng.process(render(pts, seed=i), i / 30.0)                    # 2 s
    h = eng.history(seconds=1.0)
    assert h["t"][-1] == 0 and h["t"][0] >= -1.0 and 30 <= len(h["t"]) <= 32
    assert h["primary"] == "knee" and h["target"] == 60.0 and h["tolerance"] == 3.0
    assert np.nanmax(np.abs(h["deviations"]["knee"])) < 1.0           # still at target: deviation about 0
    assert set(h["values"]) == {"knee", "hip", "ankle", "pelvic_tilt"}
    assert np.all(np.isnan(h["deviations"]["hip"]))                   # not locked yet: no reference for neighbours

    assert eng.lock_position()
    eng.process(render(pts, seed=99), 2.1)
    assert np.isfinite(eng.history(1.0)["deviations"]["hip"][-1])

    eng.process(render(pts, seed=100), 1.0)                           # time jumps back (frame stepping)
    assert eng.history(60.0)["t"].min() > -1.5                        # later samples were dropped

    eng.reset_dots()
    assert eng.history() is None


def test_history_is_bounded(tmp_path):
    from collections import deque
    eng, pts = _engine(tmp_path)
    eng._hist = deque(maxlen=25)                                      # same mechanism as the real 6000-frame buffer
    img = render(pts)
    for i in range(1, 60):
        eng.process(img, i / 30.0)
    assert len(eng._hist) == 25 and len(eng.history(60.0)["t"]) == 25


def test_live_trace_draws_both_modes(tmp_path):
    app = QApplication.instance() or QApplication([])
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(60):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 0.3 * i, ankle_df=5, pelvic_tilt=3), seed=i))
    w.release()

    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    live.target_spin.setValue(20.0)
    live.start_source(str(video))
    end = time.time() + 20
    while win.engine._last_gray is None and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    p0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5, pelvic_tilt=3)
    for name in win.engine.test.dots:
        live.view.clicked.emit(float(p0[name][0]) + 2, float(p0[name][1]) + 2)
    live.btn_lock.click()
    end = time.time() + 5
    while win.engine.frame_index < 30 and time.time() < end:
        app.processEvents()
        time.sleep(0.01)

    live.trace_mode.setCurrentIndex(0)
    live.update_trace()
    assert len(live._trace_items) == 2                                # main curve + tolerance band
    live.trace_mode.setCurrentIndex(1)
    live.update_trace()
    assert len(live._trace_items) == 5                                # four curves + band
    lo, hi = live.trace.viewRange()[1]
    assert lo < 0 < hi
    live.stop_camera()
    live.update_trace()
    assert live._trace_items == []
    win.close()
