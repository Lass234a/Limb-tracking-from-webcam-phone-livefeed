"""GUI-01 (+DOC-10): stray clicks while recording are ignored; re-marking prefers lost dots and never lands on a neighbour."""
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=70, knee_flex=60, ankle_df=5, pelvic_tilt=3)


def marked_engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    eng.process(render(PTS, seed=1), 0.05)
    return eng


def pos(eng, name):
    t = eng.trackers.trackers[name]
    return np.array([t.x, t.y])


def freeze_as_lost(eng, name, dx, dy):
    """Make `name` lost and frozen dx, dy away from where it really is (as after a long occlusion)."""
    t = eng.trackers.trackers[name]
    t.x += dx
    t.y += dy
    t.lost_frames = 5
    t.lost_radius = 2.0 * t.diam
    eng.process(render(PTS, seed=2), 0.1)                         # the dot is visible again, but too far from the frozen spot
    assert eng._last_states[name].lost


def test_a_click_beside_a_lost_dot_re_marks_the_lost_dot_not_a_neighbour(tmp_path):
    eng = marked_engine(tmp_path)
    epi_before = pos(eng, "epicondyle").copy()
    freeze_as_lost(eng, "fibular_head", 55.0, 0.0)               # the swap55 case: frozen further away than the neighbour is
    fx, fy = PTS["fibular_head"]
    assert eng.nearest_dot(fx, fy) == "fibular_head"
    assert eng.mark_dot("fibular_head", fx + 2, fy + 2)
    assert np.allclose(pos(eng, "fibular_head"), PTS["fibular_head"], atol=0.7)
    assert np.allclose(pos(eng, "epicondyle"), epi_before, atol=1.0)   # the neighbour stayed on its own dot


def test_re_marking_onto_a_dot_another_tracker_holds_is_refused_and_changes_nothing(tmp_path):
    eng = marked_engine(tmp_path)
    before = pos(eng, "fibular_head").copy()
    ex, ey = PTS["epicondyle"]
    assert eng.mark_dot("fibular_head", ex, ey) is False
    assert "epicondyle" in eng.mark_problem.lower() or "Lateral femoral" in eng.mark_problem
    assert np.allclose(pos(eng, "fibular_head"), before)
    assert eng.trackers.trackers["fibular_head"].initialised


def test_a_click_must_be_close_to_a_tracked_dot_to_re_mark_it(tmp_path):
    eng = marked_engine(tmp_path)
    x, y = PTS["malleolus"]
    assert eng.nearest_dot(x + 5, y + 5) == "malleolus"
    assert eng.nearest_dot(x + 45, y) is None                   # skin 45 px from the dot: not a re-mark
    assert eng.nearest_dot(5, 5) is None


def test_clicks_during_recording_are_ignored(tmp_path):
    app = QApplication.instance() or QApplication([])
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(90):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 15 * math.sin(math.pi * i / 90), ankle_df=5), seed=i))
    w.release()

    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
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
    assert win.engine.next_dot_to_mark() is None
    assert pump(lambda: win.engine.frame_index > 0)
    live.btn_lock.click()
    live.participant.setText("P01")
    live.btn_rec.click()
    assert win.engine.rec is not None
    assert pump(lambda: win.engine.rec is not None and win.engine.frame_index > 3)

    ex, ey = win.engine.trackers.trackers["epicondyle"].x, win.engine.trackers.trackers["epicondyle"].y
    live.view.clicked.emit(ex + 6, ey + 20)                      # a stray click on the video, 20 px from the dot
    assert "Stop recording" in live.instruction.text()
    app.processEvents()
    assert "Stop recording" in live.instruction.text()           # not overwritten by the status timer
    assert pump(lambda: win.engine.rec is None, timeout=30)
    live.stop_camera()
    metas = list((tmp_path / "recordings" / "P01").glob("*_meta.json"))
    assert len(metas) == 1
    rows = open(str(metas[0]).replace("_meta.json", "_data.csv"), encoding="utf-8").read().splitlines()[1:]
    header = open(str(metas[0]).replace("_meta.json", "_data.csv"), encoding="utf-8").readline().strip().split(",")
    lost_col = header.index("epicondyle_lost")
    assert all(r.split(",")[lost_col] == "0" for r in rows)       # the click did not lose the dot
    win.close()
