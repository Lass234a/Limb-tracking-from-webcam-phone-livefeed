"""Trails: the path of every dot over one recording. Display only."""
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import overlay  # noqa: E402
from limbtrack.engine import TRAIL_MIN_STEP_PX, Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def pose(i):
    return leg_points(hip_flex=70, knee_flex=40 + 25 * math.sin(2 * math.pi * i / 60), ankle_df=5)


def make_engine(tmp_path, trails=True):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 40.0
    eng.trails_enabled = trails
    p = pose(0)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    eng.process(render(p, seed=0), 0.001)
    return eng


def run_trial(eng, frames=40, hide=None, t0=0.1):
    eng.start_trial("P01")
    last = None
    for i in range(frames):
        h = [hide[0]] if hide and hide[1] <= i < hide[2] else []
        last = eng.process(render(pose(i), seed=i, hide=h), t0 + i / 30)
    return eng.stop_trial(), last


def test_no_trails_before_a_recording_starts(tmp_path):
    eng = make_engine(tmp_path)
    for i in range(20):
        eng.process(render(pose(i), seed=i), 0.1 + i / 30)
    assert eng._trails == {}


def test_moving_dots_leave_trails_and_still_dots_do_not_grow_one(tmp_path):
    eng = make_engine(tmp_path)
    run_trial(eng)
    t = eng._trails
    assert set(t) == set(eng.test.dots)
    assert len(t["trochanter"]) == 1 and len(t["trochanter"][0]) <= 3            # fixed dot: essentially a single point
    assert len(t["malleolus"]) == 1 and len(t["malleolus"][0]) > 20              # the ankle swings a long way
    pts = np.array(t["malleolus"][0])
    steps = np.hypot(*np.diff(pts, axis=0).T)
    assert steps.min() >= TRAIL_MIN_STEP_PX - 1e-9                               # points are only added after a real move


def test_a_lost_dot_breaks_its_trail_and_others_are_unaffected(tmp_path):
    eng = make_engine(tmp_path)
    run_trial(eng, frames=40, hide=("epicondyle", 10, 20))
    t = eng._trails
    assert len(t["epicondyle"]) == 2                                             # before and after the gap: no line across it
    assert all(len(t[n]) == 1 for n in t if n != "epicondyle")


def test_trails_survive_stop_and_reset_at_the_next_recording(tmp_path):
    eng = make_engine(tmp_path)
    run_trial(eng)
    kept = {n: sum(len(s) for s in segs) for n, segs in eng._trails.items()}
    assert kept["malleolus"] > 20
    for i in range(10):                                                          # live, not recording: nothing is added
        eng.process(render(pose(40 + i), seed=i), 2.0 + i / 30)
    assert {n: sum(len(s) for s in segs) for n, segs in eng._trails.items()} == kept
    eng.start_trial("P02")
    assert eng._trails == {}                                                     # fresh start
    eng.process(render(pose(50), seed=0), 3.0)                                  # the movement simply continues
    assert all(len(segs) == 1 and len(segs[0]) == 1 for segs in eng._trails.values())
    eng.stop_trial()


def test_changing_landmarks_clears_the_trails(tmp_path):
    eng = make_engine(tmp_path)
    run_trial(eng)
    eng.set_dot_enabled("asis", False)
    assert eng._trails == {}


def test_trails_are_display_only(tmp_path):
    """Same input with trails on and off: identical CSV, identical raw video; only the overlay picture differs."""
    out = {}
    for trails in (True, False):
        eng = make_engine(tmp_path / str(trails), trails=trails)
        out[trails] = run_trial(eng)
    rows = {k: list(csv.reader(open(v[0]["data.csv"], newline=""))) for k, v in out.items()}
    assert rows[True] == rows[False]
    caps = {k: cv2.VideoCapture(str(v[0]["raw.mp4"])) for k, v in out.items()}
    for _ in range(40):
        assert np.array_equal(caps[True].read()[1], caps[False].read()[1])
    on, off = out[True][1].overlay, out[False][1].overlay
    assert np.abs(on.astype(int) - off.astype(int)).sum() > 0                    # the trail is drawn into the picture...
    assert [a.raw_value for a in out[True][1].angles] == [a.raw_value for a in out[False][1].angles]   # ...not the numbers


def test_overlay_draws_trails_and_nothing_when_empty():
    img = np.full((200, 300, 3), 120, np.uint8)
    base = overlay.draw(img, {}, [])
    assert np.array_equal(overlay.draw(img, {}, [], trails={}), base)
    drawn = overlay.draw(img, {}, [], trails={"a": [[(10, 10), (100, 80), (200, 40)]], "b": [[(5, 150)]]})
    changed = np.abs(drawn.astype(int) - base.astype(int)).sum(axis=2) > 0
    assert changed.any() and not changed[120:, :].any()                          # a single point draws nothing


def test_review_shows_the_same_trails_up_to_the_current_frame(tmp_path):
    app = QApplication.instance() or QApplication([])
    eng = make_engine(tmp_path)
    paths, _ = run_trial(eng, frames=40, hide=("epicondyle", 10, 20))
    tr = Trial(paths["meta.json"])
    assert len(tr.trail_segments["epicondyle"]) == 2
    sizes = [sum(len(s) for s in tr.trails_upto(i)["malleolus"]) for i in (0, 10, 20, 39)]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]
    assert len(tr.trails_upto(15)["epicondyle"]) == 1                            # inside the gap: only the first part
    assert len(tr.trails_upto(30)["epicondyle"]) == 2
    with_t, without = tr.drawn(35, trails=True), tr.drawn(35, trails=False)
    assert np.abs(with_t.astype(int) - without.astype(int)).sum() > 0

    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    assert tab.chk_trails.isChecked()
    tab.slider.setValue(30)
    tab.chk_trails.setChecked(False)                                             # redraws without error
    tab.chk_trails.setChecked(True)


def test_the_live_window_has_a_trails_checkbox(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    assert win.live.chk_trails.isChecked() and win.engine.trails_enabled
    win.live.chk_trails.setChecked(False)
    assert win.engine.trails_enabled is False
    win.close()


def test_overlay_video_contains_the_trail_when_enabled(tmp_path):
    eng = make_engine(tmp_path)
    paths, last = run_trial(eng)
    cap = cv2.VideoCapture(str(paths["overlay.mp4"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 39)
    saved = cap.read()[1]
    raw = cv2.VideoCapture(str(paths["raw.mp4"]))
    raw.set(cv2.CAP_PROP_POS_FRAMES, 39)
    plain = raw.read()[1]
    pts = np.array(eng._trails["malleolus"][0]).round().astype(int)
    mid = pts[len(pts) // 2]
    saved_patch = saved[mid[1] - 2:mid[1] + 3, mid[0] - 2:mid[0] + 3].astype(int)
    plain_patch = plain[mid[1] - 2:mid[1] + 3, mid[0] - 2:mid[0] + 3].astype(int)
    assert np.abs(saved_patch - plain_patch).sum() > 100                         # the coloured path is in the saved overlay video
