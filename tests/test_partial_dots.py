"""TRK-02: a dot that is half covered or cut off is flagged (it keeps being tracked, its centre may be shifted)."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
VIDEO = ROOT / "Bundepus.mp4"


def run(radius=7.0, noise=3.0, blur=0.0, cover=None, frames=40, edge=False):
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("knee_extension")
    p = dict(leg_points(hip_flex=90, knee_flex=70, ankle_df=5))
    if edge:
        p["asis"] = np.array([300.0, 3.0])                      # a dot at the very edge of the picture
    eng.process(render(p, seed=0, dot_radius=radius, noise=noise), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    flagged = {n: 0 for n in eng.active_dots}
    last = None
    for i in range(frames):
        img = render(p, seed=i, dot_radius=radius, noise=noise)
        if blur:
            img = cv2.GaussianBlur(img, (0, 0), blur)
        if cover is not None and i >= 10:                       # hide the right-hand `cover` share of the epicondyle dot
            x, y = p["epicondyle"]
            x0 = int(x - radius + (1 - cover) * 2 * radius)
            cv2.rectangle(img, (x0, int(y - radius - 2)), (int(x + radius + 2), int(y + radius + 2)), (90, 90, 90), -1)
        last = eng.process(img, 0.1 + i / 30)
        if i >= 10:
            for n, st in last.dots.items():
                flagged[n] += int(st.partial)
    n = frames - 10
    return {k: v / n for k, v in flagged.items()}, last, eng


@pytest.mark.parametrize("radius", [4.5, 7.0])
@pytest.mark.parametrize("kw", [{}, {"noise": 6.0}, {"blur": 1.5}])
def test_clean_noisy_and_blurred_dots_are_never_flagged(radius, kw):
    rates, _, _ = run(radius=radius, **kw)
    assert max(rates.values()) == 0.0, rates


@pytest.mark.parametrize("radius", [4.5, 7.0])
def test_a_half_covered_dot_is_flagged_in_nearly_every_frame_and_only_that_dot(radius):
    rates, res, _ = run(radius=radius, cover=0.5)
    assert rates["epicondyle"] >= 0.95, rates
    assert all(v == 0.0 for k, v in rates.items() if k != "epicondyle")
    assert any("partly covered" in h and "epicondyle" in h.lower() for h in res.header)
    assert not res.dots["epicondyle"].lost                      # still tracked, just flagged


def test_a_dot_at_the_edge_of_the_picture_is_flagged():
    rates, _, _ = run(edge=True)
    assert rates["asis"] >= 0.95 and all(v == 0.0 for k, v in rates.items() if k != "asis")


def test_the_flag_changes_no_position_and_no_angle():
    from limbtrack import tracker
    a = run(cover=0.5)[1]
    saved = tracker.PARTIAL_AREA_BELOW, tracker.PARTIAL_ELONGATION
    tracker.PARTIAL_AREA_BELOW, tracker.PARTIAL_ELONGATION = 0.0, 1e9     # switch the detection off
    try:
        b = run(cover=0.5)[1]
    finally:
        tracker.PARTIAL_AREA_BELOW, tracker.PARTIAL_ELONGATION = saved
    for n in a.dots:
        assert (a.dots[n].x, a.dots[n].y) == (b.dots[n].x, b.dots[n].y)
    assert [x.raw_value for x in a.angles] == [x.raw_value for x in b.angles]


def test_csv_and_meta_record_the_flag(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    p = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(30):
        img = render(p, seed=i)
        if i >= 15:
            x, y = p["epicondyle"]
            cv2.rectangle(img, (int(x), int(y - 9)), (int(x + 9), int(y + 9)), (90, 90, 90), -1)
        eng.process(img, 0.1 + i / 30)
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert rows[5]["epicondyle_partial"] == "0" and rows[25]["epicondyle_partial"] == "1" and rows[25]["trochanter_partial"] == "0"
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["partial_dot_frames"]["epicondyle"] >= 13 and "trochanter" not in meta["partial_dot_frames"]


@pytest.mark.skipif(not VIDEO.exists(), reason="Bundepus.mp4 is not in the project folder")
def test_no_false_alarms_on_the_real_practice_video():
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("knee_extension")
    eng.set_dot_kind("black")
    for d in ["asis", "psis", "fibular_head", "mt5_base", "mt5_head"]:
        eng.set_dot_enabled(d, False)
    eng.set_three_point("trochanter", "epicondyle", "malleolus")
    cap = cv2.VideoCapture(str(VIDEO))
    clicks = {"trochanter": (1008, 128), "epicondyle": (622, 182), "malleolus": (566, 563)}
    flagged = tracked = 0
    for i in range(312):
        frame = cap.read()[1]
        res = eng.process(frame, i / 30.0)
        if i == 0:
            for n, (x, y) in clicks.items():
                eng.mark_dot(n, x + 2, y + 2)
            res = eng.process(frame, 0.0001)
        for n, st in res.dots.items():
            if n in clicks and not st.lost:
                tracked += 1
                flagged += int(st.partial)
    assert tracked > 700 and flagged / tracked < 0.01
