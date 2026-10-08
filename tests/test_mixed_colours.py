"""White dots on dark clothing and black dots on skin in the same picture: Auto decides per dot."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
WHITE_ON_DARK = {"asis", "psis", "trochanter", "epicondyle"}          # e.g. over black shorts
BLACK_ON_LIGHT = {"fibular_head", "malleolus", "mt5_base", "mt5_head"}  # e.g. on skin and socks


def picture(p, seed, radius=7.0, noise=3.0):
    rng = np.random.default_rng(seed)
    img = np.full((720, 1280, 3), 135, np.uint8)
    for n, (x, y) in p.items():
        dark = n in WHITE_ON_DARK
        cv2.circle(img, (int(x), int(y)), int(radius * 3.2), (35, 35, 35) if dark else (205, 205, 205), -1, cv2.LINE_AA)   # the cloth / skin patch
        cv2.circle(img, (int(round(x)), int(round(y))), int(radius), (245, 245, 245) if dark else (15, 15, 15), -1, cv2.LINE_AA)
    return np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape[:2] + (1,)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("radius", [5.0, 7.0])
def test_auto_marks_and_tracks_white_and_black_dots_in_one_picture(radius):
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("knee_extension")
    eng.facing_right = True
    eng.process(picture(leg_points(hip_flex=90, knee_flex=40, ankle_df=5), 0, radius), 0.0)
    p0 = leg_points(hip_flex=90, knee_flex=40, ankle_df=5)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p0[n] + 2)), n
    for n in eng.active_dots:
        assert eng.trackers.trackers[n].kind == ("white" if n in WHITE_ON_DARK else "black"), n
    worst = 0.0
    for i in range(30):
        p = leg_points(hip_flex=90, knee_flex=40 + i, ankle_df=5)
        res = eng.process(picture(p, i, radius), 0.1 + i / 30)
        for n, st in res.dots.items():
            assert not st.lost, (i, n)
            worst = max(worst, float(np.hypot(st.x - p[n][0], st.y - p[n][1])))
    assert worst < 1.0, worst
    knee = next(a for a in res.angles if a.name == "knee")
    assert knee.raw_value == pytest.approx(40 + 29, abs=3.0)
