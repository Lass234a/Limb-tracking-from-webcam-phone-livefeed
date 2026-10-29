"""TRK-01 (+CR-02): a covered dot must not take a fast-moving neighbour, whatever the order of the trackers."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402
from limbtrack.tracker import TrackerSet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def knee_at(i, start, speed_deg_per_frame, k0=40.0, k1=80.0):
    k = k0 + max(0, i - start) * speed_deg_per_frame
    return min(k, k1)


def run(hide, hide_frames, start=15, speed=4.0, radius=7.0, reverse=False, frames=45, end=80.0):
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("belt_squat")                           # (the test with the foot dots next to the malleolus; the knee test has none since 0.24.0)
    eng.target = 40.0
    p0 = leg_points(hip_flex=90, knee_flex=40.0, ankle_df=5)
    eng.process(render(p0, seed=0, dot_radius=radius), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p0[n] + 2))
    if reverse:                                          # same trackers, opposite order: the answer must not depend on it
        eng.trackers.trackers = dict(reversed(list(eng.trackers.trackers.items())))
    worst_dot, worst_knee = 0.0, 0.0
    for i in range(frames):
        k = knee_at(i, start, speed, k1=end)
        p = leg_points(hip_flex=90, knee_flex=k, ankle_df=5)
        res = eng.process(render(p, seed=i, hide=[hide] if hide_frames[0] <= i < hide_frames[1] else [], dot_radius=radius), 0.1 + i / 30)
        for n, st in res.dots.items():
            if not st.lost:
                worst_dot = max(worst_dot, float(np.hypot(st.x - p[n][0], st.y - p[n][1])))
        knee = next(a for a in res.angles if a.name == "knee")
        if knee.raw_status != "lost" and not np.isnan(knee.raw_value):
            worst_knee = max(worst_knee, abs(knee.raw_value - k))
    return worst_dot, worst_knee


@pytest.mark.parametrize("radius", [4.5, 7.0])
@pytest.mark.parametrize("speed", [4.0, 8.0])           # 120 and 240 deg/s at 30 fps
def test_a_hidden_malleolus_does_not_take_a_fast_neighbour(radius, speed):
    worst_dot, worst_knee = run("malleolus", (20, 28), speed=speed, radius=radius)
    assert worst_dot < 3.0, worst_dot                    # every dot that claims to be tracked sits on its own sticker
    assert worst_knee < 1.0, worst_knee


def test_the_same_when_the_cover_starts_with_the_motion():
    worst_dot, worst_knee = run("malleolus", (15, 24), speed=4.0, radius=4.5)
    assert worst_dot < 3.0 and worst_knee < 1.0


def test_a_hidden_foot_dot_does_not_take_the_other_foot_dot():
    worst_dot, _ = run("mt5_base", (20, 28), speed=4.0, radius=4.5)
    assert worst_dot < 3.0


def test_the_outcome_does_not_depend_on_the_order_of_the_trackers():
    a = run("malleolus", (20, 28), speed=4.0, radius=7.0)
    b = run("malleolus", (20, 28), speed=4.0, radius=7.0, reverse=True)
    assert a[0] < 3.0 and b[0] < 3.0 and a[1] < 1.0 and b[1] < 1.0


def test_the_conflict_rule_keeps_the_tracker_that_fits_its_own_expectation():
    from limbtrack.tracker import DotState, DotTracker
    ts = TrackerSet(["a", "b"], "white")
    for n, x in (("a", 100.0), ("b", 140.0)):
        t = ts.trackers[n]
        t.x, t.y, t.initialised, t.diam, t.area = x, 50.0, True, 12.0, 110.0
    ts.trackers["a"].vx = 20.0                          # a is expected at 120
    ts.trackers["b"].vx = 0.0                           # b is expected at 140
    # both trackers end up on 139: b fits its expectation, a does not
    for t, pos in ((ts.trackers["a"], 139.0), (ts.trackers["b"], 139.5)):
        t._before = (t.x, t.y, t.area, t.diam, t.contrast)
    # emulate the resolution step directly through update() with a stubbed tracker update
    def fake(n, x):
        def update(gray, claimed=()):
            ts.trackers[n].x = x
            return DotState(n, x, 50.0, False, 0, 1.0)
        return update
    ts.trackers["a"].update, ts.trackers["b"].update = fake("a", 139.0), fake("b", 139.5)
    states = ts.update(np.zeros((100, 300), np.uint8))
    assert states["a"].lost and not states["b"].lost
