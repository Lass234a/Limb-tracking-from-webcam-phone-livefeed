"""The lost-dot rule: freeze, small fixed circle, resume only on a convincing dot."""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import limbtrack.tracker as tracker
from limbtrack.synthetic import leg_points, render
from limbtrack.tracker import DotState, DotTracker, TrackerSet, to_gray

ROOT = Path(__file__).resolve().parent.parent
PTS = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
DOT = "mt5_head"          # an isolated dot, far from the others


def frame(points, **kw):
    return to_gray(render(points, **kw))


def moved(points, dx, dy):
    p = dict(points)
    p[DOT] = points[DOT] + np.array([dx, dy])
    return p


def tracker_on_dot(kind="white"):
    t = DotTracker(DOT, kind)
    assert t.init_at(frame(PTS), *(PTS[DOT] + 2))
    return t


def test_lost_dot_freezes_and_circle_is_fixed():
    t = tracker_on_dot()
    for i in range(3):                                   # moving a little, so it has a velocity
        t.update(frame(moved(PTS, 3 * (i + 1), 0), seed=i))
    last = (t.x, t.y)
    radii = []
    for i in range(12):
        st = t.update(frame(PTS, hide=[DOT], seed=10 + i))
        assert st.lost and (st.x, st.y) == last          # frozen: no prediction, no drift
        assert st.quality == 0.0
        radii.append(t.lost_radius)
    assert len(set(radii)) == 1 and radii[0] == pytest.approx(tracker.LOST_RADIUS_FACTOR * t.diam)
    assert t.vx == 0.0 and t.vy == 0.0


def test_dot_reappearing_inside_the_circle_resumes():
    t = tracker_on_dot()
    for i in range(5):
        t.update(frame(PTS, hide=[DOT], seed=i))
    assert t.lost_frames == 5
    st = t.update(frame(moved(PTS, 6, -4), seed=9))      # 7 px from where it froze; circle is about 28 px
    assert not st.lost and st.quality > 0.8
    assert math.hypot(st.x - (PTS[DOT][0] + 6), st.y - (PTS[DOT][1] - 4)) < 0.5
    assert (t.vx, t.vy) == (0.0, 0.0)                    # no velocity carried across the gap


def test_dot_reappearing_outside_the_circle_stays_lost():
    t = tracker_on_dot()
    for i in range(3):
        t.update(frame(PTS, hide=[DOT], seed=i))
    far = tracker.LOST_RADIUS_FACTOR * t.diam + 25
    for i in range(10):                                  # a perfectly good dot, but too far away
        st = t.update(frame(moved(PTS, far, 0), seed=20 + i))
        assert st.lost
    assert t.lost_radius == pytest.approx(tracker.LOST_RADIUS_FACTOR * t.diam)   # still not growing


def test_faint_blob_inside_the_circle_is_not_accepted():
    t = tracker_on_dot()
    t.update(frame(PTS, hide=[DOT], seed=0))
    assert t.update(frame(PTS, seed=1, dot_gain=0.45)).lost       # 45 % of the marked contrast
    assert not t.update(frame(PTS, seed=2, dot_gain=0.8)).lost    # 80 % is fine


def test_thin_sliver_and_oversized_blob_are_not_accepted():
    t = tracker_on_dot()
    t.update(frame(PTS, hide=[DOT], seed=0))
    x, y = int(PTS[DOT][0]), int(PTS[DOT][1])
    sliver = render(PTS, hide=[DOT], seed=1)
    cv2.rectangle(sliver, (x - 2, y - 14), (x + 2, y + 14), (245, 245, 245), -1)    # thin bright bar, bright enough
    assert t.update(to_gray(sliver)).lost
    big = render(PTS, hide=[DOT], seed=2)
    cv2.circle(big, (x, y), 15, (245, 245, 245), -1)                                  # round but about 4x the area
    assert t.update(to_gray(big)).lost
    assert not t.update(frame(PTS, seed=3)).lost                                     # the real dot is accepted


def test_black_dots_follow_the_same_rule():
    def img(pts, **kw):
        return to_gray(render(pts, kind="black", **kw))

    t = DotTracker(DOT, "black")
    assert t.init_at(img(PTS), *(PTS[DOT] + 2))
    assert t.update(img(PTS, hide=[DOT], seed=1)).lost
    assert t.update(img(moved(PTS, 200, 0), seed=2)).lost                  # too far
    assert t.update(img(PTS, seed=3, dot_gain=0.45)).lost                  # too faint
    assert not t.update(img(PTS, seed=4)).lost                             # back where it was, full contrast


def test_duplicate_blob_rule_puts_the_later_tracker_back(monkeypatch):
    ts = TrackerSet(["a", "b"], "white")
    img = frame(PTS)
    assert ts.init_dot("a", img, *(PTS["mt5_head"] + 2))
    assert ts.init_dot("b", img, *(PTS["trochanter"] + 2))
    before = (ts.trackers["b"].x, ts.trackers["b"].y)
    ax, ay = ts.trackers["a"].x, ts.trackers["a"].y
    real_update = DotTracker.update

    def fake(self, gray, claimed=()):
        st = real_update(self, gray, claimed)
        if self.name == "b":                              # pretend b ended up on a's blob
            self.x, self.y = ax + 1, ay
            return DotState("b", ax + 1, ay, False, 0, 1.0)
        return st

    monkeypatch.setattr(DotTracker, "update", fake)
    states = ts.update(img)
    assert states["b"].lost
    assert (states["b"].x, states["b"].y) == pytest.approx(before)          # back where it was, not on a's dot
    assert ts.trackers["b"].lost_radius > 0 and not states["a"].lost


# ------------------------------------------------------------------ real video (not in git: skipped if absent)
VIDEO = ROOT / "Bundepus.mp4"
CLICKS = {"trochanter": (1008, 128), "epicondyle": (622, 182), "malleolus": (566, 563)}


def _track_video():
    cap = cv2.VideoCapture(str(VIDEO))
    ts = TrackerSet(list(CLICKS), "black")
    rows = []
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        g = to_gray(cap.read()[1])
        if i == 0:
            for n, (x, y) in CLICKS.items():
                assert ts.init_dot(n, g, x + 2, y + 2)
        rows.append({n: (s.x, s.y, s.lost) for n, s in ts.update(g).items()})
    # reference: the real ankle dot, re-marked at frame 256 after it left the table leg, followed to the end
    cap.set(cv2.CAP_PROP_POS_FRAMES, 256)
    ref = DotTracker("ref", "black")
    assert ref.init_at(to_gray(cap.read()[1]), 704, 516)
    refpos = {256: (ref.x, ref.y)}
    for k in range(257, len(rows)):
        s = ref.update(to_gray(cap.read()[1]))
        refpos[k] = (s.x, s.y)
    return rows, refpos


@pytest.mark.skipif(not VIDEO.exists(), reason="Bundepus.mp4 (real test video) not present")
def test_real_video_dot_hidden_behind_table_leg_never_relocks_on_the_wrong_spot():
    """The ankle dot goes behind the table leg at frames ~238-255. The old rule re-locked 60 px away on the
    shadowed heel and stayed there for 70 frames. The new rule must never report a position off the real dot."""
    rows, ref = _track_video()
    for n in ("trochanter", "epicondyle"):
        assert not any(r[n][2] for r in rows), n
    ank = [r["malleolus"] for r in rows]
    assert not any(a[2] for a in ank[:238])
    assert all(a[2] for a in ank[241:256])                                  # hidden: lost
    for k in range(257, len(rows)):
        if not ank[k][2]:
            assert math.hypot(ank[k][0] - ref[k][0], ank[k][1] - ref[k][1]) < 8, k


@pytest.mark.skipif(not VIDEO.exists(), reason="Bundepus.mp4 (real test video) not present")
def test_real_video_with_a_larger_circle_the_dot_resumes_by_itself(monkeypatch):
    """Documents the trade-off: a circle of about 10 dot-diameters finds the real dot when it reappears."""
    monkeypatch.setattr(tracker, "LOST_RADIUS_FACTOR", 10.0)
    rows, ref = _track_video()
    ank = [r["malleolus"] for r in rows]
    assert all(a[2] for a in ank[241:252])
    assert not any(a[2] for a in ank[257:])
    for k in range(257, len(rows)):
        assert math.hypot(ank[k][0] - ref[k][0], ank[k][1] - ref[k][1]) < 8, k


# ------------------------------------------------------------------ CSV while a dot is lost
def test_csv_has_blank_xy_while_a_dot_is_lost_and_review_still_opens(tmp_path):
    import csv

    from limbtrack.engine import Engine
    from limbtrack.protocol import load_protocol
    from limbtrack.review import Trial

    eng = Engine(load_protocol(ROOT / "protocol.json"), recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    eng.process(render(PTS, seed=1), 0.01)
    eng.start_trial("P01")
    for i in range(30):
        hide = ["epicondyle"] if 10 <= i < 20 else []
        eng.process(render(PTS, seed=i, hide=hide), 0.1 + i / 30)
    paths = eng.stop_trial()

    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    lost_rows = [i for i, r in enumerate(rows) if r["epicondyle_lost"] == "1"]
    assert lost_rows == list(range(10, 20))
    for i, r in enumerate(rows):
        if i in lost_rows:
            assert r["epicondyle_x"] == "" and r["epicondyle_y"] == ""
            assert r["knee_deg"] == "" and r["hip_deg"] == ""            # angles that use it are blank too
            assert r["trochanter_x"] != "" and r["ankle_deg"] != ""      # everything else is unaffected
        else:
            assert r["epicondyle_x"] != "" and r["epicondyle_y"] != "" and r["knee_deg"] != ""

    tr = Trial(paths["meta.json"])
    assert tr.drawn(5) is not None and tr.drawn(15) is not None and tr.drawn(25) is not None
    assert np.isnan(tr.deg["knee"][15]) and np.isfinite(tr.deg["knee"][25])
