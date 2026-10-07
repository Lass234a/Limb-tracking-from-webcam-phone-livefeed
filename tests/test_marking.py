"""TRK-03: the Auto dot colour is decided on every click and is never kept from a click that failed."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402
from limbtrack.tracker import DotTracker, to_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)


@pytest.mark.parametrize("kind", ["white", "black"])
def test_a_failed_click_does_not_fix_the_colour_and_the_next_click_on_the_centre_works(kind):
    gray = to_gray(render(PTS, kind=kind, seed=0))
    cx, cy = PTS["trochanter"]
    t = DotTracker("trochanter", "auto")
    assert t.init_at(gray, cx, cy + 25) is False             # far from any dot
    assert t.kind == "auto" and not t.initialised            # nothing was learned from the failed click
    assert t.init_at(gray, cx + 1, cy + 1) is True
    assert t.kind == kind
    assert abs(t.x - cx) < 0.5 and abs(t.y - cy) < 0.5


@pytest.mark.parametrize("kind", ["white", "black"])
def test_auto_picks_the_right_colour_for_a_click_near_the_dot(kind):
    gray = to_gray(render(PTS, kind=kind, seed=0))
    for name in ("trochanter", "epicondyle", "malleolus", "mt5_head"):
        cx, cy = PTS[name]
        t = DotTracker(name, "auto")
        assert t.init_at(gray, cx + 2, cy + 2)
        assert t.kind == kind, name
        assert abs(t.x - cx) < 0.6 and abs(t.y - cy) < 0.6, name


def test_re_marking_decides_the_colour_again_in_auto_mode():
    white, black = to_gray(render(PTS, kind="white", seed=0)), to_gray(render(PTS, kind="black", seed=0))
    cx, cy = PTS["epicondyle"]
    t = DotTracker("epicondyle", "auto")
    assert t.init_at(white, cx, cy) and t.kind == "white"
    assert t.init_at(black, cx, cy) and t.kind == "black"    # e.g. the operator changed to black stickers


def test_an_explicit_colour_is_never_changed():
    gray = to_gray(render(PTS, kind="white", seed=0))
    cx, cy = PTS["trochanter"]
    t = DotTracker("trochanter", "black")
    assert t.init_at(gray, cx, cy) is False or t.kind == "black"
    assert t.kind == "black"


def test_through_the_engine_a_failed_click_then_a_good_click():
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), 0.0)
    cx, cy = PTS["trochanter"]
    assert eng.mark_dot("trochanter", cx, cy + 25) is False
    assert eng.trackers.trackers["trochanter"].kind == "auto"
    assert eng.mark_dot("trochanter", cx + 2, cy + 2) is True
    t = eng.trackers.trackers["trochanter"]
    assert t.kind == "white" and abs(t.x - cx) < 0.6 and abs(t.y - cy) < 0.6
