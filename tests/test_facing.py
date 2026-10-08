"""ANG-04 (+GUI-12, DOC-03, TST-02): a 'Participant faces' setting that contradicts the marked dots is caught."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.geometry import view_handedness  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
W = 1280


def scene(side, mirror, hip=90.0):
    p = {k: v.copy() for k, v in leg_points(hip_flex=hip, knee_flex=70, ankle_df=5, pelvic_tilt=4, facing=side).items()}
    img = render(p, seed=0)
    if mirror:
        img = cv2.flip(img, 1)
        p = {k: v * [-1, 1] + [W - 1, 0] for k, v in p.items()}
    return p, img


def truth_right(side, mirror):
    """Does the picture show the participant facing right? (a left-leg view is the mirror image of a right-leg view)"""
    return (side == "right") != mirror


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("hip", [90.0, 0.0, -20.0])
def test_the_view_is_recognised_for_either_leg_with_or_without_mirror(side, mirror, hip):
    p, _ = scene(side, mirror, hip)
    xy = {k: tuple(v) for k, v in p.items()}
    assert view_handedness(xy) == (1 if truth_right(side, mirror) else -1)
    foot_only = {k: v for k, v in xy.items() if k in ("malleolus", "fibular_head", "mt5_base", "mt5_head")}
    assert view_handedness(foot_only) == (1 if truth_right(side, mirror) else -1)   # the foot alone tells the same


def test_too_few_or_collinear_dots_give_no_answer():
    assert view_handedness({"asis": (0, 0), "psis": (10, 0)}) is None
    assert view_handedness({"asis": (0, 0), "psis": (100, 0), "trochanter": (50, 0)}) is None


def engine_for(side, mirror, setting_right):
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("knee_extension")
    eng.facing_right = setting_right
    p, img = scene(side, mirror)
    eng.process(img, 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    res = None
    for i in range(10):
        res = eng.process(img, 0.1 + i / 30)
    return eng, res, p


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("mirror", [False, True])
def test_a_correct_setting_raises_no_warning_and_gives_the_right_hip(side, mirror):
    eng, res, _ = engine_for(side, mirror, truth_right(side, mirror))
    assert not eng.facing_mismatch and not any("Participant faces" in h for h in res.header)
    hip = next(a for a in res.angles if a.name == "hip")
    assert hip.raw_value == pytest.approx(90.0, abs=0.7)
    assert eng.lock_position()


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("mirror", [False, True])
def test_a_wrong_setting_is_flagged_and_blocks_lock(side, mirror):
    eng, res, _ = engine_for(side, mirror, not truth_right(side, mirror))
    assert eng.facing_mismatch
    assert any("Participant faces" in h for h in res.header)
    assert eng.lock_position() is False
    assert "Participant faces" in eng.lock_problem() and "hip" in eng.lock_problem()
    hip = next(a for a in res.angles if a.name == "hip")
    assert hip.raw_value == pytest.approx(-90.0, abs=0.7)           # the sign the warning is about


def test_the_warning_clears_as_soon_as_the_setting_is_corrected():
    eng, res, p = engine_for("right", False, False)
    assert eng.facing_mismatch
    eng.facing_right = True
    _, img = scene("right", False)
    eng.process(img, 1.0)
    assert not eng.facing_mismatch and eng.lock_position()


def test_the_detected_side_is_saved_with_the_recording():
    import json
    eng, _, p = engine_for("left", False, False)
    eng.lock_position()
    paths = eng.start_trial("P01")
    _, img = scene("left", False)
    eng.process(img, 2.0)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["facing"] == "left" and meta["facing_detected"] == "left"
