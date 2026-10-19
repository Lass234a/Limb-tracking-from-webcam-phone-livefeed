"""ANG-04 (+GUI-12, DOC-03, TST-02): a 'Participant faces' setting that contradicts the marked dots is caught."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.geometry import KNEE_FACING_MIN_DEG, foot_handedness, knee_bend_handedness, view_handedness  # noqa: E402
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
    assert view_handedness({"iliac_crest": (0, 0), "trochanter": (10, 0)}) is None
    assert view_handedness({"malleolus": (0, 0), "fibular_head": (10, 0), "mt5_base": (50, 0), "mt5_head": (80, 0)}) is None


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("hip", [90.0, 0.0, -20.0])
def test_the_pelvis_alone_does_not_tell_the_side(side, hip):
    """With one iliac crest dot the turn crest -> trochanter -> epicondyle changes sign with the hip angle, so it is not used."""
    p, _ = scene(side, False, hip)
    pelvis_leg = {k: tuple(v) for k, v in p.items() if k in ("iliac_crest", "trochanter", "epicondyle")}
    assert view_handedness(pelvis_leg) is None


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


# ---------------------------------------------------------------- without the foot dots: the bend of the knee (2026-10-19)
KNEE_DOTS = ("trochanter", "epicondyle", "fibular_head", "malleolus")


def knee_scene(side, mirror, knee):
    p = {k: v.copy() for k, v in leg_points(hip_flex=0, knee_flex=knee, ankle_df=5, facing=side).items()}
    if mirror:
        p = {k: v * [-1, 1] + [W - 1, 0] for k, v in p.items()}
    return {k: tuple(v) for k, v in p.items() if k in KNEE_DOTS}


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("knee", [30.0, 90.0, 120.0])
def test_without_the_foot_a_bent_knee_tells_the_view(side, mirror, knee):
    assert knee_bend_handedness(knee_scene(side, mirror, knee)) == (1 if truth_right(side, mirror) else -1)
    assert view_handedness(knee_scene(side, mirror, knee)) == (1 if truth_right(side, mirror) else -1)


@pytest.mark.parametrize("knee", [0.0, 10.0, 20.0])
def test_a_nearly_straight_knee_tells_nothing(knee):
    """A knee can over-straighten by a few degrees, so a bend under KNEE_FACING_MIN_DEG either way is no evidence."""
    assert KNEE_FACING_MIN_DEG == 25.0
    for side in ("right", "left"):
        assert knee_bend_handedness(knee_scene(side, False, knee)) is None


def test_the_foot_wins_when_both_are_there():
    p = {k: tuple(v) for k, v in leg_points(hip_flex=90, knee_flex=70, ankle_df=5).items()}
    assert view_handedness(p) == foot_handedness(p) == 1


def test_lying_face_down_with_faces_right_is_caught_without_the_foot_dots():
    """P11 2026-10-15, hip extension with knee 90 lying face down: 'faces right' was chosen, the knee read -55 and -78 deg and nothing
    warned, because the foot dots were switched off. A face-down view is the mirror of a face-up one."""
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("hip_extension_k90")
    eng.disabled_dots |= {"mt5_base", "mt5_head"}
    eng.reset_dots()
    eng.facing_right = True
    p = {k: v * [-1, 1] + [W - 1, 0] for k, v in leg_points(hip_flex=-5, knee_flex=70, ankle_df=5).items()}
    img = cv2.flip(render(leg_points(hip_flex=-5, knee_flex=70, ankle_df=5), seed=0), 1)
    eng.process(img, 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2)), n
    res = None
    for i in range(8):
        res = eng.process(img, 0.1 + i / 30)
    knee = next(a for a in res.angles if a.name == "knee")
    assert knee.raw_value == pytest.approx(-70.0, abs=1.0)                      # what was recorded on 2026-10-15
    assert eng.facing_mismatch and eng.lock_position() is False
    assert "knee bends the wrong way" in eng.facing_message() and "LEFT" in eng.facing_message()
    eng.facing_right = False
    eng.process(img, 1.0)
    knee = next(a for a in eng._last_angles if a.name == "knee")
    assert not eng.facing_mismatch and knee.raw_value == pytest.approx(70.0, abs=1.0) and eng.lock_position()
