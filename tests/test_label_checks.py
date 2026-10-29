"""GUI-14 (+TRK-07): dots marked in the wrong order or twice are caught before they give plausible but wrong angles."""
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


def mark(swap=(), side="right", hip=90.0, knee=70.0, rotate=0.0, test="belt_squat"):
    """Mark all dots on a synthetic leg; `swap` pairs of names get each other's position. The belt squat has the foot dots (since 0.24.0 the
    knee and hip tests do not)."""
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test(test)
    eng.facing_right = side == "right"
    p = {k: v.copy() for k, v in leg_points(hip_flex=hip, knee_flex=knee, ankle_df=5, facing=side).items()}
    if rotate:                                              # a lying participant: the whole body turned about the picture centre
        c, a = np.array([640.0, 360.0]), np.radians(rotate)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        p = {k: (R @ (v - c)) + c for k, v in p.items()}
    img = render(p, seed=0)
    eng.process(img, 0.0)
    where = dict(p)
    for a, b in swap:
        where[a], where[b] = p[b], p[a]
    results = {n: eng.mark_dot(n, *(where[n] + 2)) for n in eng.active_dots}
    for i in range(8):
        res = eng.process(img, 0.1 + i / 30)
    return eng, res, results


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("rotate", [0.0, 90.0, -90.0, 180.0])
@pytest.mark.parametrize("knee", [10.0, 70.0, 110.0])
def test_a_correct_marking_raises_no_label_warning_in_any_posture(side, rotate, knee):
    eng, res, _ = mark(side=side, rotate=rotate, knee=knee)
    assert eng.marking_problems() == [], eng.marking_problems()


@pytest.mark.parametrize("swap, word", [
    ((("mt5_base", "mt5_head"),), "metatarsal head"),
    ((("epicondyle", "fibular_head"),), "swapped"),
    ((("iliac_crest", "epicondyle"),), "thigh"),
])
def test_swapped_landmarks_are_flagged_and_block_lock(swap, word):
    eng, res, _ = mark(swap=swap)
    problems = eng.marking_problems()
    assert problems and any(word in p for p in problems), problems
    assert any(h.startswith("CHECK LABELS") for h in res.header)
    assert eng.lock_position() is False and "mislabelled" in eng.lock_problem()


def test_the_same_sticker_cannot_be_marked_for_two_landmarks():
    eng = Engine(TESTS, recordings_dir=ROOT / "tests" / "_out")
    eng.set_test("knee_extension")
    p = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    assert eng.mark_dot("epicondyle", *(p["epicondyle"] + 2))
    assert eng.mark_dot("fibular_head", *(p["epicondyle"] + 3)) is False              # the same sticker again
    assert "already marked" in eng.mark_problem and "epicondyle" in eng.mark_problem.lower().replace("lateral femoral ", "")
    assert not eng.trackers.trackers["fibular_head"].initialised                      # nothing was learned from the wrong click
    assert eng.mark_dot("fibular_head", *(p["fibular_head"] + 2))                     # the right one still works


def test_lock_works_again_after_the_labels_are_corrected():
    eng, res, _ = mark(swap=(("mt5_base", "mt5_head"),))
    assert eng.lock_position() is False
    eng.reset_dots()
    p = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    img = render(p, seed=0)
    eng.process(img, 1.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    for i in range(6):
        eng.process(img, 1.1 + i / 30)
    assert eng.marking_problems() == [] and eng.lock_position()


def test_a_test_without_the_foot_has_no_foot_label_checks():
    """0.24.0: the knee and hip tests have no foot dots, so a foot dot cannot be mislabelled there; the other checks still run."""
    eng, res, _ = mark(test="knee_extension")
    assert "mt5_base" not in eng.active_dots and eng.marking_problems() == []
    eng, res, _ = mark(test="knee_extension", swap=(("epicondyle", "fibular_head"),))
    assert eng.marking_problems() and eng.lock_position() is False
