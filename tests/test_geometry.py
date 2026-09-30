import math

import numpy as np
import pytest

from limbtrack.geometry import (interior_angle, resolve_arm, segment_vs_vertical, spec_dots, to_display,
                                vector_angle)


@pytest.mark.parametrize("deg", [10, 30, 60, 90, 120, 150, 179])
def test_interior_angle_known(deg):
    r = math.radians(deg)
    a, b, c = (1, 0), (0, 0), (math.cos(r), math.sin(r))
    assert interior_angle(a, b, c) == pytest.approx(deg, abs=1e-9)
    assert interior_angle(c, b, a) == pytest.approx(deg, abs=1e-9)


def test_interior_angle_straight_and_degenerate():
    assert interior_angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180)
    assert math.isnan(interior_angle((0, 0), (0, 0), (1, 0)))


def test_vertical():
    assert segment_vs_vertical((0, 10), (0, 0)) == pytest.approx(0)      # straight up on screen
    assert segment_vs_vertical((0, 0), (10, 0)) == pytest.approx(90)
    assert segment_vs_vertical((0, 0), (0, 10)) == pytest.approx(180)
    assert segment_vs_vertical((0, 10), (10, 0)) == pytest.approx(45)


def test_vector_angle_signed_and_unsigned():
    assert vector_angle((0, 1), (1, 0)) == pytest.approx(90)
    assert vector_angle((0, 1), (1, 0), signed=True) == pytest.approx(-90)   # image coords: y down
    assert vector_angle((0, 1), (-1, 0), signed=True) == pytest.approx(90)
    assert vector_angle((1, 0), (2, 0)) == pytest.approx(0)
    assert math.isnan(vector_angle((0, 0), (1, 0)))


def test_resolve_arm_variants_and_facing():
    xy = {"a": (0, 0), "b": (10, 0)}
    assert np.allclose(resolve_arm(["a", "b"], xy), (10, 0))
    assert np.allclose(resolve_arm(["a", "b"], xy, facing_right=False), (-10, 0))   # mirrored
    assert np.allclose(resolve_arm({"perp": ["a", "b"]}, xy), (0, 10))              # points down
    assert np.allclose(resolve_arm({"perp": ["b", "a"]}, xy), (0, 10))              # order does not matter
    assert np.allclose(resolve_arm("forward", xy, facing_right=False), (1, 0))      # already in the mirrored frame
    assert resolve_arm(["a", "zzz"], xy) is None
    assert spec_dots({"perp": ["a", "b"]}) == ["a", "b"] and spec_dots("up") == []


def test_to_display():
    assert to_display(180, -1, 180) == 0     # straight knee = 0 flexion
    assert to_display(90, -1, 180) == 90
    assert to_display(90, -1, 90) == 0       # ankle neutral
    assert to_display(70, -1, 90) == 20      # dorsiflexed
