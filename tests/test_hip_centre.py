"""The hip joint centre of a frontal picture, computed from the two ASIS and the scale (geometry.hip_centre_frontal)."""
import math

import numpy as np
import pytest

from limbtrack.geometry import (DERIVED_KINDS, HARRINGTON_DOWN, HARRINGTON_SIDEWAYS, compute_derived, derived_input_dots,
                                hip_centre_frontal, pelvic_width_mm)
from limbtrack.synthetic import frontal_points

SPEC = {"kind": "hip_centre_frontal", "tested_asis": "asis", "other_asis": "asis_opposite"}


def test_known_answer_right_leg_at_one_pixel_per_millimetre():
    # pelvic width 250 mm: sideways 0.33 * 250 + 7.3 = 89.8 mm towards the tested side, down 0.30 * 250 + 10.9 = 85.9 mm
    tested, other = (515.0, 150.0), (765.0, 150.0)                # right leg: the tested ASIS is on the picture's left
    x, y = hip_centre_frontal(tested, other, 1.0)
    assert (x, y) == pytest.approx((640.0 - 89.8, 150.0 + 85.9), abs=1e-9)


def test_the_left_leg_is_the_mirror_image():
    x, y = hip_centre_frontal((765.0, 150.0), (515.0, 150.0), 1.0)
    assert (x, y) == pytest.approx((640.0 + 89.8, 150.0 + 85.9), abs=1e-9)


def test_the_equation_is_in_millimetres_so_the_scale_matters_not_the_picture_size():
    a, b = np.array([515.0, 150.0]), np.array([765.0, 150.0])
    one = np.array(hip_centre_frontal(a, b, 1.0))
    two = np.array(hip_centre_frontal(2 * a, 2 * b, 2.0))        # the same person photographed twice as large
    assert two == pytest.approx(2 * one, abs=1e-9)
    wide = hip_centre_frontal((590.0, 150.0), (690.0, 150.0), 0.4)           # 100 px at 0.4 px/mm = the same 250 mm pelvis
    assert wide == pytest.approx((640.0 - 89.8 * 0.4, 150.0 + 85.9 * 0.4), abs=1e-9)


@pytest.mark.parametrize("roll", [-35.0, -10.0, 0.0, 17.0, 40.0])
def test_a_rolled_picture_rolls_the_hip_centre_with_it(roll):
    """The pelvis axes come from the two ASIS, so a tilted camera or a tilted pelvis moves the point with the pelvis."""
    base = frontal_points(0.0)
    rolled = frontal_points(0.0, roll_deg=roll)
    m = np.array([640.0, 150.0])
    c0 = np.array(hip_centre_frontal(base["asis"], base["asis_opposite"], 1.0)) - m
    c1 = np.array(hip_centre_frontal(rolled["asis"], rolled["asis_opposite"], 1.0)) - m
    a = math.radians(roll)
    expected = np.array([math.cos(a) * c0[0] - math.sin(a) * c0[1], math.sin(a) * c0[0] + math.cos(a) * c0[1]])
    assert c1 == pytest.approx(expected, abs=1e-9)


def test_down_is_the_side_of_the_picture_s_bottom():
    # a pelvis whose line runs straight up the picture (a person lying down seen from above): "down" must still be a defined side
    x, y = hip_centre_frontal((300.0, 100.0), (300.0, 350.0), 1.0)
    assert math.isfinite(x) and math.isfinite(y)


@pytest.mark.parametrize("scale", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_no_usable_scale_gives_no_hip_centre(scale):
    assert hip_centre_frontal((515.0, 150.0), (765.0, 150.0), scale) is None
    assert pelvic_width_mm((515.0, 150.0), (765.0, 150.0), scale) is None


def test_the_same_point_twice_gives_no_hip_centre():
    assert hip_centre_frontal((600.0, 150.0), (600.0, 150.0), 1.0) is None


def test_the_sideways_term_agrees_with_the_cadaver_study_used_as_a_cross_check():
    """Seidel et al. (J Biomech 1995;28:995): the hip centre lies 14 % of the pelvic width medial to each ASIS, i.e. 0.5 PW - 0.14 PW from
    the midpoint. At PW = 250 mm that is 90 mm; the Harrington term gives 89.8 mm. (The check that the constants were typed right: the
    original table of Harrington et al. has not been read, see geometry.py.)"""
    pw = 250.0
    harrington = HARRINGTON_SIDEWAYS[0] * pw + HARRINGTON_SIDEWAYS[1]
    assert harrington == pytest.approx(0.5 * pw - 0.14 * pw, abs=1.0)
    assert HARRINGTON_DOWN[0] * pw + HARRINGTON_DOWN[1] == pytest.approx(85.9, abs=1e-9)


def test_derived_points_come_only_from_dots_that_are_there_and_a_known_scale():
    pts = frontal_points(0.0)
    xy = {k: tuple(v) for k, v in pts.items()}
    assert compute_derived({"hip_centre": SPEC}, xy, 1.0).keys() == {"hip_centre"}
    assert compute_derived({"hip_centre": SPEC}, xy, None) == {}                    # no ruler scale
    assert compute_derived({"hip_centre": SPEC}, {"asis": xy["asis"]}, 1.0) == {}    # the other ASIS is lost
    assert compute_derived({}, xy, 1.0) == {}


def test_a_computed_point_is_made_of_these_dots():
    assert derived_input_dots(SPEC) == ["asis", "asis_opposite"]
    assert derived_input_dots({"kind": "nonsense"}) == []
    assert set(DERIVED_KINDS) == {"hip_centre_frontal"}


def test_the_synthetic_frontal_pose_turns_the_thigh_about_the_hip_centre():
    for deg in (-20.0, 0.0, 10.0, 20.0):
        p = frontal_points(deg, tested_leg="left", roll_deg=12.0)
        hjc = np.array(hip_centre_frontal(p["asis"], p["asis_opposite"], 1.0))
        thigh = p["patella"] - hjc
        toward_other = p["asis_opposite"] - p["asis"]
        interior = math.degrees(math.acos(float(np.dot(thigh, toward_other)) / (np.hypot(*thigh) * np.hypot(*toward_other))))
        assert interior - 90.0 == pytest.approx(deg, abs=1e-9)
