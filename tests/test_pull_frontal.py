"""Strap and anchor dots in the hip abduction / adduction tests (0.25.0): offered like in the other tests, and the angle of pull and the
moment arm are measured about the computed hip centre (protocol.json "pull_joint"), not about the ASIS, which is not the joint."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.geometry import hip_centre_frontal  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402

from tests import test_frontal_engine as F  # noqa: E402  (its scene, picture and ruler helpers)
from tests.test_analysis import torque_trial  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = F.TESTS


def scene(deg=0.0, leg="right", pull_dy=0.0):
    """The frontal scene with a strap dot on the thigh and an anchor dot out to the side: a horizontal rope (pull_dy tilts it)."""
    p = F.scene(deg, leg)
    hjc = np.array(hip_centre_frontal(p["asis"], p["asis_opposite"], 1.0))
    out = -1.0 if leg == "right" else 1.0                                       # picture direction of "towards the tested side"
    p["strap"] = hjc + 0.55 * (p["patella"] - hjc)
    p["anchor"] = p["strap"] + np.array([out * 300.0, pull_dy])
    return p, hjc


def make(tmp_path, deg=0.0, leg="right", test="hip_abduction", tests=TESTS, pull_dy=0.0, ruler=True):
    eng = Engine(tests, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.tested_leg = leg
    eng.set_dot_enabled("strap", True)
    eng.set_dot_enabled("anchor", True)
    p, hjc = scene(deg, leg, pull_dy)
    eng.process(F.pic(p), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    if ruler:
        assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    return eng, p, hjc


# ---- what the protocol offers
@pytest.mark.parametrize("key", ["hip_abduction", "hip_adduction"])
def test_the_frontal_tests_offer_strap_and_anchor_off_until_ticked(tmp_path, key):
    t = TESTS[key]
    assert t.optional_dots == ["strap", "anchor"] and t.all_dots == t.dots + ["strap", "anchor"]
    assert t.dots == ["asis", "asis_opposite", "patella"] and t.pull_joint == "hip_centre"
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(key)
    assert {"strap", "anchor"} <= eng.disabled_dots and not eng.pull_active
    assert eng.pull_segment() == ("hip_centre", "patella", "hip_centre")


def test_the_sagittal_tests_still_measure_about_the_fulcrum_of_their_main_angle():
    assert all(t.pull_joint is None for t in TESTS.values() if t.view == "sagittal")


def test_a_pull_joint_that_is_not_in_the_test_is_refused_with_the_test_named(tmp_path):
    d = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    d["tests"]["hip_abduction"]["pull_joint"] = "malleolus"                      # a real dot, but not one of this test's
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="hip_abduction: pull_joint .malleolus. is not one of the test's dots"):
        load_protocol(path)


# ---- the maths through the engine, worked out by hand
@pytest.mark.parametrize("leg", ["right", "left"])
@pytest.mark.parametrize("deg", [0.0, 20.0])
def test_the_pull_is_measured_about_the_hip_centre(tmp_path, leg, deg):
    eng, p, hjc = make(tmp_path, deg, leg)
    res = F.run(eng, p)
    assert res.pull is not None
    # a horizontal rope: the moment arm is how far the strap is below the hip centre; the thigh leans out by `deg`, the rope pulls straight out
    assert res.pull["moment_arm_px"] == pytest.approx(p["strap"][1] - hjc[1], abs=1.5)
    assert res.pull["moment_arm_cm"] == pytest.approx((p["strap"][1] - hjc[1]) / 10.0, abs=0.2)     # the ruler: 10 px per cm
    assert res.pull["pull_angle_deg"] == pytest.approx(90.0 - deg, abs=0.6)
    assert abs(res.pull["moment_arm_px"] - (p["strap"][1] - p["asis"][1])) > 60.0         # the ASIS would have given a much longer arm
    assert any(line.startswith("PULL ") and " cm" in line for line in res.header)


def test_a_tilted_rope_shortens_the_moment_arm(tmp_path):
    eng, p, hjc = make(tmp_path, 10.0, "right", pull_dy=-300.0)                  # the anchor is 300 px higher than the strap: a rope at 45 deg
    res = F.run(eng, p)
    u = p["anchor"] - p["strap"]
    by_hand = abs(u[0] * (hjc[1] - p["strap"][1]) - u[1] * (hjc[0] - p["strap"][0])) / np.hypot(*u)
    assert res.pull["moment_arm_px"] == pytest.approx(by_hand, abs=1.5)
    assert by_hand < p["strap"][1] - hjc[1] - 20.0


def test_the_pivot_can_be_switched_to_the_asis_in_the_protocol_file(tmp_path):
    d = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    d["tests"]["hip_abduction"]["pull_joint"] = "asis"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    tests = load_protocol(path)
    eng, p, hjc = make(tmp_path / "r", 10.0, "right", tests=tests)
    assert eng.pull_segment() == ("asis", "patella", "asis")
    res = F.run(eng, p)
    assert res.pull["moment_arm_px"] == pytest.approx(p["strap"][1] - p["asis"][1], abs=1.5)


def test_without_the_ruler_there_is_no_hip_centre_so_no_pull_and_the_header_says_why(tmp_path):
    eng, p, _ = make(tmp_path, 10.0, ruler=False)
    res = F.run(eng, p)
    assert res.pull is None
    line = next(h for h in res.header if h.startswith("PULL"))
    assert "no ruler scale" in line and "hip centre" in line


def test_the_pull_is_off_while_the_strap_or_the_anchor_is_off(tmp_path):
    eng, p, _ = make(tmp_path, 10.0)
    eng.set_dot_enabled("anchor", False)
    res = F.run(eng, p)
    assert res.pull is None and not any(h.startswith("PULL") for h in res.header)


def test_the_angles_do_not_change_with_the_two_extra_dots(tmp_path):
    with_dots, p, _ = make(tmp_path / "a", 12.0)
    plain = Engine(TESTS, recordings_dir=tmp_path / "b")
    plain.set_test("hip_abduction")
    plain.tested_leg = "right"
    plain.process(F.pic(p), 0.0)
    for n in plain.active_dots:
        assert plain.mark_dot(n, *(p[n] + 2))
    assert plain.mark_scale_dot(0, *(p["ruler_a"] + 2)) and plain.mark_scale_dot(1, *(p["ruler_b"] + 2))
    a, b = F.run(with_dots, p), F.run(plain, p)
    for name in ("abd_goniometer", "abd_hip_centre"):
        assert F.angle(a, name).raw_value == pytest.approx(F.angle(b, name).raw_value, abs=0.2)
    assert b.pull is None


# ---- recorded, and read back in Review
def record(tmp_path, deg=15.0, hide=()):
    eng, p, hjc = make(tmp_path, deg)
    eng.target = 20.0
    paths = eng.start_trial("P01")
    for i in range(12):
        eng.process(F.render(p, dot_radius=F.DOT_R, seed=i, hide=hide if 4 <= i < 8 else ()), 0.2 + i / 30)
    eng.stop_trial()
    return paths, p, hjc


def test_the_recording_names_the_hip_centre_as_the_joint_of_the_pull(tmp_path):
    paths, p, hjc = record(tmp_path)
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert float(rows[2]["moment_arm_px"]) == pytest.approx(p["strap"][1] - hjc[1], abs=1.5)
    assert float(rows[2]["moment_arm_cm"]) == pytest.approx((p["strap"][1] - hjc[1]) / 10.0, abs=0.2)
    assert float(rows[2]["hip_centre_x"]) == pytest.approx(hjc[0], abs=1.0)           # the pivot is in the file, so another one can be worked out later
    assert {"asis_x", "asis_y", "strap_x", "anchor_x"} <= set(rows[0])
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["optional_dots_used"] == ["strap", "anchor"] and meta["dots_disabled"] == []
    pull = meta["pull"]
    assert pull["segment_from_to"] == ["hip_centre", "patella"] and pull["moment_arm_from_joint_dot"] == "hip_centre"
    assert "hip_centre_x" in pull["joint_is_computed_point"]
    assert meta["software_version"] == "0.25.0"


def test_a_covered_anchor_leaves_the_pull_blank_not_guessed(tmp_path):
    paths, _, _ = record(tmp_path, hide=("anchor",))
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert rows[2]["moment_arm_px"] != "" and rows[6]["moment_arm_px"] == "" and rows[6]["pull_angle_deg"] == ""


def test_review_reads_the_hip_centre_as_a_joint_for_the_torque(tmp_path):
    paths, p, hjc = record(tmp_path)
    tr = Trial(paths["meta.json"])
    assert list(tr.derived_tracks) == ["hip_centre"] and "hip_centre" not in tr.tracks
    assert tr.derived_tracks["hip_centre"][1][3] == pytest.approx(hjc[1], abs=1.0)
    assert tr.pull is not None and np.nanmean(tr.pull["moment_arm_px"]) == pytest.approx(p["strap"][1] - hjc[1], abs=1.5)


def frontal_torque_trial(tmp_path, strap_seen=True):
    """torque_trial's trial (200 N hold) with the geometry of a frontal one: vertical thigh from the hip centre (100, 100) to the patella,
    strap on it at y 250, rope running out to the anchor at the same height: a moment arm of 150 px = 30 cm."""
    tr = torque_trial(tmp_path, strap_seen)
    n = len(tr.t)
    const = lambda x, y: (np.full(n, float(x)), np.full(n, float(y)))              # noqa: E731
    for name in ("epicondyle", "fibular_head", "malleolus"):
        tr.tracks.pop(name, None)
    tr.tracks["patella"] = const(100, 300)
    tr.derived_tracks = {"hip_centre": const(100, 100)}
    tr.meta["pull"] = {"segment_from_to": ["hip_centre", "patella"], "moment_arm_from_joint_dot": "hip_centre"}
    return tr


def test_the_torque_of_a_frontal_trial_uses_the_hip_centre(tmp_path):
    tq = analysis.torque_summary(frontal_torque_trial(tmp_path), 9.0, 12.5)
    assert tq["arm_at_peak_force_cm"] == pytest.approx(30.0, abs=0.1)
    assert tq["peak_nm"] == pytest.approx(200.0 * 0.30, rel=0.03) and tq["pictures_with_stand_in"] == 0


def test_a_frontal_picture_without_the_strap_dot_gets_no_torque_not_a_stand_in(tmp_path):
    # in the sagittal ankle test the distal dot stands in for the strap; here the segment ends at the knee, where the strap is not
    assert analysis.torque_summary(frontal_torque_trial(tmp_path, strap_seen=False), 9.0, 12.5) is None
