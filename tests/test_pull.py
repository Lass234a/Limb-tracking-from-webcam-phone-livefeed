"""Angle of pull and moment arm from the strap and anchor dots. Nothing here is judged or changes an angle."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.pull import compute_pull  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


# ---- the maths, with numbers that can be checked by hand
def test_square_pull_is_90_degrees_and_the_moment_arm_is_the_distance_to_the_line():
    # segment points down the page (+y); strap 60 below the start, rope runs sideways; joint 50 above the segment start
    p = compute_pull(strap=(0, 60), anchor=(100, 60), seg_start=(0, 0), seg_end=(0, 100), fulcrum=(0, -50))
    assert p["pull_angle_deg"] == pytest.approx(90.0)
    assert p["pull_off90_deg"] == pytest.approx(0.0, abs=1e-9)
    assert p["moment_arm_px"] == pytest.approx(110.0)                # fulcrum to the line y = 60
    assert p["strap_joint_px"] == pytest.approx(110.0)
    assert p["moment_arm_cm"] is None and p["strap_joint_cm"] is None   # no scale given


def test_a_tilted_pull_gives_the_angle_and_a_shorter_moment_arm():
    p = compute_pull(strap=(0, 60), anchor=(100, 20), seg_start=(0, 0), seg_end=(0, 100), fulcrum=(0, -50), px_per_cm=5.0)
    assert p["pull_angle_deg"] == pytest.approx(111.801, abs=0.001)   # acos(-40 / 107.7)
    assert p["pull_off90_deg"] == pytest.approx(21.801, abs=0.001)
    assert p["moment_arm_px"] == pytest.approx(102.13, abs=0.01)      # 11000 / 107.7
    assert p["moment_arm_cm"] == pytest.approx(102.13 / 5.0, abs=0.01)
    assert p["strap_joint_cm"] == pytest.approx(22.0)
    q = compute_pull(strap=(0, 60), anchor=(100, 100), seg_start=(0, 0), seg_end=(0, 100), fulcrum=(0, -50))
    assert q["pull_angle_deg"] < 90 < p["pull_angle_deg"]             # pulling towards the feet vs towards the body


def test_degenerate_input_gives_none():
    assert compute_pull((5, 5), (5, 5), (0, 0), (0, 100), (0, -50)) is None      # strap and anchor on top of each other
    assert compute_pull((0, 60), (100, 60), (3, 3), (3, 3), (0, -50)) is None    # zero-length segment


# ---- through the engine
def scene():
    p = dict(leg_points(hip_flex=70, knee_flex=60, ankle_df=5))
    mal, fib = np.array(p["malleolus"]), np.array(p["fibular_head"])
    p["strap"] = mal + 0.35 * (fib - mal) + np.array([0.0, 0.0])               # on the shank, away from both ends
    p["anchor"] = p["strap"] + np.array([260.0, -90.0])                         # rope running up and away
    return p


def make(tmp_path, test="knee_extension", scale=None):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.target = 60.0
    eng.set_dot_enabled("strap", True)
    eng.set_dot_enabled("anchor", True)
    p = scene()
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    return eng, p


def expected(p, seg, fulcrum, ppc=None):
    return compute_pull(p["strap"], p["anchor"], p[seg[0]], p[seg[1]], p[fulcrum], ppc)


def test_the_engine_measures_the_pull_against_the_shank_in_a_knee_test(tmp_path):
    eng, p = make(tmp_path)
    for i in range(10):
        res = eng.process(render(p, seed=i), 0.1 + i / 30)
    want = expected(p, ("fibular_head", "malleolus"), "epicondyle")
    assert eng.pull_segment() == ("fibular_head", "malleolus", "epicondyle")
    assert res.pull["pull_angle_deg"] == pytest.approx(want["pull_angle_deg"], abs=0.5)
    assert res.pull["moment_arm_px"] == pytest.approx(want["moment_arm_px"], abs=1.0)
    assert any(line.startswith("PULL ") and "no scale" in line for line in res.header)


def test_the_segment_follows_the_test_thigh_for_hip_and_foot_for_ankle(tmp_path):
    eng, _ = make(tmp_path, "hip_extension_k90")
    assert eng.pull_segment() == ("trochanter", "epicondyle", "trochanter")
    eng, _ = make(tmp_path, "ankle_dorsiflexion")
    assert eng.pull_segment() == ("mt5_base", "mt5_head", "malleolus")


def test_cm_values_need_the_ruler_scale(tmp_path):
    eng, p = make(tmp_path)
    p["ruler_a"], p["ruler_b"] = np.array([120.0, 670.0]), np.array([240.0, 670.0])
    eng.scale.length_cm = 20.0
    eng.process(render(p, seed=0), 0.0)
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    for i in range(10):
        res = eng.process(render(p, seed=i), 0.1 + i / 30)
    want = expected(p, ("fibular_head", "malleolus"), "epicondyle", 6.0)
    assert res.pull["moment_arm_cm"] == pytest.approx(want["moment_arm_cm"], abs=0.3)
    assert any("moment arm" in line and " cm" in line for line in res.header)


def test_pull_changes_no_angle_and_is_off_without_the_strap_dots(tmp_path):
    eng, p = make(tmp_path)
    res = eng.process(render(p, seed=1), 0.1)
    assert res.pull is not None
    plain = Engine(TESTS, recordings_dir=tmp_path / "plain")
    plain.set_test("knee_extension")
    plain.target = 60.0
    q = {k: v for k, v in p.items() if k not in ("strap", "anchor")}
    plain.process(render(q, seed=0), 0.0)
    for n in plain.active_dots:
        assert plain.mark_dot(n, *(q[n] + 2))
    for i in range(10):
        r_plain = plain.process(render(q, seed=i), 0.1 + i / 30)
        r_pull = eng.process(render(p, seed=i), 0.1 + i / 30)
    assert r_plain.pull is None and not any(line.startswith("PULL") for line in r_plain.header)
    assert np.allclose([a.raw_value for a in r_plain.angles], [a.raw_value for a in r_pull.angles], atol=0.2)


def test_csv_and_meta_carry_the_pull_only_when_the_strap_dots_are_used(tmp_path):
    eng, p = make(tmp_path / "with")
    paths = eng.start_trial("P01")
    for i in range(30):
        eng.process(render(p, seed=i, hide=["anchor"] if 10 <= i < 15 else []), 0.2 + i / 30)
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert rows[0]["pull_angle_deg"] != "" and float(rows[0]["moment_arm_px"]) > 0
    assert rows[12]["pull_angle_deg"] == "" and rows[12]["moment_arm_px"] == ""           # anchor covered: blank, not guessed
    assert rows[0]["moment_arm_cm"] == ""                                                  # no ruler scale was set
    meta = json.loads(paths["meta.json"].read_text())["pull"]
    assert meta["segment_from_to"] == ["fibular_head", "malleolus"] and meta["moment_arm_from_joint_dot"] == "epicondyle"

    plain = Engine(TESTS, recordings_dir=tmp_path / "without")
    plain.set_test("knee_extension")
    q = {k: v for k, v in p.items() if k not in ("strap", "anchor")}
    plain.process(render(q, seed=0), 0.0)
    for n in plain.active_dots:
        assert plain.mark_dot(n, *(q[n] + 2))
    paths = plain.start_trial("P01")
    plain.process(render(q, seed=1), 0.1)
    plain.stop_trial()
    assert "pull_angle_deg" not in open(paths["data.csv"]).readline()
    assert json.loads(paths["meta.json"].read_text())["pull"] is None
