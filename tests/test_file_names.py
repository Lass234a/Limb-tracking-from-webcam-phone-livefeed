"""0.20.0: the position number (the SOP's test order, 1-21) and the trial number are in every trial's file names and meta.json.

Names until 0.19.0: P01_knee_extension_110deg_20261026-101500_raw.mp4; from 0.20.0: P01_03_knee_extension_110deg_T1_20261026-101500_raw.mp4.
Both are counted, so trials of a participant recorded before and after the change still add up."""
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol, plan_positions, position_number  # noqa: E402
from limbtrack.recorder import count_trials, trial_metas, trial_name  # noqa: E402
from limbtrack.review import Trial, after_trial_check  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")

# The SOP's test order (v0.18, "Test order"), with the decisions of 2026-10-26: No. -> (test, target)
SOP_ORDER = {1: ("knee_extension", 20), 2: ("knee_extension", 70), 3: ("knee_extension", 110), 4: ("knee_flexion", 20), 5: ("knee_flexion", 90),
             6: ("hip_extension_k0", 0), 7: ("hip_extension_k0", 45), 8: ("hip_extension_k90", 0), 9: ("hip_extension_k90", 90),
             10: ("hip_flexion", 0), 11: ("hip_flexion", 90), 12: ("hip_abduction", 0), 13: ("hip_abduction", 20), 14: ("hip_adduction", 0),
             15: ("hip_adduction", 20), 16: ("ankle_plantarflexion", 0), 17: ("ankle_plantarflexion", -20), 18: ("ankle_dorsiflexion", 0),
             19: ("ankle_dorsiflexion", -20), 20: ("belt_squat", 90), 21: ("belt_squat", 60)}


def test_the_position_numbers_are_the_sops():
    assert {i: p for i, p in enumerate(plan_positions(TESTS), 1)} == SOP_ORDER
    assert all(position_number(TESTS, k, float(t)) == i for i, (k, t) in SOP_ORDER.items())
    assert position_number(TESTS, "knee_extension", 33.0) is None                      # a typed target outside the plan
    assert trial_name("P01", "knee_extension", 110.0, 3, 2, "20261026-101500") == "P01_03_knee_extension_110deg_T2_20261026-101500"
    assert trial_name("P01", "ankle_dorsiflexion", -20.0, 19, 1, "20261026-101500") == "P01_19_ankle_dorsiflexion_-20deg_T1_20261026-101500"
    assert trial_name("P01", "knee_extension", 33.0, None, 1, "20261026-101500").startswith("P01_00_")


def record(tmp_path, test, target, participant="P01", frames=4):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.target = target
    q = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(q, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(q[n] + 2))
    paths = eng.start_trial(participant)
    for i in range(frames):
        eng.process(render(q, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    return paths


def test_a_trial_is_named_with_its_position_and_trial_number(tmp_path):
    a = record(tmp_path, "knee_extension", 110.0)
    b = record(tmp_path, "knee_extension", 110.0)                                      # the same second: still its own files, trial 2
    c = record(tmp_path, "knee_extension", 20.0)
    assert a["raw.mp4"].name.startswith("P01_03_knee_extension_110deg_T1_") and a["raw.mp4"].name.endswith("_raw.mp4")
    assert b["meta.json"].name.startswith("P01_03_knee_extension_110deg_T2_")
    assert c["meta.json"].name.startswith("P01_01_knee_extension_20deg_T1_")
    assert {k for k in a} >= {"raw.mp4", "data.csv", "meta.json"} and all(p.name.startswith("P01_03_knee_extension_110deg_T1_") for p in a.values())
    for paths, pos, n in ((a, 3, 1), (b, 3, 2), (c, 1, 1)):
        meta = json.loads(paths["meta.json"].read_text())
        assert (meta["position_number"], meta["trial_number"]) == (pos, n)
    assert count_trials(tmp_path, "P01", "knee_extension", 110.0) == 2
    assert [p.name for p in trial_metas(tmp_path, "P01", "knee_extension", 110.0)] == [a["meta.json"].name, b["meta.json"].name]
    off = record(tmp_path, "knee_extension", 33.0)
    assert off["meta.json"].name.startswith("P01_00_knee_extension_33deg_T1_")
    assert json.loads(off["meta.json"].read_text())["position_number"] is None


def test_names_from_before_and_after_the_change_are_counted_together_in_time_order(tmp_path):
    folder = tmp_path / "P01"
    folder.mkdir()
    old = folder / "P01_knee_extension_110deg_20261020-090000_meta.json"            # 0.19.0 and earlier
    old.write_text("{}")
    assert count_trials(tmp_path, "P01", "knee_extension", 110.0) == 1
    new = record(tmp_path, "knee_extension", 110.0)
    assert new["meta.json"].name.startswith("P01_03_knee_extension_110deg_T2_")      # the old trial counts: this is trial 2
    assert [p.name for p in trial_metas(tmp_path, "P01", "knee_extension", 110.0)] == [old.name, new["meta.json"].name]
    (folder / "P01_09_hip_extension_k90_90deg_T1_20261026-090000_meta.json").write_text("{}")
    (folder / "P01_07_hip_extension_k0_45deg_T1_20261026-090100_meta.json").write_text("{}")
    assert count_trials(tmp_path, "P01", "hip_extension_k0", 45.0) == 1 and count_trials(tmp_path, "P01", "hip_extension_k0", 90.0) == 0
    assert count_trials(tmp_path, "P01", "hip_extension_k90", 90.0) == 1
    assert count_trials(tmp_path, "P01", "knee_extension", 10.0) == 0                 # 110 is not 10


def test_the_export_and_the_10_percent_check_read_the_new_names(tmp_path):
    a = record(tmp_path, "knee_flexion", 90.0)
    time.sleep(1.1)
    b = record(tmp_path, "knee_flexion", 90.0)
    rows = [analysis.session_row(Trial(p["meta.json"])) for p in (a, b)]
    assert [(r["position_number"], r["trial_number"]) for r in rows] == [(5, 1), (5, 2)]
    assert analysis.SESSION_COLUMNS[:3] == ["trial", "position_number", "trial_number"]
    res = after_trial_check(b["meta.json"])
    assert res["n"] == 2 and res["trial"] == b["meta.json"].name[: -len("_meta.json")]
