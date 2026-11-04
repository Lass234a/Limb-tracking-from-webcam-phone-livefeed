"""0.34.0 (F6): the foot dots and the ankle angle are back in the knee flexion test (the gastrocnemius crosses the ankle). Table 2 gives no required
ankle angle for knee flexion, so the ankle is shown and saved but never judged: the new protocol.json key `unjudged_angles` says so.

Checks: what the protocol file now says, what the validator refuses, that no other test's dots or angles changed, that an ankle which moves
after Lock cannot make `all_ok` 0 (and does without the key), the saved trial, and Review."""
import copy
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import SimulatedForce  # noqa: E402
from limbtrack.protocol import ProtocolError, load_protocol, plan_positions, validate_protocol  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
TESTS = load_protocol(ROOT / "protocol.json")

FIVE = ["iliac_crest", "trochanter", "epicondyle", "fibular_head", "malleolus"]
SEVEN = FIVE + ["mt5_base", "mt5_head"]
FRONTAL = ["asis", "asis_opposite", "patella"]
# The dots and angles of every test: only knee flexion has changed since 0.33.0 (it had FIVE and knee + hip).
EXPECTED = {
    "knee_extension": (FIVE, ["knee", "hip"]),
    "knee_flexion": (SEVEN, ["knee", "hip", "ankle"]),
    "hip_extension_k0": (FIVE, ["hip", "knee"]),
    "hip_extension_k90": (FIVE, ["hip", "knee"]),
    "hip_flexion": (FIVE, ["hip", "knee"]),
    "hip_abduction": (FRONTAL, ["abd_goniometer", "abd_hip_centre"]),
    "hip_adduction": (FRONTAL, ["abd_goniometer", "abd_hip_centre"]),
    "ankle_plantarflexion": (SEVEN, ["ankle", "knee", "hip"]),
    "ankle_dorsiflexion": (SEVEN, ["ankle", "knee", "hip"]),
    "belt_squat": (SEVEN, ["knee", "hip", "ankle"]),
}


# ------------------------------------------------------------------------------------------------ the protocol file
def test_knee_flexion_has_the_foot_dots_and_the_ankle_and_nothing_else_changed():
    assert list(TESTS) == list(EXPECTED)
    for key, (dots, angles) in EXPECTED.items():
        t = TESTS[key]
        assert list(t.dots) == dots, key
        assert [a.name for a in t.angles] == angles, key
    kf = TESTS["knee_flexion"]
    assert kf.primary.name == "knee" and kf.targets == [20, 90] and kf.neighbour_targets == {"hip": 90}       # the rest of the test is as it was
    assert kf.optional_dots == TESTS["knee_extension"].optional_dots == ["strap", "anchor"]
    assert plan_positions(TESTS) == [(k, tg) for k, t in TESTS.items() for tg in (t.targets or [None])] and len(plan_positions(TESTS)) == 21     # 19 + 2 belt squat: the plan is unchanged


def test_the_ankle_in_knee_flexion_is_not_judged_and_in_the_ankle_tests_it_is():
    flags = {key: {a.name: a.judged for a in t.angles} for key, t in TESTS.items()}
    assert flags["knee_flexion"] == {"knee": True, "hip": True, "ankle": False}
    assert flags["ankle_plantarflexion"]["ankle"] is True and flags["ankle_dorsiflexion"]["ankle"] is True and flags["belt_squat"]["ankle"] is True
    assert all(v for key, f in flags.items() if key not in ("knee_flexion", "hip_abduction", "hip_adduction") for v in f.values())
    assert [k for k, t in TESTS.items() if "ankle" in t.neighbour_targets] == []                              # Table 2 gives no required ankle angle anywhere
    # one definition: the same arms and sign as in the ankle tests, only the judging differs
    a = next(x for x in TESTS["knee_flexion"].angles if x.name == "ankle")
    b = next(x for x in TESTS["ankle_dorsiflexion"].angles if x.name == "ankle")
    assert (a.vectors, a.sign, a.offset, a.fulcrum, a.label) == (b.vectors, b.sign, b.offset, b.fulcrum, b.label)


def test_the_file_on_disk_passes_the_checker_and_the_other_tests_are_byte_for_byte_as_before():
    assert validate_protocol(PROTOCOL) == []
    committed = _committed_protocol()
    if committed is None:
        pytest.skip("no git history to compare with")
    for key, t in committed["tests"].items():
        if key == "knee_flexion" or key.startswith("_"):
            continue
        assert PROTOCOL["tests"][key] == t, key                                                               # not one word of another test changed
    old, new = committed["tests"]["knee_flexion"], PROTOCOL["tests"]["knee_flexion"]
    assert old["angles"] == ["knee", "hip"] and new["angles"] == ["knee", "hip", "ankle"] and new["unjudged_angles"] == ["ankle"]
    assert {k: v for k, v in new.items() if k not in ("angles", "unjudged_angles", "_note")} == {k: v for k, v in old.items() if k not in ("angles", "_note")}
    added_in_0_35_0 = ("study_settings", "_study_settings_note")                                           # the frozen-settings block (F13) is not about this test
    assert {k: v for k, v in PROTOCOL.items() if k != "tests" and k not in added_in_0_35_0} == {k: v for k, v in committed.items() if k != "tests"}       # angle library, dots, pairs untouched


def _committed_protocol():
    """protocol.json as it was at 0.33.0 (commit f94a9c4), or None when git cannot say."""
    import subprocess
    try:
        out = subprocess.run(["git", "show", "f94a9c4:protocol.json"], cwd=ROOT, capture_output=True, check=True, timeout=30).stdout
        return json.loads(out.decode("utf-8"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ the checker
def broken(**changes):
    d = copy.deepcopy(PROTOCOL)
    d["tests"]["knee_flexion"].update(changes)
    return validate_protocol(d)


def test_the_checker_refuses_unjudged_angles_that_make_no_sense():
    assert broken() == []
    assert broken(unjudged_angles=[]) == []
    errs = broken(unjudged_angles=["anckle"])
    assert len(errs) == 1 and "knee_flexion.unjudged_angles" in errs[0] and "not one of this test's angles" in errs[0] and 'did you mean "ankle"' in errs[0]
    assert any("main joint" in e for e in broken(unjudged_angles=["knee"]))                                  # the main joint is always judged
    assert any("neighbour_targets" in e for e in broken(unjudged_angles=["hip"]))                           # hip has a required angle: it would never be checked
    assert any("must be a list" in e for e in broken(unjudged_angles="ankle"))
    assert any("must be a list" in e for e in broken(unjudged_angles=[3]))
    d = copy.deepcopy(PROTOCOL)
    d["tests"]["knee_flexion"]["neighbour_targets"]["ankle"] = 0                                             # a required ankle angle AND not judged
    assert any("unjudged_angles" in e and "neighbour_targets" in e for e in validate_protocol(d))
    d = copy.deepcopy(PROTOCOL)
    d["tests"]["knee_flexion"]["unjudged_angle"] = ["ankle"]                                                  # a typing slip in the key itself
    assert any('unknown field "unjudged_angle"' in e and "unjudged_angles" in e for e in validate_protocol(d))


def test_load_protocol_refuses_a_bad_key_with_the_files_name(tmp_path):
    d = copy.deepcopy(PROTOCOL)
    d["tests"]["knee_flexion"]["unjudged_angles"] = ["knee"]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ProtocolError, match="unjudged_angles"):
        load_protocol(path)
    d["tests"]["knee_flexion"]["unjudged_angles"] = ["ankle"]                                                  # and loads the good one from a file
    path.write_text(json.dumps(d), encoding="utf-8")
    assert [a.judged for a in load_protocol(path)["knee_flexion"].angles] == [True, True, False]
    del d["tests"]["knee_flexion"]["unjudged_angles"]                                                          # without the key the ankle is judged like any neighbour
    path.write_text(json.dumps(d), encoding="utf-8")
    assert [a.judged for a in load_protocol(path)["knee_flexion"].angles] == [True, True, True]


# ------------------------------------------------------------------------------------------------ the engine
def pose(ankle):
    return leg_points(hip_flex=90, knee_flex=20, ankle_df=ankle)


def ready_engine(tmp_path, tests=TESTS, ankle=5.0):
    eng = Engine(tests, recordings_dir=tmp_path)
    eng.set_test("knee_flexion")
    p = pose(ankle)
    eng.process(render(p, seed=0), 0.0)
    assert eng.next_dot_to_mark() == "iliac_crest" and len(eng.active_dots) == 7
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    for i in range(6):
        res = eng.process(render(p, seed=i), 0.1 + i / 30)
    return eng, res


def angle(res, name):
    return next(a for a in res.angles if a.name == name)


def test_the_ankle_is_shown_but_has_no_reference_before_and_after_lock(tmp_path):
    eng, res = ready_engine(tmp_path)
    ank = angle(res, "ankle")
    assert ank.raw_value == pytest.approx(5.0, abs=1.0) and ank.reference is None and ank.judged is False and ank.status != "out"
    assert angle(res, "hip").reference == 90.0 and angle(res, "knee").reference == 20.0                         # the others are judged as before
    assert eng.lock_position() and "ankle" not in eng._refs and set(eng._refs) == {"hip"}                      # the lock stores only what is judged
    p = pose(30.0)
    for i in range(6):
        res = eng.process(render(p, seed=i), 1.0 + i / 30)
    ank = angle(res, "ankle")
    assert ank.raw_value == pytest.approx(30.0, abs=1.0) and ank.reference is None and ank.raw_status != "out"        # it moved 25 deg: nothing says so


def test_lock_works_with_the_foot_dots_lost_because_the_ankle_is_not_judged(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_flexion")
    p = pose(5.0)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    for i in range(6):
        eng.process(render(p, seed=i, hide=("mt5_head",)), 0.1 + i / 30)                                      # the toe dot is covered
    assert eng.lock_position()                                                                                  # ... and the position still locks


def trial_rows(tmp_path, tests):
    eng, _ = ready_engine(tmp_path, tests)
    assert eng.lock_position()
    paths = eng.start_trial("P01")
    for i in range(14):
        eng.process(render(pose(5.0 if i < 5 else 28.0), seed=i), 1.0 + i / 30)                                 # the foot swings 23 deg after five pictures
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))                                           # the real file
    return eng, paths, rows


def test_a_moving_ankle_does_not_make_all_ok_zero_and_it_does_without_the_key(tmp_path):
    eng, paths, rows = trial_rows(tmp_path / "unjudged", TESTS)
    assert len(rows) == 14
    assert all(r["ankle_ref"] == "" and r["ankle_dev"] == "" and r["ankle_ok"] == "" for r in rows)            # shown, not judged
    ank = [float(r["ankle_deg"]) for r in rows]
    assert ank[0] == pytest.approx(5.0, abs=1.0) and ank[-1] == pytest.approx(28.0, abs=1.0)                  # the angle is saved, raw
    assert all(r["all_ok"] == "1" for r in rows), [r["all_ok"] for r in rows]                                  # the knee and hip are in their limits throughout
    d = copy.deepcopy(PROTOCOL)                                                                                 # the same movement with the ankle judged against the lock
    del d["tests"]["knee_flexion"]["unjudged_angles"]
    path = tmp_path / "judged_protocol.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    _, _, rows2 = trial_rows(tmp_path / "judged", load_protocol(path))
    assert any(r["ankle_ref"] != "" for r in rows2) and any(r["all_ok"] == "0" for r in rows2)                 # the key is what keeps the trial from failing
    assert [r["knee_deg"] for r in rows[:3]] and abs(float(rows[-1]["knee_deg"]) - 20.0) < 1.5


def test_the_saved_trial_names_the_foot_dots_and_says_the_ankle_is_not_judged(tmp_path):
    eng, paths, rows = trial_rows(tmp_path, TESTS)
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["dots_used"] == SEVEN and meta["test_key"] == "knee_flexion"
    angles = {a["name"]: a for a in meta["angles"]}
    assert list(angles) == ["knee", "hip", "ankle"] and angles["ankle"]["judged"] is False and angles["knee"]["judged"] is True
    assert meta["neighbour_targets_deg"] == {"hip": 90} and "ankle" not in meta["locked_references_deg"]
    assert meta["position_number"] == 4 and "changed_during_recording" not in meta
    assert {"mt5_base_x", "mt5_head_y", "ankle_deg"} <= set(rows[0])


# ------------------------------------------------------------------------------------------------ Review
def force_trial(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_flexion")
    p = pose(5.0)
    eng.process(render(p, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    force = SimulatedForce(rest_s=0.3, ramp_s=0.6, hold_s=1.2, peak_n=400.0)
    force.start()
    eng.force = force
    time.sleep(0.3)
    paths = eng.start_trial("P01")
    for i in range(90):
        eng.process(render(pose(5.0 + 0.2 * i), seed=i % 5), time.perf_counter())                              # the ankle drifts 18 deg during the trial
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    return paths


def test_review_opens_a_knee_flexion_trial_with_the_ankle_and_exports_it_as_an_other_angle(tmp_path):
    app = QApplication.instance() or QApplication([])
    paths = force_trial(tmp_path)
    tr = Trial(paths["meta.json"])
    assert [a["name"] for a in tr.meta["angles"]] == ["knee", "hip", "ankle"]
    row = analysis.session_row(tr)
    assert row["test_key"] == "knee_flexion" and row["main_angle"]
    others = [row["angle2_name"], row["angle3_name"]]
    assert len(set(others)) == 2 and sum("ankle" in n.lower() for n in others) == 1, others                    # the ankle is an 'other angle at peak'
    ankle_col = "angle2" if "ankle" in row["angle2_name"].lower() else "angle3"
    assert isinstance(row[f"{ankle_col}_at_peak_force_deg"], float)                                            # a number from the saved dots
    tab = ReviewTab(tmp_path)
    try:
        tab.load(paths["meta.json"])
        text = tab.angle_tree.text(everything=True)
        assert "Ankle" in text and "Knee" in text and "Hip" in text                                            # the table lists it with its dots
        assert "Mt5" in text.title() or "MT5" in text or "5th" in text.lower() or "mt5" in text.lower()
    finally:
        tab.shutdown()
    assert app is not None


def test_an_old_knee_flexion_trial_without_the_foot_still_opens_and_exports(tmp_path):
    """A trial recorded by 0.33.0 (five dots, knee and hip only) must open and export; build one by running the old protocol."""
    committed = _committed_protocol()
    if committed is None:
        pytest.skip("no git history to compare with")
    old = tmp_path / "old_protocol.json"
    old.write_text(json.dumps(committed), encoding="utf-8")
    tests = load_protocol(old)
    assert list(tests["knee_flexion"].dots) == FIVE
    eng = Engine(tests, recordings_dir=tmp_path / "rec")
    eng.set_test("knee_flexion")
    p = leg_points(hip_flex=90, knee_flex=20, ankle_df=5)
    eng.process(render(p, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    force = SimulatedForce(rest_s=0.3, ramp_s=0.6, hold_s=1.2, peak_n=400.0)
    force.start()
    eng.force = force
    time.sleep(0.3)
    paths = eng.start_trial("P01")
    for i in range(60):
        eng.process(render(p, seed=i % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    tr = Trial(paths["meta.json"])
    row = analysis.session_row(tr)
    assert row["test_key"] == "knee_flexion" and row["angle2_name"] and row["angle3_name"] == ""                # only the hip is another angle
    assert "ankle_deg" not in open(paths["data.csv"], encoding="utf-8").readline()
