"""protocol.json follows Tabel 2 of the protocol draft (2026-09-28): positions and required neighbour angles."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.results import INFO, OK, OUT  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")

# Tabel 2, sagittal rows: (measured joint positions, required neighbour angle)
TABLE2 = {
    "hip_extension_k0": ([0, 45], {"knee": 0}),
    "hip_extension_k90": ([0, 90], {"knee": 90}),
    "hip_flexion": ([0, 90], {"knee": 90}),
    "knee_extension": ([20, 70, 110], {"hip": 90}),
    "knee_flexion": ([20, 90], {"hip": 90}),
    "ankle_plantarflexion": ([0, -20], {"knee": 0}),
    "ankle_dorsiflexion": ([0, -20], {"knee": 0}),
}


@pytest.mark.parametrize("key", TABLE2)
def test_positions_and_required_neighbour_angles_match_the_table(key):
    targets, neighbour = TABLE2[key]
    assert TESTS[key].targets == targets
    assert TESTS[key].neighbour_targets == neighbour


def test_the_sagittal_rows_make_15_of_the_19_positions():
    assert sum(len(t) for t, _ in TABLE2.values()) == 15      # + 4 hip abduction/adduction positions (frontal view, since 0.15.0)


# Tabel 2, frontal rows: hip ABD and ADD at 0 and 20 deg, hip flexion 0 (not measured in the frontal view: the participant stands against a wall)
FRONTAL = {"hip_abduction": [0, 20], "hip_adduction": [0, 20]}


@pytest.mark.parametrize("key", FRONTAL)
def test_the_frontal_rows_are_there_with_their_positions_and_no_neighbour_requirement(key):
    t = TESTS[key]
    assert t.view == "frontal" and t.targets == FRONTAL[key] and t.neighbour_targets == {}
    assert t.optional_dots == []                                 # no strap or anchor in the frontal tests (no torque from the front yet)


def test_all_19_positions_of_the_protocol_are_in_the_plan():
    positions = sum(len(t.targets) for k, t in TESTS.items() if k in TABLE2 or k in FRONTAL)
    assert positions == 19
    assert sorted(k for k, t in TESTS.items() if t.view == "frontal") == sorted(FRONTAL)


def make(tmp_path, test, **leg):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    p = leg_points(**leg)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    return eng, p


def angle(res, name):
    return next(a for a in res.angles if a.name == name)


def test_the_required_neighbour_angle_is_judged_from_the_start_without_locking(tmp_path):
    eng, p = make(tmp_path / "a", "knee_extension", hip_flex=90, knee_flex=70, ankle_df=5)
    for i in range(8):
        res = eng.process(render(p, seed=i), 0.1 + i / 30)
    hip = angle(res, "hip")
    assert not eng.locked and hip.reference == 90.0 and hip.raw_status == OK            # hip at 90: green before any lock
    assert angle(res, "ankle").raw_status == INFO                                        # other neighbours still wait for Lock

    eng, p = make(tmp_path / "b", "knee_extension", hip_flex=70, knee_flex=70, ankle_df=5)
    for i in range(8):
        res = eng.process(render(p, seed=i), 0.1 + i / 30)
    assert angle(res, "hip").raw_status == OUT and angle(res, "hip").raw_deviation == pytest.approx(-20.0, abs=1.0)


def test_lock_does_not_overwrite_a_required_neighbour_angle(tmp_path):
    eng, p = make(tmp_path, "knee_extension", hip_flex=84, knee_flex=70, ankle_df=5)       # 6 deg off the required 90
    for i in range(8):
        res = eng.process(render(p, seed=i), 0.1 + i / 30)
    assert eng.lock_position()
    res = eng.process(render(p, seed=9), 0.5)
    hip, ankle = angle(res, "hip"), angle(res, "ankle")
    assert hip.reference == 90.0 and hip.raw_status == OUT                                 # judged against the protocol, not against itself
    assert ankle.reference == pytest.approx(5.0, abs=1.0) and ankle.raw_status == OK       # the others use the locked value as before


def test_csv_and_meta_record_the_required_angle(tmp_path):
    eng, p = make(tmp_path, "hip_flexion", hip_flex=0, knee_flex=90, ankle_df=5)
    eng.target = 0.0
    paths = eng.start_trial("P01")
    for i in range(6):
        eng.process(render(p, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert float(rows[0]["knee_ref"]) == 90.0
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["neighbour_targets_deg"] == {"knee": 90} and "neighbour_targets_deg" in meta["reference_basis"]


def test_the_knee_is_signed_so_hyperextension_reads_negative(tmp_path):
    eng, p = make(tmp_path, "ankle_plantarflexion", hip_flex=0, knee_flex=-8, ankle_df=0)
    for i in range(8):
        res = eng.process(render(p, seed=i), 0.1 + i / 30)
    knee = angle(res, "knee")
    assert knee.raw_value == pytest.approx(-8.0, abs=0.7) and knee.reference == 0.0 and knee.raw_status == OUT    # K 0 +/- 5: -8 is out


# ---- a test without positions (belt squat) is shown but not judged until a target is typed
def test_a_test_without_positions_has_no_target_and_is_not_judged():
    from PySide6.QtWidgets import QApplication

    from limbtrack.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        win = MainWindow(TESTS, d)
        live = win.live
        live.test_combo.setCurrentIndex(live.test_combo.findData("belt_squat"))
        assert win.engine.target is None and live.target_spin.text().startswith("no target")
        live.target_spin.setValue(60.0)                      # typing a target switches judging on
        assert win.engine.target == 60.0
        live.target_spin.setValue(live.target_spin.minimum())
        assert win.engine.target is None
        live.test_combo.setCurrentIndex(0)
        assert win.engine.target == 20.0                     # a test with positions starts on its first one
        win.close()


# ---- GUI-06: changing the test keeps the marked dots when the landmarks are the same
def test_switching_to_a_test_with_the_same_landmarks_keeps_the_marked_dots_and_the_operators_tolerances(tmp_path):
    eng, p = make(tmp_path, "knee_extension", hip_flex=90, knee_flex=70, ankle_df=5)
    for i in range(5):
        eng.process(render(p, seed=i), 0.1 + i / 30)
    eng.tolerance, eng.neighbour_tolerance = 2.0, 4.0                       # the operator's own values
    assert eng.lock_position()
    eng.set_test("knee_flexion")
    assert eng.next_dot_to_mark() is None and eng.trackers.is_ready()        # nothing to re-click
    assert not eng.locked and eng._refs == {}                                # but the lock belongs to the old test
    assert (eng.tolerance, eng.neighbour_tolerance) == (2.0, 4.0)
    assert eng.target == 20.0                                                # the new test's first position
    res = eng.process(render(p, seed=9), 1.0)
    assert not any(st.lost for st in res.dots.values())
    hip = next(a for a in res.angles if a.name == "hip")
    assert hip.reference == 90.0                                              # the new test's required neighbour angle applies at once


def test_a_test_with_other_landmarks_still_starts_afresh(tmp_path):
    import copy
    import json
    d = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    d["tests"]["knee_only"] = copy.deepcopy(d["tests"]["knee_extension"])
    d["tests"]["knee_only"].update(label="Knee only", angles=["knee"], neighbour_targets={})
    path = tmp_path / "p.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    tests = load_protocol(path)
    assert tests["knee_only"].dots != tests["knee_extension"].dots
    eng = Engine(tests, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    p = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    eng.set_test("knee_only")
    assert eng.next_dot_to_mark() is not None
