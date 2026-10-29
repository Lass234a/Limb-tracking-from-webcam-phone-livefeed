import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
OPTIONAL = {"strap", "anchor"}
PTS = leg_points(hip_flex=70, knee_flex=60, ankle_df=5, pelvic_tilt=3)


def engine(tmp_path, disabled=(), test="knee_extension"):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.target = 60.0
    for d in disabled:
        eng.set_dot_enabled(d, False)
    eng.process(render(PTS, seed=0), 0.0)
    return eng


def mark_all(eng):
    n_marked = 0
    while (name := eng.next_dot_to_mark()) is not None:
        assert eng.mark_dot(name, *(PTS[name] + 2)), name
        n_marked += 1
    eng.process(render(PTS, seed=1), 0.03)
    return n_marked


def values(res):
    return {a.name: a.value for a in res.angles}


def test_skipping_the_pelvis_dot_drops_only_the_angles_that_need_it(tmp_path):
    eng = engine(tmp_path, disabled=["iliac_crest"], test="belt_squat")        # (the belt squat has the foot; since 0.24.0 the knee test has not)
    assert eng.active_dots == ["trochanter", "epicondyle", "fibular_head", "malleolus", "mt5_base", "mt5_head"]
    assert [a.name for a in eng.active_angles] == ["knee", "ankle"]
    assert mark_all(eng) == 6                                          # the skipped dot is never asked for
    assert eng.trackers.is_ready()
    res = eng.process(render(PTS, seed=2), 0.06)
    assert set(res.dots) == set(eng.active_dots)
    v = values(res)
    assert set(v) == {"knee", "ankle"}
    assert v["knee"] == pytest.approx(60, abs=0.6) and v["ankle"] == pytest.approx(5, abs=0.6)
    avail = {t: (ok, missing) for t, ok, missing in eng.angle_availability()}
    assert avail["Knee flexion"] == (True, [])
    assert avail["Hip flex (+) / ext (-)"][0] is False and "iliac crest (side of pelvis, halfway between ASIS and PSIS)" in avail["Hip flex (+) / ext (-)"][1]


def test_trial_files_only_contain_the_chosen_landmarks(tmp_path):
    eng = engine(tmp_path, disabled=["iliac_crest"], test="belt_squat")
    mark_all(eng)
    assert eng.lock_position()
    eng.start_trial("P01")
    for i in range(10):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    paths = eng.stop_trial()
    head = next(csv.reader(open(paths["data.csv"], newline="")))
    assert "iliac_crest_x" not in head and "hip_deg" not in head
    assert "knee_deg" in head and "trochanter_x" in head
    meta = json.loads(Path(paths["meta.json"]).read_text())
    assert meta["dots_disabled"] == ["iliac_crest"] and "iliac_crest" not in meta["dots_used"]
    assert [a["name"] for a in meta["angles"]] == ["knee", "ankle"]
    from limbtrack.review import Trial
    tr = Trial(paths["meta.json"])                                      # review copes with the reduced set
    assert tr.drawn(3) is not None and set(tr.deg) == {"knee", "ankle"}


def test_main_joint_unavailable_means_no_target_but_neighbours_still_work(tmp_path):
    eng = engine(tmp_path, disabled=["fibular_head"])                   # knee and ankle both need the fibular head
    assert eng.primary_angle is None
    assert [a.name for a in eng.active_angles] == ["hip"]
    mark_all(eng)
    res = eng.process(render(PTS, seed=2), 0.06)
    assert all(not a.primary for a in res.angles)
    assert next(a for a in res.angles if a.name == "hip").reference == 90.0     # the required neighbour angle is judged from the start
    assert any("not available" in line for line in res.header)
    h = eng.history()
    assert h["target"] is None and h["primary"] == "hip"
    assert eng.lock_position()
    # PARKED-GHOST: assert eng.process(render(PTS, seed=3), 0.09).ghost is not None
    # PARKED-GHOST: assert eng._ghost_anchor() == "trochanter"                          # falls back to the hip's fulcrum
    eng.start_trial("P01")
    for i in range(5):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    paths = eng.stop_trial()
    meta = json.loads(Path(paths["meta.json"]).read_text())
    assert meta["target_deg"] is None and "deg" not in Path(paths["data.csv"]).name


def test_reenabling_a_landmark_needs_it_marked_again(tmp_path):
    eng = engine(tmp_path, disabled=["iliac_crest"])
    mark_all(eng)
    eng.lock_position()
    eng.set_dot_enabled("iliac_crest", True)
    assert eng.next_dot_to_mark() == "iliac_crest" and not eng.trackers.is_ready()
    assert eng.locked is False


def test_choice_survives_switching_test_and_lock_is_dropped(tmp_path):
    eng = engine(tmp_path)
    mark_all(eng)
    assert eng.lock_position()
    eng.set_dot_enabled("malleolus", False)
    assert eng.locked is False                                          # references would be stale
    eng.set_test("hip_extension_k90")
    assert eng.disabled_dots - OPTIONAL == {"malleolus"} and eng.primary_angle.name == "hip"
    assert "malleolus" not in eng.active_dots


def test_a_foot_dot_switched_off_is_forgotten_in_a_test_that_has_no_foot(tmp_path):
    """0.24.0: the knee and hip tests have no foot dots. A choice about a dot the test does not have is dropped (and not brought back)."""
    eng = engine(tmp_path, test="belt_squat")
    eng.set_dot_enabled("mt5_head", False)
    assert eng.disabled_dots - OPTIONAL == {"mt5_head"}
    eng.set_test("hip_extension_k90")
    assert eng.disabled_dots - OPTIONAL == set() and "mt5_head" not in eng.test.all_dots
    eng.set_test("belt_squat")
    assert "mt5_head" in eng.active_dots


def test_cannot_change_landmarks_while_recording_or_record_with_no_angles(tmp_path):
    eng = engine(tmp_path)
    mark_all(eng)
    eng.start_trial("P01")
    with pytest.raises(RuntimeError):
        eng.set_dot_enabled("iliac_crest", False)
    eng.stop_trial()

    eng2 = engine(tmp_path, disabled=["iliac_crest", "trochanter", "epicondyle", "fibular_head", "malleolus"])
    assert eng2.active_angles == []
    with pytest.raises(RuntimeError):
        eng2.start_trial("P01")
    assert eng2.history() is None


def test_landmark_checkboxes_and_presets_in_the_window(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    assert len(live.lm_checks) == 7                                     # 5 landmarks (no foot in the knee test) + the optional strap and anchor dots
    assert all(cb.isChecked() for n, cb in live.lm_checks.items() if n not in OPTIONAL)
    assert not any(live.lm_checks[n].isChecked() for n in OPTIONAL)       # the optional dots start switched off
    assert live.lm_preset.itemText(0) == "Preset: all landmarks"

    live.lm_checks["iliac_crest"].setChecked(False)
    assert win.engine.disabled_dots - OPTIONAL == {"iliac_crest"}
    assert "✗" in live.lm_info.text() and "iliac crest" in live.lm_info.text()
    assert "✓ Knee flexion" in live.lm_info.text()

    knee_preset = next(i for i in range(live.lm_preset.count()) if live.lm_preset.itemText(i).endswith("only Knee flexion"))
    live.lm_preset.setCurrentIndex(knee_preset)
    live.apply_landmark_preset(knee_preset)
    assert win.engine.active_dots == ["trochanter", "epicondyle", "fibular_head", "malleolus"]
    assert [cb.isChecked() for cb in live.lm_checks.values()] == [False, True, True, True, True, False, False]

    live.apply_landmark_preset(0)                                       # back to everything
    assert win.engine.disabled_dots == OPTIONAL                          # all landmarks, optional dots still off
    assert all(cb.isChecked() for n, cb in live.lm_checks.items() if n not in OPTIONAL)

    live.lm_checks["iliac_crest"].setChecked(False)
    live.test_combo.setCurrentIndex(3)                                  # another test keeps the choice
    assert not live.lm_checks["iliac_crest"].isChecked() and win.engine.disabled_dots - OPTIONAL == {"iliac_crest"}
    win.close()
