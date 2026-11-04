"""tools/check_session.py: the independent after-session check of the saved numbers.

A trial is recorded for real (synthetic pictures, simulated force); damaged copies of it must be caught, and the check must never write.
No real participant data is used here."""
import csv
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

_spec = importlib.util.spec_from_file_location("check_session", Path(__file__).resolve().parent.parent / "tools" / "check_session.py")        # (by path: tools/ is not put on sys.path for every other test)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

from tests.test_force import recording  # noqa: E402


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """One recorded trial with force (it takes a few seconds); each test works on its own copy of the participant folder."""
    root = tmp_path_factory.mktemp("check_session_source")
    eng, paths = recording(root, seconds=3.0)
    return paths["meta.json"].parent


@pytest.fixture
def folder(recorded, tmp_path):
    dst = tmp_path / "recordings" / recorded.name
    shutil.copytree(recorded, dst)
    return dst


def meta_of(folder):
    return next(folder.glob("*_meta.json"))


def rewrite_csv(path, change):
    """Read a CSV, let `change(rows)` alter the list of dict rows, write it back."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        names, rows = reader.fieldnames, list(reader)
    change(rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def levels(res):
    return [(lv, tx) for lv, tx in res["issues"]]


def has(res, level, words):
    return any(lv == level and words in tx for lv, tx in res["issues"])


def tree_hash(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------------------------- a good trial
def test_a_trial_as_recorded_has_no_fail(folder):
    res = cs.check_trial(meta_of(folder))
    assert not [x for x in levels(res) if x[0] == cs.FAIL], levels(res)
    assert res["facts"]["peak_n"] == pytest.approx(400.0, rel=0.1)            # (the simulated push)


# ---------------------------------------------------------------------------------------------- damage must be caught
def test_newtons_that_are_not_volts_minus_zero_times_slope_are_a_fail(folder):
    path = next(folder.glob("*_force.csv"))
    rewrite_csv(path, lambda rows: rows[len(rows) // 2].update(newtons=str(float(rows[len(rows) // 2]["newtons"]) + 5.0)))
    assert has(cs.check_trial(meta_of(folder)), cs.FAIL, "not (volts - zero) x slope")


def test_an_angle_that_is_not_what_its_dots_give_is_a_fail(folder):
    path = next(folder.glob("*_data.csv"))

    def bump(rows):
        r = rows[len(rows) // 2]
        r["knee_deg"] = str(float(r["knee_deg"]) + 3.0)
    rewrite_csv(path, bump)
    assert has(cs.check_trial(meta_of(folder)), cs.FAIL, "not what its dots give")


def test_force_and_pictures_on_two_clocks_are_a_fail(folder):
    path = next(folder.glob("*_force.csv"))
    rewrite_csv(path, lambda rows: [r.update(clock_s=f"{float(r['clock_s']) + 0.02:.5f}") for r in rows])
    assert has(cs.check_trial(meta_of(folder)), cs.FAIL, "not on one clock")


def test_a_missing_picture_row_is_a_fail(folder):
    path = next(folder.glob("*_data.csv"))
    rewrite_csv(path, lambda rows: rows.pop(5))
    assert has(cs.check_trial(meta_of(folder)), cs.FAIL, "pictures but meta.json says")


def test_picture_times_that_go_backwards_are_a_fail(folder):
    path = next(folder.glob("*_data.csv"))

    def swap(rows):
        rows[10]["t_s"], rows[11]["t_s"] = rows[11]["t_s"], rows[10]["t_s"]
    rewrite_csv(path, swap)
    assert has(cs.check_trial(meta_of(folder)), cs.FAIL, "do not always go forward")


def test_a_force_sample_rate_that_is_not_the_recorded_one_is_a_fail(folder):
    path = meta_of(folder)
    m = json.loads(path.read_text())
    m["force"]["sample_rate_hz"] = 2000.0
    path.write_text(json.dumps(m))
    assert has(cs.check_trial(path), cs.FAIL, "every force time is off")


# ---------------------------------------------------------------------------------------------- warnings
def test_a_force_that_was_not_zeroed_is_a_warning_not_a_fail(folder):
    path = meta_of(folder)
    m = json.loads(path.read_text())
    m["force"]["calibration"]["zero_set"] = False
    path.write_text(json.dumps(m))
    res = cs.check_trial(path)
    assert has(res, cs.WARN, "not zeroed") and not [x for x in levels(res) if x[0] == cs.FAIL]


def test_samples_at_the_input_limit_and_a_load_beyond_the_calibration_are_warnings(folder):
    path = next(folder.glob("*_force.csv"))
    slope = json.loads(meta_of(folder).read_text())["force"]["calibration"]
    off, k = slope["offset_v"], slope["n_per_v"]

    def clip(rows):
        mid = len(rows) // 2
        for r in rows[mid:mid + 40]:
            r["volts"] = "9.95"
            r["newtons"] = f"{(9.95 - off) * k:.3f}"
    rewrite_csv(path, clip)
    res = cs.check_trial(meta_of(folder), cal_max_n=100.0)                                 # (the simulated push is about 400 N: far beyond a 100 N calibration)
    assert has(res, cs.WARN, "at the input limit") and has(res, cs.WARN, "above the heaviest calibration load")
    assert not [x for x in levels(res) if x[0] == cs.FAIL]


def test_a_dot_lost_at_the_force_peak_is_named(folder):
    path = next(folder.glob("*_data.csv"))
    fres = cs.check_trial(meta_of(folder))
    peak_n = fres["facts"]["peak_n"]
    assert peak_n > 0
    t_peak = None
    force = cs.read_force(next(folder.glob("*_force.csv")))
    n = cs.moving_mean(force[3], 100)
    t_peak = float(force[0][int(n.argmax())])

    def lose(rows):
        for r in rows:
            if abs(float(r["t_s"]) - t_peak) <= 0.1:
                r["epicondyle_lost"] = "1"
    rewrite_csv(path, lose)
    assert has(cs.check_trial(meta_of(folder)), cs.WARN, "epicondyle lost")


def test_an_angle_far_from_the_target_is_a_warning(folder):
    path = meta_of(folder)
    m = json.loads(path.read_text())
    m["target_deg"] = 20.0                                              # (the recording is at about 70)
    path.write_text(json.dumps(m))
    assert has(cs.check_trial(path), cs.WARN, "from the target 20")


def test_a_dot_that_keeps_jumping_is_a_warning(folder):
    path = next(folder.glob("*_data.csv"))
    rewrite_csv(path, lambda rows: [r.update(knee_deg=str(float(r["knee_deg"]) + (25.0 if i % 2 else 0.0))) for i, r in enumerate(rows[:12])])
    res = cs.check_trial(meta_of(folder))
    assert has(res, cs.WARN, "jumped by more than")


# ---------------------------------------------------------------------------------------------- the angle maths on its own
def _rows(points, mirror=False):
    """CSV-like rows (one picture) of dot positions; `mirror` flips the picture left-right (a person who faces left)."""
    return [{f"{n}_x": str(-x if mirror else x) for n, (x, y) in points.items()} | {f"{n}_y": str(y) for n, (x, y) in points.items()}]


def test_the_angle_is_the_same_whichever_way_the_person_faces():
    import math
    c, s = math.cos(math.radians(70)), math.sin(math.radians(70))
    pts = {"trochanter": (100.0, 100.0), "epicondyle": (300.0, 100.0), "fibular_head": (300 + 10 * c, 100 + 10 * s), "malleolus": (300 + 200 * c, 100 + 200 * s)}
    knee = {"vectors": [["trochanter", "epicondyle"], ["fibular_head", "malleolus"]], "signed": True, "sign": 1.0, "offset": 0.0}
    right, left = _rows(pts), _rows(pts, mirror=True)
    assert cs.angle_of(knee, right, "right")[0] == pytest.approx(70.0)
    assert cs.angle_of(knee, left, "left")[0] == pytest.approx(70.0)           # the mirrored picture, read as a person who faces left
    assert cs.angle_of(knee, left, "right")[0] == pytest.approx(-70.0)         # read the wrong way round it would be the opposite angle: the setting matters


def test_hip_ankle_and_fixed_direction_arms_follow_the_protocol_signs():
    import math
    # standing, facing right: trunk and leg straight down, foot square to the shank = hip 0, knee 0, ankle 0
    pts = {"iliac_crest": (100.0, 0.0), "trochanter": (100.0, 80.0), "epicondyle": (100.0, 280.0), "fibular_head": (100.0, 290.0), "malleolus": (100.0, 480.0),
           "mt5_base": (105.0, 480.0), "mt5_head": (160.0, 480.0)}
    hip = {"vectors": [["iliac_crest", "trochanter"], ["trochanter", "epicondyle"]], "signed": True, "sign": -1, "offset": 0}
    ankle = {"vectors": [["malleolus", "fibular_head"], ["mt5_base", "mt5_head"]], "signed": False, "sign": -1, "offset": 90}
    rows = _rows(pts)
    assert cs.angle_of(hip, rows, "right")[0] == pytest.approx(0.0, abs=1e-9) and cs.angle_of(ankle, rows, "right")[0] == pytest.approx(0.0, abs=1e-9)
    pts["epicondyle"] = (100.0 + 200 * math.sin(math.radians(90)), 80.0)               # the thigh swung forward to horizontal: hip flexion 90
    assert cs.angle_of(hip, _rows(pts), "right")[0] == pytest.approx(90.0)
    pts["mt5_head"] = (105 + 55 * math.cos(math.radians(20)), 480.0 - 55 * math.sin(math.radians(20)))     # toes up 20 deg: dorsiflexion is positive
    assert cs.angle_of(ankle, _rows(pts), "right")[0] == pytest.approx(20.0)
    up = {"vectors": ["up", ["trochanter", "epicondyle"]], "signed": False, "sign": 1, "offset": 0}
    assert cs.angle_of(up, _rows(pts), "right")[0] == pytest.approx(90.0)             # an arm that is a direction, not two dots


# ---------------------------------------------------------------------------------------------- across the trials of one person
def test_an_inconsistent_ruler_scale_and_a_wandering_crest_dot_are_warned_about():
    ok = [{"thigh_cm": 40.0, "pelvis_cm": 10.0}, {"thigh_cm": 40.5, "pelvis_cm": 10.2}, {"thigh_cm": 39.5, "pelvis_cm": 9.9}]
    assert cs.check_participant(ok) == []
    bad = [{"thigh_cm": 35.0, "pelvis_cm": 6.0}, {"thigh_cm": 44.0, "pelvis_cm": 10.0}, {"thigh_cm": 49.0, "pelvis_cm": 12.0}]
    texts = [t for _, t in cs.check_participant(bad)]
    assert any("ruler scale is not consistent" in t and "35 to 49 cm" in t for t in texts)
    assert any("iliac crest dot is not a stable landmark" in t for t in texts)
    assert cs.check_participant([{"thigh_cm": 35.0}, {"thigh_cm": 49.0}]) == []             # (two trials say too little)


# ---------------------------------------------------------------------------------------------- the whole run
def test_the_run_prints_fails_sets_the_exit_code_and_never_writes(folder, capsys):
    before = tree_hash(folder.parent)
    assert cs.main([str(folder.parent)]) == 0
    clean = capsys.readouterr().out
    assert "1 trials checked: 0 FAIL" in clean and "Nothing was changed" in clean
    path = next(folder.glob("*_force.csv"))
    rewrite_csv(path, lambda rows: rows[3].update(newtons=str(float(rows[3]["newtons"]) + 9.0)))
    broken_before = tree_hash(folder.parent)
    assert cs.main([str(folder.parent)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "1 trials checked: 1 FAIL" in out
    assert tree_hash(folder.parent) == broken_before and before != broken_before                 # (only my own damage changed a file)
    assert cs.main([str(folder.parent / "no_such_folder")]) == 2


def test_the_check_reads_the_registry_for_the_heaviest_calibration_load():
    limit = cs.calibrated_limit_n()
    assert limit == pytest.approx(250 * 9.80665, rel=0.01)
    assert cs.calibrated_limit_n(Path("no_such_registry.json")) is None


def test_the_check_uses_no_analysis_code_of_the_program():
    import ast
    tree = ast.parse(Path(cs.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "limbtrack" not in imported, imported                       # (it reads the saved files only; a mistake in the program cannot also be in the check)


# ---------------------------------------------------------------------------------------------- what the review of this tool found
def test_an_empty_or_one_sample_force_file_is_reported_and_does_not_stop_the_check(folder):
    path = next(folder.glob("*_force.csv"))
    path.write_text("t_s,clock_s,volts,newtons\n", encoding="utf-8")
    assert has(cs.check_trial(meta_of(folder)), cs.WARN, "holds 0 sample(s)")
    path.write_text("t_s,clock_s,volts,newtons\n0.001,5.0,0.1,1.0\n", encoding="utf-8")
    res = cs.check_trial(meta_of(folder))
    assert has(res, cs.WARN, "holds 1 sample(s)") and "issues" in res


def test_a_folder_called_review_above_the_one_given_does_not_hide_the_trials(folder, tmp_path):
    root = tmp_path / "Review" / "recordings"                                   # (a person's own folder name, above the folder given)
    shutil.copytree(folder, root / folder.name)
    assert cs.run(root, out=lambda *_: None)[0] == 1
    shutil.copytree(folder, root / "review" / folder.name)                      # the program's own review layer below it is still left out
    assert cs.run(root, out=lambda *_: None)[0] == 1
