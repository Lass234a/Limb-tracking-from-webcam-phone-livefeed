"""tools/anybody_models.py and tools/anybody_models_run.py: the AnyBody models of the MVIC positions, written from anybody_inputs.csv.

Made-up rows go through the REAL export writer (limbtrack/anybody.py write_csv), so the columns the tool reads are the columns Review writes.
Nothing here needs AnyBody: the AnyScript is checked for balanced brackets, for includes that exist, and for names that the Python side and the
AnyScript side must agree on. That the models LOAD and RUN in AnyBody is not tested here (see the skill's first-run checklist)."""
import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

from limbtrack import anybody
from limbtrack.protocol import load_protocol

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("anybody_models")
R = _load("anybody_models_run")


# ------------------------------------------------------------------------------------------------ made-up export rows
def make_row(n=2, test="knee_extension", target=70.0, **kw):
    r = anybody.blank_row()
    r.update(participant="P01", tested_leg="right", height_m=1.82, mass_kg=78.5, fat_pct=17.5, thigh_m=0.44, shank_m=0.415,
             position_number=n, test_key=test, configuration="seated", target_deg=target, trial_id=f"T{n}", force_n=612.3, force_window_ms=500,
             force_rest_n=12.0, pull_angle_deg=84.0, strap_landmark="lateral_malleolus", strap_from_landmark_m=0.045, lever_arm_app_m=0.36,
             main_angle="knee", knee_deg_at_peak=70.9, knee_deg_at_peak_from_reference=71.4, hip_deg_at_peak=83.2, verdict="not judged",
             data_source="original", data_hash="abc123")
    r.update(kw)
    return r


def write_export(tmp_path, rows):
    path = tmp_path / "anybody_inputs.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        anybody.write_csv(fh, rows)
    return path


def derive(row=None, **kw):
    d, reason = T.derive(row or make_row(), **kw)
    assert d is not None, reason
    return d


def codes(d):
    return {f["code"] for f in d["flags"]}


# ------------------------------------------------------------------------------------------------ the contract with the export
def test_every_column_the_tool_reads_is_a_column_of_the_export():
    read = {"participant", "tested_leg", "height_m", "mass_kg", "fat_pct", "thigh_m", "shank_m", "position_number", "test_key", "target_deg", "trial_id",
            "force_n", "force_window_ms", "force_rest_n", "pull_angle_deg", "strap_landmark", "strap_from_landmark_m", "lever_arm_app_m",
            "main_angle", "main_angle_at_peak_deg", "verdict", "data_source", "data_hash", "hip_deg_at_peak", "hip_deg_at_peak_from_reference",
            "knee_deg_at_peak", "knee_deg_at_peak_from_reference", "ankle_deg_at_peak", "ankle_deg_at_peak_from_reference"}
    assert read <= set(anybody.COLUMNS)


def test_the_csv_the_export_writes_is_read_back_with_blanks_as_none(tmp_path):
    rows = T.read_rows(write_export(tmp_path, [make_row(), make_row(3, pull_angle_deg="")]))
    assert len(rows) == 2 and rows[0]["participant"] == "P01" and T.num(rows[0]["force_n"]) == pytest.approx(612.3)
    assert rows[1]["pull_angle_deg"] is None


def test_every_test_of_protocol_json_is_in_the_registry_and_the_nominal_angles_agree_with_it():
    tests = load_protocol(ROOT / "protocol.json")
    assert set(tests) <= set(T.REGISTRY), "a test of protocol.json has no entry in tools/anybody_models.py REGISTRY"
    for key, spec in T.REGISTRY.items():
        if not spec["status"].startswith("built"):
            continue
        assert tests[key].neighbour_targets["hip"] == spec["drives"]["hip"]["nominal"]
        assert spec["main_joint"] == tests[key].primary.name


# ------------------------------------------------------------------------------------------------ choosing and converting the angles
def test_the_angle_from_the_standing_reference_is_preferred_and_the_hip_is_nominal():
    d = derive()
    p = d["position"]
    assert p["KneeFlexion_deg"] == pytest.approx(71.4) and "standing reference" in d["sources"]["KneeFlexion_deg"]
    assert p["HipFlexion_deg"] == 90.0 and d["measured"]["hip_measured_deg"] == pytest.approx(83.2)      # 6.8 deg off: inside the tolerance
    assert "hip_off_nominal" not in codes(d)


def test_without_a_reference_the_dots_angle_is_used_and_said():
    d = derive(make_row(knee_deg_at_peak_from_reference=None))
    assert d["position"]["KneeFlexion_deg"] == pytest.approx(70.9) and "knee_no_reference" in codes(d)


def test_the_main_angle_and_then_the_target_are_the_fallbacks():
    d = derive(make_row(knee_deg_at_peak_from_reference=None, knee_deg_at_peak=None, main_angle_at_peak_deg=69.0))
    assert d["position"]["KneeFlexion_deg"] == pytest.approx(69.0)
    d = derive(make_row(knee_deg_at_peak_from_reference=None, knee_deg_at_peak=None))
    assert d["position"]["KneeFlexion_deg"] == 70.0 and "knee_not_measured" in codes(d)


def test_an_angle_far_from_the_target_is_a_warning_and_nominal_mode_uses_the_target():
    d = derive(make_row(knee_deg_at_peak_from_reference=95.0))
    assert "knee_off_target" in codes(d) and d["position"]["KneeFlexion_deg"] == 95.0
    assert derive(make_row(knee_deg_at_peak_from_reference=95.0), angles="nominal")["position"]["KneeFlexion_deg"] == 70.0


def test_a_hip_far_from_nominal_is_reported_but_the_nominal_is_used():
    d = derive(make_row(hip_deg_at_peak_from_reference=60.0))
    assert d["position"]["HipFlexion_deg"] == 90.0 and "hip_off_nominal" in codes(d)


def test_the_ankle_sign_is_turned_round_for_anybody():
    d = derive(make_row(), overrides=[(None, None, "ankle_deg", 12.0)])          # limbtrack: 12 deg dorsiflexion
    assert d["position"]["AnklePlantarFlexion_deg"] == -12.0
    assert derive()["position"]["AnklePlantarFlexion_deg"] == 0.0 and str(derive()["position"]["AnklePlantarFlexion_deg"]) == "0.0"


# ------------------------------------------------------------------------------------------------ force, strap and person
def test_no_force_in_newtons_means_no_model_and_a_reason():
    d, reason = T.derive(make_row(force_n=None))
    assert d is None and "no force in newtons" in reason
    assert T.derive(make_row(test="hip_flexion"))[0] is None and "hip_flexion" in T.derive(make_row(test="hip_flexion"))[1]
    assert T.derive(make_row(test="made_up"))[0] is None


def test_the_total_force_is_used_and_a_big_rest_force_is_only_noted():
    d = derive(make_row(force_rest_n=200.0))
    assert d["position"]["Force_N"] == pytest.approx(612.3) and "rest_force" in codes(d)
    assert "rest_force" not in codes(derive())


def test_a_missing_pull_angle_and_strap_place_get_the_sop_defaults_with_warnings():
    d = derive(make_row(pull_angle_deg=None, strap_landmark=None, strap_from_landmark_m=None))
    assert d["position"]["PullAngle_deg"] == 90.0 and d["position"]["StrapDist_m"] == 0.05
    assert {"pull_angle_default", "strap_default"} <= codes(d)


def test_a_strap_measured_from_the_epicondyle_is_marked_as_from_the_knee():
    d = derive(make_row(strap_landmark="lateral_epicondyle", strap_from_landmark_m=0.33))
    assert d["position"]["StrapFromKnee"] == 1 and d["position"]["StrapDist_m"] == pytest.approx(0.33)
    assert derive()["position"]["StrapFromKnee"] == 0


def test_the_tested_leg_defines_which_leg_is_left_out():
    left = T.render_position(derive(make_row(tested_leg="left")))
    assert "#define TESTED_LEG_IS_RIGHT 0" in left and "#define BM_LEG_RIGHT OFF" in left and "BM_LEG_LEFT" not in left
    right = T.render_position(derive())
    assert "#define TESTED_LEG_IS_RIGHT 1" in right and "#define BM_LEG_LEFT OFF" in right
    d = derive(make_row(tested_leg=None))
    assert d["tested_leg"] == "right" and "leg_unknown" in codes(d)


def test_data_not_judged_are_flagged_and_a_valid_trial_is_not():
    assert not derive()["reviewed"] and "not_reviewed" in codes(derive()) and "NOT REVIEWED" in T.render_position(derive())
    ok = derive(make_row(verdict="valid"))
    assert ok["reviewed"] and "not_reviewed" not in codes(ok) and "NOT REVIEWED" not in T.render_position(ok)


def test_missing_measures_are_left_out_of_participant_any_and_warned():
    d = derive(make_row(height_m=None, shank_m=None))
    text = T.render_participant("P01", d["participant_values"])
    assert "BodyHeight" in text and "// BodyHeight: not typed" in text and "ShankLength" not in text and "ThighLength = 0.44" in text
    assert "measures_missing" in codes(d)
    full = T.render_participant("P01", derive()["participant_values"])
    assert "BodyHeight = 1.82;" in full and "FatPercent = 17.5;" in full and "ShankLength = 0.415;" in full


def test_a_lever_arm_that_does_not_fit_the_geometry_is_flagged():
    assert "lever_arm_mismatch" not in codes(derive())
    assert "lever_arm_mismatch" in codes(derive(make_row(lever_arm_app_m=0.12)))


# ------------------------------------------------------------------------------------------------ the geometry (mirror of Loads.any)
def test_the_shank_hangs_down_at_knee_90_and_points_forward_when_straight():
    s = T.shank_dir_global(90, 90)
    assert s == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)
    assert T.shank_dir_global(90, 0) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert T.shank_dir_global(90, 70) == pytest.approx((math.cos(math.radians(-70)), math.sin(math.radians(-70)), 0.0))


def test_a_pull_square_to_the_shank_in_knee_extension_points_back_and_down_for_a_flexed_knee():
    s = T.shank_dir_global(90, 70)
    p = T.pull_dir_global(s, 90.0, +1)
    assert p == pytest.approx((-math.sin(math.radians(70)), -math.cos(math.radians(70)), 0.0))       # backward (-x), slightly down
    assert p[0] < 0 and T.angle_between_deg(p, s) == pytest.approx(90.0)


@pytest.mark.parametrize("alpha", [0.0, 45.0, 84.0, 90.0, 120.0])
def test_the_pull_makes_the_stated_angle_with_the_shank(alpha):
    s = T.shank_dir_global(90, 70)
    for effort in (+1, -1):
        assert T.angle_between_deg(T.pull_dir_global(s, alpha, effort), s) == pytest.approx(alpha, abs=1e-9)


def test_the_strap_always_works_against_the_effort():
    """Physics check, independent of the formula: the moment of the strap force about the knee must turn the opposite way to the effort.
    Knee extension (effort about +z): the strap moment about z is negative. Knee flexion (effort about -z): positive."""
    for knee in (20, 70, 110):
        s = T.shank_dir_global(90, knee)
        for effort in (+1, -1):
            f = T.pull_dir_global(s, 84.0, effort)
            r = tuple(0.33 * c for c in s)                                  # the strap, 0.33 m from the knee along the shank
            m_z = r[0] * f[1] - r[1] * f[0]
            assert m_z * effort < 0, (knee, effort, m_z)


def test_the_mirror_in_python_is_the_formula_in_loads_any():
    text = (ROOT / "tools" / "anybody_templates" / "SeatedKnee" / "Model" / "Loads.any").read_text(encoding="utf-8")
    assert "cross(-Main.Position.EffortAxisZ * {0, 0, 1}, ShankDir)" in text
    assert "cos(PullAngle) * ShankDir + sin(PullAngle) * Tangent" in text
    assert "(Shank.AnkleJoint.r - Shank.KneeJoint.r)" in text and "F = Main.Position.Force_N * .PullDir;" in text


# ------------------------------------------------------------------------------------------------ the files
def test_position_any_carries_the_numbers_their_sources_and_a_designvar_force():
    text = T.render_position(derive())
    assert "AnyVar KneeFlexion_deg = 71.4;" in text and "AnyVar Force_N = DesignVar(612.3);" in text
    assert "AnyVar PullAngle_deg = 84;" in text and "AnyVar EffortAxisZ = 1;" in text and "trial T2" in text and "data_hash: abc123" in text
    assert text.rstrip().endswith("};")


def test_every_position_name_the_templates_use_is_written_by_the_tool():
    written = T.render_position(derive())
    used = set()
    for f in (ROOT / "tools" / "anybody_templates" / "SeatedKnee").rglob("*.any"):
        used |= set(re.findall(r"Main\.Position\.(\w+)", f.read_text(encoding="utf-8")))
    assert used, "the templates read nothing from Main.Position?"
    for name in used:
        assert f"AnyVar {name} =" in written, f"{name} is used by a template but not written to Position.any"
    for define in ("TESTED_LEG_IS_RIGHT", "WITH_CHECKS", "WITH_MAXLOAD"):
        assert f"#define {define} " in written


def test_the_templates_and_the_generated_files_have_balanced_brackets_and_no_leftovers():
    for f in sorted((ROOT / "tools" / "anybody_templates").rglob("*.any")):
        assert T.lint_anyscript(f.read_text(encoding="utf-8")) == [], f.name
    d = derive()
    for text in (T.render_position(d), T.render_participant("P01", d["participant_values"]), T.render_main(d)):
        assert T.lint_anyscript(text) == []


def test_the_lint_finds_the_usual_typing_slips():
    assert T.lint_anyscript("A = { B = (1; };") != []
    assert T.lint_anyscript('A = "open;') != []
    assert T.lint_anyscript("/* never closed") != []
    assert T.lint_anyscript("A = {{ name }};") != []
    assert T.lint_anyscript('A = { // a comment with a } and a "quote\n B = "a } b"; /* ) */ };') == []


def test_a_hand_edited_file_is_never_overwritten_unless_forced(tmp_path):
    p = tmp_path / "Position.any"
    assert T.write_generated(p, "AnyFolder A = {};") == "written"
    assert T.write_generated(p, "AnyFolder A = {};") == "unchanged"
    assert T.write_generated(p, "AnyFolder A = { };") == "written"                    # untouched since written: the new text replaces it
    p.write_text(p.read_text(encoding="utf-8").replace("{ }", "{ AnyVar x = 5; }"), encoding="utf-8")        # a person edits it
    assert T.write_generated(p, "AnyFolder A = {};") == "kept"
    assert "x = 5" in p.read_text(encoding="utf-8")
    assert "AnyFolder A = {};" in (tmp_path / "Position.generated.any").read_text(encoding="utf-8")
    assert T.write_generated(p, "AnyFolder A = {};", force=True) == "written" and "x = 5" not in p.read_text(encoding="utf-8")


def test_a_file_saved_with_windows_line_ends_still_counts_as_untouched(tmp_path):
    p = tmp_path / "Participant.any"
    T.write_generated(p, "line one\nline two")
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert T.write_generated(p, "line one\nline two changed") == "written"


# ------------------------------------------------------------------------------------------------ the whole run
def test_build_writes_the_shared_model_once_and_small_files_per_position(tmp_path):
    rows = [make_row(1, target=20.0), make_row(2), make_row(3, target=110.0), make_row(4, test="hip_flexion", target=90.0),
            make_row(5, force_n=None), make_row(6, participant="P02", target=70.0)]
    out = tmp_path / "models"
    res = T.build(T.read_rows(write_export(tmp_path, rows)), out)
    assert [m["id"] for m in res["models"]] == ["P01_pos01_knee_extension_t20", "P01_pos02_knee_extension_t70", "P01_pos03_knee_extension_t110",
                                                "P02_pos06_knee_extension_t70"]
    assert [s["position_number"] for s in res["skipped"]] == [4, 5]
    assert (out / "SeatedKnee" / "SeatedKnee.main.any").is_file() and len(list(out.glob("SeatedKnee*"))) == 1
    assert (out / "P01" / "Participant.any").is_file() and (out / "P02" / "Participant.any").is_file()
    for m in res["models"]:
        assert (out / m["folder"] / "Position.any").is_file() and (out / m["main_file"]).is_file()
        assert T.missing_includes(out / m["main_file"]) == []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["models"]) == 4 and manifest["skipped"][1]["reason"].startswith("no force")
    assert manifest["models"][1]["expected"]["pull_dir_global"] and "knee_flexion_deg" in manifest["models"][1]["expected"]
    assert (out / "manifest.csv").read_text(encoding="utf-8").splitlines()[0].startswith("id,participant,position_number")
    assert "not reviewed" in T.summary_text(res) and "without a model" in T.summary_text(res)


def test_a_second_run_changes_nothing_and_a_changed_force_updates_only_that_position(tmp_path):
    out = tmp_path / "models"
    rows = T.read_rows(write_export(tmp_path, [make_row(1, target=20.0), make_row(2)]))
    T.build(rows, out)
    again = T.build(rows, out)
    assert set(again["files"].values()) == {"unchanged"}
    rows[1]["force_n"] = "700"
    third = T.build(rows, out)
    changed = [p for p, s in third["files"].items() if s != "unchanged"]
    assert changed == ["P01/pos02_knee_extension_t70/Position.any"]


def test_set_overrides_are_scoped_to_the_participant_and_position(tmp_path):
    ov = T.parse_overrides(["P01:2:force_n=450", "knee_deg=72", "P02:pull_angle_deg=80"])
    assert ov == [("P01", 2, "force_n", 450.0), (None, None, "knee_deg", 72.0), ("P02", None, "pull_angle_deg", 80.0)]
    out = tmp_path / "models"
    res = T.build(T.read_rows(write_export(tmp_path, [make_row(1, target=20.0), make_row(2)])), out, overrides=ov)
    first, second = res["models"]
    assert first["position"]["Force_N"] == pytest.approx(612.3) and second["position"]["Force_N"] == 450.0
    assert first["position"]["KneeFlexion_deg"] == 72.0 and second["position"]["KneeFlexion_deg"] == 72.0
    assert first["position"]["PullAngle_deg"] == 84.0                                      # P02's override did not reach P01
    assert second["overrides"] == ["force_n=450", "knee_deg=72"]
    for bad in ("force_n=abc", "nope=1", "P01:x:force_n=1"):
        with pytest.raises(SystemExit):
            T.parse_overrides([bad])


def test_selection_by_participant_test_and_position_number(tmp_path):
    rows = T.read_rows(write_export(tmp_path, [make_row(1, target=20.0), make_row(2), make_row(3, participant="P02")]))
    assert [m["position_number"] for m in T.build(rows, tmp_path / "a", participants=["P02"])["models"]] == [3]
    assert [m["position_number"] for m in T.build(rows, tmp_path / "b", positions=[1, 2])["models"]] == [1, 2]
    assert T.build(rows, tmp_path / "c", tests=["knee_flexion"])["models"] == []


def test_the_libdef_line_follows_the_ammr_path_that_is_given(tmp_path):
    T.build(T.read_rows(write_export(tmp_path, [make_row()])), tmp_path / "m", ammr_libdef="C:\\Users\\x\\AMMR\\libdef.any")
    text = (tmp_path / "m" / "SeatedKnee" / "libdef.any").read_text(encoding="utf-8")
    assert '#include "C:/Users/x/AMMR/libdef.any"' in text and "ANYBODY_PATH_INSTALLDIR" not in text


def test_the_command_line_runs_and_lists_the_tests(tmp_path, capsys):
    path = write_export(tmp_path, [make_row()])
    assert T.main(["--inputs", str(path), "--out", str(tmp_path / "o"), "--set", "force_n=500"]) == 0
    assert "1 model written" in capsys.readouterr().out
    assert T.main(["--list-tests"]) == 0 and "knee_extension" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        T.main(["--inputs", str(tmp_path / "missing.csv")])


# ------------------------------------------------------------------------------------------------ the runner
@pytest.fixture
def built(tmp_path):
    out = tmp_path / "models"
    T.build(T.read_rows(write_export(tmp_path, [make_row(1, target=20.0), make_row(2)])), out)
    return out, json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def good_result(entry, **kw):
    e = entry["expected"]
    r = {"Main.Study.Output.MaxMuscleActivity": [1.37], "Main.Check.KneeFlexion_deg": [e["knee_flexion_deg"]],
         "Main.Check.AnklePlantarFlexion_deg": [0.0], "Main.Check.PullAngle_deg": [e["pull_angle_deg"]],
         "Main.Check.PullDir": [e["pull_dir_global"]], "Main.Check.LeverArm_m": [e["lever_arm_app_m"]],
         "Main.Check.ExternalKneeMoment_Nm": [e["ext_moment_app_nm"]], "Main.MaxLoading.MaxLoad": [447.0]}
    r.update(kw)
    return r


def test_the_macro_loads_runs_dumps_and_does_the_maxload_last(built):
    out, manifest = built
    lines = R.macro_for(manifest["models"][1], out)
    assert lines[0].startswith('load "') and lines[0].endswith('pos02_knee_extension_t70/Main.any"') and lines[-1] == "exit"
    first_run, max_run = lines.index("operation Main.Study.InverseDynamics"), lines.index("operation Main.MaxLoading.Evaluate")
    dump_act = lines.index('classoperation Main.Study.Output.MaxMuscleActivity "Dump"')
    assert first_run < dump_act < max_run < lines.index('classoperation Main.MaxLoading.MaxLoad "Dump"')
    assert 'classoperation Main.Check.PullDir "Dump"' in lines
    assert not any("Check" in l for l in R.macro_for(manifest["models"][0], out, with_checks=False))


def test_only_selects_by_participant_and_position(built):
    _, manifest = built
    assert [m["position_number"] for m in R.select(manifest["models"], ["P01:2"])] == [2]
    assert len(R.select(manifest["models"], ["P01"])) == 2 and R.select(manifest["models"], None) == manifest["models"]
    assert R.select(manifest["models"], ["P09"]) == []


def test_a_result_that_matches_the_manifest_is_ok_and_gives_the_scale_numbers(built):
    _, manifest = built
    entry = manifest["models"][1]
    row = R.summarise(entry, good_result(entry))
    assert row["status"] == "ok" and row["max_muscle_activity"] == pytest.approx(1.37) and row["max_load_n"] == 447.0
    assert row["force_over_max_load"] == pytest.approx(612.3 / 447.0)
    assert row["check_knee_flexion_deg_diff"] == pytest.approx(0.0) and row["check_pull_dir_diff_deg"] == pytest.approx(0.0, abs=1e-6)


def test_a_wrong_angle_a_wrong_direction_and_a_wrong_moment_are_all_reported(built):
    _, manifest = built
    entry = manifest["models"][1]
    row = R.summarise(entry, good_result(entry, **{"Main.Check.KneeFlexion_deg": [78.0], "Main.Check.PullDir": [[0.9, 0.4, 0.0]],
                                                   "Main.Check.ExternalKneeMoment_Nm": [150.0]}))
    assert row["status"].startswith("checks differ")
    for part in ("knee_flexion_deg", "pull direction", "moment_nm"):
        assert part in row["status"]


def test_an_anybody_error_is_kept_as_the_status_and_nothing_else_is_guessed(built):
    _, manifest = built
    row = R.summarise(manifest["models"][0], {"ERROR": ["Error: model could not be loaded"]})
    assert row["status"].startswith("error: Error: model could not be loaded") and "max_muscle_activity" not in row


def test_run_collects_the_results_through_a_stand_in_for_anypytools(built):
    out, manifest = built
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen["kw"] = kw

        def start_macro(self, macros):
            seen["macros"] = macros
            return [good_result(m) for m in manifest["models"]]
    rows = R.run(out, processes=2, process_factory=Fake)
    assert seen["kw"] == {"num_processes": 2} and len(seen["macros"]) == 2 and [r["status"] for r in rows] == ["ok", "ok"]
    csv_text = (out / "results.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("id,participant") and "max_muscle_activity" in csv_text
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))[0]["id"] == manifest["models"][0]["id"]


def test_dry_run_prints_the_macros_and_runs_nothing(built, capsys):
    out, _ = built
    assert R.run(out, dry_run=True) == []
    text = capsys.readouterr().out
    assert text.count("--- P01_pos0") == 2 and "operation Main.Study.InverseDynamics" in text
