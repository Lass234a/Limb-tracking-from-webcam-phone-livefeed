"""tools/anybody_models.py and tools/anybody_models_run.py: the AnyBody models of the MVIC positions, written from anybody_inputs.csv.

Made-up rows go through the REAL export writer (limbtrack/anybody.py write_csv), so the columns the tool reads are the columns Review writes.
Nothing here needs AnyBody: the AnyScript is checked for balanced brackets, for includes that exist, and for names that the Python side and the
AnyScript side must agree on. That the models LOAD and RUN in AnyBody is not tested here (see the skill's first-run checklist); the tool's
--regression-case is the test for that, run on the computer that has AnyBody."""
import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

from limbtrack import anybody
from limbtrack.protocol import load_protocol

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "tools" / "anybody_templates" / "SeatedKnee"


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


def test_the_knee_zero_offset_is_added_only_when_asked_for_and_said():
    assert derive()["position"]["KneeFlexion_deg"] == pytest.approx(71.4)
    d = derive(knee_offset=1.5)
    assert d["position"]["KneeFlexion_deg"] == pytest.approx(72.9) and "zero offset 1.5" in d["sources"]["KneeFlexion_deg"]


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


def test_the_total_force_is_the_measured_force_the_probe_load_is_separate_and_a_big_rest_force_is_only_noted():
    d = derive(make_row(force_rest_n=200.0))
    assert d["position"]["MeasuredForce_N"] == pytest.approx(612.3) and "rest_force" in codes(d)
    assert d["position"]["ProbeLoad_N"] == 800.0 and derive(probe_load=1000.0)["position"]["ProbeLoad_N"] == 1000.0
    assert "rest_force" not in codes(derive())


def test_a_missing_pull_angle_and_strap_place_get_the_sop_defaults_with_warnings():
    d = derive(make_row(pull_angle_deg=None, strap_landmark=None, strap_from_landmark_m=None))
    assert d["position"]["PullAngle_deg"] == 90.0 and d["position"]["StrapAboveMalleolus_m"] == 0.05
    assert {"pull_angle_default", "strap_default"} <= codes(d)


def test_a_strap_place_from_another_landmark_is_not_converted():
    d = derive(make_row(strap_landmark="lateral_epicondyle", strap_from_landmark_m=0.33))
    assert d["position"]["StrapAboveMalleolus_m"] == 0.05 and "strap_default" in codes(d)
    assert derive()["position"]["StrapAboveMalleolus_m"] == pytest.approx(0.045)


def test_the_strap_pulls_backwards_in_knee_extension_and_forwards_in_knee_flexion():
    assert derive()["position"]["PullSideX"] == -1
    assert derive(make_row(test="knee_flexion", target=90.0))["position"]["PullSideX"] == 1


def test_the_tested_leg_decides_which_leg_has_muscles():
    right = T.render_position(derive())
    assert "#define TESTED_LEG_IS_RIGHT 1" in right and "#define BM_LEG_MUSCLES_RIGHT _MUSCLES_SIMPLE_" in right
    assert "#define BM_LEG_MUSCLES_LEFT _MUSCLES_NONE_" in right
    left = T.render_position(derive(make_row(tested_leg="left")))
    assert "#define TESTED_LEG_IS_RIGHT 0" in left and "#define BM_LEG_MUSCLES_LEFT _MUSCLES_SIMPLE_" in left
    assert "#define BM_LEG_MUSCLES_RIGHT _MUSCLES_NONE_" in left
    d = derive(make_row(tested_leg=None))
    assert d["tested_leg"] == "right" and "leg_unknown" in codes(d)


def test_data_not_judged_are_flagged_and_a_valid_trial_is_not():
    assert not derive()["reviewed"] and "not_reviewed" in codes(derive()) and "NOT REVIEWED" in T.render_position(derive())
    ok = derive(make_row(verdict="valid"))
    assert ok["reviewed"] and "not_reviewed" not in codes(ok) and "NOT REVIEWED" not in T.render_position(ok)


def test_the_person_is_height_mass_and_fat_and_the_tape_lengths_only_when_asked_for():
    d = derive()
    default = T.render_participant("P01", d["participant_values"])
    assert "BodyHeight = 1.82;" in default and "BodyMass = 78.5;" in default and "FatPercent = 17.5;" in default
    assert "SegmentDimensions =" not in default and "ThighLength" not in default.replace("thigh and shank length", "")
    with_tape = T.render_participant("P01", d["participant_values"], segment_lengths=True)
    assert "ThighLength = 0.44;" in with_tape and "ShankLength = 0.415;" in with_tape and "muscle strength" in with_tape


def test_missing_measures_are_left_out_of_participant_any_and_warned():
    d = derive(make_row(height_m=None, fat_pct=None))
    text = T.render_participant("P01", d["participant_values"])
    assert "// BodyHeight: not typed" in text and "// FatPercent: not typed" in text and "BodyMass = 78.5;" in text
    assert "measures_missing" in codes(d)


def test_a_lever_arm_that_does_not_fit_the_geometry_is_flagged():
    assert "lever_arm_mismatch" not in codes(derive())
    assert "lever_arm_mismatch" in codes(derive(make_row(lever_arm_app_m=0.12)))


# ------------------------------------------------------------------------------------------------ the geometry
def test_the_shank_hangs_down_at_knee_90_and_points_forward_when_straight():
    assert T.shank_dir_global(90, 90) == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)
    assert T.shank_dir_global(90, 0) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert T.shank_dir_global(90, 70) == pytest.approx((math.cos(math.radians(-70)), math.sin(math.radians(-70)), 0.0))


def test_the_pull_square_to_the_shank_in_knee_extension_is_minus_x_in_the_shank_frame_as_in_seatedkneeextmvic():
    assert T.pull_local(90.0, -1) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
    assert T.pull_local(90.0, +1) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert T.pull_local(0.0, -1) == pytest.approx((0.0, -1.0, 0.0))                 # along the shank, towards the feet (-y is distal)
    assert T.pull_local(180.0, -1)[1] == pytest.approx(1.0)                         # towards the body


@pytest.mark.parametrize("alpha", [0.0, 45.0, 84.0, 90.0, 120.0])
def test_the_pull_makes_the_stated_angle_with_the_shank(alpha):
    s = T.shank_dir_global(90, 70)
    for effort in (+1, -1):
        assert T.angle_between_deg(T.pull_dir_global(s, alpha, effort), s) == pytest.approx(alpha, abs=1e-9)


def test_the_strap_always_works_against_the_effort_in_both_the_global_and_the_shank_frame():
    """Physics check, independent of the formulas. The strap sits on the shank below the knee. Its moment about the knee must turn the opposite
    way to the effort: knee extension (effort about +z) gives a negative z-moment, knee flexion a positive one."""
    for knee in (20, 70, 110):
        s = T.shank_dir_global(90, knee)
        for effort in (+1, -1):
            f = T.pull_dir_global(s, 84.0, effort)
            r = tuple(0.33 * c for c in s)
            assert (r[0] * f[1] - r[1] * f[0]) * effort < 0, (knee, effort)
    for side, effort in ((-1, +1), (+1, -1)):                                      # PullSideX = -effort_axis_z
        fx, fy, _ = T.pull_local(84.0, side)
        m_z = 0.0 * fy - (-0.33) * fx                                              # r = (0, -0.33, 0) in the shank frame (y proximal)
        assert m_z * effort < 0


@pytest.mark.parametrize("knee", [20.0, 70.0, 110.0])
def test_the_shank_frame_formula_and_the_global_formula_give_the_same_direction(knee):
    """Loads in JointsAndDrivers.any are written in the ISB shank frame (x anterior, y proximal, z right); the manifest expects the global
    direction. They must be the same vector: y axis = minus the shank direction, x axis = y cross z."""
    s = T.shank_dir_global(90, knee)
    y_hat = (-s[0], -s[1], 0.0)
    x_hat = (y_hat[1], -y_hat[0], 0.0)
    for effort in (+1, -1):
        lx, ly, _ = T.pull_local(84.0, -effort)
        local_in_global = (lx * x_hat[0] + ly * y_hat[0], lx * x_hat[1] + ly * y_hat[1], 0.0)
        assert local_in_global == pytest.approx(T.pull_dir_global(s, 84.0, effort), abs=1e-9)


def test_the_force_in_the_model_is_the_formula_the_python_side_mirrors():
    text = (TEMPLATE / "Model" / "JointsAndDrivers.any").read_text(encoding="utf-8")
    assert "AnyVar Fx = Main.Position.PullSideX * sin(PullAngle) * Magnitude;" in text
    assert "AnyVar Fy = -cos(PullAngle) * Magnitude;" in text and "Flocal = {.Fx, .Fy, 0.0};" in text
    assert "ARel = .KneeJointAnatomicalFrame.ARel;" in text and "Main.Position.StrapAboveMalleolus_m" in text


# ------------------------------------------------------------------------------------------------ the model files
GROUP_DRIVERS = """PelvisGroundDriverPosX PelvisGroundDriverPosY PelvisGroundDriverPosZ PelvisGroundDriverRotX PelvisGroundDriverRotY PelvisGroundDriverRotZ
PostureDriverBending PostureDriverRotation PostureDriverExtension SkullThoraxFlexionDriver SkullThoraxLateralBendingDriver SkullThoraxRotationDriver
SCProtractionDriverRight SCElevationDriverRight GHDriverRightAbduction GHDriverRightFlexion GHDriverRightExternalRotation ElbowFlexionDriverRight
ElbowPronationDriverRight WristDriverRightFlexion WristDriverRightAbduction SCProtractionDriverLeft SCElevationDriverLeft GHDriverLeftAbduction
GHDriverLeftFlexion GHDriverLeftExternalRotation ElbowFlexionDriverLeft ElbowPronationDriverLeft WristDriverLeftFlexion WristDriverLeftAbduction""".split()
LEG_DRIVERS = ["HipDriver{S}Flexion", "HipDriver{S}ExternalRotation", "HipDriver{S}Abduction", "KneeDriver{S}", "AnkleDriver{S}Flexion",
               "AnkleDriver{S}Eversion"]


def test_every_driver_of_the_group_model_carries_reactions_here_too_and_only_the_tested_knee_does_not():
    """The list is the one of SeatedKneeExtMVIC/Model/JointsAndDrivers.any (left leg tested)."""
    text = (TEMPLATE / "Model" / "JointsAndDrivers.any").read_text(encoding="utf-8")
    for name in GROUP_DRIVERS:
        assert f"Drv.{name}.Reaction.Type = {{On}};" in text, name
    right_tested, left_tested = text.split("#else")[0], text.split("#else")[1]
    for tested_text, tested, other in ((right_tested, "Right", "Left"), (left_tested, "Left", "Right")):
        for pattern in LEG_DRIVERS:
            name = pattern.format(S=other)
            assert f"Drv.{name}.Reaction.Type = {{On}};" in tested_text, (tested, name)           # the relaxed leg: everything held
        assert "Drv.KneeDriver%s" % tested not in tested_text                                     # the tested knee: reaction stays off
        for pattern in LEG_DRIVERS:
            name = pattern.format(S=tested)
            if name != "KneeDriver" + tested:
                assert f"Drv.{name}.Reaction.Type = {{On}};" in tested_text, (tested, name)
    assert text.count("#ifndef MVIC_HIP_FREE") == 2 and text.count("#endif") >= 3


def test_the_mannequin_has_both_legs_in_both_branches_and_the_group_posture():
    text = (TEMPLATE / "Model" / "Mannequin.any").read_text(encoding="utf-8")
    assert text.count("Main.Position.KneeFlexion_deg") == 2 and text.count("Main.Position.OtherKneeFlexion_deg") == 2
    assert "PelvisPosY = 0.95;" in text and "ElbowPronation = -20;" in text and "GlenohumeralAbduction = 10;" in text
    assert "PostureVel" in text and "GlenohumeralAbduction = 10;\n      GlenohumeralExternalRotation" not in text.split("PostureVel")[1]


def test_the_group_configuration_is_kept():
    text = (TEMPLATE / "Model" / "BodyModelConfiguration.any").read_text(encoding="utf-8")
    for line in ("#define BM_MANNEQUIN_DRIVER_DEFAULT ON", "#define BM_MANNEQUIN_DRIVER_WEAK_SWITCH 0", "#define BM_ARM_DETAILED_HAND OFF",
                 "#define BM_SCALING _SCALING_LENGTHMASSFAT_", "#define BM_LEG_MODEL _LEG_MODEL_TLEM2_"):
        assert line in text
    assert "BM_LEG_MUSCLES" not in text.replace("Which leg has simple muscles", "")             # that is decided per position


def test_study_and_load_operations_match_the_group_model():
    main = (TEMPLATE / "SeatedKnee.main.any").read_text(encoding="utf-8")
    assert "InverseDynamics.Criterion.Type = MR_MinMaxStrict;" in main and "nStep = 1;" in main and "KinSolOverDeterminate" not in main
    assert "LoadPtr = &Main.Model.ModelEnvironmentConnection.StrapLoad.Magnitude;" in main
    ops = (TEMPLATE / "Model" / "RunAppSequence.any").read_text(encoding="utf-8")
    assert "Source = {&Main.Position.MeasuredForce_N};" in ops and "Source = {&Main.Position.ProbeLoad_N};" in ops
    assert ops.count("Target = {&Main.Model.ModelEnvironmentConnection.StrapLoad.Magnitude};") == 2


# ------------------------------------------------------------------------------------------------ the files
def test_position_any_carries_the_numbers_and_their_sources():
    text = T.render_position(derive())
    assert "AnyVar KneeFlexion_deg = 71.4;" in text and "AnyVar MeasuredForce_N = 612.3;" in text and "AnyVar ProbeLoad_N = 800;" in text
    assert "AnyVar PullAngle_deg = 84;" in text and "AnyVar PullSideX = -1;" in text and "AnyVar StrapAboveMalleolus_m = 0.045;" in text
    assert "trial T2" in text and "data_hash: abc123" in text and "DesignVar" not in text and text.rstrip().endswith("};")


def test_every_position_name_the_templates_use_is_written_by_the_tool():
    written = T.render_position(derive())
    used = set()
    for f in TEMPLATE.rglob("*.any"):
        used |= set(re.findall(r"Main\.Position\.(\w+)", f.read_text(encoding="utf-8")))
    assert used, "the templates read nothing from Main.Position?"
    for name in used:
        assert f"AnyVar {name} =" in written, f"{name} is used by a template but not written to Position.any"
    for define in ("TESTED_LEG_IS_RIGHT", "WITH_CHECKS", "BM_LEG_MUSCLES_RIGHT", "BM_LEG_MUSCLES_LEFT"):
        assert f"#define {define} " in written


def test_the_templates_and_the_generated_files_have_balanced_brackets_and_no_leftovers():
    for f in sorted((ROOT / "tools" / "anybody_templates").rglob("*.any")):
        assert T.lint_anyscript(f.read_text(encoding="utf-8")) == [], f.name
    d = derive()
    for text in (T.render_position(d), T.render_participant("P01", d["participant_values"], True), T.render_main(d), T.render_main(d, hip_free=True)):
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
def test_build_writes_the_shared_model_once_and_three_small_files_per_position(tmp_path):
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
        for name in ("Position.any", "Main.any", "Main_HipFree.any"):
            assert (out / m["folder"] / name).is_file()
        assert [v["name"] for v in m["variants"]] == ["hip_ideal", "hip_free"]
        for v in m["variants"]:
            assert T.missing_includes(out / v["main_file"]) == []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["models"]) == 4 and manifest["skipped"][1]["reason"].startswith("no force") and manifest["format"] == 2
    assert manifest["models"][1]["expected"]["pull_dir_global"] and manifest["models"][1]["variants"][1]["main_file"].endswith("Main_HipFree.any")
    assert (out / "manifest.csv").read_text(encoding="utf-8").splitlines()[0].startswith("id,participant,position_number")
    assert "not reviewed" in T.summary_text(res) and "without a model" in T.summary_text(res) and "hip free" in T.summary_text(res)


def test_only_the_free_hip_version_defines_mvic_hip_free_and_both_hand_over_the_two_folders(tmp_path):
    out = tmp_path / "models"
    res = T.build(T.read_rows(write_export(tmp_path, [make_row()])), out)
    folder = out / res["models"][0]["folder"]
    ideal = (folder / "Main.any").read_text(encoding="utf-8")
    free = (folder / "Main_HipFree.any").read_text(encoding="utf-8")
    assert "MVIC_HIP_FREE" not in ideal and "#define MVIC_HIP_FREE\n" in free
    for text in (ideal, free):
        assert '#path PARTICIPANT_DIR ".."' in text and '#path POSITION_DIR "."' in text
        assert '#include "../../SeatedKnee/SeatedKnee.main.any"' in text
    assert free.index("#define MVIC_HIP_FREE") < free.index("#include")


def test_a_second_run_changes_nothing_and_a_changed_force_updates_only_that_position(tmp_path):
    out = tmp_path / "models"
    rows = T.read_rows(write_export(tmp_path, [make_row(1, target=20.0), make_row(2)]))
    T.build(rows, out)
    again = T.build(rows, out)
    assert set(again["files"].values()) == {"unchanged"}
    rows[1]["force_n"] = "700"
    third = T.build(rows, out)
    assert [p for p, s in third["files"].items() if s != "unchanged"] == ["P01/pos02_knee_extension_t70/Position.any"]


def test_set_overrides_are_scoped_to_the_participant_and_position(tmp_path):
    ov = T.parse_overrides(["P01:2:force_n=450", "knee_deg=72", "P02:pull_angle_deg=80"])
    assert ov == [("P01", 2, "force_n", 450.0), (None, None, "knee_deg", 72.0), ("P02", None, "pull_angle_deg", 80.0)]
    out = tmp_path / "models"
    res = T.build(T.read_rows(write_export(tmp_path, [make_row(1, target=20.0), make_row(2)])), out, overrides=ov)
    first, second = res["models"]
    assert first["position"]["MeasuredForce_N"] == pytest.approx(612.3) and second["position"]["MeasuredForce_N"] == 450.0
    assert first["position"]["KneeFlexion_deg"] == 72.0 and second["position"]["KneeFlexion_deg"] == 72.0
    assert first["position"]["PullAngle_deg"] == 84.0                                      # P02's override did not reach P01
    assert second["overrides"] == ["force_n=450", "knee_deg=72"] and second["expected"]["measured_force_n"] == 450.0
    for bad in ("force_n=abc", "nope=1", "P01:x:force_n=1"):
        with pytest.raises(SystemExit):
            T.parse_overrides([bad])


def test_selection_by_participant_test_and_position_number(tmp_path):
    rows = T.read_rows(write_export(tmp_path, [make_row(1, target=20.0), make_row(2), make_row(3, participant="P02")]))
    assert [m["position_number"] for m in T.build(rows, tmp_path / "a", participants=["P02"])["models"]] == [3]
    assert [m["position_number"] for m in T.build(rows, tmp_path / "b", positions=[1, 2])["models"]] == [1, 2]
    assert T.build(rows, tmp_path / "c", tests=["knee_flexion"])["models"] == []


def test_segment_lengths_reach_participant_any_only_with_the_flag(tmp_path):
    rows = T.read_rows(write_export(tmp_path, [make_row()]))
    T.build(rows, tmp_path / "a")
    T.build(rows, tmp_path / "b", segment_lengths=True)
    assert "SegmentDimensions" not in (tmp_path / "a" / "P01" / "Participant.any").read_text(encoding="utf-8")
    assert "ShankLength = 0.415;" in (tmp_path / "b" / "P01" / "Participant.any").read_text(encoding="utf-8")


def test_the_libdef_line_follows_the_ammr_path_that_is_given(tmp_path):
    T.build(T.read_rows(write_export(tmp_path, [make_row()])), tmp_path / "m", ammr_libdef="C:\\Users\\x\\AMMR\\libdef.any")
    text = (tmp_path / "m" / "SeatedKnee" / "libdef.any").read_text(encoding="utf-8")
    assert '#include "C:/Users/x/AMMR/libdef.any"' in text and "ANYBODY_PATH_INSTALLDIR" not in text


def test_the_command_line_runs_and_lists_the_tests(tmp_path, capsys):
    path = write_export(tmp_path, [make_row()])
    assert T.main(["--inputs", str(path), "--out", str(tmp_path / "o"), "--set", "force_n=500", "--knee-offset-deg", "1.5"]) == 0
    assert "1 model written" in capsys.readouterr().out
    assert T.main(["--list-tests"]) == 0 and "knee_extension" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        T.main(["--inputs", str(tmp_path / "missing.csv")])


# ------------------------------------------------------------------------------------------------ the group's own model, as a regression case
def test_the_regression_case_carries_the_numbers_of_seatedkneeextmvic_subject_any(tmp_path):
    """Values from SeatedKneeExtMVIC/Input/Subject.any and Model/Mannequin.any (the group's tested model, left leg)."""
    row, overrides, offset = T.regression_row()
    out = tmp_path / "reg"
    res = T.build([row], out, overrides=overrides, knee_offset=offset)
    assert len(res["models"]) == 1 and res["models"][0]["id"] == "REG_pos01_knee_extension_t71p4"
    pos = (out / res["models"][0]["folder"] / "Position.any").read_text(encoding="utf-8")
    for line in ("AnyVar HipFlexion_deg = 82.2;", "AnyVar KneeFlexion_deg = 72.9;", "AnyVar AnklePlantarFlexion_deg = 25;",
                 "AnyVar StrapAboveMalleolus_m = 0.05;", "AnyVar PullAngle_deg = 90;", "AnyVar PullSideX = -1;", "AnyVar ProbeLoad_N = 800;",
                 "AnyVar OtherKneeFlexion_deg = 90;", "AnyVar OtherAnklePlantarFlexion_deg = 10;", "AnyVar HipAbduction_deg = 0;",
                 "#define TESTED_LEG_IS_RIGHT 0", "#define BM_LEG_MUSCLES_LEFT _MUSCLES_SIMPLE_", "#define BM_LEG_MUSCLES_RIGHT _MUSCLES_NONE_"):
        assert line in pos, line
    person = (out / "REG" / "Participant.any").read_text(encoding="utf-8")
    assert "BodyHeight = 2.06;" in person and "BodyMass = 106;" in person and "FatPercent = 15;" in person and "SegmentDimensions" not in person
    assert T.pull_local(90.0, -1) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)                       # Flocal = {-Magnitude, 0, 0}
    reg = res["models"][0]["expected"]["regression"]
    assert reg["hip_ideal"]["max_load_n"] == 2036.0 and reg["hip_free"]["max_load_n"] == 555.0 and reg["posture"]["knee_flexion_dots_deg"] == 71.4


def test_the_regression_case_command(tmp_path, capsys):
    assert T.main(["--regression-case", "--out", str(tmp_path / "r")]) == 0
    assert "1 model written" in capsys.readouterr().out and (tmp_path / "r" / "REG").is_dir()


# ------------------------------------------------------------------------------------------------ the runner
@pytest.fixture
def built(tmp_path):
    out = tmp_path / "models"
    T.build(T.read_rows(write_export(tmp_path, [make_row(1, target=20.0), make_row(2)])), out)
    return out, json.loads((out / "manifest.json").read_text(encoding="utf-8"))


A_, B_ = 0.0004, 0.05                                   # a made-up load line: activity = 0.0004 * load + 0.05


def good_result(entry, **kw):
    e = entry["expected"]
    r = {"Main.Study.Output.MaxMuscleActivity": [0.30], "Main.Check.KneeFlexion_deg": [e["knee_flexion_deg"]],
         "Main.Check.AnklePlantarFlexion_deg": [0.0], "Main.Check.PullAngle_deg": [e["pull_angle_deg"]],
         "Main.Check.PullDir": [e["pull_dir_global"]], "Main.Check.LeverArm_m": [e["lever_arm_app_m"]],
         "Main.Check.ExternalKneeMoment_Nm": [e["ext_moment_app_nm"]], "Main.Check.KneeFlexionLimbtrackDots_deg": [e["knee_dots_deg"] + 1.5],
         "Main.MaxLoading.MaxLoad": [(1 - B_) / A_],
         "Main.MaxLoading.Calculations.Load1": [800.0], "Main.MaxLoading.Calculations.Load2": [1600.0],
         "Main.MaxLoading.Calculations.MaxMuscleActivity1": [A_ * 800 + B_], "Main.MaxLoading.Calculations.MaxMuscleActivity2": [A_ * 1600 + B_]}
    r.update(kw)
    return r


def test_the_macro_sets_the_load_runs_dumps_and_does_the_maxload_last(built):
    out, manifest = built
    lines = R.macro_for(manifest["models"][1]["variants"][1]["main_file"], out)
    assert lines[0].startswith('load "') and lines[0].endswith('pos02_knee_extension_t70/Main_HipFree.any"') and lines[-1] == "exit"
    at = lambda s: lines.index(s)                                                                    # noqa: E731
    order = [at("operation Main.Ops.UseMeasuredForce"), at("operation Main.Study.InverseDynamics"),
             at('classoperation Main.Study.Output.MaxMuscleActivity "Dump"'), at('classoperation Main.Check.PullDir "Dump"'),
             at("operation Main.Ops.UseProbeLoad"), at("operation Main.MaxLoading.Evaluate"), at('classoperation Main.MaxLoading.MaxLoad "Dump"'),
             at('classoperation Main.MaxLoading.Calculations.MaxMuscleActivity2 "Dump"')]
    assert order == sorted(order)
    assert lines.count('classoperation Main "Update Values"') == 2
    assert not any("Check" in l for l in R.macro_for("P01/x/Main.any", out, with_checks=False))


def test_every_position_gives_two_tasks_one_for_each_hip_version(built):
    _, manifest = built
    tasks = R.tasks_of(manifest["models"])
    assert [(m["position_number"], v) for m, v, _ in tasks] == [(1, "hip_ideal"), (1, "hip_free"), (2, "hip_ideal"), (2, "hip_free")]


def test_only_selects_by_participant_and_position(built):
    _, manifest = built
    assert [m["position_number"] for m in R.select(manifest["models"], ["P01:2"])] == [2]
    assert len(R.select(manifest["models"], ["P01"])) == 2 and R.select(manifest["models"], None) == manifest["models"]
    assert R.select(manifest["models"], ["P09"]) == []


def test_the_strength_factor_is_read_from_the_load_line_at_the_measured_force(built):
    _, manifest = built
    entry = manifest["models"][1]
    row = R.summarise(entry, "hip_ideal", good_result(entry))
    f = 612.3
    assert row["status"] == "ok" and row["max_load_n"] == pytest.approx((1 - B_) / A_)
    assert row["line_slope"] == pytest.approx(A_) and row["line_intercept"] == pytest.approx(B_)
    assert row["scale_needed_line"] == pytest.approx(A_ * f + B_)                    # the exact one under a straight line, with the gravity term
    assert row["scale_needed_ratio"] == pytest.approx(f / ((1 - B_) / A_))           # the plain ratio without it
    assert row["scale_needed_line"] > row["scale_needed_ratio"]
    assert row["max_muscle_activity"] == pytest.approx(0.30) and row["knee_zero_offset_deg"] == pytest.approx(-1.5)
    assert row["check_knee_flexion_deg_diff"] == pytest.approx(0.0) and row["check_pull_dir_diff_deg"] == pytest.approx(0.0, abs=1e-6)


def test_a_wrong_angle_a_wrong_direction_and_a_wrong_moment_are_all_reported(built):
    _, manifest = built
    entry = manifest["models"][1]
    row = R.summarise(entry, "hip_ideal", good_result(entry, **{"Main.Check.KneeFlexion_deg": [78.0], "Main.Check.PullDir": [[0.9, 0.4, 0.0]],
                                                                "Main.Check.ExternalKneeMoment_Nm": [150.0]}))
    assert row["status"].startswith("checks differ")
    for part in ("knee_flexion_deg", "pull direction", "moment_nm"):
        assert part in row["status"]


def test_an_anybody_error_is_kept_as_the_status_and_nothing_else_is_guessed(built):
    _, manifest = built
    row = R.summarise(manifest["models"][0], "hip_free", {"ERROR": ["Error: model could not be loaded"]})
    assert row["status"].startswith("error: Error: model could not be loaded") and "max_load_n" not in row and row["variant"] == "hip_free"


def test_the_regression_case_is_compared_with_the_numbers_of_the_group_readme(tmp_path):
    row, overrides, offset = T.regression_row()
    res = T.build([row], tmp_path / "r", overrides=overrides, knee_offset=offset)
    entry = res["models"][0]

    def result(max_load, lever=0.438, knee_dots=71.4):
        line = {"Main.Check.StrapToKnee_m": [lever], "Main.Check.KneeFlexionLimbtrackDots_deg": [knee_dots], "Main.Check.ThighBelowHorizontal_deg": [7.8],
                "Main.Check.ShankForwardOfVertical_deg": [10.8], "Main.MaxLoading.MaxLoad": [max_load]}
        return line
    ok = R.summarise(entry, "hip_ideal", result(2036.0))
    assert ok["status"] == "ok" and ok["regression_max_load_n_diff"] == pytest.approx(0.0)
    assert R.summarise(entry, "hip_free", result(555.0))["status"] == "ok"
    assert R.summarise(entry, "hip_ideal", result(2036.0 * 1.03))["status"].startswith("checks differ")         # 3 % off: outside the 2 % tolerance
    bad = R.summarise(entry, "hip_ideal", result(2036.0, knee_dots=75.0))
    assert "regression_knee_dots_deg" in bad["status"]
    assert R.summarise(entry, "hip_free", result(2036.0))["status"].startswith("checks differ")                # the free hip must give 555, not 2036


def test_the_bracket_puts_the_two_hip_versions_side_by_side(built):
    _, manifest = built
    entry = manifest["models"][1]
    free = good_result(entry, **{"Main.MaxLoading.Calculations.MaxMuscleActivity1": [A_ * 3 * 800 + B_],       # a leg that needs 3 times the load
                                 "Main.MaxLoading.Calculations.MaxMuscleActivity2": [A_ * 3 * 1600 + B_],      # per unit of activity (a free hip)
                                 "Main.MaxLoading.MaxLoad": [(1 - B_) / (3 * A_)]})
    rows = [R.summarise(entry, "hip_ideal", good_result(entry)), R.summarise(entry, "hip_free", free)]
    (b,) = R.bracket(rows)
    assert b["id"] == entry["id"] and b["scale_needed_hip_free"] > b["scale_needed_hip_ideal"] and b["max_load_n_hip_free"] < b["max_load_n_hip_ideal"]


def test_run_collects_the_results_through_a_stand_in_for_anypytools(built):
    out, manifest = built
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen["kw"] = kw

        def start_macro(self, macros):
            seen["macros"] = macros
            return [good_result(m) for m in manifest["models"] for _ in m["variants"]]
    rows = R.run(out, processes=2, process_factory=Fake)
    assert seen["kw"] == {"num_processes": 2} and len(seen["macros"]) == 4 and [r["status"] for r in rows] == ["ok"] * 4
    assert [r["variant"] for r in rows] == ["hip_ideal", "hip_free"] * 2
    assert "max_load_n" in (out / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert (out / "results_bracket.csv").read_text(encoding="utf-8").splitlines()[0].startswith("id,participant,position_number")
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))[0]["id"] == manifest["models"][0]["id"]


def test_dry_run_prints_the_macros_and_runs_nothing(built, capsys):
    out, _ = built
    assert R.run(out, dry_run=True) == []
    text = capsys.readouterr().out
    assert text.count("--- P01_pos0") == 4 and "[hip_free]" in text and "operation Main.Ops.UseProbeLoad" in text
