"""0.34.0 (F1): the AnyBody export, anybody_inputs.csv and participant.json (limbtrack/anybody.py, analysis.anybody_rows, Review's Export menu).

The rule of the plan (M1 data contract, decisions D2-D4 on the plan's proposals) is tested on made-up rows, and the whole path on real recordings
(a real engine, a simulated load cell, real files read back): the rows, the SI numbers, the blanks and why, the hash of the recorded files, Original
against Reviewed, and that nothing recorded is ever written. The pilot trials are tried read-only when the folder is there."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis, anybody, review_layer  # noqa: E402
from limbtrack import verdict as verdicts  # noqa: E402
from limbtrack.completeness import default_plan  # noqa: E402
from limbtrack.engine import SOFTWARE_VERSION, Engine  # noqa: E402
from limbtrack.force import SimulatedForce  # noqa: E402
from limbtrack.protocol import load_protocol, plan_positions  # noqa: E402
from limbtrack.review import ReviewTab, Trial, trial_row  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PLAN = plan_positions(TESTS)
P = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
MEASURES = {"height_cm": 182.0, "mass_kg": 78.5, "fat_pct": 17.5, "thigh_length_cm": 44.0, "shank_length_cm": 41.5}
FACTS = {"strap_landmark": "lateral_malleolus", "strap_from_landmark_cm": 4.5, "supports": ["pelvic_belt", "hands_hold"], "seat_back_angle_deg": 90.0}


# ------------------------------------------------------------------------------------------------ made-up export rows
def row(trial, peak, wp=None, verdict="valid", unit="N", rise=True, **kw):
    r = {"trial": trial, "participant": "P01", "test_key": "knee_extension", "target_deg": 70.0, "force_unit": unit, "force_peak": peak,
         "force_window_peak": peak - 5.0 if wp is None else wp, "force_peak_above_rest": peak - 20.0 if rise else 0.0, "verdict": verdict}
    r.update(kw)
    return r


def test_the_highest_counted_peak_is_the_trial_and_the_other_is_the_check():
    a, b = row("T1", 400.0), row("T2", 440.0)
    s = anybody.select([a, b])
    assert s["selected"] is b and s["check"] is a and s["counted"] == 2 and s["valid"] == 2
    assert s["peak_diff_pct"] == pytest.approx(100 * 40.0 / 420.0)                      # the SOP 10.6 number: |a - b| / mean
    assert anybody.select([b, a])["selected"] is b                                       # not the order they were recorded in


def test_the_window_peak_decides_not_the_smoothed_peak_and_a_tie_goes_to_the_later_trial():
    a, b = row("T1", 500.0, wp=420.0), row("T2", 450.0, wp=440.0)
    assert anybody.select([a, b])["selected"] is b                                       # the force that goes to the model is the window peak (D4)
    c, d = row("T1", 400.0, wp=390.0), row("T2", 400.0, wp=390.0)
    assert anybody.select([c, d])["selected"] is d


def test_a_compromised_trial_never_counts_and_nor_does_one_without_a_force_rise():
    a, bad, flat = row("T1", 400.0), row("T2", 900.0, verdict="compromised"), row("T3", 800.0, rise=False)
    s = anybody.select([a, bad, flat])
    assert s["selected"] is a and s["check"] is None and s["counted"] == 1 and s["peak_diff_pct"] is None
    assert anybody.select([bad])["reason"] == "every trial is marked compromised"
    assert "no trial has a force rise" in anybody.select([flat])["reason"]
    assert anybody.select([])["reason"] == "never recorded"


def test_a_trial_not_judged_yet_is_not_used_unless_asked_and_a_compromised_one_never():
    a, b = row("T1", 400.0, verdict=""), row("T2", 440.0, verdict="")
    s = anybody.select([a, b])
    assert s["selected"] is None and "none marked valid yet" in s["reason"] and s["counted"] == 2
    t = anybody.select([a, b], judged_only=False)
    assert t["selected"] is b and t["check"] is a
    bad = row("T3", 900.0, verdict="compromised")
    assert anybody.select([a, bad], judged_only=False)["selected"] is a
    mixed = anybody.select([row("T1", 400.0), row("T2", 440.0, verdict="")])             # a valid one is used, the unjudged one is not the check
    assert mixed["selected"]["trial"] == "T1" and mixed["check"] is None


def test_newtons_come_before_volts():
    n, v = row("T1", 400.0), row("T2", 2.0, unit="V")
    assert anybody.select([n, v], judged_only=False)["selected"] is n


def test_every_position_of_the_protocol_has_a_posture_and_the_two_that_depend_on_the_target_are_told_apart():
    for key, tgt in PLAN:
        assert anybody.configuration(key, tgt) in {"seated", "prone", "supine", "standing", "belt squat"}, (key, tgt)
    assert anybody.configuration("hip_flexion", 0) == "supine" and anybody.configuration("hip_flexion", 90) == "seated"
    assert anybody.configuration("hip_extension_k90", 90) == "prone" and anybody.configuration("hip_extension_k90", 0) == "standing"
    assert anybody.configuration("belt_squat", 60) == "belt squat" and anybody.configuration("nonsense", 1) == ""


def test_the_contract_columns_are_all_there_in_the_planned_words():
    want = ["participant", "tested_leg", "height_m", "mass_kg", "fat_pct", "thigh_m", "shank_m", "test_key", "configuration", "target_deg", "trial_id",
            "force_n", "force_rest_n", "pull_angle_deg", "strap_landmark", "strap_from_landmark_m", "lever_arm_app_m", "hip_deg_at_peak",
            "knee_deg_at_peak", "ankle_deg_at_peak", "hip_standing_offset_deg", "knee_standing_offset_deg", "ankle_standing_offset_deg",
            "stab_pelvis_rot_deg", "angle_rest_to_hold_deg", "verdict", "software_version", "data_hash", "units", "angle_convention", "data_source"]
    assert [c for c in want if c not in anybody.COLUMNS] == [] and len(set(anybody.COLUMNS)) == len(anybody.COLUMNS)
    assert all(c in anybody.COLUMNS for c in analysis.stabilisation.COLUMNS)


# ------------------------------------------------------------------------------------------------ real recordings
def record_trial(eng, peak_n, total_s):
    """One real trial in real time with a simulated load cell; the push comes at the end of `total_s`."""
    force = SimulatedForce(rest_s=total_s - 2.7, ramp_s=0.6, hold_s=1.2, peak_n=peak_n)
    force.start()
    eng.force = force
    time.sleep(0.3)
    paths = eng.start_trial(eng.participant_code)
    while time.perf_counter() - force._t_start < total_s:
        eng.process(render(P, seed=int(time.perf_counter() * 30) % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    time.sleep(1.1)                                      # (trial file names carry the second)
    return paths["meta.json"]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """recordings/P01: knee extension 70: T1 400 N and T2 440 N (both valid), T3 250 N (compromised); knee extension 20: one trial, not judged."""
    root = tmp_path_factory.mktemp("anybody") / "recordings"
    eng = Engine(TESTS, recordings_dir=root)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.participant_code = "P01"
    eng.tested_leg = "right"
    eng.participant_measures = MEASURES
    eng.process(render(P, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    eng.setup_facts = FACTS
    t1 = record_trial(eng, 400.0, 6.6)
    t2 = record_trial(eng, 440.0, 6.6)
    t3 = record_trial(eng, 250.0, 3.0)
    eng.target = 20.0
    eng.setup_facts = {"strap_landmark": "other", "strap_landmark_other": "tape mark", "strap_from_landmark_cm": 6.0}
    u1 = record_trial(eng, 300.0, 3.0)
    verdicts.write_verdict(t1, verdicts.VALID)
    verdicts.write_verdict(t2, verdicts.VALID)
    verdicts.write_verdict(t3, verdicts.COMPROMISED, ["bad_force"], "strap slipped")
    return {"root": root, "folder": root / "P01", "t1": t1, "t2": t2, "t3": t3, "u1": u1}


def export_of(folder, source="original", judged_only=True, plan=PLAN, patch=None):
    """`patch(row)` may change an export row before the export is made (to stand in for a recording that has what the synthetic one lacks)."""
    from limbtrack import review_nav
    rows, paths = [], {}
    for e in review_nav.list_trials(folder):
        r, err = trial_row(e.meta_path, 0.1, 0.5, source)
        assert r is not None, err
        if patch:
            patch(r)
        rows.append(r)
        paths[r["trial"]] = e.meta_path

    def open_trial(p):
        return Trial(p, open_video=False, source=source)
    return analysis.anybody_rows(rows, lambda r: paths[r["trial"]], open_trial, plan=plan, judged_only=judged_only, source=source,
                                 export_version=SOFTWARE_VERSION)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def listing(folder):
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns, sha(p)) for p in Path(folder).iterdir() if p.is_file()}


def test_one_row_per_position_with_the_highest_valid_trial_and_the_other_as_check(session):
    res = export_of(session["folder"])
    assert [r["position_number"] for r in res["rows"]] == [2]                           # knee extension 70 is position 2 of the plan
    r = res["rows"][0]
    assert r["trial_id"] == session["t2"].name[: -len("_meta.json")] and r["check_trial_id"] == session["t1"].name[: -len("_meta.json")]
    assert r["n_trials_at_position"] == 3 and r["n_counted_trials"] == 2                 # the compromised one is in neither the choice nor the count
    a, b = (json.loads(session[k].read_text())["force"]["peak_n"] for k in ("t1", "t2"))
    assert r["check_peak_diff_pct"] == pytest.approx(100 * abs(a - b) / ((a + b) / 2), rel=0.05)         # (the smoothed peaks of the 10 % check)
    assert r["verdict"] == "valid" and r["data_source"] == "original" and r["selection"] == "highest counted peak"
    assert r["test_key"] == "knee_extension" and r["configuration"] == "seated" and r["target_deg"] == 70.0
    assert r["software_version"] and r["export_software_version"] == SOFTWARE_VERSION and r["units"] == anybody.UNITS


def test_the_participant_numbers_are_in_si_units_and_the_typed_facts_are_carried_over(session):
    r = export_of(session["folder"])["rows"][0]
    assert r["tested_leg"] == "right"
    assert (r["height_m"], r["mass_kg"], r["fat_pct"], r["thigh_m"], r["shank_m"]) == (1.82, 78.5, 17.5, 0.44, 0.415)
    assert r["strap_landmark"] == "lateral_malleolus" and r["strap_from_landmark_m"] == 0.045 and r["strap_landmark_other"] == ""
    assert r["supports"] == "pelvic_belt;hands_hold" and r["seat_back_angle_deg"] == 90.0


def test_the_force_is_the_window_peak_in_newtons_with_the_rest_beside_it_and_the_numbers_are_the_exports(session):
    res = export_of(session["folder"])
    r = res["rows"][0]
    exp = trial_row(session["t2"], 0.1, 0.5)[0]
    assert r["force_n"] == pytest.approx(exp["force_window_peak"], abs=1e-4) and r["force_window_ms"] == pytest.approx(500.0)
    assert r["force_rest_n"] == pytest.approx(exp["force_rest"], abs=1e-4) and r["force_peak_20ms_n"] == pytest.approx(exp["force_peak_20ms"], abs=1e-4)
    assert 380.0 < r["force_n"] < 460.0                                                   # the simulated 440 N
    assert r["angle_rest_to_hold_deg"] == pytest.approx(exp["angle_rest_to_hold_deg"], abs=1e-4)
    for c in analysis.stabilisation.COLUMNS:                                                 # F5 is passed on as it is in the session summary
        got = exp[c]
        if c in ("stab_dot1_shift", "stab_dot2_shift"):                                      # (except the shifts: metres, or blank without a ruler)
            want = "" if exp["stab_unit"] != "cm" or got in ("", None) else round(got / 100.0, 5)
            assert r[c] == want, c
            continue
        if c == "stab_unit":
            assert r[c] == ("m" if got == "cm" else got)
            continue
        assert (r[c] == "" and got in ("", None)) or r[c] == pytest.approx(got, abs=1e-4) or r[c] == got


def test_the_angles_are_read_at_the_peak_force_as_the_dots_read_them(session):
    r = export_of(session["folder"])["rows"][0]
    tr = Trial(session["t2"], open_video=False)
    tp = trial_row(session["t2"], 0.1, 0.5)[0]["force_t_peak_s"]
    assert r["angle_peak_time_s"] == pytest.approx(tp, abs=1e-4)
    assert r["hip_deg_at_peak"] == pytest.approx(tr.angle_near("hip", tp), abs=1e-4) and r["knee_deg_at_peak"] == pytest.approx(tr.angle_near("knee", tp), abs=1e-4)
    assert r["main_angle"] == "Knee flexion" and r["main_angle_at_peak_deg"] == pytest.approx(r["knee_deg_at_peak"], abs=1e-4)
    assert r["ankle_deg_at_peak"] == "" and "hip/knee/ankle" not in r["notes"]            # a knee test has no ankle angle: blank, not made up
    tr.close()


def test_a_blank_is_blank_and_the_notes_say_why(session):
    r = export_of(session["folder"])["rows"][0]
    assert r["pull_angle_deg"] == "" and r["lever_arm_app_m"] == ""                       # no strap and anchor dots, no ruler
    assert "pull_angle_deg: no strap and anchor dots" in r["notes"] and "lever_arm_app_m" in r["notes"]
    assert r["hip_standing_offset_deg"] == "" and "no standing reference" in r["notes"]
    assert all(r[f"{j}_deg_at_peak_from_reference"] == "" for j in anybody.JOINTS)       # no angle reference was set
    assert "participant measures not typed" not in r["notes"] and "strap place not typed" not in r["notes"]


def test_a_lever_arm_built_from_the_stand_in_dot_is_flagged_in_the_notes(session):
    def stand_in(r):
        if r["trial"] == session["t2"].name[: -len("_meta.json")]:
            r.update(arm_at_peak_force_cm=3.367, torque_stand_in_pictures=191)
    r = export_of(session["folder"], patch=stand_in)["rows"][0]
    assert r["lever_arm_app_m"] == 0.03367 and r["pull_angle_deg"] == ""                  # the number is kept, the row says what it is
    assert "lever_arm_app_m: the strap dot was not seen in 191 pictures" in r["notes"] and "may be an estimate" in r["notes"]

    def measured(r):
        r.update(arm_at_peak_force_cm=11.3, torque_stand_in_pictures=0)
    r = export_of(session["folder"], patch=measured)["rows"][0]
    assert r["lever_arm_app_m"] == 0.113 and "estimate" not in r["notes"]


def test_a_trial_that_recorded_no_leg_gets_a_blank_not_another_trials_leg(session, tmp_path):
    import shutil
    folder = tmp_path / "recordings" / "P01"
    shutil.copytree(session["folder"], folder)
    t2 = folder / session["t2"].name
    meta = json.loads(t2.read_text(encoding="utf-8"))
    del meta["tested_leg"]                                                                 # (a copy: a trial from before the leg was recorded)
    t2.write_text(json.dumps(meta), encoding="utf-8")
    res = export_of(folder)
    r = res["rows"][0]
    assert r["trial_id"] == t2.name[: -len("_meta.json")] and r["tested_leg"] == "" and "tested_leg: not recorded in this trial" in r["notes"]
    assert res["participants"]["P01"]["tested_leg"] == "right"                             # the participant's other trials still say so, in participant.json
    assert export_of(session["folder"])["rows"][0]["tested_leg"] == "right" and "tested_leg" not in export_of(session["folder"])["rows"][0]["notes"]


def test_the_stabilisation_shifts_are_in_metres_or_blank_like_the_rest_of_the_file(session):
    def with_ruler(r):
        r.update(stab_unit="cm", stab_dot1_name="iliac crest", stab_dot1_shift=1.34, stab_dot2_name="trochanter", stab_dot2_shift=0.5)
    r = export_of(session["folder"], patch=with_ruler)["rows"][0]
    assert r["stab_dot1_shift"] == 0.0134 and r["stab_dot2_shift"] == 0.005 and r["stab_unit"] == "m"
    assert r["stab_dot1_name"] == "iliac crest"

    def no_ruler(r):
        r.update(stab_unit="px", stab_dot1_name="iliac crest", stab_dot1_shift=6.7, stab_dot2_name="trochanter", stab_dot2_shift=2.0)
    r = export_of(session["folder"], patch=no_ruler)["rows"][0]
    assert r["stab_dot1_shift"] == "" and r["stab_dot2_shift"] == "" and r["stab_unit"] == "px"          # pixels are not SI: not given, and said
    assert "stab_dot1_shift/stab_dot2_shift: no ruler scale" in r["notes"]


def test_the_hash_is_of_the_recorded_files(session):
    r = export_of(session["folder"])["rows"][0]
    t = session["t2"]
    base = t.name[: -len("_meta.json")]
    each = {"meta.json": sha(t), "data.csv": sha(t.with_name(base + "_data.csv")), "force.csv": sha(t.with_name(base + "_force.csv"))}
    want = hashlib.sha256("".join(f"{k}:{each[k]}\n" for k in ("meta.json", "data.csv", "force.csv")).encode("ascii")).hexdigest()
    assert r["data_hash"] == want
    pj = export_of(session["folder"])["participants"]["P01"]
    assert pj["trials"][0]["sha256"] == each and pj["trials"][0]["data_hash"] == want


def test_positions_without_a_row_are_named_with_the_reason(session):
    res = export_of(session["folder"])
    by = {(s["test_key"], s["target_deg"]): s["reason"] for s in res["skipped"]}
    assert "none marked valid yet" in by[("knee_extension", 20.0)]                       # the trial there was never judged
    assert by[("hip_flexion", 0.0)] == "never recorded" and len(res["skipped"]) == len(PLAN) - 1
    pj = res["participants"]["P01"]
    assert {p["position_number"]: p["status"] for p in pj["positions"]}[1] == "no row" and pj["positions"][1]["status"] == "row"
    assert pj["tested_leg"] == "right" and pj["measures_si"]["height_m"] == 1.82 and pj["measures_si"]["thigh_m"] == 0.44
    assert pj["export"]["selection_rule"] == anybody.SELECTION_RULE and pj["export"]["judged_only"] is True


def test_asking_for_trials_not_judged_gives_them_a_row_that_says_so(session):
    res = export_of(session["folder"], judged_only=False)
    rows = {r["position_number"]: r for r in res["rows"]}
    assert set(rows) == {1, 2}
    u = rows[1]
    assert u["verdict"] == "not judged" and "not marked valid" in u["notes"]
    assert u["strap_landmark"] == "other" and u["strap_landmark_other"] == "tape mark" and u["strap_from_landmark_m"] == 0.06
    assert rows[2]["verdict"] == "valid"                                                  # the valid pair is as before; the compromised trial is out


def test_the_csv_and_the_json_are_written_and_read_back_whole(session, tmp_path):
    res = export_of(session["folder"])
    out = tmp_path / "review" / "P01" / "exports" / "anybody_inputs.csv"
    with review_layer.atomic(out, newline="") as fh:
        anybody.write_csv(fh, res["rows"])
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert list(rows[0]) == anybody.COLUMNS and len(rows) == 1
    assert rows[0]["force_n"] == f"{round(res['rows'][0]['force_n'], 5):g}" or float(rows[0]["force_n"]) == pytest.approx(res["rows"][0]["force_n"], abs=1e-4)
    assert rows[0]["pull_angle_deg"] == "" and "nan" not in out.read_text().lower()
    js = tmp_path / "review" / "P01" / "exports" / "participant.json"
    with review_layer.atomic(js) as fh:
        fh.write(anybody.json_text(res["participants"]["P01"]))
    back = json.loads(js.read_text())
    assert back["participant"] == "P01" and back["positions"][1]["trial_id"] == res["rows"][0]["trial_id"]


def test_reviewed_numbers_are_used_when_the_trial_has_fixes_and_the_originals_stay_as_they_were(session, tmp_path):
    import shutil
    folder = tmp_path / "recordings" / "P01"
    shutil.copytree(session["folder"], folder)
    t2 = folder / session["t2"].name
    before = listing(folder)
    orig = export_of(folder)["rows"][0]
    tr = Trial(t2, open_video=False)
    slope, zero = tr.slope_for_zero(), float(tr.force_cal["offset_v"])
    tr.close()
    rv = review_layer.Review.load(t2)
    rv.set_zero(zero + 0.05, "typed", slope)
    rv.save()
    assert listing(folder) == before                                                      # the review layer is a sibling tree; the folder is as it was
    rev = export_of(folder, source="reviewed")["rows"][0]
    assert rev["data_source"] == "reviewed" and rev["review_edits"] and rev["review_files_hash"]
    assert rev["force_n"] == pytest.approx(orig["force_n"] - 0.05 * slope["n_per_v"], abs=0.5)         # a zero 0.05 V higher: that much less force
    assert rev["data_hash"] == orig["data_hash"] and orig["review_files_hash"] == "" and orig["data_source"] == "original"
    assert listing(folder) == before


# ------------------------------------------------------------------------------------------------ Review's Export menu
def _app():
    return QApplication.instance() or QApplication([])


def _tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("anybody_export_tool", ROOT / "tools" / "anybody_export.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_command_line_tool_refuses_to_write_into_the_recordings_folder(session, tmp_path):
    import shutil
    tool = _tool()
    root = tmp_path / "recordings"
    shutil.copytree(session["root"], root)
    before = listing(root / "P01")
    for out in (root, root / "P01", root / "exports_here"):                                 # the root, a trial folder, a folder inside the root
        with pytest.raises(SystemExit, match="choose a folder of your own|holds a trial"):
            tool.run(root, out, judged_only=False)
        assert listing(root / "P01") == before
    assert not (root / "exports_here").exists() and not (root / "anybody_inputs.csv").exists()
    safe = tmp_path / "somewhere_else"
    result, failed = tool.run(root, safe, judged_only=False)
    assert (safe / "anybody_inputs.csv").is_file() and (safe / "P01" / "participant.json").is_file() and result["rows"]
    assert listing(root / "P01") == before                                                  # nothing under the recordings changed


def test_the_export_menu_writes_the_two_files_in_the_review_layer_only(session, tmp_path, monkeypatch):
    import shutil
    _app()
    root = tmp_path / "recordings"
    shutil.copytree(session["root"], root)
    msgs = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: msgs.append(("info", a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: msgs.append(("warn", a[2])))
    tab = ReviewTab(root, tests=TESTS)
    try:
        assert tab.act_export_anybody in tab.export_menu.actions()
        tab.set_participant(root / "P01")
        before = listing(root / "P01")
        out = tab.export_anybody()
        assert out == root / "review" / "P01" / "exports" / "anybody_inputs.csv" and out.is_file()
        assert (out.parent / "participant.json").is_file()
        assert listing(root / "P01") == before                                           # nothing next to the recorded files
        assert not list((root / "P01").glob("*anybody*")) and not list((root / "P01").glob("participant.json"))
        assert msgs and msgs[-1][0] == "info" and "1 position exported" in msgs[-1][1] and "without a row" in msgs[-1][1]
        rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
        assert len(rows) == 1 and rows[0]["verdict"] == "valid" and rows[0]["export_software_version"] == SOFTWARE_VERSION
    finally:
        if tab.trial is not None:
            tab.trial.close()
        tab.overview.shutdown()


def test_nothing_to_export_says_so_and_writes_nothing(session, tmp_path, monkeypatch):
    import shutil
    _app()
    root = tmp_path / "recordings"
    shutil.copytree(session["root"], root)
    for p in (root / "P01").glob("*_verdict.json"):
        p.unlink()                                                                       # (the copy: no trial judged)
    msgs = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: msgs.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: msgs.append(a[2]))
    tab = ReviewTab(root, tests=TESTS)
    try:
        tab.set_participant(root / "P01")
        assert tab.export_anybody() is None
        assert msgs and "No position has a counted trial that is marked valid" in msgs[0] and "none marked valid yet" in msgs[0]
        assert not (root / "review" / "P01" / "exports" / "anybody_inputs.csv").exists()
    finally:
        if tab.trial is not None:
            tab.trial.close()
        tab.overview.shutdown()


# ------------------------------------------------------------------------------------------------ the pilot trials (read only; skipped without the folder)
PILOT = ROOT / "Pilot"


@pytest.mark.skipif(not (PILOT / "P02").is_dir(), reason="the Pilot/ folder is not here")
def test_the_pilot_trials_export_and_nothing_under_pilot_changes(tmp_path):
    def snap():
        return sorted((str(p.relative_to(PILOT)), p.stat().st_size, p.stat().st_mtime_ns) for p in PILOT.rglob("*") if p.is_file())
    before = snap()
    from limbtrack import review_nav
    plan = default_plan()
    out = {}
    for pid in ("P02",):
        res = export_of(PILOT / pid, judged_only=False, plan=plan)
        out[pid] = res
        strict = export_of(PILOT / pid, plan=plan)
        assert strict["rows"] == [] or all(r["verdict"] == "valid" for r in strict["rows"])
    res = out["P02"]
    assert len(res["rows"]) + len(res["skipped"]) >= len(plan)                           # every position is either a row or named
    assert any(s["test_key"] == "hip_flexion" and s["target_deg"] == 0.0 and s["reason"] == "never recorded" for s in res["skipped"])
    assert all(r["force_n"] != "" for r in res["rows"]) and all(r["data_hash"] for r in res["rows"])
    assert all(r["verdict"] == "not judged" for r in res["rows"])                        # the pilot was never judged
    assert snap() == before
