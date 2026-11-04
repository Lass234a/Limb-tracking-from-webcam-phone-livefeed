#!/usr/bin/env python3
"""Write AnyBody models of the limbtrack MVIC positions: one shared model per test group (the group's SeatedKneeExtMVIC model made general), and for
every participant and position two small files with the numbers (Participant.any, Position.any) and two ways to open it (hip held ideally, hip
free). Reads `anybody_inputs.csv` (the AnyBody export of Review, limbtrack/anybody.py); needs nothing but the Python standard library; changes
nothing under the recordings.

usage (from the project folder):
    python tools/anybody_models.py --inputs anybody_inputs.csv --out AnyBody_models
        [--participants P01 ...] [--tests knee_extension ...] [--positions 3 4]
        [--angles auto|nominal] [--knee-offset-deg 0] [--probe-load 800] [--segment-lengths]
        [--set [P01:3:]KEY=VALUE ...] [--ammr-libdef PATH] [--force] [--list-tests]
    python tools/anybody_models.py --regression-case --out AnyBody_models      (the group's own model, to prove the port)

What it writes under --out:
    <Model>/                       the shared AnyScript model (tools/anybody_templates/<Model>), the same for every position
    <P>/Participant.any            height, mass and fat percent of the person (thigh and shank length only with --segment-lengths)
    <P>/pos<NN>_<test>_t<target>/  Main.any (hip held ideally by the pelvic belt and thigh strap), Main_HipFree.any (the hip muscles hold the hip),
                                   and Position.any (tested leg, joint angles, force, direction of the force, strap place)
    manifest.json, manifest.csv    every model with its inputs, where each number came from, what looks wrong, and the values AnyBody must
                                   reproduce on the first run (tools/anybody_models_run.py compares them)

To adjust a position, edit its Position.any, or run again with --set (for example --set P01:3:force_n=450 --set knee_deg=72). A file you edited by
hand is never overwritten: the new text is written beside it as <name>.generated.any unless you pass --force.

Nothing here is measured. Every number is read from the export; where the export is blank a stated default is used and the manifest says so.
The models have NOT been run in AnyBody by this tool's author: see .claude/skills/anybody-position-models/references/first-run-checklist.md.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE_ROOT = HERE / "anybody_templates"
FORMAT = 2
AMMR_TARGET = "AMMR 3.2.x on AnyBody 8.2"
HASH_TAG = "// generated-hash: "
VARIANTS = (("hip_ideal", "Main.any", False), ("hip_free", "Main_HipFree.any", True))
NOMINAL_TOL_DEG = 10.0                 # the SOP tolerance of the joint angle (step 9.6); a measured angle further off is reported
DEFAULT_PULL_ANGLE_DEG = 90.0          # SOP: the anchor is moved so that the pull is about square to the shank
DEFAULT_STRAP_DIST_M = 0.05            # SOP: the strap sits just above the malleoli (also the value in SeatedKneeExtMVIC)
DEFAULT_PROBE_LOAD_N = 800.0           # the load MaxLoading starts from (SeatedKneeExtMVIC); it must stay above about 300 N
OTHER_LEG_KNEE_DEG, OTHER_LEG_ANKLE_PF_DEG = 90.0, 10.0      # the relaxed leg hangs over the table edge (SeatedKneeExtMVIC)
LEVER_ARM_TOLERANCE = 0.20             # data check only: the app's lever arm against strap-to-knee distance times sin(pull angle)

# --------------------------------------------------------------------------------------------------------------------- the tests
# Axes: x forward (the way the participant faces), y up, z to the participant's right. effort_axis_z = the sign of the rotation about z that
# the participant's effort tends to give the segment the strap acts on: +1 knee extension (shank swings forward and up), -1 knee flexion.
# The strap works the other way round, so in the shank frame (x anterior) it pulls BACKWARDS for knee extension: PullSideX = -effort_axis_z.
# 'drives' = how each joint of the model is set: policy 'measured' = the angle at the peak force from the export (falls back to the target),
# 'nominal' = the value the protocol gives. sign converts limbtrack's convention to AnyBody's (ankle: limbtrack + = dorsiflexion,
# AnyBody AnklePlantarFlexion + = plantarflexion; knee and hip flexion are + in both).
SEATED_KNEE_DRIVES = {
    "knee": {"var": "KneeFlexion_deg", "sign": 1.0, "policy": "measured"},
    "hip": {"var": "HipFlexion_deg", "sign": 1.0, "policy": "nominal", "nominal": 90.0,
            "why": "seated, trunk upright; the app's hip angle is not trusted"},
    "ankle": {"var": "AnklePlantarFlexion_deg", "sign": -1.0, "policy": "nominal", "nominal": 0.0,
              "why": "foot hangs relaxed, no ankle angle in the protocol"},
}
REGISTRY = {
    "knee_extension": {"label": "Knee extension, hip 90", "status": "built", "model": "SeatedKnee", "main_joint": "knee", "segment": "shank",
                       "effort_axis_z": 1, "scaling_group": "knee extensors", "drives": SEATED_KNEE_DRIVES},
    "knee_flexion": {"label": "Knee flexion, hip 90", "status": "built (the same model, the strap on the other side; not looked at yet)",
                     "model": "SeatedKnee", "main_joint": "knee", "segment": "shank", "effort_axis_z": -1,
                     "scaling_group": "hamstrings", "drives": SEATED_KNEE_DRIVES},
    "hip_flexion": {"label": "Hip flexion, knee 90", "status": "planned", "segment": "thigh", "effort_axis_z": 1,
                    "scaling_group": "hip flexors", "differs": "strap on the distal thigh; supine at hip 0; pelvic belt; hip 0 and 90"},
    "hip_extension_k0": {"label": "Hip extension, knee straight", "status": "planned", "segment": "leg (thigh and shank, straight)",
                         "effort_axis_z": -1, "scaling_group": "gluteus maximus and hamstrings",
                         "differs": "standing on the other leg, trunk on the bench at hip 45; strap on the distal shank; the stance leg needs a support"},
    "hip_extension_k90": {"label": "Hip extension, knee 90", "status": "planned", "segment": "thigh (shank hangs)", "effort_axis_z": -1,
                          "scaling_group": "gluteus maximus", "differs": "standing on the other leg; hip 0 and 90 (the SOP and the protocol differ on 90)"},
    "hip_abduction": {"label": "Hip abduction, hip 0", "status": "planned", "segment": "leg", "effort_axis_z": None,
                      "scaling_group": "hip abductors", "differs": "frontal plane: the effort axis is x; camera in front; pivot not decided (ASIS or hip centre)"},
    "hip_adduction": {"label": "Hip adduction, from abduction", "status": "planned", "segment": "leg", "effort_axis_z": None,
                      "scaling_group": "hip adductors", "differs": "frontal plane, as abduction; held at 0 and 20 deg abduction"},
    "ankle_plantarflexion": {"label": "Plantarflexion, knee straight", "status": "planned", "segment": "foot", "effort_axis_z": -1,
                             "scaling_group": "triceps surae", "differs": "standing on the tested leg; strap under the foot from a hip belt; stance support"},
    "ankle_dorsiflexion": {"label": "Dorsiflexion, knee straight", "status": "planned", "segment": "foot", "effort_axis_z": 1,
                           "scaling_group": "tibialis anterior", "differs": "foot in a padded block; the knee is 90 in the pilot but 0 in the protocol (TBD)"},
    "belt_squat": {"label": "Belt squat (validation)", "status": "planned", "segment": "pelvis", "effort_axis_z": None,
                   "scaling_group": "(not used for scaling)", "differs": "whole body, both feet on a platform, load through a hip belt"},
}
LEGS = ("right", "left")
OVERRIDE_KEYS = {              # --set key -> (Position variable, factor from limbtrack's convention to AnyBody's)
    "force_n": ("MeasuredForce_N", 1.0), "probe_load_n": ("ProbeLoad_N", 1.0), "pull_angle_deg": ("PullAngle_deg", 1.0),
    "strap_dist_m": ("StrapAboveMalleolus_m", 1.0), "knee_deg": ("KneeFlexion_deg", 1.0), "hip_deg": ("HipFlexion_deg", 1.0),
    "ankle_deg": ("AnklePlantarFlexion_deg", -1.0), "hip_abduction_deg": ("HipAbduction_deg", 1.0),
}

# The group's own model (SeatedKneeExtMVIC, left leg, 2.06 m, 106 kg, 15 %), as limbtrack-style inputs, to prove the port: the generated files must carry
# the same numbers as its Input/Subject.any, and AnyBody must give the numbers of its README. See regression_row().
REGRESSION = {
    "participant": "REG",
    "expected": {"hip_ideal": {"max_load_n": 2036.0, "torque_nm": 891.0, "lever_arm_m": 0.438},
                 "hip_free": {"max_load_n": 555.0, "torque_nm": 243.0},
                 "posture": {"knee_flexion_dots_deg": 71.4, "thigh_below_horizontal_deg": 7.8, "shank_forward_of_vertical_deg": 10.8}},
    "tolerance": 0.02,
}


# --------------------------------------------------------------------------------------------------------------------- small helpers
def num(v):
    """A float from a CSV cell or None for a blank or a word."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def fmt(v):
    """A number for AnyScript: short, no exponent for the sizes used here."""
    if isinstance(v, int):
        return str(v)
    s = f"{v:.6g}"
    return "0" if s in ("-0", "-0.0") else s


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tag(x):
    """90 -> '90', -20 -> 'm20', 22.5 -> '22p5' (for folder names)."""
    return f"{x:g}".replace("-", "m").replace(".", "p")


def comment_safe(text):
    return " ".join(str(text).replace("*/", "* /").split())


# --------------------------------------------------------------------------------------------------------------------- geometry (global mirror)
def shank_dir_global(hip_flex_deg, knee_flex_deg):
    """Unit vector knee -> ankle in the sagittal plane for an upright pelvis: the thigh is horizontal forward at hip flexion 90 and the
    shank hangs down at knee flexion 90. Axes: x forward, y up. The model's own neutral angles differ by a few degrees (checked on the first run)."""
    phi = math.radians(hip_flex_deg - 90.0 - knee_flex_deg)
    return (math.cos(phi), math.sin(phi), 0.0)


def tangent_global(direction, effort_axis_z):
    """The sideways part of the pull: cross((0, 0, -effort_axis_z), direction)."""
    c = -float(effort_axis_z)
    return (-c * direction[1], c * direction[0], 0.0)


def pull_dir_global(shank, pull_angle_deg, effort_axis_z):
    a = math.radians(pull_angle_deg)
    t = tangent_global(shank, effort_axis_z)
    return tuple(math.cos(a) * s + math.sin(a) * u for s, u in zip(shank, t))


def pull_local(pull_angle_deg, pull_side_x):
    """The pull in the ISB shank frame (x anterior, y proximal): what Flocal of the strap force is, per newton. At 90 deg and PullSideX -1 it is
    (-1, 0, 0), the force of SeatedKneeExtMVIC."""
    a = math.radians(pull_angle_deg)
    return (pull_side_x * math.sin(a), -math.cos(a), 0.0)


def angle_between_deg(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return math.degrees(math.acos(max(-1.0, min(1.0, d / (na * nb)))))


# --------------------------------------------------------------------------------------------------------------------- reading the export
def read_rows(path):
    """The rows of anybody_inputs.csv as dicts; a blank cell is None."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (v if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(fh)]


def row_hash(row):
    """A short hash of the cells this tool reads, so a changed export is seen in the manifest."""
    keys = sorted(k for k in row if k not in ("notes", "_regression"))
    return sha("|".join(f"{k}={row[k]}" for k in keys))[:16]


def parse_overrides(items):
    """['P01:3:force_n=450', 'knee_deg=72'] -> [(participant or None, position number or None, key, value)]."""
    out = []
    for item in items or []:
        left, _, value = item.rpartition("=")
        parts = left.split(":")
        key = parts[-1].strip().lower()
        who = parts[0] if len(parts) >= 2 else None
        pos = parts[1] if len(parts) >= 3 else None
        if key not in OVERRIDE_KEYS:
            raise SystemExit(f"--set {item}: unknown key '{key}'. Known keys: {', '.join(sorted(OVERRIDE_KEYS))}")
        v = num(value)
        if v is None:
            raise SystemExit(f"--set {item}: '{value}' is not a number")
        pn = num(pos) if pos is not None else None
        if pos is not None and pn is None:
            raise SystemExit(f"--set {item}: position '{pos}' is not a number")
        out.append((who, None if pn is None else int(pn), key, v))
    return out


# --------------------------------------------------------------------------------------------------------------------- one position
def pick_angle(row, joint, drive, spec, angles, flag):
    """(value in limbtrack's convention, where it came from, the measured value or None) for one joint of one position."""
    j = joint
    measured, how, referenced = None, "", True
    for col, text in ((f"{j}_deg_at_peak_from_reference", "measured at the peak force, from the standing reference"),
                      (f"{j}_deg_at_peak", "measured at the peak force, dots only (no standing reference)")):
        v = num(row.get(col))
        if v is not None:
            measured, how, referenced = v, text, col.endswith("_from_reference")
            break
    if measured is None and spec["main_joint"] == j and row.get("main_angle") == j:
        v = num(row.get("main_angle_at_peak_deg"))
        if v is not None:
            measured, how, referenced = v, "main angle at the peak force", False
    target = num(row.get("target_deg")) if spec["main_joint"] == j else None
    if drive["policy"] == "nominal":
        value, source = drive["nominal"], f"nominal ({drive['why']})"
        if measured is not None and abs(measured - value) > NOMINAL_TOL_DEG:
            flag("info", f"{j}_off_nominal", f"{j}: measured {measured:.1f} deg against {value:.0f} nominal; the nominal value is used. {how}")
        return value, source, measured
    if angles == "nominal" and target is not None:
        return target, "target of the position (--angles nominal)", measured
    if measured is not None:
        if not referenced:
            flag("info", f"{j}_no_reference", f"{j}: no standing reference in this trial, so the angle is what the dots read")
        if target is not None and abs(measured - target) > NOMINAL_TOL_DEG:
            flag("warn", f"{j}_off_target", f"{j}: measured {measured:.1f} deg is {abs(measured - target):.1f} deg from the target {target:g}")
        return measured, how, measured
    if target is not None:
        flag("warn", f"{j}_not_measured", f"{j}: no measured angle in the export, the target {target:g} deg is used")
        return target, "target of the position (no measured angle in the export)", None
    return None, "", None


def derive(row, overrides=(), angles="auto", knee_offset=0.0, probe_load=DEFAULT_PROBE_LOAD_N):
    """Everything the model files and the manifest need for ONE export row: (dict, '') or (None, reason) when no model can be made."""
    key = row.get("test_key")
    spec = REGISTRY.get(key)
    if spec is None:
        return None, f"test '{key}' is not known to this tool"
    if not str(spec["status"]).startswith("built"):
        return None, f"no model yet for {key} ({spec['label']}); what is different is in references/positions.md"
    flags = []

    def flag(level, code, text):
        flags.append({"level": level, "code": code, "text": text})

    force = num(row.get("force_n"))
    if force is None or force <= 0:
        return None, "no force in newtons in the export (force in volts, no calibration, or no force recorded)"
    who, n = row.get("participant") or "", int(num(row.get("position_number")) or 0)
    target = num(row.get("target_deg"))
    leg = (row.get("tested_leg") or "").lower()
    if leg not in LEGS:
        leg = "right"
        flag("warn", "leg_unknown", "tested leg not recorded: the right leg is assumed")
    verdict = row.get("verdict") or "not judged"
    reviewed = verdict == "valid"
    if not reviewed:
        flag("warn", "not_reviewed", f"trial verdict is '{verdict}': the data are not reviewed yet")

    pos, src, measured_info = {}, {}, {}
    for joint, drive in spec["drives"].items():
        value, source, measured = pick_angle(row, joint, drive, spec, angles, flag)
        if value is None:
            return None, f"no {joint} angle (no measured angle and no target) in the export"
        if joint == "knee" and knee_offset:
            value, source = value + knee_offset, f"{source}; plus the knee zero offset {knee_offset:g} deg (--knee-offset-deg)"
        pos[drive["var"]], src[drive["var"]] = drive["sign"] * value + 0.0, source           # (+ 0.0 turns -0.0 into 0.0)
        if measured is not None:
            measured_info[f"{joint}_measured_deg"] = measured
    pos["HipAbduction_deg"], src["HipAbduction_deg"] = 0.0, "assumed: the thigh points straight forward on the bench"
    pos["HipExternalRotation_deg"], src["HipExternalRotation_deg"] = 0.0, "assumed neutral"
    pos["SubTalarEversion_deg"], src["SubTalarEversion_deg"] = 0.0, "assumed neutral"
    pos["OtherKneeFlexion_deg"], src["OtherKneeFlexion_deg"] = OTHER_LEG_KNEE_DEG, "the relaxed leg hangs over the table edge (SeatedKneeExtMVIC)"
    pos["OtherAnklePlantarFlexion_deg"], src["OtherAnklePlantarFlexion_deg"] = OTHER_LEG_ANKLE_PF_DEG, "as in SeatedKneeExtMVIC"

    pos["MeasuredForce_N"], src["MeasuredForce_N"] = force, ("total force from the load cell zero, sliding-window peak of "
                                                              f"{fmt(num(row.get('force_window_ms')) or 500)} ms (force_n)")
    pos["ProbeLoad_N"], src["ProbeLoad_N"] = probe_load, "the load MaxLoading starts from; keep it above about 300 N"
    rest = num(row.get("force_rest_n"))
    if rest is not None and rest > 0.1 * force:
        flag("info", "rest_force", f"resting force {rest:.0f} N is {100 * rest / force:.0f}% of the peak; the total force is used and the weight "
                                   "of the limb comes from gravity in the model (SOP 9.10): do not subtract the rest as well")
    pull = num(row.get("pull_angle_deg"))
    if pull is None:
        pull = DEFAULT_PULL_ANGLE_DEG
        src["PullAngle_deg"] = "default: square to the shank (no strap and anchor dots, or the strap dot was lost at the peak)"
        flag("warn", "pull_angle_default", f"no pull angle in the export: {pull:.0f} deg (square to the shank) is used")
    else:
        src["PullAngle_deg"] = "pull_angle_deg at the peak force (2D, from the strap and anchor dots)"
    pos["PullAngle_deg"] = pull
    pos["PullSideX"] = -spec["effort_axis_z"]
    src["PullSideX"] = f"fixed for {key}: -1 = the strap pulls backwards (resists knee extension), +1 forwards"

    landmark, dist = row.get("strap_landmark"), num(row.get("strap_from_landmark_m"))
    if dist is not None and landmark == "lateral_malleolus":
        pos["StrapAboveMalleolus_m"], src["StrapAboveMalleolus_m"] = dist, "typed in the set-up facts: distance above the lateral malleolus"
    else:
        pos["StrapAboveMalleolus_m"] = DEFAULT_STRAP_DIST_M
        src["StrapAboveMalleolus_m"] = "default: just above the malleoli (SOP); the strap place was not typed or is from another landmark"
        flag("warn", "strap_default", f"strap place not usable ({landmark or 'not typed'}): {DEFAULT_STRAP_DIST_M} m above the malleolus is used")

    applied = []
    for (o_who, o_pos, o_key, o_val) in overrides:
        if (o_who is None or o_who == who) and (o_pos is None or o_pos == n):
            var, factor = OVERRIDE_KEYS[o_key]
            pos[var], src[var] = factor * o_val, f"override --set {o_key}={o_val:g}"
            applied.append(f"{o_key}={o_val:g}")
            flag("info", "override", f"{o_key} set to {o_val:g} by --set")

    mm = {"height_m": num(row.get("height_m")), "mass_kg": num(row.get("mass_kg")), "fat_pct": num(row.get("fat_pct")),
          "thigh_m": num(row.get("thigh_m")), "shank_m": num(row.get("shank_m"))}
    missing = [k for k in ("height_m", "mass_kg", "fat_pct") if mm[k] is None]
    if missing:
        flag("warn", "measures_missing", "participant measures not typed: " + ", ".join(missing) + " (the AMMR default is used for each)")

    force_used = pos["MeasuredForce_N"]
    shank = shank_dir_global(pos["HipFlexion_deg"], pos["KneeFlexion_deg"])
    pdir = pull_dir_global(shank, pos["PullAngle_deg"], spec["effort_axis_z"])
    lever_app = num(row.get("lever_arm_app_m"))
    expected = {"measured_force_n": force_used, "probe_load_n": probe_load, "knee_flexion_deg": pos["KneeFlexion_deg"],
                "ankle_plantar_flexion_deg": pos["AnklePlantarFlexion_deg"], "pull_angle_deg": pos["PullAngle_deg"],
                "shank_dir_global": [round(x, 5) for x in shank], "pull_dir_global": [round(x, 5) for x in pdir], "lever_arm_app_m": lever_app,
                "knee_dots_deg": num(row.get("knee_deg_at_peak"))}
    if lever_app is not None:
        expected["ext_moment_app_nm"] = force_used * lever_app
        if mm["shank_m"] is not None:
            geom = (mm["shank_m"] - pos["StrapAboveMalleolus_m"]) * math.sin(math.radians(pos["PullAngle_deg"]))
            expected["lever_arm_geometry_m"] = geom
            if geom > 0 and abs(lever_app - geom) / geom > LEVER_ARM_TOLERANCE:
                flag("warn", "lever_arm_mismatch", f"the app's lever arm {lever_app:.3f} m is not near strap-to-knee x sin(pull angle) = {geom:.3f} m "
                                                   "(shank length, strap place or pull angle may be wrong)")
    else:
        flag("info", "lever_arm_missing", "no lever arm in the export: the first-run moment check has nothing to compare with")
    if row.get("_regression"):
        expected["regression"] = row["_regression"]

    return {"id": f"{who}_pos{n:02d}_{key}_t{tag(target) if target is not None else 'x'}", "participant": who, "position_number": n,
            "test_key": key, "target_deg": target, "trial_id": row.get("trial_id"), "verdict": verdict, "reviewed": reviewed,
            "data_source": row.get("data_source") or "", "data_hash": row.get("data_hash") or "", "row_hash": row_hash(row),
            "tested_leg": leg, "model": spec["model"], "scaling_group": spec["scaling_group"], "participant_values": mm,
            "position": pos, "sources": src, "measured": measured_info, "overrides": applied, "flags": flags, "expected": expected}, ""


# --------------------------------------------------------------------------------------------------------------------- the files
def render_position(d):
    pos, src = d["position"], d["sources"]
    right = d["tested_leg"] == "right"
    target = f"   target {d['target_deg']:g} deg" if d["target_deg"] is not None else ""
    head = [
        "// Position.any: written by tools/anybody_models.py. Edit the numbers below to adjust this position, then reload the model in AnyBody.",
        f"// {d['participant']}   position {d['position_number']}   {d['test_key']}{target}   trial {d['trial_id']}",
        f"// data: {d['data_source'] or 'unknown'}, verdict: {d['verdict']}" + ("" if d["reviewed"] else "   ** NOT REVIEWED **"),
        f"// data_hash: {d['data_hash'] or '-'}   row_hash: {d['row_hash']}",
        "// Units: degrees, newtons, metres. Angles are AnyBody's: knee and hip flexion +, ankle PLANTARflexion + (limbtrack reads dorsiflexion +).",
        "// PullAngle_deg is between the pull and the shank, read from the knee towards the ankle (90 = square to the shank).",
        "// MeasuredForce_N is the force of the trial; ProbeLoad_N is the load that MaxLoading starts from (keep it above about 300 N).",
        "",
        "// Which leg is tested; the tested leg has simple muscles, the other leg is held by joint actuators",
        f"#define TESTED_LEG_IS_RIGHT {1 if right else 0}",
        f"#define BM_LEG_MUSCLES_{'RIGHT' if right else 'LEFT'} _MUSCLES_SIMPLE_",
        f"#define BM_LEG_MUSCLES_{'LEFT' if right else 'RIGHT'} _MUSCLES_NONE_",
        "// 1 = read back the angles and the force direction (Model/Checks.any); 0 = leave out",
        "#define WITH_CHECKS 1",
        "",
        "AnyFolder Position =",
        "{",
    ]
    groups = [
        ("Tested leg (angles)", ["HipFlexion_deg", "HipAbduction_deg", "HipExternalRotation_deg", "KneeFlexion_deg", "AnklePlantarFlexion_deg",
                                 "SubTalarEversion_deg"]),
        ("The load-cell strap on the shank", ["MeasuredForce_N", "ProbeLoad_N", "PullAngle_deg", "PullSideX", "StrapAboveMalleolus_m"]),
        ("The other leg, hanging relaxed", ["OtherKneeFlexion_deg", "OtherAnklePlantarFlexion_deg"]),
    ]
    body = []
    for title, names in groups:
        body.append(f"  // {title}")
        for name in names:
            decl = f"  AnyVar {name} = {fmt(pos[name])};"
            body.append(f"{decl:<46}// {comment_safe(src.get(name, ''))}" if src.get(name) else decl)
        body.append("")
    return "\n".join(head + body + ["};"])


def render_participant(who, mm, segment_lengths=False):
    lines = [f"// Participant.any: written by tools/anybody_models.py. {who}. Typed by the operator in limbtrack (tape and scale), SI units.",
             "// A value left out was not typed: AMMR's own default is used for it (BodyHeight 1.75 m, BodyMass 75 kg, the fat percent from the BMI).",
             "// Included BEFORE the human model: the ??= defaults of the AMMR then yield to these values.",
             "HumanModel.Anthropometrics =", "{"]
    for var, key in (("BodyHeight", "height_m"), ("BodyMass", "mass_kg"), ("FatPercent", "fat_pct")):
        lines.append(f"  {var} = {fmt(mm[key])};" if mm.get(key) is not None else f"  // {var}: not typed, AMMR default")
    seg = " ".join(s for s in (f"ThighLength = {fmt(mm['thigh_m'])};" if mm.get("thigh_m") is not None else "",
                               f"ShankLength = {fmt(mm['shank_m'])};" if mm.get("shank_m") is not None else "") if s)
    if segment_lengths and seg:
        lines += ["  // Tape thigh and shank length for both legs (--segment-lengths). NOTE: in the AMMR the muscle fibre length follows the segment length and",
                  "  // the muscle force is volume / fibre length, so these two numbers change the muscle strength that is being scaled.",
                  "  SegmentDimensions =", "  {", f"    Right = {{ {seg} }};", f"    Left  = {{ {seg} }};", "  };"]
    else:
        lines.append("  // thigh and shank length: from the height (AMMR default); --segment-lengths uses the tape measures")
    lines.append("};")
    return "\n".join(lines)


def render_main(d, hip_free=False):
    variant = "the hip muscles hold the hip" if hip_free else "the hip is held ideally by the pelvic belt and thigh strap"
    lines = [f"// Opens {d['id']}: {variant}. The shared model is in ../../{d['model']}; the numbers are in ../Participant.any and Position.any (this folder)."]
    if hip_free:
        lines.append("#define MVIC_HIP_FREE")
    lines += ['#path PARTICIPANT_DIR ".."', '#path POSITION_DIR "."', f'#include "../../{d["model"]}/{d["model"]}.main.any"']
    return "\n".join(lines)


def write_generated(path, body, force=False):
    """Write `body` plus a hash line. Returns 'written', 'unchanged', or 'kept' when the file on disk was changed by hand: then the new text
    goes beside it as <name>.generated<suffix> and the file itself is left alone (unless force)."""
    body = body.rstrip("\n")
    text = f"{body}\n{HASH_TAG}{sha(body)}\n"
    path = Path(path)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == text:
            return "unchanged"
        if not force and not _untouched(old):
            path.with_name(f"{path.stem}.generated{path.suffix}").write_text(text, encoding="utf-8", newline="\n")
            return "kept"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return "written"


def _untouched(text):
    lines = text.rstrip("\n").split("\n")
    return bool(lines) and lines[-1].startswith(HASH_TAG) and lines[-1][len(HASH_TAG):] == sha("\n".join(lines[:-1]))


def copy_model(model, out, ammr_libdef=None, force=False):
    """The shared AnyScript model into out/<model>; returns {relative path: status}."""
    src_root = TEMPLATE_ROOT / model
    if not src_root.is_dir():
        raise SystemExit(f"template folder {src_root} not found")
    status = {}
    for f in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = f.relative_to(src_root)
        text = f.read_text(encoding="utf-8")
        if ammr_libdef and f.name == "libdef.any":
            text = text.replace('#include "<ANYBODY_PATH_INSTALLDIR>/AMMR/libdef.any"', f'#include "{str(ammr_libdef).replace(chr(92), "/")}"')
        status[rel.as_posix()] = write_generated(Path(out) / model / rel, text, force)
    return status


# --------------------------------------------------------------------------------------------------------------------- checks on the AnyScript text
def lint_anyscript(text):
    """Cheap checks that catch the usual typing slips: unbalanced brackets, an open string or comment, a leftover placeholder. Returns problems."""
    problems, stack, i, line, n = [], [], 0, 1, len(text)
    pairs = {")": "(", "]": "[", "}": "{"}
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                problems.append(f"line {line}: comment /* is never closed")
                break
            line += text.count("\n", i, j)
            i = j + 2
            continue
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            if j >= n or text[j] != '"':
                problems.append(f"line {line}: string is never closed")
                break
            i = j
        elif c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            if not stack or stack[-1][0] != pairs[c]:
                problems.append(f"line {line}: '{c}' has no matching opening bracket")
            else:
                stack.pop()
        i += 1
    problems += [f"line {ln}: '{c}' is never closed" for c, ln in stack]
    for m in re.finditer(r"\{\{|\}\}|\$\{|%\(", text):
        problems.append(f"leftover placeholder '{m.group(0)}' near character {m.start()}")
    return problems


def missing_includes(main_path, seen=None):
    """Relative #include files that do not exist, followed through the files that do. Paths with <NAME> are resolved by AnyBody and skipped."""
    seen = set() if seen is None else seen
    main_path = Path(main_path)
    if main_path in seen:
        return []
    seen.add(main_path)
    out = []
    for m in re.finditer(r'^\s*#include\s+"([^"]+)"', main_path.read_text(encoding="utf-8"), re.M):
        target = m.group(1)
        if "<" in target:
            continue
        p = (main_path.parent / target).resolve()
        if not p.exists():
            out.append(f"{main_path.name}: {target}")
        else:
            out += missing_includes(p, seen)
    return out


# --------------------------------------------------------------------------------------------------------------------- the group's own model
def regression_row():
    """(row, overrides, knee_offset) that make this tool write the numbers of the group's SeatedKneeExtMVIC model (Input/Subject.any): left leg,
    2.06 m, 106 kg, 15 %, knee 71.4 deg from the dots + 1.5 deg zero offset = 72.9, hip 82.2, ankle 25 deg plantarflexion, strap 0.05 m, square pull."""
    row = {"participant": REGRESSION["participant"], "tested_leg": "left", "height_m": "2.06", "mass_kg": "106", "fat_pct": "15",
           "position_number": "1", "test_key": "knee_extension", "target_deg": "71.4", "trial_id": "SeatedKneeExtMVIC", "force_n": "800",
           "force_window_ms": "500", "pull_angle_deg": "90", "strap_landmark": "lateral_malleolus", "strap_from_landmark_m": "0.05",
           "main_angle": "knee", "knee_deg_at_peak": "71.4", "knee_deg_at_peak_from_reference": "71.4", "verdict": "valid",
           "data_source": "regression (no participant)", "data_hash": "-", "_regression": REGRESSION["expected"]}
    overrides = [(None, None, "hip_deg", 82.2), (None, None, "ankle_deg", -25.0)]         # limbtrack's ankle sign: -25 = 25 deg plantarflexion
    return row, overrides, 1.5


# --------------------------------------------------------------------------------------------------------------------- the whole run
def build(rows, out, participants=None, tests=None, positions=None, angles="auto", overrides=(), ammr_libdef=None, force=False, inputs_name="",
          inputs_hash="", knee_offset=0.0, probe_load=DEFAULT_PROBE_LOAD_N, segment_lengths=False):
    """Write everything under `out`. Returns {'models': [...], 'skipped': [...], 'files': {path: status}}."""
    out = Path(out)
    result = {"models": [], "skipped": [], "files": {}}
    used, wrote_model, wrote_participant = set(), set(), {}
    for row in rows:
        who, key = row.get("participant") or "", row.get("test_key")
        n = int(num(row.get("position_number")) or 0)
        if participants and who not in participants or tests and key not in tests or positions and n not in positions:
            continue
        d, reason = derive(row, overrides, angles, knee_offset, probe_load)
        if d is None:
            result["skipped"].append({"participant": who, "position_number": n, "test_key": key, "target_deg": num(row.get("target_deg")),
                                      "reason": reason})
            continue
        if d["id"] in used:
            d["id"] += f"_{row.get('trial_id')}"
            d["flags"].append({"level": "warn", "code": "duplicate_position", "text": "two rows for this position in the export"})
        used.add(d["id"])
        folder = out / (who or "unknown") / d["id"].split(f"{who}_", 1)[-1]
        if d["model"] not in wrote_model:
            for rel, st in copy_model(d["model"], out, ammr_libdef, force).items():
                result["files"][f"{d['model']}/{rel}"] = st
            wrote_model.add(d["model"])
        wrote_participant[who] = render_participant(who, d["participant_values"], segment_lengths)
        files = [("Position.any", render_position(d))] + [(name, render_main(d, hip_free)) for _v, name, hip_free in VARIANTS]
        for name, text in files:
            result["files"][(folder / name).relative_to(out).as_posix()] = write_generated(folder / name, text, force)
        d["folder"] = folder.relative_to(out).as_posix()
        d["variants"] = [{"name": v, "main_file": f"{d['folder']}/{name}"} for v, name, _h in VARIANTS]
        d["main_file"] = d["variants"][0]["main_file"]
        result["models"].append(d)
    for who, text in wrote_participant.items():
        result["files"][f"{who or 'unknown'}/Participant.any"] = write_generated(out / (who or "unknown") / "Participant.any", text, force)
    manifest = {"format": FORMAT, "generated_by": "tools/anybody_models.py", "target": AMMR_TARGET, "inputs": inputs_name, "inputs_sha256": inputs_hash,
                "angles": angles, "knee_offset_deg": knee_offset, "segment_lengths": segment_lengths, "models": result["models"],
                "skipped": result["skipped"],
                "meaning": "Nothing is measured here. Each number in a Position.any is read from the export, or a stated default. Every position has "
                           "two variants: the hip held ideally and the hip free; the truth is between. The models have not been run in AnyBody by "
                           "this tool's author until a first run has passed (references/first-run-checklist.md)."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_manifest_csv(out / "manifest.csv", result["models"])
    return result


def write_manifest_csv(path, models):
    cols = ["id", "participant", "position_number", "test_key", "target_deg", "trial_id", "verdict", "reviewed", "tested_leg", "force_n",
            "knee_flexion_deg", "hip_flexion_deg", "pull_angle_deg", "strap_dist_m", "flags", "folder"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in models:
            p = m["position"]
            w.writerow([m["id"], m["participant"], m["position_number"], m["test_key"], m["target_deg"], m["trial_id"], m["verdict"], m["reviewed"],
                        m["tested_leg"], fmt(p["MeasuredForce_N"]), fmt(p["KneeFlexion_deg"]), fmt(p["HipFlexion_deg"]), fmt(p["PullAngle_deg"]),
                        fmt(p["StrapAboveMalleolus_m"]), "; ".join(f["code"] for f in m["flags"]), m["folder"]])


def summary_text(result):
    models, skipped = result["models"], result["skipped"]
    kept = [p for p, s in result["files"].items() if s == "kept"]
    lines = [f"{len(models)} model{'' if len(models) == 1 else 's'} written (each with the hip held ideally and the hip free)"
             + (f", {sum(1 for m in models if not m['reviewed'])} of them from data that are not reviewed" if models else "")]
    warns = sum(1 for m in models for f in m["flags"] if f["level"] == "warn")
    if warns:
        lines.append(f"{warns} warning{'' if warns == 1 else 's'} (see the flags in manifest.json)")
    if skipped:
        lines.append(f"{len(skipped)} position{'' if len(skipped) == 1 else 's'} without a model:")
        lines += [f"  {s['participant']} {s['position_number']:02d} {s['test_key']} {s['target_deg'] if s['target_deg'] is not None else ''}: {s['reason']}"
                  for s in skipped]
    if kept:
        lines.append("edited by hand, left alone (the new text is beside it as *.generated.any; --force replaces):")
        lines += [f"  {p}" for p in kept]
    return "\n".join(lines)


def list_tests():
    rows = [f"{k:<22} {v['status'][:44]:<44} {v['label']}" for k, v in REGISTRY.items()]
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", help="anybody_inputs.csv, from Review's Export menu or tools/anybody_export.py")
    ap.add_argument("--out", default="AnyBody_models", help="folder to write (default AnyBody_models, which is in .gitignore)")
    ap.add_argument("--participants", nargs="*")
    ap.add_argument("--tests", nargs="*")
    ap.add_argument("--positions", nargs="*", type=int, help="position numbers")
    ap.add_argument("--angles", choices=("auto", "nominal"), default="auto",
                    help="auto: the angle at the peak force where the export has it; nominal: the target of the position")
    ap.add_argument("--knee-offset-deg", type=float, default=0.0,
                    help="added to the knee angle: AMMR's knee zero is not the dot-to-dot zero (1.5 deg in the group's test). Default 0.")
    ap.add_argument("--probe-load", type=float, default=DEFAULT_PROBE_LOAD_N, help="the load MaxLoading starts from, N (default 800)")
    ap.add_argument("--segment-lengths", action="store_true",
                    help="use the tape thigh and shank length (they change the muscle strength in the AMMR); default: from the height")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="[P01:3:]KEY=VALUE",
                    help="override a number: " + ", ".join(sorted(OVERRIDE_KEYS)))
    ap.add_argument("--ammr-libdef", help="path of the libdef.any of your AMMR, written into the model's libdef.any")
    ap.add_argument("--force", action="store_true", help="overwrite files that were edited by hand")
    ap.add_argument("--regression-case", action="store_true", help="write the group's own SeatedKneeExtMVIC model's numbers (to prove the port)")
    ap.add_argument("--list-tests", action="store_true")
    a = ap.parse_args(argv)
    if a.list_tests:
        print(list_tests())
        return 0
    if a.regression_case:
        row, overrides, offset = regression_row()
        result = build([row], a.out, overrides=overrides, ammr_libdef=a.ammr_libdef, force=a.force, inputs_name="regression case", knee_offset=offset,
                       probe_load=a.probe_load)
        print(summary_text(result))
        print(f"written to {Path(a.out).resolve()}")
        return 0
    if not a.inputs:
        ap.error("--inputs is required")
    inputs = Path(a.inputs)
    if not inputs.is_file():
        raise SystemExit(f"{inputs} not found")
    rows = read_rows(inputs)
    result = build(rows, a.out, a.participants, a.tests, a.positions, a.angles, parse_overrides(a.sets), a.ammr_libdef, a.force, inputs.name,
                   hashlib.sha256(inputs.read_bytes()).hexdigest(), a.knee_offset_deg, a.probe_load, a.segment_lengths)
    print(summary_text(result))
    problems = [f"{m['id']}: {p}" for m in result["models"] for p in lint_anyscript((Path(a.out) / m["folder"] / "Position.any").read_text(encoding="utf-8"))]
    if problems:
        print("problems in the written files:\n  " + "\n  ".join(problems))
    print(f"written to {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
