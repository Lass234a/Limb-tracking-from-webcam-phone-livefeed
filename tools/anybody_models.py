#!/usr/bin/env python3
"""Write AnyBody models of the limbtrack MVIC positions: one shared model per test group, and for every participant and position two small
files with the numbers (Participant.any, Position.any). Reads `anybody_inputs.csv` (the AnyBody export of Review, limbtrack/anybody.py); needs
nothing but the Python standard library; changes nothing under the recordings.

usage (from the project folder):
    python tools/anybody_models.py --inputs anybody_inputs.csv --out AnyBody_models
        [--participants P01 ...] [--tests knee_extension ...] [--positions 3 4]
        [--angles auto|nominal] [--set [P01:3:]KEY=VALUE ...] [--ammr-libdef PATH] [--force] [--list-tests]

What it writes under --out:
    <Model>/                       the shared AnyScript model (tools/anybody_templates/<Model>), the same for every position
    <P>/Participant.any            height, mass, fat, thigh and shank length of the person
    <P>/pos<NN>_<test>_t<target>/  Main.any (open this in AnyBody) and Position.any (joint angles, force, direction of the force, strap place)
    manifest.json, manifest.csv    every model with its inputs, where each number came from, what looks wrong, and the values AnyBody
                                   must reproduce on the first run (tools/anybody_models_run.py compares them)

To adjust a position, edit its Position.any, or run again with --set (for example --set P01:3:force_n=450 --set knee_deg=72). A file you
edited by hand is never overwritten: the new text is written beside it as <name>.generated.any unless you pass --force.

Nothing here is measured. Every number is read from the export; where the export is blank a stated default is used and the manifest says so.
The models have NOT been run in AnyBody yet: see .claude/skills/anybody-position-models/references/first-run-checklist.md.
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
FORMAT = 1
AMMR_TARGET = "AMMR 3.2.x on AnyBody 8.2"
HASH_TAG = "// generated-hash: "
NOMINAL_TOL_DEG = 10.0                 # the SOP tolerance of the joint angle (step 9.6); a measured angle further off is reported
DEFAULT_PULL_ANGLE_DEG = 90.0          # SOP: the anchor is moved so that the pull is about square to the shank
DEFAULT_STRAP_DIST_M = 0.05            # SOP: the strap sits just above the malleoli
THIGH_FIXATION_DIST_M = 0.10           # SOP: fixation strap over the distal femoral shaft; the distance is not recorded, this is a stand-in
LEVER_ARM_TOLERANCE = 0.20             # data check only: the app's lever arm against strap-to-knee distance times sin(pull angle)

# --------------------------------------------------------------------------------------------------------------------- the tests
# Axes: x forward (the way the participant faces), y up, z to the participant's right. effort_axis_z = the sign of the rotation about z that
# the participant's effort tends to give the segment the strap acts on: +1 knee extension (shank swings forward and up), -1 knee flexion.
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
    "knee_flexion": {"label": "Knee flexion, hip 90", "status": "built (the same model, the effort turned round; not looked at yet)",
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
    "force_n": ("Force_N", 1.0), "pull_angle_deg": ("PullAngle_deg", 1.0), "strap_dist_m": ("StrapDist_m", 1.0),
    "knee_deg": ("KneeFlexion_deg", 1.0), "hip_deg": ("HipFlexion_deg", 1.0), "ankle_deg": ("AnklePlantarFlexion_deg", -1.0),
    "hip_abduction_deg": ("HipAbduction_deg", 1.0), "thigh_fixation_dist_m": ("ThighFixationDist_m", 1.0),
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


# --------------------------------------------------------------------------------------------------------------------- geometry (mirror of Loads.any)
def shank_dir_global(hip_flex_deg, knee_flex_deg):
    """Unit vector knee -> ankle in the sagittal plane for an upright pelvis: the thigh is horizontal forward at hip flexion 90 and the
    shank hangs down at knee flexion 90. Axes: x forward, y up. The model's own neutral angles differ by a few degrees (checked on the first run)."""
    phi = math.radians(hip_flex_deg - 90.0 - knee_flex_deg)
    return (math.cos(phi), math.sin(phi), 0.0)


def tangent_global(direction, effort_axis_z):
    """The sideways part of the pull: cross((0, 0, -effort_axis_z), direction), as in Loads.any."""
    c = -float(effort_axis_z)
    return (-c * direction[1], c * direction[0], 0.0)


def pull_dir_global(shank, pull_angle_deg, effort_axis_z):
    a = math.radians(pull_angle_deg)
    t = tangent_global(shank, effort_axis_z)
    return tuple(math.cos(a) * s + math.sin(a) * u for s, u in zip(shank, t))


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
    keys = sorted(k for k in row if k not in ("notes",))
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


def derive(row, overrides=(), angles="auto"):
    """Everything the model files and the manifest need for ONE export row, or (None, reason) when no model can be made."""
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

    pos, src = {}, {}
    measured_info = {}
    for joint, drive in spec["drives"].items():
        value, source, measured = pick_angle(row, joint, drive, spec, angles, flag)
        if value is None:
            return None, f"no {joint} angle (no measured angle and no target) in the export"
        pos[drive["var"]], src[drive["var"]] = drive["sign"] * value + 0.0, source      # (+ 0.0 turns -0.0 into 0.0)
        if measured is not None:
            measured_info[f"{joint}_measured_deg"] = measured
    pos["HipAbduction_deg"], src["HipAbduction_deg"] = 0.0, "assumed: the thigh points straight forward on the bench"
    pos["HipExternalRotation_deg"], src["HipExternalRotation_deg"] = 0.0, "assumed neutral"
    pos["SubTalarEversion_deg"], src["SubTalarEversion_deg"] = 0.0, "assumed neutral"

    pos["Force_N"], src["Force_N"] = force, ("total force from the load cell zero, sliding-window peak of "
                                              f"{fmt(num(row.get('force_window_ms')) or 500)} ms (force_n)")
    rest = num(row.get("force_rest_n"))
    if rest is not None and force > 0 and rest > 0.1 * force:
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

    landmark, dist = row.get("strap_landmark"), num(row.get("strap_from_landmark_m"))
    pos["StrapFromKnee"] = 0
    if dist is not None and landmark == "lateral_malleolus":
        pos["StrapDist_m"], src["StrapDist_m"] = dist, "typed in the set-up facts: distance above the lateral malleolus"
    elif dist is not None and landmark == "lateral_epicondyle":
        pos["StrapDist_m"], pos["StrapFromKnee"] = dist, 1
        src["StrapDist_m"] = "typed in the set-up facts: distance below the lateral epicondyle (the knee)"
    else:
        pos["StrapDist_m"] = DEFAULT_STRAP_DIST_M
        src["StrapDist_m"] = "default: just above the malleoli (SOP); the strap place was not typed or is from another landmark"
        flag("warn", "strap_default", f"strap place not usable ({landmark or 'not typed'}): {DEFAULT_STRAP_DIST_M} m above the ankle is used")
    pos["ThighFixationDist_m"], src["ThighFixationDist_m"] = THIGH_FIXATION_DIST_M, "stand-in: the distance is not recorded"
    pos["EffortAxisZ"], src["EffortAxisZ"] = spec["effort_axis_z"], f"fixed for {key}"

    applied = []
    for (o_who, o_pos, o_key, o_val) in overrides:
        if (o_who is None or o_who == who) and (o_pos is None or o_pos == n):
            var, factor = OVERRIDE_KEYS[o_key]
            if var in ("KneeFlexion_deg", "HipFlexion_deg", "AnklePlantarFlexion_deg") and var not in pos:
                continue
            pos[var], src[var] = factor * o_val, f"override --set {o_key}={o_val:g}"
            applied.append(f"{o_key}={o_val:g}")
            flag("info", "override", f"{o_key} set to {o_val:g} by --set")

    mm = {"height_m": num(row.get("height_m")), "mass_kg": num(row.get("mass_kg")), "fat_pct": num(row.get("fat_pct")),
          "thigh_m": num(row.get("thigh_m")), "shank_m": num(row.get("shank_m"))}
    missing = [k for k, v in mm.items() if v is None]
    if missing:
        flag("warn", "measures_missing", "participant measures not typed: " + ", ".join(missing) + " (the AMMR default is used for each)")

    shank = shank_dir_global(pos["HipFlexion_deg"], pos["KneeFlexion_deg"])
    pdir = pull_dir_global(shank, pos["PullAngle_deg"], spec["effort_axis_z"])
    lever_app, moment_app = num(row.get("lever_arm_app_m")), None
    expected = {"force_n": force, "knee_flexion_deg": pos["KneeFlexion_deg"], "ankle_plantar_flexion_deg": pos["AnklePlantarFlexion_deg"],
                "pull_angle_deg": pos["PullAngle_deg"], "shank_dir_global": [round(x, 5) for x in shank],
                "pull_dir_global": [round(x, 5) for x in pdir], "lever_arm_app_m": lever_app}
    if lever_app is not None:
        moment_app = force * lever_app
        expected["ext_moment_app_nm"] = moment_app
        if mm["shank_m"] is not None:
            strap_to_knee = (mm["shank_m"] - pos["StrapDist_m"]) if not pos["StrapFromKnee"] else pos["StrapDist_m"]
            geom = strap_to_knee * math.sin(math.radians(pos["PullAngle_deg"]))
            expected["lever_arm_geometry_m"] = geom
            if geom > 0 and abs(lever_app - geom) / geom > LEVER_ARM_TOLERANCE:
                flag("warn", "lever_arm_mismatch", f"the app's lever arm {lever_app:.3f} m is not near strap-to-knee x sin(pull angle) = {geom:.3f} m "
                                                   "(shank length, strap place or pull angle may be wrong)")
    else:
        flag("info", "lever_arm_missing", "no lever arm in the export: the first-run moment check has nothing to compare with")

    return {"id": f"{who}_pos{n:02d}_{key}_t{tag(target) if target is not None else 'x'}", "participant": who, "position_number": n,
            "test_key": key, "target_deg": target, "trial_id": row.get("trial_id"), "verdict": verdict, "reviewed": reviewed,
            "data_source": row.get("data_source") or "", "data_hash": row.get("data_hash") or "", "row_hash": row_hash(row),
            "tested_leg": leg, "model": spec["model"], "scaling_group": spec["scaling_group"], "participant_values": mm,
            "position": pos, "sources": src, "measured": measured_info, "overrides": applied, "flags": flags, "expected": expected}, ""


# --------------------------------------------------------------------------------------------------------------------- the files
def render_position(d):
    pos, src = d["position"], d["sources"]
    right = d["tested_leg"] == "right"
    head = [
        "// Position.any: written by tools/anybody_models.py. Edit the numbers below to adjust this position, then reload the model in AnyBody.",
        f"// {d['participant']}   position {d['position_number']}   {d['test_key']}   target {d['target_deg']:g} deg   trial {d['trial_id']}"
        if d["target_deg"] is not None else f"// {d['participant']}   position {d['position_number']}   {d['test_key']}   trial {d['trial_id']}",
        f"// data: {d['data_source'] or 'unknown'}, verdict: {d['verdict']}" + ("" if d["reviewed"] else "   ** NOT REVIEWED **"),
        f"// data_hash: {d['data_hash'] or '-'}   row_hash: {d['row_hash']}",
        "// Units: degrees, newtons, metres. Angles are AnyBody's: knee and hip flexion +, ankle PLANTARflexion + (limbtrack reads dorsiflexion +).",
        "// Axes: x forward, y up, z to the participant's right. PullAngle_deg is between the pull and the shank, read from the knee towards the ankle.",
        "",
        "// Which leg is tested; only that leg is modelled",
        f"#define TESTED_LEG_IS_RIGHT {1 if right else 0}",
        f"#define BM_LEG_{'LEFT' if right else 'RIGHT'} OFF",
        "// The hands hold the bench; the arms are not modelled",
        "#define BM_ARM_RIGHT OFF",
        "#define BM_ARM_LEFT OFF",
        "// 1 = read back the angles and the force direction (Model/Checks.any); 0 = leave out",
        "#define WITH_CHECKS 1",
        "// 1 = also find the load at which the highest muscle activity reaches 1 (AMMR EvaluateMaxLoad); 0 = leave out",
        "#define WITH_MAXLOAD 1",
        "",
        "AnyFolder Position =",
        "{",
    ]
    groups = [
        ("Joint angles", ["HipFlexion_deg", "HipAbduction_deg", "HipExternalRotation_deg", "KneeFlexion_deg", "AnklePlantarFlexion_deg",
                          "SubTalarEversion_deg"]),
        ("The strap (force of the load cell at the peak, its direction and place)", ["Force_N", "PullAngle_deg", "StrapDist_m", "StrapFromKnee"]),
        ("Fixed for this test", ["ThighFixationDist_m", "EffortAxisZ"]),
    ]
    body = []
    for title, names in groups:
        body.append(f"  // {title}")
        for name in names:
            if name not in pos:
                continue
            value = f"DesignVar({fmt(pos[name])})" if name == "Force_N" else fmt(pos[name])
            decl = f"  AnyVar {name} = {value};"
            body.append(f"{decl:<46}// {comment_safe(src.get(name, ''))}" if src.get(name) else decl)
        body.append("")
    return "\n".join(head + body + ["};"])


def render_participant(who, mm, leg_note=""):
    lines = [f"// Participant.any: written by tools/anybody_models.py. {who}. Typed by the operator in limbtrack (tape and scale), SI units.",
             "// A value left out was not typed: AMMR's own default is used for it (BodyHeight 1.75 m, BodyMass 75 kg, the fat percent from the BMI).",
             "// The thigh and shank length are the tape measures (trochanter to epicondyle, fibular head to malleolus); they are used for both legs.",
             "HumanModel.Anthropometrics =", "{"]
    names = (("BodyHeight", "height_m"), ("BodyMass", "mass_kg"), ("FatPercent", "fat_pct"))
    for var, key in names:
        lines.append(f"  {var} = {fmt(mm[key])};" if mm.get(key) is not None else f"  // {var}: not typed, AMMR default")
    seg = [f"ThighLength = {fmt(mm['thigh_m'])};" if mm.get("thigh_m") is not None else "",
           f"ShankLength = {fmt(mm['shank_m'])};" if mm.get("shank_m") is not None else ""]
    seg = " ".join(s for s in seg if s)
    if seg:
        lines += ["  SegmentDimensions =", "  {", f"    Right = {{ {seg} }};", f"    Left  = {{ {seg} }};", "  };"]
    else:
        lines.append("  // thigh and shank length: not typed, AMMR default from the height")
    lines.append("};")
    return "\n".join(lines)


def render_main(d):
    return "\n".join([
        f"// Opens {d['id']}. The shared model is in ../../{d['model']}; the numbers are in ../Participant.any and Position.any (this folder).",
        '#path PARTICIPANT_DIR ".."',
        '#path POSITION_DIR "."',
        f'#include "../../{d["model"]}/{d["model"]}.main.any"',
    ])


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


# --------------------------------------------------------------------------------------------------------------------- the whole run
def build(rows, out, participants=None, tests=None, positions=None, angles="auto", overrides=(), ammr_libdef=None, force=False, inputs_name="",
          inputs_hash=""):
    """Write everything under `out`. Returns {'models': [...], 'skipped': [...], 'files': {path: status}}."""
    out = Path(out)
    result = {"models": [], "skipped": [], "files": {}}
    used, wrote_model, wrote_participant = set(), set(), {}
    for row in rows:
        who, key = row.get("participant") or "", row.get("test_key")
        n = int(num(row.get("position_number")) or 0)
        if participants and who not in participants or tests and key not in tests or positions and n not in positions:
            continue
        d, reason = derive(row, overrides, angles)
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
        wrote_participant[who] = render_participant(who, d["participant_values"])
        for name, text in (("Position.any", render_position(d)), ("Main.any", render_main(d))):
            result["files"][(folder / name).relative_to(out).as_posix()] = write_generated(folder / name, text, force)
        d["folder"] = folder.relative_to(out).as_posix()
        d["main_file"] = f"{d['folder']}/Main.any"
        result["models"].append(d)
    for who, text in wrote_participant.items():
        result["files"][f"{who or 'unknown'}/Participant.any"] = write_generated(out / (who or "unknown") / "Participant.any", text, force)
    manifest = {"format": FORMAT, "generated_by": "tools/anybody_models.py", "target": AMMR_TARGET, "inputs": inputs_name, "inputs_sha256": inputs_hash,
                "angles": angles, "models": result["models"], "skipped": result["skipped"],
                "meaning": "Nothing is measured here. Each number in a Position.any is read from the export, or a stated default. The models have "
                           "not been run in AnyBody until a first run has passed (references/first-run-checklist.md)."}
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
                        m["tested_leg"], fmt(p["Force_N"]), fmt(p["KneeFlexion_deg"]), fmt(p["HipFlexion_deg"]), fmt(p["PullAngle_deg"]),
                        fmt(p["StrapDist_m"]), "; ".join(f["code"] for f in m["flags"]), m["folder"]])


def summary_text(result):
    models, skipped = result["models"], result["skipped"]
    kept = [p for p, s in result["files"].items() if s == "kept"]
    lines = [f"{len(models)} model{'' if len(models) == 1 else 's'} written"
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
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="[P01:3:]KEY=VALUE",
                    help="override a number: " + ", ".join(sorted(OVERRIDE_KEYS)))
    ap.add_argument("--ammr-libdef", help="path of the libdef.any of your AMMR, written into the model's libdef.any")
    ap.add_argument("--force", action="store_true", help="overwrite files that were edited by hand")
    ap.add_argument("--list-tests", action="store_true")
    a = ap.parse_args(argv)
    if a.list_tests:
        print(list_tests())
        return 0
    if not a.inputs:
        ap.error("--inputs is required")
    inputs = Path(a.inputs)
    if not inputs.is_file():
        raise SystemExit(f"{inputs} not found")
    rows = read_rows(inputs)
    result = build(rows, a.out, a.participants, a.tests, a.positions, a.angles, parse_overrides(a.sets), a.ammr_libdef, a.force, inputs.name,
                   hashlib.sha256(inputs.read_bytes()).hexdigest())
    print(summary_text(result))
    problems = [f"{m['id']}: {p}" for m in result["models"] for p in lint_anyscript((Path(a.out) / m["folder"] / "Position.any").read_text(encoding="utf-8"))]
    if problems:
        print("problems in the written files:\n  " + "\n  ".join(problems))
    print(f"written to {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
