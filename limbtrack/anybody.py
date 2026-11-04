"""The AnyBody export (0.34.0, plan feature F1, data contract M1 of audit/IMPROVEMENT_AND_ANYBODY_PLAN_2026-11-02.md): one row per participant and
position in `anybody_inputs.csv`, and a `participant.json` per participant, in SI units, so that a batch script (AnyPyTools, plan M4) can write one
model input per row without anybody typing numbers again.

Nothing here measures anything new and no existing measured number changes: every value is read from a number the program already has (the export
row of the trial, `analysis.session_row`, plus a few values of the trial itself) or from what was typed (participant measures, set-up facts). A value
that is missing is BLANK, never worked out from something else, and the `notes` column says why for each blank that matters. Review never writes the
recorded files; the caller writes the two files into the review layer (`review_layer.atomic`) or, for the tool, into a folder of its own.

Decisions of the plan that are not taken yet were built on the plan's PROPOSAL, and the code says so where it matters:
  * D3  which trial per position: the COUNTED trial with the highest peak; the other counted trial is the check (its id and the peak difference
        in percent, the SOP 10.6 number). Counted = marked valid (`judged_only`, the default), has a real force rise (`analysis.contracted`) and is
        not marked compromised. A position whose trials are only not judged yet, or only compromised, gets no row; it is named in participant.json
        and in the message of the button. `judged_only=False` also takes trials not judged yet (never a compromised one): used to look at data that
        was never judged, such as the pilot; the `verdict` column then says "not judged" and the notes say so.
  * D4  force: the total force from the load cell zero (not above rest), the sliding-window peak of 0.29.0 (500 ms by default; its length is in
        `force_window_ms`), and the resting force before the push beside it (`force_rest_n`), so the model's gravity rule can be applied either way
        (SOP: subtract the rest XOR model the segment's weight, never both). The 20 ms peak is given too (`force_peak_20ms_n`) for comparison.
  * D2  angles: at the peak force, the pictures within 0.1 s of the highest smoothed force, the same pictures as the existing column
        `angle_at_peak_force_deg`; the plateau mean is not exported here (it is in the session summary).
  * D9  hip reference: the standing offsets of F4 are given beside the angles; nothing is subtracted.
Angles are in the app's sign convention (`angle_convention`), as the DOTS read them; a trial recorded with an angle reference also has the angle
from the reference in a second column. The offsets are "dots minus the joint angle called then".
"""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from . import analysis, check_weight, participant_measures, review_layer, stabilisation, study_settings
from . import verdict as verdicts

FORMAT = 1
SELECTION_RULE = ("per position, the counted trial (marked valid, a real force rise, not compromised) with the highest sliding-window peak; the other "
                  "counted trial with the highest peak is the check (plan decision D3, proposal)")
FORCE_RULE = ("force_n = total force from the load cell zero (not above rest), the highest mean over the sliding window (force_window_ms), in newtons; "
              "force_rest_n = the resting force before the push (median 3.0 to 0.5 s before it); the model must take the weight of the segment from "
              "one of them only (plan decision D4, proposal)")
ANGLE_RULE = ("at the peak force: the mean of the pictures within 0.1 s of the highest smoothed force (plan decision D2, proposal); the picture delay "
              "is taken as 0")
ANGLE_CONVENTION = ("app: two-arm goniometer angles from protocol.json, in degrees, read from the dots on the skin (not AnyBody joint coordinates); "
                    "hip + = flexion, - = extension; knee + = flexion, 0 = straight, hyperextension negative; ankle + = dorsiflexion, 0 = foot square "
                    "to the shank; frontal tests: hip abduction as the goniometer reads it. standing_*_offset_deg = what the dots read minus the joint "
                    "angle called at the standing capture (0 = anatomical zero): the joint angle is about the dots' angle minus the offset")
UNITS = "m;kg;N;deg;percent;s"
JOINTS = ("hip", "knee", "ankle")
PEAK_HALF_S = 0.1
SOURCES = ("original", "reviewed")

# The posture of each position, from the protocol draft 2026-10-26 (Figures 4-13). The protocol marks some of them TBD; the entries say which.
# The default is the test's; a target listed in BY_TARGET overrides it.
CONFIGURATION = {
    "knee_extension": "seated", "knee_flexion": "seated",
    "hip_flexion": "seated", "hip_extension_k0": "standing", "hip_extension_k90": "standing",
    "hip_abduction": "standing", "hip_adduction": "standing", "ankle_plantarflexion": "seated", "ankle_dorsiflexion": "seated",
    "belt_squat": "belt squat",
}
BY_TARGET = {                                    # (test key, target) -> posture
    ("hip_flexion", 0.0): "supine",              # Figure 7 (b): hip 0, supine; (a) hip 90 is seated
    ("hip_extension_k90", 90.0): "prone",        # Figure 10 (b): hip 90, trunk lying on the bench (the protocol's TBD: first written prone)
}
CONFIGURATION_NOTE = ("configuration is taken from the protocol draft 2026-10-26 by test and target (hip extension knee 90 at hip 0 is standing in "
                      "Figure 10 but the SOP text says prone: TBD; ankle tests are taken as seated; the frontal tests as standing, a trial recorded "
                      "side-lying is not told apart)")

COLUMNS = [
    "participant", "tested_leg", "height_m", "mass_kg", "fat_pct", "thigh_m", "shank_m",
    "position_number", "test_key", "configuration", "target_deg",
    "trial_id", "check_trial_id", "check_peak_diff_pct", "n_trials_at_position", "n_counted_trials", "ten_percent_check",
    "force_n", "force_window_ms", "force_rest_n", "force_peak_20ms_n",
    "pull_angle_deg", "strap_landmark", "strap_landmark_other", "strap_from_landmark_m", "lever_arm_app_m",
    "supports", "seat_back_angle_deg", "setup_prefilled",
    "main_angle", "main_angle_at_peak_deg",
    "hip_deg_at_peak", "hip_deg_at_peak_from_reference", "hip_standing_offset_deg",
    "knee_deg_at_peak", "knee_deg_at_peak_from_reference", "knee_standing_offset_deg",
    "ankle_deg_at_peak", "ankle_deg_at_peak_from_reference", "ankle_standing_offset_deg",
    "angle_peak_time_s", "angle_rest_to_hold_deg",
    *stabilisation.COLUMNS,
    "verdict", "selection",
    "data_source", "review_edits", "software_version", "export_software_version", "data_hash", "review_files_hash",
    "units", "angle_convention", "notes",
    *check_weight.COLUMNS,                         # 0.35.0 (F8), appended so that the columns of 0.34.0 keep their places: the latest check of the force
    *study_settings.COLUMNS,                       # 0.35.0 (F13): how many of the study's frozen settings differed when the trial was recorded
]


# ------------------------------------------------------------------------------------------------ small helpers
def _num(v):
    return analysis._num(v)


def _metres(cm):
    v = _num(cm)
    return "" if v is None else round(v / 100.0, 5)


def _blank(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else v


def configuration(test_key, target):
    """The posture of a position (see CONFIGURATION); '' for a test this module does not know."""
    t = _num(target)
    if t is not None and (test_key, float(t)) in BY_TARGET:
        return BY_TARGET[(test_key, float(t))]
    return CONFIGURATION.get(test_key, "")


def same_target(a, b):
    a, b = _num(a), _num(b)
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-6)


def is_counted(row):
    """A trial that can count at all: not marked compromised and a real force rise (the 10 % check's own rule, analysis.contracted)."""
    return row.get("verdict") != verdicts.COMPROMISED and analysis.contracted(row)


def eligible(row, judged_only=True):
    return is_counted(row) and (not judged_only or row.get("verdict") == verdicts.VALID)


def _rank(row):
    """Highest first: newtons before volts, then the sliding-window peak, then the smoothed peak (a trial without a window peak ranks after one
    with it)."""
    wp, pk = _num(row.get("force_window_peak")), _num(row.get("force_peak"))
    return (row.get("force_unit") == "N", wp if wp is not None else -math.inf, pk if pk is not None else -math.inf)


def positions_of(rows, plan=None):
    """[(position number or '', test key, target, [rows of it, in the order given])]: the planned positions first (in plan order, also those
    with no row), then positions found in the rows that are not in the plan."""
    out, used = [], set()
    for i, (key, tgt) in enumerate(plan or [], 1):
        mine = [r for r in rows if r.get("test_key") == key and same_target(r.get("target_deg"), tgt)]
        used.update(id(r) for r in mine)
        out.append((i, key, tgt, mine))
    extra = {}
    for r in rows:
        if id(r) not in used:
            extra.setdefault((r.get("test_key"), _num(r.get("target_deg"))), []).append(r)
    for (key, tgt), mine in extra.items():
        out.append((_num(mine[0].get("position_number")) or "", key, tgt, mine))
    return out


def select(rows, judged_only=True):
    """Pure D3: from the export rows of ONE participant and ONE position (oldest first) -> {selected, check, counted, valid, trials, reason,
    peak_diff_pct}. `selected`/`check` are rows or None; `reason` says in plain words why there is no selected trial ('' when there is)."""
    n = len(rows)
    counted = [r for r in rows if is_counted(r)]
    valid = [r for r in counted if r.get("verdict") == verdicts.VALID]
    out = {"selected": None, "check": None, "counted": len(counted), "valid": len(valid), "trials": n, "reason": "", "peak_diff_pct": None}
    if n == 0:
        out["reason"] = "never recorded"
        return out
    if all(r.get("verdict") == verdicts.COMPROMISED for r in rows):
        out["reason"] = "every trial is marked compromised"
        return out
    if not counted:
        out["reason"] = "no trial has a force rise (or none has force)"
        return out
    pool = [r for r in counted if eligible(r, judged_only)]
    if not pool:
        out["reason"] = (f"{len(counted)} trial{'' if len(counted) == 1 else 's'} with a force rise, none marked valid yet: judge them in Review "
                         "(Next not judged), then export again")
        return out
    ordered = sorted(enumerate(pool), key=lambda p: (_rank(p[1]), p[0]), reverse=True)        # (a tie goes to the later trial)
    out["selected"] = ordered[0][1]
    if len(ordered) > 1:
        out["check"] = ordered[1][1]
        a, b = _num(out["selected"].get("force_peak")), _num(out["check"].get("force_peak"))
        if a is not None and b is not None and out["selected"].get("force_unit") == out["check"].get("force_unit") and (a + b) != 0:
            out["peak_diff_pct"] = analysis._spread_pct([a, b])         # the SOP 10.6 number: the smoothed peaks, as the 10 % check compares them
    return out


# ------------------------------------------------------------------------------------------------ from the trial itself
def _near_mean(tr, series, t_centre, half_s=PEAK_HALF_S):
    """The mean of `series` (one value per picture) in the pictures within `half_s` of `t_centre`, lost pictures left out; NaN when none is
    there (not the nearest picture: a strap dot lost at the peak must not be filled from another moment)."""
    import numpy as np
    near = abs(tr.t - t_centre) <= half_s
    vals = series[near]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if vals.size else math.nan


def trial_extras(tr, row):
    """What the export row cannot give, read from the open trial `tr` (a review.Trial; numbers only): the pull angle at the peak force, the hip,
    knee and ankle at the peak force (as the dots read them and from the angle reference when the trial has one), the time of that moment."""
    out = {}
    t_peak = _num(row.get("force_t_peak_s"))
    if t_peak is None:
        return out
    out["angle_peak_time_s"] = t_peak
    if tr.pull is not None:
        out["pull_angle_deg"] = _blank(_near_mean(tr, tr.pull["pull_angle_deg"], t_peak))
    for a in tr.angle_defs:
        if a.name in JOINTS:
            v = tr.angle_near(a.name, t_peak)
            out[f"{a.name}_deg_at_peak"] = _blank(v)
            if tr.offset_of(a.name) and not math.isnan(v):
                out[f"{a.name}_deg_at_peak_from_reference"] = v - tr.offset_of(a.name)
    return out


def data_hashes(meta_path):
    """({file key: sha256}, one combined hash) of the trial's meta.json, data.csv and force.csv. The combined hash is the SHA-256 of the lines
    'key:hash' in that order ('-' for a file the trial does not have)."""
    fp = review_layer.fingerprint(meta_path)
    each = {k: fp[k]["sha256"] for k in review_layer.HASHED if k in fp and "sha256" in fp[k]}
    text = "".join(f"{k}:{each.get(k, '-')}\n" for k in review_layer.HASHED)
    return each, hashlib.sha256(text.encode("ascii")).hexdigest()


def _read_meta(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def participant_info(metas):
    """The participant-level facts from the meta.json dicts of all the participant's trials, oldest first: the tested leg (the newest), the typed
    measures (per measure the newest typed value; `vary` lists those typed differently in different trials), the newest standing reference."""
    legs = [m.get("tested_leg") for m in metas if m.get("tested_leg")]
    values, vary = {k: None for k in participant_measures.KEYS}, []
    seen = {k: set() for k in participant_measures.KEYS}
    from_trial = {}
    for i, m in enumerate(metas):
        block = m.get("participant_measures")
        if not isinstance(block, dict):
            continue
        for k in participant_measures.KEYS:
            v = _num(block.get(k))
            if v is not None:
                values[k] = v
                seen[k].add(v)
                from_trial[k] = i
    vary = [k for k, s in seen.items() if len(s) > 1]
    standing = next((m.get("standing_reference") for m in reversed(metas) if isinstance(m.get("standing_reference"), dict)), None)
    return {"tested_leg": legs[-1] if legs else "", "legs_seen": sorted(set(legs)), "measures": values, "measures_vary": vary,
            "standing_reference": standing}


def _setup(meta):
    s = meta.get("setup")
    return s if isinstance(s, dict) else {}


def blank_row():
    return {c: "" for c in COLUMNS}


def _ten_percent(rows):
    st = analysis.position_check(rows)
    return st["state"]


# ------------------------------------------------------------------------------------------------ the whole export
def build(rows, meta_path_of, open_trial, plan=None, judged_only=True, source="original", export_version="", now=None):
    """The export of every participant found in `rows` (the export rows of their trials, oldest first, from `analysis.session_row` through
    `review.trial_row` with the same `source`).

    meta_path_of(row) -> the trial's meta.json path (for the participant facts, the set-up and the hashes);
    open_trial(meta_path) -> a review.Trial opened for numbers only with the same `source` (closed here); plan: [(test key, target)] of the
    session plan, so that positions are numbered and a position with no trial is named; judged_only: see the module text.

    Returns {'rows': [dict per position], 'participants': {code: participant.json content}, 'skipped': [{participant, position_number,
    test_key, target_deg, reason}], 'columns': COLUMNS}. Reads files only."""
    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    result = {"rows": [], "participants": {}, "skipped": [], "columns": list(COLUMNS)}
    by_person = {}
    for r in rows:
        by_person.setdefault(r.get("participant") or "", []).append(r)
    for who, mine in by_person.items():
        metas = [_read_meta(meta_path_of(r)) for r in mine]
        info = participant_info(metas)
        meta_of = {id(r): m for r, m in zip(mine, metas)}
        positions, trials_out = [], []
        for number, key, tgt, prow in positions_of(mine, plan):
            sel = select(prow, judged_only)
            pos = {"position_number": number, "test_key": key, "target_deg": _num(tgt), "configuration": configuration(key, tgt),
                   "trials": sel["trials"], "counted_trials": sel["counted"], "valid_trials": sel["valid"]}
            if sel["selected"] is None:
                pos.update(status="no row", reason=sel["reason"])
                result["skipped"].append({"participant": who, "position_number": number, "test_key": key, "target_deg": _num(tgt),
                                          "reason": sel["reason"]})
                positions.append(pos)
                continue
            s, c = sel["selected"], sel["check"]
            meta = meta_of[id(s)]
            mp = Path(meta_path_of(s))
            tr = open_trial(mp)
            try:
                extras = trial_extras(tr, s)
                reviewed = getattr(tr, "source", "original") == "reviewed"
            finally:
                tr.close()
            each, combined = data_hashes(mp)
            row = blank_row()
            setup = _setup(meta)
            notes = []
            unit = s.get("force_unit")
            row.update(participant=who, tested_leg=meta.get("tested_leg") or "",         # (this trial's own; the participant's newest leg is in participant.json)
                       height_m=_metres(info["measures"].get("height_cm")), mass_kg=_blank(info["measures"].get("mass_kg")),
                       fat_pct=_blank(info["measures"].get("fat_pct")), thigh_m=_metres(info["measures"].get("thigh_length_cm")),
                       shank_m=_metres(info["measures"].get("shank_length_cm")),
                       position_number=number, test_key=key, configuration=configuration(key, tgt), target_deg=_blank(_num(tgt)),
                       trial_id=s.get("trial"), check_trial_id=(c or {}).get("trial") or "", check_peak_diff_pct=_blank(sel["peak_diff_pct"]),
                       n_trials_at_position=sel["trials"], n_counted_trials=sel["counted"], ten_percent_check=_ten_percent(prow))
            if row["tested_leg"] == "":
                notes.append("tested_leg: not recorded in this trial (participant.json lists the legs seen in the participant's other trials)")
            if unit == "N":
                row.update(force_n=_blank(_num(s.get("force_window_peak"))), force_window_ms=_blank(_num(s.get("force_window_peak_ms"))),
                           force_rest_n=_blank(_num(s.get("force_rest"))), force_peak_20ms_n=_blank(_num(s.get("force_peak_20ms"))))
                if _num(s.get("force_window_peak")) is None:
                    notes.append("force_n: no sliding-window peak (the window was longer than the push, or the force had gaps)")
                if _num(s.get("force_rest")) is None:
                    notes.append("force_rest_n: no clear rest before the push")
            else:
                notes.append("force in volts (no calibration or zero applied): no newtons" if unit == "V" else "no force recorded")
            row.update(pull_angle_deg=extras.get("pull_angle_deg", ""), lever_arm_app_m=_metres(s.get("arm_at_peak_force_cm")))
            if row["pull_angle_deg"] == "":
                notes.append("pull_angle_deg: no strap and anchor dots, or the strap dot was lost at the peak")
            if row["lever_arm_app_m"] == "":
                notes.append("lever_arm_app_m: no ruler scale, no strap and anchor dots, or no calibrated force")
            elif (_num(s.get("torque_stand_in_pictures")) or 0) > 0:
                # analysis.torque_series uses the segment's distal dot where the strap dot is missing; the count is of the pictures in the window
                notes.append(f"lever_arm_app_m: the strap dot was not seen in {int(_num(s.get('torque_stand_in_pictures')))} pictures of the window, "
                             "the segment's distal dot was used there instead, so the lever arm may be an estimate")
            if setup.get("strap_landmark") or setup.get("strap_from_landmark_cm") is not None:
                row.update(strap_landmark=setup.get("strap_landmark") or "", strap_landmark_other=setup.get("strap_landmark_other") or "",
                           strap_from_landmark_m=_metres(setup.get("strap_from_landmark_cm")))
            else:
                notes.append("strap place not typed (set-up facts)")
            if setup:
                row.update(supports=";".join(setup.get("supports") or []), seat_back_angle_deg=_blank(_num(setup.get("seat_back_angle_deg"))),
                           setup_prefilled=bool(setup.get("prefilled")))
            row.update(main_angle=s.get("main_angle") or "", main_angle_at_peak_deg=_blank(_num(s.get("angle_at_peak_force_deg"))),
                       angle_peak_time_s=extras.get("angle_peak_time_s", ""), angle_rest_to_hold_deg=_blank(_num(s.get("angle_rest_to_hold_deg"))))
            for j in JOINTS:
                row[f"{j}_deg_at_peak"] = extras.get(f"{j}_deg_at_peak", "")
                row[f"{j}_deg_at_peak_from_reference"] = extras.get(f"{j}_deg_at_peak_from_reference", "")
                row[f"{j}_standing_offset_deg"] = _blank(_num(s.get(f"standing_{j}_offset_deg")))
            missing = [c for c in ("height_m", "mass_kg", "fat_pct", "thigh_m", "shank_m") if row[c] == ""]
            if missing:
                notes.append("participant measures not typed: " + ", ".join(missing))
            if all(row[f"{j}_standing_offset_deg"] == "" for j in JOINTS):
                notes.append("no standing reference in this trial: no standing offsets")
            if all(row[f"{j}_deg_at_peak"] == "" for j in JOINTS):
                notes.append("hip/knee/ankle: this test has none of them as a side-view angle (see main_angle)")
            for col in stabilisation.COLUMNS:
                row[col] = _blank(s.get(col))
            # The dot shifts are cm in the session summary (px without a ruler). This file is SI (units column): metres, or blank with no ruler.
            if row["stab_unit"] == "cm":
                for col in ("stab_dot1_shift", "stab_dot2_shift"):
                    row[col] = _metres(row[col])
                row["stab_unit"] = "m"
            elif row["stab_unit"] == "px":
                for col in ("stab_dot1_shift", "stab_dot2_shift"):
                    row[col] = ""
                notes.append("stab_dot1_shift/stab_dot2_shift: no ruler scale, the shifts are only in pixels and are not given (stab_unit px)")
            row.update(verdict=s.get("verdict") or "not judged", selection="highest counted peak" + ("" if c else " (the only counted trial)"),
                       data_source="reviewed" if reviewed else "original", review_edits=s.get("review_edits") or "",
                       software_version=s.get("software_version") or meta.get("software_version") or "", export_software_version=export_version,
                       data_hash=combined, review_files_hash=review_layer.trial_fingerprint(mp) if reviewed else "",
                       units=UNITS, angle_convention=ANGLE_CONVENTION)
            if row["verdict"] != verdicts.VALID:
                notes.append("trial not marked valid (exported because judged_only was off)")
            if sel["counted"] > 1 and row["check_trial_id"] == "":
                notes.append("the other counted trial is not marked valid, so it is not the check")
            for col in check_weight.COLUMNS:
                row[col] = _blank(s.get(col))
            if row["check_weight_error_pct"] == "":
                notes.append("check_weight_error_pct: no check weight in this trial's file (none made that run, or a trial of an earlier version)")
            row.update(study_settings.export_cells(meta))
            if row["study_settings_differences"] == "":
                notes.append("study_settings_differences: no frozen study settings in this trial's file (a trial of an earlier version, or protocol.json had none)")
            row["notes"] = "; ".join(notes)
            result["rows"].append(row)
            pos.update(status="row", trial_id=s.get("trial"), check_trial_id=row["check_trial_id"], check_peak_diff_pct=row["check_peak_diff_pct"])
            positions.append(pos)
            trials_out.append({"trial_id": s.get("trial"), "position_number": number, "sha256": each, "data_hash": combined,
                               "data_source": row["data_source"]})
        result["participants"][who] = participant_json(who, info, positions, trials_out, mine, judged_only, source, export_version, stamp)
    return result


def participant_json(who, info, positions, trials, rows, judged_only, source, export_version, stamp):
    """The participant.json of one participant (a dict, to be written as JSON)."""
    m = info["measures"]
    return {
        "format": FORMAT, "participant": who, "tested_leg": info["tested_leg"], "tested_legs_seen": info["legs_seen"],
        "measures_si": {"height_m": _metres(m.get("height_cm")) or None, "mass_kg": m.get("mass_kg"), "fat_pct": m.get("fat_pct"),
                        "thigh_m": _metres(m.get("thigh_length_cm")) or None, "shank_m": _metres(m.get("shank_length_cm")) or None,
                        "asis_width_m": _metres(m.get("asis_width_cm")) or None},
        "measures_typed_differently_in_different_trials": info["measures_vary"],
        "standing_reference": info["standing_reference"],
        "positions": positions, "trials": trials,
        "software_versions_recorded": sorted({str(r.get("software_version")) for r in rows if r.get("software_version")}),
        "export": {"made_at": stamp, "export_software_version": export_version, "data_source_requested": source, "judged_only": bool(judged_only),
                   "csv": "anybody_inputs.csv", "units": UNITS, "selection_rule": SELECTION_RULE, "force_rule": FORCE_RULE,
                   "angle_rule": ANGLE_RULE, "angle_convention": ANGLE_CONVENTION, "configuration_note": CONFIGURATION_NOTE,
                   "meaning": "Read from the recorded trials; a blank value was not available, never worked out from something else. Nothing recorded "
                              "was changed."},
    }


# ------------------------------------------------------------------------------------------------ writing (the caller chooses where)
def write_csv(fh, rows):
    """Write the rows as anybody_inputs.csv into an open text file (opened with newline=''). Floats are rounded to 5 decimals; blank stays blank."""
    import csv
    w = csv.writer(fh)
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow(["" if isinstance(v, float) and math.isnan(v) else round(v, 5) if isinstance(v, float) else v for v in (r[c] for c in COLUMNS)])


def json_text(data):
    return json.dumps(data, indent=2, default=str)


def summary_text(result):
    """The sentence the Export button shows: how many rows, how many positions had none and why."""
    n = len(result["rows"])
    text = f"{n} position{'' if n == 1 else 's'} exported"
    if result["skipped"]:
        text += f"; {len(result['skipped'])} without a row:\n" + "\n".join(
            f"  {s['participant']} {int(s['position_number']):02d} {s['test_key']} {s['target_deg'] if s['target_deg'] is not None else ''}: {s['reason']}"
            if isinstance(s["position_number"], (int, float)) and s["position_number"] != "" else
            f"  {s['participant']} {s['test_key']} {s['target_deg'] if s['target_deg'] is not None else ''}: {s['reason']}" for s in result["skipped"])
    return text
