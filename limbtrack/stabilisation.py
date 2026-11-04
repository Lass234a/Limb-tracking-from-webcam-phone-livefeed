"""How much the body moved from rest to the peak force (0.34.0, plan feature F5): did the pelvis and the neighbouring segment stay still?

The AnyBody model assumes the pelvis and the thigh are held (driver reactions on: hip held 2036 N, hip free 555 N, README of the seated knee
model), and SOP 10.3 asks the test leader to "watch for compensation (trunk lean, pelvic tilt, lifting off the support)" without a number.
This module puts numbers on it, from the dot tracks the trial already has. Nothing here changes a recorded or an existing measured number.

What is measured, per trial
  * the REST LEVEL of every dot: the median of its positions over the same stretch the resting force uses (Trial.force_rest: 3.0 to 0.5 s
    before the push began);
  * the PEAK POSITION of every dot: the mean of its positions in the pictures within 0.1 s of the peak force plus the picture delay, the same
    pictures that give the angle at the peak force (Trial.angle_near); the nearest picture when none of those has the dot;
  * the SHIFT of two dots of the pelvis (the iliac crest dot and the trochanter dot; in a frontal test the two ASIS dots), rest to peak:
    in centimetres at the ruler scale (right only at the ruler's distance from the camera, in the picture's plane) or, without a ruler, in
    pixels;
  * the ROTATION of the pelvis line (crest dot to trochanter dot; frontal: ASIS to ASIS) and of the neighbouring segment that must stay still
    (below), in degrees, rest to peak. The sign is "clockwise in the picture" for a participant who faces right; the picture is mirrored for
    one who faces left, so the same movement has the same sign (frontal tests are not mirrored). A flag uses the size only.
  Lost dots are left out (never filled in). When a dot or a stretch is missing the number is blank, never a made-up one.

Which segment must stay still (decided here, per test; the pelvis is always watched)
  * knee extension, knee flexion: the THIGH (trochanter to lateral epicondyle). The knee moves the shank only; a thigh that turns is the hip
    being flexed or extended by the push, which also changes the hip angle the model reads for the two-joint muscles.
  * hip extension (knee straight / 90), hip flexion: the PELVIS only. The thigh is the segment tested, and the pelvis is the other segment of the
    hip: the pelvis line is the answer. There is no second segment (the shank goes along with the thigh and its own angle is judged as the
    neighbour angle).
  * ankle plantarflexion, ankle dorsiflexion: the SHANK (fibular head to lateral malleolus). The foot is the segment tested.
  * hip abduction, hip adduction (frontal): the PELVIS line between the two ASIS only (the thigh is tested).
  * belt squat: nothing is judged. A closed chain: the pelvis, the thigh and the shank move by design. The numbers are still computed and
    exported.
  A trial of a test this module does not know gets the rule of its main joint (knee, hip, ankle) or, failing that, the pelvis only.

The limits are NOT from the SOP (it names none) and not from the protocol: they are a PROPOSAL for the group to confirm, in two named
constants below. Chosen so that a flag comes before the angle itself leaves its tolerance: the main angle's tolerance is 5 degrees, so a
pelvis or thigh that turned 3 degrees already moves the hip angle by about 3; one centimetre of the crest or trochanter dot is about 1 % of
a thigh and 20 pixel-noise widths of a dot. Without a ruler scale the shifts are in pixels and are not judged.

Tried on the 128 pilot trials (P01, P02; read only, 2026-11-02): 103 had a clear rest and could be measured; the largest pelvis-dot shift had a
median of 3.3 cm (10th to 90th percentile 1.0 to 5.3) and the pelvis line turned a median 4.3 degrees (0.4 to 8.6), so these two limits flag 87 of
the 100 trials that are judged. They are too tight to tell trials apart on that data; the group should choose them from that spread.
"""
import math
from collections import namedtuple

import numpy as np

PEAK_HALF_S = 0.1                      # the pictures within this of the peak force (plus the picture delay): as Trial.angle_near
MIN_REST_PICTURES = 3                  # fewer pictures of a dot in the rest stretch: no rest level for it
# PROPOSAL for the group to confirm (the SOP names no limit): a shift or a turn above these is flagged. Never changes a number, only a flag.
PROPOSED_DISPLACEMENT_LIMIT_CM = 1.0   # a pelvis dot (crest, trochanter; frontal: ASIS) moved more than this from its rest level
PROPOSED_ROTATION_LIMIT_DEG = 3.0      # the pelvis line or the segment that must stay still turned more than this
BASIS = (f"rest level = median of the dot positions {3.0:g} to {0.5:g} s before the push (the resting force's stretch); peak = the pictures within "
         f"{PEAK_HALF_S:g} s of the peak force plus the picture delay (the pictures of the angle at the peak force)")

Rule = namedtuple("Rule", "pelvis_dots shown_dots still_name still_dots judged why")
_SAGITTAL = ("iliac_crest", "trochanter")
_FRONTAL = ("asis_opposite", "asis")
_KNEE = Rule(_SAGITTAL, _SAGITTAL, "thigh", ("trochanter", "epicondyle"), True, "knee test: pelvis and thigh must stay still")
_HIP = Rule(_SAGITTAL, _SAGITTAL, "", None, True, "hip test: the pelvis must stay still (the thigh is the segment tested)")
_ANKLE = Rule(_SAGITTAL, _SAGITTAL, "shank", ("fibular_head", "malleolus"), True, "ankle test: pelvis and shank must stay still")
_FRONT = Rule(_FRONTAL, ("asis", "asis_opposite"), "", None, True, "frontal hip test: the pelvis must stay still (the thigh is tested)")
_CHAIN = Rule(_SAGITTAL, _SAGITTAL, "", None, False, "belt squat: closed chain, the pelvis moves by design; computed, not judged")
RULES = {"knee_extension": _KNEE, "knee_flexion": _KNEE, "hip_extension_k0": _HIP, "hip_extension_k90": _HIP, "hip_flexion": _HIP,
         "ankle_plantarflexion": _ANKLE, "ankle_dorsiflexion": _ANKLE, "hip_abduction": _FRONT, "hip_adduction": _FRONT, "belt_squat": _CHAIN}
BY_MAIN_ANGLE = {"knee": _KNEE, "hip": _HIP, "ankle": _ANKLE, "abd_goniometer": _FRONT}

COLUMNS = ["stab_unit", "stab_dot1_name", "stab_dot1_shift", "stab_dot2_name", "stab_dot2_shift", "stab_pelvis_rot_deg",
           "stab_still_segment", "stab_still_rot_deg", "stab_flags"]


def rule_for(test_key, main_angle_name=None, frontal=False):
    """The rule of a test (see the module text): by test key, else by the main joint, else the pelvis only."""
    if test_key in RULES:
        return RULES[test_key]
    if frontal:
        return _FRONT
    return BY_MAIN_ANGLE.get(main_angle_name, _HIP)


def _unavailable(why):
    return {"available": False, "why": why}


def _pts(tracks, dot, mask):
    """(x, y) arrays of `dot` in the pictures of `mask` where it is known (lost pictures left out), else None."""
    if dot not in tracks:
        return None
    x, y = tracks[dot][0][mask], tracks[dot][1][mask]
    ok = ~(np.isnan(x) | np.isnan(y))
    return (x[ok], y[ok]) if ok.any() else None


def _line_deg(a, b):
    """Direction of the line a -> b in the picture (y down), degrees; clockwise on screen is positive."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _turn(a0, b0, a1, b1, sign):
    """How far the line a-b turned from (a0, b0) to (a1, b1), degrees in (-180, 180], `sign` -1 to mirror; None when a point is missing."""
    if any(p is None for p in (a0, b0, a1, b1)):
        return None
    d = (_line_deg(a1, b1) - _line_deg(a0, b0) + 180.0) % 360.0 - 180.0
    return sign * (180.0 if d == -180.0 else d)


def measure(t, tracks, rest_from, rest_to, t_peak, px_per_cm=None, facing_right=True, rule=_HIP, frontal=False, picture_delay=0.0):
    """Rest to peak force, from the dot tracks. `t`: picture times; `tracks`: dot -> (x, y arrays, NaN where lost); `rest_from`, `rest_to`: the
    resting stretch in seconds; `t_peak`: the time of the peak force (the force clock; the picture delay is added here). Returns a dict:
    available (False with `why` when nothing can be given), unit ('cm' or 'px'), dots (a list of {name, shift, dx, dy} in the rule's order,
    shift None where a dot is missing), pelvis_rot_deg, still_segment, still_rot_deg (None where missing), flags (plain sentences, empty when
    all is within the proposed limits or the rule judges nothing), judged, rest_pictures, peak_pictures."""
    t = np.asarray(t, float)
    rest = (t >= rest_from) & (t < rest_to)
    if int(rest.sum()) < MIN_REST_PICTURES:
        return _unavailable("there are too few pictures in the resting stretch before the push")
    t_pic = t_peak + picture_delay
    near = np.abs(t - t_pic) <= PEAK_HALF_S
    if not near.any():
        near = np.zeros(t.size, bool)
        near[int(np.argmin(np.abs(t - t_pic)))] = True            # no picture within 0.1 s: the nearest one, as the angle at the peak does
    sign = 1.0 if (frontal or facing_right) else -1.0
    unit = "cm" if px_per_cm else "px"
    scale = 1.0 / px_per_cm if px_per_cm else 1.0

    def rest_xy(dot):
        p = _pts(tracks, dot, rest)
        return None if p is None or p[0].size < MIN_REST_PICTURES else (float(np.median(p[0])), float(np.median(p[1])))

    def peak_xy(dot):
        p = _pts(tracks, dot, near)                               # (a dot lost in all of those pictures has no peak position: blank, as an angle does)
        return None if p is None else (float(np.mean(p[0])), float(np.mean(p[1])))

    dots = []
    for name in rule.shown_dots:
        a, b = rest_xy(name), peak_xy(name)
        if a is None or b is None:
            dots.append({"name": name, "shift": None, "dx": None, "dy": None})
        else:
            dots.append({"name": name, "shift": math.hypot(b[0] - a[0], b[1] - a[1]) * scale, "dx": (b[0] - a[0]) * scale, "dy": (b[1] - a[1]) * scale})
    p0, p1 = rule.pelvis_dots
    pelvis = _turn(rest_xy(p0), rest_xy(p1), peak_xy(p0), peak_xy(p1), sign)
    still = None
    if rule.still_dots:
        s0, s1 = rule.still_dots
        still = _turn(rest_xy(s0), rest_xy(s1), peak_xy(s0), peak_xy(s1), sign)
    if pelvis is None and still is None and all(d["shift"] is None for d in dots):
        return _unavailable("the dots of the pelvis are not tracked in the resting stretch or at the peak force")
    flags = []
    if rule.judged:
        for d in dots:
            if unit == "cm" and d["shift"] is not None and d["shift"] > PROPOSED_DISPLACEMENT_LIMIT_CM:
                flags.append(f"the {d['name'].replace('_', ' ')} dot moved {d['shift']:.1f} cm (proposed limit {PROPOSED_DISPLACEMENT_LIMIT_CM:g} cm)")
        if pelvis is not None and abs(pelvis) > PROPOSED_ROTATION_LIMIT_DEG:
            flags.append(f"the pelvis line turned {abs(pelvis):.1f} deg (proposed limit {PROPOSED_ROTATION_LIMIT_DEG:g} deg)")
        if still is not None and abs(still) > PROPOSED_ROTATION_LIMIT_DEG:
            flags.append(f"the {rule.still_name} turned {abs(still):.1f} deg (proposed limit {PROPOSED_ROTATION_LIMIT_DEG:g} deg)")
    return {"available": True, "unit": unit, "dots": dots, "pelvis_rot_deg": pelvis, "still_segment": rule.still_name, "still_rot_deg": still,
            "flags": flags, "judged": rule.judged, "rule": rule.why, "rest_pictures": int(rest.sum()), "peak_pictures": int(near.sum()),
            "px_per_cm": px_per_cm, "basis": BASIS}


def of_trial(tr, t0, t1, picture_delay=0.0, fs=None):
    """The measure for a loaded Trial (review.Trial): the resting stretch is Trial.force_rest's and the peak is the force summary's peak in the
    window [t0, t1]. `fs`: that summary if the caller has it. {'available': False, 'why': ...} for a trial without force, without a clear
    push, or without the dots; never raises for a missing thing."""
    if not tr.has_force():
        return _unavailable("this trial has no force")
    ri = tr.force_rest()
    if ri is None or ri.get("level") is None or (isinstance(ri["level"], float) and math.isnan(ri["level"])):
        return _unavailable("there is no clear rest before the push in this trial")
    if fs is None:
        fs = tr.force_summary(t0, t1, picture_delay)
    if fs is None:
        return _unavailable("the force has no peak in the window")
    main = tr.primary_angle
    frontal = (tr.meta.get("view") == "frontal")
    rule = rule_for(tr.meta.get("test_key"), main.name if main is not None else None, frontal)
    return measure(tr.t, tr.tracks, ri["t_from"], ri["t_to"], fs["t_peak_s"], tr.px_per_cm, tr.facing_right, rule, frontal, picture_delay)


def _num(v):
    return "" if v is None else v


def export_cells(res):
    """The export row's cells (COLUMNS) for a result of `measure` / `of_trial`; all blank when it is not available."""
    out = {k: "" for k in COLUMNS}
    if not res or not res.get("available"):
        return out
    out["stab_unit"] = res["unit"]
    for n, d in zip((1, 2), res["dots"]):
        out[f"stab_dot{n}_name"], out[f"stab_dot{n}_shift"] = d["name"], _num(d["shift"])
    out["stab_pelvis_rot_deg"] = _num(res["pelvis_rot_deg"])
    out["stab_still_segment"], out["stab_still_rot_deg"] = res["still_segment"], _num(res["still_rot_deg"])
    out["stab_flags"] = "; ".join(res["flags"])
    return out


def check_item(res):
    """What the Review checks list says: (level, text, title) with level 'check' (above a proposed limit) or 'ok', or None when the measure is
    not available (then there is no line; the export cells are blank)."""
    if not res or not res.get("available"):
        return None
    parts = []
    for d in res["dots"]:
        if d["shift"] is not None:
            parts.append(f"the {d['name'].replace('_', ' ')} dot moved {d['shift']:.1f} {res['unit']}")
    if res["pelvis_rot_deg"] is not None:
        parts.append(f"the pelvis line turned {res['pelvis_rot_deg']:+.1f} deg")
    if res["still_rot_deg"] is not None:
        parts.append(f"the {res['still_segment']} turned {res['still_rot_deg']:+.1f} deg")
    if not parts:
        return None
    body = "From rest to the peak force " + ", ".join(parts) + "."
    if not res["judged"]:
        return ("ok", f"Stabilisation (not judged in this test: {res['rule']}): {body}", "Stabilisation: not judged")
    if res["flags"]:
        return ("check", f"Stabilisation: {body} Above the proposed limit (the group has not confirmed one): " + "; ".join(res["flags"])
                + ". Was the pelvis held? The model assumes it is.", "Pelvis or segment moved under load")
    note = "" if res["unit"] == "cm" else " (no ruler: the shifts are in pixels and not judged)"
    return ("ok", f"Stabilisation: {body} Within the proposed limits ({PROPOSED_DISPLACEMENT_LIMIT_CM:g} cm, {PROPOSED_ROTATION_LIMIT_DEG:g} deg){note}.",
            "Stabilisation within limits")
