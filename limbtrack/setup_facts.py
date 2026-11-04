"""Set-up facts of one position (0.34.0, F3): how the participant was held and where the strap sat, typed in the Position section.

The strength model needs to know where the strap pulls (a distance from a named landmark, not a place on a picture), which supports held
the pelvis and the thigh, and how the seat or backrest was set. Today that is written on paper and typed again later. Here it is typed
once per position, prefilled for the next trial of the same participant, test and target, and saved in every trial's meta.json as `setup`.

Nothing here is measured by the program, no measured number depends on it, and it never stops Record. An empty field is "not noted"
(None); a number outside the plausible range is not kept and the window says why. Ticks: a ticked support is in `supports`; an empty
list means nothing was ticked (the program cannot tell "none used" from "not noted", so the box says so).

The distance is the straight distance along the segment from the landmark to the MIDDLE of the strap, in cm, always positive (my choice,
2026-11-02; the SOP does not define it: the protocol only says strap locations are recorded relative to the named landmarks).
"""
import math

from . import privacy

# The named landmarks the distance can be measured from: key, label. The first four are the plan's list (F3); the superior patellar border and
# the metatarsal heads are the other landmarks the protocol names for strap locations (section 2.6), so the strap of the hip tests (just
# proximal to the femoral condyles) and the plantarflexion strap (at the metatarsal heads) can be written down without "other".
LANDMARKS = (
    ("lateral_epicondyle", "Lateral epicondyle"),
    ("lateral_malleolus", "Lateral malleolus"),
    ("trochanter", "Greater trochanter"),
    ("patellar_border", "Superior patellar border"),
    ("metatarsal_heads", "Metatarsal heads"),
    ("other", "Other (write it)"),
)
LANDMARK_KEYS = tuple(k for k, _ in LANDMARKS)
LANDMARK_LABEL = dict(LANDMARKS)

# Supports: key, label. The pelvic belt, the thigh fixation strap and the padded block are the equipment the protocol names; hands on the bench is
# how the participant holds on in the seated tests.
SUPPORTS = (
    ("pelvic_belt", "Pelvic belt"),
    ("thigh_strap", "Thigh fixation strap"),
    ("padded_block", "Padded block / pad to push against"),
    ("hands_hold", "Hands hold the bench"),
)
SUPPORT_KEYS = tuple(k for k, _ in SUPPORTS)
SUPPORT_LABEL = dict(SUPPORTS)

DISTANCE_RANGE_CM = (0.0, 80.0)          # from the landmark to the strap: longer than a thigh is a typing slip
ANGLE_RANGE_DEG = (0.0, 120.0)           # seat / backrest angle from the horizontal: 0 = lying flat, 90 = upright
OTHER_MAX_CHARS = 60
DECIMALS = 1

ANGLE_HELP = ("The angle of the backrest (or the trunk support) from horizontal: 0 = lying flat, 90 = upright. Leave empty when it does not apply.")
DISTANCE_HELP = ("Distance in cm along the segment from the named landmark to the middle of the strap. Leave empty when it was not noted.")

NOTE = ("typed by the operator, never measured by the program; strap_from_landmark_cm is the distance along the segment from the named "
        "landmark to the middle of the strap; supports lists the ticked supports (an empty list = nothing ticked, not necessarily none used); "
        "seat_back_angle_deg is the backrest from horizontal (0 lying, 90 upright); prefilled = these values were typed for an earlier trial "
        "at this position and not changed since; null = not noted")


def blank():
    return {"strap_landmark": None, "strap_landmark_other": None, "strap_from_landmark_cm": None, "supports": [], "seat_back_angle_deg": None}


def parse_number(text, lo, hi, label, unit):
    """Typed text -> (value or None, problem or None); same rules as participant_measures.parse (empty is fine, comma allowed, unit allowed)."""
    s = str(text or "").strip().lower().replace(",", ".")
    for suffix in (unit, "deg", "degrees", "°"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    if not s:
        return None, None
    try:
        v = float(s)
    except ValueError:
        return None, f"{label}: '{str(text).strip()}' is not a number."
    if not math.isfinite(v):
        return None, f"{label}: '{str(text).strip()}' is not a number."
    if not lo <= v <= hi:
        return None, f"{label}: {v:g} {unit} is outside the plausible {lo:g} to {hi:g} {unit}, so it is not kept."
    return round(v, DECIMALS), None


def parse_distance(text):
    return parse_number(text, *DISTANCE_RANGE_CM, "Strap distance", "cm")


def parse_angle(text):
    return parse_number(text, *ANGLE_RANGE_DEG, "Seat/back angle", "deg")


def clean(facts):
    """The set-up facts in their saved shape, from a dict that may hold only some of the keys. Raises ValueError for an unknown landmark or
    support, or a number outside the plausible range (the engine refuses it; the window shows the sentence)."""
    f = blank()
    facts = dict(facts or {})
    unknown = set(facts) - set(f)
    if unknown:
        raise ValueError(f"Not set-up facts: {sorted(unknown)}.")
    lm = facts.get("strap_landmark")
    if lm is not None and lm not in LANDMARK_KEYS:
        raise ValueError(f"'{lm}' is not a strap landmark ({', '.join(LANDMARK_KEYS)}).")
    f["strap_landmark"] = lm
    other = facts.get("strap_landmark_other")
    other = str(other).strip()[:OTHER_MAX_CHARS] if other is not None else ""
    problem = privacy.notes_problem(other)                     # (free text is saved with the trial: the Notes rule applies to it too)
    if problem:
        raise ValueError(f"Landmark name: {problem} Remove it: it must not identify anyone.")
    f["strap_landmark_other"] = other or None
    if f["strap_landmark"] != "other":
        f["strap_landmark_other"] = None                       # a word typed for 'other' means nothing with another landmark chosen
    for key, (lo, hi), label, unit in (("strap_from_landmark_cm", DISTANCE_RANGE_CM, "Strap distance", "cm"),
                                       ("seat_back_angle_deg", ANGLE_RANGE_DEG, "Seat/back angle", "deg")):
        v = facts.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{label}: {v!r} is not a number.")
        if not lo <= v <= hi:
            raise ValueError(f"{label}: {v:g} {unit} is outside the plausible {lo:g} to {hi:g} {unit}, so it is not kept.")
        f[key] = round(float(v), DECIMALS)
    sup = facts.get("supports") or []
    bad = [s for s in sup if s not in SUPPORT_KEYS]
    if bad:
        raise ValueError(f"{bad} are not supports ({', '.join(SUPPORT_KEYS)}).")
    f["supports"] = [s for s in SUPPORT_KEYS if s in sup]      # (in the fixed order, once each)
    return f


def is_empty(facts):
    """True when nothing was noted."""
    f = clean(facts)
    return (f["strap_landmark"] is None and f["strap_from_landmark_cm"] is None and not f["supports"] and f["seat_back_angle_deg"] is None)


def meta_block(facts, typed_at=None, prefilled=False):
    """What meta.json holds as `setup`: the facts, when they were typed, whether they are carried over from an earlier trial of the position
    unchanged, and a note. None when nothing was noted."""
    if facts is None or is_empty(facts):
        return None
    block = clean(facts)
    block["typed_at"] = typed_at
    block["prefilled"] = bool(prefilled)
    block["note"] = NOTE
    return block


def summary(facts):
    """One line for the box and the closed section: what was noted."""
    if facts is None or is_empty(facts):
        return "not noted"
    f = clean(facts)
    parts = []
    if f["strap_from_landmark_cm"] is not None or f["strap_landmark"]:
        lm = LANDMARK_LABEL.get(f["strap_landmark"], "?").lower() if f["strap_landmark"] else "a landmark not named"
        if f["strap_landmark"] == "other" and f["strap_landmark_other"]:
            lm = f["strap_landmark_other"]
        d = "?" if f["strap_from_landmark_cm"] is None else f"{f['strap_from_landmark_cm']:g} cm"
        parts.append(f"strap {d} from {lm}")
    if f["supports"]:
        parts.append(", ".join(SUPPORT_LABEL[s].lower() for s in f["supports"]))
    if f["seat_back_angle_deg"] is not None:
        parts.append(f"back {f['seat_back_angle_deg']:g} deg")
    return "; ".join(parts)
