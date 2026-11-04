"""Participant measures (0.34.0, F2): the numbers about the person that the strength scaling needs and that the camera cannot give.

Typed in the Participant section (height, mass, body fat, the thigh and shank length by tape, the ASIS-to-ASIS width), kept per participant
code like the tested leg, and saved in every trial's meta.json as `participant_measures`. Nothing here is measured by the program and no
measured number depends on it. The one thing the program does with them: when a ruler scale exists and a typed segment length differs from
the length between the matching dots in the picture by more than WARN_PCT, a quiet warning comes up (the ruler may not be in the leg's
plane, a dot may sit off its landmark, or the tape was read wrongly). It never stops Record.

Units are written into the key names (cm, kg, %): the AnyBody export converts to metres later. An empty field is "not measured" (None); a
value outside the plausible range is NOT kept (it would go into the scaling unseen), and the window says why.
"""
import math

# key, label, unit, lowest plausible, highest plausible (adults), what to measure
FIELDS = (
    ("height_cm", "Height", "cm", 120.0, 230.0, "Standing height without shoes."),
    ("mass_kg", "Mass", "kg", 30.0, 250.0, "Body mass in light clothing."),
    ("fat_pct", "Body fat", "%", 2.0, 60.0, "Body fat in percent of the body mass (from the scale or the method of the study)."),
    ("thigh_length_cm", "Thigh length", "cm", 25.0, 65.0,
     "By tape: from the greater trochanter to the lateral femoral epicondyle, along the thigh."),
    ("shank_length_cm", "Shank length", "cm", 20.0, 60.0,
     "By tape: from the head of the fibula to the lateral malleolus, along the shank."),
    ("asis_width_cm", "ASIS to ASIS", "cm", 15.0, 45.0, "By tape or calipers: the distance between the two anterior superior iliac spines."),
)
KEYS = tuple(f[0] for f in FIELDS)
LABEL = {f[0]: f[1] for f in FIELDS}
UNIT = {f[0]: f[2] for f in FIELDS}
RANGE = {f[0]: (f[3], f[4]) for f in FIELDS}
HELP = {f[0]: f[5] for f in FIELDS}

WARN_PCT = 10.0                  # a typed segment length and the picture's differ by more than this: the quiet warning
DECIMALS = 1

# The segments that can be checked against the picture: key of the typed length, the two dots whose distance is that length.
SEGMENTS = {
    "thigh": ("thigh_length_cm", "trochanter", "epicondyle"),
    "shank": ("shank_length_cm", "fibular_head", "malleolus"),
}
SEGMENT_TITLE = {"thigh": "Thigh", "shank": "Shank"}

NOTE = ("typed by the operator, never measured by the program; thigh = tape from the greater trochanter to the lateral femoral epicondyle, "
        "shank = tape from the head of the fibula to the lateral malleolus, asis_width = distance between the two ASIS; null = not measured")


def range_text(key):
    lo, hi = RANGE[key]
    return f"{lo:g} to {hi:g} {UNIT[key]}"


def parse(key, text):
    """What was typed in a box -> (value or None, problem or None). Empty is fine (None, None). A comma may be the decimal mark and the
    unit may be typed after the number ("182 cm"). A number outside the plausible range or something that is not a number gives
    (None, sentence): it is not kept."""
    s = str(text or "").strip().lower().replace(",", ".")
    for suffix in (UNIT[key], "%"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    if not s:
        return None, None
    try:
        v = float(s)
    except ValueError:
        return None, f"{LABEL[key]}: '{text.strip()}' is not a number."
    if not math.isfinite(v):
        return None, f"{LABEL[key]}: '{text.strip()}' is not a number."
    problem = check_value(key, v)
    return (None, problem) if problem else (round(v, DECIMALS), None)


def check_value(key, value):
    """None when the value is plausible for this measure, else the sentence that says why not."""
    if key not in RANGE:
        return f"'{key}' is not one of the participant measures ({', '.join(KEYS)})."
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return f"{LABEL[key]}: {value!r} is not a number."
    lo, hi = RANGE[key]
    if not lo <= value <= hi:
        return f"{LABEL[key]}: {value:g} {UNIT[key]} is outside the plausible {range_text(key)}, so it is not kept. Typed the wrong unit?"
    return None


def clean(measures):
    """{key: value or None} for every measure, from a dict that may hold only some of them. Raises ValueError for an unknown key or a value
    that is not plausible (the engine refuses it; the window shows the sentence)."""
    out = {k: None for k in KEYS}
    for k, v in dict(measures or {}).items():
        if v is None:
            if k not in out:
                raise ValueError(check_value(k, 0.0))
            continue
        problem = check_value(k, v)
        if problem:
            raise ValueError(problem)
        out[k] = round(float(v), DECIMALS)
    return out


def entered(measures):
    """Only the measures that have a value."""
    return {k: v for k, v in dict(measures or {}).items() if v is not None and k in RANGE}


def picture_check(measures, positions_px, px_per_cm):
    """The typed segment lengths against the picture. `positions_px`: dot name -> (x, y) of the dots that are tracked now; `px_per_cm`: the
    ruler scale (None without a ruler). Returns {segment: {"typed_cm", "picture_cm", "difference_pct", "warn", "dots"}} for each segment
    that has a typed length and both dots; {} when there is no ruler or nothing to compare. difference_pct = (picture - typed) / typed."""
    out = {}
    if not px_per_cm or px_per_cm <= 0:
        return out
    for name, (key, a, b) in SEGMENTS.items():
        typed = (measures or {}).get(key)
        if not typed or a not in positions_px or b not in positions_px:
            continue
        (xa, ya), (xb, yb) = positions_px[a], positions_px[b]
        picture = math.hypot(xa - xb, ya - yb) / px_per_cm
        diff = 100.0 * (picture - typed) / typed
        out[name] = {"typed_cm": float(typed), "picture_cm": round(picture, 2), "difference_pct": round(diff, 1),
                     "warn": abs(diff) > WARN_PCT, "dots": [a, b]}
    return out


def warning_text(check):
    """The quiet one-line warning for the segments that differ by more than WARN_PCT ('' when none does)."""
    parts = []
    for name, c in check.items():
        if c["warn"]:
            parts.append(f"{SEGMENT_TITLE[name].lower()} {c['typed_cm']:.1f} cm by tape, {c['picture_cm']:.1f} cm in the picture "
                         f"({c['difference_pct']:+.0f} %)")
    if not parts:
        return ""
    return ("Measures: " + "; ".join(parts) + ". Ruler in the leg's plane? Dots on the landmarks? Tape read right?")


def meta_block(measures, check=None):
    """What meta.json holds as `participant_measures`: every measure (null = not measured), the picture check made when the recording
    started (null: no ruler, or no typed length with both dots), and a note. None when nothing was typed for this participant."""
    if not entered(measures):
        return None
    block = {k: (measures or {}).get(k) for k in KEYS}
    block["picture_check"] = dict(check) if check else None
    block["picture_check_warn_pct"] = WARN_PCT
    block["note"] = NOTE
    return block
