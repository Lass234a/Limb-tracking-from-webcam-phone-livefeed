"""A standing reference (0.34.0, plan feature F4): the participant stands in the anatomical posture for 3 seconds with the dots marked, and the
hip, knee and ankle angles the dots read then are kept as the OFFSETS of those angles, together with the lengths of the leg segments in the
picture.

This is a thin layer on the angle reference of 0.33.0 (limbtrack/angle_reference.py), not a second mechanism. Pressing it does what pressing
*Set now* in the Angle reference box does for every angle the dots allow, in one go and with the dots' angle averaged over 3 s instead of
0.2 s: the joint is called 0 degrees (the anatomical zero of all three angles in protocol.json: the thigh in line with the pelvis, a straight
knee, the foot square to the shank), or the number typed from a goniometer or inclinometer if the operator read one, and the offset is what the
dots read minus that. The offsets go into the engine's angle reference, so the picture, the target and the verdict use them exactly as for a
hand-set reference, and `meta.json` `angle_reference` says so. What this module adds is the RECORD of the standing capture (`standing_reference`
in meta.json: the readings, the offsets, how steady the participant was, the segment lengths) which stays with the participant code and goes into
every later trial of that participant, so the offsets and the lengths are in the export even in a trial where the dots were marked again and the
live reference was therefore dropped.

Decisions (mine; the plan leaves them open)
  * A standing reference and one set by hand are the same kind of entry: THE LATEST WINS, per angle. Pressing the standing reference replaces a
    hand-set reference of an angle it captured (the message says which); setting one by hand afterwards replaces the standing one of that
    angle. Angles the standing capture could not read keep what they had. The meta.json block says which angles of the trial still use the
    standing offset (`in_use_in_this_trial`).
  * An angle whose pictures moved (standard deviation above MAX_SD_DEG) or were lost (fewer than MIN_VALID_SHARE of the pictures) is NOT set:
    an offset taken from a moving person is wrong in a way nobody sees afterwards. The message says which, and the others are still set.
  * The offset is only as good as the idea that it stays the same later: the crest dot is not on the pelvis' midline and the offset of a standing
    person need not equal that of a seated one (the pelvis tilts, the lower back flexes, the thigh soft tissue moves). The offsets are saved to
    be looked at against the hand instrument readings of the seated positions (reference readings, 0.16.0); they do not correct anything by
    themselves.
Raw data is untouched: data.csv holds the dots' angle, always.
"""
import copy
import math

import numpy as np

CAPTURE_S = 3.0                  # how long the participant stands still
ANGLE_NAMES = ("hip", "knee", "ankle")      # the angles of a side-view test that have an anatomical zero; 0 for all three in protocol.json
ANATOMICAL_ZERO_DEG = 0.0
MIN_PICTURES = 10                # fewer pictures with the angle readable: not set
MIN_VALID_SHARE = 0.8            # ... or readable in fewer than this share of the pictures of the capture
MAX_SD_DEG = 2.0                 # PROPOSAL: the dots' angle varied more than this (SD, 0.2 s averages) during the capture: the person moved
MAX_GAP_S = 0.5                  # no picture for this long during the capture (a break, the camera): cancelled
SEGMENTS = {"thigh": ("trochanter", "epicondyle"), "shank": ("fibular_head", "malleolus"), "foot": ("mt5_base", "mt5_head"),
            "pelvis": ("iliac_crest", "trochanter")}
NOTE = ("offset = what the dots read in the standing posture minus the joint angle called then (0 = anatomical zero, or the instrument reading "
        "typed); lengths are the distances between the two dots of a segment in the picture (cm only at the ruler's distance from the camera). "
        "data.csv holds the dots' angle; an offset is only used where an angle reference was set with it.")
COLUMNS = ["standing_taken_at", "standing_hip_offset_deg", "standing_knee_offset_deg", "standing_ankle_offset_deg", "standing_basis",
           "standing_in_use", "standing_length_unit", "standing_thigh_len", "standing_shank_len", "standing_foot_len", "standing_pelvis_len"]


class StandingError(ValueError):
    """The capture cannot be made or was cancelled; the message says why in plain English."""


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _dict(v):
    """`v` if it is a dict, else {} (a damaged meta.json block costs its numbers, never the trial)."""
    return v if isinstance(v, dict) else {}


class Capture:
    """The pictures of one standing capture. `names`: the angles to read; `readings`: {angle name: degrees the joint is at, typed from an
    instrument} (an angle without one is called 0). `feed` is called once per new picture."""

    def __init__(self, names, readings=None, instrument="", length_s=CAPTURE_S):
        self.names = [n for n in ANGLE_NAMES if n in set(names)]          # (always hip, knee, ankle: the order of the record, whatever the test lists first)
        self.readings = dict(readings or {})
        self.instrument = instrument or ""
        self.length_s = float(length_s)
        self.t0 = self.last_t = None
        self.pictures = 0
        self.values = {n: [] for n in self.names}
        self.points = {}                                     # dot -> [(x, y)] in the pictures where it was tracked

    def progress(self):
        """0..1 of the capture so far."""
        return 0.0 if self.t0 is None else min(1.0, (self.last_t - self.t0) / self.length_s)

    def feed(self, t, angles, points):
        """One picture: `angles` {name: the 0.2 s average of the unsmoothed angle, or None when it cannot be read}, `points` {dot: (x, y)} of the
        dots tracked now. True when the capture is complete. A picture with the time of the last one is the same picture again and is ignored.
        StandingError when the pictures stopped for MAX_GAP_S."""
        if self.last_t is not None:
            if t <= self.last_t:
                return False
            if t - self.last_t > MAX_GAP_S:
                raise StandingError(f"No picture came in for {t - self.last_t:.1f} s during the capture (a break, or the camera stopped): "
                                    "nothing was kept. Capture again.")
        if self.t0 is None:
            self.t0 = t
        self.last_t = t
        self.pictures += 1
        for n in self.names:
            v = angles.get(n)
            if _finite(v):
                self.values[n].append(float(v))
        for d, (x, y) in points.items():
            self.points.setdefault(d, []).append((float(x), float(y)))
        return t - self.t0 >= self.length_s

    def finish(self, px_per_cm=None, test_key="", facing="right", taken_at=""):
        """The record of the capture: {taken_at, test_key, facing, capture_s, pictures, instrument, angles, skipped, segments, note}. `angles`
        is {name: {reading_deg, reading_source, dots_deg, offset_deg, sd_deg, pictures}} for the angles that were steady and read
        in enough pictures; `skipped` is {name: why} for the others. `segments` = {unit, px_per_cm, lengths: {segment: length}, dots}."""
        angles, skipped = {}, {}
        for n in self.names:
            v = np.array(self.values[n], float)
            if v.size < MIN_PICTURES or v.size < MIN_VALID_SHARE * self.pictures:
                skipped[n] = (f"read in only {v.size} of {self.pictures} pictures (a dot of it was lost or not marked); at least {MIN_PICTURES} and "
                              f"{100 * MIN_VALID_SHARE:.0f} % are needed")
                continue
            sd = float(np.std(v, ddof=1))
            if sd > MAX_SD_DEG:
                skipped[n] = f"it moved during the capture (SD {sd:.1f} deg, above {MAX_SD_DEG:g}): keep the participant still and capture again"
                continue
            typed = self.readings.get(n)
            ref = ANATOMICAL_ZERO_DEG if typed is None else float(typed)
            mean = float(np.mean(v))
            angles[n] = {"reading_deg": round(ref, 3), "reading_source": ("anatomical zero" if typed is None else
                                                                           f"typed ({self.instrument or 'instrument'})"),
                         "dots_deg": round(mean, 3), "offset_deg": round(mean - ref, 3), "sd_deg": round(sd, 3), "pictures": int(v.size)}
        lengths, used = {}, {}
        for seg, (a, b) in SEGMENTS.items():
            pa, pb = self.points.get(a), self.points.get(b)
            if not pa or not pb or len(pa) < MIN_PICTURES or len(pb) < MIN_PICTURES:
                continue
            ma, mb = np.mean(pa, axis=0), np.mean(pb, axis=0)
            length = float(math.hypot(*(mb - ma))) / (px_per_cm if px_per_cm else 1.0)
            lengths[seg], used[seg] = round(length, 2), [a, b]
        t0, t1 = (0.0, 0.0) if self.t0 is None else (self.t0, self.last_t)
        return {"taken_at": taken_at, "test_key": test_key, "facing": facing, "capture_s": round(t1 - t0, 2), "pictures": self.pictures,
                "instrument": self.instrument if any(self.readings.get(n) is not None for n in angles) else "",
                "angles": angles, "skipped": skipped,
                "segments": {"unit": "cm" if px_per_cm else "px", "px_per_cm": round(px_per_cm, 4) if px_per_cm else None,
                             "lengths": lengths, "dots": used},
                "note": NOTE}


def meta_block(record, in_use=()):
    """What goes into meta.json `standing_reference`: a copy of the participant's record plus which of its angles this trial used (the live
    angle reference of the trial is the standing one). None without a record."""
    if not record:
        return None
    out = copy.deepcopy(record)
    out["in_use_in_this_trial"] = [n for n in out.get("angles", {}) if n in set(in_use)]
    return out


def basis_text(block):
    """One line for the export: what each angle was called ('hip, knee: anatomical zero; ankle: typed goniometer 3.0')."""
    groups = {}
    for n, a in _dict(block.get("angles")).items():
        src = str(_dict(a).get("reading_source", ""))
        if src.startswith("typed"):
            src = f"{src} {_dict(a).get('reading_deg')}"
        groups.setdefault(src, []).append(n)
    text = "; ".join(f"{', '.join(v)}: {k}" for k, v in groups.items())
    skipped = _dict(block.get("skipped"))
    return text + (("; not set: " + ", ".join(skipped)) if skipped else "")


def export_cells(meta):
    """The export row's cells (COLUMNS) from a trial's meta.json; all blank for a trial without a standing reference or with a damaged block
    (never a made-up number)."""
    out = {k: "" for k in COLUMNS}
    block = meta.get("standing_reference")
    if not isinstance(block, dict):
        return out
    out["standing_taken_at"] = block.get("taken_at") or ""
    for n, a in _dict(block.get("angles")).items():
        if n in ANGLE_NAMES and _finite(_dict(a).get("offset_deg")):
            out[f"standing_{n}_offset_deg"] = a["offset_deg"]
    out["standing_basis"] = basis_text(block)
    used = block.get("in_use_in_this_trial")
    out["standing_in_use"] = ", ".join(str(n) for n in used) if isinstance(used, list) else ""
    seg = _dict(block.get("segments"))
    out["standing_length_unit"] = str(seg.get("unit") or "")
    for name in SEGMENTS:
        v = _dict(seg.get("lengths")).get(name)
        if _finite(v):
            out[f"standing_{name}_len"] = v
    return out


def check_line(block, offsets_in_trial):
    """A sentence for Review's checks list about the standing reference in a trial, or None. `offsets_in_trial`: {angle name: offset} the trial
    was recorded with (its angle_reference). Says plainly whether the standing offsets were used in this trial."""
    if not isinstance(block, dict) or not _dict(block.get("angles")):
        return None
    shown = ", ".join(f"{n} {a['offset_deg']:+.1f}" for n, a in block["angles"].items() if _finite(_dict(a).get("offset_deg")))
    used = block.get("in_use_in_this_trial") if isinstance(block.get("in_use_in_this_trial"), list) else []
    text = f"Standing reference of this participant (taken {block.get('taken_at', '?')}): offsets in degrees, {shown}."
    if used:
        text += f" In use in this trial: {', '.join(used)} (the angles are shown from them; the dots' own angle is in data.csv)."
    elif offsets_in_trial:
        text += " Not in use in this trial: its angle reference was set by hand."
    else:
        text += " Not in use in this trial: no angle reference was set (the dots were marked again), so the angles are the dots' own."
    return text
