"""An angle reference (0.33.0): "the joint is at 0 degrees now" while the dots read 10.

The iliac-crest dot is not on the pelvis' lateral midline, so the hip angle read from the dots differs from the angle of the joint by a
roughly constant amount, and a person in neutral never reads 0 (and in a hip test never reaches extension). The operator says what the joint
is at (0 in neutral, or a number read off an inclinometer in the test posture) and presses a button: the difference between what the dots
read at that moment and what was said is kept as the OFFSET of that angle. The angle from the reference is the dots' angle minus the offset.

This module is the pure part. It does not touch a measured number: the dots' angle is what data.csv holds, always. What the offset changes is
what is shown and judged live (the target is moved onto the dots' scale: target + offset) and what Review shows next to the dots' angle.

* A reference belongs to one participant, as long as the dots stay as they were marked: not to a target (it has to survive the step from
  target 0 to 45) and not to a test (the hip tests share their dots, so the reference taken for one stands for the next). One typed before
  a participant code was given belongs to whoever is recorded first, like the reference readings of 0.16.0.
* It is cleared by the engine when a dot of its angle is marked again or removed, when the angles themselves change (a three-point angle),
  and when the dots are reset (another test with other landmarks does that).
* It differs from the reference readings of `reference.py` (a goniometer or inclinometer value kept next to the app's angle, to measure how
  far the app is from the instrument): those are never applied to anything.
"""
import math
from datetime import datetime

DEFAULT_REFERENCE_DEG = 0.0
VALUE_RANGE_DEG = (-180.0, 360.0)
NOTE = ("offset = what the dots read at that moment minus what the operator said the joint was; the angle from the reference is the dots' angle "
        "minus the offset. data.csv holds the dots' angle; its target column is the target plus the offset.")


class AngleReferenceError(ValueError):
    """What was given cannot be used as a reference; the message says why in plain English."""


def check_value(value):
    """The typed reference in degrees as a float, or AngleReferenceError."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise AngleReferenceError(f'"{value}" is not a number (the reference is in degrees)') from None
    if not math.isfinite(v) or not VALUE_RANGE_DEG[0] <= v <= VALUE_RANGE_DEG[1]:
        raise AngleReferenceError(f"{value} is not an angle (between {VALUE_RANGE_DEG[0]:g} and {VALUE_RANGE_DEG[1]:g} degrees)")
    return v


class AngleReference:
    """The references of the present set-up: `owner` = the participant code ("" before one was given) and `entries` = {angle name: entry};
    an entry is {reference_deg, dots_deg, offset_deg, taken_at}."""

    def __init__(self):
        self.owner = None
        self.entries = {}

    def clear(self, name=None):
        """Forget one angle's reference, or all of them."""
        if name is None:
            self.owner, self.entries = None, {}
        else:
            self.entries.pop(name, None)
            if not self.entries:
                self.owner = None

    def belongs_to(self, owner):
        """Do the references held belong to participant `owner`? One taken before a code was given is whoever's code comes first."""
        if self.owner is None or not self.entries:
            return False
        return not self.owner or self.owner == owner

    def adopt(self, owner):
        """A reference taken before a participant code was given belongs to the first code that comes (an earlier code is kept)."""
        if self.owner is not None and not self.owner and owner:
            self.owner = owner

    def set(self, owner, name, reference_deg, dots_deg, taken_at=None):
        """Keep the reference of one angle: the joint is at `reference_deg` while the dots read `dots_deg`. A reference of another participant
        held until now is dropped first. Returns the entry."""
        ref = check_value(reference_deg)
        dots = check_value(dots_deg)
        if not self.belongs_to(owner):
            self.owner, self.entries = owner, {}
        else:
            self.adopt(owner)
        entry = {"reference_deg": ref, "dots_deg": dots, "offset_deg": dots - ref,
                 "taken_at": taken_at or datetime.now().isoformat(timespec="seconds")}
        self.entries[name] = entry
        return dict(entry)

    def active(self, owner):
        """{angle name: entry} for participant `owner`, or {} when what is held belongs to another participant."""
        return {k: dict(v) for k, v in self.entries.items()} if self.belongs_to(owner) else {}

    def offsets(self, owner):
        """{angle name: offset in degrees} for `owner`."""
        return {k: v["offset_deg"] for k, v in self.active(owner).items()}

    def to_meta(self, owner, names=None):
        """What goes into meta.json: None without a reference, else the entries by angle name (only those in `names` when given: the
        angles of the trial) and a note on what they mean."""
        got = {k: v for k, v in self.active(owner).items() if names is None or k in names}
        if not got:
            return None
        return {"angles": {k: {"reference_deg": round(v["reference_deg"], 3), "dots_deg_at_reference": round(v["dots_deg"], 3),
                               "offset_deg": round(v["offset_deg"], 3), "taken_at": v["taken_at"]} for k, v in got.items()},
                "note": NOTE}


def offsets_from_meta(meta):
    """{angle name: offset in degrees} that a trial's meta.json says were used when it was recorded ({} for a trial without a reference)."""
    out = {}
    block = meta.get("angle_reference") or {}
    for name, v in (block.get("angles") or {}).items():
        try:
            off = float(v["offset_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(off):
            out[name] = off
    return out
