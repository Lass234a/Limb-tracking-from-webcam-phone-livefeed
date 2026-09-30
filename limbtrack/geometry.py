"""Pure angle maths. No camera or GUI code here so it is easy to test."""
import math
from dataclasses import dataclass

import numpy as np


def interior_angle(a, b, c):
    """Angle at vertex b between rays b->a and b->c, in degrees (0..180).

    Returns NaN if either segment has zero length.
    """
    ba = np.asarray(a, float) - np.asarray(b, float)
    bc = np.asarray(c, float) - np.asarray(b, float)
    if np.hypot(*ba) < 1e-9 or np.hypot(*bc) < 1e-9:
        return float("nan")
    cross = abs(ba[0] * bc[1] - ba[1] * bc[0])
    dot = float(np.dot(ba, bc))
    return math.degrees(math.atan2(cross, dot))


def segment_vs_vertical(p, q):
    """Angle between the segment p->q and the image vertical, degrees (0..180).

    0 = q directly above p, 90 = horizontal, 180 = q directly below p.
    Image y grows downward, so "up" is (0, -1).
    """
    v = np.asarray(q, float) - np.asarray(p, float)
    if np.hypot(*v) < 1e-9:
        return float("nan")
    return math.degrees(math.atan2(abs(v[0]), -v[1]))


def to_display(raw_angle, sign=1.0, offset=0.0):
    """Convert a raw geometric angle to the convention used in the protocol.

    display = sign * raw + offset
      interior angle:   sign= 1, offset=   0
      knee/hip flexion: sign=-1, offset= 180   (0 deg = fully extended)
      ankle from 90:    sign= 1, offset= -90   (+ plantarflexion, - dorsiflexion)
    """
    return sign * raw_angle + offset


# ---------------------------------------------------------------------------
# Vector-based angle definitions (goniometer style: the angle between two arms)
#
# An arm is one of:
#   "up" / "down" / "forward"          fixed directions ("forward" = the way the participant faces)
#   ["dotA", "dotB"]                   the vector from dot A to dot B
#   {"perp": ["dotA", "dotB"]}         perpendicular to A->B, pointing towards the feet (image down)
#
# Everything is computed in a frame where the participant faces RIGHT. If they face left
# the x direction of dot-based vectors is mirrored first, so signs mean the same thing
# whichever way the participant faces.
# ---------------------------------------------------------------------------
_FIXED = {"up": (0.0, -1.0), "down": (0.0, 1.0), "forward": (1.0, 0.0)}


def spec_dots(spec):
    """Names of the dots an arm depends on."""
    if isinstance(spec, str):
        return []
    if isinstance(spec, dict):
        return list(spec["perp"])
    return list(spec)


def resolve_arm(spec, xy, facing_right=True):
    """The arm as a vector in the 'facing right' frame, or None if a dot is missing."""
    flip = 1.0 if facing_right else -1.0
    if isinstance(spec, str):
        return np.array(_FIXED[spec], float)
    names = spec_dots(spec)
    if any(n not in xy for n in names):
        return None
    (xa, ya), (xb, yb) = xy[names[0]], xy[names[1]]
    d = np.array([(xb - xa) * flip, yb - ya], float)
    if isinstance(spec, dict):
        p = np.array([-d[1], d[0]])
        return p if p[1] >= 0 else -p
    return d


def vector_angle(v1, v2, signed=False):
    """Angle from v1 to v2 in degrees. Unsigned: 0..180. Signed: -180..180 (image coordinates, y down,
    positive = clockwise on screen). NaN if either vector has zero length."""
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    if np.hypot(*v1) < 1e-9 or np.hypot(*v2) < 1e-9:
        return float("nan")
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = float(np.dot(v1, v2))
    return math.degrees(math.atan2(cross if signed else abs(cross), dot))


@dataclass
class AngleGeometry:
    value: float        # in the protocol's convention (NaN if not computable)
    fulcrum: tuple      # image position where the read-out is drawn
    dir1: tuple         # unit vectors of the two arms, in image coordinates (for drawing)
    dir2: tuple


def evaluate_angle(defn, xy, facing_right=True):
    """Evaluate an angle definition (needs .vectors .signed .sign .offset .fulcrum) on dot positions
    `xy` (name -> (x, y)). Returns AngleGeometry, or None if a needed dot is missing."""
    v1 = resolve_arm(defn.vectors[0], xy, facing_right)
    v2 = resolve_arm(defn.vectors[1], xy, facing_right)
    if v1 is None or v2 is None:
        return None
    value = to_display(vector_angle(v1, v2, defn.signed), defn.sign, defn.offset)
    flip = 1.0 if facing_right else -1.0

    def unit(v):
        n = float(np.hypot(*v))
        return (0.0, 0.0) if n < 1e-9 else (v[0] * flip / n, v[1] / n)

    spec_dots_all = [d for s in defn.vectors for d in spec_dots(s)]
    f = defn.fulcrum or (spec_dots_all[0] if spec_dots_all else None)
    return AngleGeometry(value, tuple(xy[f]) if f in xy else (0.0, 0.0), unit(v1), unit(v2))


def anchored_ghost(snapshot, live_xy, anchor=None):
    """Positions of the locked pose ("ghost"), optionally shifted so the `anchor` dot sits on its live position.

    Anchored: the ghost shows only how the limb has rotated, not how the whole body has shifted in the picture.
    """
    dx = dy = 0.0
    if anchor and anchor in snapshot and anchor in live_xy:
        dx, dy = live_xy[anchor][0] - snapshot[anchor][0], live_xy[anchor][1] - snapshot[anchor][1]
    return {n: (x + dx, y + dy) for n, (x, y) in snapshot.items()}
