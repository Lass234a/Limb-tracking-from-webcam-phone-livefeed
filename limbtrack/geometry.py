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


# ---------------------------------------------------------------------------
# Computed points: points that are calculated from marked dots instead of being clicked (0.15.0).
#
# "hip_centre_frontal": the hip joint centre in a picture taken from the front, from the two ASIS dots and the picture's scale.
# There is no bone point to palpate at the hip joint (it is deep), so it is estimated from the pelvis with the regression of
#   Harrington ME, Zavatsky AB, Lawson SEM, Yuan Z, Theologis TN. Prediction of the hip joint centre in adults, children, and
#   patients with cerebral palsy based on magnetic resonance imaging. J Biomech 2007;40(3):595-602 (PubMed 16584737, DOI 10.1016/j.jbiomech.2006.02.003).
# Equation set 1, right hip, millimetres, origin at the midpoint of the two ASIS, PW = the distance between the two ASIS:
#   medio-lateral (towards the side of the hip)  =  0.33 * PW + 7.3
#   supero-inferior (downwards)                  =  0.30 * PW + 10.9
#   (antero-posterior = -0.24 * PD - 9.9 with PD the pelvic depth: it is out of the plane of a frontal picture and is not used.)
# The left hip is the mirror image; here "towards the side of the hip" is the direction from the other ASIS to the tested ASIS,
# so the same code serves either leg.
# VERIFICATION STATUS (2026-10-20): the constants were taken from a secondary source (a biomechanics wiki, via a search summary) and
# the coordinate origin from a validation paper that uses the same system (PMC5121588); the original table was NOT read. Two
# consistency checks: the medio-lateral term at PW = 250 mm is 89.8 mm, and a cadaver study puts the hip centre 14 % of the pelvic
# width medial to each ASIS, i.e. 0.5 * 250 - 0.14 * 250 = 90 mm from the midpoint (Seidel et al., J Biomech 1995;28:995, PubMed 7673267).
# Check the constants against the original paper before the numbers are reported. Reported mean errors of these equations in one
# validation (PMC5121588): about 5.9 mm medio-lateral and 5.5 mm supero-inferior; soft tissue over the ASIS adds to that.
# ---------------------------------------------------------------------------
HARRINGTON_SIDEWAYS = (0.33, 7.3)     # mm = a * PW + b
HARRINGTON_DOWN = (0.30, 10.9)        # mm = a * PW + b

DERIVED_KINDS = {"hip_centre_frontal": ("tested_asis", "other_asis")}     # kind -> the keys of its definition that name tracked dots
DERIVED_OPTIONAL = {"hip_centre_frontal": ("knee",)}     # kind -> keys naming a dot that is used when it is tracked, never required (0.19.0)


def pelvic_width_mm(tested_xy, other_xy, px_per_mm):
    """The distance between the two ASIS in millimetres, or None when it cannot be told."""
    if not (px_per_mm and px_per_mm > 0 and math.isfinite(px_per_mm)):
        return None
    d = np.asarray(tested_xy, float) - np.asarray(other_xy, float)
    pw_px = float(np.hypot(*d))
    return pw_px / px_per_mm if pw_px > 1e-9 else None


def hip_centre_frontal(tested_xy, other_xy, px_per_mm, knee_xy=None):
    """The hip joint centre of the tested side in a frontal picture, as an image position (x, y), or None.

    `tested_xy`, `other_xy`: the ASIS on the tested side and on the other side (pixels). `px_per_mm`: the picture's scale at the
    pelvis (from the ruler). `knee_xy`: the knee dot of the tested leg, or None. The pelvis axes are taken from the two ASIS
    themselves: "sideways" runs along the line from the other ASIS to the tested ASIS, "down" is square to it on the side of the
    knee (without the knee: of the picture's bottom), so a tilted camera, a tilted pelvis or a participant lying on the side does
    not change the result in the pelvis' own frame. See the note above for the equation and its status."""
    axes = pelvis_axes(tested_xy, other_xy, knee_xy)
    if axes is None or not (px_per_mm and px_per_mm > 0 and math.isfinite(px_per_mm)):
        return None
    u, v, width_px = axes
    pw = width_px / px_per_mm
    side_mm = HARRINGTON_SIDEWAYS[0] * pw + HARRINGTON_SIDEWAYS[1]
    down_mm = HARRINGTON_DOWN[0] * pw + HARRINGTON_DOWN[1]
    c = (np.asarray(tested_xy, float) + np.asarray(other_xy, float)) / 2.0 + (side_mm * u + down_mm * v) * px_per_mm
    return float(c[0]), float(c[1])


def pelvis_axes(tested_xy, other_xy, knee_xy=None):
    """The pelvis' own frame in a frontal picture, from the two ASIS: (u, v, width_px). u = the unit vector from the other ASIS to
    the tested one ("sideways", towards the tested side), v = square to it ("down") on the side of the knee when `knee_xy` is given
    and off the line between the ASIS, else on the side of the picture's bottom; width_px = the distance between the two ASIS in
    pixels. None when the two are at the same place.

    The knee is what makes this work for a participant lying on the side (P20, 2026-10-22): the line between the ASIS is then
    upright in the picture, the picture's bottom is sideways along the body, and until 0.19.0 the hip centre was put on the head
    side of the pelvis in 3 of 4 such trials. Used by the hip centre and by the engine's frontal label checks."""
    t, o = np.asarray(tested_xy, float), np.asarray(other_xy, float)
    width = float(np.hypot(*(t - o)))
    if width <= 1e-9:
        return None
    u = (t - o) / width
    v = np.array([-u[1], u[0]])
    if knee_xy is not None:
        side = float(np.dot(np.asarray(knee_xy, float) - t, v))
        if abs(side) > 1e-6 * width:
            return u, (v if side > 0 else -v), width
    return u, (v if v[1] >= 0 else -v), width


_DERIVED_COMPUTE = {"hip_centre_frontal": hip_centre_frontal}    # kind -> f(the positions of its input dots, in DERIVED_KINDS order,
                                                                 # px_per_mm, <optional key>_xy=... for each DERIVED_OPTIONAL key)


def derived_input_dots(spec):
    """The tracked dots a computed point is made from (empty for an unknown kind). Optional dots (DERIVED_OPTIONAL) are not among them."""
    keys = DERIVED_KINDS.get(spec.get("kind"), ())
    return [spec[k] for k in keys if k in spec]


def compute_derived(derived, xy, px_per_mm):
    """name -> (x, y) of every computed point of `derived` (name -> definition) whose dots are all in `xy` and whose scale is known.
    An optional dot of the definition is passed when it is in `xy` (tracked), else None."""
    out = {}
    for name, spec in derived.items():
        compute = _DERIVED_COMPUTE.get(spec.get("kind"))
        points = [xy.get(n) for n in derived_input_dots(spec)]      # the same dots that are clicked for it (AngleDef.needs)
        if compute is not None and points and all(q is not None for q in points):
            extra = {f"{k}_xy": xy.get(spec[k]) for k in DERIVED_OPTIONAL.get(spec.get("kind"), ()) if k in spec}
            p = compute(*points, px_per_mm, **extra)
            if p is not None:
                out[name] = p
    return out


# PARKED-GHOST: def anchored_ghost(snapshot, live_xy, anchor=None):
    # PARKED-GHOST: """Positions of the locked pose ("ghost"), optionally shifted so the `anchor` dot sits on its live position.

    # PARKED-GHOST: Anchored: the ghost shows only how the limb has rotated, not how the whole body has shifted in the picture.
    # PARKED-GHOST: """
    # PARKED-GHOST: dx = dy = 0.0
    # PARKED-GHOST: if anchor and anchor in snapshot and anchor in live_xy:
        # PARKED-GHOST: dx, dy = live_xy[anchor][0] - snapshot[anchor][0], live_xy[anchor][1] - snapshot[anchor][1]
    # PARKED-GHOST: return {n: (x + dx, y + dy) for n, (x, y) in snapshot.items()}


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


KNEE_FACING_MIN_DEG = 25.0      # a knee bent further than this the 'wrong' way tells the view (no knee over-straightens that far)


def view_handedness(xy):
    """Which side of the body the picture shows: +1 for the view the setting "Participant faces: right side of the image" assumes,
    -1 for the mirror-image view, None when it cannot be told.

    Uses the sign of a turning direction that does not depend on posture (upright, seated, lying) or on the leg: the foot
    (malleolus, fibular head, 5th metatarsal base and head) when those four dots are marked, otherwise the bend of the knee
    (see knee_bend_handedness). The pelvis cannot be used: with one iliac crest dot the turn from crest to trochanter to
    epicondyle changes sign with the hip angle, so it does not tell which way the person faces. Until 0.9.0 the ASIS-PSIS-
    trochanter turn was used when those dots were marked.
    """
    h = foot_handedness(xy)
    return h if h is not None else knee_bend_handedness(xy)


def foot_handedness(xy):
    """+1 / -1 from the turn of the foot (fibula to 5th metatarsal), None without the four foot dots or when the turn is nearly flat."""
    def pt(n):
        return np.array(xy[n], float) if n in xy else None

    m, f, b, h = pt("malleolus"), pt("fibular_head"), pt("mt5_base"), pt("mt5_head")
    if any(v is None for v in (m, f, b, h)):
        return None
    v1, v2 = f - m, h - b
    c = _cross(v1, v2)
    n = float(np.hypot(*v1) * np.hypot(*v2))
    if n < 1e-9 or abs(c) < 0.05 * n:
        return None
    return 1 if c > 0 else -1


def knee_bend_handedness(xy):
    """+1 / -1 from which way the knee bends, None when the knee is nearly straight or its four dots are not all marked.

    A knee bends one way only. Seen in the view 'faces right', bending turns the shank (fibular head -> malleolus) clockwise on screen
    from the thigh (trochanter -> epicondyle); in the mirror view, the other way. The angle between them is used only when it is
    larger than KNEE_FACING_MIN_DEG, more than any knee over-straightens (normal: 0 to 10 deg), so a straight knee tells nothing.
    Found on 2026-10-15: two lying-face-down trials with 'faces right' read the knee as -55 and -78 deg (the sign was wrong) and
    nothing warned, because the foot dots were switched off."""
    need = ("trochanter", "epicondyle", "fibular_head", "malleolus")
    if any(n not in xy for n in need):
        return None
    t, e, f, m = (np.array(xy[n], float) for n in need)
    thigh, shank = e - t, m - f
    if np.hypot(*thigh) < 1e-9 or np.hypot(*shank) < 1e-9:
        return None
    bend = math.degrees(math.atan2(_cross(thigh, shank), float(np.dot(thigh, shank))))
    if abs(bend) < KNEE_FACING_MIN_DEG:
        return None
    return 1 if bend > 0 else -1
