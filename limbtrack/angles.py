"""Turns an AngleDef plus dot positions into an AngleResult (value and drawing geometry)."""
from .geometry import evaluate_angle
from .results import LOST, AngleResult


def measure(defn, xy, facing_right=True):
    """Measure one angle. `xy`: name -> (x, y) for every dot that is currently trustworthy.

    If a needed dot is missing the result has status LOST and a NaN value.
    """
    r = AngleResult(defn.name, defn.title, defn.primary)
    g = evaluate_angle(defn, xy, facing_right)
    if g is None:
        r.status = LOST
        return r
    r.value, r.fulcrum, r.dir1, r.dir2 = g.value, g.fulcrum, g.dir1, g.dir2
    for spec, direction in zip(defn.vectors, (g.dir1, g.dir2)):
        if isinstance(spec, (list, tuple)):
            r.segments.append((spec[0], spec[1]))
        else:                               # 'up' / 'forward' / perpendicular: no dot-to-dot line to draw
            r.rays.append(direction)
    return r


def segment_pairs(angle_defs):
    """Unique (dotA, dotB) limb segments used by a set of angle definitions, in order."""
    out = []
    for a in angle_defs:
        for spec in a.vectors:
            if isinstance(spec, (list, tuple)):
                pair = (spec[0], spec[1])
                if pair not in out and pair[::-1] not in out:
                    out.append(pair)
    return out
