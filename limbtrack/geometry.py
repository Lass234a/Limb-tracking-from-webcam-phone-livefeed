"""Pure angle maths. No camera or GUI code here so it is easy to test."""
import math

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
