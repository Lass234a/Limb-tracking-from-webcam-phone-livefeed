"""Draws dots, limb lines and angle read-outs onto a video frame."""
import math

import cv2
import numpy as np

from .results import INFO, LOST, OK, OUT

COLOURS = {  # BGR
    OK: (80, 200, 60),
    OUT: (50, 50, 235),
    LOST: (150, 150, 150),
    INFO: (235, 235, 235),
}
DOT_OK, DOT_WEAK = (0, 215, 255), (0, 140, 255)   # yellow / orange
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale, colour, thick=2):
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def _arc(img, vertex, a, c, radius, colour, thick):
    """Draw the arc at `vertex` between the rays towards a and c."""
    ang_a = math.degrees(math.atan2(a[1] - vertex[1], a[0] - vertex[0]))
    ang_c = math.degrees(math.atan2(c[1] - vertex[1], c[0] - vertex[0]))
    diff = (ang_c - ang_a + 180) % 360 - 180  # shortest way round
    cv2.ellipse(img, (int(vertex[0]), int(vertex[1])), (radius, radius), 0,
                ang_a, ang_a + diff, colour, thick, cv2.LINE_AA)


# PARKED-GHOST: GHOST = (255, 200, 0)   # BGR bright cyan-blue
TRAIL_COLOURS = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (182, 89, 155),
                 (15, 196, 241), (156, 188, 26), (34, 126, 230), (166, 166, 127)]   # BGR, one per dot in turn


def _draw_trails(img, trails, s):
    """Thin coloured path of every dot. `trails`: dot name -> list of segments (each a list of (x, y)).

    A new segment starts whenever the dot was lost, so no line is drawn across a gap.
    """
    width = max(1, int(round(1.5 * s)))
    for k, segments in enumerate(trails.values()):
        colour = TRAIL_COLOURS[k % len(TRAIL_COLOURS)]
        for seg in segments:
            if len(seg) >= 2:
                pts = np.round(np.asarray(seg, float)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, colour, width, cv2.LINE_AA)


def _dashed(img, p, q, col, thick, dash=12):
    p, q = np.array(p, float), np.array(q, float)
    length = float(np.hypot(*(q - p)))
    if length < 1:
        return
    u = (q - p) / length
    for k in range(0, int(length), dash * 2):
        a, b = p + u * k, p + u * min(k + dash, length)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, thick, cv2.LINE_AA)


# PARKED-GHOST: def _draw_ghost(img, ghost, segments, dots, s, thick):
    # PARKED-GHOST: """Faint dashed copy of the locked pose, with a thin line from each ghost dot to its live dot."""
    # PARKED-GHOST: layer = img.copy()
    # PARKED-GHOST: for a, b in segments:
        # PARKED-GHOST: if a in ghost and b in ghost:
            # PARKED-GHOST: _dashed(layer, ghost[a], ghost[b], GHOST, thick + 2)
    # PARKED-GHOST: for n, (gx, gy) in ghost.items():
        # PARKED-GHOST: cv2.circle(layer, (int(gx), int(gy)), max(int(6 * s), 3), GHOST, thick, cv2.LINE_AA)
        # PARKED-GHOST: d = dots.get(n)
        # PARKED-GHOST: if d is not None and not d.lost and np.hypot(d.x - gx, d.y - gy) > 3:
            # PARKED-GHOST: cv2.line(layer, (int(gx), int(gy)), (int(d.x), int(d.y)), (255, 255, 255), 1, cv2.LINE_AA)
    # PARKED-GHOST: return cv2.addWeighted(layer, 0.85, img, 0.15, 0)


def draw(frame, dots, angles, header_lines=(), recording=False, trails=None):
# PARKED-GHOST: # with the ghost: def draw(frame, dots, angles, header_lines=(), recording=False, ghost=None, ghost_segments=(), trails=None)
    """Return a copy of `frame` with the overlay. `dots`: name -> object with x, y, lost.

    `trails`: dot name -> list of segments of (x, y), drawn as thin paths (or None).
    """
    img = frame.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = h / 720.0
    thick = max(1, int(round(2 * s)))
    # PARKED-GHOST: if ghost:
        # PARKED-GHOST: img = _draw_ghost(img, ghost, ghost_segments, dots, s, thick)
    if trails:
        _draw_trails(img, trails, s)

    for a in angles:
        col = COLOURS[a.status]
        for p, q in a.segments:
            dp, dq = dots.get(p), dots.get(q)
            if dp is not None and dq is not None:
                cv2.line(img, (int(dp.x), int(dp.y)), (int(dq.x), int(dq.y)), col, thick, cv2.LINE_AA)
        if a.status == LOST or a.fulcrum is None:
            continue
        fx, fy = a.fulcrum
        for d in a.rays:                     # reference directions (vertical, forward, pelvis midline)
            cv2.line(img, (int(fx), int(fy)), (int(fx + d[0] * 80 * s), int(fy + d[1] * 80 * s)), col, 1, cv2.LINE_AA)
        if a.dir1 and a.dir2:
            _arc(img, (fx, fy), (fx + a.dir1[0] * 100, fy + a.dir1[1] * 100),
                 (fx + a.dir2[0] * 100, fy + a.dir2[1] * 100), int(28 * s), col, thick)
        _text(img, f"{a.value:.1f}", (fx + 36 * s, fy + 10 * s), 0.8 * s, col, thick + 1)

    for name, d in dots.items():
        col = COLOURS[LOST] if d.lost else DOT_WEAK if getattr(d, 'weak', False) else DOT_OK
        c = (int(d.x), int(d.y))
        r = max(int(8 * s), 4)
        if d.lost:
            cv2.line(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col, thick, cv2.LINE_AA)
            cv2.line(img, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), col, thick, cv2.LINE_AA)
        else:
            cv2.circle(img, c, r, col, thick, cv2.LINE_AA)
        _text(img, name, (c[0] - r * 2, c[1] - r - 6 * s), 0.5 * s, col, 1)

    y = 34 * s
    for line in header_lines:
        _text(img, line, (12, y), 0.7 * s, (255, 255, 255), 2)
        y += 30 * s
    y += 6 * s
    for a in angles:
        col = COLOURS[a.status]
        if a.status == LOST:
            s_val = "lost"
        else:
            s_val = f"{a.value:6.1f}"
            if a.reference is not None:
                s_val += f"  ({a.deviation:+.1f} vs {a.reference:.0f})"
        _text(img, f"{a.title}: {s_val}", (12, y), 0.75 * s, col, thick)
        y += 32 * s

    if recording:
        cv2.circle(img, (w - int(28 * s), int(28 * s)), int(9 * s), (40, 40, 240), -1, cv2.LINE_AA)
        _text(img, "REC", (w - int(90 * s), int(36 * s)), 0.7 * s, (40, 40, 240), thick)
    return img
