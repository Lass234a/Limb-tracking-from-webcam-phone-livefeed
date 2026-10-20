"""Draws dots, limb lines and angle read-outs onto a video frame."""
import math
from types import SimpleNamespace

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
DERIVED_COLOUR = (255, 255, 0)                    # BGR cyan: a computed point (the hip centre of a frontal test), unlike a sticker
FONT = cv2.FONT_HERSHEY_SIMPLEX
PROMPT_PREFIX = "click dot: "                         # header line that asks for the next dot (engine.py writes it), drawn in yellow


def _text(img, s, org, scale, colour, thick=2):
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def _shade_behind(img, rows, x, y_first, keep=0.45):
    """Darken the picture behind the text lines (to `keep` of its brightness), each line only as wide as its text, so white text stays
    readable on a pale wall (seen on the P11 pictures, 2026-10-15) and the picture still shows through. `rows`: (text, scale, colour,
    thickness, advance) as drawn from baseline y_first downwards."""
    h, w = img.shape[:2]
    y = y_first
    for line, scale, _, th, advance in rows:
        (tw, tht), base = cv2.getTextSize(line, FONT, scale, th + 3)
        x0, x1 = max(int(x - 6), 0), min(int(x + tw + 6), w)
        y0, y1 = max(int(y - tht - 6), 0), min(int(y + base + 2), h)
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            roi[:] = (roi * keep).astype(img.dtype)
        y += advance


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


ROPE_COLOUR = (255, 0, 255)           # BGR magenta
SCALE_COLOUR = (255, 200, 0)      # BGR: cyan-blue, unlike any dot or angle colour
SCALE_WARN_COLOUR = (0, 165, 255)


def _draw_scale(img, scale, s):
    """The ruler: a ring on each of its two dots, a line between them and a label."""
    col = SCALE_WARN_COLOUR if scale.get("warn") else SCALE_COLOUR
    pts = scale["points"]
    for x, y, lost in pts:
        cv2.circle(img, (int(x), int(y)), int(9 * s), (128, 128, 128) if lost else col, max(1, int(2 * s)), cv2.LINE_AA)
    if len(pts) == 2 and not (pts[0][2] or pts[1][2]):
        cv2.line(img, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), col, 1, cv2.LINE_AA)
    if scale.get("text") and pts:
        x, y = pts[0][0], pts[0][1]
        _text(img, scale["text"], (x + 14 * s, y - 14 * s), 0.5 * s, col, 1)


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


BREAK_GHOST = (255, 255, 160)         # BGR: pale cyan, unlike any dot, angle, rope or ruler colour


def _dashed_circle(img, c, r, col, thick, n=14):
    for k in range(n):
        a0 = 360.0 * k / n
        cv2.ellipse(img, (int(c[0]), int(c[1])), (r, r), 0, a0, a0 + 360.0 / n * 0.55, col, thick, cv2.LINE_AA)


def _draw_break_ghost(img, ghost, segments, s, thick):
    """Pale dashed copy of the pose before a break: a ring round every dot (big enough for the sticker), dashed limb lines and the
    dot names. `ghost`: dot name -> (x, y, diameter in pixels); `segments`: pairs of dot names to join."""
    layer = img.copy()
    for a, b in segments:
        if a in ghost and b in ghost:
            _dashed(layer, ghost[a][:2], ghost[b][:2], BREAK_GHOST, thick)
    for n, (gx, gy, diam) in ghost.items():
        r = int(max(diam / 2.0 + 7 * s, 10 * s))
        _dashed_circle(layer, (gx, gy), r, BREAK_GHOST, thick + 1)
        _text(layer, n, (gx + r + 4, gy - r), 0.5 * s, BREAK_GHOST, 1)
    return cv2.addWeighted(layer, 0.85, img, 0.15, 0)


def draw(frame, dots, angles, header_lines=(), recording=False, trails=None, scale=None, ghost=None, ghost_segments=(), derived=None):
# PARKED-GHOST: # with the ghost: def draw(frame, dots, angles, header_lines=(), recording=False, ghost=None, ghost_segments=(), trails=None)
    """Return a copy of `frame` with the overlay. `dots`: name -> object with x, y, lost.

    `trails`: dot name -> list of segments of (x, y), drawn as thin paths (or None).
    `ghost`: dot name -> (x, y, diameter) of where the dots were before a break (drawn under everything else), or None.
    `derived`: computed points (name -> (x, y)), e.g. the hip centre of a frontal test: drawn as a diamond, and an angle's limb line may end there.
    """
    img = frame.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = h / 720.0
    thick = max(1, int(round(2 * s)))
    # PARKED-GHOST: if ghost:
        # PARKED-GHOST: img = _draw_ghost(img, ghost, ghost_segments, dots, s, thick)
    if ghost:
        img = _draw_break_ghost(img, ghost, ghost_segments, s, thick)
    if trails:
        _draw_trails(img, trails, s)
    if scale:
        _draw_scale(img, scale, s)
    a, b = dots.get("strap"), dots.get("anchor")          # the rope/chain between the strap marker and the anchor marker
    if a is not None and b is not None and not a.lost and not b.lost:
        cv2.line(img, (int(a.x), int(a.y)), (int(b.x), int(b.y)), ROPE_COLOUR, max(1, int(round(2 * s))), cv2.LINE_AA)

    ends = dict(dots)
    for n, (dx, dy) in (derived or {}).items():
        ends[n] = SimpleNamespace(x=dx, y=dy, lost=False)
    for a in angles:
        col = COLOURS[a.status]
        for p, q in a.segments:
            dp, dq = ends.get(p), ends.get(q)
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
        col = COLOURS[LOST] if d.lost else DOT_WEAK if (getattr(d, 'weak', False) or getattr(d, 'partial', False)) else DOT_OK
        c = (int(d.x), int(d.y))
        r = max(int(8 * s), 4)
        if d.lost:
            cv2.line(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col, thick, cv2.LINE_AA)
            cv2.line(img, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), col, thick, cv2.LINE_AA)
        else:
            cv2.circle(img, c, r, col, thick, cv2.LINE_AA)
        _text(img, name, (c[0] - r * 2, c[1] - r - 6 * s), 0.5 * s, col, 1)
    for name, (dx, dy) in (derived or {}).items():       # computed, not clicked: a diamond, so it is not taken for a sticker
        c = (int(dx), int(dy))
        cv2.drawMarker(img, c, DERIVED_COLOUR, cv2.MARKER_DIAMOND, max(int(18 * s), 8), thick, cv2.LINE_AA)
        _text(img, f"{name} (computed)", (c[0] + 12 * s, c[1] - 10 * s), 0.5 * s, DERIVED_COLOUR, 1)

    rows = []                                # (text, scale, colour, thickness, advance to the next line)
    for line in header_lines:
        if line.startswith(PROMPT_PREFIX):       # the dot to click next: the one instruction the operator must not miss
            rows.append((line, 0.8 * s, DOT_OK, 2, 34 * s))
        else:
            rows.append((line, 0.7 * s, (255, 255, 255), 2, 30 * s))
    if rows:
        rows[-1] = rows[-1][:4] + (rows[-1][4] + 6 * s,)
    for a in angles:
        col = COLOURS[a.status]
        if a.status == LOST:
            s_val = "lost"
        else:
            s_val = f"{a.value:6.1f}"
            if a.reference is not None:
                s_val += f"  ({a.deviation:+.1f} vs {a.reference:.0f})"
        rows.append((f"{a.title}: {s_val}", 0.75 * s, col, thick, 32 * s))
    y = 34 * s
    _shade_behind(img, rows, 12, y)
    for line, scale, col, th, advance in rows:
        _text(img, line, (12, y), scale, col, th)
        y += advance

    if recording:
        cv2.circle(img, (w - int(28 * s), int(28 * s)), int(9 * s), (40, 40, 240), -1, cv2.LINE_AA)
        _text(img, "REC", (w - int(90 * s), int(36 * s)), 0.7 * s, (40, 40, 240), thick)
    return img
