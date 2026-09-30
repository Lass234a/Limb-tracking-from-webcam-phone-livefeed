"""Draws dots, limb lines and angle read-outs onto a video frame."""
import math

import cv2

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


def draw(frame, dots, angles, header_lines=(), recording=False):
    """Return a copy of `frame` with the overlay. `dots`: name -> object with x, y, lost."""
    img = frame.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = h / 720.0
    thick = max(1, int(round(2 * s)))

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
