"""Privacy helpers: what is allowed to reach the disk.

Tracking always runs on the full, unmasked picture. A crop and black masks change only the COPIES of the frames that are
written to raw.mp4 and overlay.mp4. The CSV keeps full-picture pixel positions, so no number changes. The crop is a pure
translation: Review pads a cropped video back to full size (black outside the crop) and everything lines up again.
"""
import re

import numpy as np

PARTICIPANT_ID_PATTERN = r"^P[0-9]{2,3}$"          # P01 ... P999: a code only, never a name
PARTICIPANT_ID_HELP = "Use the anonymous code only, written P01, P02, ... (letter P and 2 or 3 digits). No names."

MIN_SIDE_PX = 32            # a crop or mask smaller than this is almost certainly a stray click
DOT_EDGE_MARGIN_PX = 30     # a marked dot closer than this to the crop edge (or inside a mask) stops the recording from starting
DOT_WARN_MARGIN_PX = 10     # a tracked dot closer than this to the crop edge (or inside a mask) while recording is flagged


def check_participant_id(text, pattern=PARTICIPANT_ID_PATTERN):
    """None if the ID is acceptable, else a plain-English reason."""
    if not re.match(pattern, text or ""):
        return f"'{text}' is not a valid participant code. " + PARTICIPANT_ID_HELP
    return None


def notes_problem(text):
    """None if the notes look harmless; else why they may hold personal data (an e-mail address, a long number such as a CPR number)."""
    t = text or ""
    if "@" in t:
        return "The notes contain an '@' (an e-mail address?)."
    if re.search(r"\d{6,}", re.sub(r"[\s-]", "", t)):
        return "The notes contain a long number (a personal ID number?)."
    return None


def snap_rect(x0, y0, x1, y1, width, height):
    """A dragged rectangle (any two corners, image pixels) as (x, y, w, h): integers, clamped to the picture, even width and
    height (the video writer silently drops an odd last row or column), at least MIN_SIDE_PX each way. None if too small."""
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(int(width), int(round(x1))), min(int(height), int(round(y1)))
    w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return None
    return (x0, y0, w, h)


def inside(rect, x, y):
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def edge_distance(rect, x, y):
    """Distance in pixels from (x, y) to the nearest edge of the rectangle; negative when the point is outside it."""
    rx, ry, rw, rh = rect
    d = min(x - rx, rx + rw - x, y - ry, ry + rh - y)
    return float(d)


def mask_distance(rect, x, y):
    """Distance from (x, y) to a mask rectangle: 0 or negative when inside, positive when outside."""
    rx, ry, rw, rh = rect
    dx = max(rx - x, 0, x - (rx + rw))
    dy = max(ry - y, 0, y - (ry + rh))
    if dx == 0 and dy == 0:
        return -min(x - rx, rx + rw - x, y - ry, ry + rh - y)
    return float(np.hypot(dx, dy))


def apply(frame, crop, masks):
    """The picture as it is to be saved: cropped to `crop` (x, y, w, h) and with `masks` blacked out. The input is never modified.
    With neither crop nor masks the same array is returned (no copy)."""
    if crop is None and not masks:
        return frame
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    out = frame[oy:oy + crop[3], ox:ox + crop[2]].copy() if crop else frame.copy()
    for mx, my, mw, mh in masks:
        x0, y0 = max(0, mx - ox), max(0, my - oy)
        x1, y1 = min(out.shape[1], mx + mw - ox), min(out.shape[0], my + mh - oy)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 0
    return out


def pad_to_full(img, crop, full_size):
    """Put a cropped picture back at its place on a black picture of the full size (Review)."""
    w, h = full_size
    if crop is None or (img.shape[1], img.shape[0]) == (w, h):
        return img
    x, y, cw, ch = crop
    canvas = np.zeros((h, w) + img.shape[2:], dtype=img.dtype)
    canvas[y:y + ch, x:x + cw] = img[:ch, :cw]
    return canvas
