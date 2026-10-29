"""Follows a single white or black dot from frame to frame.

Approach: the operator clicks the dot once. From then on we only search a small
window around where the dot should be. Inside the window a top-hat (white dot)
or black-hat (black dot) filter removes the background, blobs of the right size
are picked out, and the one closest to the predicted position wins. The position
is the intensity-weighted centre of the blob (sub-pixel).

If nothing suitable is found the dot is flagged as LOST. It then freezes at its last
good position (no velocity prediction) and only a small FIXED circle around that
position is searched. Tracking resumes only when a blob inside the circle is also
bright enough (see RESUME_MIN_QUALITY), so a shadow or a different object cannot be
taken for the dot.

A dot that touches a shadow, a finger or a strap merges with it into one big or elongated blob. When no blob fits, or the
best one is too big or too long (MERGED_*), the search is repeated at higher fractions of the dot's contrast, which cut a
fainter shadow or finger away from the dark (or bright) dot. The size the tracker learns for the dot is never taken from such
a merged blob. A dot that moved further than the circle while hidden stays lost
until the operator marks it again. The angle is never computed from a lost dot.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

MIN_CONTRAST = 8.0  # grey levels; below this a "dot" is indistinguishable from noise
WEAK_BELOW = 0.6    # dot contrast relative to when it was marked; below this the dot is "weak"
LOST_RADIUS_FACTOR = 2.0    # search circle for a lost dot = this x the dot's diameter when it was lost (fixed, never grows)
RESUME_MIN_QUALITY = 0.6    # a lost dot resumes only on a blob at least this fraction as contrasty as when marked
RESUME_AREA_RANGE = (0.5, 2.0)   # ...and with an area within this range of the learned dot area (normal tracking: 0.3-3.5)
RESUME_MAX_ASPECT = 1.6          # ...and roughly round: longer side at most this x the shorter side + 1 (normal tracking: 2.5x + 2)
RESUME_ASPECT_EXTRA_PX = 1

# Partly covered dot (hand, strap, clothing, image edge): the centre of what is visible is not the centre of the dot
PARTIAL_AREA_BELOW = 0.70        # visible area below this fraction of the dot's area ...
PARTIAL_AREA_FLOOR = 0.80        # ... where that area is the running area, but at least this fraction of the area when marked
PARTIAL_ELONGATION = 2.0         # or the blob is this much longer than wide (a dot cut off on one side)
PARTIAL_BORDER_FACTOR = 0.5      # or its centre is within this x diameter of the image edge

# Image preparation
BLUR_KERNEL = 3                  # Gaussian blur, pixels, before the top-hat / black-hat
KERNEL_DIAM_FACTOR = 2.5         # top-hat / black-hat kernel side = this x dot diameter ...
KERNEL_MIN_PX = 9                # ... but never less than this (then rounded up to an odd number)
MARKING_KERNEL_DIAM = 16         # diameter assumed when the operator clicks (kernel 41 px)
MARKING_WINDOW_RADIUS_PX = 40    # area looked at around the click
MARKING_PEAK_BOX_PX = 8          # the dot's brightness is the strongest value within this distance of the click ...
MARKING_BLOB_FRACTION = 0.5      # ... and the blob is everything above this fraction of it
MARKING_MAX_SNAP_PX = 15         # the blob centre must be within this distance of the click
MIN_BLOB_PIXELS = 3              # smaller specks are ignored
MARKING_KIND_BOX_PX = 4          # auto colour: white or black is decided by the response this close to the click ...
MARKING_KIND_TIE = 0.10          # ... unless white and black are within this fraction of each other there

# Following the dot
THRESHOLD_FRACTION = 0.4         # detection threshold = this x the dot's running contrast (and at least MIN_CONTRAST)
REACH_DIAM_FACTOR = 3.0          # search reach = this x diameter + REACH_EXTRA_PX, at most REACH_MAX_PX
REACH_EXTRA_PX = 12.0
REACH_MAX_PX = 160.0
VELOCITY_GAIN = 0.5              # predicted velocity = this x the latest step
AREA_RANGE = (0.40, 3.5)         # accepted blob area, as multiples of the running dot area (was 0.3 until 2026-10-09: fragments of a
                                 # dot hidden by a hand were accepted as the dot)
ASPECT_FACTOR = 2.5              # accepted blob: longer side at most this x the shorter side + ASPECT_EXTRA_PX
ASPECT_EXTRA_PX = 2
AREA_KEEP, AREA_NEW = 0.9, 0.1           # running dot area: 0.9 old + 0.1 new
CONTRAST_KEEP, CONTRAST_NEW = 0.95, 0.05  # running contrast: 0.95 old + 0.05 new
MIN_DIAMETER_PX = 4.0
NEIGHBOUR_EXCLUSION_FACTOR = 0.8  # blobs within this x diameter of another tracked dot are ignored
DUPLICATE_FACTOR = 0.5            # two dots closer than this x the smaller diameter: the later one is declared lost

# A dot touching a hand, a strap or a shadow merges with it into one big or elongated blob (found on the board recordings 2026-10-09)
MERGED_AREA_FACTOR = 1.5          # a blob larger than this x the area when marked is probably the dot plus something attached ...
MERGED_ASPECT = 1.6               # ... and so is one longer than this x its width + 1 pixel
SEPARATE_FRACTIONS = (0.6, 0.75, 0.9)   # ... then it is searched again at these (higher) fractions of the dot's contrast, which cut
                                         # a fainter shadow or finger away from the dark dot


def settings():
    """Every number that steers dot detection and tracking, for meta.json. Keep in step with the constants above."""
    return {
        "image_preparation": {
            "blur_kernel_px": BLUR_KERNEL, "kernel_diam_factor": KERNEL_DIAM_FACTOR, "kernel_min_px": KERNEL_MIN_PX,
            "marking_kernel_diam_px": MARKING_KERNEL_DIAM, "marking_window_radius_px": MARKING_WINDOW_RADIUS_PX,
            "marking_peak_box_px": MARKING_PEAK_BOX_PX, "marking_blob_fraction": MARKING_BLOB_FRACTION,
            "marking_max_snap_px": MARKING_MAX_SNAP_PX, "min_blob_pixels": MIN_BLOB_PIXELS,
            "marking_kind_box_px": MARKING_KIND_BOX_PX, "marking_kind_tie": MARKING_KIND_TIE,
        },
        "tracking": {
            "min_contrast": MIN_CONTRAST, "threshold_fraction": THRESHOLD_FRACTION,
            "reach_diam_factor": REACH_DIAM_FACTOR, "reach_extra_px": REACH_EXTRA_PX, "reach_max_px": REACH_MAX_PX,
            "velocity_gain": VELOCITY_GAIN, "area_range": list(AREA_RANGE),
            "aspect_factor": ASPECT_FACTOR, "aspect_extra_px": ASPECT_EXTRA_PX,
            "area_average": {"old": AREA_KEEP, "new": AREA_NEW}, "contrast_average": {"old": CONTRAST_KEEP, "new": CONTRAST_NEW},
            "min_diameter_px": MIN_DIAMETER_PX, "neighbour_exclusion_factor": NEIGHBOUR_EXCLUSION_FACTOR,
            "duplicate_factor": DUPLICATE_FACTOR, "weak_below_quality": WEAK_BELOW,
            "partial_area_below": PARTIAL_AREA_BELOW, "partial_area_floor": PARTIAL_AREA_FLOOR, "partial_elongation": PARTIAL_ELONGATION,
            "partial_border_factor": PARTIAL_BORDER_FACTOR,
            "merged_area_factor": MERGED_AREA_FACTOR, "merged_aspect": MERGED_ASPECT, "separate_fractions": list(SEPARATE_FRACTIONS),
            "neighbour_exclusion_centre": "predicted position, widened by the dot's step", "duplicate_resolution": "prediction_residual",
        },
        "lost_dot": {
            "search_radius_diam_factor": LOST_RADIUS_FACTOR, "resume_min_quality": RESUME_MIN_QUALITY,
            "resume_area_range": list(RESUME_AREA_RANGE), "resume_max_aspect": RESUME_MAX_ASPECT,
            "resume_aspect_extra_px": RESUME_ASPECT_EXTRA_PX,
        },
    }


def _odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


@dataclass
class DotState:
    name: str
    x: float
    y: float
    lost: bool
    lost_frames: int = 0
    quality: float = 1.0   # contrast now / contrast when marked (0 when lost, capped at 1)
    partial: bool = False  # the dot looks partly covered or cut off: its centre may be shifted (not lost, not judged differently)

    @property
    def weak(self):
        return not self.lost and self.quality < WEAK_BELOW


class DotTracker:
    def __init__(self, name, kind="auto"):
        self.name = name
        self.kind = kind  # "white" or "black"; "auto" until a click has succeeded
        self.auto = kind == "auto"   # decide the colour again on every click, never from a click that failed
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.diam = 12.0
        self.marked_area = None
        self.marked_diam = None       # dot diameter (pixels) learned when the operator marked it; used by the dot-size check
        self.area = 100.0
        self.contrast = 40.0          # slowly adapting estimate, used for the detection threshold
        self.ref_contrast = 40.0      # contrast when the operator marked the dot, used for the quality read-out
        self.lost_frames = 0
        self.lost_radius = 0.0        # fixed search circle used while lost (set on the first lost frame)
        self._before = None           # state before the latest update (so a rejected update can be undone)
        self.initialised = False

    # ------------------------------------------------------------------ helpers
    def _enhance(self, roi, diam, kind=None):
        """Top-hat / black-hat: keeps only features smaller than the kernel."""
        k = _odd(max(diam * KERNEL_DIAM_FACTOR, KERNEL_MIN_PX))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        op = cv2.MORPH_TOPHAT if (kind or self.kind) == "white" else cv2.MORPH_BLACKHAT
        return cv2.morphologyEx(cv2.GaussianBlur(roi, (BLUR_KERNEL, BLUR_KERNEL), 0), op, kernel)

    @staticmethod
    def _window(gray, cx, cy, r):
        h, w = gray.shape
        x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, w)
        y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, h)
        return x0, y0, gray[y0:y1, x0:x1]

    def _blobs(self, enh, thr):
        """Connected blobs above thr -> list of (cx, cy, area, w, h) in window coordinates (intensity-weighted centres).
        All blobs are measured at once (the same numbers as measuring them one by one, but fast when a busy scene has many)."""
        _, mask = cv2.threshold(enh, thr, 255, cv2.THRESH_BINARY)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return []
        ys, xs = np.nonzero(mask)                   # only the pixels that belong to a blob (most of the picture does not)
        lab = labels[ys, xs]
        w = enh[ys, xs].astype(np.float64)
        sw = np.bincount(lab, weights=w, minlength=n)
        sx = np.bincount(lab, weights=w * xs, minlength=n)
        sy = np.bincount(lab, weights=w * ys, minlength=n)
        area = stats[:, cv2.CC_STAT_AREA]
        keep = np.nonzero((area >= MIN_BLOB_PIXELS) & (sw > 0))[0]
        keep = keep[keep > 0]                       # label 0 is the background
        return list(zip((sx[keep] / sw[keep]).tolist(), (sy[keep] / sw[keep]).tolist(), area[keep].astype(np.float64).tolist(),
                        stats[keep, cv2.CC_STAT_WIDTH].tolist(), stats[keep, cv2.CC_STAT_HEIGHT].tolist()))

    # --------------------------------------------------------------------- init
    def init_at(self, gray, x, y):
        """Lock onto the dot nearest the click. Returns True on success."""
        gray = gray.astype(np.float32) if gray.dtype != np.uint8 else gray
        r = MARKING_WINDOW_RADIUS_PX
        x0, y0, roi = self._window(gray, x, y, r)
        if roi.size == 0:
            return False
        cx, cy = int(x - x0), int(y - y0)
        b = MARKING_PEAK_BOX_PX

        def strongest(enh):
            near = enh[max(cy - b, 0):cy + b + 1, max(cx - b, 0):cx + b + 1]
            return float(near.max()) if near.size else 0.0

        if self.auto:
            # Auto: whichever of white-dot / black-dot filtering responds more strongly right at the click (within
            # MARKING_KIND_BOX_PX). When the two are within MARKING_KIND_TIE of each other there, or neither reaches
            # MIN_CONTRAST, whichever responds more strongly within the marking box (a tie goes to white).
            # Decided anew on every click and kept only if the marking succeeds.
            enh_w, enh_b = self._enhance(roi, MARKING_KERNEL_DIAM, "white"), self._enhance(roi, MARKING_KERNEL_DIAM, "black")
            bk = MARKING_KIND_BOX_PX

            def at_click(e):
                near = e[max(cy - bk, 0):cy + bk + 1, max(cx - bk, 0):cx + bk + 1]
                return float(near.max()) if near.size else 0.0

            # Decided on the dot itself: a bright skin highlight or a dark edge a few pixels away must not decide the colour.
            wk, bk_ = at_click(enh_w), at_click(enh_b)
            if max(wk, bk_) < MIN_CONTRAST or abs(wk - bk_) <= MARKING_KIND_TIE * max(wk, bk_):
                wk, bk_ = strongest(enh_w), strongest(enh_b)       # no clear winner on the dot: the old rule
            kind = "white" if wk >= bk_ else "black"
            enh = enh_w if kind == "white" else enh_b
        else:
            kind = self.kind
            enh = self._enhance(roi, MARKING_KERNEL_DIAM, kind)
        peak = strongest(enh)
        if peak < MIN_CONTRAST:
            return False
        blobs = self._blobs(enh, MARKING_BLOB_FRACTION * peak)
        if not blobs:
            return False
        bx, by, area, _, _ = min(blobs, key=lambda b: (b[0] - cx) ** 2 + (b[1] - cy) ** 2)
        if (bx - cx) ** 2 + (by - cy) ** 2 > MARKING_MAX_SNAP_PX ** 2:
            return False
        self.x, self.y = x0 + bx, y0 + by
        self.vx = self.vy = 0.0
        self.area = area
        self.diam = max(2.0 * (area / np.pi) ** 0.5, MIN_DIAMETER_PX)
        self.marked_diam = self.diam
        self.marked_area = area
        self.contrast = self.ref_contrast = peak
        self.lost_frames = 0
        self.lost_radius = 0.0
        self.kind = kind            # only now: a failed click leaves the dot exactly as it was
        self.initialised = True
        return True

    # -------------------------------------------------------------------- track
    def _peak(self, enh, cx, cy):
        iy, ix = int(round(cy)), int(round(cx))
        return float(enh[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].max()) if enh.size else self.contrast

    def _merged(self, blob):
        """Is this blob (d, x, y, area, peak, w, h) too big or too elongated to be the dot alone?"""
        _, _, _, area, _, w, h = blob
        return (self.marked_area is not None and area > MERGED_AREA_FACTOR * self.marked_area) or max(w, h) > MERGED_ASPECT * min(w, h) + 1

    def hold(self):
        """Freeze the dot where it is, exactly as if it had just been lost (used after a pause for a break).

        From now on it searches only the small fixed circle round this position and takes a dot back only if it is round, the
        right size and nearly as contrasty as when it was marked. Nothing about how a dot is followed changes."""
        if not self.initialised:
            return
        if self.lost_frames == 0:
            self.lost_radius = LOST_RADIUS_FACTOR * self.diam
            self.lost_frames = 1
        self.vx = self.vy = 0.0

    def undo_update(self):
        """Take back the latest update and declare the dot lost at its previous position."""
        if self._before is not None:
            self.x, self.y, self.area, self.diam, self.contrast = self._before
        self.vx = self.vy = 0.0
        self.lost_frames += 1
        self.lost_radius = self.lost_radius or LOST_RADIUS_FACTOR * self.diam

    def update(self, gray, claimed=()):
        """Find the dot in a new frame. Returns a DotState.

        `claimed`: (x, y, radius) of dots that other trackers currently hold. Blobs inside those
        circles are ignored, so a hidden dot can never jump onto its neighbour.
        """
        if not self.initialised:
            return DotState(self.name, self.x, self.y, True, 0)
        self._before = (self.x, self.y, self.area, self.diam, self.contrast)
        was_lost = self.lost_frames > 0
        if was_lost:                       # frozen position, small fixed circle, no prediction
            cx0, cy0, reach = self.x, self.y, self.lost_radius
        else:
            cx0, cy0 = self.x + self.vx, self.y + self.vy
            reach = min(REACH_DIAM_FACTOR * self.diam + REACH_EXTRA_PX, REACH_MAX_PX)
        x0, y0, roi = self._window(gray, cx0, cy0, reach + self.diam)
        best = None

        def pick(enh, blobs):
            found = None
            for cx, cy, area, w, h in blobs:
                if not (AREA_RANGE[0] * self.area <= area <= AREA_RANGE[1] * self.area):
                    continue
                if max(w, h) > ASPECT_FACTOR * min(w, h) + ASPECT_EXTRA_PX:
                    continue
                if any((x0 + cx - qx) ** 2 + (y0 + cy - qy) ** 2 < qr ** 2 for qx, qy, qr in claimed):
                    continue
                d = ((x0 + cx - cx0) ** 2 + (y0 + cy - cy0) ** 2) ** 0.5
                if d > reach:
                    continue
                peak = self._peak(enh, cx, cy)
                if was_lost:               # resuming needs a convincing dot, not just something dark/bright
                    if peak < RESUME_MIN_QUALITY * self.ref_contrast:
                        continue           # too faint (shadow, edge)
                    if not (RESUME_AREA_RANGE[0] * self.area <= area <= RESUME_AREA_RANGE[1] * self.area):
                        continue           # wrong size
                    if max(w, h) > RESUME_MAX_ASPECT * min(w, h) + RESUME_ASPECT_EXTRA_PX:
                        continue           # not round (sliver, edge of an object)
                if found is None or d < found[0]:
                    found = (d, x0 + cx, y0 + cy, area, peak, w, h)
            return found

        if roi.size:
            enh = self._enhance(roi, self.diam)
            thr = max(THRESHOLD_FRACTION * self.contrast, MIN_CONTRAST)
            base_blobs = self._blobs(enh, thr)
            best = pick(enh, base_blobs)
            if best is None or self._merged(best):
                # Nothing fitting, or a blob that is too big or too long: the dot may have merged with a shadow, a finger or a
                # strap.
                # Search again with the threshold raised: the dot is darker (brighter) than what is attached to it.
                # A blob at a higher threshold always lies inside a blob at the base threshold, and has to be big enough to be
                # the dot: if no base blob near the dot is that big, the extra searches cannot find anything (saves most of
                # their cost in a busy picture).
                floor = AREA_RANGE[0] * self.area
                could_hold = any(a >= floor and (x0 + cx - cx0) ** 2 + (y0 + cy - cy0) ** 2 <= (reach + max(w, h)) ** 2
                                 for cx, cy, a, w, h in base_blobs)
                peak_here = float(enh.max())
                for frac in SEPARATE_FRACTIONS:
                    alt_thr = max(frac * self.contrast, thr)
                    if not could_hold or peak_here <= alt_thr:
                        break                  # nothing is that bright (or dark) here: a higher threshold cannot find anything
                    alt = pick(enh, self._blobs(enh, alt_thr))
                    if alt is not None and not self._merged(alt):
                        best = alt
                        break
        if best is None:
            if not was_lost:               # first miss: freeze here and fix the search circle
                self.lost_radius = LOST_RADIUS_FACTOR * self.diam
            self.vx = self.vy = 0.0
            self.lost_frames += 1
            return DotState(self.name, self.x, self.y, True, self.lost_frames, 0.0)
        _, nx, ny, area, peak, bw, bh = best
        ih, iw = gray.shape[:2]
        margin = PARTIAL_BORDER_FACTOR * self.diam
        baseline = max(self.area, PARTIAL_AREA_FLOOR * (self.marked_area or 0.0))    # a covered dot must not teach us its small area
        partial = (area < PARTIAL_AREA_BELOW * baseline or max(bw, bh) > PARTIAL_ELONGATION * min(bw, bh)
                   or nx < margin or ny < margin or nx > iw - 1 - margin or ny > ih - 1 - margin)
        if was_lost:                       # resuming after a jump: no velocity to carry over
            self.vx = self.vy = 0.0
        else:
            self.vx = VELOCITY_GAIN * (nx - self.x)
            self.vy = VELOCITY_GAIN * (ny - self.y)
        self.x, self.y = nx, ny
        if self.marked_area is None or area <= MERGED_AREA_FACTOR * self.marked_area:      # never learn the size of dot-plus-shadow
            self.area = AREA_KEEP * self.area + AREA_NEW * area
        self.diam = max(2.0 * (self.area / np.pi) ** 0.5, MIN_DIAMETER_PX)
        if peak >= MIN_CONTRAST:
            self.contrast = CONTRAST_KEEP * self.contrast + CONTRAST_NEW * peak
        self.lost_frames = 0
        return DotState(self.name, self.x, self.y, False, 0, min(peak / self.ref_contrast, 1.0), partial and not was_lost)


def to_gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class TrackerSet:
    """All dots for the current test."""

    def __init__(self, names, kind="auto"):
        self.kind = kind
        self.trackers = {n: DotTracker(n, kind) for n in names}
        self.enabled = set(names)     # dots switched off by the operator are neither tracked nor required

    def set_enabled(self, name, on):
        """Switch a dot on or off. Switching on gives a fresh tracker: the dot has to be marked again."""
        if on:
            self.trackers[name] = DotTracker(name, self.kind)
            self.enabled.add(name)
        else:
            self.enabled.discard(name)

    def clear(self, name):
        """Forget one dot: a fresh tracker, so it has to be marked again (it stays switched on)."""
        self.trackers[name] = DotTracker(name, self.kind)

    def init_dot(self, name, gray, x, y):
        return self.trackers[name].init_at(gray, x, y)

    def is_ready(self):
        return all(t.initialised for n, t in self.trackers.items() if n in self.enabled)

    def hold_all(self):
        """Freeze every marked dot where it is (see DotTracker.hold)."""
        for n, t in self.trackers.items():
            if n in self.enabled:
                t.hold()

    def update(self, gray):
        active = {n: t for n, t in self.trackers.items() if t.initialised and n in self.enabled}
        # Where each dot is expected this frame (a lost dot: where it froze). Taken before any tracker moves.
        expected = {n: ((t.x + t.vx, t.y + t.vy) if t.lost_frames == 0 else (t.x, t.y)) for n, t in active.items()}
        # A neighbour that is moving must be protected where it WILL be, with a margin for how far it moves in a frame;
        # otherwise a covered dot's search can take a fast neighbour (audit TRK-01).
        held = {n: (*expected[n], NEIGHBOUR_EXCLUSION_FACTOR * t.diam + math.hypot(t.vx, t.vy))
                for n, t in active.items() if t.lost_frames == 0}
        states = {}
        for n, t in active.items():
            claimed = [v for k, v in held.items() if k != n]
            states[n] = t.update(gray, claimed)
        # Two trackers must never sit on the same dot. The one that fits its own expectation worse gives it up (on a tie the
        # later one), is put back where it was and declared lost.
        names = list(states)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, sb = states[a], states[b]
                if sa.lost or sb.lost:
                    continue
                lim = DUPLICATE_FACTOR * min(self.trackers[a].diam, self.trackers[b].diam)
                if (sa.x - sb.x) ** 2 + (sa.y - sb.y) ** 2 < lim ** 2:
                    resid_a = math.hypot(sa.x - expected[a][0], sa.y - expected[a][1])
                    resid_b = math.hypot(sb.x - expected[b][0], sb.y - expected[b][1])
                    loser = a if resid_a > resid_b else b
                    t = self.trackers[loser]
                    t.undo_update()        # back to its previous position, lost
                    states[loser] = DotState(loser, t.x, t.y, True, t.lost_frames, 0.0)
        return states
