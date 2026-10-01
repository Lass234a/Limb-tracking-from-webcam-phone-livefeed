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
taken for the dot. A dot that moved further than the circle while hidden stays lost
until the operator marks it again. The angle is never computed from a lost dot.
"""
from dataclasses import dataclass

import cv2
import numpy as np

MIN_CONTRAST = 8.0  # grey levels; below this a "dot" is indistinguishable from noise
WEAK_BELOW = 0.6    # dot contrast relative to when it was marked; below this the dot is "weak"
LOST_RADIUS_FACTOR = 2.0    # search circle for a lost dot = this x the dot's diameter when it was lost (fixed, never grows)
RESUME_MIN_QUALITY = 0.6    # a lost dot resumes only on a blob at least this fraction as contrasty as when marked
RESUME_AREA_RANGE = (0.5, 2.0)   # ...and with an area within this range of the learned dot area (normal tracking: 0.3-3.5)
RESUME_MAX_ASPECT = 1.6          # ...and roughly round: longer side at most this x the shorter side + 1 (normal tracking: 2.5x + 2)


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

    @property
    def weak(self):
        return not self.lost and self.quality < WEAK_BELOW


class DotTracker:
    def __init__(self, name, kind="auto"):
        self.name = name
        self.kind = kind  # "white", "black" or "auto"
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.diam = 12.0
        self.area = 100.0
        self.contrast = 40.0          # slowly adapting estimate, used for the detection threshold
        self.ref_contrast = 40.0      # contrast when the operator marked the dot, used for the quality read-out
        self.lost_frames = 0
        self.lost_radius = 0.0        # fixed search circle used while lost (set on the first lost frame)
        self._before = None           # state before the latest update (so a rejected update can be undone)
        self.initialised = False

    # ------------------------------------------------------------------ helpers
    def _enhance(self, roi, diam):
        """Top-hat / black-hat: keeps only features smaller than the kernel."""
        k = _odd(max(diam * 2.5, 9))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        op = cv2.MORPH_TOPHAT if self.kind == "white" else cv2.MORPH_BLACKHAT
        return cv2.morphologyEx(cv2.GaussianBlur(roi, (3, 3), 0), op, kernel)

    @staticmethod
    def _window(gray, cx, cy, r):
        h, w = gray.shape
        x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, w)
        y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, h)
        return x0, y0, gray[y0:y1, x0:x1]

    def _blobs(self, enh, thr):
        """Connected blobs above thr -> list of (cx, cy, area, w, h) in window coordinates."""
        _, mask = cv2.threshold(enh, thr, 255, cv2.THRESH_BINARY)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 3:
                continue
            ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
            wts = enh[y:y + h, x:x + w][ys, xs].astype(np.float64)
            if wts.sum() <= 0:
                continue
            cx = x + float((xs * wts).sum() / wts.sum())
            cy = y + float((ys * wts).sum() / wts.sum())
            out.append((cx, cy, float(area), int(w), int(h)))
        return out

    # --------------------------------------------------------------------- init
    def init_at(self, gray, x, y):
        """Lock onto the dot nearest the click. Returns True on success."""
        gray = gray.astype(np.float32) if gray.dtype != np.uint8 else gray
        r = 40
        x0, y0, roi = self._window(gray, x, y, r)
        if roi.size == 0:
            return False
        if self.kind == "auto":
            cx, cy = int(x - x0), int(y - y0)
            centre = float(np.mean(roi[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3]))
            self.kind = "white" if centre >= float(np.median(roi)) else "black"
        enh = self._enhance(roi, 16)
        cx, cy = int(x - x0), int(y - y0)
        near = enh[max(cy - 8, 0):cy + 9, max(cx - 8, 0):cx + 9]
        peak = float(near.max()) if near.size else 0.0
        if peak < MIN_CONTRAST:
            return False
        blobs = self._blobs(enh, 0.5 * peak)
        if not blobs:
            return False
        bx, by, area, _, _ = min(blobs, key=lambda b: (b[0] - cx) ** 2 + (b[1] - cy) ** 2)
        if (bx - cx) ** 2 + (by - cy) ** 2 > 15 ** 2:
            return False
        self.x, self.y = x0 + bx, y0 + by
        self.vx = self.vy = 0.0
        self.area = area
        self.diam = max(2.0 * (area / np.pi) ** 0.5, 4.0)
        self.contrast = self.ref_contrast = peak
        self.lost_frames = 0
        self.lost_radius = 0.0
        self.initialised = True
        return True

    # -------------------------------------------------------------------- track
    def _peak(self, enh, cx, cy):
        iy, ix = int(round(cy)), int(round(cx))
        return float(enh[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].max()) if enh.size else self.contrast

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
            reach = min(3.0 * self.diam + 12.0, 160.0)
        x0, y0, roi = self._window(gray, cx0, cy0, reach + self.diam)
        best = None
        if roi.size:
            enh = self._enhance(roi, self.diam)
            thr = max(0.4 * self.contrast, MIN_CONTRAST)
            for cx, cy, area, w, h in self._blobs(enh, thr):
                if not (0.3 * self.area <= area <= 3.5 * self.area):
                    continue
                if max(w, h) > 2.5 * min(w, h) + 2:
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
                    if max(w, h) > RESUME_MAX_ASPECT * min(w, h) + 1:
                        continue           # not round (sliver, edge of an object)
                if best is None or d < best[0]:
                    best = (d, x0 + cx, y0 + cy, area, peak)
        if best is None:
            if not was_lost:               # first miss: freeze here and fix the search circle
                self.lost_radius = LOST_RADIUS_FACTOR * self.diam
            self.vx = self.vy = 0.0
            self.lost_frames += 1
            return DotState(self.name, self.x, self.y, True, self.lost_frames, 0.0)
        _, nx, ny, area, peak = best
        if was_lost:                       # resuming after a jump: no velocity to carry over
            self.vx = self.vy = 0.0
        else:
            self.vx = 0.5 * (nx - self.x)
            self.vy = 0.5 * (ny - self.y)
        self.x, self.y = nx, ny
        self.area = 0.9 * self.area + 0.1 * area
        self.diam = max(2.0 * (self.area / np.pi) ** 0.5, 4.0)
        if peak >= MIN_CONTRAST:
            self.contrast = 0.95 * self.contrast + 0.05 * peak
        self.lost_frames = 0
        return DotState(self.name, self.x, self.y, False, 0, min(peak / self.ref_contrast, 1.0))


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

    def init_dot(self, name, gray, x, y):
        return self.trackers[name].init_at(gray, x, y)

    def is_ready(self):
        return all(t.initialised for n, t in self.trackers.items() if n in self.enabled)

    def update(self, gray):
        active = {n: t for n, t in self.trackers.items() if t.initialised and n in self.enabled}
        held = {n: (t.x, t.y, 0.8 * t.diam) for n, t in active.items() if t.lost_frames == 0}
        states = {}
        for n, t in active.items():
            claimed = [v for k, v in held.items() if k != n]
            states[n] = t.update(gray, claimed)
        # Two trackers must never sit on the same dot: the later one is declared lost.
        names = list(states)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, sb = states[a], states[b]
                if sa.lost or sb.lost:
                    continue
                lim = 0.5 * min(self.trackers[a].diam, self.trackers[b].diam)
                if (sa.x - sb.x) ** 2 + (sa.y - sb.y) ** 2 < lim ** 2:
                    t = self.trackers[b]
                    t.undo_update()        # back to its previous position, lost
                    states[b] = DotState(b, t.x, t.y, True, t.lost_frames, 0.0)
        return states
