"""Follows a single white or black dot from frame to frame.

Approach: the operator clicks the dot once. From then on we only search a small
window around where the dot should be. Inside the window a top-hat (white dot)
or black-hat (black dot) filter removes the background, blobs of the right size
are picked out, and the one closest to the predicted position wins. The position
is the intensity-weighted centre of the blob (sub-pixel).

If nothing suitable is found the dot is flagged as LOST. The last known position
is kept for drawing, but the angle is never computed from a lost dot. The search
window grows the longer the dot stays lost, so it is picked up again when it
reappears.
"""
from dataclasses import dataclass

import cv2
import numpy as np

MIN_CONTRAST = 8.0  # grey levels; below this a "dot" is indistinguishable from noise


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


class DotTracker:
    def __init__(self, name, kind="auto"):
        self.name = name
        self.kind = kind  # "white", "black" or "auto"
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.diam = 12.0
        self.area = 100.0
        self.contrast = 40.0
        self.lost_frames = 0
        self.initialised = False

    # ------------------------------------------------------------------ helpers
    def _enhance(self, roi, diam):
        """Top-hat / black-hat: keeps only features smaller than the kernel."""
        k = _odd(max(diam * 2.5, 9))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
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
        self.contrast = peak
        self.lost_frames = 0
        self.initialised = True
        return True

    # -------------------------------------------------------------------- track
    def update(self, gray, claimed=()):
        """Find the dot in a new frame. Returns a DotState.

        `claimed`: (x, y, radius) of dots that other trackers currently hold. Blobs inside those
        circles are ignored, so a hidden dot can never jump onto its neighbour.
        """
        if not self.initialised:
            return DotState(self.name, self.x, self.y, True, 0)
        px, py = self.x + self.vx, self.y + self.vy
        reach = min(3.0 * self.diam + 12.0 + 12.0 * self.lost_frames, 160.0)
        x0, y0, roi = self._window(gray, px, py, reach + self.diam)
        found = None
        if roi.size:
            enh = self._enhance(roi, self.diam)
            thr = max(0.4 * self.contrast, MIN_CONTRAST)
            best = None
            for cx, cy, area, w, h in self._blobs(enh, thr):
                if not (0.3 * self.area <= area <= 3.5 * self.area):
                    continue
                if max(w, h) > 2.5 * min(w, h) + 2:
                    continue
                if any((x0 + cx - qx) ** 2 + (y0 + cy - qy) ** 2 < qr ** 2 for qx, qy, qr in claimed):
                    continue
                d = ((x0 + cx - px) ** 2 + (y0 + cy - py) ** 2) ** 0.5
                if d <= reach and (best is None or d < best[0]):
                    best = (d, x0 + cx, y0 + cy, area, enh, cx, cy)
            if best is not None:
                found = best
        if found is None:
            self.lost_frames += 1
            self.vx *= 0.5
            self.vy *= 0.5
            return DotState(self.name, self.x, self.y, True, self.lost_frames)
        _, nx, ny, area, enh, cx, cy = found
        self.vx = 0.5 * (nx - self.x)
        self.vy = 0.5 * (ny - self.y)
        self.x, self.y = nx, ny
        self.area = 0.9 * self.area + 0.1 * area
        self.diam = max(2.0 * (self.area / np.pi) ** 0.5, 4.0)
        peak = float(enh[int(round(cy)), int(round(cx))]) if enh.size else self.contrast
        if peak >= MIN_CONTRAST:
            self.contrast = 0.95 * self.contrast + 0.05 * peak
        self.lost_frames = 0
        return DotState(self.name, self.x, self.y, False, 0)


def to_gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class TrackerSet:
    """All dots for the current test."""

    def __init__(self, names, kind="auto"):
        self.trackers = {n: DotTracker(n, kind) for n in names}

    def init_dot(self, name, gray, x, y):
        return self.trackers[name].init_at(gray, x, y)

    def is_ready(self):
        return all(t.initialised for t in self.trackers.values())

    def update(self, gray):
        active = {n: t for n, t in self.trackers.items() if t.initialised}
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
                    t.lost_frames += 1
                    states[b] = DotState(b, t.x, t.y, True, t.lost_frames)
        return states
