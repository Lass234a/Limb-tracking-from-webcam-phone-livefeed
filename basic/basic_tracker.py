"""Basic limb-angle tracker: the smallest version of the idea, in one file.

What it does, and nothing more:
  1. opens a camera (or a video file)
  2. you click three dots once (by default hip, knee, ankle)
  3. each dot is followed from picture to picture
  4. the angle at the middle dot is shown on the picture, and can be recorded to a CSV file

What it leaves out on purpose (the full program in ../limbtrack has all of it): the protocol and target angles, locking a
position, the force from the load cell, the ruler scale, the strap and anchor dots, saved video, Review, privacy masks,
Auto dot colour, the checks on every number, and the extra rules that keep a dot from jumping onto a shadow or a finger.

Run:   python basic_tracker.py              (camera 0, black dots)
       python basic_tracker.py 1            (camera 1)
       python basic_tracker.py video.mp4    (a video file)
       python basic_tracker.py 0 --colour white --dots hip,knee,ankle
Keys:  r = start / stop recording to a CSV file   m = mark the dots again   space = pause   q = quit
"""
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---- The numbers that steer the tracking. Same meaning (and values) as in ../limbtrack/tracker.py, so the two can be compared ----
BLUR = 3                    # Gaussian blur (pixels) before the filter
KERNEL_FACTOR = 2.5         # filter size = this x the dot's width ...
KERNEL_MIN = 9              # ... but never less than this
MARK_DIAM = 16              # dot width assumed when you click
MARK_RADIUS = 40            # the area looked at around your click (pixels)
MARK_PEAK_BOX = 8           # the dot's strength = the strongest value within this distance of the click
MARK_BLOB_FRACTION = 0.5    # the dot is everything above this fraction of that strength
MARK_MAX_SNAP = 15          # the dot's centre must be within this distance of your click
MIN_CONTRAST = 8            # grey levels; weaker than this is just noise
THRESHOLD_FRACTION = 0.4    # a pixel counts as dot if it is above this fraction of the dot's usual contrast
REACH_FACTOR = 3.0          # how far from the expected position to look: this x the dot's width + REACH_EXTRA (at most REACH_MAX)
REACH_EXTRA = 12
REACH_MAX = 160
AREA_RANGE = (0.40, 3.5)    # accepted blob area, as multiples of the dot's usual area
ASPECT_FACTOR = 2.5         # accepted blob: longer side at most this x the shorter side + ASPECT_EXTRA
ASPECT_EXTRA = 2
LOST_RADIUS_FACTOR = 2.0    # a lost dot looks only inside a fixed circle this x its width, around where it was last seen
RESUME_MIN_QUALITY = 0.6    # a lost dot comes back only on a blob at least this fraction as contrasty as when you clicked it
VELOCITY_GAIN = 0.5         # expected movement = this x the last step
AREA_KEEP, CONTRAST_KEEP = 0.9, 0.95    # the dot's usual area and contrast follow slowly (0.9 old + 0.1 new, 0.95 old + 0.05 new)
MIN_BLOB_PIXELS = 3         # smaller specks are ignored
MIN_DIAMETER = 4.0


def odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def enhance(roi, diam, colour):
    """Black-hat (black dots) or top-hat (white dots): keeps only features smaller than the filter, so the background goes."""
    k = odd(max(diam * KERNEL_FACTOR, KERNEL_MIN))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    op = cv2.MORPH_TOPHAT if colour == "white" else cv2.MORPH_BLACKHAT
    return cv2.morphologyEx(cv2.GaussianBlur(roi, (BLUR, BLUR), 0), op, kernel)


def window(gray, cx, cy, r):
    """The square piece of the picture around (cx, cy). Returns its left, top and the piece."""
    h, w = gray.shape
    x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, w)
    y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, h)
    return x0, y0, gray[y0:y1, x0:x1]


def blobs(enh, thr):
    """Every connected patch above the threshold: (centre x, centre y, area, width, height). The centre is weighted by
    brightness, so it is accurate to a fraction of a pixel."""
    _, mask = cv2.threshold(enh, thr, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_BLOB_PIXELS:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        wts = enh[y:y + h, x:x + w][ys, xs].astype(np.float64)
        if wts.sum() <= 0:
            continue
        out.append((x + float((xs * wts).sum() / wts.sum()), y + float((ys * wts).sum() / wts.sum()), float(area), int(w), int(h)))
    return out


class Dot:
    """One dot on the body. Click it once with mark(); then call update() on every picture."""

    def __init__(self, name, colour):
        self.name, self.colour = name, colour
        self.marked = False
        self.found = False          # seen in the latest picture
        self.x = self.y = self.vx = self.vy = 0.0
        self.area, self.diam = 100.0, 12.0
        self.contrast = self.ref_contrast = 40.0
        self.lost_frames = 0
        self.lost_radius = 0.0

    def mark(self, gray, x, y):
        """Lock onto the dot nearest the click. Returns False if there is no dot there."""
        x0, y0, roi = window(gray, x, y, MARK_RADIUS)
        if roi.size == 0:
            return False
        cx, cy = int(x - x0), int(y - y0)
        enh = enhance(roi, MARK_DIAM, self.colour)
        near = enh[max(cy - MARK_PEAK_BOX, 0):cy + MARK_PEAK_BOX + 1, max(cx - MARK_PEAK_BOX, 0):cx + MARK_PEAK_BOX + 1]
        peak = float(near.max()) if near.size else 0.0
        if peak < MIN_CONTRAST:
            return False
        found = blobs(enh, MARK_BLOB_FRACTION * peak)
        if not found:
            return False
        bx, by, area, _, _ = min(found, key=lambda b: (b[0] - cx) ** 2 + (b[1] - cy) ** 2)
        if (bx - cx) ** 2 + (by - cy) ** 2 > MARK_MAX_SNAP ** 2:
            return False
        self.x, self.y = x0 + bx, y0 + by
        self.vx = self.vy = 0.0
        self.area = area
        self.diam = max(2.0 * math.sqrt(area / math.pi), MIN_DIAMETER)
        self.contrast = self.ref_contrast = peak
        self.lost_frames, self.lost_radius = 0, 0.0
        self.marked = self.found = True
        return True

    def update(self, gray):
        """Find the dot in a new picture. Returns True if it was found, False if it is lost."""
        was_lost = self.lost_frames > 0
        if was_lost:        # frozen where it was last seen, a small fixed circle, no prediction
            cx, cy, reach = self.x, self.y, self.lost_radius
        else:               # where it should be now: last position plus half the last step
            cx, cy = self.x + self.vx, self.y + self.vy
            reach = min(REACH_FACTOR * self.diam + REACH_EXTRA, REACH_MAX)
        x0, y0, roi = window(gray, cx, cy, reach + self.diam)
        best = None
        if roi.size:
            enh = enhance(roi, self.diam, self.colour)
            thr = max(THRESHOLD_FRACTION * self.contrast, MIN_CONTRAST)
            for bx, by, area, w, h in blobs(enh, thr):
                if not AREA_RANGE[0] * self.area <= area <= AREA_RANGE[1] * self.area:
                    continue                                    # wrong size
                if max(w, h) > ASPECT_FACTOR * min(w, h) + ASPECT_EXTRA:
                    continue                                    # not round (an edge, a strap)
                d = math.hypot(x0 + bx - cx, y0 + by - cy)
                if d > reach:
                    continue                                    # too far away
                iy, ix = int(round(by)), int(round(bx))
                peak = float(enh[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].max())
                if was_lost and peak < RESUME_MIN_QUALITY * self.ref_contrast:
                    continue                                    # coming back needs a convincing dot, not a shadow
                if best is None or d < best[0]:
                    best = (d, x0 + bx, y0 + by, area, peak)    # the nearest fitting blob wins
        if best is None:
            if not was_lost:
                self.lost_radius = LOST_RADIUS_FACTOR * self.diam
            self.vx = self.vy = 0.0
            self.lost_frames += 1
            self.found = False
            return False
        _, nx, ny, area, peak = best
        self.vx, self.vy = (0.0, 0.0) if was_lost else (VELOCITY_GAIN * (nx - self.x), VELOCITY_GAIN * (ny - self.y))
        self.x, self.y = nx, ny
        self.area = AREA_KEEP * self.area + (1 - AREA_KEEP) * area
        self.diam = max(2.0 * math.sqrt(self.area / math.pi), MIN_DIAMETER)
        if peak >= MIN_CONTRAST:
            self.contrast = CONTRAST_KEEP * self.contrast + (1 - CONTRAST_KEEP) * peak
        self.lost_frames = 0
        self.found = True
        return True


def interior_angle(a, b, c):
    """The angle at point b between the lines b->a and b->c, in degrees (180 = a straight line)."""
    ba, bc = np.subtract(a, b), np.subtract(c, b)
    return math.degrees(math.atan2(abs(ba[0] * bc[1] - ba[1] * bc[0]), float(np.dot(ba, bc))))


class App:
    """Everything except the window: the dots, the angle, the recording and the drawing. (Kept apart so it can be tested.)"""

    def __init__(self, names, colour, out_dir):
        self.names, self.colour, self.out_dir = list(names), colour, Path(out_dir)
        self.reset()
        self.gray = None
        self.message = ""
        self.angle = None
        self.record = None          # (file, csv writer, start time) while recording

    def reset(self):
        self.dots = [Dot(n, self.colour) for n in self.names]

    @property
    def next_to_mark(self):
        for d in self.dots:
            if not d.marked:
                return d
        return None

    def click(self, x, y):
        d = self.next_to_mark
        if d is None or self.gray is None or self.record:
            return
        self.message = "" if d.mark(self.gray, x, y) else f"No {self.colour} dot found there - click nearer the middle of the dot"

    def process(self, frame, t):
        """Track every dot in this picture, work out the angle, write the CSV row if recording."""
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.angle = None
        if self.next_to_mark is not None:
            return
        for d in self.dots:
            d.update(self.gray)
        if len(self.dots) >= 3 and all(d.found for d in self.dots[:3]):
            self.angle = interior_angle(*[(d.x, d.y) for d in self.dots[:3]])    # at the middle dot
        if self.record:
            f, w, t0 = self.record
            row = [round(t - t0, 4)]
            for d in self.dots:
                row += [round(d.x, 2), round(d.y, 2)] if d.found else ["", ""]   # blank while lost: nothing is guessed
            if len(self.dots) >= 3:
                row += [round(self.angle, 2), round(180 - self.angle, 2)] if self.angle is not None else ["", ""]
            w.writerow(row)
            f.flush()

    def toggle_recording(self, t):
        if self.record:
            self.record[0].close()
            self.message = "Recording stopped."
            self.record = None
            return
        if self.next_to_mark is not None:
            self.message = "Mark all the dots before recording."
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"trial_{datetime.now():%Y%m%d-%H%M%S}.csv"
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        head = ["t_s"] + [f"{n}_{axis}" for n in self.names for axis in ("x", "y")]
        if len(self.names) >= 3:
            head += ["angle_deg", "flexion_deg"]
        w.writerow(head)
        self.record = (f, w, t)
        self.message = f"Recording to {path.name}"

    def draw(self, frame):
        """The picture with the dots, lines and text drawn on it."""
        img = frame.copy()
        marked = [d for d in self.dots if d.marked]
        for d in marked:
            colour = (0, 220, 0) if d.found else (0, 0, 255)          # green = tracked, red = lost
            cv2.circle(img, (int(round(d.x)), int(round(d.y))), int(max(d.diam / 2 + 5, 9)), colour, 2, cv2.LINE_AA)
            put_text(img, d.name, int(d.x) + 14, int(d.y) - 10, colour, 0.5)
        if self.angle is not None:
            for a, b in zip(self.dots[:2], self.dots[1:3]):
                cv2.line(img, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (0, 200, 255), 2, cv2.LINE_AA)
        lines = []
        nxt = self.next_to_mark
        if nxt is not None:
            lines.append((f"Click the {nxt.name} dot ({self.dots.index(nxt) + 1} of {len(self.dots)})", (255, 255, 255)))
        elif len(self.dots) >= 3:
            if self.angle is not None:
                lines.append((f"{self.dots[1].name}: {self.angle:.1f} deg   (180 = straight, flexion {180 - self.angle:.1f})", (0, 255, 255)))
            else:
                lost = ", ".join(d.name for d in self.dots[:3] if not d.found)
                lines.append((f"No angle: dot lost ({lost})", (0, 0, 255)))
        if self.message:
            lines.append((self.message, (255, 255, 255)))
        for i, (text, colour) in enumerate(lines):
            put_text(img, text, 12, 28 + 30 * i, colour, 0.8)
        put_text(img, "[r] record   [m] mark again   [space] pause   [q] quit", 12, img.shape[0] - 12, (255, 255, 255), 0.55)
        if self.record:
            put_text(img, "REC", img.shape[1] - 80, 30, (0, 0, 255), 0.9)
        return img


def put_text(img, text, x, y, colour, scale):
    # A dark outline first, so the text reads on any picture. Both lines must be thickness 2 or more: at thickness 1 this OpenCV
    # spaces the letters differently, and the outline and the text no longer line up.
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def open_source(text):
    """A number is a camera, anything else is a video file (or a stream address). Returns (capture, is_file)."""
    is_camera = text.isdigit()
    cap = cv2.VideoCapture(int(text) if is_camera else text)
    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap, not is_camera


def main():
    ap = argparse.ArgumentParser(description="Basic limb-angle tracker")
    ap.add_argument("source", nargs="?", default="0", help="camera number (default 0) or a video file")
    ap.add_argument("--dots", default="trochanter,epicondyle,malleolus", help="names of the dots to click, comma separated; the angle is at the second")
    ap.add_argument("--colour", choices=("black", "white"), default="black", help="colour of the dots (default black)")
    args = ap.parse_args()

    cap, is_file = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open '{args.source}'. For a camera try another number (0, 1, 2 ...); for a file check the path.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    app = App([n.strip() for n in args.dots.split(",") if n.strip()], args.colour, Path(__file__).resolve().parent / "recordings")

    win = "Basic limb tracker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, lambda event, x, y, flags, param: app.click(x, y) if event == cv2.EVENT_LBUTTONDOWN else None)

    start, frame_no, frame, t, paused = time.perf_counter(), 0, None, 0.0, False
    while True:
        t_loop = time.perf_counter()
        if not paused:
            ok, new = cap.read()
            if not ok:
                if is_file and frame is not None:
                    paused, app.message = True, "End of video. Press q to quit."
                    continue
                raise SystemExit("The camera stopped delivering pictures." if not is_file else "Could not read the video.")
            frame = new
            t = frame_no / fps if is_file else time.perf_counter() - start     # a file's time comes from its own frame rate
            frame_no += 1
            app.process(frame, t)
        if frame is not None:
            cv2.imshow(win, app.draw(frame))
        wait = max(1, int(1000 / fps - (time.perf_counter() - t_loop) * 1000)) if is_file and not paused else 1
        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("m") and not app.record:
            app.reset()
            app.message = ""
        elif key == ord("r"):
            app.toggle_recording(t)
    if app.record:
        app.toggle_recording(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
