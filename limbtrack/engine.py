"""Glue: tracks dots, computes angles, judges them against the set position, records.

No GUI code here, so the whole pipeline can be tested with a video file.

How "deviation" is judged
  * primary angle   : compared with the TARGET typed by the operator (+/- tolerance).
  * neighbour angles: no judgement until the operator presses "Lock position".
                      At that moment their current values become the reference,
                      and from then on any drift beyond the neighbour tolerance turns red.
"""
import threading
from collections import deque

from dataclasses import asdict

from . import overlay
from .angles import measure
from .filters import LEVELS, make_filter
from .recorder import TrialRecorder
from .results import LOST, OK, OUT, FrameResult
from .tracker import DotState, TrackerSet, to_gray

SOFTWARE_VERSION = "0.2.0-dev"


class Engine:
    def __init__(self, tests, recordings_dir="recordings"):
        self.tests = tests
        self.recordings_dir = recordings_dir
        self.lock = threading.RLock()
        self.test = None
        self.trackers = None
        self.kind = "auto"
        self.target = None
        self.tolerance = 3.0
        self.neighbour_tolerance = 5.0
        self.locked = False
        self.smoothing = "light"     # 'off', 'light' or 'medium' (see filters.py)
        self._filters = {}
        self.facing_right = True     # which way the participant faces in the image (matters for signed angles)
        self._refs = {}
        self._last_gray = None
        self._last_states = {}
        self._last_angles = []
        self._times = deque(maxlen=30)
        self.frame_index = 0
        self.rec = None
        self.rec_info = {}
        self.camera_desc = ""
        self.set_test(next(iter(tests)))

    # ----------------------------------------------------------------- set-up
    def set_test(self, key):
        with self.lock:
            self.test = self.tests[key]
            self.tolerance = self.test.tolerance_deg
            self.neighbour_tolerance = self.test.neighbour_tolerance_deg
            self.target = self.test.targets[0] if self.test.targets else None
            self.reset_dots()

    def reset_dots(self):
        with self.lock:
            self.trackers = TrackerSet(self.test.dots, self.kind)
            self._last_states = {}
            self._last_angles = []
            self._filters = {}
            self.locked = False
            self._refs = {}

    def set_smoothing(self, level):
        if level not in LEVELS:
            raise ValueError(f"smoothing must be one of {list(LEVELS)}")
        with self.lock:
            self.smoothing = level
            self._filters = {}

    def set_dot_kind(self, kind):
        """'white', 'black' or 'auto'. Dots must be re-marked afterwards."""
        with self.lock:
            self.kind = kind
            self.reset_dots()

    def next_dot_to_mark(self):
        with self.lock:
            for n in self.test.dots:
                if not self.trackers.trackers[n].initialised:
                    return n
            return None

    def mark_dot(self, name, x, y):
        """Operator clicked a dot on the live image."""
        with self.lock:
            if self._last_gray is None:
                return False
            if not self.trackers.init_dot(name, self._last_gray, x, y):
                return False
            t = self.trackers.trackers[name]
            self._last_states[name] = DotState(name, t.x, t.y, False, 0)   # usable at once, e.g. by Lock
            self._last_angles = []
            return True

    def nearest_dot(self, x, y, max_dist=60):
        """Name of the marked dot closest to (x, y), or None. Used to re-mark a single dot."""
        with self.lock:
            best = None
            for n, st in self._last_states.items():
                d = ((st.x - x) ** 2 + (st.y - y) ** 2) ** 0.5
                if d <= max_dist and (best is None or d < best[0]):
                    best = (d, n)
            return best[1] if best else None

    def lock_position(self):
        """Store the current neighbour angles as the reference. Returns False if any angle is unavailable."""
        with self.lock:
            angles = self._last_angles or (self._compute_angles(self._last_states) if self._last_states else [])
            if not angles or any(a.status == LOST for a in angles):
                return False
            self._refs = {a.name: a.value for a in angles if not a.primary}
            self.locked = True
            return True

    def unlock_position(self):
        with self.lock:
            self.locked = False
            self._refs = {}

    # --------------------------------------------------------------- per frame
    def _smooth(self, name, raw, t):
        if t is None or self.smoothing == "off":
            return raw
        f = self._filters.get(name)
        if f is None:
            f = self._filters[name] = make_filter(self.smoothing)
        return f(raw, t)

    def _compute_angles(self, states, t=None):
        xy = {n: (st.x, st.y) for n, st in states.items() if not st.lost}
        out = []
        for a in self.test.angles:
            r = measure(a, xy, self.facing_right)
            if r.status == LOST:
                self._filters.pop(a.name, None)         # do not blend across a gap
            else:
                r.raw_value = r.value
                r.value = self._smooth(a.name, r.raw_value, t)
                if a.primary:
                    r.reference, r.tolerance = self.target, self.tolerance
                elif self.locked and a.name in self._refs:
                    r.reference, r.tolerance = self._refs[a.name], self.neighbour_tolerance
                if r.reference is not None:
                    r.deviation = r.value - r.reference
                    r.status = OK if abs(r.deviation) <= r.tolerance else OUT
            out.append(r)
        return out

    def _fps(self):
        if len(self._times) < 5:
            return 30.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 30.0

    def process(self, frame, t):
        with self.lock:
            gray = to_gray(frame)
            self._last_gray = gray
            self._times.append(t)
            states = self.trackers.update(gray)
            self._last_states = states
            angles = self._compute_angles(states, t)
            self._last_angles = angles
            first = self.test.label
            if self.target is not None:
                first += f"  |  target {self.target:g} +/-{self.tolerance:g} deg"
            head = [first, "POSITION LOCKED" if self.locked else "position not locked"]
            lost = [n for n, st in states.items() if st.lost]
            if lost:
                head.append("DOT LOST: " + ", ".join(self.test.dot_title(n) for n in lost))
            weak = [n for n, st in states.items() if st.weak]
            if weak:
                head.append("weak dot (fading): " + ", ".join(self.test.dot_title(n) for n in weak))
            nxt = self.next_dot_to_mark()
            if nxt:
                head = [self.test.label, f"click dot: {self.test.dot_title(nxt)}"]
            img = overlay.draw(frame, states, angles, head, self.rec is not None)
            if self.rec is not None:
                self.rec.write(frame, img, self.frame_index, t, self.locked, states, angles)
            res = FrameResult(self.frame_index, t, img, angles, states)
            self.frame_index += 1
            return res

    # --------------------------------------------------------------- recording
    def start_trial(self, participant, notes="", frame_size=None):
        with self.lock:
            if self.rec is not None:
                return None
            if not self.trackers.is_ready():
                raise RuntimeError("Mark all dots before recording.")
            if frame_size is None:
                h, w = self._last_gray.shape[:2]
                frame_size = (w, h)
            self.rec = TrialRecorder(
                self.recordings_dir, participant or "unnamed", self.test.key, self.target,
                self.test.dots, [a.name for a in self.test.angles], self._fps(), frame_size)
            self.rec_info = {"participant": participant, "notes": notes}
            return self.rec.paths

    def stop_trial(self):
        with self.lock:
            if self.rec is None:
                return None
            t = self.test
            meta = {
                "software_version": SOFTWARE_VERSION,
                "participant": self.rec_info.get("participant"),
                "notes": self.rec_info.get("notes"),
                "test_key": t.key,
                "test_label": t.label,
                "target_deg": self.target,
                "tolerance_deg": self.tolerance,
                "neighbour_tolerance_deg": self.neighbour_tolerance,
                "position_locked": self.locked,
                "locked_references_deg": self._refs,
                "dot_kind": self.kind,
                "camera": self.camera_desc,
                "lens_calibration": None,
                "facing": "right" if self.facing_right else "left",
                "smoothing": self.smoothing,
                "angles": [asdict(a) for a in t.angles],
            }
            paths = self.rec.close(meta)
            self.rec = None
            return paths
