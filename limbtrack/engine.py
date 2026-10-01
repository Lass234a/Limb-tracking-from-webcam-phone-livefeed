"""Glue: tracks dots, computes angles, judges them against the set position, records.

No GUI code here, so the whole pipeline can be tested with a video file.

How "deviation" is judged
  * primary angle   : compared with the TARGET typed by the operator (+/- tolerance).
  * neighbour angles: no judgement until the operator presses "Lock position".
                      At that moment their current values become the reference,
                      and from then on any drift beyond the neighbour tolerance turns red.
"""
import math
import threading
from collections import deque

import numpy as np

from dataclasses import asdict, replace

from . import overlay
from .angles import measure, segment_pairs
# PARKED-GHOST: from .geometry import anchored_ghost
from .protocol import AngleDef
from .filters import SMOOTHING_WINDOW_S, MovingAverage
from .recorder import TrialRecorder
from .results import LOST, OK, OUT, FrameResult
from .tracker import DotState, TrackerSet, to_gray

SOFTWARE_VERSION = "0.2.0"
TRAIL_MIN_STEP_PX = 0.5      # a dot adds a trail point only after moving at least this far (display only)


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
        self.disabled_dots = set()   # landmarks the operator chose not to use (kept when switching test)
        self.three_point = None      # (first, middle, last) dot names of the operator's own angle, or None
        self.three_point_main = True # judge it against the target (instead of the test's own main joint)
        # PARKED-GHOST: self.ghost_enabled = True    # draw a faint copy of the locked pose
        # PARKED-GHOST: self.ghost_anchored = True   # ...shifted so the main joint's fulcrum stays on its live position
        # PARKED-GHOST: self._ghost = None           # dot positions (px) when the position was locked
        self.trails_enabled = True   # draw the path of every dot (display only)
        self._trails = {}            # dot name -> list of segments of (x, y); filled while recording, kept until the next recording
        self._trail_open = {}        # dot name -> was it tracked on the previous recorded frame?
        self._pending_event = None   # marker to write on the next recorded frame
        self._banner = None          # (label, time) of the last marker, shown briefly on screen
        self._marker_n = 0
        self.smoothing = True        # moving average over 0.2 s for what is SHOWN (never for the CSV)
        self._filters = {}
        self.facing_right = True     # which way the participant faces in the image (matters for signed angles)
        self._refs = {}
        self._last_gray = None
        self._last_states = {}
        self._last_angles = []
        self._times = deque(maxlen=30)
        self._hist = deque(maxlen=6000)      # (t, {angle name: (value, deviation)}) for the live trace
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
            self.disabled_dots &= set(self.test.dots)
            if self.three_point and not set(self.three_point) <= set(self.test.dots):
                self.three_point = None
            self.reset_dots()

    def reset_dots(self):
        with self.lock:
            self.trackers = TrackerSet(self.test.dots, self.kind)
            self.trackers.enabled -= self.disabled_dots
            self._last_states = {}
            self._last_angles = []
            self._filters = {}
            self._hist.clear()
            self._trails, self._trail_open = {}, {}
            self.locked = False
            self._refs = {}
            # PARKED-GHOST: self._ghost = None

    # ----------------------------------------------------- landmark selection
    @property
    def active_dots(self):
        return [n for n in self.test.dots if n not in self.disabled_dots]

    def _three_point_def(self):
        if not self.three_point:
            return None
        a, b, c = self.three_point
        return AngleDef(name="three_point", label=f"Angle at {self.test.dot_title(b)}", vectors=[[b, a], [b, c]],
                        signed=False, sign=1.0, offset=0.0, primary=self.three_point_main, fulcrum=b)

    @property
    def all_angles(self):
        """The test's own angles plus the operator's three-point angle, if any.

        A three-point angle that is the main joint takes over that role from the test's own main joint.
        """
        own = list(self.test.angles)
        tp = self._three_point_def()
        if tp is None:
            return own
        if tp.primary:
            own = [replace(a, primary=False) for a in own]
        return own + [tp]

    @property
    def active_angles(self):
        """Angles whose dots are all switched on."""
        return [a for a in self.all_angles if not set(a.dots) & self.disabled_dots]

    @property
    def primary_angle(self):
        """The main joint, or None if the chosen landmarks cannot give it."""
        return next((a for a in self.active_angles if a.primary), None)

    def angle_availability(self):
        """[(angle title, available, [titles of the switched-off dots it needs])] for every angle of the test."""
        with self.lock:
            return [(a.title, not (set(a.dots) & self.disabled_dots),
                     [self.test.dot_title(d) for d in a.dots if d in self.disabled_dots]) for a in self.all_angles]

    def set_three_point(self, first, middle, last, main=True):
        """Add an angle between three landmarks of your choice: the unsigned angle at the middle one.

        A straight limb reads 180 deg. The three landmarks are switched on automatically.
        With main=True it is judged against the target; otherwise like the other joints (against the locked position).
        """
        with self.lock:
            if self.rec is not None:
                raise RuntimeError("Stop recording before changing the angles.")
            names = (first, middle, last)
            if len(set(names)) != 3:
                raise ValueError("Choose three different landmarks.")
            if any(n not in self.test.dots for n in names):
                raise ValueError("That landmark is not used by this test.")
            for n in names:
                self.set_dot_enabled(n, True)
            self.three_point, self.three_point_main = names, bool(main)
            self._angles_changed()

    def clear_three_point(self):
        with self.lock:
            if self.rec is not None:
                raise RuntimeError("Stop recording before changing the angles.")
            self.three_point = None
            self._angles_changed()

    def _angles_changed(self):
        self._last_angles = []
        self._filters = {}
        self._hist.clear()
        self.unlock_position()

    def lock_problem(self):
        """Plain-English reason why the position cannot be locked right now."""
        with self.lock:
            if not self.active_angles:
                return ("No angle can be computed with the chosen landmarks. Tick the missing landmarks, "
                        "or choose a three-point angle.")
            missing = [self.test.dot_title(n) for n in self.active_dots if n not in self._last_states]
            if missing:
                return "Mark these dots first: " + ", ".join(missing)
            lost = [self.test.dot_title(n) for n, st in self._last_states.items() if st.lost]
            if lost:
                return "These dots are not visible right now: " + ", ".join(lost)
            return "Every dot must be marked and visible to lock the position."
    def set_dot_enabled(self, name, on):
        """Use or skip one landmark (pilot testing). Not possible while recording.

        Angles that need a skipped landmark disappear; the rest keep working. The position is unlocked.
        """
        with self.lock:
            if self.rec is not None:
                raise RuntimeError("Stop recording before changing which landmarks are used.")
            if name not in self.test.dots or (name not in self.disabled_dots) == bool(on):
                return
            if on:
                self.disabled_dots.discard(name)
            else:
                self.disabled_dots.add(name)
            self.trackers.set_enabled(name, on)
            self._trails, self._trail_open = {}, {}          # the set of dots changed: old trails no longer apply
            self._last_states.pop(name, None)
            self._last_angles = []
            self._filters = {}
            self.unlock_position()

    def set_smoothing(self, on):
        """Switch the display smoothing on or off. The recorded angles are unaffected."""
        with self.lock:
            self.smoothing = bool(on)
            self._filters = {}

    def set_dot_kind(self, kind):
        """'white', 'black' or 'auto'. Dots must be re-marked afterwards."""
        with self.lock:
            self.kind = kind
            self.reset_dots()

    def next_dot_to_mark(self):
        with self.lock:
            for n in self.active_dots:
                if not self.trackers.trackers[n].initialised:
                    return n
            return None

    def mark_dot(self, name, x, y):
        """Operator clicked a dot on the live image."""
        with self.lock:
            if self._last_gray is None or name in self.disabled_dots:
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
            if not angles or any(a.status == LOST for a in angles) or len(angles) < len(self.active_angles):
                return False
            self._refs = {a.name: a.value for a in angles if not a.primary}
            # PARKED-GHOST: self._ghost = {n: (st.x, st.y) for n, st in self._last_states.items() if not st.lost}
            # PARKED-GHOST: if self.rec is not None:
                # PARKED-GHOST: self.rec_info["ghost"] = dict(self._ghost)
            self.locked = True
            return True

    def unlock_position(self):
        with self.lock:
            self.locked = False
            self._refs = {}
            # PARKED-GHOST: self._ghost = None

    # --------------------------------------------------------------- per frame
    def _smooth(self, name, raw, t):
        if t is None or not self.smoothing:
            return raw
        f = self._filters.get(name)
        if f is None:
            f = self._filters[name] = MovingAverage(SMOOTHING_WINDOW_S)
        return f(raw, t)

    def _compute_angles(self, states, t=None):
        xy = {n: (st.x, st.y) for n, st in states.items() if not st.lost}
        out = []
        for a in self.active_angles:
            r = measure(a, xy, self.facing_right)       # r.value is the raw angle here
            if r.status == LOST:
                r.raw_status = LOST
                self._filters.pop(a.name, None)         # do not average across a gap
            else:
                r.raw_value = r.value
                if a.primary:
                    r.reference, r.tolerance = self.target, self.tolerance
                elif self.locked and a.name in self._refs:
                    r.reference, r.tolerance = self._refs[a.name], self.neighbour_tolerance
                if r.reference is not None:                         # recorded verdict: from the raw angle
                    r.raw_deviation = r.raw_value - r.reference
                    r.raw_status = OK if abs(r.raw_deviation) <= r.tolerance else OUT
                r.value = self._smooth(a.name, r.raw_value, t)      # shown value (and verdict)
                if r.reference is not None:
                    r.deviation = r.value - r.reference
                    r.status = OK if abs(r.deviation) <= r.tolerance else OUT
            out.append(r)
        return out

    def add_event(self, label=None):
        """Mark the current moment in the trial ("MVIC start", "MVIC end", or a numbered marker).

        Only meaningful while recording; returns False (and does nothing) otherwise.
        """
        with self.lock:
            if self.rec is None:
                return False
            if label is None:
                self._marker_n += 1
                label = f"marker {self._marker_n}"
            self._pending_event = label
            return True

    def _update_trails(self, states):
        """Add this frame's dot positions to the trails. A lost dot ends its segment; a new one starts when it is back."""
        for n, st in states.items():
            segs = self._trails.setdefault(n, [])
            if st.lost:
                self._trail_open[n] = False
                continue
            if not self._trail_open.get(n) or not segs:
                segs.append([(st.x, st.y)])
            else:
                lx, ly = segs[-1][-1]
                if (st.x - lx) ** 2 + (st.y - ly) ** 2 >= TRAIL_MIN_STEP_PX ** 2:
                    segs[-1].append((st.x, st.y))
            self._trail_open[n] = True

    def _record_history(self, t, angles):
        while self._hist and self._hist[-1][0] >= t:      # time went backwards (frame stepping): drop the future
            self._hist.pop()
        self._hist.append((t, {a.name: (a.value, a.deviation) for a in angles}))

    def history(self, seconds=15.0):
        """The last few seconds of angles for the live trace, or None if there is nothing yet.

        Returns a dict: t (seconds relative to now, <= 0), values / deviations (angle name -> array),
        plus what the plot needs to draw the tolerance band.
        """
        with self.lock:
            if not self._hist:
                return None
            t_last = self._hist[-1][0]
            rows = [r for r in self._hist if r[0] >= t_last - seconds]
            nan = (math.nan, math.nan)
            names = [a.name for a in self.active_angles]
            main = self.primary_angle
            if not names:
                return None
            return {
                "t": np.array([r[0] - t_last for r in rows]),
                "values": {n: np.array([r[1].get(n, nan)[0] for r in rows]) for n in names},
                "deviations": {n: np.array([r[1].get(n, nan)[1] for r in rows]) for n in names},
                "titles": {a.name: a.title for a in self.active_angles},
                "primary": main.name if main else names[0],
                "target": self.target if main else None,
                "tolerance": self.tolerance,
                "neighbour_tolerance": self.neighbour_tolerance,
                "locked": self.locked,
            }

    # PARKED-GHOST: def _ghost_anchor(self):
        # PARKED-GHOST: """Dot the ghost is pinned to: the main joint's fulcrum, or the first available angle's."""
        # PARKED-GHOST: for a in ([self.primary_angle] if self.primary_angle else []) + self.active_angles:
            # PARKED-GHOST: if a.fulcrum:
                # PARKED-GHOST: return a.fulcrum
        # PARKED-GHOST: return None

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
            self._record_history(t, angles)
            first = self.test.label
            main = self.primary_angle
            if self.target is not None and main is not None:
                first += f"  |  target {self.target:g} +/-{self.tolerance:g} deg"
            head = [first, "POSITION LOCKED" if self.locked else "position not locked"]
            if main is None:
                head.append(f"main joint ({self.test.primary.title}) not available with the chosen landmarks")
            lost = [n for n, st in states.items() if st.lost]
            if lost:
                head.append("DOT LOST: " + ", ".join(self.test.dot_title(n) for n in lost))
            weak = [n for n, st in states.items() if st.weak]
            if weak:
                head.append("weak dot (fading): " + ", ".join(self.test.dot_title(n) for n in weak))
            event, self._pending_event = self._pending_event, None
            if event:
                self._banner = (event, t)
            if self._banner and 0 <= t - self._banner[1] < 1.5:
                head.append(f"MARK: {self._banner[0]}")
            nxt = self.next_dot_to_mark()
            if nxt:
                head = [self.test.label, f"click dot: {self.test.dot_title(nxt)}"]
            # PARKED-GHOST: ghost = None
            # PARKED-GHOST: if self.locked and self.ghost_enabled and self._ghost:
                # PARKED-GHOST: live = {n: (st.x, st.y) for n, st in states.items() if not st.lost}
                # PARKED-GHOST: anchor = self._ghost_anchor() if self.ghost_anchored else None
                # PARKED-GHOST: ghost = anchored_ghost(self._ghost, live, anchor)
            if self.rec is not None:
                self._update_trails(states)
            img = overlay.draw(frame, states, angles, head, self.rec is not None,
                               trails=self._trails if self.trails_enabled else None)
            # PARKED-GHOST: # (with the ghost: overlay.draw(frame, states, angles, head, self.rec is not None, ghost,
            # PARKED-GHOST: #                               segment_pairs(self.active_angles), trails=...))
            if self.rec is not None:
                self.rec.write(frame, img, self.frame_index, t, self.locked, states, angles, event or "")
            res = FrameResult(self.frame_index, t, img, angles, states, head)
            # PARKED-GHOST: # with the ghost: FrameResult(self.frame_index, t, img, angles, states, head, ghost)
            self.frame_index += 1
            return res

    # --------------------------------------------------------------- recording
    def start_trial(self, participant, notes="", frame_size=None):
        with self.lock:
            if self.rec is not None:
                return None
            if not self.active_angles:
                raise RuntimeError("None of the angles can be computed with the chosen landmarks.")
            if not self.trackers.is_ready():
                raise RuntimeError("Mark all dots before recording.")
            if frame_size is None:
                h, w = self._last_gray.shape[:2]
                frame_size = (w, h)
            self.rec = TrialRecorder(
                self.recordings_dir, participant or "unnamed", self.test.key,
                self.target if self.primary_angle else None,
                self.active_dots, [a.name for a in self.active_angles], self._fps(), frame_size)
            self._marker_n = 0
            self._pending_event = None
            self._trails, self._trail_open = {}, {}          # trails cover one recording: start afresh
            self.rec_info = {"participant": participant, "notes": notes}
            # PARKED-GHOST: # with the ghost: self.rec_info = {"participant": participant, "notes": notes,
            # PARKED-GHOST: #                                  "ghost": dict(self._ghost) if self.locked and self._ghost else None}
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
                "target_deg": self.target if self.primary_angle else None,
                "tolerance_deg": self.tolerance,
                "dots_used": self.active_dots,
                "dots_disabled": sorted(self.disabled_dots),
                "neighbour_tolerance_deg": self.neighbour_tolerance,
                "position_locked": self.locked,
                "locked_references_deg": self._refs,
                "dot_kind": self.kind,
                "camera": self.camera_desc,
                "lens_calibration": None,
                "facing": "right" if self.facing_right else "left",
                "smoothing": {
                    "enabled": self.smoothing,
                    "type": "moving_average",
                    "window_s": SMOOTHING_WINDOW_S,
                    "applies_to": "screen, live trace and overlay video only; CSV angles, deviations and flags are unsmoothed",
                },
                "trails": {"shown_in_overlay_video": self.trails_enabled, "min_step_px": TRAIL_MIN_STEP_PX},
                "reference_basis": "main joint: typed target; other joints: the value shown when the position was locked",
                # PARKED-GHOST: "locked_positions_px": self._ghost or self.rec_info.get("ghost"),
                # PARKED-GHOST: "ghost_anchor": self._ghost_anchor(),
                "three_point": list(self.three_point) if self.three_point else None,
                "angles": [asdict(a) for a in self.active_angles],
            }
            paths = self.rec.close(meta)
            self.rec = None
            return paths
