"""Glue: tracks dots, computes angles, judges them against the set position, records.

No GUI code here, so the whole pipeline can be tested with a video file.

How "deviation" is judged
  * primary angle   : compared with the TARGET typed by the operator (+/- tolerance).
  * neighbour angles: no judgement until the operator presses "Lock position".
                      At that moment their current values become the reference,
                      and from then on any drift beyond the neighbour tolerance turns red.
"""
import copy
import math
import threading
import time
from collections import deque

import numpy as np

from dataclasses import asdict, replace

from . import overlay
from .angles import measure, segment_pairs
from .geometry import view_handedness
# PARKED-GHOST: from .geometry import anchored_ghost
from .protocol import AngleDef
from .filters import SMOOTHING_WINDOW_S, MovingAverage
from .recorder import FrameSizeChanged, TrialRecorder
from .pull import compute_pull, describe as describe_pull
from .scale import MIN_DOT_DIAMETER_CM, ScaleReference, estimate_diameter_cm
from .results import LOST, OK, OUT, FrameResult
from .tracker import NEIGHBOUR_EXCLUSION_FACTOR, DotState, TrackerSet, to_gray
from .tracker import settings as tracker_settings

SOFTWARE_VERSION = "0.2.0"
LOST_CLICK_REACH_PX = 150      # a click this near a lost dot's frozen position re-marks that lost dot
RE_MARK_MIN_REACH_PX = 15      # a click must be at least this near a tracked dot (or 2 x its diameter) to re-mark it
TRAIL_MIN_STEP_PX = 0.5      # a dot adds a trail point only after moving at least this far (display only)


class _FrozenWhileRecording:
    """A setting that cannot change while a trial is being recorded (it would make the saved trial inconsistent)."""

    def __init__(self, label):
        self.label = label

    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __get__(self, obj, objtype=None):
        return self if obj is None else getattr(obj, self.name)

    def __set__(self, obj, value):
        if getattr(obj, "rec", None) is not None:
            raise RuntimeError(f"Stop recording before changing {self.label}.")
        setattr(obj, self.name, value)


# Settings copied at the start of a recording; meta.json reports these start values (see Engine.stop_trial).
SNAPSHOT_KEYS = ["participant", "notes", "test_key", "test_label", "target_deg", "tolerance_deg", "neighbour_tolerance_deg",
                 "neighbour_targets_deg", "dots_used", "dots_disabled", "optional_dots_used", "position_locked", "locked_references_deg", "dot_kind",
                 "facing", "mirrored_image", "smoothing", "trails", "three_point", "three_point_convention", "angles", "overlay_video_saved"]


class Engine:
    target = _FrozenWhileRecording("the target")
    tolerance = _FrozenWhileRecording("the tolerance")
    neighbour_tolerance = _FrozenWhileRecording("the tolerance for the other joints")
    facing_right = _FrozenWhileRecording("which way the participant faces")
    mirror = _FrozenWhileRecording("the mirror setting")
    save_overlay = _FrozenWhileRecording("whether the overlay video is saved")

    def __init__(self, tests, recordings_dir="recordings"):
        self.tests = tests
        self.recordings_dir = recordings_dir
        self.lock = threading.RLock()
        self.test = None
        self.trackers = None
        self.kind = "auto"
        self.target = None
        self._frame_stamp = None
        self.scale = ScaleReference(self.kind)   # pixel-to-cm scale from a ruler with a dot at each end (limbtrack/scale.py)
        self._scale_values = []                  # live scale per recorded frame
        self.fps_source = None                   # callable giving the video source's frame rate (set by set_source)
        self.facing_detected = None              # +1 / -1 / None: which side of the body the marked dots show (see geometry.view_handedness)
        self._facing_bad_frames = 0              # consecutive frames in which that disagrees with 'Participant faces'
        self.last_stop_reason = None             # why the engine itself ended the last recording (picture size changed), if it did
        self._notice = None                      # (text, time) shown in the header for a few seconds
        self._ref_avg, self._ref_last = {}, {}      # unsmoothed 0.2 s average per angle, used as the neighbour reference at Lock
        self._meta_start = {}                    # settings at the start of the current recording
        self.mark_problem = None     # why the last mark_dot failed, when it is not simply 'no dot there'
        self.tolerance = 3.0
        self.neighbour_tolerance = 5.0
        self.locked = False
        self._optional_seen = set()   # optional dots that have been offered once (they start switched off)
        self.disabled_dots = set()   # landmarks the operator chose not to use (kept when switching test)
        self.three_point = None      # (first, middle, last) dot names of the operator's own angle, or None
        self.three_point_main = True # judge it against the target (instead of the test's own main joint)
        self.three_point_flexion = False   # three-point angle as flexion (0 = straight) instead of the interior angle (180 = straight)
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
        self._filters, self._ref_avg = {}, {}
        self.save_overlay = True     # also save overlay.mp4 (Review can redraw it from raw.mp4 and the CSV, and export it later)
        self.mirror = False          # flip the image left-right before anything else (applied by the video thread)
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
        self.camera_info = None      # requested / granted camera settings, set by the video thread
        self.set_test(next(iter(tests)))

    # ----------------------------------------------------------------- set-up
    def set_test(self, key):
        with self.lock:
            self._forbid_while_recording("the test")
            self.test = self.tests[key]
            self.tolerance = self.test.tolerance_deg
            self.neighbour_tolerance = self.test.neighbour_tolerance_deg
            self.target = self.test.targets[0] if self.test.targets else None
            if self.three_point and self.three_point_main and not self.three_point_flexion:
                self.target = None            # the test's flexion positions do not apply to an interior angle
            self.disabled_dots &= set(self.test.all_dots)
            for n in self.test.optional_dots:               # optional dots (strap, anchor) start switched off; a choice is kept
                if n not in self._optional_seen:
                    self._optional_seen.add(n)
                    self.disabled_dots.add(n)
            if self.three_point and not set(self.three_point) <= set(self.test.dots):
                self.three_point = None
            self.reset_dots()

    def _forbid_while_recording(self, what):
        if self.rec is not None:
            raise RuntimeError(f"Stop recording before changing {what}.")

    def reset_dots(self):
        with self.lock:
            self._forbid_while_recording("the marked dots")
            self._reset_dots_core()

    def _reset_dots_core(self):
        with self.lock:
            self._facing_bad_frames, self.facing_detected = 0, None
            self.trackers = TrackerSet(self.test.all_dots, self.kind)
            self.trackers.enabled -= self.disabled_dots
            self._last_states = {}
            self._last_angles = []
            self._filters, self._ref_avg = {}, {}
            self._hist.clear()
            self._trails, self._trail_open = {}, {}
            self.locked = False
            self._refs = {}
            # PARKED-GHOST: self._ghost = None

    # ----------------------------------------------------- landmark selection
    @property
    def active_dots(self):
        return [n for n in self.test.all_dots if n not in self.disabled_dots]

    def _three_point_def(self):
        if not self.three_point:
            return None
        a, b, c = self.three_point
        if self.three_point_flexion:         # 180 - interior angle: 0 = straight, like the protocol's flexion angles
            return AngleDef(name="three_point", label=f"Flexion at {self.test.dot_title(b)} (0 = straight)", vectors=[[b, a], [b, c]],
                            signed=False, sign=-1.0, offset=180.0, primary=self.three_point_main, fulcrum=b)
        return AngleDef(name="three_point", label=f"Interior angle at {self.test.dot_title(b)} (180 = straight)", vectors=[[b, a], [b, c]],
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

    def set_three_point(self, first, middle, last, main=True, flexion=None):
        """Add an angle between three landmarks of your choice: the unsigned angle at the middle one.

        By default the interior angle (a straight limb reads 180 deg); with flexion=True it is 180 minus that (a straight limb
        reads 0 deg, like the protocol's flexion angles). The three landmarks are switched on automatically.
        With main=True it is judged against the target; otherwise like the other joints (against the locked position).
        The typed target keeps its meaning: it is converted when the convention changes and cleared when a flexion
        suggestion would otherwise be compared with an interior angle (audit ANG-05).
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
            was_main = bool(self.three_point) and self.three_point_main
            was_flexion = self.three_point_flexion
            if flexion is not None:
                self.three_point_flexion = bool(flexion)
            self.three_point, self.three_point_main = names, bool(main)
            if self.three_point_main and self.target is not None:
                if was_main and self.three_point_flexion != was_flexion:
                    self.target = 180.0 - self.target                     # same pose, other convention
                elif not was_main and not self.three_point_flexion and self.target in self.test.targets:
                    self.target = None                                    # a flexion position is not an interior angle
            self._angles_changed()

    def clear_three_point(self):
        with self.lock:
            if self.rec is not None:
                raise RuntimeError("Stop recording before changing the angles.")
            self.three_point = None
            self._angles_changed()

    def _angles_changed(self):
        self._last_angles = []
        self._filters, self._ref_avg = {}, {}
        self._hist.clear()
        self.unlock_position()

    def lock_problem(self):
        """Plain-English reason why the position cannot be locked right now."""
        with self.lock:
            if self.facing_mismatch:
                return self.facing_message() + " Change 'Participant faces' (or the mirror) to match the picture, then lock."
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
            if name not in self.test.all_dots or (name not in self.disabled_dots) == bool(on):
                return
            if on:
                self.disabled_dots.discard(name)
            else:
                self.disabled_dots.add(name)
            self.trackers.set_enabled(name, on)
            self._trails, self._trail_open = {}, {}          # the set of dots changed: old trails no longer apply
            self._last_states.pop(name, None)
            self._last_angles = []
            self._filters, self._ref_avg = {}, {}
            self.unlock_position()

    def set_smoothing(self, on):
        """Switch the display smoothing on or off. The recorded angles are unaffected."""
        with self.lock:
            self.smoothing = bool(on)
            self._filters, self._ref_avg = {}, {}

    def set_dot_kind(self, kind):
        """'white', 'black' or 'auto'. Dots must be re-marked afterwards."""
        with self.lock:
            self._forbid_while_recording("the dot colour")
            self.kind = kind
            self.scale.reset(kind)           # the ruler dots must be marked again with the new colour setting
            self.reset_dots()

    def next_dot_to_mark(self):
        with self.lock:
            for n in self.active_dots:
                if not self.trackers.trackers[n].initialised:
                    return n
            return None

    def mark_scale_dot(self, index, x, y):
        """Operator clicked ruler dot `index` (0 or 1). On failure, `mark_problem` says why when it is not simply 'no dot found'."""
        with self.lock:
            self.mark_problem = None
            if self._last_gray is None or self.rec is not None:
                return False
            ok, self.mark_problem = self.scale.mark(index, self._last_gray, x, y)
            return ok

    # ------------------------------------------------------------ angle of pull
    @property
    def pull_active(self):
        return "strap" in self.active_dots and "anchor" in self.active_dots

    def pull_segment(self):
        """(proximal dot, distal dot, fulcrum dot) the pull is measured against: the second arm of the test's main joint."""
        p = self.test.primary
        arm = p.vectors[1] if len(p.vectors) == 2 else None
        if isinstance(arm, (list, tuple)) and len(arm) == 2 and all(isinstance(d, str) for d in arm) and p.fulcrum:
            return arm[0], arm[1], p.fulcrum
        return None

    def _compute_pull(self, states):
        if not self.pull_active:
            return None
        spec = self.pull_segment()
        if spec is None:
            return None
        names = ("strap", "anchor") + tuple(spec)
        if any(n not in states or states[n].lost for n in names):
            return None
        xy = lambda n: (states[n].x, states[n].y)                       # noqa: E731
        return compute_pull(xy("strap"), xy("anchor"), xy(spec[0]), xy(spec[1]), xy(spec[2]), self.scale.px_per_cm)

    def dot_sizes_cm(self):
        """Approximate width in cm of each marked dot (from its size when marked and the ruler scale); {} without a scale."""
        with self.lock:
            ppc = self.scale.px_per_cm
            if not ppc:
                return {}
            return {n: estimate_diameter_cm(self.trackers.trackers[n].marked_diam, ppc) for n in self.active_dots
                    if self.trackers.trackers[n].marked_diam}

    def small_dots(self):
        """Dots whose approximate width is under the protocol minimum, as {name: cm}."""
        return {n: d for n, d in self.dot_sizes_cm().items() if d < MIN_DOT_DIAMETER_CM}

    def clear_scale(self):
        with self.lock:
            if self.rec is None:
                self.scale.reset()

    def mark_dot(self, name, x, y):
        """Operator clicked a dot on the live image. On failure, `mark_problem` says why (None = no dot found there)."""
        with self.lock:
            self.mark_problem = None
            if self.rec is not None:
                self.mark_problem = "Stop recording before re-marking a dot."
                return False
            if self._last_gray is None or name in self.disabled_dots:
                return False
            old = copy.deepcopy(self.trackers.trackers[name])      # a re-mark must never damage a working tracker
            if not self.trackers.init_dot(name, self._last_gray, x, y):
                return False
            t = self.trackers.trackers[name]
            if old.initialised:                                       # re-mark: never onto a dot another tracker holds
                for other, o in self.trackers.trackers.items():
                    if other == name or not o.initialised or other not in self.trackers.enabled or o.lost_frames:
                        continue
                    if (o.x - t.x) ** 2 + (o.y - t.y) ** 2 < (NEIGHBOUR_EXCLUSION_FACTOR * o.diam) ** 2:
                        self.trackers.trackers[name] = old
                        self.mark_problem = f"That is the {self.test.dot_title(other)} dot, not the {self.test.dot_title(name)} dot."
                        return False
            self._last_states[name] = DotState(name, t.x, t.y, False, 0)   # usable at once, e.g. by Lock
            self._last_angles = []
            return True

    def nearest_dot(self, x, y, lost_dist=LOST_CLICK_REACH_PX):
        """Name of the marked dot a click at (x, y) is meant to re-mark, or None. Used to re-mark a single dot.

        A lost dot sits frozen where it was last seen, so it is chosen first (the nearest lost dot within
        `lost_dist`). Otherwise the nearest tracked dot, but only if the click is close to it (within twice its
        diameter): a tracked dot has no reason to be re-marked from far away.
        """
        with self.lock:
            best_lost = best_found = None
            for n, st in self._last_states.items():
                d = ((st.x - x) ** 2 + (st.y - y) ** 2) ** 0.5
                if st.lost:
                    if d <= lost_dist and (best_lost is None or d < best_lost[0]):
                        best_lost = (d, n)
                else:
                    t = self.trackers.trackers.get(n)
                    reach = max(2.0 * (t.diam if t else 0.0), RE_MARK_MIN_REACH_PX)
                    if d <= reach and (best_found is None or d < best_found[0]):
                        best_found = (d, n)
            best = best_lost or best_found
            return best[1] if best else None

    def lock_position(self):
        """Store the current neighbour angles as the reference. Returns False if any angle is unavailable."""
        with self.lock:
            self._forbid_while_recording("the locked position")
            if self.facing_mismatch:
                return False
            angles = self._last_angles or (self._compute_angles(self._last_states) if self._last_states else [])
            if not angles or any(a.status == LOST for a in angles) or len(angles) < len(self.active_angles):
                return False
            self._refs = {a.name: self._ref_last.get(a.name, a.raw_value) for a in angles if not a.primary}   # unsmoothed, whatever the smoothing switch says
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
    def _reference_value(self, name, raw, t):
        """The value stored as a neighbour's reference at Lock: the unsmoothed angle averaged over the same 0.2 s as the
        display smoothing, but ALWAYS, so the saved reference does not depend on the smoothing switch (audit DOC-02)."""
        if t is None:
            return raw
        f = self._ref_avg.get(name)
        if f is None:
            f = self._ref_avg[name] = MovingAverage(SMOOTHING_WINDOW_S)
        return f(raw, t)

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
                self._ref_avg.pop(a.name, None)
                self._ref_last.pop(a.name, None)
            else:
                r.raw_value = r.value
                self._ref_last[a.name] = self._reference_value(a.name, r.raw_value, t)
                if a.primary:
                    r.reference, r.tolerance = self.target, self.tolerance
                elif a.name in self.test.neighbour_targets:        # required neighbour angle (Tabel 2): judged from the start
                    r.reference, r.tolerance = float(self.test.neighbour_targets[a.name]), self.neighbour_tolerance
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

    def _picture_size_changed(self, old_shape, new_shape, t):
        """The camera now delivers another picture size (a phone rotated, a virtual camera changed resolution): positions
        and videos in the old size no longer fit. End any recording (saved up to here) and ask for the dots again."""
        text = f"The picture size changed from {old_shape[1]}x{old_shape[0]} to {new_shape[1]}x{new_shape[0]}"
        if self.rec is not None:
            self.last_stop_reason = text + ". The recording was stopped and saved up to this point."
            self.stop_trial(self.last_stop_reason)
        self._reset_dots_core()
        self.scale.reset()
        self._notice = (text + ": mark the dots (and the ruler) again.", t)

    FACING_MISMATCH_FRAMES = 5

    def _check_facing(self, states):
        """Compare the side of the body the dots show with the 'Participant faces' setting (audit ANG-04)."""
        h = view_handedness({n: (st.x, st.y) for n, st in states.items() if not st.lost})
        self.facing_detected = h
        expected = 1 if self.facing_right else -1
        self._facing_bad_frames = self._facing_bad_frames + 1 if (h is not None and h != expected) else 0

    @property
    def facing_mismatch(self):
        """True when the marked dots clearly show the other side of the body than 'Participant faces' says."""
        return self._facing_bad_frames >= self.FACING_MISMATCH_FRAMES

    def facing_message(self):
        shown = "RIGHT" if self.facing_detected == 1 else "LEFT"
        chosen = "RIGHT" if self.facing_right else "LEFT"
        return (f"CHECK 'Participant faces': the dots look like a {shown}-facing view of the picture, but {chosen} is selected. "
                "The hip and pelvic-tilt signs would be wrong.")

    def seconds_since_last_frame(self):
        """Wall-clock seconds since process() last ran (infinity before the first frame). For operator warnings only."""
        return time.perf_counter() - self._frame_stamp if self._frame_stamp is not None else float("inf")

    def set_source(self, camera_desc, camera_info, fps_provider=None):
        """A new video source has started: forget the old timing. `fps_provider()` gives the source's frame rate
        (a file's own rate, or a camera's measured rate) or None."""
        with self.lock:
            self.camera_desc, self.camera_info = camera_desc, camera_info
            self.fps_source = fps_provider
            self._times.clear()

    def _fps(self):
        """Frame rate the videos of a recording are tagged with: the source's own rate when it is known (a file's rate;
        a camera's measured rate), otherwise estimated from the last frame times. Always between 1 and 240."""
        if self.fps_source is not None:
            try:
                v = self.fps_source()
            except Exception:  # noqa: BLE001 - never let a timing query stop the recording
                v = None
            if v and 1.0 <= float(v) <= 240.0:
                return float(v)
        if len(self._times) < 5:
            return 30.0
        span = self._times[-1] - self._times[0]
        est = (len(self._times) - 1) / span if span > 0 else 30.0
        return min(max(est, 1.0), 240.0)

    def process(self, frame, t, source_frame=None):
        with self.lock:
            gray = to_gray(frame)
            if self._last_gray is not None and gray.shape != self._last_gray.shape:
                self._picture_size_changed(self._last_gray.shape, gray.shape, t)
            self._last_gray = gray
            if self._times and t < self._times[-1]:         # the video stepped back: start the timing afresh
                self._times.clear()
            if not self._times or t > self._times[-1]:      # (an equal time is the same frame redrawn: not a new frame)
                self._times.append(t)
            self._frame_stamp = time.perf_counter()
            states = self.trackers.update(gray)
            self._last_states = states
            self.scale.update(gray)
            angles = self._compute_angles(states, t)
            self._check_facing(states)
            self._last_angles = angles
            self._record_history(t, angles)
            first = self.test.label
            main = self.primary_angle
            if self.target is not None and main is not None:
                first += f"  |  target {self.target:g} +/-{self.tolerance:g} deg"
            head = [first, "POSITION LOCKED" if self.locked else "position not locked"]
            if main is None:
                head.append(f"main joint ({self.test.primary.title}) not available with the chosen landmarks")
            if self.facing_mismatch:
                head.append(self.facing_message())
            pull = self._compute_pull(states)
            if self.pull_active:
                head.append(describe_pull(pull) if pull else "PULL: not available (a strap, anchor or segment dot is lost or off)")
            for problem in self.scale.problems():
                head.append("SCALE: " + problem)
            lost = [n for n, st in states.items() if st.lost]
            if lost:
                head.append("DOT LOST: " + ", ".join(self.test.dot_title(n) for n in lost))
            partial = [n for n, st in states.items() if st.partial]
            if partial:
                head.append("dot partly covered or cut off (its centre may be shifted): " + ", ".join(self.test.dot_title(n) for n in partial))
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
            if self._notice and 0 <= t - self._notice[1] < 10.0:
                head.append(self._notice[0])
            # PARKED-GHOST: ghost = None
            # PARKED-GHOST: if self.locked and self.ghost_enabled and self._ghost:
                # PARKED-GHOST: live = {n: (st.x, st.y) for n, st in states.items() if not st.lost}
                # PARKED-GHOST: anchor = self._ghost_anchor() if self.ghost_anchored else None
                # PARKED-GHOST: ghost = anchored_ghost(self._ghost, live, anchor)
            if self.rec is not None:
                self._update_trails(states)
            img = overlay.draw(frame, states, angles, head, self.rec is not None,
                               trails=self._trails if self.trails_enabled else None, scale=self.scale.overlay_info())
            # PARKED-GHOST: # (with the ghost: overlay.draw(frame, states, angles, head, self.rec is not None, ghost,
            # PARKED-GHOST: #                               segment_pairs(self.active_angles), trails=...))
            if self.rec is not None:
                live_scale = self.scale.live_px_per_cm()
                self._scale_values.append(live_scale)
                try:
                    self.rec.write(frame, img, self.rec.frames, t, self.locked, states, angles, event or "", live_scale, pull,
                                   source_frame)
                except FrameSizeChanged as e:           # (normally caught earlier; kept as a safety net)
                    self.last_stop_reason = f"{e} The recording was stopped and saved up to this point."
                    self.stop_trial(self.last_stop_reason)
            res = FrameResult(self.frame_index, t, img, angles, states, head, pull)
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
                self.active_dots, [a.name for a in self.active_angles], self._fps(), frame_size, with_pull=self.pull_active,
                save_overlay=self.save_overlay)
            self._marker_n = 0
            self._pending_event = None
            self._trails, self._trail_open = {}, {}          # trails cover one recording: start afresh
            self._scale_values = []
            self.rec_info = {"participant": participant, "notes": notes}
            self._meta_start = copy.deepcopy(self._trial_meta())      # snapshot: meta.json reports these start values
            self.last_stop_reason = None
            self.rec.write_start_meta(self._meta_start)               # a complete meta.json right away (status 'recording')
            # PARKED-GHOST: # with the ghost: self.rec_info = {"participant": participant, "notes": notes,
            # PARKED-GHOST: #                                  "ghost": dict(self._ghost) if self.locked and self._ghost else None}
            return self.rec.paths

    def _pull_meta(self):
        if not self.pull_active:
            return None
        spec = self.pull_segment()
        return {"direction": "from the strap marker towards the anchor marker",
                "segment_from_to": list(spec[:2]) if spec else None, "moment_arm_from_joint_dot": spec[2] if spec else None,
                "pull_angle": "angle between the pull direction and the segment (proximal to distal dot); 90 = square to the segment",
                "moment_arm": "shortest distance from the joint dot to the line of pull",
                "cm_use_ruler_scale": self.scale.px_per_cm is not None,
                "valid_for": "the 2D picture only; cm only at the ruler's distance from the camera"}

    def _dot_size_check(self):
        sizes = self.dot_sizes_cm()
        if not sizes:
            return None
        return {"min_dot_diameter_cm": MIN_DOT_DIAMETER_CM, "px_per_cm": round(self.scale.px_per_cm, 4),
                "dots_cm_estimate": {n: round(v, 2) for n, v in sizes.items()},
                "below_minimum": sorted(self.small_dots()),
                "note": "approximate: from the blob size when the dot was marked, minus an allowance for soft edges"}

    def _trial_meta(self):
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
            "dots_disabled": sorted(n for n in self.disabled_dots if n not in t.optional_dots),   # landmarks the operator switched off
            "optional_dots_used": [n for n in t.optional_dots if n not in self.disabled_dots],   # strap / anchor, when switched on
            "neighbour_tolerance_deg": self.neighbour_tolerance,
            "neighbour_targets_deg": dict(self.test.neighbour_targets),
            "position_locked": self.locked,
            "locked_references_deg": self._refs,
            "dot_kind": self.kind,
            "tracker_settings": tracker_settings(),
            "camera": self.camera_desc,
            "camera_settings": self.camera_info,
            "lens_calibration": None,
            "facing": "right" if self.facing_right else "left",
            "overlay_video_saved": self.save_overlay,
            "facing_detected": {1: "right", -1: "left"}.get(self.facing_detected),   # what the marked dots show; should equal facing
            "mirrored_image": self.mirror,   # True: raw.mp4, overlay.mp4 and all pixel coordinates are left-right flipped
            "smoothing": {
                "enabled": self.smoothing,
                "type": "moving_average",
                "window_s": SMOOTHING_WINDOW_S,
                "applies_to": "screen, live trace and overlay video only; CSV angles, deviations and flags are unsmoothed",
            },
            "trails": {"shown_in_overlay_video": self.trails_enabled, "min_step_px": TRAIL_MIN_STEP_PX},
            "reference_basis": "main joint: typed target; neighbour joints with a required angle (neighbour_targets_deg): that angle; other joints: their value when the position was locked",
            # PARKED-GHOST: "locked_positions_px": self._ghost or self.rec_info.get("ghost"),
            # PARKED-GHOST: "ghost_anchor": self._ghost_anchor(),
            "scale": self.scale.meta(self._scale_values),
            "dot_size_check": self._dot_size_check(),
            "pull": self._pull_meta(),
            "three_point": list(self.three_point) if self.three_point else None,
            "three_point_convention": (None if not self.three_point else
                                      "flexion (180 - interior angle, 0 = straight)" if self.three_point_flexion
                                      else "interior angle (180 = straight)"),
            "angles": [asdict(a) for a in self.active_angles],
        }
        return meta

    def stop_trial(self, reason=None):
        with self.lock:
            if self.rec is None:
                return None
            end_meta = self._trial_meta()
            meta = dict(end_meta)
            changed = {}
            for k in SNAPSHOT_KEYS:                     # report the settings as they were when the recording started
                if self._meta_start.get(k) != end_meta.get(k):
                    changed[k] = [self._meta_start.get(k), end_meta.get(k)]
                meta[k] = self._meta_start.get(k)
            if changed:
                meta["changed_during_recording"] = changed      # should never happen: the settings are frozen while recording
            if reason:
                meta["stopped_because"] = reason
            rec, self.rec = self.rec, None                      # cleared first: a failure while saving must not leave 'recording' on
            return rec.close(meta)
