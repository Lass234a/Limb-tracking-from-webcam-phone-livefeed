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
from .geometry import KNEE_FACING_MIN_DEG, foot_handedness, view_handedness
# PARKED-GHOST: from .geometry import anchored_ghost
from .protocol import AngleDef
from .filters import SMOOTHING_WINDOW_S, MovingAverage
from .recorder import FrameSizeChanged, TrialRecorder
from . import cues, privacy, provenance
from .pull import compute_pull, describe as describe_pull
from .automvic import AUTO_STOP_AFTER_END_S, DEFAULT_LENGTH_S, DEFAULT_THRESHOLD_N, DEFAULT_THRESHOLD_V, KEEP_S, TRIGGERED, AutoMvic
from .scale import MIN_DOT_DIAMETER_CM, ScaleReference, estimate_diameter_cm
from .results import LOST, OK, OUT, FrameResult
from .tracker import NEIGHBOUR_EXCLUSION_FACTOR, DotState, TrackerSet, to_gray
from .tracker import settings as tracker_settings

SOFTWARE_VERSION = "0.14.0"
META_SCHEMA = 2             # 2: provenance, settings snapshot, gaps, partial-dot counts, source_frame
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
                 "facing", "mirrored_image", "smoothing", "trails", "three_point", "three_point_convention", "angles", "overlay_video_saved", "camera_setup_checklist",
                 "privacy", "force_setup", "auto_mvic_setup", "audio_cues"]


class Engine:
    target = _FrozenWhileRecording("the target")
    tolerance = _FrozenWhileRecording("the tolerance")
    neighbour_tolerance = _FrozenWhileRecording("the tolerance for the other joints")
    facing_right = _FrozenWhileRecording("which way the participant faces")
    mirror = _FrozenWhileRecording("the mirror setting")
    save_overlay = _FrozenWhileRecording("whether the overlay video is saved")
    camera_setup = _FrozenWhileRecording("the camera set-up checklist")
    privacy_crop = _FrozenWhileRecording("the privacy crop")
    privacy_masks = _FrozenWhileRecording("the privacy masks")
    force = _FrozenWhileRecording("the force source")
    auto_mvic = _FrozenWhileRecording("the automatic MVIC markers")
    auto_mvic_threshold = _FrozenWhileRecording("the force rise that starts the automatic MVIC")
    auto_mvic_seconds = _FrozenWhileRecording("the length of the automatic MVIC")
    audio_cues = _FrozenWhileRecording("the sound cues")
    cue_push_every_s = _FrozenWhileRecording("the interval of the motivating ticks")

    def __init__(self, tests, recordings_dir="recordings"):
        self.tests = tests
        self.recordings_dir = recordings_dir
        self.protocol_path = None    # set by the window: the protocol.json in use (its hash goes into meta.json)
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
        self._facing_from_knee = False           # facing_detected came from the bend of the knee (the foot dots are not marked)
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
        self.paused = False          # a break: tracking is off, nothing can be recorded, a ghost of the dots is shown (see pause)
        self._pause_ghost = None     # dot name -> (x, y, diameter) of where the dots were when the break began; kept until they are back
        self.trails_enabled = True   # draw the path of every dot (display only)
        self._trails = {}            # dot name -> list of segments of (x, y); filled while recording, kept until the next recording
        self._trail_open = {}        # dot name -> was it tracked on the previous recorded frame?
        self._pending_event = None   # marker to write on the next recorded frame
        self._banner = None          # (label, time) of the last marker, shown briefly on screen
        self._marker_n = 0
        self.smoothing = True        # moving average over 0.2 s for what is SHOWN (never for the CSV)
        self._filters, self._ref_avg = {}, {}
        self.camera_setup = {}        # the operator's camera checklist (stabilisation off, zoom 1x, ...), saved with each trial
        self.camera_check = None     # result of the last 'Check the camera' (frame rate, longest wait)
        self.privacy_crop = None     # (x, y, w, h) in picture pixels: only this part of the picture is saved to the videos
        self.privacy_masks = []      # rectangles blacked out in the saved videos (tracking and the CSV still use the full picture)
        self._privacy_mirror = False # the mirror setting the crop and masks were drawn under
        self._privacy_hits = {}      # dot -> frames in which it was near the crop edge or inside a mask while recording
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
        self._timing = {"process": [], "arrival": []}       # per recorded frame: work time, arrival time
        self.force = None            # a limbtrack.force.ForceSource (load cell through a DAQ), or None
        self.auto_mvic = False       # mark MVIC start and end from the force (limbtrack/automvic.py); needs the force
        self.auto_mvic_threshold = DEFAULT_THRESHOLD_N   # newtons the force must rise from its rest level to call the start
        self.auto_mvic_seconds = DEFAULT_LENGTH_S        # the end marker comes this long after the start marker (protocol: 1-5-1 = 7 s)
        self._auto = None            # the AutoMvic of the present recording, or None
        self._rehearsal = None       # an AutoMvic trying the automatic markers and the beeps WITHOUT a recording (see rehearsal_start)
        self._reh_cursor, self._reh_scale, self._reh_began = 0.0, 1.0, 0.0
        self.audio_cues = True      # beeps for the recording, the ramp, the hold, the relax and the end of an MVIC (limbtrack/cues.py)
        self.cue_push_every_s = cues.PUSH_INTERVAL_S      # the motivating tick during the hold, in seconds; 0 = none
        self._cues = cues.CuePlayer()
        self._auto_cursor = 0.0      # force samples up to this time have been given to it
        self._auto_scale = 1.0
        self._pending_event_auto = False   # the marker waiting to be written was set by the automatic MVIC
        self.auto_stop_s = AUTO_STOP_AFTER_END_S   # the window stops the recording this long after the MVIC end marker; 0 = never
        self._mvic_end_t = None      # picture time of the latest MVIC end marker in this recording
        self._force_active = False   # this recording has force
        self._force_cursor = 0.0     # force samples up to this time have been handed to the recording
        self._force_stats = None
        self._force_gap_frames = 0
        self._force_clip0 = 0
        self.camera_desc = ""
        self.camera_info = None      # requested / granted camera settings, set by the video thread
        self.set_test(next(iter(tests)))

    # ----------------------------------------------------------------- set-up
    def set_test(self, key):
        with self.lock:
            self._forbid_while_recording("the test")
            old = self.test
            self.test = self.tests[key]
            if old is None or (old.tolerance_deg, old.neighbour_tolerance_deg) != (self.test.tolerance_deg, self.test.neighbour_tolerance_deg):
                self.tolerance = self.test.tolerance_deg            # (an operator's own tolerance is kept when the protocol's did not change)
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
            if old is not None and list(old.all_dots) == list(self.test.all_dots):
                # Same landmarks: the marked dots stay (the 120 s changeover is not spent re-clicking them); the position is
                # unlocked because the neighbour angles and their required values belong to the test (audit GUI-06).
                self._last_angles = []
                self._filters, self._ref_avg = {}, {}
                self._hist.clear()
                self.unlock_position()
            else:
                self._reset_dots_core()

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
            self.paused, self._pause_ghost = False, None
            # PARKED-GHOST: self._ghost = None

    # ------------------------------------------------------------------- break
    @property
    def pause_ghost(self):
        """Where the dots were when the break began (dot name -> (x, y, diameter in pixels)), or None when there is no ghost."""
        with self.lock:
            return dict(self._pause_ghost) if self._pause_ghost else None

    def pause(self):
        """Begin a break (the participant rests or leaves). Tracking stops, nothing is judged or recorded, and a ghost of the dots
        as they were shows where the stickers have to be again. Returns None if done, else the reason it was not."""
        with self.lock:
            if self.rec is not None:
                return "Stop recording before pausing: a break would leave a hole in the recording."
            if self.paused:
                return None
            lost_now = {n for n, st in self._last_states.items() if st.lost}
            self._pause_ghost = {n: (t.x, t.y, t.diam) for n, t in self.trackers.trackers.items()
                                 if n in self.trackers.enabled and t.initialised and n not in lost_now}
            self.paused = True
            return None

    def resume(self):
        """End the break. Every dot is frozen where it was and is taken back only when a convincing dot shows up inside its small
        circle (the lost-dot rule), so nothing is guessed. The ghost stays until every dot is back. Returns None."""
        with self.lock:
            if not self.paused:
                return None
            self.paused = False
            self.trackers.hold_all()
            if not self._pause_ghost:
                self._pause_ghost = None
            return None

    def cancel_pause(self):
        """Forget a break without holding the dots (the camera was stopped)."""
        with self.lock:
            self.paused, self._pause_ghost = False, None

    def _paused_frame(self, frame, t):
        """A picture during a break: nothing is tracked, recorded or judged; the ghost shows where the dots were."""
        head = ["PAUSED for a break: tracking is off and nothing is recorded.",
                "Put the stickers on the dashed circles, then press Resume (P)."]
        if self._notice and 0 <= t - self._notice[1] < 10.0:
            head.append(self._notice[0])
        img = overlay.draw(frame, {}, [], head, False, ghost=self._pause_ghost, ghost_segments=segment_pairs(self.active_angles))
        res = FrameResult(self.frame_index, t, img, [], dict(self._last_states), head, None)
        self.frame_index += 1
        return res

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
            if self.paused:
                return "The program is paused for a break. Press Resume (P) first, then lock the position."
            # Labels first: the side of the body is read from the foot dots, so swapped foot dots also look like the other side.
            bad = self.marking_problems()
            if bad:
                return "The dots look mislabelled: " + "; ".join(bad) + ". Use 'Re-mark all dots' and click them in the order asked."
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

    def dot_status(self):
        """For the list of landmarks: dot name -> 'off' (switched off), 'unmarked', 'lost' (frozen where it was last seen) or 'ok'."""
        with self.lock:
            out = {}
            for n in self.test.all_dots:
                st = self._last_states.get(n)
                out[n] = ("off" if n in self.disabled_dots else "unmarked" if not self.trackers.trackers[n].initialised
                          else "lost" if st is not None and st.lost else "ok")
            return out

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
            if self.paused:
                self.mark_problem = "Resume first (press P): dots cannot be marked during a break."
                return False
            if self._last_gray is None or name in self.disabled_dots:
                return False
            old = copy.deepcopy(self.trackers.trackers[name])      # a re-mark must never damage a working tracker
            if not self.trackers.init_dot(name, self._last_gray, x, y):
                return False
            t = self.trackers.trackers[name]
            # never onto a dot another tracker holds (first marking or re-mark)
            for other, o in self.trackers.trackers.items():
                if other == name or not o.initialised or other not in self.trackers.enabled or o.lost_frames:
                    continue
                if (o.x - t.x) ** 2 + (o.y - t.y) ** 2 < (NEIGHBOUR_EXCLUSION_FACTOR * o.diam) ** 2:
                    self.trackers.trackers[name] = old
                    self.mark_problem = (f"That is the {self.test.dot_title(other)} dot, which is already marked, not the "
                                         f"{self.test.dot_title(name)} dot.")
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
            if self.paused or self.facing_mismatch or self.marking_problems():
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
            if self._auto is not None and label in ("MVIC start", "MVIC end"):
                self._auto.manual()                       # the operator took the markers over
            self._pending_event, self._pending_event_auto = label, False
            return True

    # ------------------------------------------------------------ sound cues
    def _cue_now(self, name):
        if self.audio_cues:
            self._cues.play(name)

    def _cues_after_start(self, automatic, auto=None):
        """The ramp was registered (the MVIC start marker is being written): the cues that follow keep the 1-5-1 tempo."""
        if not self.audio_cues:
            return
        now = time.perf_counter()
        ref = now                                                   # a start marked by hand: the ramp is taken to begin now
        a = auto if auto is not None else self._auto
        if automatic and a is not None and a.trigger:
            onset = a.trigger.get("onset_t")
            if onset is not None and 0.0 <= now - onset <= 1.0:
                ref = onset                                         # the force began to rise a moment before it was registered
        self._cues.cancel()
        self._cues.schedule(cues.plan(ref, now, self.auto_mvic_seconds, push_every_s=self.cue_push_every_s))

    def warm_up_cues(self):
        """Wake the sound device (silent), so the first beep of a session is not late. Called when a camera starts and a trial begins."""
        if self.audio_cues:
            self._cues.warm_up()

    @property
    def cue_problem(self):
        """Why the latest beep could not be played, or None."""
        return self._cues.last_error

    def test_cues(self):
        """Play every cue once, in the order of a contraction, so the operator can hear them (also when they are switched off)."""
        now = time.perf_counter()
        self._cues.cancel()
        self._cues.last_error, self._cues.error_count = None, 0       # a test starts afresh: whatever fails now is reported now
        self._cues.schedule([(now + dt, name) for dt, name in ((0.0, "record"), (0.8, "ramp"), (1.6, "hold"), (2.5, "push"),
                                                              (3.0, "push"), (3.7, "relax"), (4.6, "end"))])

    def _auto_mvic_step(self, t):
        """Give the force samples that came in to the automatic MVIC, and write its start / end marker on this picture when due."""
        a = self._auto
        if a is None or self.rec is None or self.force is None or self.force.error:
            return
        ts, vs = self.force.samples_between(self._auto_cursor, t)
        if ts.size:
            self._auto_cursor = float(ts[-1]) + 1e-9
            a.feed(ts, vs * self._auto_scale)
        if self._pending_event is not None:               # a marker is already waiting for this picture: try again on the next
            return
        if a.state == TRIGGERED:
            self._pending_event, self._pending_event_auto = "MVIC start", True
            a.start_marked(t)
        elif a.end_is_due(t):
            self._pending_event, self._pending_event_auto = "MVIC end", True
            a.end_marked(t)

    # ------------------------------------------------------------ trying the automatic markers without a recording
    REHEARSAL_GIVE_UP_S = 60.0

    def rehearsal_start(self):
        """Arm the automatic MVIC and the beeps WITHOUT recording, so the whole chain (force -> start called -> beeps -> end) can be
        tried before a real trial and no empty trial is saved. Nothing is written anywhere. Returns None if armed, else the reason not."""
        with self.lock:
            if self.rec is not None:
                return "Stop recording first."
            if self.force is None or self.force.error:
                return "Connect the force first: the automatic markers read it."
            cal = self.force.calibration
            if cal.has_slope:
                unit, self._reh_scale, thr = "N", float(cal.n_per_v), self.auto_mvic_threshold
            else:
                unit, self._reh_scale, thr = "V", 1.0, DEFAULT_THRESHOLD_V
            a = AutoMvic(thr, self.auto_mvic_seconds, self.force.rate, unit)
            now = time.perf_counter()
            ts, vs = self.force.samples_between(now - KEEP_S, now)
            self._reh_cursor = float(ts[-1]) + 1e-9 if ts.size else now
            if ts.size:
                a.prime(ts, vs * self._reh_scale)
            self._rehearsal, self._reh_began = a, now
            self._cues.cancel()
            self._cues.last_error, self._cues.error_count = None, 0
            return None

    def rehearsal_stop(self):
        with self.lock:
            self._rehearsal = None
            self._cues.cancel()

    def rehearsal_step(self):
        """Call a few times a second. Returns (state, plain text): state is 'off' when no try-out is running, else the AutoMvic state
        ('waiting', 'running', 'done'); 'gave up' when no push came."""
        with self.lock:
            a = self._rehearsal
            if a is None:
                return "off", ""
            if self.rec is not None or self.force is None or self.force.error:
                self._rehearsal = None
                return "off", "The try-out stopped (a recording started or the force was lost)."
            now = time.perf_counter()
            ts, vs = self.force.samples_between(self._reh_cursor, now)
            if ts.size:
                self._reh_cursor = float(ts[-1]) + 1e-9
                a.feed(ts, vs * self._reh_scale)
            if a.state == TRIGGERED:
                a.start_marked(now)
                self._cues_after_start(True, auto=a)
            elif a.end_is_due(now):
                a.end_marked(now)
                self._cues.cancel()
                self._cue_now("end")
            if a.state == "waiting" and now - self._reh_began > self.REHEARSAL_GIVE_UP_S:
                self._rehearsal = None
                return "gave up", f"No push seen for {self.REHEARSAL_GIVE_UP_S:g} s. Press the button and push on the strap."
            tr = a.trigger or {}
            if a.state == "done":
                self._rehearsal = None
                return "done", (f"Start found {a.start_marker_t - tr.get('onset_t', a.start_marker_t):.2f} s after the force began to rise "
                                f"(a rise of {tr.get('rise_at_trigger', 0.0):.1f} {a.unit} over the rest level), end {a.length_s:g} s later. "
                                f"In a recording both markers would be written. {'Beeps played.' if self.audio_cues else 'Beeps are switched off.'}")
            if a.state == "running":
                return "running", f"Start found. {max(now - a.start_marker_t, 0.0):.1f} / {a.length_s:g} s"
            return "waiting", (f"Waiting for a push: the force has to rise by {a.threshold:g} {a.unit}" if not a.reason
                               else f"Getting ready: {a.reason}")

    def seconds_to_auto_stop(self):
        """Seconds (picture time) until the recording should stop by itself: `auto_stop_s` after the MVIC end marker, whether the
        automatic MVIC or the E key wrote it. None while that is not set (no recording, auto stop off, or no end marker yet);
        0 or less means it is due. The window does the stopping, exactly as if Stop had been pressed."""
        with self.lock:
            if self.rec is None or not self.auto_stop_s or self._mvic_end_t is None or not self._times:
                return None
            return self.auto_stop_s - (self._times[-1] - self._mvic_end_t)

    def auto_mvic_status(self):
        """(state, plain text) of the automatic MVIC for the window; ('off', '') when it is not in use in this recording."""
        with self.lock:
            a = self._auto
            if a is None or self.rec is None:
                return "off", ""
            return a.state, a.status_text(self._times[-1] if self._times else 0.0)

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
        self._privacy_crop, self._privacy_masks = None, []      # drawn on the old picture: no longer valid
        self._notice = (text + ": mark the dots (and the ruler) again. The privacy crop and masks were cleared.", t)

    # Distances that hold for any body posture (upright, seated, lying) and either leg: (longer pair, shorter pair, what is wrong,
    # smallest allowed ratio longer/shorter). They catch dots clicked in the wrong order or two labels swapped (audit GUI-14).
    LABEL_CHECKS = [
        (("malleolus", "mt5_head"), ("malleolus", "mt5_base"),
         "the 5th metatarsal head should be farther from the lateral malleolus than the base is (head and base swapped?)", 1.0),
        (("fibular_head", "malleolus"), ("mt5_base", "mt5_head"),
         "the foot (metatarsal base to head) should be shorter than the shank (fibular head to malleolus)", 1.0),
        (("trochanter", "epicondyle"), ("iliac_crest", "trochanter"),
         "the thigh (trochanter to epicondyle) should be longer than the pelvis arm (iliac crest to trochanter)", 1.0),
        (("epicondyle", "malleolus"), ("fibular_head", "malleolus"),
         "the lateral malleolus should be farther from the epicondyle than from the fibular head (epicondyle and fibular head swapped?)", 0.97),
    ]

    def marking_problems(self):
        """Plain-English reasons to doubt that the dots were marked in the right order (empty if all is plausible)."""
        with self.lock:
            xy = {n: (st.x, st.y) for n, st in self._last_states.items() if not st.lost and n in self.active_dots}
        out = []
        for (a1, a2), (b1, b2), text, ratio in self.LABEL_CHECKS:
            if all(n in xy for n in (a1, a2, b1, b2)):
                dl = math.hypot(xy[a1][0] - xy[a2][0], xy[a1][1] - xy[a2][1])
                ds = math.hypot(xy[b1][0] - xy[b2][0], xy[b1][1] - xy[b2][1])
                if ds > 1e-6 and dl < ratio * ds:
                    out.append(text)
        return out

    FACING_MISMATCH_FRAMES = 5

    def _check_facing(self, states):
        """Compare the side of the body the dots show with the 'Participant faces' setting (audit ANG-04)."""
        xy = {n: (st.x, st.y) for n, st in states.items() if not st.lost}
        h = view_handedness(xy)
        self.facing_detected = h
        self._facing_from_knee = h is not None and foot_handedness(xy) is None
        expected = 1 if self.facing_right else -1
        self._facing_bad_frames = self._facing_bad_frames + 1 if (h is not None and h != expected) else 0

    @property
    def facing_mismatch(self):
        """True when the marked dots clearly show the other side of the body than 'Participant faces' says."""
        return self._facing_bad_frames >= self.FACING_MISMATCH_FRAMES

    def facing_message(self):
        shown = "RIGHT" if self.facing_detected == 1 else "LEFT"
        chosen = "RIGHT" if self.facing_right else "LEFT"
        if self._facing_from_knee:
            return (f"CHECK 'Participant faces': the knee bends the wrong way for {chosen} (it would read as over-straightened by more "
                    f"than {KNEE_FACING_MIN_DEG:g} deg). Pick {shown}: lying face down usually needs the other side than sitting or lying "
                    "on the back. The hip and knee signs would be wrong.")
        return (f"CHECK 'Participant faces': the dots look like a {shown}-facing view of the picture, but {chosen} is selected. "
                "The hip sign would be wrong.")

    # ------------------------------------------------------------------- force
    def _force_header(self):
        """The force read-out for the picture (and the saved overlay video), or a warning when the DAQ is not delivering."""
        f = self.force
        if f is None:
            return []
        if f.error:
            return ["FORCE: " + f.error]
        if self.rec is not None and f.seconds_since_last_sample() > 0.5:
            return ["FORCE: no data from the DAQ"]
        if f.clipping_now():
            return [f"FORCE SIGNAL AT THE +-{f.range_v:g} V LIMIT: the peak may be cut off (lower the amplifier gain)"]
        n, v = f.latest_newtons(), f.latest_volts()
        if v is None:
            return ["Force: waiting for samples"]
        if n is None:
            return [f"Force {v:+.3f} V (press Zero for newtons)" if f.calibration.has_slope else f"Force {v:+.3f} V (not calibrated)"]
        peak = self._force_stats["peak"] if self._force_active and self._force_stats and not math.isnan(self._force_stats["peak"]) else None
        return [f"Force {n:.0f} N ({v:+.3f} V)" + (f"   peak {peak:.0f} N" if peak is not None else "")]

    def _take_force(self, t):
        """Hand the force samples that came in since the last picture to the recording (all remaining ones when t is None).
        Returns (mean volts, mean newtons) for the CSV row; either is None when not available (newtons: no calibration)."""
        if not self._force_active or self.rec is None:
            return None, None
        ts, vs = self.force.samples_between(self._force_cursor, float("inf") if t is None else t)
        if ts.size == 0:
            if t is not None and self.force.seconds_since_last_sample() > 0.5:
                self._force_gap_frames += 1
            return None, None
        self._force_cursor = float(ts[-1]) + 1e-9
        ns = self.force.calibration.to_newton(vs)
        self.rec.write_force(ts, vs, ns)
        st = self._force_stats
        st["n"] += int(ts.size)
        st["first"] = float(ts[0]) if st["first"] is None else st["first"]
        st["last"] = float(ts[-1])
        mean_v = float(np.mean(vs))
        if np.all(np.isnan(ns)):
            return mean_v, None
        st["sum"] += float(np.nansum(ns))
        k = int(np.nanargmax(ns))
        if math.isnan(st["peak"]) or ns[k] > st["peak"]:
            st["peak"], st["peak_clock"] = float(ns[k]), float(ts[k])
        return mean_v, float(np.nanmean(ns))

    def _force_meta(self):
        """What the force recording contained, for meta.json."""
        st, src = self._force_stats, self.force
        dur = (st["last"] - st["first"]) if st["first"] is not None and st["n"] > 1 else None
        t0 = self.rec._t0 if self.rec is not None else None
        out = dict(src.info())
        out.update({
            "samples": st["n"],
            "duration_s": round(dur, 4) if dur else None,
            "measured_rate_hz": round((st["n"] - 1) / dur, 2) if dur else None,
            "peak_n": None if math.isnan(st["peak"]) else round(st["peak"], 3),
            "peak_t_s": round(st["peak_clock"] - t0, 4) if st["peak_clock"] is not None and t0 is not None else None,
            "mean_n": round(st["sum"] / st["n"], 3) if src.calibration.calibrated and st["n"] else None,
            "frames_without_samples_over_0_5s": self._force_gap_frames,
            "samples_at_range_limit": src.clipped_total - self._force_clip0,
            "range_v": src.range_v,
            "read_error": src.error,
            "files": "force.csv has every sample: t_s (seconds after the first picture, like data.csv), clock_s, volts, newtons",
            "clock": "pictures and force samples both use time.perf_counter(). The DAQ's sample 0 is placed from the earliest chunk arrival "
                     "over the last 20 s, so the alignment is good to a few milliseconds (the smallest USB and driver delay). The MVIC "
                     "start marker is an independent check.",
            "units": "volts are always saved; newtons need a calibration (zero and a known load), otherwise they are blank",
        })
        return out

    # ----------------------------------------------------------------- privacy
    MAX_MASKS = 3

    def set_privacy_rect(self, kind, x0, y0, x1, y1):
        """Set the privacy crop ('crop') or add a black mask ('mask') from a dragged rectangle in picture pixels.
        Returns (ok, message). Only the saved videos are affected; tracking and the CSV use the full picture."""
        with self.lock:
            if self._last_gray is None:
                return False, "No picture yet."
            h, w = self._last_gray.shape[:2]
            rect = privacy.snap_rect(x0, y0, x1, y1, w, h)
            if rect is None:
                return False, f"That rectangle is too small (at least {privacy.MIN_SIDE_PX} px each way). Drag a larger one."
            if kind == "mask":
                if len(self._privacy_masks) >= self.MAX_MASKS:
                    return False, f"At most {self.MAX_MASKS} masks. Clear them first."
                self.privacy_masks = list(self._privacy_masks) + [rect]
            else:
                self.privacy_crop = rect
            self._privacy_mirror = self.mirror
            return True, ""

    def clear_privacy(self):
        with self.lock:
            self.privacy_crop, self.privacy_masks = None, []

    def privacy_problem(self):
        """Why recording must not start with the present crop and masks (empty or None if all is well)."""
        with self.lock:
            crop, masks = self._privacy_crop, self._privacy_masks
            if crop is None and not masks:
                return None
            if self._last_gray is None:
                return "No picture yet."
            h, w = self._last_gray.shape[:2]
            if self._privacy_mirror != self.mirror:
                return "The mirror setting changed after the privacy crop and masks were drawn. Clear them and draw them again."
            if crop and (crop[0] + crop[2] > w or crop[1] + crop[3] > h):
                return "The privacy crop does not fit the picture size. Clear it and draw it again."
            for n, st in self._last_states.items():
                if st.lost or n not in self.active_dots:
                    continue
                title = self.test.dot_title(n)
                if crop and privacy.edge_distance(crop, st.x, st.y) < privacy.DOT_EDGE_MARGIN_PX:
                    return (f"The {title} dot is outside or at the edge of the privacy crop (closer than {privacy.DOT_EDGE_MARGIN_PX} px). "
                            "Make the crop larger.")
                for m in masks:
                    if privacy.mask_distance(m, st.x, st.y) <= 0:
                        return f"The {title} dot is inside a black mask. Clear the masks and draw them again."
            return None

    def _privacy_check(self, states):
        """While recording: header warnings for dots that come near the crop edge or into a mask; counts frames for meta.json."""
        crop, masks = self._privacy_crop, self._privacy_masks
        out = []
        if crop is None and not masks:
            return out
        for n, st in states.items():
            if st.lost or n not in self.active_dots:
                continue
            near = bool(crop) and privacy.edge_distance(crop, st.x, st.y) < privacy.DOT_WARN_MARGIN_PX
            hidden = any(privacy.mask_distance(m, st.x, st.y) <= 0 for m in masks)
            if near or hidden:
                self._privacy_hits[n] = self._privacy_hits.get(n, 0) + 1
                out.append(f"PRIVACY: {self.test.dot_title(n)} dot " + ("inside a mask" if hidden else "at the crop edge"))
        return out

    def _privacy_meta(self):
        size = None
        if self._last_gray is not None:
            size = [int(self._last_gray.shape[1]), int(self._last_gray.shape[0])]
        return {"crop_xywh": list(self._privacy_crop) if self._privacy_crop else None,
                "masks_xywh": [list(m) for m in self._privacy_masks],
                "full_picture_size": size,
                "meaning": "Only the cropped part of the picture is saved to raw.mp4 and overlay.mp4; the masks are black in the saved "
                           "videos. Tracking ran on the full picture, and all pixel positions in the CSV are in the FULL picture "
                           "(add nothing, subtract nothing). Review pads a cropped video back to full size."}

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
        """One frame in, everything out. While recording, how long each frame took and when it arrived is kept for meta.json."""
        t_in = time.perf_counter()
        res = self._process(frame, t, source_frame)
        if self.rec is not None:
            self._timing["process"].append(time.perf_counter() - t_in)
            self._timing["arrival"].append(t_in)
        return res

    def _timing_meta(self):
        """Where the time went during the recording: the computer's work per frame against the time between frames."""
        proc = np.array(self._timing["process"]) * 1000.0
        gaps = np.diff(np.array(self._timing["arrival"])) * 1000.0
        if len(proc) < 3 or len(gaps) < 2:
            return None
        return {
            "frames": int(len(proc)),
            "process_ms": {"mean": round(float(proc.mean()), 2), "p95": round(float(np.percentile(proc, 95)), 2), "max": round(float(proc.max()), 1)},
            "frame_interval_ms": {"median": round(float(np.median(gaps)), 2), "p95": round(float(np.percentile(gaps, 95)), 2),
                                  "max": round(float(gaps.max()), 1)},
            "loop_fps": round(1000.0 / float(gaps.mean()), 2),
            "process_share_of_interval": round(float(proc.mean() / gaps.mean()), 3),
            "meaning": "process_ms = this program's work per frame (tracking, drawing, saving). If it is much smaller than the frame "
                       "interval, the computer kept up and the camera or phone delivered fewer frames than asked for.",
        }

    def _process(self, frame, t, source_frame=None):
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
            if self.paused:
                return self._paused_frame(frame, t)
            states = self.trackers.update(gray)
            self._last_states = states
            if self._pause_ghost is not None and not any(st.lost for st in states.values()):
                self._pause_ghost = None                       # after a break: every dot is back on its circle, the ghost has done its job
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
            for text in self.marking_problems():
                head.append("CHECK LABELS: " + text)
            pull = self._compute_pull(states)
            if self.pull_active:
                head.append(describe_pull(pull) if pull else "PULL: not available (a strap, anchor or segment dot is lost or off)")
            for problem in self.scale.problems():
                head.append("SCALE: " + problem)
            lost = [n for n, st in states.items() if st.lost]
            if lost:
                head.append("DOT LOST: " + ", ".join(self.test.dot_title(n) for n in lost))
            if self._pause_ghost is not None:
                head.append("After the break: put the stickers on the pale dashed circles; each dot is tracked again once it is there.")
            partial = [n for n, st in states.items() if st.partial]
            if partial:
                head.append("dot partly covered or cut off (its centre may be shifted): " + ", ".join(self.test.dot_title(n) for n in partial))
            weak = [n for n, st in states.items() if st.weak]
            if weak:
                head.append("weak dot (fading): " + ", ".join(self.test.dot_title(n) for n in weak))
            self._auto_mvic_step(t)
            event, self._pending_event = self._pending_event, None
            auto_event, self._pending_event_auto = self._pending_event_auto and bool(event), False
            if event == "MVIC start":
                self._cues_after_start(auto_event)
            elif event == "MVIC end":
                self._cues.cancel()
                self._cue_now("end")
            if event == "MVIC end" and self.rec is not None:
                self._mvic_end_t = t
            if event:
                self._banner = (event + (" (automatic)" if auto_event else ""), t)
            if self._banner and 0 <= t - self._banner[1] < 1.5:
                head.append(f"MARK: {self._banner[0]}")
            nxt = self.next_dot_to_mark()
            if nxt:
                head = [self.test.label, f"{overlay.PROMPT_PREFIX}{self.test.dot_title(nxt)}"]
            if self._notice and 0 <= t - self._notice[1] < 10.0:
                head.append(self._notice[0])
            if self.rec is not None:
                head.extend(self._privacy_check(states))
            head.extend(self._force_header())
            if self._auto is not None and self.rec is not None:
                head.append(self._auto.status_text(t))
            # PARKED-GHOST: ghost = None
            # PARKED-GHOST: if self.locked and self.ghost_enabled and self._ghost:
                # PARKED-GHOST: live = {n: (st.x, st.y) for n, st in states.items() if not st.lost}
                # PARKED-GHOST: anchor = self._ghost_anchor() if self.ghost_anchored else None
                # PARKED-GHOST: ghost = anchored_ghost(self._ghost, live, anchor)
            if self.rec is not None:
                self._update_trails(states)
            img = overlay.draw(frame, states, angles, head, self.rec is not None,
                               trails=self._trails if self.trails_enabled else None, scale=self.scale.overlay_info(),
                               ghost=self._pause_ghost, ghost_segments=segment_pairs(self.active_angles))
            # PARKED-GHOST: # (with the ghost: overlay.draw(frame, states, angles, head, self.rec is not None, ghost,
            # PARKED-GHOST: #                               segment_pairs(self.active_angles), trails=...))
            if self.rec is not None:
                live_scale = self.scale.live_px_per_cm()
                self._scale_values.append(live_scale)
                try:
                    force_v, force_n = self._take_force(t)
                    self.rec.write(frame, img, self.rec.frames, t, self.locked, states, angles, event or "", live_scale, pull,
                                   source_frame, force_n=force_n, force_v=force_v)
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
            if self.paused:
                raise RuntimeError("The program is paused for a break. Press Resume (P) first, then record.")
            if not self.active_angles:
                raise RuntimeError("None of the angles can be computed with the chosen landmarks.")
            if not self.trackers.is_ready():
                raise RuntimeError("Mark all dots before recording.")
            problem = self.privacy_problem()
            if problem:
                raise RuntimeError(problem)
            if self.auto_mvic and self.force is None:
                raise RuntimeError("The automatic MVIC markers need the force reading. Connect the force, or switch them off.")
            if self.force is not None:
                if (self.camera_info or {}).get("is_file"):
                    raise RuntimeError("Force is recorded only with a live camera: a video file is not in step with the DAQ clock. "
                                       "Disconnect the force first to record from a file.")
                if not self.force.running or self.force.error:
                    raise RuntimeError("The force reading is not running" + (f": {self.force.error}" if self.force.error else ".")
                                       + " Connect it again, or disconnect it, before recording.")
            if frame_size is None:
                h, w = self._last_gray.shape[:2]
                frame_size = (w, h)
            self._privacy_hits = {}
            self._rehearsal = None                            # a try-out of the markers ends where the real thing begins
            self._pause_ghost = None                          # a trial's overlay video must not carry the ghost of a break
            self.rec = TrialRecorder(
                self.recordings_dir, participant or "unnamed", self.test.key,
                self.target if self.primary_angle else None,
                self.active_dots, [a.name for a in self.active_angles], self._fps(), frame_size, with_pull=self.pull_active,
                save_overlay=self.save_overlay, crop=self._privacy_crop, masks=self._privacy_masks, with_force=self.force is not None)
            self._marker_n = 0
            self._pending_event = None
            self._mvic_end_t = None
            self._trails, self._trail_open = {}, {}          # trails cover one recording: start afresh
            self._scale_values = []
            self._force_active = self.force is not None
            if self._force_active:
                self.force.frozen = True                  # the zero and the calibration cannot change during the recording
                self._force_cursor = time.perf_counter()
                self._force_stats = {"n": 0, "sum": 0.0, "peak": math.nan, "peak_clock": None, "first": None, "last": None}
                self._force_gap_frames = 0
                self._force_clip0 = self.force.clipped_total
            self._arm_auto_mvic()
            self._timing = {"process": [], "arrival": []}
            self.rec_info = {"participant": participant, "notes": notes}
            self._meta_start = copy.deepcopy(self._trial_meta())      # snapshot: meta.json reports these start values
            self.last_stop_reason = None
            self.rec.write_start_meta(self._meta_start)               # a complete meta.json right away (status 'recording')
            # PARKED-GHOST: # with the ghost: self.rec_info = {"participant": participant, "notes": notes,
            # PARKED-GHOST: #                                  "ghost": dict(self._ghost) if self.locked and self._ghost else None}
            self._cues.cancel()
            self._cue_now("record")                           # only now: the recording has really started
            return self.rec.paths

    def _arm_auto_mvic(self):
        """Pressing Record arms the automatic MVIC: it starts from the last few seconds of force, so a ramp that begins at once is found."""
        self._auto = None
        if not self.auto_mvic or self.force is None:
            return
        cal = self.force.calibration
        if cal.has_slope:                                 # newtons without the zero: only the rise from the rest level is used
            unit, self._auto_scale, thr = "N", float(cal.n_per_v), self.auto_mvic_threshold
        else:
            unit, self._auto_scale, thr = "V", 1.0, DEFAULT_THRESHOLD_V
        self._auto = AutoMvic(thr, self.auto_mvic_seconds, self.force.rate, unit)
        now = time.perf_counter()
        ts, vs = self.force.samples_between(now - KEEP_S, now)
        self._auto_cursor = float(ts[-1]) + 1e-9 if ts.size else now
        if ts.size:
            self._auto.prime(ts, vs * self._auto_scale)

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
            "meta_schema": META_SCHEMA,
            "provenance": provenance.collect(str(self.protocol_path) if self.protocol_path else None),
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
            "privacy": self._privacy_meta(),
            "force_setup": self.force.info() if self.force is not None else None,
            "auto_mvic_setup": ({"enabled": True, "threshold_n": self.auto_mvic_threshold, "length_s": self.auto_mvic_seconds}
                                if self.auto_mvic else {"enabled": False}),
            "audio_cues": ({"enabled": True, "length_s": self.auto_mvic_seconds, "ramp_s": cues.RAMP_S, "relax_s": cues.RELAX_S,
                            "push_every_s": self.cue_push_every_s, "cues": sorted(cues.CUES),
                            "note": "beeps the participant may have heard; times after the ramp follow the 1-5-1 tempo from the force onset (or the S key)"}
                           if self.audio_cues else {"enabled": False}),
            "privacy_dot_warnings": {"frames_near_crop_edge_or_in_mask": dict(self._privacy_hits)},
            "camera_setup_checklist": dict(self.camera_setup),
            "camera_check": self.camera_check,
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
            due = self.seconds_to_auto_stop()
            if due is not None and due <= 0:
                meta["stopped_automatically"] = {"after_mvic_end_s": self.auto_stop_s,
                                                 "meaning": "the recording was stopped by the program this long after the MVIC end marker"}
            meta["pipeline_timing"] = self._timing_meta()
            if self._force_active:
                self._take_force(None)                    # the last samples
                meta["force"] = self._force_meta()
            if self._auto is not None:
                meta["auto_mvic"] = self._auto.result(self.rec._t0)
            if self.audio_cues and self._cues.last_error:
                meta["audio_cue_problem"] = {"error": self._cues.last_error, "times": self._cues.error_count,
                                             "meaning": "a beep could not be played during or before this recording; the participant may not have heard every cue"}
            rec, self.rec = self.rec, None                      # cleared first: a failure while saving must not leave 'recording' on
            self._cues.cancel()                                 # no cue after the recording has ended
            self._auto = None
            if self.force is not None:
                self.force.frozen = False
            self._force_active = False
            return rec.close(meta)
