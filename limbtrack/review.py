"""Review a saved trial: scrub the video with its overlay, see angle-vs-time, get summary numbers."""
import csv
import json
import math
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QLineEdit, QMenu, QSplitter, QStackedWidget, QTextEdit, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

from . import angle_reference, dot_edits, overlay, privacy, reference, review_layer, review_nav, session_memory
from . import verdict as verdicts
from .twoaxis import TwoAxisPlot
from .review_checks import ChecksChip
from .review_decide import VerdictControl
from .review_force import ForceTree
from .review_angles import AngleDotTree
from .review_fixdots import FixDotsPane
from .review_privacy import PrivacyDialog
from .review_overview import OverviewPage
from .review_style import ElidedLabel, build_stylesheet
from .displacement import window_displacements
from . import analysis
from .force import NO_FORCE_RISE_N, SMOOTH_DEFAULT_S, WINDOW_PEAK_DEFAULT_S, ForceCalibration, mains_amplitude, smooth_signal
from .loadcell import COVERAGE, GRAVITY, active_regression, find_regression
from .pull import COLUMNS as PULL_COLUMNS
from .angles import measure, segment_pairs
# PARKED-GHOST: from .geometry import anchored_ghost
from .protocol import AngleDef
from .results import INFO, LOST, OK, OUT
from .engine import SOFTWARE_VERSION, TRAIL_MIN_STEP_PX
from .tracker import WEAK_BELOW

PENS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]
CURSOR_COLOUR = "#ffd43b"        # the line that follows the video position: it was black on the black plot background until 0.14.0
EVENT_GREY = (190, 190, 190)     # MVIC start/end and other markers (was dark grey (90, 90, 90), hard to see on black)
DEFAULT_TOP_SPLIT = [630, 370]   # the trial (video, angle table) : the signal (graph, force table): the video side is the wider one since 0.33.0
DEFAULT_MAIN_SPLIT = [620, 320]  # video and its controls : the table of angles and their dots under it (0.33.0: more for the video, the table scrolls)
RIGHT_LEAST_WIDTH, FIX_LEAST_WIDTH = 460, 320   # the narrowest the graphs side and the dot tools may be made by the divider
FIX_TOP_SPLIT = [700, 300]       # while the dots are fixed: the video gets about 70 %, the dot tools the rest
LAYOUT_VERSION = 2               # 0.33.0: divider positions saved before it are not put back (they would hide the new proportions)


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def read_force_csv(path):
    """(t_s, volts, newtons) arrays of a trial's force.csv, NaN where a cell is empty or not a number (newtons are empty without a calibration).
    The same numbers the csv.DictReader gave before 0.29.0 (which took about a second for a 140 000-sample file, the largest share of
    opening a trial and of the whole participant overview); a line cut short by a crash is padded with empty cells, not dropped.
    Returns None when the file has no sample."""
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        if header is None:
            return None
        try:                                    # the usual file (every cell a number): numpy's own reader, several times faster
            usecols = [header.index(c) for c in ("t_s", "volts", "newtons")]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")           # (a file with a header only: 'input contained no data')
                arr = np.loadtxt(fh, delimiter=",", usecols=usecols, dtype=np.float64, ndmin=2)
            return None if arr.shape[0] == 0 else (arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy())
        except (ValueError, IndexError):        # an empty or odd cell, a short line: read it cell by cell below
            fh.seek(0)
            next(csv.reader(fh), None)
            rd = csv.reader(fh)
        n = len(header)
        rows = [r if len(r) == n else (r + [""] * (n - len(r)))[:n] for r in rd if r]
    if not rows:
        return None
    cols = list(zip(*rows))

    def column(name):
        values = cols[header.index(name)]
        try:
            return np.array(values, dtype=np.float64)
        except ValueError:
            return np.array([_f(v) for v in values], dtype=np.float64)
    return column("t_s"), column("volts"), column("newtons")


PEAK_20MS_S = 0.02         # 0.31.0: 'peak (20 ms)': the highest 20 ms average of the raw force, the same at any Smoothing: one mains cycle, so the
                           # 50 Hz line and every multiple of it (100, 200, 400 Hz ...) cancel exactly. The nearest thing to 'the highest single sample' that
                           # hum does not decide (a raw single sample is median 11 N, up to 45 N, too high in the 30 Oct hum: LoadCellCalibration/noise, PR 5)
PLATEAU_TRIM = 0.2         # the plateau of a hold is its middle part: the first and last 20 % of the window are left out
REST_SECONDS = 1.0         # without a clear push in the force: "rest" is the second before the window starts
REST_MIN_FRAMES = 5
# The resting level before the push (0.20.0; SOP 9.10). Until 0.19.0 it was the second before the window 'force above half of peak',
# which holds about half of the ramp: on P11 the peak above rest came out 6-16 % too low.
REST_FROM_S = 3.0          # the rest is the force from 3.0 s ...
REST_TO_S = 0.5            # ... to 0.5 s before the push begins (the same span the automatic MVIC marker takes its rest level from)
REST_MIN_S = 0.5           # less force than this before the push: no resting level (the automatic marker needs the same)
REST_FULL_S = REST_FROM_S - REST_TO_S
PUSH_RISE_FRACTION = 0.10  # the push begins where the force was last within 10 % of its rise above the low level before it ...
PRE_LEVEL_PERCENTILE = 10  # ... that low level being the 10th percentile of the force before it passed half of its peak
REST_UNSTEADY_FRACTION = 0.10   # a rest whose spread (SD) is more than this share of the rise is called not steady


class Trial:
    """A recorded trial loaded from disk."""

    def __init__(self, meta_path, open_video=True, source="original"):
        """`open_video` False: for numbers only (the participant overview and the exports of 0.29.0); no video is opened, `frame` gives None.
        `source` (0.30.0): 'original' reads the recorded numbers as they are; 'reviewed' reads them through the trial's edit list in the
        review layer (limbtrack/review_layer.py): the force zero typed or taken in Review and the dot fixes. The files on disk are never changed
        either way: the edits are applied to the copy of the numbers held here."""
        meta_path = Path(meta_path)
        self.meta_path = meta_path
        self.source = source if source in ("original", "reviewed") else "original"
        self.review = None                                               # review_layer.Review in reviewed mode
        self.edited = {}                                                 # {dot: [picture numbers]} whose position was fixed in Review
        self.edited_sets = {}                                            # the same as sets (to ask 'was this picture fixed?')
        self.zero_edit = None                                            # the force zero set in Review (a dict), or None
        self.review_notes = []                                           # plain lines about what the review changed
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.reference_side = reference.read_sidecar(meta_path)        # reference readings added or changed later in Review (never in meta.json)
        self._reference_rows = None                                      # (worked out on first use, see reference_rows)
        folder = meta_path.parent
        self.base = meta_path.name[: -len("_meta.json")]
        self.folder = folder
        with open(folder / self.meta["files"]["data.csv"], newline="", encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))
        if not self.rows:
            raise ValueError("This trial contains no frames (nothing was recorded).")
        self.t = np.array([_f(r["t_s"]) for r in self.rows])
        self.unfinished = self.meta.get("status") == "recording"     # the program stopped before it could write the final file
        if self.unfinished:
            self.meta["frames"], self.meta["duration_s"] = len(self.rows), round(float(self.t[-1]), 3)
        self.angle_defs = [AngleDef(**a) for a in self.meta["angles"]]
        self.facing_right = self.meta.get("facing", "right") == "right"
        self.dot_names = [k[:-5] for k in self.rows[0] if k.endswith("_lost")]
        # computed points (the hip centre of a frontal test, 0.15.0): positions in the CSV, none in trials recorded before
        self.derived_names = [n for n in ((self.meta.get("derived_points") or {}).get("definitions") or {}) if f"{n}_x" in self.rows[0]]
        self.deg = {a.name: np.array([_f(r[f"{a.name}_deg"]) for r in self.rows]) for a in self.angle_defs}
        self.ref = {a.name: np.array([_f(r[f"{a.name}_ref"]) for r in self.rows]) for a in self.angle_defs}
        self.dev = {a.name: np.array([_f(r[f"{a.name}_dev"]) for r in self.rows]) for a in self.angle_defs}
        self.ok = {a.name: [r[f"{a.name}_ok"] for r in self.rows] for a in self.angle_defs}
        self.events = [(i, self.t[i], r["event"]) for i, r in enumerate(self.rows) if r.get("event")]
        # dot positions over the whole trial (NaN where a dot is lost) and the ruler scale, for the displacement numbers
        self.tracks = {n: (np.array([_f(r[f"{n}_x"]) for r in self.rows]), np.array([_f(r[f"{n}_y"]) for r in self.rows]))
                       for n in self.dot_names}
        self.derived_tracks = {n: (np.array([_f(r[f"{n}_x"]) for r in self.rows]), np.array([_f(r[f"{n}_y"]) for r in self.rows]))
                               for n in self.derived_names}          # (the hip centre, kept apart from the dots: it has no lost or quality columns)
        self.px_per_cm = (self.meta.get("scale") or {}).get("px_per_cm_at_marking")
        self.ruler_dots = (self.meta.get("scale") or {}).get("ruler_dots_px_at_marking")      # two fixed points, full-picture pixels
        self.ruler_cm = (self.meta.get("scale") or {}).get("ruler_length_cm")
        self.pull = ({c: np.array([_f(r[c]) for r in self.rows]) for c in PULL_COLUMNS} if PULL_COLUMNS[0] in self.rows[0] else None)
        # trails: per dot, segments of (frame index array, xy array); a lost dot ends a segment (as when recording)
        self.trail_segments = {n: self.build_trail(n) for n in self.dot_names}
        # PARKED-GHOST: self.ghost = self.meta.get("locked_positions_px")
        # PARKED-GHOST: self.ghost_anchor = self.meta.get("ghost_anchor")
        self.cap = cv2.VideoCapture(str(folder / self.meta["files"]["raw.mp4"])) if open_video else None
        self._pos = -2
        # force (load cell through the DAQ): every sample, on the same time axis as the video (seconds after the first picture)
        self.force_t = self.force_v = self.force_n = None
        self.force_smooth_s = SMOOTH_DEFAULT_S        # moving average for the force curve and numbers (a multiple of 20 ms removes 50 Hz hum)
        self.window_peak_s = WINDOW_PEAK_DEFAULT_S    # length of the sliding window of the sliding-window peak (0.29.0, analysis.window_peak)
        fmeta = self.meta.get("force") or {}
        self.force_cal = fmeta.get("calibration")
        self.regression = find_regression((self.force_cal or {}).get("regression_id"))
        self.force_cal_note = ("calibration recorded with the trial" if (self.force_cal or {}).get("n_per_v") else
                               "no calibration yet: volts only")
        fname = (self.meta.get("files") or {}).get("force.csv")
        if fname and (folder / fname).exists():
            got = read_force_csv(folder / fname)
            if got is not None:
                self.force_t, self.force_v, self.force_n = got                    # (newtons NaN where there was no calibration)
        pv = self.meta.get("privacy") or {}        # a cropped video is put back at its place on a black full-size picture
        self.crop = tuple(pv["crop_xywh"]) if pv.get("crop_xywh") else None
        self.full_size = tuple(pv["full_picture_size"]) if pv.get("full_picture_size") else None
        self._cal_recorded = dict(self.force_cal) if self.force_cal else None      # what was recorded, kept when a zero is set in Review
        if self.source == "reviewed":
            self.apply_review(review_layer.Review.load(meta_path))
        # angle reference (0.33.0): `deg`, `ref` and `dev` stay the dots' angle as recorded; `offsets` is what the angles are shown from
        self.recorded_offsets = angle_reference.offsets_from_meta(self.meta)    # the live reference the trial was recorded with (data.csv's target column moved by it)
        self.offsets = {}
        self.reload_offsets()

    def build_trail(self, n):
        """Segments of (picture numbers, positions) of dot `n` from the rows as they are now; a lost dot ends a segment (as when recording)."""
        segs, cur = [], None
        for i, r in enumerate(self.rows):
            if r[f"{n}_x"] == "" or r[f"{n}_lost"] == "1":
                cur = None
                continue
            x, y = _f(r[f"{n}_x"]), _f(r[f"{n}_y"])
            if cur is None:
                cur = ([i], [(x, y)])
                segs.append(cur)
            elif (x - cur[1][-1][0]) ** 2 + (y - cur[1][-1][1]) ** 2 >= TRAIL_MIN_STEP_PX ** 2:
                cur[0].append(i)
                cur[1].append((x, y))
        return [(np.array(a), np.array(b)) for a, b in segs]

    # ------------------------------------------------------------------ the review layer (0.30.0)
    def apply_review(self, review):
        """Read the trial through `review` (a review_layer.Review): the force zero first (the newtons everything else uses), then the dot fixes.
        Only the copy of the numbers in this object changes."""
        self.review = review
        problems = []
        z = review.zero
        if z and self.has_force():
            problem = self._apply_zero_edit(z)
            if problem:
                problems.append(problem)
        if review.dots:
            res = dot_edits.apply(self, review)
            problems += [f"dot {n}: not a dot of this trial, the edit is ignored" for n in res["skipped"]]
        if review.exists():
            changed = review.originals_changed()
            if changed:
                problems.append("WARNING: the original file(s) changed after this review was made: " + ", ".join(changed))
        self.edited_sets = {d: set(v) for d, v in self.edited.items()}
        self.review_notes = review.summary_lines() + problems
        if not review.has_edits():
            self.source = "original"                                     # nothing was changed: this is the recorded trial, and says so

    def _apply_zero_edit(self, z):
        """Newtons from the recorded volts with the zero set in Review. Returns a sentence when it could not be applied, else None."""
        n_per_v = z.get("n_per_v")
        if not n_per_v:
            return "The force zero edit has no slope: the force stays as recorded."
        cal = ForceCalibration(n_per_v=float(n_per_v), method=f"slope {z.get('from', 'recorded')}" + (f" ({z['regression_id']})" if z.get("regression_id") else ""),
                               regression_id=z.get("regression_id") or "", zero_set=False)
        cal.set_zero(float(z["zero_v"]))
        self.apply_calibration(cal, f"applied later: zero set in Review ({z.get('source')}) {z['zero_v']:+.4f} V; slope {z.get('from', 'recorded')}"
                                    + (f", regression {z['regression_id']}" if z.get("regression_id") else ""))
        self.zero_edit = dict(z)
        return None

    def slope_for_zero(self):
        """The slope a zero set in Review is combined with: the one recorded with the trial, else the active load cell regression's, else None
        (no newtons can be given). {'n_per_v', 'regression_id', 'from'}."""
        rec = self._cal_recorded or {}
        if rec.get("n_per_v"):
            return {"n_per_v": float(rec["n_per_v"]), "regression_id": rec.get("regression_id") or "", "from": "recorded"}
        reg = active_regression()
        if reg is not None:
            return {"n_per_v": float(reg.n_per_v), "regression_id": reg.id, "from": "active"}
        return None

    @property
    def primary_angle(self):
        """The main joint's angle definition, or None."""
        return next((a for a in self.angle_defs if a.primary), None)

    @property
    def main_angle(self):
        """The main joint, or else the first angle (what the force is read against), or None."""
        return self.primary_angle or (self.angle_defs[0] if self.angle_defs else None)

    # ------------------------------------------------------------------ reference readings (typed from a hand instrument)
    def reference_rows(self):
        """One row per reading: the typed reference, the app's angle at the same moment, the difference (see limbtrack/reference.py).
        Worked out once (it may read the CSV) and again only after the readings change (reload_reference); do not change the rows."""
        if self._reference_rows is None:
            early = self.t <= reference.REST_SECONDS
            first = {a.name: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for a in self.angle_defs for v in [self.deg[a.name][early]]}
            self._reference_rows = reference.comparison(self.meta, self.reference_side, first_second=first)
        return self._reference_rows

    def reference_edit_keys(self):
        """[(key, name)] a reading can be typed for in Review: the trial's angles, then the extras of its test (the typed hip flexion of a frontal trial)."""
        return [(a.name, a.title) for a in self.angle_defs] + list((self.meta.get("reference_extras") or {}).items())

    def reload_reference(self):
        self.reference_side = reference.read_sidecar(self.meta_path)
        self._reference_rows = None
        self.reload_offsets()

    # ------------------------------------------------------------------ the angle reference (0.33.0)
    def reload_offsets(self):
        """What each angle is shown from: the reference taken when the trial was recorded (meta.json), and, for an angle the reviewer ticked
        'judge from this reading' in the reference dialog (the review layer's reference.json, key 'apply'), that reading's difference to the
        app's angle (app minus reading, at the stamp or in the first second): the dots read that much more than the joint. A reviewer's
        choice wins over the recorded one. {} for a trial with neither."""
        self.offsets = dict(self.recorded_offsets)
        apply = {k for k in ((self.reference_side or {}).get("apply") or []) if isinstance(k, str)}
        if apply:
            for row in self.reference_rows():
                if row["key"] in apply and row["key"] in self.deg and row["difference_deg"] is not None and math.isfinite(row["difference_deg"]):
                    self.offsets[row["key"]] = float(row["difference_deg"])

    def offset_of(self, name):
        """Degrees the dots read more than the joint for angle `name` (0 without a reference)."""
        return float(self.offsets.get(name, 0.0))

    def deg_view(self, name):
        """The angle `name` for every picture as it is shown: from the reference when there is one, else the dots' angle (`deg`)."""
        off = self.offset_of(name)
        return self.deg[name] - off if off else self.deg[name]

    def target_view(self, name):
        """Per picture, the target of angle `name` on the scale `deg_view` is on: data.csv's target column minus the offset the recording moved it by."""
        off = float(self.recorded_offsets.get(name, 0.0))
        return self.ref[name] - off if off else self.ref[name]

    def close(self):
        """Let go of the video file (a Trial that is replaced or finished with must not keep it open)."""
        if self.cap is not None:
            self.cap.release()

    def frame(self, i):
        if self.cap is None:
            return None
        if i != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        self._pos = i if ok else -2
        if ok and self.crop and self.full_size:
            img = privacy.pad_to_full(img, self.crop, self.full_size)
        return img if ok else None

    def trails_upto(self, i):
        """The dots' paths from the start of the recording up to frame i."""
        out = {}
        for n, segs in self.trail_segments.items():
            part = []
            for idx, xy in segs:
                if idx[0] > i:
                    break
                part.append(xy[: int(np.searchsorted(idx, i, side="right"))])
            out[n] = part
        return out

    def scale_overlay(self):
        """What overlay.draw needs to show the ruler that was marked for this trial, or None when there was none."""
        if not (self.ruler_dots and self.px_per_cm):
            return None
        px = math.hypot(self.ruler_dots[0][0] - self.ruler_dots[1][0], self.ruler_dots[0][1] - self.ruler_dots[1][1])
        return {"points": [(p[0], p[1], False) for p in self.ruler_dots],
                "text": f"ruler {self.ruler_cm:g} cm = {px:.0f} px  ({self.px_per_cm:.2f} px/cm)", "warn": False}

    def drawn(self, i, trails=True):
        img = self.frame(i)
        if img is None:
            return None
        r = self.rows[i]
        dots = {n: SimpleNamespace(x=_f(r[f"{n}_x"]), y=_f(r[f"{n}_y"]), lost=r[f"{n}_lost"] == "1",
                                   weak=r[f"{n}_lost"] != "1" and _f(r.get(f"{n}_q")) < WEAK_BELOW)
                for n in self.dot_names if r[f"{n}_x"] != ""}
        derived = {}
        for n in self.derived_names:
            x, y = _f(r[f"{n}_x"]), _f(r[f"{n}_y"])
            if not (math.isnan(x) or math.isnan(y)):
                derived[n] = (x, y)
        angles = []
        apart = [p.split(":")[0].split("-") for p in (r.get("dots_apart") or "").split(";") if p]     # 0.19.0 column; absent in older trials
        for a in self.angle_defs:
            n = a.name
            xy = {k: (d.x, d.y) for k, d in dots.items() if not d.lost}
            xy.update(derived)
            res = measure(a, xy, self.facing_right)     # geometry for drawing; numbers come from the CSV
            res.value, res.deviation = self.deg[n][i], self.dev[n][i]
            res.reference = None if math.isnan(self.ref[n][i]) else self.ref[n][i]
            flag = self.ok[n][i]
            res.status = LOST if math.isnan(res.value) else OK if flag == "1" else OUT if flag == "0" else INFO
            res.ref_offset = off = self.offset_of(n)    # the picture shows the angle from the reference first and the dots' angle beside it (0.33.0)
            if off != self.recorded_offsets.get(n, 0.0) and not math.isnan(res.value):      # a reference chosen in Review: the target on the dots' scale, judged again
                tv = self.target_view(n)[i]
                res.reference = None if math.isnan(tv) else tv + off
                if res.reference is None:
                    res.deviation, res.status = math.nan, INFO
                else:
                    res.deviation = res.value - res.reference
                    res.status = OK if abs(res.deviation) <= float(self.meta.get("tolerance_deg" if a.primary else "neighbour_tolerance_deg") or 5.0) else OUT
            if any(set(pair) & set(a.dots) for pair in apart):
                res.suspect = "dots moved apart"
            angles.append(res)
        head = f"{self.meta['test_label']} | {self.meta['participant']} | t = {self.t[i]:.2f} s"
        if r.get("label_check"):
            head += f" | CHECK LABELS: {r['label_check']}"
        # PARKED-GHOST: ghost = None
        # PARKED-GHOST: if self.ghost and r["locked"] == "1":
            # PARKED-GHOST: live = {k: (d.x, d.y) for k, d in dots.items() if not d.lost}
            # PARKED-GHOST: ghost = anchored_ghost({k: tuple(v) for k, v in self.ghost.items()}, live, self.ghost_anchor)
        img = overlay.draw(img, dots, angles, [head], trails=self.trails_upto(i) if trails else None, scale=self.scale_overlay(), derived=derived)
        for d, frames in self.edited_sets.items():              # a magenta ring round a dot whose position was fixed in Review (screen and overlay video)
            if i in frames and d in dots and not dots[d].lost and not (math.isnan(dots[d].x) or math.isnan(dots[d].y)):
                cv2.circle(img, (int(round(dots[d].x)), int(round(dots[d].y))), 17, (255, 0, 255), 2, cv2.LINE_AA)
        return img
        # PARKED-GHOST: # with the ghost: overlay.draw(img, dots, angles, [head], ghost=ghost, ghost_segments=segment_pairs(self.angle_defs), trails=...)

    def apply_calibration(self, cal, note):
        """Newtons from the saved volts with a calibration made AFTER the recording (the volts in the files are never changed)."""
        self.force_n = cal.to_newton(self.force_v)
        self.force_cal, self.force_cal_note = cal.to_dict(), note
        self.regression = find_regression(cal.regression_id)

    def recorded_zero_v(self):
        """(zero volts, where it came from): the zero the operator took for this trial, else the resting level of the recording
        (the 2nd percentile of the 100 ms smoothed signal: the strap at rest, before and after the pull)."""
        c = self._cal_recorded or {}                      # (what was recorded: a zero set in Review does not replace it)
        if c.get("zero_set", True) and c.get("zero_v") is not None:
            return float(c["zero_v"]), "the zero taken when this trial was recorded"
        rate = 1.0 / float(np.median(np.diff(self.force_t))) if len(self.force_t) > 2 else 1000.0
        return float(np.percentile(smooth_signal(self.force_v, rate, 0.1), 2)), "the resting level of the recording (no zero was taken)"

    def current_zero_v(self):
        """(zero volts, where it came from) as the numbers now use it: the zero set in Review if there is one, else the recorded one."""
        if self.zero_edit:
            return float(self.zero_edit["zero_v"]), f"the zero set in Review ({self.zero_edit.get('source')})"
        return self.recorded_zero_v()

    def zero_source(self):
        """'typed', 'stretch', 'rest' (set in Review), 'recorded' (taken when the trial was recorded), 'resting level' (none was taken) or 'none' (no force)."""
        if not self.has_force():
            return "none"
        if self.zero_edit:
            return str(self.zero_edit.get("source") or "typed")
        c = self._cal_recorded or {}
        return "recorded" if c.get("zero_set", True) and c.get("zero_v") is not None else "resting level"

    def raw_volts_mean(self, t0, t1):
        """Mean of the raw volts in [t0, t1] seconds, or NaN (used to take a zero from a stretch of the recording)."""
        if not self.has_force():
            return math.nan
        m = (self.force_t >= min(t0, t1)) & (self.force_t <= max(t0, t1)) & np.isfinite(self.force_v)
        return float(np.mean(self.force_v[m])) if m.any() else math.nan

    def relaxed_level(self, unit):
        """(level in `unit`, note): the relaxed level the operator took with Zero 2 (0.22.0), as the force reads it, or (NaN, note) when
        there is none for this trial. Newtons are worked out from the saved volts with the zero and slope now in force_cal, so a zero
        recalculated in Review carries it along. A level taken for another position or participant is not used."""
        r = (self.meta.get("force") or {}).get("relaxed")
        if not r or r.get("volts") is None:
            return math.nan, ""
        if not r.get("this_trial"):
            return math.nan, f"A relaxed level was taken for another position ({r.get('taken_for')}), so it is not used for this trial."
        if unit == "V":
            return float(r["volts"]), ""
        c = self.force_cal or {}
        if not c.get("n_per_v"):
            return math.nan, ""
        return (float(r["volts"]) - float(c.get("offset_v") or 0.0)) * float(c["n_per_v"]), ""

    def has_force(self):
        return self.force_t is not None and len(self.force_t) > 0

    def force_signal(self):
        """(unit, smoothed values, smoothed volts): newtons if calibrated (else volts), smoothed with the chosen moving average.
        The files hold every raw sample; smoothing is applied here, for the plot and the numbers."""
        rate = 1.0 / float(np.median(np.diff(self.force_t))) if len(self.force_t) > 2 else 1000.0
        volts = smooth_signal(self.force_v, rate, self.force_smooth_s)
        if np.all(np.isnan(self.force_n)):
            return "V", volts, volts
        return "N", smooth_signal(self.force_n, rate, self.force_smooth_s), volts

    def force_hum(self):
        """The 50 Hz mains interference on the raw voltage (volts, peak) or NaN."""
        rate = 1.0 / float(np.median(np.diff(self.force_t))) if len(self.force_t) > 2 else 1000.0
        return mains_amplitude(self.force_v, rate)

    def force_window(self, fraction=0.5):
        """(start, end) in seconds of the hold found from the force itself: the stretch around the peak where the smoothed
        force is above rest + fraction x (peak - rest) (default: half way, so a slow build-up or a first light push is left out). The MVIC markers are pressed by hand and can be seconds off; this
        follows the force. None without force or without a clear contraction."""
        if not self.has_force():
            return None
        unit, y, _ = self.force_signal()
        if np.all(np.isnan(y)):
            return None
        base = float(np.nanpercentile(y, 10))
        k = int(np.nanargmax(y))
        if y[k] - base < (0.01 if unit == "V" else 1.0):             # less than 10 mV (or 1 N) above rest: no contraction
            return None
        thr = base + fraction * (y[k] - base)
        above = y >= thr
        a = k
        while a > 0 and above[a - 1]:
            a -= 1
        b = k
        while b < len(y) - 1 and above[b + 1]:
            b += 1
        return float(self.force_t[a]), float(self.force_t[b])

    def force_rest(self):
        """The resting force before the push (SOP 9.10), or None without force or without a clear contraction. A dict: `level` = the
        median of the smoothed force from REST_FROM_S to REST_TO_S s before the push began (NaN when there was less than REST_MIN_S
        of force before it), `sd` its spread, `t_from` / `t_to` the stretch used, `seconds` how long it was, `onset_s` when the push
        began, `level_before` the low level the push was measured from, `note` a plain sentence when the rest was short or not steady.
        The push begins where the force was last within PUSH_RISE_FRACTION of its rise above the low level before it (the
        PRE_LEVEL_PERCENTILE-th percentile of the force before it passed half of its peak), walking back from that point. A light push
        or pre-tension before the real push therefore counts as pushing, not as rest; a strap that was already loaded before the push
        (P11 12:41: 85-160 N for 5 s) no longer puts the onset at the first sample, as the trial-wide low level did until 0.19.0.
        One rule for every trial, marked by hand or automatically; it does not depend on the analysis window."""
        if not self.has_force():
            return None
        key = (self.force_smooth_s, id(self.force_n))
        if getattr(self, "_rest_cache", (None,))[0] == key:
            return self._rest_cache[1]
        out = self._find_force_rest()
        self._rest_cache = (key, out)
        return out

    def _find_force_rest(self):
        win = self.force_window()
        if win is None:
            return None
        unit, y, _ = self.force_signal()
        t = self.force_t
        rate = 1.0 / float(np.median(np.diff(t))) if len(t) > 2 else 1000.0
        i_half = int(np.searchsorted(t, win[0]))
        in_win = (t >= win[0]) & (t <= win[1])
        peak = float(np.nanmax(y[in_win]))
        before = np.arange(t.size) < i_half
        good = ~np.isnan(y)
        pre = before & good
        if pre.sum() < 0.1 * rate:                                     # (almost) no force before the push: the recording began during it
            onset, level0 = float(t[0]), math.nan
        else:
            level0 = float(np.percentile(y[pre], PRE_LEVEL_PERCENTILE))
            thr = level0 + PUSH_RISE_FRACTION * (peak - level0)
            with np.errstate(invalid="ignore"):
                at_rest = np.nonzero(before & good & (y <= thr))[0]
            onset = float(t[at_rest[-1]]) if at_rest.size else float(t[0])
        rm = good & (t >= onset - REST_FROM_S) & (t < onset - REST_TO_S)
        seconds = float(rm.sum()) / rate
        level = float(np.median(y[rm])) if seconds >= REST_MIN_S else math.nan
        sd = float(np.std(y[rm])) if rm.sum() > 1 else math.nan
        d = 3 if unit == "V" else 0
        notes = []
        # 0.24.0: no note about a short rest any more. The SOP's 3 s at rest after Record is replaced by the resting force taken before the
        # trial (Zero 2, X), which the Live tab reminds about (softly); how long the rest in the recording was is still in `seconds`
        # (export column force_rest_s) and is not hidden.
        pushed = unit != "N" or peak - level >= NO_FORCE_RISE_N        # 0.26.0: without a push there is no rise to compare the spread with
        if pushed and not math.isnan(level) and not math.isnan(sd) and sd > REST_UNSTEADY_FRACTION * abs(peak - level):     # (P02: '0 N')
            notes.append(f"The force was not steady before the push (it varied by {sd:.{d}f} {unit}, SD): the strap was loaded or moving, "
                         "so the resting force is doubtful.")
        return {"unit": unit, "level": level, "sd": sd, "t_from": max(onset - REST_FROM_S, float(t[0])), "t_to": onset - REST_TO_S,
                "seconds": seconds, "onset_s": onset, "level_before": level0, "note": " ".join(notes)}

    def force_summary(self, t0, t1, picture_delay=0.0):
        """The force in the window: peak and when, the plateau mean, the resting level before the push (force_rest), and the main joint
        angle at the peak (the two candidate numbers for a model input: angle at peak force, and plateau mean angle). None without force.
        All force numbers come from the smoothed signal (the raw peak of a noisy signal is too high)."""
        if not self.has_force():
            return None
        unit, y, yv = self.force_signal()
        m = (self.force_t >= t0) & (self.force_t <= t1)
        if not m.any() or np.all(np.isnan(y[m])):
            return None
        yw, tw = y[m], self.force_t[m]
        k = int(np.nanargmax(yw))
        span = t1 - t0
        pm = (self.force_t >= t0 + PLATEAU_TRIM * span) & (self.force_t <= t1 - PLATEAU_TRIM * span)
        plateau = float(np.nanmean(y[pm])) if pm.any() else math.nan
        ri = self.force_rest()
        if ri is not None:
            rm = (self.force_t >= ri["t_from"]) & (self.force_t < ri["t_to"])
            rest, rest_s, rest_note = ri["level"], ri["seconds"], ri["note"]
            rest_basis = f"median {REST_FROM_S:g} to {REST_TO_S:g} s before the push"
        else:                                                          # no clear push: the second before the window, as before 0.20.0
            rm = (self.force_t >= t0 - REST_SECONDS) & (self.force_t < t0)
            rest = float(np.nanmean(y[rm])) if rm.any() else math.nan
            rest_s, rest_note = (float(rm.sum()) / (1.0 / float(np.median(np.diff(self.force_t)))) if rm.any() and len(self.force_t) > 2 else 0.0), ""
            rest_basis = "mean of the second before the window (no clear push)"
        raw = self.force_n if unit == "N" else self.force_v
        rate = 1.0 / float(np.median(np.diff(self.force_t))) if len(self.force_t) > 2 else 1000.0
        y20 = smooth_signal(raw, rate, PEAK_20MS_S)                            # 0.31.0: the highest 20 ms average, whatever the Smoothing box says
        main = self.main_angle
        at_dots = {a.title: self.angle_near(a.name, tw[k] + picture_delay) for a in self.angle_defs}      # every angle of the trial, 0.33.0
        at_peak = {a.title: at_dots[a.title] - self.offset_of(a.name) for a in self.angle_defs}            # (from the reference when there is one)
        angle = at_dots.get(main.title, math.nan) if main is not None else math.nan                        # the main joint's, as the dots read it: the old number
        relaxed, relaxed_note = self.relaxed_level(unit)
        hum = self.force_hum()
        vpeak = float(np.nanmax(yv[m]))
        vrest = float(np.nanmean(yv[rm])) if rm.any() else float(np.nanpercentile(yv, 10))
        hum_pct = 100.0 * hum / max(abs(vpeak - vrest), 1e-9) if not math.isnan(hum) else math.nan
        return {"unit": unit, "peak": float(yw[k]), "t_peak_s": float(tw[k]), "plateau_mean": plateau, "rest_mean": rest,
                "rest_seconds": rest_s, "rest_basis": rest_basis, "rest_note": rest_note,
                "peak_minus_rest": float(yw[k]) - rest if not math.isnan(rest) else math.nan,
                "relaxed": relaxed, "relaxed_note": relaxed_note,                      # Zero 2 (0.22.0): the tension taken with the participant relaxed
                "peak_minus_relaxed": float(yw[k]) - relaxed if not math.isnan(relaxed) else math.nan,
                "rest_minus_relaxed": rest - relaxed if not (math.isnan(rest) or math.isnan(relaxed)) else math.nan,
                "raw_peak": float(np.nanmax(raw[m])), "smoothing_s": self.force_smooth_s,
                "peak_20ms": float(np.nanmax(y20[m])) if np.isfinite(y20[m]).any() else math.nan,
                "mains_hum_v": hum, "mains_hum_pct_of_peak": hum_pct,
                "main_angle": main.title if main is not None else "", "main_angle_at_peak_deg": angle, "picture_delay_s": picture_delay,
                "angles_at_peak_deg": at_peak,                                          # {angle title: degrees} for every angle, the main one included (0.33.0),
                "angles_at_peak_dots_deg": at_dots,                                     # from the reference when the trial has one / as the dots read them
                "angle_offsets_deg": {a.title: self.offset_of(a.name) for a in self.angle_defs if self.offset_of(a.name)},   # (the dots read this much more than the joint)
                "samples_in_window": int(m.sum())}

    def angle_near(self, name, t_picture, half_s=0.1):
        """The angle `name` in the pictures within `half_s` seconds of the picture time `t_picture` (the mean; unsmoothed, lost pictures left out),
        else the nearest picture's value, else NaN. The angle at the peak force of every angle of the trial is read this way (0.33.0; the main joint
        was read the same way before)."""
        near = np.abs(self.t - t_picture) <= half_s
        vals = self.deg[name][near]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            i = int(np.argmin(np.abs(self.t - t_picture)))
            vals = self.deg[name][i:i + 1]
        return float(np.nanmean(vals)) if vals.size and not np.all(np.isnan(vals)) else math.nan

    def hold_window(self):
        """(start, end) seconds from the first 'MVIC start' and the next 'MVIC end' marker, or None."""
        start = next((t for _, t, lab in self.events if lab == "MVIC start"), None)
        if start is None:
            return None
        end = next((t for _, t, lab in self.events if lab == "MVIC end" and t > start), None)
        return None if end is None else (float(start), float(end))

    def displacement(self, t0, t1):
        """Per dot: how far it moved from its starting position in the window (see displacement.py). Anchor-relative
        numbers are added when the anchor marker was used."""
        ref = "anchor" if "anchor" in self.tracks else None
        return window_displacements(self.t, self.tracks, t0, t1, self.px_per_cm, ref)

    def pull_summary(self, t0, t1):
        """Rows of statistics for the angle of pull / moment arm columns within the window, or [] without strap and anchor dots."""
        if self.pull is None:
            return []
        m = (self.t >= t0) & (self.t <= t1)
        rows = []
        for col in ("pull_angle_deg", "moment_arm_cm" if self.px_per_cm else "moment_arm_px",
                    "strap_joint_cm" if self.px_per_cm else "strap_joint_px"):
            v = self.pull[col][m]
            good = v[~np.isnan(v)]
            rows.append({"quantity": col, "frames_missing": int(m.sum() - good.size),
                         "mean": float(np.mean(good)) if good.size else math.nan,
                         "min": float(np.min(good)) if good.size else math.nan,
                         "max": float(np.max(good)) if good.size else math.nan,
                         "change_start_to_end": float(np.mean(good[-5:]) - np.mean(good[:5])) if good.size >= 2 else math.nan})
        return rows

    def summary(self, t0, t1):
        """Rows of statistics per angle within [t0, t1] seconds."""
        m = (self.t >= t0) & (self.t <= t1)
        ri = self.force_rest()
        rest_win = (ri["t_from"], ri["t_to"]) if ri is not None and ri["seconds"] >= REST_MIN_S else None
        out = []
        tol_main = float(self.meta.get("tolerance_deg") or 5.0)
        tol_nb = float(self.meta.get("neighbour_tolerance_deg") or tol_main)
        for a in self.angle_defs:
            n = a.name
            # Every number is from the reference when the angle has one (0.33.0): `off` is what the dots read more than the joint, `rec` what the
            # recording moved its target column by. With the same offset as at the recording the deviations and flags in data.csv are the ones
            # that were shown; with another (a reference chosen in Review) they are worked out again here.
            off, rec = self.offset_of(n), float(self.recorded_offsets.get(n, 0.0))
            v = self.deg[n][m] - off
            valid = ~np.isnan(v)
            if off == rec:
                flags = [f for f, keep in zip(self.ok[n], m) if keep and f != ""]
                dev = self.dev[n][m]
            else:
                dev = v - (self.ref[n][m] - rec)
                flags = ["1" if abs(d) <= (tol_main if a.primary else tol_nb) else "0" for d in dev if not math.isnan(d)]
            good = v[valid]
            change = float(np.mean(good[-5:]) - np.mean(good[:5])) if good.size >= 2 else math.nan
            # The angle that matters is the one during the hold, not the one typed as the target (audit VAL-01).
            span = t1 - t0
            pm = (self.t >= t0 + PLATEAU_TRIM * span) & (self.t <= t1 - PLATEAU_TRIM * span)
            pv = self.deg[n][pm] - off
            pv = pv[~np.isnan(pv)]
            plateau = float(np.mean(pv)) if pv.size >= REST_MIN_FRAMES else math.nan
            # the angle at rest: the same stretch as the resting force (before the push), else the second before the window
            rm = ((self.t >= rest_win[0]) & (self.t < rest_win[1])) if rest_win else (self.t >= t0 - REST_SECONDS) & (self.t < t0)
            rv = self.deg[n][rm] - off
            rv = rv[~np.isnan(rv)]
            rest = float(np.mean(rv)) if rv.size >= REST_MIN_FRAMES else math.nan
            refs = self.ref[n][m] - rec
            refs = refs[~np.isnan(refs)]
            target = float(np.mean(refs)) if refs.size else math.nan
            out.append({
                "angle": a.title,
                "role": "main" if a.primary else "neighbour",
                "frames": int(m.sum()),
                "frames_lost": int((~valid).sum()),
                "mean_deg": float(np.mean(v[valid])) if valid.any() else math.nan,
                "sd_deg": float(np.std(v[valid], ddof=1)) if valid.sum() > 1 else math.nan,
                "min_deg": float(np.min(v[valid])) if valid.any() else math.nan,
                "max_deg": float(np.max(v[valid])) if valid.any() else math.nan,
                "change_start_to_end_deg": change,
                "median_deg": float(np.median(good)) if good.size else math.nan,
                "target_deg": target,
                "mean_minus_target_deg": (float(np.mean(good)) - target) if good.size and not math.isnan(target) else math.nan,
                "plateau_mean_deg": plateau,
                "offset_deg": off,                                         # the dots read this much more than the joint (0: no reference); every number above is from the reference
                "plateau_mean_dots_deg": plateau + off if not math.isnan(plateau) else math.nan,
                "rest_mean_deg": rest,
                "rest_to_hold_shift_deg": (plateau - rest) if not (math.isnan(plateau) or math.isnan(rest)) else math.nan,
                "max_abs_dev_deg": float(np.nanmax(np.abs(dev))) if np.any(~np.isnan(dev)) else math.nan,
                "pct_in_tolerance": 100.0 * sum(f == "1" for f in flags) / len(flags) if flags else math.nan,
            })
        return out


def after_trial_check(meta_path):
    """What the window says after a trial is saved (0.20.0): the SOP 10.6 10 % check over every saved trial of the same participant,
    test and target (analysis.position_check on the export rows, so the numbers are the export's), and the resting-force note of
    this trial (SOP 9.10: 3 s at rest in the strap). 0.26.0 adds, from the same rows: the main joint during the push against the target
    and the shift from rest (`angle_note`), whether that angle fits the position (`position_note`: against the earlier trials of the
    position, or for its first trial against the previous position of the same test), and whether Zero 2 fits the force just before the
    push (`relaxed_note`). Reads files only; safe to run off the window's thread. Returns a dict: participant, test_key, target_deg,
    n (trials of the position), check (position_check's dict), rest_note, angle_note, position_note, relaxed_note, trial."""
    from .recorder import trial_metas
    meta_path = Path(meta_path).resolve()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    who, key, tgt = meta.get("participant") or "", meta.get("test_key"), meta.get("target_deg")
    root = meta_path.parent.parent
    paths = [p.resolve() for p in trial_metas(root, who, key, tgt)]
    if meta_path not in paths:                         # (a file name of another form: check the trial on its own)
        paths.append(meta_path)
    rows, rest_note, unreadable, this_row, earlier = [], "", [], None, []
    for p in paths:
        try:
            tr = Trial(p, open_video=False)                    # (numbers only: no video is opened)
        except Exception:  # noqa: BLE001 - an older trial that cannot be opened (a crashed recording) is named, not hidden
            if p == meta_path:
                raise
            unreadable.append(p.name[: -len("_meta.json")])
            continue
        try:
            row = analysis.session_row(tr)
            rows.append(row)
            if p == meta_path:
                this_row = row
                if tr.has_force():
                    ri = tr.force_rest()
                    rest_note = ri["note"] if ri is not None else ""
            elif this_row is None and row.get("verdict") != verdicts.COMPROMISED:     # (the trials before this one, oldest first;
                earlier.append(analysis.row_plateau(row))                         # not one marked compromised: 0.28.0)
        finally:
            tr.close()
    check = analysis.position_check(rows)
    if unreadable:
        check["text"] = (check["text"] + " " if check["text"] else "") + f"Could not be read, so not compared: {', '.join(unreadable)}."
    tol = float(meta.get("tolerance_deg") or 5.0)
    angle_note = position_note = relaxed_note = ""
    if this_row is not None:
        angle_note = analysis.hold_angle_note(this_row, tol)
        has_earlier = any(analysis._num(v) is not None for v in earlier)
        previous = None if has_earlier else _previous_position(root, meta, meta_path)
        position_note = analysis.position_angle_note(this_row.get("main_angle"), analysis.row_plateau(this_row), earlier, tgt, tol, previous)
        relaxed_note = analysis.relaxed_mismatch(this_row.get("force_unit"), this_row.get("force_relaxed"), this_row.get("force_rest"),
                                                 this_row.get("force_peak_above_rest"))
    return {"participant": who, "test_key": key, "target_deg": tgt, "n": len(paths), "check": check,
            "compromised": sum(verdicts.is_compromised(p) for p in paths),           # (0.28.0: part of what the window's check is for)
            "rest_note": rest_note, "angle_note": angle_note, "position_note": position_note, "relaxed_note": relaxed_note,
            "trial": meta_path.name[: -len("_meta.json")]}


def trial_row(meta_path, smooth_s=SMOOTH_DEFAULT_S, window_s=WINDOW_PEAK_DEFAULT_S, source="original"):
    """The export row of one trial (analysis.session_row), worked out without opening its video: `(row, "")`, or `(None, why)` when the
    trial cannot be read. `source`: 'original' (as recorded) or 'reviewed' (read through the edits made in Review). Safe to run off the
    window's thread (it only reads files). The overview and the exports use it."""
    try:
        tr = Trial(meta_path, open_video=False, source=source)
    except Exception as e:  # noqa: BLE001 - named on the trial's line, not hidden
        return None, f"{type(e).__name__}: {e}"
    try:
        tr.force_smooth_s, tr.window_peak_s = float(smooth_s), float(window_s)
        return analysis.session_row(tr), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    finally:
        tr.close()


def _previous_position(root, meta, meta_path):
    """(target, [plateau angles]) of the position of the same participant and test recorded last before this trial (another target), or
    None. For the first trial of a position: was the participant really moved there? (0.26.0)"""
    from .recorder import trial_metas
    who, key, tgt, started = meta.get("participant") or "", meta.get("test_key"), meta.get("target_deg"), meta.get("started_at") or ""
    best = None
    for p in meta_path.parent.glob("*_meta.json"):
        if p.resolve() == meta_path:
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        t, when = m.get("target_deg"), m.get("started_at") or ""
        if ((m.get("participant") or "") != who or m.get("test_key") != key or t is None or tgt is None
                or abs(float(t) - float(tgt)) < 1e-6 or not when or when >= started):
            continue
        if best is None or when > best[0]:
            best = (when, float(t))
    if best is None:
        return None
    plateaus = []
    for p in trial_metas(root, who, key, best[1]):
        try:
            tr = Trial(p, open_video=False)
        except Exception:  # noqa: BLE001 - an unreadable trial is simply not used for this comparison
            continue
        try:
            row = analysis.session_row(tr)
            if row.get("verdict") != verdicts.COMPROMISED:          # (a trial marked compromised does not say where the position was)
                plateaus.append(analysis.row_plateau(row))
        finally:
            tr.close()
    return best[1], plateaus


class _TextPane(QTextEdit):
    """A read-only text area for a block of numbers (monospace, no wrapping); setText and text work like a label's."""

    def __init__(self, wrap=False):
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth if wrap else QTextEdit.NoWrap)      # tables stay in columns; prose wraps
        self.setStyleSheet("font-family: Consolas, monospace;")

    def setText(self, s):
        self.setPlainText(s)

    def text(self):
        return self.toPlainText()


class _CalibrationPane(QWidget):
    """The load cell regression behind the force numbers: the points and the line, how far each run is from it (with the band a reading
    is good to), where this trial's force lies, and the precision in words."""

    def __init__(self):
        super().__init__()
        self.fit = pg.PlotWidget()
        self.res = pg.PlotWidget()
        for w, left in ((self.fit, "Volts"), (self.res, "Reading - mass (kg)")):
            w.showGrid(x=True, y=True, alpha=0.25)
            w.setMouseEnabled(False, False)
            w.setLabel("left", left)
        self.res.setLabel("bottom", "Mass on the cell (kg)")
        self.text = _TextPane(wrap=True)
        self.text.setMaximumHeight(190)
        lay = QVBoxLayout(self)          # since 0.32.0 a graph view in the graph area: the two plots side by side, the words under them
        lay.setContentsMargins(0, 0, 0, 0)
        plots = QHBoxLayout()
        plots.addWidget(self.fit, 1)
        plots.addWidget(self.res, 1)
        lay.addLayout(plots, 1)
        lay.addWidget(self.text)

    def show_regression(self, reg, lines, marks_kg=()):
        """reg: a Regression or None; lines: the text under the plots; marks_kg: [(kg, label)] of this trial (peak, plateau) as vertical lines."""
        self.fit.clear()
        self.res.clear()
        self.text.setText("\n".join(lines))
        if reg is None:
            return
        kg, resid, phase = reg.residuals_kg()
        volts = np.array([p["volts"] for p in reg.points])
        top = max(float(kg.max()), max([m[0] for m in marks_kg], default=0.0)) * 1.04
        xs = np.array([0.0, top])
        self.fit.plot(xs, reg.intercept_v + reg.slope_v_per_kg * xs, pen=pg.mkPen("#555", width=2))
        self.res.addLine(y=0, pen=pg.mkPen("#555", width=1.5))
        band = COVERAGE * reg.sd_kg(np.linspace(0, top, 60))
        for sign in (1, -1):
            self.res.plot(np.linspace(0, top, 60), sign * band, pen=pg.mkPen("#2f9e44", width=1.5, style=Qt.DashLine))
        for name, colour in (("up", "#1c7ed6"), ("down", "#e8590c"), ("", "#868e96")):
            m = np.array([(p == name) if name else (p not in ("up", "down")) for p in phase])
            if m.any():
                for w, y in ((self.fit, volts), (self.res, resid)):
                    w.plot(kg[m], y[m], pen=None, symbol="o", symbolSize=7, symbolBrush=colour, symbolPen=None)
        for k, label in marks_kg:
            for w in (self.fit, self.res):
                w.addItem(pg.InfiniteLine(pos=k, angle=90, pen=pg.mkPen("#e03131", width=2, style=Qt.DashLine),
                                          label=label, labelOpts={"position": 0.9, "color": "#e03131"}))
        self.res.setXLink(self.fit)


class _VideoLabel(QLabel):
    """The picture of the trial. A click gives the place in full-picture pixels (the picture is shown scaled and centred in the label), so
    a dot can be put right on the video (0.30.0, Fix dots). Since 0.32.0 the picture is scaled to the screen's own pixels (on a screen set to
    125 or 150 % it was scaled small and stretched, so blurry), and a click takes the keyboard, so the picture keys work after it."""
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.picture_size = None                          # (width, height) of the picture last shown, in pixels
        self.view_rect = None                             # (x, y, w, h) of the part of the picture on show when it is magnified, else None (all of it)
        self.setFocusPolicy(Qt.ClickFocus)

    def show_picture(self, pm):
        """Show a pixmap made from the picture, scaled to fit this label as large as it goes, in physical pixels."""
        dpr = self.devicePixelRatioF()
        out = pm.scaled(int(self.width() * dpr), int(self.height() * dpr), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        out.setDevicePixelRatio(dpr)
        self.setPixmap(out)

    def mousePressEvent(self, e):
        pm = self.pixmap()
        if e.button() == Qt.LeftButton and self.picture_size and pm is not None and not pm.isNull():
            w, h = self.picture_size
            shown = pm.deviceIndependentSize()           # (logical size: the same as the pixmap's own on a screen at 100 %)
            off_x, off_y = (self.width() - shown.width()) / 2.0, (self.height() - shown.height()) / 2.0
            px, py = e.position().x() - off_x, e.position().y() - off_y
            if 0 <= px < shown.width() and 0 <= py < shown.height():
                if self.view_rect is not None:                 # (magnified: the shown part is a window onto the picture)
                    vx, vy, vw, vh = self.view_rect
                    self.clicked.emit(vx + px * vw / shown.width(), vy + py * vh / shown.height())
                else:
                    self.clicked.emit(px * w / shown.width(), py * h / shown.height())
                return
        super().mousePressEvent(e)


class _NoWheelSpin(QDoubleSpinBox):
    """A spin box that ignores the mouse wheel (scrolling past it must not change the number)."""

    def wheelEvent(self, e):
        e.ignore()


def fill_reference_choices(instrument_combo, resolution_combo):
    """The instrument and 'read to' choices of the reference readings (the Live tab and the Review dialog), set to the defaults."""
    for k, label in reference.INSTRUMENTS.items():
        instrument_combo.addItem(label, k)
    for v in reference.RESOLUTIONS_DEG:
        resolution_combo.addItem(f"{v:g} deg", v)
    select_reference_choices(instrument_combo, resolution_combo, None, None)


def select_reference_choices(instrument_combo, resolution_combo, instrument, resolution_deg):
    """Show `instrument` and `resolution_deg` in the two boxes (the defaults when missing or unknown), without sending the change
    signals."""
    for combo, value, default in ((instrument_combo, instrument, reference.DEFAULT_INSTRUMENT),
                                  (resolution_combo, resolution_deg, reference.DEFAULT_RESOLUTION_DEG)):
        i = combo.findData(value) if value is not None else -1
        combo.blockSignals(True)
        combo.setCurrentIndex(i if i >= 0 else max(combo.findData(default), 0))
        combo.blockSignals(False)


class ReferenceDialog(QDialog):
    """Type, change or remove the reference readings of one trial. They are saved in a small file next to the trial
    (`<trial>_reference.json`); the trial's own `meta.json` is never rewritten."""

    def __init__(self, trial, parent=None):
        super().__init__(parent)
        self.trial = trial
        self.setWindowTitle("Reference readings of this trial")
        merged = reference.merged_readings(trial.meta, trial.reference_side)
        saved = trial.meta.get("reference_readings") or {}
        side = trial.reference_side or {}
        lay = QVBoxLayout(self)
        note = QLabel("What a goniometer or an inclinometer said, in degrees, for this trial's position. Several readings of one angle: "
                      "separate them with a space. An empty box means 'not measured'. These are numbers a person typed, not measurements "
                      "of this program. They are saved next to the trial and never change its recorded files.")
        note.setWordWrap(True)
        lay.addWidget(note)
        form = QFormLayout()
        self.instrument, self.resolution = QComboBox(), QComboBox()
        fill_reference_choices(self.instrument, self.resolution)
        current = side if side.get("readings") else saved
        select_reference_choices(self.instrument, self.resolution, current.get("instrument"), current.get("resolution_deg"))
        form.addRow("Instrument", self.instrument)
        form.addRow("Read to", self.resolution)
        self.boxes = {}
        for key, title in trial.reference_edit_keys():
            box = QLineEdit(reference.format_values((merged.get(key) or {}).get("values_deg") or []))
            box.setPlaceholderText("e.g. 88 90 89")
            form.addRow(title, box)
            self.boxes[key] = box
        # 0.33.0: show and judge an angle from its reading (an angle reference taken in Review, for trials recorded without one)
        applied = {k for k in (side.get("apply") or []) if isinstance(k, str)}
        self.apply_boxes = {}
        for a in trial.angle_defs:
            chk = QCheckBox("Judge this angle from this reading")
            chk.setChecked(a.name in applied)
            chk.setToolTip("Review then shows every number of this angle from the reading: what the dots read when the reading was taken minus the reading "
                           "(the app's angle at the reading, or in the first second of the recording) is taken away. The dots' own numbers stay beside "
                           "them. Nothing recorded is changed; this is saved with the reading. It replaces a reference taken when the trial was recorded.")
            form.addRow("", chk)
            self.apply_boxes[a.name] = chk
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _save(self):
        values = {}
        for key, box in self.boxes.items():
            try:
                values[key] = reference.parse_values(box.text())
            except reference.ReadingError as e:
                QMessageBox.warning(self, "Cannot read that", f"{dict(self.trial.reference_edit_keys()).get(key, key)}: {e}")
                box.setFocus()
                return
        merged = reference.merged_readings(self.trial.meta, self.trial.reference_side)
        apply = [k for k, chk in self.apply_boxes.items() if chk.isChecked()]
        for k in apply:
            if not (values.get(k) if k in values else (merged.get(k) or {}).get("values_deg")):
                QMessageBox.warning(self, "No reading", f"{dict(self.trial.reference_edit_keys()).get(k, k)}: type the reading first, or untick "
                                    "'Judge this angle from this reading'.")
                self.boxes[k].setFocus()
                return
        try:
            reference.write_sidecar(self.trial.meta_path, self.instrument.currentData(), self.resolution.currentData(), values,
                                    recorded=self.trial.meta.get("reference_readings"), apply=apply)
        except OSError as e:                              # a locked or read-only folder: say so, keep the dialog open with what was typed
            QMessageBox.warning(self, "Not saved", f"The readings could not be saved next to the trial:\n{e}\n\n"
                                "Nothing was changed. Check that the recordings folder can be written to, then press Save again.")
            return
        self.trial.reload_reference()
        self.accept()


class ReferenceValidationDialog(QDialog):
    """How far the app's angles are from the hand instrument over all trials of a folder: a table in words and an agreement plot
    (the difference against the mean of the two, with the average difference and the 95 % limits)."""

    def __init__(self, folder, rows, parent=None):
        super().__init__(parent)
        self.folder, self.rows, self.stats = Path(folder), rows, reference.agreement(rows)
        self.setWindowTitle("Agreement with the reference readings")
        self.resize(900, 700)
        lay = QVBoxLayout(self)
        self.text = _TextPane(wrap=True)
        self.text.setText(reference.agreement_text(self.stats))
        self.text.setMaximumHeight(230)
        lay.addWidget(self.text)
        self.combo = QComboBox()
        for key, s in self.stats.items():
            self.combo.addItem(f"{s['title']}  (n = {s['n']})", key)
        self.combo.currentIndexChanged.connect(self._plot)
        lay.addWidget(self.combo)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Mean of the app and the reading (deg)")
        self.plot.setLabel("left", "App minus reference (deg)")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        lay.addWidget(self.plot, 1)
        btn_save = QPushButton("Save the table as CSV")
        btn_save.setToolTip(f"Writes reference_validation.csv into {self.folder}: one row per reading, with the app's angle, the difference and "
                            "a mark for a reading that two trials of one position share.")
        btn_save.clicked.connect(self.save_table)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.addButton(btn_save, QDialogButtonBox.ActionRole)
        lay.addWidget(buttons)
        self._plot()

    def _plot(self, *_):
        self.plot.clear()
        s = self.stats.get(self.combo.currentData())
        if s is None:
            return
        refs, apps = np.array([p[0] for p in s["pairs"]]), np.array([p[1] for p in s["pairs"]])
        self.plot.plot((refs + apps) / 2.0, apps - refs, pen=None, symbol="o", symbolSize=10, symbolBrush="#4dabf7", symbolPen=None)
        self.plot.addLine(y=0, pen=pg.mkPen((200, 200, 200), width=1))
        self.plot.addLine(y=s["bias_deg"], pen=pg.mkPen("#ffd43b", width=2))
        for v in s["limits_deg"]:
            if not math.isnan(v):
                self.plot.addLine(y=v, pen=pg.mkPen("#ff922b", width=1.5, style=Qt.DashLine))

    def save_table(self):
        target = self.folder / review_layer.REVIEW_FOLDER / "reference_validation.csv"        # (0.30.0: the review folder of the recordings folder)
        try:
            review_layer.guard(target).parent.mkdir(parents=True, exist_ok=True)
            out = reference.write_csv(self.rows, target)
        except OSError as e:                              # most often: the file is open in Excel, which locks it on Windows
            QMessageBox.warning(self, "Not saved", f"reference_validation.csv could not be written in {target.parent}:\n{e}\n\n"
                                "If the file is open in Excel or another program, close it and press Save again. "
                                "The file on disk is still the earlier one.")
            return
        QMessageBox.information(self, "Saved", f"Saved {out.name} in {target.parent}: {len(self.rows)} readings.")


class ReviewTab(QWidget):
    def __init__(self, recordings_dir):
        super().__init__()
        self._dir = Path(recordings_dir)
        self.trial = None

        # --- the header (0.29.0): choose a participant, then a trial from a drop-down; the Trial and Overview pages; exports
        self.participant_combo = QComboBox()
        self.participant_combo.setMinimumWidth(120)
        self.participant_combo.setToolTip("The participant whose trials you go through: the folders of the recordings folder that hold trials. "
                                          "'Open folder...' adds any other place, such as Pilot/P01.")
        self.participant_combo.activated.connect(self._participant_chosen)
        self.btn_open_folder = QPushButton("Open folder...")
        self.btn_open_folder.setToolTip("Pick a participant's folder (such as Pilot/P01), or a folder of participants (such as Pilot).")
        self.btn_open_folder.clicked.connect(self.open_folder_dialog)
        self.trial_combo = QComboBox()
        self.trial_combo.setMinimumWidth(180)
        self.trial_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.trial_combo.setToolTip("The trials of this participant in the order they were recorded: position, test, trial, time and your verdict "
                                    "(✓ valid, ✗ compromised, · not judged).")
        self.trial_combo.activated.connect(self._trial_chosen)
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        for b, tip in ((self.btn_prev, "The previous trial (Alt+Left)"), (self.btn_next, "The next trial (Alt+Right)")):
            b.setToolTip(tip)
            b.setFixedSize(34, self.trial_combo.sizeHint().height())     # (a symbol from another font must not make the button taller than the rest)
        self.btn_prev.clicked.connect(self.prev_trial)
        self.btn_next.clicked.connect(self.next_trial)
        self.btn_next_unjudged = QPushButton("Next not judged")
        self.btn_next_unjudged.setObjectName("nextUnjudged")
        self.btn_next_unjudged.setToolTip("Jump to the next trial that has no verdict yet (Alt+N). Go through a participant with this and the "
                                          "Valid / Dots / Position / Force / Other buttons.")
        self.btn_next_unjudged.clicked.connect(self.next_unjudged)
        self.btn_view_trial, self.btn_view_overview = QPushButton("Trial"), QPushButton("Participant overview")
        self.btn_view_trial.setObjectName("segLeft")
        self.btn_view_overview.setObjectName("segRight")
        view_group = QButtonGroup(self)
        for b in (self.btn_view_trial, self.btn_view_overview):
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            view_group.addButton(b)
        self.btn_view_trial.setChecked(True)
        self.btn_view_trial.clicked.connect(lambda: self.show_view("trial"))
        self.btn_view_overview.clicked.connect(lambda: self.show_view("overview"))
        self.btn_view_overview.setToolTip("One line per trial of this participant: verdict, force numbers, checks. Sort, filter and export.")
        # the data every number, graph and export reads (0.30.0): the recorded trial, or the recorded trial read through the edits made in Review
        self._source = "reviewed"
        self.btn_data_original, self.btn_data_reviewed = QPushButton("Original"), QPushButton("Reviewed")
        self.btn_data_original.setObjectName("segLeft")
        self.btn_data_reviewed.setObjectName("segRight")
        self.btn_data_original.setToolTip("The trial exactly as it was recorded. Everything on screen and every export uses it.")
        self.btn_data_reviewed.setToolTip("The trial read through the fixes made in Review (force zero, dot positions). The recorded files are never "
                                          "changed; with no fixes this is the same as Original.")
        data_group = QButtonGroup(self)
        for b in (self.btn_data_original, self.btn_data_reviewed):
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            data_group.addButton(b)
        self.btn_data_reviewed.setChecked(True)
        self.btn_data_original.clicked.connect(lambda: self.set_source("original"))
        self.btn_data_reviewed.clicked.connect(lambda: self.set_source("reviewed"))
        self.act_export_all = QAction("All trials of this participant (CSV)", self)
        self.act_export_all.setToolTip("One row per trial (verdict, force peak, sliding-window peak, time to peak, plateau, the angle at the peak, "
                                       "torque if possible, checks) and whether two trials of the same position differ by more than 10 %. Uses the data "
                                       "chosen under Data (Original or Reviewed). Saved in the review folder, in exports, as "
                                       "<participant>_session_summary_<original or reviewed>.csv.")
        self.act_export_valid = QAction("Valid trials only (CSV)", self)
        self.act_export_valid.setToolTip("The same, with only the trials you marked valid. Saved as <participant>_session_summary_<...>_valid.csv.")
        self.act_export_trial = QAction("This trial: summary CSV", self)
        self.act_export_trial.setToolTip("Summary of the open trial (and its dot movement) for the data chosen under Data, in the trial's review folder. "
                                         "For Reviewed data with fixes it also writes reviewed_data.csv and reviewed_force.csv: the edited numbers for other software.")
        self.act_export_video = QAction("This trial: overlay video", self)
        self.act_export_video.setToolTip("Draws the angles and trails on every frame and saves a video (for sharing or a presentation). "
                                         "Dots you fixed are drawn where you put them.")
        self.act_open_review = QAction("Open the review folder", self)
        self.act_open_review.setToolTip("Where everything Review saves is kept: fixes, verdicts, reference readings and exports. The recorded files are not in it.")
        self.act_open_review.triggered.connect(self.open_review_folder)
        self.act_refval = QAction("Reference validation...", self)
        self.act_refval.setToolTip("How far the app's angles are from the goniometer or inclinometer readings, over all trials in the recordings "
                                   "folder (average difference, spread, an agreement plot, and a table to save).")
        self.act_open_file = QAction("Open a single trial file...", self)
        self.act_export_all.triggered.connect(lambda: self.export_session())
        self.act_export_valid.triggered.connect(lambda: self.export_session(valid_only=True))
        self.act_export_trial.triggered.connect(self.export)
        self.act_export_video.triggered.connect(self.export_overlay_video)
        self.act_refval.triggered.connect(self.show_reference_validation)
        self.act_open_file.triggered.connect(self.open_dialog)
        # Export and More are in the participant overview since 0.32.0 (the menus are built here, the overview shows them)
        self.export_menu = QMenu(self)
        for a in (self.act_export_all, self.act_export_valid):
            self.export_menu.addAction(a)
        self.export_menu.addSeparator()
        for a in (self.act_export_trial, self.act_export_video):
            self.export_menu.addAction(a)
        self.export_menu.setToolTipsVisible(True)
        self.more_menu = QMenu(self)
        self.more_menu.setToolTipsVisible(True)
        self.more_menu.addAction(self.act_open_review)
        self.more_menu.addAction(self.act_refval)
        self.more_menu.addAction(self.act_open_file)
        self.title = ElidedLabel("Choose a participant, then a trial.")
        self.title.setObjectName("trialSubtitle")
        self.btn_fix = QPushButton("Fix dots")                  # opens the Fix dots workspace over the video (0.32.0)
        self.btn_fix.setObjectName("fixToggle")
        self.btn_fix.setCheckable(True)
        self.btn_fix.setFocusPolicy(Qt.NoFocus)
        self.btn_fix.setToolTip("Put a dot right after the trial: the video grows and the dot tools take the right side. Press again, or Esc, to go back.")
        self.btn_fix.setEnabled(False)
        self.btn_fix.clicked.connect(lambda on: self.set_fix_mode(on))
        self.btn_privacy = QPushButton("Privacy")                # draw a crop and masks after the recording and make protected copies of the videos (0.34.0)
        self.btn_privacy.setObjectName("privacyButton")
        self.btn_privacy.setFocusPolicy(Qt.NoFocus)
        self.btn_privacy.setToolTip("Draw a crop and black masks on the picture and make protected copies of raw.mp4 and overlay.mp4 in the review folder. "
                                    "The trial's own videos are not changed.")
        self.btn_privacy.setEnabled(False)
        self.btn_privacy.clicked.connect(lambda: self.open_privacy())
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.btn_step_back, self.btn_step_fwd = QPushButton("◀"), QPushButton("▶")                  # one picture at a time (arrow keys, and , and .)
        for b, tip in ((self.btn_step_back, "One picture back (Left arrow or ,)"), (self.btn_step_fwd, "One picture on (Right arrow or .)")):
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.setFixedSize(30, self.btn_play.sizeHint().height())
        self.btn_step_back.clicked.connect(lambda: self.step_picture(-1))
        self.btn_step_fwd.clicked.connect(lambda: self.step_picture(1))
        self.frame_label = QLabel("")                           # which picture is on screen, for going through them one by one
        self.frame_label.setObjectName("hint")
        self.frame_label.setMinimumWidth(190)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.nav_label = QLabel("")                             # where in the list the open trial is: '12 of 64 · 20 left'
        self.nav_label.setObjectName("hint")
        self.showing_label = QLabel("Showing")                  # the Data switch is in the header: it changes every number and graph
        self.showing_label.setObjectName("hint")
        self.edits_label = QLabel("")                           # '✎ 2 fixes' when the trial has fixes made in Review
        self.edits_label.setObjectName("hint")
        self.zoom_label = QLabel("Zoom")                        # only while dots are fixed (0.32.0)
        self.zoom_combo = QComboBox()
        for text, factor in (("Fit", 1.0), ("2x", 2.0), ("3x", 3.0), ("4x", 4.0)):
            self.zoom_combo.addItem(text, factor)
        self.zoom_combo.setFocusPolicy(Qt.NoFocus)
        self.zoom_combo.setToolTip("Magnify the picture around the dot you are fixing. Clicks still give the dot's place in the whole picture.")
        self.zoom_combo.currentIndexChanged.connect(lambda _: self.trial is not None and self.show_frame(self.slider.value()))
        self.zoom_label.setVisible(False)
        self.zoom_combo.setVisible(False)

        self.video = _VideoLabel()
        self.video.clicked.connect(self._video_clicked)
        self.video.setMinimumSize(320, 180)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)      # the picture is scaled to the label, not the label to the picture: the dividers can shrink it
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background: black;")

        # two plots one above the other with the same time axis: the angles on top, the force below
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Angle (deg)")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.force_plot = pg.PlotWidget()
        self.force_plot.setLabel("bottom", "Time (s)")
        self.force_plot.getAxis("left").enableAutoSIPrefix(False)
        self.force_plot.showGrid(x=True, y=True, alpha=0.25)
        self.force_plot.setXLink(self.plot)
        self.force_plot.setTitle("Blue band = analysis window (drag its edges) · yellow bar = sliding-window peak · grey stretch = where the zero is taken", size="8pt")
        self.force_plot.setVisible(False)
        self.plot.setLabel("bottom", "Time (s)")
        # the same recording as force against angle: how far the joint moved while the force rose (slack, stretch, the position slipping)
        self.fa_plot = pg.PlotWidget()
        self.fa_plot.showGrid(x=True, y=True, alpha=0.25)
        self.fa_plot.getAxis("left").enableAutoSIPrefix(False)
        self.fa_plot.setVisible(False)
        self._fa_cursor = pg.ScatterPlotItem(size=17, symbol="o", pen=pg.mkPen(CURSOR_COLOUR, width=3), brush=pg.mkBrush(0, 0, 0, 0))
        self._fa_path = None
        self._fa_dirty = True              # the path must be drawn afresh (new trial, smoothing, calibration, picture delay); else only the window part
        self._fa_cursor_shown = False
        # Made once and kept: the window's larger dots and the star only get new data when the window moves (a new plot item per mouse
        # move was slow), and the dots take their colour from one palette of 64 brushes (one brush per picture defeats pyqtgraph's
        # symbol cache: about 100 ms for a 30 s trial at 60 pictures per second).
        self._fa_dots = pg.ScatterPlotItem(size=10, pen=pg.mkPen("w", width=1.2))
        self._fa_star = pg.ScatterPlotItem(size=20, symbol="star", pen=pg.mkPen("k", width=1), brush=pg.mkBrush("#ffd43b"))
        self._fa_palette = np.array([pg.mkBrush(*c) for c in analysis.time_colours(64)], dtype=object)
        self._fa_brushes = self._fa_palette[:0]
        self.plot_view = QComboBox()
        self.plot_view.addItem("Graphs against time", "time")
        self.plot_view.addItem("Force against angle", "fa")
        self.plot_view.addItem("Force and angle together (two axes)", "both")
        self.plot_view.addItem("Load-cell calibration", "calib")
        self.plot_view.setToolTip("Force against angle draws the joint's angle on one axis and the force on the other, picture by picture, "
                                  "from purple (the start of the trial) to yellow (the end). It shows how far the joint moved while the force "
                                  "rose. Force and angle together draws both against time in one graph: the angle on the left axis, the force "
                                  "(smoothed as the Smoothing box says) on the right. Load-cell calibration shows the line the newtons come from, "
                                  "with this trial's peak and plateau on it. These need a trial with force.")
        self._plot_choice = "both"           # 0.33.0: a trial with force opens on one large graph, angle and force with two axes; the choice is kept while the tab is open
        self.plot_view.currentIndexChanged.connect(lambda _: self._plot_view_changed())
        self.plot_view.setEnabled(False)
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(CURSOR_COLOUR, width=2))
        self.force_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(CURSOR_COLOUR, width=2))
        self.window = pg.LinearRegionItem(brush=(30, 120, 220, 40))
        self.force_window_mirror = pg.LinearRegionItem(brush=(30, 120, 220, 40), movable=False)       # the same window, shown on the force plot
        self.window_peak_mark = pg.LinearRegionItem(brush=(255, 212, 59, 110), movable=False)         # the stretch of the sliding-window peak (0.29.0)
        for line in self.window_peak_mark.lines:
            line.setPen(pg.mkPen(CURSOR_COLOUR, width=1))
        self.window_peak_mark.setZValue(5)
        self.zero_region = pg.LinearRegionItem(brush=(150, 150, 150, 70), movable=True)               # where the zero is taken from (0.30.0)
        for line in self.zero_region.lines:
            line.setPen(pg.mkPen((170, 170, 170), width=1, style=Qt.DashLine))
        self.zero_region.setZValue(4)
        self.window.sigRegionChanged.connect(self._mirror_window)
        self.window.sigRegionChanged.connect(self.update_summary)
        # force and angle together, one graph with two vertical axes (0.31.0): the angle on the left, the force (smoothed as the Smoothing box says)
        # on the right, on the same time axis as the other graphs. Its window is movable too and moves the same window.
        self.both_plot = TwoAxisPlot()
        self.both_plot.setLabel("left", "Angle (deg)")
        self.both_plot.setLabel("bottom", "Time (s)")
        self.both_plot.showGrid(x=True, y=True, alpha=0.25)
        self.both_plot.getAxis("left").enableAutoSIPrefix(False)
        self.both_plot.addLegend(offset=(10, 5))
        self.both_plot.use_right(True)
        self.both_plot.setXLink(self.plot)
        self.both_plot.setVisible(False)
        self.both_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(CURSOR_COLOUR, width=2))
        self.window_both = pg.LinearRegionItem(brush=(30, 120, 220, 40))
        self.window_peak_both = pg.LinearRegionItem(brush=(255, 212, 59, 110), movable=False)
        for line in self.window_peak_both.lines:
            line.setPen(pg.mkPen(CURSOR_COLOUR, width=1))
        self.window_peak_both.setZValue(5)
        self.window_both.sigRegionChanged.connect(self._both_window_moved)
        self._both_items = []                              # the curves in the left and right view boxes (made again when the trial or the smoothing changes)
        self._both_right_items = []
        self.chk_trails = QCheckBox("Show trails")
        self.chk_trails.setChecked(True)
        self.chk_trails.toggled.connect(lambda _: self.show_frame(self.slider.value()))
        self.event_combo = QComboBox()
        self.event_combo.setPlaceholderText("Go to marker")
        self.event_combo.activated.connect(self.jump_to_event)
        self.btn_hold = QPushButton("Window: MVIC markers")
        self.btn_hold.clicked.connect(self.snap_window)

        # The checks, always in view above the graphs (0.32.0: they were the first of seven tabs, so the operator had to click back to see them),
        # and the numbers as panels under the video: Angles, Force and Dots. The reference readings are part of Angles, the calibration is a graph
        # view (Graphs: Load-cell calibration), and Fix dots is a workspace of its own (the Fix dots button).
        self.checks = ChecksChip()                             # the chip in the bottom bar; its pop-up holds the whole list
        self.checks.actionRequested.connect(self._check_action)
        self.angle_tree = AngleDotTree()                       # every angle with the dots that make it (the reference reading button is in its header)
        self.angle_tree.fixDotRequested.connect(lambda dot: self.set_fix_mode(True, dot=dot))
        self.angle_tree.referenceRequested.connect(self.edit_reference)
        self.calib = _CalibrationPane()                       # a graph view now: the line the newtons come from, with this trial's peak on it
        self.calib.setVisible(False)
        self._fix_mode = False                                 # (before the pane: it reports to the tab while it is being made)
        self.fix_dots = FixDotsPane(self)                     # put a dot right after the trial (0.30.0); a workspace since 0.32.0

        # force settings (shown only for a trial with force)
        self.picture_delay = _NoWheelSpin()
        self.picture_delay.setRange(-1.0, 2.0)
        self.picture_delay.setSingleStep(0.01)
        self.picture_delay.setDecimals(3)
        self.picture_delay.setSuffix(" s")
        self.picture_delay.setToolTip("How long after the movement a picture is time-stamped (camera, phone and USB delay). The joint angle at the "
                                      "peak force is read from the pictures stamped this long after the force peak. Measure it with a tap on the "
                                      "sensor that shows in the video; 0 = the time stamps are taken as they are.")
        self.picture_delay.valueChanged.connect(self._on_picture_delay)
        self.zero_spin = _NoWheelSpin()
        self.zero_spin.setRange(-10.0, 10.0)
        self.zero_spin.setDecimals(4)
        self.zero_spin.setSingleStep(0.001)
        self.zero_spin.setSuffix(" V")
        self.zero_spin.setToolTip("The voltage that counts as 0 N for this trial. Type it, or take it from the graph with the buttons next to it. It starts as "
                                  "the zero in use (the one set in Review, else the one taken when the trial was recorded, or the resting level of the "
                                  "recording if none was taken). Nothing changes until you press Apply zero.")
        self.btn_apply_cal = QPushButton("Apply zero")
        self.btn_apply_cal.setToolTip("Use this zero for the trial. Newtons are worked out from the recorded volts with this zero and the slope recorded with the trial "
                                      "(the active load cell regression if none was recorded). Saved in the review folder, so every number and export follows; "
                                      "the volts in the recorded files are never changed.")
        self.btn_apply_cal.clicked.connect(self.apply_force_calibration)
        self.btn_zero_stretch = QPushButton("Zero = grey stretch")
        self.btn_zero_stretch.setToolTip("Drag the grey stretch on the force graph to a place where nothing was pulling, then press this: the zero is the mean "
                                         "of the recorded volts in it (a stretch that is a multiple of 20 ms cancels 50 Hz hum exactly).")
        self.btn_zero_stretch.clicked.connect(self.zero_from_stretch)
        self.btn_zero_rest = QPushButton("Zero = rest before the push")
        self.btn_zero_rest.setToolTip("The mean of the recorded volts over the rest before the push (the stretch SOP 9.10 asks for).")
        self.btn_zero_rest.clicked.connect(self.zero_from_rest)
        self.btn_zero_reset = QPushButton("Recorded zero")
        self.btn_zero_reset.setToolTip("Take the zero set in Review back: the zero recorded with the trial is used again.")
        self.btn_zero_reset.clicked.connect(self.zero_reset)
        self.zero_label = QLabel("")
        self.zero_label.setObjectName("hint")
        self.force_smooth = _NoWheelSpin()
        self.force_smooth.setRange(0, 1000)
        self.force_smooth.setDecimals(0)
        self.force_smooth.setSingleStep(20)
        self.force_smooth.setSuffix(" ms")
        self.force_smooth.setValue(SMOOTH_DEFAULT_S * 1000)
        self.force_smooth.setToolTip("Moving average for the force curve and for the smoothed peak, plateau, rest and timing numbers. Use a multiple of "
                                     "20 ms: that removes 50 Hz mains hum exactly. 0 = raw samples. The saved files always hold the raw samples. "
                                     "The Sliding window box below is the same kind of average, used only for the Peak force number.")
        self.force_smooth.valueChanged.connect(self.on_force_smoothing)
        self.window_peak_spin = _NoWheelSpin()
        self.window_peak_spin.setRange(0, 2000)
        self.window_peak_spin.setDecimals(0)
        self.window_peak_spin.setSingleStep(20)
        self.window_peak_spin.setSuffix(" ms")
        self.window_peak_spin.setValue(WINDOW_PEAK_DEFAULT_S * 1000)
        self.window_peak_spin.setToolTip("Length of the sliding window of the sliding-window peak: the highest mean of the raw force over any "
                                         "stretch this long inside the blue window. Use a multiple of 20 ms (50 Hz hum cancels). 0 = the highest "
                                         "single sample. This is the same moving average as Smoothing, but used for one number only (Peak force): "
                                         "at the same length the two peaks are equal (unless the peak is at the very edge of the window). The peak in the PeerJ paper behind it (Slade et al. 2026) is "
                                         "the dynamometer's own peak; the paper gives no window length, 500 ms is this program's choice.")
        self.window_peak_spin.valueChanged.connect(self.on_window_peak)
        self.btn_force_win = QPushButton("Window: force > half peak")
        self.btn_force_win.setToolTip("Set the analysis window from the force itself: the stretch around the peak where the smoothed force is above half "
                                      "of the way from rest to peak (the hold). Use it when the MVIC markers were pressed early or late.")
        self.btn_force_win.clicked.connect(self.snap_to_force)
        # The force numbers and what they depend on, one table under the force graph (0.32.0): the settings widgets made above live inside its rows.
        self.force_tree = ForceTree({"sliding": self.window_peak_spin, "smoothing": self.force_smooth, "delay": self.picture_delay, "zero": self.zero_spin,
                                     "apply": self.btn_apply_cal, "zero_stretch": self.btn_zero_stretch, "zero_rest": self.btn_zero_rest,
                                     "zero_reset": self.btn_zero_reset, "zero_label": self.zero_label})
        self.force_tree.calibrationRequested.connect(lambda: self._show_plot_view("calib"))
        self.force_tree.setVisible(False)

        # The layout of 0.32.0 (from 0.14.0 the number tabs sat in a strip under video and graphs, which left both small):
        # LEFT, the trial itself: the video and its controls, the key numbers in one slim row, and under them the number panels (Angles, Force,
        # Dots). RIGHT, the signal: the checks (always in view), the graphs (the biggest part), and what acts on the analysis window and the
        # force zero. 'Fix dots' swaps the right side for the dot workspace and gives the video more room. The dividers can be dragged and
        # where they are is remembered (attach_memory). Export and More are in the participant overview.
        self.header = QFrame()
        self.header.setObjectName("reviewHeader")
        top = QHBoxLayout(self.header)
        top.setContentsMargins(0, 0, 0, 8)
        top.setSpacing(6)
        top.addWidget(QLabel("Participant"))
        top.addWidget(self.participant_combo)
        top.addWidget(self.btn_open_folder)
        top.addSpacing(10)
        top.addWidget(QLabel("Trial"))
        top.addWidget(self.trial_combo, 1)
        top.addWidget(self.btn_prev)
        top.addWidget(self.btn_next)
        top.addWidget(self.nav_label)
        top.addSpacing(10)
        top.addWidget(self.showing_label)
        top.addWidget(self.btn_data_original)
        top.addWidget(self.btn_data_reviewed)
        top.addWidget(self.edits_label)
        top.addSpacing(10)
        top.addWidget(self.btn_view_trial)
        top.addWidget(self.btn_view_overview)
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_step_back)
        ctrl.addWidget(self.btn_step_fwd)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.frame_label)
        ctrl.addWidget(self.chk_trails)
        ctrl.addWidget(self.btn_fix)
        ctrl.addWidget(self.btn_privacy)
        ctrl.addWidget(self.zoom_label)
        ctrl.addWidget(self.zoom_combo)
        self.zero_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        # the key numbers in one slim row under the video (built below, with the cards); the number panels under that
        left = QVBoxLayout()
        left.setSpacing(4)
        left.addWidget(self.video, 1)
        left.addLayout(ctrl)
        video_block = QWidget()
        left.setContentsMargins(0, 0, 6, 0)
        video_block.setLayout(left)
        self.left_split = QSplitter(Qt.Vertical)
        self.left_split.addWidget(video_block)
        self.left_split.addWidget(self.angle_tree)
        self.left_split.setStretchFactor(0, 3)
        self.left_split.setStretchFactor(1, 2)
        self.left_split.setChildrenCollapsible(False)
        self.left_split.setSizes(DEFAULT_MAIN_SPLIT)
        self._main_sizes = list(DEFAULT_MAIN_SPLIT)        # the last divider position seen while the table was on show (it is hidden, size 0, while dots are fixed)
        self.splitter = self.left_split                    # (the name the divider had before 0.32.0)
        plots = QWidget()
        pl = QVBoxLayout(plots)
        pl.setContentsMargins(6, 0, 0, 0)
        pl.setSpacing(4)
        vrow = QHBoxLayout()
        vrow.addWidget(self.plot_view)
        vrow.addStretch(1)
        vrow.addWidget(self.btn_hold)                      # the analysis window and the markers belong to the graphs: on their row
        vrow.addWidget(self.btn_force_win)
        vrow.addWidget(self.event_combo)
        pl.addLayout(vrow)
        pl.addWidget(self.plot, 3)
        pl.addWidget(self.force_plot, 2)
        pl.addWidget(self.fa_plot, 5)
        pl.addWidget(self.both_plot, 6)
        pl.addWidget(self.calib, 5)
        pl.addWidget(self.force_tree, 3)
        # the Fix dots workspace: the right side while the dots are being fixed (the Done button, or Esc, goes back)
        self.btn_fix_done = QPushButton("Done")
        self.btn_fix_done.setText("Done  Esc")
        self.btn_fix_done.setToolTip("Back to the graphs (Esc). Everything you saved stays.")
        self.btn_fix_done.clicked.connect(lambda: self.set_fix_mode(False))
        fix_page = QWidget()
        fl = QVBoxLayout(fix_page)
        fl.setContentsMargins(6, 0, 0, 0)
        fl.addWidget(self.fix_dots, 1)
        self._plots_page, self._fix_page = plots, fix_page
        for page in (plots, fix_page):                          # (a stack is as wide as its widest page: only the page on show says how narrow it may be, so the
            page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)      # divider can go further right while dots are fixed)
        plots.setMinimumWidth(RIGHT_LEAST_WIDTH)
        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(plots)
        self.right_stack.addWidget(fix_page)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.addWidget(self.left_split)
        self.top_splitter.addWidget(self.right_stack)
        self.top_splitter.setStretchFactor(0, 3)
        self.top_splitter.setStretchFactor(1, 2)
        self.top_splitter.setChildrenCollapsible(False)
        self.top_splitter.setSizes(DEFAULT_TOP_SPLIT)
        self._top_sizes = {"normal": list(DEFAULT_TOP_SPLIT), "fix": list(FIX_TOP_SPLIT)}      # the divider between video and right side, per mode
        # the operator's verdict on the open trial (0.28.0): shown, and changed here when a trial turns out compromised (or fine) later
        self.verdict_bar = VerdictControl()
        self.verdict_bar.setEnabled(False)
        self.verdict_bar.changed.connect(self._verdict_changed)
        # the bottom bar (it never moves): in Trial mode the checks chip, the verdict and Next not judged; while dots are fixed, what is being done and Done
        decide = QFrame()
        decide.setObjectName("bottomBar")
        dl = QHBoxLayout(decide)
        dl.setContentsMargins(8, 4, 8, 4)
        dl.setSpacing(8)
        dl.addWidget(self.checks, 1)
        dl.addWidget(self.verdict_bar)
        dl.addWidget(self.btn_next_unjudged)
        fixbar = QFrame()
        fixbar.setObjectName("bottomBar")
        fl2 = QHBoxLayout(fixbar)
        fl2.setContentsMargins(8, 4, 8, 4)
        fl2.setSpacing(8)
        self.fix_bar_label = ElidedLabel("")
        fl2.addWidget(self.fix_bar_label, 1)
        fl2.addWidget(self.btn_fix_done)
        self.bottom_stack = QStackedWidget()
        self.bottom_stack.addWidget(decide)
        self.bottom_stack.addWidget(fixbar)
        self.title.setObjectName("flagLine")               # only for an UNFINISHED or STOPPED trial (nothing else says it): hidden otherwise
        self.title.setVisible(False)
        self.trial_page = QWidget()
        tp = QVBoxLayout(self.trial_page)
        tp.setContentsMargins(0, 4, 0, 0)
        tp.setSpacing(4)
        tp.addWidget(self.title)
        tp.addWidget(self.top_splitter, 1)
        tp.addWidget(self.bottom_stack)
        self.overview = OverviewPage(trial_row, parallel=True)         # (the numbers are worked out in worker processes: the window stays free)
        self.overview.set_menus(self.export_menu, self.more_menu)
        self.overview.openRequested.connect(self.open_base)
        self.overview.exportRequested.connect(lambda valid_only: self.export_session(valid_only=valid_only))
        self.stack = QStackedWidget()
        self.stack.addWidget(self.trial_page)
        self.stack.addWidget(self.overview)
        lay = QVBoxLayout(self)
        lay.addWidget(self.header)
        lay.addWidget(self.stack, 1)
        # keys that work only inside the Review tab (the Live tab keeps its letter keys)
        for seq, slot in (("Alt+Left", self.prev_trial), ("Alt+Right", self.next_trial), ("Alt+N", self.next_unjudged)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        # The picture keys work on the Trial page only (the overview's table keeps its arrow keys), and not while a number is being typed (a spin
        # box or the note keeps its own keys): the arrows and , . step one picture, with Shift ten; Esc leaves the Fix dots workspace.
        for seq, d in ((",", -1), (".", 1), ("Left", -1), ("Right", 1), ("Shift+Left", -10), ("Shift+Right", 10)):
            sc = QShortcut(QKeySequence(seq), self.trial_page)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d=d: self.step_picture(d))
        sc = QShortcut(QKeySequence("F"), self.trial_page)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(lambda: self.set_fix_mode(not self._fix_mode) if self.btn_fix.isEnabled() else None)
        sc = QShortcut(QKeySequence("1"), self.trial_page)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(lambda: self.verdict_bar.btn_valid.click() if self.verdict_bar.isEnabled() and not self._fix_mode else None)
        for n, key in enumerate(verdicts.REASONS, start=2):
            sc = QShortcut(QKeySequence(str(n)), self.trial_page)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda k=key: self.verdict_bar.reason_buttons[k].click() if self.verdict_bar.isEnabled() and not self._fix_mode else None)
        sc = QShortcut(QKeySequence("Escape"), self.trial_page)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(lambda: self.set_fix_mode(False) if self._fix_mode else None)
        self._entries = []                  # review_nav.TrialEntry of the open participant
        self._folder = None                 # the open participant's folder
        self._index = -1                    # the entry that is open in the Trial view
        self._extra_folders = []            # participant folders added with 'Open folder...' (outside the recordings folder)
        self._pending_restore = None        # the remembered participant and trial, restored the first time the tab is shown
        self._syncing = False               # True while the combos are set from code (no loading then)
        self._apply_style()
        self.refresh_participants()
        self._update_nav_buttons()                          # (nothing is open yet: no previous, next or next-not-judged)
        self._redraw_timer = QTimer(self)                   # the picture is drawn again at the new size a moment after the video area is resized
        self._redraw_timer.setSingleShot(True)              # (a divider dragged, the window resized: it was cut off or too small until the next picture)
        self._redraw_timer.setInterval(60)
        self._redraw_timer.timeout.connect(lambda: self.trial is not None and self.show_frame(self.slider.value()))
        self.video.installEventFilter(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)
        self._memory_path = None
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(800)             # saved a moment after the last drag
        self._layout_timer.timeout.connect(self._save_layout)
        self.top_splitter.splitterMoved.connect(lambda *_: self._top_moved())
        self.left_split.splitterMoved.connect(lambda *_: self._layout_timer.start())

    def shutdown(self):
        """The program is closing: stop the overview's worker processes (they would end by themselves when this window's pipes close, but not at once)."""
        self.overview.shutdown()

    # ------------------------------------------------------------- remembered layout
    def _top_moved(self):
        self._top_sizes["fix" if self._fix_mode else "normal"] = self.top_splitter.sizes()
        self._layout_timer.start()

    def _main_now(self):
        """The divider between the video and the angle table: as it is while both are on show, else as it was last (while dots are fixed the
        table is hidden and its share is 0; that 0 was saved until 0.33.0 and was then ignored when the layout was put back)."""
        sizes = self.left_split.sizes()
        if not self._fix_mode and len(sizes) == 2 and all(s > 0 for s in sizes):
            self._main_sizes = sizes
        return list(self._main_sizes)

    def layout_state(self):
        """The divider positions: {'v': layout version, 'top': [left, right side], 'main': [video, number panels], 'fix': [left, right side] while fixing dots}."""
        top = dict(self._top_sizes)
        now = self.top_splitter.sizes()
        if all(s > 0 for s in now):
            top["fix" if self._fix_mode else "normal"] = now
        return {"v": LAYOUT_VERSION, "top": top["normal"], "main": self._main_now(), "fix": top["fix"]}

    def apply_layout_state(self, state):
        """Put remembered divider positions back. Anything that does not fit is ignored, and so is a layout saved by a version of the tab
        with other proportions (no 'v', or an older one)."""
        if not isinstance(state, dict) or state.get("v") != LAYOUT_VERSION:
            return
        def good(key):
            try:
                sizes = [int(x) for x in state[key]]
            except (KeyError, TypeError, ValueError):
                return None
            return sizes if len(sizes) == 2 and all(s > 0 for s in sizes) else None
        top, main, fix = good("top"), good("main"), good("fix")
        if top:
            self._top_sizes["normal"] = top
        if fix:
            self._top_sizes["fix"] = fix
        if main:
            self._main_sizes = main
            self.left_split.setSizes(main)
        self.top_splitter.setSizes(self._top_sizes["fix" if self._fix_mode else "normal"])

    def attach_memory(self, path):
        """Remember between launches (own top-level parts of settings.json): the divider positions ('review_ui') and the participant and
        trial that were open, with the participant folders added by hand ('review_nav'; put back the first time the tab is shown)."""
        self._memory_path = Path(path)
        data = session_memory.read_all(path)
        self.apply_layout_state(data.get("review_ui") or {})
        nav = data.get("review_nav")
        if isinstance(nav, dict):
            extras = [Path(p) for p in nav.get("extra_folders") or [] if isinstance(p, str) and Path(p).is_dir()]
            for f in extras:
                self._add_extra_folder(f)
            self.refresh_participants()
            folder = nav.get("participant_dir")
            if isinstance(folder, str) and Path(folder).is_dir():
                self._pending_restore = (Path(folder), nav.get("trial") if isinstance(nav.get("trial"), str) else None)

    def _save_layout(self):
        if self._memory_path is not None:
            session_memory.write_part(self._memory_path, review_ui=self.layout_state())

    def _save_nav(self):
        if self._memory_path is not None and self._folder is not None:
            base = self._entries[self._index].base if 0 <= self._index < len(self._entries) else None
            session_memory.write_part(self._memory_path, review_nav={"participant_dir": str(self._folder), "trial": base,
                                                                      "extra_folders": [str(f) for f in self._extra_folders][-20:]})

    # ------------------------------------------------------------- participants and trials (0.29.0)
    @property
    def dir(self):
        """The recordings folder: the participants are its sub-folders. Set by the Live tab when the folder is changed."""
        return self._dir

    @dir.setter
    def dir(self, path):
        self._dir = Path(path)
        self.refresh_participants()

    @staticmethod
    def _same(a, b):
        try:
            return Path(a).resolve() == Path(b).resolve()
        except OSError:
            return str(a) == str(b)

    def _add_extra_folder(self, folder):
        if not any(self._same(folder, f) for f in self._extra_folders):
            self._extra_folders.append(Path(folder))

    def refresh_participants(self):
        """Fill the participant drop-down: the folders of the recordings folder that hold trials, then any added with 'Open folder...'.
        The open participant stays selected. Nothing is loaded."""
        found = []
        if review_nav.has_trials(self._dir):                         # (trials saved straight into the recordings folder)
            found.append((self._dir.name or str(self._dir), self._dir, len(review_nav.trial_paths(self._dir))))
        found += review_nav.participant_folders(self._dir)
        for f in self._extra_folders:
            if not any(self._same(f, p) for _, p, _ in found) and review_nav.has_trials(f):
                found.append((f.name, f, len(review_nav.trial_paths(f))))
        self._syncing = True
        try:
            self.participant_combo.clear()
            self.participant_combo.addItem("Choose a participant...", None)
            for name, path, n in found:
                self.participant_combo.addItem(f"{name}  ·  {n} trial{'' if n == 1 else 's'}", str(path))
                self.participant_combo.setItemData(self.participant_combo.count() - 1, str(path), Qt.ToolTipRole)
            self._select_participant_item(self._folder)
        finally:
            self._syncing = False

    def _select_participant_item(self, folder):
        i = 0
        if folder is not None:
            for k in range(1, self.participant_combo.count()):
                if self._same(self.participant_combo.itemData(k), folder):
                    i = k
                    break
        self.participant_combo.setCurrentIndex(i)

    def _participant_chosen(self, index):
        data = self.participant_combo.itemData(index)
        if data:
            self.set_participant(Path(data))

    def open_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open a participant's folder (or a folder of participants)",
                                             str(self._folder.parent if self._folder is not None else self._dir))
        if d:
            self.open_folder(Path(d))

    def open_folder(self, folder):
        """A participant folder (such as Pilot/P01) opens that participant; a folder of participant folders (such as Pilot) adds them all to
        the participant drop-down and opens the first."""
        kind, folders = review_nav.resolve_folder(folder)
        if kind == "none":
            QMessageBox.information(self, "No trials there", f"No trial (*_meta.json) was found in {folder} or in the folders inside it.")
            return
        for f in folders:
            self._add_extra_folder(f)
        self.refresh_participants()
        self.set_participant(folders[0])

    def set_participant(self, folder, base=None, load=True):
        """Show the trials of this participant folder in the trial drop-down and in the overview, and (`load`) open `base`, else the first
        trial without a verdict, else the first."""
        folder = Path(folder)
        self._folder = folder
        self._entries = review_nav.list_trials(folder)
        self._index = -1
        if not any(self._same(folder, self.participant_combo.itemData(k)) for k in range(1, self.participant_combo.count())):
            self._add_extra_folder(folder)
            self.refresh_participants()
        self._syncing = True
        try:
            self._select_participant_item(folder)
        finally:
            self._syncing = False
        self._fill_trial_combo()
        self.overview.set_participant(folder.name, self._entries)
        self._after_view_change()
        if load and self._entries:
            i = next((k for k, e in enumerate(self._entries) if e.base == base), None) if base else None
            if i is None:
                i = review_nav.next_unjudged(self._entries, -1)
            self._open_index(0 if i is None else i)
        else:
            self._update_nav_buttons()
            self._save_nav()

    def _fill_trial_combo(self):
        self._syncing = True
        try:
            self.trial_combo.clear()
            for i, e in enumerate(self._entries):
                self.trial_combo.addItem(e.label, i)
                self.trial_combo.setItemData(i, e.base if e.readable else f"{e.base}: {e.error}", Qt.ToolTipRole)
            if not self._entries:
                self.trial_combo.addItem("No trials in this folder" if self._folder is not None else "Choose a participant first", None)
                self.trial_combo.setCurrentIndex(0)
            else:
                self.trial_combo.setCurrentIndex(self._index if 0 <= self._index < len(self._entries) else -1)
        finally:
            self._syncing = False

    def _trial_chosen(self, index):
        i = self.trial_combo.itemData(index)
        if isinstance(i, int):
            self._open_index(i)

    def _open_index(self, i):
        """Open the i-th trial of the list. A trial that cannot be opened is named, and the one before it stays on screen."""
        e = self._entries[i]
        previous = self._index
        try:
            self.load(e.meta_path)
        except Exception as err:  # noqa: BLE001 - shown to the operator
            QMessageBox.warning(self, "Cannot open trial", f"{e.base}\n\n{err}")
            self._syncing = True
            try:
                self.trial_combo.setCurrentIndex(previous if 0 <= previous < len(self._entries) else -1)
            finally:
                self._syncing = False
            return False
        return True

    def open_base(self, base):
        """Open the trial of this name (a line of the overview was activated) and show the Trial page."""
        i = next((k for k, e in enumerate(self._entries) if e.base == base), None)
        if i is not None and self._open_index(i):
            self.show_view("trial")

    def prev_trial(self):
        self._step(-1)

    def next_trial(self):
        self._step(1)

    def _step(self, d):
        i = self._index + d
        if 0 <= i < len(self._entries):
            self._open_index(i)

    def next_unjudged(self):
        i = review_nav.next_unjudged(self._entries, self._index)
        if i is not None:
            self._open_index(i)

    def _update_nav_buttons(self):
        n = len(self._entries)
        self.btn_prev.setEnabled(self._index > 0)
        self.btn_next.setEnabled(0 <= self._index < n - 1)
        left = review_nav.next_unjudged(self._entries, self._index)
        self.btn_next_unjudged.setEnabled(left is not None)
        counts = review_nav.counts(self._entries)
        self.nav_label.setText((f"{self._index + 1} of {n} · {counts['not judged']} left" if 0 <= self._index < n else f"{n} trials · {counts['not judged']} left") if n else "")
        self.btn_next_unjudged.setToolTip(
            f"Jump to the next trial that has no verdict yet (Alt+N). {counts['not judged']} of {n} not judged yet." if left is not None
            else ("Every readable trial of this participant has a verdict." if n else "Choose a participant first."))

    def _sync_navigation(self, meta_path):
        """A trial was loaded (by the drop-down, the overview, Open a single trial file, or code): make the header say which one it is."""
        meta_path = Path(meta_path)
        folder = meta_path.parent
        i = None
        if self._folder is not None and self._same(folder, self._folder):
            i = next((k for k, e in enumerate(self._entries) if self._same(e.meta_path, meta_path)), None)
        if i is None:                                     # another participant, or a trial recorded since the list was made
            fresh = self._folder is not None and self._same(folder, self._folder)
            self._folder = folder
            self._entries = review_nav.list_trials(folder)
            if not fresh and not any(self._same(folder, self.participant_combo.itemData(k)) for k in range(1, self.participant_combo.count())):
                self._add_extra_folder(folder)
                self.refresh_participants()
            self.overview.set_participant(folder.name, self._entries)
            i = next((k for k, e in enumerate(self._entries) if self._same(e.meta_path, meta_path)), -1)
            self._index = i
            self._fill_trial_combo()
        self._index = i
        self._syncing = True
        try:
            self._select_participant_item(folder)
            self.trial_combo.setCurrentIndex(i)
        finally:
            self._syncing = False
        self._update_nav_buttons()
        self._save_nav()

    def show_view(self, name):
        """'trial' or 'overview'."""
        over = name == "overview"
        if over and self._fix_mode:
            self.set_fix_mode(False)
        self.btn_view_overview.setChecked(over)
        self.btn_view_trial.setChecked(not over)
        self.btn_fix.setVisible(not over)
        self.btn_privacy.setVisible(not over)
        self.stack.setCurrentIndex(1 if over else 0)
        self._after_view_change()

    def _after_view_change(self):
        if self.stack.currentIndex() == 1:
            self.overview.set_settings(self.force_smooth.value() / 1000.0, self.window_peak_spin.value() / 1000.0, self._source)
            self.overview.ensure_rows()
            if 0 <= self._index < len(self._entries):
                self.overview.select_base(self._entries[self._index].base)

    def showEvent(self, e):
        super().showEvent(e)
        if self._pending_restore is not None:             # the participant and trial of the last time (not at start-up: the Live tab comes first)
            (folder, base), self._pending_restore = self._pending_restore, None
            try:
                self.set_participant(folder, base=base)
            except Exception:  # noqa: BLE001 - a remembered trial that cannot be opened must not stop the tab from showing
                pass
        elif self._folder is not None:                    # trials recorded since the tab was last seen
            fresh = review_nav.list_trials(self._folder)
            if [x.base for x in fresh] != [x.base for x in self._entries]:
                cur = self._entries[self._index].base if 0 <= self._index < len(self._entries) else None
                self._entries = fresh
                self._index = next((k for k, x in enumerate(fresh) if x.base == cur), -1)
                self._fill_trial_combo()
                self.overview.set_participant(self._folder.name, self._entries)
                self._update_nav_buttons()
            self.refresh_participants()

    def eventFilter(self, obj, ev):
        if obj is self.video and ev.type() == QEvent.Resize and self.trial is not None:
            self._redraw_timer.start()
        return super().eventFilter(obj, ev)

    def _apply_style(self):
        """The stylesheet from the application's palette (a light or a dark theme); set again only when it comes out different, because
        setting a stylesheet can itself touch the palette."""
        css = build_stylesheet(QApplication.palette())
        if css != getattr(self, "_css", None):
            self._css = css
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()

    # ------------------------------------------------------------- loading
    def open_dialog(self):
        """Open one trial file (More menu): the header then shows its participant."""
        path, _ = QFileDialog.getOpenFileName(self, "Open trial", str(self._folder or self._dir), "Trial (*_meta.json)")
        if path:
            try:
                self.load(path)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Cannot open trial", str(e))

    def load(self, meta_path, keep_view=False):
        """Open a trial (for the data chosen under Data). `keep_view`: the same trial again after a fix, with the blue window and the picture where they were."""
        kept_window = tuple(self.window.getRegion()) if keep_view and self.trial is not None else None
        kept_picture = self.slider.value() if keep_view and self.trial is not None else 0
        self.btn_play.setChecked(False)
        self.timer.stop()
        tr = Trial(meta_path, source=self._source)
        old, self.trial = self.trial, tr
        if old is not None:
            if not self._same(old.meta_path, tr.meta_path):
                self._close_privacy()                      # (the Privacy window reads the old trial's video)
            old.close()                                    # (the previous trial's video was never let go of before 0.29.0)
        tr.window_peak_s = float(self.window_peak_spin.value()) / 1000.0
        m = tr.meta
        self.verdict_bar.set_verdict(verdicts.read_verdict(tr.meta_path), title="Verdict on this trial:", tip=tr.base)
        self.verdict_bar.setEnabled(True)
        flags = ((["UNFINISHED: the program stopped before saving; the video may not open"] if tr.unfinished else [])
                 + ([f"STOPPED: {m['stopped_because']}"] if m.get("stopped_because") else []))
        self.title.setText("  |  ".join(flags))                  # (the one line that says what nothing else says; hidden when there is none)
        self.title.setVisible(bool(flags))
        self.video.setToolTip(f"{m['participant']}  |  {m['test_label']}"
                              + (f"  |  target {m['target_deg']:g} deg" if isinstance(m.get("target_deg"), (int, float)) else "")
                              + f"\n{m['frames']} frames, {m['duration_s']} s"
                              + (f"  |  ruler {tr.ruler_cm:g} cm = {tr.px_per_cm:.2f} px/cm" if tr.px_per_cm and tr.ruler_cm else "  |  no ruler scale")
                              + (f"\n{m['gaps']['count']} gaps with no pictures, the longest {m['gaps']['max_s']:.2f} s" if (m.get("gaps") or {}).get("count") else "")
                              + (f"\n✎ Reviewed data: {'; '.join(tr.review_notes)}" if tr.source == "reviewed" and tr.review_notes else ""))
        self._show_edits(tr)
        self.plot.clear()
        self.plot.addItem(self.cursor)
        self.plot.addItem(self.window)
        tr.force_smooth_s = float(self.force_smooth.value()) / 1000.0     # before the force is drawn (it was set after, so the strip used the default)
        self._show_force_curve(tr)
        has_f = tr.has_force()
        self.force_tree.setVisible(has_f)
        self._apply_plot_view(draw=False)
        self.plot.setLabel("bottom", "" if has_f else "Time (s)")         # the time axis label is under the lower plot when there are two
        for k, a in enumerate(tr.angle_defs):
            off = tr.offset_of(a.name)
            self.plot.plot(tr.t, tr.deg_view(a.name), pen=pg.mkPen(PENS[k % len(PENS)], width=3 if a.primary else 1.5),
                           name=a.title + (" (from reference)" if off else ""))
            if off:                                                                   # the dots' angle as a thin dashed line beside it (0.33.0)
                self.plot.plot(tr.t, tr.deg[a.name], pen=pg.mkPen(PENS[k % len(PENS)], width=1, style=Qt.DashLine), name=a.title + " (dots)")
            if a.primary and m["target_deg"] is not None:
                tol = m["tolerance_deg"]
                band = pg.LinearRegionItem(values=(m["target_deg"] - tol, m["target_deg"] + tol), orientation="horizontal",
                                           brush=(60, 180, 60, 45), movable=False)
                for line in band.lines:
                    line.setPen(pg.mkPen((60, 150, 60), style=Qt.DashLine))
                self.plot.addItem(band)
        self.event_combo.clear()
        for i, t, lab in tr.events:
            self.event_combo.addItem(f"{lab}  @ {t:.2f} s", i)
            line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(EVENT_GREY, style=Qt.DashLine),
                                   label=lab, labelOpts={"position": 0.95, "color": EVENT_GREY, "rotateAxis": (1, 0)})
            self.plot.addItem(line)
            if tr.has_force():
                self.force_plot.addItem(pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(EVENT_GREY, style=Qt.DashLine)))
        self.event_combo.setCurrentIndex(-1)                                    # (a list to jump from: nothing is chosen until it is used)
        self.btn_hold.setEnabled(tr.hold_window() is not None)
        self.btn_force_win.setEnabled(tr.has_force())
        self.force_smooth.setEnabled(tr.has_force())
        self.window_peak_spin.setEnabled(tr.has_force())
        self.btn_fix.setEnabled(tr.cap is not None)                              # (the dots are fixed on the video)
        self.btn_privacy.setEnabled(tr.cap is not None)                          # (and so are the crop and masks)
        if self._fix_mode and tr.cap is None:
            self.set_fix_mode(False)
        self._show_zero(tr)
        if not keep_view:
            self._place_zero_region(tr)
        self.window.setRegion(kept_window or tr.hold_window() or (tr.t[0], tr.t[-1]))        # the MVIC start-end window when markers exist
        self._draw_both()                                                      # the two-axis graph (shown only when chosen)
        if not keep_view and len(tr.t):                                        # 0.33.0: it is the graph a trial opens on, and the others are linked to its time axis:
            lo, hi = float(tr.t[0]), float(tr.t[-1])                            # a graph that was never on show has not found the range of the data
            if has_f and len(tr.force_t):
                lo, hi = min(lo, float(tr.force_t[0])), max(hi, float(tr.force_t[-1]))
            self.both_plot.setXRange(lo, hi, padding=0.02)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(tr.rows) - 1)
        self.slider.setValue(min(kept_picture, len(tr.rows) - 1))
        self.slider.blockSignals(False)
        self.fix_dots.set_trial(tr)
        self.show_frame(self.slider.value())
        self.update_summary()
        self._sync_navigation(tr.meta_path)                                    # the header says which participant and trial this is

    def _show_edits(self, tr):
        """The header says how many fixes this trial has in the review layer ('✎ 2 fixes'), and when they are not in the numbers shown."""
        rv = review_layer.Review.load(tr.meta_path)
        n = len(rv.dots) + (1 if rv.zero else 0)
        if not n:
            self.edits_label.setText("")
            self.edits_label.setToolTip("")
            return
        shown = tr.source == "reviewed"
        self.edits_label.setText(f"✎ {n} fix{'' if n == 1 else 'es'}" + ("" if shown else " (not shown)"))
        self.edits_label.setToolTip("; ".join(tr.review_notes) if shown and tr.review_notes else "Fixes made in Review. Choose Reviewed to see the trial with them.")

    def _show_zero(self, tr):
        """The Zero box and the line next to it: the zero in use, and where it came from."""
        if not tr.has_force():
            self.zero_label.setText("")
            return
        zero, where = tr.current_zero_v()
        self.zero_spin.setValue(zero)
        self.zero_label.setText(where)
        self.btn_zero_reset.setEnabled(tr.zero_edit is not None)

    # ------------------------------------------------------------ playback
    def _plot_view_changed(self):
        """The graph chooser was used (by the reviewer or by a button that asks for a view): remember it for the next trial."""
        if self.plot_view.isEnabled():
            self._plot_choice = self.plot_view.currentData()
        self._apply_plot_view()

    def _apply_plot_view(self, draw=True):
        """The graph on show: angle and force together on two axes (the default), the angle and the force one above the other, force against
        angle, or the calibration. All but the plain angle graph need force; a trial without force always shows the angle graph (the chooser then
        reads 'against time' and is off) and the next trial with force gets the remembered choice back.
        `draw` False: only which plots are shown (on loading a trial, the last update_summary draws it once, with the trial's window)."""
        has_f = self.trial is not None and self.trial.has_force()
        self.plot_view.setEnabled(has_f)
        i = self.plot_view.findData(self._plot_choice if has_f else "time")
        if i >= 0 and i != self.plot_view.currentIndex():
            self.plot_view.blockSignals(True)
            self.plot_view.setCurrentIndex(i)
            self.plot_view.blockSignals(False)
        fa = has_f and self.plot_view.currentData() == "fa"
        both = has_f and self.plot_view.currentData() == "both"
        calib = has_f and self.plot_view.currentData() == "calib"
        self.plot.setVisible(not (fa or both or calib))
        self.force_plot.setVisible(has_f and not (fa or both or calib))
        self.fa_plot.setVisible(fa)
        self.both_plot.setVisible(both)
        self.calib.setVisible(calib)
        if both and draw:
            self._draw_both()
        if fa and draw:
            if self._fa_dirty or self._fa_path is None:
                self._update_force_angle()
            else:
                self._update_force_angle_window()
                self._update_fa_cursor(self.slider.value())

    def _update_force_angle(self, fs=None):
        """Draw the force-against-angle path: a thin line through every picture, a dot per picture coloured by time (purple to yellow), larger
        dots inside the analysis window, a star at the peak force (with the angle read at the peak, as in the Force tab), and a ring that
        follows the video. The force is paired with the picture that shows the same moment (the picture delay box)."""
        tr, pl = self.trial, self.fa_plot
        pl.clear()
        pl.addItem(self._fa_cursor)
        self._fa_cursor.setData([], [])
        self._fa_cursor_shown, self._fa_dirty = False, False
        self._fa_path = analysis.force_angle_path(tr, self.picture_delay.value()) if tr is not None else None
        path = self._fa_path
        if path is None or not path["ok"].any():
            pl.setTitle("No picture has both an angle and a force.", color="#bbbbbb", size="10pt")
            return
        ok = path["ok"]
        x, y = path["angle"][ok], path["force"][ok]
        self._fa_brushes = self._fa_palette[np.rint(np.linspace(0, len(self._fa_palette) - 1, len(x))).astype(int)]     # purple to yellow
        pl.setLabel("bottom", f"{path['angle_name']} (deg)")
        pl.setLabel("left", f"Force ({path['unit']})")
        pl.setTitle("purple = start of the trial, yellow = its end;  large dots = the analysis window;  star = peak force",
                    color="#bbbbbb", size="9pt")
        pl.plot(x, y, pen=pg.mkPen((190, 190, 190, 110), width=1))
        pl.addItem(pg.ScatterPlotItem(x=x, y=y, size=6, pen=None, brush=list(self._fa_brushes)))
        pl.addItem(self._fa_dots)
        pl.addItem(self._fa_star)
        self._update_force_angle_window(fs)
        self._update_fa_cursor(self.slider.value())

    def _update_force_angle_window(self, fs=None):
        """What depends on the analysis window only, so dragging the window does not redraw the whole path: the larger dots of the
        pictures whose force lies in the window, and the star at the peak force (`fs`: the window's force summary, if already made).
        The window is in force time, like the Force tab: the picture stamped t shows the force of t - picture delay."""
        path = self._fa_path
        if path is None or not path["ok"].any():
            return
        ok = path["ok"]
        t0, t1 = self.window.getRegion()
        ft = path["force_t"][ok]
        inside = (ft >= t0) & (ft <= t1)
        self._fa_dots.setData(x=path["angle"][ok][inside], y=path["force"][ok][inside], brush=list(self._fa_brushes[inside]))
        if fs is None:
            fs = self.trial.force_summary(t0, t1, self.picture_delay.value())
        at_peak = (fs["angles_at_peak_deg"].get(fs["main_angle"], math.nan) if fs is not None else math.nan)    # (on the scale of the path: from the reference when there is one)
        if fs is not None and not math.isnan(at_peak):
            self._fa_star.setData(x=[at_peak], y=[fs["peak"]])
        else:
            self._fa_star.setData(x=[], y=[])

    def _on_picture_delay(self, _=None):
        self._fa_dirty = True                             # the force paired with each picture changed: the path is drawn afresh
        self.update_summary()

    def _update_fa_cursor(self, i):
        """The ring on the force-against-angle path that marks the picture shown in the video."""
        path = self._fa_path
        if path is None or self.plot_view.currentData() != "fa" or not 0 <= i < len(path["t"]) or not path["ok"][i]:
            if self._fa_cursor_shown:                     # (show_frame calls this for every picture: nothing to do when already empty)
                self._fa_cursor.setData([], [])
                self._fa_cursor_shown = False
            return
        self._fa_cursor.setData([float(path["angle"][i])], [float(path["force"][i])])
        self._fa_cursor_shown = True

    def _show_force_curve(self, tr):
        """The force in its own strip under the angle plot: every raw sample faint, the smoothed curve (the one the numbers come from) bold."""
        fp = self.force_plot
        fp.clear()
        self._fa_dirty = True                             # the force changed (new trial, smoothing, calibration): so does the force-angle path
        if not tr.has_force():
            return
        fp.addItem(self.force_cursor)
        fp.addItem(self.force_window_mirror)
        fp.addItem(self.window_peak_mark)
        fp.addItem(self.zero_region)
        unit, ys, _ = tr.force_signal()
        raw = tr.force_n if unit == "N" else tr.force_v
        faint = pg.PlotDataItem(tr.force_t, raw, pen=pg.mkPen((220, 60, 60, 45), width=1))
        faint.setDownsampling(auto=True, method="peak")
        fp.addItem(faint)
        if tr.force_smooth_s > 0:
            curve = pg.PlotDataItem(tr.force_t, ys, pen=pg.mkPen((235, 70, 70), width=2.5))
            curve.setDownsampling(auto=True, method="peak")
            fp.addItem(curve)
        fp.setLabel("left", f"Force ({unit})")
        y = ys if tr.force_smooth_s > 0 else raw
        lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
        pad = 0.06 * max(hi - lo, 1e-9)
        fp.setYRange(lo - pad, hi + pad, padding=0)
        self._mirror_window()

    def _mirror_window(self, *_):
        self.force_window_mirror.setRegion(self.window.getRegion())
        self.window_both.setRegion(self.window.getRegion())              # (a no-op when it is already there: nothing loops)

    # ------------------------------------------------------------- the review layer: what is changed is saved there, never in the trial
    def set_source(self, name):
        """'original' or 'reviewed': what every number, graph and export reads. The open trial is read again for it."""
        if name not in ("original", "reviewed"):
            return
        self._source = name
        self.btn_data_original.setChecked(name == "original")
        self.btn_data_reviewed.setChecked(name == "reviewed")
        if self.trial is not None:
            self.load(self.trial.meta_path, keep_view=True)
        self._after_view_change()

    def _save_review(self, rv):
        """Write the edit list in the review layer. A failure is shown and nothing is changed."""
        try:
            rv.save(software_version=SOFTWARE_VERSION)
        except OSError as e:                              # a locked or read-only folder (OriginalsProtected is one too)
            QMessageBox.warning(self, "Not saved", f"The change could not be saved in the review folder:\n{e}\n\nNothing was changed.")
            return False
        return True

    def _after_edit(self):
        """A fix was saved: show the reviewed data (a fix is invisible in Original), read the trial again where it was, mark it in the list."""
        if self._source != "reviewed":
            self._source = "reviewed"
            self.btn_data_original.setChecked(False)
            self.btn_data_reviewed.setChecked(True)
        path = self.trial.meta_path
        self.load(path, keep_view=True)
        if 0 <= self._index < len(self._entries) and self._same(self._entries[self._index].meta_path, path):
            e = self._entries[self._index]
            e.refresh_edited()
            self.trial_combo.setItemText(self._index, e.label)
        self._after_view_change()

    def current_index(self):
        return self.slider.value()

    def current_time(self):
        return float(self.trial.t[self.slider.value()]) if self.trial is not None else 0.0

    def step_picture(self, d):
        if self.trial is not None:
            self.slider.setValue(max(0, min(self.slider.maximum(), self.slider.value() + d)))

    def rewind_video(self):
        if self.trial is not None:
            self.trial._pos = -2                           # (something else read the video: the next picture is looked up, not assumed)

    # ------------------------------------------------------------- the Fix dots workspace and what the checks ask for (0.32.0)
    def set_fix_mode(self, on, dot=None):
        """Open (or close) the Fix dots workspace: the video takes about two thirds of the width, the dot tools replace the checks, graphs and
        number panels. `dot`: the dot to work on. Needs a trial with a video."""
        on = bool(on) and self.trial is not None and self.trial.cap is not None and self.stack.currentIndex() == 0
        if on != self._fix_mode:
            self._top_sizes["fix" if self._fix_mode else "normal"] = self.top_splitter.sizes()      # (where the divider was in the mode that ends)
            self._fix_mode = on
            self.right_stack.setCurrentIndex(1 if on else 0)
            self._plots_page.setMinimumWidth(0 if on else RIGHT_LEAST_WIDTH)
            self._fix_page.setMinimumWidth(FIX_LEAST_WIDTH if on else 0)
            self.bottom_stack.setCurrentIndex(1 if on else 0)
            self.zoom_label.setVisible(on)
            self.zoom_combo.setVisible(on)
            self.angle_tree.setVisible(not on)
            self.top_splitter.setSizes(self._top_sizes["fix" if on else "normal"])
            if on:
                self.fix_dots.refresh()
        self.btn_fix.setChecked(self._fix_mode)
        if on and dot is not None:
            i = self.fix_dots.dot.findText(dot)
            if i >= 0:
                self.fix_dots.dot.setCurrentIndex(i)
        self._update_fix_bar()
        if self.trial is not None:
            self.show_frame(self.slider.value())          # (the dot's cross and the clicks are drawn on the picture only in this mode)
        if on:
            self.video.setFocus()                           # (so the picture keys work at once)

    def _update_fix_bar(self):
        """The line in the bottom bar while dots are fixed: which dot, what is waiting to be saved, and the keys."""
        if not self._fix_mode:
            return
        dot, pending = self.fix_dots.current_dot(), len(self.fix_dots.pending)
        self.fix_bar_label.setText(f"Fixing {dot or 'a dot'}"
                                   + (f"  ·  {pending} picture{'' if pending == 1 else 's'} clicked, not saved" if pending else "")
                                   + "  ·  arrow keys or , . step one picture, Shift ten")

    def _show_plot_view(self, key):
        """Choose a graph view by its name ('time', 'fa', 'both', 'calib'), when the trial has the force it needs."""
        i = self.plot_view.findData(key)
        if i >= 0 and self.plot_view.isEnabled():
            self.plot_view.setCurrentIndex(i)

    def _check_action(self, action, arg):
        """A button on a check was pressed (review_checks.ChecksPanel): go to what puts it right."""
        if self.trial is None:
            return
        if action == "fix_dots":
            self.set_fix_mode(True, dot=arg or None)
        elif action == "zero":
            self.force_tree.open_zero()                    # the Above rest row opens and the Zero box takes the keyboard
        elif action == "window":
            self.snap_to_force()
        elif action == "calibration":
            self._show_plot_view("calib")
        elif action == "privacy":
            self.open_privacy()

    def open_privacy(self):
        """The Privacy window for the trial on screen (0.34.0): crop and masks drawn after the recording, protected copies in the review folder."""
        tr = self.trial
        if tr is None or tr.cap is None:
            return None
        self._close_privacy()
        dlg = PrivacyDialog(tr, self, changed=self._privacy_changed)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        self._privacy_dialog = dlg
        dlg.show()
        return dlg

    def _close_privacy(self):
        """Close the Privacy window if one is open (it belongs to one trial)."""
        dlg, self._privacy_dialog = getattr(self, "_privacy_dialog", None), None
        if dlg is not None:
            try:
                dlg.close()
            except RuntimeError:                           # (its C++ object is already gone)
                pass

    def _privacy_changed(self):
        """The crop, masks or protected copies of the trial changed: its checks (and the chip in the bottom bar) are worked out again.
        Nothing is reloaded: no number of the trial changed, and the Privacy window is still reading its video."""
        if self.trial is not None:
            self.update_summary()

    # ---- force zero
    def apply_force_calibration(self):
        """Use the zero in the box (typed, or put there by a button): newtons from the recorded volts with this zero and the slope recorded with the
        trial (else the active load cell regression's). Saved in the review layer, so every number and export follows."""
        self._save_zero(self.zero_spin.value(), "typed")

    def _save_zero(self, zero_v, source, region=None):
        tr = self.trial
        if tr is None or not tr.has_force():
            return False
        slope = tr.slope_for_zero()
        if slope is None:
            QMessageBox.information(self, "No calibration", "This trial has no slope (it was recorded in volts) and there is no load cell regression yet "
                                    "(limbtrack/load_cell_calibrations.json). Add one with LoadCellCalibration/add_calibration.py; "
                                    "the recordings can be converted afterwards.")
            return False
        rv = review_layer.Review.load(tr.meta_path)
        rv.set_zero(zero_v, source, slope, region)
        if not self._save_review(rv):
            return False
        self._after_edit()
        return True

    def zero_from_stretch(self):
        tr = self.trial
        if tr is None or not tr.has_force():
            return
        t0, t1 = (float(v) for v in self.zero_region.getRegion())
        v = tr.raw_volts_mean(t0, t1)
        if math.isnan(v):
            QMessageBox.information(self, "No force there", "The grey stretch holds no force samples: drag it onto the force graph.")
            return
        self._save_zero(v, "stretch", (t0, t1))
        if abs((t1 - t0) / 0.02 - round((t1 - t0) / 0.02)) > 0.05:
            self.zero_label.setText(self.zero_label.text() + f"  (the stretch is {1000 * (t1 - t0):.0f} ms: not a multiple of 20 ms, so mains hum does not cancel exactly)")

    def zero_from_rest(self):
        tr = self.trial
        if tr is None or not tr.has_force():
            return
        ri = tr.force_rest()
        if ri is None or math.isnan(ri["level"]):
            QMessageBox.information(self, "No rest found", "There is no clear rest before the push in this trial (less than half a second of force before it). "
                                    "Drag the grey stretch to a place where nothing was pulling and use 'Zero = grey stretch'.")
            return
        v = tr.raw_volts_mean(ri["t_from"], ri["t_to"])
        if not math.isnan(v):
            self._save_zero(v, "rest", (ri["t_from"], ri["t_to"]))

    def zero_reset(self):
        tr = self.trial
        if tr is None or tr.zero_edit is None:
            return
        rv = review_layer.Review.load(tr.meta_path)
        rv.clear_zero()
        if self._save_review(rv):
            self._after_edit()

    def _place_zero_region(self, tr):
        """The grey stretch starts on the rest before the push, else on the first second of the force."""
        if not tr.has_force():
            return
        ri = tr.force_rest()
        if ri is not None and not math.isnan(ri["level"]) and ri["t_to"] > ri["t_from"]:
            self.zero_region.setRegion((ri["t_from"], ri["t_to"]))
        else:
            self.zero_region.setRegion((float(tr.force_t[0]), float(tr.force_t[0]) + 1.0))

    # ---- dots
    def commit_dot_edit(self, dot, kind, rows=None, **params):
        """Save a fix of `dot` (see dot_edits.py) in the review layer and read the trial again with it. True when it was saved."""
        tr = self.trial
        if tr is None:
            return False
        rv = review_layer.Review.load(tr.meta_path)
        rv.add_dot_edit(dot, kind, rows=rows, **params)
        if not self._save_review(rv):
            return False
        self._after_edit()
        return True

    def undo_dot_edit(self, dot):
        tr = self.trial
        rv = review_layer.Review.load(tr.meta_path) if tr is not None else None
        if rv is None or not rv.undo_dot_edit(dot):
            return False
        if self._save_review(rv):
            self._after_edit()
            return True
        return False

    def reset_dot(self, dot):
        tr = self.trial
        rv = review_layer.Review.load(tr.meta_path) if tr is not None else None
        if rv is None or not any(e["dot"] == dot for e in rv.dots):
            return False
        rv.reset_dot(dot)
        if self._save_review(rv):
            self._after_edit()
            return True
        return False

    def _video_clicked(self, x, y):
        if self.trial is not None and self._fix_mode:
            self.fix_dots.on_video_click(x, y)

    def _zoom_rect(self, w, h, i):
        """(x, y, width, height) of the part of the picture to show when it is magnified (Fix dots only), centred on the dot being fixed: where you
        last clicked, else where the program has the dot in this picture, else its usual place. None for Fit. Never leaves the picture."""
        factor = float(self.zoom_combo.currentData() or 1.0)
        if not self._fix_mode or factor <= 1.0 or self.trial is None:
            return None
        cx, cy = self.fix_dots.focus_point(i) if hasattr(self.fix_dots, "focus_point") else (None, None)
        if cx is None:
            dot = self.fix_dots.current_dot()
            tr = self.trial
            if dot in tr.tracks and not math.isnan(tr.tracks[dot][0][i]):
                cx, cy = float(tr.tracks[dot][0][i]), float(tr.tracks[dot][1][i])
        if cx is None:
            cx, cy = w / 2.0, h / 2.0
        cw, ch = max(int(round(w / factor)), 16), max(int(round(h / factor)), 16)
        x0 = int(round(min(max(cx - cw / 2.0, 0), w - cw)))
        y0 = int(round(min(max(cy - ch / 2.0, 0), h - ch)))
        return x0, y0, cw, ch

    def _draw_review_marks(self, img, i):
        """What the Fix dots workspace shows on the picture: the dot it works on, and the clicks. (The ring round a dot that was fixed is drawn by Trial.drawn.)"""
        tr = self.trial
        if self._fix_mode:
            dot = self.fix_dots.current_dot()
            if dot in tr.tracks and not math.isnan(tr.tracks[dot][0][i]):
                x, y = int(round(tr.tracks[dot][0][i])), int(round(tr.tracks[dot][1][i]))
                cv2.line(img, (x - 24, y), (x + 24, y), (255, 255, 0), 1, cv2.LINE_AA)
                cv2.line(img, (x, y - 24), (x, y + 24), (255, 255, 0), 1, cv2.LINE_AA)
            for x, y, text, colour in self.fix_dots.marks(i):
                cv2.circle(img, (int(round(x)), int(round(y))), 11, colour, 2, cv2.LINE_AA)
                if text:
                    cv2.putText(img, text, (int(round(x)) + 14, int(round(y)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    def open_review_folder(self):
        """Show the folder where this participant's review files are kept (nothing is created here)."""
        meta = self.trial.meta_path if self.trial is not None else (self._entries[0].meta_path if self._entries else None)
        if meta is None:
            QMessageBox.information(self, "No participant", "Choose a participant first.")
            return
        d = review_layer.participant_review_dir(meta)
        if not d.is_dir():
            QMessageBox.information(self, "Nothing saved yet", f"Nothing has been saved in the review layer for this participant yet.\nIt will be in:\n{d}")
            return
        if hasattr(os, "startfile"):
            os.startfile(d)

    def show_frame(self, i):
        if self.trial is None:
            return
        img = self.trial.drawn(i, trails=self.chk_trails.isChecked())
        self.cursor.setValue(self.trial.t[i])
        self.force_cursor.setValue(self.trial.t[i])
        self.both_cursor.setValue(self.trial.t[i])
        self._update_fa_cursor(i)
        self.frame_label.setText(f"{i + 1} / {len(self.trial.t)}  ·  {self.trial.t[i]:.2f} / {self.trial.t[-1]:.2f} s")
        if img is None:
            return
        self._draw_review_marks(img, i)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.video.picture_size = (w, h)
        rect = self._zoom_rect(w, h, i)
        self.video.view_rect = rect
        if rect is not None:
            x0, y0, cw, ch = rect
            rgb = np.ascontiguousarray(rgb[y0:y0 + ch, x0:x0 + cw])
        ph, pw = rgb.shape[:2]
        self.video.show_picture(QPixmap.fromImage(QImage(rgb.data, pw, ph, 3 * pw, QImage.Format_RGB888).copy()))
        if self._fix_mode:
            self.fix_dots.update_where()
            self._update_fix_bar()

    def toggle_play(self):
        if self.btn_play.isChecked() and self.trial is not None:
            fps = self.trial.meta.get("measured_fps") or 30
            self.timer.start(int(1000 / fps))
            self.btn_play.setText("Pause")
        else:
            self.timer.stop()
            self.btn_play.setText("Play")

    def advance(self):
        i = self.slider.value() + 1
        if i > self.slider.maximum():
            self.btn_play.setChecked(False)
            self.toggle_play()
            return
        self.slider.setValue(i)

    # ------------------------------------------------------------- markers
    def jump_to_event(self, index):
        if self.trial is not None and index >= 0:
            self.slider.setValue(int(self.event_combo.itemData(index)))

    def on_force_smoothing(self, ms):
        if self.trial is None:
            return
        self.trial.force_smooth_s = float(ms) / 1000.0
        self._show_force_curve(self.trial)
        self._draw_both()
        self.update_summary()

    def snap_to_force(self):
        w = self.trial.force_window() if self.trial is not None else None
        if w:
            self.window.setRegion(w)

    def snap_window(self):
        w = self.trial.hold_window() if self.trial is not None else None
        if w:
            self.window.setRegion(w)

    # ------------------------------------------------------------- summary
    def _rows(self):
        t0, t1 = self.window.getRegion()
        return self.trial.summary(t0, t1), (t0, t1)

    def update_summary(self):
        if self.trial is None:
            return
        rows, (t0, t1) = self._rows()
        fs = self.trial.force_summary(t0, t1, self.picture_delay.value())      # (None without force) for the users below
        ft = analysis.force_timing(self.trial, t0, t1) if fs is not None else None
        fp = analysis.force_peaks(self.trial, t0, t1, timing=ft)               # the sliding-window peak and the times to peak (0.29.0)
        self._update_window_peak_mark(fp)
        if self.plot_view.currentData() == "fa" and self.trial.has_force():
            if self._fa_dirty:
                self._update_force_angle(fs)                 # the force or the picture delay changed: the whole path
            else:
                self._update_force_angle_window(fs)          # only the window moved: its dots and the star
        self.angle_tree.set_data(self.trial, rows, fs, (t0, t1))
        if self.trial.has_force() and fs is not None:
            self.force_tree.set_data(self.trial, t0, t1, fs, ft, fp, self.picture_delay.value())
        else:
            self.force_tree.set_none("No force was recorded in this trial." if not self.trial.has_force()
                                      else "Force was recorded but there are no samples in this window.")
        self._update_calibration(t0, t1, fs)
        self.checks.set_items(analysis.trial_check_items(self.trial, t0, t1))

    def on_window_peak(self, ms):
        if self.trial is None:
            return
        self.trial.window_peak_s = float(ms) / 1000.0
        self.update_summary()

    def _update_window_peak_mark(self, fp):
        """The yellow bar on the force plot (and on the two-axis graph): the stretch the sliding-window peak was taken from."""
        if math.isnan(fp["window_peak_start_s"]):
            self.window_peak_mark.setVisible(False)
            self.window_peak_both.setVisible(False)
            return
        for mark in (self.window_peak_mark, self.window_peak_both):
            mark.setRegion((fp["window_peak_start_s"], fp["window_peak_end_s"]))
            mark.setVisible(True)

    def _both_window_moved(self):
        r = self.window_both.getRegion()
        if tuple(r) != tuple(self.window.getRegion()):
            self.window.setRegion(r)

    def _draw_both(self):
        """Force and angle together: every angle (the main one bold, with the tolerance band) on the left axis, the force on the right axis
        (newtons if calibrated, else volts; smoothed as the Smoothing box says), the markers, the blue window (draggable here too), the yellow
        sliding-window bar and the line that follows the picture. Made again when the trial or the smoothing changes; moving the window or the
        picture only moves their lines."""
        pl, tr = self.both_plot, self.trial
        pl.clear()
        for it in self._both_right_items:
            pl.remove_right(it)
        self._both_items, self._both_right_items = [], []
        if tr is None or not tr.has_force():
            return
        m = tr.meta
        for k, a in enumerate(tr.angle_defs):
            off = tr.offset_of(a.name)
            self._both_items.append(pl.plot(tr.t, tr.deg_view(a.name), pen=pg.mkPen(PENS[k % len(PENS)], width=3 if a.primary else 1.5),
                                            name=a.title + (" (from reference)" if off else ""), connect="finite"))
            if off:                                                                   # the dots' angle as a thin dashed line beside it (0.33.0)
                pl.plot(tr.t, tr.deg[a.name], pen=pg.mkPen(PENS[k % len(PENS)], width=1, style=Qt.DashLine), name=a.title + " (dots)", connect="finite")
            if a.primary and m.get("target_deg") is not None:
                tol = m["tolerance_deg"]
                band = pg.LinearRegionItem(values=(m["target_deg"] - tol, m["target_deg"] + tol), orientation="horizontal", brush=(60, 180, 60, 45), movable=False)
                for line in band.lines:
                    line.setPen(pg.mkPen((60, 150, 60), style=Qt.DashLine))
                pl.addItem(band)
        unit, ys, _ = tr.force_signal()
        y = ys if tr.force_smooth_s > 0 else (tr.force_n if unit == "N" else tr.force_v)
        curve = pg.PlotDataItem(tr.force_t, y, pen=pg.mkPen((235, 70, 70), width=2.5))
        curve.setDownsampling(auto=True, method="peak")                          # (not setClipToView: pyqtgraph 0.14 fails when it is set before the item has a view)
        self._both_right_items.append(pl.add_right(curve))
        pl.set_right_label(f"Force ({unit}{', smoothed ' + format(tr.force_smooth_s * 1000, '.0f') + ' ms' if tr.force_smooth_s > 0 else ', raw'})", "#eb4646")
        if np.isfinite(y).any():
            lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
            pad = 0.06 * max(hi - lo, 1e-9)
            pl.set_right_range(lo - pad, hi + pad)
        for i, t, lab in tr.events:
            pl.addItem(pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(EVENT_GREY, style=Qt.DashLine)))
        for item in (self.window_both, self.window_peak_both, self.both_cursor):
            pl.addItem(item)
        self.window_both.setRegion(self.window.getRegion())
        if self.trial is not None and len(tr.t):
            self.both_cursor.setValue(float(tr.t[min(self.slider.value(), len(tr.t) - 1)]))

    def edit_reference(self):
        if self.trial is None:
            return
        if ReferenceDialog(self.trial, self).exec():
            self.load(self.trial.meta_path, keep_view=True)       # (an angle may now be shown from its reading: the graphs and the picture are made again)

    def show_reference_validation(self):
        """The agreement of the app with the hand instrument over every trial in the recordings folder."""
        folder = self.dir                                   # the recordings folder (all participants)
        rows = reference.collect_rows(folder)
        if not rows:
            QMessageBox.information(self, "No reference readings", "No trial in " + str(folder) + " has a reference reading yet. Type the "
                                    "goniometer or inclinometer reading in the Live tab (Reference reading) before recording, or add it in "
                                    "the Reference tab of a trial.")
            return
        ReferenceValidationDialog(folder, rows, self).exec()

    def _update_calibration(self, t0, t1, fs=None):
        """The Calibration tab: the regression behind this trial's newtons (or the active one when the trial has none), and its precision."""
        tr = self.trial
        if tr is None or not tr.has_force():
            self.calib.show_regression(None, ["No force was recorded in this trial."])
            return
        fs = fs if fs is not None else tr.force_summary(t0, t1, self.picture_delay.value())
        c = tr.force_cal or {}
        reg = tr.regression
        lines = []
        if fs is None or fs["unit"] != "N":
            reg = reg or active_regression()
            lines += ["This trial is in volts: no zero was taken, so there are no newtons yet.",
                      "Press Recalculate newtons (with the zero in the box) to get them from the active regression.", ""]
        elif reg is None:
            lines += [f"The newtons of this trial come from {c.get('method', 'an unknown calibration')} ({c.get('n_per_v', 0):.2f} N per volt),",
                      "which has no stored regression, so no precision can be given. Recalculate newtons to use the active regression.", ""]
            reg = active_regression()
        marks = []
        if reg is not None:
            if fs is not None and fs["unit"] == "N" and tr.regression is not None:
                lo, hi = reg.range_kg
                lines += [f"This trial (zero {c.get('offset_v', 0):+.4f} V, {c.get('n_per_v', 0):.2f} N per volt):"]
                for name, key in (("peak", "peak"), ("plateau mean", "plateau_mean")):
                    f = fs[key]
                    if f is None or math.isnan(f):
                        continue
                    kg = f / GRAVITY
                    sd = COVERAGE * float(reg.sd_newton(f))
                    lines.append(f"  {name:13s}{f:8.1f} N = {kg:6.1f} kg   +-{sd:.1f} N  ({100 * sd / max(abs(f), 1e-9):.2f} %), about 95 %")
                    marks.append((kg, name))
                    if kg > hi * 1.001:
                        lines.append(f"  !! the {name} is above the heaviest calibration load ({hi:g} kg): the line is extended, not measured there")
                lines += ["  (the calibration only; the zero adds the noise of the short average it was taken from, and a pull that is not along",
                          "   the cell's axis is not covered)", ""]
            lines += reg.summary_lines()
        else:
            lines.append("No load cell regression is stored.")
        self.calib.show_regression(reg, lines, marks)

    def export_overlay_video(self):
        """Render the overlay picture for every frame from the raw video and the CSV and save it as a video file."""
        if self.trial is None:
            return
        tr = self.trial
        out = review_layer.review_file(tr.meta_path, f"{tr.base}_overlay_{tr.source}.mp4")           # (the review layer: never beside the trial's files)
        n = len(tr.rows)
        fps = tr.meta.get("video_fps_written") or tr.meta.get("measured_fps") or 30.0
        writer, done = None, 0
        progress = QProgressDialog("Drawing the overlay video...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        part = review_layer.temp_video_path(out)
        try:
            for i in range(n):
                img = tr.drawn(i, trails=self.chk_trails.isChecked())
                if img is None:
                    break
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Could not open the video file for writing.")
                writer.write(img)
                done = i + 1
                progress.setValue(done)
                if progress.wasCanceled():
                    break
        except Exception as e:  # noqa: BLE001 - shown to the operator
            QMessageBox.warning(self, "Export failed", str(e))
            return
        finally:
            if writer is not None:
                writer.release()
            progress.close()
        if done:
            review_layer.finish_video(part, out)
        QMessageBox.information(self, "Exported", f"Saved {out.name} in the review folder ({done} of {n} frames).")

    def _verdict_changed(self, v, reasons, note):
        """The verdict was changed in Review: saved next to the trial (its history keeps what was decided on site)."""
        if self.trial is None:
            return
        try:
            verdicts.write_verdict(self.trial.meta_path, v, reasons, note, where=verdicts.IN_REVIEW)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Verdict not saved", f"The verdict could not be saved next to the trial:\n{e}")
            self.verdict_bar.set_verdict(verdicts.read_verdict(self.trial.meta_path))
            return
        if 0 <= self._index < len(self._entries) and self._entries[self._index].meta_path == self.trial.meta_path:
            e = self._entries[self._index]
            e.refresh_verdict()
            self.trial_combo.setItemText(self._index, e.label)         # the drop-down shows the verdict (✓ ✗ ·)
            self.overview.update_verdict(e.base)
            self._update_nav_buttons()
        self.update_summary()                                       # (the Check tab names the verdict)

    def export_session(self, valid_only=False):
        """One row per trial of the open participant, with the verdict and the 10 % repeatability check between trials of the same position
        (0.29.0: `valid_only` writes only the trials marked valid, to `<participant>_session_summary_valid.csv`, and the check is made among
        them). Rows the overview has already worked out are used; the others are worked out here. A trial marked compromised is in the file
        (with its reasons) but, as everywhere, not counted in the 10 % check."""
        folder = self._folder if self._folder is not None else (self.trial.folder if self.trial is not None else None)
        if folder is None:
            return
        entries = review_nav.list_trials(folder)
        smooth, window = self.force_smooth.value() / 1000.0, self.window_peak_spin.value() / 1000.0
        self.overview.cancel()
        self.overview.set_settings(smooth, window, self._source)
        rows, failed = [], []
        todo = sum(1 for e in entries if e.readable and self.overview.row_for(e.base) is None)
        progress = QProgressDialog("Working out the numbers of every trial...", "Cancel", 0, todo, self) if todo > 3 else None
        if progress is not None:
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
        done = 0
        try:
            for e in entries:
                if not e.readable:
                    failed.append(f"{e.base}: {e.error}")
                    continue
                row = self.overview.row_for(e.base)
                if row is None:
                    row, err = trial_row(e.meta_path, smooth, window, self._source)
                    if row is None:
                        failed.append(f"{e.base}: {err}")
                        continue
                    self.overview.store(e, row)
                    done += 1
                    if progress is not None:
                        progress.setValue(done)
                        QApplication.processEvents()
                        if progress.wasCanceled():
                            return
                rows.append(dict(row))
        finally:
            if progress is not None:
                progress.close()
        judged = {verdicts.VALID: 0, verdicts.COMPROMISED: 0, "not judged": 0}
        for r in rows:
            judged[r["verdict"] if r.get("verdict") in (verdicts.VALID, verdicts.COMPROMISED) else "not judged"] += 1
        if valid_only:
            rows = [r for r in rows if r.get("verdict") == verdicts.VALID]
        if not rows:
            QMessageBox.warning(self, "Nothing to export", ("No trial of this participant is marked valid yet in " if valid_only else "No readable trials in ")
                                + str(folder))
            return
        analysis.repeatability(rows)
        pid = entries[0].participant if entries and entries[0].participant else folder.name
        tag = "reviewed" if any(r.get("data_source") == "reviewed" for r in rows) else "original"     # (what the numbers are: 'reviewed' only if a trial has fixes)
        out = review_layer.exports_dir(entries[0].meta_path) / f"{pid}_session_summary_{tag}{'_valid' if valid_only else ''}.csv"
        part = review_layer.guard(out.with_name(out.name + ".tmp"))                                     # (the review layer only; the file appears whole or not at all)
        part.parent.mkdir(parents=True, exist_ok=True)
        try:
            analysis.write_session_csv(rows, part)
            os.replace(part, out)
        except OSError as e:
            QMessageBox.warning(self, "Not saved", f"{out.name} could not be written in the review folder:\n{e}\n\nIf it is open in Excel, close it and export again.")
            return
        text = f"Saved {out.name} in the review folder (exports): {len(rows)} trial{'' if len(rows) == 1 else 's'}"
        text += (" (the valid ones)." if valid_only else
                 f" ({judged['valid']} valid, {judged['compromised']} compromised, {judged['not judged']} not judged yet).")
        if judged["not judged"]:
            text += f"\n{judged['not judged']} trial{'' if judged['not judged'] == 1 else 's'} {'has' if judged['not judged'] == 1 else 'have'} no verdict yet: judge them (Next not judged), then export again."
        if failed:
            text += "\nNot readable: " + "; ".join(failed)
        QMessageBox.information(self, "Exported", text)
        return out

    def export(self):
        if self.trial is None:
            return
        rows, (t0, t1) = self._rows()
        tag = self.trial.source                                                            # 'original', or 'reviewed' when the trial has fixes
        out = review_layer.review_file(self.trial.meta_path, f"{self.trial.base}_summary_{tag}.csv")      # (the review layer: never beside the trial's files)
        with review_layer.atomic(out, newline="") as fh:
            w = csv.writer(fh)
            ident = {"participant": self.trial.meta.get("participant"), "test_key": self.trial.meta.get("test_key"),
                     "trial": self.trial.base, "tolerance_deg": self.trial.meta.get("tolerance_deg"),
                     "software_version": self.trial.meta.get("software_version"),
                     "window_source": "MVIC markers" if self.trial.hold_window() and tuple(map(float, (t0, t1))) == tuple(self.trial.hold_window())
                     else "window as set (not the MVIC markers)",
                     "data_source": tag, "review_edits": "; ".join(self.trial.review_notes),
                     "force_zero_source": self.trial.zero_source()}
            v = verdicts.read_verdict(self.trial.meta_path) or {}                          # 0.29.0: the operator's verdict goes with the numbers
            ident.update(verdict=v.get("verdict") or "", verdict_reasons=", ".join(verdicts.REASONS.get(r, r) for r in v.get("reasons") or []),
                         verdict_note=v.get("note") or "")
            fs = self.trial.force_summary(t0, t1, self.picture_delay.value())
            fnames = ["force_unit", "force_peak", "force_t_peak_s", "force_plateau_mean", "force_rest_mean", "main_angle_at_peak_force_deg",
                      "picture_delay_s", "force_n_per_v", "force_zero_v", "force_calibration_note", "force_smoothing_ms", "force_mains_hum_pct_of_peak",
                      "force_raw_peak", "force_regression_id", "force_peak_95pct_n", "force_relaxed_level", "force_peak_above_relaxed",
                      "force_window_peak", "force_window_peak_ms", "force_window_peak_at_s", "force_time_to_peak_s", "force_time_to_window_peak_s",
                      "force_peak_20ms", "main_angle_at_peak_force_from_reference_deg"]
            fp = analysis.force_peaks(self.trial, t0, t1)
            cal = self.trial.force_cal or {}
            reg = self.trial.regression
            peak_ci = (COVERAGE * float(reg.sd_newton(fs["peak"])) if reg is not None and fs is not None and fs["unit"] == "N"
                       and not math.isnan(fs["peak"]) else "")
            fvals = ([fs["unit"], fs["peak"], fs["t_peak_s"], fs["plateau_mean"], fs["rest_mean"], fs["main_angle_at_peak_deg"],
                      fs["picture_delay_s"], cal.get("n_per_v"), cal.get("offset_v"), self.trial.force_cal_note,
                      fs["smoothing_s"] * 1000.0, fs["mains_hum_pct_of_peak"], fs["raw_peak"],
                      reg.id if reg is not None else "", peak_ci, fs["relaxed"], fs["peak_minus_relaxed"],
                      fp["window_peak"], fp["window_peak_ms"], fp["window_peak_at_s"], fp["time_to_peak_s"], fp["time_to_window_peak_s"],
                      fs["peak_20ms"], (fs["angles_at_peak_deg"].get(fs["main_angle"], "") if fs["main_angle"] in fs["angle_offsets_deg"] else "")]
                     if fs is not None else [""] * len(fnames))
            fvals = ["" if isinstance(v, float) and math.isnan(v) else round(v, 3) if isinstance(v, float) else v for v in fvals]
            w.writerow(list(ident) + ["window_start_s", "window_end_s"] + fnames + list(rows[0]))
            for r in rows:
                w.writerow(list(ident.values()) + [round(t0, 3), round(t1, 3)] + fvals
                           + [round(v, 3) if isinstance(v, float) else v for v in r.values()])
        extra = review_layer.review_file(self.trial.meta_path, f"{self.trial.base}_displacement_{tag}.csv")
        drows = self.trial.displacement(t0, t1)
        keys = list(dict.fromkeys(k for r in drows for k in r))
        with review_layer.atomic(extra, newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["window_start_s", "window_end_s"] + keys)
            for r in drows:
                w.writerow([round(t0, 3), round(t1, 3)] + [("" if isinstance(r.get(k), float) and math.isnan(r[k]) else
                                                              round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys])
        written = [out.name, extra.name] + self._write_reviewed_files()
        QMessageBox.information(self, "Exported", f"Saved {' and '.join(written)} in the review folder")

    def _write_reviewed_files(self):
        """For reviewed data with fixes: the edited numbers as two files other programs can read, `reviewed_data.csv` (every picture, the dot
        columns, angles and pull as fixed, and which dots were fixed) and `reviewed_force.csv` (every force sample, newtons with the zero in use)."""
        tr = self.trial
        if tr.source != "reviewed":
            return []
        head = list(tr.rows[0]) + ["dots_edited"]
        out_d = review_layer.review_file(tr.meta_path, "reviewed_data.csv")
        with review_layer.atomic(out_d, newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            for i, r in enumerate(tr.rows):
                w.writerow([r.get(k, "") for k in head[:-1]] + [";".join(d for d, f in tr.edited_sets.items() if i in f)])
        names = [out_d.name]
        if tr.has_force():
            out_f = review_layer.review_file(tr.meta_path, "reviewed_force.csv")
            with review_layer.atomic(out_f, newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_s", "volts", "newtons", "zero_v", "zero_source"])
                zero, _ = tr.current_zero_v()
                src = tr.zero_source()
                for t, v, n in zip(tr.force_t.tolist(), tr.force_v.tolist(), tr.force_n.tolist()):
                    w.writerow([round(t, 5), round(v, 6), "" if n != n else round(n, 3), round(zero, 6), src])
            names.append(out_f.name)
        return names
