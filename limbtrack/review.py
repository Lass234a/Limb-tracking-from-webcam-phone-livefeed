"""Review a saved trial: scrub the video with its overlay, see angle-vs-time, get summary numbers."""
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QLineEdit, QSplitter, QTabWidget, QTextEdit, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

from . import overlay, privacy, reference, session_memory
from .displacement import window_displacements
from . import analysis
from .force import SMOOTH_DEFAULT_S, ForceCalibration, mains_amplitude, smooth_signal
from .loadcell import COVERAGE, GRAVITY, active_regression, find_regression
from .pull import COLUMNS as PULL_COLUMNS
from .angles import measure, segment_pairs
# PARKED-GHOST: from .geometry import anchored_ghost
from .protocol import AngleDef
from .results import INFO, LOST, OK, OUT
from .engine import TRAIL_MIN_STEP_PX
from .tracker import WEAK_BELOW

PENS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]
CURSOR_COLOUR = "#ffd43b"        # the line that follows the video position: it was black on the black plot background until 0.14.0
EVENT_GREY = (190, 190, 190)     # MVIC start/end and other markers (was dark grey (90, 90, 90), hard to see on black)
DEFAULT_TOP_SPLIT = [400, 600]   # video and controls : graphs (the graphs get the larger part)
DEFAULT_MAIN_SPLIT = [620, 380]  # video and graphs : the number tabs


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


PLATEAU_TRIM = 0.2         # the plateau of a hold is its middle part: the first and last 20 % of the window are left out
REST_SECONDS = 1.0         # "rest" is the second before the window starts
REST_MIN_FRAMES = 5


class Trial:
    """A recorded trial loaded from disk."""

    def __init__(self, meta_path):
        meta_path = Path(meta_path)
        self.meta_path = meta_path
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
        self.px_per_cm = (self.meta.get("scale") or {}).get("px_per_cm_at_marking")
        self.ruler_dots = (self.meta.get("scale") or {}).get("ruler_dots_px_at_marking")      # two fixed points, full-picture pixels
        self.ruler_cm = (self.meta.get("scale") or {}).get("ruler_length_cm")
        self.pull = ({c: np.array([_f(r[c]) for r in self.rows]) for c in PULL_COLUMNS} if PULL_COLUMNS[0] in self.rows[0] else None)
        # trails: per dot, segments of (frame index array, xy array); a lost dot ends a segment (as when recording)
        self.trail_segments = {}
        for n in self.dot_names:
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
            self.trail_segments[n] = [(np.array(a), np.array(b)) for a, b in segs]
        # PARKED-GHOST: self.ghost = self.meta.get("locked_positions_px")
        # PARKED-GHOST: self.ghost_anchor = self.meta.get("ghost_anchor")
        self.cap = cv2.VideoCapture(str(folder / self.meta["files"]["raw.mp4"]))
        self._pos = -2
        # force (load cell through the DAQ): every sample, on the same time axis as the video (seconds after the first picture)
        self.force_t = self.force_v = self.force_n = None
        self.force_smooth_s = SMOOTH_DEFAULT_S        # moving average for the force curve and numbers (a multiple of 20 ms removes 50 Hz hum)
        fmeta = self.meta.get("force") or {}
        self.force_cal = fmeta.get("calibration")
        self.regression = find_regression((self.force_cal or {}).get("regression_id"))
        self.force_cal_note = ("calibration recorded with the trial" if (self.force_cal or {}).get("n_per_v") else
                               "no calibration yet: volts only")
        fname = (self.meta.get("files") or {}).get("force.csv")
        if fname and (folder / fname).exists():
            frows = list(csv.DictReader(open(folder / fname, newline="", encoding="utf-8")))
            if frows:
                self.force_t = np.array([_f(r["t_s"]) for r in frows])
                self.force_v = np.array([_f(r["volts"]) for r in frows])
                self.force_n = np.array([_f(r["newtons"]) for r in frows])      # NaN where there was no calibration
        pv = self.meta.get("privacy") or {}        # a cropped video is put back at its place on a black full-size picture
        self.crop = tuple(pv["crop_xywh"]) if pv.get("crop_xywh") else None
        self.full_size = tuple(pv["full_picture_size"]) if pv.get("full_picture_size") else None

    # ------------------------------------------------------------------ reference readings (typed from a hand instrument)
    def reference_rows(self):
        """One row per reading: the typed reference, the app's angle at the same moment, the difference (see limbtrack/reference.py).
        Worked out once (it may read the CSV) and again only after the readings change (reload_reference); do not change the rows."""
        if self._reference_rows is None:
            csv_path = self.folder / (self.meta.get("files") or {}).get("data.csv", "data.csv")
            self._reference_rows = reference.comparison(self.meta, self.reference_side, csv_path)
        return self._reference_rows

    def reference_edit_keys(self):
        """[(key, name)] a reading can be typed for in Review: the trial's angles, then the extras of its test (the typed hip flexion of a frontal trial)."""
        return [(a.name, a.title) for a in self.angle_defs] + list((self.meta.get("reference_extras") or {}).items())

    def reload_reference(self):
        self.reference_side = reference.read_sidecar(self.meta_path)
        self._reference_rows = None

    def frame(self, i):
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
        for a in self.angle_defs:
            n = a.name
            xy = {k: (d.x, d.y) for k, d in dots.items() if not d.lost}
            xy.update(derived)
            res = measure(a, xy, self.facing_right)     # geometry for drawing; numbers come from the CSV
            res.value, res.deviation = self.deg[n][i], self.dev[n][i]
            res.reference = None if math.isnan(self.ref[n][i]) else self.ref[n][i]
            flag = self.ok[n][i]
            res.status = LOST if math.isnan(res.value) else OK if flag == "1" else OUT if flag == "0" else INFO
            angles.append(res)
        head = f"{self.meta['test_label']} | {self.meta['participant']} | t = {self.t[i]:.2f} s"
        # PARKED-GHOST: ghost = None
        # PARKED-GHOST: if self.ghost and r["locked"] == "1":
            # PARKED-GHOST: live = {k: (d.x, d.y) for k, d in dots.items() if not d.lost}
            # PARKED-GHOST: ghost = anchored_ghost({k: tuple(v) for k, v in self.ghost.items()}, live, self.ghost_anchor)
        return overlay.draw(img, dots, angles, [head], trails=self.trails_upto(i) if trails else None, scale=self.scale_overlay(), derived=derived)
        # PARKED-GHOST: # with the ghost: overlay.draw(img, dots, angles, [head], ghost=ghost, ghost_segments=segment_pairs(self.angle_defs), trails=...)

    def apply_calibration(self, cal, note):
        """Newtons from the saved volts with a calibration made AFTER the recording (the volts in the files are never changed)."""
        self.force_n = cal.to_newton(self.force_v)
        self.force_cal, self.force_cal_note = cal.to_dict(), note
        self.regression = find_regression(cal.regression_id)

    def recorded_zero_v(self):
        """(zero volts, where it came from): the zero the operator took for this trial, else the resting level of the recording
        (the 2nd percentile of the 100 ms smoothed signal: the strap at rest, before and after the pull)."""
        c = self.force_cal or {}
        if c.get("zero_set", True) and c.get("zero_v") is not None:
            return float(c["zero_v"]), "the zero taken when this trial was recorded"
        rate = 1.0 / float(np.median(np.diff(self.force_t))) if len(self.force_t) > 2 else 1000.0
        return float(np.percentile(smooth_signal(self.force_v, rate, 0.1), 2)), "the resting level of the recording (no zero was taken)"

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

    def force_summary(self, t0, t1, picture_delay=0.0):
        """The force in the window: peak and when, the plateau mean, the rest level before, and the main joint angle at the peak
        (the two candidate numbers for a model input: angle at peak force, and plateau mean angle). None without force.
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
        rm = (self.force_t >= t0 - REST_SECONDS) & (self.force_t < t0)
        plateau = float(np.nanmean(y[pm])) if pm.any() else math.nan
        rest = float(np.nanmean(y[rm])) if rm.any() else math.nan
        raw = self.force_n if unit == "N" else self.force_v
        main = next((a for a in self.angle_defs if a.primary), self.angle_defs[0] if self.angle_defs else None)
        angle = math.nan
        if main is not None:
            near = np.abs(self.t - (tw[k] + picture_delay)) <= 0.1             # a picture is stamped picture_delay after it was taken
            vals = self.deg[main.name][near]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                i = int(np.argmin(np.abs(self.t - (tw[k] + picture_delay))))
                vals = self.deg[main.name][i:i + 1]
            angle = float(np.nanmean(vals)) if vals.size and not np.all(np.isnan(vals)) else math.nan
        hum = self.force_hum()
        vpeak = float(np.nanmax(yv[m]))
        vrest = float(np.nanmean(yv[rm])) if rm.any() else float(np.nanpercentile(yv, 10))
        hum_pct = 100.0 * hum / max(abs(vpeak - vrest), 1e-9) if not math.isnan(hum) else math.nan
        return {"unit": unit, "peak": float(yw[k]), "t_peak_s": float(tw[k]), "plateau_mean": plateau, "rest_mean": rest,
                "peak_minus_rest": float(yw[k]) - rest if not math.isnan(rest) else math.nan,
                "raw_peak": float(np.nanmax(raw[m])), "smoothing_s": self.force_smooth_s,
                "mains_hum_v": hum, "mains_hum_pct_of_peak": hum_pct,
                "main_angle": main.title if main is not None else "", "main_angle_at_peak_deg": angle, "picture_delay_s": picture_delay,
                "samples_in_window": int(m.sum())}

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
        out = []
        for a in self.angle_defs:
            n = a.name
            v = self.deg[n][m]
            valid = ~np.isnan(v)
            flags = [f for f, keep in zip(self.ok[n], m) if keep and f != ""]
            dev = self.dev[n][m]
            good = v[valid]
            change = float(np.mean(good[-5:]) - np.mean(good[:5])) if good.size >= 2 else math.nan
            # The angle that matters is the one during the hold, not the one typed as the target (audit VAL-01).
            span = t1 - t0
            pm = (self.t >= t0 + PLATEAU_TRIM * span) & (self.t <= t1 - PLATEAU_TRIM * span)
            pv = self.deg[n][pm]
            pv = pv[~np.isnan(pv)]
            plateau = float(np.mean(pv)) if pv.size >= REST_MIN_FRAMES else math.nan
            rm = (self.t >= t0 - REST_SECONDS) & (self.t < t0)
            rv = self.deg[n][rm]
            rv = rv[~np.isnan(rv)]
            rest = float(np.mean(rv)) if rv.size >= REST_MIN_FRAMES else math.nan
            refs = self.ref[n][m]
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
                "rest_mean_deg": rest,
                "rest_to_hold_shift_deg": (plateau - rest) if not (math.isnan(plateau) or math.isnan(rest)) else math.nan,
                "max_abs_dev_deg": float(np.nanmax(np.abs(dev))) if np.any(~np.isnan(dev)) else math.nan,
                "pct_in_tolerance": 100.0 * sum(f == "1" for f in flags) / len(flags) if flags else math.nan,
            })
        return out


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
        lay = QHBoxLayout(self)          # side by side: this tab now sits in a wide, short strip under the graphs
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.fit, 2)
        lay.addWidget(self.res, 2)
        lay.addWidget(self.text, 3)

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


class _NoWheelSpin(QDoubleSpinBox):
    """A spin box that ignores the mouse wheel (scrolling past it must not change the number)."""

    def wheelEvent(self, e):
        e.ignore()


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
        self.instrument = QComboBox()
        for k, label in reference.INSTRUMENTS.items():
            self.instrument.addItem(label, k)
        self.resolution = QComboBox()
        for v in reference.RESOLUTIONS_DEG:
            self.resolution.addItem(f"{v:g} deg", v)
        current = side if side.get("readings") else saved
        self.instrument.setCurrentIndex(max(self.instrument.findData(current.get("instrument") or "goniometer"), 0))
        self.resolution.setCurrentIndex(max(self.resolution.findData(current.get("resolution_deg") or 1.0), 0))
        form.addRow("Instrument", self.instrument)
        form.addRow("Read to", self.resolution)
        self.boxes = {}
        for key, title in trial.reference_edit_keys():
            box = QLineEdit(reference.format_values((merged.get(key) or {}).get("values_deg") or []))
            box.setPlaceholderText("e.g. 88 90 89")
            form.addRow(title, box)
            self.boxes[key] = box
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
        try:
            reference.write_sidecar(self.trial.meta_path, self.instrument.currentData(), self.resolution.currentData(), values,
                                    recorded=self.trial.meta.get("reference_readings"))
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
        try:
            out = reference.write_csv(self.rows, self.folder / "reference_validation.csv")
        except OSError as e:                              # most often: the file is open in Excel, which locks it on Windows
            QMessageBox.warning(self, "Not saved", f"reference_validation.csv could not be written in {self.folder}:\n{e}\n\n"
                                "If the file is open in Excel or another program, close it and press Save again. "
                                "The file on disk is still the earlier one.")
            return
        QMessageBox.information(self, "Saved", f"Saved {out.name} in {self.folder}: {len(self.rows)} readings.")


class ReviewTab(QWidget):
    def __init__(self, recordings_dir):
        super().__init__()
        self.dir = Path(recordings_dir)
        self.trial = None

        btn_open = QPushButton("Open trial...")
        btn_open.clicked.connect(self.open_dialog)
        btn_session = QPushButton("Export all trials of this participant")
        btn_session.setToolTip("One row per trial (force peak, plateau, the angle at the peak, torque if possible, checks) and whether two trials of the "
                               "same position differ by more than 10 %. Saved as <participant>_session_summary.csv in the participant's folder.")
        btn_session.clicked.connect(self.export_session)
        btn_refval = QPushButton("Reference validation...")
        btn_refval.setToolTip("How far the app's angles are from the goniometer or inclinometer readings, over all trials in the recordings folder "
                              "(average difference, spread, an agreement plot, and a table to save).")
        btn_refval.clicked.connect(self.show_reference_validation)
        self.title = QLabel("Open a trial (choose a *_meta.json file in the recordings folder).")
        self.title.setWordWrap(True)
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        btn_export = QPushButton("Export summary CSV")
        btn_export.clicked.connect(self.export)
        btn_video = QPushButton("Export overlay video")
        btn_video.setToolTip("Draws the angles and trails on every frame and saves a video (for sharing or a presentation).")
        btn_video.clicked.connect(self.export_overlay_video)

        self.video = QLabel()
        self.video.setMinimumSize(480, 270)
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
        self._fa_window_items, self._fa_brushes = [], []
        self.plot_view = QComboBox()
        self.plot_view.addItem("Graphs against time", "time")
        self.plot_view.addItem("Force against angle", "fa")
        self.plot_view.setToolTip("Force against angle draws the joint's angle on one axis and the force on the other, picture by picture, "
                                  "from purple (the start of the trial) to yellow (the end). It shows how far the joint moved while the force "
                                  "rose. Needs a trial with force.")
        self.plot_view.currentIndexChanged.connect(lambda _: self._apply_plot_view())
        self.plot_view.setEnabled(False)
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(CURSOR_COLOUR, width=2))
        self.force_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(CURSOR_COLOUR, width=2))
        self.window = pg.LinearRegionItem(brush=(30, 120, 220, 40))
        self.force_window_mirror = pg.LinearRegionItem(brush=(30, 120, 220, 40), movable=False)       # the same window, shown on the force plot
        self.window.sigRegionChanged.connect(self._mirror_window)
        self.window.sigRegionChanged.connect(self.update_summary)
        self.chk_trails = QCheckBox("Show trails")
        self.chk_trails.setChecked(True)
        self.chk_trails.toggled.connect(lambda _: self.show_frame(self.slider.value()))
        self.event_combo = QComboBox()
        self.event_combo.activated.connect(self.jump_to_event)
        self.btn_hold = QPushButton("Window = MVIC start to end")
        self.btn_hold.clicked.connect(self.snap_window)

        # the numbers, one tab each
        self.check_text, self.stats, self.force_text, self.disp = _TextPane(wrap=True), _TextPane(), _TextPane(wrap=True), _TextPane()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.check_text, "Check")
        self.tabs.addTab(self.stats, "Angles")
        self.tabs.addTab(self.force_text, "Force")
        self.ref_text = _TextPane(wrap=False)                  # what a person read off a goniometer or inclinometer, beside the app's angle
        self.btn_ref_edit = QPushButton("Add or change the reference readings...")
        self.btn_ref_edit.setToolTip("Type what a goniometer or an inclinometer said for this trial. Saved in a small file next to the trial; "
                                     "the trial's own files are never rewritten.")
        self.btn_ref_edit.clicked.connect(self.edit_reference)
        ref_pane = QWidget()
        rpl = QVBoxLayout(ref_pane)
        rpl.setContentsMargins(0, 0, 0, 0)
        rpl.addWidget(self.btn_ref_edit, 0, Qt.AlignLeft)
        rpl.addWidget(self.ref_text, 1)
        self.tabs.addTab(ref_pane, "Reference")
        self.calib = _CalibrationPane()
        self.tabs.addTab(self.calib, "Calibration")
        self.tabs.addTab(self.disp, "Dots and pull")
        self.tabs.setMinimumHeight(170)       # the tabs sit in a wide strip under the graphs (0.14.0)

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
        self.zero_spin.setToolTip("The voltage that counts as 0 N for this trial. It starts as the zero taken when the trial was recorded (or the "
                                  "resting level of the recording if none was taken). Change it if the zero was wrong.")
        self.btn_apply_cal = QPushButton("Recalculate newtons")
        self.btn_apply_cal.setToolTip("Computes newtons from the recorded volts with this zero and the ACTIVE load cell regression (also for a "
                                      "regression made after the recording). The volts in the files are never changed.")
        self.btn_apply_cal.clicked.connect(self.apply_force_calibration)
        self.force_smooth = _NoWheelSpin()
        self.force_smooth.setRange(0, 1000)
        self.force_smooth.setDecimals(0)
        self.force_smooth.setSingleStep(20)
        self.force_smooth.setSuffix(" ms")
        self.force_smooth.setValue(SMOOTH_DEFAULT_S * 1000)
        self.force_smooth.setToolTip("Moving average for the force curve and for peak and plateau numbers. Use a multiple of 20 ms: that removes 50 Hz "
                                     "mains hum exactly. 0 = raw samples. The saved files always hold the raw samples.")
        self.force_smooth.valueChanged.connect(self.on_force_smoothing)
        self.btn_force_win = QPushButton("Window = force above half of peak")
        self.btn_force_win.setToolTip("Set the analysis window from the force itself: the stretch around the peak where the smoothed force is above half "
                                      "of the way from rest to peak (the hold). Use it when the MVIC markers were pressed early or late.")
        self.btn_force_win.clicked.connect(self.snap_to_force)
        self.force_box = QGroupBox("Force")
        fb = QHBoxLayout(self.force_box)
        fb.addWidget(QLabel("Smoothing"))
        fb.addWidget(self.force_smooth)
        fb.addWidget(QLabel("Picture delay"))
        fb.addWidget(self.picture_delay)
        fb.addWidget(QLabel("Zero"))
        fb.addWidget(self.zero_spin)
        fb.addWidget(self.btn_apply_cal)
        fb.addStretch(1)
        self.force_box.setVisible(False)

        # Graph-first layout (0.14.0; until then the graphs were a strip along the bottom and the number tabs sat top right):
        # video and its play controls top left, the graphs top right (the larger part), the number tabs in a wide strip below.
        # What acts on the analysis window or the force curve sits under the graphs it belongs to; the file buttons are in the top bar.
        # Both dividers can be dragged; where they are is remembered (attach_memory).
        top = QHBoxLayout()
        top.addWidget(btn_open)
        top.addWidget(self.title, 1)
        top.addWidget(btn_export)
        top.addWidget(btn_video)
        top.addWidget(btn_refval)
        top.addWidget(btn_session)
        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.chk_trails)
        erow = QHBoxLayout()
        erow.addWidget(QLabel("Markers:"))
        erow.addWidget(self.event_combo, 1)
        erow.addWidget(self.btn_hold)
        erow.addWidget(self.btn_force_win)
        hint = QLabel("Blue band on the plots = analysis window (drag its edges on the upper plot). The numbers use it.")
        hint.setWordWrap(True)
        left = QVBoxLayout()
        left.addWidget(self.video, 1)
        left.addLayout(ctrl)
        left_w = QWidget()
        left.setContentsMargins(0, 0, 6, 0)
        left_w.setLayout(left)
        plots = QWidget()
        pl = QVBoxLayout(plots)
        pl.setContentsMargins(6, 0, 0, 0)
        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("Graphs:"))
        vrow.addWidget(self.plot_view)
        vrow.addStretch(1)
        pl.addLayout(vrow)
        pl.addWidget(self.plot, 3)
        pl.addWidget(self.force_plot, 2)
        pl.addWidget(self.fa_plot, 5)
        pl.addLayout(erow)
        pl.addWidget(self.force_box)
        pl.addWidget(hint)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.addWidget(left_w)
        self.top_splitter.addWidget(plots)
        self.top_splitter.setStretchFactor(0, 2)
        self.top_splitter.setStretchFactor(1, 3)
        self.top_splitter.setChildrenCollapsible(False)
        self.top_splitter.setSizes(DEFAULT_TOP_SPLIT)
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.top_splitter)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes(DEFAULT_MAIN_SPLIT)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.splitter, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)
        self._memory_path = None
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(800)             # saved a moment after the last drag
        self._layout_timer.timeout.connect(self._save_layout)
        for sp in (self.top_splitter, self.splitter):
            sp.splitterMoved.connect(lambda *_: self._layout_timer.start())

    # ------------------------------------------------------------- remembered layout
    def layout_state(self):
        """The two divider positions: {'top': [video, graphs], 'main': [top row, number tabs]}."""
        return {"top": self.top_splitter.sizes(), "main": self.splitter.sizes()}

    def apply_layout_state(self, state):
        """Put remembered divider positions back. Anything that does not fit is ignored."""
        try:
            for sp, key in ((self.top_splitter, "top"), (self.splitter, "main")):
                sizes = [int(x) for x in state[key]]
                if len(sizes) == 2 and all(s > 0 for s in sizes):
                    sp.setSizes(sizes)
        except (KeyError, TypeError, ValueError):
            pass

    def attach_memory(self, path):
        """Remember the divider positions between launches: an own top-level part ('review_ui') of settings.json."""
        self._memory_path = Path(path)
        self.apply_layout_state(session_memory.read_all(path).get("review_ui") or {})

    def _save_layout(self):
        if self._memory_path is not None:
            session_memory.write_part(self._memory_path, review_ui=self.layout_state())

    # ------------------------------------------------------------- loading
    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open trial", str(self.dir), "Trial (*_meta.json)")
        if path:
            try:
                self.load(path)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Cannot open trial", str(e))

    def load(self, meta_path):
        self.btn_play.setChecked(False)
        self.timer.stop()
        tr = Trial(meta_path)
        self.trial = tr
        m = tr.meta
        self.title.setText(f"{m['participant']}  |  {m['test_label']}  |  target {m['target_deg']} deg  |  "
                           f"{m['frames']} frames, {m['duration_s']} s"
                           + (f"  |  ruler {tr.ruler_cm:g} cm = {tr.px_per_cm:.2f} px/cm" if tr.px_per_cm and tr.ruler_cm else "  |  no ruler scale")
                           + ("  |  UNFINISHED: the program stopped before saving; the video may not open" if tr.unfinished else "")
                           + (f"  |  STOPPED: {m['stopped_because']}" if m.get("stopped_because") else "")
                           + (f"  |  GAPS: {m['gaps']['count']} (longest {m['gaps']['max_s']:.2f} s, no frames)" if (m.get('gaps') or {}).get('count') else ""))
        self.plot.clear()
        self.plot.addItem(self.cursor)
        self.plot.addItem(self.window)
        tr.force_smooth_s = float(self.force_smooth.value()) / 1000.0     # before the force is drawn (it was set after, so the strip used the default)
        self._show_force_curve(tr)
        has_f = tr.has_force()
        self.force_box.setVisible(has_f)
        self._apply_plot_view()
        self.plot.setLabel("bottom", "" if has_f else "Time (s)")         # the time axis label is under the lower plot when there are two
        for k, a in enumerate(tr.angle_defs):
            self.plot.plot(tr.t, tr.deg[a.name], pen=pg.mkPen(PENS[k % len(PENS)], width=3 if a.primary else 1.5),
                           name=a.title)
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
        self.btn_hold.setEnabled(tr.hold_window() is not None)
        self.btn_force_win.setEnabled(tr.has_force())
        self.force_smooth.setEnabled(tr.has_force())
        if tr.has_force():
            zero, _ = tr.recorded_zero_v()
            self.zero_spin.setValue(zero)
        self.window.setRegion(tr.hold_window() or (tr.t[0], tr.t[-1]))        # the MVIC start-end window when markers exist
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(tr.rows) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.show_frame(0)
        self.update_summary()

    # ------------------------------------------------------------ playback
    def _apply_plot_view(self):
        """Time graphs (angles over the force) or force against angle. The second needs force; a trial without force always shows the first."""
        has_f = self.trial is not None and self.trial.has_force()
        self.plot_view.setEnabled(has_f)
        fa = has_f and self.plot_view.currentData() == "fa"
        self.plot.setVisible(not fa)
        self.force_plot.setVisible(has_f and not fa)
        self.fa_plot.setVisible(fa)
        if fa:
            self._update_force_angle()

    def _update_force_angle(self):
        """Draw the force-against-angle path: a thin line through every picture, a dot per picture coloured by time (purple to yellow), larger
        dots inside the analysis window, a star at the peak force (with the angle read at the peak, as in the Force tab), and a ring that
        follows the video. The force is paired with the picture that shows the same moment (the picture delay box)."""
        tr, pl = self.trial, self.fa_plot
        pl.clear()
        pl.addItem(self._fa_cursor)
        self._fa_cursor.setData([], [])
        self._fa_window_items, self._fa_brushes, self._fa_dirty = [], [], False
        self._fa_path = analysis.force_angle_path(tr, self.picture_delay.value()) if tr is not None else None
        path = self._fa_path
        if path is None or not path["ok"].any():
            pl.setTitle("No picture has both an angle and a force.", color="#bbbbbb", size="10pt")
            return
        ok = path["ok"]
        x, y = path["angle"][ok], path["force"][ok]
        self._fa_brushes = [pg.mkBrush(*(int(v) for v in c)) for c in analysis.time_colours(len(x))]
        pl.setLabel("bottom", f"{path['angle_name']} (deg)")
        pl.setLabel("left", f"Force ({path['unit']})")
        pl.setTitle("purple = start of the trial, yellow = its end;  large dots = the analysis window;  star = peak force",
                    color="#bbbbbb", size="9pt")
        pl.plot(x, y, pen=pg.mkPen((190, 190, 190, 110), width=1))
        pl.addItem(pg.ScatterPlotItem(x=x, y=y, size=6, pen=None, brush=self._fa_brushes))
        self._update_force_angle_window()
        self._update_fa_cursor(self.slider.value())

    def _update_force_angle_window(self):
        """What depends on the analysis window only, so dragging the window does not redraw the whole path: the larger dots of the
        pictures whose force lies in the window, and the star at the peak force. The window is in force time, like the Force tab:
        the picture stamped t shows the force of t - picture delay."""
        pl, path = self.fa_plot, self._fa_path
        for item in self._fa_window_items:
            pl.removeItem(item)
        self._fa_window_items = []
        if path is None or not path["ok"].any():
            return
        ok = path["ok"]
        x, y, ft = path["angle"][ok], path["force"][ok], path["force_t"][ok]
        t0, t1 = self.window.getRegion()
        inside = (ft >= t0) & (ft <= t1)
        if inside.any():
            self._fa_window_items.append(pg.ScatterPlotItem(x=x[inside], y=y[inside], size=10, pen=pg.mkPen("w", width=1.2),
                                                            brush=[b for b, keep in zip(self._fa_brushes, inside) if keep]))
        fs = self.trial.force_summary(t0, t1, self.picture_delay.value())
        if fs is not None and not math.isnan(fs["main_angle_at_peak_deg"]):
            self._fa_window_items.append(pg.ScatterPlotItem(x=[fs["main_angle_at_peak_deg"]], y=[fs["peak"]], size=20, symbol="star",
                                                            pen=pg.mkPen("k", width=1), brush=pg.mkBrush("#ffd43b")))
        for item in self._fa_window_items:
            pl.addItem(item)

    def _on_picture_delay(self, _=None):
        self._fa_dirty = True                             # the force paired with each picture changed: the path is drawn afresh
        self.update_summary()

    def _update_fa_cursor(self, i):
        """The ring on the force-against-angle path that marks the picture shown in the video."""
        path = self._fa_path
        if path is None or self.plot_view.currentData() != "fa" or not 0 <= i < len(path["t"]) or not path["ok"][i]:
            self._fa_cursor.setData([], [])
            return
        self._fa_cursor.setData([float(path["angle"][i])], [float(path["force"][i])])

    def _show_force_curve(self, tr):
        """The force in its own strip under the angle plot: every raw sample faint, the smoothed curve (the one the numbers come from) bold."""
        fp = self.force_plot
        fp.clear()
        self._fa_dirty = True                             # the force changed (new trial, smoothing, calibration): so does the force-angle path
        if not tr.has_force():
            return
        fp.addItem(self.force_cursor)
        fp.addItem(self.force_window_mirror)
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

    def apply_force_calibration(self):
        """Newtons from the saved volts with the zero in the box and the active load cell regression."""
        tr = self.trial
        if tr is None or not tr.has_force():
            return
        reg = active_regression()
        if reg is None:
            QMessageBox.information(self, "No calibration", "There is no load cell regression yet (limbtrack/load_cell_calibrations.json). "
                                    "Add one with LoadCellCalibration/add_calibration.py; the recordings can be converted afterwards.")
            return
        cal = ForceCalibration.load_cell(reg)
        cal.set_zero(self.zero_spin.value())
        _, where = tr.recorded_zero_v()
        tr.apply_calibration(cal, f"applied later: regression {reg.id}, zero {cal.offset_v:+.4f} V (the box; it started as {where})")
        self._show_force_curve(tr)
        self.update_summary()

    def show_frame(self, i):
        if self.trial is None:
            return
        img = self.trial.drawn(i, trails=self.chk_trails.isChecked())
        self.cursor.setValue(self.trial.t[i])
        self.force_cursor.setValue(self.trial.t[i])
        self._update_fa_cursor(i)
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        pm = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
        lines = [f"Window {t0:.2f} - {t1:.2f} s", f"{'':22s}{'mean':>7s}{'SD':>6s}{'min':>7s}{'max':>7s}{'start>end':>10s}{'maxdev':>8s}{'in tol':>8s}{'lost':>6s}"]
        for r in rows:
            lines.append(f"{r['angle'][:21]:22s}{r['mean_deg']:7.1f}{r['sd_deg']:6.2f}{r['min_deg']:7.1f}{r['max_deg']:7.1f}"
                         f"{r['change_start_to_end_deg']:+10.1f}{r['max_abs_dev_deg']:8.1f}{r['pct_in_tolerance']:7.0f}%{r['frames_lost']:6d}")
        lines += ["", f"Hold: the angle during the contraction, not the set angle  (plateau = middle {100 - 200 * PLATEAU_TRIM:.0f} % of the window; rest = the {REST_SECONDS:g} s before it)",
                  f"{'':22s}{'plateau':>9s}{'median':>8s}{'target':>8s}{'plat-tgt':>9s}{'rest':>8s}{'rest>hold':>10s}"]
        for r in rows:
            def f(v, w, signed=False):
                return f"{'-':>{w}s}" if math.isnan(v) else f"{v:>+{w}.1f}" if signed else f"{v:>{w}.1f}"
            pt = r["plateau_mean_deg"] - r["target_deg"] if not (math.isnan(r["plateau_mean_deg"]) or math.isnan(r["target_deg"])) else math.nan
            lines.append(f"{r['angle'][:21]:22s}{f(r['plateau_mean_deg'], 9)}{f(r['median_deg'], 8)}{f(r['target_deg'], 8)}{f(pt, 9, True)}"
                         f"{f(r['rest_mean_deg'], 8)}{f(r['rest_to_hold_shift_deg'], 10, True)}")
        if not self.trial.hold_window():
            lines.append("(no MVIC start/end markers in this trial: the numbers use the window as set)")
        self.stats.setText("\n".join(lines))
        if self.plot_view.currentData() == "fa" and self.trial.has_force():
            if self._fa_dirty:
                self._update_force_angle()                   # the force or the picture delay changed: the whole path
            else:
                self._update_force_angle_window()            # only the window moved: its dots and the star
        self.ref_text.setText(self._reference_text())
        self.force_text.setText(self._force_text(t0, t1))
        self._update_calibration(t0, t1)
        self.disp.setText(self._displacement_text(t0, t1))
        self.check_text.setHtml(self._check_html(t0, t1))

    def _reference_text(self):
        """The reference readings of the open trial beside the app's angle at the same moment."""
        tr = self.trial
        rows = tr.reference_rows()
        saved = tr.meta.get("reference_readings") or {}
        lines = ["Reference readings: numbers a person typed from a goniometer or an inclinometer. They are not measured by this program."]
        if saved.get("not_used"):
            was = saved["not_used"].get("entered_for") or {}
            who = f" (participant {was['participant']})" if was.get("participant") else ""
            lines.append(f"Readings typed for {was.get('test_key')} {was.get('target_deg')}{who} were NOT saved with this trial "
                         "(they belong to another position or participant).")
        if not rows:
            lines += ["", "No reading for this trial. Type it in the Live tab (section Reference reading) before recording, or add it here."]
            return "\n".join(lines)
        lines += ["", f"{'':26s}{'reference':>10s}{'n':>3s}{'app':>8s}{'app-ref':>9s}  instrument, from"]
        for r in rows:
            app = "-" if r["app_deg"] is None else f"{r['app_deg']:.1f}"
            diff = "-" if r["difference_deg"] is None else f"{round(r['difference_deg'], 1) + 0.0:+.1f}"          # (no "-0.0")
            lines.append(f"{r['title'][:25]:26s}{r['reference_deg']:10.1f}{r['n_readings']:3d}{app:>8s}{diff:>9s}  "
                         f"{reference.INSTRUMENTS.get(r['instrument'], r['instrument'])} (read to {r['resolution_deg']:g} deg), {r['source']}")
        bases = sorted({r["app_basis"] for r in rows if r["app_basis"]})
        lines.append("")
        if bases:
            lines += [f"The app's angle is {b}." for b in bases]
        if any(r["app_deg"] is None and r["key"] in {a.name for a in tr.angle_defs} for r in rows):
            lines.append("For an angle without an app value there was no moment kept and no usable first second.")
        if saved.get("stamped_at"):
            lines.append(f"Moment of the reading: {saved['stamped_at']}.")
        else:
            lines.append("No moment was kept ('Reading taken now' was not pressed): the first second of the recording stands in, which is only "
                         "like for like if the participant did not move between the reading and the start.")
        lines.append("A hand instrument is read to about a degree and has an error of its own of a few degrees; a difference of that size is "
                     "not an error of the app.")
        return "\n".join(lines)

    def edit_reference(self):
        if self.trial is None:
            return
        if ReferenceDialog(self.trial, self).exec():
            self.update_summary()

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

    def _force_text(self, t0, t1):
        tr = self.trial
        if not tr.has_force():
            return "No force was recorded in this trial."
        fs = tr.force_summary(t0, t1, self.picture_delay.value())
        if fs is None:
            return "Force was recorded but there are no samples in this window."
        u = fs["unit"]
        fmt = lambda v, d=1: "-" if v is None or math.isnan(v) else f"{v:.{d}f}"                    # noqa: E731
        d = 3 if u == "V" else 1
        sm = f"{fs['smoothing_s'] * 1000:.0f} ms moving average" if fs["smoothing_s"] > 0 else "raw samples"
        lines = [f"Force in the window ({u}{'' if u == 'N' else ', not calibrated'}; {sm})",
                 f"  peak                {fmt(fs['peak'], d)} at {fs['t_peak_s']:.2f} s",
                 f"  plateau mean        {fmt(fs['plateau_mean'], d)}   (middle {100 - 200 * PLATEAU_TRIM:.0f} % of the window)",
                 f"  rest before         {fmt(fs['rest_mean'], d)}      peak above rest {fmt(fs['peak_minus_rest'], d)}"]
        if fs["smoothing_s"] > 0:
            lines.append(f"  raw single-sample max {fmt(fs['raw_peak'], d)}   (a noisy single sample overstates the peak)")
        lines += ["", f"  {fs['main_angle']} at the peak force: {fmt(fs['main_angle_at_peak_deg'])} deg"
                  + (f"   (pictures read {fs['picture_delay_s']:+.3f} s after the peak)" if fs["picture_delay_s"] else "")
                  + "   (the plateau mean angle is on the Angles tab)",
                  f"  Calibration: {tr.force_cal_note}" + (f"  (regression {tr.regression.id}, see the Calibration tab)" if tr.regression else "")]
        ft = analysis.force_timing(tr, t0, t1)
        if ft is not None:
            lines += ["", "Timing and steadiness (smoothed force)",
                      f"  onset                 {ft['onset_s']:.2f} s   (force last below 10 % of its rise)",
                      f"  10 % to 90 % of rise  {ft['time_to_90pct_s']:.2f} s",
                      f"  steepest rise         {fmt(ft['rise_rate_per_s'], d)} {u}/s  (over 100 ms)",
                      f"  held above 90 % of peak for {ft['hold_above_90pct_s']:.2f} s, steadiness (SD / mean) {fmt(ft['hold_cv_pct'])} %"]
        tq = analysis.torque_summary(tr, t0, t1, self.picture_delay.value())
        if tq is not None:
            lines += ["", "Torque at the joint (force x moment arm)",
                      f"  peak {tq['peak_nm']:.1f} N.m at {tq['t_peak_s']:.2f} s   |   at the peak force {tq['at_peak_force_nm']:.1f} N.m   |   plateau {fmt(tq['plateau_nm'])} N.m",
                      f"  moment arm at the peak force {tq['arm_at_peak_force_cm']:.1f} cm (mean {tq['mean_arm_cm']:.1f} cm)"
                      + (f"; the strap dot was not seen in {tq['pictures_with_stand_in']} of {tq['pictures']} pictures: the ankle dot stood in for it"
                         if tq["pictures_with_stand_in"] else "")]
        else:
            why = ("the force is not calibrated" if u != "N" else "there is no ruler scale" if not tr.px_per_cm
                   else "the anchor dot was not used" if "anchor" not in tr.tracks else "no moment arm could be measured")
            lines += ["", f"Torque: not available ({why})."]
        if not math.isnan(fs["mains_hum_pct_of_peak"]) and fs["mains_hum_pct_of_peak"] > analysis.HUM_WARN_PCT:
            lines += ["", f"50 Hz mains interference: {fs['mains_hum_v'] * 1000:.0f} mV = {fs['mains_hum_pct_of_peak']:.0f} % of the rise above rest",
                      "  (check shielding and grounding; a smoothing window that is a multiple of 20 ms removes it)"]
        return "\n".join(lines)

    def _update_calibration(self, t0, t1):
        """The Calibration tab: the regression behind this trial's newtons (or the active one when the trial has none), and its precision."""
        tr = self.trial
        if tr is None or not tr.has_force():
            self.calib.show_regression(None, ["No force was recorded in this trial."])
            return
        fs = tr.force_summary(t0, t1, self.picture_delay.value())
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

    def _check_html(self, t0, t1):
        colours = {"ok": ("#16a34a", "OK    "), "check": ("#dc2626", "CHECK "), "info": ("#6b7280", "info  ")}
        rows = []
        for level, text in analysis.trial_checks(self.trial, t0, t1):
            c, tag = colours[level]
            esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            rows.append(f"<p style='margin:2px 0'><b style='color:{c}'>{tag.replace(' ', '&nbsp;')}</b> {esc}</p>")
        return "<div style='font-family: Segoe UI, sans-serif; font-size: 10pt'>" + "".join(rows) + "</div>"

    def export_overlay_video(self):
        """Render the overlay picture for every frame from the raw video and the CSV and save it as a video file."""
        if self.trial is None:
            return
        tr = self.trial
        out = tr.folder / f"{tr.base}_overlay_export.mp4"
        n = len(tr.rows)
        fps = tr.meta.get("video_fps_written") or tr.meta.get("measured_fps") or 30.0
        writer, done = None, 0
        progress = QProgressDialog("Drawing the overlay video...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        try:
            for i in range(n):
                img = tr.drawn(i, trails=self.chk_trails.isChecked())
                if img is None:
                    break
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
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
        QMessageBox.information(self, "Exported", f"Saved {out.name} ({done} of {n} frames).")

    def _displacement_text(self, t0, t1):
        tr = self.trial
        cm = tr.px_per_cm is not None
        unit = lambda px, c: f"{px:5.1f} px" + (f" ({c:4.2f} cm)" if cm else "")      # noqa: E731
        rows = tr.displacement(t0, t1)
        has_rel = any("rel_end_shift_px" in r for r in rows)
        lines = ["Dot movement in the window (from the starting position; px" + (", cm from the ruler scale" if cm else ", no ruler scale") + ")",
                 f"{'':16s}{'end shift':>20s}{'max shift':>20s}" + (f"{'end vs anchor':>20s}{'max vs anchor':>20s}" if has_rel else "")]
        for r in rows:
            def one(prefix):
                px, c = r.get(f"{prefix}_px", math.nan), r.get(f"{prefix}_cm", math.nan)
                return "-".rjust(20) if math.isnan(px) else unit(px, c).rjust(20)
            line = f"{r['dot'][:15]:16s}{one('end_shift')}{one('max_shift')}"
            if has_rel:
                line += (one("rel_end_shift") + one("rel_max_shift")) if "rel_end_shift_px" in r else "-".rjust(20) * 2
            lines.append(line)
        pull = tr.pull_summary(t0, t1)
        if pull:
            lines += ["", "Pull (strap and anchor dots)", f"{'':22s}{'mean':>8s}{'min':>8s}{'max':>8s}{'start>end':>11s}"]
            for q in pull:
                lines.append(f"{q['quantity']:22s}{q['mean']:8.1f}{q['min']:8.1f}{q['max']:8.1f}{q['change_start_to_end']:+11.1f}")
        return "\n".join(lines)

    def export_session(self):
        """One row per trial of the open trial's participant, with the 10 % repeatability check between trials of the same position."""
        if self.trial is None:
            return
        folder = self.trial.folder
        rows, failed = [], []
        for mp in sorted(folder.glob("*_meta.json")):
            try:
                tr = Trial(mp)
                try:
                    tr.force_smooth_s = self.trial.force_smooth_s
                    rows.append(analysis.session_row(tr))
                finally:
                    tr.cap.release()
            except Exception as e:  # noqa: BLE001 - one unreadable trial must not stop the others
                failed.append(f"{mp.name}: {e}")
        if not rows:
            QMessageBox.warning(self, "Nothing to export", "No readable trials in " + str(folder))
            return
        analysis.repeatability(rows)
        pid = self.trial.meta.get("participant") or "participant"
        out = folder / f"{pid}_session_summary.csv"
        analysis.write_session_csv(rows, out)
        QMessageBox.information(self, "Exported", f"Saved {out.name}: {len(rows)} trials." + (("\nNot readable: " + "; ".join(failed)) if failed else ""))

    def export(self):
        if self.trial is None:
            return
        rows, (t0, t1) = self._rows()
        out = self.trial.folder / f"{self.trial.base}_summary.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            ident = {"participant": self.trial.meta.get("participant"), "test_key": self.trial.meta.get("test_key"),
                     "trial": self.trial.base, "tolerance_deg": self.trial.meta.get("tolerance_deg"),
                     "software_version": self.trial.meta.get("software_version"),
                     "window_source": "MVIC markers" if self.trial.hold_window() and tuple(map(float, (t0, t1))) == tuple(self.trial.hold_window())
                     else "window as set (not the MVIC markers)"}
            fs = self.trial.force_summary(t0, t1, self.picture_delay.value())
            fnames = ["force_unit", "force_peak", "force_t_peak_s", "force_plateau_mean", "force_rest_mean", "main_angle_at_peak_force_deg",
                      "picture_delay_s", "force_n_per_v", "force_zero_v", "force_calibration_note", "force_smoothing_ms", "force_mains_hum_pct_of_peak",
                      "force_raw_peak", "force_regression_id", "force_peak_95pct_n"]
            cal = self.trial.force_cal or {}
            reg = self.trial.regression
            peak_ci = (COVERAGE * float(reg.sd_newton(fs["peak"])) if reg is not None and fs is not None and fs["unit"] == "N"
                       and not math.isnan(fs["peak"]) else "")
            fvals = ([fs["unit"], fs["peak"], fs["t_peak_s"], fs["plateau_mean"], fs["rest_mean"], fs["main_angle_at_peak_deg"],
                      fs["picture_delay_s"], cal.get("n_per_v"), cal.get("offset_v"), self.trial.force_cal_note,
                      fs["smoothing_s"] * 1000.0, fs["mains_hum_pct_of_peak"], fs["raw_peak"],
                      reg.id if reg is not None else "", peak_ci]
                     if fs is not None else [""] * len(fnames))
            fvals = ["" if isinstance(v, float) and math.isnan(v) else round(v, 3) if isinstance(v, float) else v for v in fvals]
            w.writerow(list(ident) + ["window_start_s", "window_end_s"] + fnames + list(rows[0]))
            for r in rows:
                w.writerow(list(ident.values()) + [round(t0, 3), round(t1, 3)] + fvals
                           + [round(v, 3) if isinstance(v, float) else v for v in r.values()])
        extra = self.trial.folder / f"{self.trial.base}_displacement.csv"
        drows = self.trial.displacement(t0, t1)
        keys = list(dict.fromkeys(k for r in drows for k in r))
        with open(extra, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["window_start_s", "window_end_s"] + keys)
            for r in drows:
                w.writerow([round(t0, 3), round(t1, 3)] + [("" if isinstance(r.get(k), float) and math.isnan(r[k]) else
                                                              round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys])
        QMessageBox.information(self, "Exported", f"Saved {out.name} and {extra.name}")
