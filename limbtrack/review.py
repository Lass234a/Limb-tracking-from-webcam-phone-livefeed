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
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from . import overlay, privacy
from .displacement import window_displacements
from .force import SMOOTH_DEFAULT_S, ForceCalibration, mains_amplitude, smooth_signal
from .pull import COLUMNS as PULL_COLUMNS
from .angles import measure, segment_pairs
# PARKED-GHOST: from .geometry import anchored_ghost
from .protocol import AngleDef
from .results import INFO, LOST, OK, OUT
from .engine import TRAIL_MIN_STEP_PX
from .tracker import WEAK_BELOW

PENS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]


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
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
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
        angles = []
        for a in self.angle_defs:
            n = a.name
            xy = {k: (d.x, d.y) for k, d in dots.items() if not d.lost}
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
        return overlay.draw(img, dots, angles, [head], trails=self.trails_upto(i) if trails else None, scale=self.scale_overlay())
        # PARKED-GHOST: # with the ghost: overlay.draw(img, dots, angles, [head], ghost=ghost, ghost_segments=segment_pairs(self.angle_defs), trails=...)

    def apply_calibration(self, cal, note):
        """Newtons from the saved volts with a calibration made AFTER the recording (the volts in the files are never changed)."""
        self.force_n = cal.to_newton(self.force_v)
        self.force_cal, self.force_cal_note = cal.to_dict(), note

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


class _NoWheelSpin(QDoubleSpinBox):
    """A spin box that ignores the mouse wheel (scrolling past it must not change the number)."""

    def wheelEvent(self, e):
        e.ignore()


class ReviewTab(QWidget):
    def __init__(self, recordings_dir):
        super().__init__()
        self.dir = Path(recordings_dir)
        self.trial = None

        btn_open = QPushButton("Open trial...")
        btn_open.clicked.connect(self.open_dialog)
        self.title = QLabel("Open a trial (choose a *_meta.json file in the recordings folder).")
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
        self.video.setMinimumSize(560, 315)
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background: black;")

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setLabel("left", "Angle (deg)")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", width=2))
        self.window = pg.LinearRegionItem(brush=(30, 120, 220, 40))
        self.window.sigRegionChanged.connect(self.update_summary)
        self.chk_trails = QCheckBox("Show trails")
        self.chk_trails.setChecked(True)
        self.chk_trails.toggled.connect(lambda _: self.show_frame(self.slider.value()))
        self.event_combo = QComboBox()
        self.event_combo.activated.connect(self.jump_to_event)
        self.btn_hold = QPushButton("Window = MVIC start to end")
        self.btn_hold.clicked.connect(self.snap_window)
        self.stats = QLabel()
        self.stats.setStyleSheet("font-family: Consolas, monospace;")
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.disp = QLabel()
        self.disp.setStyleSheet("font-family: Consolas, monospace;")
        self.disp.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top = QHBoxLayout()
        top.addWidget(btn_open)
        top.addWidget(self.title, 1)
        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.chk_trails)
        ctrl.addWidget(btn_export)
        ctrl.addWidget(btn_video)
        left = QVBoxLayout()
        left.addWidget(self.video, 1)
        left.addLayout(ctrl)
        erow = QHBoxLayout()
        erow.addWidget(QLabel("Markers:"))
        erow.addWidget(self.event_combo, 1)
        erow.addWidget(self.btn_hold)
        left.addLayout(erow)
        left.addWidget(QLabel("Blue band on the plot = analysis window (drag its edges). Statistics below use it."))
        left.addWidget(self.stats)
        left.addWidget(self.disp)
        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addWidget(self.plot, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(body, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)
        # force: calibration made later, and the delay between a picture's time stamp and the moment it shows
        self.force_cal_path = None
        self.picture_delay = _NoWheelSpin()
        self.picture_delay.setRange(-1.0, 2.0)
        self.picture_delay.setSingleStep(0.01)
        self.picture_delay.setDecimals(3)
        self.picture_delay.setSuffix(" s")
        self.picture_delay.setToolTip("How long after the movement a picture is time-stamped (camera, phone and USB delay). The joint angle at the "
                                      "peak force is read from the pictures stamped this long after the force peak. Measure it with a tap on the "
                                      "sensor that shows in the video; 0 = the time stamps are taken as they are.")
        self.picture_delay.valueChanged.connect(lambda _: self.update_summary())
        self.btn_apply_cal = QPushButton("Apply the saved force calibration")
        self.btn_apply_cal.setToolTip("Computes newtons from the recorded volts with the calibration in force_calibration.json (made after the "
                                      "recording). The volts in the files are never changed.")
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
        self.btn_force_win.setToolTip("Set the analysis window from the force itself: the stretch around the peak where the smoothed force is above half of the way from rest to peak (the hold). Use it when the MVIC markers were pressed early or late.")
        self.btn_force_win.clicked.connect(self.snap_to_force)
        erow.addWidget(QLabel("Force smoothing:"))
        erow.addWidget(self.force_smooth)
        erow.addWidget(self.btn_force_win)
        erow.addWidget(QLabel("Picture delay:"))
        erow.addWidget(self.picture_delay)
        erow.addWidget(self.btn_apply_cal)

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
        self._show_force_curve(tr)
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
            line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen((90, 90, 90), style=Qt.DashLine),
                                   label=lab, labelOpts={"position": 0.95, "color": (90, 90, 90), "rotateAxis": (1, 0)})
            self.plot.addItem(line)
        self.btn_hold.setEnabled(tr.hold_window() is not None)
        self.btn_force_win.setEnabled(tr.has_force())
        self.force_smooth.setEnabled(tr.has_force())
        tr.force_smooth_s = float(self.force_smooth.value()) / 1000.0
        self.window.setRegion(tr.hold_window() or (tr.t[0], tr.t[-1]))        # the MVIC start-end window when markers exist
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(tr.rows) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.show_frame(0)
        self.update_summary()

    # ------------------------------------------------------------ playback
    def _show_force_curve(self, tr):
        """The force on its own (right-hand) axis, drawn with every sample (thinned only for display, keeping the peaks)."""
        if not hasattr(self, "force_view"):
            self.force_view = pg.ViewBox()
            self.plot.showAxis("right")
            self.plot.scene().addItem(self.force_view)
            self.plot.getAxis("right").linkToView(self.force_view)
            self.force_view.setXLink(self.plot)
            self.plot.getViewBox().sigResized.connect(self._sync_force_view)
        self.force_view.clear()
        if not tr.has_force():
            self.plot.hideAxis("right")
            return
        unit, ys, _ = tr.force_signal()
        raw = tr.force_n if unit == "N" else tr.force_v
        faint = pg.PlotDataItem(tr.force_t, raw, pen=pg.mkPen((220, 60, 60, 60), width=1))          # every raw sample, faint
        faint.setDownsampling(auto=True, method="peak")
        self.force_view.addItem(faint)
        if tr.force_smooth_s > 0:
            curve = pg.PlotDataItem(tr.force_t, ys, pen=pg.mkPen((220, 60, 60), width=2))             # the smoothed curve the numbers come from
            curve.setDownsampling(auto=True, method="peak")
            self.force_view.addItem(curve)
        y = raw
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 0.05 * max(hi - lo, 1e-9)
        self.force_view.setYRange(lo - pad, hi + pad, padding=0)
        self.plot.getAxis("right").setLabel(f"Force ({unit})", color=(220, 60, 60))
        self.plot.showAxis("right")
        self._sync_force_view()

    def apply_force_calibration(self):
        tr = self.trial
        if tr is None or not tr.has_force():
            return
        cal = ForceCalibration.load(self.force_cal_path) if self.force_cal_path is not None else ForceCalibration()
        if not cal.calibrated:
            QMessageBox.information(self, "No calibration", "There is no calibration yet (force_calibration.json). In the Live tab press Zero "
                                    "and Calibrate with a known weight first; the recordings can be converted afterwards.")
            return
        tr.apply_calibration(cal, f"applied later from force_calibration.json ({cal.when or 'undated'}; {cal.method})")
        self._show_force_curve(tr)
        self.update_summary()

    def _sync_force_view(self):
        if hasattr(self, "force_view"):
            self.force_view.setGeometry(self.plot.getViewBox().sceneBoundingRect())
            self.force_view.linkedViewChanged(self.plot.getViewBox(), self.force_view.XAxis)

    def show_frame(self, i):
        if self.trial is None:
            return
        img = self.trial.drawn(i, trails=self.chk_trails.isChecked())
        self.cursor.setValue(self.trial.t[i])
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
        fs = self.trial.force_summary(t0, t1, self.picture_delay.value())
        if fs is not None:
            u = fs["unit"]
            fmt = lambda v: "-" if math.isnan(v) else f"{v:.1f}"                                  # noqa: E731
            sm = f"{fs['smoothing_s'] * 1000:.0f} ms moving average" if fs["smoothing_s"] > 0 else "raw samples"
            lines += ["", f"Force in the window ({u}{'' if u == 'N' else ', not calibrated'}; {sm}):",
                      f"  peak {fmt(fs['peak'])} at {fs['t_peak_s']:.2f} s   |   plateau mean {fmt(fs['plateau_mean'])}   |   "
                      f"rest before {fmt(fs['rest_mean'])}   |   peak above rest {fmt(fs['peak_minus_rest'])}"
                      + (f"   |   raw single-sample max {fmt(fs['raw_peak'])}" if fs["smoothing_s"] > 0 else ""),
                      f"  {fs['main_angle']} at the peak force: {fmt(fs['main_angle_at_peak_deg'])} deg   (plateau mean angle is in the table above"
                      + (f"; pictures read {fs['picture_delay_s']:+.3f} s after the peak" if fs["picture_delay_s"] else "") + ")",
                      f"  Force calibration: {self.trial.force_cal_note}"]
            if not math.isnan(fs["mains_hum_pct_of_peak"]) and fs["mains_hum_pct_of_peak"] > 3.0:
                lines.append(f"  50 Hz mains interference: {fs['mains_hum_v'] * 1000:.0f} mV = {fs['mains_hum_pct_of_peak']:.0f} % of the peak above rest "
                             "(check shielding and grounding; a smoothing window that is a multiple of 20 ms removes it)")
        elif self.trial.has_force():
            lines.append("(force was recorded but there are no samples in this window)")
        if not self.trial.hold_window():
            lines.append("(no MVIC start/end markers in this trial: the numbers use the window as set)")
        self.stats.setText("\n".join(lines))
        self.disp.setText(self._displacement_text(t0, t1))

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
                      "force_raw_peak"]
            cal = self.trial.force_cal or {}
            fvals = ([fs["unit"], fs["peak"], fs["t_peak_s"], fs["plateau_mean"], fs["rest_mean"], fs["main_angle_at_peak_deg"],
                      fs["picture_delay_s"], cal.get("n_per_v"), cal.get("offset_v"), self.trial.force_cal_note,
                      fs["smoothing_s"] * 1000.0, fs["mains_hum_pct_of_peak"], fs["raw_peak"]]
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
