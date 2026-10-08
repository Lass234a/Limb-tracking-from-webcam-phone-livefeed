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
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from . import overlay
from .displacement import window_displacements
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

    def frame(self, i):
        if i != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        self._pos = i if ok else -2
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
        return overlay.draw(img, dots, angles, [head], trails=self.trails_upto(i) if trails else None)
        # PARKED-GHOST: # with the ghost: overlay.draw(img, dots, angles, [head], ghost=ghost, ghost_segments=segment_pairs(self.angle_defs), trails=...)

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
                           + ("  |  UNFINISHED: the program stopped before saving; the video may not open" if tr.unfinished else "")
                           + (f"  |  STOPPED: {m['stopped_because']}" if m.get("stopped_because") else "")
                           + (f"  |  GAPS: {m['gaps']['count']} (longest {m['gaps']['max_s']:.2f} s, no frames)" if (m.get('gaps') or {}).get('count') else ""))
        self.plot.clear()
        self.plot.addItem(self.cursor)
        self.plot.addItem(self.window)
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
        self.window.setRegion(tr.hold_window() or (tr.t[0], tr.t[-1]))        # the MVIC start-end window when markers exist
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(tr.rows) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.show_frame(0)
        self.update_summary()

    # ------------------------------------------------------------ playback
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
            w.writerow(list(ident) + ["window_start_s", "window_end_s"] + list(rows[0]))
            for r in rows:
                w.writerow(list(ident.values()) + [round(t0, 3), round(t1, 3)]
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
