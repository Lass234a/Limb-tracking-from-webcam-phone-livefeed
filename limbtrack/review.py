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
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from . import overlay
from .results import INFO, LOST, OK, OUT, AngleResult

PENS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


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
        self.t = np.array([_f(r["t_s"]) for r in self.rows])
        self.angle_defs = self.meta["angles"]
        self.dot_names = [k[:-5] for k in self.rows[0] if k.endswith("_lost")]
        self.deg = {a["name"]: np.array([_f(r[f"{a['name']}_deg"]) for r in self.rows]) for a in self.angle_defs}
        self.ref = {a["name"]: np.array([_f(r[f"{a['name']}_ref"]) for r in self.rows]) for a in self.angle_defs}
        self.dev = {a["name"]: np.array([_f(r[f"{a['name']}_dev"]) for r in self.rows]) for a in self.angle_defs}
        self.ok = {a["name"]: [r[f"{a['name']}_ok"] for r in self.rows] for a in self.angle_defs}
        self.cap = cv2.VideoCapture(str(folder / self.meta["files"]["raw.mp4"]))
        self._pos = -2

    def frame(self, i):
        if i != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        self._pos = i if ok else -2
        return img if ok else None

    def drawn(self, i):
        img = self.frame(i)
        if img is None:
            return None
        r = self.rows[i]
        dots = {n: SimpleNamespace(x=_f(r[f"{n}_x"]), y=_f(r[f"{n}_y"]), lost=r[f"{n}_lost"] == "1")
                for n in self.dot_names if r[f"{n}_x"] != ""}
        angles = []
        for a in self.angle_defs:
            n = a["name"]
            res = AngleResult(n, a["label"], a["kind"], a["points"], a["primary"])
            res.value, res.deviation = self.deg[n][i], self.dev[n][i]
            res.reference = None if math.isnan(self.ref[n][i]) else self.ref[n][i]
            flag = self.ok[n][i]
            res.status = LOST if math.isnan(res.value) else OK if flag == "1" else OUT if flag == "0" else INFO
            angles.append(res)
        head = f"{self.meta['test_label']} | {self.meta['participant']} | t = {self.t[i]:.2f} s"
        return overlay.draw(img, dots, angles, [head])

    def summary(self, t0, t1):
        """Rows of statistics per angle within [t0, t1] seconds."""
        m = (self.t >= t0) & (self.t <= t1)
        out = []
        for a in self.angle_defs:
            n = a["name"]
            v = self.deg[n][m]
            valid = ~np.isnan(v)
            flags = [f for f, keep in zip(self.ok[n], m) if keep and f != ""]
            dev = self.dev[n][m]
            out.append({
                "angle": a["label"],
                "role": "main" if a["primary"] else "neighbour",
                "frames": int(m.sum()),
                "frames_lost": int((~valid).sum()),
                "mean_deg": float(np.mean(v[valid])) if valid.any() else math.nan,
                "sd_deg": float(np.std(v[valid], ddof=1)) if valid.sum() > 1 else math.nan,
                "min_deg": float(np.min(v[valid])) if valid.any() else math.nan,
                "max_deg": float(np.max(v[valid])) if valid.any() else math.nan,
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
        self.stats = QLabel()
        self.stats.setStyleSheet("font-family: Consolas, monospace;")
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top = QHBoxLayout()
        top.addWidget(btn_open)
        top.addWidget(self.title, 1)
        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(btn_export)
        left = QVBoxLayout()
        left.addWidget(self.video, 1)
        left.addLayout(ctrl)
        left.addWidget(QLabel("Blue band on the plot = analysis window (drag its edges). Statistics below use it."))
        left.addWidget(self.stats)
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
                           f"{m['frames']} frames, {m['duration_s']} s")
        self.plot.clear()
        self.plot.addItem(self.cursor)
        self.plot.addItem(self.window)
        for k, a in enumerate(tr.angle_defs):
            self.plot.plot(tr.t, tr.deg[a["name"]], pen=pg.mkPen(PENS[k % len(PENS)], width=3 if a["primary"] else 1.5),
                           name=a["label"])
            if a["primary"] and m["target_deg"] is not None:
                tol = m["tolerance_deg"]
                band = pg.LinearRegionItem(values=(m["target_deg"] - tol, m["target_deg"] + tol), orientation="horizontal",
                                           brush=(60, 180, 60, 45), movable=False)
                for line in band.lines:
                    line.setPen(pg.mkPen((60, 150, 60), style=Qt.DashLine))
                self.plot.addItem(band)
        self.window.setRegion((tr.t[0], tr.t[-1]))
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
        img = self.trial.drawn(i)
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

    # ------------------------------------------------------------- summary
    def _rows(self):
        t0, t1 = self.window.getRegion()
        return self.trial.summary(t0, t1), (t0, t1)

    def update_summary(self):
        if self.trial is None:
            return
        rows, (t0, t1) = self._rows()
        lines = [f"Window {t0:.2f} - {t1:.2f} s", f"{'':22s}{'mean':>7s}{'SD':>6s}{'min':>7s}{'max':>7s}{'maxdev':>8s}{'in tol':>8s}{'lost':>6s}"]
        for r in rows:
            lines.append(f"{r['angle'][:21]:22s}{r['mean_deg']:7.1f}{r['sd_deg']:6.2f}{r['min_deg']:7.1f}{r['max_deg']:7.1f}"
                         f"{r['max_abs_dev_deg']:8.1f}{r['pct_in_tolerance']:7.0f}%{r['frames_lost']:6d}")
        self.stats.setText("\n".join(lines))

    def export(self):
        if self.trial is None:
            return
        rows, (t0, t1) = self._rows()
        out = self.trial.folder / f"{self.trial.base}_summary.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["window_start_s", "window_end_s"] + list(rows[0]))
            for r in rows:
                w.writerow([round(t0, 3), round(t1, 3)] + [round(v, 3) if isinstance(v, float) else v for v in r.values()])
        QMessageBox.information(self, "Exported", f"Saved {out.name}")
