"""The window: a Live tab (camera, marking dots, lock, record) and a Review tab."""
import os
import threading
import time
from pathlib import Path

import math

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QTabWidget,
                               QVBoxLayout, QWidget)

from .camera import CameraSource, list_cameras
from .engine import Engine
from .review import ReviewTab

BIG = "font-size: 15pt; font-weight: bold;"
PENS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]


def trace_y_range(arrays, centre, half_span):
    """(low, high) for the live trace: at least centre +/- half_span, stretched to include all finite data."""
    lo, hi = centre - half_span, centre + half_span
    for a in arrays:
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size:
            lo, hi = min(lo, float(a.min()) - 1.0), max(hi, float(a.max()) + 1.0)
    return lo, hi


def bgr_to_qimage(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class VideoView(QWidget):
    """Shows an image scaled to fit; reports clicks in image pixel coordinates."""
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self._img = None

    def set_image(self, qimg):
        self._img = qimg
        self.update()

    def _target(self):
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), s

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self._img is not None:
            rect, _ = self._target()
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(rect, self._img)

    def mousePressEvent(self, e):
        if self._img is None or e.button() != Qt.LeftButton:
            return
        rect, s = self._target()
        x, y = (e.position().x() - rect.left()) / s, (e.position().y() - rect.top()) / s
        if 0 <= x < self._img.width() and 0 <= y < self._img.height():
            self.clicked.emit(x, y)


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    position = Signal(int, int)      # current frame, total frames (files only)
    ended = Signal(str)

    def __init__(self, engine, source):
        super().__init__()
        self.engine, self.source = engine, source
        self.paused = False
        self.is_file = isinstance(source, str) and not source.lower().startswith(("rtsp://", "http://", "https://"))
        self._stop = False
        self._cmds = []
        self._cmd_lock = threading.Lock()

    def stop(self):
        self._stop = True

    def command(self, name):
        """'step_fwd', 'step_back' or 'refresh' (redraw the current frame). Files only."""
        with self._cmd_lock:
            self._cmds.append(name)

    def _next_command(self):
        with self._cmd_lock:
            return self._cmds.pop(0) if self._cmds else None

    def _show(self, frame, t, cam):
        res = self.engine.process(frame, t)
        self.frame_ready.emit(bgr_to_qimage(res.overlay))
        if cam.is_file:
            self.position.emit(cam.position + 1, cam.frame_count)

    def run(self):
        try:
            cam = CameraSource(self.source)
        except Exception as e:  # noqa: BLE001 - shown to the operator
            self.ended.emit(str(e))
            return
        self.is_file = cam.is_file
        self.engine.camera_desc = cam.description
        t_start = None           # wall-clock origin for real-time playback of a file
        last = (None, None)      # last frame shown (for redraws while paused)
        msg = "Stopped."
        while not self._stop:
            cmd = self._next_command() if cam.is_file else None
            if cam.is_file and (self.paused or cmd):
                if cmd is None:
                    time.sleep(0.02)
                    continue
                t_start = None
                if cmd == "refresh":
                    if last[0] is not None:
                        self._show(last[0], last[1], cam)
                    continue
                if cmd == "step_back":
                    if cam.position < 1:
                        continue
                    cam.seek(cam.position - 1)
                frame, t = cam.read()          # step_fwd (or the frame we just seeked to)
                if frame is not None:
                    last = (frame, t)
                    self._show(frame, t, cam)
                continue                       # (at the last frame a forward step simply does nothing)
            frame, t = cam.read()
            if frame is None:
                if cam.is_file and self.engine.rec is None and last[0] is not None:
                    self.paused = True             # end of file: wait on the last frame so it can be stepped back
                    self.position.emit(cam.position + 1, cam.frame_count)
                    continue
                msg = "End of video file." if cam.is_file else "The camera stopped delivering frames."
                break
            last = (frame, t)
            self._show(frame, t, cam)
            if cam.is_file:  # play files at their own speed
                if t_start is None:
                    t_start = time.perf_counter() - t
                wait = t - (time.perf_counter() - t_start)
                if wait > 0:
                    time.sleep(wait)
        cam.close()
        self.ended.emit(msg)


class LiveTab(QWidget):
    def __init__(self, engine, recordings_dir):
        super().__init__()
        self.engine, self.recordings_dir = engine, Path(recordings_dir)
        self.worker = None
        self._rec_started = None

        self.view = VideoView()
        self.view.clicked.connect(self.on_click)

        # --- source
        self.cam_combo = QComboBox()
        self.cam_combo.addItem("Camera 0", 0)
        scan = QPushButton("Scan cameras")
        scan.clicked.connect(self.scan_cameras)
        self.btn_start = QPushButton("Start camera")
        self.btn_start.clicked.connect(self.toggle_camera)
        btn_file = QPushButton("Open video file (practice)...")
        btn_file.clicked.connect(self.open_file)
        src = QGroupBox("1. Video source")
        sl = QVBoxLayout(src)
        row = QHBoxLayout()
        row.addWidget(self.cam_combo, 1)
        row.addWidget(scan)
        sl.addLayout(row)
        sl.addWidget(self.btn_start)
        sl.addWidget(btn_file)
        self.btn_pause = QPushButton("Pause  (P)")
        self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_back = QPushButton("< Frame  (Left)")
        self.btn_back.clicked.connect(lambda: self.step("step_back"))
        self.btn_fwd = QPushButton("Frame >  (Right)")
        self.btn_fwd.clicked.connect(lambda: self.step("step_fwd"))
        self.frame_label = QLabel("")
        prow = QHBoxLayout()
        prow.addWidget(self.btn_back)
        prow.addWidget(self.btn_fwd)
        sl.addWidget(self.btn_pause)
        sl.addLayout(prow)
        sl.addWidget(self.frame_label)
        self._set_file_controls(False)

        # --- test
        self.test_combo = QComboBox()
        for k, t in engine.tests.items():
            self.test_combo.addItem(t.label, k)
        self.test_combo.currentIndexChanged.connect(self.on_test_changed)
        self.target_spin = self._spin(-90, 180, 1, " deg")
        self.target_spin.valueChanged.connect(lambda v: (setattr(engine, "target", v), self.refresh_view()))
        self.suggest = QComboBox()
        self.suggest.activated.connect(lambda i: self.target_spin.setValue(float(self.suggest.itemText(i).split()[0])))
        self.tol_spin = self._spin(0.5, 30, 0.5, " deg")
        self.tol_spin.valueChanged.connect(lambda v: (setattr(engine, "tolerance", v), self.refresh_view()))
        self.ntol_spin = self._spin(0.5, 45, 0.5, " deg")
        self.ntol_spin.valueChanged.connect(lambda v: (setattr(engine, "neighbour_tolerance", v), self.refresh_view()))
        self.kind_combo = QComboBox()
        for label, k in (("Auto (white or black)", "auto"), ("White dots", "white"), ("Black dots", "black")):
            self.kind_combo.addItem(label, k)
        self.kind_combo.currentIndexChanged.connect(self.on_kind_changed)
        self.facing_combo = QComboBox()
        self.facing_combo.addItem("Right side of the image", True)
        self.facing_combo.addItem("Left side of the image", False)
        self.facing_combo.currentIndexChanged.connect(self.on_facing_changed)
        self.chk_smooth = QCheckBox("Smooth the displayed angles (0.2 s average)")
        self.chk_smooth.setToolTip("Only changes what you see. The saved CSV always has the unsmoothed angles.")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.toggled.connect(lambda on: (engine.set_smoothing(on), self.refresh_view()))
        tg = QGroupBox("2. Test and target")
        f = QFormLayout(tg)
        f.addRow("Test", self.test_combo)
        f.addRow("Target (main joint)", self.target_spin)
        f.addRow("Suggested targets", self.suggest)
        f.addRow("Tolerance main joint", self.tol_spin)
        f.addRow("Tolerance other joints", self.ntol_spin)
        f.addRow("Dot colour", self.kind_combo)
        f.addRow("Participant faces", self.facing_combo)
        f.addRow(self.chk_smooth)

        # --- landmarks to use (pilot testing)
        self.lm_preset = QComboBox()
        self.lm_preset.activated.connect(self.apply_landmark_preset)
        self.lm_checks = {}
        self.lm_holder = QVBoxLayout()
        self.lm_info = QLabel()
        self.lm_info.setWordWrap(True)
        lm = QGroupBox("Landmarks to use (pilot)")
        lml = QVBoxLayout(lm)
        lml.addWidget(self.lm_preset)
        lml.addLayout(self.lm_holder)
        lml.addWidget(self.lm_info)

        # --- three-point angle
        self.chk_tp = QCheckBox("Use a three-point angle")
        self.tp_combos = [QComboBox() for _ in range(3)]
        self.chk_tp_main = QCheckBox("Use it as the main joint (the target above applies to it)")
        self.chk_tp_main.setChecked(True)
        self.tp_info = QLabel()
        self.tp_info.setWordWrap(True)
        tpbox = QGroupBox("Three-point angle")
        tpf = QFormLayout(tpbox)
        tpf.addRow(self.chk_tp)
        for label, combo in zip(("First dot", "Middle dot (angle is here)", "Last dot"), self.tp_combos):
            tpf.addRow(label, combo)
            combo.currentIndexChanged.connect(self.apply_three_point)
        tpf.addRow(self.chk_tp_main)
        tpf.addRow(self.tp_info)
        self.chk_tp.toggled.connect(self.apply_three_point)
        self.chk_tp_main.toggled.connect(self.apply_three_point)

        # --- marking & lock
        self.instruction = QLabel("Start the camera first.")
        self.instruction.setStyleSheet(BIG)
        self.instruction.setWordWrap(True)
        btn_remark = QPushButton("Re-mark all dots")
        btn_remark.clicked.connect(self.remark)
        self.btn_lock = QPushButton("Lock position  (Space)")
        self.btn_lock.setCheckable(True)
        self.btn_lock.clicked.connect(self.toggle_lock)
        # PARKED-GHOST: self.chk_ghost = QCheckBox("Show ghost of the locked pose")
        # PARKED-GHOST: self.chk_ghost.setChecked(True)
        # PARKED-GHOST: self.chk_ghost.toggled.connect(lambda v: (setattr(engine, "ghost_enabled", v), self.refresh_view()))
        # PARKED-GHOST: self.chk_anchor = QCheckBox("Ghost follows the main joint\n(shows angle drift only)")
        # PARKED-GHOST: self.chk_anchor.setChecked(True)
        # PARKED-GHOST: self.chk_anchor.toggled.connect(lambda v: (setattr(engine, "ghost_anchored", v), self.refresh_view()))
        self.chk_trails = QCheckBox("Show trails of the dots (from Start recording)")
        self.chk_trails.setChecked(True)
        self.chk_trails.toggled.connect(lambda v: (setattr(engine, "trails_enabled", v), self.refresh_view()))
        mk = QGroupBox("3. Mark dots, set position, lock")
        ml = QVBoxLayout(mk)
        ml.addWidget(self.instruction)
        ml.addWidget(btn_remark)
        ml.addWidget(self.btn_lock)
        # PARKED-GHOST: ml.addWidget(self.chk_ghost)
        # PARKED-GHOST: ml.addWidget(self.chk_anchor)
        ml.addWidget(self.chk_trails)

        # --- record
        self.participant = QLineEdit()
        self.participant.setPlaceholderText("e.g. P01")
        self.notes = QLineEdit()
        self.btn_rec = QPushButton("Start recording  (R)")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self.toggle_record)
        self.rec_label = QLabel("Not recording")
        btn_open = QPushButton("Open recordings folder")
        btn_open.clicked.connect(lambda: (self.recordings_dir.mkdir(exist_ok=True), os.startfile(self.recordings_dir)))
        self.btn_start_mark = QPushButton("MVIC start  (S)")
        self.btn_end_mark = QPushButton("MVIC end  (E)")
        self.btn_mark = QPushButton("Marker  (M)")
        self.btn_start_mark.clicked.connect(lambda: self.mark("MVIC start"))
        self.btn_end_mark.clicked.connect(lambda: self.mark("MVIC end"))
        self.btn_mark.clicked.connect(lambda: self.mark(None))
        mrow = QHBoxLayout()
        for b in (self.btn_start_mark, self.btn_end_mark, self.btn_mark):
            mrow.addWidget(b)
        rc = QGroupBox("4. Record")
        rf = QFormLayout(rc)
        rf.addRow("Participant ID", self.participant)
        rf.addRow("Notes", self.notes)
        rf.addRow(self.btn_rec)
        rf.addRow(mrow)
        rf.addRow(self.rec_label)
        rf.addRow(btn_open)

        # --- live trace (last 15 s)
        self.trace_mode = QComboBox()
        self.trace_mode.addItem("Main angle, last 15 s", "main")
        self.trace_mode.addItem("Drift from set position, all joints", "drift")
        self.trace = pg.PlotWidget()
        self.trace.setMinimumHeight(170)
        self.trace.setMaximumHeight(230)
        self.trace.setLabel("bottom", "Seconds ago")
        self.trace.setLabel("left", "deg")
        self.trace.showGrid(x=True, y=True, alpha=0.25)
        self.trace.setMouseEnabled(False, False)
        self.trace.addLegend(offset=(10, 5))
        self._trace_items = []
        self.trace_timer = QTimer(self)
        self.trace_timer.timeout.connect(self.update_trace)
        self.trace_timer.start(100)

        side = QVBoxLayout()
        for g in (src, tg, lm, tpbox, mk, rc):
            side.addWidget(g)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(340)
        side_scroll = QScrollArea()          # the panel is tall: scroll it on small screens
        side_scroll.setWidget(side_w)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setFixedWidth(370)
        left = QVBoxLayout()
        left.addWidget(self.view, 1)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Live trace:"))
        trow.addWidget(self.trace_mode)
        trow.addStretch(1)
        left.addLayout(trow)
        left.addWidget(self.trace)
        lay = QHBoxLayout(self)
        lay.addLayout(left, 1)
        lay.addWidget(side_scroll)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.btn_lock.click)
        QShortcut(QKeySequence(Qt.Key_R), self, activated=self.btn_rec.click)
        QShortcut(QKeySequence(Qt.Key_P), self, activated=self.btn_pause.click)
        QShortcut(QKeySequence(Qt.Key_S), self, activated=self.btn_start_mark.click)
        QShortcut(QKeySequence(Qt.Key_E), self, activated=self.btn_end_mark.click)
        QShortcut(QKeySequence(Qt.Key_M), self, activated=self.btn_mark.click)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.btn_back.click)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.btn_fwd.click)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(200)
        self.on_test_changed()

    @staticmethod
    def _spin(lo, hi, step, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setSuffix(suffix)
        s.setDecimals(1)
        return s

    # ---------------------------------------------------------- settings
    def on_test_changed(self):
        self.engine.set_test(self.test_combo.currentData())
        t = self.engine.test
        self.suggest.clear()
        for v in t.targets:
            self.suggest.addItem(f"{v:g} deg")
        self.target_spin.setValue(self.engine.target if self.engine.target is not None else 0.0)
        self.engine.target = self.target_spin.value()
        self.tol_spin.setValue(self.engine.tolerance)
        self.ntol_spin.setValue(self.engine.neighbour_tolerance)
        self.btn_lock.setChecked(False)
        self.rebuild_landmarks()

    # ------------------------------------------------- landmark selection
    def rebuild_landmarks(self):
        """One checkbox per landmark of the current test, plus presets like 'Only: Knee flexion'."""
        t = self.engine.test
        for cb in self.lm_checks.values():
            self.lm_holder.removeWidget(cb)
            cb.deleteLater()
        self.lm_checks = {}
        for n in t.dots:
            cb = QCheckBox(t.dot_title(n))
            cb.setChecked(n not in self.engine.disabled_dots)
            cb.toggled.connect(lambda on, name=n: self.on_landmark_toggled(name, on))
            self.lm_holder.addWidget(cb)
            self.lm_checks[n] = cb
        self.rebuild_three_point_choices()
        self.lm_preset.clear()
        self.lm_preset.addItem("Preset: all landmarks", list(t.dots))
        for a in t.angles:
            self.lm_preset.addItem(f"Preset: only {a.title}", list(dict.fromkeys(a.dots)))
        self.update_landmark_info()

    def rebuild_three_point_choices(self):
        t = self.engine.test
        keep = [c.currentData() for c in self.tp_combos]
        defaults = [d for d in ("trochanter", "epicondyle", "malleolus") if d in t.dots]
        if len(defaults) < 3:
            defaults = list(t.dots[:3])
        if self.engine.three_point:
            defaults = list(self.engine.three_point)
        for i, combo in enumerate(self.tp_combos):
            combo.blockSignals(True)
            combo.clear()
            for n in t.dots:
                combo.addItem(t.dot_title(n), n)
            want = keep[i] if keep[i] in t.dots and not self.engine.three_point else defaults[i]
            combo.setCurrentIndex(max(combo.findData(want), 0))
            combo.blockSignals(False)
        self.chk_tp.blockSignals(True)
        self.chk_tp.setChecked(bool(self.engine.three_point))
        self.chk_tp.blockSignals(False)
        self.update_three_point_info()

    def apply_three_point(self, *_):
        try:
            if self.chk_tp.isChecked():
                self.engine.set_three_point(*[c.currentData() for c in self.tp_combos], main=self.chk_tp_main.isChecked())
            else:
                self.engine.clear_three_point()
        except (ValueError, RuntimeError) as e:
            self.tp_info.setText(str(e))
            return
        self._sync_landmark_checks()
        self.btn_lock.setChecked(False)
        self.update_landmark_info()
        self.update_three_point_info()
        self.refresh_view()

    def update_three_point_info(self):
        e = self.engine
        main = bool(e.three_point) and e.three_point_main
        self.suggest.setEnabled(not main)         # the suggested targets belong to the test's own main joint
        if not e.three_point:
            self.tp_info.setText("")
        elif main:
            self.tp_info.setText("The target is the angle at the middle dot (a straight limb reads 180 deg). "
                                 "The test's own angles are checked against the locked position instead.")
        else:
            self.tp_info.setText("Shown with the other joints and checked against the locked position.")

    def _sync_landmark_checks(self):
        for n, cb in self.lm_checks.items():
            cb.blockSignals(True)
            cb.setChecked(n not in self.engine.disabled_dots)
            cb.blockSignals(False)

    def on_landmark_toggled(self, name, on):
        try:
            self.engine.set_dot_enabled(name, on)
        except RuntimeError as e:
            self._sync_landmark_checks()
            QMessageBox.warning(self, "Landmarks", str(e))
            return
        self.btn_lock.setChecked(False)
        self.update_landmark_info()
        self.refresh_view()

    def apply_landmark_preset(self, index):
        wanted = set(self.lm_preset.itemData(index))
        try:
            for n in self.engine.test.dots:
                self.engine.set_dot_enabled(n, n in wanted)
        except RuntimeError as e:
            QMessageBox.warning(self, "Landmarks", str(e))
        self._sync_landmark_checks()
        self.btn_lock.setChecked(False)
        self.update_landmark_info()
        self.refresh_view()

    def update_landmark_info(self):
        lines = []
        for title, ok, missing in self.engine.angle_availability():
            lines.append(f"\u2713 {title}" if ok else f"\u2717 {title}  (needs {', '.join(missing)})")
        if self.engine.primary_angle is None:
            lines.append("Main joint not available: no target is judged.")
        self.lm_info.setText("\n".join(lines))

    def on_facing_changed(self):
        self.engine.facing_right = bool(self.facing_combo.currentData())
        self.refresh_view()

    def on_kind_changed(self):
        self.engine.set_dot_kind(self.kind_combo.currentData())
        self.btn_lock.setChecked(False)

    # ------------------------------------------------------------ source
    def scan_cameras(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = list_cameras()
        finally:
            QApplication.restoreOverrideCursor()
        self.cam_combo.clear()
        for i in found:
            self.cam_combo.addItem(f"Camera {i}", i)
        if not found:
            QMessageBox.information(self, "Cameras", "No cameras found. Check the cable/app and that no other program uses it.")

    def toggle_camera(self):
        if self.worker is not None:
            self.stop_camera()
        elif self.cam_combo.currentData() is not None:
            self.start_source(self.cam_combo.currentData())

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.stop_camera()
            self.start_source(path)

    def start_source(self, source):
        self.engine.reset_dots()
        self.btn_lock.setChecked(False)
        self.worker = VideoWorker(self.engine, source)
        self.worker.frame_ready.connect(self.view.set_image)
        self.worker.position.connect(self.on_position)
        self._pos = (0, 0)
        self._set_file_controls(isinstance(source, str) and not source.lower().startswith(("rtsp://", "http")))
        self.worker.ended.connect(lambda msg, w=self.worker: self.on_ended(msg, w))
        self.worker.start()
        self.btn_start.setText("Stop camera")

    def stop_camera(self):
        if self.btn_rec.isChecked():
            self.toggle_record(force_stop=True)
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.btn_start.setText("Start camera")
        self._set_file_controls(False)

    def on_ended(self, msg, worker):
        if worker is not self.worker:      # late signal from a source that was already replaced
            return
        if self.btn_rec.isChecked():
            self.toggle_record(force_stop=True)
        if self.worker is not None:
            self.worker.wait(3000)      # 'ended' is emitted just before the thread finishes
            self.worker = None
            self.btn_start.setText("Start camera")
            self._set_file_controls(False)
            if msg != "Stopped.":
                self.instruction.setText(msg)

    # ------------------------------------------------------- live trace
    def update_trace(self):
        h = self.engine.history(15.0)
        for it in self._trace_items:
            self.trace.removeItem(it)
        self._trace_items = []
        if h is None or self.worker is None:
            return
        tol, ntol, target = h["tolerance"], h["neighbour_tolerance"], h["target"]
        drift = self.trace_mode.currentData() == "drift"
        pens = {n: PENS[k % len(PENS)] for k, n in enumerate(h["values"])}
        names = [h["primary"]] + ([n for n in h["values"] if n != h["primary"]] if drift else [])
        series = h["deviations"] if drift else h["values"]
        for n in names:
            item = self.trace.plot(h["t"], series[n], pen=pg.mkPen(pens[n], width=3 if n == h["primary"] else 1.5),
                                   name=h["titles"][n], connect="finite")
            self._trace_items.append(item)
        if drift:
            centre, span = 0.0, 2 * max(tol, ntol if len(names) > 1 else tol)
            for lo, hi, col in ((-tol, tol, (60, 180, 60, 45)),):
                band = pg.LinearRegionItem(values=(lo, hi), orientation="horizontal", brush=col, movable=False)
                self.trace.addItem(band)
                self._trace_items.append(band)
        else:
            centre, span = (target if target is not None else 0.0), 2 * tol
            if target is not None:
                band = pg.LinearRegionItem(values=(target - tol, target + tol), orientation="horizontal",
                                           brush=(60, 180, 60, 45), movable=False)
                self.trace.addItem(band)
                self._trace_items.append(band)
        if not drift and target is None:
            centre = float(np.nanmean(series[h["primary"]])) if np.isfinite(series[h["primary"]]).any() else 0.0
        lo, hi = trace_y_range([series[n] for n in names], centre, span)
        self.trace.setYRange(lo, hi, padding=0.02)
        self.trace.setXRange(-15, 0, padding=0)

    # ------------------------------------------------ pause / frame step
    def _set_file_controls(self, on):
        for b in (self.btn_pause, self.btn_back, self.btn_fwd):
            b.setEnabled(on)
        if not on:
            self.btn_pause.setChecked(False)
            self.btn_pause.setText("Pause  (P)")
            self.frame_label.setText("Pause and frame stepping work on video files only.")

    def toggle_pause(self):
        if self.worker is None or not self.worker.is_file:
            self.btn_pause.setChecked(False)
            return
        self.worker.paused = self.btn_pause.isChecked()
        self.btn_pause.setText("Play  (P)" if self.worker.paused else "Pause  (P)")
        self._update_frame_label()

    def step(self, name):
        if self.worker is None or not self.worker.is_file or self.btn_rec.isChecked():
            return
        self.btn_pause.setChecked(True)
        self.toggle_pause()
        self.worker.command(name)

    def _update_frame_label(self):
        cur, total = getattr(self, "_pos", (0, 0))
        if total:
            self.frame_label.setText(f"Frame {cur} / {total}" + ("  (paused)" if self.worker and self.worker.paused else ""))

    def on_position(self, cur, total):
        self._pos = (cur, total)
        self._update_frame_label()
        if self.worker is not None and self.worker.paused != self.btn_pause.isChecked():   # paused itself at end of file
            self.btn_pause.setChecked(self.worker.paused)
            self.btn_pause.setText("Play  (P)" if self.worker.paused else "Pause  (P)")

    def refresh_view(self):
        """Redraw the current frame after a change while paused (marking, lock, tolerances)."""
        if self.worker is not None and self.worker.is_file and self.worker.paused:
            self.worker.command("refresh")

    # ------------------------------------------------- marking & locking
    def on_click(self, x, y):
        if self.worker is None:
            return
        name = self.engine.next_dot_to_mark() or self.engine.nearest_dot(x, y)
        if name is None:
            return
        if not self.engine.mark_dot(name, x, y):
            self.instruction.setText(f"No dot found at that spot for '{self.engine.test.dot_title(name)}'. Click the centre of the dot "
                                     f"(check colour setting and lighting).")
        self.refresh_view()

    def remark(self):
        self.engine.reset_dots()
        self.btn_lock.setChecked(False)
        self.refresh_view()

    def toggle_lock(self):
        if self.btn_lock.isChecked():
            if not self.engine.lock_position():
                self.btn_lock.setChecked(False)
                QMessageBox.warning(self, "Cannot lock", self.engine.lock_problem())
        else:
            self.engine.unlock_position()
        self.refresh_view()

    # ---------------------------------------------------------- recording
    def mark(self, label):
        """Write a marker into the recording (MVIC start/end or a numbered marker)."""
        if not self.engine.add_event(label):
            self.rec_label.setText("Markers can only be added while recording.")

    def toggle_record(self, checked=None, force_stop=False):
        if self.btn_rec.isChecked() and not force_stop:
            if not self.participant.text().strip():
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Participant", "Enter a participant ID first so the files are labelled.")
                return
            if not self.engine.locked:
                r = QMessageBox.question(self, "Not locked", "The position is not locked, so the other joints are not "
                                         "checked for drift. Record anyway?")
                if r != QMessageBox.Yes:
                    self.btn_rec.setChecked(False)
                    return
            try:
                self.engine.start_trial(self.participant.text().strip(), self.notes.text())
            except Exception as e:  # noqa: BLE001
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Cannot record", str(e))
                return
            self._rec_started = time.perf_counter()
            self.btn_rec.setText("Stop recording  (R)")
            if self.worker is not None and self.worker.paused:      # recording needs frames flowing
                self.btn_pause.setChecked(False)
                self.toggle_pause()
        else:
            self.btn_rec.setChecked(False)
            paths = self.engine.stop_trial()
            self.btn_rec.setText("Start recording  (R)")
            self._rec_started = None
            self.rec_label.setText(f"Saved: {paths['data.csv'].parent.name}/{paths['raw.mp4'].name[:-8]}..." if paths
                                   else "Not recording")

    def refresh_status(self):
        recording = self._rec_started is not None
        for cb in self.lm_checks.values():
            cb.setEnabled(not recording)
        self.lm_preset.setEnabled(not recording)
        for w in (self.chk_tp, self.chk_tp_main, *self.tp_combos):
            w.setEnabled(not recording)
        if self._rec_started is not None:
            self.rec_label.setText(f"RECORDING  {time.perf_counter() - self._rec_started:5.1f} s")
        if self.worker is None:
            return
        nxt = self.engine.next_dot_to_mark()
        if nxt:
            self.instruction.setText(f"Click the {self.engine.test.dot_title(nxt)} dot on the video.")
        elif not self.engine.locked:
            self.instruction.setText("All dots marked. Set the participant's position, then lock it.")
        else:
            self.instruction.setText("Position locked.")

    def shutdown(self):
        if self.btn_rec.isChecked():
            self.toggle_record(force_stop=True)
        self.stop_camera()


class MainWindow(QWidget):
    def __init__(self, tests, root):
        super().__init__()
        self.setWindowTitle("Limb angle tracker")
        self.resize(1400, 820)
        self.engine = Engine(tests, recordings_dir=Path(root) / "recordings")
        tabs = QTabWidget()
        self.live = LiveTab(self.engine, Path(root) / "recordings")
        self.review = ReviewTab(Path(root) / "recordings")
        tabs.addTab(self.live, "Live")
        tabs.addTab(self.review, "Review")
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)

    def closeEvent(self, e):
        self.live.shutdown()
        super().closeEvent(e)
