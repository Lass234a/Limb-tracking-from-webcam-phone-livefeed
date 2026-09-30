"""The window: a Live tab (camera, marking dots, lock, record) and a Review tab."""
import os
import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QTabWidget,
                               QVBoxLayout, QWidget)

from .camera import CameraSource, list_cameras
from .engine import Engine
from .review import ReviewTab

BIG = "font-size: 15pt; font-weight: bold;"


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
        tg = QGroupBox("2. Test and target")
        f = QFormLayout(tg)
        f.addRow("Test", self.test_combo)
        f.addRow("Target (main joint)", self.target_spin)
        f.addRow("Suggested targets", self.suggest)
        f.addRow("Tolerance main joint", self.tol_spin)
        f.addRow("Tolerance other joints", self.ntol_spin)
        f.addRow("Dot colour", self.kind_combo)
        f.addRow("Participant faces", self.facing_combo)

        # --- marking & lock
        self.instruction = QLabel("Start the camera first.")
        self.instruction.setStyleSheet(BIG)
        self.instruction.setWordWrap(True)
        btn_remark = QPushButton("Re-mark all dots")
        btn_remark.clicked.connect(self.remark)
        self.btn_lock = QPushButton("Lock position  (Space)")
        self.btn_lock.setCheckable(True)
        self.btn_lock.clicked.connect(self.toggle_lock)
        mk = QGroupBox("3. Mark dots, set position, lock")
        ml = QVBoxLayout(mk)
        ml.addWidget(self.instruction)
        ml.addWidget(btn_remark)
        ml.addWidget(self.btn_lock)

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
        rc = QGroupBox("4. Record")
        rf = QFormLayout(rc)
        rf.addRow("Participant ID", self.participant)
        rf.addRow("Notes", self.notes)
        rf.addRow(self.btn_rec)
        rf.addRow(self.rec_label)
        rf.addRow(btn_open)

        side = QVBoxLayout()
        for g in (src, tg, mk, rc):
            side.addWidget(g)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(340)
        lay = QHBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addWidget(side_w)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.btn_lock.click)
        QShortcut(QKeySequence(Qt.Key_R), self, activated=self.btn_rec.click)
        QShortcut(QKeySequence(Qt.Key_P), self, activated=self.btn_pause.click)
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
                QMessageBox.warning(self, "Cannot lock", "Every dot must be marked and visible to lock the position.")
        else:
            self.engine.unlock_position()
        self.refresh_view()

    # ---------------------------------------------------------- recording
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
