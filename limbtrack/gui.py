"""The window: a Live tab (camera, marking dots, lock, record) and a Review tab."""
import json
import os
import threading
import time
import traceback
from pathlib import Path

import math

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QTabWidget,
                               QListWidget, QVBoxLayout, QWidget)

from . import privacy
from .camera import CameraSource, camera_names, classify_source, list_cameras, normalise_stream_address
from .engine import Engine
from .loadcell import GRAVITY, find_regression
from .force import ForceCalibration, ForceError, NiDaqForce, SimulatedForce
from .recorder import count_trials
from .scale import MIN_DOT_DIAMETER_CM
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


TRIALS_PER_POSITION = 2     # the session plan counts saved trials against this number


class NoWheelComboBox(QComboBox):
    """A drop-down that ignores the mouse wheel unless it has focus, so scrolling the side panel cannot change it."""

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class TextField(QLineEdit):
    """A text box that tells the window when it has focus (typed letters go into it, so letter shortcuts are off)
    and lets go of focus on Enter or Esc, so the shortcut keys work again."""

    focus_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.returnPressed.connect(self.clearFocus)

    def focusInEvent(self, e):
        super().focusInEvent(e)
        self.focus_changed.emit(True)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self.focus_changed.emit(False)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.clearFocus()
            return
        super().keyPressEvent(e)


class VideoView(QWidget):
    """Shows an image scaled to fit; reports clicks in image pixel coordinates."""
    clicked = Signal(float, float)
    rect_drawn = Signal(float, float, float, float)      # x0, y0, x1, y1 in image pixels (after set_draw_mode(True))

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self._img = None
        self._draw_mode = False
        self._drag = None                  # [x0, y0, x1, y1] while the mouse is down in draw mode
        self._crop, self._masks = None, []   # privacy shapes in image pixels (shown on the screen only)

    def set_draw_mode(self, on):
        self._draw_mode = bool(on)
        self._drag = None
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def set_privacy_shapes(self, crop, masks):
        self._crop, self._masks = crop, list(masks)
        self.update()

    def _image_xy(self, e, clamp=False):
        rect, s = self._target()
        x, y = (e.position().x() - rect.left()) / s, (e.position().y() - rect.top()) / s
        if clamp:
            x, y = min(max(x, 0.0), self._img.width()), min(max(y, 0.0), self._img.height())
        return x, y

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
            _, s = self._target()

            def box(r):
                return QRectF(rect.left() + r[0] * s, rect.top() + r[1] * s, r[2] * s, r[3] * s)
            if self._crop:                               # the part that is saved: everything outside it is dimmed
                c = box(self._crop)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(0, 0, 0, 140)))
                p.drawRect(QRectF(rect.left(), rect.top(), rect.width(), c.top() - rect.top()))
                p.drawRect(QRectF(rect.left(), c.bottom(), rect.width(), rect.bottom() - c.bottom()))
                p.drawRect(QRectF(rect.left(), c.top(), c.left() - rect.left(), c.height()))
                p.drawRect(QRectF(c.right(), c.top(), rect.right() - c.right(), c.height()))
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(0, 200, 255), 2, Qt.DashLine))
                p.drawRect(c)
            for m in self._masks:                        # black in the saved videos; shown here as a see-through red box
                p.setPen(QPen(QColor(220, 38, 38), 2))
                p.setBrush(QBrush(QColor(220, 38, 38, 90)))
                p.drawRect(box(m))
            if self._drag:
                x0, y0, x1, y1 = self._drag
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(250, 204, 21), 2, Qt.DashLine))
                p.drawRect(QRectF(rect.left() + min(x0, x1) * s, rect.top() + min(y0, y1) * s, abs(x1 - x0) * s, abs(y1 - y0) * s))

    def mousePressEvent(self, e):
        if self._img is None or e.button() != Qt.LeftButton:
            return
        if self._draw_mode:
            x, y = self._image_xy(e, clamp=True)
            self._drag = [x, y, x, y]
            return
        x, y = self._image_xy(e)
        if 0 <= x < self._img.width() and 0 <= y < self._img.height():
            self.clicked.emit(x, y)

    def mouseMoveEvent(self, e):
        if self._draw_mode and self._drag is not None and self._img is not None:
            self._drag[2:] = self._image_xy(e, clamp=True)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._draw_mode and self._drag is not None and e.button() == Qt.LeftButton:
            self._drag[2:] = self._image_xy(e, clamp=True)
            x0, y0, x1, y1 = self._drag
            self._drag = None
            self.update()
            self.rect_drawn.emit(x0, y0, x1, y1)


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    position = Signal(int, int)      # current frame, total frames (files only)
    ended = Signal(str)

    def __init__(self, engine, source):
        super().__init__()
        self.engine, self.source = engine, source
        self.paused = False
        self.camera = None
        self.lock_message = ""
        self.at_end = False      # a video file has played to its last frame (nothing more will arrive)
        self.status_note = ""     # shown to the operator while a live source is silent
        self._lock_request = False
        self.is_file = classify_source(source) == "file"
        self._stop = False
        self._cmds = []
        self._cmd_lock = threading.Lock()

    def stop(self):
        self._stop = True

    def request_lock(self):
        """Ask the video thread to lock exposure/white balance (the camera may only be touched from that thread)."""
        self._lock_request = True

    def command(self, name):
        """'step_fwd', 'step_back' or 'refresh' (redraw the current frame). Files only."""
        with self._cmd_lock:
            self._cmds.append(name)

    def _next_command(self):
        with self._cmd_lock:
            return self._cmds.pop(0) if self._cmds else None

    def _show(self, frame, t, cam):
        if self.engine.mirror:
            frame = cv2.flip(frame, 1)          # before tracking, recording and drawing: everything sees the flipped picture
        res = self.engine.process(frame, t, cam.position)
        self.at_end = False
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
        self.engine.set_source(cam.description, cam.info(),
                               (lambda: cam.file_fps) if cam.is_file else (lambda: cam.measured_fps() or (cam.requested or {}).get("fps")))
        self.camera = cam        # read-only use from the window: settings and measured frame rate
        t_start = None           # wall-clock origin for real-time playback of a file
        last = (None, None)      # last frame shown (for redraws while paused)
        msg = "Stopped."
        try:
            msg = self._loop(cam, t_start, last) or msg
        except Exception as e:  # noqa: BLE001 - the thread must never die silently
            msg = self._report_error(e)
        finally:
            cam.close()
        self.ended.emit(msg)

    def _report_error(self, e):
        """Write the full traceback to last_error.log and return a message for the operator."""
        where = ""
        try:
            folder = Path(self.engine.recordings_dir)
            folder.mkdir(parents=True, exist_ok=True)
            log = folder / "last_error.log"
            with open(log, "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc())
            where = f" Details were saved in {log}."
        except OSError:
            pass
        return f"The video stopped because of an error ({type(e).__name__}: {e}). Any recording was saved up to this point.{where}"

    REOPEN_AFTER_S = 2.0       # a live source silent for this long is closed and opened again
    GIVE_UP_AFTER_S = 6.0      # ...and the session ends if it is still silent after this long

    def _loop(self, cam, t_start, last):
        """Read, track and show frames until stopped. Returns the reason it ended."""
        msg = "Stopped."
        silent_since, reopened = None, False
        while not self._stop:
            if self._lock_request:
                self._lock_request = False
                cam.lock_exposure_wb()
                self.engine.camera_info = cam.info()
                self.lock_message = cam.lock_summary()
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
                    self.at_end = True
                    self.position.emit(cam.position + 1, cam.frame_count)
                    continue
                if cam.is_file:
                    msg = "End of video file."
                    break
                # A live source that delivers no frame is not the end: wait, try to open it again, give up only after a while.
                now = time.perf_counter()
                if silent_since is None:
                    silent_since, reopened = now, False
                cam.failed_reads += 1
                waited = now - silent_since
                self.status_note = f"No frames from the camera for {waited:.1f} s - waiting..."
                if waited > self.REOPEN_AFTER_S and not reopened:
                    reopened = True
                    self.status_note = "No frames from the camera - trying to reconnect..."
                    cam.reopen()
                if waited > self.GIVE_UP_AFTER_S:
                    msg = "The camera stopped delivering frames."
                    break
                time.sleep(0.01)
                continue
            if silent_since is not None:               # back after a silence: keep the counters with the recording
                silent_since = None
                self.status_note = ""
                self.engine.camera_info = cam.info()
            last = (frame, t)
            self._show(frame, t, cam)
            if cam.is_file:  # play files at their own speed
                if t_start is None:
                    t_start = time.perf_counter() - t
                wait = t - (time.perf_counter() - t_start)
                if wait > 0:
                    time.sleep(wait)
        return msg


class LiveTab(QWidget):
    def __init__(self, engine, recordings_dir):
        super().__init__()
        self.engine, self.recordings_dir = engine, Path(recordings_dir)
        self.worker = None
        self._rec_started = None
        self._say_until = 0.0
        self._combo_index = {"test": 0, "kind": 0, "facing": 0}     # last accepted choice, restored if the engine refuses a change
        self._marker_counts = {"start": 0, "end": 0, "other": 0}
        self._marker_msg = ("", 0.0)

        self.view = VideoView()
        self.view.clicked.connect(self.on_click)

        # --- source
        self.cam_combo = NoWheelComboBox()
        self.cam_combo.addItem("Camera 0", 0)
        self.cam_combo.setToolTip("A phone used as a webcam (Iriun, DroidCam, Camo...) appears here by name once its app is running on the phone and the PC. Press Scan cameras after starting it.")
        self.btn_scan = QPushButton("Scan cameras")
        self.btn_scan.setToolTip("Only available while no camera or video is running: scanning would switch off a running webcam.")
        self.btn_scan.clicked.connect(self.scan_cameras)
        self.btn_start = QPushButton("Start camera")
        self.btn_start.clicked.connect(self.toggle_camera)
        self.stream_box = TextField()
        self.stream_box.setPlaceholderText("stream address, e.g. http://192.168.1.20:8080/video or rtsp://...")
        self.stream_box.setToolTip("For a phone app that publishes a network stream (IP Webcam, DroidCam over Wi-Fi, an RTSP app). Wi-Fi adds delay and drops; USB is steadier.")
        self.btn_stream = QPushButton("Connect to stream")
        self.btn_stream.clicked.connect(self.connect_stream)
        btn_file = QPushButton("Open video file (practice)...")
        btn_file.clicked.connect(self.open_file)
        src = QGroupBox("Video source (set once per session)")
        sl = QVBoxLayout(src)
        row = QHBoxLayout()
        row.addWidget(self.cam_combo, 1)
        row.addWidget(self.btn_scan)
        sl.addLayout(row)
        sl.addWidget(self.btn_start)
        sl.addWidget(btn_file)
        sl.addWidget(self.stream_box)
        sl.addWidget(self.btn_stream)
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
        self.cam_info = QLabel("")
        self.cam_info.setWordWrap(True)
        sl.addWidget(self.cam_info)
        self.chk_mirror = QCheckBox("Mirror the image (flip left-right)")
        self.chk_mirror.setToolTip("Flips the picture before tracking and recording, so saved video and pixel positions are flipped too. "
                                   "Changing it clears the marked dots.")
        self.chk_mirror.toggled.connect(self.on_mirror_toggled)
        sl.addWidget(self.chk_mirror)
        self.btn_lock_cam = QPushButton("Lock exposure and white balance (switched off)")
        self.btn_lock_cam.setToolTip("Switched off: on some webcams this blacks out the picture and reports the opposite of what happened. "
                                     "Use steady lighting, or lock exposure and white balance in the phone/camera app instead.")
        self.btn_lock_cam.clicked.connect(lambda: self.worker and self.worker.request_lock())
        self.btn_lock_cam.setEnabled(False)
        sl.addWidget(self.btn_lock_cam)
        self._set_file_controls(False)

        # --- test
        self.test_combo = NoWheelComboBox()
        for k, t in engine.tests.items():
            self.test_combo.addItem(t.label, k)
        self.test_combo.currentIndexChanged.connect(self.on_test_changed)
        self.target_spin = self._spin(-91, 180, 1, " deg")
        self.target_spin.setSpecialValueText("no target")         # the lowest value means: the main joint is shown, not judged
        self.target_spin.valueChanged.connect(lambda v: self._set_setting("target", self.target_spin, v))
        self.suggest = NoWheelComboBox()
        self.suggest.activated.connect(lambda i: self.target_spin.setValue(float(self.suggest.itemText(i).split()[0])))
        self.tol_spin = self._spin(0.5, 30, 0.5, " deg")
        self.tol_spin.valueChanged.connect(lambda v: self._set_setting("tolerance", self.tol_spin, v))
        self.ntol_spin = self._spin(0.5, 45, 0.5, " deg")
        self.ntol_spin.valueChanged.connect(lambda v: self._set_setting("neighbour_tolerance", self.ntol_spin, v))
        self.kind_combo = NoWheelComboBox()
        for label, k in (("Auto (white or black)", "auto"), ("White dots", "white"), ("Black dots", "black")):
            self.kind_combo.addItem(label, k)
        self.kind_combo.currentIndexChanged.connect(self.on_kind_changed)
        self.facing_combo = NoWheelComboBox()
        self.facing_combo.addItem("Right side of the image", True)
        self.facing_combo.addItem("Left side of the image", False)
        self.facing_combo.currentIndexChanged.connect(self.on_facing_changed)
        self.chk_smooth = QCheckBox("Smooth the displayed angles (0.2 s average)")
        self.chk_smooth.setToolTip("Only changes what you see. The saved CSV always has the unsmoothed angles.")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.toggled.connect(lambda on: (engine.set_smoothing(on), self.refresh_view()))
        tg = QGroupBox("Test, target and tolerances")
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
        self.lm_preset = NoWheelComboBox()
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
        self.tp_combos = [NoWheelComboBox() for _ in range(3)]
        self.chk_tp_flex = QCheckBox("Show it as flexion (0 = straight, like the protocol) instead of the interior angle (180 = straight)")
        self.chk_tp_flex.setChecked(False)
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
        tpf.addRow(self.chk_tp_flex)
        tpf.addRow(self.tp_info)
        self.chk_tp.toggled.connect(self.apply_three_point)
        self.chk_tp_main.toggled.connect(self.apply_three_point)
        self.chk_tp_flex.toggled.connect(self.apply_three_point)

        # --- marking & lock
        self.instruction = QLabel("Start the camera first (Video source, further down in this panel).")
        self.instruction.setStyleSheet(BIG)
        self.instruction.setWordWrap(True)
        self.btn_remark = QPushButton("Re-mark all dots")
        self.btn_remark.clicked.connect(self.remark)
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
        mk = QGroupBox("Mark dots, set position, lock")
        ml = QVBoxLayout(mk)
        ml.addWidget(self.instruction)
        ml.addWidget(self.btn_remark)
        ml.addWidget(self.btn_lock)
        # PARKED-GHOST: ml.addWidget(self.chk_ghost)
        # PARKED-GHOST: ml.addWidget(self.chk_anchor)
        ml.addWidget(self.chk_trails)

        # --- record
        self.participant = TextField()
        self.participant.setPlaceholderText("P01, P02, ... (code only, no names)")
        self.notes = TextField()
        self.notes.setPlaceholderText("No names or health details")
        self.key_hint = QLabel("Typing: the letter keys are off while a text box has focus. Press Enter or Esc to leave it, "
                               "or use F5 record, F6 MVIC start, F7 MVIC end, F8 marker, F9 lock.")
        self.key_hint.setWordWrap(True)
        self.key_hint.setStyleSheet("color: #d97706; font-weight: bold;")
        self.key_hint.setVisible(False)
        for box in (self.participant, self.notes):
            box.focus_changed.connect(self._update_key_hint)
        self.chk_overlay = QCheckBox("Also save the overlay video")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setToolTip("The overlay video has the angles and trails drawn on it. Switching it off makes recording lighter on the "
                                    "computer; the Review tab can still draw the same picture from the raw video and the CSV, and can export it later.")
        self.chk_overlay.toggled.connect(self._set_overlay_saving)
        self.btn_rec = QPushButton("Start recording  (R)")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self.toggle_record)
        self.rec_label = QLabel("Not recording")
        self.trial_label = QLabel("")
        self.trial_label.setWordWrap(True)
        self.trial_label.setStyleSheet("font-weight: bold;")
        self.folder_label = QLabel(f"Saving to: {self.recordings_dir}")
        self.folder_label.setWordWrap(True)
        self.btn_folder = QPushButton("Change recordings folder...")
        self.btn_folder.setToolTip("Where trials are saved, for example an encrypted drive. Not possible while recording.")
        self.btn_folder.clicked.connect(self.choose_folder)
        btn_open = QPushButton("Open recordings folder")
        btn_open.clicked.connect(lambda: (self.recordings_dir.mkdir(exist_ok=True), os.startfile(self.recordings_dir)))
        self.btn_start_mark = QPushButton("MVIC start  (S / F6)")
        self.btn_end_mark = QPushButton("MVIC end  (E / F7)")
        self.btn_mark = QPushButton("Marker  (M / F8)")
        self.btn_start_mark.clicked.connect(lambda: self.mark("MVIC start"))
        self.btn_end_mark.clicked.connect(lambda: self.mark("MVIC end"))
        self.btn_mark.clicked.connect(lambda: self.mark(None))
        mrow = QHBoxLayout()
        for b in (self.btn_start_mark, self.btn_end_mark, self.btn_mark):
            mrow.addWidget(b)
        rc = QGroupBox("Record")
        rf = QFormLayout(rc)
        rf.addRow("Participant ID", self.participant)
        rf.addRow("Notes", self.notes)
        rf.addRow(self.trial_label)
        rf.addRow(self.btn_rec)
        rf.addRow(mrow)
        rf.addRow(self.rec_label)
        rf.addRow(self.key_hint)
        rf.addRow(btn_open)

        # --- privacy: what is written to the saved videos
        self.priv_label = QLabel()
        self.priv_label.setWordWrap(True)
        self.btn_crop = QPushButton("Draw crop...")
        self.btn_crop.setToolTip("Drag a rectangle round the leg. Only that part of the picture is saved to raw.mp4 and overlay.mp4.")
        self.btn_mask = QPushButton("Add black mask...")
        self.btn_mask.setToolTip("Drag a rectangle (for example over a face). It is black in the saved videos. At most 3.")
        self.btn_priv_clear = QPushButton("Clear")
        self.btn_crop.clicked.connect(lambda: self.start_privacy_drawing("crop"))
        self.btn_mask.clicked.connect(lambda: self.start_privacy_drawing("mask"))
        self.btn_priv_clear.clicked.connect(self.clear_privacy)
        self.view.rect_drawn.connect(self.on_privacy_rect)
        self._priv_kind = None
        pv = QGroupBox("Saved files and privacy (set once per session)")
        pvl = QVBoxLayout(pv)
        pvl.addWidget(self.priv_label)
        prow = QHBoxLayout()
        for b in (self.btn_crop, self.btn_mask, self.btn_priv_clear):
            prow.addWidget(b)
        pvl.addLayout(prow)
        pvl.addWidget(self.chk_overlay)
        pvl.addWidget(self.folder_label)
        pvl.addWidget(self.btn_folder)
        self.update_privacy_label()

        # --- scale (ruler with a dot at each end)
        self.ruler_len = self._spin(1.0, 100.0, 1.0, " cm")
        self.ruler_len.setValue(self.engine.scale.length_cm)
        self.ruler_len.setToolTip("Length between the two dot centres on the ruler (the 0 mark to the far mark).")
        self.ruler_len.valueChanged.connect(lambda v: setattr(self.engine.scale, "length_cm", float(v)))
        # --- camera set-up checklist and camera check (saved with each trial)
        self.camera_checks = {}
        cs = QGroupBox("Camera set-up (saved with each trial)")
        csl = QVBoxLayout(cs)
        for key, text in (("stabilisation_off", "Image stabilisation off"), ("zoom_1x", "Zoom 1x (no digital zoom)"),
                          ("focus_locked", "Focus locked"), ("exposure_wb_locked", "Exposure and white balance locked in the camera app"),
                          ("orientation_locked", "Orientation locked"), ("usb_connected", "Connected by USB (not Wi-Fi)")):
            cb = QCheckBox(text)
            cb.toggled.connect(self._camera_checklist_changed)
            csl.addWidget(cb)
            self.camera_checks[key] = cb
        self.camera_model = TextField()
        self.camera_model.setPlaceholderText("camera / phone model and app, e.g. Galaxy S25, Iriun")
        self.camera_model.editingFinished.connect(self._camera_checklist_changed)
        csl.addWidget(self.camera_model)
        self.btn_check_cam = QPushButton("Check the camera (5 s)")
        self.btn_check_cam.setToolTip("Tells you whether the live feed is fast and steady enough: frame rate, longest wait for a frame, format. "
                                      "Run it with the camera started, and again after changing anything.")
        self.btn_check_cam.clicked.connect(self.check_camera)
        csl.addWidget(self.btn_check_cam)
        self.camera_check_label = QLabel("")
        self.camera_check_label.setWordWrap(True)
        csl.addWidget(self.camera_check_label)

        self.btn_ruler = QPushButton("Mark the two ruler dots")
        self.btn_ruler.setToolTip("Put the ruler at the same distance from the camera as the joint, square to the camera, with a dot "
                                  "sticker centred on each end mark. Then click the two dots.")
        self.btn_ruler.clicked.connect(self.start_ruler_marking)
        self.btn_ruler_clear = QPushButton("Clear scale")
        self.btn_ruler_clear.clicked.connect(self.clear_scale)
        self.scale_label = QLabel("No scale set. Lengths in centimetres need a ruler in the picture.")
        self.scale_label.setWordWrap(True)
        self._ruler_marking = None          # 0 or 1 while the operator is clicking the two ruler dots
        sc = QGroupBox("Scale (ruler, optional)")
        sf = QFormLayout(sc)
        sf.addRow("Ruler length", self.ruler_len)
        srow = QHBoxLayout()
        srow.addWidget(self.btn_ruler)
        srow.addWidget(self.btn_ruler_clear)
        sf.addRow(srow)
        sf.addRow(self.scale_label)

        # --- live trace (last 15 s)
        self.trace_mode = NoWheelComboBox()
        self.trace_mode.addItem("Main angle, last 15 s", "main")
        self.trace_mode.addItem("Drift from set position, all joints", "drift")
        self.trace_mode.addItem("Force, last 15 s (when connected)", "force")
        self.trace_mode.addItem("Force voltage (V), last 15 s", "force_v")
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
        self.trace_timer.start(200)              # 5 redraws a second: each costs the window thread, and with it the picture thread

        # --- force: a load cell through a DAQ, sampled on its own thread (limbtrack/force.py)
        self.force_source = None
        self._force_msg = ""
        self.force_kind = NoWheelComboBox()
        for label, k in (("Off", "off"), ("NI DAQ (load cell)", "ni"), ("Simulated (test, no hardware)", "simulated")):
            self.force_kind.addItem(label, k)
        self.force_dev = NoWheelComboBox()
        self.force_dev.setEditable(True)
        self.force_dev.addItem("Dev1")
        self.force_chan = NoWheelComboBox()
        for i in range(8):
            self.force_chan.addItem(f"ai{i}")
        self.force_term = NoWheelComboBox()
        self.force_term.addItem("Differential (ai0 and ai4)", "differential")
        self.force_term.addItem("Single-ended", "single-ended")
        self.force_smooth_combo = NoWheelComboBox()
        for label, sec in (("Raw (no smoothing)", 0.0), ("20 ms", 0.02), ("100 ms", 0.1), ("200 ms", 0.2), ("500 ms", 0.5)):
            self.force_smooth_combo.addItem(label, sec)
        self.force_smooth_combo.setCurrentIndex(2)
        self.force_smooth_combo.setToolTip("What the read-out and the trace show (display only; the files hold every raw sample). A multiple of 20 ms "
                                           "removes 50 Hz mains hum exactly.")
        self.force_smooth_combo.currentIndexChanged.connect(self.on_force_smoothing)
        self.btn_force_scan = QPushButton("Scan DAQ devices")
        self.btn_force_scan.clicked.connect(self.scan_force_devices)
        self.btn_force = QPushButton("Connect")
        self.btn_force.clicked.connect(self.toggle_force)
        self.btn_force_zero = QPushButton("Zero: no pull on the strap  (Z)")
        self.btn_force_zero.setToolTip("Takes the present signal (1 s) as 0 N. Do it as often as needed: after every change of set-up "
                                       "and whenever the strap hangs differently. The zero is forgotten when the program closes. The volts of "
                                       "every recording are saved as they are, so a wrong zero can be corrected afterwards in Review.")
        self.btn_force_zero.clicked.connect(self.zero_force)
        self.btn_force_cal = QPushButton("Re-calibrate with a known weight... (this session only)")
        self.btn_force_cal.clicked.connect(self.calibrate_force)
        self.force_label = QLabel("Force: not connected")
        self.force_label.setStyleSheet(BIG)
        self.force_info = QLabel("")
        self.force_info.setWordWrap(True)
        fc = QGroupBox("Force (load cell through the DAQ)")
        fcf = QFormLayout(fc)
        fcf.addRow("Source", self.force_kind)
        fcf.addRow("Device", self.force_dev)
        fcf.addRow("Input", self.force_chan)
        fcf.addRow("Wiring", self.force_term)
        fcf.addRow("Display smoothing", self.force_smooth_combo)
        fcf.addRow(self.btn_force_scan)
        fcf.addRow(self.btn_force)
        fcf.addRow(self.force_label)
        fcf.addRow(self.btn_force_zero)
        fcf.addRow(self.btn_force_cal)
        fcf.addRow(self.force_info)

        # --- session plan: every position of the protocol with the trials already saved for this participant
        self.plan = QListWidget()
        self.plan.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plan.setTextElideMode(Qt.ElideRight)
        self.plan_rows = [(key, tgt) for key, t in engine.tests.items() for tgt in (t.targets or [None])]
        for _ in self.plan_rows:
            self.plan.addItem("")
        self.plan.setFixedHeight(min(len(self.plan_rows), 8) * (self.plan.fontMetrics().height() + 4) + 8)     # 8 rows, then scrolls
        self.plan.itemClicked.connect(self.on_plan_clicked)
        self.participant.textChanged.connect(self.refresh_plan)
        self.target_spin.valueChanged.connect(lambda _: self.refresh_plan())
        plan_box = QGroupBox("Session plan (click the next position)")
        plan_l = QVBoxLayout(plan_box)
        plan_l.addWidget(self.plan)
        self.refresh_plan()

        # --- rarely used settings, folded away until asked for
        self.btn_adv = QPushButton("Show advanced: landmarks, three-point angle")
        self.btn_adv.setCheckable(True)
        self.adv_body = QWidget()
        adv_l = QVBoxLayout(self.adv_body)
        adv_l.setContentsMargins(0, 0, 0, 0)
        adv_l.addWidget(lm)
        adv_l.addWidget(tpbox)
        self.adv_body.setVisible(False)
        self.btn_adv.toggled.connect(lambda on: (self.adv_body.setVisible(on), self.btn_adv.setText(
            "Hide advanced" if on else "Show advanced: landmarks, three-point angle")))

        side = QVBoxLayout()
        for g in (plan_box, mk, rc, tg, src, cs, fc, sc, pv, self.btn_adv, self.adv_body):
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
        QShortcut(QKeySequence(Qt.Key_Z), self, activated=self.btn_force_zero.click)
        # The F-keys work even while a text box has focus (letters would be typed into it): audit GUI-13.
        QShortcut(QKeySequence(Qt.Key_F5), self, activated=self.btn_rec.click)
        QShortcut(QKeySequence(Qt.Key_F6), self, activated=self.btn_start_mark.click)
        QShortcut(QKeySequence(Qt.Key_F7), self, activated=self.btn_end_mark.click)
        QShortcut(QKeySequence(Qt.Key_F8), self, activated=self.btn_mark.click)
        QShortcut(QKeySequence(Qt.Key_F9), self, activated=self.btn_lock.click)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.btn_back.click)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.btn_fwd.click)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_status)
        self.update_force_box()                  # button states right away, not after the first timer tick
        self.timer.start(200)
        self.on_test_changed()
        names = camera_names()                      # names only; no camera is opened
        if names:
            self.fill_cameras(list(enumerate(names)))

    @staticmethod
    def _spin(lo, hi, step, suffix):
        s = NoWheelDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setSuffix(suffix)
        s.setDecimals(1)
        return s

    # ---------------------------------------------------------- settings
    def on_test_changed(self):
        try:
            self.engine.set_test(self.test_combo.currentData())
        except RuntimeError as e:
            self._revert_combo(self.test_combo, "test", e)
            return
        self._combo_index["test"] = self.test_combo.currentIndex()
        t = self.engine.test
        self.suggest.clear()
        for v in t.targets:
            self.suggest.addItem(f"{v:g} deg")
        self.target_spin.blockSignals(True)
        self.target_spin.setValue(self.engine.target if self.engine.target is not None else self.target_spin.minimum())
        self.target_spin.blockSignals(False)                    # the engine already holds the test's first position (or none)
        self.tol_spin.setValue(self.engine.tolerance)
        self.ntol_spin.setValue(self.engine.neighbour_tolerance)
        self.btn_lock.setChecked(False)
        self.rebuild_landmarks()
        self.refresh_plan()

    # ------------------------------------------------- landmark selection
    def rebuild_landmarks(self):
        """One checkbox per landmark of the current test, plus presets like 'Only: Knee flexion'."""
        t = self.engine.test
        for cb in self.lm_checks.values():
            self.lm_holder.removeWidget(cb)
            cb.deleteLater()
        self.lm_checks = {}
        for n in t.all_dots:
            cb = QCheckBox(t.dot_title(n))
            cb.setChecked(n not in self.engine.disabled_dots)
            cb.toggled.connect(lambda on, name=n: self.on_landmark_toggled(name, on))
            self.lm_holder.addWidget(cb)
            self.lm_checks[n] = cb
        self.rebuild_three_point_choices()
        self.lm_preset.clear()
        self.lm_preset.addItem("Preset: all landmarks", list(t.dots))
        if t.optional_dots:
            self.lm_preset.addItem("Preset: all landmarks + strap and anchor", list(t.all_dots))
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
                self.engine.set_three_point(*[c.currentData() for c in self.tp_combos], main=self.chk_tp_main.isChecked(),
                                            flexion=self.chk_tp_flex.isChecked())
            else:
                self.engine.clear_three_point()
        except (ValueError, RuntimeError) as e:
            self.tp_info.setText(str(e))
            return
        self._sync_landmark_checks()
        self.btn_lock.setChecked(False)
        self.update_landmark_info()
        self.update_three_point_info()
        self.target_spin.blockSignals(True)               # the engine may have converted or cleared the target
        self.target_spin.setValue(self.target_spin.minimum() if self.engine.target is None else self.engine.target)
        self.target_spin.blockSignals(False)
        self.refresh_view()

    def update_three_point_info(self):
        e = self.engine
        main = bool(e.three_point) and e.three_point_main
        self.suggest.setEnabled(not main)         # the suggested targets belong to the test's own main joint
        if not e.three_point:
            self.tp_info.setText("")
        elif main:
            self.tp_info.setText(("The target is the flexion at the middle dot (a straight limb reads 0 deg, like the protocol). "
                                  if e.three_point_flexion else
                                  "The target is the INTERIOR angle at the middle dot (a straight limb reads 180 deg, so a knee flexed 70 deg is 110). ")
                                 + "The test's own angles are checked against the locked position instead.")
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
            for n in self.engine.test.all_dots:
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

    def _set_overlay_saving(self, on):
        try:
            self.engine.save_overlay = bool(on)
        except RuntimeError as e:
            self.chk_overlay.blockSignals(True)
            self.chk_overlay.setChecked(self.engine.save_overlay)
            self.chk_overlay.blockSignals(False)
            self.say(str(e))

    def on_mirror_toggled(self, on):
        self.engine.mirror = bool(on)
        self.engine.clear_privacy()               # drawn on the old orientation
        self.engine.reset_dots()                  # marked positions belong to the old orientation
        self.clear_scale()
        self.btn_lock.setChecked(False)
        self.refresh_view()
        self.instruction.setText("Image " + ("mirrored" if on else "not mirrored") + ". Mark the dots again, and check "
                                 "'Participant faces' against what you now see.")

    def _set_setting(self, attr, spin, value):
        """Change an engine setting from a spin box; if the engine refuses (recording), put the box back and say why."""
        if attr == "target" and value <= spin.minimum():
            value = None                                             # "no target"
        try:
            setattr(self.engine, attr, value)
        except RuntimeError as e:
            spin.blockSignals(True)
            current = getattr(self.engine, attr)
            spin.setValue(spin.minimum() if current is None else current)
            spin.blockSignals(False)
            self.say(str(e))
            return
        self.refresh_view()

    def _revert_combo(self, combo, key, error):
        combo.blockSignals(True)
        combo.setCurrentIndex(self._combo_index[key])
        combo.blockSignals(False)
        self.say(str(error))

    def on_facing_changed(self):
        try:
            self.engine.facing_right = bool(self.facing_combo.currentData())
        except RuntimeError as e:
            self._revert_combo(self.facing_combo, "facing", e)
            return
        self._combo_index["facing"] = self.facing_combo.currentIndex()
        self.refresh_view()

    def on_kind_changed(self):
        try:
            self.engine.set_dot_kind(self.kind_combo.currentData())
        except RuntimeError as e:
            self._revert_combo(self.kind_combo, "kind", e)
            return
        self._combo_index["kind"] = self.kind_combo.currentIndex()
        self.btn_lock.setChecked(False)

    # ------------------------------------------------------------ source
    def _camera_checklist_changed(self, *_):
        try:
            self.engine.camera_setup = {**{k: cb.isChecked() for k, cb in self.camera_checks.items()},
                                        "model_and_app": self.camera_model.text().strip()}
        except RuntimeError as e:
            self.say(str(e))

    def check_camera(self):
        cam = self.worker.camera if self.worker is not None else None
        if cam is None or cam.is_file:
            self.say("Start a camera first (a video file has no live feed to check).")
            return
        result = cam.health()
        if result.get("ok") is not None:
            result["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.engine.camera_check = result
        self.camera_check_label.setText(result["verdict"])
        self.camera_check_label.setStyleSheet("" if result["ok"] else "color: #d97706; font-weight: bold;")

    def connect_stream(self):
        typed = self.stream_box.text().strip()
        address = normalise_stream_address(typed)
        try:
            kind = classify_source(address) if address else None
        except ValueError:
            kind = None
        if kind != "stream":
            self.say("That is not a stream address. It should start with http://, https://, rtsp:// or similar.")
            self._source_note = "That is not a stream address. It should look like http://192.168.0.100:4747/video"
            self.update_camera_info()
            return
        if self.worker is not None:
            if self.engine.rec is not None:
                self.say("Stop recording before switching to another video source.")
                return
            self.stop_camera()                       # switching source: the running camera or video is stopped first
        if address != typed:
            self.stream_box.setText(address)        # show what is really being opened
        self.start_source(address)

    def fill_cameras(self, found):
        """Put (index, name) pairs in the camera box, keeping the same camera selected if it is still there."""
        keep = self.cam_combo.currentData()
        self.cam_combo.clear()
        for i, name in found:
            self.cam_combo.addItem(f"{i}: {name}", i)
        if keep is not None and self.cam_combo.findData(keep) >= 0:
            self.cam_combo.setCurrentIndex(self.cam_combo.findData(keep))

    def scan_cameras(self):
        if self.worker is not None:        # opening and releasing a camera that is streaming kills the stream (audit CAM-06)
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = list_cameras()
        finally:
            QApplication.restoreOverrideCursor()
        self.fill_cameras(found)
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
        self._source_note = ""
        self.engine.reset_dots()
        self.clear_scale()
        self.btn_lock.setChecked(False)
        self.worker = VideoWorker(self.engine, source)
        self.worker.frame_ready.connect(self.view.set_image)
        self.worker.position.connect(self.on_position)
        self._pos = (0, 0)
        self._set_file_controls(classify_source(source) == "file")
        self.worker.ended.connect(lambda msg, w=self.worker: self.on_ended(msg, w))
        self.worker.start()
        self.btn_start.setText("Stop camera")
        self.btn_scan.setEnabled(False)
        self.btn_stream.setEnabled(False)

    def stop_camera(self):
        if self.btn_rec.isChecked():
            self.toggle_record(force_stop=True)
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.btn_start.setText("Start camera")
        self.btn_scan.setEnabled(True)
        self.btn_stream.setEnabled(True)
        self._set_file_controls(False)
        self.update_camera_info()

    def on_ended(self, msg, worker):
        if worker is not self.worker:      # late signal from a source that was already replaced
            return
        if self.btn_rec.isChecked():
            self.toggle_record(force_stop=True)
        if self.worker is not None:
            self.worker.wait(3000)      # 'ended' is emitted just before the thread finishes
            self.worker = None
            self.btn_start.setText("Start camera")
            self.btn_scan.setEnabled(True)
            self.btn_stream.setEnabled(True)
            self._set_file_controls(False)
            self.update_camera_info()
            if msg != "Stopped.":
                self.instruction.setText(msg)
                self._source_note = msg
                try:
                    stream = classify_source(worker.source) == "stream"
                except ValueError:
                    stream = False
                if "stopped delivering" in msg or "did not answer" in msg:
                    self._source_note += ("\nFor a phone camera: the phone's PC program (DroidCam Client) must be connected to the phone "
                                          "and its own preview must be moving. Double-click Check_phone_camera.bat for a step-by-step check.")
                if stream:
                    self._source_note += ("\nCheck that the phone app is running, that the phone and this PC are on the same Wi-Fi "
                                          "(or use USB), and that the address is right (DroidCam: http://PHONE-IP:4747/video). "
                                          "Easier: install the DroidCam Client on the PC and choose 'DroidCam Video' in the camera list. "
                                          "For a step-by-step check, double-click Check_phone_camera.bat in the project folder.")
                self.update_camera_info()

    # ------------------------------------------------------- live trace
    def _draw_force_trace(self, volts_only=False):
        src = self.force_source
        if src is None:
            return
        t, v = src.recent(15.0)
        cal = src.calibration.calibrated and not volts_only
        y = src.calibration.to_newton(v) if cal else v
        self._trace_items.append(self.trace.plot(t, y, pen=pg.mkPen((220, 60, 60), width=2), name="Force (N)" if cal else "Force (V)"))
        self.trace.setLabel("left", "N" if cal else "V")
        self.trace.setXRange(-15, 0, padding=0)
        if y.size and np.isfinite(y).any():
            lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
            pad = 0.1 * max(hi - lo, 1.0 if cal else 0.01)
            self.trace.setYRange(lo - pad, hi + pad, padding=0)

    def update_trace(self):
        for it in self._trace_items:
            self.trace.removeItem(it)
        self._trace_items = []
        if self.trace_mode.currentData() in ("force", "force_v"):
            self._draw_force_trace(volts_only=self.trace_mode.currentData() == "force_v")
            return
        self.trace.setLabel("left", "deg")
        h = self.engine.history(15.0)
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
    # ------------------------------------------------------------ scale
    def start_ruler_marking(self):
        if self.worker is None:
            self.say("Start the camera or open a video first, then mark the ruler dots.")
            return
        if self.engine.rec is not None:
            self.say("The scale cannot be changed while recording.")
            return
        self.engine.clear_scale()
        self._ruler_marking = 0
        self.say("Click the ruler dot at one end of the ruler.", 8.0)

    def on_ruler_click(self, x, y):
        index = self._ruler_marking
        if self.engine.mark_scale_dot(index, x, y):
            if index == 0:
                self._ruler_marking = 1
                self.say("Now click the ruler dot at the other end.", 8.0)
            else:
                self._ruler_marking = None
                self.say(f"Scale set: {self.engine.scale.px_per_cm:.2f} px per cm ({self.ruler_len.value():g} cm ruler).")
        else:
            self.say(self.engine.mark_problem or "No dot found at that spot. Click the centre of the ruler dot.")
        self.refresh_view()

    def clear_scale(self):
        self.engine.clear_scale()
        self._ruler_marking = None
        self.refresh_view()

    def update_scale_label(self):
        sc = self.engine.scale
        recording = self._rec_started is not None
        for w in (self.ruler_len, self.btn_ruler, self.btn_ruler_clear):
            w.setEnabled(not recording)
        if not sc.configured:
            self.scale_label.setText("Marking the ruler dots..." if self._ruler_marking is not None
                                     else "No scale set. Lengths in centimetres need a ruler in the picture.")
            self.scale_label.setStyleSheet("")
            return
        text = f"Scale {sc.px_per_cm:.2f} px/cm (ruler {sc.ref_px:.0f} px = {sc.length_cm:g} cm). Fixed: mark it again if the camera or ruler moves."
        problems = sc.problems()
        if problems:
            text += "\n" + "\n".join(problems)
        sizes, small = self.engine.dot_sizes_cm(), self.engine.small_dots()
        if sizes:
            text += f"\nDots (approx.): smallest {min(sizes.values()):.1f} cm, minimum {MIN_DOT_DIAMETER_CM:g} cm"
            if small:
                text += "\nTOO SMALL: " + ", ".join(f"{self.engine.test.dot_title(n)} {v:.1f} cm" for n, v in small.items())
        self.scale_label.setText(text)
        self.scale_label.setStyleSheet("color: #d97706; font-weight: bold;" if (problems or small) else "")

    # ------------------------------------------------------------ force
    def scan_force_devices(self):
        found = NiDaqForce.devices()
        if not found:
            self._force_msg = "No DAQ device found (is the USB cable connected and the NI-DAQmx driver installed?)."
            self.update_force_box()
            return
        self.force_dev.clear()
        for name, product, chans in found:
            self.force_dev.addItem(name)
        self._force_msg = "Found: " + ", ".join(f"{n} ({p})" for n, p, _ in found)
        self.update_force_box()

    def on_force_smoothing(self, _=None):
        if self.force_source is not None:
            self.force_source.smooth_s = float(self.force_smooth_combo.currentData())

    def toggle_force(self):
        if self.force_source is not None:
            self.disconnect_force()
        else:
            self.connect_force()

    def connect_force(self):
        if self.engine.rec is not None:
            self.say("Stop recording before connecting the force.")
            return
        kind = self.force_kind.currentData()
        if kind == "off":
            self._force_msg = "Choose a source first (NI DAQ, or Simulated to try the program without hardware)."
            self.update_force_box()
            return
        if kind == "ni":
            src = NiDaqForce(device=self.force_dev.currentText().strip(), channel=self.force_chan.currentText(),
                             terminal=self.force_term.currentData())
            src.calibration = ForceCalibration.load_cell()          # the active regression's slope; no zero until the operator takes one
        else:
            src = SimulatedForce()
        try:
            src.start()
        except ForceError as e:
            self._force_msg = str(e)
            self.update_force_box()
            return
        try:
            self.engine.force = src
        except RuntimeError as e:
            src.stop()
            self.say(str(e))
            return
        self.force_source = src
        src.smooth_s = float(self.force_smooth_combo.currentData())
        self._force_msg = ""
        self.update_force_box()

    def disconnect_force(self):
        if self.engine.rec is not None:
            self.say("Stop recording before disconnecting the force.")
            return
        src, self.force_source = self.force_source, None
        self.engine.force = None
        if src is not None:
            src.stop()
        self._force_msg = ""
        self.update_force_box()

    def zero_force(self):
        src = self.force_source
        if src is None:
            return
        before = src.calibration.offset_v if src.calibration.zero_v is not None else None
        why = src.zero()
        if why:
            self._force_msg = why
        else:
            cal = src.calibration
            moved = "" if before is None or not cal.has_slope else f" (moved {(cal.offset_v - before) * cal.n_per_v:+.1f} N from the last zero)"
            self._force_msg = f"Zero set: {cal.offset_v:+.4f} V is now 0 N{moved}."
        self.update_force_box()

    def calibrate_force(self):
        src = self.force_source
        if src is None:
            return
        if src.calibration.zero_v is None:
            self._force_msg = "Press 'Zero' first, with nothing on the sensor."
            self.update_force_box()
            return
        mass, ok = QInputDialog.getDouble(self, "Known weight", "Mass that is on the sensor now (kg).\nKeep it still while it measures:",
                                          5.0, 0.01, 1000.0, 2)
        if not ok:
            return
        why = src.calibrate(mass)
        if why:
            self._force_msg = why
        else:
            self._force_msg = f"Calibrated with {mass:g} kg: {src.calibration.n_per_v:.2f} N per volt."
        self.update_force_box()

    def update_force_box(self):
        """Read-out, button states and the calibration line (called often; cheap)."""
        src, recording = self.force_source, self._rec_started is not None
        connected = src is not None
        for w in (self.force_kind, self.force_dev, self.force_chan, self.force_term, self.btn_force_scan):
            w.setEnabled(not connected and not recording)
        self.btn_force.setEnabled(not recording)
        self.btn_force.setText("Disconnect" if connected else "Connect")
        self.btn_force_zero.setEnabled(connected and not recording)
        self.btn_force_cal.setEnabled(connected and not recording)
        warn = False
        if not connected:
            self.force_label.setText("Force: not connected")
            info = self._force_msg
            warn = bool(info)
        else:
            if src.error:
                self.force_label.setText("FORCE LOST")
                info, warn = src.error, True
            else:
                v, n = src.latest_volts(), src.latest_newtons()
                self.force_label.setText("Force: waiting..." if v is None else f"Force {n:.1f} N   ({v:+.3f} V)" if n is not None
                                         else f"Force {v:+.4f} V")
                cal = src.calibration
                if cal.calibrated:
                    info = f"{cal.n_per_v:.2f} N per volt ({cal.method}), zero {cal.offset_v:+.4f} V set {cal.when[11:16]}."
                    reg = find_regression(cal.regression_id)
                    if reg is not None:
                        sd100 = 2 * float(reg.sd_newton(100 * GRAVITY))
                        info += f"\nA reading of 100 kg ({100 * GRAVITY:.0f} N) is good to +-{sd100:.1f} N (about 95 %)."
                elif cal.has_slope:
                    info, warn = "Press Zero (Z) with no pull on the strap. Until then only volts are shown and saved; newtons follow after the zero.", True
                else:
                    info = "No load cell calibration found: values are volts. Zero with nothing on the sensor, then calibrate with a known weight."
                if self._force_msg:
                    info = self._force_msg + "\n" + info
                    warn = not self._force_msg.startswith(("Zero set", "Calibrated"))        # a done zero is news, not a warning
        if info != self.force_info.text():
            self.force_info.setText(info)
        self.force_info.setStyleSheet("color: #d97706; font-weight: bold;" if warn else "")

    # ------------------------------------------------------------ session plan
    def _plan_text(self, key, tgt):
        t = self.engine.tests[key]
        return t.label.replace(" (MVIC)", "") + (f", {tgt:g} deg" if tgt is not None else ", no target")

    def refresh_plan(self):
        """Update the trial counts (read from the saved files), mark the current position and say which trial is next."""
        if not hasattr(self, "plan_rows"):
            return
        pid = self.participant.text().strip()
        valid = privacy.check_participant_id(pid) is None
        cur_key, cur_tgt = self.engine.test.key, (self.engine.target if self.engine.primary_angle else None)
        current, n_cur = None, 0
        for i, (key, tgt) in enumerate(self.plan_rows):
            n = count_trials(self.recordings_dir, pid, key, tgt) if valid else 0
            mark = "-" if not valid else ("done" if n >= TRIALS_PER_POSITION else f"{n}/{TRIALS_PER_POSITION}")
            text = f"{mark:>5}   {self._plan_text(key, tgt)}"
            item = self.plan.item(i)
            if item.text() != text:
                item.setText(text)
            if key == cur_key and ((tgt is None and cur_tgt is None) or (tgt is not None and cur_tgt is not None and abs(tgt - cur_tgt) < 1e-6)):
                current, n_cur = i, n
        self.plan.blockSignals(True)
        self.plan.setCurrentRow(-1 if current is None else current)
        self.plan.blockSignals(False)
        if current is not None:
            self.plan.scrollToItem(self.plan.item(current))
        where = self._plan_text(cur_key, cur_tgt)
        if not valid:
            self.trial_label.setText("Enter the participant code (P01, ...) to count trials.")
        elif current is None:
            self.trial_label.setText(f"{where}: not a position of the plan (trials are not counted).")
        elif n_cur >= TRIALS_PER_POSITION:
            self.trial_label.setText(f"{where}: {n_cur} trials already saved (planned {TRIALS_PER_POSITION}). Another is trial {n_cur + 1}.")
        else:
            self.trial_label.setText(f"Next: trial {n_cur + 1} of {TRIALS_PER_POSITION}\n{where}")

    def on_plan_clicked(self, item):
        key, tgt = self.plan_rows[self.plan.row(item)]
        idx = self.test_combo.findData(key)
        if idx != self.test_combo.currentIndex():
            self.test_combo.setCurrentIndex(idx)
            if self.engine.test.key != key:               # the engine refused (recording)
                self.refresh_plan()
                return
        self.target_spin.setValue(float(tgt) if tgt is not None else self.target_spin.minimum())
        self.say("Position set. Mark the dots if needed, check the angles, then lock.", 5.0)
        self.refresh_plan()

    # ------------------------------------------------------------ privacy and folder
    folder_changed = Signal(str)

    def start_privacy_drawing(self, kind):
        if self.worker is None:
            self.say("Start the camera or open a video first, then draw on the picture.")
            return
        self._priv_kind = kind
        self.view.set_draw_mode(True)
        self.say("Drag a rectangle on the video around the part to KEEP; everything outside it will not be saved." if kind == "crop"
                 else "Drag a rectangle on the video over the part to BLACK OUT in the saved videos.", 12.0)

    def on_privacy_rect(self, x0, y0, x1, y1):
        kind, self._priv_kind = self._priv_kind, None
        self.view.set_draw_mode(False)
        if kind is None:
            return
        try:
            ok, msg = self.engine.set_privacy_rect(kind, x0, y0, x1, y1)
        except RuntimeError as e:
            ok, msg = False, str(e)
        self.say(("Crop set." if kind == "crop" else "Mask added.") if ok else msg, 6.0)
        self.update_privacy_label()

    def clear_privacy(self):
        self._priv_kind = None
        self.view.set_draw_mode(False)
        try:
            self.engine.clear_privacy()
        except RuntimeError as e:
            self.say(str(e))
        self.update_privacy_label()

    def update_privacy_label(self):
        e = self.engine
        crop, masks = e.privacy_crop, e.privacy_masks
        text = (f"Saved video is cropped to {crop[2]}x{crop[3]} px." if crop else "Saved video: the whole picture.")
        if masks:
            text += f" {len(masks)} black mask" + ("s." if len(masks) > 1 else ".")
        text += "\nTracking and the CSV always use the whole picture."
        problem = e.privacy_problem()
        if problem:
            text += "\nWARNING: " + problem
        if text != self.priv_label.text():
            self.priv_label.setText(text)
            self.priv_label.setStyleSheet("color: #d97706; font-weight: bold;" if problem else "")
        self.view.set_privacy_shapes(crop, masks)

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose where recordings are saved", str(self.recordings_dir))
        if not d:
            return
        if self.engine.rec is not None:
            self.say("Stop recording before changing the recordings folder.")
            return
        self.engine.recordings_dir = Path(d)
        self.set_recordings_dir(Path(d))
        self.folder_changed.emit(str(d))
        self.refresh_plan()

    def set_recordings_dir(self, path):
        self.recordings_dir = Path(path)
        self.folder_label.setText(f"Saving to: {self.recordings_dir}")

    def say(self, text, seconds=4.0):
        """Show a message on the instruction line and keep it there (the status timer would overwrite it within 0.2 s)."""
        self.instruction.setText(text)
        self._say_until = time.monotonic() + seconds

    def on_click(self, x, y):
        if self.worker is None:
            return
        if self.engine.rec is not None:
            self.say("Stop recording to re-mark a dot. (Clicks on the video are ignored while recording.)")
            return
        if self._ruler_marking is not None:
            self.on_ruler_click(x, y)
            return
        first_marking = self.engine.next_dot_to_mark()
        name = first_marking or self.engine.nearest_dot(x, y)
        if name is None:
            self.say("Click directly on a dot to re-mark it, or use 'Re-mark all dots'.")
            return
        title = self.engine.test.dot_title(name)
        if self.engine.mark_dot(name, x, y):
            if not first_marking:
                self.say(f"Re-marked: {title}.")
        elif self.engine.mark_problem:
            self.say(self.engine.mark_problem)
        else:
            self.say(f"No dot found at that spot for '{title}'. Click the centre of the dot (check colour setting and lighting).")
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
    def _update_key_hint(self, _=None):
        self.key_hint.setVisible(self.participant.hasFocus() or self.notes.hasFocus())

    def mark(self, label):
        """Write a marker into the recording (MVIC start/end or a numbered marker) and confirm it on screen."""
        if not self.engine.add_event(label):
            self.rec_label.setText("Markers can only be added while recording.")
            return
        kind = "start" if label == "MVIC start" else "end" if label == "MVIC end" else "other"
        self._marker_counts[kind] += 1
        at = time.perf_counter() - self._rec_started if self._rec_started is not None else 0.0
        self._marker_msg = (f"{label or 'Marker'} marked at {at:.1f} s", time.monotonic() + 3.0)

    def toggle_record(self, checked=None, force_stop=False):
        if self.btn_rec.isChecked() and not force_stop:
            if self.worker is None or not self.worker.isRunning():
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Cannot record", "No video is coming in, so nothing would be recorded. "
                                    "Start the camera or open a video file first.")
                return
            if self.worker.is_file and self.worker.at_end:
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Cannot record", "The video file has already ended, so nothing would be recorded. "
                                    "Step back with the Left arrow key or open the file again.")
                return
            id_problem = privacy.check_participant_id(self.participant.text().strip())
            if id_problem:
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Participant", id_problem)
                return
            notes_problem = privacy.notes_problem(self.notes.text())
            if notes_problem:
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Notes", notes_problem + " Remove it: notes must not identify anyone.")
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
            self.refresh_plan()
            self._marker_counts = {"start": 0, "end": 0, "other": 0}
            self._marker_msg = ("", 0.0)
            self.btn_rec.setText("Stop recording  (R)")
            if self.worker is not None and self.worker.paused:      # recording needs frames flowing
                self.btn_pause.setChecked(False)
                self.toggle_pause()
        else:
            self.btn_rec.setChecked(False)
            n = self.engine.rec.frames if self.engine.rec is not None else None
            duration = time.perf_counter() - self._rec_started if self._rec_started is not None else 0.0
            try:
                paths = self.engine.stop_trial()
            except Exception as e:  # noqa: BLE001 - disk full, file open elsewhere...
                paths = None
                self.btn_rec.setText("Start recording  (R)")
                self._rec_started = None
                self.rec_label.setText("NOT SAVED PROPERLY: " + str(e))
                self.rec_label.setStyleSheet("color: #dc2626; font-weight: bold;")
                QMessageBox.critical(self, "Saving failed", f"The recording could not be saved completely:\n{e}\n\n"
                                     "The CSV written so far is in the recordings folder.")
                return
            self.btn_rec.setText("Start recording  (R)")
            self._rec_started = None
            self.refresh_plan()
            if not paths:
                self.rec_label.setText("Not recording")
            elif n == 0:
                self.rec_label.setText("NOTHING RECORDED (no video frames arrived). Repeat the trial.")
                self.rec_label.setStyleSheet("color: #dc2626; font-weight: bold;")
                QMessageBox.warning(self, "Nothing recorded", "No video frames arrived during this recording, so the trial is empty. "
                                    "Check the camera and repeat the trial.")
            else:
                short = n is not None and n < 30
                self.rec_label.setText(f"Saved: {paths['data.csv'].parent.name}/{paths['raw.mp4'].name[:-8]}... "
                                       f"({n} frames, {duration:.1f} s)" + ("  VERY SHORT RECORDING" if short else ""))
                self.rec_label.setStyleSheet("color: #d97706; font-weight: bold;" if short else "")

    def update_camera_info(self):
        """One line on what the camera really delivers; orange when it differs from what was asked for."""
        cam = self.worker.camera if self.worker is not None else None
        self.btn_lock_cam.setEnabled(False)       # switched off: on some webcams the lock blacks out the picture (audit CAM-05)
        if cam is None:
            note = getattr(self, "_source_note", "")
            self.cam_info.setText(("WARNING: " + note) if note else "")
            self.cam_info.setStyleSheet("color: #d97706; font-weight: bold;" if note else "")
            return
        g, m = cam._granted, cam.measured_fps()
        if cam.is_file:
            self.cam_info.setText(f"Video file: {g['width']}x{g['height']}, {cam.file_fps:.1f} fps")
            self.cam_info.setStyleSheet("")
            return
        text = f"Camera delivers {g['width']}x{g['height']}" + (f", {m:.1f} fps measured" if m else ", measuring frame rate...")
        problems = cam.settings_problems()
        if problems:
            text += "\nWARNING: " + "; ".join(problems)
        if self.worker.lock_message:
            text += "\n" + self.worker.lock_message
        if self.worker.status_note:
            text += "\nWARNING: " + self.worker.status_note
            problems = problems + [self.worker.status_note]
        self.cam_info.setText(text)
        self.cam_info.setStyleSheet("color: #d97706; font-weight: bold;" if problems else "")

    def refresh_status(self):
        recording = self._rec_started is not None
        self.update_force_box()
        self.plan.setEnabled(not recording)
        for w in (self.btn_crop, self.btn_mask, self.btn_priv_clear, self.btn_folder):
            w.setEnabled(not recording)
        self.update_privacy_label()
        for cb in self.lm_checks.values():
            cb.setEnabled(not recording)
        self.lm_preset.setEnabled(not recording)
        self.chk_mirror.setEnabled(not recording)
        self.facing_combo.setStyleSheet("background-color: #fcd34d;" if self.engine.facing_mismatch else "")
        self.facing_combo.setToolTip(self.engine.facing_message() if self.engine.facing_mismatch else "")
        self.chk_overlay.setEnabled(not recording)
        for w in (*self.camera_checks.values(), self.camera_model):
            w.setEnabled(not recording)
        for w in (self.test_combo, self.kind_combo, self.facing_combo, self.target_spin, self.tol_spin, self.ntol_spin,
                  self.btn_remark, self.btn_lock):
            w.setEnabled(not recording)           # nothing that changes the measurement may change during a recording
        main_three_point = bool(self.engine.three_point) and self.engine.three_point_main
        self.suggest.setEnabled(not recording and not main_three_point)
        for w in (self.chk_tp, self.chk_tp_main, self.chk_tp_flex, *self.tp_combos):
            w.setEnabled(not recording)
        if self._rec_started is not None and self.engine.rec is None:       # the engine ended the recording itself
            reason = self.engine.last_stop_reason
            self.toggle_record(force_stop=True)
            if reason:
                self.rec_label.setText("RECORDING STOPPED: " + reason)
                self.rec_label.setStyleSheet("color: #dc2626; font-weight: bold;")
                self.say(reason, 15.0)
        if self._rec_started is not None:
            stalled = (self.engine.seconds_since_last_frame() > 0.5 and not (self.worker is not None and self.worker.paused))
            if stalled:
                self.rec_label.setText("NO VIDEO - nothing is being recorded")
                self.rec_label.setStyleSheet("color: #dc2626; font-weight: bold;")
            else:
                c = self._marker_counts
                text = (f"RECORDING  {time.perf_counter() - self._rec_started:5.1f} s   "
                        f"(MVIC start {c['start']}, end {c['end']}, other {c['other']})")
                if time.monotonic() < self._marker_msg[1]:
                    text += "\n" + self._marker_msg[0]
                self.rec_label.setText(text)
                self.rec_label.setStyleSheet("")
        self.update_camera_info()
        self.update_scale_label()
        if self.worker is None:
            return
        if time.monotonic() < self._say_until:
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
        if self.force_source is not None:                # release the DAQ so other programs can use it
            self.disconnect_force()


class MainWindow(QWidget):
    def __init__(self, tests, root):
        super().__init__()
        self.setWindowTitle("Limb angle tracker")
        self.resize(1400, 820)
        self.settings_path = Path(root) / "settings.json"          # local choices of this computer (not in the repository)
        folder = Path(root) / "recordings"
        try:
            saved = json.loads(self.settings_path.read_text(encoding="utf-8")).get("recordings_dir")
            if saved:
                folder = Path(saved)
        except (OSError, ValueError, AttributeError):
            pass
        self.engine = Engine(tests, recordings_dir=folder)
        self.engine.protocol_path = Path(root) / "protocol.json"
        tabs = QTabWidget()
        self.live = LiveTab(self.engine, folder)
        self.review = ReviewTab(folder)
        self.live.folder_changed.connect(self.on_folder_changed)
        tabs.addTab(self.live, "Live")
        tabs.addTab(self.review, "Review")
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)

    def on_folder_changed(self, path):
        self.review.dir = Path(path)
        try:
            self.settings_path.write_text(json.dumps({"recordings_dir": path}, indent=2), encoding="utf-8")
        except OSError:
            pass

    def closeEvent(self, e):
        self.live.shutdown()
        super().closeEvent(e)
