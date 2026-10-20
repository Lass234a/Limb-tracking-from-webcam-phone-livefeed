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
from PySide6.QtCore import QRect, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
                               QSplitter, QTabWidget, QListWidget, QVBoxLayout, QWidget)

from . import privacy, session_memory
from .session_log import SessionLog
from .panels import Accordion, Section, StepStrip
from .camera import CameraSource, camera_names, classify_source, list_cameras, normalise_stream_address
from .engine import Engine
from .loadcell import GRAVITY, find_regression
from .force import HUM_NOTE_V, ForceCalibration, ForceError, NiDaqForce, SimulatedForce
from .recorder import count_trials
from .automvic import AUTO_STOP_AFTER_END_S, DEFAULT_LENGTH_S, DEFAULT_THRESHOLD_N
from .scale import MIN_DOT_DIAMETER_CM
from .review import ReviewTab

BIG = "font-size: 15pt; font-weight: bold;"
NOTE = "font-size: 11pt; font-weight: bold;"
IDLE_TEXT = "Enter the participant code, then pick the camera and press Start camera."
# The steps of a session, in the order they are done. The strip shows one chip each; the first one still to do is highlighted.
STEPS = (("participant", "Participant"), ("camera", "Camera"), ("privacy", "Privacy"), ("position", "Position"), ("dots", "Dots"),
         ("ruler", "Ruler"), ("lock", "Lock"), ("force", "Force"), ("markers", "Markers"), ("record", "Record"))
STEP_SECTION = {"participant": "participant", "camera": "camera", "privacy": "privacy", "position": "position", "dots": "dots",
                "ruler": "ruler", "lock": "dots", "force": "force", "markers": "markers", "record": "record"}
STEP_HINT = {"participant": "Enter the participant code (P01, ...). Code only, no names.",
             "camera": "Pick the camera and press Start camera.",
             "privacy": "Draw a crop round the leg, black out anything that identifies, or tick that nothing does.",
             "position": "Pick the position in the session plan (or the test and target).",
             "dots": "Check the dot colour, then click each dot on the video, in the order asked.",
             "ruler": "Hold the ruler at the joint, click its two dots (or tick that there is no ruler today).",
             "lock": "Put the participant in position, then press Lock position (Space).",
             "force": "Connect the force, then press Zero with no pull on the strap (Z), or tick that there is no force today.",
             "markers": "Switch the automatic markers and the beeps on, and try them (Test the beeps, Try it without recording).",
             "record": "Press Start recording (R)."}
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
        self._memory_path = None             # settings.json, once the window has attached it (see attach_memory)
        self._restoring = False
        self._save_timer = None
        self._last_source = None             # the last camera or stream that was started: {"kind": "webcam", "name": ...} / {"kind": "stream", "address": ...}
        self._remembered_source = None
        self._force_wanted = False           # the operator connected the force and did not disconnect it again
        self._auto_mvic_wanted = False
        self._step_prev = {}
        self._steps_ready = False
        self._participant_restored = ""      # a code kept from the last session and not yet confirmed (Enter in the box, or Yes when recording)
        self._arm = None                     # the landmark whose Re-mark button was pressed: the next click on the picture marks it
        self._auto_text = IDLE_TEXT          # the last instruction the program wrote itself (a message from elsewhere is not overwritten by it)
        self._first_todo = None              # the step the strip says to do next
        self._beeps_tested = False           # the beeps were played this session without an error (not remembered: every session)
        self._beeps_test_pending = False
        self._reh_timer = None
        self.log = SessionLog(self.recordings_dir)    # what the operator did and when (session_log.csv), for reviewing set-up time

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
        self.chk_autostart = QCheckBox("Open the last camera and force at start")
        self.chk_autostart.setChecked(True)
        self.chk_autostart.setToolTip("The program remembers the camera by name and the force settings. When it starts it opens them for you; "
                                      "the marked dots, the lock and the force zero are never kept: do those again each session.")
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
        sl.addWidget(self.chk_autostart)
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
        self.cam_info = QLabel("")
        self.cam_info.setWordWrap(True)
        sl.addWidget(self.cam_info)
        self.chk_mirror = QCheckBox("Mirror the image (flip left-right)")
        self.chk_mirror.setToolTip("Flips the picture before tracking and recording, so saved video and pixel positions are flipped too. "
                                   "Changing it clears the marked dots.")
        self.chk_mirror.toggled.connect(self.on_mirror_toggled)
        self.btn_lock_cam = QPushButton("Lock exposure and white balance (switched off)")
        self.btn_lock_cam.setToolTip("Switched off: on some webcams this blacks out the picture and reports the opposite of what happened. "
                                     "Use steady lighting, or lock exposure and white balance in the phone/camera app instead.")
        self.btn_lock_cam.clicked.connect(lambda: self.worker and self.worker.request_lock())
        self.btn_lock_cam.setEnabled(False)
        self.btn_lock_cam.setVisible(False)             # switched off for good (see its tooltip): not shown, so it cannot confuse
        more = QWidget()                                # what is rarely needed folds away: a video file, a network stream, the mirror
        more_l = QVBoxLayout(more)
        more_l.setContentsMargins(0, 0, 0, 0)
        more_l.addWidget(btn_file)
        more_l.addWidget(self.stream_box)
        more_l.addWidget(self.btn_stream)
        more_l.addLayout(prow)
        more_l.addWidget(self.frame_label)
        more_l.addWidget(self.chk_mirror)
        sl.addWidget(Section("more", "More: video file, network stream, mirror", [more]))
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
        self.kind_combo.setToolTip("Black discs drawn on white tape (see MARKERS.md): choose Black dots. On Auto the program can follow "
                                  "the white tape instead of a thin ink circle, without any warning. Auto is for white dots on dark "
                                  "cloth and black dots on skin in the same picture.")
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
        f.addRow("Participant faces", self.facing_combo)
        f.addRow(self.chk_smooth)

        # --- dot colour: up with the dots, because a wrong colour is the commonest reason for dots that will not mark or are followed badly
        self.kind_note = QLabel()
        self.kind_note.setWordWrap(True)
        self.kind_combo.currentIndexChanged.connect(self.update_kind_note)
        dc = QWidget()                                   # (plain widgets with a heading: group boxes inside a section lose their frame and title)
        dcf = QFormLayout(dc)
        dcf.setContentsMargins(0, 0, 0, 0)
        dcf.addRow(self._heading("Dot colour"))
        dcf.addRow("Dot colour", self.kind_combo)
        dcf.addRow(self.kind_note)

        # --- landmarks: one row each (use it / its state / Re-mark), so one dot can be redone without marking all of them again
        self.lm_preset = NoWheelComboBox()
        self.lm_preset.activated.connect(self.apply_landmark_preset)
        self.lm_checks, self.lm_status, self.lm_buttons, self.lm_rows = {}, {}, {}, {}
        self.lm_holder = QVBoxLayout()
        self.lm_holder.setSpacing(2)
        self.lm_info = QLabel()
        self.lm_info.setWordWrap(True)
        lm_hint = QLabel("To redo ONE dot, press its Re-mark button and then click that dot on the video. The other dots stay as they are.")
        lm_hint.setWordWrap(True)
        lm = QWidget()
        lml = QVBoxLayout(lm)
        lml.setContentsMargins(0, 0, 0, 0)
        lml.addWidget(self._heading("Landmarks"))
        lml.addWidget(lm_hint)
        lml.addLayout(self.lm_holder)
        lml.addWidget(self.lm_preset)
        lml.addWidget(self.lm_info)

        # --- three-point angle
        self.chk_tp = QCheckBox("Use a three-point angle")
        self.tp_combos = [NoWheelComboBox() for _ in range(3)]
        self.chk_tp_flex = QCheckBox("Show it as flexion (0 = straight)")
        self.chk_tp_flex.setToolTip("Flexion (0 = straight, like the protocol) instead of the interior angle (180 = straight).")
        self.chk_tp_flex.setChecked(False)
        self.chk_tp_main = QCheckBox("Use it as the main joint")
        self.chk_tp_main.setToolTip("The target above then applies to it.")
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
        self.instruction = QLabel(IDLE_TEXT)
        self.instruction.setStyleSheet(NOTE)
        self.instruction.setWordWrap(True)
        self.btn_remark = QPushButton("Clear all dots and start again")
        self.btn_remark.setToolTip("Forgets every marked dot (and the lock). To redo only one, use the Re-mark button of that landmark.")
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
        mk = QGroupBox("Dots, set position, lock")
        ml = QVBoxLayout(mk)
        ml.addWidget(dc)
        ml.addWidget(lm)
        ml.addWidget(self.btn_remark)
        # PARKED-GHOST: ml.addWidget(self.chk_ghost)
        # PARKED-GHOST: ml.addWidget(self.chk_anchor)
        ml.addWidget(self.chk_trails)

        # --- record
        self.participant = TextField()
        self.participant.setPlaceholderText("P01, P02, ... (code only, no names)")
        self.participant.returnPressed.connect(self._participant_entered)           # Enter only: Tab or a click into Notes must not take the box away
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
        self.chk_auto_mvic = QCheckBox("Mark the MVIC automatically from the force")
        self.chk_auto_mvic.setToolTip("Press Record to arm it. The MVIC start marker is set when the force rises from its rest level (the ramp "
                                      "begins) and the end marker the number of seconds below after it (protocol, 1-5-1 tempo: 1 s up, 5 s hold, "
                                      "1 s relax = 7 s). Needs the force connected. S and E still work: pressing one takes the markers over.")
        self.chk_auto_mvic.setEnabled(False)
        self.chk_auto_mvic.toggled.connect(self.on_auto_mvic_toggled)
        self.auto_thr_spin = self._spin(1.0, 500.0, 1.0, " N")
        self.auto_thr_spin.setValue(DEFAULT_THRESHOLD_N)
        self.auto_thr_spin.setToolTip("The start is called when the force has risen this much from its rest level and keeps rising. Small enough "
                                      "to be well inside the ramp, big enough that a touch on the strap does not count.")
        self.auto_thr_spin.valueChanged.connect(lambda v: self._set_setting("auto_mvic_threshold", self.auto_thr_spin, v))
        self.auto_len_spin = self._spin(1.0, 30.0, 0.5, " s")
        self.auto_len_spin.setValue(DEFAULT_LENGTH_S)
        self.auto_len_spin.setToolTip("The end marker comes this long after the start marker. Protocol: ramp 1 s + hold 5 s + relax 1 s = 7 s.")
        self.auto_len_spin.valueChanged.connect(lambda v: self._set_setting("auto_mvic_seconds", self.auto_len_spin, v))
        for w in (self.auto_thr_spin, self.auto_len_spin):
            w.setEnabled(False)
        self.chk_auto_stop = QCheckBox("Stop the recording by itself after the MVIC end")
        self.chk_auto_stop.setChecked(True)
        self.chk_auto_stop.setToolTip("The recording stops the number of seconds below after the MVIC end marker (automatic or the E key), exactly as "
                                      "if Stop had been pressed, so the next trial is one press of R away. Untick it to stop every recording yourself.")
        self.auto_stop_spin = self._spin(0.5, 30.0, 0.5, " s")
        self.auto_stop_spin.setValue(AUTO_STOP_AFTER_END_S)
        self.auto_stop_spin.setToolTip("How long the recording goes on after the MVIC end marker (the relax and a little rest).")
        self.chk_auto_stop.toggled.connect(self._set_auto_stop)
        self.auto_stop_spin.valueChanged.connect(self._set_auto_stop)
        self.chk_cues = QCheckBox("Sound cues (beeps)")
        self.chk_cues.setChecked(self.engine.audio_cues)
        self.chk_cues.setToolTip(
            "Record pressed: one beep. Ramp registered (the MVIC start marker, from the force or the S key): two rising beeps. One second "
            "later, when the ramp should be over: a longer beep, then a short tick every second while the force is held. One second before "
            "the end: two falling beeps (relax). End marker: one low beep. The 1-5-1 tempo follows 'MVIC length'. A participant who hears "
            "the beeps may push differently, so the setting is saved with each trial.")
        self.chk_cues.toggled.connect(self.on_cues_toggled)
        self.cue_push_spin = self._spin(0.0, 5.0, 0.5, " s")
        self.cue_push_spin.setValue(self.engine.cue_push_every_s)
        self.cue_push_spin.setToolTip("A short motivating tick this often while the force is held (0 = no ticks).")
        self.cue_push_spin.valueChanged.connect(lambda v: self._set_setting("cue_push_every_s", self.cue_push_spin, v))
        self.cue_push_spin.setEnabled(self.engine.audio_cues)
        self.btn_cue_test = QPushButton("Test the beeps")
        self.btn_cue_test.setToolTip("Plays every beep once, in the order of a contraction (also when the beeps are switched off).")
        self.btn_cue_test.clicked.connect(self.test_beeps)
        self.btn_rehearse = QPushButton("Try it without recording")
        self.btn_rehearse.setToolTip("Push on the strap as in a real MVIC. The program watches the force exactly as it does in a recording, calls the "
                                     "start, plays the beeps and says when it would have written the end marker. Nothing is saved.")
        self.btn_rehearse.clicked.connect(self.toggle_rehearsal)
        self.rehearse_label = QLabel("")
        self.rehearse_label.setWordWrap(True)
        self.rehearse_label.setVisible(False)                          # shown only while it has something to say
        self.chk_markers_hand = QCheckBox("I mark the MVIC by hand today (S and E keys)")
        self.chk_markers_hand.setToolTip("Only tells the step list that you have decided: the automatic markers are then not asked for. "
                                         "Not remembered: it is asked again every session.")
        # participant: the first thing to enter, before the position (the session plan counts this participant's saved trials)
        self.btn_participant_ok = QPushButton()
        self.btn_participant_ok.setToolTip("The code was kept from the last session. Press this (or Enter in the box) if it is the same participant; "
                                           "otherwise type the new code.")
        self.btn_participant_ok.clicked.connect(self._participant_entered)
        self.btn_participant_ok.setVisible(False)                    # only while a code from the last session waits for an answer
        pc = QGroupBox("Participant")
        pf = QFormLayout(pc)
        pf.addRow("Participant ID", self.participant)
        pf.addRow(self.btn_participant_ok)
        pf.addRow("Notes", self.notes)
        pf.addRow(self.key_hint)
        # markers and sound: set and tried before the first recording, not inside it
        mkb = QGroupBox("Markers and sound")
        mf = QFormLayout(mkb)
        mf.addRow(self.chk_auto_mvic)
        mf.addRow("Start when force rises by", self.auto_thr_spin)
        mf.addRow("MVIC length", self.auto_len_spin)
        mf.addRow(self.chk_auto_stop)
        mf.addRow("Stop this long after the end", self.auto_stop_spin)
        mf.addRow(self.chk_cues)
        mf.addRow("Tick during the hold every", self.cue_push_spin)
        mf.addRow(self.btn_cue_test)
        mf.addRow(self.btn_rehearse)
        mf.addRow(self.rehearse_label)
        mf.addRow(self.chk_markers_hand)
        rc = QGroupBox("Record")
        rf = QFormLayout(rc)
        rf.addRow(self.trial_label)
        rf.addRow(mrow)
        rf.addRow(self.rec_label)
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
        self.chk_priv_ok = QCheckBox("I looked: nothing identifies the participant")
        self.chk_priv_ok.setToolTip("For a picture that needs no crop or mask. It only tells the step list that you have looked; it changes "
                                    "nothing in the saved files. Not remembered: it is asked again every session, because the picture changes.")
        pvl.addWidget(self.chk_priv_ok)
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
                          ("focus_locked", "Focus locked"), ("exposure_wb_locked", "Exposure and white balance locked"),
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
        self.chk_ruler_skip = QCheckBox("No ruler today (no lengths in cm)")
        self.chk_ruler_skip.setToolTip("Lengths in centimetres need a ruler in the picture; angles do not. Only tells the step list that you have "
                                       "decided. Not remembered: it is asked again every session.")
        sf.addRow(self.chk_ruler_skip)

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
        self.btn_force_cal = QPushButton("Re-calibrate with a known weight...")
        self.btn_force_cal.setToolTip("For this session only: the result is not saved for the next one.")
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
        self.chk_force_skip = QCheckBox("No force today (angles only)")
        self.chk_force_skip.setToolTip("Only tells the step list that you have decided. Not remembered: it is asked again every session.")
        fcf.addRow(self.chk_force_skip)

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
        self.btn_adv = QPushButton("Show advanced: three-point angle")
        self.btn_adv.setCheckable(True)
        self.adv_body = QWidget()
        adv_l = QVBoxLayout(self.adv_body)
        adv_l.setContentsMargins(0, 0, 0, 0)
        adv_l.addWidget(tpbox)
        self.adv_body.setVisible(False)
        self.btn_adv.toggled.connect(lambda on: (self.adv_body.setVisible(on), self.btn_adv.setText(
            "Hide advanced" if on else "Show advanced: three-point angle")))

        # The panel: a strip that never scrolls (the steps of a session, what to do now, and the buttons used in every trial), and below
        # it sections of which one is open at a time. Opening a section closes the others; a closed one says what it is set to.
        self.strip = StepStrip(list(STEPS), columns=5)
        self.strip.activated.connect(self.open_step)
        self.instruction.setWordWrap(True)
        for b in (self.btn_lock, self.btn_rec, self.btn_pause):
            b.setMinimumHeight(34)
        quick = QGridLayout()
        quick.setSpacing(4)
        quick.addWidget(self.btn_lock, 0, 0)
        quick.addWidget(self.btn_rec, 0, 1)
        quick.addWidget(self.btn_pause, 1, 0, 1, 2)
        head = QVBoxLayout()
        head.setSpacing(6)
        head.addWidget(self.strip)
        head.addWidget(self.instruction)
        head.addLayout(quick)
        self.accordion = Accordion()
        for number, key, title, widgets in (("1", "participant", "Participant", [pc]),
                                            ("2", "camera", "Camera", [src]),
                                            ("3", "privacy", "Privacy and saved files", [pv]),
                                            ("4", "position", "Position and angles", [plan_box, tg]),
                                            ("5", "dots", "Dots, landmarks and lock", [mk]),
                                            ("6", "ruler", "Ruler for centimetres", [sc]),
                                            ("7", "force", "Force", [fc]),
                                            ("8", "markers", "Markers and sound", [mkb]),
                                            ("9", "record", "Record", [rc]),
                                            (None, "checklist", "Camera set-up checklist (set once)", [cs]),
                                            (None, "advanced", "Advanced: three-point angle", [self.btn_adv, self.adv_body])):
            self.accordion.add(key, title, widgets, number)
        self.btn_adv.setVisible(False)                    # the 'advanced' section is the button now; kept so that anything using it still works
        self.btn_adv.toggled.connect(lambda on: on and self.accordion.open("advanced"))
        self.accordion.opened.connect(lambda key: key == "advanced" and self.btn_adv.setChecked(True))
        side = QVBoxLayout()
        side.addWidget(self.accordion)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        self.side_scroll = QScrollArea()                 # only the opened section can ever need scrolling, and only on a small screen
        self.side_scroll.setWidget(side_w)
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QFrame.NoFrame)
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        rl.addLayout(head)
        rl.addWidget(self.side_scroll, 1)
        right.setMinimumWidth(400)
        left = QVBoxLayout()
        left.addWidget(self.view, 1)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Live trace:"))
        trow.addWidget(self.trace_mode)
        trow.addStretch(1)
        left.addLayout(trow)
        left.addWidget(self.trace)
        left_w = QWidget()
        left_w.setLayout(left)
        self.splitter = QSplitter(Qt.Horizontal)         # drag the divider to give the video or the panel more room
        self.splitter.addWidget(left_w)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([1000, 440])
        lay = QHBoxLayout(self)
        lay.addWidget(self.splitter)

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
        self.update_kind_note()
        names = camera_names()                      # names only; no camera is opened
        if names:
            self.fill_cameras(list(enumerate(names)))

    # ------------------------------------------------------------ the steps of a session
    def open_step(self, key):
        """Open the panel section that belongs to a step and bring it into view."""
        name = STEP_SECTION.get(key, key)
        self.accordion.open(name)
        sec = self.accordion.sections.get(name)
        if sec is not None:
            QTimer.singleShot(0, lambda: self._scroll_to(sec))
            QTimer.singleShot(80, lambda: self._scroll_to(sec))      # (again once the opened section has its full height)

    def _scroll_to(self, sec):
        """Put the header of the section that was just opened at the top of the panel (only matters when the panel is taller than the window)."""
        try:
            bar = self.side_scroll.verticalScrollBar()
            if bar.maximum() > 0:
                bar.setValue(sec.mapTo(self.side_scroll.widget(), sec.rect().topLeft()).y())
        except RuntimeError:                              # the window was closed in the meantime (the widgets are gone)
            pass

    def _source_text(self):
        w = self.worker
        if w is None:
            last = (self._last_source or self._remembered_source or {}).get("name")
            return f"not started (last: {last})" if last else "not started"
        if w.is_file:
            return "video file"
        if isinstance(getattr(w, "source", None), int):
            return self.cam_combo.currentText().split(": ", 1)[-1]
        return "network stream"

    def update_steps(self):
        """The chips, the one-line summaries of the closed sections, and moving on to the next section when a step has just been finished."""
        e = self.engine
        running = self.worker is not None and self.worker.isRunning()
        recording = self._rec_started is not None
        ready = running and e.trackers is not None and e.trackers.is_ready()
        names = e.active_dots
        n_marked = sum(1 for n in names if e.trackers.trackers[n].initialised) if e.trackers is not None else 0
        src = self.force_source
        kind = self.force_kind.currentData()
        zero_set = src is not None and src.calibration.zero_set
        pid = self.participant.text().strip()
        unconfirmed = self._participant_unconfirmed()
        id_ok = privacy.check_participant_id(pid) is None and not unconfirmed
        target = "no target" if e.target is None else f"{e.target:g} deg"
        shaped = bool(e.privacy_crop) or bool(e.privacy_masks)
        priv_problem = e.privacy_problem()
        priv_checked = self.chk_priv_ok.isChecked()
        ruler_set, ruler_skipped = e.scale.configured, self.chk_ruler_skip.isChecked()
        force_skipped, by_hand = self.chk_force_skip.isChecked(), self.chk_markers_hand.isChecked()
        auto_decided = self.chk_auto_mvic.isChecked() or by_hand or src is None          # without the force there is nothing to automate
        sound_decided = not self.chk_cues.isChecked() or self._beeps_tested
        if src is not None:
            force_state = ("done", "Force zeroed") if zero_set else ("todo", "Force: zero")
        elif force_skipped:
            force_state = ("done", "No force")
        else:
            force_state = ("todo", "Force" if kind == "off" else "Force: connect")
        states = {
            "participant": ("done", pid) if id_ok else ("todo", f"{pid}?") if unconfirmed else ("todo", "Participant"),
            "camera": ("done" if running else "todo", "Camera"),
            "privacy": (("off", "Privacy") if not running else ("todo", "Privacy: fix") if priv_problem
                        else ("done", "Privacy") if shaped or priv_checked else ("todo", "Privacy")),
            "position": ("done", f"Position {target}" if e.target is not None else "Position"),
            "dots": ("off" if not running else "done" if ready else "todo", f"Dots {n_marked}/{len(names)}"),
            "ruler": (("off", "Ruler") if not running else ("done", "Ruler") if ruler_set
                      else ("done", "No ruler") if ruler_skipped else ("todo", "Ruler")),
            "lock": ("off" if not ready else "done" if e.locked else "todo", "Lock"),
            "force": force_state,
            "markers": ("done" if auto_decided and sound_decided else "todo", "Markers"),
            "record": ("done", "Recording") if recording else ("off", "Record") if not (ready and id_ok) else ("todo", "Record"),
        }
        order = [k for k, _ in STEPS]
        first_todo = next((k for k in order if states[k][0] == "todo"), None)
        self._first_todo = first_todo
        for k, (state, label) in states.items():
            self.strip.set_state(k, "next" if k == first_todo else state, label, self._hint(k))
        self.btn_participant_ok.setVisible(unconfirmed)
        if unconfirmed:
            self.btn_participant_ok.setText(f"Same participant ({pid}): confirm")
        acc = self.accordion
        kind_name = self.kind_combo.currentText().split(" (")[0].lower()
        acc.set_summary("participant", (f"{pid}  (kept from last time: press Enter to confirm)" if unconfirmed else pid) or "no participant code")
        acc.set_summary("camera", self._source_text())
        acc.set_summary("privacy", ("crop" + (" + masks" if e.privacy_masks else "") if e.privacy_crop else
                                    f"{len(e.privacy_masks)} mask(s)" if e.privacy_masks else
                                    "whole picture, checked" if priv_checked else "not set")
                        + (", PROBLEM" if priv_problem else ""))
        acc.set_summary("position", f"{e.test.label}, {target}")
        acc.set_summary("checklist", f"{sum(cb.isChecked() for cb in self.camera_checks.values())}/{len(self.camera_checks)} ticked")
        acc.set_summary("dots", (f"{n_marked}/{len(names)} marked, {kind_name}" if running else f"camera not started, {kind_name}")
                        + (", locked" if e.locked else ""))
        acc.set_summary("ruler", f"set, {(e.scale.px_per_cm or 0.0):.1f} px/cm" if ruler_set else "no ruler today" if ruler_skipped else "not set")
        acc.set_summary("force", "connected, zero set" if zero_set else "connected, ZERO NEEDED" if src is not None
                        else "no force today" if force_skipped else "not connected")
        acc.set_summary("markers", ("automatic" if self.chk_auto_mvic.isChecked() else "by hand" if by_hand or src is None else "not decided")
                        + (", beeps off" if not self.chk_cues.isChecked() else ", beeps tested" if self._beeps_tested else ", beeps NOT tested"))
        acc.set_summary("record", "recording" if recording else "ready" if ready and id_ok else "not ready yet")
        if not self._steps_ready:                          # the first time: open the first thing that has to be done
            self._steps_ready = True
            self.open_step(first_todo or "record")
        else:                                              # a step was just finished while its own section was open: go on to the next
            for k in ("camera", "privacy", "dots", "ruler", "lock", "force"):      # (not the participant code, which is typed, nor the markers, which are tried out)
                if states[k][0] == "done" and self._step_prev.get(k) not in (None, "done") and acc.current() == STEP_SECTION[k]:
                    self.open_step(first_todo or "record")
                    break
        self._step_prev = {k: v[0] for k, v in states.items()}
        self.update_dot_rows()

    def _instruction_text(self):
        """What to do now, in a sentence: the dot being re-marked, else the first step that is still to do."""
        e = self.engine
        if self._rec_started is not None:
            return "Recording."
        if self._arm is not None and self.worker is not None:
            return f"Click the {e.test.dot_title(self._arm)} dot on the video to mark it again."
        step = self._first_todo
        if step == "dots" and self.worker is not None:
            nxt = e.next_dot_to_mark()
            if nxt:
                return f"Click the {e.test.dot_title(nxt)} dot on the video."
        if step == "lock":
            return "All dots marked. Set the participant's position, then lock it."
        if step is None:
            return "Position locked." if e.locked else "Ready."
        return self._hint(step)

    def _hint(self, step):
        if step == "participant" and self._participant_unconfirmed():
            return (f"{self.participant.text().strip()} was kept from the last session. Press Enter in the box if it is the same "
                    f"participant, or type the new code.")
        return STEP_HINT[step]

    def _participant_unconfirmed(self):
        """A code that came back from the last session and has not been confirmed since (a different code, typed or set, counts at once)."""
        return bool(self._participant_restored) and self.participant.text().strip() == self._participant_restored

    def _participant_entered(self):
        """Enter in the participant field: on to the next step, unless the code is not valid yet."""
        if privacy.check_participant_id(self.participant.text().strip()) is None:
            self._participant_restored = ""                # Enter confirms a code that was kept from last time
            self.update_steps()
            if self.accordion.current() == "participant":
                self.open_step(self._first_todo or "record")

    # ------------------------------------------------------------ remembering the set-up (see session_memory.py)
    def _remember(self, *_):
        if self._memory_path is None or self._restoring or self._save_timer is None:
            return
        self._save_timer.start()

    def session_state(self):
        e = self.engine
        last = self._last_source or self._remembered_source or {}
        win = self.window()
        g = win.geometry()
        return {
            "camera": dict(last),
            "autostart": self.chk_autostart.isChecked(),
            "test": self.test_combo.currentData(),
            "target": e.target,
            "tolerance": self.tol_spin.value(), "neighbour_tolerance": self.ntol_spin.value(),
            "dot_kind": self.kind_combo.currentData(), "facing_right": bool(self.facing_combo.currentData()),
            "mirror": self.chk_mirror.isChecked(), "smoothing": self.chk_smooth.isChecked(),
            "landmarks_disabled": sorted(e.disabled_dots),
            "trails": self.chk_trails.isChecked(), "overlay": self.chk_overlay.isChecked(),
            "stream": self.stream_box.text().strip(),
            "force": {"kind": self.force_kind.currentData(), "device": self.force_dev.currentText().strip(),
                      "channel": self.force_chan.currentText(), "wiring": self.force_term.currentData(),
                      "smoothing_s": self.force_smooth_combo.currentData(), "connect_at_start": self._force_wanted},
            "auto_mvic": {"on": self.chk_auto_mvic.isChecked() or self._auto_mvic_wanted, "threshold": self.auto_thr_spin.value(),
                          "length": self.auto_len_spin.value()},
            "cues": {"on": self.chk_cues.isChecked(), "push_every_s": self.cue_push_spin.value()},
            "auto_stop": {"on": self.chk_auto_stop.isChecked(), "after_s": self.auto_stop_spin.value()},
            "camera_model": self.camera_model.text().strip(),
            "ruler_cm": self.ruler_len.value(),
            "participant": self.participant.text().strip(),
            "ui": {"section": self.accordion.current(), "splitter": self.splitter.sizes(), "window": [g.x(), g.y(), g.width(), g.height()]},
        }

    def save_session(self):
        if self._memory_path is not None:
            session_memory.write_session(self._memory_path, self.session_state())

    @staticmethod
    def _pick(combo, data):
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def restore_session(self, s):
        """Put the remembered set-up back. Anything that no longer fits (a test or camera that is gone, a value out of range) is skipped."""
        self._restoring = True

        def part(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001 - one bad remembered value must never stop the program from starting
                pass
        try:
            part(lambda: self._pick(self.test_combo, s["test"]) if s.get("test") in self.engine.tests else None)
            part(lambda: self._pick(self.kind_combo, s["dot_kind"]))
            part(lambda: self._pick(self.facing_combo, bool(s["facing_right"])))
            part(lambda: self.chk_mirror.setChecked(bool(s["mirror"])))
            part(lambda: self.chk_smooth.setChecked(bool(s["smoothing"])))
            part(lambda: self.tol_spin.setValue(float(s["tolerance"])))
            part(lambda: self.ntol_spin.setValue(float(s["neighbour_tolerance"])))

            def landmarks():
                gone = set(s["landmarks_disabled"])
                for n in self.engine.test.all_dots:
                    self.engine.set_dot_enabled(n, n not in gone)
                self._sync_landmark_checks()
                self.update_landmark_info()
            part(landmarks)
            part(lambda: self.target_spin.setValue(self.target_spin.minimum() if s["target"] is None else float(s["target"])))
            part(lambda: self.chk_trails.setChecked(bool(s["trails"])))
            part(lambda: self.chk_overlay.setChecked(bool(s["overlay"])))
            part(lambda: self.stream_box.setText(str(s.get("stream", ""))))
            f = s.get("force") or {}
            part(lambda: self._pick(self.force_kind, f["kind"]))
            part(lambda: self.force_dev.setEditText(str(f["device"])) if f.get("device") else None)
            part(lambda: self.force_chan.setCurrentIndex(max(self.force_chan.findText(str(f["channel"])), 0)))
            part(lambda: self._pick(self.force_term, f["wiring"]))
            part(lambda: self._pick(self.force_smooth_combo, f["smoothing_s"]))
            self._force_wanted = bool(f.get("connect_at_start"))
            a = s.get("auto_mvic") or {}
            part(lambda: self.auto_thr_spin.setValue(float(a["threshold"])))
            part(lambda: self.auto_len_spin.setValue(float(a["length"])))
            self._auto_mvic_wanted = bool(a.get("on"))
            c = s.get("cues") or {}
            part(lambda: self.chk_cues.setChecked(bool(c["on"])))
            part(lambda: self.cue_push_spin.setValue(float(c["push_every_s"])))
            st = s.get("auto_stop") or {}
            part(lambda: self.auto_stop_spin.setValue(float(st["after_s"])))
            part(lambda: self.chk_auto_stop.setChecked(bool(st["on"])))
            part(lambda: self.camera_model.setText(str(s.get("camera_model", ""))))
            part(lambda: self.ruler_len.setValue(float(s["ruler_cm"])))
            part(lambda: self.participant.setText(session_memory.participant_if_recent(s)))
            self._participant_restored = self.participant.text().strip()               # kept from last time: confirmed again before it counts
            part(lambda: self.chk_autostart.setChecked(bool(s["autostart"])))
            self._remembered_source = s.get("camera") if isinstance(s.get("camera"), dict) else None
            ui = s.get("ui") or {}
            part(lambda: self.splitter.setSizes([int(x) for x in ui["splitter"]]) if len(ui.get("splitter", [])) == 2 else None)
            part(lambda: self._restore_window(ui["window"]))
            part(lambda: self.open_step(ui["section"]) if ui.get("section") in self.accordion.sections else None)
            self._steps_ready = bool(ui.get("section") in self.accordion.sections)
            if self._participant_unconfirmed():
                self.open_step("participant")                  # the first thing to see is the question: same participant as last time?
            self.instruction.setText(IDLE_TEXT)
        finally:
            self._restoring = False

    def _restore_window(self, geometry):
        x, y, w, h = (int(v) for v in geometry)
        rect = QRect(x, y, w, h)
        if w >= 900 and h >= 560 and any(sc.availableGeometry().intersects(rect) for sc in QGuiApplication.screens()):
            self.window().setGeometry(rect)

    def attach_memory(self, path):
        """Called by the window once everything exists: restore the remembered set-up, then keep it up to date as it changes."""
        self._memory_path = Path(path)
        s = session_memory.read_session(self._memory_path)
        if s:
            self.restore_session(s)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)                # saved a moment after the last change, so a hung or killed program has lost nothing
        self._save_timer.timeout.connect(self.save_session)
        for sig in (self.test_combo.currentIndexChanged, self.kind_combo.currentIndexChanged, self.facing_combo.currentIndexChanged,
                    self.target_spin.valueChanged, self.tol_spin.valueChanged, self.ntol_spin.valueChanged, self.chk_mirror.toggled,
                    self.chk_smooth.toggled, self.chk_trails.toggled, self.chk_overlay.toggled, self.chk_cues.toggled,
                    self.cue_push_spin.valueChanged, self.chk_auto_mvic.toggled, self.auto_thr_spin.valueChanged,
                    self.auto_len_spin.valueChanged, self.chk_auto_stop.toggled, self.auto_stop_spin.valueChanged,
                    self.force_kind.currentIndexChanged, self.force_dev.currentTextChanged,
                    self.force_chan.currentIndexChanged, self.force_term.currentIndexChanged, self.force_smooth_combo.currentIndexChanged,
                    self.cam_combo.currentIndexChanged, self.stream_box.editingFinished, self.chk_autostart.toggled,
                    self.ruler_len.valueChanged, self.participant.textChanged, self.camera_model.editingFinished,
                    self.accordion.opened):
            sig.connect(self._remember)
        self.splitter.splitterMoved.connect(self._remember)
        if s and session_memory.autostart_allowed():
            QTimer.singleShot(400, self._auto_start)

    def _auto_start(self):
        """Open the camera and the force the way they were last time (what was remembered; nothing else is started)."""
        if not self.chk_autostart.isChecked():
            return
        last = self._remembered_source or {}
        if self.worker is None and last.get("kind") == "webcam":
            name = last.get("name", "")
            i = next((k for k in range(self.cam_combo.count()) if self.cam_combo.itemText(k).split(": ", 1)[-1] == name), None)
            if i is None:
                self.say(f"The last camera ('{name}') was not found. Start it, press Scan cameras, then Start camera.", 12.0)
            else:
                self.cam_combo.setCurrentIndex(i)
                self.toggle_camera()
        elif self.worker is None and last.get("kind") == "stream" and last.get("address"):
            self.stream_box.setText(last["address"])
            self.connect_stream()
        if self._force_wanted and self.force_source is None and self.force_kind.currentData() != "off":
            self.connect_force()
            if self.force_source is not None and self._auto_mvic_wanted:
                self.chk_auto_mvic.setChecked(True)

    @staticmethod
    def _heading(text):
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; padding-top: 4px;")
        return label

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
        self._arm = None
        for row in self.lm_rows.values():
            self.lm_holder.removeWidget(row)
            row.deleteLater()
        self.lm_checks, self.lm_status, self.lm_buttons, self.lm_rows = {}, {}, {}, {}
        for n in t.all_dots:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(t.dot_title(n).split(" (")[0])                  # (the part in brackets would make the panel wider than the screen)
            cb.setToolTip(t.dot_title(n))
            cb.setChecked(n not in self.engine.disabled_dots)
            cb.toggled.connect(lambda on, name=n: self.on_landmark_toggled(name, on))
            state = QLabel("")
            state.setMinimumWidth(72)
            btn = QPushButton("Mark")
            btn.setCheckable(True)
            btn.setToolTip("Mark this dot again: press, then click the dot on the video. Only this dot changes.")
            btn.clicked.connect(lambda _=False, name=n: self.arm_dot(name))
            rl.addWidget(cb, 1)
            rl.addWidget(state)
            rl.addWidget(btn)
            self.lm_holder.addWidget(row)
            self.lm_checks[n], self.lm_status[n], self.lm_buttons[n], self.lm_rows[n] = cb, state, btn, row
        self.update_dot_rows()
        self.rebuild_three_point_choices()
        self.lm_preset.clear()
        self.lm_preset.addItem("Preset: all landmarks", list(t.dots))
        if t.optional_dots:
            self.lm_preset.addItem("Preset: all landmarks + strap and anchor", list(t.all_dots))
        for a in t.angles:
            self.lm_preset.addItem(f"Preset: only {a.title}", list(dict.fromkeys(a.dots)))
        self.update_landmark_info()

    DOT_STATE = {"ok": ("marked", "#16a34a"), "lost": ("LOST", "#d97706"), "unmarked": ("not marked", "#9ca3af"), "off": ("not used", "#9ca3af")}

    def update_dot_rows(self):
        """Per landmark: marked / LOST / not marked, and what its button does (Mark, Re-mark, or Cancel while it waits for a click)."""
        if not self.lm_rows:
            return
        status = self.engine.dot_status()
        free = self._rec_started is None and self.worker is not None and not self.engine.paused
        if self._arm is not None and status.get(self._arm, "off") == "off":
            self._arm = None                                  # switched off while waiting for a click
        for n in self.lm_rows:
            s = status.get(n, "off")
            text, colour = self.DOT_STATE[s]
            if self.lm_status[n].text() != text:
                self.lm_status[n].setText(text)
                self.lm_status[n].setStyleSheet(f"color: {colour}; font-weight: bold;")
            armed = self._arm == n
            want = "Cancel" if armed else "Re-mark" if s in ("ok", "lost") else "Mark"
            btn = self.lm_buttons[n]
            if btn.text() != want:
                btn.setText(want)
            if btn.isChecked() != armed:
                btn.setChecked(armed)
            btn.setEnabled(s != "off" and free)

    def arm_dot(self, name):
        """The button of one landmark: the next click on the picture marks that dot (again) and leaves every other dot alone.
        It stays waiting until the dot has been found, so a click that missed can simply be repeated."""
        btn = self.lm_buttons.get(name)
        if self.worker is None or self.engine.rec is not None or self.engine.paused:
            if btn is not None:
                btn.setChecked(False)
            self.say("Start the camera first (and stop recording or the break) to mark a dot." if self.worker is None
                     else "Dots cannot be marked during a recording or a break.")
            return
        self._arm = None if self._arm == name else name
        self.update_dot_rows()
        self.say(f"Click the {self.engine.test.dot_title(name)} dot on the video." if self._arm else "Cancelled: no dot will be changed.", 8.0)

    def update_kind_note(self, *_):
        kind = self.kind_combo.currentData()
        if kind == "auto":
            self.kind_note.setText("Auto can follow the white tape instead of the black ink without any warning. For black discs on white "
                                   "tape choose Black dots.")
            self.kind_note.setStyleSheet("color: #d97706; font-weight: bold;")
        else:
            self.kind_note.setText("Changing the colour clears the marked dots and the ruler: mark them again.")
            self.kind_note.setStyleSheet("")

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
        self.lm_info.setText("   ".join(lines))                       # (one paragraph: the list was a third of the section)
        self._remember()

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
        self._arm = None
        self.chk_priv_ok.setChecked(False)        # the picture is flipped: what was looked at is not what is saved now
        self.engine.clear_privacy()               # drawn on the old orientation
        self.engine.reset_dots()                  # marked positions belong to the old orientation
        self.clear_scale()
        self.btn_lock.setChecked(False)
        self.refresh_view()
        self.instruction.setText("Image " + ("mirrored" if on else "not mirrored") + ". Mark the dots again, and check "
                                 "'Participant faces' against what you now see.")

    def _set_auto_stop(self, *_):
        """Stop each recording by itself this long after the MVIC end marker (0 = the operator stops it). May change while recording."""
        self.engine.auto_stop_s = self.auto_stop_spin.value() if self.chk_auto_stop.isChecked() else 0.0
        self.auto_stop_spin.setEnabled(self.chk_auto_stop.isChecked())

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
        self._arm = None
        self.update_dot_rows()
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
        if isinstance(source, int):
            self._last_source = {"kind": "webcam", "name": self.cam_combo.currentText().split(": ", 1)[-1]}
        elif classify_source(source) == "stream":
            self._last_source = {"kind": "stream", "address": source}
        self._remember()
        self.chk_priv_ok.setChecked(False)                # 'I looked' was about the previous picture
        self.engine.reset_dots()
        self.clear_scale()
        self.btn_lock.setChecked(False)
        self.engine.warm_up_cues()                       # wake the sound device now, so the first beep of the session is on time
        self.note("video source started", Path(source).name if isinstance(source, str) and classify_source(source) == "file"
                  else self.cam_combo.currentText().split(": ", 1)[-1] if isinstance(source, int) else "stream")
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
    BREAK_TEXT = "Pause for a break  (P)"

    def _set_file_controls(self, on):
        """Video files: play / pause and frame stepping. A running camera or stream: the Pause button is a break (tracking stops and
        a ghost shows where the dots were). No source: nothing to pause."""
        live = self.worker is not None and not on
        for b in (self.btn_back, self.btn_fwd):
            b.setEnabled(on)
        self.btn_pause.setEnabled(on or live)
        if not on:
            self.btn_pause.setChecked(False)
            self.btn_pause.setText(self.BREAK_TEXT if live else "Pause  (P)")
            self.frame_label.setText("Pause is for a break: tracking stops and a ghost shows where the dots were. Frame stepping works on video files only."
                                     if live else "Pause and frame stepping work on video files only.")
            if not live:
                self.engine.cancel_pause()

    def _sync_pause_button(self):
        """After something else ended a break (Re-mark all dots), make the button say so."""
        if self.worker is not None and not self.worker.is_file and self.btn_pause.isChecked() != self.engine.paused:
            self.btn_pause.setChecked(self.engine.paused)
            self.btn_pause.setText("Resume  (P)" if self.engine.paused else self.BREAK_TEXT)

    def _toggle_break(self):
        """Pause or resume a live camera for a break."""
        if self.btn_pause.isChecked():
            why = self.engine.pause()
            if why:
                self.btn_pause.setChecked(False)
                QMessageBox.warning(self, "Cannot pause", why)
                return
            self.note("break started")
            self.btn_pause.setText("Resume  (P)")
            self.frame_label.setText("PAUSED for a break. When the participant is back, put the stickers on the dashed circles, then press Resume.")
        else:
            self.engine.resume()
            self.note("break ended")
            self.btn_pause.setText(self.BREAK_TEXT)
            self.frame_label.setText("Resumed: each dot is tracked again once it is back on its circle.")

    def toggle_pause(self):
        if self.worker is None:
            self.btn_pause.setChecked(False)
            return
        if not self.worker.is_file:
            self._toggle_break()
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
            self._force_wanted = False                  # the operator switched it off: do not open it again at the next start
        else:
            self.connect_force()
            self._force_wanted = self.force_source is not None
        self._remember()

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
        self.note("force zero refused" if why else "force zeroed", why or "")
        if why:
            self._force_msg = why
        else:
            cal = src.calibration
            moved = "" if before is None or not cal.has_slope else f" (moved {(cal.offset_v - before) * cal.n_per_v:+.1f} N from the last zero)"
            self._force_msg = f"Zero set: {cal.offset_v:+.4f} V is now 0 N{moved}." + self._zero_quality_note(cal)
        self.update_force_box()

    @staticmethod
    def _zero_quality_note(cal):
        """How noisy the zero was and how well it is known (noise averages out of a zero; hum cancels exactly in a whole-cycle window)."""
        if cal.zero_noise_v is None or cal.zero_se_v is None:
            return ""
        good = f"good to about {cal.zero_se_v * 1000:.1f} mV" + (f" (+-{cal.zero_se_n:.2f} N)" if cal.zero_se_n is not None else "")
        note = f" Noise {cal.zero_noise_v * 1000:.0f} mV, so the zero is {good}."
        if cal.zero_hum_v is not None and cal.zero_hum_v >= HUM_NOTE_V:
            note += (f" 50 Hz mains hum of {cal.zero_hum_v * 1000:.0f} mV is present: it cancels out of the zero, but it is still on the "
                     f"force signal, so check that the laptop charges from the isolated outlet.")
        return note

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
        if not connected and self.chk_auto_mvic.isChecked() and not recording:
            self.chk_auto_mvic.setChecked(False)                  # the automatic markers need the force
        for w in (self.chk_auto_mvic, self.auto_thr_spin, self.auto_len_spin):
            w.setEnabled(connected and not recording)
        trying = self._reh_timer is not None and self._reh_timer.isActive()
        self.btn_rehearse.setEnabled((connected and not recording and not src.error) or trying)
        if trying and (not connected or recording):
            self.engine.rehearsal_stop()                              # the force went away or a recording began: the try-out ends
            self._stop_rehearsal_timer("The try-out stopped (the force was disconnected or a recording started).")
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
        self.note("position chosen", f"{key} {tgt if tgt is not None else ''}".strip())
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
        self.log.folder = self.recordings_dir
        self.folder_label.setText(f"Saving to: {self.recordings_dir}")

    def say(self, text, seconds=4.0):
        """Show a message on the instruction line and keep it there (the status timer would overwrite it within 0.2 s)."""
        self.instruction.setText(text)
        self._say_until = time.monotonic() + seconds

    def on_click(self, x, y):
        if self.worker is None:
            return
        if self.engine.paused:
            self.say("Paused for a break: press Resume (P) before marking a dot.")
            return
        if self.engine.rec is not None:
            self.say("Stop recording to re-mark a dot. (Clicks on the video are ignored while recording.)")
            return
        if self._ruler_marking is not None:
            self.on_ruler_click(x, y)
            return
        armed = self._arm                                     # a landmark's own Re-mark button was pressed: the click is for that dot
        first_marking = None if armed else self.engine.next_dot_to_mark()
        name = armed or first_marking or self.engine.nearest_dot(x, y)
        if name is None:
            self.say("Click directly on a dot to re-mark it, press the Re-mark button of a landmark, or use 'Clear all dots and start again'.")
            return
        title = self.engine.test.dot_title(name)
        ok = self.engine.mark_dot(name, x, y)
        self.note("dot marked" if ok and first_marking else "dot re-marked" if ok else "dot click found no dot",
                  name + ("" if ok or not self.engine.mark_problem else f": {self.engine.mark_problem}"))
        if ok:
            if armed:
                self._arm = None
                self.say(f"Marked again: {title}. The other dots were not touched.")
            elif not first_marking:
                self.say(f"Re-marked: {title}.")
        elif self.engine.mark_problem:
            self.say(self.engine.mark_problem + (" Click again, or press its button to cancel." if armed else ""))
        else:
            self.say(f"No dot found at that spot for '{title}'. Click the centre of the dot (check colour setting and lighting)."
                     + (" Still waiting: click again, or press its button to cancel." if armed else ""))
        self.update_dot_rows()
        self.refresh_view()

    def note(self, event, detail=""):
        """One line in the session log; a log that cannot be written is said once (never silently lost)."""
        if not self.log.write(event, detail) and self.log.problem != getattr(self, "_log_problem_shown", None):
            self._log_problem_shown = self.log.problem
            self.say(self.log.problem, 10.0)

    def remark(self):
        self.note("all dots cleared")
        self._arm = None
        self.engine.reset_dots()
        self.btn_lock.setChecked(False)
        self._sync_pause_button()                       # marking again ends a break
        self.update_dot_rows()
        self.refresh_view()

    def toggle_lock(self):
        if self.btn_lock.isChecked():
            if not self.engine.lock_position():
                self.btn_lock.setChecked(False)
                self.note("lock refused", self.engine.lock_problem())
                QMessageBox.warning(self, "Cannot lock", self.engine.lock_problem())
            else:
                self.note("position locked")
        else:
            self.engine.unlock_position()
            self.note("position unlocked")
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

    def test_beeps(self):
        """Play every beep once, then say whether Windows could play them."""
        self.engine.test_cues()
        self._beeps_test_pending = True
        self.say("Playing the beeps. If you hear nothing: check the sound output and volume of this computer.", seconds=6.0)
        QTimer.singleShot(5300, self.report_beep_problem)

    def report_beep_problem(self):
        """Say, in plain English, when a beep could not be played (called after the test and after every recording)."""
        why = self.engine.cue_problem
        if why:
            self.say(f"The beeps could not be played ({why}). Check the sound output of this computer, then press 'Test the beeps' again.", seconds=20.0)
            self.cue_problem_shown = why
            self._beeps_test_pending = False
        elif getattr(self, "cue_problem_shown", None):
            self.say("The beeps played without error.")
            self.cue_problem_shown = None
        if self._beeps_test_pending and not why:
            self._beeps_tested = True                   # Windows took every beep without an error (whether you HEARD them is for you to say)
            self._beeps_test_pending = False

    # ------------------------------------------------------------ trying the automatic markers without a recording
    def toggle_rehearsal(self):
        """Start (or cancel) the try-out: push on the strap, the program calls the start, plays the beeps, says when the end would be marked."""
        if self._reh_timer is not None and self._reh_timer.isActive():
            self.engine.rehearsal_stop()
            self._stop_rehearsal_timer("Try-out cancelled.")
            return
        why = self.engine.rehearsal_start()
        if why:
            self._reh_say(why)
            return
        self._reh_say("Getting ready: collecting the rest level (keep still for a moment)...")
        self.btn_rehearse.setText("Cancel the try-out")
        if self._reh_timer is None:
            self._reh_timer = QTimer(self)
            self._reh_timer.setInterval(100)
            self._reh_timer.timeout.connect(self._rehearsal_tick)
        self._reh_timer.start()

    def _rehearsal_tick(self):
        state, text = self.engine.rehearsal_step()
        if state == "off":
            self._stop_rehearsal_timer(text or "Try-out stopped.")
            return
        self._reh_say(text)
        if state in ("done", "gave up"):
            self._stop_rehearsal_timer(text)
            if state == "done":
                QTimer.singleShot(1500, self.report_beep_problem)     # a beep that failed is said so, as after a recording

    def _reh_say(self, text):
        self.rehearse_label.setText(text)
        self.rehearse_label.setVisible(bool(text))

    def _stop_rehearsal_timer(self, text):
        if self._reh_timer is not None:
            self._reh_timer.stop()
        self._reh_say(text)
        self.btn_rehearse.setText("Try it without recording")

    def on_cues_toggled(self, on):
        try:
            self.engine.audio_cues = bool(on)
        except RuntimeError as e:
            self.chk_cues.blockSignals(True)
            self.chk_cues.setChecked(bool(self.engine.audio_cues))
            self.chk_cues.blockSignals(False)
            self.say(str(e))
        self.cue_push_spin.setEnabled(bool(self.engine.audio_cues))

    def on_auto_mvic_toggled(self, on):
        try:
            self.engine.auto_mvic = bool(on)
        except RuntimeError as e:
            self.chk_auto_mvic.blockSignals(True)
            self.chk_auto_mvic.setChecked(bool(self.engine.auto_mvic))
            self.chk_auto_mvic.blockSignals(False)
            self.say(str(e))

    def toggle_record(self, checked=None, force_stop=False):
        if self.btn_rec.isChecked() and not force_stop:
            if self.worker is None or not self.worker.isRunning():
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Cannot record", "No video is coming in, so nothing would be recorded. "
                                    "Start the camera or open a video file first.")
                return
            if self.engine.paused:
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Cannot record", "The program is paused for a break. Press Resume (P) first, then record.")
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
            if self._participant_unconfirmed():
                code = self.participant.text().strip()
                r = QMessageBox.question(self, "Participant", f"The participant code {code} was kept from the last session. Record this trial for {code}?")
                if r != QMessageBox.Yes:
                    self.btn_rec.setChecked(False)
                    return
                self._participant_restored = ""
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
            src = self.force_source
            if src is not None and not src.error and src.calibration.has_slope and not src.calibration.zero_set:
                # 4 of the 11 trials of 2026-10-15 were recorded like this: volts only, no newtons and no peak force
                r = QMessageBox.question(self, "Force not zeroed", "The force has not been zeroed since the program started, so this trial "
                                         "will have volts but no newtons (and no peak force in N). To zero: unload the strap, press Zero (Z), "
                                         "then record.\n\nRecord anyway, without newtons?")
                if r != QMessageBox.Yes:
                    self.btn_rec.setChecked(False)
                    self.open_step("force")
                    return
            try:
                self.engine.start_trial(self.participant.text().strip(), self.notes.text())
            except Exception as e:  # noqa: BLE001
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Cannot record", str(e))
                return
            self._rec_started = time.perf_counter()
            self.note("recording started", f"{self.participant.text().strip()} {self.engine.test.key} {self.engine.target if self.engine.target is not None else ''}".strip())
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
            due = self.engine.seconds_to_auto_stop()
            if self.engine.rec is not None:
                self.note("recording stopped", f"{n} frames, {duration:.1f} s, " + ("by itself after the MVIC end" if due is not None and due <= 0 else "by hand"))
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
            self.report_beep_problem()
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
        text = (f"Camera delivers {g['width']}x{g['height']}" + (f", {m:.1f} fps measured" if m else ", measuring frame rate...")
                + (f" ({cam.device_name})" if cam.device_name else ""))
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
        due = self.engine.seconds_to_auto_stop() if self._rec_started is not None else None
        if due is not None and due <= 0:
            self.toggle_record(force_stop=True)              # as if Stop had been pressed (the engine notes it in meta.json)
            self.say(f"Stopped by itself {self.engine.auto_stop_s:g} s after the MVIC end. Press R for the next trial.", 8.0)
            due = None
        if self._rec_started is not None:
            stalled = (self.engine.seconds_since_last_frame() > 0.5 and not (self.worker is not None and self.worker.paused))
            if stalled:
                self.rec_label.setText("NO VIDEO - nothing is being recorded")
                self.rec_label.setStyleSheet("color: #dc2626; font-weight: bold;")
            else:
                c = dict(self._marker_counts)
                a_state, a_text = self.engine.auto_mvic_status()
                c['start'] += a_state in ("running", "done")             # markers the automatic MVIC wrote count too
                c['end'] += a_state == "done"
                text = (f"RECORDING  {time.perf_counter() - self._rec_started:5.1f} s   "
                        f"(MVIC start {c['start']}, end {c['end']}, other {c['other']})")
                if a_text:
                    text += "\n" + a_text
                if due is not None:
                    text += f"\nStopping by itself in {max(due, 0.0):.1f} s"
                if time.monotonic() < self._marker_msg[1]:
                    text += "\n" + self._marker_msg[0]
                self.rec_label.setText(text)
                self.rec_label.setStyleSheet("")
        self.update_camera_info()
        self.update_scale_label()
        self.update_steps()
        if time.monotonic() < self._say_until:
            return
        if self.worker is None and self.instruction.text() not in (self._auto_text, IDLE_TEXT, ""):
            return                    # a message that was left when the camera ended: it stays until something replaces it
        self._auto_text = self._instruction_text()
        self.instruction.setText(self._auto_text)

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
        self.review.attach_memory(self.settings_path)           # the Review tab's divider positions (own part of settings.json)
        self.live.attach_memory(self.settings_path)             # restores the set-up of the last session (see session_memory.py); last, so that
                                                                # the tab knows which window it is in

    def on_folder_changed(self, path):
        self.review.dir = Path(path)
        session_memory.write_part(self.settings_path, recordings_dir=path)       # (the rest of the file is kept)

    def closeEvent(self, e):
        self.live.save_session()                                 # before anything is switched off: what was connected is remembered
        self.live.shutdown()
        super().closeEvent(e)
