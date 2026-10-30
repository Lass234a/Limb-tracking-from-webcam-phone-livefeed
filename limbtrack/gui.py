"""The window: a Live tab (camera, marking dots, lock, record) and a Review tab."""
import json
import os
from datetime import datetime
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
                               QSizePolicy, QSplitter, QTabWidget, QListWidget, QVBoxLayout, QWidget)

from . import privacy, reference, session_memory
from .session_log import SessionLog
from .panels import Accordion, ReminderBar, Section, StepStrip
from .camera import CameraSource, camera_names, classify_source, list_cameras, normalise_stream_address
from .engine import Engine
from .loadcell import GRAVITY, find_regression
from .force import HUM_NOTE_V, ForceCalibration, ForceError, NiDaqForce, SimulatedForce
from .protocol import plan_positions
from .recorder import count_trials, trial_metas
from .rest import REST_AFTER_SUBMAX_S, RestTimer
from .automvic import AUTO_STOP_AFTER_END_S, DEFAULT_LENGTH_S, DEFAULT_THRESHOLD_N
from .scale import MIN_DOT_DIAMETER_CM
from .review import ReviewTab, after_trial_check, fill_reference_choices, select_reference_choices

BIG = "font-size: 15pt; font-weight: bold;"
NOTE = "font-size: 11pt; font-weight: bold;"
IDLE_TEXT = "Enter the participant code, then pick the camera and press Start camera."
# The steps of a session, in the order they are done. The strip shows one chip each; the first one still to do is highlighted.
STEPS = (("camera", "Camera"), ("force", "Force"), ("markers", "Markers"), ("participant", "Participant"), ("position", "Position"),
         ("privacy", "Privacy"), ("ruler", "Ruler"), ("dots", "Dots"), ("lock", "Lock"), ("record", "Record"))
# (0.24.0: the equipment first, which can be done before the participant arrives; then the participant and what depends on the position.
# Privacy comes after the position because it is looked at again at every position change.)
STEP_SECTION = {"participant": "participant", "camera": "camera", "privacy": "privacy", "position": "position", "dots": "dots",
                "ruler": "ruler", "lock": "dots", "force": "force", "markers": "markers", "record": "record"}
STEP_HINT = {"participant": "Enter the participant code (P01, ...), the tested leg and which side of the image they face. Code only, no names.",
             "camera": "Pick the camera and press Start camera.",
             "privacy": "Look at the picture: tick that no face is visible (V), or draw a crop round the leg or a black mask.",
             "position": "Pick the position in the session plan (or the test and target).",
             "dots": "Check the dot colour, then click each dot on the video, in the order asked.",
             "ruler": "Hold the ruler at the joint, click its two dots (C), or tick that there is no ruler today.",
             "lock": "Put the participant in position, then press Lock position (Space).",
             "force": "Connect the force, then press Zero with no pull on the strap (Z), or tick that there is no force today.",
             "markers": "Switch the automatic markers and the beeps on, and try them (Test the beeps, Try it without recording).",
             "record": "Press Start recording (R)."}
# painted across the bottom of the live picture (screen only) while nobody has said that no face is visible at this position
PRIVACY_BANNER = "PRIVACY: look at the picture. Is any face visible? Press V if not, or draw a black mask."
PENS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]
# The dots as small buttons in the pinned bar: a short name each (the full title is the tooltip) and a style per state.
DOT_CHIP_NAMES = {"iliac_crest": "Crest", "trochanter": "Troch.", "epicondyle": "Epicond.", "fibular_head": "Fibula", "malleolus": "Malleol.",
                  "mt5_base": "MT5 base", "mt5_head": "MT5 head", "asis": "ASIS", "asis_opposite": "ASIS 2", "patella": "Patella",
                  "strap": "Strap", "anchor": "Anchor"}
DOT_CHIP_COLUMNS = 5
DOT_CHIP_STYLE = {"ok": "color: #4ade80; border: 1px solid #16a34a; background: rgba(22,163,74,0.12);",
                  "lost": "color: #fbbf24; border: 1px solid #d97706; background: rgba(217,119,6,0.18); font-weight: bold;",
                  "unmarked": "color: #9ca3af; border: 1px solid rgba(128,128,128,0.45); background: transparent;",
                  "armed": "color: #111; border: 2px solid #60a5fa; background: #60a5fa; font-weight: bold;"}
# while every dot is being marked again: a dot still to do has a dashed border
DOT_CHIP_PENDING = {k: v.replace(" solid ", " dashed ") for k, v in DOT_CHIP_STYLE.items()}


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


# 'Recording first' (Engine.recording_first, 0.21.0): while recording, the window does less, because its work shares Python with the
# tracking. Replaying a P11 trial through the window (2026-10-27, on battery): 25 ms of work per picture without the window, 52 ms with it.
SHOWN_EVERY_S_RECORDING = 0.1      # at most 10 pictures a second on the screen (every picture is still tracked and saved)
TRACE_MS, TRACE_MS_RECORDING = 200, 1000      # the trace: 5 redraws a second, once a second while recording
PANEL_TICKS_RECORDING = 5          # the status panel in full every 5th tick (once a second); the recording line, the automatic stop
                                   # and the 'no video' warning every tick
# (Windows priority above normal while recording was tried as part of it and left out: with 16 busy programs alongside, 26.8 pictures/s
# with it, 26.4 without.)


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
        self._banner = ""                  # a soft reminder painted across the bottom of the picture (screen only, never in a saved video)

    def set_banner(self, text):
        """A soft reminder across the bottom of the picture (screen only)."""
        if text != self._banner:
            self._banner = text
            self.update()

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
            if self._banner:
                font = p.font()
                font.setBold(True)
                font.setPointSize(max(font.pointSize(), 11))
                p.setFont(font)
                height = 2.4 * p.fontMetrics().height()
                bar = QRectF(rect.left(), rect.bottom() - height, rect.width(), height)      # (the top of the picture holds the test, the lock and the angles)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(251, 191, 36, 235)))
                p.drawRect(bar)
                p.setPen(QColor(17, 17, 17))
                p.drawText(bar.adjusted(10, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap, self._banner)

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
        self._shown_t = None     # picture time of the last picture sent to the screen

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
        held_back = (self.engine.rec is not None and self.engine.recording_first and self._shown_t is not None
                     and 0 <= t - self._shown_t < SHOWN_EVERY_S_RECORDING - 0.005)   # 'recording first': tracked and saved, not shown
        if not held_back:
            self._shown_t = t
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
    position_checked = Signal(object)        # the 10 % check of a position, worked out on a helper thread (see _check_position)

    def __init__(self, engine, recordings_dir):
        super().__init__()
        self.engine, self.recordings_dir = engine, Path(recordings_dir)
        self.worker = None
        self._rec_started = None
        self._trial_box = None               # the message about problems in the trial just saved (non-modal, see _show_trial_warnings)
        self._last_trial_warnings = []       # the problems of the trial just saved (the resting force and the 10 % check are added later)
        self._repeat = {}                    # (participant, test, target) -> (trials it was worked out for, analysis.position_check dict)
        self._repeat_running = set()         # positions whose 10 % check is being worked out
        self._repeat_announce = set()        # ... and of those, the ones whose result is said (a trial was just saved there)
        self.position_checked.connect(self._on_position_checked)
        self._say_until = 0.0
        self._combo_index = {"test": 0, "kind": 0, "facing": 0, "leg": 0}     # last accepted choice, restored if the engine refuses a change
        self._ref_box_position = None        # (test, target, participant) the reference boxes were filled for (see _fill_reference_boxes)
        self._ref_enabled = None             # whether the reference widgets are switched on (None: not yet set)
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
        self._pass = []                      # 'Redo all': the dots still to be marked again, in order (empty = no pass); _arm is one of them
        self._pass_total = 0                 # how many dots the pass began with
        self._priv_pos = (self.engine.test.key, self.engine.target)   # the position the privacy set-up was last looked at for (0.23.0)
        self._priv_recheck = False           # the position changed since: the picture must be looked at again before the next recording
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
        # frontal tests (hip abduction/adduction, camera in front of the participant): these replace 'Participant faces'
        # the tested (dominant) leg belongs to the participant (0.20.0): chosen in the Participant section, saved in every trial
        self.leg_combo = NoWheelComboBox()
        self.leg_combo.addItem("Choose the tested leg", None)
        self.leg_combo.addItem("Right leg", "right")
        self.leg_combo.addItem("Left leg", "left")
        self.leg_combo.setToolTip("The leg that is tested: the dominant leg (SOP 4.5, the leg the participant kicks a ball with). It belongs to the "
                                  "participant code: another code starts with no leg chosen, and Record asks for it. Saved in every trial. In the "
                                  "front-view tests it also checks that the two ASIS dots were not swapped (which would turn abduction into adduction): "
                                  "facing the camera, the right leg is on the left of the picture (the other way round when the image is mirrored).")
        self.leg_combo.currentIndexChanged.connect(self.on_leg_changed)
        self.tape_spin = self._spin(0.0, 500.0, 1.0, " mm")
        self.tape_spin.setSpecialValueText("not measured")
        self.tape_spin.setToolTip("Optional: the distance between the two ASIS measured with a tape or calipers. The program compares it with the "
                                  "distance in the picture (ruler scale) and warns when they differ by more than 8 %: that would mean the ruler is "
                                  "not at the pelvis' distance, the pelvis is turned, or an ASIS dot is misplaced. It does not change any angle.")
        self.tape_spin.valueChanged.connect(lambda v: self._set_setting("tape_pelvic_width_mm", self.tape_spin, v))
        self.chk_wall = QCheckBox("Hip against the bench, knee straight")
        self.chk_wall.setToolTip("How the sagittal posture (hip flexion 0, straight knee) is kept in these tests (SOP: standing with the hip against "
                                 "the treatment bench). It is not measured; this tick is saved with the trial.")
        self.chk_wall.toggled.connect(self.on_wall_toggled)
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
        f.addRow("ASIS to ASIS, tape", self.tape_spin)
        f.addRow(self.chk_wall)
        f.addRow(self.chk_smooth)
        self.settings_form = f
        self._update_view_rows()

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
        self.chip_grid = QGridLayout()                   # the same dots as small buttons in the pinned bar (built in rebuild_landmarks)
        self.chip_grid.setSpacing(3)
        self.chip_grid.setContentsMargins(0, 0, 0, 0)
        self.chips = {}
        self.lm_info = QLabel()
        self.lm_info.setWordWrap(True)
        lm_hint = QLabel("To redo ONE dot, press its Re-mark button (or its number key): its old mark is removed at once, then click that dot "
                         "on the video. The other dots stay as they are.")
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
                               "or use F5 record, F6 MVIC start, F7 MVIC end, F8 marker, F9 lock, F4 redo dots, F2/F3 position.")
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
        self.chk_rec_first = QCheckBox("Recording first: fewer pictures on screen")
        self.chk_rec_first.setChecked(True)
        self.chk_rec_first.setToolTip("While recording, the screen shows at most 10 pictures a second and the trace and this panel are refreshed "
                                      "once a second. Every picture is still tracked and saved, so more pictures per second are saved on a slow or "
                                      "busy laptop (P11 replay on battery: 23 -> 28 per second; with other programs busy: 13 -> 27). Nothing changes "
                                      "before or after a recording.")
        self.chk_rec_first.toggled.connect(self._set_recording_first)
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
        self.chk_rest_beep = QCheckBox("Beep when the rest is over")
        self.chk_rest_beep.setChecked(True)
        self.chk_rest_beep.setToolTip("Three short beeps when the rest of the SOP is over (60 s between trials at one position, 120 s before the next "
                                      "position and in the belt squat, 30 s after the submaximal contraction). The rest is counted from the moment a "
                                      "recording stops and shown under the Record button. Independent of 'Sound cues'.")
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
        pf.addRow("Tested leg", self.leg_combo)
        pf.addRow("Participant faces", self.facing_combo)        # (0.24.0: with the participant, not with the position)
        pf.addRow("Notes", self.notes)
        self.participant_form = pf
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
        mf.addRow(self.chk_rest_beep)
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
        self.chk_priv_ok.toggled.connect(lambda on: on and self.confirm_privacy("looked: nothing identifies the participant"))
        self.btn_priv_again = QPushButton("Privacy checked at this position")
        self.btn_priv_again.setToolTip("Shown after every change of position. The participant has moved, so look at the picture again: is "
                                       "the crop still round the leg, and does nothing identify the participant? Press this (V), or draw the "
                                       "crop or a mask again, or tick the box above. A reminder only: since 0.24.0 it does not stop Record, "
                                       "but each trial says whether the picture was looked at for its position.")
        self.btn_priv_again.setStyleSheet("font-weight: bold; color: #d97706;")
        self.btn_priv_again.clicked.connect(self.confirm_privacy_here)
        self.btn_priv_again.setVisible(False)
        pvl.addWidget(self.btn_priv_again)
        pvl.addWidget(self.chk_overlay)
        pvl.addWidget(self.chk_rec_first)
        pvl.addWidget(self.folder_label)
        pvl.addWidget(self.btn_folder)
        self.update_privacy_label()

        # --- reference reading: what a goniometer or an inclinometer said, typed in next to the app's own angle (0.16.0)
        self.ref_instrument, self.ref_resolution = NoWheelComboBox(), NoWheelComboBox()
        fill_reference_choices(self.ref_instrument, self.ref_resolution)
        self.ref_instrument.setToolTip("What the readings were taken with. Saved with the trial.")
        self.ref_resolution.setToolTip("What the instrument can be read to (a goniometer: about 1 deg, a digital inclinometer: 0.1 deg).")
        self.ref_instrument.currentIndexChanged.connect(self._reference_edited)
        self.ref_resolution.currentIndexChanged.connect(self._reference_edited)
        self.ref_fields, self.ref_titles, self._ref_problem = {}, {}, ""
        self.ref_form = QFormLayout()
        self.btn_ref_stamp = QPushButton("Reading taken now  (G)")
        self.btn_ref_stamp.setToolTip("Press it at the moment you read the instrument. The program keeps its own angles of that moment "
                                      "(unsmoothed, averaged over 0.2 s), so the reading is compared with the app at the same moment, "
                                      "not with the hold of the contraction.")
        self.btn_ref_stamp.clicked.connect(self.take_reference_now)
        self.btn_ref_clear = QPushButton("Clear the readings")
        self.btn_ref_clear.clicked.connect(self.clear_reference)
        self.ref_status = QLabel("")
        self.ref_status.setWordWrap(True)
        ref_hint = QLabel("Put the instrument on the participant in the position, press 'Reading taken now' as you read it, then type what "
                          "it said, in degrees. Several readings of one angle: separate them with a space. An empty box means 'not "
                          "measured'. The readings stay for the same position (the second trial) and are never saved with another one.")
        ref_hint.setWordWrap(True)
        rr = QWidget()
        rrl = QVBoxLayout(rr)
        rrl.setContentsMargins(0, 0, 0, 0)
        rrl.addWidget(ref_hint)
        rif = QFormLayout()
        rif.addRow("Instrument", self.ref_instrument)
        rif.addRow("Read to", self.ref_resolution)
        rrl.addLayout(rif)
        rrl.addLayout(self.ref_form)
        rrow = QHBoxLayout()
        rrow.addWidget(self.btn_ref_stamp)
        rrow.addWidget(self.btn_ref_clear)
        rrl.addLayout(rrow)
        rrl.addWidget(self.ref_status)
        self.rebuild_reference_fields()

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
        self.btn_ruler_fine = QPushButton("Ruler still right at this position")
        self.btn_ruler_fine.setToolTip("Shown after every change of position. Press it when the ruler and the camera have not moved: the "
                                       "scale is still right. (Not needed to record; it is a reminder.)")
        self.btn_ruler_fine.setStyleSheet("font-weight: bold; color: #d97706;")
        self.btn_ruler_fine.clicked.connect(self.ruler_is_fine)
        self.btn_ruler_fine.setVisible(False)
        sf.addRow(self.btn_ruler_fine)
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
        self.trace_timer.start(TRACE_MS)         # 5 redraws a second: each costs the window thread, and with it the picture thread
        self._panel_tick = 0                     # 'recording first': ticks of the status panel since its last full refresh

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
        self.btn_force_relax = QPushButton("Zero 2: relaxed in the strap  (X)")
        self.btn_force_relax.setToolTip("The second zero level (0.22.0). First Zero (Z) with the strap hanging free. Then the participant gets into the "
                                        "position, relaxed, the strap taut, not pushing: press this. It takes 2 s of signal as the tension that is "
                                        "there before the push, for THIS participant, test and target (take it again for the next position). "
                                        "Newtons stay the total force from the first zero; the read-out and Review also give the force ABOVE the "
                                        "relaxed level. Refused while the signal still drifts, so wait until the tension has settled. A new Zero "
                                        "clears it.")
        self.btn_force_relax.clicked.connect(self.relax_force)
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
        fcf.addRow(self.btn_force_relax)
        fcf.addRow(self.btn_force_cal)
        fcf.addRow(self.force_info)
        self.chk_force_skip = QCheckBox("No force today (angles only)")
        self.chk_force_skip.setToolTip("Only tells the step list that you have decided. Not remembered: it is asked again every session.")
        fcf.addRow(self.chk_force_skip)

        # --- the pinned bar: the position and the dots, in view whichever section is open (a change of position or a re-marking is
        # needed all the time during a session, and the sections they live in close each other)
        self._plan_current = None                        # the row of the session plan the window is at (None: not a plan position)
        self.pos_combo = NoWheelComboBox()
        self.pos_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.pos_combo.setMinimumContentsLength(10)
        self.pos_combo.setPlaceholderText("not a plan position")
        self.pos_combo.activated.connect(self.go_to_position)
        self.btn_pos_prev = QPushButton("◀")
        self.btn_pos_prev.setToolTip("Previous position of the plan  (B or F2)")
        self.btn_pos_prev.clicked.connect(lambda: self.step_position(-1))
        self.btn_pos_next = QPushButton("▶")
        self.btn_pos_next.setToolTip("Next position of the plan  (N or F3)")
        self.btn_pos_next.clicked.connect(lambda: self.step_position(+1))
        for b in (self.btn_pos_prev, self.btn_pos_next):
            b.setFocusPolicy(Qt.NoFocus)                 # (a click must not take the keyboard from the letter keys)
            b.setFixedWidth(34)
        self.pos_combo.setFocusPolicy(Qt.NoFocus)
        self.btn_redo_dots = QPushButton("Redo all (D)")              # the last cell of the dot buttons (rebuild_dot_chips)
        self.btn_redo_dots.setToolTip("Redo all dots: every marked dot and the lock are removed at once, then click the dots on the video "
                                      "one after the other, in the order asked. Pressing it again (now Stop) ends the asking; the dots "
                                      "not marked yet stay unmarked.\nA dot's own button (or number key) skips it while this runs, and "
                                      "when it does not run it removes that one dot and waits for your click.")
        self.btn_redo_dots.setFocusPolicy(Qt.NoFocus)
        self.btn_redo_dots.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.btn_redo_dots.setMinimumWidth(40)
        self.btn_redo_dots.setStyleSheet("font-weight: bold;")
        self.btn_redo_dots.clicked.connect(self.toggle_mark_again)
        quick_bar = QFrame()                                          # (kept low: every pixel of it is taken from the open section)
        quick_bar.setFrameShape(QFrame.StyledPanel)
        qb = QVBoxLayout(quick_bar)
        qb.setContentsMargins(6, 4, 6, 4)
        qb.setSpacing(3)
        pos_label = QLabel("Position")
        pos_label.setStyleSheet("font-weight: bold;")
        prow = QHBoxLayout()
        prow.setSpacing(3)
        prow.addWidget(pos_label)
        prow.addWidget(self.btn_pos_prev)
        prow.addWidget(self.pos_combo, 1)
        prow.addWidget(self.btn_pos_next)
        self.btn_ruler_pin = QPushButton("Ruler (C)")                 # the ruler is redone often: one press from any section
        self.btn_ruler_pin.setCheckable(True)                         # (checked while it waits for the two clicks)
        self.btn_ruler_pin.setFocusPolicy(Qt.NoFocus)
        self.btn_ruler_pin.setToolTip("Mark the ruler again: the old scale is removed at once, then click the ruler dot at each end of the "
                                      "ruler on the video. Press again while it waits to stop.  (C)")
        self.btn_ruler_pin.clicked.connect(self.redo_ruler)
        prow.addWidget(self.btn_ruler_pin)
        qb.addLayout(prow)
        qb.addLayout(self.chip_grid)

        # --- soft reminders (0.24.0): shown under the instruction when something should be looked at again (a position change, the
        # resting force). Each row has the buttons that answer it. None of them ever stops Record.
        self._ruler_recheck = False                      # the position changed since the ruler was last looked at
        self._relax_dismissed = ""                       # the position for which 'Not here' was pressed on the resting-force reminder
        self._soft_for_file = []                         # reminders open at Record that change what the trial's file holds (said again once it is saved)
        self.reminders = ReminderBar()
        self.btn_rem_ruler_fine = QPushButton("Fine")
        self.btn_rem_ruler_fine.setToolTip("The ruler and the camera have not moved: the scale is still right at this position.")
        self.btn_rem_ruler_fine.clicked.connect(self.ruler_is_fine)
        self.btn_rem_ruler_redo = QPushButton("Redo (C)")
        self.btn_rem_ruler_redo.clicked.connect(self.redo_ruler)
        self.btn_rem_force = QPushButton("Zero 2 (X)")
        self.btn_rem_force.setToolTip("Take the resting force for this position: the participant relaxed in the strap, the strap taut, not pushing.")
        self.btn_rem_force.clicked.connect(lambda: self.btn_force_relax.click())
        self.btn_rem_force_not = QPushButton("Not here")
        self.btn_rem_force_not.setToolTip("No resting force at this position. The reminder stays away until the position changes.")
        self.btn_rem_force_not.clicked.connect(self.relaxed_not_here)
        self.btn_rem_priv_ok = QPushButton("No face (V)")
        self.btn_rem_priv_ok.setToolTip("I looked at the picture: no face or anything else that identifies the participant is visible "
                                        "(or the crop and masks are still right).")
        self.btn_rem_priv_ok.clicked.connect(self.confirm_privacy_here)
        self.btn_rem_priv_mask = QPushButton("Mask")
        self.btn_rem_priv_mask.setToolTip("Drag a rectangle over a face (or anything else) on the video: it is black in the saved videos.")
        self.btn_rem_priv_mask.clicked.connect(lambda: self.start_privacy_drawing("mask"))
        self.btn_rem_priv_crop = QPushButton("Crop")
        self.btn_rem_priv_crop.setToolTip("Drag a rectangle round the leg on the video: only that part is saved.")
        self.btn_rem_priv_crop.clicked.connect(lambda: self.start_privacy_drawing("crop"))
        self.reminders.add_row("privacy", [self.btn_rem_priv_ok, self.btn_rem_priv_mask, self.btn_rem_priv_crop])
        self.reminders.add_row("ruler", [self.btn_rem_ruler_fine, self.btn_rem_ruler_redo])
        self.reminders.add_row("force", [self.btn_rem_force, self.btn_rem_force_not])
        self.quick_bar = quick_bar

        # --- session plan: every position of the protocol with the trials already saved for this participant
        self.plan = QListWidget()
        self.plan.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plan.setTextElideMode(Qt.ElideRight)
        self.plan_rows = plan_positions(engine.tests)     # row i is position i + 1: the number in the SOP's test order and the file names
        for _ in self.plan_rows:
            self.plan.addItem("")
        self.plan.setFixedHeight(min(len(self.plan_rows), 8) * (self.plan.fontMetrics().height() + 4) + 8)     # 8 rows, then scrolls
        self.plan.itemClicked.connect(self.on_plan_clicked)
        self.participant.textChanged.connect(self.refresh_plan)
        self.participant.textChanged.connect(self._on_participant_text)
        self.target_spin.valueChanged.connect(lambda _: self.refresh_plan())
        self.target_spin.valueChanged.connect(lambda _: self.update_reference_box())    # the target is part of the readings' position
        self.target_spin.valueChanged.connect(self._position_changed)
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
        # --- the rest between contractions (0.27.0, SOP 10.5/10.8/11.7/9.11): counted from the end of each recording, shown here
        self.rest = RestTimer()
        self.rest_label = QLabel("")
        self.rest_label.setWordWrap(True)
        self.rest_label.setVisible(False)                           # only while a rest is timed and nothing is recorded
        self.btn_rest_submax = QPushButton(f"Submax rest {REST_AFTER_SUBMAX_S:g} s  (T)")
        self.btn_rest_submax.setFocusPolicy(Qt.NoFocus)
        self.btn_rest_submax.setToolTip("After the submaximal contraction (SOP 9.11, not recorded): start the 30 s rest before the first MVIC "
                                        "trial. After a trial the rest starts by itself.")
        self.btn_rest_submax.clicked.connect(self.start_submax_rest)
        quick.addWidget(self.rest_label, 2, 0)
        quick.addWidget(self.btn_rest_submax, 2, 1)
        head = QVBoxLayout()
        head.setSpacing(6)
        head.addWidget(self.strip)
        head.addWidget(self.instruction)
        head.addWidget(self.reminders)
        head.addLayout(quick)
        head.addWidget(self.quick_bar)
        self.accordion = Accordion()
        for number, key, title, widgets in (("1", "camera", "Camera", [src]),
                                            ("2", "force", "Force", [fc]),
                                            ("3", "markers", "Markers and sound", [mkb]),
                                            ("4", "participant", "Participant", [pc]),
                                            ("5", "position", "Position and angles", [plan_box, tg]),
                                            ("6", "privacy", "Privacy and saved files", [pv]),
                                            ("7", "ruler", "Ruler for centimetres", [sc]),
                                            ("8", "dots", "Dots, landmarks and lock", [mk]),
                                            ("9", "record", "Record", [rc]),
                                            (None, "reference", "Reference reading (goniometer, inclinometer)", [rr]),
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
        QShortcut(QKeySequence(Qt.Key_X), self, activated=self.btn_force_relax.click)
        QShortcut(QKeySequence(Qt.Key_G), self, activated=self.btn_ref_stamp.click)
        QShortcut(QKeySequence(Qt.Key_D), self, activated=self.btn_redo_dots.click)
        QShortcut(QKeySequence(Qt.Key_C), self, activated=self.btn_ruler_pin.click)
        QShortcut(QKeySequence(Qt.Key_T), self, activated=self.btn_rest_submax.click)
        QShortcut(QKeySequence(Qt.Key_V), self, activated=self.confirm_privacy_if_open)
        QShortcut(QKeySequence(Qt.Key_N), self, activated=self.btn_pos_next.click)
        QShortcut(QKeySequence(Qt.Key_B), self, activated=self.btn_pos_prev.click)
        for number in range(1, 10):                       # 1..9: redo the dot with that number in the pinned bar
            QShortcut(QKeySequence(Qt.Key_0 + number), self, activated=lambda number=number: self.redo_dot_number(number))
        # The F-keys work even while a text box has focus (letters would be typed into it): audit GUI-13.
        QShortcut(QKeySequence(Qt.Key_F2), self, activated=self.btn_pos_prev.click)
        QShortcut(QKeySequence(Qt.Key_F3), self, activated=self.btn_pos_next.click)
        QShortcut(QKeySequence(Qt.Key_F4), self, activated=self.btn_redo_dots.click)
        QShortcut(QKeySequence(Qt.Key_F5), self, activated=self.btn_rec.click)
        QShortcut(QKeySequence(Qt.Key_F6), self, activated=self.btn_start_mark.click)
        QShortcut(QKeySequence(Qt.Key_F7), self, activated=self.btn_end_mark.click)
        QShortcut(QKeySequence(Qt.Key_F8), self, activated=self.btn_mark.click)
        QShortcut(QKeySequence(Qt.Key_F9), self, activated=self.btn_lock.click)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.btn_back.click)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.btn_fwd.click)
        # A held key repeats; that is wanted only for stepping through a video (Left / Right). Every other key acts once per press: in P20
        # (2026-10-22) a held Z refused the zero 45 times in 9 s, and a held R would start and stop recordings (0.19.0).
        for sc in self.findChildren(QShortcut):
            if sc.key() not in (QKeySequence(Qt.Key_Left), QKeySequence(Qt.Key_Right)):
                sc.setAutoRepeat(False)
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
        # The soft reminders (0.24.0): chips with the state 'check', and a row each under the instruction line. They are shown while
        # nothing is recording, and none of them is ever 'the next step' or a reason to refuse Record.
        if not ruler_set:
            self._ruler_recheck = False                    # nothing left to look at again
        rows = {}
        if not recording and running:
            if self._priv_recheck and not priv_problem:
                rows["privacy"] = "Privacy: the position changed. Any face visible?"
            elif not (shaped or priv_checked) and not priv_problem:
                rows["privacy"] = "Privacy: is any face visible?"
            if self._ruler_recheck and ruler_set:
                rows["ruler"] = "Ruler: still right at this position?"
            if (src is not None and zero_set and ready and id_ok and e.relaxed_here() is None
                    and self._relax_dismissed != e.relaxed_key()):
                rows["force"] = "Resting force: Zero 2 not taken here"
        if src is not None:
            force_state = (("check", "Force: Zero 2?") if "force" in rows else ("done", "Force zeroed")) if zero_set else ("todo", "Force: zero")
        elif force_skipped:
            force_state = ("done", "No force")
        else:
            force_state = ("todo", "Force" if kind == "off" else "Force: connect")
        states = {
            "participant": (("done", pid) if id_ok and e.tested_leg else ("todo", f"{pid}: leg?") if id_ok
                            else ("todo", f"{pid}?") if unconfirmed else ("todo", "Participant")),
            "camera": ("done" if running else "todo", "Camera"),
            "privacy": (("off", "Privacy") if not running else ("todo", "Privacy: fix") if priv_problem
                        else ("check", "Privacy: recheck") if self._priv_recheck
                        else ("done", "Privacy") if shaped or priv_checked else ("todo", "Privacy")),
            "position": ("done", f"Position {target}" if e.target is not None else "Position"),
            "dots": ("off" if not running else "done" if ready else "todo", f"Dots {n_marked}/{len(names)}"),
            "ruler": (("off", "Ruler") if not running else ("check", "Ruler: check") if "ruler" in rows
                      else ("done", "Ruler") if ruler_set else ("done", "No ruler") if ruler_skipped else ("todo", "Ruler")),
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
        for key in self.reminders.rows:
            self.reminders.set_row(key, rows.get(key, ""))
        ruler_due = "ruler" in rows
        if self.btn_ruler_fine.isHidden() == ruler_due:
            self.btn_ruler_fine.setVisible(ruler_due)
        pin_style = "font-weight: bold; color: #d97706;" if ruler_due else ""
        if self.btn_ruler_pin.styleSheet() != pin_style:
            self.btn_ruler_pin.setStyleSheet(pin_style)
        self.view.set_banner(PRIVACY_BANNER if "privacy" in rows else "")      # on the picture itself: screen only, never in a saved video
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
                        + (", PROBLEM" if priv_problem else ", CHECK AGAIN (new position)" if self._priv_recheck else ""))
        self.btn_priv_again.setVisible(self._priv_recheck)
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
            if self._pass:
                k = self._pass_total - len(self._pass) + 1
                return (f"Mark again ({k} of {self._pass_total}): click the {e.test.dot_title(self._arm)} dot. "
                        f"Its button skips it, D stops.")
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
        if step == "participant" and privacy.check_participant_id(self.participant.text().strip()) is None and not self.engine.tested_leg:
            return f"Choose the tested (dominant) leg of {self.participant.text().strip()} in the Participant section."
        if step == "privacy" and self._priv_recheck:
            return ("The position changed: look at the picture again. Is any face visible, and is the crop still round the leg? Press V "
                    "(or 'Privacy checked at this position') if all is well, or draw a mask or the crop again.")
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
            "recording_first": self.chk_rec_first.isChecked(),
            "stream": self.stream_box.text().strip(),
            "force": {"kind": self.force_kind.currentData(), "device": self.force_dev.currentText().strip(),
                      "channel": self.force_chan.currentText(), "wiring": self.force_term.currentData(),
                      "smoothing_s": self.force_smooth_combo.currentData(), "connect_at_start": self._force_wanted},
            "auto_mvic": {"on": self.chk_auto_mvic.isChecked() or self._auto_mvic_wanted, "threshold": self.auto_thr_spin.value(),
                          "length": self.auto_len_spin.value()},
            "cues": {"on": self.chk_cues.isChecked(), "push_every_s": self.cue_push_spin.value()},
            "auto_stop": {"on": self.chk_auto_stop.isChecked(), "after_s": self.auto_stop_spin.value()},
            "rest": {"beep": self.chk_rest_beep.isChecked()},
            "camera_model": self.camera_model.text().strip(),
            "ruler_cm": self.ruler_len.value(),
            "participant": self.participant.text().strip(),
            "tested_legs": self.engine.tested_legs,
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
            # the tested leg of each participant (0.20.0), only from a recent session, like the participant code; the single
            # 'tested_leg' of older files belonged to nobody in particular and is not used
            part(lambda: setattr(self.engine, "tested_legs", s.get("tested_legs") or {}) if session_memory.participant_if_recent(s) else None)
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
            part(lambda: self.chk_rec_first.setChecked(bool(s["recording_first"])))      # (a session from before 0.21.0: stays on)
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
            part(lambda: self.chk_rest_beep.setChecked(bool((s.get("rest") or {})["beep"])))
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
                    self.leg_combo.currentIndexChanged,
                    self.target_spin.valueChanged, self.tol_spin.valueChanged, self.ntol_spin.valueChanged, self.chk_mirror.toggled,
                    self.chk_smooth.toggled, self.chk_trails.toggled, self.chk_overlay.toggled, self.chk_rec_first.toggled, self.chk_cues.toggled,
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
        self._update_view_rows()
        self.rebuild_landmarks()
        self.rebuild_reference_fields()
        self.refresh_plan()
        self._position_changed()                                # (the target box was set with its signals blocked)

    # ------------------------------------------------- reference readings (a hand instrument, typed in)
    def rebuild_reference_fields(self):
        """One box per angle of the test (and per extra the test offers, such as the typed hip flexion of a frontal test)."""
        while self.ref_form.rowCount():
            self.ref_form.removeRow(0)
        keys = self.engine.reference_keys()
        self.ref_fields, self.ref_titles = {}, dict(keys)
        for key, title in keys:
            box = TextField()
            box.setPlaceholderText("e.g. 88 90 89")
            box.setToolTip("Degrees, as the instrument says them. Several readings: separate with a space. Empty = not measured.")
            box.focus_changed.connect(self._update_key_hint)
            box.editingFinished.connect(self._reference_edited)
            self.ref_form.addRow(title, box)
            self.ref_fields[key] = box
        self._ref_enabled = None                               # new boxes: switch them on or off at the next update
        self._fill_reference_boxes()
        self.update_reference_box()

    def _fill_reference_boxes(self):
        """Show in the boxes what the engine holds for the position the window is at now (test, target, participant), and nothing when
        the readings belong to another one: a box must never show a reading that the next edit would save for the wrong position."""
        e = self.engine
        r = e.reference
        mine = e.reference_for_this_position()
        for key, box in self.ref_fields.items():
            box.setText(reference.format_values(r.values.get(key, [])) if mine else "")
            box.setStyleSheet("")
        if mine:
            select_reference_choices(self.ref_instrument, self.ref_resolution, r.instrument, r.resolution_deg)
        self._ref_problem = ""
        self._ref_box_position = e.reference_position()

    def _reference_edited(self, *_):
        """What was typed (or the instrument) changed: hand it to the engine. A box that cannot be read is marked and keeps the old value.
        Nothing is sent when nothing changed (a box that lets go of the focus as a recording starts must not be refused), and never what
        the boxes show for another position or participant (they are refilled instead)."""
        e = self.engine
        if self._ref_box_position != e.reference_position():
            self._fill_reference_boxes()
            self.update_reference_box()
            self.say("The position or the participant changed: the reference boxes now show what belongs to it. Type the reading again.")
            return
        values, problems = {}, []
        for key, box in self.ref_fields.items():
            try:
                values[key] = reference.parse_values(box.text())
                box.setStyleSheet("")
            except reference.ReadingError as err:
                box.setStyleSheet("border: 1px solid #dc2626;")
                problems.append(f"{self.ref_titles.get(key, key)}: {err}")
        r = e.reference
        held = r.values if e.reference_for_this_position() else {}
        instrument, resolution = self.ref_instrument.currentData(), self.ref_resolution.currentData()
        if any(v != held.get(k, []) for k, v in values.items()) or (instrument, resolution) != (r.instrument, r.resolution_deg):
            try:
                e.set_reference(values=values, instrument=instrument, resolution_deg=resolution)
            except (RuntimeError, ValueError) as err:
                self.say(str(err))
        self._ref_problem = "\n".join(problems)
        self.update_reference_box()

    def _commit_reference_boxes(self):
        """Before a recording starts: hand what is typed in the boxes to the engine. F5 starts a recording while a box still has the focus,
        before the box has let go of its text, so a reading shown only in the box would be missing from the trial. Returns a reason not to
        record, or ''."""
        self._reference_edited()
        if self._ref_problem:
            return ("A reference reading cannot be read, so it would not be saved with this trial:\n" + self._ref_problem
                    + "\n\nCorrect it, or empty the box (empty = not measured), then record.")
        return ""

    def _on_participant_text(self, text):
        """Reference readings belong to one participant: the engine is told the code, and the boxes show only that participant's readings.
        The tape measure of the pelvis and the wall tick belong to one participant too."""
        self.engine.participant_code = text.strip()
        self.update_reference_box()
        self._show_cleared_pelvis_facts()
        self._show_leg()

    def _show_leg(self):
        """The leg box shows the leg chosen for the participant code in the window (none for a new code; 0.20.0)."""
        i = max(self.leg_combo.findData(self.engine.tested_leg), 0)
        if i != self.leg_combo.currentIndex():
            self.leg_combo.blockSignals(True)
            self.leg_combo.setCurrentIndex(i)
            self.leg_combo.blockSignals(False)
            self._combo_index["leg"] = i
        self.leg_combo.setStyleSheet("" if self.engine.tested_leg else "color: #d97706; font-weight: bold;")

    def _show_cleared_pelvis_facts(self):
        """The engine clears the tape measure of the pelvis and the wall tick when the participant changes (after a recording, if the
        code was changed during it): show the cleared boxes, and say so."""
        owner = self.engine.pelvis_facts_cleared_for
        if owner is None:
            return
        self.engine.pelvis_facts_cleared_for = None
        for w in (self.tape_spin, self.chk_wall):
            w.blockSignals(True)
        self.tape_spin.setValue(self.tape_spin.minimum())
        self.chk_wall.setChecked(False)
        for w in (self.tape_spin, self.chk_wall):
            w.blockSignals(False)
        if self.engine.test.frontal:
            self.say(f"Another participant: the tape measure of the pelvis and the wall tick (entered for {owner}) were cleared.", 8.0)

    def take_reference_now(self):
        try:
            app = self.engine.take_reference_stamp()
        except (RuntimeError, ValueError) as e:
            self.say(str(e))
            return
        self.say("Reading moment kept (" + ", ".join(f"{self.ref_titles.get(k, k)} {v:.1f}" for k, v in app.items() if v is not None)
                 + "). Now type what the instrument said.", 8.0)
        first = next(iter(self.ref_fields.values()), None)
        if first is not None:
            first.setFocus()
        self.update_reference_box()

    def clear_reference(self):
        try:
            self.engine.clear_reference()
        except RuntimeError as e:
            self.say(str(e))
            return
        for box in self.ref_fields.values():
            box.clear()
            box.setStyleSheet("")
        self._ref_problem = ""
        self.update_reference_box()

    def update_reference_box(self, recording=None):
        """The status under the reference boxes (what is kept, what is typed, whether it belongs to this position) and the section's summary."""
        if recording is None:
            recording = self._rec_started is not None
        e = self.engine
        r = e.reference
        if self._ref_box_position != e.reference_position():
            self._fill_reference_boxes()                # the target, test or participant changed: the boxes must not show the old readings
        if self._ref_enabled != (not recording):       # (this runs every 200 ms: touch the widgets only on a change)
            for w in (self.ref_instrument, self.ref_resolution, self.btn_ref_stamp, self.btn_ref_clear, *self.ref_fields.values()):
                w.setEnabled(not recording)
            self._ref_enabled = not recording
        lines, warn, summary, mismatch = [], False, "none yet", e.reference_mismatch()
        if not r.has_data():
            lines.append("No reading yet for this position.")
        elif mismatch:
            key, target, who = r.position
            if mismatch == "participant":
                lines.append(f"These readings were typed for participant {who or '(no code)'} and will NOT be saved with a trial of "
                             f"{e.reference_position()[2] or 'a participant without a code'}. Clear them, or type new ones.")
                warn, summary = True, "for another participant"
            else:
                was = e.tests.get(key)
                lines.append(f"These readings were typed for {was.label if was else key}"
                             + ("" if target is None else f", {target:g} deg") + (f" (participant {who})" if who else "")
                             + " and will NOT be saved with a trial of this position. Clear them, or type new ones.")
                warn, summary = True, "for another position"
        else:
            if r.stamp:
                when = datetime.fromisoformat(r.stamp["taken_at"])
                age = (datetime.now() - when).total_seconds()
                ago = f"{age:.0f} s" if age < 90 else f"{age / 60:.0f} min"
                kept = ", ".join(f"{self.ref_titles.get(k, k)} {v:.1f}" for k, v in r.stamp["app_deg"].items() if v is not None)
                lines.append(f"Moment kept at {when:%H:%M:%S} ({ago} ago). The app said: {kept}.")
            else:
                lines.append("No moment kept: press 'Reading taken now' as you read the instrument. Without it the first second of the "
                             "recording stands in for the comparison.")
            typed = [f"{self.ref_titles.get(k, k)} {sum(v) / len(v):.1f}" + (f" ({len(v)} readings)" if len(v) > 1 else "") for k, v in r.values.items()]
            lines.append("Typed: " + ", ".join(typed) + "." if typed else "Nothing typed yet.")
            summary = f"{len(r.values)} typed" + (", moment kept" if r.stamp else ", no moment kept")
        if self._ref_problem:
            lines.append(self._ref_problem)
            warn = True
        text = "\n".join(lines)
        if text != self.ref_status.text():
            self.ref_status.setText(text)
        style = "color: #d97706; font-weight: bold;" if warn else ""
        if style != self.ref_status.styleSheet():             # a new style sheet re-styles the label: only when it changes
            self.ref_status.setStyleSheet(style)
        if hasattr(self, "accordion"):                  # (not yet built while the boxes are first made)
            self.accordion.set_summary("reference", summary)

    def _update_view_rows(self):
        """A frontal test (camera in front of the participant) has no 'Participant faces': it offers the optional tape measure of the
        pelvis and the posture tick instead. (The tested leg is in the Participant section for every test since 0.20.0.)"""
        frontal = self.engine.test.frontal
        for w, show, form in ((self.facing_combo, not frontal, getattr(self, "participant_form", None)),     # (built later than the first call)
                              (self.tape_spin, frontal, self.settings_form), (self.chk_wall, frontal, self.settings_form)):
            w.setVisible(show)
            label = form.labelForField(w) if form is not None else None     # (None for the tick box: it spans the row)
            if label is not None:
                label.setVisible(show)

    def on_leg_changed(self):
        try:
            self.engine.tested_leg = self.leg_combo.currentData()
        except RuntimeError as e:
            self._revert_combo(self.leg_combo, "leg", e)
            return
        self._combo_index["leg"] = self.leg_combo.currentIndex()
        self._show_leg()
        self.refresh_view()

    def on_wall_toggled(self, on):
        try:
            self.engine.wall_posture = bool(on)
        except RuntimeError as e:
            self._revert_check(self.chk_wall, self.engine.wall_posture, e)

    # ------------------------------------------------- landmark selection
    def rebuild_landmarks(self):
        """One checkbox per landmark of the current test, plus presets like 'Only: Knee flexion'."""
        t = self.engine.test
        self._stop_marking()
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
        self.rebuild_dot_chips()
        self.update_dot_rows()
        self.rebuild_three_point_choices()
        self.lm_preset.clear()
        self.lm_preset.addItem("Preset: all landmarks", list(t.dots))
        if t.optional_dots:
            self.lm_preset.addItem("Preset: all landmarks + strap and anchor", list(t.all_dots))
        for a in t.angles:
            self.lm_preset.addItem(f"Preset: only {a.title}", list(dict.fromkeys(a.dots)))
        self.update_landmark_info()

    def rebuild_dot_chips(self):
        """The pinned bar's dot buttons: one per landmark of the current test, numbered (the number key re-marks that dot). Each does what
        the Mark / Re-mark button in the landmark list does (arm_dot), so there is one way of redoing a dot."""
        t = self.engine.test
        for chip in self.chips.values():
            self.chip_grid.removeWidget(chip)
            chip.deleteLater()
        self.chips = {}
        for i, n in enumerate(t.all_dots):
            short = DOT_CHIP_NAMES.get(n) or t.dot_title(n).split(" (")[0][:10]
            chip = QPushButton(f"{i + 1} {short}" if i < 9 else short)
            chip.setFocusPolicy(Qt.NoFocus)
            chip.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)          # (never makes the panel wider than the window)
            chip.setMinimumWidth(40)
            chip.clicked.connect(lambda _=False, name=n: self.arm_dot(name))
            self.chip_grid.addWidget(chip, i // DOT_CHIP_COLUMNS, i % DOT_CHIP_COLUMNS)
            self.chips[n] = chip
        cells = max(len(t.all_dots), 2 * DOT_CHIP_COLUMNS - 1)          # (Redo all sits in the last cell: bottom right)
        self.chip_grid.removeWidget(self.btn_redo_dots)
        self.chip_grid.addWidget(self.btn_redo_dots, cells // DOT_CHIP_COLUMNS, cells % DOT_CHIP_COLUMNS)
        for c in range(DOT_CHIP_COLUMNS):
            self.chip_grid.setColumnStretch(c, 1)

    DOT_STATE = {"ok": ("marked", "#16a34a"), "lost": ("LOST", "#d97706"), "unmarked": ("not marked", "#9ca3af"), "off": ("not used", "#9ca3af")}

    def update_dot_rows(self):
        """Per landmark: marked / LOST / not marked, and what its button does (Mark, Re-mark, or Cancel while it waits for a click)."""
        if not self.lm_rows:
            return
        status = self.engine.dot_status()
        free = self._rec_started is None and self.worker is not None and not self.engine.paused
        if self._pass:
            if not free:                                      # a recording, a break or no camera ends the pass (no dot is marked or removed by it)
                self._stop_marking()
            else:
                self._pass = [n for n in self._pass if status.get(n, "off") != "off"]      # a dot switched off in the middle is not done
                if not self._pass:
                    self._stop_marking()
                elif self._arm is None or status.get(self._arm, "off") == "off":
                    self._arm = self._pass[0]
        if self._arm is not None and status.get(self._arm, "off") == "off":
            self._arm = None                                  # switched off while waiting for a click
        want_redo = "Stop (D)" if self._pass else "Redo all (D)"
        if self.btn_redo_dots.text() != want_redo:
            self.btn_redo_dots.setText(want_redo)
        for n in self.lm_rows:
            s = status.get(n, "off")
            text, colour = self.DOT_STATE[s]
            if self.lm_status[n].text() != text:
                self.lm_status[n].setText(text)
                self.lm_status[n].setStyleSheet(f"color: {colour}; font-weight: bold;")
            armed = self._arm == n
            pending = n in self._pass and not armed
            want = ("Skip" if self._pass else "Cancel") if armed else "Re-mark" if s in ("ok", "lost") else "Mark"
            btn = self.lm_buttons[n]
            if btn.text() != want:
                btn.setText(want)
            if btn.isChecked() != armed:
                btn.setChecked(armed)
            btn.setEnabled(s != "off" and free)
            chip = self.chips.get(n)
            if chip is not None:
                chip.setVisible(s != "off")                     # a dot that is not used has no button
                style = (DOT_CHIP_STYLE if not pending else DOT_CHIP_PENDING)["armed" if armed else s if s != "off" else "unmarked"]
                if chip.styleSheet() != style:
                    chip.setStyleSheet(style)
                tip = (f"{self.engine.test.dot_title(n)}: "
                       f"{'waiting for your click on the video' if armed else text + ', still to be marked again' if pending else text}")
                if chip.toolTip() != tip:
                    chip.setToolTip(tip)
                chip.setEnabled(s != "off" and free)

    def arm_dot(self, name):
        """The button (or number key) of one landmark: its old mark is removed at once, and the next click on the picture marks that dot
        again; every other dot is left alone. It stays waiting until the dot has been found, so a click that missed can simply be repeated."""
        btn = self.lm_buttons.get(name)
        if self.worker is None or self.engine.rec is not None or self.engine.paused:
            if btn is not None:
                btn.setChecked(False)
            self.say("Start the camera first (and stop recording or the break) to mark a dot." if self.worker is None
                     else "Dots cannot be marked during a recording or a break.")
            return
        self._leave_ruler_marking()
        title = self.engine.test.dot_title(name)
        if self._pass:                                        # in the middle of 'Redo all': its own dot is skipped, another one is taken next
            if name == self._arm:
                self._pass.remove(name)
                self._pass_next(f"Skipped: the {title} dot is not marked.")
            else:
                self._take_away(name)                         # (a dot marked earlier in this pass is taken away again)
                self._arm = name
                self.update_dot_rows()
                self.say(f"Click the {title} dot on the video.", 8.0)
            return
        if self._arm == name:                                 # pressed again while it waits: the click is no longer asked for
            self._arm = None
            self.update_dot_rows()
            self.say(f"Cancelled. The {title} dot stays removed: press its button to mark it.", 8.0)
            return
        self._take_away(name)
        self._arm = name
        self.update_dot_rows()
        self.say(f"Click the {title} dot on the video.", 8.0)

    def _take_away(self, name):
        """The old mark of one dot goes before it is marked again (the engine keeps the lock, see Engine.clear_dot)."""
        if self.engine.clear_dot(name):
            self.note("dot removed", name)

    # ------------------------------------------------- 'Redo all': every dot removed, then marked one after the other
    def _stop_marking(self):
        """Nothing waits for a click any more: neither one dot nor a pass (a dot that was not marked stays unmarked)."""
        self._arm, self._pass, self._pass_total = None, [], 0

    def toggle_mark_again(self):
        """Redo all (D): every marked dot and the lock are removed at once (the participant has been moved, so no old mark may linger on
        the picture), then the next clicks on the video mark the dots one after the other in the order asked. A dot can be skipped (press
        its button: it stays unmarked) or taken out of turn (press another dot's button). Pressing it again (now Stop) ends the asking."""
        if self._pass:
            self._stop_marking()
            self.note("marking all dots again stopped")
            self.update_dot_rows()
            left = self.engine.next_dot_to_mark()
            self.say("Stopped. " + ("All the dots are marked." if left is None else "The dots not marked yet stay unmarked: use their "
                                                                                    "buttons or number keys."), 6.0)
            return
        names = list(self.engine.active_dots)
        if self.worker is None or self.engine.rec is not None or self.engine.paused:
            self.say("Start the camera first (and stop recording or the break) to mark the dots." if self.worker is None
                     else "Dots cannot be marked during a recording or a break.")
            return
        if not names:
            self.say("No landmark is switched on: there is no dot to mark.")
            return
        self._leave_ruler_marking()
        self.engine.reset_dots()                              # every old mark and the lock go now, before the first click
        self.btn_lock.setChecked(False)
        self._pass, self._pass_total, self._arm = names, len(names), names[0]
        self.note("marking all dots again started", "all dots removed")
        self.update_dot_rows()
        self.refresh_view()
        self.say(self._instruction_text(), 8.0)

    def _pass_done(self, name):
        """The dot `name` was marked again in a pass: the next dot is asked for."""
        if name in self._pass:
            self._pass.remove(name)
        self._pass_next("")                                   # (the dot's button turning green says it was marked)

    def _pass_next(self, lead):
        """Wait for the next dot still to be marked again, or finish."""
        if not self._pass:
            total = self._pass_total
            unmarked = [n for n, s in self.engine.dot_status().items() if s == "unmarked"]
            self._stop_marking()
            self.note("marking all dots again finished", f"{len(unmarked)} not marked" if unmarked else "")
            self.update_dot_rows()
            tail = (f"All {total} dots marked. Set the position, then lock it." if not unmarked
                    else f"{len(unmarked)} of {total} dots are not marked: use their buttons or number keys.")
            self.say(f"{lead} {tail}".strip(), 8.0)
            return
        self._arm = self._pass[0]
        self.update_dot_rows()
        self.say(f"{lead} {self._instruction_text()}".strip(), 8.0)

    def redo_dot_number(self, number):
        """The number key of a dot button in the pinned bar: the same as pressing it (a dot that is not used has no button)."""
        names = list(self.engine.test.all_dots)
        if not 1 <= number <= len(names):
            return
        chip = self.chips.get(names[number - 1])
        if chip is not None and chip.isEnabled() and not chip.isHidden():
            chip.click()

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
        self.rebuild_reference_fields()                   # the angles of the test changed: so did what a reading can be typed for
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
            self._revert_check(self.chk_overlay, self.engine.save_overlay, e)

    def _set_recording_first(self, on):
        try:
            self.engine.recording_first = bool(on)
        except RuntimeError as e:
            self._revert_check(self.chk_rec_first, self.engine.recording_first, e)

    def _start_recording_view(self):
        """'Recording first' (0.21.0): the trace once a second while recording (the picture: VideoWorker._show; the panel: refresh_status)."""
        if self.engine.recording_first:
            self.trace_timer.setInterval(TRACE_MS_RECORDING)
            self._panel_tick = 0

    def _end_recording_view(self):
        self.trace_timer.setInterval(TRACE_MS)

    def on_mirror_toggled(self, on):
        self.engine.mirror = bool(on)
        self._stop_marking()
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
        if spin.specialValueText() and value <= spin.minimum():
            value = None                                             # the box's lowest value: "no target" / "not measured"
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

    def _revert_check(self, chk, value, error):
        """Put a tick box back to what the engine kept (it refused the change, for example while recording) and say why."""
        chk.blockSignals(True)
        chk.setChecked(bool(value))
        chk.blockSignals(False)
        self.say(str(error))

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
        had_ruler = self.engine.scale.configured
        try:
            self.engine.set_dot_kind(self.kind_combo.currentData())
        except RuntimeError as e:
            self._revert_combo(self.kind_combo, "kind", e)
            return
        self._combo_index["kind"] = self.kind_combo.currentIndex()
        self._stop_marking()
        self.update_dot_rows()
        self.btn_lock.setChecked(False)
        if had_ruler:                                       # (the ruler dots are followed in the dot colour; 0.24.0: the ruler is marked first)
            self._ruler_marking = None
            self.say("The dot colour changed, so the ruler was cleared too: mark it again (C).", 8.0)

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
        self._priv_recheck = False                        # (the whole privacy step is asked afresh for a new picture)
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
        self._stop_marking()                                  # (the ruler is served first: a dot waiting for a click would never get it)
        self.update_dot_rows()
        self._ruler_marking = 0
        self.note("ruler marking started")
        self.say("Click the ruler dot at one end of the ruler.", 8.0)

    def redo_ruler(self):
        """The pinned 'Ruler (C)' button and the C key (0.24.0): mark the ruler again from whichever section is open; pressed again while it
        waits for the clicks, it stops (the old scale is gone either way, like a single dot that was cancelled)."""
        if self._ruler_marking is not None:
            self._ruler_marking = None
            self.update_scale_label()
            self.say("Ruler marking stopped. Press Ruler (C) to mark it again.")
            return
        self.start_ruler_marking()
        self.update_scale_label()

    def ruler_is_fine(self):
        """The reminder 'still right at this position?' is answered: the ruler and the camera have not moved."""
        if self._ruler_recheck:
            self._ruler_recheck = False
            e = self.engine
            self.note("ruler checked", f"{e.test.key} {'' if e.target is None else format(e.target, 'g')} (still right)".strip())
        self.update_steps()

    def relaxed_not_here(self):
        """'Not here' on the resting-force reminder: no Zero 2 at this position; the reminder stays away until the position changes."""
        self._relax_dismissed = self.engine.relaxed_key()
        self.note("resting force reminder dismissed", self._relax_dismissed)
        self.update_steps()

    def confirm_privacy_if_open(self):
        """The V key: 'I looked, no face is visible' (with a crop or masks: they are still right). Works whenever the picture is there."""
        if self.worker is None or self._rec_started is not None:
            self.say("Start the camera first, then look at the picture." if self.worker is None else "Not while recording.")
            return
        self.confirm_privacy_here()
        self.say("Privacy: looked at this position.")

    def on_ruler_click(self, x, y):
        index = self._ruler_marking
        if self.engine.mark_scale_dot(index, x, y):
            if index == 0:
                self._ruler_marking = 1
                self.say("Now click the ruler dot at the other end.", 8.0)
            else:
                self._ruler_marking = None
                self._ruler_recheck = False                       # marked just now: that is the look at this position
                self.note("ruler marked", f"{self.engine.scale.px_per_cm:.2f} px/cm")
                self.say(f"Scale set: {self.engine.scale.px_per_cm:.2f} px per cm ({self.ruler_len.value():g} cm ruler).")
        else:
            self.say(self.engine.mark_problem or "No dot found at that spot. Click the centre of the ruler dot.")
        self.refresh_view()

    def _leave_ruler_marking(self):
        """A dot is about to wait for the next click on the picture: the ruler dots, which are served first, no longer may."""
        if self._ruler_marking is not None:
            self._ruler_marking = None
            self.update_scale_label()

    def clear_scale(self):
        self.engine.clear_scale()
        self._ruler_marking = None
        self.refresh_view()

    def update_scale_label(self):
        sc = self.engine.scale
        recording = self._rec_started is not None
        for w in (self.ruler_len, self.btn_ruler, self.btn_ruler_clear, self.btn_ruler_pin):
            w.setEnabled(not recording)
        waiting = self._ruler_marking is not None
        if self.btn_ruler_pin.isChecked() != waiting:
            self.btn_ruler_pin.setChecked(waiting)           # (checked = it waits for the clicks; a click on the button itself is redo_ruler)
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
        had_relaxed = src.calibration.relaxed_v is not None
        why = src.zero()
        self.note("force zero refused" if why else "force zeroed", why or "")
        if why:
            self._force_msg = why
        else:
            cal = src.calibration
            moved = "" if before is None or not cal.has_slope else f" (moved {(cal.offset_v - before) * cal.n_per_v:+.1f} N from the last zero)"
            self._force_msg = (f"Zero set: {cal.offset_v:+.4f} V is now 0 N{moved}." + self._zero_quality_note(cal)
                               + (" The relaxed level was cleared: take it again (X) with the participant relaxed in the strap." if had_relaxed else ""))
        self.update_force_box()

    def relax_force(self):
        """The second zero level (0.22.0): the tension with the participant relaxed in the strap, for this position."""
        src = self.force_source
        if src is None:
            return
        why = src.relax(self.engine.relaxed_key())
        self.note("force relaxed level refused" if why else "force relaxed level taken", why or "")
        if why:
            self._force_msg = why
        else:
            cal = src.calibration
            n, se = cal.relaxed_n, cal.relaxed_se_v
            where = f"{n:+.1f} N above the zero ({cal.relaxed_v:+.4f} V)" if n is not None else f"{cal.relaxed_v:+.4f} V"
            self._force_msg = (f"Relaxed level set: {where}, good to about {se * 1000:.1f} mV." if se is not None else f"Relaxed level set: {where}.")
            if n is not None and n < -2.0:
                self._force_msg += (" That is BELOW the zero: was the zero taken with the strap pulled or the cell loaded? Zero again (Z) with the "
                                    "strap hanging free, then take the relaxed level.")
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
        self.btn_force_relax.setEnabled(connected and not recording)
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
                rel = self.engine.relaxed_here()                # the second zero level, when it was taken for this position
                above = f"\nabove relaxed {n - rel.relaxed_n:.1f} N" if rel is not None and rel.relaxed_n is not None and n is not None else ""
                self.force_label.setText("Force: waiting..." if v is None else f"Force {n:.1f} N   ({v:+.3f} V)" + above if n is not None
                                         else f"Force {v:+.4f} V")
                cal = src.calibration
                if cal.calibrated:
                    info = f"{cal.n_per_v:.2f} N per volt ({cal.method}), zero {cal.offset_v:+.4f} V set {cal.when[11:16]}."
                    if rel is not None and rel.relaxed_n is not None:
                        info += f"\nRelaxed level {rel.relaxed_n:+.1f} N above the zero, set {rel.relaxed_when[11:16]} for this position."
                    elif cal.relaxed_v is not None:
                        info += (f"\nThe relaxed level ({cal.relaxed_when[11:16]}) is for another position or participant ({cal.relaxed_for}): "
                                 "take it again (X) for this one.")
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
                    warn = (not self._force_msg.startswith(("Zero set", "Calibrated", "Relaxed level set"))        # a done zero is news, not a warning
                            or "BELOW" in self._force_msg)
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
        pinned = []                                       # the same rows for the pinned dropdown: "Knee extension, 70 deg   (1/2)"
        for i, (key, tgt) in enumerate(self.plan_rows):
            n = count_trials(self.recordings_dir, pid, key, tgt) if valid else 0
            mark = "-" if not valid else ("done" if n >= TRIALS_PER_POSITION else f"{n}/{TRIALS_PER_POSITION}")
            if valid and n == TRIALS_PER_POSITION and (self._repeat_state(pid, key, tgt, n) or {}).get("state") == "third":
                mark = "3rd!"                             # the two peaks differ by more than 10 %: a third trial (SOP 10.6, 0.20.0)
            text = f"{mark:>5}   {i + 1:02d}  {self._plan_text(key, tgt)}"
            pinned.append(f"{i + 1:02d}  {self._plan_text(key, tgt)}" + (f"   ({mark})" if valid else ""))
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
        self._plan_current = current
        pc = self.pos_combo
        pc.blockSignals(True)                             # (only a pick by the operator, 'activated', goes on to change the position)
        if pc.count() != len(pinned):
            pc.clear()
            pc.addItems(pinned)
        else:
            for i, text in enumerate(pinned):
                if pc.itemText(i) != text:
                    pc.setItemText(i, text)
        pc.setCurrentIndex(-1 if current is None else current)
        pc.blockSignals(False)
        where = self._plan_text(cur_key, cur_tgt)
        if not valid:
            self.trial_label.setText("Enter the participant code (P01, ...) to count trials.")
        elif current is None:
            self.trial_label.setText(f"{where}: not a position of the plan (trials are not counted).")
        elif n_cur >= TRIALS_PER_POSITION:
            chk = self._repeat_state(pid, cur_key, cur_tgt, n_cur)
            if chk is None and self._rec_started is None:   # not known yet (a restart): work it out for this position, quietly
                metas = trial_metas(self.recordings_dir, pid, cur_key, cur_tgt)
                if metas:
                    self._check_position(metas[-1], self._position_key(pid, cur_key, cur_tgt), announce=False)
            state = (chk or {}).get("state")
            if state == "third":
                self.trial_label.setText(f"{where}: THIRD TRIAL NEEDED, the two peaks differ by {chk['pct']:.1f} % (limit 10 %). "
                                         f"Next is trial {n_cur + 1}.")
            elif state == "deviation":
                self.trial_label.setText(f"{where}: {n_cur} trials and no two within 10 %: note it in the deviation log (no more trials).")
            elif state == "no_rise":                       # 0.26.0
                self.trial_label.setText(f"{where}: {n_cur} trials saved, none with a force rise: the 10 % check cannot be made. "
                                         f"Another is trial {n_cur + 1}.")
            else:
                ok = " 10 % check: fine." if state == "ok" else ""
                self.trial_label.setText(f"{where}: {n_cur} trials already saved (planned {TRIALS_PER_POSITION}).{ok} "
                                         f"Another is trial {n_cur + 1}.")
        else:
            self.trial_label.setText(f"Next: trial {n_cur + 1} of {TRIALS_PER_POSITION}\n{where}")
        pc.setToolTip(self.trial_label.text().split("\n")[0] + "\nPick another position to set the test and the target in one step.")

    def on_plan_clicked(self, item):
        self.go_to_position(self.plan.row(item))

    def go_to_position(self, row):
        """Set the test and the target of one row of the session plan: from the plan list, the pinned dropdown or the previous/next keys.
        The marked dots stay when the new test has the same landmarks (the engine decides, see Engine.set_test)."""
        if not 0 <= row < len(self.plan_rows):
            return
        key, tgt = self.plan_rows[row]
        if self.engine.rec is not None:
            self.say("Stop recording before changing the position.")
            self.refresh_plan()
            return
        idx = self.test_combo.findData(key)
        if idx != self.test_combo.currentIndex():
            self.test_combo.setCurrentIndex(idx)
            if self.engine.test.key != key:               # the engine refused
                self.refresh_plan()
                return
        self.target_spin.setValue(float(tgt) if tgt is not None else self.target_spin.minimum())
        self.note("position chosen", f"{key} {tgt if tgt is not None else ''}".strip())
        self.say("Position set. Mark the dots if needed, check the angles, then lock.", 5.0)
        self.refresh_plan()

    def step_position(self, delta):
        """The previous (-1) or next (+1) row of the session plan. It stops at the ends; from a position that is not in the plan it
        starts at the first (next) or the last (previous) row."""
        rows, cur = len(self.plan_rows), self._plan_current
        row = (0 if delta > 0 else rows - 1) if cur is None else cur + delta
        if not 0 <= row < rows:
            self.say("This is the last position of the plan." if delta > 0 else "This is the first position of the plan.", 4.0)
            return
        self.go_to_position(row)

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
        if ok:                                                # (drawing it meant looking at the picture of this position)
            self.confirm_privacy("crop drawn" if kind == "crop" else "mask added")

    def _position_changed(self, *_):
        """The participant is in another position (another test or target). Two things may no longer be right, and both come up as soft
        reminders (0.24.0; nothing stops Record): the picture the privacy set-up was looked at on (the crop may no longer cover the leg, a
        face may have come into view; the 'I looked' tick is cleared, 0.23.0), and the ruler (it may have been moved or taken away).
        The same position again, trial after trial, asks nothing. (The resting force follows its own per-position rule.)"""
        e = self.engine
        old, self._priv_pos = self._priv_pos, (e.test.key, e.target)
        if old == self._priv_pos or self._restoring:
            return
        where = f"{e.test.key} {'' if e.target is None else format(e.target, 'g')}".strip()
        self._priv_recheck = True
        if self.chk_priv_ok.isChecked():
            self.chk_priv_ok.setChecked(False)                # ('I looked' was about the previous position)
        self.note("privacy to be checked again", where)
        if e.scale.configured:
            self._ruler_recheck = True
            self.note("ruler to be checked again", where)
        self.update_steps()

    def confirm_privacy(self, how):
        """The operator looked at the picture of this position and says the privacy set-up is right for it (saved with the next trials)."""
        self.engine.note_privacy_check(how)
        if self._priv_recheck:
            self._priv_recheck = False
            e = self.engine
            self.note("privacy checked again", f"{e.test.key} {'' if e.target is None else format(e.target, 'g')} ({how})".strip())
        self.update_steps()

    def confirm_privacy_here(self):
        """The button shown after a change of position. With a crop or masks it says they are still right; without, it is the same as
        ticking 'I looked: nothing identifies the participant' (the step list wants one of the two)."""
        if self.engine.privacy_crop or self.engine.privacy_masks or self.chk_priv_ok.isChecked():
            self.confirm_privacy("looked: the crop and masks are still right")
        else:
            self.chk_priv_ok.setChecked(True)                 # (its toggled signal confirms)

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
        in_pass = bool(armed and self._pass)                  # (the old marks of the dots still to do must not stop this one being marked)
        ok = self.engine.mark_dot(name, x, y, replacing=[n for n in self._pass if n != name] if in_pass else ())
        self.note("dot marked" if ok and first_marking else "dot re-marked" if ok else "dot click found no dot",
                  name + ("" if ok or not self.engine.mark_problem else f": {self.engine.mark_problem}"))
        leave = "skip it" if in_pass else "cancel"
        if ok:
            if in_pass:
                self._pass_done(name)
            elif armed:
                self._arm = None
                self.say(f"Marked again: {title}. The other dots were not touched.")
            elif not first_marking:
                self.say(f"Re-marked: {title}.")
        elif self.engine.mark_problem:
            self.say(self.engine.mark_problem + (f" Click again, or press its button to {leave}." if armed else ""))
        else:
            self.say(f"No dot found at that spot for '{title}'. Click the centre of the dot (check colour setting and lighting)."
                     + (f" Still waiting: click again, or press its button to {leave}." if armed else ""))
        self.update_dot_rows()
        self.refresh_view()

    def note(self, event, detail=""):
        """One line in the session log; a log that cannot be written is said once (never silently lost)."""
        if not self.log.write(event, detail) and self.log.problem != getattr(self, "_log_problem_shown", None):
            self._log_problem_shown = self.log.problem
            self.say(self.log.problem, 10.0)

    def remark(self):
        self.note("all dots cleared")
        self._stop_marking()
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
            self._revert_check(self.chk_cues, self.engine.audio_cues, e)
        self.cue_push_spin.setEnabled(bool(self.engine.audio_cues))

    def on_auto_mvic_toggled(self, on):
        try:
            self.engine.auto_mvic = bool(on)
        except RuntimeError as e:
            self._revert_check(self.chk_auto_mvic, self.engine.auto_mvic, e)

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
            if not self.engine.tested_leg and not self._ask_tested_leg():
                self.btn_rec.setChecked(False)
                return
            notes_problem = privacy.notes_problem(self.notes.text())
            if notes_problem:
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Notes", notes_problem + " Remove it: notes must not identify anyone.")
                return
            ref_problem = self._commit_reference_boxes()
            if ref_problem:
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Reference reading", ref_problem)
                self.open_step("reference")
                return
            # What is still open as a soft reminder does not stop the recording (0.24.0): it is written to the session log and to
            # meta.json, and what changes the file itself is said again with the trial's warnings once it is saved.
            reminders = self._open_reminders()
            self.engine.reminders_open = [text for text, _ in reminders]
            self._soft_for_file = [text for text, in_file in reminders if in_file]
            self.engine.rest_before = self.rest.meta(self._position_now(), self._rest_position_done())
            try:
                self.engine.start_trial(self.participant.text().strip(), self.notes.text())
            except Exception as e:  # noqa: BLE001
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Cannot record", str(e))
                return
            self.rest.stop()                                        # (a new rest starts when this recording stops)
            self._update_rest()
            self._rec_started = time.perf_counter()
            self._start_recording_view()
            self.note("recording started", f"{self.participant.text().strip()} {self.engine.test.key} {self.engine.target if self.engine.target is not None else ''}".strip())
            if reminders:
                self.note("recording started with reminders open", " | ".join(text for text, _ in reminders))
            self.refresh_plan()
            self._marker_counts = {"start": 0, "end": 0, "other": 0}
            self._marker_msg = ("", 0.0)
            self.btn_rec.setText("Stop recording  (R)")
            if self.worker is not None and self.worker.paused:      # recording needs frames flowing
                self.btn_pause.setChecked(False)
                self.toggle_pause()
        else:
            self.btn_rec.setChecked(False)
            self._end_recording_view()
            n = self.engine.rec.frames if self.engine.rec is not None else None
            duration = time.perf_counter() - self._rec_started if self._rec_started is not None else 0.0
            if self._rec_started is not None:                       # the rest after this contraction begins now (0.27.0; also when the
                self.rest.start("trial", self._position_now(), self.engine.test.key)     # engine had to end the recording itself)
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
                warnings = list(self.engine.last_trial_warnings) + [f"Recorded while this was open: {t}" for t in self._soft_for_file]
                self._soft_for_file = []
                self.rec_label.setText(f"Saved: {paths['data.csv'].parent.name}/{paths['raw.mp4'].name[:-8]}... "
                                       f"({n} frames, {duration:.1f} s)" + ("  VERY SHORT RECORDING" if short else "")
                                       + ("\nCHECK THIS TRIAL: " + warnings[0] if warnings else ""))
                self.rec_label.setStyleSheet("color: #dc2626; font-weight: bold;" if warnings else "color: #d97706; font-weight: bold;" if short else "")
                self._last_trial_warnings = warnings
                if warnings:
                    self._show_trial_warnings(warnings)
                try:                                       # the 10 % check and the resting force follow from the saved files (0.20.0)
                    m = json.loads(paths["meta.json"].read_text(encoding="utf-8"))
                    key = self._position_key(m.get("participant"), m.get("test_key"), m.get("target_deg"))
                except (OSError, ValueError) as e:
                    self._add_after_trial([f"The 10 % check could not be made: {e}"], warn=True)
                else:
                    self._check_position(paths["meta.json"], key, announce=True)

    def _open_reminders(self):
        """What is still open when Record is pressed, as (sentence, changes_what_the_file_holds) pairs (0.24.0). None of it stops the
        recording. The first four used to be questions or a refusal; the last two were already only reminders on the screen."""
        e = self.engine
        out = []
        shaped = bool(e.privacy_crop) or bool(e.privacy_masks)
        if self._priv_recheck:
            out.append(("Privacy: the picture was not looked at again after the position changed.", True))
        elif not (shaped or self.chk_priv_ok.isChecked()):
            out.append(("Privacy: nobody said that no face is visible.", True))
        if not e.locked:
            out.append(("The position is not locked, so the other joints are not checked for drift.", True))
        main_problem = e.main_angle_problem()
        if main_problem:
            out.append((main_problem + " The trial may have no (or a doubtful) main angle.", True))
        src = self.force_source
        if src is not None and not src.error and src.calibration.has_slope and not src.calibration.zero_set:
            # 4 of the 11 trials of 2026-10-15 were recorded like this: volts only, no newtons and no peak force (Review can apply a later zero)
            out.append(("The force has not been zeroed since the program started: this trial has volts but no newtons.", True))
        if self._ruler_recheck and e.scale.configured:
            out.append(("Ruler: not looked at again after the position changed.", False))
        if src is not None and src.calibration.zero_set and e.relaxed_here() is None and self._relax_dismissed != e.relaxed_key():
            out.append(("Resting force: Zero 2 was not taken for this position.", False))
        rest = self.rest.reminder(self._position_now(), self._rest_position_done())          # 0.27.0: the rest is in the file (rest_before)
        if rest:
            out.append((rest, True))
        return out

    def _ask_tested_leg(self):
        """Record with no tested leg chosen for this participant (0.20.0): ask Right / Left / Cancel. True when a leg was chosen."""
        code = self.participant.text().strip()
        box = QMessageBox(QMessageBox.Question, "Tested leg",
                          f"Which leg is tested for {code}?\n\nIt is the dominant leg (SOP 4.5: the leg the participant kicks a ball with). "
                          f"It is kept for {code} and saved in every trial.", QMessageBox.Cancel, self)
        right = box.addButton("Right leg", QMessageBox.AcceptRole)
        left = box.addButton("Left leg", QMessageBox.AcceptRole)
        box.exec()
        clicked = box.clickedButton()
        leg = "right" if clicked is right else "left" if clicked is left else None
        if leg is None:
            self.open_step("participant")
            return False
        self.leg_combo.setCurrentIndex(self.leg_combo.findData(leg))
        return self.engine.tested_leg == leg

    # ------------------------------------------------------------ the rest between contractions (SOP 10.5, 10.8, 11.7, 9.11; 0.27.0)
    def _position_now(self):
        return self._position_key(self.participant.text().strip(), self.engine.test.key, self.engine.target)

    def _rest_position_done(self):
        """Whether the 10 % check needs no further trial at the position the rest follows (then the next is a new position: 120 s)."""
        got = self._repeat.get(self.rest.position) if self.rest.position is not None else None
        return got is not None and got[1].get("state") in ("ok", "deviation")

    def start_submax_rest(self):
        """The 30 s after the submaximal contraction (not recorded), started by hand."""
        if self._rec_started is not None:
            self.say("A rest cannot start during a recording.")
            return
        self.rest.start("submax")
        self.note("rest started", "after the submaximal contraction")
        self._update_rest()

    def _update_rest(self):
        """The rest line under Record: amber while the rest runs, green when the SOP's time is over (with a beep, if wanted). Hidden while
        recording and when no rest is timed."""
        st = None if self._rec_started is not None else self.rest.status(self._position_now(), self._rest_position_done())
        if st is None:
            if self.rest_label.isVisible():
                self.rest_label.setVisible(False)
            return
        style = "color: #16a34a; font-weight: bold;" if st["done"] else "color: #b45309; font-weight: bold; font-size: 15px;"
        if self.rest_label.text() != st["text"]:
            self.rest_label.setText(st["text"])
        if self.rest_label.styleSheet() != style:
            self.rest_label.setStyleSheet(style)
        if not self.rest_label.isVisible():
            self.rest_label.setVisible(True)
        if self.rest.beep_due(self._position_now(), self._rest_position_done()):
            self.note("rest over", st["text"])
            if self.chk_rest_beep.isChecked():
                self.engine.rest_beep()

    # ------------------------------------------------------------ the 10 % check after each trial (SOP 10.6; 0.20.0)
    @staticmethod
    def _position_key(participant, test_key, target):
        return (participant or "", test_key, None if target is None else round(float(target), 6))

    def _check_position(self, meta_path, key, announce):
        """Work out the 10 % check of the position of `meta_path` (every saved trial of that participant, test and target) on a helper
        thread: reading the trials takes 0.1-0.2 s each and the live picture must not wait. The result comes back through
        position_checked. `announce`: say it (after a trial just saved), else only update the plan (a position reached after a restart)."""
        if announce:
            self._repeat_announce.add(key)
        if key in self._repeat_running:                   # (one is on its way: it reads the same files, the newest trial included)
            return
        self._repeat_running.add(key)

        def work():
            res = {"key": key, "meta_path": meta_path}
            try:
                res.update(after_trial_check(meta_path))
            except Exception as e:  # noqa: BLE001 - said on screen and in the session log, never hidden
                res["error"] = f"{type(e).__name__}: {e}"
            try:
                self.position_checked.emit(res)
            except RuntimeError:                          # the window was closed meanwhile: nobody left to tell
                pass
        threading.Thread(target=work, name="position-check", daemon=True).start()

    def _on_position_checked(self, res):
        key = res["key"]
        self._repeat_running.discard(key)
        announce = key in self._repeat_announce
        self._repeat_announce.discard(key)
        if "error" in res:
            text = "The 10 % check could not be made: " + res["error"]
            self.note("position check failed", text)
            n = len(trial_metas(Path(res["meta_path"]).parent.parent, *key))
            self._repeat[key] = (n, {"state": "error", "text": text, "pct": math.nan, "n": n})    # not tried again for these trials
            if announce:
                self._add_after_trial([text], warn=True)
            return
        chk = res["check"]
        self._repeat[key] = (res["n"], chk)
        if announce:
            saved = trial_metas(Path(res["meta_path"]).parent.parent, *key)
            if res["n"] < len(saved):                 # a check that began before this trial was saved: once more, with it
                self._check_position(saved[-1], key, announce=True)
                return
        if announce:
            lines, warn = [], False
            if res.get("angle_note"):                 # 0.26.0: the main joint during the push, the shift from rest, where to set it
                self.note("hold angle", res["angle_note"])
                lines.append(res["angle_note"])
            if res.get("position_note"):              # 0.26.0: an angle that does not fit the position (wrong position chosen?)
                self.note("angle check", res["position_note"])
                lines.append("CHECK: " + res["position_note"])
                warn = True
            if res.get("relaxed_note"):               # 0.26.0: Zero 2 does not describe this trial
                self.note("trial warning", res["relaxed_note"])
                lines.append(res["relaxed_note"])
                warn = True
            if res["rest_note"]:
                self.note("trial warning", res["rest_note"])
                lines.append(res["rest_note"])
                warn = True
            if chk["text"]:
                self.note("position check", chk["text"])
                lines.append(chk["text"])
                warn = warn or chk["state"] in ("third", "deviation", "units", "no_rise")
            if lines:
                self._add_after_trial(lines, warn)
        self.refresh_plan()

    def _add_after_trial(self, lines, warn):
        """Add lines to what the window says about the trial just saved: the record line, and (when one of them asks for something)
        the non-blocking message together with the trial's other warnings. Not while the next trial is already recording."""
        if self._rec_started is not None:
            return
        self.rec_label.setText(self.rec_label.text() + "".join("\n" + ln for ln in lines))
        if warn:
            self.rec_label.setStyleSheet("color: #dc2626; font-weight: bold;")
            self._show_trial_warnings(self._last_trial_warnings + lines, log=False)

    def _repeat_state(self, pid, key, tgt, n):
        """The 10 % check of a position when it is known for the trials now saved there (n), else None."""
        got = self._repeat.get(self._position_key(pid, key, tgt))
        return got[1] if got is not None and got[0] == n else None

    def _show_trial_warnings(self, warnings, log=True):
        """Problems found in the trial just saved (no force rise, dots that moved apart, CHECK LABELS; 0.19.0; the resting force and the 10 %
        check, 0.20.0): a message that does not block, so the next trial can still be started with R, and a line each in the session log."""
        for w in warnings if log else ():
            self.note("trial warning", w)
        if self._trial_box is not None:
            self._trial_box.close()
        box = QMessageBox(QMessageBox.Warning, "Check this trial", "\n\n".join(warnings), QMessageBox.Ok, self)
        box.setModal(False)
        box.show()
        self._trial_box = box

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
        if recording and self.engine.recording_first:
            self._panel_tick += 1                        # 'recording first': the rest of the panel once a second, the first tick of a
            if (self._panel_tick - 1) % PANEL_TICKS_RECORDING:     # recording in full (it greys out what must not change)
                self._refresh_recording()
                return
        self.update_force_box()
        self.update_reference_box(recording)
        self._update_rest()
        self._show_cleared_pelvis_facts()            # (the engine clears them when a recording ends with another participant code)
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
        self.chk_rec_first.setEnabled(not recording)
        for w in (*self.camera_checks.values(), self.camera_model):
            w.setEnabled(not recording)
        for w in (self.test_combo, self.kind_combo, self.facing_combo, self.leg_combo, self.tape_spin, self.chk_wall, self.target_spin,
                  self.tol_spin, self.ntol_spin, self.btn_remark, self.btn_lock, self.btn_redo_dots, self.pos_combo, self.btn_pos_prev,
                  self.btn_pos_next):
            w.setEnabled(not recording)           # nothing that changes the measurement may change during a recording
        main_three_point = bool(self.engine.three_point) and self.engine.three_point_main
        self.suggest.setEnabled(not recording and not main_three_point)
        for w in (self.chk_tp, self.chk_tp_main, self.chk_tp_flex, *self.tp_combos):
            w.setEnabled(not recording)
        self._refresh_recording()
        self.update_camera_info()
        self.update_scale_label()
        self.update_steps()
        if time.monotonic() < self._say_until:
            return
        if self.worker is None and self.instruction.text() not in (self._auto_text, IDLE_TEXT, ""):
            return                    # a message that was left when the camera ended: it stays until something replaces it
        self._auto_text = self._instruction_text()
        self.instruction.setText(self._auto_text)

    def _refresh_recording(self):
        """The recording line, the automatic stop and the 'no video' warning (every tick of the status timer)."""
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
