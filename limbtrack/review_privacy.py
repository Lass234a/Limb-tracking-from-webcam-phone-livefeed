"""The Privacy window of Review (0.34.0): draw a crop and black masks on a picture of a recorded trial and make protected copies of its videos.

The same two tools as in the Live tab (a crop that keeps only part of the picture, black masks over faces and name tags), drawn after the
recording on the full picture. The protected copies go in the trial's review folder (`limbtrack/privacy_fix.py`); the trial's own
`raw.mp4` and `overlay.mp4` are never written, and no tracked position and no number changes.

Drag on the picture: with *Crop* chosen the rectangle becomes the crop (a new drag replaces it), with *Mask* it is added as a black mask
(at most `MAX_MASKS`). Dots are drawn small on the picture so that a crop or mask that would cut one off can be seen before the copies are made.
"""
import cv2
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QRadioButton, QSizePolicy, QSlider,
                               QVBoxLayout, QWidget)

from . import privacy, privacy_fix
from .engine import Engine

MAX_MASKS = Engine.MAX_MASKS
CROP_COLOUR, MASK_COLOUR, DOT_COLOUR, HIT_COLOUR = "#f59e0b", "#111827", "#22c55e", "#ef4444"


class PictureDraw(QWidget):
    """The picture of one frame with the crop and masks drawn over it; a drag with the mouse gives a rectangle in picture pixels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.setObjectName("privacyPicture")
        self.image = None                    # QImage of the full picture
        self.crop, self.masks = None, []     # what is drawn, in picture pixels
        self.recorded_crop = None
        self.dots = []                       # [(x, y, hit)] of the frame shown
        self._start = self._now = None       # the drag in picture pixels
        self.on_rect = None                  # called with (x0, y0, x1, y1) when a drag ends

    # ---- geometry
    def _scale(self):
        if self.image is None:
            return 1.0, 0.0, 0.0
        s = min(self.width() / self.image.width(), self.height() / self.image.height())
        return s, (self.width() - s * self.image.width()) / 2.0, (self.height() - s * self.image.height()) / 2.0

    def to_picture(self, pos):
        s, ox, oy = self._scale()
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    def to_widget(self, x, y):
        s, ox, oy = self._scale()
        return QPointF(ox + x * s, oy + y * s)

    def _rect(self, r):
        a, b = self.to_widget(r[0], r[1]), self.to_widget(r[0] + r[2], r[1] + r[3])
        return QRectF(a, b)

    # ---- mouse
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.image is not None:
            self._start = self._now = self.to_picture(e.position())
            self.update()

    def mouseMoveEvent(self, e):
        if self._start is not None:
            self._now = self.to_picture(e.position())
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._start is not None:
            x0, y0 = self._start
            x1, y1 = self.to_picture(e.position())
            self._start = self._now = None
            self.update()
            if self.on_rect is not None:
                self.on_rect(x0, y0, x1, y1)

    # ---- drawing
    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1f2937"))
        if self.image is None:
            return
        s, ox, oy = self._scale()
        p.drawImage(QRectF(ox, oy, s * self.image.width(), s * self.image.height()), self.image)
        if self.recorded_crop:                                               # what the video holds at all: the rest is black already
            p.setPen(QPen(QColor("#9ca3af"), 1, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(self._rect(self.recorded_crop))
        if self.crop:                                                        # outside the crop is shaded: it will not be in the copy
            shade = QColor(0, 0, 0, 150)
            full = QRectF(ox, oy, s * self.image.width(), s * self.image.height())
            c = self._rect(self.crop)
            for r in (QRectF(full.left(), full.top(), full.width(), c.top() - full.top()),
                      QRectF(full.left(), c.bottom(), full.width(), full.bottom() - c.bottom()),
                      QRectF(full.left(), c.top(), c.left() - full.left(), c.height()),
                      QRectF(c.right(), c.top(), full.right() - c.right(), c.height())):
                p.fillRect(r, shade)
            p.setPen(QPen(QColor(CROP_COLOUR), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(c)
        for m in self.masks:
            p.fillRect(self._rect(m), QColor(MASK_COLOUR))
            p.setPen(QPen(QColor("#e5e7eb"), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(self._rect(m))
        for x, y, hit in self.dots:
            p.setPen(QPen(QColor(HIT_COLOUR if hit else DOT_COLOUR), 2))
            p.setBrush(Qt.NoBrush)
            c = self.to_widget(x, y)
            p.drawEllipse(c, 7, 7)
        if self._start is not None and self._now is not None:
            p.setPen(QPen(QColor("white"), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            a, b = self.to_widget(*self._start), self.to_widget(*self._now)
            p.drawRect(QRectF(a, b).normalized())


class PrivacyDialog(QDialog):
    """Draw the crop and masks of one trial and make its protected copies. `trial` is a review.Trial (its video is read, nothing is written
    except in the review layer). `changed` is called after the fix or the copies changed, so Review can show its checks again."""

    def __init__(self, trial, parent=None, changed=None):
        super().__init__(parent)
        self.trial = trial
        self.changed = changed
        self.setWindowTitle(f"Privacy: {trial.base}")
        self.setModal(False)
        self.resize(900, 640)
        self.meta = trial.meta
        self.recorded_crop, self.recorded_masks = privacy_fix.recorded(self.meta)
        self.full_size = tuple(trial.full_size) if trial.full_size else None
        fix = privacy_fix.read(trial.meta_path)
        self.crop, self.masks = privacy_fix.drawn(fix)
        self.crop = tuple(self.crop) if self.crop else None
        self.masks = [tuple(m) for m in self.masks]
        self.index = 0

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("privacyInfo")
        self.picture = PictureDraw()
        self.picture.on_rect = self.rect_drawn
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(len(trial.rows) - 1, 0))
        self.slider.valueChanged.connect(self.show_frame)
        self.frame_label = QLabel()
        self.mode_crop = QRadioButton("Crop (keep only this part)")
        self.mode_mask = QRadioButton("Mask (black rectangle)")
        self.mode_mask.setChecked(True)
        self.modes = QButtonGroup(self)
        self.modes.addButton(self.mode_crop)
        self.modes.addButton(self.mode_mask)
        self.btn_clear_crop = QPushButton("Remove crop")
        self.btn_undo_mask = QPushButton("Remove last mask")
        self.btn_clear_all = QPushButton("Clear drawing")
        self.btn_make = QPushButton("Make protected copies")
        self.btn_make.setDefault(True)
        self.btn_close = QPushButton("Close")
        self.btn_clear_crop.clicked.connect(self.remove_crop)
        self.btn_undo_mask.clicked.connect(self.remove_last_mask)
        self.btn_clear_all.clicked.connect(self.clear_drawing)
        self.btn_make.clicked.connect(self.make_copies)
        self.btn_close.clicked.connect(self.close)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        self.problem.setObjectName("privacyProblem")
        self.note = QLabel("The copies are saved in this trial's review folder as <trial>_raw_protected.mp4 and <trial>_overlay_protected.mp4. "
                           "The trial's own raw.mp4 and overlay.mp4 are not changed and stay on the disk until you decide what to do with them. "
                           "Nothing is deleted: a copy that a new drawing replaces is moved to the folder 'superseded'. No dot position and no number changes.")
        self.note.setWordWrap(True)
        self.note.setObjectName("privacyNote")

        modes = QHBoxLayout()
        for w in (self.mode_crop, self.mode_mask):
            modes.addWidget(w)
        modes.addStretch(1)
        for w in (self.btn_clear_crop, self.btn_undo_mask, self.btn_clear_all):
            modes.addWidget(w)
        scrub = QHBoxLayout()
        scrub.addWidget(self.slider, 1)
        scrub.addWidget(self.frame_label)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(self.btn_make)
        bottom.addWidget(self.btn_close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addLayout(modes)
        lay.addWidget(self.picture, 1)
        lay.addLayout(scrub)
        lay.addWidget(self.problem)
        lay.addWidget(self.note)
        lay.addLayout(bottom)
        self.show_frame(0)

    # ---- the picture
    def _frame_image(self, i):
        img = self.trial.frame(i)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if self.full_size is None:
            self.full_size = (w, h)
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def show_frame(self, i):
        self.index = int(i)
        self.picture.image = self._frame_image(self.index)
        self.picture.recorded_crop = self.recorded_crop
        self.frame_label.setText(f"picture {self.index + 1} of {len(self.trial.rows)}")
        self.refresh()

    def _dots_in(self, i):
        out = []
        if not 0 <= i < len(self.trial.rows):
            return out
        r = self.trial.rows[i]
        for d in self.trial.dot_names:
            if r.get(f"{d}_x", "") == "" or r.get(f"{d}_lost") == "1":
                continue
            try:
                out.append((d, float(r[f"{d}_x"]), float(r[f"{d}_y"])))
            except (KeyError, ValueError):
                continue
        return out

    def problems(self):
        """Dots the drawing would cut off or cover, over the whole trial: {dot: {...}} (`privacy_fix.dot_problems`)."""
        return privacy_fix.dot_problems(self.trial.rows, self.trial.dot_names, self.crop, self.masks)

    def refresh(self):
        pic = self.picture
        pic.crop, pic.masks = self.crop, list(self.masks)
        hit = set(self.problems())
        pic.dots = [(x, y, d in hit) for d, x, y in self._dots_in(self.index)]
        pic.update()
        st = privacy_fix.status(self.trial.meta_path, self.meta)
        self.info.setText(self._info_text(st))
        drawn = bool(self.crop or self.masks)
        self.btn_make.setEnabled(drawn)
        self.btn_clear_crop.setEnabled(self.crop is not None)
        self.btn_undo_mask.setEnabled(bool(self.masks))
        self.btn_clear_all.setEnabled(drawn)
        lines = []
        for d, c in self.problems().items():
            what = []
            if c["inside_mask"]:
                what.append(f"under a mask in {c['inside_mask']} pictures")
            if c["at_edge"]:
                what.append(f"outside or at the edge of the crop in {c['at_edge']} pictures")
            lines.append(f"The {d} dot is " + " and ".join(what) + ".")
        if lines:
            lines.append("The copy hides it there (the numbers in the CSV are not changed). Check that this is what you want.")
        if privacy_fix.outside_recorded_crop(self.crop, self.recorded_crop):
            lines.append("The crop reaches outside the crop this video was recorded with: that part was never saved and comes out black.")
        self.problem.setText("\n".join(lines))
        self.problem.setVisible(bool(lines))

    def _info_text(self, st):
        extra = ""
        if self.recorded_crop or self.recorded_masks:
            extra = " Already applied when recording: " + ", ".join(
                (["a crop"] if self.recorded_crop else []) + ([f"{len(self.recorded_masks)} mask(s)"] if self.recorded_masks else [])) + " (dotted line)."
        return st["text"] + extra

    # ---- drawing
    def rect_drawn(self, x0, y0, x1, y1):
        if self.full_size is None:
            return
        r = privacy.snap_rect(x0, y0, x1, y1, self.full_size[0], self.full_size[1])
        if r is None:
            self.problem.setText(f"That rectangle is too small (at least {privacy.MIN_SIDE_PX} px each way). Drag a larger one.")
            self.problem.setVisible(True)
            return
        if self.mode_crop.isChecked():
            self.crop = r
        else:
            if len(self.masks) >= MAX_MASKS:
                self.problem.setText(f"At most {MAX_MASKS} masks. Remove one first.")
                self.problem.setVisible(True)
                return
            self.masks.append(r)
        self.refresh()

    def remove_crop(self):
        self.crop = None
        self.refresh()

    def remove_last_mask(self):
        if self.masks:
            self.masks.pop()
        self.refresh()

    def clear_drawing(self):
        self.crop, self.masks = None, []
        self.refresh()

    # ---- the copies
    def make_copies(self):
        if not (self.crop or self.masks):
            return
        total_videos = sum(1 for k in privacy_fix.KINDS if privacy_fix.original_path(self.trial.meta_path, self.meta, k) is not None)
        if not total_videos:
            QMessageBox.warning(self, "No video", "This trial has no saved video to protect.")
            return
        progress = QProgressDialog("Making the protected copies...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        done_kinds = []

        def step(kind, done, total):
            progress.setLabelText(f"Making the protected {kind} video ({done} of {total or '?'})...")
            progress.setMaximum(total or 0)
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            result = privacy_fix.make_copies(self.trial.meta_path, self.meta, self.crop, self.masks, progress=step)
            done_kinds = [k for k, n in result.items() if n]
        except Exception as e:  # noqa: BLE001 - shown to the operator
            progress.close()
            QMessageBox.warning(self, "Not made", f"The protected copies could not be made:\n{e}")
            self.refresh()
            return
        progress.close()
        self.refresh()
        if self.changed:
            self.changed()
        st = privacy_fix.status(self.trial.meta_path, self.meta)
        QMessageBox.information(self, "Protected copies", ("Made: " + ", ".join(done_kinds) + ". " if done_kinds else "") + st["text"])
