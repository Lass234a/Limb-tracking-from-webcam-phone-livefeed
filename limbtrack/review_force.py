"""The force numbers of an open trial as a tree (0.32.0): five bold numbers (the peak force, the force above rest, the plateau mean, the peak
torque and the load cell's plus or minus), each folding open to what it is made of, with the things that change it as real controls inside
the rows (the sliding window, the smoothing, the picture delay and the zero). Until 0.31.0 these were the force cards, a block of force
settings under the graphs and the Force tab (`ForcePanel` in review_panels.py).

    ForceTree(controls)   the widget. `controls` are the live widgets the Review tab owns and has connected (see CONTROLS); they are moved
                          into the rows once and stay alive, so the rows are built once and only their texts change.

Nothing here reads a file or changes a trial; the tree only shows what it is given (the same arguments as `ForcePanel.set_data`). `text()` is a
plain-text copy of what a person can read in it, so a test can look for a fact without knowing where it sits on the screen.

The picture keys (the arrows and , . to step) are installed by the Review tab: nothing here may keep the keyboard. The tree never takes it, and
a spin box gives it back when a number was typed or its little arrows were used (as `review_fixdots._TimeSpin` does).
"""
import math

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QHBoxLayout, QHeaderView, QLabel, QMenu, QPushButton, QSizePolicy,
                               QStyle, QStyleOptionSpinBox, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import analysis
from .loadcell import COVERAGE, GRAVITY
from .review_panels import UNKNOWN, _plateau_percent, num, strong_colour
from .review_style import ACCENT, WARN, ElidedLabel, border_colour, muted_colour, tint

TITLE = "Force numbers and what they depend on"
METRICS = ("Peak force", "Above rest", "Plateau mean", "Peak torque", "Load cell")      # the bold rows, in order
OPEN_AT_START = ("Peak force",)                                                          # the rows that are open until someone folds them
CONTROLS = ("sliding", "smoothing", "delay", "zero", "apply", "zero_stretch", "zero_rest", "zero_reset", "zero_label")
ZERO_FROM = (("zero_stretch", "grey stretch"), ("zero_rest", "rest before the push"), ("zero_reset", "recorded zero"))
ZERO_WORDS = {"typed": "typed in Review", "stretch": "grey stretch", "rest": "rest before the push", "recorded": "recorded",
              "resting level": "resting level, none was taken", "none": "none"}          # Trial.zero_source() in words
POINTER = "Strap and anchor dots: see the angle table"                                   # the line under the torque that points to the angle table
KEY_ROLE = Qt.UserRole + 1                                                               # where a row keeps its name (the first cell may be empty)
VALUE_PT = 14.0                                                                          # the size of the number of a bold row
VALUE_WIDTH, NAME_SHARE, NAME_MIN, NAME_MAX = 100, 0.33, 150, 190                        # the column widths (the third gets the rest)
ROW_HEIGHT, TEXT_ROW_HEIGHT = 28, 24                                                     # a row that holds a control, a row of text only


# ------------------------------------------------------------------------------------------------ small helpers
def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _font(base, bold=False, points=None, shrink=0.0):
    f = QFont(base)
    f.setBold(bold)
    if points is not None:
        f.setPointSizeF(points)
    elif shrink:
        size = f.pointSizeF()
        f.setPointSizeF(max((size if size > 0 else 9.0) - shrink, 7.0))
    return f


def _blend(a, b, share):
    """`share` of colour a over colour b, as a QColor."""
    a, b = QColor(a), QColor(b)
    return QColor(round(a.red() * share + b.red() * (1 - share)), round(a.green() * share + b.green() * (1 - share)),
                  round(a.blue() * share + b.blue() * (1 - share)))


def warning_texts(fs):
    """The sentences that go on top: 50 Hz hum over the limit, the note about the rest before the push, a Zero 2 that does not fit it."""
    out = []
    hum = fs["mains_hum_pct_of_peak"]
    if not _nan(hum) and hum > analysis.HUM_WARN_PCT:
        out.append(f"50 Hz mains interference: {fs['mains_hum_v'] * 1000:.0f} mV = {hum:.0f} % of the rise above rest "
                   "(check shielding and grounding; a smoothing window that is a multiple of 20 ms removes it).")
    if fs["rest_note"]:
        out.append(fs["rest_note"])
    mismatch = analysis.relaxed_mismatch(fs["unit"], fs["relaxed"], fs["rest_mean"], fs["peak_minus_rest"])
    if mismatch:
        out.append(mismatch)
    return out


def torque_reason(trial, unit):
    """Why there is no torque, in the words ForcePanel used."""
    return ("the force is not calibrated" if unit != "N" else "there is no ruler scale" if not trial.px_per_cm
            else "the anchor dot was not used" if "anchor" not in trial.tracks else "no moment arm could be measured")


# ------------------------------------------------------------------------------------------------ parts that sit in the rows
class _Cell(QWidget):
    """A small box that holds one control (or two) inside one cell of the tree. `layoutChanged` is sent when something inside asks for a new size
    (a label whose text was changed by the Review tab), so the row can be made as tall as the text needs."""

    layoutChanged = Signal()

    def __init__(self, *widgets, margins=(2, 1, 4, 1), stretch=False, parent=None):
        super().__init__(parent)
        self._height = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(*margins)
        lay.setSpacing(5)
        for w in widgets:
            lay.addWidget(w)
        if stretch:
            lay.addStretch(1)
        self.setMinimumWidth(0)

    def pin_height(self, h):
        """The tree makes a row as tall as its widget's size hint asks; a wrapping label or an opened warning would ask for the height of a
        narrow guess, so the tree says how tall the row really is."""
        if self._height != h:
            self._height = h
            self.updateGeometry()

    def sizeHint(self):
        s = super().sizeHint()
        return s if self._height is None else QSize(s.width(), self._height)

    def minimumSizeHint(self):
        s = super().minimumSizeHint()
        return s if self._height is None else QSize(s.width(), self._height)

    def event(self, e):
        if e.type() == QEvent.LayoutRequest:
            self.layoutChanged.emit()
        return super().event(e)


class _WarningRow(QFrame):
    """One amber line at the top of the tree; clicked, it shows the whole sentence over several lines."""

    changed = Signal()

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setObjectName("forceWarning")
        self.text = text
        self.setToolTip(text)
        self.setCursor(Qt.PointingHandCursor)
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.expanded = False
        self.label = QLabel(text, self)
        self.label.setObjectName("forceWarningText")
        self.label.setWordWrap(False)
        self.label.setTextInteractionFlags(Qt.NoTextInteraction)

    def _text_width(self, width):
        return max(width - 14, 20)

    def heightForWidth(self, width):
        fm = self.label.fontMetrics()
        if not self.expanded:
            return fm.height() + 6
        return fm.boundingRect(0, 0, self._text_width(width), 10000, int(Qt.TextWordWrap), self.text).height() + 6

    def sizeHint(self):
        return QSize(200, self.heightForWidth(200))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.label.setGeometry(10, 3, self._text_width(e.size().width()), e.size().height() - 6)
        self._elide()

    def _elide(self):
        if self.expanded:
            self.label.setWordWrap(True)
            self.label.setText(self.text)
        else:
            self.label.setWordWrap(False)
            self.label.setText(self.label.fontMetrics().elidedText(self.text, Qt.ElideRight, self.label.width()))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.expanded = not self.expanded
            self.changed.emit()
        super().mouseReleaseEvent(e)


class _Tree(QTreeWidget):
    """The tree; `resized` is sent when its viewport changes size (also when its scroll bar comes or goes)."""

    resized = Signal()

    def viewportEvent(self, e):
        done = super().viewportEvent(e)
        if e.type() == QEvent.Resize:
            self.resized.emit()
        return done

    def settle(self):
        """Lay the rows out now (an unfolded row's controls are only shown after this)."""
        self.executeDelayedItemsLayout()
        self.updateEditorGeometries()


class _Watch(QObject):
    """Calls `fn` when the enabled state of a watched widget changes."""

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def eventFilter(self, obj, e):
        if e.type() == QEvent.EnabledChange:
            self.fn()
        return False


class _GiveBack(QObject):
    """A spin box gives the keyboard back to the picture after its little arrows were pressed (typing a number ends with editingFinished)."""

    def eventFilter(self, spin, e):
        if e.type() == QEvent.MouseButtonRelease and self._on_arrows(spin, e.position().toPoint()):
            QTimer.singleShot(0, spin, spin.clearFocus)                     # (after the spin box has finished with the click)
        return False

    @staticmethod
    def _on_arrows(spin, pos):
        opt = QStyleOptionSpinBox()
        opt.initFrom(spin)
        opt.rect = spin.rect()
        opt.buttonSymbols = spin.buttonSymbols()
        opt.frame = spin.hasFrame()
        opt.subControls = QStyle.SC_All
        part = spin.style().hitTestComplexControl(QStyle.CC_SpinBox, opt, pos, spin)
        return part in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown)


# ------------------------------------------------------------------------------------------------ the tree
class ForceTree(QWidget):
    """The force of the analysis window as a tree of bold numbers that fold open (see the top of this file).

    `ForceTree(controls)`: `controls` is a dict of the live widgets of the Review tab: 'sliding', 'smoothing', 'delay' and 'zero' (spin boxes),
    'apply' (the Apply zero button), 'zero_stretch', 'zero_rest' and 'zero_reset' (three buttons the Review tab handles; they are kept as hidden
    children and the 'Zero from' menu presses them) and 'zero_label' (where the zero came from).

    `set_data(trial, t0, t1, fs, ft, fp, picture_delay)` as `ForcePanel.set_data`: `fs` = trial.force_summary(...), `ft` =
    analysis.force_timing(...) (None when there is no clear push) and `fp` = analysis.force_peaks(...). `set_none(text)` for a trial without
    force or a window without samples. `open_zero()` unfolds the zero and gives its box the keyboard. `set_fold(name, open)` folds a row; the
    state of every fold is kept while the widget lives (also over set_data and over other trials). Signal `calibrationRequested` when 'Show the
    calibration graph' is pressed. `rows` maps the five bold names (METRICS) to their items, `items` every row by name, `warnings` are the
    amber lines on top. `text(everything=False)` is what a person can read, as plain lines (everything=True: also the rows of folded numbers)."""

    calibrationRequested = Signal()

    def __init__(self, controls, parent=None):
        super().__init__(parent)
        missing = [k for k in CONTROLS if controls.get(k) is None]
        if missing:
            raise KeyError("ForceTree needs these controls: " + ", ".join(missing))
        self._controls = dict(controls)
        self._css = None
        self._last = None
        self._none = True
        self._syncing = False
        self._fitting = False
        self._open = {name: name in OPEN_AT_START for name in METRICS}
        self.warnings = []
        self._warning_items = []
        self.rows, self.items, self._kind = {}, {}, {}
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setSizePolicy(policy)
        self.setMinimumWidth(0)

        self.title = ElidedLabel(TITLE, self)
        self.title.setObjectName("forceTitle")
        self.title.setFont(_font(self.font(), bold=True))
        self.open_all_link = QPushButton("Open all", self)
        self.open_all_link.setObjectName("forceLink")
        self.open_all_link.setFlat(True)
        self.open_all_link.setCursor(Qt.PointingHandCursor)
        self.open_all_link.setFocusPolicy(Qt.NoFocus)
        self.open_all_link.setToolTip("Unfold every number to what it is made of, or fold them all up again.")
        self.open_all_link.clicked.connect(self._open_all_clicked)
        self.caption = ElidedLabel("", self)
        self.caption.setObjectName("forceCaption")
        self.empty = QLabel("", self)
        self.empty.setObjectName("panelEmpty")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)
        self.empty.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.tree = _Tree(self)
        self._setup_tree()
        self._build()

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.title, 1)
        head.addWidget(self.open_all_link)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addLayout(head)
        lay.addWidget(self.caption)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.empty, 1)

        self._wire_controls()
        self._apply_style()
        self._paint_rows()
        self._sync_folds()
        self.set_none("No trial is open.")

    # ------------------------------------------------------------ what the lead calls
    def set_none(self, text):
        self._last = None
        self._none = True
        self._set_warnings([])
        self.empty.setText(text)
        self.empty.show()
        self.tree.hide()
        self.caption.hide()
        self.open_all_link.hide()

    def set_data(self, trial, t0, t1, fs, ft, fp, picture_delay):
        if fs is None:
            self.set_none("Force was recorded but there are no samples in this window.")
            return
        self._last = (trial, t0, t1, fs, ft, fp, picture_delay)
        if fp is None:
            ft = analysis.force_timing(trial, t0, t1)
            fp = analysis.force_peaks(trial, t0, t1, timing=ft)
        u = fs["unit"]
        d = 3 if u == "V" else 1
        v = lambda x: UNKNOWN if _nan(x) else f"{num(x, d)} {u}"                    # noqa: E731 - a number with its unit, '-' for none
        self._none = False
        self.empty.hide()
        self.caption.show()
        self.tree.show()
        self.open_all_link.show()
        smoothing = f"{fs['smoothing_s'] * 1000:.0f} ms moving average" if fs["smoothing_s"] > 0 else "raw samples"
        self.caption.setText(f"Force in the window: {u}{'' if u == 'N' else ', not calibrated'}, {smoothing}")
        self._set_warnings(warning_texts(fs))
        self._fill_peak(trial, fs, ft, fp, picture_delay, v, smoothing)
        self._fill_rest(trial, fs, v)
        self._fill_plateau(fs, ft, v, u, d)
        self._fill_torque(trial, t0, t1, picture_delay, u)
        self._fill_load_cell(trial, fs, u, v)
        self._sync_folds()
        self._fit()

    def open_zero(self):
        """Unfold 'Above rest' and give the zero box the keyboard (the 'Set zero' check action)."""
        self.set_fold("Above rest", True)
        self.tree.settle()
        self.tree.scrollToItem(self.items["Zero"])
        spin = self._controls["zero"]
        spin.setFocus(Qt.OtherFocusReason)
        spin.selectAll()

    def set_fold(self, name, open):  # noqa: A002 - the word the Review tab uses
        if name not in self._kind or self._kind[name] != "fold":
            raise KeyError(f"'{name}' is not a row that folds: {', '.join(n for n, k in self._kind.items() if k == 'fold')}")
        self._open[name] = bool(open)
        self._sync_folds()

    def text(self, everything=False):
        """Plain lines of everything a person can read: the title, the caption, the warnings in full, then each row as 'name: value  note' (the
        rows under a folded number only with everything=True)."""
        if self._none:
            return self.empty.text()
        lines = [f"{self.title.text()}  [{self.open_all_link.text()}]", self.caption.text()]
        lines += [f"Warning: {w.text}" for w in self.warnings]
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item not in self._warning_items:
                self._lines(item, 0, lines, everything)
        return "\n".join(lines)

    # ------------------------------------------------------------ the rows
    def _setup_tree(self):
        t = self.tree
        t.setObjectName("forceTree")
        t.setColumnCount(3)
        t.setHeaderLabels(["", "Value", "How"])
        t.setRootIsDecorated(True)
        t.setIndentation(14)
        t.setAnimated(False)
        t.setUniformRowHeights(False)
        t.setExpandsOnDoubleClick(False)
        t.setSelectionMode(QAbstractItemView.NoSelection)
        t.setFocusPolicy(Qt.NoFocus)                                       # the arrow keys step the picture: the tree must not take them
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)             # (a scroll bar of its own, up and down, only when the room is too short)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        t.setTextElideMode(Qt.ElideRight)
        t.setWordWrap(False)
        t.setMinimumWidth(0)
        t.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        head = t.header()
        head.setSectionResizeMode(QHeaderView.Fixed)
        head.setStretchLastSection(False)
        head.setSectionsMovable(False)
        head.setSectionsClickable(False)
        head.setHighlightSections(False)
        t.itemClicked.connect(self._row_clicked)
        t.itemExpanded.connect(lambda item: self._fold_changed(item, True))
        t.itemCollapsed.connect(lambda item: self._fold_changed(item, False))
        t.resized.connect(self._fit)

    def _add(self, parent, key, kind="leaf", value=None, how=None, bold=False, shown_name=True):
        """One row. `kind`: 'fold' (it has rows under it), 'leaf', 'note' (one muted line over the whole width). `value` / `how` are controls to
        put in those cells. `shown_name` False: the first cell holds a control over the whole width, so no name is written under it."""
        item = QTreeWidgetItem(parent if parent is not None else self.tree, [key if shown_name else "", "", ""])
        item.setFlags(Qt.ItemIsEnabled)
        item.setChildIndicatorPolicy(QTreeWidgetItem.DontShowIndicator)
        self.items[key] = item
        self._kind[key] = kind
        item.setData(0, KEY_ROLE, key)
        small = _font(self.tree.font(), shrink=1.0)
        item.setFont(2, small)
        if bold:
            item.setFont(0, _font(self.tree.font(), bold=True, points=max(self.tree.font().pointSizeF(), 9.0) + 1.0))
            item.setFont(1, _font(self.tree.font(), bold=True, points=VALUE_PT))
        if kind == "note":
            item.setFirstColumnSpanned(True)
        for col, widgets in ((1, value), (2, how)):
            if widgets:
                widgets = widgets if isinstance(widgets, tuple) else (widgets,)
                self.tree.setItemWidget(item, col, _Cell(*widgets, stretch=col == 2))
        if kind != "note":
            item.setSizeHint(0, QSize(0, ROW_HEIGHT if value or how else TEXT_ROW_HEIGHT))
        return item

    def _build(self):
        c = self._controls
        self.zero_from = QToolButton()
        self.zero_from.setObjectName("zeroFrom")
        self.zero_from.setText("Zero from")
        self.zero_from.setToolTip("Take the zero from the grey stretch on the force graph, from the rest before the push, or go back to the zero recorded "
                                  "with the trial.")
        self.zero_from.setPopupMode(QToolButton.InstantPopup)
        self.zero_from.setFocusPolicy(Qt.NoFocus)
        self.zero_menu = QMenu(self.zero_from)
        self.zero_menu.setToolTipsVisible(True)
        self.zero_actions = {}
        for key, text in ZERO_FROM:
            action = QAction(text, self.zero_menu)
            action.setToolTip(c[key].toolTip())
            action.triggered.connect(lambda _checked=False, b=c[key]: b.click())      # (the Review tab handles the click of its own button)
            self.zero_menu.addAction(action)
            self.zero_actions[key] = action
        self.zero_from.setMenu(self.zero_menu)
        self.zero_menu.aboutToShow.connect(self._sync_zero_menu)
        self._watch = _Watch(self._sync_zero_menu, self)
        for key, _text in ZERO_FROM:                                                    # the three buttons stay alive, hidden, and are pressed by the menu
            b = c[key]
            b.setParent(self)
            b.hide()
            b.installEventFilter(self._watch)
        self.button_calibration = QPushButton("Show the calibration graph")
        self.button_calibration.setObjectName("sectionButton")
        self.button_calibration.setToolTip("Show the line the newtons come from, with this trial's peak on it.")
        self.button_calibration.setFocusPolicy(Qt.NoFocus)
        self.button_calibration.clicked.connect(self.calibrationRequested)
        c["zero_label"].setWordWrap(True)
        c["zero_label"].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        c["zero_label"].setMinimumWidth(0)

        peak = self._add(None, "Peak force", "fold", bold=True)
        self._add(peak, "Sliding window", value=c["sliding"])
        self._add(peak, "Smoothing", value=c["smoothing"])
        self._add(peak, "Picture delay", value=c["delay"])
        self._add(peak, "Smoothed peak")
        ttp = self._add(peak, "Time to peak", "fold")
        for key in ("Onset", "10 to 90 % of the rise", "Steepest rise", "To the sliding peak"):
            self._add(ttp, key)
        rest = self._add(None, "Above rest", "fold", bold=True)
        self._add(rest, "Rest before the push")
        self._add(rest, "Zero", value=c["zero"], how=(c["apply"], self.zero_from))
        zero_note = self._add(rest, "Zero comes from", "note", shown_name=False)
        self.tree.setItemWidget(zero_note, 0, _Cell(c["zero_label"], margins=(4, 0, 4, 0)))
        self.tree.itemWidget(zero_note, 0).layoutChanged.connect(self._later_fit)
        self._add(rest, "Relaxed (Zero 2)")
        self._add(rest, "Peak above relaxed")
        plateau = self._add(None, "Plateau mean", "fold", bold=True)
        self._add(plateau, "Held above 90 %")
        self._add(plateau, "Steadiness")
        torque = self._add(None, "Peak torque", "fold", bold=True)
        self._add(torque, "At the peak force")
        self._add(torque, "Plateau")
        self._add(torque, "Moment arm")
        self._add(torque, "Stand-in dot")
        self._add(torque, POINTER, "note")
        load = self._add(None, "Load cell", "fold", bold=True)
        self._add(load, "Calibration")
        self._add(load, "Newtons")
        self._add(load, "Regression")
        self._add(load, "Peak plus or minus")
        self._add(load, "Plateau plus or minus")
        self._add(load, "Peak above the line")
        self._add(load, "Plateau above the line")
        button = self._add(load, "Show the calibration graph", "note", shown_name=False)
        self.tree.setItemWidget(button, 0, _Cell(self.button_calibration, margins=(4, 2, 4, 2), stretch=True))
        for name in METRICS:
            self.rows[name] = self.items[name]
        self._hosted = {"Sliding window": ("sliding", None), "Smoothing": ("smoothing", None), "Picture delay": ("delay", None),
                        "Zero": ("zero", "apply")}
        self._static_notes()

    def _static_notes(self):
        """The notes that never change (what the spin boxes do); the rest are filled by set_data."""
        notes = {"Smoothing": "0 = raw; use steps of 20 ms"}
        for key, note in notes.items():
            self._put(key, note=note)
        pointer = self.items[POINTER]
        pointer.setFont(0, _font(self.tree.font(), shrink=1.0))
        self._put(POINTER, muted=True, tip="The strap and anchor dots the moment arm is measured from are in the angle table.")

    def _wire_controls(self):
        """The keyboard goes back to the picture after a number was typed or the little arrows were used."""
        self._give_back = _GiveBack(self)
        for key in ("sliding", "smoothing", "delay", "zero"):
            spin = self._controls[key]
            spin.setFocusPolicy(Qt.ClickFocus)                             # (a spin box takes the keyboard only when it is clicked, never by itself)
            spin.editingFinished.connect(spin.clearFocus)
            spin.installEventFilter(self._give_back)
        self._controls["apply"].setFocusPolicy(Qt.NoFocus)                  # (a button never takes the keyboard either)

    # ------------------------------------------------------------ putting texts on rows
    def _put(self, key, value=None, note=None, state=None, tip=None, muted=False):
        """Set the texts of a row (None leaves a text as it is). `state` 'warn' colours the value and the note amber; `muted` greys the whole row.
        The tooltip is the whole text (the cells cut what does not fit)."""
        item = self.items[key]
        if value is not None:
            item.setText(1, value)
        if note is not None:
            item.setText(2, note)
        shown = " - ".join(s for s in (item.text(1), item.text(2)) if s)
        tip = tip if tip is not None else shown
        for col in range(3):
            item.setToolTip(col, tip)
        item.setData(0, Qt.UserRole, (state, muted))
        self._colour(item, state, muted)

    def _colour(self, item, state, muted):
        warn, grey = QColor(strong_colour(WARN, QApplication.palette())), QColor(muted_colour(QApplication.palette()))
        item.setForeground(0, QBrush(grey) if muted else QBrush())
        item.setForeground(1, QBrush(warn) if state == "warn" else QBrush(grey) if muted else QBrush())
        item.setForeground(2, QBrush(warn) if state == "warn" else QBrush(grey))

    def _show(self, key, shown):
        self.items[key].setHidden(not shown)

    def _fill_peak(self, trial, fs, ft, fp, picture_delay, v, smoothing):
        if math.isnan(fp["window_peak"]):
            note = f"window shorter than {trial.window_peak_s * 1000:.0f} ms, or gaps in the force"
            self._put("Sliding window", note="no stretch this long in the window", state="warn")
            self._put("Peak force", UNKNOWN, note, "warn", tip=f"There is no sliding-window peak: the analysis window is shorter than the sliding window "
                      f"({trial.window_peak_s * 1000:.0f} ms), or the force has gaps. Make the window longer or the sliding window shorter.")
        else:
            span = f"{fp['window_peak_start_s']:.2f} to {fp['window_peak_end_s']:.2f} s"
            self._put("Sliding window", note=f"best stretch {span}")
            self._put("Peak force", v(fp["window_peak"]), f"{fp['window_peak_ms']:.0f} ms sliding-window average",
                      tip=f"The sliding-window peak: the highest average of the raw force over any {fp['window_peak_ms']:.0f} ms stretch inside the "
                          f"window. This stretch: {span}.")
        angle = fs["main_angle_at_peak_deg"]
        name = f" ({fs['main_angle']})" if fs["main_angle"] else ""
        delay = f", pictures read {fs['picture_delay_s']:+.3f} s after the peak" if fs["picture_delay_s"] else ""
        self._put("Picture delay", note=f"angle at the peak {num(angle)} deg{name}{delay}" if not _nan(angle) else f"angle at the peak not known{delay}",
                  tip="How long after the movement a picture is time-stamped. The angle at the peak force is read from the pictures stamped this long "
                      "after the force peak.")
        raw = f"; raw single sample {v(fs['raw_peak'])}" if fs["smoothing_s"] > 0 else ""
        self._put("Smoothed peak", v(fs["peak"]), f"peak 20 ms {v(fs['peak_20ms'])}{raw}; at {fs['t_peak_s']:.2f} s",
                  tip=f"The highest value of the smoothed force ({smoothing}), at {fs['t_peak_s']:.2f} s. Peak 20 ms is the highest 20 ms average of the "
                      "raw force: one mains cycle, so hum cancels, and it is the same at any Smoothing."
                      + (" The raw single sample is the highest single sample: a noisy sample overstates the peak." if raw else ""))
        timing = ft is not None
        to_window = not _nan(fp["time_to_window_peak_s"])
        to_peak = fp["time_to_peak_s"] if not _nan(fp["time_to_peak_s"]) else ft["time_to_peak_s"] if timing else math.nan
        if not _nan(to_peak):
            self._put("Time to peak", f"{to_peak:.2f} s", "from the onset to the peak",
                      tip="How long the push took from its onset to the highest smoothed force. Open it for the onset, the rise and the sliding-window peak.")
        else:
            self._put("Time to peak", UNKNOWN, "no clear push in the window", "warn")
        for key in ("Onset", "10 to 90 % of the rise", "Steepest rise"):
            self._show(key, timing)
        self._show("To the sliding peak", to_window)
        if to_window:
            self._put("To the sliding peak", f"{fp['time_to_window_peak_s']:.2f} s", "from the onset to the middle of the best stretch")
        if timing:
            d = 3 if fs["unit"] == "V" else 1
            self._put("Onset", f"{ft['onset_s']:.2f} s", "start of the push (10 % of the rise)",
                      tip="The start of the push: where the force was last within 10 % of its rise above the level before the push.")
            self._put("10 to 90 % of the rise", f"{ft['time_to_90pct_s']:.2f} s", "")
            self._put("Steepest rise", f"{num(ft['rise_rate_per_s'], d)} {fs['unit']}/s", "over 100 ms")

    def _fill_rest(self, trial, fs, v):
        zero, where = None, ""
        try:
            zero, where = trial.current_zero_v()
            where = ZERO_WORDS.get(trial.zero_source(), where)
        except Exception:  # noqa: BLE001 - a trial without a zero to tell: the note simply leaves it out
            zero = None
        zero_text = "" if zero is None else f" - zero {zero:.4f} V ({where})"
        rest = v(fs["rest_mean"]) if not _nan(fs["rest_mean"]) else "not known"
        self._put("Above rest", v(fs["peak_minus_rest"]), f"rest {rest}{zero_text}",
                  tip=f"The peak force minus the resting force just before the push. Rest {rest}; the zero in use"
                      + ("" if zero is None else f" is {zero:.4f} V ({where})") + ".")
        self._put("Rest before the push", v(fs["rest_mean"]), f"{fs['rest_basis']}, {fs['rest_seconds']:.1f} s of force",
                  "warn" if fs["rest_note"] else None)
        relaxed = not _nan(fs["relaxed"])
        if relaxed:
            ease = fs["rest_minus_relaxed"]
            note = "taken with the participant relaxed in the strap, before the trial" + (
                "" if _nan(ease) else f"; the rest before the push was {abs(ease):.{3 if fs['unit'] == 'V' else 1}f} {fs['unit']} "
                                      f"{'higher' if ease > 0 else 'lower'} than that")
            self._put("Relaxed (Zero 2)", v(fs["relaxed"]), note)
            self._put("Peak above relaxed", v(fs["peak_minus_relaxed"]), "")
        elif fs["relaxed_note"]:
            self._put("Relaxed (Zero 2)", "not used", fs["relaxed_note"])
        self._show("Relaxed (Zero 2)", relaxed or bool(fs["relaxed_note"]))
        self._show("Peak above relaxed", relaxed)

    def _fill_plateau(self, fs, ft, v, u, d):
        self._put("Plateau mean", v(fs["plateau_mean"]), f"middle {_plateau_percent():.0f} % of the window",
                  tip=f"The mean force over the middle {_plateau_percent():.0f} % of the window.")
        for key in ("Held above 90 %", "Steadiness"):
            self._show(key, ft is not None)
        if ft is not None:
            self._put("Held above 90 %", f"{ft['hold_above_90pct_s']:.2f} s", "of the peak")
            self._put("Steadiness", f"{num(ft['hold_cv_pct'])} %", "SD / mean while held")

    def _fill_torque(self, trial, t0, t1, picture_delay, u):
        tq = analysis.torque_summary(trial, t0, t1, picture_delay)
        children = ("At the peak force", "Plateau", "Moment arm", "Stand-in dot", POINTER)
        if tq is None:
            why = torque_reason(trial, u)
            self.items["Peak torque"].setFont(1, _font(self.tree.font()))
            self._put("Peak torque", "not available", why, muted=True, tip=f"Torque is not available: {why}.")
            for key in children:
                self._show(key, False)
            return
        self.items["Peak torque"].setFont(1, _font(self.tree.font(), bold=True, points=VALUE_PT))
        self._put("Peak torque", f"{tq['peak_nm']:.1f} N.m", f"at {tq['t_peak_s']:.2f} s; force x arm {tq['arm_at_peak_force_cm']:.1f} cm",
                  tip="The highest torque in the window: the force times the moment arm, picture by picture.")
        self._put("At the peak force", f"{tq['at_peak_force_nm']:.1f} N.m", "")
        self._put("Plateau", f"{num(tq['plateau_nm'])} N.m", "")
        self._put("Moment arm", f"{tq['arm_at_peak_force_cm']:.1f} cm", f"at the peak force, mean {tq['mean_arm_cm']:.1f} cm")
        stand_in = tq["pictures_with_stand_in"]
        if stand_in:
            self._put("Stand-in dot", f"{stand_in} of {tq['pictures']}", f"the strap dot was not seen in {stand_in} of {tq['pictures']} pictures: "
                      "the ankle dot stood in for it", "warn")
        for key in children:
            self._show(key, key != "Stand-in dot" or bool(stand_in))

    def _fill_load_cell(self, trial, fs, u, v):
        reg, c = trial.regression, trial.force_cal or {}
        children = ("Calibration", "Newtons", "Regression", "Peak plus or minus", "Plateau plus or minus", "Peak above the line",
                    "Plateau above the line", "Show the calibration graph")
        shown = {"Calibration": True, "Show the calibration graph": True}
        self._put("Calibration", "recorded" if u == "N" else "none", trial.force_cal_note, "warn" if u != "N" else None)
        if u != "N":
            self._put("Load cell", UNKNOWN, "not calibrated: volts only", tip="The force is in volts, so there is no precision to give yet.")
            self._put("Newtons", "not yet", "no zero was taken: use Apply zero to get newtons")
            shown["Newtons"] = True
        elif reg is None:
            self._put("Load cell", UNKNOWN, "no regression stored: no precision can be given",
                      tip="The calibration has no regression stored with it, so no precision can be given.")
            self._put("Regression", "none stored", f"{c.get('method', 'unknown calibration')}, {c.get('n_per_v', 0):.2f} N per volt: "
                      "no precision can be given")
            shown["Regression"] = True
        else:
            date = getattr(reg, "date", "")
            label = f"calibration {reg.id}" + (f", {date}" if date and date != reg.id else "")
            sd_peak = COVERAGE * float(reg.sd_newton(fs["peak"])) if not _nan(fs["peak"]) else math.nan
            self._put("Load cell", UNKNOWN if _nan(sd_peak) else f"± {sd_peak:.1f} N", label,
                      tip="How far off the peak can be: the load cell's own scatter at the peak, at about 95 % certainty, from the calibration line "
                          f"({label}).")
            self._put("Regression", str(reg.id), f"{c.get('n_per_v', 0):.2f} N per volt, zero {c.get('offset_v', 0):+.4f} V")
            shown["Regression"] = True
            lo, hi = reg.range_kg
            for name, key in (("Peak", "peak"), ("Plateau", "plateau_mean")):
                f = fs[key]
                if _nan(f):
                    continue
                sd = COVERAGE * float(reg.sd_newton(f))
                kg = f / GRAVITY
                self._put(f"{name} plus or minus", f"{sd:.1f} N", f"{100 * sd / max(abs(f), 1e-9):.2f} % at about 95 % ({kg:.1f} kg)")
                shown[f"{name} plus or minus"] = True
                if kg > hi * 1.001:
                    self._put(f"{name} above the line", f"{kg:.0f} kg", f"the heaviest calibration load is {hi:g} kg: the line is extended there, "
                              "not measured", "warn")
                    shown[f"{name} above the line"] = True
        for key in children:
            self._show(key, shown.get(key, False))

    # ------------------------------------------------------------ warnings
    def _set_warnings(self, texts):
        if [w.text for w in self.warnings] == texts:
            return
        for item in self._warning_items:
            self.tree.removeItemWidget(item, 0)
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
        self.warnings, self._warning_items = [], []
        for i, text in enumerate(texts):
            w = _WarningRow(text)
            w.changed.connect(self._warning_toggled)
            item = QTreeWidgetItem(["", "", ""])
            item.setFlags(Qt.ItemIsEnabled)
            self.tree.insertTopLevelItem(i, item)
            item.setFirstColumnSpanned(True)
            self.tree.setItemWidget(item, 0, _Cell(w, margins=(4, 2, 4, 2)))
            self.warnings.append(w)
            self._warning_items.append(item)
        self._fit_rows()

    def _warning_toggled(self):
        for w in self.warnings:
            w._elide()
        self._fit_rows()

    # ------------------------------------------------------------ folds
    def _foldable(self):
        """The names of the rows that have something under them just now."""
        return [n for n, k in self._kind.items() if k == "fold" and self.items[n].childIndicatorPolicy() == QTreeWidgetItem.ShowIndicator]

    def _sync_folds(self):
        """Fold every row as it was last left (a row with nothing under it just now has no arrow and stays shut; what was chosen is kept)."""
        self._syncing = True
        try:
            for name, kind in self._kind.items():
                if kind != "fold":
                    continue
                item = self.items[name]
                has = any(not item.child(i).isHidden() for i in range(item.childCount()))
                item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator if has else QTreeWidgetItem.DontShowIndicator)
                item.setExpanded(has and self._open.setdefault(name, False))
        finally:
            self._syncing = False
        self._update_link()
        self._later_fit()

    def _fold_changed(self, item, is_open):
        if self._syncing:
            return
        self._open[item.data(0, KEY_ROLE)] = is_open
        self._update_link()
        self._later_fit()

    def _row_clicked(self, item, _column):
        """A click anywhere on a row that folds opens or closes it (not only on the little arrow)."""
        if self._kind.get(item.data(0, KEY_ROLE)) == "fold" and item.childIndicatorPolicy() == QTreeWidgetItem.ShowIndicator:
            item.setExpanded(not item.isExpanded())

    def _all_open(self):
        names = self._foldable()
        return bool(names) and all(self.items[n].isExpanded() for n in names)

    def _update_link(self):
        self.open_all_link.setText("Close all" if self._all_open() else "Open all")

    def _open_all_clicked(self):
        target = not self._all_open()
        for name in self._foldable():
            self._open[name] = target
        self._sync_folds()

    # ------------------------------------------------------------ the zero menu
    def _sync_zero_menu(self):
        """The three entries are as enabled as the buttons they press."""
        for key, _text in ZERO_FROM:
            self.zero_actions[key].setEnabled(self._controls[key].isEnabled())
        self.zero_from.setEnabled(any(a.isEnabled() for a in self.zero_actions.values()))

    # ------------------------------------------------------------ the size of things
    def _later_fit(self):
        QTimer.singleShot(0, self, self._fit)                              # (once the rows have been laid out; never after the widget is gone)

    def _fit(self):
        """The three columns for this width, and the rows that wrap as tall as their text needs."""
        vp = self.tree.viewport().width()
        if self._fitting or vp <= 0:
            return
        name_w = max(NAME_MIN, min(NAME_MAX, int(vp * NAME_SHARE)))
        head = self.tree.header()
        for col, w in enumerate((name_w, VALUE_WIDTH, max(vp - name_w - VALUE_WIDTH, 40))):
            if head.sectionSize(col) != w:
                head.resizeSection(col, w)
        self._fit_rows()

    def _fit_rows(self):
        """Warnings and the 'zero comes from' line wrap to the room they have; their rows are as tall as that needs."""
        if self._fitting:
            return
        self._fitting = True
        try:
            vp = self.tree.viewport().width()
            if vp <= 0:
                return
            indent = self.tree.indentation()
            for item, w in zip(self._warning_items, self.warnings):
                self._height(item, w.heightForWidth(vp - indent - 8) + 4, vp)
            label = self._controls["zero_label"]
            note = self.items["Zero comes from"]
            note.setHidden(not label.text())
            if label.text():
                room = vp - 3 * indent - 8
                self._height(note, label.heightForWidth(room) + 2, vp)
        finally:
            self._fitting = False

    def _height(self, item, h, width):
        """The row of `item` is h tall (its widget says so too: the tree takes the taller of the two)."""
        self.tree.itemWidget(item, 0).pin_height(h)
        if item.sizeHint(0).height() != h:
            item.setSizeHint(0, QSize(width, h))

    def sizeHint(self):
        return QSize(520, 420)

    def minimumSizeHint(self):
        return QSize(300, 140)

    # ------------------------------------------------------------ what a person can read
    def _lines(self, item, depth, lines, everything):
        if item.isHidden():
            return
        name = item.data(0, KEY_ROLE)
        pad = "  " * depth
        kind = self._kind.get(name)
        if name in self._hosted:
            value, how = self._hosted[name]
            note = f"[{self._controls['apply'].text()}]  [{self.zero_from.text()}: " + ", ".join(t for _k, t in ZERO_FROM) + "]" if how else item.text(2)
            lines.append(f"{pad}{name}: {self._controls[value].text()}" + (f"  {note}" if note else ""))
        elif name == "Zero comes from":
            if self._controls["zero_label"].text():
                lines.append(f"{pad}{self._controls['zero_label'].text()}")
        elif name == "Show the calibration graph":
            lines.append(f"{pad}[{self.button_calibration.text()}]")
        elif kind == "note":
            lines.append(f"{pad}{name}")
        else:
            lines.append(f"{pad}{name}: {item.text(1)}" + (f"  {item.text(2)}" if item.text(2) else ""))
        if item.childCount() and (everything or item.isExpanded()):
            for i in range(item.childCount()):
                self._lines(item.child(i), depth + 1, lines, everything)

    # ------------------------------------------------------------ the look
    def _paint_rows(self):
        """The bold rows have a quiet tint across the row (the palette's text colour a little over its background)."""
        pal = QApplication.palette()
        band = QBrush(_blend(pal.color(QPalette.WindowText), pal.color(QPalette.Base), 0.06))
        for name in METRICS:
            for col in range(3):
                self.items[name].setBackground(col, band)

    def _recolour(self):
        for item in self.items.values():
            state, muted = item.data(0, Qt.UserRole) or (None, False)
            self._colour(item, state, muted)

    def _apply_style(self):
        pal = QApplication.palette()
        muted, border = muted_colour(pal), border_colour(pal)
        css = f"""
QLabel#forceCaption, QLabel#panelEmpty, QLabel#hint {{ color: {muted}; }}
QPushButton#forceLink {{ border: none; background: transparent; color: {ACCENT}; padding: 1px 4px; }}
QPushButton#forceLink:hover {{ text-decoration: underline; }}
QPushButton#sectionButton {{ border: 1px solid {border}; border-radius: 4px; padding: 2px 10px; }}
QPushButton#sectionButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QToolButton#zeroFrom {{ padding: 1px 20px 1px 9px; }}
QToolButton#zeroFrom::menu-indicator {{ subcontrol-origin: padding; subcontrol-position: center right; right: 6px; }}
QTreeWidget#forceTree {{ border: 1px solid {border}; border-radius: 6px; }}
QTreeWidget#forceTree QHeaderView::section {{ background: transparent; color: {muted}; border: none; border-bottom: 1px solid {border}; padding: 2px 6px; }}
QFrame#forceWarning {{ background: {tint(WARN, 0.14)}; border: 1px solid {tint(WARN, 0.5)}; border-radius: 5px; }}
QLabel#forceWarningText {{ color: {strong_colour(WARN, pal, 0.8)}; background: transparent; border: none; }}
"""
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()
            self._paint_rows()
            self._recolour()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._later_fit()
