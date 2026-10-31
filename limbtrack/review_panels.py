"""The numbers of an open trial as panels instead of text (0.32.0): the angles, with the angle at the peak force first, and the force, in
sections that re-flow with the width and never need a scroll bar. Until 0.31.0 these were monospace text in tabs.

    FactGrid    rows of "label  value  note" (the building block of both panels)
    AnglePanel  the angle at the peak force large, a few plain facts, a table of every angle, and the reference readings
    ForcePanel  Peak, Level, Timing, Torque and Load cell as small sections, warnings on top

Nothing here reads a file or changes a trial; the panels only show what they are given. Every panel has `text()`: a plain-text dump of what a
person can read in it (one line per row, 'label: value  note'), so a test can look for a fact without knowing where it sits on the screen.
"""
import itertools
import math

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QBoxLayout, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import analysis, reference
from .loadcell import COVERAGE, GRAVITY
from .review_style import ACCENT, STATE_COLOURS, WARN, ElidedLabel, border_colour, muted_colour, tint

UNKNOWN = "-"


# ------------------------------------------------------------------------------------------------ small helpers
def strong_colour(colour, pal=None, share=0.72):
    """`colour` pulled toward the palette's text colour, for text on a pale background: a darker green or amber in a light theme, a lighter
    one in a dark theme (the plain GOOD and WARN are too faint as small text on white)."""
    pal = pal or QApplication.palette()
    c, t = QColor(colour), pal.color(QPalette.WindowText)
    return QColor(round(c.red() * share + t.red() * (1 - share)), round(c.green() * share + t.green() * (1 - share)),
                  round(c.blue() * share + t.blue() * (1 - share))).name()


def num(v, d=1, signed=False):
    """A number for the screen: '-' for nothing or NaN, never '-0.0'."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return UNKNOWN
    if math.isnan(x):
        return UNKNOWN
    x = round(x, d) + 0.0
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _small(font):
    f = QFont(font)
    size = f.pointSizeF()
    f.setPointSizeF(max((size if size > 0 else 9.0) - 1.0, 7.0))
    return f


def _bold(font):
    f = QFont(font)
    f.setBold(True)
    return f


def _plateau_percent():
    """How much of the window the hold angle and the plateau force are taken from (the middle part): read from review.py when it is loaded."""
    try:
        from . import review
        return 100 - 200 * review.PLATEAU_TRIM
    except Exception:  # noqa: BLE001 - only a wording
        return 60


class _Note(QLabel):
    """A few lines of small muted text that wrap to the width they are given (and never widen the window). `text()` is the whole text."""

    def __init__(self, text="", parent=None, name="panelHint"):
        super().__init__(text, parent)
        self.setObjectName(name)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        self.setFont(_small(self.font()))


# ------------------------------------------------------------------------------------------------ FactGrid
class FactGrid(QWidget):
    """Rows of a muted label, a bold value and a small muted note. `set_rows([(label, value, note, state)])`: `state` None, 'good', 'warn' or
    'bad' colours the value. A note goes beside the value when the whole of it fits there; otherwise on its own lines under the row (wrapped,
    so nothing is cut off). A very long label or value is cut with '...' and its whole text is the tooltip: the grid never widens its parent.
    It lays its labels out itself so that its height can be worked out for a width (`heightForWidth`). `text()` gives one line per row,
    'label: value  note'."""

    GAP = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._labels, self._values, self._notes = [], [], []
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumWidth(100)
        self._css = None
        self._apply_style()

    # ------------------------------------------------------------ what is shown
    @property
    def rows(self):
        return list(self._rows)

    def value_text(self, i):
        return self._values[i].text()

    def label_text(self, i):
        return self._labels[i].text()

    def note_text(self, i):
        return self._notes[i].text()

    def text(self, everything=False):
        """One line per row: 'label: value  note' (the note left out when there is none)."""
        return "\n".join(f"{label}: {value}" + (f"  {note}" if note else "") for label, value, note, _state in self._rows)

    def set_rows(self, rows):
        rows = [tuple(r) + (None,) * (4 - len(r)) for r in rows]
        if rows == self._rows:
            return
        self._rows = rows
        while len(self._labels) < len(rows):
            self._labels.append(self._make_label("factLabel", self.font()))
            self._values.append(self._make_label("factValue", _bold(self.font())))
            note = _Note("", self, "factNote")
            self._notes.append(note)
        for i, (label, value, note, state) in enumerate(rows):
            tip = f"{label}: {value}" + (f"\n{note}" if note else "")
            self._labels[i].setText(label, tip)
            self._values[i].setText(str(value), tip)
            self._notes[i].setText(note or "")
            self._notes[i].setToolTip(tip)
            for w in (self._labels[i], self._values[i], self._notes[i]):
                w.show()
        for i in range(len(rows), len(self._labels)):
            for w in (self._labels[i], self._values[i], self._notes[i]):
                w.hide()
        self._colour_values()
        self.updateGeometry()
        self._place(self.width())

    def _make_label(self, name, font):
        w = ElidedLabel("", self)
        w.setObjectName(name)
        w.setFont(font)
        w.setMinimumWidth(0)
        w.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return w

    def _colour_values(self):
        pal = QApplication.palette()
        for (_label, _value, _note, state), w in zip(self._rows, self._values):
            colour = STATE_COLOURS.get(state)
            w.setStyleSheet(f"color: {strong_colour(colour, pal)};" if colour else "")

    # ------------------------------------------------------------ layout
    def _plan(self, width):
        """(label width, value width, room beside the value, row height, [(note beside?, height of the note under the row)])."""
        rows = self._rows
        fl, fv, fn = self.fontMetrics(), self.fontMetrics(), self.fontMetrics()
        if self._labels:
            fl, fv, fn = self._labels[0].fontMetrics(), self._values[0].fontMetrics(), self._notes[0].fontMetrics()
        label_nat = max((fl.horizontalAdvance(r[0]) for r in rows), default=0) + 2
        value_nat = max((fv.horizontalAdvance(str(r[1])) for r in rows), default=0) + 2
        lw = min(label_nat, int(width * 0.45))
        vw = max(min(value_nat, width - lw - self.GAP), 30)
        room = width - lw - vw - 2 * self.GAP
        row_h = max(fl.height(), fv.height()) + 4
        notes = []
        for _label, _value, note, _state in rows:
            if not note:
                notes.append((False, 0))
            elif fn.horizontalAdvance(note) + 4 <= room:
                notes.append((True, 0))
            else:
                notes.append((False, fn.boundingRect(0, 0, max(width, 20), 10000, int(Qt.TextWordWrap), note).height() + 1))
        return lw, vw, room, row_h, notes

    def heightForWidth(self, width):
        if not self._rows:
            return 0
        _lw, _vw, _room, row_h, notes = self._plan(max(width, 1))
        return sum(row_h + under + (2 if under else 0) for _beside, under in notes)

    def _place(self, width):
        if not self._rows or width <= 0:
            return
        lw, vw, room, row_h, notes = self._plan(width)
        y = 0
        for i, (beside, under) in enumerate(notes):
            self._labels[i].setGeometry(0, y, lw, row_h)
            self._values[i].setGeometry(lw + self.GAP, y, vw, row_h)
            note = self._notes[i]
            if beside:
                note.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                note.setGeometry(lw + self.GAP + vw + self.GAP, y, max(room, 0), row_h)
                note.show()
            elif under:
                note.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                note.setGeometry(0, y + row_h - 2, width, under)
                note.show()
            else:
                note.hide()
            y += row_h + under + (2 if under else 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place(e.size().width())

    def sizeHint(self):
        if not self._rows:
            return QSize(100, 0)
        lw, vw, _room, _row_h, _notes = self._plan(10000)
        note = min(max((self._notes[i].fontMetrics().horizontalAdvance(r[2]) for i, r in enumerate(self._rows)), default=0), 240)
        w = lw + vw + 2 * self.GAP + (note if any(r[2] for r in self._rows) else 0)
        return QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self):
        return QSize(100, len(self._rows) * (self.fontMetrics().height() + 4))

    # ------------------------------------------------------------ the look
    def _apply_style(self):
        css = f"QLabel#factLabel, QLabel#factNote {{ color: {muted_colour(QApplication.palette())}; }}"
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()
            self._colour_values()


# ------------------------------------------------------------------------------------------------ the angle panel
def _off(r):
    return r["plateau_mean_deg"] - r["target_deg"] if not (_nan(r["plateau_mean_deg"]) or _nan(r["target_deg"])) else math.nan


def _main_columns(tol):
    """[(title, text(row), state(row) or None, unit for text(), tooltip)] of the table of every angle (the columns after the first are
    right-aligned)."""
    trim = _plateau_percent()

    def off_state(r):
        d = _off(r)
        return None if math.isnan(d) else "good" if abs(d) <= tol else "warn"
    return [
        ("Angle", lambda r: r["angle"], None, "", "The angles of this trial; the main joint is in bold."),
        ("Hold", lambda r: num(r["plateau_mean_deg"]), None, " deg",
         f"The angle during the contraction (the middle {trim:.0f} % of the window), not the angle that was set."),
        ("Target", lambda r: num(r["target_deg"]), None, " deg", "The angle that was asked for."),
        ("Off target", lambda r: num(_off(r), signed=True), off_state, " deg", "The hold angle minus the target. Green is within the tolerance."),
        ("In tolerance", lambda r: UNKNOWN if _nan(r["pct_in_tolerance"]) else f"{r['pct_in_tolerance']:.0f} %", None, "",
         "The share of the pictures in the window whose angle was within the tolerance."),
        ("Lost", lambda r: str(r["frames_lost"]), None, "", "Pictures in the window without this angle (a dot was lost)."),
    ]


def _more_measures():
    """[(title, text(row), tooltip)] of the numbers behind 'More numbers': one row of the second table each, one column per angle."""
    return [
        ("Mean", lambda r: num(r["mean_deg"]), "The mean of every picture in the window."),
        ("SD", lambda r: num(r["sd_deg"], 2), "The spread (standard deviation) of the angle in the window."),
        ("Min", lambda r: num(r["min_deg"]), "The lowest angle in the window."),
        ("Max", lambda r: num(r["max_deg"]), "The highest angle in the window."),
        ("Start to end", lambda r: num(r["change_start_to_end_deg"], signed=True), "How much the angle changed from the first to the last pictures of the window."),
        ("Max deviation", lambda r: num(r["max_abs_dev_deg"]), "The largest distance from the target in the window."),
        ("Median", lambda r: num(r["median_deg"]), "The middle value of the angle in the window."),
        ("Rest", lambda r: num(r["rest_mean_deg"]), "The angle at rest, before the push (the same stretch as the resting force)."),
        ("Rest to hold", lambda r: num(r["rest_to_hold_shift_deg"], signed=True), "How far the angle moved from rest to the hold."),
    ]


def reference_cells(rows):
    """[(title, reference, app, difference, instrument)] as texts, one tuple per reading of `trial.reference_rows()`: the words the reference
    block and the angle tree (review_angles.py, 0.32.0) both show."""
    specs = []
    for row in rows:
        diff = row["difference_deg"]
        inst = reference.INSTRUMENTS.get(row["instrument"], row["instrument"])
        instrument = f"{inst} (read to {row['resolution_deg']:g} deg)" if row.get("resolution_deg") is not None else str(inst)
        instrument += f", {row['source']}" if row.get("source") else ""
        ref = f"{num(row['reference_deg'])} deg" + (f" (mean of {row['n_readings']})" if row["n_readings"] > 1 else "")
        app = UNKNOWN if row["app_deg"] is None else f"{row['app_deg']:.1f} deg"
        difference = UNKNOWN if diff is None else f"{num(diff, signed=True)} deg"
        specs.append((row["title"], ref, app, difference, instrument))
    return specs


def reference_notice(rows, saved):
    """The sentence(s) above the readings: readings typed for another position that were not kept, and 'no reading for this trial'."""
    notice = ""
    if saved.get("not_used"):
        was = saved["not_used"].get("entered_for") or {}
        who = f" (participant {was['participant']})" if was.get("participant") else ""
        notice = (f"Readings typed for {was.get('test_key')} {was.get('target_deg')}{who} were NOT saved with this trial "
                  "(they belong to another position or participant).")
    if not rows:
        notice = (notice + " " if notice else "") + "No reading for this trial. Type it in the Live tab (section Reference reading) before recording, or add it here."
    return notice


class _ReferenceBlock(QFrame):
    """The reference readings (typed from a goniometer or an inclinometer) beside the app's angle: one line per reading with the reference, the
    app's angle and the difference, the instrument under it, and a button to add or change them. `text()` gives the same as lines."""

    editRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("referenceBlock")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 6)
        lay.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.title = QLabel("Reference readings")
        self.title.setFont(_bold(self.title.font()))
        self.button = QPushButton("Add or change the reference readings...")
        self.button.setObjectName("refEdit")
        self.button.setFocusPolicy(Qt.NoFocus)
        self.button.clicked.connect(self.editRequested)
        head.addWidget(self.title, 0)
        head.addStretch(1)
        head.addWidget(self.button, 0)
        lay.addLayout(head)
        self.subtitle = _Note("Typed from a goniometer or an inclinometer, not measured by this program.", name="refSubtitle")
        lay.addWidget(self.subtitle)
        self.notice = _Note("", name="refNotice")
        lay.addWidget(self.notice)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 2, 0, 0)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(0)
        self.grid.setColumnStretch(0, 1)
        lay.addWidget(self.grid_host)
        self.basis = _Note("", name="refBasis")
        lay.addWidget(self.basis)
        self.line_widgets = []
        self.n_rows = 0
        self._lines = []

    def set_trial(self, trial):
        rows = trial.reference_rows()
        saved = trial.meta.get("reference_readings") or {}
        for w in self.line_widgets:
            self.grid.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self.line_widgets = []
        self._lines = []
        self.n_rows = len(rows)
        notice = reference_notice(rows, saved)
        self.notice.setText(notice)
        self.notice.setVisible(bool(notice))
        self.grid_host.setVisible(bool(rows))
        self.button.setText("Add or change the reference readings..." if rows else "Add the reference readings...")
        if rows:
            for col, name in enumerate(("Angle", "Reference", "App", "Difference")):
                head = QLabel(name)
                head.setObjectName("refHead")
                head.setFont(_small(head.font()))
                if col:
                    head.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.grid.addWidget(head, 0, col)
                self.line_widgets.append(head)
            specs = reference_cells(rows)
            shared = len({s[4] for s in specs}) == 1             # one instrument for every reading: said once, under them
            r = 1
            for cells in specs:
                tip = f"{cells[0]}: reference {cells[1]}, app {cells[2]}, difference {cells[3]}.\n{cells[4]}"
                for col, text in enumerate(cells[:4]):
                    if col == 0:
                        label = ElidedLabel(text)
                        label.setFont(_bold(label.font()))
                        label.setMinimumWidth(40)
                    else:
                        label = QLabel(text)
                        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    label.setObjectName("refCell")
                    label.setToolTip(tip)
                    self.grid.addWidget(label, r, col)
                    self.line_widgets.append(label)
                r += 1
                if not shared:
                    self._add_instrument(cells[4], r)
                    r += 1
                self._lines.append(f"{cells[0]}: reference {cells[1]}, app {cells[2]}, difference {cells[3]}  {cells[4]}")
            if shared:
                self._add_instrument(specs[0][4], r)
        basis = self._explanation(trial, rows, saved)
        self.basis.setText(basis)
        self.basis.setVisible(bool(basis))
        self.setToolTip("")

    def _add_instrument(self, text, row):
        """A small line under the readings (or under one reading) saying which instrument it was read with, how finely, and where it came from."""
        what = _Note(text, name="refInstrument")
        self.grid.addWidget(what, row, 0, 1, 4)
        self.line_widgets.append(what)

    @staticmethod
    def _explanation(trial, rows, saved):
        """The sentences about how the readings and the app's angle belong together (empty when there is no reading)."""
        if not rows:
            return ""
        lines = []
        for basis in sorted({r["app_basis"] for r in rows if r["app_basis"]}):
            lines.append(f"The app's angle is {basis}.")
        if any(r["app_deg"] is None and r["key"] in {a.name for a in trial.angle_defs} for r in rows):
            lines.append("For an angle without an app value there was no moment kept and no usable first second.")
        if saved.get("stamped_at"):
            lines.append(f"Moment of the reading: {saved['stamped_at']}.")
        else:
            lines.append("No moment was kept ('Reading taken now' was not pressed): the first second of the recording stands in, which is only "
                         "like for like if the participant did not move between the reading and the start.")
        lines.append("A hand instrument is read to about a degree and has an error of its own of a few degrees; a difference of that size is "
                     "not an error of the app.")
        return " ".join(lines)

    def text(self, everything=False):
        out = [self.title.text(), self.subtitle.text()]
        if self.notice.isVisibleTo(self):
            out.append(self.notice.text())
        out += self._lines
        if self.basis.isVisibleTo(self):
            out.append(self.basis.text())
        out.append(f"[{self.button.text()}]")
        return "\n".join(out)


class AnglePanel(QWidget):
    """The angles of the open trial. `set_data(trial, rows, fs, window=None)`: `rows` = trial.summary(t0, t1), `fs` = trial.force_summary(...) (None
    for a trial without force; then the headline is the hold angle), `window` = (t0, t1) for the caption (optional). `set_none(text)` shows a quiet
    message instead. Signal `editReferenceRequested` when the button for the reference readings is pressed. The 'More numbers' switch adds a
    second table with the minor statistics. `text()` is everything on the screen as plain lines (`text(everything=True)` also holds the
    more numbers when the switch is off)."""

    editReferenceRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(0)
        self._last = None
        self._css = None
        self._sig = None
        self._main_row = []
        self._headline_state = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(6)
        self.empty = QLabel("")
        self.empty.setObjectName("panelEmpty")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)
        outer.addWidget(self.empty)
        self.content = QWidget()
        outer.addWidget(self.content)
        lay = QVBoxLayout(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # the headline and the plain facts beside it
        self.top = QBoxLayout(QBoxLayout.LeftToRight)
        self.top.setSpacing(12)
        self.headline = QFrame()
        self.headline.setObjectName("angleHeadline")
        hl = QVBoxLayout(self.headline)
        hl.setContentsMargins(12, 6, 14, 7)
        hl.setSpacing(0)
        self.headline_caption = ElidedLabel("")
        self.headline_caption.setObjectName("headlineCaption")
        self.headline_value = QLabel(UNKNOWN)
        self.headline_value.setObjectName("headlineValue")
        font = _bold(self.headline_value.font())
        font.setPointSizeF(24.0)
        self.headline_value.setFont(font)
        self.headline_target = ElidedLabel("")
        self.headline_target.setObjectName("headlineSub")
        self.headline_off = ElidedLabel("")
        self.headline_off.setObjectName("headlineSub")
        for w in (self.headline_caption, self.headline_target, self.headline_off):
            w.setFont(_small(w.font()))
            w.setMinimumWidth(60)
        hl.addWidget(self.headline_caption)
        hl.addWidget(self.headline_value)
        hl.addWidget(self.headline_target)
        hl.addWidget(self.headline_off)
        self.headline.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.facts = FactGrid()
        self.top.addWidget(self.headline, 0, Qt.AlignTop)
        self.top.addWidget(self.facts, 1, Qt.AlignTop)
        lay.addLayout(self.top)

        # the table of every angle, and the minor numbers behind a switch
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.window_label = ElidedLabel("")
        self.window_label.setObjectName("panelHint")
        self.more_check = QCheckBox("More numbers")
        self.more_check.setToolTip("Add the mean, the spread, the lowest and highest angle, the change from start to end, the largest deviation, the median, "
                                   "the rest angle and the shift from rest to hold.")
        self.more_check.setFocusPolicy(Qt.NoFocus)
        self.more_check.toggled.connect(self._more_toggled)
        bar.addWidget(self.window_label, 1)
        bar.addWidget(self.more_check, 0)
        lay.addLayout(bar)
        self.table = self._make_table("angleTable")
        lay.addWidget(self.table)
        self.more_table = self._make_table("moreTable")
        self.more_table.setVisible(False)
        lay.addWidget(self.more_table)
        self.note = _Note("")
        lay.addWidget(self.note)

        # the reference readings
        self.reference = _ReferenceBlock()
        self.reference.editRequested.connect(self.editReferenceRequested)
        lay.addWidget(self.reference)
        lay.addStretch(1)
        self._apply_style()
        self.set_none("No trial is open.")

    @staticmethod
    def _make_table(name):
        t = QTableWidget(0, 0)
        t.setObjectName(name)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QAbstractItemView.NoSelection)
        t.setFocusPolicy(Qt.NoFocus)
        t.setShowGrid(False)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)          # (the columns share the width: nothing to scroll sideways)
        t.verticalHeader().hide()
        t.verticalHeader().setDefaultSectionSize(22)
        t.horizontalHeader().setFixedHeight(24)
        t.horizontalHeader().setMinimumSectionSize(36)
        t.horizontalHeader().setHighlightSections(False)
        t.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        t.setMinimumWidth(0)
        t.setTextElideMode(Qt.ElideRight)
        return t

    # ------------------------------------------------------------ what the lead calls
    def set_none(self, text):
        self._last = None
        self._sig = None
        self.empty.setText(text)
        self.empty.setVisible(True)
        self.content.setVisible(False)

    def set_data(self, trial, rows, fs, window=None):
        self._last = (trial, rows, fs, window)
        self.empty.setVisible(False)
        self.content.setVisible(True)
        main = next((r for r in rows if r["role"] == "main"), rows[0] if rows else None)
        tol = float(trial.meta.get("tolerance_deg") or 5.0)
        self._fill_headline(main, fs, tol)
        self._fill_facts(main, fs, tol)
        self._fill_tables(rows, tol)
        self.window_label.setText("" if window is None else f"Window {window[0]:.2f} to {window[1]:.2f} s")
        no_markers = trial.hold_window() is None
        self.note.setText(f"Hold is the angle during the contraction (the middle {_plateau_percent():.0f} % of the window), not the angle that was set."
                          + (" No MVIC start/end markers in this trial: the numbers use the window as set." if no_markers else ""))
        self.reference.set_trial(trial)

    # ------------------------------------------------------------ the parts
    def _fill_headline(self, main, fs, tol):
        at_peak = fs is not None and not _nan(fs["main_angle_at_peak_deg"])
        if at_peak:
            value, caption = fs["main_angle_at_peak_deg"], f"{fs['main_angle'] or (main['angle'] if main else 'Angle')} at the peak force"
            tip = "The joint angle in the pictures at the moment of the highest force."
            if fs.get("picture_delay_s"):
                tip += f" The pictures are read {fs['picture_delay_s']:+.3f} s after the force peak."
        elif main is not None:
            value, caption = main["plateau_mean_deg"], f"{main['angle']} during the hold"
            tip = "The angle during the contraction (the middle part of the window). There is no force in this trial, so there is no peak force."
        else:
            value, caption, tip = math.nan, "No angle", ""
        tgt = main["target_deg"] if main is not None else math.nan
        dev = value - tgt if not (_nan(value) or _nan(tgt)) else math.nan
        state = None if math.isnan(dev) else "good" if abs(dev) <= tol else "warn"
        self.headline_caption.setText(caption)
        self.headline_value.setText(UNKNOWN if _nan(value) else f"{value:.1f} deg")
        self.headline_target.setText(f"Target {tgt:g} deg, tolerance {tol:g}" if not _nan(tgt) else "No target for this angle")
        if math.isnan(dev):
            off = ""
        elif abs(dev) < 0.05:
            off = "On the target"
        else:
            off = f"{abs(dev):.1f} deg {'above' if dev > 0 else 'below'} the target"
        self.headline_off.setText(off)
        self.headline.setToolTip(tip)
        self._headline_state = state
        self._size_headline()
        self._paint_headline()

    def _size_headline(self):
        """The card is as wide as its longest line (so the caption is not cut), up to a limit."""
        widest = max(w.fontMetrics().horizontalAdvance(w.text()) for w in (self.headline_caption, self.headline_value, self.headline_target, self.headline_off))
        self.headline.setFixedWidth(max(min(widest + 12 + 14 + 6, 300), 150))

    def _paint_headline(self):
        pal = QApplication.palette()
        colour = STATE_COLOURS.get(self._headline_state)
        strong = strong_colour(colour, pal, 0.8) if colour else ""
        self.headline_value.setStyleSheet(f"color: {strong};" if colour else "")
        self.headline_off.setStyleSheet(f"color: {strong};" if colour else "")
        self.headline.setStyleSheet(
            f"QFrame#angleHeadline {{ background: {tint(colour, 0.12) if colour else tint(ACCENT, 0.06)}; "
            f"border: 1px solid {tint(colour, 0.55) if colour else border_colour(pal)}; border-radius: 8px; }}")

    def _fill_facts(self, main, fs, tol):
        if main is None:
            self.facts.set_rows([])
            return
        plateau, tgt, shift = main["plateau_mean_deg"], main["target_deg"], main["rest_to_hold_shift_deg"]
        dev = plateau - tgt if not (_nan(plateau) or _nan(tgt)) else math.nan
        rows = []
        if fs is not None:                                         # with force the headline is the angle at the peak: the hold angle is a fact
            if math.isnan(dev):
                note, state = "no target", None
            else:
                note, state = f"{abs(dev):.1f} deg {'above' if dev > 0 else 'below'} the target {tgt:g}", "good" if abs(dev) <= tol else "warn"
            rows.append(("Hold angle", UNKNOWN if _nan(plateau) else f"{plateau:.1f} deg", note, state))
        else:
            rows.append(("Target", UNKNOWN if _nan(tgt) else f"{tgt:g} deg", f"tolerance {tol:g} deg", None))
        rows.append(("Rest to hold", UNKNOWN if _nan(shift) else f"{num(shift, signed=True)} deg",
                     "" if _nan(main["rest_mean_deg"]) else f"at rest {main['rest_mean_deg']:.1f} deg", None))
        frames, lost = main["frames"], main["frames_lost"]
        lost_pct = 100.0 * lost / frames if frames else 0.0
        rows.append(("Pictures lost", f"{lost} of {frames}", "no angle in these" if lost else "", "warn" if lost_pct > analysis.LOST_WARN_PCT else None))
        rows.append(("In tolerance", UNKNOWN if _nan(main["pct_in_tolerance"]) else f"{main['pct_in_tolerance']:.0f} %", "of the pictures in the window", None))
        self.facts.set_rows(rows)

    def _fill_tables(self, rows, tol):
        """Both tables, made again only when what they show changed (the window moves many times a second)."""
        cols, more = _main_columns(tol), _more_measures()
        cells = [[text(row) for _t, text, _s, _u, _tip in cols] for row in rows]
        states = [[(s(row) if s else None) for _t, _x, s, _u, _tip in cols] for row in rows]
        more_cells = [[text(row) for row in rows] for _t, text, _tip in more]
        sig = (cells, states, more_cells, [(r["role"], r["angle"]) for r in rows])
        if sig == self._sig:
            self._apply_more()
            return
        self._sig = sig
        self._main_row = [row["role"] == "main" for row in rows]
        pal = QApplication.palette()
        accent = QColor(ACCENT)
        accent.setAlphaF(0.12)
        flags = {False: Qt.AlignLeft | Qt.AlignVCenter, True: Qt.AlignRight | Qt.AlignVCenter}
        t = self.table
        t.setUpdatesEnabled(False)
        t.clear()
        t.setColumnCount(len(cols))
        t.setRowCount(len(rows))
        for c, (title, _text, _state, _unit, tip) in enumerate(cols):
            item = QTableWidgetItem(title)
            item.setToolTip(tip)
            item.setTextAlignment(flags[c > 0])
            t.setHorizontalHeaderItem(c, item)
        for r, row in enumerate(rows):
            for c in range(len(cols)):
                item = QTableWidgetItem(cells[r][c])
                item.setTextAlignment(flags[c > 0])
                if row["role"] == "main":
                    item.setFont(_bold(item.font()))
                    item.setBackground(QBrush(accent))
                if states[r][c]:
                    item.setForeground(QColor(strong_colour(STATE_COLOURS[states[r][c]], pal)))
                if c == 0:
                    item.setToolTip(row["angle"] + (" (the main joint)" if row["role"] == "main" else ""))
                t.setItem(r, c, item)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(cols)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        t.setUpdatesEnabled(True)
        t.setVisible(bool(rows))

        m = self.more_table                                           # one row per measure, one column per angle: it always fits the width
        m.setUpdatesEnabled(False)
        m.clear()
        m.setColumnCount(1 + len(rows))
        m.setRowCount(len(more))
        head = QTableWidgetItem("")
        m.setHorizontalHeaderItem(0, head)
        for k, row in enumerate(rows, start=1):
            item = QTableWidgetItem(row["angle"])
            item.setToolTip(row["angle"] + (" (the main joint)" if row["role"] == "main" else ""))
            item.setTextAlignment(flags[False])
            m.setHorizontalHeaderItem(k, item)
        muted = QColor(muted_colour(pal))
        for i, (title, _text, tip) in enumerate(more):
            label = QTableWidgetItem(title)
            label.setToolTip(tip)
            label.setForeground(muted)
            m.setItem(i, 0, label)
            for k, row in enumerate(rows, start=1):
                item = QTableWidgetItem(more_cells[i][k - 1])
                item.setToolTip(f"{title}: {row['angle']}")
                item.setTextAlignment(flags[False])
                if row["role"] == "main":
                    item.setFont(_bold(item.font()))
                    item.setBackground(QBrush(accent))
                m.setItem(i, k, item)
        mh = m.horizontalHeader()
        mh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for k in range(1, 1 + len(rows)):
            mh.setSectionResizeMode(k, QHeaderView.Stretch)
        m.setUpdatesEnabled(True)
        self._apply_more()

    def _more_toggled(self, _on):
        self._apply_more()

    def _apply_more(self):
        self.more_table.setVisible(self.more_check.isChecked() and self.more_table.rowCount() > 0 and self.more_table.columnCount() > 1)
        self._fit_tables()

    def _fit_tables(self):
        for t in (self.table, self.more_table):
            t.setFixedHeight(t.horizontalHeader().height() + t.rowCount() * t.verticalHeader().defaultSectionSize() + 2 * t.frameWidth())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.top.setDirection(QBoxLayout.TopToBottom if e.size().width() < 430 else QBoxLayout.LeftToRight)

    # ------------------------------------------------------------ what a person can read
    def text(self, everything=False):
        """Plain lines of everything the panel shows. The 'More numbers' lines only while the switch is on (or with `everything`)."""
        if not self.content.isVisibleTo(self):
            return self.empty.text()
        lines = [f"{self.headline_caption.text()}: {self.headline_value.text()}  {self.headline_target.text()}"
                 + (f"  {self.headline_off.text()}" if self.headline_off.text() else "")]
        lines += self.facts.text().splitlines()
        if self.window_label.text():
            lines.append(self.window_label.text())
        t = self.table
        units = [spec[3] for spec in _main_columns(1.0)]
        for r in range(t.rowCount()):
            name = t.item(r, 0).text() + (" (main joint)" if self._main_row[r] else "")
            parts = []
            for c in range(1, t.columnCount()):
                cell = t.item(r, c).text()
                parts.append(f"{t.horizontalHeaderItem(c).text()} {cell}" + (units[c] if cell != UNKNOWN else ""))
            lines.append(f"{name}: " + ", ".join(parts))
        m = self.more_table
        if (everything or self.more_check.isChecked()) and m.rowCount():
            lines.append("More numbers")
            for i in range(m.rowCount()):
                parts = [f"{m.horizontalHeaderItem(k).text()} {m.item(i, k).text()}" + ("" if m.item(i, k).text() == UNKNOWN else " deg")
                         for k in range(1, m.columnCount())]
                lines.append(f"{m.item(i, 0).text()}: " + ", ".join(parts))
        lines.append(self.note.text())
        lines += self.reference.text(everything).splitlines()
        return "\n".join(lines)

    # ------------------------------------------------------------ the look
    def _apply_style(self):
        pal = QApplication.palette()
        muted, border = muted_colour(pal), border_colour(pal)
        css = f"""
QLabel#headlineCaption, QLabel#headlineSub {{ color: {muted}; background: transparent; border: none; }}
QLabel#headlineValue {{ background: transparent; border: none; }}
QLabel#panelHint, QLabel#panelEmpty, QLabel#refSubtitle, QLabel#refHead, QLabel#refInstrument, QLabel#refBasis {{ color: {muted}; }}
QLabel#refNotice {{ color: {strong_colour(WARN, pal)}; }}
QFrame#referenceBlock {{ border: 1px solid {border}; border-radius: 6px; }}
QTableWidget#angleTable, QTableWidget#moreTable {{ border: 1px solid {border}; border-radius: 6px; background: transparent; }}
QTableWidget#angleTable::item, QTableWidget#moreTable::item {{ padding: 0px 6px; }}
QTableWidget#angleTable QHeaderView::section, QTableWidget#moreTable QHeaderView::section {{ padding: 2px 6px; border: none; border-bottom: 1px solid {border}; font-weight: 600; background: transparent; }}
QPushButton#refEdit {{ border: 1px solid {border}; border-radius: 4px; padding: 2px 10px; }}
QPushButton#refEdit:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
"""
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()
            if self._last is not None:
                self._sig = None                                       # (the colours of the cells follow the palette)
                self.set_data(*self._last)


# ------------------------------------------------------------------------------------------------ the force panel
class _Warning(QFrame):
    """One amber line at the top of the force panel; clicked, it shows the whole sentence over several lines."""

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
        rect = fm.boundingRect(0, 0, self._text_width(width), 10000, int(Qt.TextWordWrap), self.text)
        return rect.height() + 6

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


class _Section(QFrame):
    """One block of the force panel: a small title over a FactGrid (and, for the load cell, a button)."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("forceSection")
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(9, 5, 9, 7)
        lay.setSpacing(3)
        self.title = QLabel(title)
        self.title.setObjectName("sectionTitle")
        self.title.setFont(_bold(self.title.font()))
        self.grid = FactGrid()
        lay.addWidget(self.title)
        lay.addWidget(self.grid)
        self.extra = None

    def add_button(self, text, tip):
        self.extra = QPushButton(text)
        self.extra.setObjectName("sectionButton")
        self.extra.setToolTip(tip)
        self.extra.setFocusPolicy(Qt.NoFocus)
        self.layout().addWidget(self.extra)
        return self.extra

    def text(self):
        lines = [self.title.text()] + ["  " + line for line in self.grid.text().splitlines()]
        if self.extra is not None:
            lines.append(f"  [{self.extra.text()}]")
        return "\n".join(lines)


class ForcePanel(QWidget):
    """The force in the analysis window as sections that flow into 1 to 4 columns by width (no scroll bar). `set_data(trial, t0, t1, fs, ft, fp,
    picture_delay)` with `fs` = trial.force_summary(t0, t1, picture_delay), `ft` = analysis.force_timing(...) (None when there is no clear push) and
    `fp` = analysis.force_peaks(...); `set_none(text)` for a trial without force or a window without samples. Signal `calibrationRequested`
    when 'Show the calibration graph' is pressed. The height depends on the width (`heightForWidth`): the sections keep their order and are
    divided over the columns so that the columns are about as tall as each other. `text()` is everything on the screen as plain lines."""

    calibrationRequested = Signal()
    SECTION_MIN_WIDTH = 214
    MARGIN, GAP = 8, 8
    MAX_COLUMNS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumWidth(0)
        self._css = None
        self._last = None
        self.empty = QLabel("", self)
        self.empty.setObjectName("panelEmpty")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)
        self.caption = ElidedLabel("", self)
        self.caption.setObjectName("panelHint")
        self.warnings = []
        self.sections = {name: _Section(name, self) for name in ("Peak", "Level", "Timing", "Torque", "Load cell")}
        self.button_calibration = self.sections["Load cell"].add_button("Show the calibration graph", "Show the line the newtons come from, with this trial's peak on it.")
        self.button_calibration.clicked.connect(self.calibrationRequested)
        self._shown = []                                  # the sections in use, in order
        self._apply_style()
        self.set_none("No trial is open.")

    # ------------------------------------------------------------ what the lead calls
    def set_none(self, text):
        self._last = None
        self._clear_warnings()
        self.empty.setText(text)
        self.empty.show()
        self.caption.hide()
        for s in self.sections.values():
            s.hide()
        self._shown = []
        self.updateGeometry()
        self._flow(self.width())

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
        self.empty.hide()
        self.caption.show()
        smoothing = f"{fs['smoothing_s'] * 1000:.0f} ms moving average" if fs["smoothing_s"] > 0 else "raw samples"
        self.caption.setText(f"Force in the window: {u}{'' if u == 'N' else ', not calibrated'}, {smoothing}")
        self._set_warnings(self._warning_texts(fs))
        self.sections["Peak"].grid.set_rows(self._peak_rows(trial, fs, fp, u, d, smoothing))
        self.sections["Level"].grid.set_rows(self._level_rows(fs, u, d))
        self.sections["Timing"].grid.set_rows(self._timing_rows(ft, fp, u, d))
        self.sections["Torque"].grid.set_rows(self._torque_rows(trial, t0, t1, picture_delay, u))
        self.sections["Load cell"].grid.set_rows(self._load_cell_rows(trial, fs, u))
        self._shown = list(self.sections.values())
        for s in self._shown:
            s.show()
        self.updateGeometry()
        self._flow(self.width())

    # ------------------------------------------------------------ the numbers
    @staticmethod
    def _warning_texts(fs):
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

    @staticmethod
    def _peak_rows(trial, fs, fp, u, d, smoothing):
        rows = [("Peak", f"{num(fs['peak'], d)} {u}", f"at {fs['t_peak_s']:.2f} s ({smoothing})", None),
                ("Peak (20 ms)", f"{num(fs['peak_20ms'], d)} {u}",
                 "highest 20 ms average of the raw force: one mains cycle, so hum cancels; the same at any Smoothing", None)]
        if math.isnan(fp["window_peak"]):
            rows.append(("Sliding-window peak", UNKNOWN, f"window shorter than {trial.window_peak_s * 1000:.0f} ms, or gaps in the force", "warn"))
        else:
            rows.append(("Sliding-window peak", f"{num(fp['window_peak'], d)} {u}",
                         f"highest mean over {fp['window_peak_ms']:.0f} ms of the raw samples, {fp['window_peak_start_s']:.2f} to {fp['window_peak_end_s']:.2f} s", None))
        if fs["smoothing_s"] > 0:
            rows.append(("Raw single sample", f"{num(fs['raw_peak'], d)} {u}", "a noisy single sample overstates the peak", None))
        delay = f", pictures read {fs['picture_delay_s']:+.3f} s after the peak" if fs["picture_delay_s"] else ""
        rows.append(("Angle at the peak", f"{num(fs['main_angle_at_peak_deg'])} deg", f"{fs['main_angle'] or ''}{delay}".strip(", "), None))
        return rows

    @staticmethod
    def _level_rows(fs, u, d):
        rows = [("Rest before", f"{num(fs['rest_mean'], d)} {u}", f"{fs['rest_basis']}, {fs['rest_seconds']:.1f} s of force", "warn" if fs["rest_note"] else None),
                ("Peak above rest", f"{num(fs['peak_minus_rest'], d)} {u}", "", None),
                ("Plateau mean", f"{num(fs['plateau_mean'], d)} {u}", f"middle {_plateau_percent():.0f} % of the window", None)]
        if not _nan(fs["relaxed"]):
            ease = fs["rest_minus_relaxed"]
            note = "taken with the participant relaxed in the strap, before the trial" + (
                "" if _nan(ease) else f"; the rest before the push was {abs(ease):.{d}f} {u} {'higher' if ease > 0 else 'lower'} than that")
            rows += [("Relaxed (Zero 2)", f"{num(fs['relaxed'], d)} {u}", note, None),
                     ("Peak above relaxed", f"{num(fs['peak_minus_relaxed'], d)} {u}", "", None)]
        elif fs["relaxed_note"]:
            rows.append(("Relaxed (Zero 2)", "not used", fs["relaxed_note"], None))
        return rows

    @staticmethod
    def _timing_rows(ft, fp, u, d):
        if ft is None:
            return [("Timing", UNKNOWN, "no clear push in the window", "warn")]
        also = "" if _nan(fp["time_to_window_peak_s"]) else f"; to the middle of the sliding-window peak {fp['time_to_window_peak_s']:.2f} s"
        return [("Onset", f"{ft['onset_s']:.2f} s", "force last within 10 % of its rise above the level before the push", None),
                ("Time to peak", f"{ft['time_to_peak_s']:.2f} s", "from the onset to the peak" + also, None),
                ("10 to 90 % of the rise", f"{ft['time_to_90pct_s']:.2f} s", "", None),
                ("Steepest rise", f"{num(ft['rise_rate_per_s'], d)} {u}/s", "over 100 ms", None),
                ("Held above 90 %", f"{ft['hold_above_90pct_s']:.2f} s", "of the peak", None),
                ("Steadiness", f"{num(ft['hold_cv_pct'])} %", "SD / mean while held", None)]

    @staticmethod
    def _torque_rows(trial, t0, t1, picture_delay, u):
        tq = analysis.torque_summary(trial, t0, t1, picture_delay)
        if tq is None:
            why = ("the force is not calibrated" if u != "N" else "there is no ruler scale" if not trial.px_per_cm
                   else "the anchor dot was not used" if "anchor" not in trial.tracks else "no moment arm could be measured")
            return [("Torque", "not available", why, None)]
        stand_in = (f"the strap dot was not seen in {tq['pictures_with_stand_in']} of {tq['pictures']} pictures: the ankle dot stood in for it"
                    if tq["pictures_with_stand_in"] else "")
        return [("Peak", f"{tq['peak_nm']:.1f} N.m", f"at {tq['t_peak_s']:.2f} s", None),
                ("At the peak force", f"{tq['at_peak_force_nm']:.1f} N.m", "", None),
                ("Plateau", f"{num(tq['plateau_nm'])} N.m", "", None),
                ("Moment arm", f"{tq['arm_at_peak_force_cm']:.1f} cm", f"at the peak force, mean {tq['mean_arm_cm']:.1f} cm", None)] + (
                [("Stand-in dot", f"{tq['pictures_with_stand_in']} of {tq['pictures']}", stand_in, "warn")] if stand_in else [])

    @staticmethod
    def _load_cell_rows(trial, fs, u):
        reg, c = trial.regression, trial.force_cal or {}
        rows = [("Calibration", "recorded" if u == "N" else "none", trial.force_cal_note, "warn" if u != "N" else None)]
        if u != "N":
            rows.append(("Newtons", "not yet", "no zero was taken: use Apply zero to get newtons", None))
            return rows
        if reg is None:
            rows.append(("Regression", "none stored", f"{c.get('method', 'unknown calibration')}, {c.get('n_per_v', 0):.2f} N per volt: no precision can be given", None))
            return rows
        rows.append(("Regression", str(reg.id), f"{c.get('n_per_v', 0):.2f} N per volt, zero {c.get('offset_v', 0):+.4f} V", None))
        lo, hi = reg.range_kg
        for name, key in (("Peak", "peak"), ("Plateau mean", "plateau_mean")):
            f = fs[key]
            if _nan(f):
                continue
            sd = COVERAGE * float(reg.sd_newton(f))
            kg = f / GRAVITY
            rows.append((f"{name} plus or minus", f"{sd:.1f} N", f"{100 * sd / max(abs(f), 1e-9):.2f} % at about 95 % ({kg:.1f} kg)", None))
            if kg > hi * 1.001:
                rows.append((f"{name} above the line", f"{kg:.0f} kg", f"the heaviest calibration load is {hi:g} kg: the line is extended there, not measured", "warn"))
        return rows

    # ------------------------------------------------------------ warnings
    def _clear_warnings(self):
        for w in self.warnings:
            w.setParent(None)
            w.deleteLater()
        self.warnings = []

    def _set_warnings(self, texts):
        if [w.text for w in self.warnings] == texts:
            return
        self._clear_warnings()
        for text in texts:
            w = _Warning(text, self)
            w.changed.connect(self._warning_toggled)
            w.show()
            self.warnings.append(w)

    def _warning_toggled(self):
        for w in self.warnings:
            w._elide()
        self.updateGeometry()
        self._flow(self.width())

    # ------------------------------------------------------------ the flow
    def columns_for(self, width):
        """How many columns the sections go in at this width: the most that fit (each at least SECTION_MIN_WIDTH wide), or fewer if that is lower."""
        return self._plan(width)[0]

    def _plan(self, width):
        """(columns, [the sections of each column], height of each column's sections, section width) for this width. The sections stay in order;
        the way of cutting the row of sections into columns that makes the tallest column shortest wins (the fewest columns when it is a tie)."""
        inner = max(width - 2 * self.MARGIN, 40)
        most = max(1, min(self.MAX_COLUMNS, len(self._shown), (inner + self.GAP) // (self.SECTION_MIN_WIDTH + self.GAP)))
        best = None
        for cols in range(1, most + 1):
            colw = (inner - (cols - 1) * self.GAP) // cols
            heights = [s.heightForWidth(colw) for s in self._shown]
            for cuts in itertools.combinations(range(1, len(heights)), cols - 1):
                bounds = (0, *cuts, len(heights))
                groups = [list(range(a, b)) for a, b in zip(bounds, bounds[1:])]
                tallest = max(sum(heights[i] + self.GAP for i in g) for g in groups)
                if best is None or tallest < best[0]:
                    best = (tallest, cols, groups, colw, heights)
        if best is None:
            return 1, [], [], inner, []
        _tallest, cols, groups, colw, heights = best
        return cols, groups, [sum(heights[i] + self.GAP for i in g) - self.GAP for g in groups], colw, heights

    def _flow(self, width, apply=True):
        """Put the parts in place for this width; returns the height they need."""
        if width <= 0:
            width = self.sizeHint().width()
        m, gap = self.MARGIN, self.GAP
        inner = max(width - 2 * m, 40)
        y = m
        if not self._shown:
            h = self.empty.heightForWidth(inner) if self.empty.hasHeightForWidth() else self.empty.sizeHint().height()
            h = max(h, 40)
            if apply:
                self.empty.setGeometry(m, y, inner, h)
            return y + h + m
        h = self.caption.sizeHint().height()
        if apply:
            self.caption.setGeometry(m, y, inner, h)
        y += h + 4
        for w in self.warnings:
            h = w.heightForWidth(inner)
            if apply:
                w.setGeometry(m, y, inner, h)
            y += h + 3
        if self.warnings:
            y += 2
        cols, groups, col_heights, colw, heights = self._plan(width)
        if apply:
            for k, group in enumerate(groups):
                yy = y
                for i in group:
                    self._shown[i].setGeometry(m + k * (colw + gap), yy, colw, heights[i])
                    yy += heights[i] + gap
        return y + max(col_heights, default=0) + m

    def heightForWidth(self, width):
        return self._flow(width, apply=False)

    def sizeHint(self):
        w = 2 * self.SECTION_MIN_WIDTH + self.GAP + 2 * self.MARGIN
        return QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self):
        return QSize(self.SECTION_MIN_WIDTH, 60)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._flow(e.size().width())

    # ------------------------------------------------------------ what a person can read
    def text(self, everything=False):
        """Plain lines of everything the panel shows: the caption, the warnings in full, then each section as its title and its rows."""
        if not self._shown:
            return self.empty.text()
        lines = [self.caption.text()]
        lines += [f"Warning: {w.text}" for w in self.warnings]
        for s in self._shown:
            lines += s.text().splitlines()
        return "\n".join(lines)

    # ------------------------------------------------------------ the look
    def _apply_style(self):
        pal = QApplication.palette()
        muted, border = muted_colour(pal), border_colour(pal)
        css = f"""
QLabel#panelHint, QLabel#panelEmpty {{ color: {muted}; }}
QLabel#sectionTitle {{ color: {muted}; }}
QFrame#forceSection {{ border: 1px solid {border}; border-radius: 6px; }}
QFrame#forceWarning {{ background: {tint(WARN, 0.14)}; border: 1px solid {tint(WARN, 0.5)}; border-radius: 5px; }}
QLabel#forceWarningText {{ color: {strong_colour(WARN, pal, 0.8)}; background: transparent; border: none; }}
QPushButton#sectionButton {{ border: 1px solid {border}; border-radius: 4px; padding: 2px 10px; }}
QPushButton#sectionButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
"""
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()
