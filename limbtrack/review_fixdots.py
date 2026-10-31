"""The Fix dots workspace of the Review tab (0.30.0; a workspace of its own since 0.32.0): put a dot right after the trial. The fixes are kept in
the review layer (never in the trial's files), see limbtrack/dot_edits.py for what each does and what is recomputed.

  Hold in place   a fixed dot (anchor, ruler) stays at one position over a time range
  Mark lost       the dot is called lost over a time range
  By hand         click the dot picture by picture (it steps on after each click); a straight line is laid between two clicks
  Re-track        the program's tracker follows the dot from the picture you clicked it on

From the top: the dots of the trial in a table (how much each is lost, where it jumped, how far it moved, what was fixed) with the pull numbers
under it, a small timeline of the chosen dot (where it was lost, where it jumped, where it was fixed; a click on it moves the video there), and
the tools. The table (`DotTable`) and the table with the pull numbers (`PullTable`, both in `DotsPanel`) are also usable on their own, for
the 'Dots' panel under the video; each has `text()`, a plain-text copy of what a person reads in it.

The picture keys (the arrows and , . to step, Esc to leave) are installed by ReviewTab on the Trial page. Nothing here may keep the keyboard: the
buttons and tables never take it, a spin box gives it back to the picture when a number was typed or its little arrows were used (`release_keyboard`).
"""
import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QStyle, QStyleOptionSpinBox,
                               QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from . import dot_edits
from .analysis import LOST_WARN_PCT
from .review_style import ACCENT, BAD, GOOD, WARN, ElidedLabel, border_colour, muted_colour

HINT = ("Fix a dot after the trial. Your fixes are saved in the review folder; the recorded files are never changed. "
        "Only the pictures you fix change, and everything worked out from the dot (angles, pull, torque, checks, the overlay video) follows.")
HEADING = "Click a dot in the list to work on it. Fixes are saved as you go. Press Done (or Esc) when you have finished."
STEP_KEYS = ("Keys: Left and Right arrow (or the , and . keys) step one picture back and on. Hold Shift to step ten pictures. "
             "The buttons under the video step too.")                       # the keys ReviewTab installs for step_picture

JUMP_CM = 2.0                      # a dot that moves farther than this between two pictures has jumped (when the trial has a ruler scale) ...
JUMP_PICTURE_FRACTION = 0.01       # ... else farther than this share of the picture's width (the knee and ankle dots move a few millimetres per picture)
TIMELINE_HEIGHT = 100
TIMELINE_MAX_LIMITS = 6.0          # the timeline's scale stops at this many jump limits: a jump of 700 px must not flatten the rest (it runs off the top)
CURSOR_COLOUR = "#ffd43b"          # the same yellow line as the graphs of the Review tab
STEP_COLOUR = "#74c0fc"
MAX_TABLE_ROWS = 10                # more dots than this scroll inside their table
ROW_HEIGHT = 20
MIN_DOT_COLUMN = 56                # the dot's name is cut ('fibular_h...') before the numbers are squeezed

PULL_NAMES = {"pull_angle_deg": "Angle of pull (deg)", "moment_arm_cm": "Moment arm (cm)", "moment_arm_px": "Moment arm (px)",
              "strap_joint_cm": "Strap to joint (cm)", "strap_joint_px": "Strap to joint (px)"}


# ---------------------------------------------------------------------------------------------------------- the numbers (no Qt)
def jump_limit(tr):
    """(px, words): how far a dot may move between two pictures before it counts as a jump. 2 cm from the ruler scale, else 1 % of the picture's width."""
    if tr.px_per_cm:
        return JUMP_CM * float(tr.px_per_cm), f"{JUMP_CM:g} cm"
    width = (tr.full_size[0] if tr.full_size else None) or (tr.meta.get("frame_size") or [0])[0] or 1280
    return JUMP_PICTURE_FRACTION * float(width), f"{100 * JUMP_PICTURE_FRACTION:g} % of the picture's width"


def runs(mask):
    """[(first, last)] picture numbers of every stretch where `mask` is True."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()))


def band_edges(t, stretches, min_width=0.0):
    """[(x0, x1)] seconds of stretches of pictures [(first, last)]: from half a picture before the first to half a picture after the last (a band
    is at least `min_width` wide, so one lost picture can still be seen)."""
    out, n = [], len(t)
    for a, b in stretches:
        x0 = t[a] - (t[a] - t[a - 1]) / 2.0 if a > 0 else t[a]
        x1 = t[b] + (t[b + 1] - t[b]) / 2.0 if b + 1 < n else t[b]
        if x1 - x0 < min_width:
            mid = (x0 + x1) / 2.0
            x0, x1 = mid - min_width / 2.0, mid + min_width / 2.0
        out.append((float(x0), float(x1)))
    return out


@dataclass
class DotStat:
    """What the table and the timeline say about one dot of a trial (the shifts, which follow the blue window, are kept apart)."""
    name: str
    lost: np.ndarray             # True in the pictures where the dot is lost
    lost_pct: float              # share of the pictures, as the Check list counts it
    step: np.ndarray             # px the dot moved since the picture before (NaN where either picture has no position)
    jumps: int                   # pictures where that was more than the jump limit
    fixed: np.ndarray            # True in the pictures whose position was fixed in Review
    n_fixed: int

    @property
    def lost_runs(self):
        return runs(self.lost)

    @property
    def fixed_runs(self):
        return runs(self.fixed)

    @property
    def needs_attention(self):
        return self.lost_pct > LOST_WARN_PCT or self.jumps > 0


def trial_dot_stats(tr):
    """A DotStat for every dot of the trial (the computed hip centre is not a dot of the trial). Read from the trial as it is shown, so a fix shows."""
    limit, _ = jump_limit(tr)
    n = len(tr.t)
    out = []
    for d in tr.dot_names:
        x, y = tr.tracks[d]
        lost = np.fromiter((r[f"{d}_lost"] == "1" for r in tr.rows), bool, n)
        step = np.full(n, np.nan)
        if n > 1:
            step[1:] = np.hypot(np.diff(x), np.diff(y))
        with np.errstate(invalid="ignore"):
            jumps = int(np.count_nonzero(step > limit))
        fixed = np.zeros(n, bool)
        idx = tr.edited.get(d)
        if idx:
            fixed[list(idx)] = True
        out.append(DotStat(d, lost, 100.0 * float(lost.mean()) if n else 0.0, step, jumps, fixed, int(fixed.sum())))
    return out


def window_shifts(tr, t0, t1):
    """{dot: row of `Trial.displacement`} over [t0, t1]: end and max shift in px and cm (cm NaN without a ruler scale), and the same relative to the
    anchor (`rel_...`) for every dot but the anchor itself, when the trial has an anchor dot."""
    return {r["dot"]: r for r in tr.displacement(t0, t1)}


def trial_signature(tr):
    """Changes when the fixes of the shown trial changed (the numbers are then worked out again; the caller also compares the trial itself)."""
    if tr is None:
        return None
    rv = tr.review
    return (len(tr.t), tuple(sorted((d, len(v)) for d, v in tr.edited.items())), (len(rv.dots), len(rv.dot_rows)) if rv is not None else (0, 0))


def has_video(tr):
    """True when the trial's video can be read. A trial opened for numbers only has none, and nor has one whose video file is missing."""
    cap = getattr(tr, "cap", None)
    try:
        return cap is not None and bool(cap.isOpened())
    except Exception:  # noqa: BLE001 - a broken capture object is just 'no video'
        return False


def _distance(v, unit):
    return "-" if v is None or math.isnan(v) else f"{v:.1f} {unit}" if abs(v) < 10 else f"{v:.0f} {unit}"


def _shift(row, key, cm):
    """(text of the cell, its tooltip, the plain text) of one shift of a displacement row; `key` is 'end_shift', 'max_shift', 'rel_end_shift' or
    'rel_max_shift'. With a ruler scale the cell says cm and the pixels are in the tooltip and the plain text."""
    px, cmv = row.get(f"{key}_px", math.nan), row.get(f"{key}_cm", math.nan)
    if math.isnan(px):
        return "-", "", "-"
    if cm and not math.isnan(cmv):
        shown = _distance(cmv, "cm")
        return shown, f"{cmv:.2f} cm = {px:.1f} px", f"{shown} ({px:.1f} px)"
    shown = _distance(px, "px")
    return shown, "", shown


def _dump(head, rows):
    """Plain text of a table: a header line and a line for every row, the first column to the left and the others to the right."""
    lines = [head] + rows
    widths = [max(len(r[c]) for r in lines) for c in range(len(head))]
    return "\n".join("  ".join(cell.ljust(w) if c == 0 else cell.rjust(w) for c, (cell, w) in enumerate(zip(r, widths))).rstrip() for r in lines)


def _pct(p):
    return "0 %" if p == 0 else "<1 %" if p < 1 else f"{p:.0f} %"


def _window_of(tab, tr):
    """The blue window of the tab, else the whole trial."""
    try:
        t0, t1 = tab.window.getRegion()
        return float(t0), float(t1)
    except (AttributeError, TypeError):
        return float(tr.t[0]), float(tr.t[-1])


def _table_qss(pal):
    return (f"QTableWidget {{ border: 1px solid {border_colour(pal)}; border-radius: 4px; }} QTableWidget::item {{ padding: 0px 4px; }} "
            f"QHeaderView::section {{ padding: 2px 4px; border: none; border-bottom: 1px solid {border_colour(pal)}; font-weight: 600; }}")


def _style_table(t, select=True):
    """The look of the small number tables: no row numbers, no grid, whole rows selected, and never the keyboard (the picture keys stay free)."""
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
    t.verticalHeader().setMinimumSectionSize(ROW_HEIGHT - 4)
    t.horizontalHeader().setHighlightSections(False)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection if select else QAbstractItemView.NoSelection)
    t.setFocusPolicy(Qt.NoFocus)
    t.setWordWrap(False)
    t.setTextElideMode(Qt.ElideRight)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    _apply_table_qss(t)


def _apply_table_qss(t):
    """The table's stylesheet for the application's palette. True when it came out different (set only then: setting one can itself touch the palette)."""
    css = _table_qss(QApplication.palette())
    if css == getattr(t, "_qss", None):
        return False
    t._qss = css
    t.setStyleSheet(css)
    return True


def _align_headers(table, count):
    """The names of the columns sit over their numbers: the first to the left, the others to the right."""
    for c in range(count):
        table.horizontalHeaderItem(c).setTextAlignment((Qt.AlignLeft if c == 0 else Qt.AlignRight) | Qt.AlignVCenter)


def _fit_height(table):
    """The table is exactly as tall as its rows (up to MAX_TABLE_ROWS), so nothing wastes room above the tools."""
    rows = min(table.rowCount(), MAX_TABLE_ROWS)
    table.setFixedHeight(table.horizontalHeader().sizeHint().height() + rows * ROW_HEIGHT + 2 * table.frameWidth() + 2)


def _set_cell(table, row, col, text, right=False, bold=False, colour=None, background=None, tip=None):
    it = table.item(row, col)
    if it is None:
        it = QTableWidgetItem()
        table.setItem(row, col, it)
    it.setText(text)
    it.setTextAlignment((Qt.AlignRight if right else Qt.AlignLeft) | Qt.AlignVCenter)
    f = it.font()
    f.setBold(bold)
    it.setFont(f)
    it.setForeground(QBrush(QColor(colour)) if colour else QBrush())
    it.setBackground(background if background is not None else QBrush())
    it.setToolTip(tip or "")
    return it


# ---------------------------------------------------------------------------------------------------------- the dot table
class DotTable(QTableWidget):
    """One row per dot of the trial: Dot / Lost / Jumps / End shift / Max shift / (End and Max shift vs anchor) / Fixed. A row of a dot that needs a
    look (lost in more than 10 % of the pictures, or it jumped) is amber. `dotChosen(name)` is sent when a row is clicked, `dotActivated(name)`
    when it is double-clicked. `tab` (a ReviewTab) is optional: with it `refresh()` reads the trial and the blue window from the tab. `fit_height`
    makes the table exactly as tall as its rows. The two columns 'vs anchor' (how far a dot moved against the anchor dot: what the whole picture
    shares, a camera that was bumped, is taken away) show when the trial has an anchor dot and `relative` is True (the narrow workspace leaves them out).
    `text()` is a plain-text copy of the table."""
    dotChosen = Signal(str)
    dotActivated = Signal(str)
    COLUMNS = ["Dot", "Lost", "Jumps", "End shift", "Max shift", "End\nvs anchor", "Max\nvs anchor", "Fixed"]
    C_DOT, C_LOST, C_JUMPS, C_END, C_MAX, C_REL_END, C_REL_MAX, C_FIXED = range(8)
    RELATIVE = (C_REL_END, C_REL_MAX)

    def __init__(self, tab=None, parent=None, fit_height=False, relative=True):
        super().__init__(0, len(self.COLUMNS), parent)
        self.tab, self.trial, self.window, self.fit_height, self.relative = tab, None, None, fit_height, relative
        self.stats = {}                  # dot -> DotStat
        self.shifts = {}                 # dot -> row of Trial.displacement, for the blue window
        self.limit_px, self.limit_words = 0.0, ""
        self._sig = None                 # trial_signature of what is shown (a refresh for the same trial only moves the window)
        self._plain = []                 # per row: the plain text of every column (text())
        self.setObjectName("dotTable")
        self.setHorizontalHeaderLabels(self.COLUMNS)
        _style_table(self)
        _align_headers(self, len(self.COLUMNS))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed if fit_height else QSizePolicy.Expanding)
        hh = self.horizontalHeader()
        fm = self.fontMetrics()
        hh.setSectionResizeMode(self.C_DOT, QHeaderView.Stretch)
        for c, sample in ((self.C_LOST, "100 %"), (self.C_JUMPS, "Jumps"), (self.C_END, "End shift"), (self.C_MAX, "Max shift"),
                          (self.C_REL_END, "12 cm"), (self.C_REL_MAX, "12 cm"), (self.C_FIXED, "✎ 9999")):
            hh.setSectionResizeMode(c, QHeaderView.Fixed)
            head = max(fm.horizontalAdvance(line) for line in self.COLUMNS[c].split("\n"))
            hh.resizeSection(c, max(fm.horizontalAdvance(sample), head + 6) + 16)
        hh.setMinimumSectionSize(40)
        for c in self.RELATIVE:
            self.setColumnHidden(c, True)
        self.cellClicked.connect(lambda r, _c: self.dotChosen.emit(self._name(r)))
        self.cellDoubleClicked.connect(lambda r, _c: self.dotActivated.emit(self._name(r)))
        self._tips()
        if fit_height:
            _fit_height(self)

    def minimumSizeHint(self):
        """As wide as its numbers need (the dot's name is cut before they are squeezed), so a panel holding it knows its smallest width."""
        hh = self.horizontalHeader()
        w = MIN_DOT_COLUMN + sum(hh.sectionSize(c) for c in range(1, self.columnCount()) if not self.isColumnHidden(c)) + 2 * self.frameWidth()
        return QSize(w, ROW_HEIGHT * 3)

    # ---- what is shown
    def _tips(self):
        tips = ["The dots of this trial.",
                "How many of the pictures the program could not find the dot in. More than 10 % is amber: the angles, pull and checks that need the dot are missing there.",
                (f"Pictures where the dot moved more than {self.limit_words} since the picture before. A jump is usually the tracker taking hold of something else "
                 "(or losing the dot and finding it again in the wrong place)." if self.limit_words else
                 "Pictures where the dot moved far in one step (2 cm, or 1 % of the picture's width without a ruler scale): usually the tracker taking hold of something else."),
                "How far the dot ended from where it started, inside the blue window (cm from the ruler scale, else pixels).",
                "The farthest the dot got from where it started, anywhere inside the blue window.",
                "How far the dot ended from where it started, compared with the anchor dot: movement the whole picture shares (a camera that was bumped) is taken away.",
                "The farthest the dot got from where it started, compared with the anchor dot.",
                "Pictures of this dot that you fixed in Review (a pencil)."]
        for c, tip in enumerate(tips):
            self.horizontalHeaderItem(c).setToolTip(tip)

    def set_trial(self, trial, window=None):
        """Fill the table from `trial` (None: empty). `window`: (t0, t1) seconds for the shift columns, else the tab's blue window, else the whole trial."""
        keep = self.selected_dot()
        self.trial = trial
        if trial is None:
            self.stats, self.shifts, self.window, self._sig, self._plain = {}, {}, None, None, []
            self.setRowCount(0)
            if self.fit_height:
                _fit_height(self)
            return
        self.limit_px, self.limit_words = jump_limit(trial)
        self._tips()
        self._sig = trial_signature(trial)
        self.stats = {s.name: s for s in trial_dot_stats(trial)}
        self.window = tuple(window) if window is not None else _window_of(self.tab, trial)
        self.shifts = window_shifts(trial, *self.window)
        self._show_relative()
        self.setRowCount(len(self.stats))
        self._plain = [[""] * self.columnCount() for _ in self.stats]
        for row, s in enumerate(self.stats.values()):
            self._fill_row(row, s)
        if self.fit_height:
            _fit_height(self)
        self.select_dot(keep)

    def _show_relative(self):
        """The 'vs anchor' columns show only for a trial with an anchor dot (and when asked for)."""
        want = self.relative and any("rel_end_shift_px" in r for r in self.shifts.values())
        for c in self.RELATIVE:
            self.setColumnHidden(c, not want)

    def _fill_row(self, row, s):
        back = None
        if s.needs_attention:
            c = QColor(WARN)
            c.setAlpha(70)
            back = QBrush(c)
        tip = (f"{s.name}: lost in {s.lost_pct:.0f} % of the pictures, {s.jumps} jump{'' if s.jumps == 1 else 's'}, "
               f"{s.n_fixed} picture{'' if s.n_fixed == 1 else 's'} fixed" + (". Needs a look." if s.needs_attention else "."))
        muted = muted_colour(QApplication.palette())
        plain = self._plain[row]
        it = _set_cell(self, row, self.C_DOT, s.name, background=back, tip=tip)
        it.setData(Qt.UserRole, s.name)
        _set_cell(self, row, self.C_LOST, _pct(s.lost_pct), True, bold=s.lost_pct > LOST_WARN_PCT, colour=muted if s.lost_pct == 0 else None, background=back, tip=tip)
        _set_cell(self, row, self.C_JUMPS, str(s.jumps), True, bold=s.jumps > 0, colour=muted if s.jumps == 0 else None, background=back, tip=tip)
        for col in (self.C_END, self.C_MAX, self.C_REL_END, self.C_REL_MAX):
            _set_cell(self, row, col, "-", True, background=back, tip=tip)
        _set_cell(self, row, self.C_FIXED, f"✎ {s.n_fixed}" if s.n_fixed else "-", True, bold=bool(s.n_fixed), colour=ACCENT if s.n_fixed else muted, background=back,
                  tip=tip if not s.n_fixed else f"{s.n_fixed} picture{'' if s.n_fixed == 1 else 's'} of {s.name} fixed in Review.")
        plain[self.C_DOT], plain[self.C_LOST], plain[self.C_JUMPS] = s.name, _pct(s.lost_pct), str(s.jumps)
        plain[self.C_FIXED] = f"✎ {s.n_fixed}" if s.n_fixed else "-"
        self._fill_shifts(row, s.name)

    def _fill_shifts(self, row, name):
        """The shift columns of one row, from the blue window (the only part that changes when the window moves)."""
        r = self.shifts.get(name, {})
        cm = bool(self.trial.px_per_cm) if self.trial is not None else False
        for col, key in ((self.C_END, "end_shift"), (self.C_MAX, "max_shift"), (self.C_REL_END, "rel_end_shift"), (self.C_REL_MAX, "rel_max_shift")):
            shown, tip, plain = _shift(r, key, cm)
            it = self.item(row, col)
            it.setText(shown)
            if tip:
                it.setToolTip(f"{name}, {self.COLUMNS[col].replace(chr(10), ' ').lower()}: {tip}")
            self._plain[row][col] = plain

    def set_window(self, t0, t1):
        """The blue window moved: only the shift columns are worked out again."""
        self.window = (float(t0), float(t1))
        if self.trial is None:
            return
        self.shifts = window_shifts(self.trial, *self.window)
        for row in range(self.rowCount()):
            self._fill_shifts(row, self._name(row))
        self._show_relative()

    def shows(self, tr):
        """True when the table already shows this trial as it is now (no other trial, no fix since)."""
        return tr is not None and tr is self.trial and trial_signature(tr) == self._sig

    def refresh(self):
        """Read the trial and the blue window from the tab again. For the trial that is already shown (no fix since) only the shifts are worked out
        again, which is what a dragged window needs; another trial, or a fixed one, is read afresh."""
        tr = self.tab.trial if self.tab is not None else self.trial
        if self.shows(tr):
            self.set_window(*_window_of(self.tab, tr))
        else:
            self.set_trial(tr)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange) and _apply_table_qss(self) and self.trial is not None:
            self.set_trial(self.trial, self.window)               # (the amber rows and the muted zeros are drawn in the palette's colours)

    def text(self):
        """A plain-text copy of what a person reads in the table: a header line and one line per dot (dot, lost, jumps, end shift, max shift, the two
        'vs anchor' shifts when shown, fixed). With a ruler scale a shift is given in cm and, in brackets, in px. '' when no trial is shown."""
        if self.trial is None or not self._plain:
            return ""
        cols = [c for c in range(self.columnCount()) if not self.isColumnHidden(c)]
        return _dump([self.COLUMNS[c].replace("\n", " ") for c in cols], [[p[c] for c in cols] for p in self._plain])

    # ---- the chosen dot
    def _name(self, row):
        it = self.item(row, 0)
        return it.data(Qt.UserRole) if it is not None else ""

    def select_dot(self, name):
        """Mark a dot's row as the chosen one (nothing is sent)."""
        for row in range(self.rowCount()):
            if self._name(row) == name:
                self.selectRow(row)
                return
        self.clearSelection()

    def selected_dot(self):
        rows = self.selectionModel().selectedRows() if self.selectionModel() is not None else []
        return self._name(rows[0].row()) if rows else None


class PullTable(QTableWidget):
    """The pull numbers (angle of pull, moment arm, strap to joint) in the blue window: mean, lowest, highest and the change from start to end.
    Empty, and hidden, for a trial without strap and anchor dots. `text()` is a plain-text copy."""
    COLUMNS = ["Pull", "Mean", "Min", "Max", "Change"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLUMNS), parent)
        self.setObjectName("pullTable")
        _style_table(self, select=False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        _align_headers(self, len(self.COLUMNS))
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        fm = self.fontMetrics()
        for c in range(1, len(self.COLUMNS)):
            hh.setSectionResizeMode(c, QHeaderView.Fixed)
            hh.resizeSection(c, max(fm.horizontalAdvance("-123.4"), fm.horizontalAdvance(self.COLUMNS[c]) + 6) + 14)
        hh.setMinimumSectionSize(40)
        for c, tip in enumerate(["The angle between the line of pull (strap dot to anchor dot) and the lower leg segment, and the distances that make the torque.",
                                 "Mean over the blue window.", "Lowest value in the blue window.", "Highest value in the blue window.",
                                 "From the start to the end of the blue window (each end averaged over five pictures)."]):
            self.horizontalHeaderItem(c).setToolTip(tip)
        self._plain = []
        self.setVisible(False)

    def minimumSizeHint(self):
        hh = self.horizontalHeader()
        return QSize(MIN_DOT_COLUMN + sum(hh.sectionSize(c) for c in range(1, self.columnCount())) + 2 * self.frameWidth(), ROW_HEIGHT * 2)

    def set_trial(self, trial, window):
        rows = trial.pull_summary(*window) if trial is not None and window is not None else []
        self.setRowCount(len(rows))
        self._plain = []
        for r, q in enumerate(rows):
            name = PULL_NAMES.get(q["quantity"], q["quantity"])
            _set_cell(self, r, 0, name, tip=name)
            plain = [name]
            for c, key, signed in ((1, "mean", False), (2, "min", False), (3, "max", False), (4, "change_start_to_end", True)):
                v = q[key]
                text = "-" if math.isnan(v) else f"{v:+.1f}" if signed else f"{v:.1f}"
                _set_cell(self, r, c, text, True, tip=f"{q['frames_missing']} pictures of the window have no value" if q["frames_missing"] and c == 1 else None)
                plain.append(text)
            self._plain.append(plain)
        self.setFixedHeight(self.horizontalHeader().sizeHint().height() + len(rows) * ROW_HEIGHT + 2 * self.frameWidth() + 2)
        self.setVisible(bool(rows))

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            _apply_table_qss(self)

    def text(self):
        """A plain-text copy: a header line and one line per pull number (mean, min, max, change). '' when the trial has no strap and anchor dots."""
        return _dump(list(self.COLUMNS), self._plain) if self._plain else ""


class DotsPanel(QWidget):
    """The dot table with the pull numbers under it and a line saying which stretch they are for. It follows the blue window of the tab. Use it
    where a panel of dot numbers is wanted (the 'Dots' tab under the video): `set_trial(trial)` when a trial is loaded, `refresh()` to read tab and
    window again (cheap when only the window moved). `scrolls` (the default): when the panel is shorter than its numbers it scrolls, and never
    forces the room it sits in to be taller; the workspace puts it in a scrolling column of its own and passes False. `relative` shows the 'vs anchor'
    columns. `text()` is a plain-text copy of what a person reads: the line over the table, the table, the pull table."""
    dotChosen = Signal(str)
    dotActivated = Signal(str)
    windowChanged = Signal(float, float)                 # the blue window moved (after the numbers were worked out again)

    def __init__(self, tab=None, parent=None, scrolls=True, relative=True):
        super().__init__(parent)
        self.tab, self._hooked = tab, False
        self.caption = ElidedLabel("")
        self.caption.setObjectName("hint")
        self.table = DotTable(tab, fit_height=True, relative=relative)
        self.pull = PullTable()
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.caption)
        lay.addWidget(self.table)
        lay.addWidget(self.pull)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        if scrolls:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.NoFrame)
            area.setFocusPolicy(Qt.NoFocus)                # (the picture keys never end up in a scroll area)
            area.setWidget(body)
            outer.addWidget(area)
        else:
            outer.addWidget(body)
        self.table.dotChosen.connect(self.dotChosen)
        self.table.dotActivated.connect(self.dotActivated)
        self.set_trial(None)

    @property
    def trial(self):
        return self.table.trial

    def _hook_window(self):
        """Follow the tab's blue window (once; the tab may not have made it yet when this panel is made)."""
        sig = getattr(getattr(self.tab, "window", None), "sigRegionChanged", None) if self.tab is not None else None
        if sig is not None and not self._hooked:
            sig.connect(self._window_moved)
            self._hooked = True

    def _window_moved(self, *_):
        tr = self.table.trial
        if tr is not None:
            self.set_window(*_window_of(self.tab, tr))

    def set_trial(self, trial, window=None):
        self._hook_window()
        self.table.set_trial(trial, window)
        self._show_window()

    def set_window(self, t0, t1):
        self.table.set_window(t0, t1)
        self.pull.set_trial(self.table.trial, self.table.window)
        self._caption()
        self.windowChanged.emit(float(t0), float(t1))

    def _show_window(self):
        self.pull.set_trial(self.table.trial, self.table.window)
        self._caption()

    def _caption(self):
        tr, w = self.table.trial, self.table.window
        if tr is None or w is None:
            self.caption.setText("")
            return
        self.caption.setText(f"Movement in the blue window, {w[0]:.2f} to {w[1]:.2f} s ({'cm from the ruler scale' if tr.px_per_cm else 'pixels: no ruler scale'})")

    def refresh(self):
        """Read the trial and the blue window from the tab again (for the trial already shown, with no fix since, only the window is taken again)."""
        tr = self.tab.trial if self.tab is not None else self.table.trial
        if self.table.shows(tr):
            self.set_window(*_window_of(self.tab, tr))
        else:
            self.set_trial(tr)

    def select_dot(self, name):
        self.table.select_dot(name)

    def text(self):
        """What a person reads, as plain text: the line over the table, one line per dot, then (when there is a strap and an anchor dot) one line per pull number."""
        parts = [p for p in (self.caption.text(), self.table.text()) if p]
        pull = self.pull.text()
        return "\n".join(parts) + ("\n\n" + pull if pull else "")


# ---------------------------------------------------------------------------------------------------------- the timeline of one dot
class DotTimeline(pg.PlotWidget):
    """Time along the bottom, along the side how far the dot moved since the picture before (px). Red bands: the dot was lost. Green bands: its
    positions were fixed. The dashed line is the jump limit. The yellow line follows the picture on screen; a click or a drag on the graph sends
    `picked(seconds)`, so the video can go there."""
    picked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(TIMELINE_HEIGHT)
        self.setFocusPolicy(Qt.NoFocus)                     # (the picture keys must not scroll this graph)
        pi = self.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.setMouseEnabled(x=False, y=False)
        pi.showGrid(x=True, y=True, alpha=0.2)
        pi.setLabel("left", "Jump (px)")
        pi.setLabel("bottom", "Time (s)")
        for ax in ("left", "bottom"):
            pi.getAxis(ax).setStyle(tickTextOffset=2)
            pi.getAxis(ax).setTickFont(self.font())
        pi.getAxis("left").setWidth(48)
        pi.layout.setContentsMargins(0, 2, 6, 0)
        pi.enableAutoRange(False)
        nothing = pg.mkPen(None)
        self.window_region = pg.LinearRegionItem(values=(0, 1), movable=False, brush=(30, 120, 220, 45), pen=nothing)
        self.window_region.setZValue(-10)
        for line in self.window_region.lines:
            line.setPen(nothing)
        self.lost_bars = pg.BarGraphItem(x0=[0], x1=[1], y0=[0], y1=[1], brush=pg.mkBrush(BAD + "99"), pen=nothing)
        self.fixed_bars = pg.BarGraphItem(x0=[0], x1=[1], y0=[0], y1=[1], brush=pg.mkBrush(GOOD + "88"), pen=nothing)
        self.lost_bars.setZValue(-6)
        self.fixed_bars.setZValue(-5)
        self.curve = pg.PlotCurveItem(pen=pg.mkPen(STEP_COLOUR, width=1.5))
        self.limit_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(WARN, width=1.2, style=Qt.DashLine))
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(CURSOR_COLOUR, width=2))
        self.cursor.setZValue(10)
        for item in (self.window_region, self.lost_bars, self.fixed_bars, self.curve, self.limit_line, self.cursor):
            pi.addItem(item)
        self.t = None
        self.clear_data()

    def clear_data(self):
        self.t = None
        for item in (self.lost_bars, self.fixed_bars, self.curve, self.limit_line, self.window_region, self.cursor):
            item.setVisible(False)

    def set_data(self, t, step, lost_stretches, fixed_stretches, limit_px, window=None):
        """Draw a dot: `t` seconds of the pictures, `step` px per picture (NaN where unknown), stretches [(first, last)] picture numbers."""
        if t is None or len(t) < 2:
            self.clear_data()
            return
        self.t = np.asarray(t, float)
        span = float(self.t[-1] - self.t[0])
        finite = step[np.isfinite(step)]
        top = max(2.0 * limit_px, min(float(finite.max()) * 1.1 if finite.size else 0.0, TIMELINE_MAX_LIMITS * limit_px))      # (a very big jump runs off the top)
        pi = self.getPlotItem()
        pi.setXRange(float(self.t[0]), float(self.t[-1]), padding=0)
        pi.setYRange(0.0, top, padding=0)
        self.curve.setData(self.t, step, connect="finite")
        self.limit_line.setValue(limit_px)
        for bars, stretches in ((self.lost_bars, lost_stretches), (self.fixed_bars, fixed_stretches)):
            edges = band_edges(self.t, stretches, span / 250.0)
            if edges:
                x0, x1 = np.array([e[0] for e in edges]), np.array([e[1] for e in edges])
                bars.setOpts(x0=x0, x1=x1, y0=np.zeros(len(x0)), y1=np.full(len(x0), top))
            bars.setVisible(bool(edges))
        self.curve.setVisible(True)
        self.limit_line.setVisible(True)
        self.cursor.setVisible(True)
        self.set_window(window)

    def set_window(self, window):
        """Show the blue analysis window (the stretch the shift numbers are for)."""
        if window is None:
            self.window_region.setVisible(False)
            return
        self.window_region.setRegion((float(window[0]), float(window[1])))
        self.window_region.setVisible(self.t is not None)

    def set_cursor(self, seconds):
        """Move the yellow line (cheap: called for every picture)."""
        self.cursor.setValue(float(seconds))

    def _seconds_at(self, e):
        return float(self.getPlotItem().vb.mapSceneToView(self.mapToScene(e.position().toPoint())).x())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.t is not None:
            self.picked.emit(self._seconds_at(e))
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton and self.t is not None:
            self.picked.emit(self._seconds_at(e))
            e.accept()
            return
        super().mouseMoveEvent(e)


# ---------------------------------------------------------------------------------------------------------- small widgets of the pane
class _TimeSpin(QDoubleSpinBox):
    """A time in seconds. It takes the keyboard only while a number is typed in it: `arrowUsed` is sent after its little arrows were pressed."""
    arrowUsed = Signal()

    def __init__(self):
        super().__init__()
        self.setDecimals(2)
        self.setSingleStep(0.1)
        self.setSuffix(" s")
        self.setFocusPolicy(Qt.ClickFocus)

    def wheelEvent(self, e):
        e.ignore()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        opt = QStyleOptionSpinBox()
        self.initStyleOption(opt)
        if self.style().hitTestComplexControl(QStyle.CC_SpinBox, opt, e.position().toPoint(), self) in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown):
            self.arrowUsed.emit()


class CollapsibleBox(QWidget):
    """A flat button with an arrow that opens and closes what is under it (closed to begin with)."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setAutoRaise(True)
        self.button.setArrowType(Qt.RightArrow)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setFocusPolicy(Qt.NoFocus)
        self.button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        f = self.button.font()
        f.setBold(True)
        self.button.setFont(f)
        self.body = QWidget()
        self.body.setVisible(False)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(14, 0, 0, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.button)
        lay.addWidget(self.body)
        self.button.toggled.connect(self.set_expanded)

    def set_expanded(self, on):
        self.button.blockSignals(True)
        self.button.setChecked(bool(on))
        self.button.blockSignals(False)
        self.button.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(bool(on))

    def is_expanded(self):
        return self.button.isChecked()


class _Wrapped(QLabel):
    """A label that wraps to the width it has and never widens the pane. Its smallest height is one line: how many lines it needs is worked out
    from the width (height for width), not from the narrowest the pane could be, which would make the pane look taller than it is."""

    def __init__(self, text=""):
        super().__init__(text)
        self.setWordWrap(True)
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)       # (Minimum would make the layout's smallest height the sizeHint's: three lines)
        policy.setHeightForWidth(True)                                          # (a new policy forgets the word wrap's own: the layout then ignores the width)
        self.setSizePolicy(policy)

    def minimumSizeHint(self):
        return QSize(10, self.fontMetrics().lineSpacing())


def _hint(text, wrap=True):
    lb = _Wrapped(text) if wrap else QLabel(text)
    lb.setObjectName("hint")
    return lb


# ---------------------------------------------------------------------------------------------------------- the pane
class FixDotsPane(QScrollArea):
    """The Fix dots workspace: a narrow column (380 to 600 px wide) that fills the right side while dots are fixed. A scroll area only as a
    fallback for a short window; at about 900 px of height nothing needs scrolling.

    What the ReviewTab calls: `set_trial(trial)` after every load (also the reload after a fix), `refresh()`, `current_dot()` and the combo `dot` (the
    dot to work on; `set_fix_mode(True, dot=...)` sets it), `on_video_click(x, y)` for a click on the video, `marks(i)` for what to draw on picture i,
    `update_where()` for every picture shown (cheap)."""

    def __init__(self, tab):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tab = tab
        self.click = None                      # (x, y, picture number) of the last click on the video (of this trial)
        self.pending = {}                      # picture -> (x, y): positions clicked by hand for the dot on screen, not saved yet
        self._stash = {}                       # dot -> pending positions of a dot that was left for another before they were saved
        self._pending_dot = None               # the dot `pending` belongs to
        self._shown = None                     # the trial the table and the timeline show, and trial_signature of it as it was worked out
        self._sig = None
        self._trial_path = None                # which trial that is (a reload after a fix is the same trial: clicks not saved yet stay)
        self._want_default = False             # a trial was opened: start on the dot that needs a look most
        self._video_ok = False                 # the trial's video can be read
        self._timeline_key = None
        self._usual = None
        self._usual_key = None
        inner = QWidget()
        self.setWidget(inner)

        # (1) the heading
        self.heading = _hint(HEADING)
        self.heading.setToolTip(HINT)

        # (2) the dots, and the pull numbers under them; (3) the timeline of the chosen dot
        self.dots = DotsPanel(tab, scrolls=False, relative=False)       # (no 'vs anchor' columns: the column is narrow, and here the question is which dot is wrong)
        self.table, self.pull = self.dots.table, self.dots.pull
        self.dots.dotChosen.connect(self._row_chosen)
        self.dot = QComboBox(inner)            # (the state behind the table: the chosen dot; hidden since the table chooses)
        self.dot.setVisible(False)
        self.dot.setFocusPolicy(Qt.NoFocus)                      # (the picture keys must never end up in it)
        self.dot.view().setFocusPolicy(Qt.NoFocus)
        self.dot.currentIndexChanged.connect(lambda _: self.refresh())
        self.dot_title = QLabel("")
        f = self.dot_title.font()
        f.setBold(True)
        self.dot_title.setFont(f)
        self.where = ElidedLabel("")
        self.where.setObjectName("hint")
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(self.dot_title)
        title_row.addWidget(self.where, 1)
        self.timeline = DotTimeline()
        self.timeline.picked.connect(self._timeline_picked)
        self.dots.windowChanged.connect(lambda a, b: self.timeline.set_window((a, b)))        # (the blue band follows the window)
        self.timeline.setToolTip("How far the chosen dot moved from one picture to the next (px). Red = lost, green = fixed in Review, dashed = the jump limit, "
                                 "blue = the blue window. A very big jump runs off the top. Click or drag to move the video there.")
        self.timeline_hint = ElidedLabel("Red = lost. Green = fixed. Dashed = jump limit. Click to move the video.")
        self.timeline_hint.setObjectName("hint")

        # (4a) hold in place or mark lost
        self.t_from, self.t_to = _TimeSpin(), _TimeSpin()
        self.btn_from = self._button("this picture", "Take the time of the picture on screen.", lambda: self.t_from.setValue(self.tab.current_time()))
        self.btn_to = self._button("this picture", "Take the time of the picture on screen.", lambda: self.t_to.setValue(self.tab.current_time()))
        self.btn_window = self._button("Blue window", "The stretch of time inside the blue window of the graphs.", self._range_from_window)
        self.btn_hold_usual = self._button("Hold at its usual position", "The place the dot sat when the program saw it well (the middle of those positions). "
                                           "For the anchor and the ruler dots, which do not move.", lambda: self._hold(use_click=False))
        self.btn_hold_click = self._button("Hold where I clicked", "Click the dot on the picture first, then press this: it stays there over the range.",
                                           lambda: self._hold(use_click=True))
        self.btn_lost = self._button("Mark lost", "No position is better than a wrong one: the dot counts as lost over the range.", self._mark_lost)
        self.box_hold = QGroupBox("Hold in place or mark lost")
        g1 = QGridLayout(self.box_hold)
        g1.setContentsMargins(8, 6, 8, 8)
        g1.setHorizontalSpacing(6)
        g1.setVerticalSpacing(5)
        g1.addWidget(_hint("Choose the stretch of time, then press a button."), 0, 0, 1, 3)
        for r, (label, spin, btn) in enumerate((("From", self.t_from, self.btn_from), ("To", self.t_to, self.btn_to)), start=1):
            row = QHBoxLayout()                                 # (the box and its button stay together however wide the pane is)
            row.setSpacing(6)
            row.addWidget(spin)
            row.addWidget(btn)
            row.addStretch(1)
            g1.addWidget(QLabel(label), r, 0)
            g1.addLayout(row, r, 1)
        g1.addWidget(self.btn_window, 1, 2, 2, 1)
        hold_row = QHBoxLayout()
        hold_row.setSpacing(6)
        hold_row.addWidget(self.btn_hold_usual, 1)
        hold_row.addWidget(self.btn_hold_click, 1)
        g1.addLayout(hold_row, 3, 0, 1, 3)
        g1.addWidget(self.btn_lost, 4, 0, 1, 3)
        g1.setColumnStretch(1, 1)

        # (4b) by hand
        self.chk_hand = QCheckBox("Set by hand: click the dot on each picture")
        self.chk_hand.setFocusPolicy(Qt.NoFocus)
        self.chk_hand.toggled.connect(self._hand_toggled)
        self.chk_hand.clicked.connect(self.release_keyboard)
        self.chk_step = QCheckBox("step on to the next picture after each click")
        self.chk_step.setFocusPolicy(Qt.NoFocus)
        self.chk_step.setChecked(True)
        self.chk_step.clicked.connect(self.release_keyboard)
        self.pending_label = ElidedLabel("")
        self.btn_save_hand = self._button("Save these positions", "Keep the clicked positions (a straight line is laid between two clicks).", self._save_hand)
        self.btn_drop_hand = self._button("Discard", "Forget the positions clicked and not saved.", self._drop_hand)
        self.keys_label = _hint(STEP_KEYS)                      # (which keys step, said where they are needed)
        self.box_hand = QGroupBox("By hand, picture by picture")
        v2 = QVBoxLayout(self.box_hand)
        v2.setContentsMargins(8, 6, 8, 8)
        v2.setSpacing(4)
        v2.addWidget(self.chk_hand)
        v2.addWidget(self.chk_step)
        v2.addWidget(self.keys_label)
        save_row = QHBoxLayout()
        save_row.setSpacing(6)
        save_row.addWidget(self.pending_label, 1)
        save_row.addWidget(self.btn_save_hand)
        save_row.addWidget(self.btn_drop_hand)
        v2.addLayout(save_row)

        # (4c) re-track, closed to begin with
        self.btn_back, self.btn_fwd, self.btn_both = (self._button(t, "") for t in ("Backward", "Forward", "Both ways"))      # (no arrow symbols: another font makes the buttons tall)
        for b, d in ((self.btn_back, "backward"), (self.btn_fwd, "forward"), (self.btn_both, "both")):
            b.clicked.connect(lambda _=False, direction=d: self._retrack(direction))
            b.setToolTip("Click the dot on the picture first. The program's tracker (the same as when recording) follows it over the saved video from that "
                         "picture, until the next picture you fixed by hand or the end. It helps where a dot was hidden and can be picked up again; where "
                         "the dot's picture is poor (blurred, low contrast) it fails the way it did when recording: use Hold or set it by hand then.")
        self.box_retrack = CollapsibleBox("Re-track with the program's tracker")
        self.box_retrack.body_layout.addWidget(_hint("Click the dot on the picture, then press a button. The tracker follows the dot through the video from that picture."))
        rt = QHBoxLayout()
        for b in (self.btn_back, self.btn_fwd, self.btn_both):
            rt.addWidget(b)
        self.box_retrack.body_layout.addLayout(rt)

        # (5) the fixes of this dot, and what just happened
        self.edits_title = QLabel("Fixes of this dot")
        self.edits_title.setFont(f)
        self.edits = QListWidget()
        self.edits.setFocusPolicy(Qt.NoFocus)
        self.edits.setVisible(False)                           # (only when this dot has fixes)
        self.edits_none = _hint("No fixes of this dot yet.", wrap=False)
        self.btn_undo = self._button("Take back the last fix", "Take back the last fix of this dot.", self._undo)
        self.btn_reset = self._button("Reset this dot", "Back to the positions recorded with the trial.", self._reset)
        undo_row = QHBoxLayout()
        undo_row.setSpacing(6)
        undo_row.addWidget(self.btn_undo)
        undo_row.addWidget(self.btn_reset)
        self.status = _Wrapped("")

        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(self.heading)
        lay.addWidget(self.dots)
        lay.addLayout(title_row)
        lay.addWidget(self.timeline)
        lay.addWidget(self.timeline_hint)
        lay.addWidget(self.box_hold)
        lay.addWidget(self.box_hand)
        lay.addWidget(self.box_retrack)
        lay.addWidget(self.edits_title)
        lay.addWidget(self.edits)
        lay.addWidget(self.edits_none)
        lay.addLayout(undo_row)
        lay.addWidget(self.status)
        lay.addStretch(1)
        # the keyboard goes back to the picture after a number was typed or its arrows were used (a spin box would keep , . and the arrows)
        self.t_from.editingFinished.connect(self.release_keyboard)
        self.t_to.editingFinished.connect(self.release_keyboard)
        self.t_from.arrowUsed.connect(self.release_keyboard)
        self.t_to.arrowUsed.connect(self.release_keyboard)
        self.box_retrack.button.clicked.connect(self.release_keyboard)
        for w in (self, inner):                                # (a click on empty room in the column must not take the picture keys either)
            w.setFocusPolicy(Qt.NoFocus)
        self._controls = (self.t_from, self.t_to, self.btn_from, self.btn_to, self.btn_window, self.btn_hold_usual, self.btn_hold_click, self.btn_lost,
                          self.btn_back, self.btn_fwd, self.btn_both, self.chk_hand, self.chk_step, self.btn_undo, self.btn_reset)
        self.set_trial(None)

    def minimumSizeHint(self):
        """As wide as the column's contents need (the splitter must not squeeze it below that: nothing here scrolls sideways)."""
        inner = self.widget()
        w = inner.minimumSizeHint().width() if inner is not None else 200
        return QSize(w + self.verticalScrollBar().sizeHint().width() + 2 * self.frameWidth(), 120)

    def _button(self, text, tip="", slot=None):
        """A pane button never takes the keyboard (the picture keys keep working after a click on it); the keyboard is given back to the picture first."""
        b = QPushButton(text)
        b.setFocusPolicy(Qt.NoFocus)
        if tip:
            b.setToolTip(tip)
        b.clicked.connect(self.release_keyboard)
        if slot is not None:
            b.clicked.connect(lambda _=False: slot())
        return b

    def release_keyboard(self, *_):
        """Take the keyboard from a spin box (or any other control of this pane) and give it to the picture, so the arrow keys and , . step the video."""
        fw = QApplication.focusWidget()
        if fw is None or not (fw is self or self.isAncestorOf(fw)):
            return
        video = getattr(self.tab, "video", None)
        if isinstance(video, QWidget):
            video.setFocus(Qt.OtherFocusReason)
        else:
            fw.clearFocus()

    # ---------------------------------------------------------------------------------------------------------- state
    def set_trial(self, trial):
        """A trial was loaded (or reloaded after a fix): fill the dot list, the table and the timeline, and show its fixes. The same trial again (a fix
        was saved) keeps the dot, the stretch of time, the click and the positions clicked by hand and not saved yet; another trial starts afresh, on
        the dot that needs a look most."""
        path = None if trial is None else trial.meta_path
        same = trial is not None and path == self._trial_path
        keep = self.dot.currentText() if same else ""
        self.dot.blockSignals(True)
        self.dot.clear()
        if trial is not None:
            self.dot.addItems(trial.dot_names)
            if self.dot.count():
                self.dot.setCurrentIndex(max(self.dot.findText(keep), 0))
            last = float(trial.t[-1])
            for sp in (self.t_from, self.t_to):
                sp.setRange(0.0, last)
            if not same:
                self.t_from.setValue(0.0)
                self.t_to.setValue(last)
        self.dot.blockSignals(False)
        if not same:
            self.pending, self._stash, self._pending_dot, self.click = {}, {}, None, None
            self._want_default = trial is not None
        self._trial_path = path
        self._shown = None                             # (the table and the timeline are made again)
        self._usual_key = None
        self.refresh()

    @property
    def trial(self):
        return self.tab.trial

    def current_dot(self):
        return self.dot.currentText() or None

    def focus_point(self, i):
        """Where a magnified picture is centred while this dot is fixed (0.32.0): the position clicked by hand on the nearest picture at or before `i`,
        else the last click, else None (the tab then uses where the program has the dot)."""
        earlier = [k for k in self.pending if k <= i]
        if earlier:
            return self.pending[max(earlier)]
        if self.click is not None:
            return self.click[0], self.click[1]
        return None, None

    def refresh(self):
        tr = self.trial
        self._video_ok = has_video(tr)
        has = self._video_ok and bool(tr.dot_names)
        for w in self._controls:
            w.setEnabled(has)
        self._sync_numbers(tr)
        self.edits.clear()
        dot = self.current_dot()
        self._sync_pending(dot)
        review = tr.review if tr is not None else None
        mine = [e for e in (review.dots if review is not None else []) if dot is not None and e["dot"] == dot]
        for e in mine:
            self.edits.addItem(self._describe(e, review))
        self.edits.setVisible(bool(mine))
        self.edits.setFixedHeight(min(len(mine), 4) * ROW_HEIGHT + 4)
        self.edits_none.setVisible(not mine)
        self.btn_undo.setEnabled(has and bool(mine))
        self.btn_reset.setEnabled(has and bool(mine))
        n = len(self.pending) if has else 0
        self.pending_label.setText(f"{n} picture{'' if n == 1 else 's'} clicked, not saved" if n else "")
        self.btn_save_hand.setEnabled(n > 0)
        self.btn_drop_hand.setEnabled(n > 0)
        bar = getattr(self.tab, "_update_fix_bar", None)          # (the line in the bottom bar says how many clicks wait to be saved)
        if bar is not None:
            bar()
        if not has or dot is None:
            self.where.setText("Open a trial with a video to fix a dot." if tr is None else
                               "This trial's video cannot be opened, so its dots cannot be fixed here." if not self._video_ok else "")
            return
        key = (self._sig, dot)
        if key != self._usual_key:
            self._usual, self._usual_key = dot_edits.usual_position(tr, dot), key
        self.update_where()

    def _sync_numbers(self, tr):
        """The table, the pull numbers and the timeline, worked out again only when another trial is shown or its fixes changed (not for every click)."""
        sig = trial_signature(tr)
        if tr is not self._shown or sig != self._sig:
            self._shown, self._sig = tr, sig
            self._timeline_key = None
            self.dots.set_trial(tr)
            if self._want_default and tr is not None:
                self._want_default = False
                self._choose_default()
        dot = self.current_dot()
        self.dots.select_dot(dot)
        if (sig, dot) != self._timeline_key:
            self._timeline_key = (sig, dot)
            self._show_timeline(tr, dot)

    def _choose_default(self):
        """A trial was opened: start on the dot that needs a look most (the most pictures lost, then the most jumps), else on the anchor, else the first."""
        bad = [s for s in self.table.stats.values() if s.needs_attention]
        name = max(bad, key=lambda s: (s.lost_pct, s.jumps)).name if bad else "anchor"
        i = self.dot.findText(name)
        if i >= 0 and i != self.dot.currentIndex():
            self.dot.blockSignals(True)
            self.dot.setCurrentIndex(i)
            self.dot.blockSignals(False)

    def _show_timeline(self, tr, dot):
        s = self.table.stats.get(dot) if dot is not None else None
        self.dot_title.setText(dot or "")
        if tr is None or s is None:
            self.timeline.clear_data()
            return
        self.timeline.set_data(tr.t, s.step, s.lost_runs, s.fixed_runs, self.table.limit_px, self.table.window)
        self.timeline.set_cursor(tr.t[min(max(self.tab.current_index(), 0), len(tr.t) - 1)])

    def _sync_pending(self, dot):
        """Clicked positions belong to the dot they were clicked for: they are kept aside when another dot is chosen, and come back with it."""
        if dot == self._pending_dot:
            return
        if self._pending_dot is not None and self.pending:
            self._stash[self._pending_dot] = self.pending
            self.say(f"{self._pending_dot}: {len(self.pending)} clicked positions are kept, not saved. Choose {self._pending_dot} again to save or discard them.")
        self.pending = self._stash.pop(dot, {})
        self._pending_dot = dot

    def _row_chosen(self, name):
        """A row of the table was clicked: that dot is the one to work on, and the cross on the picture moves to it."""
        i = self.dot.findText(name)
        if i >= 0:
            self.dot.setCurrentIndex(i)
            self.release_keyboard()
            if self._video_ok:
                self.tab.show_frame(self.tab.current_index())

    def _timeline_picked(self, seconds):
        """A click on the timeline: the video goes to the nearest picture."""
        self.release_keyboard()
        if self.trial is not None:
            self.tab.slider.setValue(self._nearest(seconds))

    def _nearest(self, seconds):
        """The number of the picture nearest to a time."""
        t = self.trial.t
        k = int(np.searchsorted(t, seconds))
        if k > 0 and (k >= len(t) or seconds - t[k - 1] <= t[k] - seconds):
            k -= 1
        return int(min(max(k, 0), len(t) - 1))

    def update_where(self):
        """The line beside the dot's name (its usual position and where it is on the picture on screen) and the yellow line of the timeline. Cheap:
        the tab calls this for every picture."""
        tr, dot = self.trial, self.current_dot()
        if tr is None or dot is None or dot not in tr.tracks or not self._video_ok:
            return
        i = min(max(self.tab.current_index(), 0), len(tr.t) - 1)
        x, y = tr.tracks[dot][0][i], tr.tracks[dot][1][i]
        usual = self._usual
        self.where.setText((f"usual position ({usual[0]:.1f}, {usual[1]:.1f}); " if usual else "") +
                           ("on this picture: lost" if math.isnan(x) else f"on this picture: ({x:.1f}, {y:.1f})"))
        self.timeline.set_cursor(tr.t[i])

    @staticmethod
    def _describe(e, review):
        if e["kind"] == "hold":
            return f"held at ({e['x']:.1f}, {e['y']:.1f}), {e['from_s']:.2f} to {e['to_s']:.2f} s"
        if e["kind"] == "lost":
            return f"marked lost, {e['from_s']:.2f} to {e['to_s']:.2f} s"
        k = sum(1 for r in review.dot_rows if r["edit"] == e["id"])
        return f"{'re-tracked' if e['kind'] == 'retrack' else 'set by hand'}: {k} picture{'' if k == 1 else 's'}"

    def say(self, text):
        self.status.setText(text)

    # ---------------------------------------------------------------------------------------------------------- the picture was clicked
    def hand_mode(self):
        return self.chk_hand.isChecked()

    def on_video_click(self, x, y):
        """The tab passes a click on the video here, in full-picture pixels."""
        self.release_keyboard()                # (a spin box left with focus would swallow the picture keys)
        if self.trial is None or self.current_dot() is None:
            return
        i = self.tab.current_index()
        self.click = (float(x), float(y), i)
        if self.hand_mode():
            self.pending[i] = (float(x), float(y))
            self.refresh()
            if self.chk_step.isChecked() and i + 1 < len(self.trial.t):
                self.tab.slider.setValue(i + 1)
            else:
                self.tab.show_frame(i)
        else:
            self.say(f"Clicked ({x:.1f}, {y:.1f}) on picture {i + 1}. Now press 'Hold where I clicked', or a re-track button.")       # (the number the picture counter shows)
            self.tab.show_frame(i)

    def marks(self, i):
        """What the tab should draw on the video for picture i: [(x, y, text, colour BGR)] (the clicked positions waiting to be saved, the last click)."""
        out = [(x, y, "", (0, 215, 255)) for f, (x, y) in self.pending.items() if f == i]
        if self.click is not None and self.click[2] == i and not self.hand_mode():
            out.append((self.click[0], self.click[1], "click", (0, 215, 255)))
        return out

    # ---------------------------------------------------------------------------------------------------------- actions
    def _range_from_window(self):
        """The first and the last picture inside the blue window."""
        t0, t1 = (float(v) for v in self.tab.window.getRegion())
        t = self.trial.t
        self.t_from.setValue(float(t[min(int(np.searchsorted(t, t0, side="left")), len(t) - 1)]))
        self.t_to.setValue(float(t[max(int(np.searchsorted(t, t1, side="right")) - 1, 0)]))

    def _range(self):
        """(from, to) seconds for a fix. The boxes show hundredths, so each is moved to the picture nearest to it (the picture on screen, or the last
        one of the trial, must not be missed because its time rounds the wrong way) and the stored times are widened to the thousandth."""
        a, b = sorted((self.t_from.value(), self.t_to.value()))
        t = self.trial.t
        return math.floor(float(t[self._nearest(a)]) * 1000.0) / 1000.0, math.ceil(float(t[self._nearest(b)]) * 1000.0) / 1000.0

    def _hold(self, use_click):
        dot = self.current_dot()
        if dot is None:
            return
        if use_click:
            if self.click is None:
                self.say("Click the dot on the picture first.")
                return
            x, y = self.click[0], self.click[1]
        else:
            usual = dot_edits.usual_position(self.trial, dot)
            if usual is None:
                self.say("This dot was never seen: click it on the picture and use 'Hold where I clicked'.")
                return
            x, y = usual
        t0, t1 = self._range()
        if self.tab.commit_dot_edit(dot, "hold", from_s=round(t0, 3), to_s=round(t1, 3), x=round(x, 2), y=round(y, 2)):
            self.say(f"{dot} is held at ({x:.1f}, {y:.1f}) from {t0:.2f} to {t1:.2f} s.")

    def _mark_lost(self):
        dot = self.current_dot()
        if dot is None:
            return
        t0, t1 = self._range()
        if self.tab.commit_dot_edit(dot, "lost", from_s=round(t0, 3), to_s=round(t1, 3)):
            self.say(f"{dot} counts as lost from {t0:.2f} to {t1:.2f} s.")

    def _retrack(self, direction):
        tr, dot = self.trial, self.current_dot()
        if tr is None or dot is None:
            return
        i = self.tab.current_index()
        if self.click is not None and self.click[2] == i:
            seed = (self.click[0], self.click[1])
        else:
            x, y = tr.tracks[dot][0][i], tr.tracks[dot][1][i]
            if math.isnan(x):
                self.say("Click the dot on this picture first.")
                return
            seed = (float(x), float(y))
        stop = set()
        if tr.review is not None:
            by_id = {e["id"]: e for e in tr.review.dots}
            stop = {r["frame"] for r in tr.review.dot_rows if r["dot"] == dot and by_id.get(r["edit"], {}).get("kind") == "manual"}
        n = len(tr.t)
        progress = QProgressDialog(f"Following {dot} through the video...", "Stop", 0, n, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        state = {"stopped": False}

        def tick(done, total):
            progress.setMaximum(total)
            progress.setValue(done)
            if progress.wasCanceled():
                state["stopped"] = True
                return False
            return True
        rows, why = dot_edits.retrack(tr, dot, i, seed, direction, stop_at=stop, progress=tick)
        progress.close()
        self.tab.rewind_video()                                # (the tracker moved the video's read position)
        if why:
            self.say(why)
            return
        lost = sum(1 for r in rows if math.isnan(r["x"]))
        if self.tab.commit_dot_edit(dot, "retrack", rows=rows, seed_frame=i, seed_xy=[round(seed[0], 2), round(seed[1], 2)], direction=direction):
            msg = f"{dot} re-tracked over {len(rows)} pictures ({lost} of them it could not find: they now count as lost)." + (" Stopped early." if state["stopped"] else "")
            if rows and lost > 0.3 * len(rows):
                msg += (" The tracker could not follow this dot well (its picture may be blurred or low in contrast). For a fixed dot use 'Hold at its usual "
                        "position'; otherwise set it by hand, or take this fix back.")
            self.say(msg)

    def _hand_toggled(self, on):
        if not on and self.pending:
            self.say(f"{len(self.pending)} clicked positions are waiting: press 'Save these positions' or 'Discard'.")
        self.say(("Click the dot on this picture; it steps on to the next." if self.chk_step.isChecked() else "Click the dot on this picture.") if on else self.status.text())
        if self._video_ok:
            self.tab.show_frame(self.tab.current_index())       # (the ring of the last click comes off the picture, or the clicked positions go on)

    def _save_hand(self):
        dot = self.current_dot()
        if dot is None or not self.pending:
            return
        tr = self.trial
        rows = [{"frame": f, "t_s": float(tr.t[f]), "x": x, "y": y} for f, (x, y) in sorted(self.pending.items())]
        if self.tab.commit_dot_edit(dot, "manual", rows=rows):
            self.say(f"{dot}: {len(rows)} position{'' if len(rows) == 1 else 's'} saved (a straight line between them).")
            self.pending = {}
            self.refresh()

    def _drop_hand(self):
        self.pending = {}
        self.refresh()
        self.tab.show_frame(self.tab.current_index())

    def _undo(self):
        dot = self.current_dot()
        if dot is not None and self.tab.undo_dot_edit(dot):
            self.say(f"The last fix of {dot} is taken back.")

    def _reset(self):
        dot = self.current_dot()
        if dot is not None and self.tab.reset_dot(dot):
            self.say(f"{dot} is back to the recorded positions.")
