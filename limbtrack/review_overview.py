"""The participant overview of the Review tab (0.29.0): one line per trial of the open participant, with its verdict, its force numbers and
how many automatic checks it fails, so a whole participant can be looked through, sorted and exported without opening the trials one by
one. The numbers are the export's (analysis.session_row / analysis.repeatability); they are worked out in a background thread, once,
and kept until the trial's files or the smoothing / sliding-window settings change.
"""
import math
import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QHBoxLayout, QHeaderView, QLabel, QProgressBar, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import analysis, review_nav as nav
from . import verdict as verdicts
from .review_style import BAD, GOOD, WARN

FILTERS = [("all", "All trials"), ("unjudged", "Not judged yet"), ("valid", "Valid"), ("compromised", "Compromised"),
           ("checks", "With checks to look at")]
COLUMNS = ["Verdict", "Pos", "Trial", "Test", "Target", "Time", "Peak", "Window peak", "Time to peak (s)", "Angle at peak (deg)", "Checks", "10 % check"]
RIGHT_ALIGNED = {"Pos", "Trial", "Target", "Time", "Peak", "Window peak", "Time to peak (s)", "Angle at peak (deg)", "Checks"}
_BASE_ROLE = Qt.UserRole + 1
_RENDER_DELAY_MS = 120


def file_stamp(meta_path):
    """The modification times of the files a row depends on (meta, verdict, reference readings): a changed one makes the row out of date."""
    p = Path(meta_path)
    stem = p.name[: -len(nav.META_SUFFIX)]
    out = []
    for name in (p.name, stem + "_verdict.json", stem + "_reference.json"):
        try:
            out.append(os.stat(p.with_name(name)).st_mtime_ns)
        except OSError:
            out.append(0)
    return tuple(out)


class _Signals(QObject):
    produced = Signal(int, str, object, str)         # generation, trial base, row (or None), error text
    finished = Signal(int)


class _Job(QRunnable):
    """Works out the rows of a list of trials one after the other, off the window's thread, and stops when told to."""

    def __init__(self, generation, items, row_fn, smooth_s, window_s, signals, cancelled):
        super().__init__()
        self.setAutoDelete(True)
        self.generation, self.items, self.row_fn = generation, items, row_fn
        self.smooth_s, self.window_s, self.signals, self.cancelled = smooth_s, window_s, signals, cancelled

    def run(self):
        try:
            for base, path in self.items:
                if self.cancelled():
                    break
                try:
                    row, err = self.row_fn(path, self.smooth_s, self.window_s)
                except Exception as e:  # noqa: BLE001 - one trial must not stop the others; the reason is shown on its line
                    row, err = None, f"{type(e).__name__}: {e}"
                self.signals.produced.emit(self.generation, base, row, err)
        finally:
            self.signals.finished.emit(self.generation)


class _SortItem(QTableWidgetItem):
    """A table cell that sorts by a number (blank last) or by text, not by what it shows."""

    def __lt__(self, other):
        a, b = self.data(Qt.UserRole), other.data(Qt.UserRole)
        a = math.inf if a is None or (isinstance(a, float) and math.isnan(a)) else a
        b = math.inf if b is None or (isinstance(b, float) and math.isnan(b)) else b
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


def _num(v):
    return analysis._num(v)


def _fmt(v, unit, decimals=None):
    v = _num(v)
    if v is None:
        return "-"
    d = decimals if decimals is not None else (3 if unit == "V" else 0)
    return f"{v:.{d}f}"


class OverviewPage(QWidget):
    """The table. `row_fn(meta_path, smooth_s, window_s) -> (row dict or None, error text)` does the work (review.trial_row)."""

    openRequested = Signal(str)          # the base name of the trial that was double-clicked / activated
    exportRequested = Signal(bool)       # True: valid trials only

    def __init__(self, row_fn, parent=None):
        super().__init__(parent)
        self.row_fn = row_fn
        self.entries = []
        self.name = ""
        self.smooth_s, self.window_s = 0.1, 0.5
        self._cache = {}                 # base -> (stamp + settings, row or None, error)
        self._generation = 0
        self._running = 0                # jobs not yet finished
        self._cancel = False
        self._pending = 0                # rows asked for in the current run, and how many came back
        self._done = 0
        self._stamps = {}                # base -> the stamp the rows of the current run are made for
        self._again = False              # a run was asked for while a cancelled one was still finishing
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.signals = _Signals(self)
        self.signals.produced.connect(self._on_row)
        self.signals.finished.connect(self._on_finished)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(_RENDER_DELAY_MS)
        self._render_timer.timeout.connect(self.render)

        self.summary = QLabel("Choose a participant.")
        self.summary.setObjectName("trialTitle")
        self.filter = QComboBox()
        for key, text in FILTERS:
            self.filter.addItem(text, key)
        self.filter.currentIndexChanged.connect(lambda _: self.render())
        self.btn_all = QPushButton("Export all trials (CSV)")
        self.btn_all.setToolTip("One row per trial of this participant, with the verdict, its reasons and note, and every force number. "
                                "Trials marked compromised are in the file, marked as such.")
        self.btn_all.clicked.connect(lambda: self.exportRequested.emit(False))
        self.btn_valid = QPushButton("Export valid trials only (CSV)")
        self.btn_valid.setToolTip("The same file, with only the trials you marked valid. Not judged and compromised trials are left out.")
        self.btn_valid.clicked.connect(lambda: self.exportRequested.emit(True))
        top = QHBoxLayout()
        top.addWidget(self.summary, 1)
        top.addWidget(QLabel("Show:"))
        top.addWidget(self.filter)
        top.addWidget(self.btn_all)
        top.addWidget(self.btn_valid)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("overviewTable")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        hh = self.table.horizontalHeader()
        hh.setSortIndicator(-1, Qt.AscendingOrder)                 # recording order until a heading is clicked
        self.table.setSortingEnabled(True)
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open_row(r))
        self.table.itemActivated.connect(lambda it: self._open_row(it.row()))
        self.progress = QProgressBar()
        self.progress.setObjectName("overviewProgress")
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.hint = QLabel("Double-click a line (or press Enter) to open the trial. Click a heading to sort. Numbers are worked out once "
                           "and kept; a changed verdict, smoothing or sliding window refreshes them.")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.progress)
        lay.addWidget(self.hint)

    # ----------------------------------------------------------------------------------------- what is shown
    def set_participant(self, name, entries):
        """Show these trials (a review_nav list). Nothing is worked out until `ensure_rows`."""
        self.cancel()
        self.name, self.entries = name, list(entries)
        self.render()

    def set_settings(self, smooth_s, window_s):
        self.smooth_s, self.window_s = float(smooth_s), float(window_s)

    def _stamp(self, entry):
        return file_stamp(entry.meta_path) + (round(self.smooth_s, 4), round(self.window_s, 4))

    def row_for(self, base, allow_stale=False):
        """The cached row of a trial if it is current (its files and the settings are as they were when it was worked out), else None.
        `allow_stale`: also an out-of-date one (the table shows the old numbers until the new ones are in; an export never does)."""
        hit = self._cache.get(base)
        if not hit:
            return None
        if allow_stale:
            return hit[1]
        e = next((x for x in self.entries if x.base == base), None)
        return hit[1] if e is not None and hit[0] == self._stamp(e) else None

    def stale_entries(self):
        """The readable trials without a current cached result."""
        out = []
        for e in self.entries:
            if not e.readable:
                continue
            hit = self._cache.get(e.base)
            if not hit or hit[0] != self._stamp(e):
                out.append(e)
        return out

    def ensure_rows(self):
        """Start working out the rows that are missing or out of date, in the background."""
        todo = self.stale_entries()
        if not todo:
            return
        if self._running:                # a cancelled run is still finishing: start again when it has
            self._again = True
            return
        self._cancel = False
        self._generation += 1
        self._pending, self._done = len(todo), 0
        self._running += 1
        self.progress.setRange(0, self._pending)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        stamps = {e.base: self._stamp(e) for e in todo}
        self._stamps = stamps
        self.pool.start(_Job(self._generation, [(e.base, e.meta_path) for e in todo], self.row_fn, self.smooth_s, self.window_s,
                             self.signals, lambda: self._cancel))

    def cancel(self):
        """Stop a run in progress (the rows already done are kept)."""
        self._cancel = True

    def shutdown(self):
        self.cancel()
        self.pool.waitForDone(60000)

    def wait_until_done(self, timeout_ms=60000):
        """Let the window's own thread deliver what the background thread has produced, until it is finished (used by tests and by exports)."""
        end = time.monotonic() + timeout_ms / 1000.0
        while self._running and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        QApplication.processEvents()
        return not self._running

    def _on_row(self, generation, base, row, err):
        if generation != self._generation:
            return
        stamp = self._stamps.get(base)
        e = next((x for x in self.entries if x.base == base), None)
        if stamp is not None and e is not None:
            self._cache[base] = (stamp, row, err)
        self._done += 1
        self.progress.setValue(self._done)
        self._render_timer.start()

    def _on_finished(self, generation):
        if generation != self._generation:
            return
        self._running = max(0, self._running - 1)
        self.progress.setVisible(False)
        self.render()
        if self._again:
            self._again = False
            self.ensure_rows()

    def store(self, entry, row, err=""):
        """Keep a row worked out elsewhere (an export that did the missing ones itself)."""
        self._cache[entry.base] = (self._stamp(entry), row, err)

    def update_verdict(self, base):
        """A verdict was saved: show it at once. The line's checks name the verdict, so its numbers count as out of date and are worked out
        again the next time the overview is shown (`ensure_rows`); until then the old ones stay."""
        entry = next((e for e in self.entries if e.base == base), None)
        if entry is None:
            return
        entry.refresh_verdict()
        hit = self._cache.get(base)
        if hit and hit[1] is not None:
            v = verdicts.read_verdict(entry.meta_path)
            row = dict(hit[1])
            row["verdict"] = (v or {}).get("verdict") or ""
            row["verdict_reasons"] = ", ".join(verdicts.REASONS.get(r, r) for r in (v or {}).get("reasons") or [])
            row["verdict_note"] = (v or {}).get("note") or ""
            self._cache[base] = (hit[0], row, hit[2])
        self.render()

    # ----------------------------------------------------------------------------------------- drawing
    def _visible(self, entry, row):
        f = self.filter.currentData()
        if f == "unjudged":
            return entry.readable and entry.verdict is None
        if f == "valid":
            return entry.verdict == verdicts.VALID
        if f == "compromised":
            return entry.verdict == verdicts.COMPROMISED
        if f == "checks":
            return (not entry.readable) or bool(row and row.get("checks_to_look_at"))
        return True

    def render(self):
        """Redraw the table from the entries and the cached rows (the 10 % check is worked out over what is there)."""
        self._render_timer.stop()
        rows = {e.base: self.row_for(e.base, allow_stale=True) for e in self.entries}
        copies = {b: dict(r) for b, r in rows.items() if r is not None}
        analysis.repeatability(list(copies.values()))
        sort_col, sort_order = self.table.horizontalHeader().sortIndicatorSection(), self.table.horizontalHeader().sortIndicatorOrder()
        selected = self._selected_base()
        self.table.setSortingEnabled(False)
        shown = [e for e in self.entries if self._visible(e, rows.get(e.base))]
        self.table.setRowCount(len(shown))
        for r, e in enumerate(shown):
            for c, (text, key, tip, colour, bold) in enumerate(self._cells(e, copies.get(e.base))):
                it = _SortItem(text)
                it.setData(Qt.UserRole, key)
                it.setData(_BASE_ROLE, e.base)
                if tip:
                    it.setToolTip(tip)
                if colour:
                    it.setForeground(QBrush(QColor(colour)))
                if bold:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if not e.readable:
                    f = it.font()
                    f.setItalic(True)
                    it.setFont(f)
                if e.verdict == verdicts.COMPROMISED:
                    it.setBackground(QBrush(QColor(220, 38, 38, 26)))
                it.setTextAlignment(Qt.AlignVCenter | (Qt.AlignRight if COLUMNS[c] in RIGHT_ALIGNED else Qt.AlignLeft))
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        if sort_col >= 0:
            self.table.sortItems(sort_col, sort_order)
        if selected:
            self.select_base(selected)
        counts = nav.counts(self.entries)
        text = f"{self.name}: {len(self.entries)} trials · {counts['valid']} valid · {counts['compromised']} compromised · {counts['not judged']} not judged"
        if counts["unreadable"]:
            text += f" · {counts['unreadable']} cannot be read"
        if self._running:
            text += f"   (working out the numbers: {self._done} of {self._pending})"
        self.summary.setText(text if self.entries else "Choose a participant.")

    def _cells(self, e, row):
        """[(text, sort key, tooltip, colour, bold)] of one line."""
        verdict_text = f"{nav.GLYPHS[e.verdict]} {nav.WORDS[e.verdict]}"
        v = verdicts.read_verdict(e.meta_path)
        tip = verdicts.describe(v) if v else "not judged yet"
        colour = {verdicts.VALID: GOOD, verdicts.COMPROMISED: BAD}.get(e.verdict)
        order = {verdicts.COMPROMISED: 0, None: 1, verdicts.VALID: 2}[e.verdict]
        blank = ("", None, "", None, False)
        if not e.readable:
            return [("cannot be read", -1, e.error, WARN, False), blank, blank, (e.base, e.base.lower(), e.error, None, False)] + [blank] * (len(COLUMNS) - 4)
        pos = e.position_number if e.position_number not in (None, "") else None
        name = nav._short_test(e.test_label) or e.test_key
        target = _num(e.target_deg)
        cells = [(verdict_text, order, tip, colour, e.verdict is not None),
                 (f"{int(pos):02d}" if pos is not None else "", int(pos) if pos is not None else None, "", None, False),
                 (f"T{e.trial_number}" if e.trial_number not in (None, "") else "", _num(e.trial_number), "", None, False),
                 (name + (" (unfinished)" if e.unfinished else ""), name.lower(), e.base, None, False),
                 (f"{target:g}°" if target is not None else "", target, "", None, False),
                 (e.clock, e.started, e.started.replace("T", " "), None, False)]
        if row is None:                                   # not worked out yet ('…'), or it failed (the reason is on the line)
            err = self._cache[e.base][2] if e.base in self._cache else ""
            first = ("could not be read", None, err, WARN, False) if err else ("…", None, "", None, False)
            return cells + [first] + [blank] * (len(COLUMNS) - len(cells) - 1)
        unit = row.get("force_unit") or ""
        sfx = f" {unit}" if unit else ""
        has_force = bool(unit)
        peak, wpeak = _num(row.get("force_peak")), _num(row.get("force_window_peak"))
        cells += [(_fmt(peak, unit) + (sfx if peak is not None else ""), peak, "", None, False) if has_force else ("no force", None, "", None, False),
                  (_fmt(wpeak, unit) + (sfx if wpeak is not None else ""), wpeak,
                   f"highest mean over {row.get('force_window_peak_ms'):.0f} ms" if _num(row.get("force_window_peak_ms")) else "", None, False),
                  (_fmt(row.get("force_time_to_peak_s"), unit, 2), _num(row.get("force_time_to_peak_s")), "from the start of the push to the peak", None, False),
                  (_fmt(row.get("angle_at_peak_force_deg"), unit, 1), _num(row.get("angle_at_peak_force_deg")), str(row.get("main_angle") or ""), None, False)]
        checks = [c for c in (row.get("checks_to_look_at") or "").split(" | ") if c]
        cells.append((str(len(checks)) if checks else "0", len(checks), "\n".join(checks), WARN if checks else None, bool(checks)))
        pct, third = _num(row.get("repeatability_pct_same_position")), row.get("third_trial_needed")
        if pct is None:
            cells.append(("", None, "", None, False))
        else:
            cells.append((f"{pct:.1f} %" + (" · 3rd trial needed" if third is True else ""), pct,
                          "How far the peaks of the trials of this position are apart, as a share of their mean (SOP 10.6)",
                          BAD if third is True else None, third is True))
        return cells

    def _selected_base(self):
        items = self.table.selectedItems()
        return items[0].data(_BASE_ROLE) if items else None

    def select_base(self, base):
        """Highlight the line of this trial (the one that is open in the Trial view)."""
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.data(_BASE_ROLE) == base:
                self.table.selectRow(r)
                self.table.scrollToItem(it)
                return

    def _open_row(self, r):
        it = self.table.item(r, 0)
        if it is not None:
            self.openRequested.emit(it.data(_BASE_ROLE))
