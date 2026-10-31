"""The participant overview of the Review tab (0.29.0; rebuilt in 0.32.0): one line per trial of the open participant, with its verdict, its
force numbers and how many automatic checks it fails, so a whole participant can be looked through, sorted and exported without opening the
trials one by one. The numbers are the export's (analysis.session_row / analysis.repeatability).

How the first load stays quick (measured on 64 trials of P02, see CHANGELOG 0.32.0):
  * the table is filled line by line, only the lines that changed are touched, and the columns are fitted once per batch (the old table
    asked Qt to fit every column after every cell, which froze the window for 6-7 s with 64 lines);
  * the rows are worked out off the window's thread, in a thread, and for a big participant also in a few worker processes
    (overview_worker.py), whoever is free takes the next trial; a worker that dies loses only the trial it was reading;
  * finished rows are kept in the participant's review folder (review/<participant>/overview_cache.json, through review_layer), keyed by
    the trial's files, the settings and the program version, so the next load needs no work and a changed file redoes only that trial.
The approval of a participant (review_layer.write_approval) is made here too.
"""
import json
import math
import os
import signal
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from . import analysis, overview_worker, review_layer, review_nav as nav
from . import verdict as verdicts
from .engine import SOFTWARE_VERSION
from .loadcell import REGISTRY_FILE
from .review_style import ACCENT, BAD, GOOD, WARN, ElidedLabel, tint

FILTERS = [("all", "All trials"), ("unjudged", "Not judged yet"), ("valid", "Valid"), ("compromised", "Compromised"),
           ("checks", "With checks to look at"), ("edited", "Edited in Review"), ("changed", "Changed since approved")]
# 0.32.0: the time of the test is among the first columns, the angle at the peak sits beside the target, and the sliding-window peak, the time
# to peak and the edits are at the right
COLUMNS = ["Verdict", "Trial", "Time", "Pos", "Test", "Target", "Angle at peak (deg)", "Other angle at peak", "Peak", "Checks", "10 % check",
           "Window peak", "Time to peak (s)", "Edits"]                 # (0.33.0: the other joints of the test at the peak force sit beside the main one)
RIGHT_ALIGNED = {"Trial", "Time", "Pos", "Target", "Angle at peak (deg)", "Peak", "Checks", "Window peak", "Time to peak (s)"}
COLUMN_TIPS = {
    "Verdict": "What you judged: valid, compromised (the reasons are in the tip of the cell) or not judged yet.",
    "Trial": "The trial number of its position (T1, T2, T3).",
    "Time": "When the trial was recorded.",
    "Pos": "The position number in the session plan.",
    "Test": "The test. Hover a cell for the file name.",
    "Target": "The joint angle the participant was asked to hold.",
    "Angle at peak (deg)": "The joint angle at the moment of the peak force. Hover a cell for the joint.",
    "Other angle at peak": "The other joint or joints of the test at the same moment (the hip in a knee test, the knee in a hip test). "
                           "Hover a cell for the full names.",
    "Peak": "The highest force of the trial, as in the export.",
    "Checks": "How many automatic checks say this trial needs a look. Hover a number to read them.",
    "10 % check": "How far apart the peaks of the trials of one position are, as a share of their mean (SOP 10.6). "
                  "Trials marked compromised are not counted.",
    "Window peak": "The highest average force over the sliding window (its length is set in the Trial view).",
    "Time to peak (s)": "Seconds from the start of the push to the peak.",
    "Edits": "A pencil: this trial was edited in Review (a force zero or dot fixes). The originals are never changed.",
}
_BASE_ROLE = Qt.UserRole + 1
_FLUSH_MS = 100                      # the table is brought up to date at most this often while rows arrive
BLANK = ("", None, "", None, False)
MAX_PROCESSES = 4                    # worker processes at most (measured: more than 4-6 does not help, each holds ~100 MB)
PROCESS_MIN_TODO = 40                # fewer trials than this are worked out in the thread alone: a worker needs ~2.7 s to start, in which the
                                     # thread has done about 35 trials by itself
TRIALS_PER_PROCESS = 8               # and no more workers than one per this many trials
READY_TIMEOUT_MS = 60000             # a worker that has not said it is ready by then (a stuck import) is given up
JOB_TIMEOUT_MS = 180000              # a trial a worker has not answered by then is given up (a trial takes 0.1-0.3 s)
STOP_GRACE_MS = 3000                 # a worker told to quit gets this long before it is ended
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def file_stamp(meta_path):
    """The modification times of the files a row depends on (meta, the on-site verdict, an older reference file, and everything in the review
    layer: edits, verdict, reference readings): a changed one makes the row out of date."""
    p = Path(meta_path)
    stem = p.name[: -len(nav.META_SUFFIX)]
    out = []
    for name in (p.name, stem + "_verdict.json", stem + "_reference.json"):
        try:
            out.append(os.stat(p.with_name(name)).st_mtime_ns)
        except OSError:
            out.append(0)
    return tuple(out) + review_layer.stamp(p)


def settings_stamp(smooth_s, window_s, source):
    """The rest of a row's key: the smoothing, the sliding window, original or reviewed, the program version (a new version may work the
    numbers out differently) and the load cell calibrations (a row's newtons come from them)."""
    try:
        registry = os.stat(REGISTRY_FILE).st_mtime_ns
    except OSError:
        registry = 0
    return (round(float(smooth_s), 4), round(float(window_s), 4), str(source), SOFTWARE_VERSION, registry)


def plain_row(row):
    """A row as plain JSON values (numpy numbers become Python ones): the same whether it was worked out here, in a worker or read back from
    the cache, so the three can be compared and shown alike."""
    return None if row is None else json.loads(overview_worker.dumps(row))


class _Signals(QObject):
    produced = Signal(int, str, object, str, str)    # generation, trial base, row (or None), error text, who worked it out
    finished = Signal(int)


class _Todo:
    """The trials still to be worked out. The thread and the worker processes take from it, whoever is free first; `cancel` empties it."""

    def __init__(self, items):
        self._items = deque(items)
        self._lock = threading.Lock()
        self.cancelled = False

    def take(self):
        with self._lock:
            return None if self.cancelled or not self._items else self._items.popleft()

    def cancel(self):
        with self._lock:
            self.cancelled = True

    def remaining(self):
        with self._lock:
            return list(self._items)


class _Job(QRunnable):
    """Works out trials one after the other, off the window's thread, until there are no more or it is cancelled."""

    def __init__(self, generation, todo, row_fn, smooth_s, window_s, source, signals):
        super().__init__()
        self.setAutoDelete(True)
        self.generation, self.todo, self.row_fn = generation, todo, row_fn
        self.smooth_s, self.window_s, self.source, self.signals = smooth_s, window_s, source, signals

    def run(self):
        try:
            while True:
                item = self.todo.take()
                if item is None:
                    break
                base, path = item
                try:
                    row, err = self.row_fn(path, self.smooth_s, self.window_s, self.source)
                except Exception as e:  # noqa: BLE001 - one trial must not stop the others; the reason is shown on its line
                    row, err = None, f"{type(e).__name__}: {e}"
                self.signals.produced.emit(self.generation, base, row, err, "thread")
        finally:
            self.signals.finished.emit(self.generation)


# ------------------------------------------------------------------------------------------------ worker processes
class _Worker:
    """One worker process as the window sees it (see overview_worker.py for what it says and hears)."""

    def __init__(self, number, proc):
        self.number, self.proc = number, proc
        self.state = "starting"          # starting, idle, busy, quitting, dead
        self.buf = b""
        self.job = None                  # (job id, trial base) while it is reading a trial
        self.pid = 0                     # the interpreter itself (in a virtual environment a small launcher stands in front of it)
        self.rows = 0
        self.memory_mb = self.peak_mb = 0
        self.err = b""                   # the end of what it wrote to stderr: the reason shown if it dies
        self.reason = ""                 # why it died, if it did
        self.timer = QTimer()
        self.timer.setSingleShot(True)

    @property
    def who(self):
        return f"process {self.number}"


class _ProcessPool(QObject):
    """A few worker processes taking trials from a shared `_Todo`. Driven by the window's event loop: nothing here blocks except `stop(wait_ms)`.
    A worker that dies (crashes, is killed, does not answer in time) loses only the trial it was reading: that trial is reported with the reason;
    the others carry on with the rest."""

    rowReady = Signal(str, object, str, str)         # trial base, row (or None), error text, who
    changed = Signal()                               # a worker became ready, answered or died: the run may be finished

    def __init__(self, spec, todo, settings, count, python=None, cwd=None, extra_path=(), ready_ms=READY_TIMEOUT_MS, job_ms=JOB_TIMEOUT_MS,
                 parent=None):
        super().__init__(parent)
        self.spec, self.todo, self.settings, self.count = spec, todo, settings, int(count)
        self.python = str(python or overview_worker.python_command())
        self.cwd = Path(cwd) if cwd else Path(__file__).resolve().parent.parent
        self.extra_path = [str(p) for p in extra_path]
        self.ready_ms, self.job_ms = int(ready_ms), int(job_ms)
        self.workers = []
        self._next_id = 0

    # ---- starting
    def start(self):
        for n in range(1, self.count + 1):
            self._spawn(n)

    def _spawn(self, number):
        proc = QProcess(self)
        w = _Worker(number, proc)
        proc.setProgram(self.python)
        proc.setArguments(["-m", overview_worker.MODULE, "--row-fn", self.spec])
        proc.setWorkingDirectory(str(self.cwd))
        env = QProcessEnvironment.systemEnvironment()
        old = env.value("PYTHONPATH", "")
        env.insert("PYTHONPATH", os.pathsep.join([str(self.cwd), *self.extra_path] + ([old] if old else [])))
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(lambda w=w: self._read(w))
        proc.readyReadStandardError.connect(lambda w=w: self._read_err(w))
        proc.finished.connect(lambda code, status, w=w: self._finished(w, code, status))
        proc.errorOccurred.connect(lambda err, w=w: self._error(w, err))
        w.timer.timeout.connect(lambda w=w: self._timeout(w))
        self.workers.append(w)
        w.timer.start(self.ready_ms)
        proc.start()

    # ---- what the workers say
    def _read(self, w):
        w.buf += bytes(w.proc.readAllStandardOutput())
        while b"\n" in w.buf:
            line, w.buf = w.buf.split(b"\n", 1)
            self._line(w, line)

    def _read_err(self, w):
        w.err = (w.err + bytes(w.proc.readAllStandardError()))[-4000:]

    def _line(self, w, line):
        try:
            msg = json.loads(line.decode("ascii", "replace"))
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        if "ready" in msg:
            if msg["ready"] and w.state == "starting":
                w.pid, w.memory_mb, w.state = int(msg.get("pid") or 0), msg.get("memory_mb") or 0, "idle"
                w.timer.stop()
                self._feed(w)
            elif not msg["ready"]:
                self._dead(w, f"could not load the row function ({msg.get('error', '')})")
        elif "id" in msg:
            if w.job is None or msg["id"] != w.job[0]:
                return
            base = w.job[1]
            w.job, w.state = None, "idle"
            w.timer.stop()
            w.rows += 1
            self.rowReady.emit(base, msg.get("row"), str(msg.get("error") or ""), w.who)
            self._feed(w)
        elif msg.get("bye"):
            w.peak_mb = msg.get("peak_mb") or 0
        self.changed.emit()

    def _feed(self, w):
        """Give an idle worker the next trial, or tell it to quit when there is none."""
        if w.state != "idle":
            return
        item = self.todo.take()
        if item is None:
            self._quit(w)
            return
        base, path = item
        self._next_id += 1
        w.job, w.state = (self._next_id, base), "busy"
        w.timer.start(self.job_ms)
        smooth, window, source = self.settings
        w.proc.write((overview_worker.dumps({"id": self._next_id, "path": str(path), "smooth": smooth, "window": window, "source": source})
                      + "\n").encode("ascii"))

    def _quit(self, w):
        if w.state in ("quitting", "dead"):
            return
        w.state = "quitting"
        w.timer.stop()
        w.proc.write(b'{"cmd":"quit"}\n')
        w.proc.closeWriteChannel()
        w.timer.start(STOP_GRACE_MS)                     # (a worker that has not left by then is ended)

    # ---- how workers end
    def _finished(self, w, code, status):
        self._read(w)                                    # whatever it said last
        if w.state == "dead":
            self.changed.emit()
            return
        clean = w.state == "quitting" and status == QProcess.ExitStatus.NormalExit
        if clean:
            w.state = "dead"
            w.timer.stop()
            self.changed.emit()
            return
        tail = w.err.decode("utf-8", "replace").strip().splitlines()
        why = "crashed" if status == QProcess.ExitStatus.CrashExit else f"stopped (exit code {code})"
        self._dead(w, f"the worker process {why}" + (f": {tail[-1][:200]}" if tail else ""))

    def _error(self, w, err):
        if err == QProcess.ProcessError.FailedToStart:
            self._dead(w, f"the worker process could not be started ({w.proc.errorString()})")
        elif err == QProcess.ProcessError.WriteError:
            self._dead(w, "the worker process could not be reached")
        # (a crash is followed by `finished`, which says it)

    def _timeout(self, w):
        if w.state == "quitting":
            self._end_if_running(w)
        elif w.state == "starting":
            self._dead(w, f"the worker process did not start within {self.ready_ms / 1000:.0f} s", kill=True)
        elif w.state == "busy":
            self._dead(w, f"the worker process did not answer within {self.job_ms / 1000:.0f} s", kill=True)

    def _dead(self, w, reason, kill=False):
        if w.state == "dead":
            return
        job, w.job, w.state, w.reason = w.job, None, "dead", reason
        w.timer.stop()
        if job is not None:                              # the only trial lost with it
            self.rowReady.emit(job[1], None, f"Could not be read: {reason}", w.who)
        if kill or w.proc.state() != QProcess.ProcessState.NotRunning:
            self._hard_kill(w)
        self.changed.emit()

    def _hard_kill(self, w):
        """End the worker for good: its launcher and, in a virtual environment, the interpreter behind it."""
        running = w.proc.state() != QProcess.ProcessState.NotRunning
        if running:
            w.proc.kill()
            if w.pid:
                try:
                    os.kill(w.pid, signal.SIGTERM)       # (TerminateProcess on Windows)
                except (OSError, ValueError):
                    pass

    def _end_if_running(self, w):
        if w.proc.state() != QProcess.ProcessState.NotRunning:
            self._hard_kill(w)
            if w.state != "dead":
                w.state = "dead"
                self.changed.emit()

    # ---- what the window asks
    def busy(self):
        return sum(1 for w in self.workers if w.state == "busy")

    def usable(self):
        """Workers that are, or will be, able to take a trial."""
        return sum(1 for w in self.workers if w.state in ("starting", "idle", "busy"))

    def alive(self):
        return sum(1 for w in self.workers if w.state != "dead" or w.proc.state() != QProcess.ProcessState.NotRunning)

    def ready(self):
        return sum(1 for w in self.workers if w.pid)

    def stop(self, wait_ms=0):
        """End the pool. Workers that are still starting are ended at once; idle ones are told to quit; busy ones quit after their trial (the todo
        is empty or cancelled by then). `wait_ms`: wait for them that long, then end whatever is left."""
        for w in self.workers:
            if w.state == "starting":
                w.state = "dead"
                w.timer.stop()
                self._hard_kill(w)
            elif w.state == "idle":
                self._quit(w)
        if wait_ms:
            end = time.monotonic() + wait_ms / 1000.0
            for w in self.workers:
                if w.proc.state() != QProcess.ProcessState.NotRunning:
                    w.proc.waitForFinished(max(1, int((end - time.monotonic()) * 1000)))
            for w in self.workers:
                if w.proc.state() != QProcess.ProcessState.NotRunning:
                    self._hard_kill(w)
                    w.proc.waitForFinished(1000)
                w.state = "dead"
                w.timer.stop()

    def stats(self):
        return {"started": len(self.workers), "ready": self.ready(), "rows": [w.rows for w in self.workers],
                "peak_mb": [w.peak_mb or w.memory_mb for w in self.workers], "reasons": [w.reason for w in self.workers if w.reason]}


# ------------------------------------------------------------------------------------------------ the table
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


def when_text(iso):
    """'2026-10-30T17:12:04' -> '30 Oct 17:12'."""
    try:
        t = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return str(iso or "")
    return f"{t.day} {_MONTHS[t.month - 1]} {t:%H:%M}"


def _confirm_box(parent, title, text, detail=""):
    """The plain question the approval asks (replaceable: `OverviewPage.confirm`). True: approve."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText(detail)
    approve = box.addButton("Approve", QMessageBox.AcceptRole)
    box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(approve)
    box.exec()
    return box.clickedButton() is approve


def _warn_box(parent, title, text):
    QMessageBox.warning(parent, title, text)


class OverviewPage(QWidget):
    """The table. `row_fn(meta_path, smooth_s, window_s, source) -> (row dict or None, error text)` does the work (review.trial_row); `source` is
    'original' or 'reviewed' (the trial read through the edits made in Review). `parallel=True` also uses worker processes for a big
    participant (`row_fn` must then be a function a worker can import: overview_worker.spec_for); the default works in a thread only."""

    openRequested = Signal(str)          # the base name of the trial that was double-clicked / activated
    exportRequested = Signal(bool)       # True: valid trials only (the Export menu of 0.32.0 calls the exports itself; kept for code that listens)
    approvalChanged = Signal()           # the participant was approved, or the trials changed since the approval (the label says which)

    def __init__(self, row_fn, parent=None, parallel=False):
        super().__init__(parent)
        self.row_fn = row_fn
        self.parallel = bool(parallel)
        self.process_min_todo, self.max_processes, self.trials_per_process = PROCESS_MIN_TODO, MAX_PROCESSES, TRIALS_PER_PROCESS
        self.use_thread = True           # False: only the worker processes work (the thread steps in only if none can); for a quiet background run
        self.worker_python, self.worker_extra_path = None, []         # (tests: another interpreter, more places to import the row function from)
        self.ready_timeout_ms, self.job_timeout_ms = READY_TIMEOUT_MS, JOB_TIMEOUT_MS
        self.confirm, self.warn = _confirm_box, _warn_box             # (tests: replaced by functions that answer without a window)
        self.entries = []
        self.name = ""
        self.smooth_s, self.window_s, self.source = 0.1, 0.5, "original"
        self._cache = {}                 # base -> (stamp + settings, row or None, error); the current result of every trial worked out
        self._by_base = {}
        self._disk = {}                  # base -> [{'key', 'row'}, ...] newest first: the participant's overview cache as read / to be written
        self._disk_dirty = False
        self.cache_error = ""            # why the cache could not be saved, if so
        self.cache_hits = 0              # rows taken from the cache on disk (not worked out) since the participant was chosen
        self._generation = 0
        self._running = 0                # 1 while a run is in progress
        self._todo = None
        self._procs = None
        self._retiring = []              # pools whose workers are still ending
        self._thread_started = False
        self._thread_done = True
        self._cancel = False
        self._pending = 0                # rows asked for in the current run, and how many came back
        self._done = 0
        self._stamps = {}                # base -> the stamp the rows of the current run are made for
        self._again = False              # a run was asked for while a cancelled one was still finishing
        self._run_started = 0.0
        self._run_who = {}               # who worked out how many rows of this run
        self.last_run = {}               # what the latest run did: mode, seconds, workers, rows by whom (for tests and the log)
        self._shown = []                 # bases of the lines in the table
        self._signatures = {}            # base -> what its line shows
        self._verdict_tip = {}           # base -> the tooltip of its verdict cell (read once, not on every redraw)
        self._fingerprints = {}          # base -> (file stamp, fingerprint of the trial's review files)
        self._approvals = []             # every approval record of this participant, oldest first
        self._approval = {}              # the latest of them, {} if none
        self._changed_since = set()      # bases of the trials that differ from what was approved
        self._user_widths = set()        # columns the user dragged: they are not fitted again
        self._fitting = False
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.signals = _Signals(self)
        self.signals.produced.connect(self._on_row)
        self.signals.finished.connect(self._on_finished)
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(_FLUSH_MS)
        self._flush_timer.timeout.connect(self.flush)

        # top bar: what is shown, what to show, and the three buttons at the right (Approve review, Export, More)
        self.summary = ElidedLabel("Choose a participant.")
        self.summary.setObjectName("trialTitle")
        self.filter = QComboBox()
        for key, text in FILTERS:
            self.filter.addItem(text, key)
        self.filter.currentIndexChanged.connect(lambda _: self.flush(full=True))
        self.approve_button = QPushButton("Approve review")
        self.approve_button.setObjectName("approveButton")
        self.approve_button.setFocusPolicy(Qt.NoFocus)
        self.approve_button.setEnabled(False)
        self.approve_button.setToolTip("Marks this participant's review as looked through, and keeps a note of which state of the review you "
                                       "accepted (the verdicts and edits of every trial). It changes no number and no recorded file, and the "
                                       "exports work with or without it. If a trial changes afterwards, this page says so.")
        self.approve_button.setStyleSheet(
            f"QPushButton {{ border: 1px solid {ACCENT}; color: {ACCENT}; border-radius: 6px; padding: 4px 12px; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {tint(ACCENT, 0.12)}; }}"
            "QPushButton:disabled { border-color: palette(mid); color: palette(mid); font-weight: normal; }")
        self.approve_button.clicked.connect(lambda: self.request_approval())
        self.export_button, self.more_button = QToolButton(), QToolButton()
        for b, text, tip in ((self.export_button, "Export", "Save this participant's numbers (all trials, or the valid ones) or the open trial's files."),
                             (self.more_button, "More", "Open the review folder, reference validation, or a single trial file.")):
            b.setText(text)
            b.setToolTip(tip)
            b.setPopupMode(QToolButton.InstantPopup)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setFocusPolicy(Qt.NoFocus)
            b.setVisible(False)                                # shown by set_menus
            b.setStyleSheet("QToolButton { border: 1px solid palette(mid); border-radius: 6px; padding: 4px 22px 4px 12px; }"
                            "QToolButton:hover { background: palette(midlight); }"
                            "QToolButton::menu-indicator { subcontrol-origin: padding; subcontrol-position: center right; right: 7px; }")
        self.approval_label = ElidedLabel("Not approved yet")
        self.approval_label.setObjectName("hint")
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self.summary, 1)
        top.addWidget(QLabel("Show:"))
        top.addWidget(self.filter)
        top.addWidget(self.approve_button)
        top.addWidget(self.export_button)
        top.addWidget(self.more_button)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("overviewTable")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        for c, name in enumerate(COLUMNS):
            self.table.horizontalHeaderItem(c).setToolTip(COLUMN_TIPS[name])
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
        # (not ResizeToContents: Qt then measures the whole column after every cell that is set, which froze the window for 6-7 s with 64
        # lines. The columns are fitted by `_fit_columns` once per batch, and the user can still drag them.)
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setStretchLastSection(True)
        hh.sectionResized.connect(self._section_resized)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open_row(r))
        self.table.itemActivated.connect(lambda it: self._open_row(it.row()))
        self.progress = QProgressBar()
        self.progress.setObjectName("overviewProgress")
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.hint = QLabel("Double-click a line (or press Enter) to open the trial. Click a heading to sort. The numbers are worked out once and "
                           "kept for next time; a changed verdict, edit, smoothing or sliding window refreshes them.")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(4)
        lay.addLayout(top)
        lay.addWidget(self.approval_label)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.progress)
        lay.addWidget(self.hint)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)                 # no worker process outlives the program
        self.flush(full=True)

    def set_menus(self, export_menu, more_menu):
        """Show the Export and More menus (made by the Review tab, which keeps them) as two drop-down buttons at the top right."""
        for button, menu in ((self.export_button, export_menu), (self.more_button, more_menu)):
            button.setMenu(menu)
            button.setVisible(menu is not None)

    # ----------------------------------------------------------------------------------------- what is shown
    def set_participant(self, name, entries):
        """Show these trials (a review_nav list). Nothing is worked out until `ensure_rows`; the rows kept from last time are read at once."""
        self.cancel()
        self._save_disk_cache()                                    # (the participant being left)
        self.name, self.entries = name, list(entries)
        self._by_base = {e.base: e for e in self.entries}
        self._verdict_tip.clear()
        self._cache = {b: h for b, h in self._cache.items() if h[1] is not None}       # a trial that could not be read is tried again
        folder = self._folder()
        self._disk = {b: list(v) for b, v in (review_layer.read_overview_cache(folder).get("trials") or {}).items()} if folder else {}
        self._disk_dirty = False
        self.cache_hits = 0
        self._approvals = review_layer.read_approvals(folder) if folder else []
        self._approval = self._approvals[-1] if self._approvals else {}
        self._refresh_approval(emit=False)
        self.flush(full=True)

    def _folder(self):
        return self.entries[0].meta_path.parent if self.entries else None

    def set_settings(self, smooth_s, window_s, source=None):
        self.smooth_s, self.window_s = float(smooth_s), float(window_s)
        if source in ("original", "reviewed"):
            self.source = source

    def _stamp(self, entry, settings=None):
        return file_stamp(entry.meta_path) + (settings or settings_stamp(self.smooth_s, self.window_s, self.source))

    def _hit(self, entry, settings=None):
        """(stamp, row, error) of a trial if it is current (its files and the settings are as they were when it was worked out): in memory, else
        from the cache on disk (which then is the one in memory). None when it must be worked out."""
        stamp = self._stamp(entry, settings)
        hit = self._cache.get(entry.base)
        if hit and hit[0] == stamp:
            return hit
        for v in self._disk.get(entry.base, ()):
            if isinstance(v, dict) and isinstance(v.get("row"), dict) and tuple(v.get("key") or ()) == stamp:
                self._cache[entry.base] = hit = (stamp, v["row"], "")
                self.cache_hits += 1
                self._flush_soon()
                return hit
        return None

    def row_for(self, base, allow_stale=False):
        """The cached row of a trial if it is current, else None. `allow_stale`: also an out-of-date one (the table shows the old numbers until
        the new ones are in; an export never does)."""
        if allow_stale:
            hit = self._cache.get(base)
            return hit[1] if hit else None
        e = self._by_base.get(base)
        hit = self._hit(e) if e is not None else None
        return hit[1] if hit else None

    def stale_entries(self):
        """The readable trials without a current result (neither in memory nor in the cache on disk)."""
        settings = settings_stamp(self.smooth_s, self.window_s, self.source)
        return [e for e in self.entries if e.readable and self._hit(e, settings) is None]

    # ----------------------------------------------------------------------------------------- working the rows out
    def _process_count(self, n):
        """How many worker processes a run of `n` trials gets (0: the thread alone, and why not is in `last_run`)."""
        if not self.parallel:
            return 0, "processes not asked for"
        if not overview_worker.processes_allowed():
            return 0, "processes are switched off"
        if n < self.process_min_todo:
            return 0, f"only {n} trials: a thread is quicker than starting processes"
        if overview_worker.spec_for(self.row_fn) is None:
            return 0, "the row function cannot be imported by a worker"
        cpus = os.cpu_count() or 2
        return max(0, min(overview_worker.configured_workers(cpus - 1), self.max_processes, math.ceil(n / self.trials_per_process))), ""

    def ensure_rows(self):
        """Start working out the rows that are missing or out of date, in the background."""
        self._refresh_approval()
        todo = self.stale_entries()
        if not todo:
            self.flush()
            return
        if self._running:                # a cancelled run is still finishing: start again when it has
            self._again = True
            return
        self._cancel = False
        self._generation += 1
        self._pending, self._done = len(todo), 0
        self._running = 1
        self._run_started = time.perf_counter()
        self._run_who = {}
        self.progress.setRange(0, self._pending)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        settings = settings_stamp(self.smooth_s, self.window_s, self.source)
        self._stamps = {e.base: self._stamp(e, settings) for e in todo}
        self._todo = _Todo([(e.base, str(e.meta_path)) for e in todo])
        count, why = self._process_count(len(todo))
        self.last_run = {"trials": len(todo), "mode": "thread", "why_not_processes": why, "processes": 0}
        self._procs = None
        if count:
            spec = overview_worker.spec_for(self.row_fn)
            pool = _ProcessPool(spec, self._todo, (self.smooth_s, self.window_s, self.source), count, python=self.worker_python,
                                extra_path=self.worker_extra_path, ready_ms=self.ready_timeout_ms, job_ms=self.job_timeout_ms, parent=self)
            gen = self._generation
            pool.rowReady.connect(lambda base, row, err, who, gen=gen: self._on_row(gen, base, row, err, who))
            pool.changed.connect(self._try_finish)
            self._procs = pool
            self.last_run.update(mode="thread and processes", processes=count, why_not_processes="")
        self._thread_started, self._thread_done = False, True
        if self._procs is None or self.use_thread:
            self._start_thread()
        if self._procs is not None:
            self._procs.start()
        self.flush()

    def _start_thread(self):
        if not self._thread_started:
            self._thread_started, self._thread_done = True, False
            self.pool.start(_Job(self._generation, self._todo, self.row_fn, self.smooth_s, self.window_s, self.source, self.signals))

    def cancel(self):
        """Stop a run in progress (the rows already done are kept)."""
        self._cancel = True
        if self._todo is not None:
            self._todo.cancel()

    def shutdown(self):
        """Stop everything and wait for it (a trial being read is finished first): the program is closing, or a test is done."""
        self.cancel()
        self.pool.waitForDone(60000)
        for pool in ([self._procs] if self._procs is not None else []) + list(self._retiring):
            pool.stop(wait_ms=STOP_GRACE_MS)
        self._retiring.clear()
        self._save_disk_cache()

    def wait_until_done(self, timeout_ms=60000):
        """Let the window's own thread deliver what the background workers have produced, until they are finished (used by tests and by exports)."""
        end = time.monotonic() + timeout_ms / 1000.0
        while self._running and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()
        return not self._running

    def _on_row(self, generation, base, row, err, who="thread"):
        if generation != self._generation:
            return
        stamp = self._stamps.get(base)
        if stamp is not None and base in self._by_base:
            row = plain_row(row)
            self._cache[base] = (stamp, row, err)
            if row is not None and not err:
                self._remember(base, stamp, row)
        self._done += 1
        self._run_who[who] = self._run_who.get(who, 0) + 1
        self.progress.setValue(self._done)
        self._flush_soon()

    def _on_finished(self, generation):
        """The thread has no more trials to take."""
        if generation != self._generation:
            return
        self._thread_done = True
        self._try_finish()

    def _try_finish(self):
        for pool in list(self._retiring):
            if not pool.alive():
                self._retiring.remove(pool)
                pool.deleteLater()
        if not self._running or not self._thread_done:
            return
        left = [] if self._todo is None or self._todo.cancelled else self._todo.remaining()
        if left and self._procs is not None and self._procs.usable():
            return                                                 # the workers are starting or reading: they take the rest
        if left and not self._thread_started:                      # no worker is left (none could start, all died): the thread does it
            self.last_run["fell_back_to_thread"] = True
            self._start_thread()
            return
        if self._procs is not None and self._procs.busy():
            return
        # the todo is empty (or the run was cancelled) and nobody is reading a trial
        for base, _path in left:                                   # (only if every worker died and the thread with them)
            self._cache[base] = (self._stamps.get(base), None, "Could not be read: the numbers were not worked out")
        pool, self._procs = self._procs, None
        if pool is not None:
            self.last_run.update(pool.stats())
            pool.stop()
            self._retiring.append(pool)
        self._running = 0
        self.last_run.update(seconds=round(time.perf_counter() - self._run_started, 2), by=dict(self._run_who), cancelled=self._cancel,
                             cache_hits=self.cache_hits)
        self.progress.setVisible(False)
        self._save_disk_cache()
        self.flush()
        self._refresh_approval()
        if self._again:
            self._again = False
            self.ensure_rows()

    def store(self, entry, row, err=""):
        """Keep a row worked out elsewhere (an export that did the missing ones itself)."""
        stamp = self._stamp(entry)
        row = plain_row(row)
        self._cache[entry.base] = (stamp, row, err)
        if row is not None and not err:
            self._remember(entry.base, stamp, row)
        self._flush_soon()

    def update_verdict(self, base):
        """A verdict was saved: show it at once. The line's checks name the verdict, so its numbers count as out of date and are worked out
        again the next time the overview is shown (`ensure_rows`); until then the old ones stay."""
        entry = self._by_base.get(base)
        if entry is None:
            return
        entry.refresh_verdict()
        self._verdict_tip.pop(base, None)
        hit = self._cache.get(base)
        if hit and hit[1] is not None:
            v = verdicts.read_verdict(entry.meta_path)
            row = dict(hit[1])
            row["verdict"] = (v or {}).get("verdict") or ""
            row["verdict_reasons"] = ", ".join(verdicts.REASONS.get(r, r) for r in (v or {}).get("reasons") or [])
            row["verdict_note"] = (v or {}).get("note") or ""
            self._cache[base] = (hit[0], row, hit[2])
        self._refresh_approval()
        self.flush()

    # ----------------------------------------------------------------------------------------- the rows kept for next time
    def _remember(self, base, stamp, row):
        keep = [v for v in self._disk.get(base, []) if tuple(v.get("key") or ()) != tuple(stamp)]
        self._disk[base] = [{"key": list(stamp), "row": row}] + keep[: review_layer.OVERVIEW_CACHE_VARIANTS - 1]
        self._disk_dirty = True

    def _save_disk_cache(self):
        """Write the finished rows into the participant's review folder (whole or not at all). A failure is kept in `cache_error`, not hidden."""
        if not self._disk_dirty or not self.entries:
            return
        self._disk_dirty = False
        try:
            review_layer.write_overview_cache(self._folder(), {b: v for b, v in self._disk.items() if b in self._by_base})
            self.cache_error = ""
        except (OSError, ValueError, TypeError) as e:
            self.cache_error = f"{type(e).__name__}: {e}"
        self._update_hint()

    def _update_hint(self):
        base = ("Double-click a line (or press Enter) to open the trial. Click a heading to sort. The numbers are worked out once and "
                "kept for next time; a changed verdict, edit, smoothing or sliding window refreshes them.")
        self.hint.setText(base + (f"\nThe numbers could not be kept for next time: {self.cache_error}" if self.cache_error else ""))

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
        if f == "edited":
            return entry.edited
        if f == "changed":
            return entry.base in self._changed_since
        return True

    def _flush_soon(self):
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def render(self):
        """Draw the whole table again."""
        self.flush(full=True)

    def flush(self, full=False):
        """Bring the table up to date with the entries and the rows worked out so far. Only the lines that differ from what they show are
        written (a row arriving touches its line, and the 10 % check of the lines of its position); all lines are drawn again only when
        the set of lines changes (a filter, a participant) or `full`. The columns are fitted once at the end."""
        self._flush_timer.stop()
        rows = {e.base: self.row_for(e.base, allow_stale=True) for e in self.entries}
        copies = {b: dict(r) for b, r in rows.items() if r is not None}
        analysis.repeatability(list(copies.values()))              # (the 10 % check is worked out over what is there)
        shown = [e for e in self.entries if self._visible(e, rows.get(e.base))]
        cells = {e.base: self._cells(e, copies.get(e.base)) for e in shown}
        hh = self.table.horizontalHeader()
        sort_col, sort_order = hh.sortIndicatorSection(), hh.sortIndicatorOrder()
        selected = self._selected_base()
        wrote = False
        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)
        try:
            if full or {e.base for e in shown} != set(self._shown) or self.table.rowCount() != len(shown):
                self.table.setRowCount(len(shown))
                self._signatures = {}
                where = {e.base: r for r, e in enumerate(shown)}
                wrote = True
            else:
                where = {}
                for r in range(self.table.rowCount()):
                    it = self.table.item(r, 0)
                    if it is not None:
                        where[it.data(_BASE_ROLE)] = r
            for e in shown:
                sig = (tuple(cells[e.base]), e.readable, e.verdict == verdicts.COMPROMISED)
                if self._signatures.get(e.base) != sig and e.base in where:
                    self._write_row(where[e.base], e, cells[e.base])
                    self._signatures[e.base] = sig
                    wrote = True
            self._shown = [e.base for e in shown]
        finally:
            self.table.setSortingEnabled(True)
            if sort_col >= 0:
                self.table.sortItems(sort_col, sort_order)
            if wrote:
                self._fit_columns()
            self.table.setUpdatesEnabled(True)
        if selected:
            self.select_base(selected)
        self._update_summary()

    def _write_row(self, r, e, cells):
        for c, (text, key, tip, colour, bold) in enumerate(cells):
            it = self.table.item(r, c)
            if it is None:
                it = _SortItem()
                self.table.setItem(r, c, it)
            it.setText(text)
            it.setData(Qt.UserRole, key)
            it.setData(_BASE_ROLE, e.base)
            it.setToolTip(tip or "")
            it.setForeground(QBrush(QColor(colour)) if colour else QBrush())
            if bold or not e.readable:
                f = it.font()
                f.setBold(bold)
                f.setItalic(not e.readable)
                it.setFont(f)
            else:
                it.setData(Qt.FontRole, None)
            it.setBackground(QBrush(QColor(220, 38, 38, 26)) if e.verdict == verdicts.COMPROMISED else QBrush())
            it.setTextAlignment(Qt.AlignVCenter | (Qt.AlignRight if COLUMNS[c] in RIGHT_ALIGNED else Qt.AlignLeft))

    def _fit_columns(self):
        """Fit each column to its widest cell, except those the user dragged."""
        self._fitting = True
        try:
            for c in range(len(COLUMNS)):
                if c not in self._user_widths:
                    self.table.resizeColumnToContents(c)
        finally:
            self._fitting = False

    def _section_resized(self, index, _old, _new):
        if not self._fitting and QApplication.mouseButtons() & Qt.LeftButton:       # dragged by the user, not by the last column stretching
            self._user_widths.add(index)

    def _update_summary(self):
        counts = nav.counts(self.entries)
        text = f"{self.name}: {len(self.entries)} trials · {counts['valid']} valid · {counts['compromised']} compromised · {counts['not judged']} not judged"
        if counts["unreadable"]:
            text += f" · {counts['unreadable']} cannot be read"
        if self._running:
            text += f"   (working out the numbers: {self._done} of {self._pending})"
        self.summary.setText(text if self.entries else "Choose a participant.")

    def _verdict_cell(self, e):
        tip = self._verdict_tip.get(e.base)
        if tip is None:
            v = verdicts.read_verdict(e.meta_path)
            tip = self._verdict_tip[e.base] = verdicts.describe(v) if v else "not judged yet"
        return tip

    def _cells(self, e, row):
        """[(text, sort key, tooltip, colour, bold)] of one line, in the order of COLUMNS."""
        c = {name: BLANK for name in COLUMNS}
        if not e.readable:
            c["Verdict"] = ("cannot be read", -1, e.error, WARN, False)
            c["Test"] = (e.base, e.base.lower(), e.error, None, False)
            return [c[n] for n in COLUMNS]
        colour = {verdicts.VALID: GOOD, verdicts.COMPROMISED: BAD}.get(e.verdict)
        order = {verdicts.COMPROMISED: 0, None: 1, verdicts.VALID: 2}[e.verdict]
        pos = e.position_number if e.position_number not in (None, "") else None
        name = nav._short_test(e.test_label) or e.test_key
        target = _num(e.target_deg)
        c["Verdict"] = (f"{nav.GLYPHS[e.verdict]} {nav.WORDS[e.verdict]}", order, self._verdict_cell(e), colour, e.verdict is not None)
        c["Pos"] = (f"{int(pos):02d}" if pos is not None else "", int(pos) if pos is not None else None, "", None, False)
        c["Trial"] = (f"T{e.trial_number}" if e.trial_number not in (None, "") else "", _num(e.trial_number), "", None, False)
        c["Test"] = (name + (" (unfinished)" if e.unfinished else ""), name.lower(), e.base, None, False)
        c["Target"] = (f"{target:g}°" if target is not None else "", target, "", None, False)
        c["Time"] = (e.clock, e.started, e.started.replace("T", " "), None, False)
        c["Edits"] = ("✎" if e.edited else "", 1 if e.edited else 0, (row or {}).get("review_edits") or ("edited in Review" if e.edited else ""),
                      ACCENT if e.edited else None, e.edited)
        if row is None:                                   # not worked out yet ('…'), or it failed (the reason is on the line)
            err = self._cache[e.base][2] if e.base in self._cache else ""
            c["Peak"] = ("could not be read", None, err, WARN, False) if err else ("…", None, "", None, False)
            return [c[n] for n in COLUMNS]
        unit = row.get("force_unit") or ""
        sfx = f" {unit}" if unit else ""
        peak, wpeak = _num(row.get("force_peak")), _num(row.get("force_window_peak"))
        c["Peak"] = (_fmt(peak, unit) + (sfx if peak is not None else ""), peak, "", None, False) if unit else ("no force", None, "", None, False)
        c["Window peak"] = (_fmt(wpeak, unit) + (sfx if wpeak is not None else ""), wpeak,
                            f"highest mean over {row.get('force_window_peak_ms'):.0f} ms" if _num(row.get("force_window_peak_ms")) else "", None, False)
        c["Time to peak (s)"] = (_fmt(row.get("force_time_to_peak_s"), unit, 2), _num(row.get("force_time_to_peak_s")),
                                 "from the start of the push to the peak", None, False)
        dots_peak, ref_peak = _num(row.get("angle_at_peak_force_deg")), _num(row.get("angle_at_peak_force_from_reference_deg"))
        shown_peak = ref_peak if ref_peak is not None else dots_peak                # from the reference when the trial has one (0.33.0), the dots' angle in the tip
        c["Angle at peak (deg)"] = (_fmt(shown_peak, unit, 1), shown_peak,
                                    str(row.get("main_angle") or "") + (f"\nfrom the reference; the dots read {dots_peak:.1f}" if ref_peak is not None and dots_peak is not None else ""),
                                    None, False)
        others = []                                                   # (name, shown number, the dots' number or None when it is the same)
        for n in (2, 3):
            if row.get(f"angle{n}_name"):
                dots, ref = _num(row.get(f"angle{n}_at_peak_force_deg")), _num(row.get(f"angle{n}_at_peak_from_reference_deg"))
                others.append((str(row[f"angle{n}_name"]), ref if ref is not None else dots, dots if ref is not None else None))
        if others:                                                    # "Hip flex 41.7°"; two others (the ankle tests) side by side, the full names in the tip
            shown = [(name.split(" (")[0], v) for name, v, _d in others]
            c["Other angle at peak"] = (" · ".join(f"{n} {v:.1f}°" if v is not None else f"{n} -" for n, v in shown), others[0][1],
                                        "\n".join((f"{name}: {v:.1f}°" if v is not None else f"{name}: not known")
                                                  + (f" (from the reference; the dots read {d:.1f}°)" if d is not None and v is not None else "") for name, v, d in others),
                                        None, False)
        checks = [x for x in (row.get("checks_to_look_at") or "").split(" | ") if x]
        c["Checks"] = (str(len(checks)), len(checks), "\n".join(checks), WARN if checks else None, bool(checks))
        pct, third = _num(row.get("repeatability_pct_same_position")), row.get("third_trial_needed")
        if pct is not None:
            c["10 % check"] = (f"{pct:.1f} %" + (" · 3rd trial needed" if third is True else ""), pct,
                               "How far the peaks of the trials of this position are apart, as a share of their mean (SOP 10.6)",
                               BAD if third is True else None, third is True)
        return [c[n] for n in COLUMNS]

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

    def request_export(self, valid_only=False):
        self.exportRequested.emit(bool(valid_only))

    def showEvent(self, e):
        super().showEvent(e)
        self._refresh_approval()

    def closeEvent(self, e):
        self.shutdown()
        super().closeEvent(e)

    # ----------------------------------------------------------------------------------------- approval
    def _fingerprint(self, e):
        stamp = file_stamp(e.meta_path)
        hit = self._fingerprints.get(e.base)
        if hit is None or hit[0] != stamp:
            hit = self._fingerprints[e.base] = (stamp, review_layer.trial_fingerprint(e.meta_path))
        return hit[1]

    def _counts_now(self):
        c = nav.counts(self.entries)
        return {"trials": len(self.entries), "valid": c["valid"], "compromised": c["compromised"], "not_judged": c["not judged"],
                "unreadable": c["unreadable"], "edited": sum(1 for e in self.entries if e.edited)}

    def approval_state(self):
        """Where the approval of this participant stands, as a dict: `approved` (ever), `approved_at`, `label` (the text of the label), `changed`
        (bases of the trials that are new or whose review differs from what was approved), `removed` (approved trials that are gone), `counts`
        (as approved), `now` (the counts now), `approvals` (how many approvals the file holds)."""
        now = self._counts_now()
        rec = self._approval
        if not rec:
            return {"approved": False, "approved_at": "", "label": "Not approved yet", "changed": [], "removed": [], "counts": {}, "now": now,
                    "approvals": 0}
        old = {t.get("base"): t for t in rec.get("trials") or [] if isinstance(t, dict)}
        changed = [e.base for e in self.entries if e.base not in old or old[e.base].get("fingerprint") != self._fingerprint(e)]
        removed = [b for b in old if b not in self._by_base]
        n = len(changed) + len(removed)
        counts = rec.get("counts") or {}
        if n:
            label = f"Approved {when_text(rec.get('approved_at'))} - {n} trial{'' if n == 1 else 's'} changed since"
        else:
            label = f"Approved {when_text(rec.get('approved_at'))} - {counts.get('valid', 0)} valid, {counts.get('compromised', 0)} compromised"
            if counts.get("not_judged"):
                label += f", {counts['not_judged']} not judged"
        return {"approved": True, "approved_at": rec.get("approved_at", ""), "label": label, "changed": changed, "removed": removed,
                "counts": counts, "now": now, "approvals": len(self._approvals)}

    def refresh_approval(self):
        """Look at the trials again (their review files may have changed) and update the label, the button and the 'Changed since approved' filter."""
        self._refresh_approval()

    def _refresh_approval(self, emit=True):
        state = self.approval_state()
        before = (self.approval_label.text(), self._changed_since)
        self._changed_since = set(state["changed"])
        self.approval_label.setText(state["label"])
        colour = (WARN if state["changed"] or state["removed"] else GOOD) if state["approved"] else None
        self.approval_label.setStyleSheet(f"color: {colour};" if colour else "")
        self.approve_button.setEnabled(bool(self.entries))
        if emit and (self.approval_label.text(), self._changed_since) != before:
            if self.filter.currentData() == "changed":
                self.flush()
            self.approvalChanged.emit()
        return state

    def approval_summary(self):
        """What the question before approving says, as a dict: `title`, `text`, `detail` and the counts it is made from."""
        c = self._counts_now()
        judged = c["valid"] + c["compromised"]
        lines = [f"{judged} of {c['trials']} trials are judged ({c['valid']} valid, {c['compromised']} compromised)."]
        if c["not_judged"]:
            n = c["not_judged"]
            lines.append(f"{n} trial{' is' if n == 1 else 's are'} NOT judged yet. You can approve anyway; {'it stays' if n == 1 else 'they stay'} marked not judged.")
        if c["unreadable"]:
            n = c["unreadable"]
            lines.append(f"{n} trial{' cannot' if n == 1 else 's cannot'} be read.")
        if c["edited"]:
            n = c["edited"]
            lines.append(f"{n} trial{' was' if n == 1 else 's were'} edited in Review (a force zero or dot fixes).")
        detail = ("Approving keeps a note of this state of the review. It changes no number and no recorded file, and you can still export. "
                  "If a trial changes afterwards, this page says so.")
        return {"title": f"Approve the review of {self.name}?", "text": "\n".join(lines), "detail": detail, **c, "judged": judged}

    def request_approval(self):
        """The Approve review button: asks first (`self.confirm`), then approves. Returns the record, or None if it was not approved."""
        if not self.entries:
            return None
        s = self.approval_summary()
        if not self.confirm(self, s["title"], s["text"], s["detail"]):
            return None
        try:
            return self.commit_approval()
        except OSError as e:
            self.warn(self, "Not approved", f"The approval could not be saved in the review folder:\n{e}")
            return None

    def commit_approval(self):
        """Approve now, without asking: writes review/<participant>/approval.json (the earlier approvals stay in it) and returns the record. Raises
        OSError when it cannot be written."""
        if not self.entries:
            raise ValueError("there are no trials to approve")
        record = {"approved_at": datetime.now().isoformat(timespec="seconds"), "software_version": SOFTWARE_VERSION,
                  "participant": (self.entries[0].participant or self.name),
                  "counts": self._counts_now(),
                  "trials": [{"base": e.base, "verdict": e.verdict if e.readable else None, "edited": bool(e.edited),
                              "fingerprint": self._fingerprint(e)} for e in self.entries]}
        saved = review_layer.write_approval(self._folder(), record)
        self._approvals = review_layer.read_approvals(self._folder())
        self._approval = saved
        self._refresh_approval(emit=False)
        self.approvalChanged.emit()
        return saved
