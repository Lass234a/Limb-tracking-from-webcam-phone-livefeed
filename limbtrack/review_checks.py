"""The checks of a trial as a panel that is always in view (0.32.0), and (round 2 of 0.32.0) as a chip in the bottom bar with the panel in a pop-up.

Until 0.31.0 the checks were the first of seven tabs: a coloured text list that was hidden as soon as another tab was open, so the operator had
to click back to see whether the trial was fine. Now a short summary sits above the graphs: how many things are fine, how many to look at, how
many are only notes; the things to look at each get a line with a plain headline, the sentence in muted text and, where Review can do something
about it, a small button. The fine ones and the notes are one click away behind their number.

Red is not used for a check (red is for a compromised verdict): a check is amber.
"""
import time

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .review_panels import strong_colour
from .review_style import ACCENT, GOOD, WARN, ElidedLabel, border_colour, muted_colour, tint

INFO_GREY = "#6b7280"
MAX_HEIGHT = 280              # about 40 % of a 700 px column: taller than this the lines scroll inside the panel
COMPACT_BELOW = 520           # a panel narrower than this puts the sentence of a line under its headline
QWIDGETSIZE_MAX = 16777215    # (Qt's "no maximum")
PILL_HEIGHT = 20
MARGIN_V, SPACING, SIDE_MARGIN = 5, 3, 8
ACTION_LABELS = {"fix_dots": "Show dot", "zero": "Set zero", "window": "Set window", "calibration": "Calibration", "privacy": "Privacy"}
ACTION_TIPS = {"fix_dots": "Open the Fix dots workspace", "zero": "Go to the force zero", "window": "Go to the analysis window",
               "calibration": "Show the calibration graph", "privacy": "Draw a crop and masks and make protected copies of the videos"}
KIND_COLOURS = {"check": WARN, "ok": GOOD, "info": INFO_GREY}
CHECKS_KEY = "C"              # the key that opens and closes the list of checks (the Review tab binds it; the chip says so in its tooltip)
MAX_KEYED_BUTTONS = 9         # inside the open list the keys 1 to 9 press the buttons of its lines


class _Marker(QWidget):
    """A small round marker: amber with '!', green with a tick, grey with 'i'. Drawn, so it looks the same on every computer."""

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(16, 16)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(KIND_COLOURS[self.kind]))
        p.drawEllipse(QRectF(1, 1, 14, 14))
        p.setPen(QPen(QColor("white"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if self.kind == "ok":
            tick = QPainterPath()
            tick.moveTo(4.6, 8.4)
            tick.lineTo(7.0, 10.8)
            tick.lineTo(11.4, 5.6)
            p.drawPath(tick)
        elif self.kind == "check":
            p.drawLine(QPointF(8, 4.2), QPointF(8, 8.8))
            p.drawPoint(QPointF(8, 11.6))
        else:
            p.drawPoint(QPointF(8, 4.6))
            p.drawLine(QPointF(8, 7.4), QPointF(8, 11.8))


class _Headline(ElidedLabel):
    """The bold headline of a line: as wide as its text when there is room, cut with '...' when there is not."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        font = self.font()
        font.setBold(True)
        self.setFont(font)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumWidth(56)
        self._fit()

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())


def action_label(item):
    """The plain label of the button a check has ('' when it has none): 'Show dot', 'Set zero', 'Set window' or 'Calibration'."""
    action = item.get("action")
    if action not in ACTION_LABELS:
        return ""
    return "Show dots" if action == "fix_dots" and not item.get("arg") else ACTION_LABELS[action]


class _Row(QFrame):
    """One check: marker, headline, the sentence on one line (the whole sentence when the line is clicked), and a button if there is something
    to do about it. In a narrow panel (`set_compact(True)`) the sentence goes on a second line under the headline, where it has the room.
    `item` is the dict from analysis.trial_check_items, `kind` its level."""

    changed = Signal()              # the line became taller or shorter (the panel fits itself again)
    actionClicked = Signal(str, str)

    def __init__(self, item, expanded=False, parent=None):
        super().__init__(parent)
        self.item = item
        self.kind = item["level"]
        self.compact = False
        self.setObjectName("checkRow")
        self.setProperty("kind", self.kind)
        self.setToolTip(item["text"])
        self.setCursor(Qt.PointingHandCursor)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(6, 1, 4, 1)
        self.grid.setHorizontalSpacing(7)
        self.grid.setVerticalSpacing(1)
        self.grid.setColumnStretch(2, 1)
        self.marker = _Marker(self.kind)
        self.title_label = _Headline(item["title"])
        self.title_label.setToolTip(item["text"])
        self.sentence = ElidedLabel(item["text"])
        self.sentence.setObjectName("checkSentence")
        self.button = None
        action = item.get("action")
        if action in ACTION_LABELS:
            self.button = QPushButton(action_label(item))
            self.button.setObjectName("checkAction")
            self.button.setFocusPolicy(Qt.NoFocus)             # (the keys of Review, such as the next picture, stay with the video)
            self.button.setToolTip(ACTION_TIPS[action] + (f": {item['arg']}" if action == "fix_dots" and item.get("arg") else ""))
            self.button.clicked.connect(lambda: self.actionClicked.emit(action, item.get("arg") or ""))
        self.detail = QLabel(item["text"])
        self.detail.setObjectName("checkDetail")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.NoTextInteraction)
        self.detail.setContentsMargins(0, 0, 4, 2)
        self.expanded = False
        self._arrange()
        self.set_expanded(expanded, announce=False)

    def _arrange(self):
        """Put the parts in their cells: all on one line, or (compact) the sentence on a second line."""
        for w in (self.marker, self.title_label, self.sentence, self.detail) + ((self.button,) if self.button else ()):
            self.grid.removeWidget(w)
        if self.compact:
            self.grid.addWidget(self.marker, 0, 0, Qt.AlignTop)
            self.grid.addWidget(self.title_label, 0, 1, 1, 2)
            self.grid.addWidget(self.sentence, 1, 1, 1, 3)
            self.grid.addWidget(self.detail, 2, 1, 1, 3)
        else:
            self.grid.addWidget(self.marker, 0, 0, Qt.AlignVCenter)
            self.grid.addWidget(self.title_label, 0, 1)
            self.grid.addWidget(self.sentence, 0, 2)
            self.grid.addWidget(self.detail, 1, 1, 1, 3)
        if self.button:
            self.grid.addWidget(self.button, 0, 3, Qt.AlignVCenter)
        self.sentence.setMinimumWidth(0 if self.compact else 40)

    def set_compact(self, on):
        if bool(on) != self.compact:
            self.compact = bool(on)
            self._arrange()
            self.set_expanded(self.expanded, announce=False)

    def set_expanded(self, on, announce=True):
        self.expanded = bool(on)
        self.detail.setVisible(self.expanded)
        self.sentence.setVisible(not self.expanded)                  # (the whole sentence takes the place of the one line)
        self.updateGeometry()                                        # (so the panel's lines know at once how tall this one is now)
        if announce:
            self.changed.emit()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.set_expanded(not self.expanded)
        super().mouseReleaseEvent(e)


class _Pill(QPushButton):
    """A summary count such as '4 to look at' that can be clicked for the lines behind it."""

    def __init__(self, name, tip, parent=None):
        super().__init__(parent)
        self.setObjectName(name)
        self.setCheckable(True)
        self.setToolTip(tip)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(PILL_HEIGHT)


class _RowsScroll(QScrollArea):
    """The lines of the panel: no frame, no side scroll; it scrolls up and down only when the lines are taller than the panel may be."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)      # (never widens the window)
        self.setMinimumWidth(0)
        self.viewport().setAutoFillBackground(False)

    def minimumSizeHint(self):
        return QSize(0, self.height())


class ChecksPanel(QFrame):
    """The checks of the open trial, always in view and compact.

    `set_items(items)` takes the list of dicts of `analysis.trial_check_items` (level, text, title, area, action, arg). Above: a summary of
    pills, a green '5 fine', an amber '4 to look at' and a grey '2 notes'. Under it a line for every item that needs a look (level 'check');
    the fine ones and the notes are shown when their pill is clicked. A line with an action has a button; pressing it emits
    `actionRequested(action, arg)` with the action ('fix_dots', 'zero', 'window' or 'calibration') and the argument ('' when there is none; for
    'fix_dots' the dot name). Clicking a line shows its whole sentence. With nothing to look at: one calm green line. With no items at all (no
    trial open): a quiet grey line. `counts()` gives {'ok', 'check', 'info'}; `rows` are the visible lines that need a look."""

    actionRequested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("checksPanel")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.setMaximumHeight(MAX_HEIGHT)
        self.setMinimumWidth(0)
        self._items, self._built = [], False
        self._list = None                     # which of the quiet lists is open: None, 'ok' or 'info'
        self._expand_all = False
        self._expanded_keys = set()
        self._compact = False                 # narrow: the sentence of a line goes under its headline
        self.rows, self.extra_rows = [], []
        self._css = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(SIDE_MARGIN, MARGIN_V, SIDE_MARGIN, MARGIN_V)
        lay.setSpacing(SPACING)
        self.summary = QHBoxLayout()
        self.summary.setContentsMargins(0, 0, 0, 0)
        self.summary.setSpacing(6)
        self.pill_check = _Pill("pillCheck", "Things to look at. Click to read all the sentences in full.")
        self.pill_ok = _Pill("pillOk", "Checks that are fine. Click to see them.")
        self.pill_info = _Pill("pillInfo", "Notes that need no action. Click to see them.")
        for pill in (self.pill_check, self.pill_ok, self.pill_info):
            self.summary.addWidget(pill)
        self.hint = ElidedLabel("")
        self.hint.setObjectName("checkHint")
        self.hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.summary.addWidget(self.hint, 1)
        lay.addLayout(self.summary)
        self.scroll = _RowsScroll()
        self.host = QWidget()
        self.host.setObjectName("checksHost")
        self.host_lay = QVBoxLayout(self.host)
        self.host_lay.setContentsMargins(0, 0, 0, 0)
        self.host_lay.setSpacing(1)
        self.scroll.setWidget(self.host)
        lay.addWidget(self.scroll)
        lay.addStretch(1)                       # (room the panel is given beyond its own height goes under the lines, not between the parts)
        self.pill_check.clicked.connect(self._toggle_expand_all)
        self.pill_ok.clicked.connect(lambda: self._open_list("ok"))
        self.pill_info.clicked.connect(lambda: self._open_list("info"))
        self._apply_style()
        self._rebuild()

    # ------------------------------------------------------------------ what the lead calls
    def set_items(self, items):
        """Show these checks (the dicts of analysis.trial_check_items). The same list again changes nothing (so it can be called on every
        move of the analysis window without the lines flickering or closing)."""
        items = [dict(i) for i in (items or [])]
        if self._built and items == self._items:
            return
        self._items = items
        self._rebuild()

    def counts(self):
        out = {"ok": 0, "check": 0, "info": 0}
        for i in self._items:
            out[i["level"]] = out.get(i["level"], 0) + 1
        return out

    # ------------------------------------------------------------------ building
    def _rebuild(self):
        self._built = True
        c = self.counts()
        for pill, n, one, many, level in ((self.pill_check, c["check"], "to look at", "to look at", "check"),
                                           (self.pill_ok, c["ok"], "fine", "fine", "ok"),
                                           (self.pill_info, c["info"], "note", "notes", "info")):
            pill.setVisible(n > 0)
            pill.setText(f"{n} {one if n == 1 else many}")
        if self._list and not c[self._list]:
            self._list = None
        if not c["check"]:
            self._expand_all = False
        self._expanded_keys &= {(i["title"], i["text"]) for i in self._items}
        self._fill_rows()

    def _fill_rows(self):
        """The lines under the summary: made again when the items or the open list change."""
        while self.host_lay.count():
            w = self.host_lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.rows, self.extra_rows = [], []
        c = self.counts()
        self.pill_check.setChecked(self._expand_all)
        self.pill_ok.setChecked(self._list == "ok")
        self.pill_info.setChecked(self._list == "info")
        if not self._items:
            self.hint.setText("")
            self._add(self._plain_line("No checks to show", muted=True))
        else:
            for item in (i for i in self._items if i["level"] == "check"):
                row = self._make_row(item, self._expand_all or (item["title"], item["text"]) in self._expanded_keys)
                self.rows.append(row)
                self._add(row)
            if not c["check"]:
                self._add(self._plain_line("Nothing to look at", good=True))
            if self._list:
                for item in (i for i in self._items if i["level"] == self._list):
                    row = self._make_row(item, False)
                    self.extra_rows.append(row)
                    self._add(row)
            self.hint.setText("Click a number or a line for more" if (self.rows or c["ok"] or c["info"]) else "")
        self.host_lay.addStretch(1)
        self._align()
        self._fit()

    def _align(self):
        """The same width for every headline and every button, so the sentences start at one place; a headline is at most 42 % of the line.
        (In a narrow panel the headline has the line to itself, over the sentence: only the buttons are made alike.)"""
        rows = self.rows + self.extra_rows
        if not rows:
            return
        title_w = min(max(r.title_label.sizeHint().width() for r in rows), int(self._inner_width() * 0.42))
        button_w = max((r.button.sizeHint().width() for r in rows if r.button), default=0)
        for r in rows:
            if self._compact:
                r.title_label.setMinimumWidth(56)
                r.title_label.setMaximumWidth(QWIDGETSIZE_MAX)
            else:
                r.title_label.setFixedWidth(title_w)
            if r.button:
                r.button.setFixedWidth(button_w)

    def _add(self, widget):
        self.host_lay.addWidget(widget)
        widget.show()                         # (after it is in the layout: shown before, it would open as a window of its own)

    def _make_row(self, item, expanded):
        row = _Row(item, expanded)
        row.set_compact(self._compact)
        row.changed.connect(self._row_toggled)
        row.actionClicked.connect(self.actionRequested)
        return row

    def _plain_line(self, text, good=False, muted=False):
        """A line without a button: 'Nothing to look at' (calm green) or 'No checks to show' (grey)."""
        box = QFrame()
        box.setObjectName("checkRow")
        box.setProperty("kind", "ok" if good else "none")
        h = QHBoxLayout(box)
        h.setContentsMargins(6, 2, 4, 2)
        h.setSpacing(7)
        if good:
            h.addWidget(_Marker("ok"), 0, Qt.AlignVCenter)
        label = QLabel(text)
        label.setObjectName("checkGood" if good else "checkSentence")
        font = label.font()
        font.setBold(good)
        label.setFont(font)
        h.addWidget(label, 1)
        return box

    # ------------------------------------------------------------------ clicks
    def _toggle_expand_all(self):
        self._expand_all = self.pill_check.isChecked()
        for row in self.rows:
            row.set_expanded(self._expand_all, announce=False)
        self._expanded_keys = set()
        self._fit()

    def _open_list(self, which):
        self._list = None if self._list == which else which
        self._fill_rows()

    def _row_toggled(self):
        self._expanded_keys = {(r.item["title"], r.item["text"]) for r in self.rows if r.expanded}
        self._expand_all = bool(self.rows) and all(r.expanded for r in self.rows)
        self.pill_check.setChecked(self._expand_all)
        self._fit()

    # ------------------------------------------------------------------ size
    def _inner_width(self):
        """The width the lines have: the panel's own width less its margins and frame (worked out from the panel, not from the scroll area, which
        is only resized after the panel is)."""
        return max(self.width() - 2 * SIDE_MARGIN - 2, 120)

    def _rows_height(self, width):
        lay = self.host_lay
        return lay.totalHeightForWidth(width) if lay.hasHeightForWidth() else lay.sizeHint().height()

    def _fit(self):
        """Make the lines exactly as tall as they are, up to what the panel may be; taller than that, they scroll."""
        width = self._inner_width()
        room = MAX_HEIGHT - PILL_HEIGHT - 2 * MARGIN_V - SPACING - 2        # (2: the frame)
        need = self._rows_height(width)
        if need > room:                                                      # (a scroll bar takes some width: the sentences wrap a little more)
            need = self._rows_height(width - self.scroll.verticalScrollBar().sizeHint().width())
        self.scroll.setFixedHeight(max(min(need, room), 0))
        self.updateGeometry()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if e.size().width() != e.oldSize().width():
            compact = e.size().width() < COMPACT_BELOW
            flipped = compact != self._compact
            self._compact = compact
            if flipped:
                for row in self.rows + self.extra_rows:
                    row.set_compact(compact)
            self._align()
            if flipped or any(r.expanded for r in self.rows):
                self._fit()                       # (a sentence shown in full wraps differently at another width; a second line makes a line taller)

    # ------------------------------------------------------------------ what a person can read
    def text(self, everything=False):
        """Plain lines of what the panel shows: the summary, then the things to look at, one line each, 'headline: sentence  [button]'; the
        fine ones and the notes only while their number is open (or with `everything`)."""
        if not self._items:
            return "No checks to show"
        c = self.counts()
        parts = [f"{c['check']} to look at"] if c["check"] else []
        parts += [f"{c['ok']} fine"] if c["ok"] else []
        parts += [f"{c['info']} {'note' if c['info'] == 1 else 'notes'}"] if c["info"] else []
        lines = ["Checks: " + ", ".join(parts)]

        def line(item):
            label = action_label(item)
            return f"  {item['title']}: {item['text']}" + (f"  [{label}]" if label else "")
        if c["check"]:
            lines.append("To look at")
            lines += [line(i) for i in self._items if i["level"] == "check"]
        else:
            lines.append("Nothing to look at")
        for level, head in (("ok", "Fine"), ("info", "Notes")):
            if c[level] and (everything or self._list == level):
                lines.append(head)
                lines += [line(i) for i in self._items if i["level"] == level]
        return "\n".join(lines)

    # ------------------------------------------------------------------ the look
    def _apply_style(self):
        pal = QApplication.palette()
        muted, border = muted_colour(pal), border_colour(pal)
        warn, good = strong_colour(WARN, pal), strong_colour(GOOD, pal)
        css = f"""
QFrame#checksPanel {{ border: 1px solid {border}; border-radius: 6px; }}
QWidget#checksHost {{ background: transparent; }}
QPushButton#pillCheck, QPushButton#pillOk, QPushButton#pillInfo {{ border-radius: 9px; padding: 1px 10px; font-weight: 600; border: 1px solid transparent; }}
QPushButton#pillCheck {{ background: {tint(WARN, 0.16)}; color: {warn}; border-color: {tint(WARN, 0.55)}; }}
QPushButton#pillOk {{ background: {tint(GOOD, 0.14)}; color: {good}; border-color: {tint(GOOD, 0.5)}; }}
QPushButton#pillInfo {{ background: {tint(INFO_GREY, 0.16)}; color: {muted}; border-color: {tint(INFO_GREY, 0.5)}; }}
QPushButton#pillCheck:checked {{ background: {tint(WARN, 0.34)}; }}
QPushButton#pillOk:checked {{ background: {tint(GOOD, 0.3)}; }}
QPushButton#pillInfo:checked {{ background: {tint(INFO_GREY, 0.34)}; }}
QLabel#checkHint, QLabel#checkSentence {{ color: {muted}; }}
QLabel#checkGood {{ color: {good}; }}
QFrame#checkRow {{ border: none; border-left: 3px solid transparent; border-radius: 0; }}
QFrame#checkRow[kind="check"] {{ border-left-color: {WARN}; }}
QFrame#checkRow[kind="ok"] {{ border-left-color: {GOOD}; }}
QFrame#checkRow[kind="info"] {{ border-left-color: {INFO_GREY}; }}
QFrame#checkRow:hover {{ background: rgba(128, 128, 128, 0.12); }}
QPushButton#checkAction {{ border: 1px solid {border}; border-radius: 4px; padding: 0px 9px; min-height: 18px; max-height: 18px; background: transparent; }}
QPushButton#checkAction:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
"""
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()


class _Popup(QFrame):
    """The pop-up that holds the checks panel: a window of its own that closes when anything outside it is clicked. While it is open it owns the
    keyboard: no key of the Review page acts through it (the key 1 is "valid" on the page, and "the first button" in this list)."""

    def __init__(self, parent):
        super().__init__(parent, Qt.Popup)
        self.closed_at = 0.0

    def hideEvent(self, e):
        self.closed_at = time.monotonic()
        super().hideEvent(e)

    def event(self, e):
        if e.type() == QEvent.ShortcutOverride:      # (Qt asks the window before it looks for a shortcut: an accepted override keeps the page's keys from firing)
            e.accept()
            return True
        return super().event(e)

    def keyPressEvent(self, e):
        """The keys of the open list: 1 to 9 press the button of that line, Return reads every sentence in full (or folds them again), and the
        key that opened the list closes it. Escape closes it too (Qt)."""
        chip, key = self.parent(), e.key()
        if chip is not None and Qt.Key_1 <= key <= Qt.Key_9:
            buttons = chip.keyed_buttons()
            if key - Qt.Key_1 < len(buttons):
                buttons[key - Qt.Key_1].click()
            return
        if chip is not None and key in (Qt.Key_Return, Qt.Key_Enter):
            if chip.panel.pill_check.isVisibleTo(self):
                chip.panel.pill_check.click()
            return
        if key == getattr(Qt, f"Key_{CHECKS_KEY}"):
            self.hide()
            return
        super().keyPressEvent(e)


class ChecksChip(QWidget):
    """The checks in the bottom bar: a chip ('4 to look at', amber; 'Nothing to look at', green; grey with no trial) and, beside it, a one-line
    preview of the things to look at. Clicking the chip opens the whole `ChecksPanel` in a pop-up above it (title, sentence, action button for
    every issue, and the fine and note counts); pressing an action button closes the pop-up and emits `actionRequested(action, arg)`.
    It answers `set_items`, `counts`, `text` and `rows` like the panel, so what reads the panel can read this."""
    actionRequested = Signal(str, str)
    POPUP_WIDTH = 480

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("checksChipBox")
        self._css = None
        self.panel = ChecksPanel()
        self.popup = _Popup(self)
        self.popup.setObjectName("checksPopup")
        pl = QVBoxLayout(self.popup)
        pl.setContentsMargins(4, 4, 4, 4)
        pl.addWidget(self.panel)
        self.panel.actionRequested.connect(self._action)
        self.chip = QPushButton("No checks")
        self.chip.setObjectName("checksChip")
        self.chip.setFocusPolicy(Qt.NoFocus)
        self.chip.setCursor(Qt.PointingHandCursor)
        self.chip.setProperty("kind", "none")
        self.chip.clicked.connect(self.toggle_popup)
        self.preview = ElidedLabel("")
        self.preview.setObjectName("checkHint")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self.chip)
        lay.addWidget(self.preview, 1)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._apply_style()

    # ---- what the window calls (the panel's interface)
    @property
    def rows(self):
        return self.panel.rows

    def set_items(self, items):
        self.panel.set_items(items)
        self._refresh()

    def counts(self):
        return self.panel.counts()

    def text(self, everything=False):
        return self.panel.text(everything)

    def _refresh(self):
        c = self.counts()
        items = self.panel._items
        if not items:
            kind, text = "none", "No checks"
        elif c["check"]:
            kind, text = "check", f"⚠ {c['check']} to look at ▴"
        else:
            kind, text = "ok", "✓ Nothing to look at ▴"
        self.chip.setText(text)
        self.chip.setProperty("kind", kind)
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)
        titles = [i["title"] for i in items if i["level"] == "check"]
        self.preview.setText(" · ".join(titles), " · ".join(f"{i['title']}: {i['text']}" for i in items if i["level"] == "check"))
        self.chip.setToolTip(f"The checks of this trial. Click to read every one and to go to what puts it right. Key: {CHECKS_KEY}; in the list, "
                             "1 to 9 press the buttons and Return reads every sentence." if items else "")

    # ---- the pop-up
    def toggle_popup(self):
        if self.popup.isVisible():
            self.popup.hide()
        elif time.monotonic() - self.popup.closed_at > 0.25:          # (the click that closed it by clicking outside must not open it again)
            self.show_popup()

    def keyed_buttons(self):
        """The action buttons of the list, in the order of its lines: the first nine have a key each (1 to 9)."""
        return [r.button for r in self.panel.rows + self.panel.extra_rows if r.button][:MAX_KEYED_BUTTONS]

    def show_popup(self):
        for n, button in enumerate(self.keyed_buttons(), start=1):          # (say the key of each button in its tooltip; the first tip is kept, so this can be done again)
            base = button.property("baseTip") or button.toolTip()
            button.setProperty("baseTip", base)
            button.setToolTip(f"{base} (key {n})")
        self.popup.setFixedWidth(self.POPUP_WIDTH)
        self.popup.layout().activate()
        self.panel._fit()
        self.popup.adjustSize()
        self.popup.show()
        self._place()                                       # (after it is shown: its lines only find their final height then)

    def _place(self):
        """Put the pop-up directly above the chip (below it only when there is no room above), inside the screen."""
        top_left = self.chip.mapToGlobal(QPoint(0, 0))
        self.popup.adjustSize()
        y = top_left.y() - self.popup.height() - 4
        x = top_left.x()
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen is not None:
            if y < screen.top() + 4:
                y = min(top_left.y() + self.chip.height() + 4, screen.bottom() - self.popup.height() - 4)
            x = min(x, screen.right() - self.popup.width() - 4)
        self.popup.move(max(x, 0), y)

    def _action(self, action, arg):
        self.popup.hide()
        self.actionRequested.emit(action, arg)

    # ---- the look
    def _apply_style(self):
        pal = QApplication.palette()
        muted, border = muted_colour(pal), border_colour(pal)
        warn, good = strong_colour(WARN, pal), strong_colour(GOOD, pal)
        css = f"""
QPushButton#checksChip {{ border-radius: 11px; padding: 2px 12px; font-weight: 600; border: 1px solid {border}; color: {muted}; }}
QPushButton#checksChip[kind="check"] {{ background: {tint(WARN, 0.18)}; color: {warn}; border-color: {tint(WARN, 0.6)}; }}
QPushButton#checksChip[kind="ok"] {{ background: {tint(GOOD, 0.16)}; color: {good}; border-color: {tint(GOOD, 0.55)}; }}
QLabel#checkHint {{ color: {muted}; }}
QFrame#checksPopup {{ border: 1px solid {border}; border-radius: 8px; background: {pal.color(pal.ColorRole.Window).name()}; }}
"""
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)
            self.popup.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()
