"""Building blocks for the Live tab's right-hand panel: sections that open one at a time, and a strip of step chips.

The panel used to be one long column of eleven boxes (about three screens tall). Now the things used in every session sit in a
strip that never scrolls, and the rest is a list of sections of which one is open at a time; a closed section still says in one
line what it is set to. Nothing here knows about cameras or angles: the window gives it widgets and states.
"""
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from .verdict import COMPROMISED, REASONS, VALID

HEADER_STYLE = ("QPushButton { text-align: left; padding: 7px 10px; font-weight: bold; border: 1px solid rgba(128,128,128,0.35);"
                " border-radius: 4px; }"
                "QPushButton:checked { background: rgba(120,160,255,0.18); }")
CHIP_STYLES = {
    "done": "color: #4ade80; border: 1px solid #16a34a; background: rgba(22,163,74,0.12);",
    "todo": "color: #fbbf24; border: 1px solid #d97706; background: rgba(217,119,6,0.12);",
    "next": "color: #111; border: 2px solid #fbbf24; background: #fbbf24; font-weight: bold;",
    "off": "color: #9ca3af; border: 1px solid rgba(128,128,128,0.35); background: transparent;",
    # a soft reminder (0.24.0): done, but look at it again; it never becomes 'the next step' and never stops a recording
    "check": "color: #fbbf24; border: 1px dashed #d97706; background: rgba(217,119,6,0.12); font-weight: bold;",
}
MARKS = {"done": "✓ ", "todo": "", "next": "▶ ", "off": "", "check": "? "}


class ElidedButton(QPushButton):
    """A button whose text is cut short with '...' when it does not fit, instead of making the whole panel wider than the window.
    The full text is the tooltip. `text()` is the full text until the button is on screen and has a width of its own."""
    PAD = 34                          # the padding and border of HEADER_STYLE plus a little room

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""

    def setFullText(self, text):
        self._full = text
        self.setToolTip(text)
        self._fit()

    def fullText(self):
        return self._full

    def minimumSizeHint(self):
        return QSize(60, super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        room = self.width() - self.PAD
        text = self._full
        if self.isVisible() and room > 40:
            text = self.fontMetrics().elidedText(self._full, Qt.ElideRight, room)
        if text != self.text():
            self.setText(text)


class Section(QWidget):
    """A header button and a body that is shown when it is open."""
    toggled = Signal(str, bool)

    def __init__(self, key, title, widgets, number=None, parent=None):
        super().__init__(parent)
        self.key, self.title, self.number, self.summary = key, title, number, ""
        self.header = ElidedButton()
        self.header.setCheckable(True)
        self.header.setStyleSheet(HEADER_STYLE)
        self.header.setCursor(Qt.PointingHandCursor)
        self.body = QWidget()
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(2, 4, 2, 8)
        for w in widgets:
            if isinstance(w, QGroupBox):                 # the section header already names it: drop the box's own frame and title
                w.setTitle("")
                w.setFlat(True)
                w.setStyleSheet("QGroupBox { border: none; margin: 0; padding: 0; }")
            bl.addWidget(w)
        self.body.setVisible(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self.header.toggled.connect(self._on_toggled)
        self._refresh_header()

    def _refresh_header(self):
        arrow = "▾" if self.header.isChecked() else "▸"
        head = f"{self.number}  {self.title}" if self.number else self.title
        self.header.setFullText(f"{arrow}  {head}" + (f"   —   {self.summary}" if self.summary and not self.header.isChecked() else ""))

    def _on_toggled(self, on):
        self.body.setVisible(on)
        self._refresh_header()
        self.toggled.emit(self.key, on)

    def is_open(self):
        return self.header.isChecked()

    def set_open(self, on):
        self.header.setChecked(bool(on))

    def set_summary(self, text):
        if text != self.summary:
            self.summary = text
            self._refresh_header()


class Accordion(QWidget):
    """Sections of which at most one is open (opening one closes the others)."""
    opened = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(4)
        self.sections = {}
        self._busy = False

    def add(self, key, title, widgets, number=None):
        sec = Section(key, title, widgets, number)
        sec.toggled.connect(self._on_section_toggled)
        self.sections[key] = sec
        self._lay.addWidget(sec)
        return sec

    def _on_section_toggled(self, key, on):
        if self._busy:
            return
        if on:
            self._busy = True
            try:
                for k, s in self.sections.items():
                    if k != key and s.is_open():
                        s.set_open(False)
            finally:
                self._busy = False
            self.opened.emit(key)

    def open(self, key):
        if key in self.sections:
            self.sections[key].set_open(True)

    def current(self):
        return next((k for k, s in self.sections.items() if s.is_open()), None)

    def set_summary(self, key, text):
        if key in self.sections:
            self.sections[key].set_summary(text)


class ReminderBar(QFrame):
    """Soft reminders (0.24.0): one row each, with the buttons that answer it. Rows come and go as the window decides; the bar is
    hidden when no row is open. Nothing here ever stops a recording: it only makes sure the operator has seen the question."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("reminders")
        self.setStyleSheet("QFrame#reminders { border: 1px solid #d97706; border-radius: 4px; background: rgba(217,119,6,0.10); }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(5, 2, 5, 2)
        lay.setSpacing(2)
        self.rows = {}                                       # key -> (row widget, label)
        self.setVisible(False)

    def add_row(self, key, buttons):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        label = QLabel()
        label.setWordWrap(True)
        label.setStyleSheet("color: #d97706; font-weight: bold;")
        h.addWidget(label, 1)
        for b in buttons:
            b.setFocusPolicy(Qt.NoFocus)                     # (a click must not take the keyboard from the letter keys)
            b.setStyleSheet("padding: 2px 7px;")
            h.addWidget(b)
        row.setVisible(False)
        self.layout().addWidget(row)
        self.rows[key] = (row, label)

    def set_row(self, key, text):
        """Show the row with this text, or hide it when the text is empty."""
        row, label = self.rows[key]
        if text and label.text() != text:
            label.setText(text)
        if row.isHidden() == bool(text):
            row.setVisible(bool(text))
        want = any(not r.isHidden() for r, _ in self.rows.values())
        if self.isHidden() == want:
            self.setVisible(want)

    def open_keys(self):
        return [k for k, (r, _) in self.rows.items() if not r.isHidden()]


class StepStrip(QWidget):
    """Chips for the steps of a session. Each shows done / to do / next / off and opens its section when clicked."""
    activated = Signal(str)

    def __init__(self, steps, columns=3, parent=None):
        super().__init__(parent)
        self.chips, self.labels, self.states = {}, {}, {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for i, (key, label) in enumerate(steps):
            chip = QPushButton()
            chip.setCursor(Qt.PointingHandCursor)
            chip.setFixedHeight(30)                         # (a symbol that comes from another font must not make one chip taller than the rest)
            chip.clicked.connect(lambda _=False, k=key: self.activated.emit(k))
            self.chips[key], self.labels[key] = chip, label
            grid.addWidget(chip, i // columns, i % columns)
            self.set_state(key, "off")

    def set_state(self, key, state, text=None, tip=""):
        """state: 'done' (green tick), 'todo' (amber), 'next' (the one to do now), 'off' (not needed yet or not used)."""
        label = text or self.labels[key]
        chip = self.chips[key]
        if self.states.get(key) == (state, label, tip):
            return
        self.states[key] = (state, label, tip)
        chip.setText(MARKS[state] + label)
        chip.setStyleSheet("QPushButton { padding: 3px 6px; border-radius: 4px; " + CHIP_STYLES[state] + " }")
        chip.setToolTip(tip)

    def state_of(self, key):
        return self.states[key][0]


VERDICT_STYLES = {                              # the frame's colour says the verdict at a glance
    VALID: ("#16a34a", "rgba(22,163,74,0.10)"),
    COMPROMISED: ("#dc2626", "rgba(220,38,38,0.10)"),
    None: ("rgba(128,128,128,0.45)", "transparent"),
}
VERDICT_BUTTONS = {"wrong_dots": "Dots", "wrong_position": "Position", "bad_force": "Force", "other": "Other"}
VERDICT_TIPS = {
    "wrong_dots": "Compromised: a dot was placed or marked in the wrong place, jumped, swapped or was lost when it mattered.",
    "wrong_position": "Compromised: the participant was not in the position chosen in the program (or the wrong one was chosen).",
    "bad_force": "Compromised: the strap was not pulling, slipped or jerked, the signal was cut off, or the zero was off.",
    "other": "Compromised for another reason: write it in the note.",
}


class VerdictBar(QFrame):
    """The operator's verdict on one trial (0.28.0, limbtrack/verdict.py): *Valid*, or one or more reasons it is compromised, and a
    note. Under Record in the Live tab (the trial just saved) and in Review (the trial that is open). It only shows and reports; the
    window saves. `changed(verdict, reasons, note)` comes on every click and when the note is finished; `set_verdict` shows a saved
    verdict without it."""

    changed = Signal(object, list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("verdict")
        lay = QVBoxLayout(self)                               # two rows, kept low: it sits in the panel's pinned part after each trial
        lay.setContentsMargins(5, 3, 5, 3)
        lay.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(3)
        self.label = QLabel()                                 # which trial: "Trial T2:" (the position is in its tooltip)
        self.label.setStyleSheet("font-weight: bold;")
        row.addWidget(self.label)
        self.btn_valid = QPushButton("Valid")
        self.btn_valid.setToolTip("The trial was completely well done and is valid.")
        self.reason_buttons = {}
        for b in [self.btn_valid] + [QPushButton(VERDICT_BUTTONS[k]) for k in REASONS]:
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)                     # (a click must not take the keyboard from the letter keys)
            b.setStyleSheet("QPushButton { padding: 2px 6px; } QPushButton:checked { font-weight: bold; }")
            row.addWidget(b)
        for k, b in zip(REASONS, [row.itemAt(i).widget() for i in range(2, row.count())]):
            b.setToolTip(VERDICT_TIPS[k])
            self.reason_buttons[k] = b
            b.clicked.connect(lambda on, key=k: self._reason_clicked(key, on))
        self.btn_valid.clicked.connect(self._valid_clicked)
        lay.addLayout(row)
        second = QHBoxLayout()
        second.setSpacing(4)
        self.state = QLabel()                                 # "not judged", "VALID", "COMPROMISED (not counted)"
        second.addWidget(self.state)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Note (optional), Enter saves it")
        self.note.editingFinished.connect(self._note_finished)
        second.addWidget(self.note, 1)
        lay.addLayout(second)
        self._saved_note = ""
        self._show_state()

    def current(self):
        """(verdict, reasons, note) as the buttons and the note stand."""
        reasons = [k for k, b in self.reason_buttons.items() if b.isChecked()]
        v = VALID if self.btn_valid.isChecked() else COMPROMISED if reasons else None
        return v, reasons, self.note.text().strip()

    def set_verdict(self, saved, title=None, tip=""):
        """Show a saved verdict (a dict from verdict.read_verdict, or None) without reporting a change."""
        if title is not None:
            self.label.setText(title)
            self.label.setToolTip(tip)
        saved = saved or {}
        self.btn_valid.setChecked(saved.get("verdict") == VALID)
        for k, b in self.reason_buttons.items():
            b.setChecked(saved.get("verdict") == COMPROMISED and k in (saved.get("reasons") or []))
        self.note.blockSignals(True)
        self.note.setText(saved.get("note") or "")
        self.note.blockSignals(False)
        self._saved_note = self.note.text().strip()
        self._show_state()

    def _valid_clicked(self, on):
        if on:
            for b in self.reason_buttons.values():
                b.setChecked(False)
        self._report()

    def _reason_clicked(self, key, on):
        if on:
            self.btn_valid.setChecked(False)
        self._report()

    def _note_finished(self):
        if self.note.text().strip() != self._saved_note:     # (editingFinished also comes when the box only loses the focus)
            self._report()

    def _report(self):
        v, reasons, note = self.current()
        self._saved_note = note
        self._show_state()
        self.changed.emit(v, reasons, note)

    def _show_state(self):
        v, reasons, _ = self.current()
        text = "VALID" if v == VALID else "COMPROMISED (not counted)" if v == COMPROMISED else "not judged"
        border, back = VERDICT_STYLES[v]
        self.state.setText(text)
        self.state.setStyleSheet(f"color: {border}; font-weight: bold;" if v else "color: #9ca3af;")
        self.setStyleSheet(f"QFrame#verdict {{ border: 1px solid {border}; border-radius: 4px; background: {back}; }}")
