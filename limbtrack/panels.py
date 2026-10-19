"""Building blocks for the Live tab's right-hand panel: sections that open one at a time, and a strip of step chips.

The panel used to be one long column of eleven boxes (about three screens tall). Now the things used in every session sit in a
strip that never scrolls, and the rest is a list of sections of which one is open at a time; a closed section still says in one
line what it is set to. Nothing here knows about cameras or angles: the window gives it widgets and states.
"""
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QGroupBox, QPushButton, QVBoxLayout, QWidget

HEADER_STYLE = ("QPushButton { text-align: left; padding: 7px 10px; font-weight: bold; border: 1px solid rgba(128,128,128,0.35);"
                " border-radius: 4px; }"
                "QPushButton:checked { background: rgba(120,160,255,0.18); }")
CHIP_STYLES = {
    "done": "color: #4ade80; border: 1px solid #16a34a; background: rgba(22,163,74,0.12);",
    "todo": "color: #fbbf24; border: 1px solid #d97706; background: rgba(217,119,6,0.12);",
    "next": "color: #111; border: 2px solid #fbbf24; background: #fbbf24; font-weight: bold;",
    "off": "color: #9ca3af; border: 1px solid rgba(128,128,128,0.35); background: transparent;",
}
MARKS = {"done": "✓ ", "todo": "", "next": "▶ ", "off": ""}


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
