"""The Live tab's "Measures" boxes in the Participant section (0.34.0, F2): height, mass, body fat, thigh and shank length by tape, ASIS width.

The engine keeps them per participant code (limbtrack/participant_measures.py, Engine.participant_measures); this box only asks for them and
says what was not kept. Nothing here is measured and nothing here ever stops Record: an empty box is "not measured", and a number outside
the plausible range is not kept (the box turns amber and says why).
"""
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from . import participant_measures as pm
from .textfield import TextField

COLUMNS = 3
BAD_STYLE = "background-color: #fcd34d; color: #000000;"        # (an implausible or unreadable entry)


def _g(value):
    return "" if value is None else f"{value:g}"


class ParticipantMeasuresBox(QWidget):
    """`engine`: the Engine. `code_ok()`: whether the participant code in the window is a usable one (the measures belong to a code)."""

    def __init__(self, engine, code_ok=None):
        super().__init__()
        self.engine = engine
        self._code_ok = code_ok or (lambda: True)
        self.edits = {}                       # measure key -> TextField (Enter or Esc leaves it, as in the Notes box)
        self._problems = {}                   # measure key -> why what is typed was not kept
        self._code = None                     # the participant code the boxes show
        self._stored = None                   # what the engine held when the boxes were last filled or edited
        self._enabled = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        head = QLabel("Measures (typed, optional)")
        head.setStyleSheet("font-weight: bold;")
        head.setToolTip("Kept for this participant code and saved in every trial. Nothing here is measured by the program, and an empty box "
                        "never stops Record. The thigh and shank lengths are also compared with the picture when a ruler is marked: a "
                        "difference above 10 % comes up as a quiet reminder.")
        lay.addWidget(head)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        for i, (key, label, unit, lo, hi, help_text) in enumerate(pm.FIELDS):
            r, c = divmod(i, COLUMNS)
            title = QLabel(f"{label} ({unit})")
            edit = TextField()
            edit.setPlaceholderText(unit)
            edit.setToolTip(f"{help_text} Plausible: {pm.range_text(key)}. Empty = not measured.")
            edit.setMinimumWidth(48)
            edit.textEdited.connect(lambda text, k=key: self._edited(k, text))
            grid.addWidget(title, 2 * r, c)
            grid.addWidget(edit, 2 * r + 1, c)
            self.edits[key] = edit
        for c in range(COLUMNS):
            grid.setColumnStretch(c, 1)
        lay.addLayout(grid)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #d97706; font-weight: bold;")
        self.status.setVisible(False)
        lay.addWidget(self.status)
        self.refresh(False)

    # ------------------------------------------------------------------------------------------ typing
    def _edited(self, key, text):
        value, problem = pm.parse(key, text)
        try:
            self.engine.set_participant_measure(key, value)
        except (RuntimeError, ValueError) as err:          # (a recording started in the meantime; or a value the parser let through)
            value, problem = None, str(err)
        self._problems[key] = problem
        self._stored = self.engine.participant_measures
        self._show_problems()

    def _show_problems(self):
        for key, edit in self.edits.items():
            style = BAD_STYLE if self._problems.get(key) else ""
            if edit.styleSheet() != style:
                edit.setStyleSheet(style)
        text = " ".join(p for p in self._problems.values() if p)
        if text != self.status.text():
            self.status.setText(text)
        if self.status.isHidden() == bool(text):
            self.status.setVisible(bool(text))

    # ------------------------------------------------------------------------------------------ the engine's values
    def _load(self, code, values):
        for key, edit in self.edits.items():
            edit.setText(_g(values.get(key)))             # (setText does not count as typing)
        self._problems = {}
        self._code, self._stored = code, dict(values)
        self._show_problems()

    def refresh(self, recording=False):
        """Fill the boxes from the engine when the participant code (or the engine's values) changed, and switch them off while recording
        or without a usable code. Runs every 200 ms: a widget is touched only when it has to change."""
        code, values = self.engine.participant_code, self.engine.participant_measures
        if code != self._code or values != self._stored:
            self._load(code, values)
        enabled = bool(self._code_ok()) and not recording
        if enabled != self._enabled:
            self._enabled = enabled
            for edit in self.edits.values():
                edit.setEnabled(enabled)
            tip = "" if enabled else ("Not while recording." if recording else "Type the participant code first: the measures belong to it.")
            self.setToolTip(tip)
