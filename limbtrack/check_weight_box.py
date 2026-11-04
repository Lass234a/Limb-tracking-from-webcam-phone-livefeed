"""The Live tab's "Check weight" rows in the Force box (0.35.0, F8): the known mass in kg, one button, and the result as green, amber or red.

The engine does the work (Engine.make_check_weight, limbtrack/check_weight.py); this box only asks for the mass, says what is missing (the force
not connected, no zero, a swinging weight) and shows the latest result. Nothing here ever stops Record, and the check changes no zero, no setting and no
measured number. The mass is remembered between launches by the window (it is the operator's own weight, not a measurement); the result is not (a
restart loses the zero it was made against).
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import check_weight as cw
from .textfield import TextField

BAD_STYLE = "background-color: #fcd34d; color: #000000;"
LEVEL_STYLE = {"green": "color: #15803d; font-weight: bold;", "amber": "color: #d97706; font-weight: bold;", "red": "color: #dc2626; font-weight: bold;"}
WARN_STYLE = "color: #d97706; font-weight: bold;"


def _g(value):
    return "" if value is None else f"{value:g}"


class CheckWeightBox(QWidget):
    """`engine`: the Engine. `source()`: the force source in use (or None). `note(event, detail)`: one line to the session log."""
    mass_changed = Signal()                   # the typed mass changed (the window remembers it)

    def __init__(self, engine, source, note=None):
        super().__init__()
        self.engine, self._source, self._note = engine, source, note or (lambda event, detail="": None)
        self.mass_kg = None                   # the mass typed, when it is usable
        self._problem = None                  # why what is typed is not kept
        self._why = ""                        # why the last press made no check (until something changes)
        self._why_state = None                # (connected, zero set) when it was said
        self._enabled = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Check weight (kg)"))
        self.edit = TextField()
        self.edit.setPlaceholderText("kg")
        self.edit.setMaximumWidth(70)
        self.edit.setToolTip(f"The known mass you hang on the rig, {cw.MASS_RANGE_KG[0]:g} to {cw.MASS_RANGE_KG[1]:g} kg (a comma works as the decimal mark). "
                             "Remembered.")
        self.edit.textEdited.connect(self._edited)
        row.addWidget(self.edit)
        row.addStretch(1)
        lay.addLayout(row)
        self.button = QPushButton("Check weight: hung, rig zeroed")
        self.button.setToolTip("First Zero (Z) with nothing but the hook or strap hanging, then hang the known mass and press this. The program averages "
                               f"the zeroed force over {cw.AVERAGE_S:g} s (the 50 Hz hum cancels out), compares it with mass x 9.80665 N and shows the "
                               f"error: green up to {cw.AMBER_ABOVE_PCT:g} %, amber up to {cw.RED_ABOVE_PCT:g} %, red above (a proposal the group must "
                               "confirm). The result goes to the session log and into every following trial. It changes no zero and no number.")
        self.button.clicked.connect(self.check)
        lay.addWidget(self.button)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        lay.addWidget(self.label)
        self.refresh(False)

    # ------------------------------------------------------------------------------------------ the mass
    def set_mass(self, kg):
        """Put a remembered mass back (a value that is not usable now is ignored)."""
        if kg is None or cw.mass_problem(kg):
            return
        self.mass_kg = round(float(kg), 2)
        self.edit.setText(_g(self.mass_kg))
        self._problem = None
        self.refresh(False)

    def _edited(self, text):
        self.mass_kg, self._problem = cw.parse_mass(text)
        self.mass_changed.emit()
        self.refresh(False)

    # ------------------------------------------------------------------------------------------ the check
    def check(self):
        """The button: check the force against the typed mass."""
        text = self.edit.text()
        self.mass_kg, self._problem = cw.parse_mass(text)
        if self.mass_kg is None:
            self._why = self._problem or f"Type the known mass first ({cw.MASS_RANGE_KG[0]:g} to {cw.MASS_RANGE_KG[1]:g} kg)."
            self._why_state = self._state()
            self.refresh(False)
            return None
        try:
            result, why = self.engine.make_check_weight(self.mass_kg)
        except RuntimeError as e:             # (a recording started in the meantime)
            result, why = None, str(e)
        if result is None:
            self._why, self._why_state = why, self._state()
            self._note("check weight refused", f"mass_kg={self.mass_kg:g}; {why}")
        else:
            self._why, self._why_state = "", None
            self._note("check weight", cw.log_detail(result))
        self.refresh(False)
        return result

    # ------------------------------------------------------------------------------------------ showing it
    def _state(self):
        src = self._source()
        return (src is not None, src is not None and src.calibration.zero_set)

    def refresh(self, recording=False):
        """Show the latest check or what is missing. Runs every 200 ms: a widget is touched only when it has to change."""
        src = self._source()
        connected, zero = self._state()
        if self._why and self._why_state != (connected, zero):
            self._why = ""                    # connecting or zeroing since: the old refusal no longer applies
        enabled = connected and not recording
        if enabled != self._enabled:
            self._enabled = enabled
            self.button.setEnabled(enabled)
        check = self.engine.check_weight
        parts, style = [], ""
        if self._problem:
            parts.append(self._problem)
            style = WARN_STYLE
        if self._why:
            parts.append(self._why)
            style = WARN_STYLE
        elif not connected:
            parts.append("The force is not connected: connect it first (the check needs a source with a load cell calibration and a zero).")
            style = WARN_STYLE
        elif not zero:
            parts.append("Not zeroed yet: press Zero (Z) with nothing but the hook or strap hanging, then hang the weight.")
            style = WARN_STYLE
        if check is not None:
            parts.append(cw.summary_text(check, src.calibration if src is not None else None))
            if not style:
                style = LEVEL_STYLE.get(check.get("level"), "")
        text = "\n".join(parts)
        edit_style = BAD_STYLE if self._problem else ""
        if self.edit.styleSheet() != edit_style:
            self.edit.setStyleSheet(edit_style)
        if text != self.label.text():
            self.label.setText(text)
        if self.label.styleSheet() != style:
            self.label.setStyleSheet(style)
        self.edit.setEnabled(not recording)
