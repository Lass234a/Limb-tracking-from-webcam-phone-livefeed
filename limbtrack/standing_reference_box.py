"""The Live tab's "Standing reference" box (0.34.0, F4): the participant stands in the anatomical posture, the operator presses one button, and
3 seconds later the hip, knee and ankle offsets are kept (as angle references) together with the leg segment lengths in the picture.

The engine does the work (Engine.start_standing_reference, limbtrack/standing_reference.py, which sets the offsets through the same angle
reference as the Angle reference box); this box only asks for it, shows the progress and says what was kept. Nothing here changes a measured
number, and nothing here ever refuses Record.
"""
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout

from . import standing_reference as sr
from .angle_reference import AngleReferenceError, check_value
from .reference import INSTRUMENTS
from .textfield import NoWheelCombo, TextField

HINT = ("With the participant standing in the anatomical position (feet under the hips, knees straight, arms down), all dots marked (the foot dots "
        "too, for the ankle), press Capture. For 3 seconds the program reads the hip, knee and ankle angles of the dots and keeps what they read "
        "as the offset of each angle: the joint is called 0 degrees (the anatomical zero), or the number you type here if you read the joint "
        "with a goniometer or an inclinometer. The segment lengths in the picture are kept too. The offsets are set as angle references, so the "
        "angles are shown from them next to the dots' own angle; the latest reference of an angle wins, one set by hand too. The participant must "
        "keep still: an angle that moved is not set.")
BAD_STYLE = "background-color: #fcd34d; color: #000000;"
TITLES = {"hip": "Hip", "knee": "Knee", "ankle": "Ankle"}


class StandingReferenceBox(QGroupBox):
    """`engine`: the Engine. `say`: how a message is shown to the operator (LiveTab.say). `code_ok()`: the participant code in the window is usable."""

    def __init__(self, engine, say=None, code_ok=None):
        super().__init__("Standing reference (the participant stands in the anatomical position)")
        self.engine = engine
        self._say = say or (lambda text, seconds=4.0: None)
        self._code_ok = code_ok or (lambda: True)
        self._last_message = ""
        self._enabled = None
        self.edits = {}
        lay = QVBoxLayout(self)
        hint = QLabel(HINT)
        hint.setWordWrap(True)
        lay.addWidget(hint)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(1)
        for c, name in enumerate(sr.ANGLE_NAMES):
            edit = TextField()
            edit.setPlaceholderText("optional")
            edit.setToolTip(f"Optional. What an instrument reads for the {name} in this posture, in degrees. Empty: the joint is called 0 (the anatomical zero).")
            edit.setMinimumWidth(48)
            edit.textEdited.connect(lambda _text, e=edit: e.setStyleSheet(""))
            self.edits[name] = edit
            grid.addWidget(QLabel(f"{TITLES[name]} (deg)"), 0, c)          # (short: a long label here made the whole Position section wider than the panel)
            grid.addWidget(edit, 1, c)
            grid.setColumnStretch(c, 1)
        lay.addLayout(grid)
        row = QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.instrument = NoWheelCombo()      # (the wheel must not change the saved instrument while the side panel is scrolled)
        for key, title in INSTRUMENTS.items():
            self.instrument.addItem(title, key)
        self.instrument.setToolTip("The instrument the typed readings come from (saved with the standing reference).")
        self.capture_btn = QPushButton(f"Capture {sr.CAPTURE_S:g} s")
        self.capture_btn.setToolTip("Read the angles for 3 seconds with the participant standing still. Press again during the capture to cancel it.")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Forget this participant's standing reference and the angle references it set.")
        self.capture_btn.clicked.connect(self.capture)
        self.clear_btn.clicked.connect(self.clear)
        row.addWidget(QLabel("Instrument"), 0, 0)
        row.addWidget(self.instrument, 0, 1)
        row.addWidget(self.capture_btn, 0, 2)
        row.addWidget(self.clear_btn, 0, 3)
        row.setColumnStretch(1, 1)
        lay.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.refresh(False)

    # ------------------------------------------------------------------------------------------ the buttons
    def _readings(self):
        """{angle: degrees} of the boxes that hold a number; ValueError, with a sentence, for one that is not an angle (that box turns amber)."""
        out = {}
        for name, edit in self.edits.items():
            text = edit.text().strip().replace(",", ".")
            if not text:
                continue
            try:
                out[name] = check_value(text)
            except AngleReferenceError as err:
                edit.setStyleSheet(BAD_STYLE)
                raise ValueError(f"{TITLES[name]} reading: {err}") from None
        return out

    def capture(self):
        if self.engine.standing_state()["state"] == "capturing":
            self.engine.cancel_standing_reference()
            self.refresh(False)
            return
        try:
            self.engine.start_standing_reference(self._readings(), self.instrument.currentData())
        except (RuntimeError, ValueError) as err:
            self._say(str(err), 8.0)
            self.status.setText(str(err))
            return
        self._last_message = ""
        self.refresh(False)

    def clear(self):
        try:
            self.engine.clear_standing_reference()
        except RuntimeError as err:
            self._say(str(err))
            return
        self._last_message = ""
        self.refresh(False)

    # ------------------------------------------------------------------------------------------ what is shown
    @staticmethod
    def summary(state):
        """The plain sentence about the record held for the participant ('' without one)."""
        rec = state.get("record")
        if not rec or not rec.get("angles"):
            return ""
        offs = ", ".join(f"{TITLES.get(n, n)} {a['offset_deg']:+.1f}" for n, a in rec["angles"].items())
        text = f"Kept (taken {rec.get('taken_at', '?')[11:19] or '?'}): the dots read more than the joint by {offs} degrees."
        lengths = (rec.get("segments") or {}).get("lengths") or {}
        if lengths:
            unit = (rec.get("segments") or {}).get("unit", "")
            text += " Segments: " + ", ".join(f"{k} {v:.1f} {unit}" for k, v in lengths.items()) + ("" if unit == "cm" else " (no ruler: pixels)") + "."
        used = state.get("in_use") or []
        text += (f" Used as the angle reference of: {', '.join(used)}." if used else
                 " Not used as an angle reference at the moment (the dots were marked again, or the reference was changed).")
        return text

    def refresh(self, recording=False):
        """Runs every 200 ms: a widget is touched only when its text or state changes."""
        st = self.engine.standing_state()
        capturing = st["state"] == "capturing"
        if capturing:
            text = f"Capturing: keep the participant still... {st['progress'] * st['length_s']:.1f} / {st['length_s']:g} s"
            button = "Cancel"
        else:
            text = " ".join(x for x in (st["message"], self.summary(st)) if x) or "No standing reference yet."
            button = f"Capture {sr.CAPTURE_S:g} s"
            if st["message"] and st["message"] != self._last_message:
                self._last_message = st["message"]
                self._say(st["message"], 12.0)
        if text != self.status.text():
            self.status.setText(text)
        if button != self.capture_btn.text():
            self.capture_btn.setText(button)
        enabled = bool(self._code_ok()) and not recording
        if enabled != self._enabled:
            self._enabled = enabled
            for w in (*self.edits.values(), self.instrument, self.capture_btn):
                w.setEnabled(enabled)
            self.setToolTip("" if enabled else ("Not while recording." if recording else "Type the participant code first: the standing reference belongs to a person."))
        self.clear_btn.setEnabled(enabled and bool(st["record"]))
