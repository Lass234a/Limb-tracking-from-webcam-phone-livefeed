"""The Live tab's "Angle reference" box (0.33.0): one row per angle of the test, to say "the joint is at 0 degrees now" while the dots read 10.

The engine keeps the offset (limbtrack/angle_reference.py, Engine.set_angle_reference); this box only asks for it and says what is kept.
Nothing here changes a measured number: the picture shows the angle from the reference first and the dots' angle beside it, the target is
judged on the angle from the reference, and data.csv holds the dots' angle.
"""
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .angle_reference import DEFAULT_REFERENCE_DEG, VALUE_RANGE_DEG

HINT = ("The dots do not always read the angle of the joint: the crest dot of the hip is not on the pelvis' midline, so a person in neutral reads "
        "10 degrees or more and can never reach extension. Put the joint in a position you know, type what it is (0 in neutral, or what an "
        "inclinometer shows in the test posture) and press Set now. From then on the angle is shown and judged from the reference "
        "(the target means degrees from it) and the dots' angle stays beside it. The recording keeps the dots' angle and the reference. "
        "It is for this participant (it stays from one test to the next while the dots stay) and goes when a dot of that angle is marked again.")


class _Spin(QDoubleSpinBox):
    """A number box that ignores the mouse wheel unless it has focus, so scrolling the side panel cannot change it."""

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class AngleReferenceBox(QGroupBox):
    """`engine`: the Engine. `say`: how a refusal is shown to the operator (LiveTab.say)."""

    def __init__(self, engine, say=None):
        super().__init__("Angle reference (the joint is at 0 now, the dots read more)")
        self.engine = engine
        self._say = say or (lambda text, seconds=4.0: None)
        self._rows = {}                       # angle name -> {"spin", "set", "clear", "status"}
        self._names = None                    # the angles the rows are for
        self._enabled = None
        self._lay = QVBoxLayout(self)
        hint = QLabel(HINT)
        hint.setWordWrap(True)
        self._lay.addWidget(hint)
        self._holder = QVBoxLayout()
        self._holder.setSpacing(4)
        self._lay.addLayout(self._holder)
        self.refresh(False)

    # ------------------------------------------------------------------------------------------ the rows
    def _clear_rows(self):
        while self._holder.count():
            item = self._holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._rows = {}

    def _build(self, state):
        typed = {name: row["spin"].value() for name, row in self._rows.items()}          # (what was typed stays when the rows are made again)
        self._clear_rows()
        for name, title, entry, _live in state:
            box = QWidget()
            col = QVBoxLayout(box)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(1)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold;")
            spin = _Spin()
            spin.setRange(*VALUE_RANGE_DEG)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix(" deg")
            spin.setValue(typed.get(name, DEFAULT_REFERENCE_DEG))
            spin.setToolTip("What the joint is at now: 0 in neutral, or the number an inclinometer shows.")
            set_btn = QPushButton("Set now")
            set_btn.setToolTip("Keep what the dots read at this moment (the unsmoothed angle averaged over 0.2 s) as this reference.")
            clear_btn = QPushButton("Clear")
            clear_btn.setToolTip("Judge this angle on the dots again.")
            set_btn.clicked.connect(lambda _=False, n=name: self.set_now(n))
            clear_btn.clicked.connect(lambda _=False, n=name: self.clear(n))
            top.addWidget(label, 1)
            top.addWidget(spin)
            top.addWidget(set_btn)
            top.addWidget(clear_btn)
            status = QLabel("")
            status.setWordWrap(True)
            col.addLayout(top)
            col.addWidget(status)
            self._holder.addWidget(box)
            self._rows[name] = {"spin": spin, "set": set_btn, "clear": clear_btn, "status": status}
        self._names = [s[0] for s in state]
        self._enabled = None

    def refresh(self, recording=False):
        """Make the rows match the angles of the test and say what is kept. Runs every 200 ms: a widget is touched only when its text changes."""
        state = self.engine.angle_reference_state()
        names = [s[0] for s in state]
        if names != self._names:
            self._build(state)
        for name, title, entry, live in state:
            row = self._rows[name]
            text = self._status_text(entry, live)
            if text != row["status"].text():
                row["status"].setText(text)
            row["clear"].setEnabled(entry is not None and not recording)
        if self._enabled != (not recording):
            for row in self._rows.values():
                row["spin"].setEnabled(not recording)
                row["set"].setEnabled(not recording)
            self._enabled = not recording
            for name, title, entry, live in state:
                self._rows[name]["clear"].setEnabled(entry is not None and not recording)

    @staticmethod
    def _status_text(entry, live):
        if entry is None:
            return "None: this angle is judged on the dots." + ("" if live is None else f" The dots read {live:.1f} now.")
        when = datetime.fromisoformat(entry["taken_at"]).strftime("%H:%M:%S")
        text = (f"Called {entry['reference_deg']:.1f} while the dots read {entry['dots_deg']:.1f} (at {when}): the dots read "
                f"{entry['offset_deg']:+.1f} more than the joint.")
        if live is not None:
            text += f" Now the dots read {live:.1f}, from the reference {live - entry['offset_deg']:.1f}."
        return text

    # ------------------------------------------------------------------------------------------ the buttons
    def set_now(self, name):
        row = self._rows.get(name)
        if row is None:
            return
        try:
            entry = self.engine.set_angle_reference(name, row["spin"].value())
        except (RuntimeError, ValueError) as err:
            self._say(str(err))
            return
        self._say(f"Reference kept: the joint is {entry['reference_deg']:.1f}, the dots read {entry['dots_deg']:.1f}. From now on this angle is "
                  f"shown and judged from the reference (the dots read {entry['offset_deg']:+.1f} more than the joint).", 8.0)
        self.refresh(False)

    def clear(self, name):
        try:
            self.engine.clear_angle_reference(name)
        except RuntimeError as err:
            self._say(str(err))
            return
        self.refresh(False)
