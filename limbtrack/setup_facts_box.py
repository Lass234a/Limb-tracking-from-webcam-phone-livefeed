"""The Live tab's "Set-up at this position" box in the Position section (0.34.0, F3): where the strap sits, what held the participant, the seat angle.

The engine keeps the facts per participant, test and target (limbtrack/setup_facts.py, Engine.setup_facts), so the next trial of the same
position starts with what was typed for the one before (prefilled, and the box says so). Nothing here is measured and nothing here ever
stops Record: an empty box is "not noted", and a number outside the plausible range is not kept (the box turns amber and says why).
"""
from PySide6.QtWidgets import QCheckBox, QGridLayout, QGroupBox, QLabel, QVBoxLayout

from . import setup_facts as sf
from .textfield import NoWheelCombo, TextField

BAD_STYLE = "background-color: #fcd34d; color: #000000;"


def _g(value):
    return "" if value is None else f"{value:g}"


class SetupFactsBox(QGroupBox):
    """`engine`: the Engine. `code_ok()`: whether the participant code in the window is usable (the facts belong to a participant)."""

    def __init__(self, engine, code_ok=None):
        super().__init__("Set-up at this position (typed, optional)")
        self.engine = engine
        self._code_ok = code_ok or (lambda: True)
        self._key = None                      # (participant, test, target) the widgets show
        self._stored = None                   # what the engine held when the widgets were last filled or edited
        self._enabled = None
        self._problems = {}
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setSpacing(3)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        self.landmark = NoWheelCombo()
        self.landmark.addItem("not noted", None)
        for key, label in sf.LANDMARKS:
            self.landmark.addItem(label, key)
        self.landmark.setToolTip("The landmark the strap distance is measured from. The protocol records strap locations relative to the "
                                 "superior patellar border, the femoral epicondyles, the malleoli or the metatarsal heads.")
        self.other = TextField()
        self.other.setPlaceholderText("which landmark")
        self.other.setMaxLength(sf.OTHER_MAX_CHARS)
        self.distance = TextField()
        self.distance.setPlaceholderText("cm")
        self.distance.setToolTip(sf.DISTANCE_HELP + f" Plausible: {sf.DISTANCE_RANGE_CM[0]:g} to {sf.DISTANCE_RANGE_CM[1]:g} cm.")
        self.angle = TextField()
        self.angle.setPlaceholderText("deg")
        self.angle.setToolTip(sf.ANGLE_HELP + f" Plausible: {sf.ANGLE_RANGE_DEG[0]:g} to {sf.ANGLE_RANGE_DEG[1]:g} deg.")
        grid.addWidget(QLabel("Strap from"), 0, 0)
        grid.addWidget(self.landmark, 0, 1)
        grid.addWidget(self.other, 0, 2)
        grid.addWidget(QLabel("Strap distance (cm)"), 1, 0)
        grid.addWidget(self.distance, 1, 1)
        grid.addWidget(QLabel("Seat/back angle (deg)"), 2, 0)
        grid.addWidget(self.angle, 2, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid)
        self.checks = {}
        for key, label in sf.SUPPORTS:
            cb = QCheckBox(label)
            cb.toggled.connect(self._changed)
            lay.addWidget(cb)
            self.checks[key] = cb
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.landmark.currentIndexChanged.connect(self._changed)
        self.other.textEdited.connect(self._changed)
        self.distance.textEdited.connect(self._changed)
        self.angle.textEdited.connect(self._changed)
        self.setToolTip("How the participant was held and where the strap sat at this position. Kept for this participant, test and target, "
                        "so the next trial starts with it (change what differs). Saved in every trial. Nothing here is measured by the "
                        "program, and an empty box never stops Record.")
        self.refresh(False)

    # ------------------------------------------------------------------------------------------ typing
    def _read(self):
        """The widgets -> (facts as typed, problems). A number that is not plausible is left out and named."""
        problems = {}
        dist, problems["distance"] = sf.parse_distance(self.distance.text())
        ang, problems["angle"] = sf.parse_angle(self.angle.text())
        facts = {"strap_landmark": self.landmark.currentData(),
                 "strap_landmark_other": self.other.text(),
                 "strap_from_landmark_cm": dist,
                 "supports": [k for k, cb in self.checks.items() if cb.isChecked()],
                 "seat_back_angle_deg": ang}
        return facts, problems

    def _changed(self, *_):
        if self._loading:
            return
        facts, self._problems = self._read()
        self.other.setVisible(self.landmark.currentData() == "other")
        try:
            self.engine.setup_facts = facts
        except (RuntimeError, ValueError) as err:          # (a recording started in the meantime)
            self._problems["engine"] = str(err)
        self._stored = self.engine.setup_facts
        self._show()

    def _show(self):
        for edit, key in ((self.distance, "distance"), (self.angle, "angle")):
            style = BAD_STYLE if self._problems.get(key) else ""
            if edit.styleSheet() != style:
                edit.setStyleSheet(style)
        problems = " ".join(p for p in self._problems.values() if p)
        state = self.engine.setup_state
        if problems:
            text, style = problems, "color: #d97706; font-weight: bold;"
        elif state is None:
            text, style = "Not noted for this position yet.", ""
        else:
            when = state[0][11:16] if len(state[0]) >= 16 else state[0]
            if state[1]:
                text = (f"Carried over from the earlier trial at this position (typed {when}). Change what differs; "
                        f"a change is saved with the next trial.")
            else:
                text = f"Typed {when}. The next trial at this position starts with it."
            style = ""
        if text != self.status.text():
            self.status.setText(text)
        if self.status.styleSheet() != style:
            self.status.setStyleSheet(style)

    # ------------------------------------------------------------------------------------------ the engine's values
    def _load(self, key, facts):
        self._loading = True
        try:
            f = sf.clean(facts) if facts else sf.blank()
            self.landmark.setCurrentIndex(max(self.landmark.findData(f["strap_landmark"]), 0))
            self.other.setText(f["strap_landmark_other"] or "")
            self.other.setVisible(f["strap_landmark"] == "other")
            self.distance.setText(_g(f["strap_from_landmark_cm"]))
            self.angle.setText(_g(f["seat_back_angle_deg"]))
            for k, cb in self.checks.items():
                cb.setChecked(k in f["supports"])
        finally:
            self._loading = False
        self._problems = {}
        self._key, self._stored = key, (dict(facts) if facts else None)
        self._show()

    def refresh(self, recording=False):
        """Fill the widgets from the engine when the position (participant, test, target) or the engine's facts changed, and switch them off
        while recording or without a usable participant code. Runs every 200 ms: a widget is touched only when it has to change."""
        key, facts = self.engine.setup_key(), self.engine.setup_facts
        if key != self._key or facts != self._stored:
            self._load(key, facts)
        elif self.engine.setup_state is not None:
            self._show()                      # ('carried over' turns into 'typed' or back when a trial used the facts)
        enabled = bool(self._code_ok()) and not recording
        if enabled != self._enabled:
            self._enabled = enabled
            for w in (self.landmark, self.other, self.distance, self.angle, *self.checks.values()):
                w.setEnabled(enabled)
            self.setToolTip(self.toolTip().split(" [")[0] + ("" if enabled else
                            (" [Not while recording.]" if recording else " [Type the participant code first.]")))
