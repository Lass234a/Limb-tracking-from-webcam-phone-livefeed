"""The decision part of the Review tab's bottom bar (0.32.0): the verdict on the open trial as 'Valid' and a 'Compromised' menu (the reasons) plus a
note, in one row. It has the same interface as panels.VerdictBar (which the Live tab keeps): `changed(verdict, reasons, note)`, `set_verdict`,
`current`, and the attributes `btn_valid`, `reason_buttons`, `note` and `state`, so what ReviewTab and its tests do with the verdict is unchanged. The
reason buttons are kept (hidden) as the state behind the menu; the menu's checkable actions and the keys 1 to 5 go through them.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QMenu, QPushButton, QToolButton, QWidget, QWidgetAction

from .panels import VERDICT_BUTTONS, VERDICT_TIPS
from .review_style import BAD, GOOD, ElidedLabel, border_colour, muted_colour, tint
from .verdict import COMPROMISED, REASONS, VALID

STATE_WORDS = {VALID: "valid", COMPROMISED: "compromised (not counted)", None: "not judged"}


class VerdictControl(QWidget):
    """Valid, Compromised (menu of reasons) and Note, in one row. Only shows and reports; the window saves."""
    changed = Signal(object, list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._css = None
        self._saved_note = ""
        self.label = ElidedLabel("")                           # (kept for the interface: the title VerdictBar showed; here it is the Valid button's tooltip)
        self.label.hide()
        self.btn_valid = QPushButton("✓ Valid  1")
        self.btn_valid.setObjectName("verdictValid")
        self.btn_valid.setCheckable(True)
        self.btn_valid.setFocusPolicy(Qt.NoFocus)                  # (a click must not take the keyboard from the picture keys)
        self.btn_valid.setToolTip("The trial was completely well done and is valid (key 1).")
        self.btn_valid.clicked.connect(self._valid_clicked)
        self.reason_buttons, self.reason_actions = {}, {}
        self.menu = QMenu(self)
        self.menu.setToolTipsVisible(True)
        for n, key in enumerate(REASONS, start=2):
            b = QPushButton(VERDICT_BUTTONS[key], self)        # the state behind the menu action; never shown
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(VERDICT_TIPS[key])
            b.hide()
            b.clicked.connect(lambda on, k=key: self._reason_clicked(k, on))
            self.reason_buttons[key] = b
            act = QAction(f"{VERDICT_BUTTONS[key]}   (key {n})", self.menu)
            act.setCheckable(True)
            act.setToolTip(VERDICT_TIPS[key])
            act.toggled.connect(lambda on, k=key: self._action_toggled(k, on))
            self.menu.addAction(act)
            self.reason_actions[key] = act
        self.btn_comp = QToolButton()
        self.btn_comp.setObjectName("verdictComp")
        self.btn_comp.setText("Compromised")
        self.btn_comp.setPopupMode(QToolButton.InstantPopup)
        self.btn_comp.setMenu(self.menu)
        self.btn_comp.setFocusPolicy(Qt.NoFocus)
        self.btn_comp.setToolTip("Choose why the trial is compromised (keys 2 to 5). It is then not counted.")
        self.note = QLineEdit()
        self.note.setPlaceholderText("Note (optional), Enter saves it")
        self.note.setMinimumWidth(240)
        self.note.editingFinished.connect(self._note_finished)
        self.note.returnPressed.connect(self.note_menu_close)
        self.note_menu = QMenu(self)
        wa = QWidgetAction(self.note_menu)
        wa.setDefaultWidget(self.note)
        self.note_menu.addAction(wa)
        self.btn_note = QToolButton()
        self.btn_note.setObjectName("verdictNote")
        self.btn_note.setText("Note…")
        self.btn_note.setPopupMode(QToolButton.InstantPopup)
        self.btn_note.setMenu(self.note_menu)
        self.btn_note.setFocusPolicy(Qt.NoFocus)
        self.note_menu.aboutToShow.connect(lambda: self.note.setFocus())
        self.state = ElidedLabel("not judged")                 # 'not judged' / 'valid' / 'compromised (not counted)'
        self.state.setObjectName("verdictState")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        for w in (self.state, self.btn_valid, self.btn_comp, self.btn_note):
            lay.addWidget(w)
        self.state.setMinimumWidth(70)
        self._apply_style()
        self._show_state()

    # ---- the interface of VerdictBar
    def current(self):
        """(verdict, reasons, note) as the buttons and the note stand."""
        reasons = [k for k, b in self.reason_buttons.items() if b.isChecked()]
        v = VALID if self.btn_valid.isChecked() else COMPROMISED if reasons else None
        return v, reasons, self.note.text().strip()

    def set_verdict(self, saved, title=None, tip=""):
        """Show a saved verdict (a dict from verdict.read_verdict, or None) without reporting a change."""
        if title is not None:
            self.label.setText(title, tip)
        saved = saved or {}
        self.btn_valid.setChecked(saved.get("verdict") == VALID)
        for k, b in self.reason_buttons.items():
            b.setChecked(saved.get("verdict") == COMPROMISED and k in (saved.get("reasons") or []))
        self._sync_actions()
        self.note.blockSignals(True)
        self.note.setText(saved.get("note") or "")
        self.note.blockSignals(False)
        self._saved_note = self.note.text().strip()
        self._show_state()

    # ---- what the clicks do (the logic of VerdictBar)
    def _valid_clicked(self, on):
        if on:
            for b in self.reason_buttons.values():
                b.setChecked(False)
            self._sync_actions()
        self._report()

    def _reason_clicked(self, key, on):
        if on:
            self.btn_valid.setChecked(False)
        self._sync_actions()
        self._report()

    def _action_toggled(self, key, on):
        b = self.reason_buttons[key]
        if b.isChecked() != on:
            b.setChecked(on)
            self._reason_clicked(key, on)

    def _sync_actions(self):
        for k, act in self.reason_actions.items():
            act.blockSignals(True)
            act.setChecked(self.reason_buttons[k].isChecked())
            act.blockSignals(False)

    def note_menu_close(self):
        self.note_menu.hide()

    def _note_finished(self):
        if self.note.text().strip() != self._saved_note:     # (editingFinished also comes when the box only loses the focus)
            self._report()

    def _report(self):
        v, reasons, note = self.current()
        self._saved_note = note
        self._show_state()
        self.changed.emit(v, reasons, note)

    def _show_state(self):
        v, reasons, note = self.current()
        self.state.setText(STATE_WORDS[v])
        self.state.setProperty("verdict", v or "none")
        n = len(reasons)
        self.btn_comp.setText("Compromised" + (f" · {n}" if n else ""))
        self.btn_comp.setProperty("on", bool(n))
        self.btn_note.setText("Note ✎" if note else "Note…")
        self.btn_note.setToolTip(note or "Add a note to this verdict.")
        for w in (self.state, self.btn_comp):
            w.style().unpolish(w)
            w.style().polish(w)

    # ---- the look
    def _apply_style(self):
        from PySide6.QtWidgets import QApplication
        pal = QApplication.palette()
        border, muted = border_colour(pal), muted_colour(pal)
        css = f"""
QPushButton#verdictValid, QToolButton#verdictComp, QToolButton#verdictNote {{ border: 1px solid {border}; border-radius: 6px; padding: 3px 12px; min-height: 18px; }}
QToolButton#verdictComp::menu-indicator {{ subcontrol-position: right center; right: 4px; }}
QToolButton#verdictComp {{ padding-right: 18px; }}
QPushButton#verdictValid:checked {{ background: {tint(GOOD, 0.22)}; border-color: {GOOD}; font-weight: 600; }}
QToolButton#verdictComp[on="true"] {{ background: {tint(BAD, 0.18)}; border-color: {BAD}; font-weight: 600; }}
QLabel#verdictState {{ color: {muted}; }}
QLabel#verdictState[verdict="valid"] {{ color: {GOOD}; font-weight: 600; }}
QLabel#verdictState[verdict="compromised"] {{ color: {BAD}; font-weight: 600; }}
"""
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        from PySide6.QtCore import QEvent
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()
