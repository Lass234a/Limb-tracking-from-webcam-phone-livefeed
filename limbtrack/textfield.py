"""Two small widgets the Live tab's text boxes and drop-downs share (kept apart from gui.py so the boxes in the Position and Participant sections
can use them without importing the window)."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QLineEdit


class TextField(QLineEdit):
    """A text box that tells the window when it has focus (typed letters go into it, so letter shortcuts are off)
    and lets go of focus on Enter or Esc, so the shortcut keys work again."""

    focus_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.returnPressed.connect(self.clearFocus)

    def focusInEvent(self, e):
        super().focusInEvent(e)
        self.focus_changed.emit(True)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self.focus_changed.emit(False)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.clearFocus()
            return
        super().keyPressEvent(e)


class NoWheelCombo(QComboBox):
    """A drop-down that ignores the mouse wheel unless it has focus, so scrolling the side panel cannot change it."""

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()
