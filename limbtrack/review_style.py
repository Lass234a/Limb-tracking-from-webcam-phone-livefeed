"""The look of the Review tab (0.29.0): one stylesheet for the tab, built from the window's own palette so it is right in a light and in a
dark Windows theme, and the small widgets it uses (a key-number card, a coloured verdict text). Nothing here touches the Live tab.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout

GOOD, WARN, BAD, ACCENT = "#16a34a", "#d97706", "#dc2626", "#2563eb"       # the greens, ambers and reds of the rest of the program
STATE_COLOURS = {"good": GOOD, "warn": WARN, "bad": BAD, None: None}


def _mix(a, b, share):
    """`share` of colour a over colour b, as '#rrggbb'."""
    return QColor(round(a.red() * share + b.red() * (1 - share)), round(a.green() * share + b.green() * (1 - share)),
                  round(a.blue() * share + b.blue() * (1 - share))).name()


def muted_colour(pal):
    """The text colour for captions and hints: the text colour pulled 40 % toward the background (readable on both)."""
    return _mix(pal.color(QPalette.WindowText), pal.color(QPalette.Window), 0.62)


def border_colour(pal):
    return _mix(pal.color(QPalette.WindowText), pal.color(QPalette.Window), 0.2)


def tint(colour, alpha=0.12):
    """'rgba(r,g,b,alpha)' of a '#rrggbb' colour, for a quiet background that does not fight the text on it."""
    c = QColor(colour)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def build_stylesheet(pal):
    """The Review tab's stylesheet for this palette. Only objectNames made in review.py / review_overview.py are styled, so the Live tab
    and the plots are left as they are."""
    muted, border = muted_colour(pal), border_colour(pal)
    base = pal.color(QPalette.Base).name()
    return f"""
QFrame#reviewHeader {{ border: none; border-bottom: 1px solid {border}; }}
QLabel#trialTitle {{ font-size: 10.5pt; font-weight: 600; }}
QLabel#trialSubtitle, QLabel#hint {{ color: {muted}; font-size: 9pt; }}
QFrame#card {{ background: {base}; border: 1px solid {border}; border-radius: 6px; }}
QLabel#cardCaption {{ color: {muted}; font-size: 8pt; }}
QLabel#cardValue {{ font-size: 13pt; font-weight: 600; }}
QLabel#cardSub {{ color: {muted}; font-size: 8pt; }}
QPushButton#fixToggle {{ border: 1px solid {ACCENT}; color: {ACCENT}; border-radius: 6px; padding: 4px 12px; font-weight: 600; }}
QPushButton#fixToggle:hover {{ background: {tint(ACCENT, 0.12)}; }}
QPushButton#fixToggle:checked {{ background: {ACCENT}; color: white; }}
QPushButton#fixToggle:disabled {{ border-color: {border}; color: {muted}; font-weight: normal; }}
QFrame#bottomBar {{ border: none; border-top: 1px solid {border}; }}
QLabel#flagLine {{ color: {WARN}; font-weight: 600; padding: 1px 4px; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QSplitter::handle:hover {{ background: {tint(ACCENT, 0.25)}; }}
QTabWidget#numberTabs QTabBar::tab {{ padding: 4px 14px; }}
QPushButton#segLeft, QPushButton#segRight {{ padding: 4px 14px; border: 1px solid {border}; background: transparent; }}
QPushButton#segLeft {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QPushButton#segRight {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; border-left: none; }}
QPushButton#segLeft:checked, QPushButton#segRight:checked {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
QPushButton#nextUnjudged {{ border: 1px solid {ACCENT}; color: {ACCENT}; border-radius: 6px; padding: 4px 12px; font-weight: 600; }}
QPushButton#nextUnjudged:hover {{ background: {tint(ACCENT, 0.12)}; }}
QPushButton#nextUnjudged:disabled {{ border-color: {border}; color: {muted}; font-weight: normal; }}
QTableWidget#overviewTable {{ gridline-color: transparent; border: 1px solid {border}; border-radius: 6px; }}
QTableWidget#overviewTable::item {{ padding: 3px 6px; }}
QTableWidget#overviewTable QHeaderView::section {{ padding: 4px 6px; border: none; border-bottom: 1px solid {border}; font-weight: 600; }}
QTabWidget#numberTabs::pane {{ border: 1px solid {border}; border-radius: 6px; top: -1px; }}
QProgressBar#overviewProgress {{ max-height: 8px; border: none; border-radius: 4px; background: {tint(ACCENT, 0.15)}; text-align: center; }}
QProgressBar#overviewProgress::chunk {{ background: {ACCENT}; border-radius: 4px; }}
"""


class ElidedLabel(QLabel):
    """A one-line label that cuts its text with '...' to the room it has (it never widens the window); the whole text is its tooltip.
    `text()` gives the whole text, not the cut one."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, text, tip=None):
        self._full = text
        self.setToolTip(text if tip is None else tip)
        self._fit()

    def text(self):
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        QLabel.setText(self, self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(self.width() - 2, 10)))


class MetricCard(QFrame):
    """One key number: a small caption and the number large. The line under it (when and how it was found) is in the tooltip since 0.32.0:
    the strip of cards is half as tall. `set_value(value, sub, state)`: `state` None, 'good', 'warn' or 'bad' colours the number."""

    def __init__(self, caption, tip="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._tip = tip
        self.setToolTip(tip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(0)
        self.caption, self.value, self.sub = QLabel(caption), QLabel("-"), QLabel("")
        self.caption.setObjectName("cardCaption")
        self.value.setObjectName("cardValue")
        self.sub.setObjectName("cardSub")
        for w in (self.caption, self.value, self.sub):
            w.setTextInteractionFlags(Qt.NoTextInteraction)
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # a long line is cut off (and in the tooltip), it never widens the window
            lay.addWidget(w)
        self.sub.setWordWrap(False)
        self.sub.setVisible(False)
        self.setMinimumWidth(80)
        self._state = None

    def set_value(self, value, sub="", state=None):
        self.value.setText(value)
        self.sub.setText(sub)
        self.setToolTip((self._tip + ("\n\n" + sub if sub else "")).strip())
        if state != self._state:
            colour = STATE_COLOURS.get(state)
            self.value.setStyleSheet(f"color: {colour};" if colour else "")
            self._state = state
