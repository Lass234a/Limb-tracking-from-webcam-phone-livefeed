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
QFrame#card {{ background: {base}; border: 1px solid {border}; border-radius: 8px; }}
QLabel#cardCaption {{ color: {muted}; font-size: 8.5pt; }}
QLabel#cardValue {{ font-size: 17pt; font-weight: 600; }}
QLabel#cardSub {{ color: {muted}; font-size: 8.5pt; }}
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


class MetricCard(QFrame):
    """One key number: a small caption, the number large, and a line under it (when and how it was found). `set_value(value, sub, state)`:
    `state` None, 'good', 'warn' or 'bad' colours the number."""

    def __init__(self, caption, tip="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setToolTip(tip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
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
        self.setMinimumWidth(90)
        self._state = None

    def set_value(self, value, sub="", state=None):
        self.value.setText(value)
        self.sub.setText(sub)
        self.sub.setToolTip(sub)
        if state != self._state:
            colour = STATE_COLOURS.get(state)
            self.value.setStyleSheet(f"color: {colour};" if colour else "")
            self._state = state
