"""Accessibility of the two main screens, checked on the real windows: a screen reader can name every control, a control with a key says
which, every control of Review that never takes the keyboard has a key of its own, and the Privacy window of Review can be opened from the
keyboard. (Colour contrast is in test_theme_contrast.py; what each Review key does is in test_review_keys.py.)"""
import os
import re
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QShortcut  # noqa: E402
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QLabel, QLineEdit, QPushButton, QRadioButton,  # noqa: E402
                               QSlider, QTextEdit, QToolButton, QWidget)

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.review import ReviewTab  # noqa: E402

from tests.test_force import recording  # noqa: E402
from tests.test_pilot_checks import recording_at  # noqa: E402
from tests.test_steps import TESTS  # noqa: E402

CONTROLS = (QPushButton, QCheckBox, QRadioButton, QComboBox, QSlider, QAbstractSpinBox, QLineEdit, QToolButton, QAbstractItemView, QTextEdit)


def _inner(w):
    """Qt's own parts of a control (the editor inside a spin box or a combo box, a header, the scroll bars) are not controls of their own."""
    return (w.objectName().startswith("qt_") or type(w).__name__ in ("QHeaderView", "QTabBar", "QScrollBar")
            or (isinstance(w.parent(), (QAbstractSpinBox, QComboBox)) and isinstance(w, QLineEdit)))


def unnamed(root):
    """The visible controls under `root` with no accessible name, no text and no label pointing at them: what a screen reader cannot say."""
    buddies = {lab.buddy() for lab in root.findChildren(QLabel) if lab.buddy() is not None}
    out = []
    for w in root.findChildren(QWidget):
        if not isinstance(w, CONTROLS) or not w.isVisible() or _inner(w):
            continue
        text = "" if isinstance(w, (QLineEdit, QTextEdit, QAbstractItemView, QComboBox, QSlider)) else (getattr(w, "text", lambda: "")() or "")
        if not (text.strip() or w.accessibleName() or w in buddies):
            out.append((type(w).__name__, w.objectName(), w.toolTip()[:40]))
    return out


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def live_window(app, tmp_path):
    w = MainWindow(TESTS, tmp_path)
    w.resize(1500, 950)
    w.show()
    QApplication.processEvents()
    yield w
    w.live.worker = None
    w.close()


@pytest.fixture(scope="module")
def trial_meta(tmp_path_factory):
    eng, paths = recording(tmp_path_factory.mktemp("a11y_trial"), seconds=2.0)
    return paths["meta.json"]


@pytest.fixture
def review(app, trial_meta):
    tab = ReviewTab(trial_meta.parent.parent)
    tab.resize(1500, 900)
    tab.show()
    tab.load(str(trial_meta))
    QApplication.processEvents()
    yield tab
    tab.trial.close()
    tab.close()


def test_every_control_of_the_live_tab_can_be_named_by_a_screen_reader(live_window):
    assert unnamed(live_window) == []


def test_every_control_of_the_review_tab_can_be_named_by_a_screen_reader(review):
    assert unnamed(review) == []


def test_the_arrow_buttons_are_named_by_what_they_do_not_by_their_arrow(live_window, review):
    assert (live_window.live.btn_pos_prev.accessibleName(), live_window.live.btn_pos_next.accessibleName()) == ("Previous position", "Next position")
    assert (review.btn_prev.accessibleName(), review.btn_next.accessibleName()) == ("Previous trial", "Next trial")
    assert (review.btn_step_back.accessibleName(), review.btn_step_fwd.accessibleName()) == ("One picture back", "One picture on")
    assert review.slider.accessibleName() == "Picture of the video"


def test_in_the_live_tab_every_control_the_keyboard_cannot_tab_to_names_its_key(live_window):
    """Controls that never take the keyboard (so a click leaves the letter keys alone) must then have a key of their own, said in the label or the tooltip."""
    live = live_window.live
    mouse_only = []
    for w in live.findChildren(QWidget):
        if isinstance(w, CONTROLS) and w.isVisible() and w.isEnabled() and w.focusPolicy() == Qt.NoFocus and not _inner(w):
            said = (getattr(w, "text", lambda: "")() or "") + " " + w.toolTip()
            if not re.search(r"\(\s*[A-Za-z0-9/ ]*[A-Za-z0-9]\s*\)|\bkey\b|\bF[1-9]\b", said):
                mouse_only.append((type(w).__name__, said.strip()[:50]))
    assert mouse_only == []


def test_the_privacy_window_opens_from_the_keyboard(review, monkeypatch):
    shortcuts = {sc.key().toString() for sc in review.trial_page.findChildren(QShortcut)}
    assert "P" in shortcuts and "Key: P" in review.btn_privacy.toolTip()
    opened = []
    monkeypatch.setattr(ReviewTab, "open_privacy", lambda self: opened.append(1))
    sc = next(s for s in review.trial_page.findChildren(QShortcut) if s.key().toString() == "P")
    sc.activated.emit()
    assert opened == [1]
    review.set_fix_mode(True)                                  # while dots are being fixed the key does nothing (as F, 1 to 5 and the rest)
    sc.activated.emit()
    assert opened == [1]


# ------------------------------------------------------------------------------------------- Review: a key for every control the keyboard skips
SAYS_A_KEY = re.compile(r"\bKey: |\(keys? |\barrow\b|\bEsc\b")        # 'Key: M.'  '(key 1)'  '(keys 2 to 5)'  '(Left arrow or ,)'  'Esc'


def mouse_only_without_a_key(tab):
    """(the controls looked at, the ones among them that name no key): the visible, enabled controls under `tab` that never take the keyboard (a click
    on one must not take the picture keys away). Left out: item views (the two lists of numbers: their folds open and close together with the key U)
    and the Fix dots workspace (its dots are clicked on the picture, which is what it is for)."""
    controls = tuple(c for c in CONTROLS if c is not QAbstractItemView)
    seen, missing = [], []
    for w in tab.findChildren(QWidget):
        if not isinstance(w, controls) or _inner(w) or not (w.isVisible() and w.isEnabled()) or w.focusPolicy() != Qt.NoFocus or tab.fix_dots.isAncestorOf(w):
            continue
        text = getattr(w, "text", None)
        said = (text() if callable(text) else "") + " " + w.toolTip()
        seen.append(w)
        if not SAYS_A_KEY.search(said):
            missing.append((type(w).__name__, w.objectName(), said.strip()[:60]))
    return seen, missing


@pytest.fixture(scope="module")
def full_recordings(tmp_path_factory):
    root = tmp_path_factory.mktemp("a11y_keys")
    recording_at(root, 70.0, seconds=1.5)
    return root


@pytest.fixture
def full_review(app, full_recordings, tmp_path):
    """Review on a trial that has a video and a force (so every button is enabled), the participant open."""
    dest = tmp_path / "recordings"
    shutil.copytree(full_recordings, dest)
    tab = ReviewTab(dest)
    tab.resize(1500, 900)
    tab.show()
    tab.set_participant(dest / "P01")
    QApplication.processEvents()
    yield tab
    if tab.trial is not None:
        tab.trial.close()
    tab.overview.shutdown()
    tab.close()


def test_on_the_review_trial_page_every_control_the_keyboard_cannot_tab_to_names_its_key(full_review):
    r = full_review
    seen, missing = mouse_only_without_a_key(r)
    assert missing == []
    looked_at = {id(w) for w in seen}
    must = (r.btn_view_trial, r.btn_view_overview, r.btn_data_original, r.btn_data_reviewed, r.btn_fix, r.btn_privacy, r.btn_step_back, r.btn_step_fwd,
            r.checks.chip, r.verdict_bar.btn_valid, r.verdict_bar.btn_comp, r.verdict_bar.btn_note, r.angle_tree.more_check, r.angle_tree.reference_button,
            r.force_tree.open_all_link)
    assert [w.objectName() or type(w).__name__ for w in must if id(w) not in looked_at] == []        # (it did look at the controls that matter: finding nothing proves nothing)


def test_on_the_review_overview_every_control_the_keyboard_cannot_tab_to_names_its_key(full_review):
    r = full_review
    r.show_view("overview")
    assert r.overview.wait_until_done(120000)
    QApplication.processEvents()
    seen, missing = mouse_only_without_a_key(r)
    assert missing == []
    looked_at = {id(w) for w in seen}
    ov = r.overview
    must = (r.btn_view_trial, r.btn_view_overview, r.btn_data_original, r.btn_data_reviewed, ov.approve_button, ov.session_button, ov.export_button, ov.more_button)
    assert [w.objectName() or type(w).__name__ for w in must if id(w) not in looked_at] == []
