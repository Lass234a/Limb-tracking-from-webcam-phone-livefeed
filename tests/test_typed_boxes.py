"""The typed boxes added in 0.34.0 (Measures, Set-up at this position, Standing reference) behave like the Notes box: Enter or Esc lets go of the
keyboard so the letter keys work again, the 'letters are off' hint shows while one has focus, the drop-downs ignore the mouse wheel without focus,
and none of the new boxes is what makes the Position section wider than the panel.

Real key and wheel events are sent to the real widgets offscreen (a test that only calls setText never sees a key)."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.textfield import TextField  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


@pytest.fixture
def win(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow(TESTS, tmp_path)
    w.resize(1400, 820)
    w.show()
    w.activateWindow()
    w.app = app
    yield w
    w.live.worker = None
    w.close()


def fields(live):
    return {"height": (live.measures_box.edits["height_cm"], "participant"), "distance": (live.setup_box.distance, "position"),
            "angle": (live.setup_box.angle, "position"), "hip reading": (live.standing_box.edits["hip"], "position"),
            "check weight": (live.check_box.edit, "force")}


def test_the_new_text_boxes_are_the_apps_text_fields(win):
    live = win.live
    assert all(isinstance(e, TextField) for e, _ in fields(live).values())
    assert isinstance(live.setup_box.other, TextField)
    assert isinstance(live.check_box.edit, TextField)
    assert live.check_box.edit in live._typing_fields()                                   # (the letter shortcuts and the key hint look at this list)
    assert all(isinstance(e, TextField) for e in live.standing_box.edits.values())
    assert all(isinstance(e, TextField) for e in live.measures_box.edits.values())


@pytest.mark.parametrize("name", ["height", "distance", "angle", "hip reading", "check weight"])
def test_enter_and_esc_let_go_and_the_hint_follows_the_focus(win, name):
    live, app = win.live, win.app
    live.participant.setText("P01")
    live.refresh_status()
    edit, section = fields(live)[name]
    live.open_step(section)
    app.processEvents()
    edit.setFocus()
    app.processEvents()
    assert edit.hasFocus() and (not live.key_hint.isHidden())                    # the operator is told the letters are off (the hint's own flag: it sits in the Participant section)
    QTest.keyClicks(edit, "17")
    QTest.keyClick(edit, Qt.Key_Return)
    app.processEvents()
    assert not edit.hasFocus() and live.key_hint.isHidden()
    assert edit.text() == "17"                                              # (the typed number stays; the letter R after Enter is not typed into it)
    QTest.keyClick(live, Qt.Key_R)
    assert edit.text() == "17"
    edit.setFocus()
    app.processEvents()
    assert (not live.key_hint.isHidden())
    QTest.keyClick(edit, Qt.Key_Escape)
    app.processEvents()
    assert not edit.hasFocus() and live.key_hint.isHidden()


def wheel(widget):
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)


def test_the_mouse_wheel_does_not_change_the_saved_instrument_or_the_landmark_without_focus(win):
    live = win.live
    for combo in (live.standing_box.instrument, live.setup_box.landmark):
        assert combo.count() > 2
        assert not combo.hasFocus()
        before = combo.currentIndex()
        wheel(combo)
        assert combo.currentIndex() == before


def test_the_new_boxes_are_not_what_makes_the_position_section_wide(win):
    """The Position section was already wider than the panel (the plan's long drop-down); the new boxes must not add to that."""
    live = win.live
    live.open_step("position")
    win.app.processEvents()
    body = live.accordion.sections["position"].body
    older = max(c.minimumSizeHint().width() for c in body.findChildren(QWidget)
                if c not in (live.setup_box, live.angle_ref_box, live.standing_box) and not any(
                    b.isAncestorOf(c) for b in (live.setup_box, live.angle_ref_box, live.standing_box)) and c is not body
                and not c.isAncestorOf(live.standing_box))
    for box in (live.setup_box, live.standing_box):
        assert box.minimumSizeHint().width() <= older, (box, box.minimumSizeHint().width(), older)
