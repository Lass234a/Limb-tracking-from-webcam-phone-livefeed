"""The pinned bar of the Live tab (0.18.0): the position dropdown and the dot buttons stay in view whichever section is open, and
D / N / B / 1-9 (F4 / F3 / F2 while a text box has focus) work them from the keyboard."""
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QSizePolicy  # noqa: E402

from limbtrack.gui import DOT_CHIP_COLUMNS, DOT_CHIP_STYLE, MainWindow, TRIALS_PER_POSITION  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)


class RunningCamera:
    """Stands in for a running camera so the window thinks a live source is open (enough for refresh_status too)."""
    is_file = False
    paused = False
    at_end = False
    source = 0
    camera = None
    status_note = ""
    lock_message = ""

    def stop(self):
        pass

    def wait(self, ms=0):
        return True

    def isRunning(self):
        return True


@pytest.fixture
def win(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(TESTS, tmp_path)
    w.live.worker = RunningCamera()
    w.engine.process(render(PTS, seed=0), 0.0)
    yield w
    w.engine.rec = None
    w.live.worker = None
    w.close()


def row_of(live, key, tgt):
    return live.plan_rows.index((key, tgt))


def mark_all(win):
    for n in win.engine.active_dots:
        assert win.engine.mark_dot(n, *(PTS[n] + 2))
    win.live.update_dot_rows()


def shortcuts_for(live, *keys):
    wanted = {QKeySequence(k) for k in keys}
    return [s for s in live.findChildren(QShortcut) if s.key() in wanted]


# ------------------------------------------------------------------------------------------------ the bar stays in view
def test_the_bar_is_not_in_a_section_and_stays_in_view_whichever_section_is_open(win):
    live = win.live
    assert not live.accordion.isAncestorOf(live.quick_bar) and not live.side_scroll.isAncestorOf(live.quick_bar)
    win.resize(1400, 820)
    win.show()
    QApplication.processEvents()
    for key in live.accordion.sections:
        live.accordion.open(key)
        QApplication.processEvents()
        assert live.quick_bar.isVisibleTo(win), key
        assert live.pos_combo.isVisibleTo(win) and live.btn_redo_dots.isVisibleTo(win), key
        assert all(c.isVisibleTo(win) for n, c in live.chips.items() if n in win.engine.active_dots), key
    win.hide()


def test_nothing_in_the_bar_can_make_the_panel_wider_than_the_window(win):
    """The offscreen platform has no real fonts, so widths cannot be measured here (see test_steps.py); the rule is on how the parts
    are built: short button texts, buttons and the dropdown that never ask for the width of their text, a caption that wraps."""
    live = win.live
    for n in win.engine.test.all_dots:                              # strap and anchor on as well: nine buttons
        live.lm_checks[n].setChecked(True)
    live.update_dot_rows()
    assert len([c for c in live.chips.values() if not c.isHidden()]) == len(win.engine.test.all_dots)
    for chip in live.chips.values():
        assert len(chip.text()) <= 12 and chip.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored, chip.text()
    for b in (live.btn_redo_dots, live.btn_pos_prev, live.btn_pos_next):
        assert len(b.text()) <= 12 and b.focusPolicy() == Qt.NoFocus
    assert live.pos_combo.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon
    assert live.btn_redo_dots.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert DOT_CHIP_COLUMNS * 2 > len(win.engine.test.all_dots)              # two rows of buttons at most, Redo all in the last cell
    grid = live.chip_grid
    assert grid.rowCount() == 2 and grid.itemAtPosition(1, DOT_CHIP_COLUMNS - 1).widget() is live.btn_redo_dots


# ------------------------------------------------------------------------------------------------ the position dropdown
def test_the_dropdown_lists_the_plan_and_follows_the_position(win):
    live = win.live
    assert live.pos_combo.count() == len(live.plan_rows)
    assert live.pos_combo.currentIndex() == live._plan_current == row_of(live, "knee_extension", 20)
    live.on_plan_clicked(live.plan.item(row_of(live, "knee_flexion", 90)))          # the list in the Position section
    assert live.pos_combo.currentIndex() == row_of(live, "knee_flexion", 90)
    live._pick(live.test_combo, "hip_flexion")                                      # the old test box
    assert live.pos_combo.currentIndex() == live._plan_current == row_of(live, "hip_flexion", win.engine.target)
    live.target_spin.setValue(0.0)
    assert live.pos_combo.currentIndex() == row_of(live, "hip_flexion", 0)


def test_the_dropdown_shows_the_trial_counts_of_the_typed_participant(win):
    live = win.live
    assert f"(0/{TRIALS_PER_POSITION})" not in live.pos_combo.itemText(0)           # no valid code yet: no counts
    folder = Path(win.engine.recordings_dir) / "P01"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "P01_knee_extension_20deg_20260101-000000_meta.json").write_text("{}")
    live.participant.setText("P01")
    assert live.pos_combo.itemText(row_of(live, "knee_extension", 20)).endswith(f"(1/{TRIALS_PER_POSITION})")
    assert live.pos_combo.itemText(row_of(live, "knee_extension", 70)).endswith(f"(0/{TRIALS_PER_POSITION})")
    assert live.pos_combo.toolTip().startswith("Next: trial 2 of")


def test_picking_an_item_sets_test_and_target_like_a_click_on_the_plan(win):
    live = win.live
    row = row_of(live, "hip_extension_k90", 90)
    live.pos_combo.activated.emit(row)
    assert win.engine.test.key == "hip_extension_k90" and win.engine.target == 90
    assert "Position set" in live.instruction.text()
    assert live.pos_combo.currentIndex() == row


def test_a_target_outside_the_plan_leaves_the_dropdown_empty_and_says_so(win):
    live = win.live
    live.participant.setText("P01")                                               # (without a code the caption asks for one first)
    live.target_spin.setValue(33.0)
    assert live._plan_current is None and live.pos_combo.currentIndex() == -1
    assert "not a position of the plan" in live.pos_combo.toolTip()


def test_previous_and_next_walk_the_plan_and_stop_at_the_ends(win):
    live = win.live
    n = len(live.plan_rows)
    live.go_to_position(0)
    live.btn_pos_prev.click()
    assert live._plan_current == 0 and "first position" in live.instruction.text()
    live.btn_pos_next.click()
    assert live._plan_current == 1 and live.pos_combo.currentIndex() == 1
    assert (win.engine.test.key, win.engine.target) == (live.plan_rows[1][0], live.plan_rows[1][1])
    live.go_to_position(n - 1)
    live.btn_pos_next.click()
    assert live._plan_current == n - 1 and "last position" in live.instruction.text()


def test_from_a_position_outside_the_plan_next_starts_at_the_top_and_previous_at_the_bottom(win):
    live = win.live
    live.target_spin.setValue(33.0)
    live.btn_pos_next.click()
    assert live._plan_current == 0
    live.target_spin.setValue(33.0)
    live.btn_pos_prev.click()
    assert live._plan_current == len(live.plan_rows) - 1


def test_the_position_cannot_change_during_a_recording(win):
    live = win.live
    before = (win.engine.test.key, win.engine.target)
    win.engine.rec = object()                                                   # "a trial is being recorded"
    live.go_to_position(row_of(live, "knee_flexion", 90))
    assert (win.engine.test.key, win.engine.target) == before
    assert "Stop recording" in live.instruction.text()
    assert live.pos_combo.currentIndex() == row_of(live, *before)               # the dropdown snapped back
    live._rec_started = time.monotonic()
    try:
        live.refresh_status()
        assert not live.pos_combo.isEnabled() and not live.btn_pos_prev.isEnabled() and not live.btn_pos_next.isEnabled()
        assert not live.btn_redo_dots.isEnabled()
    finally:
        live._rec_started = None
        win.engine.rec = None


def test_same_landmarks_keep_the_marked_dots_when_the_position_changes(win):
    live = win.live
    mark_all(win)
    live.go_to_position(row_of(live, "knee_extension", 70))                    # same test, other target
    assert set(win.engine.dot_status().values()) - {"off"} == {"ok"}
    assert all(not c.isHidden() and c.styleSheet() == DOT_CHIP_STYLE["ok"] for n, c in live.chips.items() if n in win.engine.active_dots)


# ------------------------------------------------------------------------------------------------ the dot buttons
def test_there_is_one_button_per_landmark_and_it_follows_the_test(win):
    live = win.live
    names = list(win.engine.test.all_dots)
    assert list(live.chips) == list(live.lm_rows) == names
    assert live.chips[names[0]].text().startswith("1 ") and live.chips[names[4]].text().startswith("5 ")
    off = [n for n in names if n not in win.engine.active_dots]
    assert off and all(live.chips[n].isHidden() for n in off)                    # strap and anchor are not used: no button
    live._pick(live.test_combo, "hip_abduction")                                 # another set of landmarks
    assert list(live.chips) == list(live.lm_rows) == list(win.engine.test.all_dots)
    assert max(i // DOT_CHIP_COLUMNS for i in range(len(live.chips))) <= 1       # two rows at most


def test_the_buttons_show_marked_lost_and_not_marked(win):
    live = win.live
    names = list(win.engine.active_dots)
    live.update_dot_rows()
    assert all(live.chips[n].styleSheet() == DOT_CHIP_STYLE["unmarked"] for n in names)
    mark_all(win)
    assert all(live.chips[n].styleSheet() == DOT_CHIP_STYLE["ok"] for n in names)
    gone = names[2]
    for i in range(1, 25):
        win.engine.process(render(PTS, seed=i, hide=[gone]), i / 30)
    live.update_dot_rows()
    assert live.chips[gone].styleSheet() == DOT_CHIP_STYLE["lost"] and "LOST" in live.chips[gone].toolTip()
    assert live.chips[names[0]].styleSheet() == DOT_CHIP_STYLE["ok"]


def test_a_button_does_what_the_landmark_row_button_does(win):
    live = win.live
    mark_all(win)
    a, c = list(win.engine.active_dots)[0], list(win.engine.active_dots)[2]
    before_c = np.array([win.engine.trackers.trackers[c].x, win.engine.trackers.trackers[c].y])
    live.chips[a].click()
    assert live._arm == a and live.chips[a].styleSheet() == DOT_CHIP_STYLE["armed"] and live.lm_buttons[a].text() == "Cancel"
    live.on_click(*(PTS[a] + 3))
    assert live._arm is None and live.chips[a].styleSheet() == DOT_CHIP_STYLE["ok"]
    after_c = np.array([win.engine.trackers.trackers[c].x, win.engine.trackers.trackers[c].y])
    assert np.allclose(before_c, after_c)                                         # only that dot changed
    live.chips[a].click()
    live.chips[a].click()                                                         # pressed again: cancelled
    assert live._arm is None


def test_the_buttons_are_off_during_a_break_and_without_a_camera(win):
    live = win.live
    mark_all(win)
    a = list(win.engine.active_dots)[0]
    assert live.chips[a].isEnabled()
    win.engine.pause()
    live.update_dot_rows()
    assert not live.chips[a].isEnabled()
    win.engine.resume()
    live.worker = None
    live.update_dot_rows()
    assert not live.chips[a].isEnabled()


# ------------------------------------------------------------------------------------------------ redo all dots
def test_redo_all_starts_a_pass_that_changes_nothing_until_the_first_click(win):
    """What a pass does dot by dot is in test_mark_again.py; here: the bar's button starts and stops it."""
    live = win.live
    mark_all(win)
    assert win.engine.lock_position()
    live.btn_redo_dots.click()
    assert live._pass == list(win.engine.active_dots) and live._arm == live._pass[0] and live.btn_redo_dots.text() == "Stop (D)"
    assert win.engine.locked and set(win.engine.dot_status().values()) - {"off"} == {"ok"}          # nothing changed yet
    live.btn_redo_dots.click()
    assert not live._pass and live._arm is None and live.btn_redo_dots.text() == "Redo all (D)" and win.engine.locked


# ------------------------------------------------------------------------------------------------ keys
def test_each_new_key_has_exactly_one_shortcut(win):
    live = win.live
    for k in (Qt.Key_D, Qt.Key_N, Qt.Key_B, Qt.Key_F2, Qt.Key_F3, Qt.Key_F4, *(Qt.Key_0 + i for i in range(1, 10))):
        assert len(shortcuts_for(live, k)) == 1, k


def test_d_and_f4_redo_the_dots_n_and_b_f3_and_f2_move_the_position(win):
    live = win.live
    win.show()
    live.setFocus()
    QApplication.processEvents()
    mark_all(win)
    QTest.keyClick(live, Qt.Key_D)
    assert live._pass                                                              # D starts it ...
    QTest.keyClick(live, Qt.Key_F4)
    assert not live._pass                                                          # ... F4 is the same key: it stops it
    here = live._plan_current
    QTest.keyClick(live, Qt.Key_N)
    assert live._plan_current == here + 1
    QTest.keyClick(live, Qt.Key_F3)
    assert live._plan_current == here + 2
    QTest.keyClick(live, Qt.Key_B)
    QTest.keyClick(live, Qt.Key_F2)
    assert live._plan_current == here
    win.hide()


def test_a_number_key_redoes_that_dot_and_only_waits_for_a_dot_that_is_in_use(win):
    live = win.live
    win.show()
    live.setFocus()
    QApplication.processEvents()
    mark_all(win)
    names = list(win.engine.test.all_dots)
    QTest.keyClick(live, Qt.Key_3)
    assert live._arm == names[2]
    QTest.keyClick(live, Qt.Key_3)                                                 # again: cancelled, like the button
    assert live._arm is None
    QTest.keyClick(live, Qt.Key_8)                                                 # the strap dot is not used
    assert live._arm is None
    win.hide()


def test_the_letters_are_typed_into_the_notes_box_and_the_f_keys_still_work(win):
    live = win.live
    win.show()
    QApplication.processEvents()
    mark_all(win)
    live.notes.setFocus()
    QTest.keyClicks(live.notes, "bad dn 3")                                        # contains b, d, n and a digit
    assert live.notes.text() == "bad dn 3"
    assert set(win.engine.dot_status().values()) - {"off"} == {"ok"} and live._plan_current == 0 and live._arm is None and not live._pass
    QTest.keyClick(live.notes, Qt.Key_F3)                                          # F-keys work from inside the box
    assert live._plan_current == 1
    QTest.keyClick(live.notes, Qt.Key_F4)
    assert live._pass                                                              # Redo all started
    QTest.keyClick(live.notes, Qt.Key_F4)
    assert not live._pass
    assert "F4" in live.key_hint.text() and "F2/F3" in live.key_hint.text()
    win.hide()
