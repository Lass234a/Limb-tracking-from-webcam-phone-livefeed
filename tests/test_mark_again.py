"""Redo all (0.18.1): every dot is marked again, one after the other in the order asked, while the old dots keep tracking until each is
replaced. Nothing changes before the first click; the lock is released at the first dot marked again; a dot can be skipped (its own
button) or taken out of turn (another dot's button); a recording, a break or a stopped camera ends the pass without touching any dot."""
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.gui import DOT_CHIP_PENDING, DOT_CHIP_STYLE, MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)
MOVED = leg_points(hip_flex=80, knee_flex=40, ankle_df=10)               # the participant has been put in another position


class RunningCamera:
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
    for n in w.engine.active_dots:
        assert w.engine.mark_dot(n, *(PTS[n] + 2))
    w.live.update_dot_rows()                                                 # (the buttons know now that a camera is running)
    yield w
    w.engine.rec = None
    w.live.worker = None
    w.close()


def pos(win, name):
    t = win.engine.trackers.trackers[name]
    return np.array([t.x, t.y])


def move_the_participant(win, frames=3):
    """Pictures of the participant in another position: the old trackers lose or follow their dots."""
    for i in range(1, frames + 1):
        win.engine.process(render(MOVED, seed=i), i / 30)


# ------------------------------------------------------------------------------------------------ start, stop, nothing changes
def test_it_asks_for_the_first_dot_and_changes_nothing_yet(win):
    live, e = win.live, win.engine
    names = list(e.active_dots)
    assert e.lock_position()
    before = {n: pos(win, n) for n in names}
    live.btn_redo_dots.click()
    assert live._pass == names and live._arm == names[0] and live._pass_total == len(names)
    assert f"1 of {len(names)}" in live.instruction.text() and "skips it, D stops" in live.instruction.text()
    assert e.locked and live.btn_redo_dots.text() == "Stop (D)"                   # the lock stays until a dot has really been marked again
    assert all(np.allclose(before[n], pos(win, n)) for n in names)
    assert live.chips[names[0]].styleSheet() == DOT_CHIP_STYLE["armed"]
    assert live.chips[names[1]].styleSheet() == DOT_CHIP_PENDING["ok"] and "dashed" in DOT_CHIP_PENDING["ok"]    # still to do
    assert "still to be marked again" in live.chips[names[1]].toolTip()
    assert live.lm_buttons[names[0]].text() == "Skip"


def test_pressing_it_again_stops_and_nothing_was_lost(win):
    live, e = win.live, win.engine
    assert e.lock_position()
    before = {n: pos(win, n) for n in e.active_dots}
    live.btn_redo_dots.click()
    live.btn_redo_dots.click()
    assert not live._pass and live._arm is None and live.btn_redo_dots.text() == "Redo all (D)"
    assert e.locked and all(np.allclose(before[n], pos(win, n)) for n in e.active_dots)
    assert "keep their old marks" in live.instruction.text()
    assert all(live.chips[n].styleSheet() == DOT_CHIP_STYLE["ok"] for n in e.active_dots)


def test_without_a_camera_or_during_a_break_it_does_not_start(win):
    live = win.live
    win.engine.pause()
    live.btn_redo_dots.click()
    assert not live._pass and "break" in live.instruction.text()
    win.engine.resume()
    live.worker = None
    live.btn_redo_dots.click()
    assert not live._pass and "camera" in live.instruction.text().lower()


# ------------------------------------------------------------------------------------------------ a whole pass
def test_a_pass_marks_every_dot_at_its_new_place_in_order_and_releases_the_lock(win):
    live, e = win.live, win.engine
    names = list(e.active_dots)
    live.btn_lock.click()                                                           # the real Lock button, not only the engine
    assert e.locked and live.btn_lock.isChecked()
    move_the_participant(win)
    live.btn_redo_dots.click()
    for i, n in enumerate(names):
        assert live._arm == n and live._pass == names[i:]
        assert e.locked == (i == 0)                                                 # still locked until the first dot is marked again
        live.on_click(*(MOVED[n] + 2))
        assert not e.locked
    assert not e.locked and not live.btn_lock.isChecked()
    assert live._arm is None and not live._pass and live.btn_redo_dots.text() == "Redo all (D)"
    assert all(np.allclose(pos(win, n), MOVED[n], atol=2.5) for n in names)
    assert "All" in live.instruction.text() and "lock" in live.instruction.text().lower()
    assert set(e.dot_status().values()) - {"off"} == {"ok"}


def test_the_lock_is_kept_until_the_first_dot_has_really_been_marked_again(win):
    live, e = win.live, win.engine
    names = list(e.active_dots)
    assert e.lock_position()
    live.btn_redo_dots.click()
    live.on_click(5.0, 5.0)                                                         # a click that finds no dot
    assert e.locked and live._arm == names[0] and "Still waiting" in live.instruction.text() and "skip it" in live.instruction.text()
    live.on_click(*(PTS[names[0]] + 2))
    assert not e.locked


def test_what_was_worked_out_from_the_old_dots_is_forgotten_at_the_first_dot(win):
    live, e = win.live, win.engine
    for i in range(1, 8):
        e.process(render(PTS, seed=i), i / 30)
    assert e._filters and e._ref_avg
    live.btn_redo_dots.click()
    live.on_click(*(PTS[live._arm] + 2))
    assert not e._filters and not e._ref_avg and e.facing_detected is None


# ------------------------------------------------------------------------------------------------ the old marks do not get in the way
def test_an_old_mark_that_sits_on_a_sticker_does_not_stop_it_being_marked_in_a_pass(win):
    live, e = win.live, win.engine
    a, b = list(e.active_dots)[:2]
    assert not e.mark_dot(a, *(PTS[b] + 2)) and "already marked" in (e.mark_problem or "")        # the usual guard, outside a pass
    assert e.mark_dot(a, *(PTS[b] + 2), replacing=[b])                                           # b is about to be marked again
    live.btn_redo_dots.click()
    assert e.mark_dot(a, *(PTS[a] + 2))                                                          # put a back for the pass
    t = e.trackers.trackers[b]
    t.x, t.y = PTS[a][0], PTS[a][1]                                                              # b has slipped onto a's sticker
    live.on_click(*(PTS[a] + 2))                                                                 # a is marked again: not refused
    assert live._pass[0] == b and np.allclose(pos(win, a), PTS[a], atol=2.5)
    live.on_click(*(PTS[b] + 2))                                                                 # now b, at its own sticker
    assert np.allclose(pos(win, b), PTS[b], atol=2.5)


def test_a_wrong_click_in_a_pass_is_caught_when_the_other_dot_is_marked_and_can_be_fixed(win):
    live, e = win.live, win.engine
    a, b = list(e.active_dots)[:2]
    live.btn_redo_dots.click()
    live.on_click(*(PTS[b] + 2))                                                    # a was clicked on b's sticker (b is still to do: allowed)
    assert live._arm == b
    live.on_click(*(PTS[b] + 2))                                                    # now b's own sticker: a already holds it
    assert live._arm == b and "already marked" in live.instruction.text()
    live.chips[a].click()                                                           # jump back to a ...
    assert live._arm == a
    live.on_click(*(PTS[a] + 2))                                                    # ... mark it where it belongs
    assert live._arm == b
    live.on_click(*(PTS[b] + 2))                                                    # and b is accepted now
    assert np.allclose(pos(win, a), PTS[a], atol=2.5) and np.allclose(pos(win, b), PTS[b], atol=2.5)


# ------------------------------------------------------------------------------------------------ skip and jump
def test_the_button_of_the_dot_that_is_asked_for_skips_it_and_it_keeps_its_mark(win):
    live, e = win.live, win.engine
    names = list(e.active_dots)
    before = pos(win, names[0])
    live.btn_redo_dots.click()
    live.chips[names[0]].click()                                                    # the chip, or the row's button: the same
    assert live._arm == names[1] and live._pass == names[1:] and "Skipped" in live.instruction.text()
    assert np.allclose(pos(win, names[0]), before) and e.dot_status()[names[0]] == "ok"
    live.lm_buttons[names[1]].click()                                               # and the button in the Dots section
    assert live._arm == names[2]


def test_skipping_every_dot_ends_the_pass_and_keeps_the_lock(win):
    live, e = win.live, win.engine
    assert e.lock_position()
    live.btn_redo_dots.click()
    for _ in range(len(e.active_dots)):
        live.chips[live._arm].click()
    assert not live._pass and live._arm is None and e.locked
    assert "handled" in live.instruction.text()


def test_another_dots_button_takes_that_dot_next_and_the_pass_goes_on_from_the_first_to_do(win):
    live, e = win.live, win.engine
    names = list(e.active_dots)
    live.btn_redo_dots.click()
    live.chips[names[3]].click()
    assert live._arm == names[3] and live._pass == names                              # nothing is done yet; the order of the rest is kept
    live.on_click(*(PTS[names[3]] + 2))
    assert names[3] not in live._pass and live._arm == names[0] and live._pass == [n for n in names if n != names[3]]
    live.on_click(*(PTS[names[0]] + 2))
    assert live._arm == names[1]


def test_a_number_key_works_in_a_pass_like_the_button(win):
    live = win.live
    names = list(win.engine.active_dots)
    live.btn_redo_dots.click()
    live.redo_dot_number(1)                                                           # the asked-for dot: skipped
    assert live._arm == names[1]
    live.redo_dot_number(4)
    assert live._arm == names[3]


# ------------------------------------------------------------------------------------------------ other tests, other dots
def test_a_pass_with_the_strap_and_anchor_dots_on_asks_for_all_nine(win):
    live = win.live
    for n in win.engine.test.all_dots:
        live.lm_checks[n].setChecked(True)
    live.update_dot_rows()
    live.btn_redo_dots.click()
    assert live._pass == list(win.engine.test.all_dots) and len(live._pass) == 9 and live._arm == live._pass[0]
    assert f"1 of 9" in live.instruction.text()
    assert all(not c.isHidden() for c in live.chips.values())
    live.btn_redo_dots.click()
    assert not live._pass


def test_a_pass_in_a_frontal_test_asks_for_that_tests_dots(win):
    live = win.live
    live._pick(live.test_combo, "hip_abduction")
    names = list(win.engine.active_dots)
    assert names and set(names) != set(TESTS["knee_extension"].dots)                  # other landmarks than the leg test
    live.btn_redo_dots.click()
    assert live._pass == names and live._arm == names[0]
    assert list(live.chips) == list(win.engine.test.all_dots) == list(live.lm_rows)


# ------------------------------------------------------------------------------------------------ the ruler and the dots share the next click
def test_starting_a_pass_takes_over_from_the_ruler_marking(win):
    live = win.live
    live.start_ruler_marking()
    assert live._ruler_marking == 0
    live.btn_redo_dots.click()
    assert live._ruler_marking is None and live._pass                                  # the next click is for the dot, not the ruler
    first = live._arm
    live.on_click(*(PTS[first] + 2))
    assert first not in live._pass and win.engine.scale.px_per_cm is None             # (no ruler dot was marked by it)


def test_arming_one_dot_takes_over_from_the_ruler_marking(win):
    live = win.live
    a = list(win.engine.active_dots)[0]
    live.start_ruler_marking()
    live.chips[a].click()
    assert live._ruler_marking is None and live._arm == a


def test_starting_the_ruler_marking_ends_a_pass_and_a_waiting_dot(win):
    live = win.live
    a = list(win.engine.active_dots)[0]
    live.btn_redo_dots.click()
    assert live._pass
    live.start_ruler_marking()
    assert not live._pass and live._arm is None and live._ruler_marking == 0 and live.btn_redo_dots.text() == "Redo all (D)"
    live.chips[a].click()
    live.start_ruler_marking()
    assert live._arm is None and live._ruler_marking == 0


# ------------------------------------------------------------------------------------------------ the engine's part
def test_the_engine_releases_the_lock_only_and_refuses_during_a_recording(win):
    e = win.engine
    names = list(e.active_dots)
    assert e.lock_position()
    before = {n: pos(win, n) for n in names}
    e.start_remarking()
    assert not e.locked and set(e.dot_status().values()) - {"off"} == {"ok"}          # no dot was forgotten
    assert all(np.allclose(before[n], pos(win, n)) for n in names)
    e.rec = object()
    with pytest.raises(RuntimeError, match="Stop recording"):
        e.start_remarking()


def test_replacing_is_only_what_the_caller_names(win):
    e = win.engine
    a, b, c = list(e.active_dots)[:3]
    assert not e.mark_dot(a, *(PTS[b] + 2), replacing=[c])                            # b is not on the list: still refused
    assert "already marked" in (e.mark_problem or "")
    assert e.mark_dot(a, *(PTS[b] + 2), replacing=[b, c])


# ------------------------------------------------------------------------------------------------ what ends a pass
def test_a_break_a_recording_or_a_stopped_camera_ends_the_pass_and_touches_no_dot(win):
    live, e = win.live, win.engine
    names = list(e.active_dots)
    before = {n: pos(win, n) for n in names}
    for how in ("break", "recording", "camera"):
        live.btn_redo_dots.click()
        assert live._pass
        if how == "break":
            e.pause()
        elif how == "recording":
            live._rec_started = time.monotonic()
        else:
            live.worker = None
        live.update_dot_rows()
        assert not live._pass and live._arm is None, how
        e.resume()
        live._rec_started = None
        live.worker = RunningCamera()
        live.update_dot_rows()
    assert all(np.allclose(before[n], pos(win, n)) for n in names)


def test_a_click_during_a_recording_does_nothing_in_a_pass(win):
    live, e = win.live, win.engine
    names = list(e.active_dots)
    before = pos(win, names[0])
    live.btn_redo_dots.click()
    e.rec = object()
    live.on_click(*(MOVED[names[0]] + 2))
    assert np.allclose(pos(win, names[0]), before) and "recording" in live.instruction.text().lower()


def test_another_test_clear_all_or_a_new_colour_ends_the_pass(win):
    live = win.live
    live.btn_redo_dots.click()
    live._pick(live.test_combo, "hip_abduction")
    assert not live._pass and live._arm is None
    live.btn_redo_dots.click()
    assert live._pass
    live.remark()                                                                    # 'Clear all dots and start again' in the Dots section
    assert not live._pass and live._arm is None and live.btn_redo_dots.text() == "Redo all (D)"


def test_a_dot_switched_off_in_the_middle_is_dropped_and_the_pass_ends_when_none_is_left(win):
    live, e = win.live, win.engine
    names = list(e.active_dots)
    live.btn_redo_dots.click()
    live.lm_checks[names[1]].setChecked(False)
    live.update_dot_rows()
    assert names[1] not in live._pass and live._arm == names[0]
    live.lm_checks[names[0]].setChecked(False)                                       # the one waiting for a click
    live.update_dot_rows()
    assert live._arm == names[2] and live._pass[0] == names[2]
    for n in list(live._pass):
        live.lm_checks[n].setChecked(False)
    live.update_dot_rows()
    assert not live._pass and live._arm is None


def test_the_pass_does_not_start_when_no_landmark_is_in_use(win):
    live = win.live
    for n in list(win.engine.active_dots):
        live.lm_checks[n].setChecked(False)
    live.btn_redo_dots.click()
    assert not live._pass and "No landmark" in live.instruction.text()


def test_a_single_dot_outside_a_pass_still_keeps_the_lock_and_cancels_with_its_button(win):
    live, e = win.live, win.engine
    a = list(e.active_dots)[1]
    assert e.lock_position()
    live.chips[a].click()
    assert live._arm == a and live.lm_buttons[a].text() == "Cancel" and not live._pass
    live.on_click(*(PTS[a] + 3))
    assert e.locked and live._arm is None                                            # as before 0.18.1
    live.chips[a].click()
    live.chips[a].click()
    assert live._arm is None
