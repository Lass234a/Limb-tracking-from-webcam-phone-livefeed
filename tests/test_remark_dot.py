"""Redoing ONE dot from the landmark list: each landmark has its state and a Mark / Re-mark button; pressing it makes the next click on the
picture mark that dot (and nothing else), and it keeps waiting until the dot has been found."""
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)


class RunningCamera:
    is_file = False
    paused = False
    at_end = False
    source = 0
    camera = None
    status_note = ""

    def stop(self):
        pass

    def wait(self, ms=0):
        return True

    def isRunning(self):
        return True


@pytest.fixture()
def win(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(TESTS, tmp_path)
    w.live.worker = RunningCamera()
    w.engine.process(render(PTS, seed=0), 0.0)
    yield w
    w.live.worker = None
    w.close()


def status(win):
    win.live.update_dot_rows()
    return {n: win.live.lm_status[n].text() for n in win.live.lm_rows}


def position(win, name):
    t = win.engine.trackers.trackers[name]
    return np.array([t.x, t.y])


def test_every_landmark_has_a_row_with_its_state_and_a_button(win):
    live = win.live
    names = list(win.engine.test.all_dots)
    assert list(live.lm_rows) == names and set(live.lm_checks) == set(names) == set(live.lm_buttons) == set(live.lm_status)
    assert live.accordion.sections["dots"].body.isAncestorOf(live.lm_rows[names[0]])             # in the Dots section, not in 'advanced'
    st = status(win)
    assert [st[n] for n in win.engine.active_dots] == ["not marked"] * len(win.engine.active_dots)
    assert all(live.lm_buttons[n].text() == "Mark" for n in win.engine.active_dots)
    off = [n for n in names if n not in win.engine.active_dots]
    assert off and all(st[n] == "not used" and not live.lm_buttons[n].isEnabled() for n in off)    # the strap and anchor dots are off
    assert "full" not in live.lm_checks[names[0]].text() and "(" not in live.lm_checks[names[0]].text()      # short name; the full one is the tooltip
    assert "halfway between" in live.lm_checks["iliac_crest"].toolTip()


def test_marked_and_lost_dots_say_so(win):
    live = win.live
    names = list(win.engine.active_dots)
    for n in names:
        assert win.engine.mark_dot(n, *(PTS[n] + 2))
    assert set(status(win).values()) - {"not used"} == {"marked"}
    assert all(live.lm_buttons[n].text() == "Re-mark" for n in names)
    gone = names[2]
    for i in range(1, 25):
        win.engine.process(render(PTS, seed=i, hide=[gone]), i / 30)
    assert status(win)[gone] == "LOST" and status(win)[names[0]] == "marked"
    assert win.engine.dot_status()[gone] == "lost"


def test_one_wrongly_marked_dot_is_redone_without_touching_the_others(win):
    live = win.live
    a, b, c = list(win.engine.active_dots)[:3]
    assert win.engine.mark_dot(a, *(PTS[b] + 2))                       # the first dot was clicked on the second one's sticker
    assert not win.engine.mark_dot(b, *(PTS[b] + 2))                   # ... so the second is refused: "already marked"
    assert "already marked" in (win.engine.mark_problem or "")
    assert win.engine.mark_dot(c, *(PTS[c] + 2))
    before_c = position(win, c)
    live.arm_dot(a)                                                    # its own button
    assert live._arm == a and live.lm_buttons[a].text() == "Cancel" and live.lm_buttons[a].isChecked()
    assert "mark it again" in live._instruction_text()
    live.on_click(*(PTS[a] + 2))                                       # the real sticker
    assert live._arm is None
    assert np.allclose(position(win, a), PTS[a], atol=2.0)             # now where it belongs
    assert np.allclose(position(win, c), before_c)                     # the others were not touched
    assert "The other dots were not touched" in live.instruction.text()
    assert win.engine.mark_dot(b, *(PTS[b] + 2))                       # and the second can now be marked
    assert live.lm_buttons[a].text() == "Re-mark" and not live.lm_buttons[a].isChecked()


def test_a_click_that_misses_keeps_waiting_until_the_dot_is_found(win):
    live = win.live
    a = list(win.engine.active_dots)[0]
    live.arm_dot(a)
    live.on_click(5.0, 5.0)                                            # nowhere near a dot
    assert live._arm == a and "Still waiting" in live.instruction.text()
    assert live.lm_buttons[a].text() == "Cancel"
    live.on_click(*(PTS[a] + 2))
    assert live._arm is None and status(win)[a] == "marked"


def test_pressing_the_button_again_cancels(win):
    live = win.live
    a = list(win.engine.active_dots)[0]
    live.arm_dot(a)
    live.arm_dot(a)
    assert live._arm is None and live.lm_buttons[a].text() == "Mark" and not live.lm_buttons[a].isChecked()
    live.on_click(*(PTS[a] + 2))                                       # a plain click still marks the next dot in order
    assert status(win)[a] == "marked"


def test_only_one_dot_waits_at_a_time(win):
    live = win.live
    a, b = list(win.engine.active_dots)[:2]
    live.arm_dot(a)
    live.arm_dot(b)
    assert live._arm == b and not live.lm_buttons[a].isChecked() and live.lm_buttons[b].isChecked()


def test_the_lock_is_kept_when_one_dot_is_marked_again(win):
    live = win.live
    for n in win.engine.active_dots:
        assert win.engine.mark_dot(n, *(PTS[n] + 2))
    assert win.engine.lock_position()
    a = list(win.engine.active_dots)[1]
    live.arm_dot(a)
    live.on_click(*(PTS[a] + 3))
    assert win.engine.locked                                           # the reference pose is still the one that was locked: drift stays visible


def test_nothing_waits_without_a_camera_or_while_recording_and_a_change_of_colour_cancels(win):
    live = win.live
    a = list(win.engine.active_dots)[0]
    live.worker = None
    live.arm_dot(a)
    assert live._arm is None and "camera" in live.instruction.text().lower()
    live.worker = RunningCamera()
    live.arm_dot(a)
    assert live._arm == a
    live._pick(live.kind_combo, "black")                               # the colour changed: every dot is cleared, so is the wait
    assert live._arm is None
    live.arm_dot(a)
    live.remark()                                                      # 'Clear all dots and start again'
    assert live._arm is None
    assert not live.lm_buttons[a].isChecked()


def test_switching_a_landmark_off_while_it_waits_cancels_the_wait(win):
    live = win.live
    a = list(win.engine.active_dots)[0]
    live.arm_dot(a)
    live.lm_checks[a].setChecked(False)
    live.update_dot_rows()
    assert live._arm is None and live.lm_status[a].text() == "not used"
