"""The step chips, the one-line summaries and the panel moving on by itself: what the Live tab tells the operator to do next.

The order of a session (limbtrack/gui.py STEPS, since 0.24.0): camera, force, markers, participant, position, privacy, ruler, dots, lock, record.
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox, QPushButton  # noqa: E402

from limbtrack.force import ForceCalibration  # noqa: E402
from limbtrack.gui import STEP_HINT, STEPS, MainWindow  # noqa: E402
from limbtrack.panels import ElidedButton  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)


@pytest.fixture()
def win(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(TESTS, tmp_path)
    yield w
    w.live.worker = None
    w.close()


class RunningCamera:
    """Stands in for a running camera (the window only asks whether one is running)."""
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


def states(live):
    return {k: v[0] for k, v in live.strip.states.items()}


def mark_all(win):
    win.engine.process(render(PTS, seed=0), 0.0)
    for n in win.engine.active_dots:
        assert win.engine.mark_dot(n, *(PTS[n] + 2))


def next_step(live):
    return next((k for k, s in states(live).items() if s == "next"), None)


def test_the_steps_are_in_the_order_of_a_session_equipment_first_then_the_participant(win):
    keys = [k for k, _ in STEPS]
    assert keys == ["camera", "force", "markers", "participant", "position", "privacy", "ruler", "dots", "lock", "record"]
    assert max(keys.index(k) for k in ("camera", "force", "markers")) < keys.index("participant")     # done before the participant arrives
    assert keys.index("participant") < keys.index("position")                      # the plan counts THIS participant's trials
    assert keys.index("position") < keys.index("privacy")                          # privacy is looked at again at every position change
    assert keys.index("ruler") < keys.index("dots")                                # 0.24.0: the ruler first (its colour is the one set under Dots)
    assert set(STEP_HINT) == set(keys)
    numbers = {k: s.number for k, s in win.live.accordion.sections.items() if s.number}
    assert [numbers[k] for k in ("camera", "force", "markers", "participant", "position", "privacy", "ruler", "dots", "record")] == \
        [str(i) for i in range(1, 10)]                                              # the numbers in the panel follow the same order


def test_at_the_start_the_camera_is_the_next_step_and_nothing_that_needs_the_picture_is_asked_for(win):
    live = win.live
    live.update_steps()
    assert states(live) == {"camera": "next", "force": "todo", "markers": "todo", "participant": "todo", "position": "done",
                            "privacy": "off", "ruler": "off", "dots": "off", "lock": "off", "record": "off"}
    assert live.accordion.current() == "camera"                                   # the first thing to do is in front of you


def test_every_step_in_turn_becomes_the_next_one(win):
    live = win.live
    live.update_steps()
    live.worker = RunningCamera()
    live.update_steps()
    assert states(live)["camera"] == "done" and next_step(live) == "force"
    live.chk_force_skip.setChecked(True)
    live.update_steps()
    assert states(live)["force"] == "done" and "No force" in live.strip.chips["force"].text() and next_step(live) == "markers"
    live._beeps_tested = True                                                      # (the beeps are on, so they have to have been tried)
    live.update_steps()
    assert states(live)["markers"] == "done" and next_step(live) == "participant"
    live.participant.setText("P01")
    live.update_steps()
    assert states(live)["participant"] == "next" and live.strip.chips["participant"].text().endswith("P01: leg?")   # 0.20.0: the leg too
    assert "tested (dominant) leg of P01" in live._hint("participant")
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live.update_steps()
    assert states(live)["participant"] == "done" and "P01" in live.strip.chips["participant"].text() and next_step(live) == "privacy"
    live.chk_priv_ok.setChecked(True)                                              # nothing in the picture identifies anyone
    live.update_steps()
    assert states(live)["privacy"] == "done" and next_step(live) == "ruler" and "0/5" in live.strip.chips["dots"].text()   # (no foot dots here)
    live.chk_ruler_skip.setChecked(True)
    live.update_steps()
    assert states(live)["ruler"] == "done" and "No ruler" in live.strip.chips["ruler"].text() and next_step(live) == "dots"
    mark_all(win)
    live.update_steps()
    assert states(live)["dots"] == "done" and next_step(live) == "lock"
    assert win.engine.lock_position()
    live.update_steps()
    assert states(live)["lock"] == "done" and next_step(live) == "record" and states(live)["record"] == "next"
    assert live.strip.chips["record"].text().endswith("Record")


def test_the_panel_moves_on_by_itself_when_a_step_is_finished_in_its_own_section(win):
    live = win.live
    live.update_steps()
    assert live.accordion.current() == "camera"
    live.worker = RunningCamera()
    live.update_steps()
    assert live.accordion.current() == "force"                                     # the camera runs: on to the next thing
    live.chk_force_skip.setChecked(True)
    live.update_steps()
    assert live.accordion.current() == "markers"
    live._beeps_tested = True
    live.update_steps()
    assert live.accordion.current() == "markers"                                   # the markers are tried out: the panel does not move on by itself
    live.open_step("participant")
    live.participant.setText("P01")
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live.update_steps()
    assert live.accordion.current() == "participant"                               # typing a code never takes the box away from you
    live.participant.editingFinished.emit()                                        # Tab or a click into Notes: the box stays
    assert live.accordion.current() == "participant"
    live.participant.returnPressed.emit()                                          # Enter
    assert live.accordion.current() == "privacy"
    live.chk_priv_ok.setChecked(True)
    live.update_steps()
    assert live.accordion.current() == "ruler"
    live.chk_ruler_skip.setChecked(True)
    live.update_steps()
    assert live.accordion.current() == "dots"
    mark_all(win)
    live.update_steps()
    assert live.accordion.current() == "dots"                                      # Lock is the next step, and lives with the dots
    win.engine.lock_position()
    live.update_steps()
    assert live.accordion.current() == "record"


def test_a_section_the_operator_opened_is_not_closed_behind_their_back(win):
    live = win.live
    live.update_steps()
    live.open_step("force")                                                        # they went to look at the force box
    live.worker = RunningCamera()
    live.update_steps()
    assert live.accordion.current() == "force"                                     # the camera section was not the open one: leave it


def test_clicking_a_chip_opens_its_section(win):
    live = win.live
    live.update_steps()
    live.strip.chips["force"].click()
    assert live.accordion.current() == "force"
    live.strip.chips["lock"].click()
    assert live.accordion.current() == "dots"                                      # Lock lives in the dots section
    live.strip.chips["markers"].click()
    assert live.accordion.current() == "markers"
    live.strip.chips["privacy"].click()
    assert live.accordion.current() == "privacy"
    live.strip.chips["ruler"].click()
    assert live.accordion.current() == "ruler"
    live.strip.chips["camera"].click()
    assert live.accordion.current() == "camera"


def test_the_privacy_chip_wants_a_crop_a_mask_or_a_look(win):
    live = win.live
    live.worker = RunningCamera()
    mark_all(win)
    live.participant.setText("P01")
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live.update_steps()
    assert states(live)["privacy"] == "todo" and "not set" in live.accordion.sections["privacy"].summary       # (the force comes first now)
    ok, _ = win.engine.set_privacy_rect("mask", 0, 0, 200, 200)                    # a black mask over a face
    assert ok
    live.update_steps()
    assert states(live)["privacy"] == "done" and "mask" in live.accordion.sections["privacy"].summary
    win.engine.clear_privacy()
    live.update_steps()
    assert states(live)["privacy"] != "done"
    live.chk_priv_ok.setChecked(True)
    live.update_steps()
    assert states(live)["privacy"] == "done" and "checked" in live.accordion.sections["privacy"].summary


def test_the_ruler_chip_asks_every_session_and_follows_the_scale(win):
    live = win.live
    live.update_steps()
    assert states(live)["ruler"] == "off"                                          # no picture, no ruler
    live.worker = RunningCamera()
    mark_all(win)
    live.update_steps()
    assert states(live)["ruler"] in ("todo", "next") and "not set" in live.accordion.sections["ruler"].summary
    win.engine.scale.ref_px = 120.0                                                # both ruler dots marked
    live.update_steps()
    assert states(live)["ruler"] == "done" and "px/cm" in live.accordion.sections["ruler"].summary
    win.engine.scale.ref_px = None
    live.chk_ruler_skip.setChecked(True)
    live.update_steps()
    assert states(live)["ruler"] == "done" and "no ruler today" in live.accordion.sections["ruler"].summary


def test_the_force_chip_follows_the_choice_the_connection_and_the_zero(win):
    live = win.live
    live.update_steps()
    assert states(live)["force"] == "todo" and live.strip.chips["force"].text() == "Force"      # asked for: not silently 'off'
    live.chk_force_skip.setChecked(True)
    live.update_steps()
    assert states(live)["force"] == "done" and "No force" in live.strip.chips["force"].text()
    live.chk_force_skip.setChecked(False)
    live._pick(live.force_kind, "simulated")
    live.update_steps()
    assert states(live)["force"] == "todo" and "connect" in live.strip.chips["force"].text().lower()
    live.btn_force.click()
    live.force_source.calibration = ForceCalibration(n_per_v=526.95, zero_set=False)          # like the load cell: a slope, but no zero yet
    live.update_steps()
    assert "zero" in live.strip.chips["force"].text().lower() and "ZERO NEEDED" in live.accordion.sections["force"].summary
    live.force_source.calibration.set_zero(0.1)
    live.update_steps()
    assert states(live)["force"] == "done" and "zero set" in live.accordion.sections["force"].summary
    live.shutdown()


def test_the_markers_chip_wants_the_automatic_markers_decided_and_the_beeps_tried(win):
    live = win.live
    live.update_steps()
    assert live.chk_cues.isChecked() and states(live)["markers"] == "todo"          # beeps on, never tried
    assert "NOT tested" in live.accordion.sections["markers"].summary
    live.test_beeps()
    assert not live._beeps_tested                                                  # not before the test has run to the end
    live.report_beep_problem()                                                     # (what the timer calls 5 s later)
    assert live._beeps_tested
    live.update_steps()
    assert states(live)["markers"] == "done"                                       # (no force: nothing to automate)
    assert "beeps tested" in live.accordion.sections["markers"].summary
    live._pick(live.force_kind, "simulated")                                       # with the force connected the automatic markers are a choice
    live.btn_force.click()
    live.update_steps()
    assert states(live)["markers"] == "todo"
    live.chk_markers_hand.setChecked(True)
    live.update_steps()
    assert states(live)["markers"] == "done" and "by hand" in live.accordion.sections["markers"].summary
    live.chk_markers_hand.setChecked(False)
    live.chk_auto_mvic.setChecked(True)
    live.update_steps()
    assert states(live)["markers"] == "done" and "automatic" in live.accordion.sections["markers"].summary
    live.shutdown()


def test_a_beep_that_could_not_be_played_does_not_count_as_tried(win):
    live = win.live
    live.test_beeps()
    live.engine._cues.last_error = "Cannot play"
    live.report_beep_problem()
    assert not live._beeps_tested
    live.chk_cues.setChecked(False)                                                # no beeps at all: nothing to try
    live.update_steps()
    assert states(live)["markers"] == "done"


def test_the_instruction_follows_the_next_step(win):
    live = win.live
    live.refresh_status()
    assert live.instruction.text() == STEP_HINT["camera"]
    live.worker = RunningCamera()
    live.refresh_status()
    assert live.instruction.text() == STEP_HINT["force"]
    live.chk_force_skip.setChecked(True)
    live._beeps_tested = True
    live.refresh_status()
    assert live.instruction.text() == STEP_HINT["participant"]
    live.participant.setText("P01")
    live.refresh_status()
    assert "tested (dominant) leg of P01" in live.instruction.text()
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live.refresh_status()
    assert live.instruction.text() == STEP_HINT["privacy"]
    live.chk_priv_ok.setChecked(True)
    live.refresh_status()
    assert live.instruction.text() == STEP_HINT["ruler"]
    live.chk_ruler_skip.setChecked(True)
    live.refresh_status()
    assert "dot on the video" in live.instruction.text()                            # the dot to click next, by name
    mark_all(win)
    live.refresh_status()
    assert "lock" in live.instruction.text().lower()


def test_a_message_left_by_a_camera_that_ended_is_not_overwritten_by_the_hint(win):
    live = win.live
    live.refresh_status()
    live.instruction.setText("The camera stopped delivering frames.")
    live.refresh_status()
    assert live.instruction.text() == "The camera stopped delivering frames."


def test_closed_sections_say_what_they_are_set_to(win):
    live = win.live
    live.update_steps()
    heads = {k: s.summary for k, s in live.accordion.sections.items()}
    assert "no participant code" in heads["participant"] and "not started" in heads["camera"] and "not set" in heads["privacy"]
    assert "Knee extension" in heads["position"] and "20 deg" in heads["position"]
    assert "camera not started" in heads["dots"] and "not set" in heads["ruler"] and "not connected" in heads["force"]
    assert "not ready yet" in heads["record"]
    live.participant.setText("P12")
    live.update_steps()
    assert "P12" in live.accordion.sections["participant"].summary or live.accordion.sections["participant"].is_open()


def test_the_dot_colour_is_in_the_dots_section_and_named_in_its_summary(win):
    live = win.live
    dots = live.accordion.sections["dots"].body
    position = live.accordion.sections["position"].body
    assert dots.isAncestorOf(live.kind_combo) and not position.isAncestorOf(live.kind_combo)
    assert dots.isAncestorOf(live.lm_preset) and dots.isAncestorOf(live.btn_remark)
    live._pick(live.kind_combo, "black")
    live.update_steps()
    assert "black dots" in live.accordion.sections["dots"].summary
    assert "Auto can follow" not in live.kind_note.text()
    live._pick(live.kind_combo, "auto")
    assert "Auto can follow" in live.kind_note.text()                               # the one choice that can silently follow the tape


def test_the_buttons_used_in_every_trial_are_in_the_strip_and_the_old_clutter_is_folded_away(win):
    live = win.live
    for w in (live.btn_lock, live.btn_rec, live.btn_pause, live.instruction):
        assert w.parent() is not None and live.accordion.isAncestorOf(w) is False       # not inside a section: they never scroll away
    assert not live.btn_lock_cam.isVisibleTo(live)                               # the permanently switched-off exposure button is gone from view
    camera_body = live.accordion.sections["camera"].body
    assert live.btn_start.parent() is not None and camera_body.isAncestorOf(live.btn_start)
    assert not camera_body.isAncestorOf(live.stream_box) or not live.stream_box.isVisibleTo(live)
    assert live.accordion.sections["participant"].body.isAncestorOf(live.participant)
    assert live.accordion.sections["markers"].body.isAncestorOf(live.btn_cue_test)
    assert live.accordion.sections["markers"].body.isAncestorOf(live.chk_auto_mvic)
    assert live.accordion.sections["privacy"].body.isAncestorOf(live.btn_crop)
    assert live.accordion.sections["ruler"].body.isAncestorOf(live.btn_ruler)


def test_the_advanced_button_still_works_and_opens_its_section(win):
    live = win.live
    assert not live.adv_body.isVisibleTo(win) and live.btn_adv.text().startswith("Show")
    live.btn_adv.click()
    assert live.accordion.current() == "advanced" and live.adv_body.isVisibleTo(win)
    live.open_step("camera")
    live.open_step("advanced")
    assert live.btn_adv.isChecked()


def test_no_label_in_the_panel_is_too_long_for_it(win):
    """Qt neither wraps a check box nor cuts a button's text: one long label makes the whole panel wider than the window and the sides are
    clipped (seen on a real screen). Measuring widths here would depend on the fonts of the test machine, so the rule is on the length."""
    content = win.live.side_scroll.widget()
    for cb in content.findChildren(QCheckBox):
        assert len(cb.text()) <= 48, cb.text()
    for b in content.findChildren(QPushButton):
        if not isinstance(b, ElidedButton):
            assert len(b.text()) <= 42, b.text()


def test_a_header_is_cut_short_instead_of_widening_the_panel(win):
    live = win.live
    live.timer.stop()                                                              # (the status timer would write the real summary back)
    live.update_steps()
    win.resize(1400, 820)
    win.show()
    QApplication.processEvents()
    sec = live.accordion.sections["position"]
    sec.set_open(False)
    sec.set_summary("a summary that is far, far too long to fit into the header of a panel that is only four hundred pixels wide " * 2)
    QApplication.processEvents()
    head = sec.header
    assert head.minimumSizeHint().width() < 100                                    # it does not insist on its full text
    assert head.text().endswith("…") and len(head.text()) < len(head.fullText())    # cut short
    assert "far, far too long" in head.toolTip()                                    # the whole text is one hover away
    win.hide()


def test_the_short_sections_need_no_scrolling(win):
    live = win.live
    live.update_steps()
    win.resize(1400, 820)
    win.show()
    QApplication.processEvents()
    # The pinned bar (0.18.0) keeps its height for good, taken from the sections: the window gets that much more, so the test still says
    # that the sections themselves fit in the room they had before it.
    live.quick_bar.setVisible(False)
    QApplication.processEvents()
    room = live.side_scroll.viewport().height()
    live.quick_bar.setVisible(True)
    QApplication.processEvents()
    win.resize(1400, 820 + room - live.side_scroll.viewport().height())
    QApplication.processEvents()
    for key in ("participant", "camera"):
        live.accordion.open(key)
        QApplication.processEvents()
        QApplication.processEvents()
        # (the scroll range itself: a section that only just fits makes the panel flip between two widths when its scroll bar comes and
        # goes, and the section's size hint at the narrower width is then taller than what is really needed)
        assert live.side_scroll.verticalScrollBar().maximum() <= 40, key                                  # no real scrolling
    win.hide()


def test_the_camera_ticks_and_the_privacy_look_are_claims_about_today_and_are_asked_again(win, tmp_path):
    live = win.live
    live.update_steps()
    assert "0/6 ticked" in live.accordion.sections["checklist"].summary
    live.camera_checks["zoom_1x"].setChecked(True)
    live.camera_checks["focus_locked"].setChecked(True)
    live.update_steps()
    assert "2/6 ticked" in live.accordion.sections["checklist"].summary
    state = live.session_state()
    assert "camera_checklist" not in state and "camera_model" in state              # the model is remembered, the ticks are not
    live.chk_priv_ok.setChecked(True)
    live.on_mirror_toggled(True)                                                    # the picture is flipped: the look no longer counts
    assert not live.chk_priv_ok.isChecked()
    live.chk_priv_ok.setChecked(True)
    video = tmp_path / "tiny.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(10):
        out.write(render(PTS, seed=i))
    out.release()
    live.start_source(str(video))                                                   # a new picture: 'I looked' was about the old one
    try:
        assert not live.chk_priv_ok.isChecked()
    finally:
        live.stop_camera()


def test_a_participant_code_kept_from_last_time_counts_only_after_it_is_confirmed(win, monkeypatch):
    live = win.live
    answers = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: answers.append(a[2]) or QMessageBox.No)
    win.engine.tested_legs = {"P07": "left"}                                        # remembered with the code (restore_session, 0.20.0)
    live.participant.setText("P07")
    assert live.leg_combo.currentData() == "left"
    live._participant_restored = "P07"                                              # what restore_session does with a code from the last session
    live.update_steps()
    assert states(live)["participant"] == "todo" and live.strip.chips["participant"].text().endswith("P07?")      # (the camera comes first now)
    assert "kept from the last session" in live.instruction.text() or "kept from the last session" in live._hint("participant")
    assert "press Enter to confirm" in live.accordion.sections["participant"].summary
    live.worker = RunningCamera()
    mark_all(win)
    live.update_steps()
    assert states(live)["record"] == "off"                                          # not ready to record for a participant nobody has confirmed
    live.btn_rec.setChecked(True)
    live.toggle_record()                                                            # pressing Record asks first, and No means no recording
    assert answers and "P07" in answers[0] and win.engine.rec is None and not live.btn_rec.isChecked()
    live.open_step("participant")                                                   # (restore_session opens it for a kept code: the camera is first now)
    assert live.btn_participant_ok.isVisibleTo(live) and "P07" in live.btn_participant_ok.text()    # a button for it, not only the Enter key
    live.btn_participant_ok.click()                                                 # yes, the same participant
    live.update_steps()
    assert states(live)["participant"] == "done" and live.strip.chips["participant"].text().endswith("P07")
    assert not live._participant_unconfirmed() and not live.btn_participant_ok.isVisibleTo(live)


def test_another_code_counts_at_once_and_so_does_the_old_one_after_it_is_confirmed_by_recording(win, monkeypatch):
    live = win.live
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    live.participant.setText("P07")
    live._participant_restored = "P07"
    live.participant.setText("P08")                                                 # a new participant: typed, so no question
    assert win.engine.tested_leg is None and live.leg_combo.currentData() is None   # ... and no leg yet (it was P07's)
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live.update_steps()
    assert states(live)["participant"] == "done" and not live._participant_unconfirmed()
    live.participant.setText("P07")                                                 # back to the old code: still the kept one
    assert live._participant_unconfirmed()
