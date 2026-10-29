"""0.24.0: soft reminders. After a change of position the ruler, the privacy of the picture and (with the force zeroed) the resting force
(Zero 2) come up as rows under the instruction line, each with the buttons that answer it, as amber chips and, for privacy, as a banner
on the picture. None of them stops Record; what is open is written to the session log and to meta.json (reminders_open_at_start).
Also here: the Ruler (C) button and key, "Participant faces" in the Participant section, and the foot dots only in the ankle tests."""
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.gui import PRIVACY_BANNER, MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
P0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)


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


@pytest.fixture
def win(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow(TESTS, tmp_path)
    w.live.worker = RunningCamera()
    w.engine.process(render(P0, seed=0), 0.0)
    w.live.participant.setText("P01")
    w.live.leg_combo.setCurrentIndex(w.live.leg_combo.findData("right"))
    w.live.update_steps()
    yield w
    w.live.worker = None
    w.close()


def another_target(live):
    t = live.target_spin.value()
    live.target_spin.setValue(t + 10 if t < 100 else t - 10)


def open_rows(live):
    return sorted(live.reminders.open_keys())


def set_ruler(win):
    win.engine.scale.ref_px = 120.0                      # both ruler dots marked


# ------------------------------------------------------------------------------------------------ the rows
def test_a_position_change_opens_the_privacy_and_ruler_rows_and_the_banner(win):
    live = win.live
    set_ruler(win)
    live.chk_priv_ok.setChecked(True)
    live.update_steps()
    assert open_rows(live) == [] and live.reminders.isHidden() and live.view._banner == ""
    another_target(live)
    live.update_steps()
    assert open_rows(live) == ["privacy", "ruler"] and not live.reminders.isHidden()
    assert live.strip.states["privacy"][0] == "check" and live.strip.states["ruler"][0] == "check"     # amber chips, not 'next'
    assert live.view._banner == PRIVACY_BANNER                                         # on the picture itself
    assert not live.btn_ruler_fine.isHidden() and "amber" not in live.btn_ruler_pin.text()
    assert live.btn_ruler_pin.styleSheet() != ""                                        # the pinned Ruler button is amber too


def test_no_ruler_row_when_there_is_no_ruler_or_none_is_wanted_today(win):
    live = win.live
    live.chk_priv_ok.setChecked(True)
    another_target(live)
    live.update_steps()
    assert open_rows(live) == ["privacy"]                                               # no ruler marked: the chip itself asks for one
    live.chk_ruler_skip.setChecked(True)
    live.update_steps()
    assert open_rows(live) == ["privacy"] and live.strip.states["ruler"][0] == "done"


def test_the_first_look_at_the_privacy_is_also_a_row_before_anything_is_said(win):
    live = win.live
    live.update_steps()
    assert open_rows(live) == ["privacy"] and live.view._banner == PRIVACY_BANNER
    live.confirm_privacy_if_open()                                                      # the V key
    assert open_rows(live) == [] and live.view._banner == ""


def test_the_same_position_again_asks_nothing(win):
    live = win.live
    set_ruler(win)
    live.chk_priv_ok.setChecked(True)
    live.target_spin.setValue(live.target_spin.value())
    live.update_steps()
    assert open_rows(live) == []


def test_the_buttons_answer_their_own_row(win):
    live = win.live
    set_ruler(win)
    live.chk_priv_ok.setChecked(True)
    another_target(live)
    live.update_steps()
    live.btn_rem_ruler_fine.click()                                                     # 'Fine'
    assert open_rows(live) == ["privacy"] and not live._ruler_recheck and live.btn_ruler_fine.isHidden()
    live.btn_rem_priv_ok.click()                                                        # 'No face (V)'
    assert open_rows(live) == [] and live.reminders.isHidden() and live.view._banner == ""
    assert live.chk_priv_ok.isChecked()
    another_target(live)
    live.btn_rem_priv_mask.click()                                                      # 'Mask': the next drag on the picture is a mask
    live.view.rect_drawn.emit(0, 0, 200, 200)
    assert win.engine.privacy_masks and "privacy" not in open_rows(live)                # drawing it counted as looking
    another_target(live)
    live.btn_rem_priv_crop.click()
    live.view.rect_drawn.emit(100, 100, 900, 600)
    assert win.engine.privacy_crop and "privacy" not in open_rows(live)


def test_marking_the_ruler_again_answers_the_ruler_row(win):
    live = win.live
    scene = {**P0, "ruler_a": np.array([120.0, 670.0]), "ruler_b": np.array([360.0, 670.0])}       # a 20 cm ruler would be 240 px here
    win.engine.process(render(scene, seed=1), 0.1)
    live.chk_priv_ok.setChecked(True)
    assert win.engine.mark_scale_dot(0, *(scene["ruler_a"] + 2)) and win.engine.mark_scale_dot(1, *(scene["ruler_b"] + 2))
    another_target(live)
    live.update_steps()
    assert "ruler" in open_rows(live)
    live.btn_rem_ruler_redo.click()                                                     # 'Redo (C)': the old scale goes, two clicks wait
    assert live._ruler_marking == 0 and not win.engine.scale.configured
    live.update_steps()
    assert "ruler" not in open_rows(live)                                               # nothing left to look at again; the chip asks for a ruler
    assert live.strip.states["ruler"][0] in ("todo", "next")
    live.on_click(*(scene["ruler_a"] + 2))
    assert live._ruler_marking == 1
    live._ruler_recheck = True                                                          # (as if it were still open: the second click answers it)
    live.on_click(*(scene["ruler_b"] + 2))
    assert live._ruler_marking is None and win.engine.scale.configured and not live._ruler_recheck
    live.update_steps()
    assert "ruler" not in open_rows(live) and live.strip.states["ruler"][0] == "done"


def test_a_row_is_hidden_while_recording(win):
    live = win.live
    live.update_steps()
    assert open_rows(live) == ["privacy"]
    live._rec_started = time.perf_counter()
    try:
        live.update_steps()
        assert open_rows(live) == [] and live.view._banner == ""
    finally:
        live._rec_started = None


# ------------------------------------------------------------------------------------------------ the resting force
def test_the_resting_force_row_follows_the_zero_and_the_position(win, tmp_path):
    from limbtrack.force import ForceCalibration
    app = QApplication.instance()
    live = win.live
    live.chk_priv_ok.setChecked(True)
    mark = lambda: [win.engine.mark_dot(n, *(P0[n] + 2)) for n in win.engine.active_dots]          # noqa: E731
    assert all(mark())
    live._pick(live.force_kind, "simulated")
    live.btn_force.click()
    live.force_source.rest_s = 60.0                                                     # the participant does not push
    end = time.time() + 10
    while time.time() < end and live.force_source.average_volts(3.0)[1] < 2400:
        app.processEvents()
        time.sleep(0.01)
    live.force_source.calibration.zero_set = False                                      # (the simulated force comes with a zero already)
    live.update_steps()
    assert "force" not in open_rows(live)                                               # not zeroed: nothing to take Zero 2 on
    live.btn_force_zero.click()
    live.update_steps()
    assert "force" in open_rows(live) and live.strip.states["force"][0] == "check"
    assert "Zero 2" in live.strip.chips["force"].text()
    live.btn_rem_force.click()                                                          # 'Zero 2 (X)'
    live.update_steps()
    assert "force" not in open_rows(live) and live.strip.states["force"][0] == "done"
    win.engine.target = 90.0                                                            # another position: the level is not used there
    live.update_steps()
    assert "force" in open_rows(live)
    live.btn_rem_force_not.click()                                                      # 'Not here'
    assert "force" not in open_rows(live)
    win.engine.target = 70.0                                                            # the next position: asked again
    live.update_steps()
    assert "force" in open_rows(live)
    live.shutdown()


# ------------------------------------------------------------------------------------------------ Record is never refused by them
@pytest.fixture
def real(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.Yes)
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(150):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 10 * math.sin(math.pi * i / 150), ankle_df=5), seed=i))
    w.release()
    window = MainWindow(TESTS, tmp_path)

    def pump(cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    live = window.live
    live.start_source(str(video))
    assert pump(lambda: window.engine._last_gray is not None)
    for name in window.engine.test.dots:
        live.view.clicked.emit(float(P0[name][0]) + 2, float(P0[name][1]) + 2)
    assert window.engine.next_dot_to_mark() is None
    live.participant.setText("P01")
    yield window, live, warnings, asked, pump, tmp_path
    window.close()


def test_record_goes_ahead_without_the_lock_and_asks_nothing(real):
    """Not locked, no look at the privacy: until 0.23.0 a question or a refusal; now a recording with the reminders written down."""
    window, live, warnings, asked, pump, tmp = real
    set_ruler(window)
    live._ruler_recheck = True
    assert not window.engine.locked
    live.btn_rec.click()
    assert window.engine.rec is not None and not warnings and not asked, (warnings, asked)
    assert pump(lambda: window.engine.rec.frames > 5)
    live.btn_rec.click()
    meta = json.loads(next((tmp / "recordings").rglob("*_meta.json")).read_text())
    texts = " | ".join(meta["reminders_open_at_start"])
    assert "not locked" in texts and "Privacy" in texts and "Ruler" in texts, texts
    assert meta["position_locked"] is False
    shown = " | ".join(live._last_trial_warnings)
    assert "Recorded while this was open: The position is not locked" in shown and "Privacy" in shown
    assert "Ruler" not in shown                                                         # the ruler does not change what the file holds
    log = (tmp / "recordings" / "session_log.csv").read_text(encoding="utf-8")
    assert "recording started with reminders open" in log


def test_a_recording_with_everything_answered_has_an_empty_list(real):
    window, live, warnings, asked, pump, tmp = real
    live.chk_priv_ok.setChecked(True)
    live.btn_lock.click()
    assert window.engine.locked
    live.btn_rec.click()
    assert window.engine.rec is not None
    assert pump(lambda: window.engine.rec.frames > 5)
    live.btn_rec.click()
    meta = json.loads(next((tmp / "recordings").rglob("*_meta.json")).read_text())
    assert meta["reminders_open_at_start"] == []
    assert not any("Recorded while this was open" in w for w in live._last_trial_warnings)


def test_what_still_stops_a_recording(real):
    window, live, warnings, asked, pump, tmp = real
    live.participant.setText("Peter Jones")                                             # a name instead of a code
    live.btn_rec.click()
    assert window.engine.rec is None and warnings and not live.btn_rec.isChecked()
    live.participant.setText("P01")
    live.notes.setText("mail me at peter@example.com")                                 # notes must not identify anyone
    warnings.clear()
    live.btn_rec.click()
    assert window.engine.rec is None and warnings


def test_the_force_not_zeroed_is_a_reminder_that_changes_the_file(win):
    from limbtrack.force import ForceCalibration
    live = win.live

    class Src:
        error = None
        calibration = ForceCalibration(n_per_v=526.95, zero_set=False)
    live.force_source = Src()
    texts = [t for t, in_file in live._open_reminders() if in_file]
    assert any("not been zeroed" in t for t in texts)
    live.force_source = None


# ------------------------------------------------------------------------------------------------ the ruler button and key
def test_the_ruler_button_and_key_mark_it_again_from_any_section(win):
    live = win.live
    live.open_step("force")                                                             # another section is open
    shortcut = next(sc for sc in live.findChildren(QShortcut) if sc.key() == QKeySequence(Qt.Key_C))
    assert not shortcut.autoRepeat()
    set_ruler(win)
    assert win.engine.scale.configured
    shortcut.activated.emit()
    assert live._ruler_marking == 0 and not win.engine.scale.configured and live.btn_ruler_pin.isChecked()   # the old scale goes at once
    live.update_scale_label()
    assert live.btn_ruler_pin.isChecked()
    live.btn_ruler_pin.click()                                                          # pressed again while it waits: stops
    assert live._ruler_marking is None and not live.btn_ruler_pin.isChecked()


def test_the_ruler_button_says_so_when_there_is_no_picture(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(TESTS, tmp_path)
    try:
        w.live.btn_ruler_pin.click()
        assert w.live._ruler_marking is None and not w.live.btn_ruler_pin.isChecked()
        assert "Start the camera" in w.live.instruction.text()
    finally:
        w.close()


def test_a_dot_colour_change_clears_the_ruler_and_says_so(win):
    live = win.live
    set_ruler(win)
    live._pick(live.kind_combo, "white")
    assert not win.engine.scale.configured and "mark it again (C)" in live.instruction.text()


# ------------------------------------------------------------------------------------------------ where things are
def test_participant_faces_is_in_the_participant_section_not_the_position_section(win):
    live = win.live
    participant = live.accordion.sections["participant"].body
    position = live.accordion.sections["position"].body
    assert participant.isAncestorOf(live.facing_combo) and not position.isAncestorOf(live.facing_combo)
    assert participant.isAncestorOf(live.leg_combo)
    assert not live.facing_combo.isHidden()
    live._pick(live.test_combo, "hip_abduction")                                        # camera in front: nobody 'faces' left or right
    assert live.facing_combo.isHidden() and live.participant_form.labelForField(live.facing_combo).isHidden()
    assert not live.tape_spin.isHidden()
    live._pick(live.test_combo, "knee_extension")
    assert not live.facing_combo.isHidden() and not live.participant_form.labelForField(live.facing_combo).isHidden()


def test_the_foot_dots_are_tracked_only_in_the_ankle_tests_and_the_belt_squat():
    foot = {"mt5_base", "mt5_head"}
    for key in ("knee_extension", "knee_flexion", "hip_extension_k0", "hip_extension_k90", "hip_flexion"):
        t = TESTS[key]
        assert not foot & set(t.all_dots) and [a.name for a in t.angles if a.name == "ankle"] == [], key
        assert len(t.dots) == 5
    for key in ("ankle_plantarflexion", "ankle_dorsiflexion", "belt_squat"):
        assert foot <= set(TESTS[key].dots) and len(TESTS[key].dots) == 7, key
    assert TESTS["knee_extension"].all_dots == TESTS["hip_flexion"].all_dots            # same dots: changing test keeps them marked


def test_switching_between_knee_and_hip_tests_keeps_the_marked_dots(win):
    live = win.live
    e = win.engine
    for n in e.active_dots:
        assert e.mark_dot(n, *(P0[n] + 2))
    live._pick(live.test_combo, "hip_flexion")
    assert e.next_dot_to_mark() is None and e.trackers.is_ready()                       # nothing to click again
    live._pick(live.test_combo, "ankle_plantarflexion")
    assert e.next_dot_to_mark() is not None                                             # the foot dots are new: marked afresh
