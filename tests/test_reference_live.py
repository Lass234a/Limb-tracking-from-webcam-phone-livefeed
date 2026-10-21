"""Reference readings in the Live tab (0.16.0): the section, typing a reading, 'reading taken now' (button and G), readings that belong to
another position or participant (0.17.1: the boxes follow the position at once), a reading typed just before F5, nothing changeable
while recording."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

from tests.test_pause import RunningCamera  # noqa: E402  (stands in for a running camera)

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)


@pytest.fixture
def win(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(TESTS, tmp_path)
    live = w.live
    live.test_combo.setCurrentIndex(live.test_combo.findData("knee_extension"))
    live.target_spin.setValue(60.0)
    eng = w.engine
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(8):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    yield w
    w.close()


def test_the_section_is_in_the_panel_with_a_box_for_every_angle(win):
    live = win.live
    assert "reference" in live.accordion.sections
    assert list(live.ref_fields) == ["knee", "hip", "ankle"]
    assert live.accordion.sections["reference"].summary == "none yet"
    assert "No reading yet" in live.ref_status.text()


def test_a_frontal_test_has_boxes_for_the_typed_hip_flexion_and_knee_too(win):
    live = win.live
    live.test_combo.setCurrentIndex(live.test_combo.findData("hip_abduction"))
    assert list(live.ref_fields) == ["abd_goniometer", "abd_hip_centre", "hip_flexion", "knee_flexion"]


def test_what_is_typed_reaches_the_engine_with_the_instrument(win):
    live = win.live
    live.ref_fields["knee"].setText("61,5 62")
    live.ref_fields["knee"].editingFinished.emit()
    live.ref_instrument.setCurrentIndex(live.ref_instrument.findData("inclinometer"))
    live.ref_resolution.setCurrentIndex(live.ref_resolution.findData(0.1))
    r = win.engine.reference
    assert r.values == {"knee": [61.5, 62.0]} and r.instrument == "inclinometer" and r.resolution_deg == 0.1
    assert r.position == ("knee_extension", 60.0, None)                        # no participant code in the window yet
    assert "Typed: Knee flexion 61.8 (2 readings)" in live.ref_status.text() and live.accordion.sections["reference"].summary.startswith("1 typed")
    live.ref_fields["knee"].setText("")
    live.ref_fields["knee"].editingFinished.emit()
    assert win.engine.reference.values == {}


def test_a_box_that_cannot_be_read_is_marked_and_keeps_the_old_value(win):
    live = win.live
    live.ref_fields["knee"].setText("61")
    live.ref_fields["knee"].editingFinished.emit()
    live.ref_fields["knee"].setText("sixty")
    live.ref_fields["knee"].editingFinished.emit()
    assert win.engine.reference.values == {"knee": [61.0]}
    assert "not a number" in live.ref_status.text() and "dc2626" in live.ref_fields["knee"].styleSheet()
    live.ref_fields["knee"].setText("62")
    live.ref_fields["knee"].editingFinished.emit()
    assert win.engine.reference.values == {"knee": [62.0]} and live.ref_fields["knee"].styleSheet() == "" and "not a number" not in live.ref_status.text()


def test_reading_taken_now_keeps_the_apps_angles_by_button_and_by_the_g_key(win):
    live = win.live
    live.btn_ref_stamp.click()
    stamp = win.engine.reference.stamp
    assert stamp is not None and stamp["app_deg"]["knee"] == pytest.approx(60.0, abs=0.7)
    assert "Moment kept at" in live.ref_status.text() and "Knee flexion 60" in live.ref_status.text()
    assert live.accordion.sections["reference"].summary == "0 typed, moment kept"
    win.engine.clear_reference()
    shortcuts = [s for s in live.findChildren(QShortcut) if s.key() == QKeySequence(Qt.Key_G)]
    assert len(shortcuts) == 1
    shortcuts[0].activated.emit()
    assert win.engine.reference.stamp is not None


def test_reading_taken_now_without_dots_says_why(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(TESTS, tmp_path)
    try:
        said = []
        w.live.say = lambda text, seconds=5.0: said.append(text)
        w.live.btn_ref_stamp.click()
        assert said and "mark the dots first" in said[-1] and w.engine.reference.stamp is None
    finally:
        w.close()


def test_readings_for_another_position_are_flagged_and_not_shown_in_the_boxes(win):
    live = win.live
    live.ref_fields["knee"].setText("61")
    live.ref_fields["knee"].editingFinished.emit()
    live.target_spin.setValue(70.0)                                            # the participant is moved to the next position
    assert live.ref_fields["knee"].text() == ""                                # at once: the window empties the boxes itself
    assert "NOT be saved" in live.ref_status.text() and live.accordion.sections["reference"].summary == "for another position"
    live.target_spin.setValue(60.0)                                            # back: the readings of that position show again
    assert live.ref_fields["knee"].text() == "61"
    live.target_spin.setValue(70.0)
    live.ref_fields["knee"].setText("69")
    live.ref_fields["knee"].editingFinished.emit()                             # a new reading at the new position replaces the old one
    assert win.engine.reference.values == {"knee": [69.0]} and "NOT be saved" not in live.ref_status.text()


def test_an_edit_after_a_new_target_never_saves_the_old_positions_reading(win):
    live = win.live
    live.ref_fields["knee"].setText("61")
    live.ref_fields["knee"].editingFinished.emit()
    live.target_spin.setValue(70.0)
    live.ref_fields["hip"].setText("90")                                       # only the hip is typed at the new position
    live.ref_fields["hip"].editingFinished.emit()
    r = win.engine.reference
    assert r.values == {"hip": [90.0]} and r.position[:2] == ("knee_extension", 70.0)
    live.ref_instrument.setCurrentIndex(live.ref_instrument.findData("inclinometer"))
    assert win.engine.reference.values == {"hip": [90.0]}                     # the instrument alone brings nothing back either


def test_another_participant_empties_the_boxes_and_never_gets_the_readings(win):
    live = win.live
    live.participant.setText("P01")
    live.ref_fields["knee"].setText("61")
    live.ref_fields["knee"].editingFinished.emit()
    assert win.engine.reference.position == ("knee_extension", 60.0, "P01")
    live.participant.setText("P02")                                            # the next participant, same position
    assert live.ref_fields["knee"].text() == "" and not win.engine.reference_for_this_position()
    assert "participant P01" in live.ref_status.text() and live.accordion.sections["reference"].summary == "for another participant"
    live.participant.setText("P01")                                            # a typing slip put right: the readings show again
    assert live.ref_fields["knee"].text() == "61" and win.engine.reference_for_this_position()


def test_readings_typed_before_the_code_go_with_the_code_typed_after(win):
    live = win.live
    live.ref_fields["knee"].setText("61")
    live.ref_fields["knee"].editingFinished.emit()
    live.participant.setText("P01")
    assert live.ref_fields["knee"].text() == "61" and win.engine.reference_for_this_position()


@pytest.fixture
def recordable(win, monkeypatch):
    """The window with a 'running camera', a participant code, and the questions before recording answered Yes."""
    shown = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    win.live.worker = RunningCamera()
    win.live.participant.setText("P01")
    yield win, shown
    if win.engine.rec is not None:
        win.engine.stop_trial()
    win.live.worker = None


def test_a_reading_typed_just_before_f5_is_saved_with_the_trial(recordable):
    win, shown = recordable
    live = win.live
    live.ref_fields["knee"].setText("88")                                      # typed; the box has not let go of it (no editingFinished)
    live.btn_rec.click()                                                       # what F5 does: the box keeps the focus
    assert win.engine.rec is not None
    assert win.engine._meta_start["reference_readings"]["readings"]["knee"]["values_deg"] == [88.0]
    said = []
    live.say = lambda text, seconds=4.0: said.append(text)
    live.update_reference_box()                                                # (the 200 ms refresh) the box is switched off for the recording ...
    live.ref_fields["knee"].editingFinished.emit()                             # ... and lets go of the focus: nothing changed, nothing refused
    assert not said and not live.ref_fields["knee"].isEnabled()
    live.btn_rec.click()
    assert win.engine.rec is None


def test_a_reading_that_cannot_be_read_stops_the_recording_from_starting(recordable):
    win, shown = recordable
    live = win.live
    live.ref_fields["knee"].setText("sixty")
    live.btn_rec.click()
    assert win.engine.rec is None and not live.btn_rec.isChecked()
    assert shown and "cannot be read" in shown[-1] and "not a number" in shown[-1]
    live.ref_fields["knee"].setText("")                                        # emptied: 'not measured', and the recording may start
    live.btn_rec.click()
    assert win.engine.rec is not None and win.engine._meta_start["reference_readings"] is None


def test_clear_empties_the_boxes_and_the_engine(win):
    live = win.live
    live.ref_fields["knee"].setText("61")
    live.ref_fields["knee"].editingFinished.emit()
    live.btn_ref_stamp.click()
    live.btn_ref_clear.click()
    assert not win.engine.reference.has_data() and live.ref_fields["knee"].text() == "" and "No reading yet" in live.ref_status.text()


def test_nothing_can_be_changed_while_recording(win):
    live = win.live
    live.update_reference_box(True)
    assert not live.btn_ref_stamp.isEnabled() and not live.btn_ref_clear.isEnabled() and not live.ref_fields["knee"].isEnabled()
    assert not live.ref_instrument.isEnabled() and not live.ref_resolution.isEnabled()
    live.update_reference_box(False)
    assert live.btn_ref_stamp.isEnabled() and live.ref_fields["knee"].isEnabled()


def test_the_200_ms_refresh_leaves_the_section_alone_when_nothing_changed(win, monkeypatch):
    live = win.live
    live.update_reference_box(False)
    calls = []
    monkeypatch.setattr(live.ref_status, "setStyleSheet", lambda s: calls.append(("style", s)))
    monkeypatch.setattr(live.ref_fields["knee"], "setEnabled", lambda on: calls.append(("enabled", on)))
    for _ in range(3):
        live.update_reference_box(False)
    assert calls == []
    live.update_reference_box(True)                                            # a recording starts: now the boxes are switched off
    assert ("enabled", False) in calls


def test_a_three_point_angle_gets_its_own_box(win):
    live = win.live
    live.chk_tp.setChecked(True)
    live.apply_three_point()
    assert "three_point" in live.ref_fields
