"""What the program remembers between launches (settings.json): the file helpers, a full round trip through two windows, the
things that are deliberately NOT remembered, and opening the last camera and force when the program starts."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import session_memory as sm  # noqa: E402
from limbtrack.gui import LiveTab, MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def window(root):
    return MainWindow(TESTS, root)


# ------------------------------------------------------------------------------------------------ the file
def test_writing_a_part_keeps_the_other_parts_and_a_bad_file_reads_as_empty(tmp_path):
    path = tmp_path / "settings.json"
    assert sm.read_all(path) == {} and sm.read_session(path) == {}
    path.write_text(json.dumps({"recordings_dir": "D:/data"}), encoding="utf-8")
    assert sm.write_session(path, {"test": "knee_extension"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["recordings_dir"] == "D:/data" and data["session"]["test"] == "knee_extension" and data["session"]["format"] == sm.FORMAT
    assert sm.write_part(path, recordings_dir="E:/other") and sm.read_session(path)["test"] == "knee_extension"      # the session survives
    assert sm.read_all(path)["recordings_dir"] == "E:/other"
    path.write_text("{not json", encoding="utf-8")
    assert sm.read_all(path) == {} and sm.read_session(path) == {}
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert sm.read_all(path) == {}
    path.write_text(json.dumps({"session": {"format": 999, "test": "x"}}), encoding="utf-8")
    assert sm.read_session(path) == {}                                                                                 # an unknown format is ignored


def test_an_unwritable_file_does_not_raise(tmp_path):
    assert sm.write_part(tmp_path / "no_such_folder" / "settings.json", a=1) is False


def test_a_participant_code_comes_back_only_within_twelve_hours():
    now = 1_000_000.0
    assert sm.participant_if_recent({"participant": "P07", "saved_at": now - 3600}, now) == "P07"
    assert sm.participant_if_recent({"participant": "P07", "saved_at": now - 13 * 3600}, now) == ""
    assert sm.participant_if_recent({"participant": "P07", "saved_at": now + 5}, now) == ""            # from the future: not trusted
    assert sm.participant_if_recent({"participant": 7, "saved_at": now}, now) == ""
    assert sm.participant_if_recent({}, now) == ""


def test_nothing_is_opened_by_itself_without_a_screen_or_when_asked_not_to(monkeypatch):
    assert sm.autostart_allowed() is False                                    # these tests run offscreen
    monkeypatch.setenv("QT_QPA_PLATFORM", "windows")
    monkeypatch.delenv("LIMBTRACK_NO_AUTOSTART", raising=False)
    assert sm.autostart_allowed() is True
    monkeypatch.setenv("LIMBTRACK_NO_AUTOSTART", "1")
    assert sm.autostart_allowed() is False


# ------------------------------------------------------------------------------------------------ a round trip through two windows
def test_the_set_up_comes_back_in_a_new_window(app, tmp_path):
    w1 = window(tmp_path)
    a = w1.live
    a._pick(a.test_combo, "hip_extension_k90")
    a._pick(a.kind_combo, "black")
    a._pick(a.facing_combo, False)
    a.tol_spin.setValue(2.5)
    a.ntol_spin.setValue(7.0)
    a.target_spin.setValue(45.0)
    a.lm_checks["iliac_crest"].setChecked(False)
    a.chk_cues.setChecked(False)
    a.cue_push_spin.setValue(2.0)
    a.ruler_len.setValue(20.0)
    a.camera_model.setText("Galaxy S25, DroidCam")
    a.camera_checks["zoom_1x"].setChecked(True)
    a._pick(a.force_kind, "simulated")
    a.participant.setText("P07")
    a.auto_len_spin.setValue(5.0)
    a.accordion.open("force")
    a.save_session()
    w1.close()

    w2 = window(tmp_path)
    b = w2.live
    assert b.test_combo.currentData() == "hip_extension_k90" and w2.engine.test.key == "hip_extension_k90"
    assert b.kind_combo.currentData() == "black" and w2.engine.kind == "black"
    assert b.facing_combo.currentData() is False and w2.engine.facing_right is False
    assert b.tol_spin.value() == 2.5 and w2.engine.tolerance == 2.5
    assert b.ntol_spin.value() == 7.0 and w2.engine.neighbour_tolerance == 7.0
    assert w2.engine.target == 45.0 and b.target_spin.value() == 45.0
    assert "iliac_crest" in w2.engine.disabled_dots and not b.lm_checks["iliac_crest"].isChecked()
    assert not b.chk_cues.isChecked() and not w2.engine.audio_cues and b.cue_push_spin.value() == 2.0
    assert b.ruler_len.value() == 20.0 and w2.engine.scale.length_cm == 20.0
    assert b.camera_model.text() == "Galaxy S25, DroidCam"
    assert not b.camera_checks["zoom_1x"].isChecked()                                     # a claim about the camera: made again every session
    assert b.force_kind.currentData() == "simulated" and b.force_source is None            # chosen, but not connected here
    assert b.participant.text() == "P07"                                                   # used a moment ago
    assert b.auto_len_spin.value() == 5.0
    assert b.accordion.current() == "participant"                                          # the kept participant is asked about first
    w2.close()


def test_what_must_not_be_remembered_is_not(app, tmp_path):
    w1 = window(tmp_path)
    a = w1.live
    a.notes.setText("some note")
    a.save_session()
    saved = json.dumps(sm.read_all(tmp_path / "settings.json"))
    for forbidden in ("some note", "crop", "mask", "zero", "position_locked", "positions", "dots_marked"):
        assert forbidden not in saved, forbidden
    w1.close()
    w2 = window(tmp_path)
    assert w2.live.notes.text() == "" and w2.engine.locked is False and w2.engine.privacy_crop is None and w2.engine.privacy_masks == []
    assert not w2.live.engine.trackers.is_ready()
    w2.close()


def test_a_participant_from_yesterday_is_not_offered(app, tmp_path):
    w1 = window(tmp_path)
    w1.live.participant.setText("P03")
    w1.live.save_session()
    w1.close()
    data = sm.read_all(tmp_path / "settings.json")
    data["session"]["saved_at"] = time.time() - 20 * 3600
    (tmp_path / "settings.json").write_text(json.dumps(data), encoding="utf-8")
    w2 = window(tmp_path)
    assert w2.live.participant.text() == ""
    w2.close()


def test_values_that_no_longer_fit_are_skipped_and_the_rest_still_comes_back(app, tmp_path):
    session = {"format": sm.FORMAT, "saved_at": time.time(), "test": "a_test_that_was_removed", "tolerance": 4.0,
               "dot_kind": "purple", "target": "not a number", "neighbour_tolerance": 99999, "participant": "P02",
               "landmarks_disabled": ["no_such_dot", "iliac_crest"], "force": {"kind": "teleporter", "channel": "ai99"},
               "ui": {"splitter": [1], "window": [10, 10, 5, 5], "section": "no_such_section"},
               "camera": "not a dict", "cues": {"on": "yes please", "push_every_s": "soon"}}
    (tmp_path / "settings.json").write_text(json.dumps({"session": session}), encoding="utf-8")
    w = window(tmp_path)                                                                  # must not raise
    b = w.live
    assert b.tol_spin.value() == 4.0                                                      # the good value was applied
    assert b.participant.text() == "P02"
    assert "iliac_crest" in w.engine.disabled_dots
    assert w.engine.test.key == next(iter(TESTS))                                         # the unknown test left the default
    assert b.kind_combo.currentData() == "auto"
    w.close()


def test_the_window_shape_comes_back_but_never_off_screen_or_tiny(app, tmp_path):
    w1 = window(tmp_path)
    w1.resize(1111, 777)
    w1.live.save_session()
    w1.close()
    w2 = window(tmp_path)
    assert (w2.width(), w2.height()) == (1111, 777)
    w2.close()
    data = sm.read_all(tmp_path / "settings.json")
    for bad in ([100000, 100000, 1200, 800], [0, 0, 300, 200]):
        data["session"]["ui"]["window"] = bad
        (tmp_path / "settings.json").write_text(json.dumps(data), encoding="utf-8")
        w3 = window(tmp_path)
        assert (w3.width(), w3.height()) != tuple(bad[2:])
        w3.close()


# ------------------------------------------------------------------------------------------------ saved as it changes
def test_a_change_is_saved_a_moment_later_without_closing_the_program(app, tmp_path):
    w = window(tmp_path)
    assert sm.read_session(tmp_path / "settings.json") == {}                              # nothing yet
    w.live.tol_spin.setValue(1.5)
    pump(app, 1.3)
    assert sm.read_session(tmp_path / "settings.json")["tolerance"] == 1.5                # (a killed program has lost nothing)
    w.close()


def test_changing_the_recordings_folder_keeps_the_session(app, tmp_path):
    w = window(tmp_path)
    w.live.tol_spin.setValue(2.0)
    w.live.save_session()
    w.on_folder_changed(str(tmp_path / "elsewhere"))
    data = sm.read_all(tmp_path / "settings.json")
    assert data["recordings_dir"] == str(tmp_path / "elsewhere") and data["session"]["tolerance"] == 2.0
    w.close()


def test_the_force_is_opened_again_only_if_it_was_connected_and_not_switched_off(app, tmp_path):
    w1 = window(tmp_path)
    a = w1.live
    a._pick(a.force_kind, "simulated")
    a.btn_force.click()                                                                   # connected by the operator
    assert a.force_source is not None and a._force_wanted
    a.save_session()
    a.shutdown()                                                                          # the program closes: this is NOT 'switched off'
    assert a._force_wanted
    a.save_session()
    assert sm.read_session(tmp_path / "settings.json")["force"]["connect_at_start"] is True
    w1.close()
    w2 = window(tmp_path)
    assert w2.live._force_wanted
    w2.live._pick(w2.live.force_kind, "simulated")
    w2.live.btn_force.click()                                                             # connect ...
    w2.live.btn_force.click()                                                             # ... and switch off on purpose
    assert not w2.live._force_wanted
    w2.live.save_session()
    assert sm.read_session(tmp_path / "settings.json")["force"]["connect_at_start"] is False
    w2.live.shutdown()
    w2.close()


# ------------------------------------------------------------------------------------------------ opening the last camera and force
def remembered(tmp_path, **extra):
    session = {"format": sm.FORMAT, "saved_at": time.time(), "camera": {"kind": "webcam", "name": "DroidCam Video"}, "autostart": True,
               "force": {"kind": "off", "connect_at_start": False}}
    session.update(extra)
    (tmp_path / "settings.json").write_text(json.dumps({"session": session}), encoding="utf-8")


def window_with_cameras(tmp_path, monkeypatch, allowed=True):
    monkeypatch.setattr(sm, "autostart_allowed", lambda: allowed)
    started = []
    monkeypatch.setattr(LiveTab, "start_source", lambda self, source: started.append(source))
    monkeypatch.setattr("limbtrack.gui.camera_names", lambda: ["Integrated Camera", "DroidCam Video"])
    w = window(tmp_path)
    return w, started


def test_the_last_camera_is_found_by_name_not_by_number(app, tmp_path, monkeypatch):
    remembered(tmp_path)
    w, started = window_with_cameras(tmp_path, monkeypatch)
    pump(app, 0.9)
    assert started == [1]                                                                 # DroidCam is number 1 today; it was 0 last time
    w.close()


def test_a_camera_that_is_not_there_is_reported_and_nothing_else_is_opened(app, tmp_path, monkeypatch):
    remembered(tmp_path, camera={"kind": "webcam", "name": "Camo"})
    w, started = window_with_cameras(tmp_path, monkeypatch)
    pump(app, 0.9)
    assert started == [] and "('Camo') was not found" in w.live.instruction.text()
    w.close()


def test_a_network_stream_is_reopened_too(app, tmp_path, monkeypatch):
    remembered(tmp_path, camera={"kind": "stream", "address": "http://192.168.1.20:8080/video"})
    w, started = window_with_cameras(tmp_path, monkeypatch)
    pump(app, 0.9)
    assert started == ["http://192.168.1.20:8080/video"] and w.live.stream_box.text() == "http://192.168.1.20:8080/video"
    w.close()


def test_the_box_can_switch_opening_by_itself_off(app, tmp_path, monkeypatch):
    remembered(tmp_path, autostart=False)
    w, started = window_with_cameras(tmp_path, monkeypatch)
    pump(app, 0.9)
    assert started == [] and not w.live.chk_autostart.isChecked()
    w.close()


def test_nothing_starts_in_the_automatic_tests(app, tmp_path, monkeypatch):
    remembered(tmp_path)
    w, started = window_with_cameras(tmp_path, monkeypatch, allowed=False)
    pump(app, 0.9)
    assert started == []
    w.close()


def test_the_first_ever_launch_starts_nothing(app, tmp_path, monkeypatch):
    w, started = window_with_cameras(tmp_path, monkeypatch)
    pump(app, 0.9)
    assert started == [] and w.live.chk_autostart.isChecked()
    w.close()


def test_the_force_is_connected_at_start_when_it_was_wanted(app, tmp_path, monkeypatch):
    remembered(tmp_path, force={"kind": "simulated", "connect_at_start": True})
    w, started = window_with_cameras(tmp_path, monkeypatch)
    pump(app, 0.9)
    assert w.live.force_source is not None and w.engine.force is w.live.force_source
    w.live.shutdown()
    w.close()


def test_the_participant_of_the_last_session_comes_back_as_a_question_not_as_an_answer(app, tmp_path):
    w1 = window(tmp_path)
    w1.live.participant.setText("P07")
    w1.live.save_session()
    w1.close()
    w2 = window(tmp_path)
    b = w2.live
    assert b.participant.text() == "P07" and b._participant_unconfirmed()
    assert b.accordion.current() == "participant"                                           # the question is the first thing on screen
    b.update_steps()
    assert b.strip.states["participant"][0] == "next" and b.strip.chips["participant"].text().endswith("P07?")
    b.participant.returnPressed.emit()
    assert not b._participant_unconfirmed()
    w2.close()


def test_the_open_section_comes_back_when_there_is_no_participant_to_ask_about(app, tmp_path):
    w1 = window(tmp_path)
    w1.live.accordion.open("force")
    w1.live.save_session()
    w1.close()
    w2 = window(tmp_path)
    assert w2.live.accordion.current() == "force" and w2.live.participant.text() == ""
    w2.close()
