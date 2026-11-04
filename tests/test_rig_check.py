"""0.35.0 (F9): the rig check, one list before the participant arrives (limbtrack/rig_check.py, limbtrack/rig_check_dialog.py, the Live tab's Rig check
button). Each item is green, amber or grey ("not checked") with a one-line reason: the camera and the picture rate the live loop really runs at, the force
connected and its samples, the zero noise with the hum, the check weight (F8), the beeps (the real cues through limbtrack/cues.py, ticked as heard),
the ruler and the lamp. The list is read on a helper thread, saved on request to the session log and into the meta.json of the following trials, and it
never blocks Record or anything else.

The equipment is the real thing where it can be: the real Engine and its live loop, the real force source code (its ring buffer; a simulated load cell
running in real time for the DAQ read), the real window and dialog offscreen, the real cue player (also once with the real Windows sound call), and real
files (session_log.csv, meta.json)."""
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import check_weight as cw  # noqa: E402
from limbtrack import cues, rig_check as rc  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.rig_check_dialog import RigCheckDialog  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402
from tests.test_check_weight import LOAD_N, MASS_KG, N_PER_V, LoadedForce, RunningCamera, connect, made, source, volts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
P = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


def live_camera(eng, name="Galaxy S25 (1280x720)", kind="webcam", fps=30.0, frames=40):
    """The engine as if a camera had been started and had delivered `frames` pictures at `fps`."""
    eng.set_source(name, {"source_kind": kind, "is_file": kind == "file"}, None)
    t0 = time.perf_counter()
    for i in range(frames):
        eng.process(render(P, seed=i % 3), t0 + i / fps)


def facts_of(eng, **kw):
    return rc.gather(eng, **kw)


def state(result, key):
    return result["items"][key]["state"]


# ------------------------------------------------------------------------------------------------ the live loop's picture rate
def test_the_picture_rate_comes_from_the_live_loop_and_is_unknown_until_ten_pictures(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    assert eng.live_fps() is None
    live_camera(eng, frames=6)
    assert eng.live_fps() is None
    live_camera(eng, fps=30.0, frames=40)
    assert eng.live_fps() == pytest.approx(30.0, rel=0.01)
    live_camera(eng, fps=12.5, frames=40)
    assert eng.live_fps() == pytest.approx(12.5, rel=0.01)


# ------------------------------------------------------------------------------------------------ the camera item
def test_the_camera_is_green_with_a_name_and_a_good_rate_and_says_why_when_it_is_not(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    f = facts_of(eng)
    assert state(rc.evaluate(f), "camera") == "grey" and "No camera is started" in rc.evaluate(f)["items"]["camera"]["reason"]
    live_camera(eng, fps=30.0)
    r = rc.evaluate(facts_of(eng, camera_name="Galaxy S25, DroidCam"))
    assert state(r, "camera") == "green" and "Galaxy S25, DroidCam" in r["items"]["camera"]["reason"] and "30.0 pictures a second" in r["items"]["camera"]["reason"]
    r = rc.evaluate(facts_of(eng, camera_name=""))
    assert state(r, "camera") == "amber" and "not known" in r["items"]["camera"]["reason"]           # opened by number
    live_camera(eng, fps=15.0)
    r = rc.evaluate(facts_of(eng, camera_name="Galaxy S25"))
    assert state(r, "camera") == "amber" and "only 15.0 pictures a second" in r["items"]["camera"]["reason"]
    live_camera(eng, kind="file")
    assert state(rc.evaluate(facts_of(eng, camera_name="x")), "camera") == "amber" and "video file" in rc.evaluate(facts_of(eng))["items"]["camera"]["reason"]
    live_camera(eng, fps=30.0)
    eng._frame_stamp -= 5.0                                                              # no picture for five seconds
    r = rc.evaluate(facts_of(eng, camera_name="Galaxy S25"))
    assert state(r, "camera") == "amber" and "No picture for" in r["items"]["camera"]["reason"]
    live_camera(eng, kind="stream", fps=30.0)
    assert state(rc.evaluate(facts_of(eng, camera_name="network stream")), "camera") == "green"
    eng.set_source("Galaxy S25 (1280x720)", {"source_kind": "webcam", "is_file": False}, None)           # a new source: the timing starts afresh
    assert state(rc.evaluate(facts_of(eng, camera_name="Galaxy S25")), "camera") == "grey"          # frame rate not measured yet


def test_no_picture_yet_is_said_so_and_not_as_infinity(tmp_path):
    """R2: before the first picture or the first sample the silence is infinite; the reason must be a sentence."""
    c = {"started": True, "kind": "webcam", "name": "Cam", "fps": None, "silent_s": float("inf")}
    st, why = rc._camera(c)
    assert st == "amber" and why == "No picture has arrived yet from the camera." and "inf" not in why
    f = {"connected": True, "kind": "simulated", "error": None, "silent_s": float("inf"), "samples_last_s": 0, "nominal_hz": 50.0}
    st, why = rc._force(f)
    assert st == "amber" and why == "No force sample has arrived yet." and "inf" not in why
    assert rc._force({**f, "silent_s": 2.0})[1] == "No force sample for 2.0 s."               # a finite silence keeps its number
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_source("Galaxy S25 (1280x720)", {"source_kind": "webcam", "is_file": False}, None)      # started, nothing arrived (the engine's real infinity)
    r = rc.evaluate(facts_of(eng, camera_name="Galaxy S25"))
    assert state(r, "camera") == "amber" and "inf" not in r["items"]["camera"]["reason"] and "has arrived yet" in r["items"]["camera"]["reason"]


def test_the_camera_name_comes_from_the_driver_and_a_placeholder_or_a_number_match_is_not_a_name(tmp_path):
    """T7: the engine's own facts (capture_device_name, capture_matched_by_name) decide, not the drop-down's text."""
    eng = Engine(TESTS, recordings_dir=tmp_path)

    def start(info, name):
        eng.set_source("0 (1280x720)", {"source_kind": "webcam", "is_file": False, **info}, None)
        t0 = time.perf_counter()
        for i in range(40):
            eng.process(render(P, seed=i % 3), t0 + i / 30)
        return rc.evaluate(facts_of(eng, camera_name=name))["items"]["camera"]
    got = start({"capture_device_name": "DroidCam Video", "capture_matched_by_name": True}, "Camera 0")      # the driver's name wins over the drop-down
    assert got["state"] == "green" and "DroidCam Video" in got["reason"]
    got = start({"capture_device_name": None, "capture_matched_by_name": False}, "0: something")            # opened by the same number: names were unavailable
    assert got["state"] == "amber" and "not known" in got["reason"]
    got = start({}, "Camera 0")                                                                              # the program's own stand-in when names cannot be read
    assert got["state"] == "amber" and "not known" in got["reason"]
    got = start({}, "Galaxy S25")
    assert got["state"] == "green" and "Galaxy S25" in got["reason"]


# ------------------------------------------------------------------------------------------------ the force and the zero noise
def test_the_force_item_follows_the_connection_the_samples_and_the_error(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    r = rc.evaluate(facts_of(eng))
    assert state(r, "force") == "amber" and "not connected" in r["items"]["force"]["reason"] and state(r, "zero_noise") == "grey"
    r = rc.evaluate(facts_of(eng, force_skipped=True))
    assert state(r, "force") == "grey" and "No force today" in r["items"]["force"]["reason"]
    src = source(volts(load_n=0.0, hum_n=2.0))
    eng.force = src
    r = rc.evaluate(facts_of(eng))
    assert state(r, "force") == "green" and "samples in the last second" in r["items"]["force"]["reason"]
    src.error = "The DAQ stopped sending samples (cable unplugged or device reset)."
    r = rc.evaluate(facts_of(eng))
    assert state(r, "force") == "amber" and "cable unplugged" in r["items"]["force"]["reason"]
    stale = source(volts(load_n=0.0))
    stale._last_arrival = time.perf_counter() - 3.0                                         # the last chunk came three seconds ago
    eng.force = stale
    assert state(rc.evaluate(facts_of(eng)), "force") == "amber" and "No force sample" in rc.evaluate(facts_of(eng))["items"]["force"]["reason"]
    thin = source(volts(seconds=0.3))
    eng.force = thin
    assert "Only" in rc.evaluate(facts_of(eng))["items"]["force"]["reason"]
    eng.force = None


def test_the_zero_noise_is_green_when_quiet_amber_with_noise_or_hum_and_grey_without_a_zero(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.force = source(volts(load_n=0.0, hum_n=3.0, noise_v=0.002))
    r = rc.evaluate(facts_of(eng))
    z = r["items"]["zero_noise"]
    assert z["state"] == "green" and "mV" in z["reason"] and "hum" in z["reason"] and " N)" in z["reason"]
    eng.force = source(volts(load_n=0.0, hum_n=3.0, noise_v=0.02))                            # a third of the refusal limit, twice over
    z = rc.evaluate(facts_of(eng))["items"]["zero_noise"]
    assert z["state"] == "amber" and "above the 10 mV" in z["reason"]
    eng.force = source(volts(load_n=0.0, hum_n=25.0, noise_v=0.002))                          # 47 mV of hum
    z = rc.evaluate(facts_of(eng))["items"]["zero_noise"]
    assert z["state"] == "amber" and "isolated outlet" in z["reason"]
    eng.force = source(volts(load_n=0.0), zero_set=False)
    z = rc.evaluate(facts_of(eng))["items"]["zero_noise"]
    assert z["state"] == "grey" and "not zeroed" in z["reason"]
    f = facts_of(eng)
    assert f["zero_noise"] == {}                                                              # nothing was read: the zero is missing
    eng.force = None


def test_the_noise_and_the_hum_are_what_was_put_into_the_signal():
    src = source(volts(load_n=0.0, hum_n=10.0, noise_v=0.004))
    z = rc.zero_noise(src)
    assert z["noise_v"] == pytest.approx(0.004, rel=0.1) and z["hum_v"] == pytest.approx(10.0 / N_PER_V, rel=0.05)
    assert z["noise_n"] == pytest.approx(0.004 * N_PER_V, rel=0.1) and z["samples"] % 20 == 0 and z["seconds"] == 2.0
    assert "why" in rc.zero_noise(source(volts(seconds=0.5)))


def test_the_simulated_load_cell_in_real_time_goes_through_the_whole_force_reading(tmp_path):
    """The real reader thread of a source (the simulated one): its samples are counted, its noise and hum are read, and reading moves nothing."""
    eng = Engine(TESTS, recordings_dir=tmp_path)
    src = LoadedForce(hum_n=4.0, rest_s=1e6, period_s=1e7)
    src.start()
    eng.force = src
    try:
        time.sleep(2.5)
        assert src.zero() is None
        before = (src.calibration.offset_v, src.calibration.n_per_v, src.calibration.zero_set, src.calibration.when)
        time.sleep(2.2)
        f = facts_of(eng)
        after = (src.calibration.offset_v, src.calibration.n_per_v, src.calibration.zero_set, src.calibration.when)
    finally:
        src.stop()
    assert before == after                                                                    # the check only reads
    assert 900 <= f["force"]["samples_last_s"] <= 1100 and f["force"]["kind"] == "simulated"
    z = f["zero_noise"]
    assert z["noise_v"] == pytest.approx(0.002, rel=0.2) and z["hum_v"] == pytest.approx(4.0 / N_PER_V, rel=0.1), z
    r = rc.evaluate(f)
    assert state(r, "force") == "green" and state(r, "zero_noise") == "green"


# ------------------------------------------------------------------------------------------------ check weight, ruler, lamp
def test_the_check_weight_item_is_the_f8_standing(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.force = source(volts(load_n=LOAD_N, hum_n=3.0))
    r = rc.evaluate(facts_of(eng))
    assert state(r, "check_weight") == "grey" and "No check weight this run" in r["items"]["check_weight"]["reason"]
    res, why = eng.make_check_weight(MASS_KG)
    assert why is None
    r = rc.evaluate(facts_of(eng))
    assert state(r, "check_weight") == "green" and "25 kg" in r["items"]["check_weight"]["reason"]
    res, _ = eng.make_check_weight(24.6)                                                      # a wrong typed mass: +1.6 % is amber
    r = rc.evaluate(facts_of(eng))
    assert state(r, "check_weight") == "amber" and "AMBER" in r["items"]["check_weight"]["reason"]
    eng.check_weight = made(14.0)[0]
    r = rc.evaluate(facts_of(eng))
    assert state(r, "check_weight") == "amber" and "over 12 h old" in r["items"]["check_weight"]["reason"]
    eng.check_weight = made(1.0, level="red", err=-3.1)[0]
    assert "RED" in rc.evaluate(facts_of(eng))["items"]["check_weight"]["reason"]
    eng.force.calibration.n_per_v *= 1.05                                                     # another slope since
    eng.check_weight = made(1.0)[0]
    assert "another calibration" in rc.evaluate(facts_of(eng))["items"]["check_weight"]["reason"]
    eng.force = None


def test_the_ruler_and_the_lamp(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    r = rc.evaluate(facts_of(eng))
    assert state(r, "ruler") == "grey" and "Not marked" in r["items"]["ruler"]["reason"]
    assert state(r, "lamp") == "grey" and "Not ticked" in r["items"]["lamp"]["reason"]
    eng.scale.length_cm = 40.0
    eng.scale.ref_px = 200.0                                                                  # both ruler dots marked
    r = rc.evaluate(facts_of(eng), {"lamp": True})
    assert state(r, "ruler") == "green" and "5.00 px per cm" in r["items"]["ruler"]["reason"] and state(r, "lamp") == "green"
    f = facts_of(eng)
    f["ruler"]["problems"] = ["The ruler is only 90 px long."]
    assert rc.evaluate(f)["items"]["ruler"]["state"] == "amber"


# ------------------------------------------------------------------------------------------------ the beeps
class RecordingBackend:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, name):
        if self.fail:
            raise RuntimeError("Failed to play sound")
        self.calls.append(name)


def test_the_three_beeps_go_through_the_cue_player_and_an_error_is_kept_not_hidden(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.audio_cues = False                                                                    # the check plays them also when they are off for recordings
    backend = RecordingBackend()
    eng._cues = cues.CuePlayer(backend=backend)
    seconds = eng.beep_check()
    assert 2.0 < seconds < 3.0
    time.sleep(seconds)
    names, err = eng.beep_check_result()
    assert names == list(rc.BEEP_NAMES) == backend.calls and err is None
    bad = RecordingBackend(fail=True)
    eng._cues = cues.CuePlayer(backend=bad)
    time.sleep(0.1)
    time.sleep(eng.beep_check())
    names, err = eng.beep_check_result()
    assert err == "RuntimeError: Failed to play sound" and eng._cues.error_count == 3
    r = rc.evaluate({"beeps_silent": False}, {"played": True, "names": names, "error": err, "heard": True})
    assert state(r, "beeps") == "amber" and "Failed to play sound" in r["items"]["beeps"]["reason"]


def test_a_second_press_only_counts_the_new_beeps(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    backend = RecordingBackend()
    eng._cues = cues.CuePlayer(backend=backend)
    time.sleep(eng.beep_check())
    time.sleep(eng.beep_check())
    names, err = eng.beep_check_result()
    assert names == list(rc.BEEP_NAMES) and err is None and len(backend.calls) == 6


def test_the_beeps_item_needs_them_played_without_error_and_heard_and_is_grey_where_sound_is_off():
    f = {"beeps_silent": False}
    assert rc.evaluate(f, {})["items"]["beeps"]["state"] == "grey"
    played = {"played": True, "names": list(rc.BEEP_NAMES), "error": None, "heard": False}
    assert rc.evaluate(f, played)["items"]["beeps"]["state"] == "amber" and "not ticked" in rc.evaluate(f, played)["items"]["beeps"]["reason"]
    assert rc.evaluate(f, {**played, "heard": True})["items"]["beeps"]["state"] == "green"
    assert rc.evaluate(f, {**played, "names": ["record"], "heard": True})["items"]["beeps"]["state"] == "amber"
    assert rc.evaluate({"beeps_silent": True}, {**played, "heard": True})["items"]["beeps"]["state"] == "grey"      # a silent test run proves nothing


def test_the_real_windows_sound_call_plays_the_three_cues_without_an_error(tmp_path, monkeypatch):
    """The one call that is not faked: cues.py's own winsound call, the way a recording makes it. (You will hear three short beeps.) A spy wraps
    winsound.PlaySound and calls straight through, so the test shows that Windows WAS asked three times, with the cues' own sound data. That the
    sound is audible on the lab computer is not something a test can hear."""
    winsound = pytest.importorskip("winsound")
    if sys.platform != "win32":
        pytest.skip("the Windows sound call")
    monkeypatch.setattr(cues, "silent", lambda: False)
    asked = []
    real = winsound.PlaySound

    def spy(sound, flags):
        asked.append((sound, flags))
        return real(sound, flags)                                                             # (the real call, not a stand-in)
    monkeypatch.setattr(winsound, "PlaySound", spy)
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng._cues = cues.CuePlayer()                                                              # the default backend: winsound.PlaySound
    time.sleep(eng.beep_check() + 0.3)
    names, err = eng.beep_check_result()
    assert err is None, err
    assert names == list(rc.BEEP_NAMES)
    assert [a[0] for a in asked] == [cues.wav_bytes(n) for n in rc.BEEP_NAMES] and all(a[1] == winsound.SND_MEMORY for a in asked)
    assert rc.gather(eng)["beeps_silent"] is False


# ------------------------------------------------------------------------------------------------ the list as a whole
def test_all_green_only_when_every_item_is_and_the_summary_counts_them():
    green = {"camera": {"started": True, "kind": "webcam", "name": "Cam", "fps": 30.0, "silent_s": 0.0},
             "force": {"connected": True, "kind": "simulated", "error": None, "silent_s": 0.0, "samples_last_s": 1000, "nominal_hz": 1000.0, "zero_set": True},
             "zero_noise": {"noise_v": 0.002, "hum_v": 0.003, "noise_n": 1.0, "hum_n": 1.5, "samples": 2000, "seconds": 2.0},
             "check_weight": {"check": made(1.0)[0], "state": "ok", "age_h": 1.0},
             "ruler": {"configured": True, "px_per_cm": 5.0, "length_cm": 40.0, "problems": []}, "beeps_silent": False, "time": "2026-11-04T08:00:00"}
    manual = {"played": True, "names": list(rc.BEEP_NAMES), "error": None, "heard": True, "lamp": True}
    r = rc.evaluate(green, manual)
    assert r["all_green"] is True and r["overall"] == "green" and r["counts"] == {"green": 7, "amber": 0, "grey": 0}
    assert [k for k, *_ in rc.rows(r)] == [k for k, _ in rc.ITEMS] and rc.summary_line(r) == "7 of 7 ready."
    r = rc.evaluate(green, {**manual, "lamp": False})
    assert r["overall"] == "grey" and r["counts"]["grey"] == 1 and "1 not checked" in rc.summary_line(r)
    green["camera"]["fps"] = 12.0
    r = rc.evaluate(green, {**manual, "lamp": False})
    assert r["overall"] == "amber" and "1 to look at" in rc.summary_line(r)
    assert r["thresholds"]["min_camera_fps"] == 25.0 and "proposal" in r["note"]


def test_the_log_detail_reads_back_item_by_item():
    r = rc.evaluate({"camera": {"started": False}, "force": {"connected": False}, "time": "2026-11-04T08:00:00"}, {})
    got = rc.parse_log_detail(rc.log_detail(r))
    assert got["overall"][0] == "amber"
    for key, _title, st, reason in rc.rows(r):
        assert got[key] == (st, reason), key
    assert list(got)[1:] == [k for k, _ in rc.ITEMS]


# ------------------------------------------------------------------------------------------------ the window and the dialog
@pytest.fixture
def win(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    (tmp_path / "recordings").mkdir()
    w = MainWindow(TESTS, tmp_path)
    w.app = app
    yield w
    w.live.worker = None
    w.close()


def pump(app, cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def log_rows(win, event):
    path = win.live.recordings_dir / "session_log.csv"
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["event"] == event] if path.exists() else []


def mark_dots(eng):
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(P, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))


def trial(eng, frames=8):
    eng.participant_code = "P01"
    paths = eng.start_trial("P01")
    t0 = time.perf_counter()
    for i in range(frames):
        eng.process(render(P, seed=i % 3), t0 + i / 30)
    eng.stop_trial()
    time.sleep(1.1)
    return paths["meta.json"]


def test_the_rig_check_button_is_in_the_live_tab_beside_the_session_check(win):
    live = win.live
    assert live.btn_rig_check.text() == "Rig check" and live.btn_session_check.text() == "Session check"
    assert live.btn_rig_check.parent() is live.btn_session_check.parent()
    assert live.accordion.sections["position"].body.isAncestorOf(live.btn_rig_check)


def test_the_dialog_reads_on_a_helper_thread_judges_the_ticks_at_once_and_saves_a_line_and_the_block(win, monkeypatch):
    live, eng, app = win.live, win.engine, win.app
    monkeypatch.setattr(cues, "silent", lambda: False)                                        # (the beeps are faked below: the real call has its own test)
    backend = RecordingBackend()
    eng._cues = cues.CuePlayer(backend=backend)
    live.worker = RunningCamera()
    live._last_source = {"kind": "webcam", "name": "Galaxy S25"}                              # (what start_source keeps: the name the camera was started with)
    live_camera(eng, name="Galaxy S25 (1280x720)", fps=30.0)
    src = source(volts(load_n=LOAD_N, hum_n=3.0))
    connect(win, src)
    live.check_box.edit.setText("25")
    live.check_box.edit.textEdited.emit("25")
    assert live.check_box.check()["level"] == "green"
    calibration_before = dict(src.calibration.to_dict())
    threads = []
    real_gather = rc.gather

    def spy(engine, **kw):
        threads.append(threading.current_thread())
        return real_gather(engine, **kw)
    monkeypatch.setattr(rc, "gather", spy)
    src._push(volts(load_n=LOAD_N, hum_n=3.0), time.perf_counter())                          # fresh samples for the zero noise
    live.open_rig_check()
    dlg = live._rig_dialog
    assert dlg is not None and dlg.isVisible()
    assert "Working out" in dlg.view.toPlainText() and dlg.facts is None                      # the window answers at once; the equipment is read meanwhile
    assert pump(app, lambda: dlg.facts is not None)
    assert threads and threads[0] is not threading.main_thread()                              # the equipment was read off the window's thread
    text = dlg.view.toPlainText()
    for title in ("Camera", "Force connected", "Zero noise", "Check weight", "Beeps", "Ruler", "Lamp"):
        assert title in text, title
    r = dlg.result
    assert state(r, "camera") == "green" and state(r, "force") == "green" and state(r, "zero_noise") == "green" and state(r, "check_weight") == "green"
    assert state(r, "beeps") == "grey" and state(r, "ruler") == "grey" and state(r, "lamp") == "grey" and r["overall"] == "grey"
    assert "Not saved yet" in dlg.status.text()
    dlg.chk_lamp.setChecked(True)                                                             # a tick re-judges at once, nothing is read again
    assert state(dlg.result, "lamp") == "green" and len(threads) == 1
    assert not dlg.chk_heard.isEnabled()
    dlg.btn_beeps.click()                                                                     # the real cue player, a recording backend
    assert state(dlg.result, "beeps") == "amber" and "only 0 of the 3" in dlg.result["items"]["beeps"]["reason"]      # still sounding
    assert not dlg.btn_beeps.isEnabled() and "Playing" in dlg.status.text()
    assert pump(app, lambda: dlg.chk_heard.isEnabled(), timeout=6)
    assert backend.calls == list(rc.BEEP_NAMES) and state(dlg.result, "beeps") == "amber" and "not ticked" in dlg.result["items"]["beeps"]["reason"]
    dlg.chk_heard.setChecked(True)
    assert state(dlg.result, "beeps") == "green"
    eng.scale.length_cm, eng.scale.ref_px = 40.0, 200.0                                       # the ruler gets marked meanwhile; "Check again" sees it
    live_camera(eng, name="Galaxy S25 (1280x720)", fps=30.0)                                   # (the pictures and the samples of a minute ago would be stale)
    src = source(volts(load_n=LOAD_N, hum_n=3.0))                                             # (a new source object: a burst pushed into the old one would be mis-timed)
    connect(win, src)
    dlg.btn_again.click()
    assert pump(app, lambda: dlg.btn_again.isEnabled() and state(dlg.result, "ruler") == "green")
    assert dlg.result["all_green"] is True, {k: (v["state"], v["reason"]) for k, v in dlg.result["items"].items() if v["state"] != "green"}
    assert "7 of 7 ready" in dlg.status.text()
    assert eng.rig_check is None
    dlg.btn_save.click()
    assert dlg.saved is not None and eng.rig_check == dlg.saved and "Saved at" in dlg.status.text()
    assert dlg.saved["saved_at"][:10] == datetime.now().isoformat()[:10] and dlg.saved["all_green"] is True
    lines = log_rows(win, "rig check")
    assert len(lines) == 1 and list(lines[0]) == ["time", "seconds_since_start", "event", "detail"]
    got = rc.parse_log_detail(lines[0]["detail"])
    assert got["overall"][0] == "green" and all(got[k] == (dlg.saved["items"][k]["state"], dlg.saved["items"][k]["reason"]) for k, _ in rc.ITEMS)
    assert src.calibration.to_dict() == calibration_before                                    # the list moved no zero and no setting
    dlg.chk_lamp.setChecked(False)
    assert "changed since" in dlg.status.text()                                              # what is on screen is no longer what was saved
    dlg.close()
    live.force_source = None
    eng.force = None


def test_the_saved_check_goes_into_the_following_trials_and_a_trial_before_it_says_null(win):
    live, eng = win.live, win.engine
    mark_dots(eng)
    before = json.loads(trial(eng).read_text(encoding="utf-8"))
    assert "rig_check" in before and before["rig_check"] is None
    block = rc.evaluate({"camera": {"started": False}, "force": {"connected": False}, "time": "2026-11-04T08:00:00"}, {"lamp": True})
    assert live._save_rig_check(dict(block, saved_at="2026-11-04T08:01:00")) is None
    after = json.loads(trial(eng).read_text(encoding="utf-8"))
    got = after["rig_check"]
    assert got["saved_at"] == "2026-11-04T08:01:00" and got["items"]["lamp"]["state"] == "green" and got["items"]["force"]["state"] == "amber"
    assert got["overall"] == "amber" and got["all_green"] is False and "changed_during_recording" not in after
    assert [k for k in got["items"]] == [k for k, _ in rc.ITEMS]
    older = json.loads(trial(eng).read_text(encoding="utf-8"))["rig_check"]                   # and the one after that
    assert older["saved_at"] == "2026-11-04T08:01:00"
    assert log_rows(win, "rig check")[0]["detail"].startswith("overall=amber | camera=grey")


def test_record_is_never_blocked_by_the_rig_check_and_a_save_cannot_happen_during_a_recording(win):
    live, eng = win.live, win.engine
    mark_dots(eng)
    eng.participant_code = "P01"
    eng.start_trial("P01")
    try:
        problem = live._save_rig_check({"time": "t", "items": {}})
        assert problem and "Stop recording" in problem and eng.rig_check is None
        assert eng.beep_check_result() == ([], None)
        # the button of the dialog does not play over a recording's cues
        live.open_rig_check()
        dlg = live._rig_dialog
        assert pump(win.app, lambda: dlg.facts is not None)
        dlg.btn_beeps.click()
        assert "cannot be played now" in dlg.status.text() and dlg.manual["played"] is False
        dlg.close()
    finally:
        eng.stop_trial()
    eng.start_trial("P01")                                                                    # after a failed save and an amber list, Record still works
    assert eng.rec is not None
    eng.stop_trial()


def test_a_gather_that_fails_is_said_in_the_window_not_hidden(win):
    app = win.app

    def broken(ctx):
        raise RuntimeError("the camera object is gone")
    dlg = RigCheckDialog(win.live, "Rig check", broken, lambda: None, lambda: ([], None), lambda r: None)
    assert pump(app, lambda: bool(dlg.error))
    assert "RuntimeError: the camera object is gone" in dlg.view.toPlainText() and dlg.result is None
    dlg.btn_save.click()                                                                      # nothing to save
    assert dlg.saved is None
    dlg.close()


def test_the_context_is_read_on_the_window_thread_and_handed_to_the_helper(win):
    app = win.app
    seen = {}

    def context():
        seen["context_thread"] = threading.current_thread()
        return {"camera_name": "Cam"}

    def gather(ctx):
        seen["gather_thread"], seen["ctx"] = threading.current_thread(), ctx
        return {"time": "2026-11-04T08:00:00"}
    dlg = RigCheckDialog(win.live, "Rig check", gather, lambda: 1.0, lambda: ([], None), lambda r: None, context=context)
    assert pump(app, lambda: dlg.facts is not None)
    assert seen["context_thread"] is threading.main_thread() and seen["gather_thread"] is not threading.main_thread() and seen["ctx"] == {"camera_name": "Cam"}
    dlg.close()


def test_the_dialog_can_run_without_a_helper_thread_for_the_automatic_tests(win):
    dlg = RigCheckDialog(win.live, "Rig check", lambda ctx: {"time": "2026-11-04T08:00:00"}, lambda: 1.0, lambda: ([], None), lambda r: None, threaded=False)
    assert dlg.facts is not None and dlg.result["overall"] in ("amber", "grey")
    dlg.close()


def test_the_window_gives_the_name_the_camera_was_started_with_not_the_drop_downs_text_now(win):
    """T7 at the window: the drop-down was changed while the camera runs; the rig check still names the camera that runs."""
    live = win.live
    live.fill_cameras([(0, "Galaxy S25"), (1, "Integrated Camera")])
    live.cam_combo.setCurrentIndex(0)
    live._last_source = {"kind": "webcam", "name": "Galaxy S25"}
    live.worker = RunningCamera()
    live.worker.source = 0                                                                    # (a webcam is opened by its number)
    live_camera(win.engine, name="0 (1280x720)", fps=30.0)
    live.cam_combo.setCurrentIndex(1)                                                         # (nothing stops the drop-down while a camera runs)
    seen = []
    real_gather = rc.gather
    rc.gather = lambda engine, **kw: (seen.append(kw), real_gather(engine, **kw))[1]
    try:
        live.open_rig_check()
        dlg = live._rig_dialog
        assert pump(win.app, lambda: dlg.facts is not None)
    finally:
        rc.gather = real_gather
    assert seen and seen[-1]["camera_name"] == "Galaxy S25"
    assert "Galaxy S25" in dlg.result["items"]["camera"]["reason"] and "Integrated" not in dlg.result["items"]["camera"]["reason"]
    dlg.close()
    live.worker = None


def test_pressing_rig_check_twice_keeps_one_window_and_runs_it_again(win):
    """T9: two windows would share the engine's one beep test; the second press raises and re-runs the open one."""
    live = win.live
    live.open_rig_check()
    first = live._rig_dialog
    assert first.isVisible()
    runs = []
    first.run = lambda: runs.append(1)
    live.open_rig_check()
    assert live._rig_dialog is first and runs == [1]
    assert len([w for w in win.findChildren(RigCheckDialog) if w.isVisible()]) == 1
    first.close()
    live.open_rig_check()                                                                     # closed: a new one is fine
    assert live._rig_dialog is not first and live._rig_dialog.isVisible()
    live._rig_dialog.close()
