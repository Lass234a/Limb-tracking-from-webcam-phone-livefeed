"""Sound cues for an MVIC: the plan (pure), the generated tones, the player, and the engine hooks. No sound is ever made here:
the player is given a backend that only writes the names down."""
import io
import json
import os
import time
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import cues  # noqa: E402
from limbtrack.cues import CUES, CuePlayer, duration_s, plan, wav_bytes  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


def names(items):
    return [n for _, n in items]


# ------------------------------------------------------------------------------------------ the plan
def test_the_protocol_tempo_gives_ramp_hold_ticks_and_relax():
    items = plan(ref=100.0, now=100.0, length_s=7.0)
    assert items == [(100.0, "ramp"), (101.0, "hold"), (102.0, "push"), (103.0, "push"), (104.0, "push"), (105.0, "push"), (106.0, "relax")]


def test_an_onset_estimate_before_the_registration_moves_the_later_cues_earlier():
    items = plan(ref=99.85, now=100.0, length_s=7.0)                        # the force began 0.15 s before the ramp was registered
    assert items[0] == (100.0, "ramp")
    assert items[1][0] == pytest.approx(100.85) and items[1][1] == "hold"
    assert items[-1][0] == pytest.approx(105.85) and items[-1][1] == "relax"


def test_no_ticks_when_the_interval_is_zero_and_a_faster_tick_when_asked():
    assert "push" not in names(plan(100.0, 100.0, 7.0, push_every_s=0))
    assert names(plan(100.0, 100.0, 7.0, push_every_s=0.5)).count("push") == 9      # 1.5, 2.0 ... 5.5 (not within 0.25 s of the relax)


def test_a_cue_that_is_long_past_is_dropped_and_no_cue_starts_on_top_of_another():
    late = plan(ref=98.5, now=100.0, length_s=7.0)                          # the onset estimate is 1.5 s old: the hold was due 0.5 s ago
    assert "hold" not in names(late) and names(late)[0] == "ramp"
    close = plan(ref=99.1, now=100.0, length_s=7.0)                         # the hold would fall 0.1 s after the ramp began, inside its 0.2 s
    (t_ramp, _), (t_hold, n) = close[0], close[1]
    assert n == "hold" and t_hold >= t_ramp + duration_s("ramp") + cues.GAP_S - 1e-9
    for (t1, n1), (t2, _) in zip(close, close[1:]):
        assert t2 >= t1 + duration_s(n1) - 1e-9


def test_a_short_contraction_has_no_hold_cues():
    items = plan(100.0, 100.0, length_s=2.0)                                # ramp 1 s, relax 1 s: nothing in between
    assert names(items) == ["ramp", "relax"] and items[1][0] == pytest.approx(101.0)


# ------------------------------------------------------------------------------------------ the tones
@pytest.mark.parametrize("name", sorted(CUES))
def test_every_cue_is_a_valid_short_wav_without_clicks(name):
    with wave.open(io.BytesIO(wav_bytes(name))) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == cues.SAMPLE_RATE
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert len(frames) / cues.SAMPLE_RATE == pytest.approx(duration_s(name), abs=0.01)
    assert duration_s(name) < 0.5                                           # every cue is short
    assert abs(int(frames[0])) <= 1 and abs(int(frames[-1])) <= 1           # it starts and ends at silence: no click
    assert 0.3 * 32767 < np.abs(frames).max() < 0.6 * 32767                 # audible, not at full scale


def test_the_cues_can_be_told_apart_by_their_pitch_pattern():
    assert [f for f, _ in CUES["ramp"]] == sorted(f for f, _ in CUES["ramp"])                        # rising
    assert [f for f, _ in CUES["relax"]] == sorted((f for f, _ in CUES["relax"]), reverse=True)      # falling
    assert max(ms for _, ms in CUES["push"]) < min(ms for _, ms in CUES["hold"])                     # a tick is shorter than the hold beep
    assert CUES["end"][0][0] < min(f for f, _ in CUES["record"])                                     # the end is the lowest


def test_nothing_is_played_on_a_machine_without_a_screen():
    assert cues.silent()                                                    # these tests run offscreen: no sound can leave the computer
    cues._play_with_windows("ramp")                                         # (returns at once, no error)


# ------------------------------------------------------------------------------------------ the player
def test_the_player_plays_cues_in_order_at_their_time_and_can_be_cancelled():
    heard = []
    player = CuePlayer(backend=heard.append)
    now = time.perf_counter()
    player.schedule([(now + 0.30, "hold"), (now + 0.10, "ramp"), (now + 5.0, "relax")])
    time.sleep(0.6)
    assert heard == ["ramp", "hold"]                                        # in time order, the far one still waiting
    for name, due, played in player.played:
        assert 0 <= played - due < 0.08, (name, played - due)              # on time, whatever the camera does
    assert names(player.pending()) == ["relax"]
    player.cancel()
    assert player.pending() == []
    player.play("end")
    time.sleep(0.15)
    assert heard[-1] == "end"
    player.close()


def test_a_failing_sound_does_not_stop_the_player():
    heard = []

    def backend(name):
        heard.append(name)
        if name == "ramp":
            raise OSError("no sound card")
    player = CuePlayer(backend=backend)
    now = time.perf_counter()
    player.schedule([(now + 0.05, "ramp"), (now + 0.15, "hold")])
    time.sleep(0.4)
    assert heard == ["ramp", "hold"]
    player.close()


# ------------------------------------------------------------------------------------------ the engine
def ready_engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    heard = []
    eng._cues = CuePlayer(backend=lambda name: heard.append((name, time.perf_counter())))
    return eng, heard


def run(eng, seconds):
    for i in range(int(seconds * 30)):
        eng.process(render(PTS, seed=i % 5), time.perf_counter())
        time.sleep(1 / 30)


def test_a_marked_contraction_is_cued_from_record_to_the_end_marker(tmp_path):
    eng, heard = ready_engine(tmp_path)
    eng.auto_mvic_seconds = 3.0                                             # ramp 1 s, hold 1 s, relax 1 s
    eng.cue_push_every_s = 0.0
    paths = eng.start_trial("P01")
    t_record = time.perf_counter()
    run(eng, 0.5)
    eng.add_event("MVIC start")                                             # the S key
    t_start = time.perf_counter()
    run(eng, 2.6)
    eng.add_event("MVIC end")                                               # the E key
    run(eng, 0.4)
    eng.stop_trial()
    time.sleep(0.1)
    order = [n for n, _ in heard]
    assert order == ["record", "ramp", "hold", "relax", "end"], order
    at = {n: t for n, t in heard}
    assert at["record"] - t_record < 0.2
    assert 0 <= at["ramp"] - t_start < 0.2                                  # on the picture that carries the marker
    assert at["hold"] - at["ramp"] == pytest.approx(1.0, abs=0.12)          # the ramp is over one second after it began
    assert at["relax"] - at["ramp"] == pytest.approx(2.0, abs=0.12)         # the relax begins one second before the 3 s are over
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["audio_cues"]["enabled"] is True and meta["audio_cues"]["length_s"] == 3.0 and meta["audio_cues"]["push_every_s"] == 0.0
    assert "changed_during_recording" not in meta


def test_the_motivating_ticks_come_every_second_during_the_hold(tmp_path):
    eng, heard = ready_engine(tmp_path)
    eng.auto_mvic_seconds = 5.0                                             # ramp 1, hold 3, relax 1
    eng.cue_push_every_s = 1.0
    eng.start_trial("P01")
    eng.add_event("MVIC start")
    run(eng, 4.4)
    eng.stop_trial()
    order = [n for n, _ in heard]
    assert order == ["record", "ramp", "hold", "push", "push", "relax"], order       # ticks at +2 s and +3 s; none within 0.25 s of the relax


def test_stopping_the_recording_silences_the_cues_still_to_come(tmp_path):
    eng, heard = ready_engine(tmp_path)
    eng.start_trial("P01")
    eng.add_event("MVIC start")
    run(eng, 0.3)
    assert names(eng._cues.pending())                                       # hold, ticks and relax are waiting
    eng.stop_trial()
    assert eng._cues.pending() == []
    n = len(heard)
    time.sleep(1.5)
    assert len(heard) == n                                                  # the hold beep never came


def test_switched_off_nothing_is_played_and_meta_says_so(tmp_path):
    eng, heard = ready_engine(tmp_path)
    eng.audio_cues = False
    paths = eng.start_trial("P01")
    eng.add_event("MVIC start")
    run(eng, 1.5)
    eng.stop_trial()
    assert heard == []
    assert json.loads(paths["meta.json"].read_text())["audio_cues"] == {"enabled": False}


def test_the_sound_settings_cannot_change_while_recording(tmp_path):
    eng, heard = ready_engine(tmp_path)
    eng.start_trial("P01")
    with pytest.raises(RuntimeError, match="sound cues"):
        eng.audio_cues = False
    with pytest.raises(RuntimeError, match="ticks"):
        eng.cue_push_every_s = 2.0
    eng.stop_trial()
    eng.audio_cues = False


def test_an_automatically_registered_ramp_is_cued_from_the_force_onset(tmp_path):
    from limbtrack.force import SimulatedForce
    eng, heard = ready_engine(tmp_path)
    force = SimulatedForce(rest_s=4.0, ramp_s=0.6, hold_s=2.0, peak_n=400.0, period_s=30.0)
    force.start()
    eng.force = force
    time.sleep(1.5)
    eng.auto_mvic, eng.auto_mvic_seconds = True, 3.0
    eng.cue_push_every_s = 0.0
    eng.start_trial("P01")
    run(eng, 6.5)
    eng.stop_trial()
    force.stop()
    order = [n for n, _ in heard]
    assert order == ["record", "ramp", "hold", "relax", "end"], order
    at = {n: t for n, t in heard}
    assert at["relax"] - at["ramp"] == pytest.approx(2.0, abs=0.35)         # counted from the onset, so a little under 2 s after the registration


# ------------------------------------------------------------------------------------------ the window
def test_the_record_box_has_the_sound_controls(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    assert live.chk_cues.isChecked() and win.engine.audio_cues
    assert live.cue_push_spin.value() == cues.PUSH_INTERVAL_S and live.cue_push_spin.isEnabled()
    live.chk_cues.setChecked(False)
    assert not win.engine.audio_cues and not live.cue_push_spin.isEnabled()
    live.chk_cues.setChecked(True)
    live.cue_push_spin.setValue(2.0)
    assert win.engine.cue_push_every_s == 2.0
    heard = []
    win.engine._cues = CuePlayer(backend=heard.append)
    live.btn_cue_test.click()
    time.sleep(0.2)
    assert heard == ["record"]                                              # the first beep is at once ...
    assert names(win.engine._cues.pending()) == ["ramp", "hold", "push", "push", "relax", "end", "rest"]     # ... the rest follow in the order of a contraction (0.27.0: then the rest-over beep)
    win.engine._cues.cancel()
    win.engine.rec = object()                                               # a trial is being recorded: the box refuses and says why
    live.chk_cues.setChecked(False)
    assert live.chk_cues.isChecked() and win.engine.audio_cues
    win.engine.rec = None
    win.close()


# ------------------------------------------------------------------------------------------ the real sound call (no sound is made)
class FakeWinsound:
    """Behaves like Python's winsound where it matters: an in-memory sound cannot be played asynchronously."""
    SND_ASYNC, SND_MEMORY, SND_PURGE = 1, 4, 64

    def __init__(self):
        self.calls = []

    def PlaySound(self, sound, flags):
        if flags & self.SND_MEMORY and flags & self.SND_ASYNC:
            raise RuntimeError("Cannot play asynchronously from memory")
        self.calls.append((sound, flags))


def test_the_real_sound_call_is_made_the_way_windows_allows(monkeypatch):
    """The first version used SND_MEMORY | SND_ASYNC, which Python refuses (RuntimeError), and the error was swallowed: no beep was
    ever heard on the real computer, because every other test runs silent. This pins the call itself."""
    import sys
    fake = FakeWinsound()
    monkeypatch.setitem(sys.modules, "winsound", fake)
    monkeypatch.setattr(cues, "silent", lambda: False)
    for name in sorted(CUES) + ["silence"]:
        cues._play_with_windows(name)                                        # would raise with the old flags
    assert [c[0] for c in fake.calls] == [wav_bytes(n) for n in sorted(CUES) + ["silence"]]
    assert all(flags & fake.SND_MEMORY and not flags & fake.SND_ASYNC for _, flags in fake.calls)


def test_python_really_refuses_an_in_memory_sound_played_asynchronously():
    winsound = pytest.importorskip("winsound")                              # (Windows only; this call raises before any sound starts)
    with pytest.raises(RuntimeError, match="asynchronously from memory"):
        winsound.PlaySound(wav_bytes("push"), winsound.SND_MEMORY | winsound.SND_ASYNC)


def test_a_sound_that_cannot_be_played_is_kept_as_an_error_not_hidden():
    def broken(name):
        raise RuntimeError("Cannot play asynchronously from memory")
    player = CuePlayer(backend=broken)
    assert player.last_error is None and player.error_count == 0
    now = time.perf_counter()
    player.schedule([(now + 0.02, "ramp"), (now + 0.05, "hold")])
    time.sleep(0.3)
    assert player.last_error == "RuntimeError: Cannot play asynchronously from memory" and player.error_count == 2
    player.close()


def test_a_recording_says_in_meta_that_a_beep_could_not_be_played(tmp_path):
    eng, heard = ready_engine(tmp_path)

    def broken(name):
        raise OSError("no sound device")
    eng._cues = CuePlayer(backend=broken)
    paths = eng.start_trial("P01")
    time.sleep(0.2)
    assert "no sound device" in eng.cue_problem
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["audio_cue_problem"]["error"] == "OSError: no sound device" and meta["audio_cue_problem"]["times"] >= 1
    assert "changed_during_recording" not in meta


def test_the_window_reports_a_beep_problem_and_clears_it_when_it_is_gone(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    win.engine._cues.last_error = "RuntimeError: Cannot play asynchronously from memory"
    live.report_beep_problem()
    assert "could not be played" in live.instruction.text() and "Cannot play" in live.instruction.text()
    win.engine._cues.last_error = None
    live.report_beep_problem()
    assert "played without error" in live.instruction.text()
    win.close()


def test_the_test_button_starts_afresh_so_an_old_error_cannot_hide_a_new_success(tmp_path):
    eng, heard = ready_engine(tmp_path)
    eng._cues.last_error, eng._cues.error_count = "OSError: old", 3
    eng.test_cues()
    assert eng.cue_problem is None and eng._cues.error_count == 0
    eng._cues.cancel()


# ------------------------------------------------------------------------------------------ the sound device is woken before the first real beep
def test_a_silent_warm_up_is_played_once_and_not_again_while_the_device_is_awake():
    heard = []
    player = CuePlayer(backend=heard.append)
    player.warm_up()
    player.warm_up()                                                        # a second call at once does nothing
    time.sleep(0.2)
    assert heard == ["silence"]
    assert "silence" not in [n for n, _, _ in player.played]                # the diagnostics list holds real cues only
    player.play("ramp")
    time.sleep(0.2)
    player.warm_up()                                                        # a cue was just played: the device is awake
    time.sleep(0.2)
    assert heard == ["silence", "ramp"]
    player.close()


def test_the_warm_up_is_a_silent_buffer_and_the_engine_asks_for_it_only_when_the_beeps_are_on(tmp_path):
    with wave.open(io.BytesIO(wav_bytes("silence"))) as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert len(frames) / cues.SAMPLE_RATE == pytest.approx(cues.SILENCE_MS / 1000.0, abs=0.002) and not frames.any()
    eng, heard = ready_engine(tmp_path)
    eng.warm_up_cues()
    time.sleep(0.15)
    assert [n for n, _ in heard] == ["silence"]
    eng.audio_cues = False
    eng._cues._last_sound = None
    eng.warm_up_cues()
    time.sleep(0.15)
    assert [n for n, _ in heard] == ["silence"]                             # switched off: nothing more


def test_the_plan_leaves_room_for_the_sound_calls_own_overhead():
    assert cues.GAP_S >= 0.1                                                # a measured call returns about 0.1 s after the cue ends
    close = plan(ref=99.1, now=100.0, length_s=7.0)
    assert close[1][0] >= close[0][0] + duration_s("ramp") + 0.1
