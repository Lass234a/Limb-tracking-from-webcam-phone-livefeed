"""Pause for a break: tracking stops, a ghost shows where the dots were, and a dot comes back only onto its ghost circle.

Nothing here changes a measured number: pausing is only possible when no trial is being recorded, and after the break every dot
has to be found again, inside a small fixed circle round where it was, before it is tracked (the existing lost-dot rule)."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402
from limbtrack.tracker import DotTracker, to_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)


def ready_engine(tmp_path, lock=True):
    """An engine with every dot marked and (by default) the position locked. Returns it and the next free time."""
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2)), n
    t = 0.05
    for i in range(6):
        eng.process(render(PTS, seed=i + 1), t)
        t += 1 / 30
    if lock:
        assert eng.lock_position()
    return eng, t


def empty_room(seed=0):
    """Nobody in the picture: the participant has left for a break."""
    rng = np.random.default_rng(seed)
    img = np.full((720, 1280, 3), 90.0) + rng.normal(0, 3, (720, 1280, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def run_empty(eng, t, n=10):
    res = None
    for i in range(n):
        res = eng.process(empty_room(i), t)
        t += 1 / 30
    return res, t


# ------------------------------------------------------------------------------------------ the pause itself
def test_pause_stops_tracking_and_nothing_is_flagged_while_the_participant_is_away(tmp_path):
    eng, t = ready_engine(tmp_path)
    before = {n: (tr.x, tr.y, tr.area) for n, tr in eng.trackers.trackers.items() if n in eng.active_dots}
    n_points = len(eng.history()["t"])
    assert eng.pause() is None and eng.paused
    res, t = run_empty(eng, t, 15)
    assert any("PAUSED" in h for h in res.header)
    assert not any("LOST" in h for h in res.header)                              # no alarm for dots nobody is wearing
    assert res.angles == []                                                      # no angle from a paused picture
    after = {n: (tr.x, tr.y, tr.area) for n, tr in eng.trackers.trackers.items() if n in eng.active_dots}
    assert after == before                                                       # the trackers were not touched
    assert len(eng.history()["t"]) == n_points                                   # no new points in the angle trace
    assert eng.locked                                                            # the locked position survives the break


def test_the_ghost_shows_where_every_dot_was(tmp_path):
    eng, t = ready_engine(tmp_path)
    shown_before = {n: (eng.trackers.trackers[n].x, eng.trackers.trackers[n].y) for n in eng.active_dots}
    assert eng.pause_ghost is None
    eng.pause()
    ghost = eng.pause_ghost
    assert set(ghost) == set(eng.active_dots)
    for n, (x, y, diam) in ghost.items():
        assert (x, y) == pytest.approx(shown_before[n]) and diam > 0
    room = empty_room(0)
    res = eng.process(room, t)
    x, y, _ = ghost["malleolus"]
    r = 30
    ring = np.abs(res.overlay[int(y) - r:int(y) + r, int(x) - r:int(x) + r].astype(int) - room[int(y) - r:int(y) + r, int(x) - r:int(x) + r].astype(int))
    assert ring.max() > 60                                                       # a circle and the dot's name are drawn there
    far = (slice(600, 700), slice(900, 1200))                                    # nowhere near a dot or the text
    assert np.array_equal(res.overlay[far], room[far])


def test_a_dot_marked_late_is_in_the_ghost_only_if_it_was_marked(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), 0.0)
    assert eng.mark_dot("epicondyle", *(PTS["epicondyle"] + 2))
    eng.process(render(PTS, seed=1), 0.05)
    assert eng.pause() is None
    assert set(eng.pause_ghost) == {"epicondyle"}


# ------------------------------------------------------------------------------------------ coming back
def test_back_in_the_same_place_every_dot_is_found_again_and_the_numbers_are_the_same(tmp_path):
    eng, t = ready_engine(tmp_path)
    before = {a.name: a.raw_value for a in eng.process(render(PTS, seed=30), t).angles}
    t += 1 / 30
    eng.pause()
    _, t = run_empty(eng, t, 20)
    assert eng.resume() is None and not eng.paused
    assert eng.pause_ghost is not None                                           # the ghost stays until the dots are back
    res = eng.process(render(PTS, seed=31), t)
    assert not any(d.lost for d in res.dots.values())
    assert eng.pause_ghost is None                                               # all dots back: the ghost goes
    for a in res.angles:
        assert a.raw_value == pytest.approx(before[a.name], abs=0.1), a.name
    assert eng.locked


def test_a_participant_who_comes_back_in_another_place_is_not_tracked_until_the_dots_are_on_the_ghost(tmp_path):
    eng, t = ready_engine(tmp_path)
    eng.pause()
    _, t = run_empty(eng, t)
    eng.resume()
    shift = np.array([0.0, 150.0])
    moved = {k: v + shift for k, v in PTS.items()}
    old = np.array([PTS[n] for n in eng.active_dots])
    assert all(np.hypot(*(old - m).T).min() > 40 for m in moved.values())       # no dot lands near another dot's old place
    for i in range(4):
        res = eng.process(render(moved, seed=i), t)
        t += 1 / 30
    assert all(d.lost for d in res.dots.values())                                # nothing is taken for a dot it is not
    assert any("DOT LOST" in h for h in res.header) and eng.pause_ghost is not None
    assert all(a.status == "lost" for a in res.angles if a.name in ("knee", "hip"))
    res = eng.process(render(PTS, seed=9), t)                                    # now the stickers are back on the circles
    assert not any(d.lost for d in res.dots.values()) and eng.pause_ghost is None


def test_the_ghost_goes_when_the_dots_are_marked_again(tmp_path):
    eng, t = ready_engine(tmp_path)
    eng.pause()
    eng.resume()
    assert eng.pause_ghost is not None
    eng.reset_dots()
    assert eng.pause_ghost is None and not eng.paused


# ------------------------------------------------------------------------------------------ what a break must not allow
def test_no_pause_while_recording_and_no_recording_locking_or_marking_while_paused(tmp_path):
    eng, t = ready_engine(tmp_path)
    eng.start_trial("P01")
    assert "recording" in eng.pause().lower() and not eng.paused
    eng.stop_trial()

    assert eng.pause() is None
    with pytest.raises(RuntimeError, match="paused"):
        eng.start_trial("P01")
    assert eng.lock_position() is False and "paused" in eng.lock_problem().lower()
    assert eng.mark_dot("epicondyle", *(PTS["epicondyle"] + 2)) is False and "Resume" in eng.mark_problem
    assert eng.pause() is None                                                   # pausing twice is harmless
    assert eng.resume() is None
    assert eng.resume() is None                                                  # so is resuming twice
    eng.process(render(PTS, seed=3), t)
    eng.start_trial("P01")
    eng.stop_trial()


def test_the_ghost_is_not_in_the_recording(tmp_path):
    eng, t = ready_engine(tmp_path)
    eng.pause()
    eng.resume()
    moved = {k: v + np.array([0.0, 150.0]) for k, v in PTS.items()}
    eng.process(render(moved, seed=1), t)                                        # dots lost: the ghost is still up
    assert eng.pause_ghost is not None
    eng.start_trial("P01")
    assert eng.pause_ghost is None                                               # a trial's overlay video must not carry it
    eng.stop_trial()


# ------------------------------------------------------------------------------------------ the tracker's part
def test_hold_freezes_a_dot_so_it_comes_back_only_close_to_where_it_was():
    t = DotTracker("d", "white")
    assert t.init_at(to_gray(render(PTS, seed=1)), *(PTS["epicondyle"] + 2))
    x0, y0 = t.x, t.y
    t.hold()
    far = {k: v.copy() for k, v in PTS.items()}
    far["epicondyle"] = PTS["epicondyle"] + np.array([60.0, 0.0])
    st = t.update(to_gray(render(far, seed=2)))
    assert st.lost and (t.x, t.y) == (x0, y0)                                    # frozen where it was
    near = {k: v.copy() for k, v in PTS.items()}
    near["epicondyle"] = PTS["epicondyle"] + np.array([10.0, 0.0])
    st = t.update(to_gray(render(near, seed=3)))
    assert not st.lost and abs(t.x - near["epicondyle"][0]) < 0.5 and abs(t.y - near["epicondyle"][1]) < 0.5


# ------------------------------------------------------------------------------------------ the window
class RunningCamera:
    """Stands in for a running camera so the window thinks a live source is open."""
    is_file = False
    paused = False
    at_end = False

    def stop(self):
        pass

    def wait(self, ms=0):
        return True

    def isRunning(self):
        return True


def test_the_pause_button_is_a_break_pause_for_a_camera_and_the_ghost_follows(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    assert not live.btn_pause.isEnabled()                                        # no source: nothing to pause
    live.worker = RunningCamera()
    live._set_file_controls(False)
    assert live.btn_pause.isEnabled() and not live.btn_back.isEnabled() and "Pause" in live.btn_pause.text()
    live.btn_pause.click()
    assert win.engine.paused and "Resume" in live.btn_pause.text() and live.btn_pause.isChecked()
    live.btn_pause.click()
    assert not win.engine.paused and "Pause" in live.btn_pause.text() and not live.btn_pause.isChecked()
    live.btn_pause.click()                                                       # stopping the camera ends the break
    live.worker = None
    live._set_file_controls(False)
    assert not win.engine.paused and not live.btn_pause.isEnabled() and not live.btn_pause.isChecked()
    win.close()


def test_the_pause_button_says_why_when_recording(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    shown = []
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    live.worker = RunningCamera()
    live._set_file_controls(False)
    win.engine.rec = object()                                                    # "a trial is being recorded"
    live.btn_pause.click()
    assert shown and "recording" in shown[0].lower()
    assert not win.engine.paused and not live.btn_pause.isChecked()
    win.engine.rec = None
    live.worker = None
    win.close()
