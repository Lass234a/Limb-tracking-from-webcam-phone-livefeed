"""'Recording first' (0.21.0): while recording, the window does less (at most 10 pictures a second on the screen, the trace and the
status panel once a second), so the picture thread tracks and saves more pictures per second.

Found on 2026-10-27 by replaying a P11 trial (7 dots, force, automatic MVIC) through the real window on battery: the program's own
work was 25 ms per picture without the window and 52 ms with it, because the window's trace (5 redraws a second, ~40 ms each), status
panel (5 a second, ~29 ms each) and picture (converted and painted for every picture) share Python with the tracking."""
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow, VideoWorker  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
FRAME = np.zeros((10, 10, 3), np.uint8)


class StubEngine:
    mirror = False

    def __init__(self, recording, first):
        self.rec = object() if recording else None
        self.recording_first = first
        self.n = 0

    def process(self, frame, t, source_frame=None):
        self.n += 1

        class R:
            overlay = FRAME
        return R()


class Cam:
    is_file = False
    position = 0


def shown_of(recording, first, fps=30.0, seconds=3.0):
    """(pictures processed, pictures sent to the screen) for `seconds` of pictures arriving at `fps`."""
    eng = StubEngine(recording, first)
    w = VideoWorker(eng, 0)
    shown = []
    w.frame_ready.connect(lambda img: shown.append(img))
    for k in range(int(fps * seconds)):
        w._show(FRAME, 100.0 + k / fps, Cam())
    return eng.n, len(shown)


def test_while_recording_first_every_picture_is_processed_but_ten_a_second_are_shown():
    processed, shown = shown_of(recording=True, first=True)
    assert processed == 90                       # tracked and saved: all of them
    assert shown == 30                           # on the screen: every third picture at 30 per second


def test_slower_pictures_are_shown_at_most_ten_a_second_and_never_fewer_than_every_other():
    assert shown_of(recording=True, first=True, fps=17.0) == (51, 26)       # 17 per second (P11 replay): every other picture
    assert shown_of(recording=True, first=True, fps=8.0) == (24, 24)        # slower than 10 per second: every picture


@pytest.mark.parametrize("recording, first", [(False, True), (True, False), (False, False)])
def test_otherwise_every_picture_is_shown(recording, first):
    assert shown_of(recording, first) == (90, 90)


def test_the_setting_cannot_change_during_a_recording_and_is_saved_with_the_timing(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    assert eng.recording_first is True                                      # on unless switched off
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(5):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    paths = eng.start_trial("P01")
    with pytest.raises(RuntimeError, match="Stop recording"):
        eng.recording_first = False
    for i in range(5):
        eng.process(render(PTS, seed=i), 0.3 + i / 30)
    eng.stop_trial()
    import json
    meta = json.loads(paths["meta.json"].read_text(encoding="utf-8"))
    assert meta["pipeline_timing"]["recording_first"] is True
    eng.recording_first = False                                             # allowed again afterwards
    assert eng.recording_first is False


def recording_window(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(TESTS, tmp_path)
    live, eng = win.live, win.engine
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(5):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    return app, win, live, eng


def test_the_window_refreshes_the_panel_and_the_trace_once_a_second_while_recording_first(tmp_path, monkeypatch):
    app, win, live, eng = recording_window(tmp_path)
    assert live.chk_rec_first.isChecked() and eng.recording_first
    assert live.trace_timer.interval() == 200 and live.timer.interval() == 200
    full = []
    monkeypatch.setattr(live, "update_steps", lambda: full.append(1))
    eng.start_trial("P01")
    live._start_recording_view()
    live._rec_started = time.perf_counter()
    assert live.trace_timer.interval() == 1000
    live.refresh_status()
    assert len(full) == 1 and not live.test_combo.isEnabled()      # the first tick in full: what must not change is greyed out at once
    for _ in range(9):
        live.refresh_status()
    assert len(full) == 2                                          # then once in five ticks (once a second)
    assert "RECORDING" in live.rec_label.text()                    # the recording line: every tick
    assert not live.chk_rec_first.isEnabled()                      # fixed during a recording, like the overlay video
    eng.stop_trial()
    live._rec_started = None
    live._end_recording_view()
    assert live.trace_timer.interval() == 200
    full.clear()
    for _ in range(5):
        live.refresh_status()
    assert len(full) == 5 and live.chk_rec_first.isEnabled()
    win.close()


def test_switched_off_the_window_works_as_before_and_the_choice_is_remembered(tmp_path, monkeypatch):
    app, win, live, eng = recording_window(tmp_path)
    live.chk_rec_first.setChecked(False)
    assert eng.recording_first is False
    full = []
    monkeypatch.setattr(live, "update_steps", lambda: full.append(1))
    eng.start_trial("P01")
    live._start_recording_view()
    live._rec_started = time.perf_counter()
    assert live.trace_timer.interval() == 200
    for _ in range(5):
        live.refresh_status()
    assert len(full) == 5
    eng.stop_trial()
    live._rec_started = None
    live._end_recording_view()
    assert live.session_state()["recording_first"] is False
    live.save_session()
    win.close()
    win2 = MainWindow(TESTS, tmp_path)                                      # the next session starts with the same choice
    assert not win2.live.chk_rec_first.isChecked() and win2.engine.recording_first is False
    win2.close()
