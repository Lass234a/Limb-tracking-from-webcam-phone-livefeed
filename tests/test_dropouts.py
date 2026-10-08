"""CR-05, CAM-02: a camera that goes quiet for a moment does not end the session; gaps in a recording are measured."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import VideoWorker  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


class FakeLive:
    """A live camera stand-in: a script of frames and None (a failed read)."""

    is_file = False
    is_live = True
    source = 0
    requested = {"width": 1280, "height": 720, "fps": 30}
    description = "fake"
    _granted = {"width": 1280, "height": 720, "fps": 30.0, "fourcc": "MJPG"}

    def __init__(self, script, recover_on_reopen=False):
        self.script = list(script)
        self.failed_reads = 0
        self.reconnects = 0
        self.position = 0
        self.recover_on_reopen = recover_on_reopen
        self.t = 0.0

    def read(self):
        if not self.script:
            return None, None
        item = self.script.pop(0)
        if item is None:
            return None, None
        self.t += 1 / 30
        self.position += 1
        return item, self.t

    def reopen(self):
        self.reconnects += 1
        return True

    def info(self):
        return {"failed_reads": self.failed_reads, "reconnects": self.reconnects}

    def measured_fps(self):
        return 30.0

    def close(self):
        pass


class StubEngine:
    rec = None
    mirror = False

    def __init__(self):
        self.n = 0
        self.camera_info = None
        self.watch, self.notes = None, []

    def process(self, frame, t, source_frame=None):
        self.n += 1
        if self.watch is not None:
            self.notes.append(self.watch.status_note)

        class R:
            overlay = np.zeros((10, 10, 3), np.uint8)
        return R()


def run(script, **timing):
    eng = StubEngine()
    w = VideoWorker(eng, 0)
    for k, v in timing.items():
        setattr(w, k, v)
    cam = FakeLive(script)
    msg = w._loop(cam, None, (None, None))
    return msg, eng, cam, w


FRAME = np.zeros((10, 10, 3), np.uint8)


def test_a_short_silence_does_not_end_the_session():
    msg, eng, cam, w = run([FRAME] * 5 + [None] * 20 + [FRAME] * 5 + [None] * 10_000, GIVE_UP_AFTER_S=0.4, REOPEN_AFTER_S=10.0)
    assert eng.n == 10                                   # frames before AND after the silence were processed
    assert cam.failed_reads >= 20 and eng.camera_info["failed_reads"] >= 20     # counted and kept for meta.json
    assert msg == "The camera stopped delivering frames."    # only the long silence at the end finishes the session


def test_a_long_silence_makes_it_reconnect_and_then_give_up():
    msg, eng, cam, w = run([FRAME] * 3 + [None] * 100_000, REOPEN_AFTER_S=0.1, GIVE_UP_AFTER_S=0.5)
    assert cam.reconnects == 1                           # one attempt per silence
    assert msg == "The camera stopped delivering frames." and eng.n == 3
    assert "waiting" in w.status_note or "reconnect" in w.status_note


def test_recovery_after_a_reconnect_clears_the_warning():
    script = [FRAME] * 2 + [None] * 5000
    cam = FakeLive(script)
    eng = StubEngine()
    w = VideoWorker(eng, 0)
    w.REOPEN_AFTER_S, w.GIVE_UP_AFTER_S = 0.05, 5.0
    eng.watch = w

    import threading

    def revive():
        time.sleep(0.3)
        cam.script[:] = [FRAME] * 3                      # the camera comes back
        time.sleep(0.2)
        w._stop = True
    threading.Thread(target=revive, daemon=True).start()
    w._loop(cam, None, (None, None))
    assert eng.n == 5 and cam.reconnects >= 1
    assert eng.notes == [""] * 5                         # the warning was gone as soon as frames flowed again


# ---- gaps in the recording
def recorded(tmp_path, times):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    paths = eng.start_trial("P01")
    for i, t in enumerate(times):
        eng.process(render(PTS, seed=i % 5), t)
    eng.stop_trial()
    return paths


def test_a_hole_in_the_frames_is_listed_in_meta_and_shown_in_review(tmp_path):
    times = [i / 30 for i in range(30)] + [1.0 + 1.5 + i / 30 for i in range(30)]       # a 1.5 s hole after frame 29
    paths = recorded(tmp_path, times)
    gaps = json.loads(paths["meta.json"].read_text())["gaps"]
    assert gaps["count"] == 1 and gaps["max_s"] == pytest.approx(1.5 + 1 / 30, abs=0.01)
    assert gaps["list"][0]["frame"] == 30 and gaps["list"][0]["t_s"] == pytest.approx(1.0 + 1.5, abs=0.05)
    assert Trial(paths["meta.json"]).meta["gaps"]["count"] == 1


def test_a_steady_recording_has_no_gaps_and_a_slow_but_steady_one_neither(tmp_path):
    assert json.loads(recorded(tmp_path / "a", [i / 30 for i in range(40)])["meta.json"].read_text())["gaps"]["count"] == 0
    assert json.loads(recorded(tmp_path / "b", [i / 12 for i in range(40)])["meta.json"].read_text())["gaps"]["count"] == 0
