"""CR-06 (+CAM-03, COD-10): the videos are tagged with the source's own frame rate; CR-08: 'frame' is the video frame index."""
import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


def make(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    return eng


def test_repeated_redraws_and_steps_do_not_change_the_estimated_rate(tmp_path):
    eng = make(tmp_path)
    for i in range(20):
        eng.process(render(PTS, seed=i), i / 30)
    base = eng._fps()
    assert base == pytest.approx(30.0, rel=0.02)
    for _ in range(8):                                             # a paused file: every click redraws the same frame
        eng.process(render(PTS, seed=1), 19 / 30)
    assert eng._fps() == pytest.approx(base)
    for i in range(30):                                            # stepping back and forth
        eng.process(render(PTS, seed=1), (10 + (i % 2)) / 30)
    assert 1.0 <= eng._fps() <= 240.0


def test_the_source_rate_wins_over_the_estimate(tmp_path):
    eng = make(tmp_path)
    for i in range(10):
        eng.process(render(PTS, seed=i), i / 50)                   # frame times say 50 fps
    eng.set_source("demo.mp4", {"is_file": True}, lambda: 25.0)    # the file says 25
    assert eng._fps() == 25.0
    eng.set_source("camera", None, lambda: None)                   # nothing known: falls back to the estimate, never 0
    assert 1.0 <= eng._fps() <= 240.0
    eng.set_source("camera", None, lambda: 0.0)
    assert eng._fps() >= 1.0


def test_a_new_source_forgets_the_old_timing(tmp_path):
    eng = make(tmp_path)
    for i in range(10):
        eng.process(render(PTS, seed=i), 1000.0 + i / 30)          # file-like times
    eng.set_source("camera", None, None)
    for i in range(10):
        eng.process(render(PTS, seed=i), 5.0 + i / 30)             # a camera clock: no mixed-up span
    assert eng._fps() == pytest.approx(30.0, rel=0.05)


def test_the_videos_carry_the_source_rate_and_the_csv_frame_column_counts_from_zero(tmp_path):
    eng = make(tmp_path)
    eng.set_source("demo.mp4", {"is_file": True}, lambda: 24.0)
    for i in range(6):
        eng.process(render(PTS, seed=i), 0.1 + i / 30, source_frame=100 + i)
    for _ in range(5):
        eng.process(render(PTS, seed=7), 0.4)                      # redraws before recording must not count
    paths = eng.start_trial("P01")
    for i in range(8):
        eng.process(render(PTS, seed=i), 1.0 + i / 24, source_frame=200 + i)
    eng.stop_trial()
    assert cv2.VideoCapture(str(paths["raw.mp4"])).get(cv2.CAP_PROP_FPS) == pytest.approx(24.0, abs=0.1)
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert [int(r["frame"]) for r in rows] == list(range(8))        # the index in raw.mp4 / overlay.mp4
    assert [int(r["source_frame"]) for r in rows] == list(range(200, 208))
    assert list(rows[0])[-1] == "source_frame"
