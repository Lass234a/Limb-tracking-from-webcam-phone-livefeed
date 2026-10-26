"""DAT-01 (low-risk part), DAT-06/CAM-10/COD-03, COD-04: a recording survives crashes, size changes and failures while saving."""
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.recorder import FrameSizeChanged  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


def ready_engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(3):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    assert eng.lock_position()
    return eng


def test_csv_rows_reach_the_disk_one_by_one_and_a_complete_meta_exists_from_the_start(tmp_path):
    eng = ready_engine(tmp_path)
    paths = eng.start_trial("P01")
    meta = json.loads(paths["meta.json"].read_text())                       # exists before a single frame is written
    assert meta["status"] == "recording" and meta["test_key"] == "knee_extension" and meta["angles"] and meta["frames"] is None
    for i in range(1, 8):
        eng.process(render(PTS, seed=i), 0.2 + i / 30)
        rows = list(csv.reader(open(paths["data.csv"], newline="")))
        assert len(rows) - 1 == i                                            # the row is already on disk (header + i rows)
    eng.stop_trial()
    final = json.loads(paths["meta.json"].read_text())
    assert final["status"] == "complete" and final["frames"] == 7
    assert not list(tmp_path.glob("**/*.tmp"))                               # the atomic write left nothing behind


def test_two_trials_started_in_the_same_second_never_write_over_each_other(tmp_path, monkeypatch):
    """The file name has the time to the second; since 0.20.0 it also has the trial number (T1, T2, T3), which keeps trials of one
    position in the same second apart. The '-2', '-3' of 0.17.1 is still there for a name that is taken anyway (shown below)."""
    from datetime import datetime
    import limbtrack.recorder as recorder

    class Frozen:                                                             # every trial starts at 12:00:00
        @staticmethod
        def now():
            return datetime(2026, 10, 21, 12, 0, 0)
    monkeypatch.setattr(recorder, "datetime", Frozen)
    eng = ready_engine(tmp_path)
    metas = []
    for k in range(3):
        paths = eng.start_trial("P01")
        eng.process(render(PTS, seed=k), 1.0 + k)
        eng.stop_trial()
        metas.append(paths["meta.json"])
    assert [m.name.split("deg_")[1] for m in metas] == ["T1_20261021-120000_meta.json", "T2_20261021-120000_meta.json",
                                                         "T3_20261021-120000_meta.json"]
    assert len(list((tmp_path / "P01").glob("*_meta.json"))) == 3
    assert all(json.loads(m.read_text())["frames"] == 1 for m in metas)                # each trial kept its own files
    for m in metas[1:]:                                                                # trial 2 and 3 removed: the next is trial 2 again,
        for f in (tmp_path / "P01").glob(m.name[: -len("_meta.json")] + "_*"):         # and its name is free
            f.unlink()
    (tmp_path / "P01" / metas[1].name.replace("_meta.json", "_raw.mp4")).write_bytes(b"")   # ... unless a video of that name was left
    paths = eng.start_trial("P01")
    eng.stop_trial()
    assert paths["meta.json"].name.endswith("T2_20261021-120000-2_meta.json")         # never written over


def test_a_trial_cut_short_can_still_be_opened_in_review(tmp_path):
    eng = ready_engine(tmp_path)
    paths = eng.start_trial("P01")
    for i in range(1, 12):
        eng.process(render(PTS, seed=i), 0.2 + i / 30)
    # the program "dies" here: stop_trial is never called, the videos are never finalised
    tr = Trial(paths["meta.json"])
    assert tr.unfinished and tr.meta["frames"] == 11 and len(tr.rows) == 11
    assert tr.summary(float(tr.t[0]), float(tr.t[-1]))                       # statistics work on the rows that were saved
    tr.frame(3)                                                              # an unreadable video gives no picture, no crash
    eng.stop_trial()


def test_a_picture_size_change_stops_and_saves_the_recording_and_asks_for_the_dots_again(tmp_path):
    eng = ready_engine(tmp_path)
    paths = eng.start_trial("P01")
    for i in range(1, 10):
        eng.process(render(PTS, seed=i), 0.2 + i / 30)
    small = cv2.resize(render(PTS, seed=11), (960, 540))
    res = eng.process(small, 0.6)
    assert eng.rec is None and "picture size changed from 1280x720 to 960x540" in eng.last_stop_reason
    assert any("mark the dots" in line for line in res.header)
    assert not eng.trackers.is_ready()                                       # the old positions are meaningless now
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["status"] == "complete" and meta["frames"] == 9 and "picture size changed" in meta["stopped_because"]
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    cap = cv2.VideoCapture(str(paths["raw.mp4"]))
    n = 0
    while cap.read()[0]:
        n += 1
    assert len(rows) == n == 9                                               # CSV rows and video frames still match


def test_a_size_change_without_recording_also_clears_the_dots(tmp_path):
    eng = ready_engine(tmp_path)
    eng.process(cv2.resize(render(PTS, seed=3), (960, 540)), 0.5)
    assert not eng.trackers.is_ready() and not eng.locked


def test_the_recorder_refuses_a_frame_of_another_size(tmp_path):
    eng = ready_engine(tmp_path)
    eng.start_trial("P01")
    odd = np.zeros((540, 960, 3), np.uint8)
    with pytest.raises(FrameSizeChanged):
        eng.rec.write(odd, odd, 0, 0.0, False, {}, [])
    eng.stop_trial()


def test_a_failure_while_closing_still_clears_the_recording_and_writes_meta(tmp_path):
    eng = ready_engine(tmp_path)
    paths = eng.start_trial("P01")
    eng.process(render(PTS, seed=5), 0.3)
    real = eng.rec._raw

    class FailsOnClose:
        def release(self):
            real.release()                                                   # (so the test leaves no file open)
            raise OSError("No space left on device")

    eng.rec._raw = FailsOnClose()
    eng.stop_trial()
    assert eng.rec is None                                                   # the next recording can start
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["status"] == "complete" and any("No space left" in e for e in meta["close_errors"])
    assert eng.start_trial("P01") is not None
    eng.stop_trial()


def test_the_window_notices_when_the_engine_ends_the_recording(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live, eng = win.live, win.engine
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(3):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    eng.lock_position()
    live.participant.setText("P01")
    eng.start_trial("P01")
    live._rec_started = time.perf_counter()
    live.btn_rec.setChecked(True)
    eng.process(cv2.resize(render(PTS, seed=4), (960, 540)), 0.5)             # the camera changes size
    live.refresh_status()
    assert live._rec_started is None and not live.btn_rec.isChecked()
    assert "RECORDING STOPPED" in live.rec_label.text() and "picture size changed" in live.rec_label.text()
    win.close()
