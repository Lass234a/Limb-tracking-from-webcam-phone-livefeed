"""CAM-07 (+COD-08, DAT-10): video encoding runs on its own thread, never drops a frame, and the overlay video is optional."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


def make(tmp_path, overlay=True):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.save_overlay = overlay
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    return eng


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    return n


def test_every_recorded_frame_reaches_both_videos_through_the_writer_thread(tmp_path):
    eng = make(tmp_path)
    paths = eng.start_trial("P01")
    n = 150                                                        # more than the queue holds: the producer waits, nothing is dropped
    for i in range(n):
        eng.process(render(PTS, seed=i % 7), 0.1 + i / 30)
    eng.stop_trial()
    assert count_frames(paths["raw.mp4"]) == count_frames(paths["overlay.mp4"]) == n
    assert len(list(csv.DictReader(open(paths["data.csv"], newline="")))) == n
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["video_writer"]["frames_encoded"] == n and meta["video_writer"]["overlay_video_saved"] is True
    assert 1 <= meta["video_writer"]["queue_peak"] <= meta["video_writer"]["queue_capacity"]


def test_close_waits_for_the_queued_frames(tmp_path):
    eng = make(tmp_path)
    paths = eng.start_trial("P01")
    for i in range(40):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    eng.stop_trial()                                               # right after the last frame: the queue may still hold some
    assert count_frames(paths["raw.mp4"]) == 40


def test_without_the_overlay_only_raw_video_csv_and_meta_are_written(tmp_path):
    eng = make(tmp_path, overlay=False)
    paths = eng.start_trial("P01")
    for i in range(20):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    assert "overlay.mp4" not in paths and not list(Path(paths["raw.mp4"]).parent.glob("*_overlay.mp4"))
    assert count_frames(paths["raw.mp4"]) == 20
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["overlay_video_saved"] is False and "overlay.mp4" not in meta["files"]
    eng.start_trial("P01")
    with pytest.raises(RuntimeError, match="Stop recording"):
        eng.save_overlay = True
    eng.stop_trial()


def test_a_writer_failure_is_reported_not_swallowed(tmp_path):
    eng = make(tmp_path)
    eng.start_trial("P01")
    eng.process(render(PTS, seed=1), 0.1)

    class Broken:
        def write(self, frame):
            raise OSError("No space left on device")

        def release(self):
            pass

    eng.rec._video.raw = Broken()
    with pytest.raises(RuntimeError, match="Saving the video failed"):
        for i in range(60):                                        # the thread hits the error on its next frame; write() then raises
            eng.process(render(PTS, seed=i), 0.2 + i / 30)
            import time
            time.sleep(0.02)
    paths = eng.rec.paths
    eng.stop_trial()
    assert "No space left" in json.loads(paths["meta.json"].read_text())["close_errors"][0]


def test_review_exports_the_overlay_video_from_raw_and_csv(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    eng = make(tmp_path, overlay=False)
    paths = eng.start_trial("P01")
    for i in range(12):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    tab.export_overlay_video()
    out = tmp_path / "P01" / f"{tab.trial.base}_overlay_export.mp4"
    assert out.exists() and count_frames(out) == 12


def test_the_window_has_an_overlay_checkbox_that_is_frozen_while_recording(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    assert live.chk_overlay.isChecked() and win.engine.save_overlay is True
    live.chk_overlay.setChecked(False)
    assert win.engine.save_overlay is False
    live.chk_overlay.setChecked(True)
    win.close()
