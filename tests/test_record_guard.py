"""COD-01: Record is refused when no video is coming in, empty trials are called out, Review explains an empty trial."""
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(45):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + 15 * math.sin(math.pi * i / 45), ankle_df=5), seed=i))
    w.release()
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)

    def pump(cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    def start_and_mark():
        live = win.live
        live.start_source(str(video))
        assert pump(lambda: win.engine._last_gray is not None)
        p0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)
        for name in win.engine.test.dots:
            live.view.clicked.emit(float(p0[name][0]) + 2, float(p0[name][1]) + 2)
        assert win.engine.next_dot_to_mark() is None
        assert pump(lambda: win.engine.frame_index > 0)
        live.participant.setText("P01")

    return app, win, warnings, pump, start_and_mark, tmp_path


def test_record_is_refused_after_the_source_was_stopped(setup):
    app, win, warnings, pump, start_and_mark, tmp = setup
    start_and_mark()
    live = win.live
    live.btn_lock.click()
    live.stop_camera()
    live.btn_rec.click()
    assert win.engine.rec is None and not live.btn_rec.isChecked()
    assert warnings and "No video is coming in" in warnings[-1]
    assert not (tmp / "recordings" / "P01").exists()
    win.close()


def test_record_is_refused_at_the_end_of_a_file_and_allowed_after_stepping_back(setup):
    app, win, warnings, pump, start_and_mark, tmp = setup
    start_and_mark()
    live = win.live
    live.btn_lock.click()
    assert pump(lambda: live.worker is not None and live.worker.at_end, timeout=30)
    live.btn_rec.click()
    assert win.engine.rec is None and "already ended" in warnings[-1]
    live.btn_back.click()
    assert pump(lambda: not live.worker.at_end)
    live.btn_rec.click()
    assert win.engine.rec is not None                                   # a file stepped back from the end can be recorded
    live.btn_rec.click()                                                # stop
    win.close()


def test_a_stalled_recording_says_no_video(setup):
    app, win, warnings, pump, start_and_mark, tmp = setup
    start_and_mark()
    live = win.live
    live.btn_lock.click()
    live.btn_pause.click()                                              # paused mid-file is fine: recording un-pauses it
    live.btn_rec.click()
    assert win.engine.rec is not None
    live.worker.stop()
    live.worker.wait(3000)                                              # no more frames arrive, but the recording is still on
    time.sleep(0.7)
    live.refresh_status()
    assert "NO VIDEO" in live.rec_label.text()
    live.btn_rec.click()
    win.close()


def test_stopping_with_zero_frames_warns_and_review_explains(setup):
    app, win, warnings, pump, start_and_mark, tmp = setup
    start_and_mark()
    live = win.live
    live.btn_lock.click()
    live.btn_pause.click()
    pump(lambda: False, 0.3)
    paths = win.engine.start_trial("P01")                               # a recording that never receives a frame
    live._rec_started = time.perf_counter()
    live.btn_rec.setChecked(True)
    live.toggle_record(force_stop=True)
    assert "NOTHING RECORDED" in live.rec_label.text()
    assert warnings and "empty" in warnings[-1]
    with pytest.raises(ValueError, match="contains no frames"):
        Trial(paths["meta.json"])
    win.close()
