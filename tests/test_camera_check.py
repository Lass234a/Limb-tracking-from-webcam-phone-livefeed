"""ARC-06, ARC-08: a 5-second camera check with a plain verdict, and a camera set-up checklist saved with each trial."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.camera import CameraSource  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def camera(stamps, fourcc="MJPG", size=(1280, 720)):
    cam = object.__new__(CameraSource)
    cam.source, cam.is_file, cam.is_live = 0, False, True
    cam._stamps = list(stamps)
    cam._granted = {"width": size[0], "height": size[1], "fps": 30.0, "fourcc": fourcc}
    cam.failed_reads, cam.reconnects = 0, 0
    return cam


def steady(fps, seconds=6.0):
    return [i / fps for i in range(int(seconds * fps))]


def test_a_steady_30_fps_feed_passes():
    h = camera(steady(30.0)).health()
    assert h["ok"] is True and h["fps"] == pytest.approx(30.0, abs=0.2) and h["max_gap_s"] < 0.05
    assert h["verdict"].startswith("Camera OK") and "1280x720" in h["verdict"]


def test_a_slow_feed_fails_and_says_why():
    h = camera(steady(12.0)).health()
    assert h["ok"] is False and "12.0 fps" in h["verdict"] and h["verdict"].startswith("Camera NOT good enough")


def test_a_feed_with_a_stall_fails_even_if_the_average_is_fine():
    stamps = steady(30.0, 2.5) + [2.5 + 0.6 + i / 30 for i in range(100)]            # a 0.6 s wait in the middle
    h = camera(stamps).health()
    assert h["ok"] is False and "wait of 0.6" in h["verdict"]


def test_an_uncompressed_format_at_large_size_is_named():
    h = camera(steady(30.0), fourcc="YUY2").health()
    assert h["ok"] is False and "YUY2" in h["verdict"]


def test_too_little_data_gives_no_verdict_yet():
    h = camera(steady(30.0, 2.0)).health()
    assert h["ok"] is None and "Not enough frames" in h["verdict"]


def test_the_checklist_and_the_check_are_saved_with_the_trial(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live, eng = win.live, win.engine
    live.camera_checks["stabilisation_off"].setChecked(True)
    live.camera_checks["usb_connected"].setChecked(True)
    live.camera_model.setText("Galaxy S25, Iriun")
    live.camera_model.editingFinished.emit()
    assert eng.camera_setup["stabilisation_off"] is True and eng.camera_setup["zoom_1x"] is False
    assert eng.camera_setup["model_and_app"] == "Galaxy S25, Iriun"
    eng.camera_check = {"ok": True, "fps": 29.9, "verdict": "Camera OK"}

    p = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    paths = eng.start_trial("P01")
    eng.process(render(p, seed=1), 0.1)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["camera_setup_checklist"]["usb_connected"] is True and meta["camera_setup_checklist"]["model_and_app"] == "Galaxy S25, Iriun"
    assert meta["camera_check"]["fps"] == 29.9
    win.close()


def test_the_checklist_cannot_change_during_a_recording(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    p = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        eng.mark_dot(n, *(p[n] + 2))
    eng.start_trial("P01")
    with pytest.raises(RuntimeError, match="Stop recording"):
        eng.camera_setup = {"zoom_1x": True}
    eng.stop_trial()


def test_the_check_button_needs_a_running_camera(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    win.live.check_camera()
    assert "Start a camera first" in win.live.instruction.text()
    win.close()


def test_a_camera_directshow_cannot_open_is_opened_through_media_foundation(monkeypatch):
    """DroidCam's virtual camera opens by index only through Media Foundation (found with a real DroidCam on 2026-10-08)."""
    import cv2
    from limbtrack import camera as cam_mod

    calls = []

    class Cap:
        def __init__(self, source, api=None, *a):
            self.api, self.props = api, {}
            calls.append(api)

        def isOpened(self):
            return self.api == cv2.CAP_MSMF

        def set(self, k, v):
            self.props[k] = v
            return True

        def get(self, k):
            return {cv2.CAP_PROP_FRAME_WIDTH: 1280.0, cv2.CAP_PROP_FRAME_HEIGHT: 720.0, cv2.CAP_PROP_FPS: 30.0}.get(k, self.props.get(k, 0.0))

        def read(self):
            return True, None

        def release(self):
            pass

    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", Cap)
    cam = cam_mod.CameraSource(1)
    assert cam.backend == "MSMF" and calls[:2] == [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    assert cam.info()["capture_backend"] == "MSMF"
    assert cam.settings_problems() == []

    monkeypatch.setattr(Cap, "isOpened", lambda self: False)            # neither backend opens it: the old error message
    with pytest.raises(RuntimeError, match="Could not open video source"):
        cam_mod.CameraSource(2)
