"""The window and meta.json report what the camera really delivers, and warn when it is not what was asked for."""
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.camera import CameraSource  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class FakeCap:
    def __init__(self, w, h, fps):
        self.v = {cv2.CAP_PROP_FRAME_WIDTH: w, cv2.CAP_PROP_FRAME_HEIGHT: h, cv2.CAP_PROP_FPS: fps}

    def get(self, k):
        return self.v.get(k, 0)


def fake_camera(granted, requested=(1280, 720, 30), arrival_fps=30.0):
    cam = object.__new__(CameraSource)
    cam.cap = FakeCap(*granted)
    cam.source, cam.is_file, cam.is_live = 0, False, True
    cam.requested = {"width": requested[0], "height": requested[1], "fps": requested[2]}
    cam._granted = cam.granted()
    cam._stamps = [i / arrival_fps for i in range(40)]
    return cam


def test_no_warning_when_the_camera_delivers_what_was_asked():
    cam = fake_camera((1280, 720, 30), arrival_fps=29.8)
    assert cam.settings_problems() == []
    assert abs(cam.measured_fps() - 29.8) < 0.01


def test_warns_about_wrong_resolution_and_low_frame_rate():
    cam = fake_camera((640, 480, 30), arrival_fps=15.0)
    problems = " | ".join(cam.settings_problems())
    assert "1280x720" in problems and "640x480" in problems
    assert "15.0 fps" in problems


def test_frame_rate_unknown_until_enough_frames_have_arrived():
    cam = fake_camera((1280, 720, 30))
    cam._stamps = cam._stamps[:5]
    assert cam.measured_fps() is None and cam.settings_problems() == []


def test_info_for_meta_json_lists_requested_and_granted():
    info = fake_camera((640, 480, 30)).info()
    assert info["requested"] == {"width": 1280, "height": 720, "fps": 30}
    assert info["granted_by_driver"]["width"] == 640 and info["is_file"] is False


def test_file_source_reports_its_own_size_in_the_window_and_meta(tmp_path):
    app = QApplication.instance() or QApplication([])
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(30):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + i, ankle_df=5), seed=i))
    w.release()

    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    live.start_source(str(video))
    end = time.time() + 20
    while time.time() < end and "1280x720" not in live.cam_info.text():
        app.processEvents()
        time.sleep(0.01)
    assert "Video file: 1280x720" in live.cam_info.text()
    assert win.engine.camera_info["requested"] is None and win.engine.camera_info["is_file"] is True

    live.stop_camera()
    assert live.cam_info.text() == ""                        # nothing shown when no source is running

    # the record in meta.json (the engine as the video thread would have filled it in)
    eng = win.engine
    p0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)
    eng.process(render(p0, seed=0), 0.0)
    for name in eng.test.dots:
        assert eng.mark_dot(name, *(p0[name] + 2))
    eng.process(render(p0, seed=0), 0.001)
    paths = eng.start_trial("P01")
    eng.process(render(p0, seed=0), 0.1)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["camera_settings"]["is_file"] is True
    assert meta["camera_settings"]["granted_by_driver"]["width"] == 1280
    win.close()


class SettableCap(FakeCap):
    """A stand-in webcam. accept=False behaves like a driver that returns True from set() but ignores it."""

    def __init__(self, accept):
        super().__init__(1280, 720, 30)
        self.accept = accept
        self.v.update({cv2.CAP_PROP_AUTO_EXPOSURE: 0.75, cv2.CAP_PROP_EXPOSURE: -6.0,
                       cv2.CAP_PROP_AUTO_WB: 1, cv2.CAP_PROP_WB_TEMPERATURE: 4600})

    def set(self, k, value):
        if self.accept:
            self.v[k] = value
        return True


def webcam_with(cap):
    cam = fake_camera((1280, 720, 30))
    cam.cap = cap
    return cam


def test_lock_is_reported_as_accepted_when_the_camera_takes_it():
    cap = SettableCap(accept=True)
    cam = webcam_with(cap)
    r = cam.lock_exposure_wb()
    assert r["exposure_locked"] and r["white_balance_locked"]
    assert cap.v[cv2.CAP_PROP_EXPOSURE] == -6.0 and cap.v[cv2.CAP_PROP_WB_TEMPERATURE] == 4600   # held at the current values
    assert "locked" in cam.lock_summary()
    assert cam.info()["exposure_wb_lock"]["readback"]["auto_exposure"] == 0.25                   # raw read-back is kept for the record


def test_lock_is_reported_as_refused_when_the_camera_ignores_it():
    cam = webcam_with(SettableCap(accept=False))
    r = cam.lock_exposure_wb()
    assert not r["exposure_locked"] and not r["white_balance_locked"]
    assert "did not accept" in cam.lock_summary()


def test_lock_button_is_only_available_for_a_live_camera(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    assert not win.live.btn_lock_cam.isEnabled()                       # nothing running
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(20):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + i, ankle_df=5), seed=i))
    w.release()
    win.live.start_source(str(video))
    end = time.time() + 20
    while time.time() < end and "Video file" not in win.live.cam_info.text():
        app.processEvents()
        time.sleep(0.01)
    assert not win.live.btn_lock_cam.isEnabled()                       # a file has no exposure to lock
    win.live.stop_camera()
    win.close()


def test_scan_cameras_is_disabled_while_a_source_runs_and_never_probes_then(tmp_path, monkeypatch):
    """CAM-06: releasing a probe handle on the streaming webcam kills the stream."""
    import limbtrack.gui as gui

    app = QApplication.instance() or QApplication([])
    probed = []
    monkeypatch.setattr(gui, "list_cameras", lambda: probed.append(1) or [])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    assert live.btn_scan.isEnabled()
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(30):
        w.write(render(leg_points(hip_flex=80, knee_flex=20 + i, ankle_df=5), seed=i))
    w.release()
    live.start_source(str(video))
    assert not live.btn_scan.isEnabled()
    live.scan_cameras()                                                  # even if called directly
    assert probed == []
    live.stop_camera()
    assert live.btn_scan.isEnabled()
    win.close()


# ---- CAM-04: set the size before the video format, report the format, and do not blame light for a format problem
def fourcc_int(text):
    return sum(ord(c) << (8 * i) for i, c in enumerate(text))


def test_size_and_frame_rate_are_set_before_the_video_format(monkeypatch):
    import limbtrack.camera as camera

    calls = []

    class RecordingCap:
        def __init__(self, *a, **k):
            pass

        def set(self, prop, value):
            calls.append(prop)
            return True

        def get(self, prop):
            return 0

        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(camera.cv2, "VideoCapture", RecordingCap)
    camera.CameraSource(0)
    i = calls.index
    assert i(cv2.CAP_PROP_FOURCC) > max(i(cv2.CAP_PROP_FRAME_WIDTH), i(cv2.CAP_PROP_FRAME_HEIGHT), i(cv2.CAP_PROP_FPS))


def test_the_video_format_is_reported_and_saved():
    cam = fake_camera((1280, 720, 30))
    cam.cap.v[cv2.CAP_PROP_FOURCC] = fourcc_int("MJPG")
    assert cam.granted()["fourcc"] == "MJPG"
    cam._granted = cam.granted()
    assert cam.info()["granted_by_driver"]["fourcc"] == "MJPG"
    cam.cap.v[cv2.CAP_PROP_FOURCC] = 0
    assert cam.granted()["fourcc"] == ""                                 # unknown is not a format


def test_a_slow_uncompressed_format_is_named_instead_of_blaming_light():
    cam = fake_camera((1280, 720, 30), arrival_fps=10.0)
    cam.cap.v[cv2.CAP_PROP_FOURCC] = fourcc_int("YUY2")
    cam._granted = cam.granted()
    text = " ".join(cam.settings_problems())
    assert "uncompressed" in text and "YUY2" in text and "low light" not in text


def test_slow_mjpg_or_unknown_format_gets_the_general_message():
    for fourcc in (fourcc_int("MJPG"), 0):
        cam = fake_camera((1280, 720, 30), arrival_fps=15.0)
        cam.cap.v[cv2.CAP_PROP_FOURCC] = fourcc
        cam._granted = cam.granted()
        text = " ".join(cam.settings_problems())
        assert "dim light" in text and "cannot keep up" in text and "uncompressed" not in text


def test_a_webcam_is_opened_on_automatic_exposure_and_white_balance(monkeypatch):
    """CAM-05: manual settings survive in the driver between sessions and can leave the picture black."""
    import limbtrack.camera as camera

    calls = []

    class RecordingCap:
        def __init__(self, *a, **k):
            pass

        def set(self, prop, value):
            calls.append((prop, value))
            return True

        def get(self, prop):
            return 0

        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(camera.cv2, "VideoCapture", RecordingCap)
    camera.CameraSource(0)
    assert (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) in calls and (cv2.CAP_PROP_AUTO_WB, 1) in calls


# ---- cameras are listed by name (a phone used as a webcam shows up as "Iriun Webcam", "DroidCam", ...)
def test_cameras_are_listed_by_name_without_opening_any(monkeypatch):
    import limbtrack.camera as camera

    opened = []
    monkeypatch.setattr(camera, "camera_names", lambda: ["Integrated Camera", "Iriun Webcam"])
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError("opened")))
    assert camera.list_cameras() == [(0, "Integrated Camera"), (1, "Iriun Webcam")]
    assert opened == []


def test_without_a_name_list_the_indexes_are_probed_and_called_camera_n(monkeypatch):
    import limbtrack.camera as camera

    class Probe:
        def __init__(self, index, api=None):
            self.index = index

        def isOpened(self):
            return self.index in (0, 2)

        def release(self):
            pass

    monkeypatch.setattr(camera, "camera_names", lambda: None)
    monkeypatch.setattr(camera.cv2, "VideoCapture", Probe)
    assert camera.list_cameras() == [(0, "Camera 0"), (2, "Camera 2")]


def test_the_camera_box_shows_names_and_keeps_the_choice(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    live.fill_cameras([(0, "Integrated Camera"), (1, "Iriun Webcam"), (2, "DroidCam Source 3")])
    live.cam_combo.setCurrentIndex(1)
    assert live.cam_combo.itemText(1) == "1: Iriun Webcam" and live.cam_combo.currentData() == 1
    live.fill_cameras([(0, "Integrated Camera"), (1, "Iriun Webcam")])          # scanned again: still on the phone
    assert live.cam_combo.currentData() == 1
    live.fill_cameras([(0, "Integrated Camera")])                               # the phone is gone
    assert live.cam_combo.count() == 1 and live.cam_combo.currentData() == 0
    win.close()
