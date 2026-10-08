"""CAM-08: network streams (a phone app that publishes a stream) are recognised, opened with time-outs and read without lag."""
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.camera import CameraSource, classify_source  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("source, kind", [
    (0, "webcam"), (2, "webcam"),
    ("http://192.168.1.20:8080/video", "stream"), ("HTTPS://phone.local/stream", "stream"), ("rtsp://10.0.0.5/live", "stream"),
    ("rtmp://x/y", "stream"), ("udp://@:5000", "stream"), ("srt://x:9000", "stream"), ("tcp://x:1", "stream"),
    (r"C:\Users\someone\Desktop\leg.mp4", "file"), ("demo/leg.mp4", "file"), ("http.mp4", "file"), ("rtsp.mkv", "file"),
])
def test_sources_are_told_apart(source, kind):
    assert classify_source(source) == kind


def test_nonsense_is_refused():
    with pytest.raises(ValueError):
        classify_source(None)


class Mjpeg(BaseHTTPRequestHandler):
    fps = 30

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        i = 0
        t0 = time.perf_counter()
        try:
            while not self.server.stop:
                img = np.full((120, 160, 3), (i * 7) % 255, np.uint8)
                ok, jpg = cv2.imencode(".jpg", img)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                i += 1
                time.sleep(max(0.0, t0 + i / self.fps - time.perf_counter()))
        except Exception:  # noqa: BLE001 - client went away
            pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Mjpeg)
    srv.daemon_threads, srv.stop = True, False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.stop = True
    srv.shutdown()


def test_a_stream_is_opened_as_a_live_source_and_frames_flow(server):
    cam = CameraSource(server)
    try:
        assert cam.kind == "stream" and cam.is_live and not cam.is_file
        got = [cam.read() for _ in range(15)]
        assert all(f is not None for f, _ in got) and got[0][0].shape[:2] == (120, 160)
        assert all(b[1] > a[1] for a, b in zip(got, got[1:]))
        assert cam.info()["source_kind"] == "stream"
    finally:
        cam.close()


def test_a_slow_consumer_never_lags_behind_the_stream(server):
    """The consumer needs 60 ms per frame but the stream delivers every 33 ms: it must see recent frames, not a backlog."""
    cam = CameraSource(server)
    try:
        ages = []
        for _ in range(40):
            frame, t = cam.read()
            assert frame is not None
            ages.append(time.perf_counter() - t)
            time.sleep(0.06)
        assert float(np.median(ages[10:])) < 0.08, ages           # without the latest-frame reader this grows to ~0.3 s
        assert cam.frames_discarded > 5                            # frames were skipped instead of queued
    finally:
        cam.close()


def test_a_closed_stream_gives_no_frames_instead_of_hanging(server):
    cam = CameraSource(server)
    try:
        assert cam.read()[0] is not None
        cam._grabber.stopped = True                                # the stream stops delivering
        cam._grabber.join(timeout=3)
        cam._grabber.stopped = False
        cam._grabber.seq = cam._grabber.taken                      # (nothing new is waiting)
        t0 = time.perf_counter()
        frame, t = cam._grabber.next(timeout=0.5)
        assert frame is None and t is None and time.perf_counter() - t0 < 2.0
    finally:
        cam.close()


def test_the_window_connects_only_to_stream_addresses(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    live.stream_box.setText("my_video.mp4")
    live.connect_stream()
    assert "not a stream address" in live.instruction.text() and live.worker is None
    live.stream_box.setText("")
    live.connect_stream()
    assert live.worker is None
    win.close()
