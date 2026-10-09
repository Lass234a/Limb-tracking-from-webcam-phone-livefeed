"""The phone-camera check: its verdicts in plain English (no network or camera is used here)."""
from limbtrack.diagnose import verdict, http_probe, tcp_reachable

NOPIC = {"frames": 0, "fps": 0.0, "size": None, "backend": None, "error": ""}
PIC = {"frames": 90, "fps": 30.0, "size": (1280, 720), "backend": "MSMF", "error": ""}


def text(f):
    return "\n".join(verdict(f))


def test_no_client_and_no_camera_says_to_start_the_pc_program():
    assert "not running" in text({"client_running": False, "phone_cameras": []})


def test_a_listed_camera_without_pictures_says_the_client_is_not_connected_to_the_phone():
    t = text({"client_running": True, "phone_cameras": [(0, "DroidCam Video", NOPIC)]})
    assert "NOT connected to the phone" in t and "preview" in t


def test_a_working_camera_is_confirmed_with_its_size_and_rate():
    t = text({"client_running": True, "phone_cameras": [(0, "DroidCam Video", PIC)]})
    assert t.startswith("OK:") and "1280x720" in t and "MSMF" in t


def test_an_unreachable_phone_lists_the_usual_causes():
    t = text({"phone_cameras": [], "client_running": True, "address": {"host": "192.168.0.100", "port": 4747},
              "reach": (False, "no answer within 3 seconds")})
    assert "SAME Wi-Fi" in t and "IP address" in t and "foreground" in t


def test_a_reachable_phone_that_serves_no_stream_points_to_the_browser_and_single_viewer():
    f = {"phone_cameras": [], "client_running": True, "address": {"host": "h", "port": 4747}, "reach": (True, ""),
         "paths": [{"path": "/video", "url": "u", "probe": {"stream": False}, "read": NOPIC}]}
    t = text(f)
    assert "browser" in t and "ONE viewer" in t


def test_a_working_stream_address_is_given_back_to_type_in():
    f = {"phone_cameras": [], "client_running": True, "address": {"host": "h", "port": 4747}, "reach": (True, ""),
         "paths": [{"path": "/mjpegfeed", "url": "http://h:4747/mjpegfeed", "probe": {"stream": True}, "read": PIC}]}
    t = text(f)
    assert "http://h:4747/mjpegfeed" in t and "Connect to stream" in t


def test_a_stream_the_program_cannot_read_suggests_the_client_route():
    f = {"phone_cameras": [], "client_running": True, "address": {"host": "h", "port": 4747}, "reach": (True, ""),
         "paths": [{"path": "/video", "url": "http://h:4747/video", "probe": {"stream": True}, "read": {**NOPIC, "error": "Could not open"}}]}
    assert "DroidCam Client route" in text(f)


def test_probes_fail_politely_on_a_closed_port_and_a_bad_address():
    ok, why = tcp_reachable("127.0.0.1", 1, timeout=1.0)
    assert ok is False and why
    r = http_probe("http://127.0.0.1:1/video", timeout=1.0)
    assert r["stream"] is False and r["error"]
