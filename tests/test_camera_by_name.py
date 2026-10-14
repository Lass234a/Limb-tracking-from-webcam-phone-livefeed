"""A camera that DirectShow cannot open is opened through Media Foundation by its NAME, never by its number.

Found on 2026-10-14 with a real DroidCam: DirectShow lists [DroidCam Video, Integrated Camera] but Media Foundation lists
[Integrated Camera, DroidCam Video]. The old fallback asked Media Foundation for the same number (0) and so opened the laptop webcam
instead of the phone, silently. No camera hardware is used here: OpenCV and the two name lists are faked."""
import cv2
import pytest

from limbtrack import camera as cam_mod

DIRECTSHOW = ["DroidCam Video", "Integrated Camera"]
MEDIA_FOUNDATION = ["Integrated Camera", "DroidCam Video"]


class Recorder:
    """A fake cv2.VideoCapture: opens through the backends in `opens`, remembers every (index, backend) it was asked for."""
    calls = []
    opens = ()

    def __init__(self, index, api=None, *a):
        self.index, self.api, self.props = index, api, {}
        Recorder.calls.append((index, api))

    def isOpened(self):
        return self.api in Recorder.opens

    def set(self, k, v):
        self.props[k] = v
        return True

    def get(self, k):
        return {cv2.CAP_PROP_FRAME_WIDTH: 1280.0, cv2.CAP_PROP_FRAME_HEIGHT: 720.0, cv2.CAP_PROP_FPS: 30.0}.get(k, self.props.get(k, 0.0))

    def read(self):
        return True, None

    def release(self):
        pass


@pytest.fixture
def fakes(monkeypatch):
    Recorder.calls, Recorder.opens = [], (cv2.CAP_MSMF,)              # DirectShow cannot open it, Media Foundation can
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", Recorder)
    monkeypatch.setattr(cam_mod, "camera_names", lambda: list(DIRECTSHOW))
    monkeypatch.setattr(cam_mod, "msmf_camera_names", lambda: list(MEDIA_FOUNDATION))
    return monkeypatch


def msmf_calls():
    return [i for i, api in Recorder.calls if api == cv2.CAP_MSMF]


def test_the_media_foundation_camera_with_the_same_name_is_opened_not_the_one_with_the_same_number(fakes):
    cam = cam_mod.CameraSource(0)                                      # 'DroidCam Video' in the list
    assert cam.backend == "MSMF" and msmf_calls() == [1]               # Media Foundation's number for it is 1, not 0
    assert cam.device_name == "DroidCam Video" and cam.msmf_index == 1 and cam.name_matched is True
    info = cam.info()
    assert info["capture_device_name"] == "DroidCam Video" and info["capture_media_foundation_index"] == 1 and info["capture_matched_by_name"]
    Recorder.calls.clear()
    cam = cam_mod.CameraSource(1)                                      # 'Integrated Camera'
    assert msmf_calls() == [0] and cam.device_name == "Integrated Camera"


def test_a_camera_that_media_foundation_does_not_list_is_not_replaced_by_another_one(fakes):
    fakes.setattr(cam_mod, "msmf_camera_names", lambda: ["Integrated Camera"])
    with pytest.raises(RuntimeError, match="Media Foundation does not list it.*Integrated Camera"):
        cam_mod.CameraSource(0)
    assert msmf_calls() == []                                          # no Media Foundation camera was opened at all


def test_a_camera_neither_backend_can_open_says_so(fakes):
    Recorder.opens = ()
    with pytest.raises(RuntimeError, match="Could not open video source.*Media Foundation could not open 'DroidCam Video' either"):
        cam_mod.CameraSource(0)


def test_when_the_names_cannot_be_read_the_same_number_is_used_as_before(fakes):
    fakes.setattr(cam_mod, "camera_names", lambda: None)
    cam = cam_mod.CameraSource(1)
    assert msmf_calls() == [1] and cam.name_matched is False and cam.device_name is None


def test_a_camera_directshow_opens_never_goes_near_media_foundation(fakes):
    Recorder.opens = (cv2.CAP_DSHOW,)
    fakes.setattr(cam_mod, "msmf_camera_names", lambda: pytest.fail("the names are only needed for the fallback"))
    cam = cam_mod.CameraSource(0)
    assert cam.backend == "DSHOW" and msmf_calls() == [] and cam.device_name is None


def test_listing_the_media_foundation_names_opens_no_camera_and_never_raises():
    names = cam_mod.msmf_camera_names()                                # on this computer: a list of names (or None without Media Foundation)
    assert names is None or all(isinstance(n, str) for n in names)
