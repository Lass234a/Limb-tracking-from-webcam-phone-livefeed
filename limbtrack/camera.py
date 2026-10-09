"""Frame sources. Everything the app needs from a camera is: open, read() -> (frame, time), close.

Phones will appear here as well: apps such as Iriun / Camo / DroidCam show up as an
ordinary webcam index, and a network stream is just a URL string.
"""
import re
import threading
from urllib.parse import urlsplit, urlunsplit
import time
from pathlib import Path

import cv2


STREAM_SCHEMES = ("rtsp", "rtsps", "http", "https", "rtmp", "rtmps", "udp", "tcp", "srt", "rtp")


def normalise_stream_address(address):
    """What the operator typed, made into an address OpenCV can open. Phone apps such as DroidCam (port 4747) and IP Webcam
    (port 8080) serve their picture at /video; the bare address (http://192.168.0.100:4747) is a web page, not a picture."""
    a = (address or "").strip()
    if a and "://" not in a and re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?(/.*)?$", a):
        a = "http://" + a
    p = urlsplit(a)
    if p.scheme.lower() in ("http", "https") and p.netloc and p.path in ("", "/"):
        a = urlunsplit(p._replace(path="/video"))
    return a


OPEN_TIMEOUT_S = 20.0       # a camera that has not opened by then is reported instead of waiting for ever


class CameraOpenTimeout(RuntimeError):
    """The camera did not open in time."""


def call_with_timeout(fn, seconds, what="The camera"):
    """Run fn() on a helper thread and give up after `seconds` (a driver call that never returns cannot be interrupted, so the
    thread is left behind; if it finishes later, whatever it opened is released). Raises CameraOpenTimeout."""
    box = {"abandoned": False}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the caller
            box["error"] = e
        if box["abandoned"] and "value" in box:
            try:
                box["value"].release()
            except Exception:  # noqa: BLE001
                pass

    t = threading.Thread(target=run, daemon=True, name="camera-open")
    t.start()
    t.join(seconds)
    if t.is_alive():
        box["abandoned"] = True
        raise CameraOpenTimeout(f"{what} did not answer within {seconds:g} s. For a phone camera the usual cause is that the phone's PC "
                                "program (for example DroidCam Client) is not connected to the phone: connect it there first and wait "
                                "until its preview moves. Also close other programs that may be using the camera (browser, Teams, OBS).")
    if "error" in box:
        raise box["error"]
    return box["value"]


def classify_source(source):
    """'webcam' (an integer index), 'stream' (a network address such as http://... or rtsp://...) or 'file'."""
    if isinstance(source, int):
        return "webcam"
    if isinstance(source, str):
        if "://" in source and source.split("://", 1)[0].lower() in STREAM_SCHEMES:
            return "stream"
        return "file"
    raise ValueError(f"not a camera, stream address or file: {source!r}")


class _LatestFrame(threading.Thread):
    """Reads a network stream as fast as it delivers and keeps only the newest frame, so a slow consumer never falls behind
    the stream (a backlog of old frames would make the live picture lag by a third of a second or more)."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.cond = threading.Condition()
        self.frame, self.t, self.seq, self.taken = None, None, 0, 0
        self.discarded = 0           # frames that were replaced before anyone read them
        self.stopped = False

    def run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cond:
                if self.seq > self.taken:
                    self.discarded += 1                        # the previous frame was never read: it is replaced
                self.frame, self.t, self.seq = frame, time.perf_counter(), self.seq + 1
                self.cond.notify_all()

    def next(self, timeout=1.5):
        """The newest frame that has not been returned yet, or (None, None) after `timeout` seconds without one."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > self.taken or self.stopped, timeout):
                return None, None
            self.taken = self.seq
            return self.frame, self.t


class CameraSource:
    failed_reads = 0       # reads that returned no frame on a live source (counted, never silent)
    reconnects = 0
    kind = "webcam"        # 'webcam', 'stream' or 'file' (set when the source is opened)
    backend = "DSHOW"      # how a webcam was opened: DirectShow, or Media Foundation when DirectShow could not open it
    _grabber = None

    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.kind = classify_source(source)
        self.is_file = self.kind == "file"
        self.is_live = not self.is_file
        self._size_fps = (width, height, fps)
        self.failed_reads = 0       # reads that returned no frame on a live source (counted, never silent)
        self.reconnects = 0
        self.cap = self._open_capture()
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self._grabber = self._start_grabber()
        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.requested = {"width": width, "height": height, "fps": fps} if isinstance(source, int) else None
        self._t0 = time.perf_counter()
        self._n = 0
        self._granted = self.granted()      # read once; the window may ask for it from another thread
        self._stamps = []          # recent frame arrival times (live sources): the real frame rate

    def _open_capture(self):
        if isinstance(self.source, int):
            return call_with_timeout(self._open_webcam, OPEN_TIMEOUT_S, "Opening the camera")
        return self._open_other()

    def _open_webcam(self):
        source, (width, height, fps) = self.source, self._size_fps
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        # Order matters on DirectShow: asking for compressed MJPG *before* the size makes some webcams fall back
        # to uncompressed YUY2 when the size is set, which runs at only ~10 fps at 1280x720 (audit CAM-04).
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Start from automatic exposure and white balance: a webcam keeps manual settings from an earlier session, and a
        # manual exposure can leave the picture black (audit CAM-05).
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)      # DirectShow: 0.75 = automatic
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        if not cap.isOpened():
            # Some virtual cameras (DroidCam's "DroidCam Video") cannot be opened through DirectShow by index but work through
            # Windows Media Foundation. Only the size and rate are requested there; the format is not under our control.
            cap.release()
            alt = cv2.VideoCapture(source, cv2.CAP_MSMF)
            if alt.isOpened():
                alt.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                alt.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                alt.set(cv2.CAP_PROP_FPS, fps)
                self.backend = "MSMF"
                return alt
            alt.release()
            return cv2.VideoCapture(source, cv2.CAP_DSHOW)       # (not opened: the caller reports it)
        self.backend = "DSHOW"
        return cap

    def _open_other(self):
        source = self.source
        if self.kind == "stream":   # network stream: do not wait half a minute on a stalled connection
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000])
        return cv2.VideoCapture(source)

    def _start_grabber(self):
        if self.kind != "stream":
            return None
        g = _LatestFrame(self.cap)
        g.start()
        return g

    def _stop_grabber(self):
        g, self._grabber = getattr(self, "_grabber", None), None
        if g is not None:
            g.stopped = True
            with g.cond:
                g.cond.notify_all()
            g.join(timeout=3.0)

    @property
    def frames_discarded(self):
        """Frames of a network stream that arrived faster than they could be processed and were skipped (0 for other sources)."""
        g = getattr(self, "_grabber", None)
        return g.discarded if g is not None else 0

    def reopen(self):
        """Close and open the same live source again (after it stopped delivering frames). Returns True if it opened.
        A webcam comes back on its automatic settings, so an exposure/white-balance lock has to be pressed again."""
        self._stop_grabber()
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass
        self.reconnects += 1
        try:
            self.cap = self._open_capture()
        except CameraOpenTimeout:
            return False
        ok = self.cap.isOpened()
        if ok:
            self._grabber = self._start_grabber()
            self._granted = self.granted()
            self._stamps = []
            if getattr(self, "lock_report", None) and self.lock_report.get("requested"):
                self.lock_report = None
        return ok

    @property
    def description(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        name = Path(self.source).name if self.is_file else self.source      # a file's name only: no user folders in meta.json
        return f"{name} ({w}x{h})"

    def granted(self):
        """What the camera driver says it is delivering. (Drivers can report the requested value without honouring it.)"""
        return {"width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(self.cap.get(cv2.CAP_PROP_FPS) or 0.0, 3), "fourcc": self._fourcc_str()}

    def _fourcc_str(self):
        """The video format the camera reports (for example 'MJPG' or 'YUY2'); '' if unknown."""
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
        text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
        return text.strip() if code and all(32 <= ord(c) < 127 for c in text) else ""

    def measured_fps(self):
        """Frames per second actually arriving over the last few seconds (live sources only), or None until known."""
        stamps = list(self._stamps)[-90:]
        if len(stamps) < 10 or stamps[-1] <= stamps[0]:
            return None
        return (len(stamps) - 1) / (stamps[-1] - stamps[0])

    def health(self, seconds=5.0, min_fps=25.0, max_gap_s=0.15):
        """How the live feed behaved over the last `seconds`: frame rate, longest wait for a frame and how uneven the
        spacing was. Returns a dict with a plain-English `verdict` and `ok` (None when there is not enough data yet)."""
        stamps = [s for s in list(self._stamps) if s >= (self._stamps[-1] - seconds)] if self._stamps else []
        g = self._granted
        out = {"seconds": seconds, "width": g["width"], "height": g["height"], "fourcc": g.get("fourcc", ""),
               "failed_reads": self.failed_reads, "reconnects": self.reconnects}
        if len(stamps) < 10 or stamps[-1] - stamps[0] < 0.8 * seconds:
            out.update(ok=None, verdict=f"Not enough frames yet: let the camera run for {seconds:g} s and check again.")
            return out
        dts = [b - a for a, b in zip(stamps, stamps[1:])]
        mean = sum(dts) / len(dts)
        out.update(frames=len(stamps), fps=round(1.0 / mean, 2), max_gap_s=round(max(dts), 3),
                   jitter_ms=round(1000.0 * (sum((d - mean) ** 2 for d in dts) / len(dts)) ** 0.5, 1))
        problems = []
        if out["fps"] < min_fps:
            problems.append(f"only {out['fps']:.1f} fps (at least {min_fps:g} wanted)")
        if out["max_gap_s"] > max_gap_s:
            problems.append(f"a wait of {out['max_gap_s']:.2f} s for one frame")
        if self.backend == "DSHOW" and out["fourcc"] and out["fourcc"] != "MJPG" and not self.is_file and out["width"] * out["height"] > 640 * 480:
            problems.append(f"the camera sends {out['fourcc']} (uncompressed), which is slow at this size")
        out["ok"] = not problems
        out["verdict"] = (f"Camera OK: {out['width']}x{out['height']}, {out['fps']:.1f} fps, longest wait {out['max_gap_s']:.2f} s."
                          if not problems else "Camera NOT good enough: " + "; ".join(problems) + ".")
        return out

    def settings_problems(self):
        """Plain-English warnings when the camera is not delivering what was asked for (live cameras only)."""
        if self.requested is None:
            return []
        g, r, out = self._granted, self.requested, []
        if (g["width"], g["height"]) != (r["width"], r["height"]):
            out.append(f"asked for {r['width']}x{r['height']} but the camera delivers {g['width']}x{g['height']}")
        m = self.measured_fps()
        if m is not None and m < 0.85 * r["fps"]:
            fmt = g.get("fourcc", "")
            if self.backend == "DSHOW" and fmt and fmt != "MJPG" and r["width"] * r["height"] > 640 * 480:
                out.append(f"asked for {r['fps']:g} fps but only {m:.1f} fps are arriving: the camera is sending uncompressed video "
                           f"({fmt}), which is limited to about 10 fps at this size; try a lower resolution or another camera")
            else:
                out.append(f"asked for {r['fps']:g} fps but only {m:.1f} fps are arriving: the camera may slow down in dim light, "
                           f"or this computer cannot keep up (frames are dropped while recording)")
        return out

    def lock_exposure_wb(self):
        """Switch automatic exposure and white balance off and hold their current values.

        Must be called from the thread that reads frames. Returns (and remembers) a report with the raw numbers read
        back from the driver, because many webcams accept the request without honouring it.
        """
        if not isinstance(self.source, int):
            self.lock_report = {"requested": False, "note": "not a webcam: nothing to lock"}
            return self.lock_report
        cap = self.cap
        exp0, wbt0 = cap.get(cv2.CAP_PROP_EXPOSURE), cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        ok_ae = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)       # DirectShow: 0.25 = manual, 0.75 = automatic
        ok_exp = cap.set(cv2.CAP_PROP_EXPOSURE, exp0)           # hold the value the camera had settled on
        ok_awb = cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        ok_wbt = cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wbt0)
        ae, awb = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), cap.get(cv2.CAP_PROP_AUTO_WB)
        exposure_locked = bool(ok_ae and ok_exp) and ae in (0, 0.25)
        wb_locked = bool(ok_awb and ok_wbt) and awb == 0
        self.lock_report = {
            "requested": True,
            "exposure_locked": exposure_locked, "white_balance_locked": wb_locked,
            "readback": {"auto_exposure": ae, "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
                         "auto_wb": awb, "wb_temperature": cap.get(cv2.CAP_PROP_WB_TEMPERATURE)},
        }
        return self.lock_report

    def lock_summary(self):
        """One plain-English sentence about the last lock request, or '' if none was made."""
        r = getattr(self, "lock_report", None)
        if not r:
            return ""
        if not r["requested"]:
            return r["note"]
        e, w = r["exposure_locked"], r["white_balance_locked"]
        if e and w:
            return "Exposure and white balance are locked (as far as the camera reports)."
        if not e and not w:
            return "This camera did not accept the lock: exposure and white balance stay automatic. Use even, constant lighting."
        return ("Exposure locked; white balance stayed automatic." if e
                else "White balance locked; exposure stayed automatic.")

    def info(self):
        """Settings record for meta.json."""
        return {"requested": self.requested, "granted_by_driver": self._granted, "is_file": self.is_file,
                "exposure_wb_lock": getattr(self, "lock_report", None),
                "failed_reads": self.failed_reads, "reconnects": self.reconnects,
                "source_kind": self.kind, "frames_discarded": self.frames_discarded,
                "capture_backend": self.backend if isinstance(self.source, int) else None}

    def read(self):
        """Returns (frame, t) or (None, None). For files, t follows the video's own frame rate."""
        if self._grabber is not None:
            frame, t = self._grabber.next()
            ok = frame is not None
        else:
            ok, frame = self.cap.read()
            t = time.perf_counter()
        if not ok:
            return None, None
        self._n += 1
        if self.is_file:
            t = (self._n - 1) / self.file_fps
        if self.is_live:
            self._stamps.append(t)
            if len(self._stamps) > 400:
                del self._stamps[0]
        return frame, t

    @property
    def position(self):
        """Index of the frame most recently returned by read() (files only)."""
        return self._n - 1

    @property
    def frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_file else 0

    def seek(self, index):
        """Make the next read() return frame `index` (files only)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self._n = index

    def close(self):
        self._stop_grabber()
        self.cap.release()


def camera_names():
    """Names of the video devices Windows knows (webcams, and phone apps such as Iriun, DroidCam or Camo that appear as
    webcams), in the order OpenCV's DirectShow indexes use. Opens no camera. None if they cannot be listed."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return [str(n) for n in FilterGraph().get_input_devices()]
    except Exception:  # noqa: BLE001 - pygrabber missing or COM not available
        return None


def list_cameras(max_index=6):
    """[(index, name)] of the cameras. By name where Windows can list them (fast, opens nothing); otherwise each index is
    tried (slow, about a second per camera) and named 'Camera N'. Never call this while a camera is streaming."""
    names = camera_names()
    if names is not None:
        return list(enumerate(names))
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append((i, f"Camera {i}"))
        cap.release()
    return found
