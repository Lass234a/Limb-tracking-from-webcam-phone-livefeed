"""Frame sources. Everything the app needs from a camera is: open, read() -> (frame, time), close.

Phones will appear here as well: apps such as Iriun / Camo / DroidCam show up as an
ordinary webcam index, and a network stream is just a URL string.
"""
import time
from pathlib import Path

import cv2


class CameraSource:
    failed_reads = 0       # reads that returned no frame on a live source (counted, never silent)
    reconnects = 0

    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.is_file = isinstance(source, str) and not source.lower().startswith(("rtsp://", "http://", "https://"))
        self.is_live = not self.is_file
        self._size_fps = (width, height, fps)
        self.failed_reads = 0       # reads that returned no frame on a live source (counted, never silent)
        self.reconnects = 0
        self.cap = self._open_capture()
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.requested = {"width": width, "height": height, "fps": fps} if isinstance(source, int) else None
        self._t0 = time.perf_counter()
        self._n = 0
        self._granted = self.granted()      # read once; the window may ask for it from another thread
        self._stamps = []          # recent frame arrival times (live sources): the real frame rate

    def _open_capture(self):
        source, (width, height, fps) = self.source, self._size_fps
        if isinstance(source, int):
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
            return cap
        if not self.is_file:        # network stream: do not wait half a minute on a stalled connection
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000])
        return cv2.VideoCapture(source)

    def reopen(self):
        """Close and open the same live source again (after it stopped delivering frames). Returns True if it opened.
        A webcam comes back on its automatic settings, so an exposure/white-balance lock has to be pressed again."""
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass
        self.reconnects += 1
        self.cap = self._open_capture()
        ok = self.cap.isOpened()
        if ok:
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
        stamps = list(self._stamps)
        if len(stamps) < 10 or stamps[-1] <= stamps[0]:
            return None
        return (len(stamps) - 1) / (stamps[-1] - stamps[0])

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
            if fmt and fmt != "MJPG" and r["width"] * r["height"] > 640 * 480:
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
                "failed_reads": self.failed_reads, "reconnects": self.reconnects}

    def read(self):
        """Returns (frame, t) or (None, None). For files, t follows the video's own frame rate."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        self._n += 1
        t = time.perf_counter() if self.is_live else (self._n - 1) / self.file_fps
        if self.is_live:
            self._stamps.append(t)
            if len(self._stamps) > 90:
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
        self.cap.release()


def list_cameras(max_index=6):
    """Indexes of webcams that open. Slow (about a second per camera), so call it on demand."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
