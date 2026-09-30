"""Frame sources. Everything the app needs from a camera is: open, read() -> (frame, time), close.

Phones will appear here as well: apps such as Iriun / Camo / DroidCam show up as an
ordinary webcam index, and a network stream is just a URL string.
"""
import time

import cv2


class CameraSource:
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.is_file = isinstance(source, str) and not source.lower().startswith(("rtsp://", "http://", "https://"))
        self.is_live = not self.is_file
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._t0 = time.perf_counter()
        self._n = 0

    @property
    def description(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"{self.source} ({w}x{h})"

    def read(self):
        """Returns (frame, t) or (None, None). For files, t follows the video's own frame rate."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        self._n += 1
        t = time.perf_counter() if self.is_live else (self._n - 1) / self.file_fps
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
