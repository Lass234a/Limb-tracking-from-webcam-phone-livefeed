"""Writes one trial to disk: raw video, overlay video, per-frame CSV and a small JSON."""
import csv
import json
import math
import os
import queue
import re
import statistics
import threading
from datetime import datetime
from pathlib import Path

import cv2

from .pull import COLUMNS as PULL_COLUMNS
from .results import LOST, OK, OUT


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "x"


def _num(v, nd=3):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


class FrameSizeChanged(RuntimeError):
    """A frame arrived with another size than the videos were opened for (phone rotated, camera changed resolution)."""


class _VideoThread(threading.Thread):
    """Encodes the frames of one trial off the tracking thread, so tracking and encoding overlap. Frames are never
    dropped: the queue is bounded, and when it is full the caller waits (a slow disk slows the program down instead of
    silently desynchronising the video from the CSV)."""

    QUEUE_FRAMES = 90

    def __init__(self, raw, ovl):
        super().__init__(daemon=True)
        self.raw, self.ovl = raw, ovl
        self.q = queue.Queue(maxsize=self.QUEUE_FRAMES)
        self.error = None
        self.peak = 0
        self.written = 0

    def put(self, frame, overlay, timeout=10.0):
        self.q.put((frame, overlay), timeout=timeout)       # raises queue.Full if the writer is stuck for `timeout` seconds
        self.peak = max(self.peak, self.q.qsize())

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if self.error is not None:
                continue                                      # keep draining so the producer is never blocked for good
            try:
                self.raw.write(item[0])
                if self.ovl is not None:
                    self.ovl.write(item[1])
                self.written += 1
            except Exception as e:  # noqa: BLE001 - reported to the recording through write()/close()
                self.error = e


class TrialRecorder:
    def __init__(self, root, participant, test_key, target, dot_names, angle_names, fps, frame_size, with_pull=False,
                 save_overlay=True):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.started_at = datetime.now().isoformat(timespec="milliseconds")
        folder = Path(root) / _safe(participant)
        folder.mkdir(parents=True, exist_ok=True)
        tgt = "" if target is None else f"_{target:g}deg"
        self.base = folder / f"{_safe(participant)}_{_safe(test_key)}{tgt}_{stamp}"
        self.save_overlay = bool(save_overlay)
        names = ("raw.mp4", "overlay.mp4", "data.csv", "meta.json") if self.save_overlay else ("raw.mp4", "data.csv", "meta.json")
        self.paths = {k: Path(f"{self.base}_{k}") for k in names}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._raw = cv2.VideoWriter(str(self.paths["raw.mp4"]), fourcc, fps, frame_size)
        self._ovl = cv2.VideoWriter(str(self.paths["overlay.mp4"]), fourcc, fps, frame_size) if self.save_overlay else None
        if not (self._raw.isOpened() and (self._ovl is None or self._ovl.isOpened())):
            raise RuntimeError(f"Could not open the video writers at {fps:.1f} fps, {frame_size[0]}x{frame_size[1]} "
                               "(check disk space and folder permissions).")
        self._video = _VideoThread(self._raw, self._ovl)
        self._video.start()
        self._fh = open(self.paths["data.csv"], "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._fh)
        self.dot_names, self.angle_names = list(dot_names), list(angle_names)
        head = ["frame", "t_s", "clock_s", "locked"]
        for d in self.dot_names:
            head += [f"{d}_x", f"{d}_y", f"{d}_lost", f"{d}_q"]
        for a in self.angle_names:
            head += [f"{a}_deg", f"{a}_ref", f"{a}_dev", f"{a}_ok"]
        head += ["all_ok", "event", "scale_px_per_cm"]
        self.with_pull = with_pull
        if with_pull:
            head += PULL_COLUMNS
        head.append("source_frame")        # frame number in the source (file frame, or camera read count)
        head += [f"{d}_partial" for d in self.dot_names]      # 1 = the dot looked partly covered / cut off in this frame
        self._csv.writerow(head)
        self.frames = 0
        self.events = []          # {frame, t_s, label} markers added during the trial
        self.last_t = 0.0
        self._t0 = None
        self.gaps = []             # {frame, t_s, gap_s}: stretches where no frame arrived
        self._partial = {d: 0 for d in self.dot_names}      # frames in which each dot looked partly covered
        self._dts, self._prev_t = [], None
        self.fps = fps
        self.frame_size = frame_size

    def _write_meta(self, data):
        """Write meta.json in one piece: a crash cannot leave half a file."""
        tmp = self.paths["meta.json"].with_name(self.paths["meta.json"].name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.paths["meta.json"])

    def write_start_meta(self, meta):
        """A complete meta.json at the start (status 'recording'), so a trial that is cut short can still be reviewed."""
        data = dict(meta)
        data.update({"status": "recording", "started_at": self.started_at, "frames": None, "duration_s": None,
                     "video_fps_written": round(self.fps, 3), "frame_size": list(self.frame_size),
                     "files": {k: p.name for k, p in self.paths.items()}})
        self._write_meta(data)

    def write(self, frame, overlay, frame_index, t, locked, dots, angles, event="", scale_px_per_cm=None, pull=None, source_frame=None):
        if (frame.shape[1], frame.shape[0]) != tuple(self.frame_size) or overlay.shape[:2] != frame.shape[:2]:
            raise FrameSizeChanged(f"The picture size changed from {self.frame_size[0]}x{self.frame_size[1]} to "
                                   f"{frame.shape[1]}x{frame.shape[0]}.")
        if self._t0 is None:
            self._t0 = t
        if self._prev_t is not None:
            dt = t - self._prev_t
            if len(self._dts) >= 5 and dt > max(3.0 * statistics.median(self._dts), 0.15):
                self.gaps.append({"frame": self.frames, "t_s": round(t - self._t0, 4), "gap_s": round(dt, 4)})
            else:
                self._dts = (self._dts + [dt])[-60:]
        self._prev_t = t
        if self._video.error is not None:
            raise RuntimeError(f"Saving the video failed: {self._video.error}")
        try:
            self._video.put(frame, overlay)
        except queue.Full:
            raise RuntimeError("Saving the video is not keeping up (the disk may be full or very slow).") from None
        row = [frame_index, round(t - self._t0, 5), round(t, 5), int(locked)]
        for d in self.dot_names:
            st = dots.get(d)
            if st is None:
                row += ["", "", "", ""]
            elif st.lost:                       # no position is known while a dot is lost: leave x/y blank
                row += ["", "", 1, round(st.quality, 2)]
            else:
                row += [round(st.x, 2), round(st.y, 2), 0, round(st.quality, 2)]
        by_name = {a.name: a for a in angles}
        oks = []
        for n in self.angle_names:
            a = by_name[n]
            ok = 1 if a.raw_status == OK else 0 if a.raw_status == OUT else ""
            if a.raw_status in (OUT, LOST):
                oks.append(False)
            row += [_num(a.raw_value), _num(a.reference), _num(a.raw_deviation), ok]       # unsmoothed values
        row.append(0 if oks else 1)
        row.append(event or "")
        row.append(_num(scale_px_per_cm, 4))        # live ruler scale this frame (blank: no ruler marked, or a ruler dot lost)
        if self.with_pull:                          # angle of pull / moment arm; blank when the strap, anchor or segment dots are lost
            row += [_num((pull or {}).get(c), 2) for c in PULL_COLUMNS]
        row.append("" if source_frame is None else int(source_frame))
        for d in self.dot_names:
            st = dots.get(d)
            row.append("" if st is None or st.lost else int(bool(getattr(st, "partial", False))))
        if event:
            self.events.append({"frame": self.frames, "t_s": round(t - self._t0, 4), "label": event})
        for d in self.dot_names:
            st = dots.get(d)
            if st is not None and getattr(st, 'partial', False):
                self._partial[d] += 1
        self._csv.writerow(row)
        self.frames += 1
        self.last_t = t - self._t0
        self._fh.flush()        # every row: a crash loses at most the frame being written

    def close(self, meta):
        errors = []
        self._video.q.put(None)                                # the writer thread finishes everything queued, then stops
        self._video.join(timeout=120)
        if self._video.is_alive():
            errors.append("video writer: did not finish within 2 minutes")
        if self._video.error is not None:
            errors.append(f"video writer: {self._video.error}")
        closers = [("raw video", self._raw.release)] + ([("overlay video", self._ovl.release)] if self._ovl is not None else []) \
            + [("csv", self._fh.close)]
        for what, fn in closers:
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - keep closing the rest, and still write meta.json
                errors.append(f"{what}: {e}")
        meta = dict(meta)
        meta.update({
            "status": "complete",
            "started_at": self.started_at,
            "events": self.events,
            "frames": self.frames,
            "duration_s": round(self.last_t, 4),
            "video_fps_written": round(self.fps, 3),
            "measured_fps": round((self.frames - 1) / self.last_t, 3) if self.frames > 1 and self.last_t > 0 else None,
            "frame_size": list(self.frame_size),
            "files": {k: p.name for k, p in self.paths.items()},
            "partial_dot_frames": {d: self._partial[d] for d in self.dot_names if self._partial[d]},
            "gaps": {"count": len(self.gaps), "total_s": round(sum(g["gap_s"] for g in self.gaps), 4),
                     "max_s": max((g["gap_s"] for g in self.gaps), default=0.0), "list": self.gaps,
                     "meaning": "stretches longer than 3 normal frame intervals (and 0.15 s) with no frame; the angles during a gap are unknown"},
            "video_writer": {"overlay_video_saved": self.save_overlay, "frames_encoded": self._video.written,
                             "queue_peak": self._video.peak, "queue_capacity": _VideoThread.QUEUE_FRAMES},
        })
        if errors:
            meta["close_errors"] = errors
        self._write_meta(meta)
        return self.paths
