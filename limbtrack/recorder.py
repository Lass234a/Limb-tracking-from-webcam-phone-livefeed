"""Writes one trial to disk: raw video, overlay video, per-frame CSV and a small JSON."""
import csv
import json
import math
import os
import re
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


class TrialRecorder:
    def __init__(self, root, participant, test_key, target, dot_names, angle_names, fps, frame_size, with_pull=False):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.started_at = datetime.now().isoformat(timespec="milliseconds")
        folder = Path(root) / _safe(participant)
        folder.mkdir(parents=True, exist_ok=True)
        tgt = "" if target is None else f"_{target:g}deg"
        self.base = folder / f"{_safe(participant)}_{_safe(test_key)}{tgt}_{stamp}"
        self.paths = {k: Path(f"{self.base}_{k}") for k in ("raw.mp4", "overlay.mp4", "data.csv", "meta.json")}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._raw = cv2.VideoWriter(str(self.paths["raw.mp4"]), fourcc, fps, frame_size)
        self._ovl = cv2.VideoWriter(str(self.paths["overlay.mp4"]), fourcc, fps, frame_size)
        if not (self._raw.isOpened() and self._ovl.isOpened()):
            raise RuntimeError("Could not open the video writers (check disk space and folder permissions).")
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
        head.append("source_frame")        # last column: frame number in the source (file frame, or camera read count)
        self._csv.writerow(head)
        self.frames = 0
        self.events = []          # {frame, t_s, label} markers added during the trial
        self.last_t = 0.0
        self._t0 = None
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
        self._raw.write(frame)
        self._ovl.write(overlay)
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
        if event:
            self.events.append({"frame": self.frames, "t_s": round(t - self._t0, 4), "label": event})
        self._csv.writerow(row)
        self.frames += 1
        self.last_t = t - self._t0
        self._fh.flush()        # every row: a crash loses at most the frame being written

    def close(self, meta):
        errors = []
        for what, fn in (("raw video", self._raw.release), ("overlay video", self._ovl.release), ("csv", self._fh.close)):
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
        })
        if errors:
            meta["close_errors"] = errors
        self._write_meta(meta)
        return self.paths
