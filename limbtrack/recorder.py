"""Writes one trial to disk: raw video, overlay video, per-frame CSV and a small JSON."""
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import cv2

from .results import LOST, OK, OUT


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "x"


def _num(v, nd=3):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


class TrialRecorder:
    def __init__(self, root, participant, test_key, target, dot_names, angle_names, fps, frame_size):
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
        head += ["all_ok", "event"]
        self._csv.writerow(head)
        self.frames = 0
        self.events = []          # {frame, t_s, label} markers added during the trial
        self.last_t = 0.0
        self._t0 = None
        self.fps = fps
        self.frame_size = frame_size

    def write(self, frame, overlay, frame_index, t, locked, dots, angles, event=""):
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
        if event:
            self.events.append({"frame": self.frames, "t_s": round(t - self._t0, 4), "label": event})
        self._csv.writerow(row)
        self.frames += 1
        self.last_t = t - self._t0
        if self.frames % 30 == 0:
            self._fh.flush()

    def close(self, meta):
        self._raw.release()
        self._ovl.release()
        self._fh.close()
        meta = dict(meta)
        meta.update({
            "started_at": self.started_at,
            "events": self.events,
            "frames": self.frames,
            "duration_s": round(self.last_t, 4),
            "video_fps_written": round(self.fps, 3),
            "measured_fps": round((self.frames - 1) / self.last_t, 3) if self.frames > 1 and self.last_t > 0 else None,
            "frame_size": list(self.frame_size),
            "files": {k: p.name for k, p in self.paths.items()},
        })
        self.paths["meta.json"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return self.paths
