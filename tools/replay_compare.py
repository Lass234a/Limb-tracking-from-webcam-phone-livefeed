"""Replay saved trials through the dot tracker and compare, without any picture leaving the computer.

For every trial in the recordings folder (`..._meta.json` + `..._raw.mp4` + `..._data.csv`) the tracker is started on the dots
as the program had them, run over the saved video, and its track is compared with the one saved in data.csv. Only numbers are
written: no picture, no participant code, no file name (trials are called T01, T02, ...; `replay_key.csv` on this computer says
which is which and should NOT be shared).

Run from the project folder:
    .venv\\Scripts\\python.exe tools\\replay_compare.py                      (looks in recordings\\)
    .venv\\Scripts\\python.exe tools\\replay_compare.py D:\\other\\recordings
    .venv\\Scripts\\python.exe tools\\replay_compare.py --candidate C:\\path\\to\\other\\limbtrack\\tracker.py

--candidate runs a second version of the tracker over the same videos and compares both with the saved track and with each
other: this is how a change to the tracker is tried on real footage before it is accepted.

How to read it. The saved track is what the program followed during the trial, NOT the truth. 'differs' therefore means 'the
replay put the dot somewhere else', not 'the replay is wrong'. With the same tracker as the recording, almost every frame should
agree; what is left comes from the saved video being compressed (mp4v) and from clicks that were made while a dot was marked.
A candidate is better when it loses fewer visible dots, fills gaps sooner, and does not move dots to other places than the saved
track unless the saved track was wrong (look at the video yourself in Review for those frames: 'first_diff_frame').
"""
import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from limbtrack import privacy                       # noqa: E402
from limbtrack import tracker as base_tracker       # noqa: E402


def load_tracker(path):
    spec = importlib.util.spec_from_file_location("candidate_tracker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def read_trial(meta_path):
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    files = meta.get("files") or {}
    folder = Path(meta_path).parent
    raw, data = folder / files.get("raw.mp4", ""), folder / files.get("data.csv", "")
    if not raw.is_file() or not data.is_file():
        return meta, None, "video or data.csv missing"
    with open(data, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < 5:
        return meta, None, "fewer than 5 frames"
    return meta, rows, None


def dots_of(rows):
    return [k[:-5] for k in rows[0] if k.endswith("_lost")]


def start_frame(rows, dots):
    """First row in which every dot has a position and none is lost: where the replay marks them."""
    for i, r in enumerate(rows):
        if all(r[f"{d}_lost"] == "0" and r[f"{d}_x"] != "" for d in dots):
            return i
    return None


def replay(tracker_mod, meta, rows, video, dots, i0, crop, full_size, masks):
    """Track from row i0 to the end. Returns {dot: [(x, y, lost)] per row from i0}, the dots that could not be marked, and each
    dot's diameter (pixels) as learned when it was marked."""
    cap = cv2.VideoCapture(str(video))
    kind = meta.get("dot_kind") or "auto"
    ts = tracker_mod.TrackerSet(dots, kind=kind)
    out = {d: [] for d in dots}
    failed = set()
    n = len(rows)
    for i in range(n):
        ok, img = cap.read()
        if not ok:
            for d in dots:
                out[d].append((float("nan"), float("nan"), True))
            continue
        if i < i0:
            continue
        if crop and full_size:
            img = privacy.pad_to_full(img, crop, full_size)
        gray = tracker_mod.to_gray(img)
        if i == i0:
            for d in dots:
                if not ts.init_dot(d, gray, _f(rows[i][f"{d}_x"]), _f(rows[i][f"{d}_y"])):
                    failed.add(d)
                    ts.trackers[d].initialised = False
            for d in dots:
                t = ts.trackers[d]
                out[d].append((t.x, t.y, d in failed))
            continue
        states = ts.update(gray)
        for d in dots:
            s = states.get(d)
            out[d].append((s.x, s.y, s.lost) if s else (float("nan"), float("nan"), True))
    cap.release()
    diam = {d: (ts.trackers[d].marked_diam or 10.0) for d in dots}
    return out, failed, diam


def compare(track, rows, i0, dot, tol, masks):
    """Replay track of one dot against the saved one. Frames inside a privacy mask are left out (the video is blacked out there)."""
    agree = differs = new_lost = ref_lost_new_found = both_lost = masked = 0
    dist, longest, run, first_diff = [], 0, 0, None
    for k, (x, y, lost) in enumerate(track):
        r = rows[i0 + k]
        rx, ry, rlost = _f(r[f"{dot}_x"]), _f(r[f"{dot}_y"]), r[f"{dot}_lost"] != "0" or r[f"{dot}_x"] == ""
        if not rlost and any(privacy.inside(m, rx, ry) for m in masks):
            masked += 1
            continue
        run = run + 1 if lost else 0
        longest = max(longest, run)
        if rlost and lost:
            both_lost += 1
        elif rlost:
            ref_lost_new_found += 1
        elif lost:
            new_lost += 1
        else:
            d = float(np.hypot(x - rx, y - ry))
            if d <= tol:
                agree += 1
                dist.append(d)
            else:
                differs += 1
                if first_diff is None:
                    first_diff = i0 + k
    return {"agree": agree, "differs": differs, "replay_lost_saved_ok": new_lost, "saved_lost_replay_found": ref_lost_new_found,
            "both_lost": both_lost, "masked": masked, "longest_lost_run": longest, "first_diff_frame": first_diff,
            "median_dist_px": round(statistics.median(dist), 3) if dist else None,
            "p95_dist_px": round(float(np.percentile(dist, 95)), 3) if dist else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="?", default=str(HERE.parent / "recordings"))
    ap.add_argument("--candidate", help="path to another version of tracker.py to compare with the one in this folder")
    ap.add_argument("--out", default="replay_report.csv", help="the numbers (safe to share)")
    ap.add_argument("--key", default="replay_key.csv", help="which T-number is which trial (keep private)")
    ap.add_argument("--tol", type=float, default=0.75, help="positions within this x the dot's diameter count as the same (default 0.75)")
    ap.add_argument("--max-trials", type=int, default=0, help="stop after this many trials (0 = all)")
    args = ap.parse_args()

    metas = sorted(Path(args.recordings).rglob("*_meta.json"))
    if not metas:
        print(f"No *_meta.json found under {args.recordings}")
        return 1
    versions = {"current": base_tracker}
    if args.candidate:
        versions["candidate"] = load_tracker(args.candidate)
    report, key, tno = [], [], 0
    for mp in metas:
        if args.max_trials and tno >= args.max_trials:
            break
        meta, rows, why = read_trial(mp)
        label = f"T{tno + 1:02d}"
        info = {"trial": label, "test": meta.get("test"), "target": meta.get("target_deg", meta.get("target")),
                "software": meta.get("software_version"), "frames": len(rows) if rows else 0}
        if rows is None:
            print(f"{label} skipped: {why}")
            continue
        pv = meta.get("privacy") or {}
        crop = tuple(pv["crop_xywh"]) if pv.get("crop_xywh") else None
        full = tuple(pv["full_picture_size"]) if pv.get("full_picture_size") else None
        masks = [tuple(m) for m in pv.get("masks_xywh") or []]
        dots = dots_of(rows)
        i0 = start_frame(rows, dots)
        if i0 is None:
            print(f"{label} skipped: no frame where every dot is marked")
            continue
        tno += 1
        key.append({"trial": label, "meta_file": str(mp)})
        folder = mp.parent
        video = folder / meta["files"]["raw.mp4"]
        tracks = {}
        for vname, mod in versions.items():
            tracks[vname], failed, diam = replay(mod, meta, rows, video, dots, i0, crop, full, masks)
            for d in dots:
                res = compare(tracks[vname][d], rows, i0, d, args.tol * diam[d], masks)
                report.append({**info, "version": vname, "dot": d, "marked_at_frame": i0, "init_failed": d in failed, **res})
        if "candidate" in versions:         # the two versions against each other
            for d in dots:
                a, b = tracks["current"][d], tracks["candidate"][d]
                moved = sum(1 for p, q in zip(a, b) if not p[2] and not q[2] and np.hypot(p[0] - q[0], p[1] - q[1]) > 2.0)
                report.append({**info, "version": "current_vs_candidate", "dot": d, "marked_at_frame": i0,
                               "differs": moved, "replay_lost_saved_ok": sum(1 for p, q in zip(a, b) if not p[2] and q[2]),
                               "saved_lost_replay_found": sum(1 for p, q in zip(a, b) if p[2] and not q[2])})
        print(f"{label} {info['test']} {len(rows)} frames, {len(dots)} dots done")
    if not report:
        print("Nothing to compare.")
        return 1
    cols = sorted({k for r in report for k in r}, key=lambda c: ["trial", "test", "target", "software", "frames", "version", "dot"].index(c)
                  if c in ("trial", "test", "target", "software", "frames", "version", "dot") else 99)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(report)
    with open(args.key, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "meta_file"])
        w.writeheader()
        w.writerows(key)
    print(f"\nSummary per version (all dots, all trials):")
    for vname in list(versions) + (["current_vs_candidate"] if "candidate" in versions else []):
        rs = [r for r in report if r["version"] == vname]
        tot = {k: sum(int(r.get(k) or 0) for r in rs) for k in ("agree", "differs", "replay_lost_saved_ok", "saved_lost_replay_found")}
        print(f"  {vname:22s} {tot}")
    print(f"\nWrote {args.out} (numbers only: safe to share) and {args.key} (keep private).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
