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

Two measures that do not depend on the saved track being right:
  * marking_offset_px (per dot): how far the tracker's marking at the first picture lands from the position the program saved
    there (empty = the marking failed). A colour or snapping fault shows here at once.
  * apart_frames (one row per pair of dots on one bone, `rigid_pairs` of protocol.json, dot column "a+b"): the pictures in which
    the distance between the two changed by more than 25 % from when they were first seen together: the program's own CHECK DOTS
    rule (engine.RIGID_PAIR_CHANGE), here applied to each tracker's replay. saved_apart_frames is what the live program flagged
    in data.csv ('dots_apart'); apart_both is the overlap. Fewer apart_frames with the same saved_apart_frames overlap = fewer jumps.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def replay_all(mods, meta, rows, video, dots, i0, crop, full_size):
    """Track from row i0 to the end with every tracker version in `mods` ({name: module}) over ONE pass through the video (the
    picture is decoded once, not once per version). Returns, per version name: {dot: [(x, y, lost)] per row from i0}, the dots that
    could not be marked, each dot's diameter (pixels) as learned when it was marked, and how far each marking landed from the saved
    position at row i0 (pixels; a dot that could not be marked has no entry)."""
    cap = cv2.VideoCapture(str(video))
    kind = meta.get("dot_kind") or "auto"
    sets = {v: m.TrackerSet(dots, kind=kind) for v, m in mods.items()}
    to_gray = next(iter(mods.values())).to_gray
    out = {v: {d: [] for d in dots} for v in mods}
    failed = {v: set() for v in mods}
    offset = {v: {} for v in mods}
    nan = float("nan")
    for i in range(len(rows)):
        ok, img = cap.read()
        if not ok:
            for v in mods:
                for d in dots:
                    out[v][d].append((nan, nan, True))
            continue
        if i < i0:
            continue
        if crop and full_size:
            img = privacy.pad_to_full(img, crop, full_size)
        gray = to_gray(img)
        for v, ts in sets.items():
            if i == i0:
                for d in dots:
                    sx, sy = _f(rows[i][f"{d}_x"]), _f(rows[i][f"{d}_y"])
                    if ts.init_dot(d, gray, sx, sy):
                        offset[v][d] = float(np.hypot(ts.trackers[d].x - sx, ts.trackers[d].y - sy))
                    else:
                        failed[v].add(d)
                        ts.trackers[d].initialised = False
                for d in dots:
                    t = ts.trackers[d]
                    out[v][d].append((t.x, t.y, d in failed[v]))
            else:
                states = ts.update(gray)
                for d in dots:
                    s = states.get(d)
                    out[v][d].append((s.x, s.y, s.lost) if s else (nan, nan, True))
    cap.release()
    diam = {v: {d: (sets[v].trackers[d].marked_diam or 10.0) for d in dots} for v in mods}
    return {v: (out[v], failed[v], diam[v], offset[v]) for v in mods}


def replay(tracker_mod, meta, rows, video, dots, i0, crop, full_size, masks):
    """One tracker version over the video (see replay_all). Returns the track, the dots that could not be marked and each dot's
    diameter. `masks` is accepted for older callers and not used."""
    out, failed, diam, _ = replay_all({"v": tracker_mod}, meta, rows, video, dots, i0, crop, full_size)["v"]
    return out, failed, diam


# ---- dots on one bone: the program's own CHECK DOTS rule (protocol.json rigid_pairs, engine.RIGID_PAIR_CHANGE)
RIGID_PAIR_CHANGE = 0.25
_APART = re.compile(r"([a-z0-9_]+)-([a-z0-9_]+):[+-]?\d+")


def read_rigid_pairs(protocol_path):
    try:
        return [tuple(p) for p in json.loads(Path(protocol_path).read_text(encoding="utf-8")).get("rigid_pairs", [])]
    except (OSError, ValueError):
        return []


def apart_frames(tracks, pairs, i0, tol=RIGID_PAIR_CHANGE):
    """{(a, b): rows in which the distance of the pair differs from its distance when both were first seen by more than `tol`}.
    Only pictures in which both dots are tracked count (a lost dot has no position)."""
    out = {}
    for a, b in pairs:
        if a not in tracks or b not in tracks:
            continue
        ref, flagged = None, set()
        for k, (pa, pb) in enumerate(zip(tracks[a], tracks[b])):
            if pa[2] or pb[2]:
                continue
            d = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            if ref is None:
                ref = d
            elif ref > 1e-6 and abs(d / ref - 1.0) > tol:
                flagged.add(i0 + k)
        out[(a, b)] = flagged
    return out


def saved_apart(rows, pairs):
    """{(a, b): rows the live program flagged CHECK DOTS for that pair} from data.csv's 'dots_apart' column (e.g. "fibular_head-malleolus:+40")."""
    out = {tuple(p): set() for p in pairs}
    for r, row in enumerate(rows):
        for m in _APART.finditer(row.get("dots_apart") or ""):
            key = (m.group(1), m.group(2))
            if key in out:
                out[key].add(r)
    return out


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


def code_signature(candidate, protocol=None):
    """Changes whenever the tracker code, this script or the protocol file (rigid_pairs) change, so cached numbers are never reused
    for different code."""
    h = hashlib.sha1()
    extra = ([Path(candidate)] if candidate else []) + ([Path(protocol)] if protocol and Path(protocol).is_file() else [])
    for p in sorted((HERE.parent / "limbtrack").glob("*.py")) + [Path(__file__)] + extra:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def trial_signature(meta_path, meta, tol, code_sig):
    """Changes whenever the trial's files, the tolerance or the code change."""
    folder = Path(meta_path).parent
    parts = [code_sig, repr(tol)]
    for p in [Path(meta_path)] + [folder / v for v in (meta.get("files") or {}).values()]:
        try:
            st = p.stat()
            parts.append(f"{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append("missing")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def process_trial(job):
    """One trial through every version. Runs in a worker process. Returns (label, report rows, printed line)."""
    label, mp, i0, candidate, tol_factor, protocol = job
    versions = {"current": base_tracker}
    if candidate:
        versions["candidate"] = load_tracker(candidate)
    meta, rows, _ = read_trial(mp)
    info = {"trial": label, "test": meta.get("test"), "target": meta.get("target_deg", meta.get("target")),
            "software": meta.get("software_version"), "frames": len(rows)}
    pv = meta.get("privacy") or {}
    crop = tuple(pv["crop_xywh"]) if pv.get("crop_xywh") else None
    full = tuple(pv["full_picture_size"]) if pv.get("full_picture_size") else None
    masks = [tuple(m) for m in pv.get("masks_xywh") or []]
    dots = dots_of(rows)
    video = Path(mp).parent / meta["files"]["raw.mp4"]
    pairs = [p for p in read_rigid_pairs(protocol) if p[0] in dots and p[1] in dots]
    saved = saved_apart(rows, pairs)
    ran = replay_all(versions, meta, rows, video, dots, i0, crop, full)
    report, tracks = [], {}
    for vname in versions:
        tracks[vname], failed, diam, offset = ran[vname]
        for d in dots:
            res = compare(tracks[vname][d], rows, i0, d, tol_factor * diam[d], masks)
            report.append({**info, "version": vname, "dot": d, "marked_at_frame": i0, "init_failed": d in failed,
                           "marking_offset_px": round(offset[d], 2) if d in offset else None, **res})
        for pair, flagged in apart_frames(tracks[vname], pairs, i0).items():
            s = saved[pair]
            report.append({**info, "version": vname, "dot": f"{pair[0]}+{pair[1]}", "marked_at_frame": i0,
                           "apart_frames": len(flagged), "saved_apart_frames": len(s), "apart_both": len(flagged & s)})
    if "candidate" in versions:             # the two versions against each other
        for d in dots:
            a, b = tracks["current"][d], tracks["candidate"][d]
            moved = sum(1 for p, q in zip(a, b) if not p[2] and not q[2] and np.hypot(p[0] - q[0], p[1] - q[1]) > 2.0)
            report.append({**info, "version": "current_vs_candidate", "dot": d, "marked_at_frame": i0,
                           "differs": moved, "replay_lost_saved_ok": sum(1 for p, q in zip(a, b) if not p[2] and q[2]),
                           "saved_lost_replay_found": sum(1 for p, q in zip(a, b) if p[2] and not q[2])})
    return label, report, f"{label} {info['test']} {len(rows)} frames, {len(dots)} dots done"


def read_key(path):
    """The T-numbers handed out by earlier runs: a trial keeps its number when more recordings are added."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return {r["meta_file"]: r["trial"] for r in csv.DictReader(fh)}
    except OSError:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="?", default=str(HERE.parent / "recordings"))
    ap.add_argument("--candidate", help="path to another version of tracker.py to compare with the one in this folder")
    ap.add_argument("--out", default="replay_report.csv", help="the numbers (safe to share)")
    ap.add_argument("--key", default="replay_key.csv", help="which T-number is which trial (keep private)")
    ap.add_argument("--cache", default="replay_cache.json", help="numbers of trials already replayed (same code, same files: not run again)")
    ap.add_argument("--no-cache", action="store_true", help="replay every trial again")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="trials replayed at the same time (default: processors - 1)")
    ap.add_argument("--tol", type=float, default=0.75, help="positions within this x the dot's diameter count as the same (default 0.75)")
    ap.add_argument("--protocol", default=str(HERE.parent / "protocol.json"), help="protocol.json whose rigid_pairs give the dots on one bone")
    ap.add_argument("--max-trials", type=int, default=0, help="stop after this many trials (0 = all)")
    args = ap.parse_args()

    metas = sorted(Path(args.recordings).rglob("*_meta.json"))
    if not metas:
        print(f"No *_meta.json found under {args.recordings}")
        return 1

    # Pass 1 (quick): which trials can be replayed, and what T-number each has. Old numbers are kept, new ones come after them.
    old = read_key(args.key)
    used = {int(v[1:]) for v in old.values() if v[1:].isdigit()}
    nxt = max(used, default=0) + 1
    code_sig = code_signature(args.candidate, args.protocol)
    plan = []                               # (label, meta path, i0, signature)
    for mp in metas:
        if args.max_trials and len(plan) >= args.max_trials:
            break
        meta, rows, why = read_trial(mp)
        if rows is None:
            print(f"skipped one trial: {why}")
            continue
        i0 = start_frame(rows, dots_of(rows))
        if i0 is None:
            print("skipped one trial: no frame where every dot is marked")
            continue
        label = old.get(str(mp))
        if label is None:
            label = f"T{nxt:02d}"
            nxt += 1
        plan.append((label, mp, i0, trial_signature(mp, meta, args.tol, code_sig)))

    cache = {}
    if not args.no_cache:
        try:
            cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    results, todo = {}, []
    for label, mp, i0, sig in plan:
        hit = cache.get(label)
        if hit and hit.get("sig") == sig:
            results[label] = hit["report"]
        else:
            todo.append((label, mp, i0, args.candidate, args.tol, args.protocol))
    print(f"{len(plan)} trials: {len(results)} unchanged (from cache), {len(todo)} to replay"
          f"{f' on {min(args.jobs, len(todo))} processes' if len(todo) > 1 else ''}")

    sigs = {label: sig for label, _, _, sig in plan}
    def done(label, report, line):
        results[label] = report
        cache[label] = {"sig": sigs[label], "report": report}
        print(line, flush=True)
    if args.jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(todo))) as pool:
            for fut in as_completed([pool.submit(process_trial, j) for j in todo]):
                done(*fut.result())
    else:
        for j in todo:
            done(*process_trial(j))

    report = [r for label, _, _, _ in plan for r in results.get(label, [])]
    if not report:
        print("Nothing to compare.")
        return 1
    if not args.no_cache:
        Path(args.cache).write_text(json.dumps({l: cache[l] for l, *_ in plan}), encoding="utf-8")
    first = ["trial", "test", "target", "software", "frames", "version", "dot"]
    cols = sorted({k for r in report for k in r}, key=lambda c: (first.index(c) if c in first else 99, c))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(report)
    with open(args.key, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "meta_file"])
        w.writeheader()
        w.writerows({"trial": label, "meta_file": str(mp)} for label, mp, _, _ in plan)
    versions = ["current"] + (["candidate", "current_vs_candidate"] if args.candidate else [])
    print("\nSummary per version (all dots, all trials):")
    for vname in versions:
        rs = [r for r in report if r["version"] == vname]
        tot = {k: sum(int(r.get(k) or 0) for r in rs) for k in ("agree", "differs", "replay_lost_saved_ok", "saved_lost_replay_found")}
        print(f"  {vname:22s} {tot}")
    print("\nDots on one bone that move apart (the program's CHECK DOTS rule, pictures) and markings off by more than 3 px or failed:")
    for vname in versions[:2] if args.candidate else versions[:1]:
        rs = [r for r in report if r["version"] == vname]
        pairs_ = [r for r in rs if "+" in str(r["dot"])]
        dots_ = [r for r in rs if "+" not in str(r["dot"])]
        off = sum(1 for r in dots_ if r.get("init_failed") or (r.get("marking_offset_px") not in (None, "") and float(r["marking_offset_px"]) > 3.0))
        print(f"  {vname:22s} apart in {sum(int(r.get('apart_frames') or 0) for r in pairs_)} pictures "
              f"(the live program flagged {sum(int(r.get('saved_apart_frames') or 0) for r in pairs_)}, in both {sum(int(r.get('apart_both') or 0) for r in pairs_)}); "
              f"markings off or failed: {off} of {len(dots_)} dots")
    print(f"\nWrote {args.out} (numbers only: safe to share) and {args.key} (keep private).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
