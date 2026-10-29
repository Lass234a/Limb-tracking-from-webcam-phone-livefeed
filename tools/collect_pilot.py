#!/usr/bin/env python3
"""Collect the correct (clean) MVIC measurements of the limbtrack pilot (P01, P02) into one local collection.

Standard library only. Reads <pilot>/**/*_meta.json + *_data.csv (+ *_force.csv), writes
  measurements.csv  one row per correct measurement: position, target, force, tempo, angle at the hold, dots, file paths
  dots.csv          one row per dot per correct measurement: pictures, lost, quality, partly covered, position at the hold
  positions.csv     one row per participant and position: trials, correct trials, spread of the hold angle
  excluded.csv      every trial that is NOT in the collection, with the reason
  README.md         the criteria, the columns and the caveats
  data/             a copy of meta.json, data.csv and force.csv of every correct measurement (numbers only; no video is copied)

usage (from the project folder): python tools/collect_pilot.py [--pilot Pilot] [--out Pilot_collection] [--participants P01 P02] [--no-copy]
"""
import argparse
import csv
import json
import re
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MAIN_ANGLE = {"knee_extension": "knee_deg", "knee_flexion": "knee_deg", "hip_extension_k0": "hip_deg", "hip_extension_k90": "hip_deg",
              "hip_flexion": "hip_deg", "hip_abduction": "abd_goniometer_deg", "hip_adduction": "abd_goniometer_deg", "belt_squat": "knee_deg"}
NAME = re.compile(r"^(P\d+)_(\d+)_(.+?)(?:_(-?\d+)deg)?_(T\d+)_(\d{8}-\d{6})_meta\.json$")
PULL_DOTS = ("strap", "anchor")


def find_pilot(given):
    """The pilot folder: the one given, else Pilot/ in the project folder (this script may sit in tools/), else Pilot/ in the current folder."""
    if given:
        p = Path(given)
        if not p.is_dir():
            sys.exit(f"pilot folder not found: {p}")
        return p
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "Pilot", here.parent.parent / "Pilot", Path.cwd() / "Pilot"):
        if cand.is_dir():
            return cand
    sys.exit("no Pilot/ folder found: give it with --pilot FOLDER")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    return statistics.pstdev(v) if len(v) > 1 else None


def r(x, n=2):
    return "" if x is None else round(x, n)


def analyse(mp, args):
    """-> (row, dot_rows, reason). reason is a list of why the trial is excluded (empty list = correct)."""
    m = NAME.match(mp.name)
    who, pos_no, test, tgt, trial, stamp = m.groups()
    meta = json.loads(mp.read_text(encoding="utf-8"))
    base = mp.name[:-len("_meta.json")]
    data_p = mp.with_name(base + "_data.csv")
    force_p = mp.with_name(base + "_force.csv")
    ident = {"id": base, "participant": who, "position_no": int(pos_no), "test": test, "target_deg": tgt if tgt is not None else "", "trial": trial,
             "started": stamp}
    if not data_p.is_file():
        return ident, [], ["no data.csv"]
    with open(data_p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < 20:
        return ident, [], ["fewer than 20 pictures"]
    t = [num(x["t_s"]) for x in rows]
    ev = [x.get("event", "") or "" for x in rows]
    starts = [i for i, e in enumerate(ev) if e.startswith("MVIC start")]
    ends = [i for i, e in enumerate(ev) if e.startswith("MVIC end")]
    reasons = []
    if not (starts and ends and ends[0] > starts[0]):
        return ident, [], ["no MVIC start and end marker"]
    a, b = starts[0], ends[0]
    window = t[b] - t[a]
    if not (args.window[0] <= window <= args.window[1]):
        reasons.append(f"MVIC window {window:.1f} s outside {args.window[0]}-{args.window[1]} s")
    dots = [k[:-5] for k in rows[0] if k.endswith("_lost")]
    core = [d for d in dots if d not in PULL_DOTS]
    win = range(a, b + 1)
    lost = {d: sum(1 for i in win if rows[i][f"{d}_lost"] != "0") for d in dots}
    core_lost = sum(lost[d] for d in core)
    if core_lost:
        reasons.append(f"a landmark dot was lost in {core_lost} dot-pictures of the MVIC window")
    flagged = sum(1 for i in win if rows[i].get("dots_apart") or rows[i].get("label_check"))
    if flagged:
        reasons.append(f"CHECK DOTS / CHECK LABELS in {flagged} pictures of the window")
    force = [num(x.get("force_n")) for x in rows]
    seg = [f for f in force[a:b + 1] if f is not None]
    allf = [f for f in force if f is not None]
    if len(seg) < 5:
        reasons.append("no force in newtons in the window")
        return ident, [], reasons
    rest = pct(allf, 5)
    peak = max(seg)
    if peak - rest < args.min_rise:
        reasons.append(f"force rise {peak - rest:.0f} N under {args.min_rise:.0f} N")
    if reasons:
        return ident, [], reasons

    # tempo from the force curve
    lo, hi, hi80 = rest + 0.1 * (peak - rest), rest + 0.9 * (peak - rest), rest + 0.8 * (peak - rest)
    sf = [f if f is not None else float("-inf") for f in force[a:b + 1]]
    i10 = next((i for i, f in enumerate(sf) if f >= lo), None)
    i90 = next((i for i, f in enumerate(sf) if f >= hi), None)
    rise_s = t[a + i90] - t[a + i10] if i10 is not None and i90 is not None else None
    dts = [t[i + 1] - t[i] for i in range(len(t) - 1) if t[i + 1] > t[i]]
    dt = statistics.median(dts) if dts else 0.033
    plateau = [a + i for i, f in enumerate(sf) if f >= hi80]
    hold80 = len(plateau) * dt
    tempo = ("close to 1-5-1" if rise_s is not None and 0.5 <= rise_s <= 1.5 and hold80 >= 4.0
             else "roughly 1-5-1" if rise_s is not None and 0.5 <= rise_s <= 2.5 and hold80 >= 3.0 else "other")

    # main angle at the hold
    col = MAIN_ANGLE.get(test)
    ang = [num(x.get(col)) if col in rows[0] else None for x in rows] if col else [None] * len(rows)
    pl_ang = [ang[i] for i in plateau if ang[i] is not None]
    ip = a + max(range(len(sf)), key=lambda i: sf[i])
    near_peak = [ang[i] for i in range(len(rows)) if ang[i] is not None and abs(t[i] - t[ip]) <= 0.1]
    pre = [ang[i] for i in range(max(a - 15, 0), a) if ang[i] is not None]
    plateau_angle = mean(pl_ang)
    rest_angle = statistics.median(pre) if pre else None
    target = num(tgt)
    dev = plateau_angle - target if plateau_angle is not None and target is not None else None

    partial = sum(1 for i in win for d in dots if num(rows[i].get(f"{d}_partial")) not in (None, 0.0) and rows[i].get(f"{d}_partial") not in ("", "0"))
    row = dict(ident)
    row.update({
        "software_version": meta.get("software_version", ""), "dot_kind": meta.get("dot_kind", ""), "pictures": len(rows),
        "window_s": r(window, 1), "peak_n": r(peak, 0), "rest_n": r(rest, 0), "rise_above_rest_n": r(peak - rest, 0),
        "rise_10_90_s": r(rise_s, 1), "hold_at_80pct_s": r(hold80, 1), "tempo": tempo,
        "main_angle_column": col if col and col in rows[0] else "", "hold_angle_deg": r(plateau_angle, 1),
        "hold_angle_sd_deg": r(sd(pl_ang), 1), "angle_at_peak_deg": r(mean(near_peak), 1), "rest_angle_deg": r(rest_angle, 1),
        "hold_minus_rest_deg": r(plateau_angle - rest_angle if plateau_angle is not None and rest_angle is not None else None, 1),
        "dev_from_target_deg": r(dev, 1), "within_5_deg_of_target": "" if dev is None else int(abs(dev) <= 5),
        "dots": " ".join(dots), "landmark_dots_lost_in_window": 0, "pull_dots_lost_in_window": sum(lost[d] for d in dots if d in PULL_DOTS),
        "partly_covered_dot_pictures_in_window": partial,
        "raw_video": str(mp.with_name(base + "_raw.mp4")) if mp.with_name(base + "_raw.mp4").is_file() else "",
        "overlay_video": str(mp.with_name(base + "_overlay.mp4")) if mp.with_name(base + "_overlay.mp4").is_file() else "",
        "force_csv": str(force_p) if force_p.is_file() else "",
    })
    dot_rows = []
    for d in dots:
        q = [num(rows[i].get(f"{d}_q")) for i in win if rows[i][f"{d}_lost"] == "0"]
        q = [x for x in q if x is not None]
        xs = [num(rows[i].get(f"{d}_x")) for i in plateau if rows[i][f"{d}_lost"] == "0"]
        ys = [num(rows[i].get(f"{d}_y")) for i in plateau if rows[i][f"{d}_lost"] == "0"]
        xs, ys = [x for x in xs if x is not None], [y for y in ys if y is not None]
        pp = sum(1 for i in win if rows[i].get(f"{d}_partial") not in (None, "", "0"))
        dot_rows.append({"id": base, "participant": who, "position_no": int(pos_no), "test": test, "target_deg": ident["target_deg"], "trial": trial,
                         "dot": d, "kind": "pull (strap/anchor)" if d in PULL_DOTS else "landmark", "pictures_in_window": len(win),
                         "lost_pictures": lost[d], "median_quality": r(statistics.median(q), 2) if q else "", "partly_covered_pictures": pp,
                         "x_at_hold_px": r(statistics.median(xs), 1) if xs else "", "y_at_hold_px": r(statistics.median(ys), 1) if ys else ""})
    return row, dot_rows, []


def write_csv(path, rows, cols=None):
    cols = cols or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


README = """# Pilot collection: the correct measurements

Made by the limbtrack-pilot-collection skill ({n_ok} correct measurements of {n_all} trials; participants {who}).
Numbers only. The videos are NOT copied: `raw_video` and `overlay_video` in measurements.csv point to the originals.
Keep this folder out of git and do not share it outside the project: it holds participant codes and dates.

## What "correct" means here
A trial is in the collection when ALL of this holds. It means technically complete and clean, NOT validated:
1. both MVIC markers exist and the window between them is {w0}-{w1} s (the automatic end marker is always 7.0 s after the start,
   so the window length says nothing about the tempo; see `tempo`)
2. no landmark dot (strap and anchor excluded) was lost in any picture of the window
3. the program flagged no CHECK DOTS or CHECK LABELS in the window
4. the force in newtons rises at least {rise:.0f} N above its rest level (5th percentile of the trial) inside the window

## Things to know before the numbers are used
- The angles have been tested in several rounds but are not directly validated against a goniometer or inclinometer. Low spread inside a hold (hold_angle_sd_deg) means the
  tracking is steady, not that the angle is right. Compare `hold_angle_deg` with `target_deg` and with the other trials of the position.
- Ankle tests have no main angle in data.csv (blank angle columns). The hip tests read far from their targets (a known pelvis-line
  definition issue in HANDOFF.md); treat hip angles with care.
- `hold_angle_deg` = mean angle while the force is at 80 % of the peak or more. `angle_at_peak_deg` = mean within 0.1 s of the peak.
  `hold_minus_rest_deg` = how far the joint moved under load.
- `tempo`: rise = 10 to 90 % of peak; hold = time at 80 % of peak or more. "close to 1-5-1": rise 0.5-1.5 s and hold >= 4 s.

## Files
measurements.csv, dots.csv, positions.csv, excluded.csv, data/<participant>/<position>/<trial>/ (meta.json, data.csv, force.csv)
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default=None, help="folder with the pilot trials (default: Pilot/ in the project folder, or in the current folder)")
    ap.add_argument("--out", default=None, help="default: <pilot folder>_collection next to the pilot folder")
    ap.add_argument("--participants", nargs="*", default=None, help="default: every participant found in the folder (P01, P02)")
    ap.add_argument("--window", nargs=2, type=float, default=[6.0, 8.0])
    ap.add_argument("--min-rise", type=float, default=60.0)
    ap.add_argument("--no-copy", action="store_true", help="do not copy meta/data/force files into out/data")
    args = ap.parse_args()
    pilot = find_pilot(args.pilot)
    if args.participants is None:
        args.participants = sorted({NAME.match(p.name).group(1) for p in pilot.rglob("*_meta.json") if NAME.match(p.name)})
    out = Path(args.out) if args.out else pilot.with_name(pilot.name + "_collection")
    marker = out / ".limbtrack_collection"
    if out.exists() and not marker.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and was not made by this script: choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    marker.write_text("made by limbtrack-pilot-collection\n", encoding="utf-8")
    if (out / "data").exists():
        shutil.rmtree(out / "data")                  # our own folder: rewritten from scratch so no stale trial is left

    ok_rows, dot_rows, excl = [], [], []
    metas = [p for p in sorted(pilot.rglob("*_meta.json")) if NAME.match(p.name) and NAME.match(p.name).group(1) in args.participants]
    for mp in metas:
        row, dots, why = analyse(mp, args)
        if why:
            excl.append({"id": row["id"], "participant": row["participant"], "position_no": row["position_no"], "test": row["test"],
                         "target_deg": row["target_deg"], "trial": row["trial"], "reason": "; ".join(why)})
            continue
        ok_rows.append(row)
        dot_rows += dots
        if not args.no_copy:
            d = out / "data" / row["participant"] / f"{row['position_no']:02d}_{row['test']}_{row['target_deg']}" / row["trial"]
            d.mkdir(parents=True, exist_ok=True)
            for suffix in ("meta.json", "data.csv", "force.csv"):
                src = mp.with_name(row["id"] + "_" + suffix)
                if src.is_file():
                    shutil.copy2(src, d / src.name)

    by = defaultdict(list)
    for x in ok_rows:
        by[(x["participant"], x["position_no"], x["test"], x["target_deg"])].append(x)
    total = defaultdict(int)
    for mp in metas:
        g = NAME.match(mp.name).groups()
        total[(g[0], int(g[1]))] += 1
    pos_rows = []
    for (who, no, test, tgt), xs in sorted(by.items()):
        h = [num(x["hold_angle_deg"]) for x in xs if x["hold_angle_deg"] != ""]
        d = [num(x["dev_from_target_deg"]) for x in xs if x["dev_from_target_deg"] != ""]
        pos_rows.append({"participant": who, "position_no": no, "test": test, "target_deg": tgt, "trials_recorded": total[(who, no)],
                         "trials_correct": len(xs), "correct_trials": " ".join(x["trial"] for x in xs),
                         "mean_peak_n": r(mean([num(x["peak_n"]) for x in xs]), 0), "mean_hold_angle_deg": r(mean(h), 1),
                         "hold_angle_spread_deg": r(max(h) - min(h), 1) if len(h) > 1 else "", "mean_dev_from_target_deg": r(mean(d), 1)})
    write_csv(out / "measurements.csv", ok_rows)
    write_csv(out / "dots.csv", dot_rows)
    write_csv(out / "positions.csv", pos_rows)
    write_csv(out / "excluded.csv", excl, ["id", "participant", "position_no", "test", "target_deg", "trial", "reason"])
    (out / "README.md").write_text(README.format(n_ok=len(ok_rows), n_all=len(metas), who=", ".join(args.participants), w0=args.window[0],
                                                 w1=args.window[1], rise=args.min_rise), encoding="utf-8")
    ang = [x for x in ok_rows if x["hold_angle_deg"] != ""]
    close = sum(1 for x in ok_rows if x["tempo"] == "close to 1-5-1")
    print(f"trials read: {len(metas)} | correct: {len(ok_rows)} | excluded: {len(excl)} | positions with a correct trial: {len(pos_rows)}")
    print(f"correct trials with a main angle: {len(ang)} | close to 1-5-1: {close}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
