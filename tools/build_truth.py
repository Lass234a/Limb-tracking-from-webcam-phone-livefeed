"""Weak ground truth from the session log: where did the operator see the tracker go wrong?

The tracker is only as good as the facit it is measured against. `replay_compare.py` compares with the track the program saved, which
is the old tracker's own opinion. The session log (session_log.csv) holds what a person noticed:

  * `trial warning` CHECK DOTS / CHECK LABELS: the program saw two dots move apart (or swap labels) in a stated time range of a stated
    trial. These are frame ranges where a dot probably jumped.
  * `dot re-marked` and `marking all dots again` between two recordings: the operator had to put that dot back, so it was lost or
    wrong before the next trial started (the video of that stretch does not exist: clicks are ignored while recording).

This script links both to the trial files and writes `truth_candidates.csv`: one row per episode, with the frames of the trial it
concerns. Nothing here is a coordinate. The rows say where to LOOK (contact sheets, Review) and what to count; a person confirms them.
`truth_candidates.csv` names trial files and participant codes: keep it on this computer (it is in .gitignore).

Run from the project folder:
    .venv\\Scripts\\python.exe tools\\build_truth.py C:\\path\\to\\session_log.csv C:\\limbtrack\\Pilot
"""
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

STAMP = re.compile(r"(\d{8}-\d{6})")
CHECK_DOTS = re.compile(r"CHECK DOTS in (\d+) pictures of this trial \(([\d.]+) to ([\d.]+) s\): the dots (\w+) and (\w+) moved apart")
CHECK_LABELS = re.compile(r"CHECK LABELS was shown in (\d+) pictures")


def trials_by_start(folder):
    """{start time: meta path}, from the time stamp in the file name (the log's `recording started` is that second)."""
    out = {}
    for mp in sorted(Path(folder).rglob("*_meta.json")):
        m = STAMP.search(mp.name)
        if m:
            out[datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")] = mp
    return out


def frames_between(data_csv, t0, t1):
    """First and last frame number whose t_s lies in [t0, t1] (None, None if the file cannot be read)."""
    try:
        with open(data_csv, newline="", encoding="utf-8") as fh:
            ts = [(int(r["frame"]), float(r["t_s"])) for r in csv.DictReader(fh) if r.get("t_s") not in (None, "")]
    except (OSError, ValueError, KeyError):
        return None, None
    inside = [f for f, t in ts if t0 <= t <= t1]
    return (inside[0], inside[-1]) if inside else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_log")
    ap.add_argument("trials", help="folder with the trials (searched for *_meta.json)")
    ap.add_argument("--out", default="truth_candidates.csv")
    args = ap.parse_args()

    with open(args.session_log, newline="", encoding="utf-8") as fh:
        log = list(csv.DictReader(fh))
    starts = trials_by_start(args.trials)

    def trial_at(when):
        for t, mp in starts.items():
            if abs((t - when).total_seconds()) <= 2:
                return mp
        return None

    out, current, last_stop, remarked = [], None, None, {}
    for ev in log:
        when = datetime.fromisoformat(ev["time"])
        e, d = ev["event"], ev["detail"]
        if e == "dot re-marked":
            remarked[d] = remarked.get(d, 0) + 1
        elif e == "marking all dots again finished":
            remarked["(all dots)"] = remarked.get("(all dots)", 0) + 1
        elif e == "recording started":
            current = trial_at(when)
            for dot, n in remarked.items():
                if current:
                    out.append({"trial": current.name[:-len("_meta.json")], "kind": "remarked_before", "dots": dot, "n": n,
                                "frame_from": "", "frame_to": ""})
            remarked = {}
        elif e == "trial warning" and current:
            m = CHECK_DOTS.search(d)
            if m:
                n, t0, t1, a, b = int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4), m.group(5)
                f0, f1 = frames_between(current.with_name(current.name.replace("_meta.json", "_data.csv")), t0, t1)
                out.append({"trial": current.name[:-len("_meta.json")], "kind": "check_dots", "dots": f"{a}+{b}", "n": n,
                            "frame_from": f0 if f0 is not None else "", "frame_to": f1 if f1 is not None else ""})
            m = CHECK_LABELS.search(d)
            if m:
                out.append({"trial": current.name[:-len("_meta.json")], "kind": "check_labels", "dots": "", "n": int(m.group(1)),
                            "frame_from": "", "frame_to": ""})
    if not out:
        print("No episodes found: does the log belong to these trials?")
        return 1
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "kind", "dots", "n", "frame_from", "frame_to"])
        w.writeheader()
        w.writerows(out)
    kinds = {}
    for r in out:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"{len(out)} episodes in {len({r['trial'] for r in out})} trials of {len(starts)}: {kinds}")
    print(f"Wrote {args.out} (names trial files: keep private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
