"""Turn the LogFile_*.csv files of a calibration session into a points file for add_calibration.py.

    python logs_to_points.py FOLDER -o points.csv [--skip 6]

For each file the first --skip seconds (loading and settling) are dropped, then a whole number of 20 ms periods is averaged, so 50 Hz
mains hum cancels. The mass is read from the file name (LogFile_75kg_..., LogFile_2,5kg_...). Up or down is decided by the time
order: runs before the heaviest one are 'up', after it 'down'.

Names and labels are not always right, so the script flags what looks wrong and sets keep = 0 for it; CHECK THE FILE before using it:
  * a run shorter than 15 s, or a very unsteady one (SD above 20 mV): a load put on or taken off during the file
  * a run more than 1 kg off a first line through the other runs: a wrong name (a '0 kg' file that is really the 50 kg run)
Edit the file by hand where the notes are wrong (change kg or keep). Needs only numpy.
"""
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RATE = 1000.0
MIN_SECONDS = 15.0
MAX_SD_V = 0.02
OUTLIER_KG = 1.0


def read_log(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        hdr = next(r)
        iv = next((i for i, h in enumerate(hdr) if "ai" in h.lower() and "/" in h), 1)
        t, v = [], []
        for row in r:
            try:
                t.append(datetime.strptime(row[0], "%d/%m/%Y %H:%M:%S.%f").timestamp())
                v.append(float(row[iv]))
            except (ValueError, IndexError):
                continue
    return np.array(t), np.array(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("-o", "--out", default="points.csv")
    ap.add_argument("--skip", type=float, default=6.0, help="seconds dropped at the start of every run")
    a = ap.parse_args(argv)
    runs = []
    for p in sorted(Path(a.folder).glob("LogFile_*.csv")):
        m = re.match(r"LogFile_(-?\d+(?:[.,]\d+)?)kg_", p.name)
        if not m:
            continue
        t, v = read_log(p)
        if v.size < 50:
            continue
        s = v[int(a.skip * RATE):]
        s = s[: s.size // 20 * 20]
        runs.append(dict(run=p.stem[8:], kg=float(m.group(1).replace(",", ".")), t0=float(t[0]), dur=float(t[-1] - t[0]),
                         volts=float(s.mean()) if s.size else float(v.mean()), sd=float(s.std()) if s.size else float("nan"), notes=[]))
    if len(runs) < 4:
        sys.exit(f"Only {len(runs)} usable LogFile_<mass>kg_*.csv files found in {a.folder}")
    runs.sort(key=lambda r: r["t0"])
    t_peak = max(runs, key=lambda r: r["kg"])["t0"]
    for r in runs:
        r["phase"] = "up" if r["t0"] < t_peak else "peak" if r["t0"] == t_peak else "down"
        if r["dur"] < MIN_SECONDS:
            r["notes"].append(f"only {r['dur']:.0f} s long")
        if not r["sd"] <= MAX_SD_V:
            r["notes"].append(f"unsteady (SD {r['sd'] * 1000:.0f} mV)")
    # a wrongly named run bends a plain line and makes good runs look wrong, so the worst one is set aside first and the line refitted
    ok = [r for r in runs if not r["notes"]]
    while len(ok) >= 5:
        slope, icpt = np.polyfit([r["kg"] for r in ok], [r["volts"] for r in ok], 1)
        off = [(r["volts"] - icpt) / slope - r["kg"] for r in ok]
        worst = int(np.argmax(np.abs(off)))
        if abs(off[worst]) <= OUTLIER_KG:
            break
        ok[worst]["notes"].append(f"{off[worst]:+.1f} kg from the line through the other runs: is the name right?")
        del ok[worst]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "kg", "volts", "phase", "keep", "seconds", "sd_mv", "note"])
        for r in runs:
            w.writerow([r["run"], f"{r['kg']:g}", f"{r['volts']:.5f}", r["phase"], 0 if r["notes"] else 1, f"{r['dur']:.0f}",
                        f"{r['sd'] * 1000:.1f}", "; ".join(r["notes"])])
    flagged = [r for r in runs if r["notes"]]
    print(f"{len(runs)} runs written to {a.out}; {len(flagged)} flagged (keep = 0):")
    for r in flagged:
        print(f"  {r['run']}: {'; '.join(r['notes'])}")
    print(f'Check the file, then:  python add_calibration.py {a.out} --id <name> --description "..." --activate')


if __name__ == "__main__":
    main()
