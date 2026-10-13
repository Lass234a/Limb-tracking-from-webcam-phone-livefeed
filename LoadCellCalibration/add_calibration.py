"""Add a load cell calibration to the program (no code change needed).

    python add_calibration.py POINTS.csv --id 2026-10-13 --description "0-250 kg, chain and strap on the cell" [--activate]

POINTS.csv has one row per steady run, with the columns
    kg      the mass that was hanging on the cell (plates; the chain and strap that stayed on count as the zero)
    volts   the mean amplifier voltage of the run
    phase   up / down / peak (optional: loading or unloading; gives the hysteresis)
    run     a name for the run (optional)
    keep    0 to leave a row out (optional; rows with keep = 0 are ignored but stay in the file as a record)
`logs_to_points.py` makes such a file from the LogFile_*.csv files of a session; check it, then add it here.

The fit is stored with its points in limbtrack/load_cell_calibrations.json. With --activate (or when it is the first one) the program
uses it from the next start; older recordings keep the id of the regression they were measured with. Needs only numpy.
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from limbtrack import loadcell  # noqa: E402


def read_points(path):
    kg, volts, phase, runs = [], [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row.get("keep", "1")).strip().lower() in ("0", "no", "false"):
                continue
            kg.append(float(row["kg"]))
            volts.append(float(row["volts"]))
            phase.append((row.get("phase") or "").strip())
            runs.append((row.get("run") or "").strip())
    return kg, volts, phase, runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("points", help="CSV with the columns kg, volts [, phase, run, keep]")
    ap.add_argument("--id", required=True, help="name of this calibration, e.g. its date; the same id replaces an older fit")
    ap.add_argument("--description", default="")
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--activate", action="store_true", help="the program uses this calibration from now on")
    ap.add_argument("--registry", default=None, help="registry file (default: limbtrack/load_cell_calibrations.json)")
    a = ap.parse_args(argv)
    kg, volts, phase, runs = read_points(a.points)
    reg = loadcell.fit_points(kg, volts, phase, id=a.id, date=a.date, description=a.description, runs=runs)
    loadcell.add_regression(reg, activate=a.activate, path=a.registry)
    active = loadcell.load_registry(a.registry)[0]
    print("\n".join(reg.summary_lines()))
    print(f"\nStored as '{reg.id}'. Active calibration: '{active}'" + ("" if active == reg.id else "  (run again with --activate to use the new one)"))
    return reg


if __name__ == "__main__":
    main()
