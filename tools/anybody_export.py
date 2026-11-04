#!/usr/bin/env python3
"""Write the AnyBody inputs (anybody_inputs.csv and one participant.json per participant) for every participant folder under a root, into a
folder of your own. Reads the recorded trials only; nothing under the root is changed.

usage (from the project folder, with the project's Python):
    .venv/Scripts/python.exe tools/anybody_export.py --root Pilot --out SOME_FOLDER [--source original|reviewed] [--include-unjudged]
                                                    [--participants P01 P02]

Without --include-unjudged only trials marked valid are used (the rule of the plan, decision D3); a participant whose trials were never judged then
gets no rows. --include-unjudged also takes trials not judged yet (never a compromised one) and says so in the verdict column and the notes.
The same code is behind the Export menu of Review's participant overview (limbtrack/anybody.py).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from limbtrack import anybody, review_nav                      # noqa: E402
from limbtrack.completeness import default_plan                # noqa: E402
from limbtrack.engine import SOFTWARE_VERSION                  # noqa: E402
from limbtrack.review import Trial, trial_row                  # noqa: E402


def run(root, out, source="original", judged_only=True, participants=None, smooth_s=0.1, window_s=0.5):
    root, out = Path(root), Path(out)
    if any(out.glob("*_meta.json")):
        raise SystemExit(f"{out} holds a trial's own files: choose another folder")
    folders = [p for name, p, _n in review_nav.participant_folders(root) if not participants or name in participants]
    if not folders and review_nav.has_trials(root):
        folders = [root]
    # participant.json goes to <out>/<participant>: that must not be a folder of recorded trials (--out equal to the root, or inside it)
    here = out.resolve()
    for f in [root, *folders]:
        if here == f.resolve() or f.resolve() in here.parents:
            raise SystemExit(f"{out} is the recordings folder {f} or inside it: choose a folder of your own")
    rows, paths, failed = [], {}, []
    for folder in folders:
        for e in review_nav.list_trials(folder):
            if not e.readable:
                failed.append(f"{e.base}: {e.error}")
                continue
            row, err = trial_row(e.meta_path, smooth_s, window_s, source)
            if row is None:
                failed.append(f"{e.base}: {err}")
                continue
            rows.append(row)
            paths[(row.get("participant"), row["trial"])] = e.meta_path

    def open_trial(meta_path):
        tr = Trial(meta_path, open_video=False, source=source)
        tr.force_smooth_s, tr.window_peak_s = float(smooth_s), float(window_s)
        return tr

    result = anybody.build(rows, lambda r: paths[(r.get("participant"), r["trial"])], open_trial, plan=default_plan(),
                           judged_only=judged_only, source=source, export_version=SOFTWARE_VERSION)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "anybody_inputs.csv", "w", newline="", encoding="utf-8") as fh:
        anybody.write_csv(fh, result["rows"])
    for who, data in result["participants"].items():
        d = out / (who or "unknown")
        d.mkdir(exist_ok=True)
        (d / "participant.json").write_text(anybody.json_text(data), encoding="utf-8")
    return result, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=anybody.SOURCES, default="original")
    ap.add_argument("--include-unjudged", action="store_true")
    ap.add_argument("--participants", nargs="*")
    a = ap.parse_args()
    result, failed = run(a.root, a.out, a.source, not a.include_unjudged, a.participants)
    print(anybody.summary_text(result))
    if failed:
        print("not readable:\n  " + "\n  ".join(failed))
    print(f"written to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
