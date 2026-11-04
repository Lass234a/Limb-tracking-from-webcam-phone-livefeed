#!/usr/bin/env python3
"""Run the models written by tools/anybody_models.py in AnyBody (through AnyPyTools) and collect the results next to the manifest.

usage (on a computer with AnyBody 8.2 and `pip install anypytools`):
    python tools/anybody_models_run.py AnyBody_models [--only P01:2 ...] [--processes 2] [--dry-run] [--anybodycon PATH]

For every model it loads Main.any, runs the inverse dynamics at the measured force, reads the results, then runs the AMMR MaxLoad evaluation.
Output: results.csv and results.json in the models folder. This is the FIRST thing to run on a new computer, on one position:
the columns that start with check_ say whether the model got the angles, the force direction and the moment that the manifest expects.

What the numbers mean (see the skill anybody-position-models):
  max_muscle_activity   the highest muscle activity needed to hold the measured force, in a model with the AMMR default strength. 1.0 = the
                        model is just strong enough. 1.4 = the muscles would have to be 40% stronger. It is the whole-leg number: the same factor for all
                        muscles; the groups of the study are scaled later.
  max_load_n            the force at the strap at which the activity would reach 1 (from two runs, assuming activity is linear in the load)
  force_over_max_load   measured force / max_load_n; below 1 the model is stronger than the person at this position

NOT TRIED AGAINST ANYBODY: the AnyPyTools calls follow its documentation; run with --dry-run first and read the macro.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ANGLE_TOL_DEG = 1.0                # a read-back angle further off than this is reported
DIRECTION_TOL_DEG = 2.0            # so is a force direction
MOMENT_TOL_FRAC = 0.10             # and a moment

STUDY = "Main.Study.InverseDynamics"
DUMPS = ["Main.Study.Output.MaxMuscleActivity"]
CHECK_DUMPS = ["Main.Check.KneeFlexion_deg", "Main.Check.AnklePlantarFlexion_deg", "Main.Check.PullAngle_deg", "Main.Check.PullDir",
               "Main.Check.ShankDir", "Main.Check.LeverArm_m", "Main.Check.ExternalKneeMoment_Nm", "Main.Check.ShankLength_m",
               "Main.Check.StrapToKnee_m"]
MAXLOAD_RUN = "Main.MaxLoading.Evaluate"
MAXLOAD_DUMP = "Main.MaxLoading.MaxLoad"


def macro_for(entry, root, with_checks=True, with_maxload=True):
    """The AnyBody macro (a list of lines) for one model of the manifest. The MaxLoad evaluation runs last: it runs the study again at two
    other loads, so the other results must be dumped before it."""
    main = (Path(root) / entry["main_file"]).resolve().as_posix()
    lines = [f'load "{main}"', f"operation {STUDY}", "run"]
    lines += [f'classoperation {name} "Dump"' for name in DUMPS + (CHECK_DUMPS if with_checks else [])]
    if with_maxload:
        lines += [f"operation {MAXLOAD_RUN}", "run", f'classoperation {MAXLOAD_DUMP} "Dump"']
    return lines + ["exit"]


def _first(x):
    """The first number of a number, a list or an array of arrays; None when empty."""
    while hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        if len(x) == 0:
            return None
        x = x[0]
    return None if x is None else float(x)


def _vector(x):
    """A 3-vector from a dumped AnyVec3 (it may come as [[x, y, z]] or [x, y, z])."""
    while hasattr(x, "__len__") and len(x) == 1 and hasattr(x[0], "__len__"):
        x = x[0]
    return [float(v) for v in x] if hasattr(x, "__len__") and len(x) == 3 else None


def _angle_between(a, b):
    d = sum(p * q for p, q in zip(a, b))
    n = math.sqrt(sum(p * p for p in a)) * math.sqrt(sum(q * q for q in b))
    return math.degrees(math.acos(max(-1.0, min(1.0, d / n)))) if n else None


def summarise(entry, result):
    """One row of results from the manifest entry and the dict AnyPyTools returned for its macro."""
    exp, p = entry["expected"], entry["position"]
    row = {"id": entry["id"], "participant": entry["participant"], "position_number": entry["position_number"], "test_key": entry["test_key"],
           "target_deg": entry["target_deg"], "trial_id": entry["trial_id"], "reviewed": entry["reviewed"], "force_n": p["Force_N"]}
    errors = result.get("ERROR") or []
    if errors:
        row.update(status="error: " + " | ".join(str(e) for e in errors)[:300])
        return row
    act = _first(result.get(DUMPS[0]))
    maxload = _first(result.get(MAXLOAD_DUMP))
    row.update(max_muscle_activity=act, max_load_n=maxload,
               force_over_max_load=(p["Force_N"] / maxload if maxload else None))
    problems = []

    def check(name, got, want, tol, kind):
        row[f"check_{name}"] = got
        if got is None or want is None:
            return
        d = abs(got - want) / abs(want) if kind == "frac" and want else abs(got - want)
        row[f"check_{name}_diff"] = d
        if d > tol:
            problems.append(f"{name}: model {got:.3f}, expected {want:.3f}")

    check("knee_flexion_deg", _first(result.get("Main.Check.KneeFlexion_deg")), exp.get("knee_flexion_deg"), ANGLE_TOL_DEG, "abs")
    check("ankle_plantar_flexion_deg", _first(result.get("Main.Check.AnklePlantarFlexion_deg")), exp.get("ankle_plantar_flexion_deg"),
          ANGLE_TOL_DEG, "abs")
    check("pull_angle_deg", _first(result.get("Main.Check.PullAngle_deg")), exp.get("pull_angle_deg"), ANGLE_TOL_DEG, "abs")
    pdir, want_dir = _vector(result.get("Main.Check.PullDir")), exp.get("pull_dir_global")
    if pdir and want_dir:
        row["check_pull_dir_diff_deg"] = _angle_between(pdir, want_dir)
        if row["check_pull_dir_diff_deg"] > DIRECTION_TOL_DEG:
            problems.append(f"pull direction is {row['check_pull_dir_diff_deg']:.1f} deg from the expected one")
    check("lever_arm_m", _first(result.get("Main.Check.LeverArm_m")), exp.get("lever_arm_app_m"), MOMENT_TOL_FRAC, "frac")
    check("moment_nm", _first(result.get("Main.Check.ExternalKneeMoment_Nm")), exp.get("ext_moment_app_nm"), MOMENT_TOL_FRAC, "frac")
    row["status"] = "ok" if act is not None and not problems else ("checks differ: " + "; ".join(problems) if problems else "no result")
    return row


def select(models, only):
    """--only P01:2 P02 -> the manifest entries of participant P01 position 2, and all of P02."""
    if not only:
        return list(models)
    wanted = []
    for item in only:
        who, _, n = item.partition(":")
        wanted.append((who, int(n) if n else None))
    return [m for m in models if any(m["participant"] == w and (n is None or m["position_number"] == n) for w, n in wanted)]


def run(root, only=None, processes=1, dry_run=False, anybodycon=None, process_factory=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    models = select(manifest["models"], only)
    macros = [macro_for(m, root, with_checks=True, with_maxload=True) for m in models]
    if dry_run:
        for m, mac in zip(models, macros):
            print(f"--- {m['id']}\n" + "\n".join(mac))
        return []
    if process_factory is None:
        try:
            from anypytools import AnyPyProcess
        except ImportError:
            raise SystemExit("AnyPyTools is not installed here: pip install anypytools (and AnyBody 8.2 must be installed)")
        process_factory = AnyPyProcess
    kwargs = {"num_processes": processes}
    if anybodycon:
        kwargs["anybodycon_path"] = anybodycon
    results = process_factory(**kwargs).start_macro(macros)
    rows = [summarise(m, r) for m, r in zip(models, results)]
    write_results(root, rows)
    return rows


def write_results(root, rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(Path(root) / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (Path(root) / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the folder written by tools/anybody_models.py (holds manifest.json)")
    ap.add_argument("--only", nargs="*", help="P01:2 = participant P01, position 2; P02 = all of P02")
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="print the macros and run nothing")
    ap.add_argument("--anybodycon", help="path of AnyBodyConsole.exe if AnyPyTools does not find it")
    a = ap.parse_args(argv)
    rows = run(a.root, a.only, a.processes, a.dry_run, a.anybodycon)
    for r in rows:
        print(f"{r['id']}: {r['status']}" + (f"   max activity {r['max_muscle_activity']:.3f}" if r.get("max_muscle_activity") is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
