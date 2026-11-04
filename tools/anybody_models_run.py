#!/usr/bin/env python3
"""Run the models written by tools/anybody_models.py in AnyBody (through AnyPyTools) and collect the results next to the manifest.

usage (on a computer with AnyBody 8.2 and `pip install anypytools`):
    python tools/anybody_models_run.py AnyBody_models [--only P01:2 ...] [--processes 2] [--dry-run] [--anybodycon PATH]

Every position is run twice: with the hip held ideally by the pelvic belt and thigh strap (Main.any) and with the hip muscles holding the hip
(Main_HipFree.any). For each run it sets the strap load to the measured force, runs the inverse dynamics and reads the model back, then sets the
probe load and runs the AMMR MaxLoad evaluation (two runs, extrapolated to 100 % muscle activity).
Output: results.csv and results.json (one row per position and variant) and results_bracket.csv (one row per position: the two ends side by side).

FIRST THING TO RUN on a new computer: `python tools/anybody_models.py --regression-case --out AnyBody_models`, then this tool on it. The case is the
group's own SeatedKneeExtMVIC model; its README says 2036 N (hip ideal) and 555 N (hip free). If this tool does not reproduce them to 2 %, the
port is wrong somewhere and nothing else should be trusted yet. After that, run ONE real position and read the check_ columns.

What the numbers mean (see the skill anybody-position-models):
  max_load_n            the strap force at which the model's highest muscle activity reaches 1, with the AMMR default muscle strength
  scale_needed_line     the strength factor the leg muscles need so that the model's capacity equals the measured force. The load line of the two
                        runs is m = a*F + b (b is gravity and the like), so the factor is a*F_measured + b. 1.0 = the default strength is right;
                        0.4 = the person's muscles are 40 % as strong as the default.
  scale_needed_ratio    F_measured / max_load_n: the same without the gravity term (the group's first idea). Close to the line value when b is small.
  max_muscle_activity   the activity at the measured force itself; NOT the strength factor: the trunk's baseline activity (about 13 %) hides the
                        legs when the load is low. Kept as a cross-check only.
  check_*               what the model read back, against what the manifest expects. If any differs, status says so and the numbers must not be used.

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
REGRESSION_TOL_FRAC = 0.02         # the group's own model must be reproduced to this
POSTURE_TOL_DEG = 0.5

STUDY = "Main.Study.InverseDynamics"
ACTIVITY = "Main.Study.Output.MaxMuscleActivity"
CHECK_DUMPS = ["Main.Check.KneeFlexion_deg", "Main.Check.AnklePlantarFlexion_deg", "Main.Check.PullAngle_deg", "Main.Check.PullDir",
               "Main.Check.LeverArm_m", "Main.Check.StrapToKnee_m", "Main.Check.ExternalKneeMoment_Nm", "Main.Check.ShankLength_m",
               "Main.Check.KneeFlexionLimbtrackDots_deg", "Main.Check.ThighBelowHorizontal_deg", "Main.Check.ShankForwardOfVertical_deg"]
MAXLOAD = "Main.MaxLoading.MaxLoad"
LINE_DUMPS = {"l1": "Main.MaxLoading.Calculations.Load1", "l2": "Main.MaxLoading.Calculations.Load2",
              "m1": "Main.MaxLoading.Calculations.MaxMuscleActivity1", "m2": "Main.MaxLoading.Calculations.MaxMuscleActivity2"}
UPDATE = 'classoperation Main "Update Values"'


def macro_for(main_file, root, with_checks=True):
    """The AnyBody macro (a list of lines) for one Main file. The MaxLoad evaluation runs last: it runs the study again at two other loads, so
    the other results must be dumped before it."""
    main = (Path(root) / main_file).resolve().as_posix()
    lines = [f'load "{main}"', "operation Main.Ops.UseMeasuredForce", "run", UPDATE, f"operation {STUDY}", "run",
             f'classoperation {ACTIVITY} "Dump"']
    if with_checks:
        lines += [f'classoperation {name} "Dump"' for name in CHECK_DUMPS]
    lines += ["operation Main.Ops.UseProbeLoad", "run", UPDATE, "operation Main.MaxLoading.Evaluate", "run", f'classoperation {MAXLOAD} "Dump"']
    lines += [f'classoperation {name} "Dump"' for name in LINE_DUMPS.values()]
    return lines + ["exit"]


def _first(x):
    """The first number of a number, a list or an array of arrays; None when empty or missing."""
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


def load_line(result):
    """(slope a, intercept b) of the straight line  activity = a * load + b  through the two runs of MaxLoading, or None."""
    g = {k: _first(result.get(name)) for k, name in LINE_DUMPS.items()}
    if None in g.values() or g["l2"] == g["l1"]:
        return None
    a = (g["m2"] - g["m1"]) / (g["l2"] - g["l1"])
    return a, g["m1"] - a * g["l1"]


def summarise(entry, variant, result):
    """One row of results from the manifest entry, the variant name and the dict AnyPyTools returned for its macro."""
    exp, p = entry["expected"], entry["position"]
    row = {"id": entry["id"], "variant": variant, "participant": entry["participant"], "position_number": entry["position_number"],
           "test_key": entry["test_key"], "target_deg": entry["target_deg"], "trial_id": entry["trial_id"], "reviewed": entry["reviewed"],
           "force_n": p["MeasuredForce_N"]}
    errors = result.get("ERROR") or []
    if errors:
        row.update(status="error: " + " | ".join(str(e) for e in errors)[:300])
        return row
    maxload = _first(result.get(MAXLOAD))
    line = load_line(result)
    force = p["MeasuredForce_N"]
    row.update(max_load_n=maxload, scale_needed_ratio=(force / maxload if maxload else None),
               line_slope=line[0] if line else None, line_intercept=line[1] if line else None,
               scale_needed_line=(line[0] * force + line[1] if line else None), max_muscle_activity=_first(result.get(ACTIVITY)))
    problems = []

    def check(name, got, want, tol, kind="abs", column=None):
        col = column or f"check_{name}"
        row[col] = got
        if got is None or want is None:
            return
        d = abs(got - want) / abs(want) if kind == "frac" and want else abs(got - want)
        row[f"{col}_diff"] = d
        if d > tol:
            problems.append(f"{name}: model {got:.3f}, expected {want:.3f}")

    first = lambda name: _first(result.get(f"Main.Check.{name}"))               # noqa: E731
    check("knee_flexion_deg", first("KneeFlexion_deg"), exp.get("knee_flexion_deg"), ANGLE_TOL_DEG)
    check("ankle_plantar_flexion_deg", first("AnklePlantarFlexion_deg"), exp.get("ankle_plantar_flexion_deg"), ANGLE_TOL_DEG)
    check("pull_angle_deg", first("PullAngle_deg"), exp.get("pull_angle_deg"), ANGLE_TOL_DEG)
    pdir, want_dir = _vector(result.get("Main.Check.PullDir")), exp.get("pull_dir_global")
    if pdir and want_dir:
        row["check_pull_dir_diff_deg"] = _angle_between(pdir, want_dir)
        if row["check_pull_dir_diff_deg"] > DIRECTION_TOL_DEG:
            problems.append(f"pull direction is {row['check_pull_dir_diff_deg']:.1f} deg from the expected one")
    check("lever_arm_m", first("LeverArm_m"), exp.get("lever_arm_app_m"), MOMENT_TOL_FRAC, "frac")
    check("moment_nm", first("ExternalKneeMoment_Nm"), exp.get("ext_moment_app_nm"), MOMENT_TOL_FRAC, "frac")
    row["knee_dots_model_deg"] = first("KneeFlexionLimbtrackDots_deg")
    if row["knee_dots_model_deg"] is not None and exp.get("knee_dots_deg") is not None:
        row["knee_zero_offset_deg"] = exp["knee_dots_deg"] - row["knee_dots_model_deg"]       # to add to the knee angle if it is more than a degree
    reg = (exp.get("regression") or {})
    if reg:
        want = reg.get(variant, {})
        check("regression_max_load_n", maxload, want.get("max_load_n"), REGRESSION_TOL_FRAC, "frac", "regression_max_load_n")
        check("regression_lever_arm_m", first("StrapToKnee_m"), want.get("lever_arm_m"), 0.01, "frac", "regression_lever_arm_m")
        post = reg.get("posture", {})
        check("regression_knee_dots_deg", row["knee_dots_model_deg"], post.get("knee_flexion_dots_deg"), POSTURE_TOL_DEG, "abs", "regression_knee_dots_deg")
        check("regression_thigh_deg", first("ThighBelowHorizontal_deg"), post.get("thigh_below_horizontal_deg"), POSTURE_TOL_DEG, "abs",
              "regression_thigh_deg")
        check("regression_shank_deg", first("ShankForwardOfVertical_deg"), post.get("shank_forward_of_vertical_deg"), POSTURE_TOL_DEG, "abs",
              "regression_shank_deg")
    if maxload is None:
        row["status"] = "no result"
    else:
        row["status"] = "ok" if not problems else "checks differ: " + "; ".join(problems)
    return row


def bracket(rows):
    """One row per position with the two ends of the hip assumption side by side."""
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["variant"]] = r
    out = []
    for key, v in by.items():
        i, f = v.get("hip_ideal", {}), v.get("hip_free", {})
        any_row = i or f
        out.append({"id": key, "participant": any_row.get("participant"), "position_number": any_row.get("position_number"),
                    "test_key": any_row.get("test_key"), "target_deg": any_row.get("target_deg"), "reviewed": any_row.get("reviewed"),
                    "force_n": any_row.get("force_n"),
                    "scale_needed_hip_ideal": i.get("scale_needed_line"), "scale_needed_hip_free": f.get("scale_needed_line"),
                    "max_load_n_hip_ideal": i.get("max_load_n"), "max_load_n_hip_free": f.get("max_load_n"),
                    "status_hip_ideal": i.get("status"), "status_hip_free": f.get("status")})
    return out


def select(models, only):
    """--only P01:2 P02 -> the manifest entries of participant P01 position 2, and all of P02."""
    if not only:
        return list(models)
    wanted = []
    for item in only:
        who, _, n = item.partition(":")
        wanted.append((who, int(n) if n else None))
    return [m for m in models if any(m["participant"] == w and (n is None or m["position_number"] == n) for w, n in wanted)]


def tasks_of(models):
    """[(manifest entry, variant name, main file)] in the order they are run."""
    return [(m, v["name"], v["main_file"]) for m in models for v in m["variants"]]


def run(root, only=None, processes=1, dry_run=False, anybodycon=None, process_factory=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tasks = tasks_of(select(manifest["models"], only))
    macros = [macro_for(main, root) for _m, _v, main in tasks]
    if dry_run:
        for (m, v, _main), mac in zip(tasks, macros):
            print(f"--- {m['id']} [{v}]\n" + "\n".join(mac))
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
    rows = [summarise(m, v, r) for (m, v, _main), r in zip(tasks, results)]
    write_results(root, rows)
    return rows


def _write_csv(path, rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_results(root, rows):
    _write_csv(Path(root) / "results.csv", rows)
    _write_csv(Path(root) / "results_bracket.csv", bracket(rows))
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
        extra = f"   max load {r['max_load_n']:.0f} N, strength factor {r['scale_needed_line']:.2f}" if r.get("scale_needed_line") is not None else ""
        print(f"{r['id']} [{r['variant']}]: {r['status']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
