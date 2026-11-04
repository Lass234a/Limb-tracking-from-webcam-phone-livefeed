#!/usr/bin/env python3
"""After-session check of the saved numbers: are they what they claim to be? (audit of 2026-11-04)

An independent second opinion. It reads only the files a recording leaves (`*_meta.json`, `*_data.csv`, `*_force.csv`) and
works every number out again with its own short code; it imports nothing from the program's analysis, so a mistake in the
program cannot also be in the check. It never writes and never changes a file.

usage:  python tools/check_session.py <folder with the trials, or a folder of participant folders> [--cal-max-n NEWTONS]

  FAIL   the numbers disagree with each other: do not use this trial until it is understood
         (picture rows lost, time going backwards, newtons not equal to (volts - zero) x slope, force and pictures not on one clock,
         an angle that is not what its dots give).
  WARN   the numbers are consistent but may not mean what they seem to
         (no zero, input limit reached, load beyond the calibration, a dot lost / moved apart / mislabelled at the force peak,
         a resting force far from zero, the angle far from the position's target, a dot that jumped, an inconsistent ruler scale,
         an iliac crest dot that is not a stable landmark).
The exit code is 1 when there is a FAIL, else 0. Standard library and numpy only.
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

GRAVITY = 9.80665
FAIL, WARN = "FAIL", "WARN"
ANGLE_TOL_DEG = 0.1            # the CSV keeps dot positions to 0.01 px, which moves an angle on a 15 px arm by up to 0.04 deg
NEWTON_TOL = 0.01              # newtons are written to 0.001 N
CLOCK_TOL_S = 0.001            # force and pictures share one clock: samples are stamped to 0.1 ms
RATE_TOL = 0.02                # the force sample rate may differ from the one in meta.json by this fraction
PICTURE_GAP_S = 0.2
MIN_FPS = 25.0
FORCE_GAP_S = 0.02
REST_LOW_N, REST_HIGH_N = -20.0, 50.0       # the resting level in the recording (5th percentile of the 100 ms force)
NO_RISE_N = 60.0                            # a push moves the force at least this far above rest
PEAK_ZONE_S = 0.1                           # the pictures used for "the angle at the peak force"
WEAK_QUALITY = 0.6
POSITION_TOL_DEG = 10.0
JUMP_DEG, JUMPS_WARN = 10.0, 3              # a picture-to-picture step this big, this many times: a dot jumped
SCALE_SPREAD = 0.08                         # one person's thigh should measure the same in every trial, within this relative SD
PELVIS_RANGE = 1.3                          # crest-to-trochanter distance (cm) may not range more than this factor between trials
FIXED = {"up": (0.0, -1.0), "down": (0.0, 1.0), "forward": (1.0, 0.0)}


# ------------------------------------------------------------------------------------------------ reading
def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def col(rows, name):
    """A column as floats; a blank or missing cell is NaN."""
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = r.get(name, "")
        if v not in ("", None):
            try:
                out[i] = float(v)
            except ValueError:
                pass
    return out


def read_force(path):
    """(t_s, clock_s, volts, newtons) as float arrays, or None when the file has another layout."""
    with open(path, newline="", encoding="utf-8") as fh:
        head = fh.readline().strip().split(",")
    if head[:4] != ["t_s", "clock_s", "volts", "newtons"]:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                         # (an empty file is reported below, not as a numpy warning)
        a = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(0, 1, 2, 3))
    a = np.atleast_2d(a)
    if a.size == 0 or a.shape[1] < 4:
        return (np.array([]),) * 4
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def median(a):
    """The median of the finite values, NaN when there are none."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def moving_mean(y, n):
    """Centred moving average over n samples (the ends repeat the end value)."""
    n = max(int(n), 1)
    left = n // 2
    p = np.pad(y, (left, n - 1 - left), mode="edge")
    c = np.cumsum(np.insert(p, 0, 0.0))
    return (c[n:] - c[:-n]) / n


# ------------------------------------------------------------------------------------------------ angles (own implementation)
def arm(spec, rows, flip):
    """The arm of an angle as one vector per picture: 'up' / 'down' / 'forward', [dotA, dotB], or {'perp': [dotA, dotB]}."""
    if isinstance(spec, str):
        return np.tile(np.array(FIXED[spec], float), (len(rows), 1))
    a, b = spec["perp"] if isinstance(spec, dict) else spec
    v = np.column_stack([(col(rows, f"{b}_x") - col(rows, f"{a}_x")) * flip, col(rows, f"{b}_y") - col(rows, f"{a}_y")])
    if isinstance(spec, dict):
        p = np.column_stack([-v[:, 1], v[:, 0]])
        v = np.where((p[:, 1] >= 0)[:, None], p, -p)
    return v


def angle_of(defn, rows, facing):
    flip = 1.0 if facing == "right" else -1.0
    v1, v2 = arm(defn["vectors"][0], rows, flip), arm(defn["vectors"][1], rows, flip)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    raw = np.degrees(np.arctan2(cross if defn.get("signed") else np.abs(cross), (v1 * v2).sum(1)))
    return defn.get("sign", 1.0) * raw + defn.get("offset", 0.0)


def dots_of(defn):
    out = []
    for s in defn["vectors"]:
        if not isinstance(s, str):
            out += list(s["perp"] if isinstance(s, dict) else s)
    return out


# ------------------------------------------------------------------------------------------------ one trial
def calibrated_limit_n(path=None):
    """The heaviest load the load cell was calibrated to, in newtons (the active regression of the program's registry), or None."""
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "limbtrack" / "load_cell_calibrations.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        reg = next(r for r in d["calibrations"] if r["id"] == d["active"])
        return float(reg["range_kg"][1]) * GRAVITY
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return None


def _check_pictures(meta, rows, t):
    """Issues about the pictures: complete, in time order, a steady rate."""
    out = []
    if meta.get("status") != "complete":
        out.append((WARN, f"the recording is marked '{meta.get('status')}', not 'complete' (the program stopped before saving)"))
    if len(rows) != meta.get("frames"):
        out.append((FAIL, f"data.csv has {len(rows)} pictures but meta.json says {meta.get('frames')}"))
    dt = np.diff(t)
    if len(dt) and not (dt > 0).all():
        out.append((FAIL, f"picture times do not always go forward ({int((dt <= 0).sum())} steps)"))
    elif len(dt):
        if 1.0 / np.median(dt) < MIN_FPS:
            out.append((WARN, f"only {1.0 / np.median(dt):.1f} pictures per second (fast movement is sampled coarsely)"))
        if dt.max() > PICTURE_GAP_S:
            out.append((WARN, f"a gap of {dt.max():.2f} s without pictures"))
    return out


def _check_force_time(fm, rows, t, ft, fclock):
    """Issues about the force samples' time: forward, at the recorded rate, without gaps, on the pictures' clock. Returns (issues, rate)."""
    out = []
    fdt = np.diff(ft)
    rate = 1.0 / np.median(fdt) if len(fdt) else float("nan")
    if len(fdt) and not (fdt > 0).all():
        out.append((FAIL, "force sample times do not always go forward"))
    if fm.get("sample_rate_hz") and abs(rate / fm["sample_rate_hz"] - 1.0) > RATE_TOL:
        out.append((FAIL, f"the force was sampled at {rate:.1f} Hz but meta.json says {fm['sample_rate_hz']:g} Hz: every force time is off"))
    if len(fdt) and fdt.max() > FORCE_GAP_S:
        out.append((WARN, f"a gap of {fdt.max() * 1000:.0f} ms in the force samples"))
    if len(rows) and np.isfinite(t).any():
        err = np.abs((fclock - col(rows, "clock_s")[0]) - ft).max()
        if err > CLOCK_TOL_S:
            out.append((FAIL, f"force and pictures are not on one clock (off by up to {err * 1000:.1f} ms): the angle at the force peak is not trustworthy"))
    return out, rate


def _check_force_numbers(fm, ft, volts, newtons, rate, cal_max_n):
    """Issues about the force values: newtons from volts, the input limit, the calibration range, the push and the rest.
    Returns (issues, facts, peak_t); peak_t is the time of the (100 ms smoothed) force peak or None."""
    out, facts, peak_t = [], {}, None
    cal = fm.get("calibration") or {}
    n = np.full(len(volts), np.nan)
    if cal.get("zero_set") and cal.get("n_per_v"):
        expect = (volts - cal["offset_v"]) * cal["n_per_v"]
        ok = np.isfinite(newtons)
        if ok.any() and np.abs(expect[ok] - newtons[ok]).max() > NEWTON_TOL:
            out.append((FAIL, f"newtons are not (volts - zero) x slope (up to {np.abs(expect[ok] - newtons[ok]).max():.3f} N apart): the force may be wrong"))
        n = expect
    else:
        out.append((WARN, "the force was not zeroed (volts only): there are no newtons for this trial"))
    clipped = int((np.abs(volts) >= fm.get("range_v", 10.0) * 0.99).sum())
    if clipped:
        out.append((WARN, f"{clipped} force samples at the input limit: the real peak may be higher than the number says"))
    if np.isfinite(n).any():
        s = moving_mean(n, round(0.1 * rate))
        k = int(np.nanargmax(s))
        peak_t, peak = float(ft[k]), float(s[k])
        rest = float(np.percentile(s, 5))
        facts.update(peak_n=peak, rest_n=rest)
        if cal_max_n and peak > cal_max_n:
            out.append((WARN, f"the peak ({peak:.0f} N) is above the heaviest calibration load ({cal_max_n:.0f} N): the line is extended there, not measured"))
        if peak - rest < NO_RISE_N:
            out.append((WARN, f"the force never rose (peak {peak:.0f} N, rest {rest:.0f} N): no push was recorded"))
        elif rest < REST_LOW_N or rest > REST_HIGH_N:
            out.append((WARN, f"the resting force in the recording is {rest:+.0f} N: the zero was taken under load or has drifted, or the strap was loaded"))
    return out, facts, peak_t


def _check_force(base, meta, rows, t, cal_max_n):
    """Everything about force.csv. Returns (issues, facts, peak_t)."""
    path = Path(base + "_force.csv")
    if not path.is_file():
        return [], {}, None
    got = read_force(path)
    if got is None:
        return [(WARN, "force.csv has an unexpected layout: the force was not checked")], {}, None
    ft, fclock, volts, newtons = got
    if len(ft) < 2:
        return [(WARN, f"force.csv holds {len(ft)} sample(s): the force was not checked (the force was connected but nothing was recorded)")], {}, None
    fm = meta.get("force") or {}
    timing, rate = _check_force_time(fm, rows, t, ft, fclock)
    numbers, facts, peak_t = _check_force_numbers(fm, ft, volts, newtons, rate, cal_max_n)
    return timing + numbers, facts, peak_t


def _check_angles(meta, rows):
    """Every angle recomputed from its dots, and whether it jumps. Returns (issues, main) with main = (angle definition, saved values) of the main joint."""
    out, main = [], None
    facing = meta.get("facing", "right")
    for a in meta.get("angles") or []:
        cname = f"{a['name']}_deg"
        if not rows or cname not in rows[0] or any(f"{d}_x" not in rows[0] for d in dots_of(a)):
            continue
        mine, saved = angle_of(a, rows, facing), col(rows, cname)
        ok = np.isfinite(mine) & np.isfinite(saved)
        label = a.get("label", a["name"])
        if ok.any() and np.abs(mine[ok] - saved[ok]).max() > ANGLE_TOL_DEG:
            out.append((FAIL, f"{label}: the saved angle is not what its dots give (up to {np.abs(mine[ok] - saved[ok]).max():.2f} deg apart)"))
        step = np.abs(np.diff(saved[np.isfinite(saved)]))
        if (step > JUMP_DEG).sum() >= JUMPS_WARN:
            out.append((WARN, f"{label} jumped by more than {JUMP_DEG:g} deg {int((step > JUMP_DEG).sum())} times: a dot probably jumped to something else"))
        if a.get("primary"):
            main = (a, saved)
    return out, main


def _check_dots_at_peak(rows, near):
    """The dots among the pictures used for the angle at the force peak: lost, weak, moved apart or mislabelled."""
    bad = []
    for d in sorted({c[:-5] for c in rows[0] if c.endswith("_lost")}):
        lost, q = col(rows, f"{d}_lost")[near], col(rows, f"{d}_q")[near]
        if (lost == 1).any():
            bad.append(f"{d} lost")
        elif np.isfinite(q).any() and np.nanmin(q) < WEAK_QUALITY:
            bad.append(f"{d} weak")
    for flag, words in (("dots_apart", "dots moved apart"), ("label_check", "labels may be swapped")):
        if any(r.get(flag) for r, m in zip(rows, near) if m):
            bad.append(words)
    return [(WARN, "at the force peak: " + ", ".join(bad) + " (the angle at the peak rests on pictures the program itself doubted)")] if bad else []


def _check_position(meta, main, near):
    """The main angle at the peak against the position's target. Returns (issues, facts)."""
    if main is None or not isinstance(meta.get("target_deg"), (int, float)):
        return [], {}
    v = main[1][near]
    if not np.isfinite(v).any():
        return [], {}
    dev = float(np.nanmean(v)) - meta["target_deg"]
    out = []
    if abs(dev) > POSITION_TOL_DEG:
        out.append((WARN, f"{main[0].get('label', main[0]['name'])} at the peak is {np.nanmean(v):.1f} deg, {dev:+.1f} from the target {meta['target_deg']:g}: "
                          "was this the right position, or does the dot placement read differently from the joint?"))
    return out, {"main_dev_deg": dev}


def _ruler_facts(rows):
    """Lengths in cm from the ruler scale (for the checks across trials): the thigh and the crest-to-trochanter arm."""
    sc = median(col(rows, "scale_px_per_cm")) if rows and "scale_px_per_cm" in rows[0] else float("nan")
    facts = {}
    if np.isfinite(sc) and sc > 0 and rows:
        for key, a, b in (("thigh_cm", "trochanter", "epicondyle"), ("pelvis_cm", "iliac_crest", "trochanter")):
            if f"{a}_x" in rows[0] and f"{b}_x" in rows[0]:
                facts[key] = median(np.hypot(col(rows, f"{b}_x") - col(rows, f"{a}_x"), col(rows, f"{b}_y") - col(rows, f"{a}_y"))) / sc
    return facts


def check_trial(meta_path, cal_max_n=None):
    """One trial: {"issues": [(level, text)], "facts": {...}} (the facts feed check_participant)."""
    meta_path = Path(meta_path)
    base = str(meta_path)[: -len("_meta.json")]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = read_rows(base + "_data.csv")
    except (OSError, ValueError) as e:
        return {"issues": [(FAIL, f"cannot be read: {e}")], "facts": {}}
    t = col(rows, "t_s")
    issues = _check_pictures(meta, rows, t)
    force_issues, facts, peak_t = _check_force(base, meta, rows, t, cal_max_n)
    angle_issues, main = _check_angles(meta, rows)
    issues += force_issues + angle_issues
    if peak_t is not None and rows:
        near = np.abs(t - peak_t) <= PEAK_ZONE_S
        issues += _check_dots_at_peak(rows, near)
        position_issues, position_facts = _check_position(meta, main, near)
        issues += position_issues
        facts.update(position_facts)
    facts.update(_ruler_facts(rows))
    return {"issues": issues, "facts": facts}


def check_participant(facts_list):
    """Warnings across the trials of ONE participant: [(level, text)]. `facts_list`: the 'facts' dicts of check_trial."""
    out = []
    th = np.array([f["thigh_cm"] for f in facts_list if "thigh_cm" in f])
    if len(th) >= 3 and th.std() / th.mean() > SCALE_SPREAD:
        out.append((WARN, f"the ruler scale is not consistent: the same thigh measures {th.min():.0f} to {th.max():.0f} cm in different trials "
                          f"(SD {100 * th.std() / th.mean():.0f} %). Lever arms and torque carry that error. Is the ruler in the leg's plane, and was it marked each time?"))
    pv = np.array([f["pelvis_cm"] for f in facts_list if "pelvis_cm" in f])
    if len(pv) >= 3 and pv.max() / pv.min() > PELVIS_RANGE:
        out.append((WARN, f"the iliac crest dot is not a stable landmark: its distance to the trochanter ranges from {pv.min():.1f} to {pv.max():.1f} cm between trials, "
                          "so the hip angle has a different offset in each set-up"))
    return out


# ------------------------------------------------------------------------------------------------ the whole session
def find_trials(root):
    root = Path(root)
    return sorted(root.rglob("*_meta.json"), key=lambda p: (p.parent.name, p.name)) if root.is_dir() else []


def run(root, cal_max_n=None, out=print):
    """Check every trial under `root`; print FAIL and WARN lines; return (n_trials, n_fail, n_warn)."""
    root = Path(root)
    metas = [m for m in find_trials(root) if "review" not in {part.lower() for part in m.relative_to(root).parts}]         # (only folders below the one given)
    nf = nw = 0
    by_person = {}
    for m in metas:
        res = check_trial(m, cal_max_n)
        for level, text in res["issues"]:
            out(f"{level}  {m.name[:-len('_meta.json')]}: {text}")
            nf += level == FAIL
            nw += level == WARN
        by_person.setdefault(m.parent.name, []).append(res["facts"])
    for person, facts in by_person.items():
        for level, text in check_participant(facts):
            out(f"{level}  {person} (all trials): {text}")
            nf += level == FAIL
            nw += level == WARN
    out(f"\n{len(metas)} trials checked: {nf} FAIL, {nw} WARN. " + ("A FAIL means the numbers disagree with each other: do not use that trial before it is understood. "
        if nf else "No FAIL: the numbers agree with each other. ") + "Nothing was changed.")
    return len(metas), nf, nw


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independent check of a session's saved numbers.")
    ap.add_argument("folder", help="a folder with the trials, or a folder of participant folders")
    ap.add_argument("--cal-max-n", type=float, default=None, help="heaviest calibration load in newtons (default: from limbtrack/load_cell_calibrations.json)")
    a = ap.parse_args(argv)
    if not Path(a.folder).is_dir():
        print(f"{a.folder} is not a folder", file=sys.stderr)
        return 2
    n, nf, _ = run(a.folder, a.cal_max_n if a.cal_max_n else calibrated_limit_n())
    return 1 if nf or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
