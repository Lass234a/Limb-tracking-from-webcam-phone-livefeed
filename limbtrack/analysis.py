"""What can be read out of a recorded trial beyond the angle table: a plain check list, force timing and steadiness, torque from the
force and the moment arm, and one summary row per trial for a whole participant.

Everything here only READS a loaded trial (limbtrack/review.py `Trial`); nothing is written into the recording. The numbers use the
smoothed force (a multiple of 20 ms removes 50 Hz mains hum) and say which assumptions they rest on.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

from .pull import compute_pull

HUM_WARN_PCT = 3.0              # 50 Hz hum above this share of the force rise is flagged
FPS_WARN = 20.0                 # fewer pictures per second than this are flagged
LOST_WARN_PCT = 10.0            # a dot lost in more than this share of the pictures is flagged
MARKER_LAG_WARN_S = 1.0         # an MVIC marker this far from where the force really rose/fell is flagged
ANCHOR_WANDER_WARN_PX = 2.0     # an anchor that wanders more than this has moved (camera or anchor)
STRAP_SEEN_WARN_PCT = 80.0
DRIFT_WARN_DEG = 2.0            # the joint angle moving this much during the hold is flagged
REPEAT_WARN_PCT = 10.0          # two trials of the same position differing by more than this: a third is needed (protocol, audit note)


# ------------------------------------------------------------------------------------------------ force timing
def force_timing(tr, t0, t1):
    """Onset, rise time, rate of force development and steadiness of the hold in the window (from the smoothed force).
    Values are in the force unit (N, or V when not calibrated). None without force or without a clear contraction."""
    if not tr.has_force():
        return None
    unit, y, _ = tr.force_signal()
    t = tr.force_t
    m = (t >= t0) & (t <= t1)
    if not m.any() or np.all(np.isnan(y[m])):
        return None
    k_local = int(np.nanargmax(y[m]))
    k = int(np.nonzero(m)[0][0]) + k_local
    # the rise is measured from the level just before the push (Trial.force_rest, 0.20.0); until 0.19.0 from the trial's 10th percentile,
    # which lies below a strap that is already loaded before the push and put the onset at the first sample (P11 12:41)
    ri = tr.force_rest()
    before = ri["level_before"] if ri is not None and not math.isnan(ri["level_before"]) else math.nan
    base = before if not math.isnan(before) else float(np.nanpercentile(y, 10))
    peak = float(y[k])
    rise = peak - base
    if rise < (0.01 if unit == "V" else 1.0):
        return None
    # onset: walk back from the peak to where the force is last within 10 % of the rise; t90: first time at 90 % after that
    lo = k
    while lo > 0 and y[lo - 1] >= base + 0.10 * rise:
        lo -= 1
    hi = lo
    while hi < k and y[hi] < base + 0.90 * rise:
        hi += 1
    onset, t90 = float(t[lo]), float(t[hi])
    # steepest rise over 100 ms between the onset and the peak
    rate = float(t[1] - t[0]) if len(t) > 1 else 0.001
    step = max(1, int(round(0.1 / rate)))
    seg = y[lo:k + 1]
    rfd = float(np.nanmax((seg[step:] - seg[:-step]) / (step * rate))) if seg.size > step else math.nan
    # steadiness of the hold: the middle 60 % of the stretch above 90 % of the peak
    above = y >= base + 0.90 * rise
    a = k
    while a > 0 and above[a - 1]:
        a -= 1
    b = k
    while b < len(y) - 1 and above[b + 1]:
        b += 1
    span = b - a
    mid = y[a + int(0.2 * span): b - int(0.2 * span) + 1] if span >= 10 else y[a:b + 1]
    plateau90 = float(np.nanmean(mid)) if mid.size else math.nan
    cv = 100.0 * float(np.nanstd(mid)) / abs(plateau90) if mid.size > 3 and plateau90 else math.nan
    return {"unit": unit, "peak": peak, "rest": base, "onset_s": onset, "time_to_90pct_s": t90 - onset,
            "rise_rate_per_s": rfd, "hold_above_90pct_s": float(t[b] - t[a]), "hold_mean": plateau90, "hold_cv_pct": cv,
            "offset_s": float(t[b])}


# ------------------------------------------------------------------------------------------------ torque
def torque_series(tr, picture_delay=0.0):
    """Joint torque (N.m) for every picture: smoothed force (N) x moment arm (m). Needs a calibrated force, a cm scale and the anchor.
    The moment arm is the distance from the joint to the line from the strap dot to the anchor dot; when the strap dot is not
    tracked in a picture, the distal dot of the segment (the ankle: the strap sits next to it) stands in, and that is reported.
    Returns None, or a dict with t, torque, arm_cm, estimated (True where the stand-in was used), unit note."""
    if not tr.has_force() or not tr.px_per_cm:
        return None
    unit, y, _ = tr.force_signal()
    if unit != "N":
        return None
    spec = (tr.meta.get("pull") or {})
    seg, fulcrum = spec.get("segment_from_to"), spec.get("moment_arm_from_joint_dot")
    if not seg or not fulcrum or "anchor" not in tr.tracks or fulcrum not in tr.tracks:
        return None
    n = len(tr.t)
    arm = np.full(n, np.nan)
    est = np.zeros(n, bool)
    for i in range(n):
        def xy(name):
            x, yv = tr.tracks[name][0][i], tr.tracks[name][1][i]
            return None if (math.isnan(x) or math.isnan(yv)) else (float(x), float(yv))
        anchor, joint, a, b = xy("anchor"), xy(fulcrum), xy(seg[0]), xy(seg[1])
        strap = xy("strap") if "strap" in tr.tracks else None
        if strap is None:
            strap, est[i] = b, True                    # the stand-in
        if None in (anchor, joint, a, b, strap):
            continue
        p = compute_pull(strap, anchor, a, b, joint, tr.px_per_cm)
        if p is not None and p["moment_arm_cm"] is not None:
            arm[i] = p["moment_arm_cm"]
    f_at = np.interp(tr.t - picture_delay, tr.force_t, y)          # a picture stamped t shows the scene at t - delay
    return {"t": tr.t, "torque": f_at * arm / 100.0, "arm_cm": arm, "estimated": est, "force_n": f_at}


TIME_COLOUR_STOPS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]      # purple -> teal -> green -> yellow (viridis)


def time_colours(n):
    """`n` colours (r, g, b) running from purple (the first picture) to yellow (the last), for a path drawn in time order."""
    if n <= 0:
        return []
    stops = np.array(TIME_COLOUR_STOPS, float)
    pos, idx = np.linspace(0.0, len(stops) - 1.0, n), np.arange(len(stops))
    rgb = np.rint(np.column_stack([np.interp(pos, idx, stops[:, c]) for c in range(3)])).astype(int)     # (round half to even, as round())
    return [tuple(c) for c in rgb.tolist()]


def force_angle_path(tr, picture_delay=0.0):
    """The main joint's angle against the force, picture by picture, for the 'force against angle' view of Review.

    x = the main joint's angle of each picture (unsmoothed, from the CSV; blank where its dots were lost), y = the force at the moment the
    picture shows: a picture is stamped `picture_delay` seconds after it was taken, so the force at `t - picture_delay` goes with the picture
    stamped `t` (the same rule as the angle at peak force). The force is the one Review shows (smoothed with the chosen moving average,
    newtons when calibrated, else volts). Nothing is smoothed or invented for the angle. Returns None without force or without an angle;
    otherwise a dict: t, angle, force (all per picture), force_t (the moment of the force each picture is paired with, t - picture_delay:
    the time the analysis window is in), ok (both numbers there), unit, angle_name."""
    if not tr.has_force():
        return None
    main = tr.main_angle
    if main is None:
        return None
    unit, ys, _ = tr.force_signal()
    t = np.asarray(tr.t, float)
    angle = np.asarray(tr.deg[main.name], float)
    force = np.interp(t - picture_delay, tr.force_t, ys, left=np.nan, right=np.nan)           # no force is made up outside what was recorded
    return {"t": t, "angle": angle, "force": force, "force_t": t - picture_delay, "ok": np.isfinite(angle) & np.isfinite(force), "unit": unit,
            "angle_name": main.title}


def torque_summary(tr, t0, t1, picture_delay=0.0):
    """Peak torque, torque at the peak force and the plateau torque in the window; None when torque cannot be computed."""
    ts = torque_series(tr, picture_delay)
    if ts is None:
        return None
    m = (ts["t"] >= t0) & (ts["t"] <= t1) & ~np.isnan(ts["torque"])
    if m.sum() < 2:
        return None
    idx = np.nonzero(m)[0]
    k = idx[int(np.argmax(ts["torque"][idx]))]
    kf = idx[int(np.argmax(ts["force_n"][idx]))]
    span = t1 - t0
    pm = m & (ts["t"] >= t0 + 0.2 * span) & (ts["t"] <= t1 - 0.2 * span)
    return {"peak_nm": float(ts["torque"][k]), "t_peak_s": float(ts["t"][k]), "at_peak_force_nm": float(ts["torque"][kf]),
            "arm_at_peak_force_cm": float(ts["arm_cm"][kf]),
            "plateau_nm": float(np.nanmean(ts["torque"][pm])) if pm.any() else math.nan,
            "mean_arm_cm": float(np.nanmean(ts["arm_cm"][m])), "pictures_with_stand_in": int(ts["estimated"][m].sum()),
            "pictures": int(m.sum())}


# ------------------------------------------------------------------------------------------------ the check list
def trial_checks(tr, t0=None, t1=None):
    """[(level, text)] with level 'ok', 'check' or 'info': a quick go / no-go for the trial in plain English."""
    out = []
    m = tr.meta

    def add(level, text):
        out.append((level, text))

    # pictures and the computer
    fps = m.get("measured_fps")
    if fps is not None:
        add("check" if fps < FPS_WARN else "ok",
            f"Pictures: {fps:.1f} per second" + (f" (below {FPS_WARN:g}: fast movements are sampled coarsely)" if fps < FPS_WARN else ""))
    pt = m.get("pipeline_timing") or {}
    if pt and pt.get("process_share_of_interval", 0) > 0.8 and fps and fps < 26:
        add("check", f"The computer was the limit: its work took {pt['process_ms']['mean']:.0f} ms per picture while the pictures came "
                     f"{pt['frame_interval_ms']['median']:.0f} ms apart. Close other programs, or use fewer dots.")
    g = m.get("gaps") or {}
    if g.get("count"):
        add("check", f"{g['count']} gap(s) with no pictures, longest {g.get('max_s', 0):.2f} s: the angles in them are unknown.")
    # tracking
    n = len(tr.rows)
    for d in tr.dot_names:
        lost = sum(r[f"{d}_lost"] == "1" for r in tr.rows)
        pct = 100.0 * lost / max(n, 1)
        if pct > LOST_WARN_PCT:
            add("check", f"Dot '{d}' was lost in {pct:.0f} % of the pictures" + (" (the strap dot is often hidden by the buckle or a fold: put a bright sticker where it stays visible)" if d == "strap" else "."))
    if m.get("partial_dot_frames"):
        add("info", "Dots seen only partly (the centre may be a little off): " + ", ".join(f"{k} {v} pictures" for k, v in m["partial_dot_frames"].items()))
    checks = m.get("checks") or {}                                     # 0.19.0
    for pair, c in (checks.get("dots_apart_frames") or {}).items():
        add("check", f"Dots {pair.replace('-', ' and ')} moved apart in {c['frames']} pictures ({c['first_t_s']:.1f} to {c['last_t_s']:.1f} s): "
                     "a dot may have jumped to something else, so the angles using them are doubtful there. Look at the video.")
    texts = checks.get("label_checks") or {}
    for code, c in (checks.get("label_check_frames") or {}).items():
        add("check", f"CHECK LABELS in {c['frames']} pictures ({c['first_t_s']:.1f} to {c['last_t_s']:.1f} s): {texts.get(code, code)}.")
    small = (m.get("dot_size_check") or {}).get("below_minimum")
    if small:
        add("check", "Dots under the 2 cm minimum: " + ", ".join(small))
    # position during the hold
    w = (t0, t1) if t0 is not None else (tr.force_window() if tr.has_force() else None) or tr.hold_window() or (float(tr.t[0]), float(tr.t[-1]))
    main = tr.primary_angle
    if main is not None and m.get("target_deg") is not None:
        mask = (tr.t >= w[0]) & (tr.t <= w[1])
        vals = tr.deg[main.name][mask]
        vals = vals[~np.isnan(vals)]
        if vals.size >= 3:
            core = vals[int(0.2 * vals.size): max(int(0.8 * vals.size), int(0.2 * vals.size) + 1)]
            hold = float(np.mean(core))
            tol = m.get("tolerance_deg", 3.0)
            dev = hold - m["target_deg"]
            add("ok" if abs(dev) <= tol else "check", f"{main.title} during the hold: {hold:.1f} deg, {dev:+.1f} from the target {m['target_deg']:g} (tolerance {tol:g})")
            drift = float(np.mean(vals[-5:]) - np.mean(vals[:5]))
            if abs(drift) > DRIFT_WARN_DEG:
                add("check", f"The joint moved {drift:+.1f} deg between the start and the end of the hold (slack in the strap, or the position slipped).")
    # force
    if tr.has_force():
        f = m.get("force") or {}
        if not f.get("calibrated") and np.all(np.isnan(tr.force_n)):
            add("info", "Force is in volts: no calibration was applied yet (Review can apply a later one).")
        fs = tr.force_summary(*w)
        if fs is not None and fs["rest_note"]:                         # 0.20.0: SOP 9.10, 3 s at rest in the strap before each trial
            add("check", fs["rest_note"])
        if fs is not None:
            hum = fs["mains_hum_pct_of_peak"]
            if not math.isnan(hum) and hum > HUM_WARN_PCT:
                add("check", f"50 Hz mains interference is {hum:.0f} % of the force rise: check shielding and grounding (smoothing removes it from the numbers).")
        if f.get("samples_at_range_limit"):
            add("check", f"The force signal touched the edge of the input range ({f['samples_at_range_limit']} samples): the peak may be cut off.")
        if f.get("read_error"):
            add("check", "The DAQ stopped during the recording: " + str(f["read_error"]))
        if f.get("frames_without_samples_over_0_5s"):
            add("check", f"No force samples for more than 0.5 s in {f['frames_without_samples_over_0_5s']} pictures.")
        if f.get("no_rise_warning"):                                   # 0.19.0
            add("check", f"The force hardly moved during the MVIC (at most {f['rise_n']:.0f} N): was the strap on the load cell and the "
                         "amplifier on? The newtons of this trial are not a contraction.")
        fw = tr.force_window()
        hw = tr.hold_window()
        if fw and hw:
            lag = fw[0] - hw[0]
            if abs(lag) > MARKER_LAG_WARN_S:
                add("check", f"The MVIC start marker was pressed {abs(lag):.1f} s {'before' if lag > 0 else 'after'} the force rose. "
                             "Use 'Window = force above half of peak' for the hold numbers.")
            else:
                add("ok", f"MVIC marker and force onset agree within {abs(lag):.1f} s.")
        am = m.get("auto_mvic")
        if am:
            if am.get("start_marker_t_s") is not None:
                late = am.get("start_marker_after_onset_s")
                add("info", "The MVIC start and end were marked automatically from the force"
                    + (f": the start {late:.2f} s after the force began to rise, the end {am['length_s']:g} s after the start." if late is not None else "."))
            elif am.get("state") == "manual":
                add("info", "The automatic MVIC markers were on, but a marker was pressed by hand and took over.")
            else:
                add("check", f"The automatic MVIC markers were on, but the force never rose by {am.get('threshold')} {am.get('unit')}: no start was marked. "
                              "Use 'Window = force above half of peak'.")
    # strap and anchor
    if "anchor" in tr.tracks:
        ax, ay = tr.tracks["anchor"]
        if np.any(~np.isnan(ax)):
            wander = float(max(np.nanstd(ax), np.nanstd(ay)))
            add("check" if wander > ANCHOR_WANDER_WARN_PX else "ok", f"Anchor dot wandered {wander:.1f} px" + (" (the camera or the anchor moved)" if wander > ANCHOR_WANDER_WARN_PX else " (steady)"))
    if "strap" in tr.tracks:
        sx = tr.tracks["strap"][0]
        seen = 100.0 * float(np.sum(~np.isnan(sx))) / max(len(sx), 1)
        add("check" if seen < STRAP_SEEN_WARN_PCT else "ok", f"Strap dot seen in {seen:.0f} % of the pictures"
            + ("; the pull angle and moment arm are missing where it was not seen." if seen < STRAP_SEEN_WARN_PCT else "."))
    if tr.pull is not None and not tr.px_per_cm:
        add("info", "No ruler scale: the moment arm is in pixels only, so no torque can be given.")
    if not out:
        add("info", "Nothing to check.")
    return out


# ------------------------------------------------------------------------------------------------ one row per trial, per participant
SESSION_COLUMNS = ["trial", "position_number", "trial_number", "participant", "test_key", "target_deg", "software_version", "window",
                   "window_start_s", "window_end_s",
                   "pictures_per_s", "force_unit", "force_peak", "force_t_peak_s", "force_plateau", "force_rest", "force_rest_s",
                   "force_peak_above_rest", "force_relaxed", "force_peak_above_relaxed", "force_onset_s",
                   "force_time_to_90pct_s", "force_hold_cv_pct", "hum_pct_of_peak", "main_angle", "angle_at_peak_force_deg", "plateau_angle_deg",
                   "angle_rest_to_hold_deg", "torque_peak_nm", "torque_at_peak_force_nm", "arm_at_peak_force_cm", "torque_stand_in_pictures",
                   "checks_to_look_at", "repeatability_pct_same_position", "third_trial_needed",
                   "reference_main_deg", "reference_instrument", "app_main_at_reading_deg", "app_minus_reference_main_deg", "reference_app_basis"]


def session_row(tr):
    """The key numbers of one trial (window from the force when there is force, else the MVIC markers, else the whole trial)."""
    m = tr.meta
    if tr.has_force() and tr.force_window():
        win, src = tr.force_window(), "force above half of peak"
    elif tr.hold_window():
        win, src = tr.hold_window(), "MVIC markers"
    else:
        win, src = (float(tr.t[0]), float(tr.t[-1])), "whole trial"
    row = {k: "" for k in SESSION_COLUMNS}
    row.update(trial=tr.base, position_number=m.get("position_number") or "", trial_number=m.get("trial_number") or "",   # (0.20.0)
               participant=m.get("participant"), test_key=m.get("test_key"), target_deg=m.get("target_deg"),
               software_version=m.get("software_version"), window=src, window_start_s=win[0], window_end_s=win[1],
               pictures_per_s=m.get("measured_fps"))
    ang = tr.summary(*win)
    main = next((r for r in ang if r["role"] == "main"), ang[0] if ang else None)
    if main is not None:
        row.update(main_angle=main["angle"], plateau_angle_deg=main["plateau_mean_deg"], angle_rest_to_hold_deg=main["rest_to_hold_shift_deg"])
    fs = tr.force_summary(*win)
    if fs is not None:
        row.update(force_unit=fs["unit"], force_peak=fs["peak"], force_t_peak_s=fs["t_peak_s"], force_plateau=fs["plateau_mean"],
                   force_rest=fs["rest_mean"], force_rest_s=fs["rest_seconds"], force_peak_above_rest=fs["peak_minus_rest"],
                   force_relaxed=fs["relaxed"], force_peak_above_relaxed=fs["peak_minus_relaxed"],
                   angle_at_peak_force_deg=fs["main_angle_at_peak_deg"], hum_pct_of_peak=fs["mains_hum_pct_of_peak"])
        ft = force_timing(tr, *win)
        if ft is not None:
            row.update(force_onset_s=ft["onset_s"], force_time_to_90pct_s=ft["time_to_90pct_s"], force_hold_cv_pct=ft["hold_cv_pct"])
    tq = torque_summary(tr, *win)
    if tq is not None:
        row.update(torque_peak_nm=tq["peak_nm"], torque_at_peak_force_nm=tq["at_peak_force_nm"], arm_at_peak_force_cm=tq["arm_at_peak_force_cm"],
                   torque_stand_in_pictures=tq["pictures_with_stand_in"])
    row["checks_to_look_at"] = " | ".join(text for lvl, text in trial_checks(tr, *win) if lvl == "check")
    primary = tr.primary_angle                       # the typed reference of the main joint, beside the app's angle at the same moment
    ref = next((r for r in tr.reference_rows() if r["key"] == primary.name), None) if primary is not None else None
    if ref is not None:
        row.update(reference_main_deg=ref["reference_deg"], reference_instrument=ref["instrument"], app_main_at_reading_deg=ref["app_deg"],
                   app_minus_reference_main_deg=ref["difference_deg"], reference_app_basis=ref["app_basis"] or "")
    return row


def repeatability(rows):
    """Adds, to rows of the same test and target, how much their peak force differs and whether a third trial is needed
    (more than REPEAT_WARN_PCT apart). Uses the force peak; rows without force are left alone."""
    groups = {}
    for r in rows:
        if r["force_peak"] != "" and not (isinstance(r["force_peak"], float) and math.isnan(r["force_peak"])):
            groups.setdefault((r["test_key"], r["target_deg"], r["force_unit"]), []).append(r)
    for g in groups.values():
        if len(g) < 2:
            continue
        peaks = [float(r["force_peak"]) for r in g]
        mean = float(np.mean(peaks))
        spread = 100.0 * (max(peaks) - min(peaks)) / mean if mean else math.nan
        for r in g:
            r["repeatability_pct_same_position"] = spread
            r["third_trial_needed"] = bool(spread > REPEAT_WARN_PCT) if len(g) == 2 else ""
    return rows


MAX_TRIALS_PER_POSITION = 3     # SOP 10.6: no more than three trials


def _peak(r):
    v = r.get("force_peak")
    return None if v in ("", None) or (isinstance(v, float) and math.isnan(v)) else float(v)


def position_check(rows):
    """The SOP 10.6 check for the saved trials of ONE position (export rows from session_row, oldest first), after each trial: are the
    peaks within 10 % of their mean, or is a third trial needed? The total peak (`force_peak`) is compared, the same number and rule as
    `repeatability` and the export (the group's choice, 2026-10-26); the peak above rest is shown beside it for information.
    Returns {'state', 'text', 'pct', 'n'}: state 'one' (only one trial with force), 'ok', 'third' (a third trial is needed),
    'deviation' (three or more trials and no two within 10 %), 'units' (newtons and volts cannot be compared), 'none' (no force)."""
    with_force = [r for r in rows if _peak(r) is not None]
    n = len(with_force)
    if n == 0:
        return {"state": "none", "text": "", "pct": math.nan, "n": 0}
    units = sorted({r["force_unit"] for r in with_force})
    if len(units) > 1:
        volts = [i + 1 for i, r in enumerate(with_force) if r["force_unit"] == "V"]
        return {"state": "units", "pct": math.nan, "n": n,
                "text": f"The 10 % check cannot be made: trial {', '.join(map(str, volts))} of this position has volts only (not zeroed), "
                        "the other newtons. Compare them in Review after recalculating newtons."}
    u = units[0]
    d = 0 if u == "N" else 3
    peaks = [_peak(r) for r in with_force]
    fmt = lambda v: f"{v:.{d}f} {u}"                                              # noqa: E731
    above = [r.get("force_peak_above_rest") for r in with_force]
    above_txt = ("; above rest " + ", ".join("-" if a in ("", None) or (isinstance(a, float) and math.isnan(a)) else fmt(float(a)) for a in above)
                 if any(a not in ("", None) for a in above) else "")
    listing = ", ".join(f"trial {i + 1} {fmt(p)}" for i, p in enumerate(peaks)) + above_txt
    if n == 1:
        return {"state": "one", "pct": math.nan, "n": 1, "text": f"Peak force {fmt(peaks[0])} (trial 1 of this position; the 10 % check follows after trial 2)."}
    mean = float(np.mean(peaks))
    spread = 100.0 * (max(peaks) - min(peaks)) / mean if mean else math.nan       # the same as repeatability()
    if n == 2:
        if spread > REPEAT_WARN_PCT:
            return {"state": "third", "pct": spread, "n": 2,
                    "text": f"THIRD TRIAL NEEDED: the two peaks differ by {spread:.1f} % of their mean (limit {REPEAT_WARN_PCT:g} %): {listing}. "
                            "Rest 60 s, then record trial 3 (SOP 10.6)."}
        return {"state": "ok", "pct": spread, "n": 2,
                "text": f"10 % check: the two peaks differ by {spread:.1f} % of their mean (limit {REPEAT_WARN_PCT:g} %): {listing}. No third trial needed."}
    best = min((100.0 * abs(a - b) / ((a + b) / 2.0), i, j) for i, a in enumerate(peaks) for j, b in enumerate(peaks) if j > i and (a + b) > 0)
    more = f" {n} trials are saved at this position (SOP 10.6: at most {MAX_TRIALS_PER_POSITION})." if n > MAX_TRIALS_PER_POSITION else ""
    if best[0] > REPEAT_WARN_PCT:
        return {"state": "deviation", "pct": spread, "n": n,
                "text": f"No two of the {n} peaks are within {REPEAT_WARN_PCT:g} % of their mean (closest: trials {best[1] + 1} and {best[2] + 1}, "
                        f"{best[0]:.1f} %): {listing}. Note it in the deviation log; no further trial (SOP 10.6).{more}"}
    return {"state": "ok", "pct": spread, "n": n,
            "text": f"10 % check: trials {best[1] + 1} and {best[2] + 1} differ by {best[0]:.1f} % of their mean (all {n}: {spread:.1f} %): {listing}.{more}"}


def write_session_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SESSION_COLUMNS)
        for r in rows:
            w.writerow(["" if isinstance(v, float) and math.isnan(v) else round(v, 4) if isinstance(v, float) else v
                        for v in (r[c] for c in SESSION_COLUMNS)])
    return path
