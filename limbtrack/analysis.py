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

from .force import NO_FORCE_RISE_N
from .loadcell import GRAVITY
from .pull import compute_pull

HUM_WARN_PCT = 3.0              # 50 Hz hum above this share of the force rise is flagged
FPS_WARN = 20.0                 # fewer pictures per second than this are flagged
LOST_WARN_PCT = 10.0            # a dot lost in more than this share of the pictures is flagged
MARKER_LAG_WARN_S = 1.0         # an MVIC marker this far from where the force really rose/fell is flagged
ANCHOR_WANDER_WARN_PX = 2.0     # an anchor that wanders more than this has moved (camera or anchor)
STRAP_SEEN_WARN_PCT = 80.0
DRIFT_WARN_DEG = 2.0            # the joint angle moving this much during the hold is flagged
REPEAT_WARN_PCT = 10.0          # two trials of the same position differing by more than this: a third is needed (protocol, audit note)
ANGLE_CHECK_DEG = 15.0          # 0.26.0: the main joint during the push differing this much from the earlier trials of its position is called
                                # out (P01/P02 2026-10-28/29: six trials saved under the previous position read 24-38 deg off)
RELAXED_MISMATCH_N = 20.0       # 0.26.0: Zero 2 and the force just before the push differing by more than this (or by more than
RELAXED_MISMATCH_FRACTION = 0.05  # this share of the rise) is called out (P02 belt squat 90: Zero 2 9 N, rest before the push 423 N)


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
    tracked in a picture, the distal dot of the segment (the ankle: the strap sits next to it) stands in, and that is reported. Not
    when the joint is a computed point (the hip centre of a frontal test): there the segment ends at the knee, which is not where the
    strap sits, so a picture without the strap dot gets no torque.
    Returns None, or a dict with t, torque, arm_cm, estimated (True where the stand-in was used), unit note."""
    if not tr.has_force() or not tr.px_per_cm:
        return None
    unit, y, _ = tr.force_signal()
    if unit != "N":
        return None
    spec = (tr.meta.get("pull") or {})
    seg, fulcrum = spec.get("segment_from_to"), spec.get("moment_arm_from_joint_dot")
    tracks = {**tr.tracks, **getattr(tr, "derived_tracks", {})}       # (a computed joint, the hip centre of a frontal test, is not a dot)
    if not seg or not fulcrum or "anchor" not in tracks or fulcrum not in tracks or any(d not in tracks for d in seg):
        return None            # (0.26.0: a segment dot switched off, P01's foot dots on 2026-10-28, raised KeyError and Review could not open the trial)
    stand_in = fulcrum not in getattr(tr, "derived_tracks", {})
    n = len(tr.t)
    arm = np.full(n, np.nan)
    est = np.zeros(n, bool)
    for i in range(n):
        def xy(name):
            x, yv = tracks[name][0][i], tracks[name][1][i]
            return None if (math.isnan(x) or math.isnan(yv)) else (float(x), float(yv))
        anchor, joint, a, b = xy("anchor"), xy(fulcrum), xy(seg[0]), xy(seg[1])
        strap = xy("strap") if "strap" in tr.tracks else None
        if strap is None and stand_in:
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
        if f.get("no_rise_warning"):                                   # 0.19.0 (0.26.0: also without MVIC markers, over the whole recording)
            where = "during the whole recording (no MVIC start was marked)" if f.get("rise_basis") == "whole recording" else "during the MVIC"
            add("check", f"The force hardly moved {where} (at most {f['rise_n']:.0f} N): was the strap on the load cell and the "
                         "amplifier on? The newtons of this trial are not a contraction.")
        reg = getattr(tr, "regression", None)
        if fs is not None and fs["unit"] == "N" and reg is not None and fs["peak"] / GRAVITY > reg.range_kg[1] * 1.001:     # 0.26.0
            add("check", f"The peak force ({fs['peak']:.0f} N = {fs['peak'] / GRAVITY:.0f} kg) is above the heaviest calibration load "
                         f"({reg.range_kg[1]:g} kg): the newtons there are the calibration line extended, not measured.")
        if fs is not None:
            rm = relaxed_mismatch(fs["unit"], fs["relaxed"], fs["rest_mean"], fs["peak_minus_rest"])                        # 0.26.0
            if rm:
                add("check", rm)
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


# ------------------------------------------------------------------------------------------------ after a trial (0.26.0)
def _whole(v, sign=False):
    """A whole number for a message, never '-0' (round() gives an int, which has no negative zero)."""
    n = round(v)
    return f"{n:+d}" if sign else f"{n:d}"


def relaxed_mismatch(unit, relaxed, rest, rise):
    """A sentence when Zero 2 (the relaxed level, 0.22.0) does not describe the trial: it and the force just before the push differ by more
    than RELAXED_MISMATCH_N (or RELAXED_MISMATCH_FRACTION of the rise); else ''. Newtons only. On 2026-10-29 Zero 2 was often taken before
    the participant was in the final posture: belt squat 90 9 N against 423 N before the push, ankle plantarflexion 0 -17 N against 300-356 N,
    so 'above relaxed' was off by up to 414 N."""
    relaxed, rest, rise = _num(relaxed), _num(rest), _num(rise)
    if unit != "N" or relaxed is None or rest is None:
        return ""
    limit = max(RELAXED_MISMATCH_N, RELAXED_MISMATCH_FRACTION * abs(rise) if rise is not None else 0.0)
    if abs(rest - relaxed) <= limit:
        return ""
    return (f"Zero 2 (relaxed) was {_whole(relaxed)} N, but the force just before this push was {_whole(rest)} N ({_whole(rest - relaxed, sign=True)} N): Zero 2 "
            "does not describe this trial (the posture or the strap changed after it was taken), so 'above relaxed' is off by that much. "
            "Use the peak above rest, or press X again with the participant relaxed in the final position.")


def hold_angle_note(row, tolerance):
    """What the main joint did in a trial (an export row from session_row): its angle during the push against the target, the angle at rest
    before the push and the shift under load, and, when the shift is larger than the tolerance, the angle to set at rest so that the push
    lands on the target (target minus the shift). '' without a push window or a main angle. Nothing is changed: these are the numbers
    Review shows. On 2026-10-28/29 only 36 of 116 trials were within 5 deg of the target during the push; the knee moved a median 19 deg
    (extension) and 18 deg (flexion) from rest under load, the ankle 20 deg (dorsiflexion); the live tab judges the angle at rest."""
    main, p = row.get("main_angle"), _num(row.get("plateau_angle_deg"))
    if not main or p is None or row.get("window") == "whole trial":
        return ""
    shift, tgt = _num(row.get("angle_rest_to_hold_deg")), _num(row.get("target_deg"))
    text = f"{main} during the push: {_whole(p)} deg"
    if tgt is not None:
        dev = p - tgt
        text += (f" (target {tgt:g} +-{tolerance:g}: "
                 + ("within" if abs(dev) <= tolerance else f"{_whole(abs(dev))} deg {'above' if dev > 0 else 'below'}") + ")")
    text += "."
    if shift is not None:
        text += f" At rest before the push it was {_whole(p - shift)} deg, so it moved {_whole(shift, sign=True)} deg under load."
        if tgt is not None and abs(shift) > tolerance:
            text += f" To be at {tgt:g} deg during the push, set about {_whole(tgt - shift)} deg at rest (the target minus this shift)."
    return text


def position_angle_note(main, plateau, earlier, target, tolerance, previous=None):
    """A sentence when a trial's main joint during the push does not fit its position; else ''. `earlier`: the plateau angles of the earlier
    trials of this position; `previous`: for the first trial of a position, (target, [plateau angles]) of the previous position of the same
    test, or None. Called out: more than ANGLE_CHECK_DEG from the median of the earlier trials, or (first trial) within the tolerance of the
    previous position while the two targets are more than ANGLE_CHECK_DEG apart. Not judged against the target itself (the hip reads
    20-60 deg below it in every sitting position, so that would always fire). On 2026-10-28/29 six trials were saved under the previous
    position this way (the position was changed in the program only afterwards); a dot that jumped or a changed set-up also shows here."""
    plateau = _num(plateau)
    if not main or plateau is None:
        return ""
    good = [v for v in (_num(x) for x in earlier) if v is not None]
    if good:
        med = float(np.median(good))
        if abs(plateau - med) <= ANGLE_CHECK_DEG:
            return ""
        span = f"{_whole(min(good))}" if len(good) == 1 else f"{_whole(min(good))} to {_whole(max(good))}"
        return (f"{main} during the push of this trial: {_whole(plateau)} deg; the earlier trials of this position: {span} deg ({_whole(plateau - med, sign=True)} deg). "
                "Is the right position chosen in the program, and is the participant in it? (A dot that jumped, or a changed set-up, also "
                "gives this: look at the video.)")
    target = _num(target)
    if previous is None or target is None:
        return ""
    ptgt, pvals = _num(previous[0]), [v for v in (_num(x) for x in previous[1]) if v is not None]
    if ptgt is None or not pvals or abs(ptgt - target) <= ANGLE_CHECK_DEG:
        return ""
    med = float(np.median(pvals))
    if abs(plateau - med) > tolerance:
        return ""
    return (f"{main} during the push: {_whole(plateau)} deg, the same as at the previous position of this test (target {ptgt:g}: {_whole(med)} deg), "
            f"although the targets are {abs(ptgt - target):g} deg apart. Was the participant moved to the new position before Record, and is "
            "the right position chosen in the program?")


# ------------------------------------------------------------------------------------------------ one row per trial, per participant
SESSION_COLUMNS = ["trial", "position_number", "trial_number", "participant", "test_key", "target_deg", "software_version", "window",
                   "window_start_s", "window_end_s",
                   "pictures_per_s", "force_unit", "force_peak", "force_t_peak_s", "force_plateau", "force_rest", "force_rest_s",
                   "force_peak_above_rest", "force_relaxed", "force_peak_above_relaxed", "force_onset_s",
                   "force_time_to_90pct_s", "force_hold_cv_pct", "hum_pct_of_peak", "force_samples_at_limit", "main_angle", "angle_at_peak_force_deg", "plateau_angle_deg",
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
                   angle_at_peak_force_deg=fs["main_angle_at_peak_deg"], hum_pct_of_peak=fs["mains_hum_pct_of_peak"],
                   force_samples_at_limit=(m.get("force") or {}).get("samples_at_range_limit", ""))      # (0.26.0: > 0 = the peak is cut off)
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


def _num(v):
    return None if v in ("", None) or (isinstance(v, float) and math.isnan(v)) else float(v)


def _peak(r):
    return _num(r.get("force_peak"))


def contracted(r):
    """Whether the trial of an export row holds a real contraction (0.26.0): its rise (the peak above the rest before the push, else the
    peak itself) is at least NO_FORCE_RISE_N in newtons, or above zero in volts (not zeroed: no newton limit can be applied). Trials
    without one are left out of the 10 % check: on 2026-10-29 two P02 trials with the strap not pulling (peaks -1 and -3 N) were judged
    'differ by -73.5 %, no third trial needed'."""
    peak = _peak(r)
    if peak is None:
        return False
    rise = _num(r.get("force_peak_above_rest"))
    rise = peak if rise is None else rise
    return rise >= NO_FORCE_RISE_N if r.get("force_unit") == "N" else rise > 0


def _spread_pct(peaks):
    """(largest - smallest) as a percentage of the mean (of its size: a mean below zero must not turn a spread negative)."""
    mean = float(np.mean(peaks))
    return 100.0 * (max(peaks) - min(peaks)) / abs(mean) if mean else math.nan


def _listed(nums):
    """1 -> '1', [1, 2] -> '1 and 2', [1, 2, 4] -> '1, 2 and 4'."""
    nums = [str(x) for x in nums]
    return nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]


def repeatability(rows):
    """Adds, to rows of the same test and target, how much their peak force differs and whether a third trial is needed
    (more than REPEAT_WARN_PCT apart). Uses the force peak; rows without force, or without a contraction (`contracted`), are left alone."""
    groups = {}
    for r in rows:
        if contracted(r):
            groups.setdefault((r["test_key"], r["target_deg"], r["force_unit"]), []).append(r)
    for g in groups.values():
        if len(g) < 2:
            continue
        spread = _spread_pct([float(r["force_peak"]) for r in g])
        for r in g:
            r["repeatability_pct_same_position"] = spread
            r["third_trial_needed"] = bool(spread > REPEAT_WARN_PCT) if len(g) == 2 else ""
    return rows


MAX_TRIALS_PER_POSITION = 3     # SOP 10.6: no more than three trials


def position_check(rows):
    """The SOP 10.6 check for the saved trials of ONE position (export rows from session_row, oldest first), after each trial: are the
    peaks within 10 % of their mean, or is a third trial needed? The total peak (`force_peak`) is compared, the same number and rule as
    `repeatability` and the export (the group's choice, 2026-10-26); the peak above rest is shown beside it for information. Trials
    without a contraction (`contracted`, 0.26.0) are named and not compared; a peak cut off at the input limit is called 'at least'.
    Returns {'state', 'text', 'pct', 'n'}: state 'one' (only one trial with a contraction), 'ok', 'third' (a third trial is needed),
    'deviation' (three or more trials and no two within 10 %), 'units' (newtons and volts cannot be compared), 'no_rise' (force, but no
    contraction in any trial), 'none' (no force)."""
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
    fmt = lambda v: f"{v:.{d}f} {u}"                                              # noqa: E731
    nums = [i + 1 for i, r in enumerate(with_force) if contracted(r)]             # trial numbers (counted among the trials with force)
    flat = [i + 1 for i, r in enumerate(with_force) if not contracted(r)]
    limit = f"{NO_FORCE_RISE_N:g} N" if u == "N" else "0 V"
    if not nums:
        return {"state": "no_rise", "pct": math.nan, "n": n,
                "text": f"No force rise in trial {_listed(flat)} of this position (less than {limit} above the rest), so there is nothing to "
                        "compare. Is the strap hooked to the load cell and the amplifier on?"}
    peaks = [_peak(with_force[i - 1]) for i in nums]
    cut = [i for i in nums if (_num(with_force[i - 1].get("force_samples_at_limit")) or 0) > 0]
    shown = {i: ("at least " if i in cut else "") + fmt(p) for i, p in zip(nums, peaks)}
    above = [with_force[i - 1].get("force_peak_above_rest") for i in nums]
    above_txt = ("; above rest " + ", ".join("-" if a in ("", None) or (isinstance(a, float) and math.isnan(a)) else fmt(float(a)) for a in above)
                 if any(a not in ("", None) for a in above) else "")
    listing = ", ".join(f"trial {i} {shown[i]}" for i in nums) + above_txt
    notes = ""
    if flat:
        notes += f" Not compared: trial {_listed(flat)} (no force rise: less than {limit} above the rest)."
    if cut:
        notes += (f" Trial {_listed(cut)} reached the input limit, so {'its' if len(cut) == 1 else 'their'} true peak was higher than saved: "
                  "this comparison is a lower bound.")
    if len(nums) == 1:
        return {"state": "one", "pct": math.nan, "n": n,
                "text": f"Peak force {shown[nums[0]]} (trial {nums[0]} of this position; the 10 % check follows after trial {n + 1}).{notes}"}
    spread = _spread_pct(peaks)                                                   # the same as repeatability()
    if len(nums) == 2:
        if spread > REPEAT_WARN_PCT:
            return {"state": "third", "pct": spread, "n": n,
                    "text": f"THIRD TRIAL NEEDED: the two peaks differ by {spread:.1f} % of their mean (limit {REPEAT_WARN_PCT:g} %): {listing}. "
                            f"Rest 60 s, then record trial {n + 1} (SOP 10.6).{notes}"}
        return {"state": "ok", "pct": spread, "n": n,
                "text": f"10 % check: the two peaks differ by {spread:.1f} % of their mean (limit {REPEAT_WARN_PCT:g} %): {listing}. "
                        f"No third trial needed.{notes}"}
    best = min(((100.0 * abs(a - b) / abs((a + b) / 2.0), nums[i], nums[j]) for i, a in enumerate(peaks) for j, b in enumerate(peaks)
                if j > i and (a + b) != 0), default=(math.inf, nums[0], nums[1]))
    k = len(nums)
    more = f" {n} trials are saved at this position (SOP 10.6: at most {MAX_TRIALS_PER_POSITION})." if n > MAX_TRIALS_PER_POSITION else ""
    if best[0] > REPEAT_WARN_PCT:
        return {"state": "deviation", "pct": spread, "n": n,
                "text": f"No two of the {k} peaks are within {REPEAT_WARN_PCT:g} % of their mean (closest: trials {best[1]} and {best[2]}, "
                        f"{best[0]:.1f} %): {listing}. Note it in the deviation log; no further trial (SOP 10.6).{more}{notes}"}
    return {"state": "ok", "pct": spread, "n": n,
            "text": f"10 % check: trials {best[1]} and {best[2]} differ by {best[0]:.1f} % of their mean (all {k}: {spread:.1f} %): {listing}.{more}{notes}"}


def write_session_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SESSION_COLUMNS)
        for r in rows:
            w.writerow(["" if isinstance(v, float) and math.isnan(v) else round(v, 4) if isinstance(v, float) else v
                        for v in (r[c] for c in SESSION_COLUMNS)])
    return path
