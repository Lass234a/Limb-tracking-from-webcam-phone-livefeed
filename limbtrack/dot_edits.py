"""Fixing dot positions after the trial (0.30.0), on a loaded `review.Trial`, from the edit list of `review_layer.Review`.

Four kinds of edit, all per dot:
  hold      a stationary dot (the anchor, a ruler dot) stays at one position over a time range
  lost      the dot is called lost over a time range (it is hidden, or the tracker is somewhere wrong and no position is better than a wrong one)
  manual    positions clicked picture by picture (linear interpolation between clicks)
  retrack   the tracker follows the dot from a picture where you clicked it (positions are stored, so the numbers never depend on re-running it)

`apply` changes ONLY the pictures that were edited: their dot columns, and what is worked out from those dots, with the same functions
and rules the recording used: the angles (`angles.measure`), deviation and in-tolerance flag against the reference saved in each picture,
the computed hip centre (`geometry.compute_derived`), the angle of pull and moment arm (`pull.compute_pull`), "dots apart" for the rigid
pairs, and `all_ok`. Every other picture keeps the numbers that were recorded. The files on disk are never touched: this works on the
copy of the numbers that `Trial` holds in memory.
"""
import math

import numpy as np

from .angles import measure
from .geometry import DERIVED_OPTIONAL, compute_derived, derived_input_dots
from .pull import COLUMNS as PULL_COLUMNS
from .pull import compute_pull
from .results import LOST

RIGID_PAIR_CHANGE_DEFAULT = 0.25


def _fmt(v, nd):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


def frames_between(t, from_s, to_s):
    return np.nonzero((t >= min(from_s, to_s)) & (t <= max(from_s, to_s)))[0]


def overlay_positions(tr, review):
    """Start from the recorded positions and lay the edits on in order (a later edit wins). Returns {dot: dict(x, y, lost, edited)} for the
    dots that have an edit, with full-length arrays; edits of a dot the trial does not have are skipped (named in the second result)."""
    pos, skipped = {}, []
    by_id = {}
    for r in review.dot_rows:
        by_id.setdefault(r["edit"], {})[r["frame"]] = r
    n = len(tr.t)
    for e in sorted(review.dots, key=lambda e: e.get("id", 0)):
        d = e["dot"]
        if d not in tr.tracks:
            skipped.append(d)
            continue
        if d not in pos:
            x, y = tr.tracks[d]
            pos[d] = {"x": x.copy(), "y": y.copy(), "lost": np.array([r[f"{d}_lost"] == "1" or r[f"{d}_x"] == "" for r in tr.rows]),
                      "edited": np.zeros(n, bool)}
        p = pos[d]
        kind = e["kind"]
        if kind in ("hold", "lost"):
            idx = frames_between(tr.t, e["from_s"], e["to_s"])
            if kind == "hold":
                p["x"][idx], p["y"][idx], p["lost"][idx] = float(e["x"]), float(e["y"]), False
            else:
                p["x"][idx], p["y"][idx], p["lost"][idx] = math.nan, math.nan, True
            p["edited"][idx] = True
        elif kind in ("manual", "retrack"):
            rows = by_id.get(e["id"], {})
            frames = sorted(f for f in rows if 0 <= f < n)
            for f in frames:
                x, y = rows[f]["x"], rows[f]["y"]
                if math.isnan(x) or math.isnan(y):
                    p["x"][f], p["y"][f], p["lost"][f] = math.nan, math.nan, True
                else:
                    p["x"][f], p["y"][f], p["lost"][f] = x, y, False
                p["edited"][f] = True
            if kind == "manual":                                      # between two clicks: a straight line from one to the other
                for a, b in zip(frames[:-1], frames[1:]):
                    if b - a > 1 and not (math.isnan(rows[a]["x"]) or math.isnan(rows[b]["x"])):
                        k = np.arange(a + 1, b)
                        w = (k - a) / float(b - a)
                        p["x"][k] = rows[a]["x"] + w * (rows[b]["x"] - rows[a]["x"])
                        p["y"][k] = rows[a]["y"] + w * (rows[b]["y"] - rows[a]["y"])
                        p["lost"][k], p["edited"][k] = False, True
    return pos, skipped


def usual_position(tr, dot, min_quality=0.8):
    """Where a dot usually is: the median position over the pictures where it was seen well (not lost, quality at least `min_quality`).
    For a fixed dot this is the place to hold it at. None when it was never seen well."""
    if dot not in tr.tracks:
        return None
    x, y = tr.tracks[dot]
    q = np.array([float(r.get(f"{dot}_q") or "nan") for r in tr.rows])
    ok = np.isfinite(x) & np.isfinite(y) & (np.nan_to_num(q, nan=0.0) >= min_quality)
    if not ok.any():
        ok = np.isfinite(x) & np.isfinite(y)
    return (float(np.median(x[ok])), float(np.median(y[ok]))) if ok.any() else None


def _inputs(angle, derived_defs):
    """The dots an angle depends on, looking through computed points (the hip centre) to the dots they are made from."""
    out = set()
    for name in angle.dots:
        spec = derived_defs.get(name)
        if spec is None:
            out.add(name)
        else:
            out.update(derived_input_dots(spec))
            out.update(spec[k] for k in DERIVED_OPTIONAL.get(spec.get("kind"), ()) if k in spec)
    return out


def apply(tr, review):
    """Put the review's dot edits into the Trial `tr` (in memory). Returns a dict: edited ({dot: [picture numbers]}), skipped (dots the trial
    does not have), notes (plain lines about what was recomputed)."""
    pos, skipped = overlay_positions(tr, review)
    derived_defs = ((tr.meta.get("derived_points") or {}).get("definitions")) or {}
    edited = {d: np.nonzero(p["edited"])[0] for d, p in pos.items()}
    tr.edited = {d: [int(i) for i in idx] for d, idx in edited.items() if idx.size}
    if not tr.edited:
        return {"edited": {}, "skipped": skipped, "notes": []}
    # 1. the dot columns of the edited pictures, in the form the recording wrote them
    for d, p in pos.items():
        for i in edited[d]:
            r = tr.rows[i]
            if p["lost"][i]:
                r[f"{d}_x"], r[f"{d}_y"], r[f"{d}_lost"], r[f"{d}_q"] = "", "", "1", 0.0
                if f"{d}_partial" in r:
                    r[f"{d}_partial"] = ""
            else:
                r[f"{d}_x"], r[f"{d}_y"], r[f"{d}_lost"], r[f"{d}_q"] = _fmt(p["x"][i], 2), _fmt(p["y"][i], 2), "0", ""     # (no tracker quality: set by hand)
                if f"{d}_partial" in r:
                    r[f"{d}_partial"] = "0"
        tr.tracks[d] = (p["x"], p["y"])
        tr.trail_segments[d] = tr.build_trail(d)

    def where(dot, i):
        if dot in pos:
            return None if pos[dot]["lost"][i] else (float(pos[dot]["x"][i]), float(pos[dot]["y"][i]))
        x, y = tr.tracks[dot][0][i], tr.tracks[dot][1][i]
        return None if (math.isnan(x) or math.isnan(y) or tr.rows[i][f"{dot}_lost"] == "1") else (float(x), float(y))

    def scale_px_per_mm(i):
        v = tr.rows[i].get("scale_px_per_cm")
        try:
            return float(v) / 10.0 if v not in ("", None) else None
        except ValueError:
            return None

    # 2. the computed points (the hip centre of a frontal test)
    for name, spec in derived_defs.items():
        if name not in tr.derived_tracks:
            continue
        ins = set(derived_input_dots(spec)) | {spec[k] for k in DERIVED_OPTIONAL.get(spec.get("kind"), ()) if k in spec}
        frames = sorted({int(i) for d in ins if d in edited for i in edited[d]})
        if not frames:
            continue
        dx, dy = tr.derived_tracks[name]
        dx, dy = dx.copy(), dy.copy()
        for i in frames:
            xy = {dn: where(dn, i) for dn in tr.dot_names}
            got = compute_derived({name: spec}, {k: v for k, v in xy.items() if v is not None}, scale_px_per_mm(i))
            p = got.get(name)
            dx[i], dy[i] = (p if p is not None else (math.nan, math.nan))
            tr.rows[i][f"{name}_x"], tr.rows[i][f"{name}_y"] = (("", "") if p is None else (_fmt(p[0], 2), _fmt(p[1], 2)))
        tr.derived_tracks[name] = (dx, dy)

    # 3. the angles, their deviation from the reference saved in the picture, and the in-tolerance flag
    tol_main = float(tr.meta.get("tolerance_deg") or 5.0)
    tol_nb = float(tr.meta.get("neighbour_tolerance_deg") or tol_main)
    angle_frames = set()
    for a in tr.angle_defs:
        ins = _inputs(a, derived_defs)
        frames = sorted({int(i) for d in ins if d in edited for i in edited[d]})
        angle_frames.update(frames)
        for i in frames:
            xy = {dn: where(dn, i) for dn in tr.dot_names}
            xy = {k: v for k, v in xy.items() if v is not None}
            if derived_defs:
                xy.update(compute_derived(derived_defs, xy, scale_px_per_mm(i)))
            res = measure(a, xy, tr.facing_right)
            deg = math.nan if res.status == LOST else float(res.value)
            ref = tr.ref[a.name][i]
            dev = deg - ref if not (math.isnan(deg) or math.isnan(ref)) else math.nan
            if math.isnan(dev):
                ok = ""
            else:
                ok = "1" if abs(dev) <= (tol_main if a.primary else tol_nb) else "0"
            tr.deg[a.name][i], tr.dev[a.name][i], tr.ok[a.name][i] = deg, dev, ok
            r = tr.rows[i]
            r[f"{a.name}_deg"], r[f"{a.name}_dev"], r[f"{a.name}_ok"] = _fmt(deg, 3), _fmt(dev, 3), ok

    # 4. dots on one bone that moved apart (the rigid pairs the recording checked)
    checks = tr.meta.get("checks") or {}
    pairs = [tuple(p) for p in checks.get("rigid_pairs") or []]
    ref_px = checks.get("rigid_pair_reference_px") or {}
    limit = float(checks.get("rigid_pair_change_limit_pct") or 100 * RIGID_PAIR_CHANGE_DEFAULT) / 100.0
    apart_frames = set()
    if pairs:
        for i in sorted({int(i) for d in pos for i in edited[d]}):
            parts = []
            for a_dot, b_dot in pairs:
                ref = ref_px.get(f"{a_dot}-{b_dot}")
                pa, pb = where(a_dot, i) if a_dot in tr.tracks else None, where(b_dot, i) if b_dot in tr.tracks else None
                if ref and pa is not None and pb is not None:
                    change = math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / float(ref) - 1.0
                    if abs(change) > limit:
                        parts.append(f"{a_dot}-{b_dot}:{100 * change:+.0f}")
            tr.rows[i]["dots_apart"] = ";".join(parts)
            if parts:
                apart_frames.add(i)

    # 5. all_ok, as the recording worked it out
    for i in sorted(angle_frames | apart_frames):
        r = tr.rows[i]
        apart = [p.split(":")[0].split("-") for p in (r.get("dots_apart") or "").split(";") if p]
        bad = False
        for a in tr.angle_defs:
            if not a.judged:
                continue
            lost_or_out = math.isnan(tr.deg[a.name][i]) or tr.ok[a.name][i] == "0"
            suspect = any(set(pair) & set(a.dots) for pair in apart)
            bad = bad or lost_or_out or suspect
        r["all_ok"] = "0" if bad else "1"

    # 6. the angle of pull and the moment arm
    spec = (tr.meta.get("pull") or {})
    seg, fulcrum = spec.get("segment_from_to"), spec.get("moment_arm_from_joint_dot")
    if tr.pull is not None and seg and fulcrum and "strap" in tr.tracks and "anchor" in tr.tracks:
        needs = {"strap", "anchor", *seg, fulcrum}
        frames = {int(i) for d in needs if d in edited for i in edited[d]}
        if fulcrum in derived_defs:                                   # a computed joint: its input dots move it
            ins = set(derived_input_dots(derived_defs[fulcrum])) | {derived_defs[fulcrum][k] for k in DERIVED_OPTIONAL.get(derived_defs[fulcrum].get("kind"), ())
                                                                    if k in derived_defs[fulcrum]}
            frames |= {int(i) for d in ins if d in edited for i in edited[d]}
        frames = sorted(frames)
        for c in PULL_COLUMNS:
            tr.pull[c] = tr.pull[c].copy()
        for i in frames:
            xy = {dn: where(dn, i) for dn in tr.dot_names}
            xy = {k: v for k, v in xy.items() if v is not None}
            if derived_defs:
                xy.update(compute_derived(derived_defs, xy, scale_px_per_mm(i)))
            got = None
            if all(k in xy for k in ("strap", "anchor", seg[0], seg[1], fulcrum)):
                got = compute_pull(xy["strap"], xy["anchor"], xy[seg[0]], xy[seg[1]], xy[fulcrum], tr.px_per_cm)
            for c in PULL_COLUMNS:
                v = (got or {}).get(c)
                tr.pull[c][i] = math.nan if v is None else float(v)
                tr.rows[i][c] = _fmt(v, 2)
    # 7. what the Check tab counts from the recording's own summary: the pairs that moved apart
    if pairs:
        found = {}
        for i, r in enumerate(tr.rows):
            for part in (r.get("dots_apart") or "").split(";"):
                if part:
                    key = part.split(":")[0]
                    c = found.setdefault(key, {"frames": 0, "first_t_s": float(tr.t[i]), "last_t_s": float(tr.t[i])})
                    c["frames"] += 1
                    c["last_t_s"] = float(tr.t[i])
        tr.meta.setdefault("checks", {})["dots_apart_frames"] = found
    notes = []
    for d, idx in tr.edited.items():
        notes.append(f"{d}: {len(idx)} picture{'' if len(idx) == 1 else 's'} changed ({tr.t[idx[0]]:.2f}-{tr.t[idx[-1]]:.2f} s)")
    return {"edited": tr.edited, "skipped": skipped, "notes": notes}


# ------------------------------------------------------------------------------------------------ the tracker, run after the fact
def retrack(tr, dot, seed_frame, seed_xy, direction="forward", stop_at=(), progress=None):
    """Follow `dot` through the raw video with the program's own dot tracker, starting from where you clicked on picture `seed_frame`.
    `direction`: 'forward', 'backward' or 'both'. `stop_at`: picture numbers already fixed by hand (the run stops before them). The other dots
    are passed as 'claimed' (where they are in each picture) so the dot cannot jump onto a neighbour. Returns (rows, why): rows =
    [{frame, t_s, x, y}] (NaN where the tracker lost the dot), why = '' or the reason it could not start. `progress(i, n)` may return False to stop."""
    from .tracker import NEIGHBOUR_EXCLUSION_FACTOR, DotTracker, to_gray
    if tr.cap is None:
        return [], "The video is not open."
    n = len(tr.t)
    stop = set(stop_at)

    def start():
        img = tr.frame(seed_frame)
        if img is None:
            return None, "The picture could not be read from the video."
        trk = DotTracker(dot, kind=tr.meta.get("dot_kind") or "auto")
        if not trk.init_at(to_gray(img), float(seed_xy[0]), float(seed_xy[1])):
            return None, "No dot was found where you clicked: click closer to the middle of the dot."
        return trk, ""

    def others(i):
        out = []
        for name in tr.dot_names:
            if name == dot:
                continue
            x, y = tr.tracks[name][0][i], tr.tracks[name][1][i]
            if not (math.isnan(x) or math.isnan(y)):
                out.append((float(x), float(y), NEIGHBOUR_EXCLUSION_FACTOR * 12.0))
        return out

    rows = []
    sweeps = []
    if direction in ("forward", "both"):
        sweeps.append(range(seed_frame + 1, n))
    if direction in ("backward", "both"):
        sweeps.append(range(seed_frame - 1, -1, -1))
    total = sum(len(s) for s in sweeps) or 1
    done = 0
    for sweep in sweeps:
        trk, why = start()
        if trk is None:
            return [], why
        if not any(r["frame"] == seed_frame for r in rows):
            rows.append({"frame": seed_frame, "t_s": float(tr.t[seed_frame]), "x": trk.x, "y": trk.y})
        for i in sweep:
            if i in stop:
                break
            img = tr.frame(i)
            if img is None:
                break
            st = trk.update(to_gray(img), claimed=others(i))
            rows.append({"frame": i, "t_s": float(tr.t[i]), "x": math.nan if st.lost else st.x, "y": math.nan if st.lost else st.y})
            done += 1
            if progress is not None and progress(done, total) is False:
                return sorted(rows, key=lambda r: r["frame"]), ""
    return sorted(rows, key=lambda r: r["frame"]), ""
