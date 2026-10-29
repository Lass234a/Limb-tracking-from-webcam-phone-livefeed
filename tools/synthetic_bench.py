"""Tracker stress benchmark on synthetic scenes with known truth (no camera, no participants).

usage: python tools/synthetic_bench.py [REPO_DIR] [--quick] [--out results.json]
REPO_DIR defaults to this project; give another copy of the project (e.g. a branch) to try a changed tracker on the same scenes.
Every scenario: mark the dots on frame 0 (click jittered by up to 3 px), track N frames, compare with truth.
Metrics per scenario (over all dots, all frames where truth is visible):
  rmse_px   : rms position error on frames the tracker reports not-lost
  false_px  : fraction of not-lost frames with error > 0.75 x dot diameter (tracker is confidently wrong)
  lost_vis  : fraction of frames where the dot is visible but tracker says lost
  recover   : mean frames to come back after a dot reappears (inf if never; capped)
  swaps     : frames where a not-lost dot sits within 1 diameter of a DIFFERENT dot's truth
"""
import itertools
import json
import math
import sys
import time
import zlib

import cv2
import numpy as np

from pathlib import Path

args_ = [a for a in sys.argv[1:] if not a.startswith("--") and a != (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None)]
repo = args_[0] if args_ else str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, repo)
from limbtrack import synthetic as syn          # noqa: E402
from limbtrack.tracker import TrackerSet, to_gray   # noqa: E402

QUICK = "--quick" in sys.argv
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
N = 45 if QUICK else 90
SIZE = (1280, 720)


def traj(kind, n):
    """Joint angle paths. Returns list of (hip, knee, ankle, tilt) per frame."""
    out = []
    for i in range(n):
        p = i / max(n - 1, 1)
        if kind == "still":
            out.append((80, 40, 5, 4))
        elif kind == "knee_slow":
            out.append((80, 20 + 80 * (0.5 - 0.5 * math.cos(2 * math.pi * p)), 5, 4))
        elif kind == "knee_fast":
            out.append((80, 10 + 100 * (0.5 - 0.5 * math.cos(4 * math.pi * p)), 5, 4))
        elif kind == "knee_very_fast":
            out.append((80, 5 + 110 * (0.5 - 0.5 * math.cos(6 * math.pi * p)), 5, 4))
        elif kind == "hip_sweep":
            out.append((10 + 80 * (0.5 - 0.5 * math.cos(2 * math.pi * p)), 90, 5, 4))
        elif kind == "ankle":
            out.append((80, 5, -20 + 40 * (0.5 - 0.5 * math.cos(4 * math.pi * p)), 4))
        elif kind == "all_moving":
            out.append((50 + 30 * math.sin(2 * math.pi * p), 60 + 40 * math.sin(4 * math.pi * p), 10 * math.sin(6 * math.pi * p), 6 * math.sin(2 * math.pi * p)))
    return out


def degrade(img, i, d):
    if d.get("blur"):
        s = d["blur"]
        img = cv2.GaussianBlur(img, (0, 0), s)
    if d.get("mblur"):
        k = d["mblur"]
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, ker)
    if d.get("flicker"):
        g = 1.0 + d["flicker"] * math.sin(i * 1.7)
        img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
    if d.get("gradient"):                       # light falls off across the picture, drifting over time
        h, w = img.shape[:2]
        ramp = np.linspace(1 - d["gradient"], 1 + d["gradient"], w, dtype=np.float32)[None, :, None]
        ramp = np.roll(ramp, int(i * 4), axis=1)
        img = np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    if d.get("jpeg"):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, d["jpeg"]])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if d.get("shake"):
        dx, dy = d["shake"] * math.sin(i * 2.3), d["shake"] * math.cos(i * 1.9)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return img


def run(sc):
    rng = np.random.default_rng(sc.get("seed", 1))
    n = N
    path = traj(sc["motion"], n)
    kind = sc.get("kind", "white")
    facing = sc.get("facing", "right")
    names = None
    truths, frames = [], []
    hide_fn = sc.get("hide")              # f(i, name) -> bool
    for i, (h, k, a, t) in enumerate(path):
        pts = syn.leg_points(hip_flex=h, knee_flex=k, ankle_df=a, pelvic_tilt=t, facing=facing)
        if sc.get("camera_shift"):
            sh = np.array([sc["camera_shift"] * math.sin(i / 7.0), 0.0])
            pts = {kk: v + sh for kk, v in pts.items()}
        names = list(pts)
        hide = tuple(nm for nm in names if hide_fn and hide_fn(i, nm))
        img = syn.render(pts, kind=kind, dot_radius=sc.get("radius", 7.0), noise=sc.get("noise", 3.0), seed=i + 1000 * sc.get("seed", 1),
                         distractors=sc.get("distractors", True), dot_gain=sc.get("gain", 1.0), hide=hide, patch=sc.get("patch"))
        img = degrade(img, i, sc)
        truths.append((pts, set(hide)))
        frames.append(img)
    ts = TrackerSet(names, kind=sc.get("track_kind", kind))
    g0 = to_gray(frames[0])
    for nm in names:
        x, y = truths[0][0][nm] + rng.uniform(-3, 3, 2)
        if not ts.init_dot(nm, g0, x, y):
            ts.trackers[nm].initialised = False
    n_vis = n_err = sq = lost_vis = false_n = swaps = 0
    rec_times, since, hidden_now = [], {}, set()
    t0 = time.perf_counter()
    for i in range(1, n):
        states = ts.update(to_gray(frames[i]))
        pts, hid = truths[i]
        for nm in names:
            st = states.get(nm)
            if st is None:
                continue
            vis = nm not in hid
            diam = max(ts.trackers[nm].marked_diam or 2 * sc.get("radius", 7.0), 4.0)
            if vis:
                n_vis += 1
                if st.lost:
                    lost_vis += 1
                else:
                    e = float(np.hypot(st.x - pts[nm][0], st.y - pts[nm][1]))
                    n_err += 1
                    sq += e * e
                    if e > 0.75 * diam:
                        false_n += 1
            if not st.lost:
                for other in names:
                    if other != nm and np.hypot(st.x - pts[other][0], st.y - pts[other][1]) < diam and \
                            np.hypot(st.x - pts[nm][0], st.y - pts[nm][1]) > 1.5 * diam:
                        swaps += 1
                        break
            # recovery: frames from reappearing until the tracker holds the dot again
            if not vis:
                hidden_now.add(nm)
            else:
                if nm in hidden_now:
                    hidden_now.discard(nm)
                    since[nm] = 0
                if since.get(nm) is not None:
                    if not st.lost and np.hypot(st.x - pts[nm][0], st.y - pts[nm][1]) < 0.75 * diam:
                        rec_times.append(since[nm])
                        since[nm] = None
                    else:
                        since[nm] += 1
    for v in since.values():
        if v is not None:
            rec_times.append(999)
    dt = (time.perf_counter() - t0) / max(n - 1, 1)
    return {
        "rmse_px": round(math.sqrt(sq / n_err), 3) if n_err else None,
        "false": round(false_n / max(n_err, 1), 4),
        "lost_vis": round(lost_vis / max(n_vis, 1), 4),
        "recover": round(float(np.mean(rec_times)), 1) if rec_times else None,
        "swaps": swaps,
        "ms_frame": round(dt * 1000, 2),
    }


def scenarios():
    S = []
    base = {"seed": 1}
    motions = ["still", "knee_slow", "knee_fast", "knee_very_fast", "hip_sweep", "ankle", "all_moving"]
    for kind, m in itertools.product(["white", "black"], motions):
        S.append({**base, "name": f"motion/{kind}/{m}", "kind": kind, "motion": m})
    for kind in ["white", "black"]:
        for facing in ["right", "left"]:
            S.append({**base, "name": f"facing/{kind}/{facing}", "kind": kind, "motion": "knee_slow", "facing": facing})
    for r in [3, 4, 5, 6, 9, 12, 16]:
        for kind in ["white", "black"]:
            S.append({**base, "name": f"radius/{kind}/{r}", "kind": kind, "motion": "knee_slow", "radius": r})
    for nz in [0, 6, 10, 15, 25]:
        for kind in ["white", "black"]:
            S.append({**base, "name": f"noise/{kind}/{nz}", "kind": kind, "motion": "knee_slow", "noise": nz})
    for g in [0.15, 0.25, 0.4, 0.6]:
        for kind in ["white", "black"]:
            S.append({**base, "name": f"lowcontrast/{kind}/{g}", "kind": kind, "motion": "knee_slow", "gain": g})
    for b in [0.8, 1.5, 2.5]:
        S.append({**base, "name": f"blur/white/{b}", "motion": "knee_fast", "blur": b})
        S.append({**base, "name": f"blur/black/{b}", "kind": "black", "motion": "knee_fast", "blur": b})
    for k in [5, 9, 15]:
        S.append({**base, "name": f"motionblur/white/{k}", "motion": "knee_fast", "mblur": k})
    for f in [0.15, 0.3]:
        S.append({**base, "name": f"flicker/white/{f}", "motion": "knee_slow", "flicker": f})
        S.append({**base, "name": f"flicker/black/{f}", "kind": "black", "motion": "knee_slow", "flicker": f})
    for gr in [0.3, 0.5]:
        S.append({**base, "name": f"gradient/white/{gr}", "motion": "knee_slow", "gradient": gr})
        S.append({**base, "name": f"gradient/black/{gr}", "kind": "black", "motion": "knee_slow", "gradient": gr})
    for q in [30, 10]:
        S.append({**base, "name": f"jpeg/white/{q}", "motion": "knee_slow", "jpeg": q})
        S.append({**base, "name": f"jpeg/black/{q}", "kind": "black", "motion": "knee_slow", "jpeg": q})
    for sh in [2, 5, 10]:
        S.append({**base, "name": f"shake/white/{sh}", "motion": "knee_slow", "shake": sh})
    for patch in [12, 16]:
        S.append({**base, "name": f"tapepatch/black/{patch}", "kind": "black", "motion": "knee_slow", "patch": patch})
        S.append({**base, "name": f"tapepatch/white/{patch}", "kind": "white", "motion": "knee_slow", "patch": patch})
    # occlusions: a dot hidden for a stretch, reappears in place
    for nm in ["malleolus", "epicondyle", "mt5_head", "fibular_head"]:
        for dur in [5, 15]:
            S.append({**base, "name": f"occlude/white/{nm}/{dur}", "motion": "knee_slow",
                      "hide": (lambda i, x, nm=nm, dur=dur: x == nm and 20 <= i < 20 + dur)})
    S.append({**base, "name": "occlude/white/two_dots", "motion": "knee_slow", "hide": lambda i, x: x in ("malleolus", "fibular_head") and 20 <= i < 30})
    S.append({**base, "name": "occlude/black/malleolus/10", "kind": "black", "motion": "knee_slow", "hide": lambda i, x: x == "malleolus" and 20 <= i < 30})
    S.append({**base, "name": "occlude/white/fast_knee/malleolus/8", "motion": "knee_fast", "hide": lambda i, x: x == "malleolus" and 20 <= i < 28})
    # auto colour with mixed
    S.append({**base, "name": "autokind/white", "kind": "white", "track_kind": "auto", "motion": "knee_slow"})
    S.append({**base, "name": "autokind/black", "kind": "black", "track_kind": "auto", "motion": "knee_slow"})
    # combos (realistic bad day)
    S.append({**base, "name": "combo/white/noise10+blur1+jpeg30+r5", "motion": "knee_fast", "noise": 10, "blur": 1.0, "jpeg": 30, "radius": 5})
    S.append({**base, "name": "combo/black/noise10+blur1+jpeg30+r5", "kind": "black", "motion": "knee_fast", "noise": 10, "blur": 1.0, "jpeg": 30, "radius": 5})
    S.append({**base, "name": "combo/black/gain.4+flicker+r6", "kind": "black", "motion": "all_moving", "gain": 0.4, "flicker": 0.2, "radius": 6})
    S.append({**base, "name": "combo/white/gain.4+gradient+shake", "motion": "all_moving", "gain": 0.4, "gradient": 0.3, "shake": 3})
    return S


if __name__ == "__main__":
    res = {}
    sc_list = scenarios()
    if QUICK:
        sc_list = [s for s in sc_list if zlib.crc32(s["name"].encode()) % 3 == 0]
    for sc in sc_list:
        sc = dict(sc)
        try:
            r = run(sc)
        except Exception as e:                      # a scenario that crashes is itself a finding
            r = {"error": repr(e)}
        res[sc["name"]] = r
        print(f"{sc['name']:46s} {r}", flush=True)
    if OUT:
        json.dump(res, open(OUT, "w"), indent=1)
