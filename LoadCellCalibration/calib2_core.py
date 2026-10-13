"""Load cell calibration, session 2 (13 Oct 2026, ceiling rig, 0 to 250 kg up and down) plus the comparison with session 1 (12 Oct).

All the numbers come from here; `analyse_calibration_v2.py` turns them into the HTML report.  Needs only numpy.

How the runs are classified (the file names and the `weight_1` column are not always right, so the voltage is used to check them):
  ZERO   no chain and no strap on the cell
  HW     chain and strap hanging on the cell, no plates ("1 kg" in the notes)
  PLATES chain and strap plus plates of the stated mass
The first SKIP seconds of every run are dropped (loading and settling), then a whole number of 20 ms periods is averaged,
so 50 Hz hum averages out exactly.
"""
import csv
import glob
import os
import re
from datetime import datetime

import numpy as np

DAY1 = r"C:\Users\llind\Desktop\Studie\KVT3\Kalibrerings data"
DAY2 = DAY1 + r"\Omgang 2"
RATE = 1000.0
SKIP = 6.0
G = 9.80665
PEAK_STAMP = "15-52-38"          # the 250 kg run: runs before it are 'up', after it 'down'

# classification of the day-2 files (stem after 'LogFile_' without .csv) -> (kind, plates kg, why)
CLASSIFY = {
    "0kg_13-10-13-15-00-46": ("ZERO", 0, ""),
    "1kg_13-10-13-15-02-55": ("HW", 0, "labelled 1 kg: chain and strap only"),
    "1kg_13-10-13-16-10-05": ("HW", 0, "labelled 1 kg: chain and strap only"),
    "0kg_13-10-13-15-14-30": ("DROP", None, "6 s file named 0 kg but the signal is at the 50 kg level: the start of the 50 kg run"),
    "lassekg_13-10-13-15-45-58": ("DROP", None, "load applied and removed during the file (signal jumps by 3 V): not a steady run"),
    "200kg_13-10-13-15-56-30": ("DROP", None, "only 8 s long with a start-up disturbance; repeated in the next file"),
    "0kg_13-10-13-16-11-12": ("HW", 0, "named 0 kg, but the file's own label is 1 and its voltage equals the other 1 kg run: chain and strap only"),
    "0kg_13-10-13-16-12-09": ("ZERO", 0, ""),
    "175kg_13-10-13-15-39-56": ("PLATES", 175, "the label column inside the file still says 150, the voltage equals the other 175 kg runs"),
}


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        hdr = next(r)
        iv = hdr.index("Dev1/ai0")
        il = hdr.index("weight_1") if "weight_1" in hdr else None
        t, v, lab = [], [], []
        for row in r:
            if len(row) <= iv:
                continue
            try:
                tt = datetime.strptime(row[0], "%d/%m/%Y %H:%M:%S.%f").timestamp()
                vv = float(row[iv])
            except ValueError:
                continue
            t.append(tt)
            v.append(vv)
            lab.append(float(row[il]) if il is not None and len(row) > il and row[il] != "" else np.nan)
    return np.array(t), np.array(v), np.array(lab)


def steady(v, skip=SKIP):
    v = v[int(skip * RATE):]
    return v[: v.size // 20 * 20]


def hum_peak(v):
    x = v - v.mean()
    w = np.hanning(x.size)
    sp = np.abs(np.fft.rfft(x * w)) * 2 / w.sum()
    f = np.fft.rfftfreq(x.size, 1 / RATE)
    return float(sp[(f > 48) & (f < 52)].max())


def smooth_sd(v, n=100):
    c = np.cumsum(np.insert(v - v.mean(), 0, 0.0))
    return float(((c[n:] - c[:-n]) / n).std())


def ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = r @ r / dof
    return beta, s2 * np.linalg.inv(X.T @ X), r, float(np.sqrt(s2)), dof


def load_day2():
    runs = []
    for p in sorted(glob.glob(os.path.join(DAY2, "LogFile_*.csv")), key=lambda q: re.search(r"13-10-13-(\d\d-\d\d-\d\d)", q).group(1)):
        stem = os.path.basename(p)[8:-4]
        t, v, lab = load_csv(p)
        m = re.match(r"(\d+)kg", stem)
        name_kg = float(m.group(1)) if m else None
        if stem in CLASSIFY:
            kind, plates, why = CLASSIFY[stem]
        else:
            kind, plates, why = "PLATES", name_kg, ""
        sv = steady(v)
        runs.append(dict(stem=stem, t0=float(t[0]), dur=float(t[-1] - t[0]), n=int(v.size), kind=kind, plates=plates, why=why, name_kg=name_kg,
                         label_in_file=float(np.nanmax(lab)) if np.isfinite(lab).any() else None,
                         mean=float(sv.mean()) if kind != "DROP" and sv.size > 3000 else float(v.mean()), hum=hum_peak(sv) if sv.size > 3000 else float("nan"),
                         sd_raw=float(sv.std()) if sv.size else float("nan"), sd_smooth=smooth_sd(sv) if sv.size > 3000 else float("nan"),
                         up=PEAK_STAMP not in stem and None))
    t_peak = next(r["t0"] for r in runs if PEAK_STAMP in r["stem"])
    for r in runs:
        r["phase"] = "up" if r["t0"] < t_peak else ("peak" if r["t0"] == t_peak else "down")
    return runs


def load_day1():
    out = []
    files = sorted(glob.glob(os.path.join(DAY1, "LogFile_*.csv")), key=lambda p: re.search(r"(\d\d-\d\d-\d\d-\d\d-\d\d-\d\d)", p).group(1)[9:])
    for p in files:
        with open(p, newline="", encoding="utf-8-sig") as f:
            if "weight_1" not in next(csv.reader(f)):
                continue
        t, v, lab = load_csv(p)
        n = v.size // 20 * 20
        out.append(dict(stem=os.path.basename(p)[8:-4], t0=float(t[0]), kg=float(np.nanmax(lab)), mean=float(v[:n].mean())))
    return out


def analyse():
    R = {}
    runs = load_day2()
    R["runs"] = runs
    used = [r for r in runs if r["kind"] in ("HW", "PLATES")]
    P = np.array([r["plates"] for r in used], float)
    V = np.array([r["mean"] for r in used])
    T = np.array([(r["t0"] - runs[0]["t0"]) / 60 for r in used])
    R["used"] = used

    # main fit: chain+strap attached, plates 0..250 kg
    X = np.column_stack([np.ones_like(P), P])
    beta, cov, res, rmse, dof = ols(X, V)
    a, b = beta
    R.update(a=a, b=b, se_a=float(np.sqrt(cov[0, 0])), se_b=float(np.sqrt(cov[1, 1])), rmse_v=rmse, rmse_kg=rmse / b, dof=dof, resid_v=res, resid_kg=res / b,
             P=P, V=V, T=T, n_used=len(used))
    R["r2"] = 1 - (res @ res) / ((V - V.mean()) ** 2).sum()
    x250 = np.array([1.0, 250.0])
    R["se250_kg"] = float(np.sqrt(x250 @ cov @ x250) / b)
    R["n_per_v"] = G / b
    R["kg_per_v"] = 1 / b

    # curvature
    X2 = np.column_stack([np.ones_like(P), P, P ** 2])
    b2, c2, r2_, rm2, d2 = ols(X2, V)
    R.update(quad=b2[2], quad_se=float(np.sqrt(c2[2, 2])), quad_t=float(b2[2] / np.sqrt(c2[2, 2])), quad_kg250=float(b2[2] * 250 ** 2 / b),
             quad_kg250_se=float(np.sqrt(c2[2, 2]) * 250 ** 2 / b), rmse_quad_kg=rm2 / b2[1])
    # a straight line only through plates >= 25 kg, then see where the hardware-only runs land
    m25 = P >= 25
    b25, c25, r25, rm25, d25 = ols(X[m25], V[m25])
    R["fit25"] = dict(a=b25[0], b=b25[1], rmse_kg=rm25 / b25[1], hw_pred=b25[0], hw_meas=float(np.mean([r["mean"] for r in used if r["kind"] == "HW"])))
    R["hw_gap_kg"] = (R["fit25"]["hw_meas"] - b25[0]) / b25[1]

    # up versus down and time drift
    up = np.array([r["phase"] == "up" for r in used]).astype(float)
    dn = np.array([r["phase"] == "down" for r in used]).astype(float)
    X3 = np.column_stack([np.ones_like(P), P, dn])
    b3, c3, r3, rm3, d3 = ols(X3, V)
    R.update(down_offset_v=b3[2], down_offset_se_v=float(np.sqrt(c3[2, 2])), down_offset_kg=b3[2] / b3[1], rmse_updown_kg=rm3 / b3[1], slope_updown=b3[1])
    X4 = np.column_stack([np.ones_like(P), P, T])
    b4, c4, r4, rm4, d4 = ols(X4, V)
    R.update(drift_v_per_min=b4[2], drift_se=float(np.sqrt(c4[2, 2])), rmse_drift_kg=rm4 / b4[1], slope_drift=b4[1])
    X5 = np.column_stack([np.ones_like(P), P, dn, T])
    b5, c5, r5, rm5, d5 = ols(X5, V)
    R.update(rmse_both_kg=rm5 / b5[1], slope_both=b5[1], both_down=b5[2] / b5[1], both_drift=b5[3])
    # slope from the ascending runs only and from the descending runs only (day 1 went up only)
    mu = np.array([r["phase"] in ("up", "peak") for r in used])
    md = np.array([r["phase"] in ("down", "peak") for r in used])
    bu, cu, _, rmu, _ = ols(X[mu], V[mu])
    bd, cd, _, rmd, _ = ols(X[md], V[md])
    R.update(slope_up=bu[1], slope_up_se=float(np.sqrt(cu[1, 1])), slope_down=bd[1], slope_down_se=float(np.sqrt(cd[1, 1])),
             rmse_up_kg=rmu / bu[1], rmse_down_kg=rmd / bd[1], icpt_up=bu[0], icpt_down=bd[0])
    # paired up / down at the same plate mass
    pairs = []
    for k in sorted(set(P)):
        u = [r["mean"] for r in used if r["plates"] == k and r["phase"] == "up"]
        d = [r["mean"] for r in used if r["plates"] == k and r["phase"] == "down"]
        if u and d:
            pairs.append((k, np.mean(u), np.mean(d)))
    R["pairs"] = pairs

    # zero and hardware-only repeats
    R["zeros"] = [(r["t0"], r["mean"]) for r in runs if r["kind"] == "ZERO"]
    R["hws"] = [(r["t0"], r["mean"]) for r in runs if r["kind"] == "HW"]
    z0, z1 = R["zeros"][0][1], R["zeros"][-1][1]
    R["zero_drift_v"] = z1 - z0
    R["zero_minutes"] = (R["zeros"][-1][0] - R["zeros"][0][0]) / 60
    R["hw_mass_kg"] = (np.mean([v for _, v in R["hws"]]) - np.mean([v for _, v in R["zeros"]])) / b
    R["hw_mass_kg_first"] = (R["hws"][0][1] - z0) / b
    R["hw_mass_kg_last"] = (R["hws"][-1][1] - z1) / b

    # day 1 against day 2
    d1 = load_day1()
    k1 = np.array([r["kg"] for r in d1])
    v1 = np.array([r["mean"] for r in d1])
    X1 = np.column_stack([np.ones_like(k1), k1])
    bb1, cc1, rr1, rm1, dd1 = ols(X1, v1)
    R["d1"] = dict(a=bb1[0], b=bb1[1], se_b=float(np.sqrt(cc1[1, 1])), rmse_kg=rm1 / bb1[1], n=len(d1), kg=k1, v=v1, resid_kg=rr1 / bb1[1])
    diff = (b - bb1[1]) / bb1[1]
    se_diff = np.sqrt(R["se_b"] ** 2 + R["d1"]["se_b"] ** 2) / bb1[1]
    R["slope_diff_pct"] = 100 * diff
    R["slope_diff_se_pct"] = 100 * se_diff
    # pooled: shared slope, one intercept per day
    Pp = np.concatenate([P, k1])
    Vp = np.concatenate([V, v1])
    day = np.concatenate([np.zeros_like(P), np.ones_like(k1)])
    Xp = np.column_stack([1 - day, day, Pp])
    bp, cp, rp, rmp, dp = ols(Xp, Vp)
    R["pool"] = dict(b=bp[2], se_b=float(np.sqrt(cp[2, 2])), a2=bp[0], a1=bp[1], rmse_kg=rmp / bp[2], n=len(Vp))
    R["pool_slope_change_pct"] = 100 * (bp[2] - b) / b
    R["pool_se_change_pct"] = 100 * (np.sqrt(cp[2, 2]) - R["se_b"]) / R["se_b"]
    R["pool_pred250_shift_kg"] = (bp[2] - b) * 250 / b
    # how much of the slope information day 1 carries
    Sxx2 = ((P - P.mean()) ** 2).sum()
    Sxx1 = ((k1 - k1.mean()) ** 2).sum()
    R["info_day1_share"] = float(Sxx1 / (Sxx1 + Sxx2))
    # day 1 shape check with day 2 slope: fit only an intercept to day 1 and see the scatter
    a1_only = np.mean(v1 - b * k1)
    r1_only = (v1 - a1_only - b * k1) / b
    R["d1_with_d2_slope_rmse_kg"] = float(np.sqrt(np.sum(r1_only ** 2) / (len(v1) - 1)))
    R["d1_with_d2_slope_resid"] = r1_only
    return R


if __name__ == "__main__":
    R = analyse()
    print(f"runs used {R['n_used']}: V = {R['a']:.5f} + {R['b']*1000:.4f} mV/kg x plates   se(b) = {R['se_b']*1000:.4f} mV/kg ({100*R['se_b']/R['b']:.3f} %)   R2 = {R['r2']:.7f}   rmse = {R['rmse_kg']:.3f} kg")
    print(f"N/V = {R['n_per_v']:.3f}   kg/V = {R['kg_per_v']:.3f}   se of line at 250 kg = {R['se250_kg']:.3f} kg")
    print(f"quad term: {R['quad_kg250']:+.2f} kg at 250 +- {R['quad_kg250_se']:.2f} (t={R['quad_t']:+.2f}); rmse straight {R['rmse_kg']:.3f} vs curved {R['rmse_quad_kg']:.3f} kg")
    f = R["fit25"]
    print(f"fit on plates>=25 only: intercept {f['a']:.5f} V; hardware-only runs measured {f['hw_meas']:.5f} V -> gap {R['hw_gap_kg']:+.3f} kg (rmse {f['rmse_kg']:.3f})")
    print(f"down - up offset: {R['down_offset_v']*1000:+.2f} +- {R['down_offset_se_v']*1000:.2f} mV = {R['down_offset_kg']:+.3f} kg; rmse with it {R['rmse_updown_kg']:.3f} kg")
    print(f"time drift term: {R['drift_v_per_min']*1000:+.3f} +- {R['drift_se']*1000:.3f} mV/min; rmse with it {R['rmse_drift_kg']:.3f} kg; both: rmse {R['rmse_both_kg']:.3f}, down {R['both_down']:+.3f} kg, drift {R['both_drift']*1000:+.3f} mV/min")
    print(f"ascending only: slope {R['slope_up']*1000:.4f} +- {R['slope_up_se']*1000:.4f} mV/kg rmse {R['rmse_up_kg']:.3f} kg; descending only: slope {R['slope_down']*1000:.4f} +- {R['slope_down_se']*1000:.4f} rmse {R['rmse_down_kg']:.3f} kg; intercepts {R['icpt_up']:.5f} / {R['icpt_down']:.5f}")
    print("paired up/down (kg, up V, down V, diff mV, diff kg):")
    for k, u, d in R["pairs"]:
        print(f"   {k:5.0f} {u:.5f} {d:.5f} {1000*(d-u):+.2f} mV {(d-u)/R['b']:+.3f} kg")
    print(f"zero: {[round(v,5) for _, v in R['zeros']]} drift {R['zero_drift_v']*1000:+.2f} mV in {R['zero_minutes']:.0f} min = {R['zero_drift_v']/R['b']:+.3f} kg")
    print(f"hardware-only: {[round(v,5) for _, v in R['hws']]}")
    print(f"chain+strap mass from cell: {R['hw_mass_kg']:.3f} kg (first {R['hw_mass_kg_first']:.3f}, last {R['hw_mass_kg_last']:.3f})")
    print("residuals (kg):")
    for r_, p_, ph in zip(R["resid_kg"], R["P"], [u["phase"] for u in R["used"]]):
        print(f"   {p_:6.0f} {ph:5s} {r_:+.3f}")
    d1 = R["d1"]
    print(f"day 1: slope {d1['b']*1000:.4f} +- {d1['se_b']*1000:.4f} mV/kg ({d1['n']} runs, rmse {d1['rmse_kg']:.3f} kg)")
    print(f"slope day2 vs day1: {R['slope_diff_pct']:+.2f} % +- {R['slope_diff_se_pct']:.2f} % (formal)")
    p = R["pool"]
    print(f"pooled slope {p['b']*1000:.4f} +- {p['se_b']*1000:.4f}; change vs day2 alone {R['pool_slope_change_pct']:+.3f} %, se change {R['pool_se_change_pct']:+.1f} %, shift at 250 kg {R['pool_pred250_shift_kg']:+.3f} kg; pooled rmse {p['rmse_kg']:.3f} kg; day 1 share of slope information {100*R['info_day1_share']:.1f} %")
    print(f"day 1 with day 2 slope: rmse {R['d1_with_d2_slope_rmse_kg']:.3f} kg (own slope {d1['rmse_kg']:.3f})")
    print("\nrun table")
    for r in R["runs"]:
        print(f"  {r['stem'][:30]:30s} {r['kind']:6s} plates={r['plates']} name={r['name_kg']} label={r['label_in_file']} {r['phase']:4s} mean={r['mean']:.5f} hum={r['hum']*1000:.1f} mV sd_raw={r['sd_raw']*1000:.1f} {r['why']}")
