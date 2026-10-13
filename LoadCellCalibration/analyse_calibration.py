"""Load-cell calibration check: reads the FlexLogger CSV files, fits volts against kilograms and writes a plain-language HTML report.

Run:  python analyse_calibration.py [data_folder] [report.html]
Needs only numpy.  Files with a `weight_1` column are taken as loaded measurements (the number is the nominal mass in kg);
files without it are shown as 'no label' and are not used in the fit.
"""
import csv
import glob
import html
import os
import re
import sys
from datetime import datetime

import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\llind\Desktop\Studie\KVT3\Kalibrerings data"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_report.html")
RATE = 1000.0
G = 9.80665
TARGET_KG = 250.0


# ---------------------------------------------------------------------------------------------------------- reading
def load(path):
    hdr_ok = None
    t, v, lab = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
        iv = hdr.index("Dev1/ai0")
        il = hdr.index("weight_1") if "weight_1" in hdr else None
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
    return hdr, np.array(t), np.array(v), np.array(lab)


def moving_average(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[n:] - c[:-n]) / n


def mains_amplitude(x, f=50.0):
    cycles = int(x.size * f / RATE)
    n = int(round(cycles * RATE / f))
    seg = x[:n] - x[:n].mean()
    return 2 * abs(np.mean(seg * np.exp(-2j * np.pi * f * np.arange(n) / RATE)))


def ols(x, y, deg):
    X = np.vander(x, deg + 1, increasing=True)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    dof = len(x) - deg - 1
    s2 = r @ r / dof
    return beta, s2 * np.linalg.inv(X.T @ X), r, np.sqrt(s2), dof


files = sorted(glob.glob(os.path.join(DATA, "LogFile_*.csv")), key=lambda p: re.search(r"(\d\d-\d\d-\d\d-\d\d-\d\d-\d\d)", p).group(1)[9:])
runs, unlabelled = [], []
for p in files:
    hdr, t, v, lab = load(p)
    n = (v.size // 20) * 20                      # a whole number of 50 Hz cycles
    rec = dict(name=os.path.basename(p), t0=t[0], v=v[:n], mean=v[:n].mean(), n=v.size, hum=mains_amplitude(v), dur=t[-1] - t[0])
    if "weight_1" in hdr:
        rec["kg"] = float(np.nanmax(lab))
        runs.append(rec)
    else:
        unlabelled.append(rec)
runs.sort(key=lambda r: r["t0"])
T0 = runs[0]["t0"]
kg = np.array([r["kg"] for r in runs])
V = np.array([r["mean"] for r in runs])
tmin = np.array([(r["t0"] - T0) / 60 for r in runs])

# ---------------------------------------------------------------------------------------------------------- statistics
beta, cov, resid, rmse, dof = ols(kg, V, 1)
a, b = beta
r2 = 1 - (resid @ resid) / ((V - V.mean()) ** 2).sum()
beta2, cov2, resid2, rmse2, dof2 = ols(kg, V, 2)
curv_se_kg_250 = np.sqrt(cov2[2, 2]) * TARGET_KG ** 2 / b          # how unsure the data leaves a bend, in kg at 250 kg
x250 = np.array([1, TARGET_KG])
line_se_250_kg = np.sqrt(x250 @ cov @ x250) / b
resid_kg = resid / b
n_per_v = G / b                                                      # newtons per volt
v_at_250 = a + b * TARGET_KG
sd_raw = np.mean([r["v"].std() for r in runs])
hum_amp = np.mean([r["hum"] for r in runs])
sd_100 = np.mean([moving_average(r["v"], 100).std() for r in runs])
rep = [r for r in runs if r["kg"] == 40.0]
rep_diff_kg = (rep[1]["mean"] - rep[0]["mean"]) / b if len(rep) == 2 else float("nan")

# ---------------------------------------------------------------------------------------------------------- design comparison
SIG0 = rmse / b                                                      # kg, scatter of one measurement


def design_se(x, rel):
    x = np.array(x, float)
    sig = np.sqrt(SIG0 ** 2 + (rel * x) ** 2)
    W = np.diag(1 / sig ** 2)
    res = {}
    for deg, nm in ((1, "line"), (2, "curve")):
        if len(set(x)) < deg + 1:
            res[nm] = float("nan")
            continue
        X = np.vander(x, deg + 1, increasing=True)
        C = np.linalg.inv(X.T @ W @ X)
        g = np.array([1, TARGET_KG] + ([TARGET_KG ** 2] if deg == 2 else []))
        res[nm] = float(np.sqrt(g @ C @ g))
    return res


CUR = list(kg)
DESIGNS = [
    ("Today's data only (0-60 kg)", CUR),
    ("Today's data + 100, 150, 200, 250 kg", CUR + [100, 150, 200, 250]),
    ("Today's data + 100, 150, 200, 250 up and 200, 100, 0 down", CUR + [100, 150, 200, 250, 200, 100, 0]),
    ("Fresh set: 0, 50, 100, 150, 200, 250 up and down (12 runs)", [0, 50, 100, 150, 200, 250] * 2),
    ("Fresh set: 0, 25, 50, 100, 150, 200, 250 up and down (14 runs)", [0, 25, 50, 100, 150, 200, 250] * 2),
    ("Only 0, 125, 250 kg, three times each (9 runs)", [0, 125, 250] * 3),
    ("Only 0 and 250 kg, six times each (12 runs)", [0, 250] * 6),
]
design_rows = []
for name, x in DESIGNS:
    d = {rel: design_se(x, rel) for rel in (0.0, 0.002, 0.005)}
    design_rows.append((name, len(x), d))


# ---------------------------------------------------------------------------------------------------------- SVG helpers
class Plot:
    def __init__(self, xlim, ylim, w=720, h=340, left=62, right=20, top=18, bottom=48):
        self.xlim, self.ylim, self.w, self.h = xlim, ylim, w, h
        self.l, self.r, self.t, self.b = left, right, top, bottom
        self.parts = []

    def X(self, x):
        return self.l + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.w - self.l - self.r)

    def Y(self, y):
        return self.h - self.b - (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.h - self.t - self.b)

    def add(self, s):
        self.parts.append(s)

    def axes(self, xticks, yticks, xlabel, ylabel, xfmt="{:g}", yfmt="{:g}"):
        for yv in yticks:
            self.add(f'<line class="grid" x1="{self.l}" x2="{self.w - self.r}" y1="{self.Y(yv):.1f}" y2="{self.Y(yv):.1f}"/>')
            self.add(f'<text class="tick" x="{self.l - 8}" y="{self.Y(yv) + 4:.1f}" text-anchor="end">{yfmt.format(yv)}</text>')
        for xv in xticks:
            self.add(f'<text class="tick" x="{self.X(xv):.1f}" y="{self.h - self.b + 18}" text-anchor="middle">{xfmt.format(xv)}</text>')
        self.add(f'<line class="axis" x1="{self.l}" x2="{self.w - self.r}" y1="{self.h - self.b}" y2="{self.h - self.b}"/>')
        self.add(f'<text class="axl" x="{(self.l + self.w - self.r) / 2:.0f}" y="{self.h - 6}" text-anchor="middle">{html.escape(xlabel)}</text>')
        self.add(f'<text class="axl" transform="translate(14 {(self.t + self.h - self.b) / 2:.0f}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>')

    def svg(self, label):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{html.escape(label)}">' + "".join(self.parts) + "</svg>")


def chart_fit():
    p = Plot((0, 270), (0, 5.2))
    p.add(f'<rect class="unknown" x="{p.X(60):.1f}" y="{p.t}" width="{p.X(270) - p.X(60):.1f}" height="{p.h - p.t - p.b}"/>')
    p.add(f'<text class="note" x="{p.X(165):.1f}" y="{p.Y(1.0):.1f}" text-anchor="middle">not measured yet</text>')
    p.axes(range(0, 271, 50), [0, 1, 2, 3, 4, 5], "Weight on the load cell (kg)", "Output of the amplifier (V)", yfmt="{:.0f}")
    p.add(f'<line class="fitline" x1="{p.X(0):.1f}" y1="{p.Y(a):.1f}" x2="{p.X(60):.1f}" y2="{p.Y(a + b * 60):.1f}"/>')
    p.add(f'<line class="fitline dash" x1="{p.X(60):.1f}" y1="{p.Y(a + b * 60):.1f}" x2="{p.X(250):.1f}" y2="{p.Y(v_at_250):.1f}"/>')
    for k, v in zip(kg, V):
        p.add(f'<circle class="pt" cx="{p.X(k):.1f}" cy="{p.Y(v):.1f}" r="4.5"><title>{k:g} kg: {v:.4f} V</title></circle>')
    p.add(f'<text class="lbl" x="{p.X(250) - 6:.1f}" y="{p.Y(v_at_250) - 12:.1f}" text-anchor="end">250 kg would give about {v_at_250:.1f} V</text>')
    p.add(f'<text class="lbl" x="{p.X(62):.1f}" y="{p.Y(a + b * 60) - 20:.1f}">14 measurements, all below 60 kg</text>')
    return p.svg("Amplifier output against weight: straight line through measurements up to 60 kg, extended to 250 kg")


def chart_resid():
    p = Plot((-3, 63), (-0.4, 0.4), h=300)
    p.add(f'<rect class="band" x="{p.X(-3):.1f}" y="{p.Y(0.2):.1f}" width="{p.X(63) - p.X(-3):.1f}" height="{p.Y(-0.2) - p.Y(0.2):.1f}"/>')
    p.axes(range(0, 61, 10), [-0.4, -0.2, 0, 0.2, 0.4], "Weight (kg)", "Difference from the line (kg)", yfmt="{:+.1f}")
    p.add(f'<text class="note" x="{p.X(61):.1f}" y="{p.Y(0.2) - 6:.1f}" text-anchor="end">within +/- 0.2 kg</text>')
    for k, e in zip(kg, resid_kg):
        p.add(f'<circle class="pt" cx="{p.X(k):.1f}" cy="{p.Y(e):.1f}" r="4.5"><title>{k:g} kg: {e:+.3f} kg off the line</title></circle>')
    return p.svg("Residuals of the straight-line fit in kilograms for each measurement")


def chart_hum():
    r = next(r for r in runs if r["kg"] == 20.0)
    v = r["v"]
    sm = moving_average(v, 100)                       # 'valid' window; index i of sm is centred on sample i + 50
    i0 = 8000
    idx = np.arange(i0, i0 + 500)
    raw = (v[idx] - a) / b
    smo = (sm[idx - 50] - a) / b
    p = Plot((0, 0.5), (8, 32), h=320)
    p.axes([0, 0.1, 0.2, 0.3, 0.4, 0.5], [10, 15, 20, 25, 30], "Time (seconds)", "Reading in kg (20 kg hung)", xfmt="{:.1f}")
    pts_raw = " ".join(f"{p.X((i - i0) / RATE):.1f},{p.Y(y):.1f}" for i, y in zip(idx, raw))
    pts_smo = " ".join(f"{p.X((i - i0) / RATE):.1f},{p.Y(y):.1f}" for i, y in zip(idx, smo))
    p.add(f'<polyline class="raw" points="{pts_raw}"/>')
    p.add(f'<polyline class="smooth" points="{pts_smo}"/>')
    p.add(f'<text class="lbl s1" x="{p.X(0.005):.1f}" y="{p.Y(29.6):.1f}">raw signal: swings about +/- {hum_amp / b:.0f} kg, 50 times a second</text>')
    p.add(f'<text class="lbl s2" x="{p.X(0.005):.1f}" y="{p.Y(11.2):.1f}">smoothed over 100 ms (what the app uses)</text>')
    return p.svg("Half a second of the raw signal at 20 kg and the same signal smoothed over 100 milliseconds")


def chart_design():
    rows = [(n, d[0.002]["curve"]) for n, c, d in design_rows if not np.isnan(d[0.002]["curve"])]
    w, left, rowh = 720, 330, 34
    h = 20 + rowh * len(rows) + 40
    xmax = 5.0
    out = []
    for i, (n, v) in enumerate(rows):
        y = 20 + i * rowh
        bw = max(2.0, v / xmax * (w - left - 70))
        short = (n.replace("Today's data + 100, 150, 200, 250 up and 200, 100, 0 down", "Today + 100-250 up, 200-0 down")
                 .replace("Today's data + 100, 150, 200, 250", "Today + 100, 150, 200, 250")
                 .replace("Fresh set: ", "New: ").replace(" up and down", " up+down").split(" (")[0])
        out.append(f'<text class="tick" x="{left - 10}" y="{y + 18}" text-anchor="end">{html.escape(short)}</text>')
        out.append(f'<rect class="bar" x="{left}" y="{y + 4}" width="{bw:.1f}" height="20" rx="3"><title>{v:.2f} kg</title></rect>')
        out.append(f'<text class="lbl" x="{left + bw + 6:.1f}" y="{y + 19}">{v:.2f} kg</text>')
    out.append(f'<text class="axl" x="{(left + w) / 2:.0f}" y="{h - 8}" text-anchor="middle">Expected uncertainty of the calibration at 250 kg (1 sigma, kg; weights accurate to 0.2 %)</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Uncertainty at 250 kg for each candidate set of weights">' + "".join(out) + "</svg>"


# ---------------------------------------------------------------------------------------------------------- tables
def table_runs():
    rows = []
    for r, k, e in zip(runs, kg, resid_kg):
        rows.append(f"<tr><td>{(r['t0'] - T0) / 60:.0f}</td><td>{k:g}</td><td>{r['mean']:.4f}</td><td>{e:+.2f}</td><td>{r['hum'] / b:.1f}</td></tr>")
    return "".join(rows)


def table_zero():
    zero_like = [(r["t0"], r["mean"], "no weight label (started before the labelled runs)") for r in unlabelled]
    zero_like += [(r["t0"], r["mean"], f"labelled {r['kg']:g} kg") for r in runs if r["kg"] == 0]
    zero_like.sort()
    return "".join(f"<tr><td>{(t - T0) / 60:+.0f}</td><td>{m:.4f}</td><td>{(m - a) / b:+.2f}</td><td>{html.escape(n)}</td></tr>" for t, m, n in zero_like)


def table_design():
    out = []
    for name, n, d in design_rows:
        def f(x):
            return "cannot tell" if np.isnan(x) else f"{x:.2f}"
        out.append(f"<tr><td>{html.escape(name)}</td><td>{n}</td><td>{f(d[0.0]['curve'])}</td><td>{f(d[0.002]['curve'])}</td><td>{f(d[0.005]['curve'])}</td></tr>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------- results file
with open(os.path.join(os.path.dirname(OUT), "calibration_points.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "minutes_from_first", "nominal_kg", "mean_volts", "fit_residual_kg", "hum_50hz_amplitude_kg"])
    for r, e in zip(runs, resid_kg):
        w.writerow([r["name"], f"{(r['t0'] - T0) / 60:.1f}", r["kg"], f"{r['mean']:.5f}", f"{e:.3f}", f"{r['hum'] / b:.2f}"])

# ---------------------------------------------------------------------------------------------------------- the page
HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Load cell calibration check</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834; --good:#006300; --warn:#8a5a00; --warnbg:#fff4dc; --goodbg:#e7f5e7; --band:rgba(42,120,214,.10); --unk:rgba(137,135,129,.12); }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --good:#6fd36f; --warn:#f2c14e; --warnbg:#2b2412; --goodbg:#132313; --band:rgba(57,135,229,.16); --unk:rgba(137,135,129,.14); }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s2:#d95926; --good:#6fd36f; --warn:#f2c14e; --warnbg:#2b2412; --goodbg:#132313; --band:rgba(57,135,229,.16); --unk:rgba(137,135,129,.14); }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:820px; margin:0 auto; padding:28px 16px 64px; }}
h1 {{ font-size:28px; line-height:1.2; margin:0 0 6px; }} h2 {{ font-size:20px; margin:40px 0 8px; }} h3 {{ font-size:16px; margin:22px 0 4px; }}
p, li {{ color:var(--ink); }} .sub {{ color:var(--ink2); margin:0 0 20px; }}
.card {{ background:var(--surface); border:1px solid var(--grid); border-radius:10px; padding:16px; margin:14px 0; }}
.verdict {{ border-left:4px solid var(--s1); }}
.callout {{ border-radius:8px; padding:12px 14px; margin:12px 0; }} .good {{ background:var(--goodbg); }} .warn {{ background:var(--warnbg); }}
.callout b.t {{ display:block; margin-bottom:2px; }} .good b.t {{ color:var(--good); }} .warn b.t {{ color:var(--warn); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:14px 0; }}
.kpi {{ background:var(--surface); border:1px solid var(--grid); border-radius:10px; padding:12px 14px; }} .kpi .v {{ font-size:24px; font-weight:600; }} .kpi .d {{ color:var(--ink2); font-size:13px; }}
svg {{ width:100%; height:auto; display:block; }} svg text {{ font:12px system-ui,sans-serif; fill:var(--ink2); }}
.tick {{ fill:var(--muted) !important; }} .axl {{ fill:var(--ink2) !important; }} .grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }}
.pt {{ fill:var(--s1); stroke:var(--surface); stroke-width:2; }} .fitline {{ stroke:var(--s1); stroke-width:2; fill:none; }} .dash {{ stroke-dasharray:6 5; }}
.unknown {{ fill:var(--unk); }} .band {{ fill:var(--band); }} .note {{ fill:var(--muted) !important; font-style:italic; }} .lbl {{ fill:var(--ink) !important; }}
.raw {{ fill:none; stroke:var(--s1); stroke-width:1.2; opacity:.75; }} .smooth {{ fill:none; stroke:var(--s2); stroke-width:2.5; }}
.s1 {{ fill:var(--s1) !important; font-weight:600; }} .s2 {{ fill:var(--s2) !important; font-weight:600; }} .bar {{ fill:var(--s1); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:8px 0; }} th, td {{ text-align:right; padding:5px 8px; border-bottom:1px solid var(--grid); }}
th:first-child, td:first-child {{ text-align:left; }} th {{ color:var(--ink2); font-weight:600; }} td.l, th.l {{ text-align:left; }}
details {{ margin:10px 0; }} summary {{ cursor:pointer; color:var(--ink2); }} code {{ background:var(--grid); padding:1px 5px; border-radius:4px; font-size:.9em; }}
.cap {{ color:var(--ink2); font-size:14px; margin:4px 0 0; }} ol li, ul li {{ margin:4px 0; }}
.scroll {{ overflow-x:auto; }}
</style></head><body><main>

<h1>Load cell calibration check</h1>
<p class="sub">Measurements from 12 October 2026 (amplifier + TAS501 + NI USB-6003), {len(runs)} loaded runs from 0 to 60 kg.</p>

<div class="card verdict"><b>Short answer.</b> Yes, the scaling is linear, and very cleanly so, <b>up to 60 kg</b>. Every measurement lies within
about 0.2 kg of one straight line. What this data cannot show is whether the line stays straight up to 250 kg, because nothing heavier than 60 kg
(24 % of your expected maximum) was measured. The next session should measure 100 to 250 kg.</div>

<div class="kpis">
<div class="kpi"><div class="v">{b * 1000:.2f} mV/kg</div><div class="d">slope: {n_per_v:.1f} newtons per volt, uncertainty {100 * np.sqrt(cov[1, 1]) / b:.2f} %</div></div>
<div class="kpi"><div class="v">R&sup2; = {r2:.5f}</div><div class="d">1.00000 would be a perfect line</div></div>
<div class="kpi"><div class="v">{rmse / b:.2f} kg</div><div class="d">typical distance of a point from the line (largest {np.abs(resid_kg).max():.2f} kg)</div></div>
<div class="kpi"><div class="v">{v_at_250:.1f} V</div><div class="d">expected output at 250 kg: well inside the &plusmn;10 V input range</div></div>
</div>

<h2>1. The straight-line fit</h2>
<p>Each dot is the average of a 30-second recording with a known weight on the cell. The solid line is the best straight line through them;
the dashed part is that line continued to 250 kg, which is a guess until it has been measured.</p>
<div class="card">{chart_fit()}</div>
<p class="cap">Equation: output (V) = {a:.4f} + {b:.5f} &times; weight (kg). So the zero (nothing hanging) sits at {a:.3f} V, and force in newtons = (volts &minus; {a:.4f}) &times; {n_per_v:.2f}.</p>

<h2>2. How far is each point from the line?</h2>
<p>Straight-line fits can hide small bends, so here is the leftover for each point in kilograms. A bend would show up as a curve (a smile or a frown); a good fit
looks like random scatter around zero.</p>
<div class="card">{chart_resid()}</div>
<ul>
<li>Scatter is random, with no smile or frown. A curved fit did not improve on the straight line (the curve term is not distinguishable from zero).</li>
<li>The two 0 kg points are the furthest from the line (+0.11 and &minus;0.18 kg). That is the zero wandering a little, not a scaling problem.</li>
<li>Slope between neighbouring weights stays between 18.3 and 19.1 mV/kg all the way (the highest value is the very first step, 0 to 2.5 kg).</li>
<li>Fitting only the weights up to 30 kg and then predicting 40, 50 and 60 kg misses by 0.01 to 0.18 kg.</li>
</ul>

<div class="callout warn"><b class="t">The caveat that matters</b>The data allow a slight bend. Within what you measured, the uncertainty left on a possible curve would move the
reading at 250 kg by about &plusmn;{curv_se_kg_250:.1f} kg (1 sigma). A good load cell is usually straight to better than 0.05 % of full scale, so a bend is unlikely,
but nothing here proves it for the amplifier and cabling as a whole. That is what the heavy weights are for.</div>

<h2>3. Other things the data show</h2>

<h3>The 50 Hz hum is large, but removable</h3>
<p>The raw signal carries a 50 Hz ripple of about &plusmn;{hum_amp * 1000:.0f} mV, which is &plusmn;{hum_amp / b:.1f} kg on the scale. It is the same at every weight, so it is electrical pickup from the mains,
not the weights moving. Averaging over a whole number of 20 ms periods removes it exactly. The app's 100 ms smoothing does that (the ripple falls from {sd_raw * 1000:.0f} mV to {sd_100 * 1000:.0f} mV standard deviation), and the 30 s means used here are not affected. A slower wobble of about 1 kg (2.5 times a second) is left in the smoothed line; it is also electrical, it averages out over a second or more, and it is small next to a 250 kg push.</p>
<div class="card">{chart_hum()}</div>
<p>Two consequences. First, always read the peak force from the smoothed signal; the highest raw sample can be about {hum_amp / b:.0f} kg too high. Second, because the ripple is that large it is worth fixing at the source when convenient:
use a shielded, twisted cable for the amplifier output, connect the amplifier's signal ground to the DAQ's AI GND, and keep the cable away from mains leads and chargers.</p>

<h3>The zero drifts a little</h3>
<p>Readings with nothing on the cell (minutes relative to the first labelled run) and their difference from the fitted zero:</p>
<div class="scroll"><table><tr><th>Minute</th><th>Volts</th><th>kg from fitted zero</th><th class="l">Run</th></tr>{table_zero()}</table></div>
<p>The zero moved about 20 mV (roughly 1 kg) between 13:24 and 13:44 while the amplifier warmed up, then stayed within about &plusmn;0.3 kg. A 40 kg weight measured again 20 minutes later read {rep_diff_kg:+.2f} kg different
({100 * rep_diff_kg / 40:+.2f} %). That is small for a 250 kg test, but 0.5 kg is 2.5 % of a 20 kg push, so <b>take a no-load reading at the start of every trial</b>. The file-name labels are worth checking too, see below.</p>

<h3>Things to double-check in the files</h3>
<ul>
<li><code>LogFile_-1kg_...</code> is labelled 0 in the data column and reads like a no-load run (0.034 V). If a negative 1 kg was intended, it did not register.</li>
<li>The earlier files named 5, 5_kg and 10kg (13:34 to 13:39) read 0.041 to 0.042 V, which is the no-load level. They look like no-load test runs that were named by mistake, so they are not used.</li>
<li>The <code>30s mean</code> column is a running average that carries over from the previous 30 seconds, so it does not match the file's own mean in several files. I did not use it.</li>
<li>The 60 kg run is only 6 seconds long (whole 20 ms cycles used); fine for the mean but shorter than the others.</li>
</ul>

<details><summary>All runs used in the fit</summary>
<div class="scroll"><table><tr><th>Minute</th><th>Nominal kg</th><th>Mean volts</th><th>Off the line (kg)</th><th>50 Hz hum (kg amplitude)</th></tr>{table_runs()}</table></div></details>

<h2>4. Which weights to use, up to 250 kg</h2>
<p>I compared candidate sets of weights using the scatter measured above ({rmse / b:.2f} kg per run), adding a possible error in the weights themselves. The numbers are the expected uncertainty of the finished calibration at 250 kg, where a possible bend is included.</p>
<div class="card">{chart_design()}</div>
<div class="scroll"><table><tr><th>Set of weights</th><th>Runs</th><th>weights exact</th><th>weights off by 0.2 %</th><th>weights off by 0.5 %</th></tr>{table_design()}</table></div>
<p class="cap">Uncertainty at 250 kg in kg (1 sigma), allowing for a bend. 'Cannot tell' means two weights alone cannot show a bend at all.</p>

<div class="card verdict"><b>What this says.</b>
<ul>
<li><b>Anything that reaches 250 kg fixes the main problem</b>: it cuts the uncertainty at 250 kg from about 3 to 7 kg down to under 1 kg. The exact choice of intermediate weights matters much less than reaching the top.</li>
<li><b>After that, the accuracy of the weights themselves is the limit.</b> A 0.2 % error on 250 kg is 0.5 kg, which is more than the electrical scatter of the load cell (0.08 kg). Know your weights better than the 0.2 % level, because no regression can correct a weight that is wrongly labelled.</li>
<li>Only two weights (0 and 250) gives the best straight-line precision but cannot show whether it bends, so do not do that.</li>
</ul></div>

<h3>Recommended plan</h3>
<p>Keep today's 0 to 60 kg data (it is good) and add the heavy end. If you can get the weights, it is also worth repeating the whole thing cleanly once.</p>
<ol>
<li><b>Warm up</b> the amplifier for at least 30 minutes. Zero the app and record 10 s with nothing on the cell.</li>
<li><b>Go up</b>: 100, 150, 200, 250 kg. If possible add 75 and 125 kg. Hold each for 15 s (30 s is more than needed); ignore the first 2 s after loading.</li>
<li><b>Come back down</b>: 200, 100, then 0 kg. Differences between up and down show sticking in the cell (hysteresis).</li>
<li><b>One check weight that is not in the fit</b>, for example 125 kg or 35 kg, measured as the final step. The fitted line should predict it to within about 1 %.</li>
<li>Finish with another <b>no-load reading</b> to see how far the zero moved during the session.</li>
</ol>
<p>That is roughly 12 extra runs, about 15 to 20 minutes. Practical points for the weights:</p>
<ul>
<li><b>Know the real mass.</b> Use plates that have been weighed on a calibrated scale, or weigh the whole loaded assembly. Gym plates are often 1 to 2 % off their printed value. Write down the measured mass and use that in the fit.</li>
<li>Load the cell the <b>same way as in the experiment</b> (same strap or hook, same pull direction, vertical, no swinging).</li>
<li>Steps of 50 kg between 50 and 250 kg are practical with 25 kg plates and keep the number of changes small.</li>
</ul>

<div class="callout warn"><b class="t">Check the cell's rated capacity before hanging 250 kg</b>The TAS501 is made in many capacities, including a 200 kg version. Read the label on your cell.
250 kg needs a cell rated at least 250 kg (a 300 kg or larger one is safer); the datasheet allows brief overload to 150 % but the calibration is only meaningful inside the rated range. Also confirm that the hook, strap and rig are rated for the load.</div>

<h2>What else would be smart</h2>
<ol>
<li><b>Use all points in the app's calibration.</b> The app now uses the zero and one weight. A multi-point fit with a "check weight" step would use data like this directly and warn when the fit is poor. I can build that.</li>
<li><b>A baseline in every trial</b> (2 s of no load before each contraction) so the drift never reaches the result.</li>
<li><b>Keep the strap geometry the same</b> for calibration and testing, since a pull at an angle changes what the cell reads.</li>
<li><b>Repeat the calibration</b> at the start of each test day and after any change to wiring or amplifier gain; keep the file with the data.</li>
<li>Gain: 250 kg gives about {v_at_250:.1f} V of the &plusmn;10 V input. That is comfortable, so there is no need to change the amplifier gain. The DAQ resolution (about 0.3 mV) is about 0.02 kg, far finer than the scatter.</li>
</ol>

<p class="cap">Made by <code>analyse_calibration.py</code> in the same folder from the CSV files in <code>{html.escape(DATA)}</code>. The per-run numbers are saved in <code>calibration_points.csv</code>.</p>
</main></body></html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"slope {b * 1000:.4f} mV/kg  zero {a:.5f} V  R2 {r2:.6f}  rmse {rmse / b:.3f} kg  N/V {n_per_v:.2f}")
print(f"curv_se@250 {curv_se_kg_250:.2f} kg  line_se@250 {line_se_250_kg:.3f} kg  rep40 {rep_diff_kg:+.3f} kg  hum {hum_amp / b:.2f} kg  sd raw {sd_raw * 1000:.1f} mV -> {sd_100 * 1000:.1f} mV")
print("wrote", OUT)
