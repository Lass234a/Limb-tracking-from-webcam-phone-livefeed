"""Builds calibration_report_v2.html (session 2: 0-250 kg up and down) from calib2_core.analyse().  Needs only numpy."""
import html
import os

import numpy as np

import calib2_core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_report_v2.html")
R = C.analyse()
a, b = R["a"], R["b"]
P, V = R["P"], R["V"]
phase = [u["phase"] for u in R["used"]]
d1 = R["d1"]
pool = R["pool"]
T0 = R["runs"][0]["t0"]


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
        return f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{html.escape(label)}">' + "".join(self.parts) + "</svg>"


def legend(p, items, x, y):
    for i, (cls, text) in enumerate(items):
        yy = y + i * 18
        p.add(f'<circle class="{cls}" cx="{x}" cy="{yy}" r="5"/>')
        p.add(f'<text class="lbl" x="{x + 12}" y="{yy + 4}">{html.escape(text)}</text>')


def chart_fit():
    p = Plot((0, 265), (0, 5.2))
    p.axes(range(0, 251, 50), [0, 1, 2, 3, 4, 5], "Plates on the cell (kg)", "Amplifier output (V)", yfmt="{:.0f}")
    p.add(f'<line class="fitline" x1="{p.X(0):.1f}" y1="{p.Y(a):.1f}" x2="{p.X(250):.1f}" y2="{p.Y(a + b * 250):.1f}"/>')
    for k, v, ph in zip(P, V, phase):
        cls = "pt" if ph in ("up", "peak") else "pt2"
        p.add(f'<circle class="{cls}" cx="{p.X(k):.1f}" cy="{p.Y(v):.1f}" r="4.5"><title>{k:g} kg ({ph}): {v:.4f} V</title></circle>')
    legend(p, [("pt", "going up"), ("pt2", "coming down")], p.X(8), p.Y(4.8))
    p.add(f'<text class="lbl" x="{p.X(250) - 8:.1f}" y="{p.Y(a + b * 250) - 12:.1f}" text-anchor="end">{a + b * 250:.2f} V at 250 kg</text>')
    return p.svg("Amplifier output against plate mass, 0 to 250 kg, going up and coming down, with the straight-line fit")


def chart_resid():
    p = Plot((-8, 258), (-0.5, 0.5), h=320)
    p.add(f'<rect class="band" x="{p.X(-8):.1f}" y="{p.Y(0.25):.1f}" width="{p.X(258) - p.X(-8):.1f}" height="{p.Y(-0.25) - p.Y(0.25):.1f}"/>')
    p.axes(range(0, 251, 50), [-0.4, -0.2, 0, 0.2, 0.4], "Plates on the cell (kg)", "Difference from the one line (kg)", yfmt="{:+.1f}")
    p.add(f'<text class="note" x="{p.X(256):.1f}" y="{p.Y(0.25) - 6:.1f}" text-anchor="end">within +/- 0.25 kg (0.1 % of 250 kg)</text>')
    for k, e, ph in zip(P, R["resid_kg"], phase):
        cls = "pt" if ph in ("up", "peak") else "pt2"
        p.add(f'<circle class="{cls}" cx="{p.X(k):.1f}" cy="{p.Y(e):.1f}" r="4.5"><title>{k:g} kg ({ph}): {e:+.3f} kg</title></circle>')
    legend(p, [("pt", "going up"), ("pt2", "coming down")], p.X(150), p.Y(-0.33))
    return p.svg("Difference of each measurement from the fitted line in kilograms, going up and coming down")


def clock(t):
    import datetime
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M")


def table_files():
    rows = []
    for r in R["runs"]:
        kind = {"ZERO": "zero (no chain, no strap)", "HW": "chain + strap only", "PLATES": "plates", "DROP": "not used"}[r["kind"]]
        use = "" if r["kind"] != "PLATES" else f"{r['plates']:g} kg"
        flag = ""
        if r["why"]:
            flag = f'<b>Fixed:</b> {html.escape(r["why"])}' if r["kind"] != "DROP" else f'<b>Left out:</b> {html.escape(r["why"])}'
        elif r["hum"] > 0.02:
            flag = "laptop on battery: 50 Hz hum, average unaffected"
        rows.append(f"<tr><td>{clock(r['t0'])}</td><td class='l'>{html.escape(r['stem'].split('_13-10')[0])}</td><td class='l'>{kind}</td><td>{use}</td><td>{r['mean']:.4f}</td><td class='l'>{flag}</td></tr>")
    return "".join(rows)


def table_pairs():
    return "".join(f"<tr><td>{k:g}</td><td>{u:.4f}</td><td>{d:.4f}</td><td>{1000 * (d - u):+.1f}</td><td>{(d - u) / b:+.2f}</td></tr>" for k, u, d in R["pairs"])


def table_zero():
    rows = [f"<tr><td>{clock(t)}</td><td class='l'>no chain, no strap</td><td>{v:.4f}</td><td>{(v - R['zeros'][0][1]) / b:+.2f}</td></tr>" for t, v in R["zeros"]]
    rows += [f"<tr><td>{clock(t)}</td><td class='l'>chain + strap</td><td>{v:.4f}</td><td>{(v - R['hws'][0][1]) / b:+.2f}</td></tr>" for t, v in R["hws"]]
    rows.sort()
    return "".join(rows)


MIS = next(r for r in R["runs"] if r["stem"].startswith("0kg_13-10-13-16-11"))
MIS_V = MIS["mean"]
MIS_GAP_MV = 1000 * (MIS_V - R["zeros"][-1][1])
MIS_GAP_KG = (MIS_V - R["zeros"][-1][1]) / b
hum_off = [r for r in R["runs"] if r["kind"] in ("PLATES",) and r["hum"] > 0.02]
hum_on = [r for r in R["runs"] if r["kind"] in ("PLATES", "HW", "ZERO") and r["hum"] <= 0.02]
hum_on_max = max(r["hum"] for r in hum_on) * 1000
hum_off_vals = ", ".join(f"{r['hum'] * 1000:.0f} mV ({r['plates']:g} kg, {clock(r['t0'])})" for r in hum_off)

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Load cell calibration v2</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834; --good:#006300; --warn:#8a5a00; --warnbg:#fff4dc; --goodbg:#e7f5e7; --band:rgba(42,120,214,.10); }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --good:#6fd36f; --warn:#f2c14e; --warnbg:#2b2412; --goodbg:#132313; --band:rgba(57,135,229,.16); }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s2:#d95926; --good:#6fd36f; --warn:#f2c14e; --warnbg:#2b2412; --goodbg:#132313; --band:rgba(57,135,229,.16); }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:820px; margin:0 auto; padding:28px 16px 64px; }}
h1 {{ font-size:28px; line-height:1.2; margin:0 0 6px; }} h2 {{ font-size:20px; margin:40px 0 8px; }} h3 {{ font-size:16px; margin:22px 0 4px; }}
.sub {{ color:var(--ink2); margin:0 0 20px; }}
.card {{ background:var(--surface); border:1px solid var(--grid); border-radius:10px; padding:16px; margin:14px 0; }} .verdict {{ border-left:4px solid var(--s1); }}
.callout {{ border-radius:8px; padding:12px 14px; margin:12px 0; }} .good {{ background:var(--goodbg); }} .warn {{ background:var(--warnbg); }}
.callout b.t {{ display:block; margin-bottom:2px; }} .good b.t {{ color:var(--good); }} .warn b.t {{ color:var(--warn); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:14px 0; }}
.kpi {{ background:var(--surface); border:1px solid var(--grid); border-radius:10px; padding:12px 14px; }} .kpi .v {{ font-size:24px; font-weight:600; }} .kpi .d {{ color:var(--ink2); font-size:13px; }}
svg {{ width:100%; height:auto; display:block; }} svg text {{ font:12px system-ui,sans-serif; fill:var(--ink2); }}
.tick {{ fill:var(--muted) !important; }} .axl {{ fill:var(--ink2) !important; }} .grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }}
.pt {{ fill:var(--s1); stroke:var(--surface); stroke-width:2; }} .pt2 {{ fill:var(--s2); stroke:var(--surface); stroke-width:2; }} .fitline {{ stroke:var(--ink2); stroke-width:1.5; fill:none; }}
.band {{ fill:var(--band); }} .note {{ fill:var(--muted) !important; font-style:italic; }} .lbl {{ fill:var(--ink) !important; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:8px 0; }} th, td {{ text-align:right; padding:5px 8px; border-bottom:1px solid var(--grid); vertical-align:top; }}
th:first-child, td:first-child {{ text-align:left; }} th {{ color:var(--ink2); font-weight:600; }} td.l, th.l {{ text-align:left; }}
details {{ margin:10px 0; }} summary {{ cursor:pointer; color:var(--ink2); }} code {{ background:var(--grid); padding:1px 5px; border-radius:4px; font-size:.9em; }}
.cap {{ color:var(--ink2); font-size:14px; margin:4px 0 0; }} ol li, ul li {{ margin:4px 0; }} .scroll {{ overflow-x:auto; }}
</style></head><body><main>

<h1>Load cell calibration, session 2</h1>
<p class="sub">13 October 2026, ceiling rig with chain and strap, plates 0 to 250 kg going up and coming down. {R['n_used']} steady runs used.</p>

<div class="card verdict"><b>Short answer.</b> The load cell is linear all the way to 250 kg. One straight line fits all {R['n_used']} runs to within
{np.abs(R['resid_kg']).max():.2f} kg (0.1 % of 250 kg), and a possible bend is now ruled out: it is {R['quad_kg250']:+.2f} &plusmn; {R['quad_kg250_se']:.2f} kg at 250 kg, not different from zero.
Yesterday's data should <b>not</b> be mixed in. It would change the slope by {abs(R['pool_slope_change_pct']):.2f} %, which is nothing, and it was measured with a different rig and
a lot of hum. Keep it as an independent check: its slope agrees with today's to {abs(R['slope_diff_pct']):.2f} %.</div>

<div class="kpis">
<div class="kpi"><div class="v">{R['n_per_v']:.2f} N/V</div><div class="d">{R['kg_per_v']:.3f} kg per volt, or {b * 1000:.3f} mV per kg. Uncertainty of the slope {100 * R['se_b'] / b:.3f} %</div></div>
<div class="kpi"><div class="v">R&sup2; = {R['r2']:.6f}</div><div class="d">{R['rmse_kg']:.2f} kg typical distance from the line, {R['rmse_updown_kg']:.2f} kg once the up/down step is allowed for</div></div>
<div class="kpi"><div class="v">{R['down_offset_kg']:+.2f} kg</div><div class="d">readings on the way down sit this much above the way up (see section 3)</div></div>
<div class="kpi"><div class="v">{R['hw_mass_kg']:.2f} kg</div><div class="d">chain and strap, as measured by the cell itself</div></div>
</div>

<h2>1. Which files were corrected or left out</h2>
<p>The file names and the label column inside the files disagree in a few places, so I classified every run by its voltage as well. The corrections below follow what you told me
(a file named 0 kg that actually has the chain on is a "1 kg" run) and every one is backed by the numbers.</p>
<ul>
<li><b>Named 0 kg at 16:11 is really chain + strap.</b> Its own label says 1 and its voltage ({MIS_V:.4f} V) equals the 1 kg run a minute earlier ({R['hws'][1][1]:.4f} V), and is {MIS_GAP_MV:.1f} mV ({MIS_GAP_KG:.2f} kg) above the true 0 kg run a minute later.</li>
<li><b>Named 0 kg at 15:14 is the start of the 50 kg run</b> (6 s long, voltage at the 50 kg level). Left out.</li>
<li><b>175 kg at 15:39</b> has 150 in its label column but the voltage equals the other 175 kg runs, so it is 175 kg.</li>
<li><b>Left out:</b> the "lasse" file (the load goes on and off inside it) and the 8 s long 200 kg file (disturbed start; the next file repeats it).</li>
</ul>
<details><summary>Every file and how it was treated</summary>
<div class="scroll"><table><tr><th>Time</th><th class="l">File</th><th class="l">What it is</th><th>Plates</th><th>Volts</th><th class="l">Note</th></tr>{table_files()}</table></div></details>

<h2>2. The fit</h2>
<p>Each dot is the average of 24 s of a run (the first 6 s are dropped while the load settles). All runs with the chain and strap on count, including the chain-only runs as 0 kg of plates.
The line is the best straight line through all of them.</p>
<div class="card">{chart_fit()}</div>
<p class="cap">Output (V) = {a:.5f} + {b:.6f} &times; plates (kg). The {a:.4f} V is the chain-and-strap zero. Force in newtons = (volts &minus; zero) &times; {R['n_per_v']:.2f}.</p>

<h2>3. How straight is it, and what is the up/down step?</h2>
<p>The dots below are the leftover distance from the line. A bend would show as an arch or a bowl. Instead the points split in two groups: runs on the way up sit a little below the line and runs on the way down a little above it.</p>
<div class="card">{chart_resid()}</div>
<ul>
<li><b>No bend:</b> the curve term is {R['quad_kg250']:+.2f} &plusmn; {R['quad_kg250_se']:.2f} kg at 250 kg. The straight line is all you need.</li>
<li><b>Going down reads {R['down_offset_kg']:.2f} kg (about {100 * R['down_offset_kg'] / 250:.2f} % of 250 kg) higher than going up</b>, at every load, even at zero load with the chain on
({R['pairs'][0][2] - R['pairs'][0][1]:+.4f} V, {(R['pairs'][0][2] - R['pairs'][0][1]) / b:+.2f} kg). Allowing for that one step, the scatter falls from {R['rmse_kg']:.2f} kg to {R['rmse_updown_kg']:.2f} kg.</li>
<li><b>The slope is the same both ways</b> ({R['slope_up'] * 1000:.3f} going up, {R['slope_down'] * 1000:.3f} mV/kg coming down, {abs(100 * (R['slope_down'] / R['slope_up'] - 1)):.2f} % apart). Only the zero moved, so it is a shift of the zero, not a change in sensitivity.</li>
<li>It cannot be said from one loop whether the step is the cell remembering the load (hysteresis), the zero drifting over time, or both. The two repeated zero runs show the zero rising {R['zero_drift_v'] * 1000:.1f} mV ({R['zero_drift_v'] / b:+.2f} kg) in {R['zero_minutes']:.0f} minutes, which accounts for about half of it.</li>
</ul>
<div class="scroll"><table><tr><th>Plates (kg)</th><th>Going up (V)</th><th>Coming down (V)</th><th>Difference (mV)</th><th>Difference (kg)</th></tr>{table_pairs()}</table></div>
<p class="cap">The 0 kg row is the chain-and-strap run at the start and the average of the two at the end.</p>
<div class="callout good"><b class="t">What it means for the experiment</b>Press <i>Zero</i> with the chain and strap hanging before every trial. That removes this step. Averaging up and down, as this fit does, puts the line in the middle, so the leftover bias is about &plusmn;{R['down_offset_kg'] / 2:.2f} kg at any load, even without a fresh zero.</div>

<h2>4. Zero and the chain</h2>
<div class="scroll"><table><tr><th>Time</th><th class="l">State</th><th>Volts</th><th>Change from first (kg)</th></tr>{table_zero()}</table></div>
<p>The chain and strap together read {R['hw_mass_kg']:.2f} kg ({R['hw_mass_kg_first']:.2f} kg early, {R['hw_mass_kg_last']:.2f} kg late). The chain-only runs land on the line fitted from 25 to 250 kg only to within
{R['hw_gap_kg']:+.2f} kg, which is what confirms that the "1 kg" files are the chain and strap alone, not a 1 kg weight on top of them. If you weigh the chain and strap on a scale and get about 1.1 kg, the
zero, the slope and the labelling are all consistent with each other.</p>

<h2>5. The 50 Hz hum</h2>
<p>With the laptop charging, the hum is between 1.0 and {hum_on_max:.1f} mV in every run, which is below 0.2 kg. With the laptop on battery it was {hum_off_vals}. The averages of those runs fall in line with all the others,
so the hum does not bias a 24 s average. It only matters for live readouts and for peaks, where the app's 100 ms smoothing removes it. Keep the laptop charging from the isolated outlet.</p>

<h2>6. Should yesterday's measurements be added?</h2>
<p>No, it would add noise more than information. In numbers:</p>
<div class="scroll"><table>
<tr><th class="l"></th><th>Slope (mV/kg)</th><th>Range</th><th>Runs</th></tr>
<tr><td class="l">Yesterday (rack rig, 0 to 60 kg, up only)</td><td>{d1['b'] * 1000:.3f} &plusmn; {d1['se_b'] * 1000:.3f}</td><td>0 to 60 kg</td><td>{d1['n']}</td></tr>
<tr><td class="l">Today going up only</td><td>{R['slope_up'] * 1000:.3f} &plusmn; {R['slope_up_se'] * 1000:.3f}</td><td>0 to 250 kg</td><td>13</td></tr>
<tr><td class="l">Today, all runs</td><td>{b * 1000:.3f} &plusmn; {R['se_b'] * 1000:.3f}</td><td>0 to 250 kg</td><td>{R['n_used']}</td></tr>
<tr><td class="l">Both days together (own zero for each day)</td><td>{pool['b'] * 1000:.3f} &plusmn; {pool['se_b'] * 1000:.3f}</td><td>0 to 250 kg</td><td>{pool['n']}</td></tr>
</table></div>
<ul>
<li><b>Yesterday's range is only a quarter of today's.</b> It carries {100 * R['info_day1_share']:.1f} % of the information about the slope. Pooling moves the slope by {R['pool_slope_change_pct']:+.3f} %, which is {R['pool_pred250_shift_kg']:+.2f} kg at 250 kg.</li>
<li><b>The slopes differ by {R['slope_diff_pct']:+.2f} % &plusmn; {R['slope_diff_se_pct']:.2f} %</b> (today higher). That is small but about twice its own uncertainty, a sign that yesterday had extra scatter: a different rig, a warming-up amplifier and 100 mV of hum.
Using yesterday's slope alone at 250 kg would read about {abs(R['slope_diff_pct']) * 2.5:.1f} kg low.</li>
<li><b>Different zero.</b> Yesterday's rack rig had a different zero and different hardware, so it needs its own intercept and cannot share one with today's data.</li>
<li><b>Useful as a check.</b> Two different sessions, rigs and hum levels giving the same sensitivity to {abs(R['slope_diff_pct']):.2f} % shows the amplifier gain is stable. If the plates were the same both days, this does not check the plate masses.</li>
</ul>

<h2>7. What to put in the app, and what is left</h2>
<div class="card verdict">
<b>Calibration to use:</b> {R['n_per_v']:.2f} newtons per volt ({R['kg_per_v']:.3f} kg per volt), with the zero taken with the chain and strap on (about {R['a']:.4f} V in this session; the app sets it itself when you press Zero).<br>
<b>Uncertainty from the electronics at 250 kg:</b> about &plusmn;{max(R['se250_kg'], R['down_offset_kg'] / 2):.2f} kg ({100 * max(R['se250_kg'], R['down_offset_kg'] / 2) / 250:.2f} %), or &plusmn;{R['se250_kg']:.2f} kg if you zero before every trial.
</div>
<ol>
<li><b>Weigh the plates.</b> The electronics are now known to about 0.1 %, so the real masses are the limit. Nothing here proves the plates are as heavy as their labels, and a 1 % error in the plates is a 1 % error in every force. Check a few on a calibrated scale, and weigh the chain and strap.</li>
<li><b>A second full loop</b> (up and down again) would show whether the +{R['down_offset_kg']:.2f} kg step repeats at the same size, which tells hysteresis from drift. A 10-minute no-load run with the chain on would give the pure drift.</li>
<li><b>Write the zero-with-rig rule into the SOP:</b> chain and strap on, laptop charging from the isolated outlet, Zero before every trial and a no-load check at the end.</li>
</ol>

<p class="cap">Made by <code>analyse_calibration_v2.py</code> and <code>calib2_core.py</code> from <code>{html.escape(C.DAY2)}</code> and, for the comparison, the session 1 files in <code>{html.escape(C.DAY1)}</code>.</p>
</main></body></html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", OUT)
