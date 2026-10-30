"""Noise analysis of the load-cell (ADC) signal in saved trials: one CSV row per trial, one per 1 s window, amplitude spectra and the operator's
force/zero events. Standalone (numpy only, no limbtrack import). Reads recordings; changes nothing.

With --anonymise the output holds NO participant code and NO force level (no peak, no rise, no force-over-time curve): only noise measures, so it can
be shared. The code <-> session map is then written to --private-map and must never be committed.

Example (PowerShell):
  python noise_export.py --source A=path\\to\\trials_A --source B=path\\to\\trials_B ^
      --session-log project=path\\to\\session_log.csv --anonymise --private-map ..\\private_map.csv --out out_dir
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

LINES = (50, 100, 150, 200, 250, 300, 400)    # Hz: mains and the harmonics that matter
MAINS_HZ = 50.0

# columns that carry a force level or a participant's strength: left out of an anonymised export
DROP_INDEX = {"rise_from_first_V", "rise_from_last_V", "rise_from_first_N", "rise_from_last_N", "level_highest_1s_median_V",
              "meta_peak_n", "peak_smoothed_20ms_V", "peak_raw_sample_V", "volts_max"}
DROP_WINDOWS = {"level_median_V"}


def program_mains_amplitude(x, rate, mains_hz=MAINS_HZ):
    """Exactly limbtrack.force.mains_amplitude: lock-in at the exact mains frequency over a whole number of cycles (volts, peak)."""
    x = np.asarray(x, dtype=np.float64)
    cycles = int(x.size * mains_hz / rate)
    if cycles < 5:
        return math.nan
    n = int(round(cycles * rate / mains_hz))
    seg = x[:n] - x[:n].mean()
    t = np.arange(n) / rate
    return float(2.0 * abs(np.mean(seg * np.exp(-2j * np.pi * mains_hz * t))))


def fit_lines(t, x, freqs):
    """Least-squares fit of a constant, a linear trend and sine+cosine at each frequency. Returns (amplitudes, residual)."""
    cols = [np.ones_like(t), t - t.mean()]
    for f in freqs:
        cols += [np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)]
    A = np.stack(cols, axis=1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return [float(np.hypot(c[2 + 2 * k], c[3 + 2 * k])) for k in range(len(freqs))], x - A @ c


def load_force(path):
    t, v = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_s"])); v.append(float(r["volts"]))
    return np.array(t), np.array(v)


def analyse(dataset, meta_path):
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    fp = meta_path.with_name(meta_path.name.replace("_meta.json", "_force.csv"))
    if not fp.exists():
        return None
    t, v = load_force(fp)
    if len(v) < 2000:
        return None
    fs = 1.0 / float(np.median(np.diff(t)))
    n = int(round(fs))
    tid = re.sub(r"_\d{8}-\d{6}_meta\.json$", "", meta_path.name)
    force = m.get("force", {}) or {}
    cal = force.get("calibration", {}) or {}
    dt = np.diff(t)
    nz = np.unique(np.round(np.abs(np.diff(v)), 9)); nz = nz[nz > 0]
    wins = []
    for i in range(0, len(v) - n + 1, n // 2):
        tt, s = t[i:i + n], v[i:i + n]
        amps, resid_all = fit_lines(tt, s, LINES)
        _, r5010 = fit_lines(tt, s, (50, 100))
        hum = program_mains_amplitude(s, fs)
        hum = hum if math.isfinite(hum) else 0.0
        wob = math.sqrt(max(float(s.std()) ** 2 - hum ** 2 / 2.0, 0.0))        # the program's own 'wobble' (force.py _steady_mean)
        wins.append(dict(i=i, t_start=float(tt[0]), level=float(np.median(s)), std=float(s.std()), amps=amps, resid=float(resid_all.std()),
                         wobble_prog=wob, wobble_no50_100=float(r5010.std())))
    resid = np.array([w["resid"] for w in wins])
    quiet = resid <= np.median(resid)                        # the quietest half: windows with no push in progress
    for w, q in zip(wins, quiet):
        w["quiet"] = int(q)
    qi = [w for w in wins if w["quiet"]]

    def med(key):
        return float(np.median([w[key] for w in qi]))
    amps_med = [float(np.median([w["amps"][k] for w in qi])) for k in range(len(LINES))]
    hann = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    for w in qi:
        seg = v[w["i"]:w["i"] + n]
        a = np.abs(np.fft.rfft((seg - seg.mean()) * hann)) * 2 / hann.sum()
        acc += a ** 2
    spectrum_mv = np.sqrt(acc / len(qi)) * 1000
    f50 = f100 = math.nan
    if t[-1] - t[0] >= 4.0:
        N = 1 << 20
        sp = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v)), N)); fr = np.fft.rfftfreq(N, 1 / fs)
        mk = (fr > 48) & (fr < 52); f50 = float(fr[mk][np.argmax(sp[mk])])
        mk = (fr > 98) & (fr < 102); f100 = float(fr[mk][np.argmax(sp[mk])])
    meds = np.array([w["level"] for w in wins])
    first, last, top = float(np.median(v[:n])), float(np.median(v[-n:])), float(meds.max())
    k20 = np.ones(max(int(round(0.020 * fs)), 1)) / max(int(round(0.020 * fs)), 1)
    sm20 = np.convolve(v, k20, mode="same")
    prog_hum = program_mains_amplitude(v, fs)
    n_per_v = cal.get("n_per_v")
    rise_first = top - first
    row = dict(
        trial_id=tid, dataset=dataset, session_date=m["started_at"][:10], start_time=m["started_at"][11:19], software_version=m.get("software_version"),
        test=m.get("test_key"), target_deg=m.get("target_deg"), rest_only_trial=int(rise_first < 0.05),
        n_samples=len(v), duration_s=round(float(t[-1] - t[0]), 3), sample_rate_hz=round(fs, 3), meta_measured_rate_hz=force.get("measured_rate_hz"),
        steps_over_1p5ms=int((dt > 1.5 * np.median(dt)).sum()), longest_step_ms=round(float(dt.max()) * 1000, 2),
        adc_device=force.get("device"), adc_channel=force.get("channel"), adc_terminal=force.get("terminal"), adc_range_v=force.get("range_v"),
        smallest_step_mV=round(float(nz.min()) * 1000, 4) if len(nz) else "", volts_min=round(float(v.min()), 5), volts_max=round(float(v.max()), 5),
        calibrated=force.get("calibrated"), zero_set=cal.get("zero_set"), n_per_v=n_per_v, zero_v=cal.get("zero_v"), regression_id=cal.get("regression_id"),
        meta_zero_noise_mV=None if cal.get("zero_noise_v") is None else round(cal["zero_noise_v"] * 1000, 3),
        meta_zero_hum_mV=None if cal.get("zero_hum_v") is None else round(cal["zero_hum_v"] * 1000, 3),
        meta_peak_n=force.get("peak_n"),
        quiet_windows=int(quiet.sum()), windows=len(wins),
        **{f"amp_{f}Hz_mV": round(a * 1000, 3) for f, a in zip(LINES, amps_med)},
        mains_freq_hz=round(f50, 4) if math.isfinite(f50) else "", second_harmonic_freq_hz=round(f100, 4) if math.isfinite(f100) else "",
        std_quiet_mV=round(med("std") * 1000, 3), wobble_program_rule_mV=round(med("wobble_prog") * 1000, 3),
        wobble_without_50_100_mV=round(med("wobble_no50_100") * 1000, 3), residual_after_all_lines_mV=round(med("resid") * 1000, 3),
        program_hum_wholetrial_lockin_mV=round(prog_hum * 1000, 3) if math.isfinite(prog_hum) else "",
        level_first_1s_V=round(first, 5), level_last_1s_V=round(last, 5), level_highest_1s_median_V=round(top, 5),
        baseline_first_minus_last_mV=round((first - last) * 1000, 2), rise_from_first_V=round(rise_first, 5), rise_from_last_V=round(top - last, 5),
        rise_from_first_N=None if n_per_v is None else round(rise_first * n_per_v, 1), rise_from_last_N=None if n_per_v is None else round((top - last) * n_per_v, 1),
        peak_smoothed_20ms_V=round(float(sm20.max()), 5), peak_raw_sample_V=round(float(v.max()), 5), raw_minus_smoothed_peak_mV=round(float(v.max() - sm20.max()) * 1000, 2),
        hum_regime=("clean" if amps_med[0] * 1000 < 10 and amps_med[1] * 1000 < 2 else "50Hz_only" if amps_med[1] * 1000 < 2 else "50+100Hz"),
        passes_program_zero_rule=int(med("wobble_prog") <= 0.030), passes_zero_rule_if_harmonics_removed=int(med("wobble_no50_100") <= 0.030),
        hum_pct_of_rise_true=round(100 * amps_med[0] / rise_first, 2) if rise_first >= 0.2 else "",
        hum_pct_of_rise_program_method=round(100 * prog_hum / rise_first, 2) if (rise_first >= 0.2 and math.isfinite(prog_hum)) else "",
    )
    window_rows = [dict(trial_id=tid, t_start_s=round(w["t_start"], 3), quiet=w["quiet"], level_median_V=round(w["level"], 5), std_mV=round(w["std"] * 1000, 3),
                        **{f"amp_{f}Hz_mV": round(a * 1000, 3) for f, a in zip(LINES, w["amps"])}, residual_after_lines_mV=round(w["resid"] * 1000, 3),
                        wobble_program_rule_mV=round(w["wobble_prog"] * 1000, 3), wobble_without_50_100_mV=round(w["wobble_no50_100"] * 1000, 3)) for w in wins]
    return row, window_rows, spectrum_mv


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def anonymise_id(tid, session):
    """'Pxx_02_knee_extension_70deg_T1' -> 'S1_pos02_knee_extension_70deg_T1' (the participant code is dropped)."""
    mt = re.match(r"^P\d+_(\d+)_(.+)_T(\d+)$", tid)
    if not mt:
        raise ValueError(f"unexpected trial name: {tid}")
    return f"{session}_pos{mt.group(1)}_{mt.group(2)}_T{mt.group(3)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, metavar="LABEL=FOLDER", help="a folder of trials (*_force.csv + *_meta.json); repeatable")
    ap.add_argument("--session-log", action="append", default=[], metavar="LABEL=FILE", help="a session_log.csv; repeatable")
    ap.add_argument("--since", default="", help="session-log rows from this date (YYYY-MM-DD) on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--anonymise", action="store_true", help="no participant codes, no force levels (see the module text)")
    ap.add_argument("--private-map", help="with --anonymise: where to write the session <-> label map (keep it private)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = [s.split("=", 1) for s in a.source]
    sess = {label: f"S{k + 1}" for k, (label, _) in enumerate(sources)}
    index, windows, spectra = [], [], {}
    for label, folder in sources:
        metas = sorted(Path(folder).glob("*_meta.json"), key=lambda p: (re.sub(r"_T\d+_.*", "", p.name), int(re.search(r"_T(\d+)_", p.name).group(1))))
        k = 0
        for mp in metas:
            res = analyse(label, mp)
            if res is None:
                print("skipped (no force file or too short):", mp.name)
                continue
            row, wrows, spec = res
            if a.anonymise:
                new = anonymise_id(row["trial_id"], sess[label])
                for r in [row] + wrows:
                    r["trial_id"] = new
                row["dataset"] = sess[label]
            index.append(row); windows += wrows; spectra[row["trial_id"]] = spec
            k += 1
        print(f"{label} ({sess[label]}): {k} trials")
    ids = [r["trial_id"] for r in index]
    assert len(ids) == len(set(ids)), "trial ids are not unique"
    idx_fields = [c for c in index[0].keys() if not (a.anonymise and c in DROP_INDEX)]
    win_fields = [c for c in windows[0].keys() if not (a.anonymise and c in DROP_WINDOWS)]
    write_csv(out / "noise_index.csv", index, idx_fields)
    write_csv(out / "noise_windows.csv", windows, win_fields)
    n_bins = min(len(s) for s in spectra.values())
    with open(out / "noise_spectra.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["freq_hz"] + ids)
        for k in range(n_bins):
            w.writerow([k] + [round(float(spectra[i][k]), 4) for i in ids])
    ev = []
    keep = re.compile(r"zero|relaxed|force|calibrat|usb|daq|recording (started|stopped)$", re.I)
    for spec in a.session_log:
        label, path = spec.split("=", 1)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if a.since and r["time"][:10] < a.since:
                    continue
                text = r["event"] + " " + (r.get("detail") or "")
                if keep.search(text) and "trial warning" not in r["event"]:
                    detail = (r.get("detail") or "")[:400]
                    if a.anonymise:
                        detail = re.sub(r"\bP\d{2,3}\b", "[code]", detail)
                    ev.append(dict(source=label, time=r["time"], seconds_since_start=r["seconds_since_start"], event=r["event"], detail=detail))
    if ev:
        write_csv(out / "session_force_events.csv", ev, ["source", "time", "seconds_since_start", "event", "detail"])
    if a.anonymise and a.private_map:
        write_csv(Path(a.private_map), [dict(session=sess[l], source_label=l, folder=f) for l, f in sources])
    print(f"index {len(index)} trials, {len(windows)} windows, {n_bins} spectrum bins, {len(ev)} events -> {out}")


if __name__ == "__main__":
    sys.exit(main())
