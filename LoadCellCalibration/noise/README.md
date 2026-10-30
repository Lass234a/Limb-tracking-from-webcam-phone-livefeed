# Load-cell / ADC noise analysis (anonymised) - start here

**Purpose.** Data and a script to analyse the noise on the load cell (dynamometer) signal and to build noise-reducing features in limbtrack.
Measured on 2026-10-30 in four sessions: **S1** a participant session (22 trials, several tests), **S2** a noise test (13 rest trials with
changed set-ups), **S3** and **S4** the pilot sessions of 28-29 Oct (64 trials each). The priority is **the force measurement**: the accuracy
of the force numbers matters more than anything else here.

**What is NOT here (on purpose).** The project rule is *participant data - never commit* (`.gitignore`), and this repository is public. So there are
**no raw signals**, **no participant codes**, and **no force levels** (no peak, no rise, no force-over-time curve); trials are named `S1_pos02_...`.
The map from `S1..S4` to participant codes and the raw files stay with the operator. Be aware that `hum_pct_of_rise_*` together with the hum amplitude
would let a reader back-calculate the force rise of one anonymous trial; there is no link to a person. If that is not acceptable, drop those two columns.

## 1. Files

| File | What it is |
|---|---|
| `noise_index.csv` | **One row per trial (163).** Noise measures at 50/100/150/200/250/300/400 Hz, mains frequency, the program's zero rule, calibration state, rest levels, hum regime. |
| `noise_windows.csv` | One row per 1 s window (3665): the line amplitudes, residual and the zero-rule wobble inside each window (time resolved). |
| `noise_spectra.csv` | Amplitude spectrum 0-500 Hz (1 Hz bins, mV) of each trial's quiet windows. Column = `trial_id`. |
| `session_force_events.csv` | Operator events about force: zero taken/refused with the program's reason, relaxed level, USB error, recording start/stop (participant codes replaced by `[code]`). |
| `noise_export.py` | Rebuilds all of it from saved trials (`*_force.csv` + `*_meta.json`), numpy only. `--anonymise` gives exactly the files above. |

Load: `idx = pd.read_csv("noise_index.csv")`. Column groups in `noise_index.csv`: identity (`trial_id`, `dataset` = session, date/time, test, target), recording
(`sample_rate_hz`, steps, ADC device/channel/terminal/range, `smallest_step_mV`), calibration (`calibrated`, `zero_set`, `n_per_v`, `zero_v`, `regression_id`, the program's own
`meta_zero_noise_mV`, `meta_zero_hum_mV`), noise (`amp_*Hz_mV`, `mains_freq_hz`, `std_quiet_mV`, `wobble_program_rule_mV`, `wobble_without_50_100_mV`,
`residual_after_all_lines_mV`, `program_hum_wholetrial_lockin_mV`, `raw_minus_smoothed_peak_mV`), levels (`level_first_1s_V`, `level_last_1s_V`, `baseline_first_minus_last_mV`) and
classification (`hum_regime`, `rest_only_trial`, `passes_program_zero_rule`, `passes_zero_rule_if_harmonics_removed`, `hum_pct_of_rise_true` / `_program_method`).

## 2. How the signal is recorded

- NI DAQmx, device `Dev1`, channel `ai0`, **differential**, range +-10 V, **1000 samples/s** (measured 999.9; 1.0 ms steps with a few 1.5-2.4 ms gaps). Nyquist 500 Hz.
- Smallest step in the signal about 0.32 mV. In the quietest windows about 2-3 mV rms remains after removing the hum lines (`residual_after_all_lines_mV` is larger whenever the force changes).
- Volts -> newtons: `N = (volts - zero_v) * n_per_v`, `n_per_v` = 526.95 N/V (load cell regression 2026-10-13; **never checked with a known weight**: `known_mass_kg` is empty).
  A knee-extension contraction is roughly 0.8-1.3 V, so **10 mV is about 5 N** and 1 % of a contraction is about 12 mV.
- The saved samples are raw (never smoothed). Review/analysis smooth with a moving average (0.1 s default, the peak uses 20 ms).

## 3. Findings so far

**3.1 The noise is mains hum, and it changes with the set-up/time.** Mains is 50 Hz (measured 49.948-50.051 Hz, drifting). `hum_regime`: `clean` = 50 Hz < 10 mV and 100 Hz < 2 mV; `50Hz_only`; `50+100Hz`.

| Time (30 Oct) | Trials | 50 Hz | 100 Hz | 200 Hz | Regime |
|---|---|---|---|---|---|
| 11:38-11:46 | S1 knee extension 70 (3 trials) | 47-57 mV | 42-45 mV | 8-9 mV | 50+100 Hz |
| 12:12-12:19 | S2 (13 rest trials) | 18-73 mV (see 3.6) | < 0.1 mV | ~0 | 50 Hz only |
| 12:44-13:22 | S1 knee extension, hip flexion | 67-75 mV | 23-26 mV | 4-5 mV | 50+100 Hz |
| 13:35-13:49 | S1 knee flexion 90 and 20 | 3.9-4.3 mV | 0.13-0.28 mV | ~0.05 mV | clean |

Pilot days (S3, S4; 28-29 Oct): 50 Hz median 20 mV (range 1.5-66 mV), 100 Hz median 0.1 mV (max 2.0 mV). So a strong 50 Hz line is not new; the **100 Hz line (23-45 mV) is**.
**What changed between 11:46 and 12:12 and between 13:22 and 13:35 is not recorded** (candidates: USB port or cable - a USB transfer error occurred at 11:55 -, lamps, charger, cable routing,
where the participant sits, the test type changed from hip flexion to knee flexion). With 50+100 Hz the 100 Hz line is as strong as the 50 Hz line and 200 Hz is clearly present, while the odd harmonics
are smaller (150 Hz 0.7-3.6 mV, 250 Hz about 1 mV): the even harmonics dominate, which suggests a rectified/switch-mode or lamp-type source rather than only magnetic pick-up from a distribution
panel (the operator asked whether a nearby electrical cabinet is the source: not proven either way). Whether the participant's coupling to the rig changes the hum is an untested hypothesis.

**3.2 The program's zero rule is blind to 100 Hz hum.** `force.py::_steady_mean` refuses a zero when the signal "wobbles" more than `MAX_STEADY_SD_V` = 30 mV, after removing **only the 50 Hz line**.
With 100/200 Hz present the wobble is 21-35 mV (median 25), so the zero was refused 6 times for "wobbles 32-34 mV, limit 30" (11:17-11:47; two more refusals were a USB error at 11:55 and a +10 mV
drift at 11:56), and S1's first three trials were saved with `zero_set=False`, i.e. no newtons. After removing the 50 and 100 Hz lines the same windows wobble only 7.6-19 mV (median 13):
`wobble_without_50_100_mV` vs `wobble_program_rule_mV`; `passes_program_zero_rule` / `passes_zero_rule_if_harmonics_removed` (13 of 163 trials fail the rule, 10 of them would pass with the harmonics
removed). The zero is a mean over 1.000 s, so a 100 Hz (or any 50 Hz multiple) line averages out of the mean exactly, like the 50 Hz line; refusing the zero because of it costs the operator time, not accuracy.

**3.3 The program's hum estimate is sometimes far too low.** `force.py::mains_amplitude` is a lock-in at exactly 50.000 Hz over the whole trial. When the real mains is 49.96 Hz and the trial is long, the
phase drifts and the amplitude is underestimated: a 60 s trial of S1 reported 4.8 mV (0.4 % of the force rise) against a true ~57 mV (4.7 %), so the 3 % warning (`analysis.HUM_WARN_PCT`) did not fire.
Over 14 contraction trials in the hum regime the program/true ratio is median 0.96 but down to 0.08 (`hum_pct_of_rise_program_method` vs `hum_pct_of_rise_true`; `program_hum_wholetrial_lockin_mV` vs `amp_50Hz_mV`).

**3.4 What the hum does to the force numbers.** The 20 ms and 100 ms moving averages cancel 50/100/200/400 Hz almost exactly (residual < 0.1 % even at 49.96 Hz): the smoothed peak changes < 0.5 % between 20 and 500 ms
windows. A **single raw sample** peak is 78 mV (mean) to 86 mV (max) higher in the 50+100 Hz regime, i.e. about 41-45 N too high (`raw_minus_smoothed_peak_mV`): never use a raw peak for results.

**3.5 The baseline ("rest") level is the bigger force uncertainty.** In the first three S1 trials the level before the push was 19-38 mV above the level after it, which moves the peak above rest by about +-3 %,
far more than the noise effect. Columns `level_first_1s_V`, `level_last_1s_V`, `baseline_first_minus_last_mV` use simple definitions (**not** the program's rest window; when the push starts within the first second they are
meaningless, hence the wide range -224..781 mV). In S2 the rest sat 0-10 N below the stored zero. The program's own rest is `review.py::Trial.force_rest` (median 3 to 0.5 s before the push); the "relaxed level"
(Zero 2) is in `force.py`.

**3.6 S2 hum levels (conditions NOT confirmed).** The plan was 6 conditions x 2 (original set-up, without light, without charger and light, changed set-up with charger, a second changed set-up line, light and
changed set-up); it does not map 1:1 to the 13 trials (no notes were saved). By hum plateau: T1-T4 about 19 mV, T5-T6 about 73 mV, T7-T9 about 31 mV, T10-T11 about 43.5 mV, T12-T13 about 30.5 mV; 100 Hz is ~0 in all.
If the plan order holds: light has no effect on 50 Hz, no charger made it ~4x worse, a changed set-up is worse than the original.

## 4. Where the relevant code is (limbtrack 0.28.0)

- `limbtrack/force.py`: `SAMPLE_RATE_HZ`, `smooth_signal` (comb-exact moving average), `mains_amplitude` (the lock-in), `MAX_STEADY_SD_V`, `MAX_DRIFT_V` (10 mV between halves), `HUM_NOTE_V`, `ZERO_SECONDS` (1 s),
  `RELAXED_SECONDS` (2 s), `_steady_mean` (zero / relaxed-level gate), `ForceCalibration`.
- `limbtrack/analysis.py`: `HUM_WARN_PCT` = 3 (share of the force rise), `repeatability` / `REPEAT_WARN_PCT` (the 10 % check).
- `limbtrack/review.py`: `Trial.force_hum`, `force_rest`, `force_window`, `force_summary` (`mains_hum_v`, `mains_hum_pct_of_peak`), `REST_UNSTEADY_FRACTION`.
- Conventions (`HANDOFF.md`, `FEATURES.md`, `CHANGELOG.md`): numbers are named constants; a feature states whether it alters measured numbers; tests in `tests/` (e.g. `test_force*.py`, `test_p20_checks.py` for the hum).

## 5. Ideas for noise-reducing features (suggestions with the evidence above, not decisions)

1. **Zero/steadiness gate that ignores all mains lines** (3.2): compute the wobble after removing 50, 100, 150, 200 ... Hz (or on a 20 ms-averaged signal). The zero stays exact (whole-cycle averaging) and needless refusals stop.
   Check that the drift test (`MAX_DRIFT_V`) still catches a person leaning on the rig.
2. **A better hum estimate** (3.3): a per-window fit with the frequency searched in 49.8-50.2 Hz (or the median over 1 s windows) instead of one lock-in at 50.000 Hz; also report 100 and 200 Hz; make the 3 % warning fire on the true amplitude.
3. **Live hum meter** in the force panel (50/100/200 Hz amplitude in mV from the last 2 s, threshold tied to the expected contraction: 3 % of 1.2 V = 36 mV, of 0.8 V = 24 mV) with plain advice
   (charger, lamps, USB cable/port, cable routing, grounding), so the operator sees the problem before the trial.
4. **Hum lines in Review/export**: per-trial 50/100/200 Hz amplitude next to the hum % (never change the saved raw samples).
5. **Baseline guard**: show the level before and after the push and flag a difference above e.g. 20 mV (~10 N); push the operator to take the relaxed level (Zero 2) per position. This is the largest force uncertainty found (3.5).
6. **Known-weight check** in the force panel (the calibration has never been verified with a known mass).
7. Outside software: wiring (differential, bias resistors for a floating source), shielding/grounding, USB cable, lamps/chargers/other mains devices near the cable. An extra notch/comb filter on the **saved** data is not needed for peak
   numbers (the moving averages already cancel the lines); it would only help live display and operator feedback.

## 6. Caveats

- The S2 conditions are a guess from the plan and the hum plateaus (3.6); ask the operator.
- `rest_only_trial = 1` marks trials with no real contraction (rise < 50 mV); the hum percentages are blank for them.
- Quiet windows = the lower half of the 1 s windows by residual after removing the hum lines (no push in progress). Amplitudes are medians over those windows (least-squares sine fits with a linear trend removed).
- One lab, one rig, mains 50 Hz, one participant in the regime comparison. Do not generalise without more sessions.

Regenerate (raw trials needed, kept by the operator): `python noise_export.py --source S1=<folder> --source S2=<folder> ... --session-log project=<session_log.csv> --anonymise --private-map <outside the repo>\map.csv --out <dir>`.
It only reads the recordings. **Never commit the private map or any raw signal.**
