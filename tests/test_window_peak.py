"""The sliding-window peak and the time to peak force (0.29.0): the highest mean force over any stretch of W seconds in the analysis
window, from the raw samples; and the seconds from the start of the push to the peak. Built on the known force signal of test_force."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.force import WINDOW_PEAK_DEFAULT_S, ForceCalibration  # noqa: E402

from tests import test_force as T  # noqa: E402  (humming_trial: rest, a light push, a ramp, a hold at 1.0 V from 8.6 to 13 s, release)


def test_the_default_window_is_half_a_second_and_a_multiple_of_20_ms():
    assert WINDOW_PEAK_DEFAULT_S == 0.5 and round(WINDOW_PEAK_DEFAULT_S * 1000) % 20 == 0


def test_the_window_peak_of_a_known_hold_is_the_hold(tmp_path):
    tr, _ = T.humming_trial(tmp_path)                      # 1.0 V hold + 50 Hz hum of 0.09 V + noise
    wp = analysis.window_peak(tr, 8.0, 13.5)
    assert wp["unit"] == "V" and wp["value"] == pytest.approx(1.0, abs=0.01)     # the hum is gone: 0.5 s is 25 whole mains cycles
    assert wp["window_s"] == pytest.approx(0.5) and wp["end_s"] - wp["start_s"] == pytest.approx(0.499, abs=0.002)
    assert 8.6 <= wp["start_s"] and wp["end_s"] <= 13.0 and wp["at_s"] == pytest.approx((wp["start_s"] + wp["end_s"]) / 2)


def test_the_window_peak_is_not_above_the_peak_of_the_smoothed_curve_and_not_below_the_plateau(tmp_path):
    tr, _ = T.humming_trial(tmp_path)
    fs = tr.force_summary(8.0, 13.5)
    wp = analysis.window_peak(tr, 8.0, 13.5)
    assert wp["value"] <= fs["peak"] + 1e-9                # a longer average cannot be higher than a shorter one's highest value
    assert wp["value"] >= fs["plateau_mean"] - 0.01


def test_a_short_spike_counts_less_in_a_long_window_than_in_a_short_one(tmp_path):
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    tr.force_v = tr.force_v.copy()
    k = int(np.searchsorted(tr.force_t, 10.0))
    tr.force_v[k:k + 20] += 1.0                            # a 20 ms spike of +1 V on the hold
    short = analysis.window_peak(tr, 8.0, 13.5, 0.02)["value"]
    long = analysis.window_peak(tr, 8.0, 13.5, 0.5)["value"]
    assert short == pytest.approx(2.0, abs=0.05)           # 20 ms window: the spike on top of the hold
    assert long == pytest.approx(1.0 + 1.0 * 0.02 / 0.5, abs=0.02)    # 500 ms window: the spike is 4 % of it


def test_window_zero_is_the_highest_single_sample(tmp_path):
    tr, _ = T.humming_trial(tmp_path)
    m = (tr.force_t >= 8.0) & (tr.force_t <= 13.5)
    assert analysis.window_peak(tr, 8.0, 13.5, 0.0)["value"] == pytest.approx(float(np.max(tr.force_v[m])))


def test_a_band_shorter_than_the_window_gives_nothing_not_a_wrong_number(tmp_path):
    tr, _ = T.humming_trial(tmp_path)
    assert analysis.window_peak(tr, 10.0, 10.3, 0.5) is None
    assert analysis.window_peak(tr, 10.0, 10.3, 0.2) is not None
    assert analysis.window_peak(tr, 30.0, 40.0) is None   # no sample in the band
    fp = analysis.force_peaks(tr, 10.0, 10.3)
    assert np.isnan(fp["window_peak"]) and np.isnan(fp["time_to_window_peak_s"])


def test_stretches_with_a_missing_sample_are_skipped(tmp_path):
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    tr.force_v = tr.force_v.copy()
    a, b = np.searchsorted(tr.force_t, [9.0, 9.2])
    tr.force_v[a:b] = np.nan                               # a gap in the middle of the hold
    wp = analysis.window_peak(tr, 8.6, 13.0)
    assert wp["value"] == pytest.approx(1.0, abs=0.02)
    assert wp["end_s"] < 9.0 or wp["start_s"] > 9.2        # the stretch does not contain the gap
    tr.force_v[:] = np.nan
    assert analysis.window_peak(tr, 8.6, 13.0) is None


def test_newtons_are_used_when_the_force_is_calibrated(tmp_path):
    tr, _ = T.humming_trial(tmp_path)
    cal = ForceCalibration()
    cal.set_zero(0.1)
    cal.n_per_v = 200.0 / 0.9                              # 0.9 V above rest = 200 N
    tr.apply_calibration(cal, "test calibration")
    wp = analysis.window_peak(tr, 8.0, 13.5)
    assert wp["unit"] == "N" and wp["value"] == pytest.approx(200.0, abs=3.0)


def _ramp_trial(tmp_path):
    """No noise: rest 0.1 V to 2 s, a straight ramp to 1.0 V at 4 s, a drop to 0.5 V and a hold. The highest value is at 4.0 s."""
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    t = np.arange(-1.0, 12.0, 0.001)
    v = np.interp(t, [-1, 2, 4, 4.1, 12], [0.1, 0.1, 1.0, 0.5, 0.5])
    tr.force_t, tr.force_v, tr.force_n = t, v, np.full(t.size, np.nan)
    tr.force_smooth_s = 0.1
    return tr


def test_time_to_peak_runs_from_the_start_of_the_push_to_the_peak(tmp_path):
    tr = _ramp_trial(tmp_path)
    ft = analysis.force_timing(tr, 0.0, 11.0)
    assert ft["t_peak_s"] == pytest.approx(4.0, abs=0.06)
    assert ft["onset_s"] == pytest.approx(2.2, abs=0.1)                        # the force last within 10 % of its rise
    assert ft["time_to_peak_s"] == pytest.approx(ft["t_peak_s"] - ft["onset_s"]) and ft["time_to_peak_s"] == pytest.approx(1.8, abs=0.12)
    fp = analysis.force_peaks(tr, 0.0, 11.0)
    assert fp["time_to_peak_s"] == pytest.approx(ft["time_to_peak_s"]) and fp["onset_s"] == pytest.approx(ft["onset_s"])
    # the 0.5 s window that is highest ends at the top of the ramp: its middle lies before the peak, so it is reached sooner
    assert fp["window_peak_at_s"] < ft["t_peak_s"] and 0 < fp["time_to_window_peak_s"] < fp["time_to_peak_s"]
    assert fp["window_peak"] < ft["peak"]                                      # the mean of a ramp's last 0.5 s is below its top


def test_a_trial_without_force_gives_blanks(tmp_path):
    tr, _ = T.humming_trial(tmp_path)
    tr.force_t = tr.force_v = tr.force_n = None
    fp = analysis.force_peaks(tr, 0.0, 10.0)
    assert all(np.isnan(v) for v in fp.values()) and analysis.window_peak(tr, 0.0, 10.0) is None


def test_the_trials_own_window_setting_is_used(tmp_path):
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    tr.window_peak_s = 1.0
    assert analysis.force_peaks(tr, 8.0, 13.5)["window_peak_ms"] == pytest.approx(1000.0)
    assert analysis.force_peaks(tr, 8.0, 13.5, window_s=0.2)["window_peak_ms"] == pytest.approx(200.0)


def test_the_session_row_and_its_columns_carry_the_new_numbers(tmp_path):
    tr, _ = T.humming_trial(tmp_path)
    row = analysis.session_row(tr)
    for c in ("force_window_peak", "force_window_peak_ms", "force_window_peak_at_s", "force_time_to_peak_s", "force_time_to_window_peak_s"):
        assert c in analysis.SESSION_COLUMNS
    assert row["force_window_peak"] == pytest.approx(1.0, abs=0.02) and row["force_window_peak_ms"] == pytest.approx(500.0)
    assert row["force_time_to_peak_s"] > 0 and row["force_time_to_window_peak_s"] > 0
    assert row["force_window_peak"] <= row["force_peak"] + 1e-9
