"""Taking the zero: the 50 Hz mains hum no longer blocks it (it averages out of the mean exactly), real noise up to 30 mV is accepted,
a drifting signal is still refused, and what the zero was taken from is saved. Samples are made up: no DAQ is needed."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.force import (HUM_NOTE_V, MAX_DRIFT_V, MAX_STEADY_SD_V, ForceCalibration, ForceSource,  # noqa: E402
                             make_source)
from limbtrack.gui import LiveTab  # noqa: E402

RATE = 1000
TRUE_ZERO = 0.1234


def source_with(volts):
    """A force source that has received `volts` (1000 samples per second) a moment ago, with the load cell's slope but no zero yet."""
    src = ForceSource()
    src.calibration = ForceCalibration(n_per_v=526.95, zero_set=False)
    for k in range(0, volts.size, 10):
        src._push(volts[k:k + 10], time.perf_counter())
    return src


def signal(seconds=1.6, hum=0.0, noise=0.0, ramp=0.0, seed=1):
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    return TRUE_ZERO + hum * np.sin(2 * np.pi * 50 * t + 0.7) + rng.normal(0, noise, t.size) + ramp * t / seconds


def test_the_limits_are_what_the_notes_say():
    assert MAX_STEADY_SD_V == 0.03 and MAX_DRIFT_V == 0.01 and HUM_NOTE_V == 0.01


@pytest.mark.parametrize("hum", [0.034, 0.09])
def test_mains_hum_does_not_stop_the_zero_and_does_not_move_it(hum):
    """34 mV was the hum on 14 Oct 13:36 (a raw spread of 26 mV: refused at the old 20 mV limit); 90 mV the worst seen on 9 Oct."""
    src = source_with(signal(hum=hum, noise=0.003))
    assert src.zero() is None
    cal = src.calibration
    assert cal.zero_set and cal.offset_v == pytest.approx(TRUE_ZERO, abs=0.002)          # the hum cancelled out of the mean
    assert cal.zero_hum_v == pytest.approx(hum, rel=0.05)                                  # and it is reported
    assert cal.zero_noise_v == pytest.approx(0.003, abs=0.001)                             # the noise is what is left without the hum


def test_noise_up_to_the_limit_is_accepted_and_the_zero_is_still_good_to_about_a_millivolt():
    src = source_with(signal(noise=0.028, seed=3))
    assert src.zero() is None
    cal = src.calibration
    assert cal.offset_v == pytest.approx(TRUE_ZERO, abs=0.004)                             # inside 4 standard errors
    assert cal.zero_noise_v == pytest.approx(0.028, rel=0.1)
    assert cal.zero_se_v == pytest.approx(cal.zero_noise_v / np.sqrt(src.last_reading["samples"]))   # noise / sqrt(samples)
    assert cal.zero_se_n == pytest.approx(cal.zero_se_v * 526.95)
    assert cal.zero_se_n < 1.0                                                              # under one newton


def test_too_much_noise_is_still_refused_and_says_so():
    src = source_with(signal(noise=0.05, seed=4))
    why = src.zero()
    assert why and "not steady enough" in why and "50 Hz" in why
    assert not src.calibration.zero_set


@pytest.mark.parametrize("ramp, accepted", [(0.012, True), (0.06, False)])
def test_a_drifting_signal_is_refused_because_averaging_cannot_remove_drift(ramp, accepted):
    """The ramp is over the whole 1.6 s; the zero looks at the last 1.0 s, whose halves then differ by 0.5/1.6 of it."""
    src = source_with(signal(noise=0.002, ramp=ramp))
    why = src.zero()
    if accepted:
        assert why is None
    else:
        assert why and "not steady" in why and "moved" in why


def test_hum_and_drift_together_are_judged_on_the_drift_only():
    ok = source_with(signal(hum=0.09, noise=0.003, ramp=0.012))
    assert ok.zero() is None
    bad = source_with(signal(hum=0.09, noise=0.003, ramp=0.06))
    assert "moved" in (bad.zero() or "")


def test_a_known_load_is_read_with_the_same_rule():
    """Calibrating with a known mass uses the same reading: 34 mV of hum on top of the load no longer blocks it."""
    src = source_with(signal(hum=0.034, noise=0.003))
    assert src.zero() is None
    src2 = source_with(signal(hum=0.034, noise=0.003) + 0.8)          # 0.8 V more with a known mass on the sensor
    src2.calibration = src.calibration
    assert src2.calibrate(10.0, seconds=1.0) is None and src2.calibration.n_per_v == pytest.approx(10.0 * 9.80665 / 0.8, rel=0.01)


def test_the_zero_description_is_saved_and_old_files_still_load():
    cal = ForceCalibration(n_per_v=526.95, zero_set=False)
    cal.set_zero(0.1234, noise_v=0.02, hum_v=0.034, samples=1000)
    d = cal.to_dict()
    assert d["zero_noise_v"] == 0.02 and d["zero_hum_v"] == 0.034 and d["zero_se_v"] == pytest.approx(0.02 / np.sqrt(1000))
    assert ForceCalibration.from_dict(d).to_dict() == d
    old = {k: v for k, v in d.items() if not k.startswith("zero_") or k == "zero_v"}                    # a file from before 0.10.0
    back = ForceCalibration.from_dict(old)
    assert back.zero_noise_v is None and back.zero_se_n is None and back.calibrated
    plain = ForceCalibration()
    plain.set_zero(0.2)                                                                                  # a zero set by hand (no window)
    assert plain.zero_se_v is None


def test_the_window_says_how_good_the_zero_is_and_mentions_the_hum_only_when_it_is_there():
    cal = ForceCalibration(n_per_v=526.95, zero_set=False)
    assert LiveTab._zero_quality_note(cal) == ""                                                         # no description: nothing to say
    cal.set_zero(0.1234, noise_v=0.009, hum_v=0.003, samples=1000)
    quiet = LiveTab._zero_quality_note(cal)
    assert "Noise 9 mV" in quiet and "good to about 0.3 mV" in quiet and "N)" in quiet and "hum" not in quiet
    cal.set_zero(0.1234, noise_v=0.009, hum_v=0.034, samples=1000)
    humming = LiveTab._zero_quality_note(cal)
    assert "34 mV" in humming and "cancels out of the zero" in humming and "isolated outlet" in humming


def test_zero_through_the_simulated_source_still_works():
    src = make_source("simulated")
    src.calibration = ForceCalibration(n_per_v=526.95, zero_set=False)
    src.start()
    try:
        time.sleep(1.4)
        assert src.zero() is None and src.calibration.zero_se_v is not None and src.calibration.zero_se_v < 0.001
    finally:
        src.stop()
