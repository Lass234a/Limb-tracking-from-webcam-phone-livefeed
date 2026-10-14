"""Automatic MVIC markers: the detection of the ramp (limbtrack/automvic.py) on made-up force traces. No camera, no DAQ.

The protocol's 1-5-1 tempo (ramp 1 s, hold 5 s, relax 1 s) is the reference: the start is called when the force clearly leaves
its rest level, the end is placed 7 s after the start marker.
"""
import numpy as np
import pytest

from limbtrack.automvic import (DONE, MANUAL, RUNNING, TRIGGERED, WAITING, AutoMvic)

RATE = 1000.0
T0 = 5000.0                      # like time.perf_counter(): the clock does not start at zero


def trace(seconds=14.0, onset=5.0, ramp=1.0, hold=5.0, relax=1.0, peak=400.0, rest=30.0, noise=0.2, hum=0.0, seed=0, drift=None,
          spike=None, wobble=0.0):
    """(times, force in N): rest level `rest`, then the 1-5-1 contraction starting at `onset`, with noise, 50 Hz hum and optional extras."""
    t = np.arange(int(seconds * RATE)) / RATE
    up = np.clip((t - onset) / ramp, 0, 1)
    down = np.clip(1.0 - (t - onset - ramp - hold) / relax, 0, 1)
    shape = np.where(t < onset, 0.0, np.where(t < onset + ramp + hold, up, down))
    f = rest + peak * shape + np.random.default_rng(seed).normal(0, noise, t.size) + hum * np.sin(2 * np.pi * 50.0 * t)
    f = f + wobble * np.sin(2 * np.pi * 1.0 * t)
    if drift is not None:                                   # (start s, end s, N per s): a slow pre-tension
        a, b, k = drift
        f = f + k * (np.clip(t, a, b) - a)
    if spike is not None:                                   # (at s, length s, N): a knock
        at, ln, amp = spike
        f = f + amp * ((t >= at) & (t < at + ln))
    return t + T0, f


def run(t, f, am, block=0.033):
    """Feed in picture-sized blocks. Returns the picture time (end of the block) at which the start was called, or None."""
    step = int(block * RATE)
    for a in range(0, len(t), step):
        if am.feed(t[a:a + step], f[a:a + step]):
            return float(t[min(a + step, len(t)) - 1])
    return None


def test_the_start_is_called_a_fraction_of_a_second_after_the_force_begins_to_rise():
    t, f = trace(onset=5.0, peak=400.0)
    am = AutoMvic(threshold=10.0)
    called = run(t, f, am)
    assert called is not None and am.state == TRIGGERED
    tr = am.trigger
    assert tr["rest_level"] == pytest.approx(30.0, abs=0.5) and tr["threshold_used"] == pytest.approx(10.0)
    assert T0 + 5.0 < tr["trigger_t"] < T0 + 5.25                       # 10 N of a 400 N/s ramp takes 25 ms, plus the 70 ms check
    assert tr["onset_t"] == pytest.approx(T0 + 5.0, abs=0.06)           # the estimate of where the force really began to rise
    assert called - (T0 + 5.0) < 0.35
    assert tr["direction"] == "up"


@pytest.mark.parametrize("peak, ramp", [(80.0, 1.0), (200.0, 3.0), (1000.0, 0.5)])
def test_weak_slow_and_fast_contractions_are_all_found(peak, ramp):
    t, f = trace(onset=4.0, peak=peak, ramp=ramp)
    am = AutoMvic(threshold=10.0)
    assert run(t, f, am) is not None
    assert am.trigger["onset_t"] == pytest.approx(T0 + 4.0, abs=0.25)


def test_a_rest_with_50_hz_hum_does_not_trigger_and_the_ramp_still_does():
    hum = 47.0                                                         # 90 mV of hum at 527 N/V: far above the 10 N threshold
    t, f = trace(onset=100.0, hum=hum, seconds=12.0)
    assert run(t, f, AutoMvic(threshold=10.0)) is None
    t, f = trace(onset=6.0, hum=hum, seconds=12.0)
    am = AutoMvic(threshold=10.0)
    assert run(t, f, am) is not None and am.trigger["onset_t"] == pytest.approx(T0 + 6.0, abs=0.1)


def test_a_slow_pre_tension_is_not_a_ramp():
    t, f = trace(onset=100.0, drift=(3.0, 8.0, 8.0), seconds=10.0)      # +8 N per second for 5 s (40 N in all)
    assert run(t, f, AutoMvic(threshold=10.0)) is None


def test_a_short_knock_is_not_a_ramp_but_a_real_push_after_it_is():
    t, f = trace(onset=9.0, spike=(5.0, 0.03, 150.0), seconds=13.0)
    am = AutoMvic(threshold=10.0)
    called = run(t, f, am)
    assert called is not None and called > T0 + 9.0                     # not the knock at 5 s


def test_a_load_cell_that_reads_negative_works_the_same():
    t, f = trace(onset=5.0, peak=-400.0, rest=-30.0)
    am = AutoMvic(threshold=10.0)
    assert run(t, f, am) is not None
    assert am.trigger["direction"] == "down" and am.trigger["rise_at_trigger"] >= 10.0
    assert am.trigger["onset_t"] == pytest.approx(T0 + 5.0, abs=0.06)


def test_an_unsteady_rest_raises_the_threshold():
    t, f = trace(onset=7.0, peak=15.0, wobble=6.0, seconds=11.0, noise=0.1)       # the rest sways +-6 N once a second
    am = AutoMvic(threshold=10.0)
    assert run(t, f, am) is None                                        # a 15 N push cannot be told from the sway
    t, f = trace(onset=7.0, peak=400.0, wobble=6.0, seconds=11.0, noise=0.1)
    am = AutoMvic(threshold=10.0)
    assert run(t, f, am) is not None and am.trigger["threshold_used"] > 10.0


def test_nothing_is_called_before_there_is_enough_rest_signal():
    t, f = trace(onset=0.3, seconds=0.45)                               # 0.45 s of signal in all: too little to know the rest level
    am = AutoMvic(threshold=10.0)
    assert run(t, f, am) is None and am.state == WAITING and "rest level" in am.reason
    assert "rest level" in am.status_text(T0)


def test_a_ramp_less_than_a_second_after_arming_is_found_from_the_history_before_arming():
    t, f = trace(onset=4.5, seconds=8.0)
    am = AutoMvic(threshold=10.0)
    arm = int(4.0 * RATE)                                               # armed at 4.0 s; the history before is handed over at once
    assert am.feed(t[:arm], f[:arm]) is False and am.state == WAITING
    called = run(t[arm:], f[arm:], am)
    assert called is not None and called - (T0 + 4.5) < 0.35


def test_one_start_one_end_after_the_set_length_and_nothing_more():
    t, f = trace(onset=5.0, seconds=16.0)
    am = AutoMvic(threshold=10.0, length_s=7.0)
    called = run(t, f, am)
    assert called is not None
    assert not am.end_is_due(called)
    am.start_marked(called)
    assert am.state == RUNNING and am.end_due_t == pytest.approx(called + 7.0)
    assert not am.end_is_due(called + 6.9) and am.end_is_due(called + 7.0)
    assert "/ 7 s" in am.status_text(called + 3.0)
    am.end_marked(called + 7.03)
    assert am.state == DONE
    assert am.feed(*trace(onset=5.0, seconds=16.0)) is False            # no second start in the same recording
    res = am.result(t0=T0)
    assert res["start_marker_t_s"] == pytest.approx(called - T0, abs=1e-3) and res["end_marker_t_s"] == pytest.approx(called - T0 + 7.03, abs=1e-3)
    assert res["length_s"] == 7.0 and res["force_onset_estimate_t_s"] == pytest.approx(5.0, abs=0.06)
    assert 0.0 < res["start_marker_after_onset_s"] < 0.4


def test_a_marker_pressed_by_hand_takes_over():
    t, f = trace(onset=9.0, seconds=14.0)
    am = AutoMvic(threshold=10.0)
    am.feed(t[:3000], f[:3000])
    am.manual()
    assert am.state == MANUAL and run(t[3000:], f[3000:], am) is None
    assert "by hand" in am.status_text(T0 + 3.0)


def test_the_volts_fallback_works_in_volts():
    t, f = trace(onset=5.0, peak=0.8, rest=0.05, noise=0.0004)         # a load cell without a slope: the signal is in volts
    am = AutoMvic(threshold=0.02, unit="V")
    assert run(t, f, am) is not None and am.trigger["onset_t"] == pytest.approx(T0 + 5.0, abs=0.1)
