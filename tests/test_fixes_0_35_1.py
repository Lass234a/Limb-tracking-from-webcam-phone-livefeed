"""The fixes of 0.35.1 (an audit of 0.35.0), each with the case that showed it:

* no force sample is lost when the DAQ clock estimate steps back (samples got the time of the sample before and were never handed to force.csv);
* the zero and the relaxed level average a whole number of 50 Hz cycles, so the hum really cancels out of them;
* the rate of force development uses the median sample interval, not the first one (which could be zero, or 1 % off after rounding);
* the cache of the resting force follows the force arrays themselves, not their id();
* a library angle may use the old 'kind' / 'points' spelling, which the protocol check already accepted;
* the freeze check names the library versions.
Everything is made up here: no DAQ, camera or recording is needed."""
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from limbtrack import analysis, force, provenance
from limbtrack.force import ForceCalibration, ForceSource, whole_mains_cycles
from limbtrack.protocol import load_protocol, validate_protocol
from limbtrack.review import Trial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import freeze_check  # noqa: E402

RATE = 1000.0


# ------------------------------------------------------------------------------------------------ force sample times
def _hand_over(src, cursor, until):
    """What Engine._take_force does: the samples from the cursor up to `until`, then the cursor just after the last one handed over."""
    ts, _ = src.samples_between(cursor, until)
    return ts.size, (float(ts[-1]) + 1e-9 if ts.size else cursor)


@pytest.mark.parametrize("drop_at", [2500, 2501, 2502])
def test_no_force_sample_is_lost_when_the_usb_delay_drops(drop_at):
    """The USB delay drops from about 6 ms to about 3 ms (the computer became less busy): the clock estimate steps back 3 ms. Until 0.35.1 the
    first samples of the next chunk got the time of the sample before, and when the chunk before had already been handed to the recording
    those samples were never written (2 of 40 000 when the step came right after a picture)."""
    src = ForceSource(rate=RATE)
    rng = np.random.default_rng(2)
    cursor, handed, pushed = -math.inf, 0, 0
    for k in range(4000):                                          # 40 s of 10 ms chunks, a picture after every third
        delay = 0.006 if k < drop_at else 0.003
        arrival = 100.0 + (k + 1) * 0.010 + delay + rng.uniform(0, 0.0005)
        src._push(np.zeros(10), arrival)
        pushed += 10
        if k % 3 == 2:
            n, cursor = _hand_over(src, cursor, arrival)
            handed += n
    n, _ = _hand_over(src, cursor, math.inf)
    assert handed + n == pushed
    t = np.concatenate([c[0] for c in src._chunks])
    assert np.all(np.diff(t) > 0)                                  # never two samples with the same time
    assert np.min(np.diff(t)) >= force.MIN_SAMPLE_STEP_S * 0.999


def test_the_time_of_a_sample_moves_by_at_most_the_step_back():
    """Keeping time running forward only touches the samples whose estimate fell behind: the step is taken up within its own length."""
    src = ForceSource(rate=RATE)
    for k in range(3000):
        src._push(np.zeros(10), 100.0 + (k + 1) * 0.010 + (0.006 if k < 2000 else 0.003))
    t = np.concatenate([c[0] for c in src._chunks])
    late = t[20000:]
    assert np.allclose(np.diff(late[10:]), 1.0 / RATE, atol=1e-9)  # (after the step: back to one sample per millisecond, on the new estimate)
    assert late[-1] == pytest.approx(100.0 + 3000 * 0.010 + 0.003, abs=1e-9)


# ------------------------------------------------------------------------------------------------ the zero over whole hum cycles
def _fixed_clock(monkeypatch, now):
    monkeypatch.setattr(force, "time", types.SimpleNamespace(perf_counter=lambda: now))


def test_whole_mains_cycles_keeps_the_newest_whole_cycles():
    v = np.arange(987.0)
    w = whole_mains_cycles(v, RATE)
    assert w.size == 980 and w[-1] == 986.0 and w[0] == 7.0         # 49 cycles of 20 samples, the newest kept
    assert whole_mains_cycles(np.arange(99.0), RATE) is None        # fewer than 5 cycles


def test_the_zero_is_not_moved_by_a_part_cycle_of_hum(monkeypatch):
    """The 1 s window of a zero holds about 985 to 1000 samples (the newest are still on their way from the DAQ). With the 90 mV hum of
    9 Oct, 987 samples and the worst phase of the hum, the plain mean was off by 0.52 mV (0.27 N); over the 49 whole cycles it is the true
    zero."""
    now, zero_v, hum = 1000.0, 0.1234, 0.09
    n = 3000
    t = np.arange(n) / RATE
    volts = zero_v + hum * np.sin(2 * np.pi * 50 * t + 2.83)
    src = ForceSource(rate=RATE)
    src.calibration = ForceCalibration(n_per_v=526.95, zero_set=False)
    arrival = now - 0.0137                                         # the newest 13.7 ms have not arrived: 987 samples in the last second
    src._push(volts, arrival)
    _fixed_clock(monkeypatch, now)
    _, window = src.samples_between(now - 1.0, now + 1.0)
    assert window.size == 987
    assert abs(window.mean() - zero_v) > 3e-4                      # what the zero was until 0.35.1
    assert src.zero() is None
    assert src.calibration.offset_v == pytest.approx(zero_v, abs=1e-9)
    assert src.last_reading["samples"] == 980


def test_the_relaxed_level_also_uses_whole_cycles(monkeypatch):
    now, level, hum = 1000.0, 0.5, 0.09
    t = np.arange(4000) / RATE
    src = ForceSource(rate=RATE)
    src.calibration = ForceCalibration(n_per_v=526.95, zero_set=False)
    src.calibration.set_zero(0.1)
    src._push(level + hum * np.sin(2 * np.pi * 50 * t + 0.4), now - 0.0113)
    _fixed_clock(monkeypatch, now)
    assert src.relax("P01 | knee_extension | 70") is None
    assert src.calibration.relaxed_v == pytest.approx(level, abs=1e-9)


# ------------------------------------------------------------------------------------------------ rate of force development
def _ramp_trial(t):
    """A trial-like object with a ramp of exactly 1000 N/s from 1.0 to 1.5 s, a hold at 500 N and a release."""
    y = np.clip((t - 1.0) * 1000.0, 0.0, 500.0)
    y[t > 2.5] = 0.0
    return types.SimpleNamespace(has_force=lambda: True, force_signal=lambda: ("N", y, y), force_t=t, force_rest=lambda: None)


@pytest.mark.parametrize("first_gap", [0.0, 0.00101, 0.00099])
def test_the_rate_of_force_development_does_not_hang_on_the_first_sample_interval(first_gap):
    """force.csv rounds the times to 10 us, so the first interval of a file is 0.99 to 1.01 ms; until 0.35.1 it set the 100 ms step (99 to 101
    samples, the rate about 1 % off), and two samples with the same time (the clock estimate stepping back) raised ZeroDivisionError."""
    t = np.arange(3000) / RATE
    t[1] = t[0] + first_gap
    ft = analysis.force_timing(_ramp_trial(t), 0.0, 3.0)
    assert ft["rise_rate_per_s"] == pytest.approx(1000.0, rel=1e-6)


# ------------------------------------------------------------------------------------------------ the cache of the resting force
N_PER_V = 526.95


def _numbers_only_trial(seconds=3.0):
    """The force part of a Trial only (no files are read): 0.1 V at rest, a push of 1.2 V in the middle, the recorded zero 0 V."""
    tr = Trial.__new__(Trial)
    tr.force_t = np.arange(int(seconds * RATE)) / RATE
    tr.force_v = 0.1 + np.clip((tr.force_t - seconds / 2) * 4.0, 0.0, 1.2) * (tr.force_t < 0.8 * seconds)
    tr.force_n = ForceCalibration(n_per_v=N_PER_V, offset_v=0.0).to_newton(tr.force_v)
    tr.force_smooth_s = force.SMOOTH_DEFAULT_S
    return tr


def test_the_resting_force_follows_two_zeros_set_one_after_the_other(monkeypatch):
    """Two zeros set one after the other, with no look at the rest in between (the newtons replaced as Trial.apply_calibration replaces them).
    Until 0.35.1 the cache was keyed on id(force_n), and Python gives a freed array's id to a new one: on a 3 s trial the rest of the FIRST
    newtons came back in 27 of 30 tries of a loop (how often depends on the memory allocator). Here every array has the same id, the worst
    case of that reuse; the cache now keeps the arrays it was made from, so a new array is never taken for the old one."""
    from limbtrack import review
    monkeypatch.setattr(review, "id", lambda obj: 0, raising=False)
    tr = _numbers_only_trial()
    assert tr.force_rest()["level"] == pytest.approx(0.1 * N_PER_V)
    for zero in (0.05, 0.02):
        cal = ForceCalibration(n_per_v=N_PER_V)
        cal.set_zero(zero)
        tr.force_n = cal.to_newton(tr.force_v)
    assert tr.force_rest()["level"] == pytest.approx((0.1 - 0.02) * N_PER_V)
    assert tr._rest_cache[1] is tr.force_n
    tr.force_smooth_s = 0.2                                        # another smoothing: worked out again too
    assert tr.force_rest() is tr._rest_cache[3] and tr._rest_cache[0] == 0.2


# ------------------------------------------------------------------------------------------------ the protocol file
def test_a_library_angle_may_use_the_old_kind_points_spelling(tmp_path):
    """The check accepted {"kind": "joint", "points": [...]} in angle_library, but loading it then failed (AngleDef has no 'kind')."""
    data = {"dot_order": ["trochanter", "epicondyle", "malleolus"],
            "angle_library": {"knee3": {"kind": "joint", "points": ["trochanter", "epicondyle", "malleolus"], "sign": -1, "offset": 180}},
            "tests": {"knee_only": {"label": "Knee", "primary": "knee3", "targets": [90], "angles": ["knee3"]}}}
    assert validate_protocol(data) == []
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    a = load_protocol(p)["knee_only"].angles[0]
    assert a.vectors == [["epicondyle", "trochanter"], ["epicondyle", "malleolus"]] and a.fulcrum == "epicondyle" and a.primary


# ------------------------------------------------------------------------------------------------ the verdict is not told by colour alone
def test_an_angle_outside_its_tolerance_says_out_in_words():
    """WCAG 2.1 1.4.1 (use of colour): the picture said 'outside the tolerance' only by turning the line red; for a red-green colour-blind
    operator that red and the green are 1.5 : 1 apart in brightness. The line now also says OUT; an angle inside says nothing extra."""
    from limbtrack.overlay import angle_line
    from limbtrack.results import LOST, OK, OUT, AngleResult

    def angle(status, value=75.2, reference=70.0, **kw):
        a = AngleResult("knee", "Knee flexion", True, value=value, deviation=value - reference, reference=reference, status=status)
        for k, v in kw.items():
            setattr(a, k, v)
        return a
    assert angle_line(angle(OUT)) == "Knee flexion:   75.2  (+5.2 vs 70)  OUT"
    assert angle_line(angle(OK, value=71.0)) == "Knee flexion:   71.0  (+1.0 vs 70)"
    assert angle_line(angle(LOST)) == "Knee flexion: lost"
    assert angle_line(angle(OUT, ref_offset=10.0, value=85.2, reference=80.0)) == "Knee flexion:   75.2  (+5.2 vs 70)  OUT   dots 85.2"
    assert angle_line(angle(OUT, suspect="moved apart")).endswith("OUT  CHECK DOTS")


# ------------------------------------------------------------------------------------------------ the freeze check
def test_the_freeze_report_names_the_library_versions():
    info = {"commit": "abc", "branch": "main", "tags": [], "changed": [], "untracked": []}
    libs = provenance.libraries()
    assert set(libs) == {"python", "numpy", "opencv", "pyside6", "pyqtgraph"} and libs["numpy"] == np.__version__
    text = "\n".join(freeze_check.report(info, "1", "c", "d", None, [], now=0, libraries=libs))
    assert f"numpy {np.__version__}" in text and "opencv" in text
    assert "Libraries" not in "\n".join(freeze_check.report(info, "1", "c", "d", None, [], now=0))     # (the old call still works)
    rec = provenance.collect(None)
    assert {k: rec[k] for k in libs} == libs                       # meta.json keeps the same keys
