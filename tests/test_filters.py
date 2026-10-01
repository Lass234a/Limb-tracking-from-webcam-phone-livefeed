"""Display smoothing: a 0.2 s moving average that never touches the recorded angles."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from limbtrack.engine import Engine
from limbtrack.filters import SMOOTHING_WINDOW_S, MovingAverage
from limbtrack.protocol import load_protocol
from limbtrack.synthetic import leg_points, render

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def test_window_is_a_fifth_of_a_second():
    assert SMOOTHING_WINDOW_S == 0.2


def test_moving_average_is_the_mean_of_the_last_0_2_seconds():
    f = MovingAverage(0.2)
    out = [f(v, i / 30) for i, v in enumerate([10, 20, 30, 40, 50, 60, 70, 80])]    # 30 fps: 6-7 samples fit in 0.2 s
    assert out[0] == 10 and out[1] == 15 and out[2] == 20                          # still filling up
    # at t = 7/30 the window holds samples with t > 7/30 - 0.2 = 1/30 : samples 2..7 (six values)
    assert out[7] == pytest.approx(np.mean([30, 40, 50, 60, 70, 80]))


def test_window_is_in_seconds_not_frames():
    slow, fast = MovingAverage(0.2), MovingAverage(0.2)
    for i in range(40):
        slow(i, i / 10)          # 10 fps: only two samples fit in 0.2 s
        last_fast = fast(i, i / 100)    # 100 fps: about 20 samples fit
    assert slow(40, 4.0) == pytest.approx(39.5)        # mean of the last two samples
    assert last_fast < 39 - 5


def test_constant_unchanged_noise_calmed_and_lag_is_half_the_window():
    rng = np.random.default_rng(0)
    f = MovingAverage()
    assert all(f(45.0, i / 30) == 45.0 for i in range(20))
    f = MovingAverage()
    raw, out = [], []
    for i in range(600):                                   # still angle with 0.3 deg jitter at 30 fps
        x = 45 + rng.normal(0, 0.3)
        raw.append(x)
        out.append(f(x, i / 30))
    assert np.std(raw[60:]) / np.std(out[60:]) > 2.0       # about sqrt(6)
    f = MovingAverage()                                    # steady 30 deg/s: shown value trails by about 0.1 s = 3 deg
    lag = [(20 + 30 * i / 30) - f(20 + 30 * i / 30, i / 30) for i in range(120)]
    assert np.mean(lag[30:]) == pytest.approx(30 * 0.1, abs=0.5)
    f = MovingAverage()                                    # slow creep of 1 deg/s: 0.1 deg late
    lag = [(20 + i / 30) - f(20 + i / 30, i / 30) for i in range(300)]
    assert np.mean(lag[30:]) == pytest.approx(0.1, abs=0.05)


def test_time_going_backwards_starts_afresh():
    f = MovingAverage()
    f(10.0, 5.0)
    f(12.0, 5.03)
    assert f(50.0, 1.0) == 50.0          # frame stepping backwards: no mixing with the future


def _record(tmp_path, smoothing, frames=60):
    eng = Engine(TESTS, recordings_dir=tmp_path / ("on" if smoothing else "off"))
    eng.set_test("knee_extension")
    eng.set_smoothing(smoothing)
    eng.target = 40
    seq = [leg_points(hip_flex=70, knee_flex=40 + 25 * math.sin(2 * math.pi * i / 90), ankle_df=5)
           for i in range(frames)]
    eng.process(render(seq[0], seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(seq[0][n] + 2))
    eng.process(render(seq[0], seed=0), 0.0)
    eng.start_trial("P01")
    shown, raw = [], []
    for i, pts in enumerate(seq):
        res = eng.process(render(pts, seed=i), i / 30)
        k = next(a for a in res.angles if a.name == "knee")
        shown.append(k.value)
        raw.append(k.raw_value)
    return eng.stop_trial(), np.array(shown), np.array(raw), seq


def test_csv_always_holds_the_unsmoothed_angle(tmp_path):
    truth = lambda seq: np.array([40 + 25 * math.sin(2 * math.pi * i / 90) for i in range(len(seq))])   # noqa: E731
    for smoothing in (True, False):
        paths, shown, raw, seq = _record(tmp_path, smoothing)
        rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
        assert "knee_raw_deg" not in rows[0] and "knee_deg" in rows[0]
        csv_deg = np.array([float(r["knee_deg"]) for r in rows])
        assert np.allclose(csv_deg, raw, atol=0.001)                       # CSV == raw, whatever the screen shows
        assert np.abs(csv_deg - truth(seq)).max() < 1.0                    # and it follows the real movement without lag
        dev = np.array([float(r["knee_dev"]) for r in rows])
        assert np.allclose(dev, csv_deg - 40.0, atol=0.001)                # deviation computed from the raw angle
        meta = json.loads(Path(paths["meta.json"]).read_text())
        assert meta["smoothing"]["enabled"] is smoothing and meta["smoothing"]["window_s"] == 0.2
        if smoothing:
            assert np.abs(shown - raw).max() > 1.5                         # the screen lags during the fast sweep
        else:
            assert np.array_equal(shown, raw)


def test_screen_verdict_uses_smoothed_value_and_csv_flag_the_raw_one(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0                                      # tolerance 3
    steady = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(steady, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(steady[n] + 2))
    for i in range(10):                                    # long enough for the average to settle at 60
        eng.process(render(steady, seed=i), i / 30)
    res = eng.process(render(leg_points(hip_flex=70, knee_flex=64, ankle_df=5), seed=99), 10 / 30)   # sudden +4 deg
    k = next(a for a in res.angles if a.name == "knee")
    assert k.raw_value == pytest.approx(64, abs=0.6) and k.raw_status == "out"       # recorded: out of tolerance at once
    assert 60 < k.value < 63 and k.status == "ok"                                    # screen: average has not caught up yet
