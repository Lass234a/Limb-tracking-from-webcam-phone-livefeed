import math
from pathlib import Path

import csv
import numpy as np
import pytest

from limbtrack.engine import Engine
from limbtrack.filters import LEVELS, OneEuroFilter, make_filter
from limbtrack.protocol import load_protocol
from limbtrack.synthetic import leg_points, render

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("level,min_ratio,max_sweep_lag", [("light", 1.7, 1.0), ("medium", 2.2, 1.3)])
def test_filter_calms_noise_without_much_lag(level, min_ratio, max_sweep_lag):
    rng = np.random.default_rng(0)
    f = make_filter(level)
    raw, out = [], []
    for i in range(600):                                   # still angle with 0.3 deg jitter
        x = 45 + rng.normal(0, 0.3)
        raw.append(x)
        out.append(f(x, i / 30))
    assert np.std(raw[60:]) / np.std(out[60:]) > min_ratio

    f = make_filter(level)                                  # fast sweep (peak ~50 deg/s)
    lag = max(abs((40 + 25 * math.sin(2 * math.pi * i / 90)) - f(40 + 25 * math.sin(2 * math.pi * i / 90), i / 30))
              for i in range(90))
    assert lag < max_sweep_lag

    f = make_filter(level)                                  # slow creep of 1 deg/s must be followed almost exactly
    errs = [(20 + i / 30) - f(20 + i / 30, i / 30) for i in range(300)]
    assert abs(np.mean(errs[150:])) < 0.2


def test_filter_restarts_after_gap_or_backwards_time():
    f = OneEuroFilter()
    f(10.0, 0.0)
    assert f(50.0, 2.0) == 50.0          # long gap: no blending with the old value
    assert f(20.0, 1.0) == 20.0          # time went backwards (frame stepping): restart
    assert make_filter("off") is None and set(LEVELS) == {"off", "light", "medium"}


def test_csv_holds_raw_and_smoothed_and_off_means_identical(tmp_path):
    tests = load_protocol(ROOT / "protocol.json")
    for level in ("light", "off"):
        eng = Engine(tests, recordings_dir=tmp_path / level)
        eng.set_test("knee_extension")
        eng.set_smoothing(level)
        eng.target = 40
        pts = leg_points(hip_flex=70, knee_flex=40, ankle_df=5)
        eng.process(render(pts, seed=0), 0.0)
        for n in eng.test.dots:
            assert eng.mark_dot(n, *(pts[n] + 2))
        eng.process(render(pts, seed=0), 0.0)
        eng.start_trial("P01")
        for i in range(60):
            eng.process(render(pts, seed=i), i / 30)
        paths = eng.stop_trial()
        rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
        assert "knee_raw_deg" in rows[0] and "knee_deg" in rows[0]
        filt = np.array([float(r["knee_deg"]) for r in rows])
        raw = np.array([float(r["knee_raw_deg"]) for r in rows])
        if level == "off":
            assert np.array_equal(filt, raw)
        else:
            assert np.std(filt[10:]) < np.std(raw[10:])
