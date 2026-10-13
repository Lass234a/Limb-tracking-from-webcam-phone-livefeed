"""The load cell regressions: fitting, the precision model, the registry and the script that adds a new calibration."""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from limbtrack import loadcell
from limbtrack.loadcell import GRAVITY, fit_points

ROOT = Path(__file__).resolve().parent.parent


def synthetic(slope=0.0186, icpt=0.087, noise=0.0, hyst_v=0.0, seed=0):
    kg = np.array([0, 50, 100, 150, 200, 250, 200, 150, 100, 50, 0], float)
    phase = ["up"] * 5 + ["peak"] + ["down"] * 5
    rng = np.random.RandomState(seed)
    v = icpt + slope * kg + np.where(np.array(phase) == "down", hyst_v, 0.0) + rng.normal(0, noise, kg.size)
    return kg, v, phase


def test_the_fit_recovers_the_slope_the_scatter_and_the_hysteresis():
    kg, v, phase = synthetic(noise=0.001, hyst_v=0.0056, seed=3)
    reg = fit_points(kg, v, phase, id="t")
    assert reg.slope_v_per_kg == pytest.approx(0.0186, rel=0.01) and reg.n_per_v == pytest.approx(GRAVITY / 0.0186, rel=0.01)
    assert reg.hysteresis_kg == pytest.approx(0.0056 / 0.0186, abs=0.1)
    kg0, v0, ph0 = synthetic()
    exact = fit_points(kg0, v0, ph0, id="exact")
    assert exact.rmse_kg == pytest.approx(0.0, abs=1e-9) and exact.r2 == pytest.approx(1.0)
    assert exact.range_kg == [0.0, 250.0] and exact.n_runs == 11 and len(exact.points) == 11


def test_precision_is_the_scatter_plus_a_slope_error_that_grows_with_the_load():
    reg = fit_points(*synthetic(noise=0.002, seed=1), id="t")
    assert float(reg.sd_kg(0.0)) == pytest.approx(reg.rmse_kg)
    assert float(reg.sd_kg(250.0)) == pytest.approx(np.hypot(reg.rmse_kg, 250.0 * reg.slope_rel_se))
    assert float(reg.sd_kg(250.0)) > float(reg.sd_kg(10.0))
    assert float(reg.sd_newton(100 * GRAVITY)) == pytest.approx(float(reg.sd_kg(100.0)) * GRAVITY)
    assert reg.sd_kg(np.array([1.0, -1.0, 5.0])).shape == (3,)


def test_a_fit_that_cannot_be_right_is_refused():
    with pytest.raises(ValueError, match="At least 4"):
        fit_points([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="must rise"):
        fit_points([0, 50, 100, 150], [1.0, 0.9, 0.8, 0.7])


def test_the_registry_keeps_every_regression_and_one_is_active(tmp_path):
    path = tmp_path / "reg.json"
    assert loadcell.load_registry(path) == (None, {}) and loadcell.active_regression(path) is None
    a = fit_points(*synthetic(), id="a")
    b = fit_points(*synthetic(slope=0.0190), id="b")
    loadcell.add_regression(a, path=path)
    loadcell.add_regression(b, path=path)                                   # the first one stays active unless asked
    assert loadcell.active_regression(path).id == "a" and set(loadcell.load_registry(path)[1]) == {"a", "b"}
    loadcell.add_regression(b, activate=True, path=path)
    assert loadcell.active_regression(path).id == "b" and loadcell.find_regression("a", path).slope_v_per_kg == pytest.approx(0.0186)
    assert loadcell.find_regression("nope", path) is None and loadcell.find_regression("", path) is None
    assert loadcell.load_registry(path)[1]["b"].points == b.points         # the points are kept with the fit


def test_the_calibration_that_ships_with_the_program_is_the_one_from_13_october():
    reg = loadcell.active_regression()
    assert reg is not None and reg.id == "2026-10-13"
    assert reg.n_per_v == pytest.approx(526.95, abs=0.05) and reg.n_runs == 24 and reg.range_kg == [0.0, 250.0]
    assert reg.rmse_kg == pytest.approx(0.154, abs=0.005) and reg.hysteresis_kg == pytest.approx(0.29, abs=0.02)
    assert COVERAGE_OK(reg)


def COVERAGE_OK(reg):
    return float(2 * reg.sd_newton(100 * GRAVITY)) < 4.0          # a reading of 100 kg is good to a few newtons


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "LoadCellCalibration" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_add_calibration_reads_a_points_file_skips_rows_marked_keep_0_and_stores_the_fit(tmp_path, capsys):
    kg, v, phase = synthetic(noise=0.001, seed=2)
    pts, reg_file = tmp_path / "points.csv", tmp_path / "reg.json"
    with open(pts, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "kg", "volts", "phase", "keep"])
        for i, (k, x, p) in enumerate(zip(kg, v, phase)):
            w.writerow([f"run{i}", k, x, p, 1])
        w.writerow(["bad", 999, 0.0, "up", 0])                              # a flagged run: ignored
        w.writerow(["bad2", "", 5.0, "up", 0])                              # a flagged run without a mass: ignored too
    mod = load_script("add_calibration")
    reg = mod.main([str(pts), "--id", "new", "--description", "test", "--registry", str(reg_file), "--activate"])
    assert reg.n_runs == 11 and reg.slope_v_per_kg == pytest.approx(0.0186, rel=0.01)
    assert loadcell.active_regression(reg_file).id == "new" and "Stored as 'new'" in capsys.readouterr().out


def test_logs_to_points_flags_a_mislabelled_run(tmp_path):
    from datetime import datetime, timedelta
    mod = load_script("logs_to_points")
    folder = tmp_path / "logs"
    folder.mkdir()
    t0 = datetime(2026, 10, 13, 15, 0, 0)
    plan = [("0", 0), ("25", 25), ("50", 50), ("0", 50), ("75", 75), ("100", 100), ("75", 75)]     # the second '0 kg' file is really the 50 kg run
    for i, (label, true_kg) in enumerate(plan):
        with open(folder / f"LogFile_{label}kg_13-10-13-15-{i:02d}-00.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Time", "Dev1/ai0", "weight_1"])
            for n in range(20000):                                          # 20 s at 1000 Hz
                w.writerow([(t0 + timedelta(minutes=i, milliseconds=n)).strftime("%d/%m/%Y %H:%M:%S.%f"), 0.087 + 0.0186 * true_kg, label])
    out = tmp_path / "points.csv"
    mod.main([str(folder), "-o", str(out)])
    rows = list(csv.DictReader(open(out, newline="")))
    assert len(rows) == 7
    kept = [r for r in rows if r["keep"] == "1"]
    assert len(kept) == 6 and [r for r in rows if r["keep"] == "0"][0]["kg"] == "0" and "is the name right" in [r for r in rows if r["keep"] == "0"][0]["note"]
    assert rows[5]["phase"] == "peak" and rows[6]["phase"] == "down" and float(rows[2]["volts"]) == pytest.approx(0.087 + 0.0186 * 50)
