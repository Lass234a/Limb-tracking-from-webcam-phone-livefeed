"""0.35.0 (F8): the check weight. A known mass (20 to 50 kg) is hung on the zeroed rig; the program averages the zeroed force over 3 s (a whole number of
50 Hz cycles, never a single sample), compares it with mass x 9.80665 N and shows the error in percent as green, amber or red. The result goes to the
session log and into the meta.json of every following trial (`check_weight`), a soft reminder comes up when none was made in the last hours or the last
one was amber or red, and the session summary and the AnyBody export get two appended columns.

The reading is tried on the real force source code: a source that gets its samples through the real ring buffer (no thread, instant), and a
simulated load cell that runs in real time with a constant hanging mass, a 50 Hz hum and a contraction. Files (meta.json, session_log.csv,
settings.json, the export CSV) are written and read back; the widgets are real, offscreen."""
import csv
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis, anybody, check_weight as cw, session_memory  # noqa: E402
from limbtrack import force as force_mod  # noqa: E402
from limbtrack import verdict as verdicts  # noqa: E402
from limbtrack.engine import SOFTWARE_VERSION, Engine  # noqa: E402
from limbtrack.force import ForceCalibration, ForceSource, SimulatedForce  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.loadcell import GRAVITY  # noqa: E402
from limbtrack.protocol import load_protocol, plan_positions  # noqa: E402
from limbtrack.review import Trial, trial_row  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PLAN = plan_positions(TESTS)
P = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
N_PER_V = 527.0                         # about the slope of the 2026-10-13 regression (0.0186 V per kg)
OFFSET_V = 0.0123
MASS_KG = 25.0
LOAD_N = MASS_KG * GRAVITY              # 245.17 N


# ------------------------------------------------------------------------------------------------ a source fed from an array
def volts(seconds=3.2, load_n=0.0, hum_n=0.0, noise_v=0.002, step_n=0.0, seed=0):
    """`seconds` of 1000-per-second volts of a load with a 50 Hz hum, noise, and optionally a step half way (a swinging weight)."""
    n = int(seconds * 1000)
    t = np.arange(n) / 1000.0
    newtons = load_n + hum_n * np.sin(2 * np.pi * 50 * t + 0.7) + step_n * (t >= seconds / 2)
    return newtons / N_PER_V + OFFSET_V + np.random.RandomState(seed).normal(0.0, noise_v, n)


def source(v, regression_id="2026-10-13", zero_set=True, n_per_v=N_PER_V):
    """The real ForceSource with its real ring buffer, given `v` as if the DAQ had just delivered it (no thread: instant)."""
    s = ForceSource()
    s.calibration = ForceCalibration(offset_v=OFFSET_V, n_per_v=n_per_v, method=f"load cell regression {regression_id}" if regression_id else "zero and 5 kg",
                                     regression_id=regression_id, zero_set=zero_set, zero_v=OFFSET_V, when="2026-11-04T09:00:00")
    s._push(np.asarray(v, dtype=np.float64), time.perf_counter())
    return s


class LoadedForce(SimulatedForce):
    """The simulated load cell in real time with a hanging mass (`load_n`, settable while it runs), a 50 Hz hum, and the usual contraction."""

    def __init__(self, hum_n=10.0, **kw):
        super().__init__(n_per_v=N_PER_V, offset_v=OFFSET_V, **kw)
        self.load_n, self.hum_n = 0.0, hum_n

    def newtons_at(self, t):
        return super().newtons_at(t) + self.load_n + self.hum_n * np.sin(2 * np.pi * 50 * np.asarray(t, dtype=np.float64))


# ------------------------------------------------------------------------------------------------ the pure rules
def test_typing_accepts_a_comma_and_the_unit_and_keeps_only_a_usable_mass():
    assert cw.parse_mass("25") == (25.0, None) and cw.parse_mass(" 24,5 kg ") == (24.5, None) and cw.parse_mass("") == (None, None)
    v, why = cw.parse_mass("2.5")
    assert v is None and "outside the 20 to 50 kg" in why
    v, why = cw.parse_mass("25000")                                                          # grams typed
    assert v is None and "wrong unit" in why
    assert cw.parse_mass("heavy")[0] is None and "not a number" in cw.parse_mass("heavy")[1]
    for bad in ("nan", "inf"):
        assert cw.parse_mass(bad)[0] is None and cw.parse_mass(bad)[1]
    assert cw.parse_mass("20")[0] == 20.0 and cw.parse_mass("50")[0] == 50.0                  # the limits themselves are fine
    assert cw.expected_n(20) == pytest.approx(196.133)                                       # mass x 9.80665


def test_the_levels_are_green_up_to_one_percent_amber_up_to_two_and_red_above_whichever_the_sign():
    assert (cw.AMBER_ABOVE_PCT, cw.RED_ABOVE_PCT) == (1.0, 2.0)
    assert [cw.level_of(e) for e in (0.0, 1.0, -1.0, 1.0001, -1.5, 2.0, -2.0, 2.0001, -7.0)] == \
        ["green", "green", "green", "amber", "amber", "amber", "amber", "red", "red"]


def test_a_known_hanging_mass_reads_within_a_tenth_of_a_percent_and_a_single_sample_does_not():
    """A 15 N hum on a 245 N load: any ONE sample is up to 6 % off; the 3 s average (150 whole mains cycles) is not."""
    v = volts(load_n=LOAD_N, hum_n=15.0)
    src = source(v)
    singles = (v[-2900:] - OFFSET_V) * N_PER_V
    assert np.max(np.abs(singles - LOAD_N)) / LOAD_N > 0.04                                  # what one sample would have said
    res, why = cw.measure(src, MASS_KG)
    assert why is None
    assert abs(res["error_pct"]) < 0.1 and res["level"] == "green" and res["ok"] is True
    assert res["expected_n"] == pytest.approx(LOAD_N, abs=0.001) and res["measured_n"] == pytest.approx(LOAD_N, abs=0.3)
    assert res["samples"] % 20 == 0 and 2700 <= res["samples"] <= 3000 and res["seconds"] == 3.0       # a whole number of 20 ms: the hum cancels
    assert res["hum_v"] == pytest.approx(15.0 / N_PER_V, abs=0.002)                          # the hum was there and was seen
    assert res["zero_v"] == OFFSET_V and res["calibration_id"] == "2026-10-13" and res["note"] == ""
    assert res["time"][:10] == datetime.now().isoformat()[:10]


def test_the_error_is_signed_and_the_level_follows_it():
    src = source(volts(load_n=LOAD_N, hum_n=5.0))
    want = {24.8: (+0.806, "green"), 24.7: (+1.215, "amber"), 24.4: (+2.459, "red"), 25.5: (-1.961, "amber"), 26.0: (-3.846, "red")}
    for typed, (err, level) in want.items():
        res, why = cw.measure(src, typed)
        assert why is None and res["error_pct"] == pytest.approx(err, abs=0.05), typed
        assert res["level"] == level and res["ok"] is (level == "green"), typed
    off = source(volts(load_n=LOAD_N, hum_n=5.0), n_per_v=N_PER_V * 1.015)                    # a slope that is 1.5 % too steep
    res, _ = cw.measure(off, MASS_KG)
    assert res["error_pct"] == pytest.approx(1.5, abs=0.05) and res["level"] == "amber"


def test_everything_that_makes_a_check_meaningless_is_refused_with_a_sentence_and_nothing_is_changed():
    good = volts(load_n=LOAD_N, hum_n=5.0)
    ok = source(good)
    cases = {
        "mass": (ok, 5.0, "outside the 20 to 50 kg"),
        "not connected": (None, MASS_KG, "not connected"),
        "no zero": (source(good, zero_set=False), MASS_KG, "not zeroed"),
        "no slope": (source(good, n_per_v=None), MASS_KG, "no load cell calibration"),
        "too few samples": (source(volts(seconds=1.0, load_n=LOAD_N)), MASS_KG, "Not enough force samples"),
        "swinging": (source(volts(load_n=LOAD_N, step_n=60.0)), MASS_KG, "moved"),
        "noisy": (source(volts(load_n=LOAD_N, noise_v=0.06)), MASS_KG, "wobbles"),
    }
    for name, (src, mass, text) in cases.items():
        res, why = cw.measure(src, mass)
        assert res is None and text in why, (name, why)
    ok.error = "The DAQ stopped sending samples (cable unplugged or device reset)."
    assert cw.measure(ok, MASS_KG)[1] == ok.error
    ok.error, ok.frozen = None, True
    assert "Stop recording first" in cw.measure(ok, MASS_KG)[1]
    assert ok.calibration.offset_v == OFFSET_V and ok.calibration.zero_v == OFFSET_V        # reading the check moved nothing


def test_a_slow_creep_is_refused_by_the_drift_limit_alone_and_a_big_hum_is_not_counted_as_wobble():
    """T1: a 20 N creep over the 3 s is quiet (noise far under the limit) but moves the second half 19 mV above the first: only the drift limit
    sees it. T2: a 30 N hum would make a plain spread 40 mV (over the 30 mV limit), yet the hanging weight is steady and is accepted."""
    n = int(3.2 * 1000)
    creep = volts(load_n=LOAD_N, hum_n=5.0) + np.linspace(0.0, 20.0 / N_PER_V, n)
    assert float(creep.std()) < force_mod.MAX_STEADY_SD_V                          # the noise limit cannot be what refuses it
    res, why = cw.measure(source(creep), MASS_KG)
    assert res is None and "moved" in why and "not steady enough" not in why
    humming = volts(load_n=LOAD_N, hum_n=30.0)
    assert float(humming.std()) > force_mod.MAX_STEADY_SD_V                        # a plain spread would have refused it
    res, why = cw.measure(source(humming), MASS_KG)
    assert why is None and abs(res["error_pct"]) < 0.1
    assert res["noise_v"] < 0.005 and res["hum_v"] == pytest.approx(30.0 / N_PER_V, abs=0.003)


def test_a_slope_that_came_from_a_known_weight_is_said_not_to_check_the_regression():
    res, _ = cw.measure(source(volts(load_n=LOAD_N), regression_id=""), MASS_KG)
    assert "did not come from the load cell's regression" in res["note"] and res["calibration_id"] == ""
    assert "regression" in cw.summary_text(res)


# ------------------------------------------------------------------------------------------------ the reminder
def made(hours_ago=0.0, level="green", err=0.3, now=None, cal=None):
    cal = cal or ForceCalibration(offset_v=OFFSET_V, n_per_v=N_PER_V, regression_id="2026-10-13", zero_set=True)
    when = (now or datetime.now()) - timedelta(hours=hours_ago)
    return {"mass_kg": 25.0, "expected_n": LOAD_N, "measured_n": LOAD_N * (1 + err / 100), "error_pct": err, "level": level, "ok": level == "green",
            "time": when.isoformat(timespec="seconds"), "zero_v": OFFSET_V, "calibration_id": cal.regression_id, "n_per_v": cal.n_per_v}, cal


def test_the_reminder_follows_the_age_the_level_and_the_calibration_and_only_while_the_force_is_connected():
    now = datetime(2026, 11, 4, 12, 0, 0)
    fresh, cal = made(2.0, now=now)
    assert cw.reminder(fresh, cal, True, now) == "" and cw.standing(fresh, cal, now)[0] == "ok"
    assert cw.reminder(None, cal, False, now) == ""                                          # no force: nothing to check
    assert "none made in the last 12 h" in cw.reminder(None, cal, True, now)
    old, _ = made(12.5, now=now)
    assert "over 12 h old" in cw.reminder(old, cal, True, now)
    edge, _ = made(11.9, now=now)
    assert cw.reminder(edge, cal, True, now) == ""
    amber, _ = made(1.0, level="amber", err=1.4, now=now)
    assert "AMBER (+1.4 %" in cw.reminder(amber, cal, True, now)
    red, _ = made(1.0, level="red", err=-2.6, now=now)
    assert "RED (-2.6 %" in cw.reminder(red, cal, True, now)
    other = ForceCalibration(offset_v=OFFSET_V, n_per_v=N_PER_V * 1.01, regression_id="2026-10-13", zero_set=True)
    assert "another calibration" in cw.reminder(fresh, other, True, now)
    future, _ = made(-5.0, now=now)
    assert cw.standing(future, cal, now)[0] == "stale"                                       # a clock that jumped back is not trusted
    assert cw.standing({"time": "not a time"}, cal, now)[0] == "stale"
    assert cw.reminder(amber, cal, True, now) == cw.reminder(amber, cal, True, now + timedelta(hours=3))      # no changing numbers: 'Noted' holds


def test_the_meta_block_says_whether_the_calibration_is_still_the_one_it_was_made_with():
    check, cal = made(0.5)
    block = cw.meta_block(check, cal)
    assert block["same_calibration"] is True and block["mass_kg"] == 25.0 and "note_on_values" in block
    cal.n_per_v *= 1.02
    assert cw.meta_block(check, cal)["same_calibration"] is False
    assert cw.meta_block(check, None)["same_calibration"] is None and cw.meta_block(None, cal) is None


def test_a_new_zero_or_a_zero_lost_on_reconnect_makes_the_check_old_news_but_never_refuses_anything():
    """R1: the check was made against one zero. A re-zero (Z) keeps the regression and the slope, so only the zero tells."""
    now = datetime(2026, 11, 4, 12, 0, 0)
    check, cal = made(1.0, now=now)
    assert cw.same_zero(check, cal) is True and cw.standing(check, cal, now)[0] == "ok"
    assert cw.meta_block(check, cal)["same_zero"] is True and list(cw.meta_block(check, cal))[-1] == "same_zero"      # the new key is last
    cal.set_zero(OFFSET_V + 20.0 / N_PER_V)                              # the real re-zero with 20 N of strap tension on
    assert cw.same_calibration(check, cal) is True                       # (the slope did not change: that is why the zero needs its own look)
    assert cw.same_zero(check, cal) is False and cw.standing(check, cal, now)[0] == "zero"
    text = cw.reminder(check, cal, True, now)
    assert "zero was set again" in text and cw.reminder(check, cal, False, now) == ""
    assert "zero was set again" in cw.summary_text(check, cal)
    assert cw.meta_block(check, cal)["same_zero"] is False and cw.meta_block(check, cal)["same_calibration"] is True
    cal.set_zero(OFFSET_V)                                               # back on the old zero volts: the same newtons, the same check
    assert cw.same_zero(check, cal) is True and cw.standing(check, cal, now)[0] == "ok"
    cal.zero_set = False                                                 # a reconnect that has not been zeroed again
    assert cw.same_zero(check, cal) is False
    assert cw.same_zero(check, None) is None and cw.meta_block(check, None)["same_zero"] is None
    old = {k: v for k, v in check.items() if k != "zero_v"}              # a check without its zero cannot be judged: no invented warning
    cal.zero_set = True
    assert cw.same_zero(old, cal) is None and cw.meta_block(old, cal)["same_zero"] is None


def test_a_re_zero_after_the_check_reaches_the_trial_the_reminder_and_the_rig_check(tmp_path):
    """R1 through the engine, the real meta.json and the rig check list."""
    from limbtrack import rig_check
    src = source(volts(load_n=LOAD_N, hum_n=5.0))
    eng = Engine(TESTS, tmp_path)
    eng.force = src
    result, why = eng.make_check_weight(MASS_KG)
    assert why is None
    assert eng._check_weight_meta()["same_zero"] is True
    src.calibration.set_zero(OFFSET_V + 20.0 / N_PER_V)
    block = eng._check_weight_meta()
    assert block["same_zero"] is False and block["level"] == "green" and block["zero_v"] == OFFSET_V
    facts = rig_check.gather(eng)
    assert facts["check_weight"]["state"] == "zero"
    res = rig_check.evaluate(facts)
    assert res["items"]["check_weight"]["state"] == "amber" and "zero was set again" in res["items"]["check_weight"]["reason"]


def test_the_export_cells_are_blank_without_a_check_and_for_a_damaged_block():
    assert cw.COLUMNS == ["check_weight_error_pct", "check_weight_time"]
    assert cw.export_cells({}) == {"check_weight_error_pct": "", "check_weight_time": ""}
    assert cw.export_cells({"check_weight": None}) == {"check_weight_error_pct": "", "check_weight_time": ""}
    assert cw.export_cells({"check_weight": "oops"}) == {"check_weight_error_pct": "", "check_weight_time": ""}
    assert cw.export_cells({"check_weight": {"error_pct": float("nan"), "time": "T"}}) == {"check_weight_error_pct": "", "check_weight_time": "T"}
    assert cw.export_cells({"check_weight": {"error_pct": -0.4, "time": "2026-11-04T09:30:00"}}) == \
        {"check_weight_error_pct": -0.4, "check_weight_time": "2026-11-04T09:30:00"}


def test_the_log_detail_reads_back_to_the_same_numbers():
    res, _ = cw.measure(source(volts(load_n=LOAD_N, hum_n=5.0)), MASS_KG)
    got = cw.parse_log_detail(cw.log_detail(res))
    assert float(got["mass_kg"]) == res["mass_kg"] and float(got["expected_n"]) == res["expected_n"] and float(got["measured_n"]) == res["measured_n"]
    assert float(got["error_pct"]) == res["error_pct"] and got["level"] == "green" and got["ok"] == "yes" and float(got["zero_v"]) == res["zero_v"]
    assert got["calibration_id"] == "2026-10-13" and got["time"] == res["time"]


def test_the_clean_function_refuses_what_is_not_a_result():
    good, _ = made()
    assert cw.clean(good) == good and cw.clean(None) is None
    for bad in ("x", {}, {**good, "level": "blue"}, {**good, "error_pct": float("nan")}, {**good, "time": 5}):
        with pytest.raises(ValueError):
            cw.clean(bad)


# ------------------------------------------------------------------------------------------------ the engine, in real time
def record(eng, seconds):
    """One real recording of `seconds`: synthetic pictures on the perf_counter clock (the clock of the force), returns the meta.json path."""
    paths = eng.start_trial(eng.participant_code)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        eng.process(render(P, seed=int(time.perf_counter() * 30) % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    time.sleep(1.1)                                      # (trial file names carry the second)
    return paths["meta.json"]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """A real-time session: zero with nothing hanging, a trial before any check, the 25 kg mass hung, the check, then a trial with a contraction."""
    root = tmp_path_factory.mktemp("checkweight") / "recordings"
    eng = Engine(TESTS, recordings_dir=root)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.participant_code = "P01"
    eng.tested_leg = "right"
    eng.process(render(P, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    src = LoadedForce(peak_n=400.0, rest_s=1e6, ramp_s=0.6, hold_s=1.2, period_s=1e7)       # (no contraction until the test asks for one)
    src.start()
    eng.force = src
    out = {"eng": eng, "src": src, "root": root, "folder": root / "P01"}
    try:
        time.sleep(1.4)
        assert src.zero() is None                                  # nothing hanging, the 10 N hum on the signal
        out["before"] = record(eng, 0.6)                           # a trial made before any check (the rig is still unloaded)
        out["no_check_block"] = eng._check_weight_meta()
        src.load_n = LOAD_N                                        # the mass is hung
        time.sleep(3.7)                                            # (the window must hold only the loaded signal)
        out["result"], out["why"] = eng.make_check_weight(MASS_KG)
        src.rest_s = src._next / src.rate + 0.7                    # the push begins 0.7 s into the next recording
        out["after"] = record(eng, 3.5)
        verdicts.write_verdict(out["after"], verdicts.VALID)
    finally:
        src.stop()
    return out


def test_the_real_time_check_reads_the_hanging_mass_through_hum_and_noise(session):
    res, why = session["result"], session["why"]
    assert why is None and res is not None
    assert abs(res["error_pct"]) < 0.3 and res["level"] == "green" and res["ok"] is True, res
    assert res["expected_n"] == pytest.approx(LOAD_N, abs=0.001) and res["measured_n"] == pytest.approx(LOAD_N, rel=0.003)
    assert res["samples"] % 20 == 0 and res["samples"] >= 2700
    assert res["hum_v"] > 0.012                                                  # the 10 N hum (19 mV at this slope) was on the signal
    assert res["zero_v"] == pytest.approx(OFFSET_V, abs=0.002) and "regression" in res["note"]       # (the simulated slope is not a regression's)
    assert session["eng"].check_weight["error_pct"] == res["error_pct"]


def test_the_engine_keeps_the_latest_check_and_refuses_a_change_while_recording(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    check, _ = made()
    eng.check_weight = check
    assert eng.check_weight == check and eng._check_weight_meta()["same_calibration"] is None        # (no force connected here)
    eng.check_weight = None
    assert eng.check_weight is None and eng._check_weight_meta() is None
    with pytest.raises(ValueError):
        eng.check_weight = {"level": "green"}
    res, why = eng.make_check_weight(MASS_KG)
    assert res is None and "not connected" in why and eng.check_weight is None               # a refused check leaves nothing behind
    eng.process(render(P, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    eng.participant_code = "P01"
    eng.start_trial("P01")
    try:
        with pytest.raises(RuntimeError, match="Stop recording"):
            eng.check_weight = check
        with pytest.raises(RuntimeError, match="Stop recording"):
            eng.make_check_weight(MASS_KG)
    finally:
        eng.stop_trial()


def test_a_trial_carries_the_latest_check_and_one_made_before_any_check_says_null(session):
    before = json.loads(session["before"].read_text(encoding="utf-8"))
    after = json.loads(session["after"].read_text(encoding="utf-8"))
    assert "check_weight" in before and before["check_weight"] is None                       # the key is there, the value says none
    block = after["check_weight"]
    for k in ("mass_kg", "expected_n", "measured_n", "error_pct", "zero_v", "time", "calibration_id", "ok", "level", "same_calibration"):
        assert k in block, k
    assert block["mass_kg"] == 25.0 and block["same_calibration"] is True and block["ok"] is True
    assert block["error_pct"] == session["result"]["error_pct"] and block["time"] == session["result"]["time"]
    assert after["software_version"] == SOFTWARE_VERSION
    assert "changed_during_recording" not in after and "changed_during_recording" not in before        # the setting is frozen like the others
    assert after["force"]["peak_n"] > 450                                                    # (a real contraction on top of the hanging mass)


def test_the_measured_force_of_the_trial_is_not_touched_by_the_check(session):
    """The check only reads: the zero in the trial is the one that was taken, the newtons are volts through the same zero and slope."""
    after = json.loads(session["after"].read_text(encoding="utf-8"))
    cal = after["force_setup"]["calibration"]
    assert cal["offset_v"] == pytest.approx(session["result"]["zero_v"], abs=1e-6) and cal["n_per_v"] == N_PER_V
    force = np.genfromtxt(str(session["after"]).replace("_meta.json", "_force.csv"), delimiter=",", names=True)
    v, n = force["volts"], force["newtons"]
    assert len(v) > 2000 and np.nanmax(np.abs(n - (v - cal["offset_v"]) * cal["n_per_v"])) < 5e-3          # (the file rounds volts to 6 and newtons to 3 decimals)


def test_the_session_summary_and_the_anybody_export_get_the_two_columns_at_the_end(session, tmp_path):
    cols = analysis.SESSION_COLUMNS
    assert cols[-2:] == cw.COLUMNS and cols[:3] == ["trial", "position_number", "trial_number"] and len(cols) == len(set(cols)) == 66 + 9 + 11 + 2
    assert anybody.COLUMNS[-3:-1] == cw.COLUMNS and anybody.COLUMNS[-4] == "notes" and anybody.COLUMNS[0] == "participant"      # (0.35.0 F13 adds its column after these)
    check = json.loads(session["after"].read_text(encoding="utf-8"))["check_weight"]
    row_before, _ = trial_row(session["before"], 0.1, 0.5)
    row_after, _ = trial_row(session["after"], 0.1, 0.5)
    assert row_before["check_weight_error_pct"] == "" and row_before["check_weight_time"] == ""
    assert row_after["check_weight_error_pct"] == check["error_pct"] and row_after["check_weight_time"] == check["time"]
    csv_path = analysis.write_session_csv([row_before, row_after], tmp_path / "session.csv")
    back = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert list(back[0])[-2:] == cw.COLUMNS and back[0]["check_weight_error_pct"] == "" and float(back[1]["check_weight_error_pct"]) == pytest.approx(check["error_pct"], abs=1e-4)
    # the AnyBody export: the trial is valid and has a force rise, so it is the row of its position
    from limbtrack import review_nav
    rows, paths = [], {}
    for e in review_nav.list_trials(session["folder"]):
        r, err = trial_row(e.meta_path, 0.1, 0.5)
        assert r is not None, err
        rows.append(r)
        paths[r["trial"]] = e.meta_path
    res = analysis.anybody_rows(rows, lambda r: paths[r["trial"]], lambda p: Trial(p, open_video=False), plan=PLAN, judged_only=True,
                                export_version=SOFTWARE_VERSION)
    assert len(res["rows"]) == 1
    exported = res["rows"][0]
    assert exported["check_weight_error_pct"] == check["error_pct"] and exported["check_weight_time"] == check["time"]
    assert "no check weight" not in exported["notes"]
    out = tmp_path / "anybody_inputs.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        anybody.write_csv(fh, res["rows"])
    got = list(csv.DictReader(open(out, encoding="utf-8")))
    assert list(got[0])[-3:-1] == cw.COLUMNS and float(got[0]["check_weight_error_pct"]) == pytest.approx(check["error_pct"], abs=1e-4)


def test_a_trial_of_an_earlier_version_without_the_key_opens_and_exports_with_blanks(session, tmp_path):
    meta = json.loads(session["after"].read_text(encoding="utf-8"))
    old = tmp_path / "old"
    old.mkdir()
    for p in session["folder"].glob(session["after"].name[: -len("_meta.json")] + "*"):
        (old / p.name).write_bytes(p.read_bytes())
    path = old / session["after"].name
    del meta["check_weight"]
    meta["software_version"] = "0.34.0"
    path.write_text(json.dumps(meta), encoding="utf-8")
    row, err = trial_row(path, 0.1, 0.5)
    assert row is not None, err
    assert row["check_weight_error_pct"] == "" and row["check_weight_time"] == "" and row["software_version"] == "0.34.0"
    tr = Trial(path, open_video=False)
    assert len(tr.t) > 0
    tr.close()


# ------------------------------------------------------------------------------------------------ the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    (tmp_path / "recordings").mkdir()                    # (the session log never creates the folder; a recording would)
    w = MainWindow(TESTS, tmp_path)
    yield w
    w.live.worker = None
    w.close()


class RunningCamera:
    is_file = False
    paused = False
    at_end = False
    source = 0
    camera = None
    status_note = ""

    def stop(self):
        pass

    def wait(self, ms=0):
        return True

    def isRunning(self):
        return True


def connect(win, src):
    win.live.force_source = src
    win.engine.force = src


def typed(edit, text):
    edit.setText(text)
    edit.textEdited.emit(text)


def log_lines(win):
    path = win.live.recordings_dir / "session_log.csv"
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def test_the_box_is_in_the_force_section_and_says_what_is_missing(win):
    live = win.live
    box = live.check_box
    assert live.accordion.sections["force"].body.isAncestorOf(box)
    live.refresh_status()
    assert not box.button.isEnabled() and "not connected" in box.label.text()
    typed(box.edit, "25")
    assert box.button.isEnabled() is False                                                   # nothing to check without a force source
    connect(win, source(volts(load_n=LOAD_N), zero_set=False))
    live.refresh_status()
    assert box.button.isEnabled() and "Not zeroed yet" in box.label.text()
    assert box.check() is None and "not zeroed" in box.label.text().lower()
    assert win.engine.check_weight is None                                                   # a refused check leaves nothing behind
    lines = log_lines(win)
    assert lines and lines[-1]["event"] == "check weight refused" and "not zeroed" in lines[-1]["detail"]
    live.force_source = None
    win.engine.force = None


def test_a_mass_that_is_not_usable_is_not_kept_and_the_box_says_why(win):
    box = win.live.check_box
    connect(win, source(volts(load_n=LOAD_N)))
    typed(box.edit, "2.5")
    assert box.mass_kg is None and "outside the 20 to 50 kg" in box.label.text() and box.edit.styleSheet()
    assert box.check() is None and win.engine.check_weight is None
    typed(box.edit, "")
    assert box.check() is None and "Type the known mass" in box.label.text()
    typed(box.edit, "24,9 kg")
    assert box.mass_kg == 24.9 and not box.edit.styleSheet()
    win.live.force_source = None
    win.engine.force = None


def test_pressing_the_button_shows_the_result_and_writes_a_line_to_the_session_log_that_reads_back(win):
    live = win.live
    box = live.check_box
    src = source(volts(load_n=LOAD_N, hum_n=8.0))
    connect(win, src)
    typed(box.edit, "25")
    res = box.check()
    assert res is not None and res["level"] == "green"
    assert "GREEN" in box.label.text() and "25 kg" in box.label.text() and "error" in box.label.text()
    lines = [r for r in log_lines(win) if r["event"] == "check weight"]
    assert len(lines) == 1
    assert list(lines[0]) == ["time", "seconds_since_start", "event", "detail"] and lines[0]["time"][:10] == datetime.now().isoformat()[:10]
    got = cw.parse_log_detail(lines[0]["detail"])
    assert float(got["mass_kg"]) == 25.0 and float(got["expected_n"]) == pytest.approx(LOAD_N, abs=0.001)
    assert float(got["measured_n"]) == res["measured_n"] and float(got["error_pct"]) == res["error_pct"] and got["ok"] == "yes"
    assert float(got["zero_v"]) == OFFSET_V and got["calibration_id"] == "2026-10-13" and got["time"] == res["time"]
    typed(box.edit, "24.4")                                                                   # a wrong typed mass: the log says red
    res2 = box.check()
    assert res2["level"] == "red" and "RED" in box.label.text()
    lines = [r for r in log_lines(win) if r["event"] == "check weight"]
    assert len(lines) == 2 and cw.parse_log_detail(lines[1]["detail"])["ok"] == "no" and cw.parse_log_detail(lines[1]["detail"])["level"] == "red"
    assert win.engine.check_weight["level"] == "red"                                         # the latest one is kept
    live.force_source = None
    win.engine.force = None


def test_the_reminder_row_is_soft_noted_by_a_button_and_gone_after_a_green_check(win):
    live, e = win.live, win.engine
    live.worker = RunningCamera()
    e.process(render(P, seed=0), 0.0)
    live.update_steps()
    assert "checkweight" not in live.reminders.open_keys()                                   # no force connected: no reminder
    src = source(volts(load_n=LOAD_N, hum_n=5.0))
    connect(win, src)
    live.update_steps()
    assert "checkweight" in live.reminders.open_keys() and "none made" in live.reminders.rows["checkweight"][1].text()
    live.btn_rem_check.click()                                                               # Noted
    assert "checkweight" not in live.reminders.open_keys()
    assert any(r["event"] == "check weight reminder dismissed" for r in log_lines(win))
    typed(live.check_box.edit, "24.7")                                                       # the hung mass is 25 kg: +1.2 % is amber
    live.check_box.check()
    assert e.check_weight["level"] == "amber"
    live.update_steps()
    assert "checkweight" in live.reminders.open_keys() and "AMBER" in live.reminders.rows["checkweight"][1].text()
    typed(live.check_box.edit, "25")
    src._push(volts(load_n=LOAD_N, hum_n=5.0), time.perf_counter())
    live.check_box.check()
    assert e.check_weight["level"] == "green"
    live.update_steps()
    assert "checkweight" not in live.reminders.open_keys()
    e.check_weight = made(13.0)[0]                                                           # a check from yesterday evening
    live.update_steps()
    assert "checkweight" in live.reminders.open_keys() and "over 12 h old" in live.reminders.rows["checkweight"][1].text()
    live.force_source = None
    e.force = None


def test_the_reminder_is_written_down_when_record_is_pressed_and_never_stops_it(win):
    live, e = win.live, win.engine
    live.worker = RunningCamera()
    e.process(render(P, seed=0), 0.0)
    connect(win, source(volts(load_n=LOAD_N)))
    texts = [t for t, in_file in live._open_reminders()]
    assert any(t.startswith("Check weight:") for t in texts)
    assert not [t for t, in_file in live._open_reminders() if in_file and t.startswith("Check weight")]    # (the file holds the check itself)
    live.force_source = None
    e.force = None


def test_a_real_recording_with_the_reminder_open_is_not_refused_and_lists_it(tmp_path):
    """Engine level, real time: Record is pressed with no check weight; the trial is recorded and its file says the reminder was open."""
    eng = Engine(TESTS, recordings_dir=tmp_path / "recordings")
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.participant_code = "P01"
    eng.process(render(P, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    src = LoadedForce(peak_n=300.0)
    src.start()
    eng.force = src
    try:
        time.sleep(1.3)
        assert src.zero() is None
        eng.reminders_open = [cw.reminder(eng.check_weight, src.calibration, True)]
        meta = json.loads(record(eng, 0.5).read_text(encoding="utf-8"))
    finally:
        src.stop()
    assert meta["check_weight"] is None and meta["reminders_open_at_start"][0].startswith("Check weight: none made")


def test_the_mass_is_remembered_through_settings_json_and_the_result_is_not(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    (tmp_path / "recordings").mkdir()
    w1 = MainWindow(TESTS, tmp_path)
    try:
        connect(w1, source(volts(load_n=LOAD_N)))
        typed(w1.live.check_box.edit, "25,5")
        w1.live.check_box.check()
        assert w1.engine.check_weight is not None
        w1.live.save_session()
        w1.live.force_source = None
        w1.engine.force = None
    finally:
        w1.close()
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["session"]            # the real file
    assert saved["check_weight_kg"] == 25.5
    assert "zero" not in json.dumps(saved) and "check_weight" not in saved                   # the result (made against a zero) is not remembered
    w2 = MainWindow(TESTS, tmp_path)
    try:
        assert w2.live.check_box.mass_kg == 25.5 and w2.live.check_box.edit.text() == "25.5"
        assert w2.engine.check_weight is None                                                # a new run starts without a check
    finally:
        w2.close()


def test_a_remembered_mass_that_is_no_longer_usable_is_ignored(win):
    win.live.restore_session({"check_weight_kg": 5.0, "format": session_memory.FORMAT, "saved_at": time.time()})
    assert win.live.check_box.mass_kg is None and win.live.check_box.edit.text() == ""
    win.live.restore_session({"check_weight_kg": "heavy", "format": session_memory.FORMAT, "saved_at": time.time()})
    assert win.live.check_box.mass_kg is None


def test_the_buttons_are_off_while_recording(win):
    live = win.live
    connect(win, source(volts(load_n=LOAD_N)))
    live.check_box.refresh(False)
    assert live.check_box.button.isEnabled() and live.check_box.edit.isEnabled()
    live.check_box.refresh(True)
    assert not live.check_box.button.isEnabled() and not live.check_box.edit.isEnabled()
    live.force_source = None
    win.engine.force = None


def test_the_window_switches_the_check_weight_box_off_while_recording_and_on_again(win):
    """T6: not the box's own refresh(True), but the window's refresh_status with a recording running."""
    live = win.live
    connect(win, source(volts(load_n=LOAD_N)))
    live.refresh_status()
    assert live.check_box.button.isEnabled() and live.check_box.edit.isEnabled()
    live._rec_started = time.perf_counter()                                                   # a recording is running (the window's own flag)
    try:
        live.refresh_status()
        assert not live.check_box.button.isEnabled() and not live.check_box.edit.isEnabled()
    finally:
        live._rec_started = None
    live.refresh_status()
    assert live.check_box.button.isEnabled() and live.check_box.edit.isEnabled()
    assert not [r for r in log_lines(win) if r["event"] == "check weight refused"]
    live.force_source = None
    win.engine.force = None
