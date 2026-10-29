"""0.20.0: the resting force before the push (SOP 9.10) and the 10 % check after each trial (SOP 10.6).

Until 0.19.0 the resting force was the second before the window 'force above half of peak', which holds about half of the ramp: on the
P11 trials the peak above rest came out 6-16 % too low. The 10 % check was only in the export."""
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.recorder import count_trials, trial_metas  # noqa: E402
from limbtrack.review import REST_FROM_S, REST_TO_S, Trial, after_trial_check  # noqa: E402

from tests import test_force as T  # noqa: E402

N_PER_V = 500.0


def with_force(tr, t, newtons):
    """Give a saved trial a known force (newtons; the volts follow from a made-up slope)."""
    tr.force_t, tr.force_n, tr.force_v = t, np.asarray(newtons, float), np.asarray(newtons, float) / N_PER_V
    tr.t = np.linspace(float(t[0]), float(t[-1]), len(tr.t))
    return tr


def ramp(t, rest, start, top, ramp_s=1.0, hold_end=11.0):
    """rest, a linear ramp of ramp_s from `start` to `top`, the hold, release over 0.5 s."""
    return np.interp(t, [t[0], start, start + ramp_s, hold_end, hold_end + 0.5, t[-1]], [rest, rest, top, top, rest, rest])


@pytest.fixture
def trial(tmp_path):
    tr, _ = T.humming_trial(tmp_path, hum=0.0)
    return tr


# ------------------------------------------------------------------------------------------------ the resting force
def test_the_rest_is_taken_before_the_push_not_inside_the_ramp(trial):
    t = np.arange(-1.0, 16.0, 0.001)
    tr = with_force(trial, t, ramp(t, 50.0, 5.0, 400.0))
    ri = tr.force_rest()
    assert ri["level"] == pytest.approx(50.0, abs=0.5) and ri["note"] == ""
    assert ri["onset_s"] == pytest.approx(5.1, abs=0.05)                       # where the force was last within 10 % of its rise
    assert ri["seconds"] == pytest.approx(REST_FROM_S - REST_TO_S, abs=0.01)
    assert (ri["t_from"], ri["t_to"]) == pytest.approx((ri["onset_s"] - REST_FROM_S, ri["onset_s"] - REST_TO_S))
    row = analysis.session_row(tr)
    assert row["window"] == "force above half of peak" and row["window_start_s"] == pytest.approx(5.5, abs=0.05)
    assert row["force_rest"] == pytest.approx(50.0, abs=0.5) and row["force_rest_s"] == pytest.approx(2.5, abs=0.01)
    assert row["force_peak_above_rest"] == pytest.approx(350.0, abs=1.0)
    # the old rule, the mean of the second before the window, would have taken half of the ramp: about 94 N instead of 50
    old = float(np.mean(tr.force_signal()[1][(t >= row["window_start_s"] - 1.0) & (t < row["window_start_s"])]))
    assert old > 85.0


def test_the_angle_at_rest_is_taken_from_the_same_stretch(trial):
    t = np.arange(-1.0, 16.0, 0.001)
    tr = with_force(trial, t, ramp(t, 50.0, 5.0, 400.0))
    tr.deg["knee"] = np.where(tr.t < 5.0, 70.0, 64.0)                          # the knee gives 6 deg as soon as the push starts
    main = next(r for r in tr.summary(*tr.force_window()) if r["angle"] == tr.primary_angle.title)
    assert main["rest_mean_deg"] == pytest.approx(70.0, abs=0.01) and main["rest_to_hold_shift_deg"] == pytest.approx(-6.0, abs=0.01)


def test_a_strap_loaded_before_the_push_no_longer_hides_the_push(trial):
    """P11 12:41: 85-160 N for 5 s before the push; the onset used to be put at the first sample and the rise time at 6 s."""
    t = np.arange(0.0, 16.0, 0.001)
    pre = 100.0 + 25.0 * np.cos(1.4 * np.pi * t)                               # 75-125 N, at its low point (75 N) when the push begins
    tr = with_force(trial, t, np.where(t < 5.0, pre, ramp(t, 75.0, 5.0, 250.0)))
    ri = tr.force_rest()
    assert 4.9 < ri["onset_s"] < 5.3 and "not steady" in ri["note"]
    ft = analysis.force_timing(tr, *tr.force_window())
    assert ft["onset_s"] == pytest.approx(ri["onset_s"], abs=0.01) and ft["time_to_90pct_s"] < 1.2
    assert any(lvl == "check" and "not steady" in text for lvl, text in analysis.trial_checks(tr))


@pytest.mark.parametrize("start, known, seconds", [(1.0, True, 0.6), (0.4, False, 0.0)])
def test_a_short_rest_is_no_longer_reported_and_too_short_a_rest_gives_no_level(trial, start, known, seconds):
    """0.24.0: the SOP's 3 s of rest after Record is replaced by the resting force taken before the trial (Zero 2, with a reminder in the
    Live tab). The numbers stay (level, seconds of rest, peak above rest) but nothing says "only X s at rest" any more."""
    t = np.arange(0.0, 12.0, 0.001)
    tr = with_force(trial, t, ramp(t, 20.0, start, 300.0))
    ri = tr.force_rest()
    assert ri["seconds"] == pytest.approx(seconds, abs=0.05)
    assert ri["note"] == ""                                                      # no note about a short rest, however short
    if known:
        assert ri["level"] == pytest.approx(20.0, abs=0.5)
    else:
        assert math.isnan(ri["level"])
        assert analysis.session_row(tr)["force_peak_above_rest"] != analysis.session_row(tr)["force_peak_above_rest"]   # NaN
    assert not any("at rest" in text or "resting force is not known" in text for _, text in analysis.trial_checks(tr))


def test_review_says_where_the_rest_came_from(trial, tmp_path):
    from limbtrack.review import ReviewTab
    QApplication.instance() or QApplication([])
    t = np.arange(-1.0, 16.0, 0.001)
    tr = with_force(trial, t, ramp(t, 50.0, 5.0, 400.0))
    tab = ReviewTab(tmp_path)
    tab.trial = tr
    text = tab._force_text(*tr.force_window())
    assert "rest before" in text and f"median {REST_FROM_S:g} to {REST_TO_S:g} s before the push, 2.5 s of force" in text


# ------------------------------------------------------------------------------------------------ the 10 % rule on export rows
def rows(*peaks, unit="N", above=None):
    return [{"force_peak": p, "force_unit": unit, "force_peak_above_rest": "" if above is None else above[i]} for i, p in enumerate(peaks)]


def test_two_trials_within_and_beyond_ten_percent():
    ok = analysis.position_check(rows(400.0, 380.0))
    assert ok["state"] == "ok" and ok["pct"] == pytest.approx(5.13, abs=0.01) and "No third trial needed" in ok["text"]
    third = analysis.position_check(rows(400.0, 340.0, above=[380.0, 330.0]))
    assert third["state"] == "third" and third["pct"] == pytest.approx(16.2, abs=0.1)
    assert "THIRD TRIAL NEEDED" in third["text"] and "trial 1 400 N, trial 2 340 N; above rest 380 N, 330 N" in third["text"]
    assert analysis.position_check(rows(400.0, 362.0))["state"] == "ok"          # 9.97 %: the limit itself is fine
    # the same number as the export's repeatability column
    r = [{"test_key": "k", "target_deg": 1, **x} for x in rows(400.0, 340.0)]
    analysis.repeatability(r)
    assert r[0]["repeatability_pct_same_position"] == pytest.approx(third["pct"]) and r[0]["third_trial_needed"] is True


def test_three_trials_are_judged_on_the_closest_two():
    good = analysis.position_check(rows(400.0, 340.0, 395.0))
    assert good["state"] == "ok" and "trials 1 and 3 differ by 1.3 %" in good["text"]
    bad = analysis.position_check(rows(400.0, 340.0, 290.0))
    assert bad["state"] == "deviation" and "deviation log" in bad["text"] and "closest: trials 2 and 3" in bad["text"]
    four = analysis.position_check(rows(400.0, 340.0, 290.0, 395.0))
    assert "4 trials are saved at this position (SOP 10.6: at most 3)" in four["text"]


def test_one_trial_mixed_units_and_no_force():
    assert analysis.position_check(rows(400.0))["state"] == "one"
    mixed = rows(400.0) + rows(0.8, unit="V")
    assert analysis.position_check(mixed)["state"] == "units" and "trial 2" in analysis.position_check(mixed)["text"]
    assert analysis.position_check([{"force_peak": "", "force_unit": ""}])["state"] == "none"


# ------------------------------------------------------------------------------------------------ on real saved trials
def test_after_a_trial_the_saved_trials_of_its_position_are_compared(tmp_path):
    root = tmp_path / "recordings"
    _, a = T.recording(root, seconds=3.0)                                      # knee extension 70, P01, simulated 400 N
    first = after_trial_check(a["meta.json"])
    assert first["n"] == 1 and first["check"]["state"] == "one" and (first["participant"], first["test_key"]) == ("P01", "knee_extension")
    time.sleep(1.1)                                                            # (a new file name: they carry the second)
    _, b = T.recording(root, seconds=3.0, peak_n=330.0)
    res = after_trial_check(b["meta.json"])
    assert res["n"] == 2 == count_trials(root, "P01", "knee_extension", 70.0)
    assert res["check"]["state"] == "third" and res["check"]["pct"] == pytest.approx(19.2, abs=2.0)
    assert res["trial"] == Path(b["meta.json"]).name[: -len("_meta.json")]
    assert "at rest" not in res["rest_note"]              # the simulated push begins about when Record is pressed, but since 0.24.0 a short rest is not reported
    # the same numbers as 'Export all trials of this participant'
    exported = [analysis.session_row(Trial(p)) for p in trial_metas(root, "P01", "knee_extension", 70.0)]
    analysis.repeatability(exported)
    assert exported[0]["repeatability_pct_same_position"] == pytest.approx(res["check"]["pct"])


def test_trials_started_in_the_same_second_are_listed_in_order(tmp_path):
    folder = tmp_path / "P01"
    folder.mkdir()
    for stem in ("20260101-000001-2", "20260101-000001", "20260101-000000", "20260101-000001-10"):
        (folder / f"P01_knee_extension_20deg_{stem}_meta.json").write_text("{}")
    names = [p.name.split("20deg_")[1][:-10] for p in trial_metas(tmp_path, "P01", "knee_extension", 20)]
    assert names == ["20260101-000000", "20260101-000001", "20260101-000001-2", "20260101-000001-10"]
    assert count_trials(tmp_path, "P01", "knee_extension", 20) == 4          # (a same-second trial was not counted before 0.20.0)


# ------------------------------------------------------------------------------------------------ in the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow(T.TESTS, tmp_path)
    yield app, w
    w.close()


def pump(app, cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_the_window_says_third_trial_needed_and_marks_the_plan(win, tmp_path, monkeypatch):
    app, w = win
    live = w.live
    shown = []
    monkeypatch.setattr(live, "_show_trial_warnings", lambda warnings, log=True: shown.append(list(warnings)))
    root = live.recordings_dir
    _, a = T.recording(root, seconds=3.0)
    time.sleep(1.1)
    _, b = T.recording(root, seconds=3.0, peak_n=330.0)
    live.participant.setText("P01")
    live.go_to_position(live.plan_rows.index(("knee_extension", 70)))
    live.rec_label.setText("Saved: ...")
    key = live._position_key("P01", "knee_extension", 70.0)
    assert pump(app, lambda: key not in live._repeat_running)                  # (going to the position starts a quiet check)
    live._check_position(b["meta.json"], key, announce=True)
    assert pump(app, lambda: key not in live._repeat_running and live._repeat.get(key, (0,))[0] == 2)
    assert "THIRD TRIAL NEEDED" in live.rec_label.text() and shown and "THIRD TRIAL NEEDED" in shown[-1][-1]
    row = live.plan.item(live.plan_rows.index(("knee_extension", 70))).text()
    assert row.strip().startswith("3rd!") and "THIRD TRIAL NEEDED" in live.trial_label.text()
    log = (root / "session_log.csv").read_text(encoding="utf-8")
    assert "position check" in log and "THIRD TRIAL NEEDED" in log


def test_after_a_restart_the_position_is_checked_quietly_when_the_window_goes_there(win, monkeypatch):
    app, w = win
    live = w.live
    shown = []
    monkeypatch.setattr(live, "_show_trial_warnings", lambda warnings, log=True: shown.append(warnings))
    root = live.recordings_dir
    T.recording(root, seconds=3.0)
    time.sleep(1.1)
    T.recording(root, seconds=3.0, peak_n=330.0)
    live.participant.setText("P01")
    live.go_to_position(live.plan_rows.index(("knee_extension", 70)))
    key = live._position_key("P01", "knee_extension", 70.0)
    assert pump(app, lambda: live._repeat.get(key, (0,))[0] == 2)
    assert live.plan.item(live.plan_rows.index(("knee_extension", 70))).text().strip().startswith("3rd!")
    assert not shown                                                           # quiet: no message for trials of an earlier run


def test_a_check_that_fails_is_said_and_not_repeated(win, tmp_path):
    app, w = win
    live = w.live
    bad = tmp_path / "recordings" / "P05" / "P05_knee_extension_70deg_20260101-000000_meta.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps({"participant": "P05", "test_key": "knee_extension", "target_deg": 70}))   # no data files
    key = live._position_key("P05", "knee_extension", 70)
    live.rec_label.setText("Saved: ...")
    live._check_position(bad, key, announce=True)
    assert pump(app, lambda: key in live._repeat)
    assert live._repeat[key][1]["state"] == "error" and "could not be made" in live.rec_label.text()
