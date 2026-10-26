"""0.19.0, from the review of the P20 session (2026-10-22): two dots on one bone that move apart are caught (a tracker had jumped to
something dark next to its dot in 2 of 6 ankle trials without any warning), the label checks are saved per picture, the force peak leaves
the 50 Hz hum out, a trial whose force never rose is called out, and the frontal tests work with the participant lying on the side."""
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack import review  # noqa: E402
from limbtrack.analysis import trial_checks  # noqa: E402
from limbtrack.engine import NO_FORCE_RISE_N, RIGID_PAIR_CHANGE, Engine  # noqa: E402
from limbtrack.force import ForceCalibration  # noqa: E402
from limbtrack.geometry import pelvis_axes  # noqa: E402
from limbtrack.protocol import load_protocol, validate_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import frontal_points, leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PAIR = ("mt5_base", "mt5_head")


def leg():
    return {k: v.copy() for k, v in leg_points(hip_flex=0.0, knee_flex=0.0, ankle_df=0.0, foot=60.0).items()}   # a 5th metatarsal about a third of the shank, as on a person


def ankle_engine(tmp_path, swap=()):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("ankle_dorsiflexion")
    p = leg()
    eng.process(render(p, seed=0), 0.0)
    where = dict(p)
    for a, b in swap:
        where[a], where[b] = p[b], p[a]
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(where[n] + 2))
    return eng, p


def toe_moved(p, px):
    """The same leg with the 5th metatarsal head dot `px` pixels further along the foot (as if its tracker had slid onto something)."""
    q = dict(p)
    u = (p["mt5_head"] - p["mt5_base"]) / np.hypot(*(p["mt5_head"] - p["mt5_base"]))
    q["mt5_head"] = p["mt5_head"] + px * u
    return q


def drift_toe(eng, p, total_px, t0, steps=12, record=None):
    """Move the toe dot outwards a few pixels per picture (the tracker follows it) and return the last frame result."""
    res = None
    for k in range(1, steps + 1):
        res = eng.process(render(toe_moved(p, total_px * k / steps), seed=k), t0 + k / 30)
    return res


def angle(res, name):
    return next(a for a in res.angles if a.name == name)


# ------------------------------------------------------------------------------------------------ dots on one bone
def test_every_test_checks_the_pairs_of_its_own_dots():
    assert TESTS["ankle_dorsiflexion"].rigid_pairs == [("fibular_head", "malleolus"), PAIR, ("trochanter", "epicondyle")]
    assert TESTS["hip_abduction"].rigid_pairs == [("asis", "asis_opposite")]
    assert all(("iliac_crest", "trochanter") not in t.rigid_pairs for t in TESTS.values())      # spans the hip: changes with the hip angle


def test_a_toe_dot_that_moves_away_from_its_base_marks_the_ankle_and_blocks_lock(tmp_path):
    eng, p = ankle_engine(tmp_path)
    length = float(np.hypot(*(p["mt5_head"] - p["mt5_base"])))
    res = eng.process(render(p, seed=1), 0.05)
    assert eng.dots_apart() == {} and all(a.suspect is None for a in res.angles)
    assert eng.lock_position()
    eng.unlock_position()
    res = drift_toe(eng, p, 0.5 * length, 0.1)                       # +50 %, as in the dorsiflexion -20 trial of P20
    apart = eng.dots_apart()
    assert set(apart) == {PAIR} and apart[PAIR] == pytest.approx(0.5, abs=0.08)
    assert "farther apart" in angle(res, "ankle").suspect
    assert angle(res, "knee").suspect is None and angle(res, "hip").suspect is None   # they use no foot dot
    assert not math.isnan(angle(res, "ankle").raw_value)                               # the value is kept
    assert any(h.startswith("CHECK DOTS") and "5th metatarsal" in h for h in res.header)
    assert eng.lock_position() is False and "farther apart" in eng.lock_problem()
    assert "may be wrong" in eng.main_angle_problem()
    q = toe_moved(p, 0.5 * length)                                    # marking the toe dot again where it is now: a new reference
    assert eng.mark_dot("mt5_head", *(q["mt5_head"] + 2))
    res = eng.process(render(q, seed=20), 1.0)
    assert eng.dots_apart() == {} and angle(res, "ankle").suspect is None and eng.lock_position()


def test_a_small_change_is_not_called_a_jump(tmp_path):
    eng, p = ankle_engine(tmp_path)
    length = float(np.hypot(*(p["mt5_head"] - p["mt5_base"])))
    res = drift_toe(eng, p, 0.6 * RIGID_PAIR_CHANGE * length, 0.1)    # skin moving or the foot turning a little out of the picture's plane
    assert eng.dots_apart() == {} and angle(res, "ankle").suspect is None


def test_the_check_is_saved_per_picture_and_summed_up_in_meta_and_review(tmp_path, monkeypatch):
    eng, p = ankle_engine(tmp_path)
    length = float(np.hypot(*(p["mt5_head"] - p["mt5_base"])))
    eng.target = 0.0
    paths = eng.start_trial("P01")
    for i in range(6):
        eng.process(render(p, seed=i), 0.1 + i / 30)
    drift_toe(eng, p, 0.5 * length, 0.4)
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    flagged = [r for r in rows if r["dots_apart"]]
    assert rows[0]["dots_apart"] == "" and rows[0]["label_check"] == ""
    assert flagged and all(r["dots_apart"].startswith("mt5_base-mt5_head:+") and r["all_ok"] == "0" for r in flagged)
    assert all(r["ankle_deg"] != "" for r in flagged)                  # the angle itself is still saved
    meta = json.loads(paths["meta.json"].read_text())
    c = meta["checks"]
    assert c["dots_apart_frames"]["mt5_base-mt5_head"]["frames"] == len(flagged) and c["label_check_frames"] == {}
    assert ["mt5_base", "mt5_head"] in c["rigid_pairs"] and c["rigid_pair_reference_px"]["mt5_base-mt5_head"] == pytest.approx(length, abs=3)
    assert c["rigid_pair_change_limit_pct"] == 25 and "knee_order" in c["label_checks"]
    assert any("CHECK DOTS" in w for w in eng.last_trial_warnings)
    tr = Trial(paths["meta.json"])
    assert any(level == "check" and "moved apart" in text for level, text in trial_checks(tr))
    drawn = []
    monkeypatch.setattr(review.overlay, "draw", lambda img, dots, angles, *a, **k: drawn.append(angles) or img)
    tr.drawn(len(rows) - 1)
    tr.drawn(0)
    assert next(a for a in drawn[0] if a.name == "ankle").suspect and next(a for a in drawn[0] if a.name == "knee").suspect is None
    assert next(a for a in drawn[1] if a.name == "ankle").suspect is None


def test_a_clean_trial_has_empty_check_columns(tmp_path):
    eng, p = ankle_engine(tmp_path)
    paths = eng.start_trial("P01")
    for i in range(8):
        eng.process(render(p, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert all(r["dots_apart"] == "" and r["label_check"] == "" for r in rows)
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["checks"]["dots_apart_frames"] == {} and meta["checks"]["label_check_frames"] == {} and eng.last_trial_warnings == []


def test_swapped_knee_dots_are_saved_in_every_picture_they_are_shown(tmp_path):
    eng, p = ankle_engine(tmp_path, swap=(("epicondyle", "fibular_head"),))
    paths = eng.start_trial("P01")
    for i in range(6):
        res = eng.process(render(p, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    assert any(h.startswith("CHECK LABELS") for h in res.header)
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert all(r["label_check"] == "knee_order" for r in rows)
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["checks"]["label_check_frames"]["knee_order"]["frames"] == len(rows)
    assert any("CHECK LABELS" in w for w in eng.last_trial_warnings)
    assert any("CHECK LABELS in" in text for _, text in trial_checks(Trial(paths["meta.json"])))


def test_the_main_angle_problem_names_a_lost_dot(tmp_path):
    eng, p = ankle_engine(tmp_path)
    assert eng.main_angle_problem() is None
    hidden = render(p, seed=3, hide=("mt5_base",))
    for i in range(40):
        eng.process(hidden, 0.1 + i / 30)
    assert "cannot be measured" in eng.main_angle_problem() and "5th metatarsal base" in eng.main_angle_problem()


@pytest.mark.parametrize("pairs, word", [
    ([["mt5_base"]], "two dot names"),
    ([["mt5_base", "mt5_base"]], "both dots"),
    ([["mt5_base", "toe_tip"]], "unknown dot"),
    ([["asis", "hip_centre"]], "computed point"),
    ("mt5_base", "must be a list"),
])
def test_wrong_rigid_pairs_are_reported(pairs, word):
    data = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    data["rigid_pairs"] = pairs
    errs = validate_protocol(data)
    assert any(word in e for e in errs), errs


# ------------------------------------------------------------------------------------------------ force
class FakeForce:
    """A force source on the picture clock with a known signal: `newtons(seconds since t0)` plus a 50 Hz hum of `hum_v` volts."""
    kind = "fake"
    rate, range_v, running, error, clipped_total = 1000.0, 10.0, True, None, 0

    def __init__(self, t0, seconds, newtons, hum_v=0.0, n_per_v=500.0, offset_v=0.1):
        self.frozen = False
        self.calibration = ForceCalibration(offset_v=offset_v, n_per_v=n_per_v, method="test")
        self.t = t0 + np.arange(int(seconds * self.rate)) / self.rate
        self.v = offset_v + newtons(self.t - t0) / n_per_v + hum_v * np.sin(2.0 * np.pi * 50.0 * self.t)

    def samples_between(self, a, b):
        m = (self.t >= a) & (self.t < b)
        return self.t[m], self.v[m]

    def seconds_since_last_sample(self):
        return 0.0

    def latest_volts(self):
        return float(self.v[0])

    def latest_newtons(self):
        return float(self.calibration.to_newton(self.v[0]))

    def clipping_now(self):
        return False

    def info(self):
        return {"source": self.kind, "sample_rate_hz": self.rate, "calibration": self.calibration.to_dict(), "calibrated": self.calibration.calibrated}


def contraction(peak):
    """Rest until 3 s, a 0.5 s ramp, a hold until 6 s, release."""
    return lambda s: peak * np.clip((s - 3.0) / 0.5, 0.0, 1.0) * (s < 6.0)


def force_trial(tmp_path, newtons, hum_v=0.0, markers=True, end_marker=True, seconds=12.0):
    eng, p = ankle_engine(tmp_path)
    img = render(p, seed=0)
    T = time.perf_counter()
    eng.force = FakeForce(T - 2.0, seconds, newtons, hum_v=hum_v)    # the signal began 2 s before now: 'now' is 2 s into it
    paths = eng.start_trial("P01")
    res = None
    for i in range(150):                                               # 5 s of pictures from 0.1 s after now
        if markers and i == 25:
            eng.add_event("MVIC start")                                # (written on the next picture: ~1.0 s after now, 3.0 s into the signal)
        if markers and end_marker and i == 140:
            eng.add_event("MVIC end")
        res = eng.process(img, T + 0.1 + i / 30)
    eng.stop_trial()
    return eng, json.loads(paths["meta.json"].read_text()), paths, res


def test_the_peak_leaves_the_mains_hum_out(tmp_path):
    eng, meta, paths, res = force_trial(tmp_path, lambda s: 0.0 * s, hum_v=0.5)    # no load at all, 0.5 V of hum: P20
    f = meta["force"]
    assert f["peak_raw_n"] == pytest.approx(250.0, abs=5.0)            # 0.5 V x 500 N/V: what the 'peak' said until 0.18.2
    assert abs(f["peak_n"]) < 2.0 and "hum is not counted" in f["peak_meaning"]
    assert not any("peak 2" in h for h in res.header)                 # the live read-out shows the hum-free peak too


def test_a_trial_without_a_force_rise_is_called_out(tmp_path):
    eng, meta, paths, _ = force_trial(tmp_path, lambda s: 0.0 * s, hum_v=0.5)
    f = meta["force"]
    assert f["no_rise_warning"] is True and f["rise_n"] < 5.0 and f["no_rise_limit_n"] == NO_FORCE_RISE_N
    assert f["rise_window_t_s"][0] == pytest.approx(0.9, abs=0.1)
    assert any("No force rise" in w for w in eng.last_trial_warnings)
    assert any("hardly moved" in text for _, text in trial_checks(Trial(paths["meta.json"])))


def test_the_hum_at_the_very_end_of_the_force_is_not_taken_for_a_rise(tmp_path):
    """No end marker and the force stops with the last picture: the average must not be taken over a window cut off by the end (the edge
    value repeated there leaves part of the 0.5 V hum in, about 100 N)."""
    eng, meta, paths, _ = force_trial(tmp_path, lambda s: 0.0 * s, hum_v=0.5, end_marker=False, seconds=2.0 + 0.1 + 149 / 30)
    f = meta["force"]
    assert f["no_rise_warning"] is True and f["rise_n"] < 5.0


def test_a_real_contraction_is_not_called_out(tmp_path):
    eng, meta, paths, _ = force_trial(tmp_path, contraction(300.0), hum_v=0.05)
    f = meta["force"]
    assert f["no_rise_warning"] is False and f["rise_n"] == pytest.approx(300.0, abs=10.0)
    assert f["peak_n"] == pytest.approx(300.0, abs=3.0) and f["peak_raw_n"] == pytest.approx(325.0, abs=5.0)
    assert eng.last_trial_warnings == []
    assert not any("hardly moved" in text for _, text in trial_checks(Trial(paths["meta.json"])))


def test_no_start_marker_no_rise_check(tmp_path):
    eng, meta, paths, _ = force_trial(tmp_path, lambda s: 0.0 * s, markers=False)
    assert "rise_n" not in meta["force"] and "no_rise_warning" not in meta["force"] and eng.last_trial_warnings == []


# ------------------------------------------------------------------------------------------------ frontal tests lying on the side
def test_down_is_towards_the_knee_when_the_knee_is_given():
    t, o = (100.0, 100.0), (100.0, 350.0)                              # the line between the ASIS upright in the picture: lying on the side
    for knee, sign in (((500.0, 220.0), 1.0), ((-300.0, 220.0), -1.0)):
        u, v, width = pelvis_axes(t, o, knee)
        assert width == pytest.approx(250.0) and v[0] == pytest.approx(sign)
    u, v, _ = pelvis_axes((100.0, 100.0), (350.0, 100.0))               # no knee: the picture's bottom, as before
    assert v[1] == pytest.approx(1.0)


def frontal_engine(tmp_path, roll, leg_side="right", deg=12.0, test="hip_abduction"):
    p = dict(frontal_points(deg, tested_leg=leg_side, roll_deg=roll, centre=(640.0, 360.0)))
    p["ruler_a"], p["ruler_b"] = np.array([120.0, 690.0]), np.array([270.0, 690.0])
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.tested_leg = leg_side
    img = render(p, dot_radius=11.0, seed=0)
    eng.process(img, 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    res = None
    for i in range(6):
        res = eng.process(render(p, dot_radius=11.0, seed=i), 0.1 + i / 30)
    return eng, p, res


@pytest.mark.parametrize("roll", [90.0, -90.0, 80.0, 100.0, -75.0])
@pytest.mark.parametrize("leg_side", ["right", "left"])
def test_lying_on_the_side_gives_the_same_angles_and_no_false_label_warning(tmp_path, roll, leg_side):
    eng, p, res = frontal_engine(tmp_path, roll, leg_side)
    hjc = angle(res, "abd_hip_centre")
    assert hjc.raw_value == pytest.approx(12.0, abs=0.3)               # the hip centre on the knee side: the thigh's true turn about it
    a, b = p["asis_opposite"] - p["asis"], p["patella"] - p["asis"]
    by_hand = math.degrees(math.acos(float(np.dot(a, b)) / (np.hypot(*a) * np.hypot(*b)))) - 90.0
    assert angle(res, "abd_goniometer").raw_value == pytest.approx(by_hand, abs=0.3)
    assert eng.marking_problems() == [] and eng.lock_position()


@pytest.mark.parametrize("roll", [90.0, -90.0])
def test_lying_on_the_side_with_the_other_tested_leg_set_is_caught(tmp_path, roll):
    eng, p, res = frontal_engine(tmp_path, roll, "right")
    eng.tested_leg = "left"
    for i in range(3):
        eng.process(render(p, dot_radius=11.0, seed=10 + i), 1.0 + i / 30)
    items = eng.marking_problem_items()
    assert [c for c, _ in items] == ["asis_side"] and "lying on the side" in items[0][1]
    assert not eng.lock_position()


def test_side_lying_trials_save_the_hip_centre_on_the_knee_side(tmp_path):
    eng, p, _ = frontal_engine(tmp_path, 90.0, "right", deg=15.0)
    paths = eng.start_trial("P01")
    for i in range(5):
        eng.process(render(p, dot_radius=11.0, seed=i), 0.2 + i / 30)
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    mid = (p["asis"] + p["asis_opposite"]) / 2.0
    for r in rows:
        h = np.array([float(r["hip_centre_x"]), float(r["hip_centre_y"])])
        assert float(np.dot(h - mid, p["patella"] - mid)) > 0          # towards the knee, not the head
        assert float(r["abd_hip_centre_deg"]) == pytest.approx(15.0, abs=0.3)
