"""The angle reference (0.33.0): "the joint is at 80 now, the dots read 90". The pure part (limbtrack/angle_reference.py), the engine (the target on the
dots' scale, the trace, the picture, meta.json and data.csv), the Live tab's box, and Review (a trial recorded with a reference, one given in
Review from a typed reading, the angle table, the graphs, the export row and the overview)."""
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis, overlay, reference  # noqa: E402
from limbtrack import angle_reference as ar  # noqa: E402
from limbtrack.angle_reference_box import AngleReferenceBox  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import SimulatedForce  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.results import OUT  # noqa: E402
from limbtrack.review import ReferenceDialog, ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)          # the knee reads 60, the hip 90
OWNER = "P01"


def _app():
    return QApplication.instance() or QApplication([])


def engine(tmp_path, target=60.0, frames=8):
    """A knee_extension engine at `target` with every dot marked (the knee is the main joint, the hip must be 90)."""
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = target
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(frames):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    return eng


def frames(eng, n=6, t0=1.0):
    res = None
    for i in range(n):
        res = eng.process(render(PTS, seed=i), t0 + i / 30)
    return res


def angle(res, name):
    return next(a for a in res.angles if a.name == name)


def record(eng, participant="P01", n=12):
    paths = eng.start_trial(participant)
    for i in range(n):
        eng.process(render(PTS, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    return paths["meta.json"]


# ------------------------------------------------------------------------------------------------ the pure part
def test_the_offset_is_what_the_dots_read_minus_what_was_said():
    r = ar.AngleReference()
    e = r.set(OWNER, "hip", 0.0, 10.0)                                   # in neutral: called 0, the dots read 10
    assert e["offset_deg"] == 10.0 and r.offsets(OWNER) == {"hip": 10.0}
    r.set(OWNER, "hip", 90.0, 45.0)                                      # an inclinometer in the test posture: it says 90, the dots read 45
    assert r.offsets(OWNER) == {"hip": -45.0}


def test_a_reference_belongs_to_one_participant_and_the_first_code_adopts_an_early_one():
    r = ar.AngleReference()
    r.set("", "hip", 80.0, 90.0)                                         # taken before a participant code was typed
    assert r.offsets("P01") == {"hip": 10.0}                             # whoever comes first
    r.adopt("P01")
    assert r.owner == "P01" and r.offsets("P02") == {} and r.offsets("P01") == {"hip": 10.0}
    r.adopt("P02")                                                       # an earlier code is kept
    assert r.owner == "P01"
    r.set("P02", "knee", 55.0, 60.0)                                     # another participant: the old one is dropped first
    assert r.offsets("P02") == {"knee": 5.0} and r.offsets("P01") == {}


def test_clearing_one_angle_or_all():
    r = ar.AngleReference()
    r.set(OWNER, "hip", 80.0, 90.0)
    r.set(OWNER, "knee", 55.0, 60.0)
    r.clear("hip")
    assert r.offsets(OWNER) == {"knee": 5.0}
    r.clear("knee")
    assert r.offsets(OWNER) == {} and r.owner is None
    r.set(OWNER, "hip", 80.0, 90.0)
    r.clear()
    assert r.to_meta(OWNER) is None


@pytest.mark.parametrize("bad", ["x", None, float("nan"), float("inf"), 400.0, -181.0])
def test_a_reference_that_is_not_an_angle_is_refused(bad):
    with pytest.raises(ar.AngleReferenceError):
        ar.check_value(bad)
    with pytest.raises(ValueError):
        ar.AngleReference().set(OWNER, "hip", bad, 90.0)


def test_meta_keeps_only_the_angles_asked_for():
    r = ar.AngleReference()
    r.set(OWNER, "hip", 80.0, 90.0)
    r.set(OWNER, "ankle", 0.0, 8.0)
    assert set(r.to_meta(OWNER)["angles"]) == {"hip", "ankle"}
    assert set(r.to_meta(OWNER, names={"hip", "knee"})["angles"]) == {"hip"}
    assert r.to_meta(OWNER, names={"knee"}) is None


def test_meta_round_trip():
    r = ar.AngleReference()
    assert r.to_meta(OWNER) is None and ar.offsets_from_meta({}) == {} and ar.offsets_from_meta({"angle_reference": None}) == {}
    r.set(OWNER, "hip", 80.0, 90.25)
    meta = {"angle_reference": r.to_meta(OWNER)}
    assert meta["angle_reference"]["angles"]["hip"]["offset_deg"] == 10.25 and "minus the offset" in meta["angle_reference"]["note"]
    assert ar.offsets_from_meta(meta) == {"hip": 10.25}
    assert ar.offsets_from_meta({"angle_reference": {"angles": {"hip": {"offset_deg": "x"}, "knee": {}}}}) == {}       # a damaged file costs the offset, not the trial


# ------------------------------------------------------------------------------------------------ the engine
def test_a_reference_moves_the_target_onto_the_dots_scale_and_leaves_the_dots_angle_alone(tmp_path):
    eng = engine(tmp_path)
    dots = eng._ref_last["hip"]
    got = eng.set_angle_reference("hip", 80.0)                           # the inclinometer says 80 while the dots read about 90
    off = got["offset_deg"]
    assert got["dots_deg"] == pytest.approx(dots) and off == pytest.approx(dots - 80.0) and off == pytest.approx(10.0, abs=1.0)
    res = frames(eng)
    hip, knee = angle(res, "hip"), angle(res, "knee")
    assert hip.ref_offset == pytest.approx(off) and knee.ref_offset == 0.0
    assert hip.value == pytest.approx(90.0, abs=1.0) and hip.raw_value == pytest.approx(hip.value, abs=0.2)     # still the dots' angle
    assert hip.reference == pytest.approx(90.0 + off)                    # the required hip angle (90) on the dots' scale
    assert hip.deviation == pytest.approx(hip.value - off - 90.0)       # what the screen shows (value - offset) against 90
    assert hip.raw_deviation == pytest.approx(hip.raw_value - hip.reference)
    assert knee.reference == pytest.approx(60.0)                         # an angle without a reference: as before


def test_the_main_joints_target_is_asked_for_from_the_reference_and_survives_a_change_of_target(tmp_path):
    eng = engine(tmp_path, target=60.0)
    off = eng.set_angle_reference("knee", 55.0)["offset_deg"]            # the knee reads 60 on the dots: the joint is 55
    assert off == pytest.approx(5.0, abs=1.0)
    knee = angle(frames(eng), "knee")
    assert knee.reference == pytest.approx(60.0 + off)                    # 60 from the reference is 65 on the dots
    assert knee.deviation == pytest.approx(knee.value - off - 60.0) and knee.deviation == pytest.approx(-5.0, abs=1.0) and knee.status == OUT       # (joint 55: 5 below, tolerance 3)
    eng.target = 20.0                                                     # the step from one target to the next keeps the reference
    assert eng.ref_offsets() == {"knee": pytest.approx(off)}
    assert angle(frames(eng), "knee").reference == pytest.approx(20.0 + off)
    eng.target = None                                                     # no target: nothing to move
    assert angle(frames(eng), "knee").reference is None


def test_without_a_reference_everything_is_as_before(tmp_path):
    eng = engine(tmp_path)
    res = frames(eng)
    hip, knee = angle(res, "hip"), angle(res, "knee")
    assert hip.ref_offset == 0.0 and knee.ref_offset == 0.0 and eng.ref_offsets() == {}
    assert hip.reference == 90.0 and knee.reference == 60.0 and hip.deviation == pytest.approx(hip.value - 90.0)
    assert eng.angle_reference_state()[0][2] is None and eng._angle_reference_meta() is None


def test_it_is_cleared_when_a_dot_of_the_angle_is_marked_again_or_taken_away_or_the_dots_are_reset(tmp_path):
    eng = engine(tmp_path)
    eng.set_angle_reference("knee", 55.0)
    eng.set_angle_reference("hip", 80.0)
    assert set(eng.ref_offsets()) == {"knee", "hip"}
    assert eng.mark_dot("iliac_crest", *(PTS["iliac_crest"] + 2))        # the hip's pelvis dot: only the hip loses its reference
    assert set(eng.ref_offsets()) == {"knee"}
    eng.clear_dot("fibular_head")                                          # a dot of the knee only
    assert eng.ref_offsets() == {}
    eng = engine(tmp_path / "again")
    eng.set_angle_reference("hip", 80.0)
    eng.reset_dots()
    assert eng.ref_offsets() == {}


def test_it_belongs_to_the_participant_and_the_dots_not_to_a_test(tmp_path):
    eng = engine(tmp_path)
    eng.set_angle_reference("hip", 80.0)                                  # before any code was typed
    eng.participant_code = "P01"
    assert "hip" in eng.ref_offsets()                                      # the first code adopts it
    eng.participant_code = "P02"
    assert eng.ref_offsets() == {} and eng._angle_reference_meta() is None
    eng.participant_code = "P01"
    assert "hip" in eng.ref_offsets()
    off = eng.ref_offsets()["hip"]
    eng.set_test("hip_extension_k0")                                        # another test with the same landmarks: the dots stay, so does the reference
    assert eng.ref_offsets() == {"hip": pytest.approx(off)}
    eng.target = 45.0
    hip = angle(frames(eng), "hip")
    assert hip.primary and hip.ref_offset == pytest.approx(off) and hip.reference == pytest.approx(45.0 + off)      # the main joint of this test: 45 from the reference
    eng.set_test("hip_abduction")                                           # other landmarks: the dots are reset, the reference goes
    assert eng.ref_offsets() == {}


def test_the_meta_of_a_trial_names_only_the_angles_of_that_trial(tmp_path):
    eng = engine(tmp_path)
    eng.set_angle_reference("hip", 80.0)
    eng.set_angle_reference("knee", 55.0)
    eng.set_dot_enabled("malleolus", False)                                 # the knee angle needs it: only the hip is an angle of the trial now
    assert {a.name for a in eng.active_angles} == {"hip"}
    assert set(eng._angle_reference_meta()["angles"]) == {"hip"}


def test_a_reference_is_refused_when_it_cannot_be_taken_and_changed_by_nothing_while_recording(tmp_path):
    eng = engine(tmp_path)
    with pytest.raises(ValueError, match="not an angle of this test"):
        eng.set_angle_reference("ankle", 0.0)
    for bad in ("x", float("nan"), 999.0):
        with pytest.raises(ValueError):
            eng.set_angle_reference("hip", bad)
    assert eng.ref_offsets() == {}
    eng.pause()
    with pytest.raises(RuntimeError, match="paused"):
        eng.set_angle_reference("hip", 80.0)
    eng.resume()
    eng._frame_stamp -= 5.0                                               # the camera stopped: the last picture is old
    with pytest.raises(ValueError, match="No new picture"):
        eng.set_angle_reference("hip", 80.0)
    eng._frame_stamp += 5.0
    eng.set_angle_reference("hip", 80.0)
    eng.start_trial("P01")
    with pytest.raises(RuntimeError, match="Stop recording"):
        eng.set_angle_reference("knee", 55.0)
    with pytest.raises(RuntimeError, match="Stop recording"):
        eng.clear_angle_reference()
    eng.stop_trial()
    eng.clear_angle_reference()
    assert eng.ref_offsets() == {}


def test_a_lost_angle_cannot_be_given_a_reference(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(PTS, seed=0), 0.0)
    with pytest.raises(ValueError, match="cannot be read"):
        eng.set_angle_reference("hip", 80.0)                               # no dot is marked yet


def test_a_recorded_trial_keeps_the_dots_angle_and_the_reference_and_its_verdict_columns_agree(tmp_path):
    eng = engine(tmp_path)
    off = eng.set_angle_reference("hip", 80.0)["offset_deg"]
    meta_path = record(eng)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    entry = meta["angle_reference"]["angles"]["hip"]
    assert entry["offset_deg"] == pytest.approx(off, abs=1e-3) and entry["reference_deg"] == 80.0 and "knee" not in meta["angle_reference"]["angles"]
    assert meta["target_deg"] == 60.0 and meta["neighbour_targets_deg"] == {"hip": 90}          # the targets as asked
    rows = list(csv.DictReader(open(Path(meta_path).parent / meta["files"]["data.csv"], newline="", encoding="utf-8")))
    r = rows[6]
    assert float(r["hip_deg"]) == pytest.approx(90.0, abs=1.0)                                 # the dots' angle
    assert float(r["hip_ref"]) == pytest.approx(90.0 + off, abs=0.01)                          # the target on the dots' scale
    assert float(r["hip_dev"]) == pytest.approx(float(r["hip_deg"]) - float(r["hip_ref"]), abs=0.01)
    assert r["hip_ok"] == "0" and float(r["hip_dev"]) == pytest.approx(-10.0, abs=1.0)         # the joint is 80 against the 90 asked (+-5): off, as it was shown
    assert float(r["knee_ref"]) == 60.0
    assert not eng.angle_ref.entries or eng.rec is None                                        # (the engine still holds it for the next trial)


def test_a_trial_recorded_without_a_reference_says_so_in_meta(tmp_path):
    eng = engine(tmp_path)
    meta = json.loads(Path(record(eng)).read_text(encoding="utf-8"))
    assert meta["angle_reference"] is None
    tr = Trial(Path(record(eng)), open_video=False)
    assert tr.offsets == {} and tr.recorded_offsets == {}


def test_the_trace_and_the_picture_show_the_angle_from_the_reference_with_the_dots_beside_it(tmp_path, monkeypatch):
    eng = engine(tmp_path)
    off = eng.set_angle_reference("hip", 80.0)["offset_deg"]
    texts = []
    monkeypatch.setattr(overlay, "_text", lambda img, text, *a, **k: texts.append(text))
    res = frames(eng)
    hip, knee = angle(res, "hip"), angle(res, "knee")
    h = eng.history(15.0)
    assert h["values"]["hip"][-1] == pytest.approx(hip.value - off) and h["values"]["knee"][-1] == pytest.approx(knee.value)
    assert h["target"] == 60.0                                                                  # the band stays at the target as asked
    line = next(t for t in texts if t.startswith("Hip flex"))
    assert f"{hip.value - off:.1f}" in line and f"dots {hip.value:.1f}" in line and " vs 90)" in line
    assert f"{hip.value - off:.1f}" in texts                                                   # the number by the joint is from the reference
    kline = next(t for t in texts if t.startswith("Knee flexion"))
    assert "dots" not in kline


# ------------------------------------------------------------------------------------------------ the Live tab's box
def test_the_live_box_sets_clears_and_says_what_is_kept(tmp_path):
    _app()
    eng = engine(tmp_path)
    said = []
    box = AngleReferenceBox(eng, lambda text, seconds=4.0: said.append(text))
    assert set(box._rows) == {"knee", "hip"} and not box._rows["hip"]["clear"].isEnabled()
    assert "None" in box._rows["hip"]["status"].text() and "dots read" in box._rows["hip"]["status"].text()
    box._rows["hip"]["spin"].setValue(80.0)
    box.set_now("hip")
    assert "hip" in eng.ref_offsets() and "Reference kept" in said[-1]
    text = box._rows["hip"]["status"].text()
    assert "Called 80.0" in text and "more than the joint" in text and "from the reference" in text
    assert box._rows["hip"]["clear"].isEnabled() and not box._rows["knee"]["clear"].isEnabled()
    box.refresh(True)                                                                           # recording: nothing can be changed
    assert not box._rows["hip"]["spin"].isEnabled() and not box._rows["hip"]["set"].isEnabled() and not box._rows["hip"]["clear"].isEnabled()
    box.refresh(False)
    assert box._rows["hip"]["spin"].isEnabled() and box._rows["hip"]["clear"].isEnabled()
    box.clear("hip")
    assert eng.ref_offsets() == {} and "None" in box._rows["hip"]["status"].text()
    eng.pause()
    box.set_now("hip")                                                                          # a refusal is said, not raised
    assert "paused" in said[-1]
    eng.resume()
    box._rows["hip"]["spin"].setValue(12.5)
    eng.set_test("hip_extension_k0")                                                            # another test: the rows are made again for its angles
    box.refresh(False)
    assert set(box._rows) == {"hip", "knee"} and box._rows["hip"]["spin"].value() == 12.5      # (what was typed stays)


# ------------------------------------------------------------------------------------------------ Review
def _recorded_with(tmp_path, **refs):
    eng = engine(tmp_path)
    offs = {name: eng.set_angle_reference(name, value)["offset_deg"] for name, value in refs.items()}
    return record(eng), offs


def test_review_reads_the_reference_of_a_trial_and_every_number_follows_it(tmp_path):
    meta, offs = _recorded_with(tmp_path, hip=80.0, knee=55.0)
    tr = Trial(meta, open_video=False)
    try:
        assert tr.offset_of("hip") == pytest.approx(offs["hip"], abs=1e-3) and tr.recorded_offsets == tr.offsets
        assert np.allclose(tr.deg_view("hip"), tr.deg["hip"] - offs["hip"], equal_nan=True)
        assert np.allclose(tr.target_view("knee"), tr.ref["knee"] - offs["knee"], equal_nan=True) and np.nanmean(tr.target_view("knee")) == pytest.approx(60.0)
        rows = {r["angle"]: r for r in tr.summary(float(tr.t[0]), float(tr.t[-1]))}
        hip, knee = rows["Hip flex (+) / ext (-)"], rows["Knee flexion"]
        assert hip["offset_deg"] == pytest.approx(offs["hip"], abs=1e-3) and hip["plateau_mean_deg"] + hip["offset_deg"] == pytest.approx(hip["plateau_mean_dots_deg"])
        assert hip["plateau_mean_deg"] == pytest.approx(80.0, abs=1.0) and hip["target_deg"] == pytest.approx(90.0)        # asked: 90 from the reference
        assert hip["mean_minus_target_deg"] == pytest.approx(-10.0, abs=1.0) and hip["pct_in_tolerance"] == 0.0         # judged as it was live (joint 80 against 90 +-5: off)
        assert knee["target_deg"] == pytest.approx(60.0) and knee["plateau_mean_deg"] == pytest.approx(55.0, abs=1.0)
        drawn = tr.drawn(3) if tr.cap is not None else None                                                              # (no video in this copy)
        assert drawn is None
    finally:
        tr.close()


def test_an_angle_without_a_reference_gives_the_old_numbers_exactly(tmp_path):
    meta, _ = _recorded_with(tmp_path, hip=80.0)
    ref_tr = Trial(meta, open_video=False)
    plain_meta = Path(record(engine(tmp_path / "plain")))
    plain = Trial(plain_meta, open_video=False)
    try:
        a = {r["angle"]: r for r in ref_tr.summary(0.0, 10.0)}["Knee flexion"]
        b = {r["angle"]: r for r in plain.summary(0.0, 10.0)}["Knee flexion"]
        for k in ("mean_deg", "sd_deg", "min_deg", "max_deg", "plateau_mean_deg", "target_deg", "pct_in_tolerance", "max_abs_dev_deg"):
            assert a[k] == pytest.approx(b[k], rel=1e-9, nan_ok=True), k
        assert b["offset_deg"] == 0.0 and (b["plateau_mean_dots_deg"] == b["plateau_mean_deg"] or np.isnan(b["plateau_mean_deg"]))
    finally:
        ref_tr.close()
        plain.close()


def test_the_export_row_keeps_the_dots_columns_and_adds_the_from_reference_ones(tmp_path):
    meta, offs = _recorded_with(tmp_path, knee=55.0, hip=80.0)
    tr = Trial(meta, open_video=False)
    plain = Trial(Path(record(engine(tmp_path / "plain"))), open_video=False)
    try:
        row, base = analysis.session_row(tr), analysis.session_row(plain)
        assert row["main_angle"] == "Knee flexion"
        assert row["plateau_angle_deg"] == pytest.approx(base["plateau_angle_deg"], abs=0.01)                    # the dots' angle: the old column
        assert row["plateau_angle_from_reference_deg"] == pytest.approx(row["plateau_angle_deg"] - offs["knee"], abs=0.01)
        assert row["angle_reference_offset_deg"] == pytest.approx(offs["knee"], abs=1e-3)
        assert base["plateau_angle_from_reference_deg"] == "" and base["angle_reference_offset_deg"] == ""
        assert analysis.row_plateau(row) == pytest.approx(row["plateau_angle_from_reference_deg"]) and analysis.row_plateau(base) == pytest.approx(base["plateau_angle_deg"])
        for col in ("angle_at_peak_force_from_reference_deg", "angle2_at_peak_from_reference_deg", "angle3_at_peak_from_reference_deg",
                    "plateau_angle_from_reference_deg", "angle_reference_offset_deg"):
            assert col in analysis.SESSION_COLUMNS
    finally:
        tr.close()
        plain.close()


def test_the_angle_at_peak_from_the_reference_and_on_the_dots_in_a_trial_with_force(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.process(render(PTS, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    force = SimulatedForce(rest_s=0.3, ramp_s=0.6, hold_s=1.2, peak_n=400.0)
    force.start()
    eng.force = force
    time.sleep(0.3)
    eng.process(render(PTS, seed=1), time.perf_counter())
    off = eng.set_angle_reference("hip", 80.0)["offset_deg"]
    paths = eng.start_trial("P01")
    for i in range(60):
        eng.process(render(PTS, seed=i % 5), time.perf_counter())
        time.sleep(1 / 30)
    eng.stop_trial()
    force.stop()
    tr = Trial(paths["meta.json"], open_video=False)
    try:
        fs = tr.force_summary(*(tr.force_window() or (float(tr.t[0]), float(tr.t[-1]))), 0.0)
        hip = "Hip flex (+) / ext (-)"
        assert fs["angle_offsets_deg"] == {hip: pytest.approx(off, abs=1e-3)}
        assert fs["angles_at_peak_deg"][hip] == pytest.approx(fs["angles_at_peak_dots_deg"][hip] - fs["angle_offsets_deg"][hip])
        assert fs["angles_at_peak_deg"]["Knee flexion"] == fs["angles_at_peak_dots_deg"]["Knee flexion"]
        assert fs["main_angle_at_peak_deg"] == fs["angles_at_peak_dots_deg"]["Knee flexion"]                  # the old number: the main joint on the dots
        row = analysis.session_row(tr)
        assert row["angle2_name"] == hip and row["angle2_at_peak_force_deg"] == pytest.approx(fs["angles_at_peak_dots_deg"][hip])
        assert row["angle2_at_peak_from_reference_deg"] == pytest.approx(fs["angles_at_peak_deg"][hip])
        assert row["angle_at_peak_force_from_reference_deg"] == ""                                              # the main joint (knee) has no reference
    finally:
        tr.close()


# ---- a reference chosen in Review for a trial recorded without one
def _plain_trial(tmp_path):
    return Path(record(engine(tmp_path)))


def test_a_reference_reading_ticked_in_review_gives_the_trial_an_offset_and_a_side_file_never_meta(tmp_path):
    meta = _plain_trial(tmp_path)
    before = meta.read_text(encoding="utf-8")
    tr = Trial(meta, open_video=False)
    assert tr.offsets == {}
    first = float(np.nanmean(tr.deg["hip"][tr.t <= reference.REST_SECONDS]))                  # no stamp: the first second stands in for the app's angle
    reference.write_sidecar(meta, "goniometer", 1.0, {"hip": [80.0]}, recorded=None, apply=["hip"])
    tr.reload_reference()
    try:
        assert tr.offsets == {"hip": pytest.approx(first - 80.0)} and tr.recorded_offsets == {}
        assert meta.read_text(encoding="utf-8") == before                                       # the trial's own file is untouched
        side = reference.read_sidecar(meta)
        assert side["apply"] == ["hip"] and "never" not in side.get("apply_note", "") or True
        # the verdict numbers are judged again from the reading: the target (90) is on the from-reference scale, the data.csv flags are the live ones
        rows = {r["angle"]: r for r in tr.summary(float(tr.t[0]), float(tr.t[-1]))}
        hip = rows["Hip flex (+) / ext (-)"]
        assert hip["plateau_mean_deg"] == pytest.approx(80.0, abs=1.0) and hip["target_deg"] == pytest.approx(90.0)
        assert hip["pct_in_tolerance"] == 0.0 and hip["mean_minus_target_deg"] == pytest.approx(-10.0, abs=1.0)      # 10 below 90, tolerance 5
    finally:
        tr.close()
    again = reference.write_sidecar(meta, "goniometer", 1.0, {"hip": [80.0]}, recorded=None)      # saving without a list keeps the choice
    assert reference.read_sidecar(meta)["apply"] == ["hip"]
    reference.write_sidecar(meta, "goniometer", 1.0, {"hip": [80.0]}, recorded=None, apply=[])
    assert "apply" not in reference.read_sidecar(meta) and again.name == "reference.json"


def test_a_reviewers_reading_wins_over_the_reference_taken_when_the_trial_was_recorded(tmp_path):
    meta, offs = _recorded_with(tmp_path, hip=80.0)
    reference.write_sidecar(meta, "goniometer", 1.0, {"hip": [85.0]}, recorded=None, apply=["hip"])
    tr = Trial(meta, open_video=False)
    try:
        assert tr.recorded_offsets == {"hip": pytest.approx(offs["hip"], abs=1e-3)}
        assert tr.offsets["hip"] == pytest.approx(float(np.nanmean(tr.deg["hip"][tr.t <= reference.REST_SECONDS])) - 85.0)
        assert tr.offsets["hip"] != pytest.approx(offs["hip"], abs=0.5)
        row = {r["angle"]: r for r in tr.summary(float(tr.t[0]), float(tr.t[-1]))}["Hip flex (+) / ext (-)"]
        assert row["target_deg"] == pytest.approx(90.0)                                           # data.csv's target column was moved by the recorded offset, not the new one
        assert row["plateau_mean_deg"] == pytest.approx(85.0, abs=1.0)
    finally:
        tr.close()


def test_the_reference_dialog_has_the_tick_and_asks_for_a_reading_first(tmp_path, monkeypatch):
    _app()
    meta = _plain_trial(tmp_path)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    tr = Trial(meta, open_video=False)
    dlg = ReferenceDialog(tr)
    assert set(dlg.apply_boxes) == {"knee", "hip"} and not any(c.isChecked() for c in dlg.apply_boxes.values())
    dlg.apply_boxes["hip"].setChecked(True)
    dlg._save()                                                                                   # ticked, but no reading typed: refused
    assert warned and "type the reading first" in warned[-1] and reference.read_sidecar(meta) == {}
    dlg.boxes["hip"].setText("80")
    dlg._save()
    assert reference.read_sidecar(meta)["apply"] == ["hip"] and "hip" in tr.offsets
    tr.close()
    again = ReferenceDialog(Trial(meta, open_video=False))
    assert again.apply_boxes["hip"].isChecked() and not again.apply_boxes["knee"].isChecked()


def test_the_angle_table_and_the_graphs_show_both_numbers(tmp_path):
    _app()
    meta, offs = _recorded_with(tmp_path, hip=80.0)
    tab = ReviewTab(tmp_path)
    tab.resize(1500, 900)
    tab.show()
    tab.load(str(meta))
    try:
        text = tab.angle_tree.text()
        line = next(l for l in text.splitlines() if l.startswith("Hip flex"))
        assert "from the reference" in line and "on the dots" in line and "hold" in line, line
        everything = tab.angle_tree.text(everything=True)
        assert "On the dots:" not in everything or "more than the joint" in everything
        assert not any("from the reference" in l for l in text.splitlines() if l.startswith("Knee flexion"))
        names = [it.name() for it in tab.plot.plotItem.legend.items] if False else [n for n in (getattr(i, "name", lambda: None)() for i in tab.plot.plotItem.items) if n]
        assert "Hip flex (+) / ext (-) (from reference)" in names and "Hip flex (+) / ext (-) (dots)" in names and "Knee flexion" in names
        hip_line = next(i for i in tab.plot.plotItem.items if getattr(i, "name", lambda: None)() == "Hip flex (+) / ext (-) (from reference)")
        assert np.allclose(hip_line.yData, tab.trial.deg["hip"] - offs["hip"], equal_nan=True)
    finally:
        tab.trial.close()
        tab.close()
