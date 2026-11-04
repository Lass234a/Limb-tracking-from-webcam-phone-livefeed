"""0.34.0 (F4): the standing reference. One button captures the hip, knee and ankle angles of the dots for 3 s with the participant standing, sets
their offsets through the angle reference of 0.33.0 (a thin layer, not a second mechanism), keeps the record and the segment lengths for the
participant, saves them in every later trial's meta.json and exports them. Pure rules, the engine, real recordings and files, the real box and
window offscreen, and Review."""
import csv
import json
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis, session_memory  # noqa: E402
from limbtrack import standing_reference as sr  # noqa: E402
from limbtrack.engine import SNAPSHOT_KEYS, Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.standing_reference_box import StandingReferenceBox  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
POSE = dict(hip_flex=12, knee_flex=3, ankle_df=6)            # a "standing" leg whose dots read 12, 3 and 6 degrees (the joints are at 0)
P = leg_points(**POSE)
RULER_PX, RULER_CM = 200.0, 40.0                             # 5 px per cm: thigh 190 px = 38.0 cm, shank 135 px = 27.0 cm, foot 110 px = 22.0 cm, pelvis 95 px = 19.0 cm
FPS = 20


def engine(tmp_path, test="ankle_dorsiflexion", ruler=True, code="P01"):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.process(render(P, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    if ruler:
        eng.scale.length_cm = RULER_CM
        eng.scale.ref_px = RULER_PX
    eng.participant_code = code
    for i in range(6):
        eng.process(render(P, seed=i), 0.1 + i / FPS)
    return eng


def run(eng, seconds, pose=None, t0=10.0, start=0):
    """Feed `seconds` of pictures at 20 per second; `pose(i)` gives the leg of picture i (default: the standing leg)."""
    for i in range(start, start + int(seconds * FPS)):
        pts = P if pose is None else leg_points(**pose(i))
        eng.process(render(pts, seed=i % 7), t0 + i / FPS)


def captured(tmp_path, **kw):
    eng = engine(tmp_path, **kw)
    eng.start_standing_reference()
    run(eng, sr.CAPTURE_S + 0.3)
    assert eng.standing_state()["state"] == "idle", eng.standing_state()
    return eng


def record(eng, code="P01", n=8, t0=100.0):
    paths = eng.start_trial(code)
    for i in range(n):
        eng.process(render(P, seed=i), t0 + i / FPS)
    eng.stop_trial()
    return paths["meta.json"]


# ------------------------------------------------------------------------------------------------ the pure part
def feed_all(cap, values, seconds=3.2, fps=20, points=None):
    for i in range(int(seconds * fps)):
        done = cap.feed(5.0 + i / fps, {n: values(i) if callable(values) else values for n in cap.names}, points or {})
        if done:
            return True
    return False


def test_the_capture_takes_three_seconds_and_the_offset_is_the_mean_minus_the_anatomical_zero():
    cap = sr.Capture(["hip", "knee", "ankle"])
    assert not cap.feed(0.0, {"hip": 10.0}, {}) and cap.progress() == 0.0
    for k in range(1, 30):
        assert not cap.feed(k * 0.05, {"hip": 10.0}, {})
    assert cap.progress() == pytest.approx(1.45 / 3.0)
    for k in range(30, 60):
        assert not cap.feed(k * 0.05, {"hip": 10.0}, {})
    assert cap.feed(3.0, {"hip": 10.0}, {})                       # 3.0 s after the first picture: complete
    cap = sr.Capture(["hip", "knee"])
    assert feed_all(cap, lambda i: 12.0 + (0.2 if i % 2 else -0.2))
    rec = cap.finish(taken_at="2026-11-02T10:00:00")
    assert set(rec["angles"]) == {"hip", "knee"} and rec["skipped"] == {}
    hip = rec["angles"]["hip"]
    assert hip["reading_deg"] == 0.0 and hip["reading_source"] == "anatomical zero" and hip["offset_deg"] == pytest.approx(12.0, abs=0.01)
    assert hip["dots_deg"] == pytest.approx(12.0, abs=0.02) and hip["sd_deg"] == pytest.approx(0.2, abs=0.02) and hip["pictures"] == 61
    assert rec["capture_s"] == pytest.approx(3.0, abs=0.1) and rec["pictures"] == 61 and rec["instrument"] == "" and "note" in rec


def test_a_typed_instrument_reading_is_what_the_joint_is_called_instead_of_zero():
    cap = sr.Capture(["hip", "knee"], {"hip": 2.0}, "goniometer")
    feed_all(cap, 12.0)
    rec = cap.finish()
    assert rec["angles"]["hip"]["offset_deg"] == pytest.approx(10.0) and rec["angles"]["hip"]["reading_source"] == "typed (goniometer)"
    assert rec["angles"]["knee"]["offset_deg"] == pytest.approx(12.0) and rec["angles"]["knee"]["reading_source"] == "anatomical zero"
    assert rec["instrument"] == "goniometer"
    assert sr.basis_text(rec) == "hip: typed (goniometer) 2.0; knee: anatomical zero"


def test_an_angle_that_moved_or_was_hardly_read_is_not_set_and_the_others_are():
    cap = sr.Capture(["hip", "knee", "ankle"])
    for i in range(64):
        t = 5.0 + i / 20
        cap.feed(t, {"hip": 12.0 + 6.0 * math.sin(i / 3.0), "knee": 3.0, "ankle": 6.0 if i % 5 == 0 else None}, {})
    rec = cap.finish()
    assert set(rec["angles"]) == {"knee"}
    assert "it moved during the capture" in rec["skipped"]["hip"] and "SD" in rec["skipped"]["hip"]
    assert "read in only 13 of 64 pictures" in rec["skipped"]["ankle"]
    assert sr.basis_text(rec).endswith("not set: hip, ankle")


def test_pictures_that_stop_cancel_the_capture_and_the_same_picture_twice_counts_once():
    cap = sr.Capture(["hip"])
    cap.feed(1.0, {"hip": 5.0}, {})
    assert cap.feed(1.0, {"hip": 99.0}, {}) is False and cap.pictures == 1                # the same frame redrawn
    cap.feed(1.1, {"hip": 5.0}, {})
    with pytest.raises(sr.StandingError, match="No picture came in for 1.4 s"):
        cap.feed(2.5, {"hip": 5.0}, {})


def test_segment_lengths_are_in_cm_with_a_ruler_and_in_pixels_without():
    pts = {"trochanter": (100.0, 100.0), "epicondyle": (100.0, 290.0), "fibular_head": (0.0, 0.0), "malleolus": (0.0, 135.0)}
    for px_per_cm, unit, thigh in ((5.0, "cm", 38.0), (None, "px", 190.0)):
        cap = sr.Capture(["knee"])
        feed_all(cap, 3.0, points=pts)
        seg = cap.finish(px_per_cm)["segments"]
        assert seg["unit"] == unit and seg["lengths"]["thigh"] == pytest.approx(thigh) and seg["dots"]["thigh"] == ["trochanter", "epicondyle"]
        assert seg["lengths"]["shank"] == pytest.approx(135.0 / (px_per_cm or 1.0)) and "foot" not in seg["lengths"] and "pelvis" not in seg["lengths"]


def test_the_meta_block_says_which_angles_the_trial_used_and_the_export_cells_read_it_defensively():
    rec = {"taken_at": "T", "angles": {"hip": {"offset_deg": 12.0, "reading_deg": 0.0, "reading_source": "anatomical zero"},
                                       "knee": {"offset_deg": 3.0, "reading_deg": 0.0, "reading_source": "anatomical zero"}},
           "skipped": {}, "segments": {"unit": "cm", "lengths": {"thigh": 38.0, "shank": 27.0}}}
    assert sr.meta_block(None) is None and sr.meta_block({}) is None
    block = sr.meta_block(rec, ["hip", "ankle"])
    assert block["in_use_in_this_trial"] == ["hip"] and "in_use_in_this_trial" not in rec                   # (a copy)
    cells = sr.export_cells({"standing_reference": block})
    assert cells["standing_hip_offset_deg"] == 12.0 and cells["standing_knee_offset_deg"] == 3.0 and cells["standing_ankle_offset_deg"] == ""
    assert cells["standing_in_use"] == "hip" and cells["standing_length_unit"] == "cm" and cells["standing_thigh_len"] == 38.0
    assert cells["standing_foot_len"] == "" and cells["standing_pelvis_len"] == "" and cells["standing_taken_at"] == "T"
    assert set(cells) == set(sr.COLUMNS)
    for meta in ({}, {"standing_reference": None}, {"standing_reference": "damaged"}, {"standing_reference": {"angles": "x"}},
                 {"standing_reference": {"angles": {"hip": {"offset_deg": "x"}}, "segments": {"lengths": {"thigh": "x"}}}}):
        got = sr.export_cells(meta)
        assert set(got) == set(sr.COLUMNS) and got["standing_hip_offset_deg"] == "" and got["standing_thigh_len"] == ""      # nothing invented


# ------------------------------------------------------------------------------------------------ the engine
def test_pressing_it_sets_the_offsets_of_every_angle_through_the_angle_reference(tmp_path):
    eng = engine(tmp_path)
    before = {n: eng._ref_last[n] for n in ("hip", "knee", "ankle")}
    eng.start_standing_reference()
    assert eng.standing_state()["state"] == "capturing"
    run(eng, 1.5)
    assert eng.standing_state()["state"] == "capturing" and 0.4 < eng.standing_state()["progress"] < 0.7
    assert eng.ref_offsets() == {}                                  # nothing is set until the 3 s are over
    run(eng, 1.9, start=int(1.5 * FPS))
    st = eng.standing_state()
    assert st["state"] == "idle" and "Standing reference kept" in st["message"]
    rec = st["record"]
    assert set(rec["angles"]) == {"hip", "knee", "ankle"} and st["in_use"] == ["hip", "knee", "ankle"]
    offsets = eng.ref_offsets()
    assert offsets == {n: pytest.approx(rec["angles"][n]["offset_deg"]) for n in ("hip", "knee", "ankle")}          # the SAME offsets, in the angle reference
    for n, want in (("hip", 12), ("knee", 3), ("ankle", 6)):
        assert offsets[n] == pytest.approx(want, abs=1.5) and offsets[n] == pytest.approx(before[n], abs=0.3)      # what the dots read, the joint called 0
    held = eng.angle_ref.active("P01")
    assert all(held[n]["reference_deg"] == 0.0 and held[n]["taken_at"] == rec["taken_at"] for n in held)
    assert rec["segments"]["unit"] == "cm"
    assert rec["segments"]["lengths"] == {"thigh": pytest.approx(38.0, abs=0.3), "shank": pytest.approx(27.0, abs=0.3),
                                          "foot": pytest.approx(22.0, abs=0.3), "pelvis": pytest.approx(19.0, abs=0.3)}
    res = eng.process(render(P, seed=3), 40.0)
    hip = next(a for a in res.angles if a.name == "hip")
    assert hip.ref_offset == pytest.approx(offsets["hip"]) and hip.raw_value == pytest.approx(before["hip"], abs=0.3)      # shown from the reference; the dots' angle kept


def test_a_typed_reading_sets_the_offset_from_that_reading(tmp_path):
    eng = engine(tmp_path)
    eng.start_standing_reference({"hip": 2.0, "knee": None}, "inclinometer")
    run(eng, 3.3)
    rec = eng.standing_state()["record"]
    assert rec["angles"]["hip"]["reading_deg"] == 2.0 and rec["angles"]["hip"]["offset_deg"] == pytest.approx(10.0, abs=1.5)
    assert rec["angles"]["knee"]["reading_deg"] == 0.0 and rec["instrument"] == "inclinometer"
    assert eng.ref_offsets()["hip"] == pytest.approx(rec["angles"]["hip"]["offset_deg"])


def test_only_the_angles_of_the_test_are_read_so_a_five_dot_test_has_no_ankle(tmp_path):
    eng = engine(tmp_path, test="knee_extension")
    eng.start_standing_reference()
    run(eng, 3.3)
    rec = eng.standing_state()["record"]
    assert set(rec["angles"]) == {"hip", "knee"} and "foot" not in rec["segments"]["lengths"]
    assert sr.export_cells({"standing_reference": rec})["standing_ankle_offset_deg"] == ""


def test_the_latest_reference_of_an_angle_wins_standing_over_hand_and_hand_over_standing(tmp_path):
    eng = engine(tmp_path)
    eng.set_angle_reference("hip", 5.0)                              # set by hand first
    hand_hip = eng.ref_offsets()["hip"]
    eng.start_standing_reference()
    run(eng, 3.3)
    st = eng.standing_state()
    assert "It replaced the reference set earlier for hip" in st["message"]
    assert eng.ref_offsets()["hip"] == pytest.approx(st["record"]["angles"]["hip"]["offset_deg"]) and eng.ref_offsets()["hip"] != pytest.approx(hand_hip, abs=1.0)
    assert st["in_use"] == ["hip", "knee", "ankle"]
    time.sleep(0.01)
    eng.set_angle_reference("knee", 1.0)                             # then by hand: the knee is the hand's now
    st = eng.standing_state()
    assert st["in_use"] == ["hip", "ankle"] and eng.ref_offsets()["knee"] == pytest.approx(st["record"]["angles"]["knee"]["offset_deg"] - 1.0, abs=0.01)
    meta = json.loads(record(eng).read_text())
    assert meta["standing_reference"]["in_use_in_this_trial"] == ["hip", "ankle"]
    assert meta["angle_reference"]["angles"]["knee"]["reference_deg"] == 1.0                       # what the trial was shown from
    assert meta["standing_reference"]["angles"]["knee"]["offset_deg"] != meta["angle_reference"]["angles"]["knee"]["offset_deg"]
    assert "hip" in sr.export_cells(meta)["standing_in_use"] and "knee" not in sr.export_cells(meta)["standing_in_use"]


def test_an_angle_the_standing_capture_could_not_read_keeps_the_reference_it_had(tmp_path):
    eng = engine(tmp_path)
    eng.set_angle_reference("ankle", 2.0)
    hand_ankle = eng.ref_offsets()["ankle"]
    eng.start_standing_reference()
    for i in range(int(3.3 * FPS)):                                    # the foot dots are hidden (covered) during the whole capture
        eng.process(render(P, seed=i % 7, hide=("mt5_base", "mt5_head")), 10.0 + i / FPS)
    st = eng.standing_state()
    assert "ankle" not in st["record"]["angles"] and set(st["record"]["angles"]) == {"hip", "knee"}
    assert "Ankle not set: read in only 0 of" in st["message"]
    assert eng.ref_offsets()["ankle"] == hand_ankle                    # untouched


def test_a_person_who_moves_gets_no_offset_for_the_angle_that_moved(tmp_path):
    eng = engine(tmp_path)
    eng.set_angle_reference("hip", 0.0)
    before = dict(eng.ref_offsets())
    eng.start_standing_reference()
    run(eng, 3.3, pose=lambda i: {**POSE, "hip_flex": 12 + 7.0 * math.sin(i / 3.0)})
    st = eng.standing_state()
    assert "Hip not set: it moved during the capture" in st["message"]
    assert set((st["record"] or {}).get("angles", {})) == {"knee", "ankle"}
    assert eng.ref_offsets()["hip"] == before["hip"]


def test_it_cannot_begin_in_these_cases_and_says_why(tmp_path):
    eng = engine(tmp_path, code="")
    with pytest.raises(ValueError, match="participant code first"):
        eng.start_standing_reference()
    eng.participant_code = "P01"
    with pytest.raises(ValueError, match="is not a number"):
        eng.start_standing_reference({"hip": "abc"})
    with pytest.raises(ValueError, match="not an angle"):
        eng.start_standing_reference({"hip": 900})
    with pytest.raises(ValueError, match="takes a reading for"):
        eng.start_standing_reference({"elbow": 3.0})
    assert eng.standing_state()["state"] == "idle"
    fresh = Engine(TESTS, recordings_dir=tmp_path / "x")
    fresh.set_test("ankle_dorsiflexion")
    fresh.participant_code = "P01"
    with pytest.raises(ValueError, match="No hip, knee or ankle angle can be read"):
        fresh.start_standing_reference()
    eng.start_trial("P01")
    with pytest.raises(RuntimeError, match="Stop recording"):
        eng.start_standing_reference()
    with pytest.raises(RuntimeError, match="Stop recording"):
        eng.clear_standing_reference()
    eng.stop_trial()
    eng.paused = True
    with pytest.raises(RuntimeError, match="paused for a break"):
        eng.start_standing_reference()


def test_a_capture_is_cancelled_by_a_change_of_test_a_gap_or_the_start_of_a_recording(tmp_path):
    eng = engine(tmp_path)
    eng.start_standing_reference()
    run(eng, 1.0)
    eng.set_test("knee_extension")                                    # other landmarks: the dots are reset
    run(eng, 0.5, start=20)
    st = eng.standing_state()
    assert st["state"] == "idle" and "changed during the standing capture" in st["message"] and eng.ref_offsets() == {} and st["record"] is None
    eng = engine(tmp_path / "p")
    eng.start_standing_reference()
    run(eng, 1.0)
    eng.participant_code = "P02"                                      # another person is typed in the window
    run(eng, 0.5, start=20)
    st = eng.standing_state()
    assert st["state"] == "idle" and "participant changed" in st["message"] and eng.ref_offsets() == {} and eng.standing_references_all == {}
    eng = engine(tmp_path / "b")
    eng.start_standing_reference()
    run(eng, 1.0)
    eng.process(render(P, seed=1), 14.0)                              # nothing for 3 s
    st = eng.standing_state()
    assert st["state"] == "idle" and "No picture came in for" in st["message"] and eng.ref_offsets() == {}
    eng = engine(tmp_path / "c")
    eng.start_standing_reference()
    run(eng, 1.0)
    eng.start_trial("P01")
    assert eng.standing_state()["state"] == "idle"
    eng.stop_trial()
    eng2 = engine(tmp_path / "d")
    eng2.start_standing_reference()
    eng2.cancel_standing_reference()
    assert eng2.standing_state()["state"] == "idle" and "cancelled" in eng2.standing_state()["message"]


def test_clear_forgets_the_record_and_its_references_but_not_a_hand_set_one(tmp_path):
    eng = captured(tmp_path)
    eng.set_angle_reference("ankle", 1.0)
    eng.clear_standing_reference()
    st = eng.standing_state()
    assert st["record"] is None and set(eng.ref_offsets()) == {"ankle"} and eng.standing_references_all == {}
    meta = json.loads(record(eng).read_text())
    assert meta["standing_reference"] is None


def test_it_belongs_to_one_participant(tmp_path):
    eng = captured(tmp_path)
    assert set(eng.standing_references_all) == {"P01"}
    eng.participant_code = "P02"
    assert eng.standing_state()["record"] is None and eng.ref_offsets() == {}
    meta = json.loads(record(eng, "P02").read_text())
    assert meta["standing_reference"] is None
    eng.participant_code = "P01"
    assert eng.standing_state()["record"] is not None


# ------------------------------------------------------------------------------------------------ real files
def test_a_recorded_trial_has_the_block_the_dots_angle_is_unchanged_and_the_reference_is_the_same_as_by_hand(tmp_path):
    plain = engine(tmp_path / "plain")
    plain_meta = json.loads(record(plain).read_text())
    assert plain_meta["standing_reference"] is None and plain_meta["angle_reference"] is None
    eng = captured(tmp_path / "with")
    meta_path = record(eng)
    meta = json.loads(meta_path.read_text())
    block = meta["standing_reference"]
    assert "standing_reference" in SNAPSHOT_KEYS and "changed_during_recording" not in meta
    assert set(block["angles"]) == {"hip", "knee", "ankle"} and block["in_use_in_this_trial"] == ["hip", "knee", "ankle"] and block["segments"]["unit"] == "cm"
    assert block["test_key"] == "ankle_dorsiflexion" and block["facing"] == "right" and block["capture_s"] >= 3.0 and block["pictures"] > 55
    for n, a in block["angles"].items():                             # the live angle reference carries the very same offsets
        assert meta["angle_reference"]["angles"][n]["offset_deg"] == a["offset_deg"] and meta["angle_reference"]["angles"][n]["reference_deg"] == 0.0
    rows = lambda p: list(csv.DictReader(open(Path(p).parent / json.loads(Path(p).read_text())["files"]["data.csv"], newline="", encoding="utf-8")))     # noqa: E731
    a, b = rows(meta_path), rows(next((tmp_path / "plain" / "P01").glob("*_meta.json")))
    for n in ("hip", "knee", "ankle"):                               # data.csv holds the dots' angle, with or without the reference
        assert [float(r[f"{n}_deg"]) for r in a] == pytest.approx([float(r[f"{n}_deg"]) for r in b], abs=0.1)
    assert abs(float(a[3]["hip_deg"]) - POSE["hip_flex"]) < 1.5


def test_the_export_has_the_standing_columns_at_the_end_and_blanks_for_a_trial_without(tmp_path):
    eng = captured(tmp_path / "a")
    with_meta = record(eng)
    without_meta = record(engine(tmp_path / "b"))
    cols = analysis.SESSION_COLUMNS
    assert cols[-len(sr.COLUMNS) - 2: -2] == sr.COLUMNS and cols[-2:] == ["check_weight_error_pct", "check_weight_time"] and cols[:3] == ["trial", "position_number", "trial_number"]
    t1, t2 = Trial(with_meta, open_video=False), Trial(without_meta, open_video=False)
    try:
        r1, r2 = analysis.session_row(t1), analysis.session_row(t2)
        rec = json.loads(Path(with_meta).read_text())["standing_reference"]
        assert r1["standing_hip_offset_deg"] == rec["angles"]["hip"]["offset_deg"] and r1["standing_ankle_offset_deg"] == rec["angles"]["ankle"]["offset_deg"]
        assert r1["standing_in_use"] == "hip, knee, ankle" and r1["standing_basis"] == "hip, knee, ankle: anatomical zero"
        assert r1["standing_length_unit"] == "cm" and r1["standing_thigh_len"] == pytest.approx(38.0, abs=0.3) and r1["standing_pelvis_len"] == pytest.approx(19.0, abs=0.3)
        assert r1["angle_reference_offset_deg"] == "" or isinstance(r1["angle_reference_offset_deg"], float)       # the 0.33.0 column is untouched
        assert all(r2[c] == "" for c in sr.COLUMNS)
        out = analysis.write_session_csv([r1, r2], tmp_path / "both.csv")
        back = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
        assert list(back[0]) == cols and float(back[0]["standing_knee_offset_deg"]) == pytest.approx(rec["angles"]["knee"]["offset_deg"], abs=1e-3)
        assert back[1]["standing_knee_offset_deg"] == "" and back[1]["standing_thigh_len"] == ""
    finally:
        t1.close()
        t2.close()


def test_marking_the_dots_again_drops_the_live_reference_but_the_record_and_the_export_stay(tmp_path):
    eng = captured(tmp_path)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 3))                           # the dots are marked again (the participant sat down)
    assert eng.ref_offsets() == {} and eng.standing_state()["in_use"] == [] and eng.standing_state()["record"] is not None
    meta_path = record(eng)
    meta = json.loads(meta_path.read_text())
    assert meta["angle_reference"] is None and meta["standing_reference"]["in_use_in_this_trial"] == [] and set(meta["standing_reference"]["angles"]) == {"hip", "knee", "ankle"}
    tr = Trial(meta_path, open_video=False)
    try:
        row = analysis.session_row(tr)
        assert row["standing_hip_offset_deg"] != "" and row["standing_in_use"] == "" and row["angle_reference_offset_deg"] == ""
        assert tr.offsets == {}                                       # Review shows the dots' own angles: the standing offset was not applied to this trial
        line = next(i for i in analysis.trial_check_items(tr) if i["title"] == "Standing reference in the file")
        assert line["level"] == "info" and "Not in use in this trial: no angle reference was set" in line["text"]
    finally:
        tr.close()


def test_review_shows_a_trial_with_the_standing_offsets_beside_the_dots_angle(tmp_path):
    app = QApplication.instance() or QApplication([])
    eng = captured(tmp_path)
    meta_path = record(eng, n=14)
    tr = Trial(meta_path, open_video=False)
    try:
        hip = tr.offsets["hip"]
        assert hip == pytest.approx(eng.ref_offsets()["hip"]) and np.nanmean(tr.deg["hip"]) - np.nanmean(tr.deg_view("hip")) == pytest.approx(hip)
        line = next(i for i in analysis.trial_check_items(tr) if i["title"] == "Standing reference in the file")
        assert "In use in this trial: hip, knee, ankle" in line["text"] and f"hip {hip:+.1f}" in line["text"]
    finally:
        tr.close()
    tab = ReviewTab(tmp_path)
    tab.load(str(meta_path))
    try:
        assert "Standing reference of this participant" in tab.checks.text(everything=True)
        line = next(l for l in tab.angle_tree.text().splitlines() if l.startswith("Hip flex"))
        assert "from the reference" in line and "on the dots" in line, line      # the 0.33.0 display: the angle from the reference first, the dots' own beside it
    finally:
        tab.trial.close()
        tab.close()
    assert app is not None


def test_a_trial_recorded_before_this_feature_opens_and_exports_with_blanks(tmp_path):
    meta_path = record(engine(tmp_path))
    meta = json.loads(meta_path.read_text())
    del meta["standing_reference"]                                    # as recorded by 0.33.0 and earlier
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    tr = Trial(meta_path, open_video=False)
    try:
        row = analysis.session_row(tr)
        assert all(row[c] == "" for c in sr.COLUMNS) and not any(i["title"] == "Standing reference in the file" for i in analysis.trial_check_items(tr))
    finally:
        tr.close()


# ------------------------------------------------------------------------------------------------ the box and the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow(TESTS, tmp_path)
    w.app = app
    yield w
    w.live.worker = None
    w.close()


def mark_in_window(win):
    live, e = win.live, win.engine
    e.set_test("ankle_dorsiflexion")
    e.process(render(P, seed=0), 0.0)
    for n in e.active_dots:
        assert e.mark_dot(n, *(P[n] + 2))
    e.scale.length_cm = RULER_CM
    e.scale.ref_px = RULER_PX
    live.participant.setText("P01")
    live.refresh_status()
    return live, e


def typed(edit, text):
    edit.setText(text)
    edit.textEdited.emit(text)


def test_the_box_is_in_the_position_section_and_waits_for_a_participant_code(win):
    live = win.live
    box = live.standing_box
    assert isinstance(box, StandingReferenceBox) and live.accordion.sections["position"].body.isAncestorOf(box)
    live.participant.setText("")
    live.refresh_status()
    assert not box.capture_btn.isEnabled() and not any(e.isEnabled() for e in box.edits.values()) and "participant code" in box.toolTip()
    live.participant.setText("P01")
    live.refresh_status()
    assert box.capture_btn.isEnabled() and all(e.isEnabled() for e in box.edits.values()) and not box.clear_btn.isEnabled()
    box.refresh(True)
    assert not box.capture_btn.isEnabled() and "Not while recording" in box.toolTip()


def test_pressing_capture_shows_the_progress_then_what_was_kept_and_clear_takes_it_away(win):
    live, e = mark_in_window(win)
    box = live.standing_box
    for i in range(4):
        e.process(render(P, seed=i), 0.1 + i / FPS)
    typed(box.edits["hip"], "2,0")
    box.instrument.setCurrentIndex(box.instrument.findData("goniometer"))
    box.capture_btn.click()
    assert e.standing_state()["state"] == "capturing" and box.capture_btn.text() == "Cancel" and "Capturing" in box.status.text()
    run(e, 1.0)
    box.refresh(False)
    assert re.fullmatch(r"Capturing: keep the participant still\.\.\. [01]\.\d / 3 s", box.status.text()), box.status.text()
    run(e, 2.5, start=int(1.0 * FPS))
    box.refresh(False)
    assert box.capture_btn.text() == "Capture 3 s" and "Standing reference kept" in box.status.text() and "Hip +" in box.status.text()
    assert "Segments: thigh 38.0 cm" in box.status.text() and "Used as the angle reference of: hip, knee, ankle." in box.status.text()
    assert "Standing reference kept" in live.instruction.text()        # said on the instruction line too
    rec = e.standing_state()["record"]
    assert rec["angles"]["hip"]["reading_deg"] == 2.0 and rec["instrument"] == "goniometer"
    assert box.clear_btn.isEnabled()
    box.clear_btn.click()
    assert e.standing_state()["record"] is None and e.ref_offsets() == {} and "No standing reference yet" in box.status.text()


def test_a_second_press_cancels_and_a_bad_reading_turns_the_box_amber_and_does_not_start(win):
    live, e = mark_in_window(win)
    box = live.standing_box
    for i in range(4):
        e.process(render(P, seed=i), 0.1 + i / FPS)
    box.capture_btn.click()
    box.capture_btn.click()
    assert e.standing_state()["state"] == "idle" and "cancelled" in box.status.text()
    typed(box.edits["knee"], "abc")
    box.capture_btn.click()
    assert e.standing_state()["state"] == "idle" and "Knee reading" in box.status.text() and "fcd34d" in box.edits["knee"].styleSheet()
    typed(box.edits["knee"], "0")
    assert box.edits["knee"].styleSheet() == ""


def test_the_record_is_remembered_with_a_recent_session_only_and_the_live_reference_is_not(win, tmp_path):
    live, e = mark_in_window(win)
    for i in range(4):
        e.process(render(P, seed=i), 0.1 + i / FPS)
    e.start_standing_reference()
    run(e, 3.3)
    assert e.ref_offsets()
    live.save_session()
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["session"]            # the real file
    assert set(saved["standing_references"]) == {"P01"} and set(saved["standing_references"]["P01"]["angles"]) == {"hip", "knee", "ankle"}
    w2 = MainWindow(TESTS, tmp_path)
    try:
        assert w2.engine.standing_references_all["P01"]["angles"] == saved["standing_references"]["P01"]["angles"]
        assert w2.engine.ref_offsets() == {}                       # the dots are not remembered, so neither is the reference they made
        w2.live.refresh_status()
        assert "Not used as an angle reference at the moment" in w2.live.standing_box.status.text()
    finally:
        w2.close()
    old = {"participant": "P09", "standing_references": {"P09": saved["standing_references"]["P01"]}, "saved_at": time.time() - 3 * 24 * 3600,
           "format": session_memory.FORMAT}
    win.engine.standing_references_all = {}
    live.restore_session(old)
    assert win.engine.standing_references_all == {}                # a new day starts blank, like the code and the leg
    win.engine.standing_references_all = {"P01": {"angles": {}}, "": {"angles": {"hip": {}}}, "P02": "damaged"}
    assert win.engine.standing_references_all == {}                # a damaged remembered entry is dropped, not kept
