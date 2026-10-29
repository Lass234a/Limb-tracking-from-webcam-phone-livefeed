"""Reference readings typed from a goniometer or inclinometer (0.16.0): parsing, what is saved, the moment of a reading, the side file,
the comparison with the app and the agreement over many trials."""
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from limbtrack import reference as ref  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import frontal_points, leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


# --------------------------------------------------------------------------------------------------------------- what is typed
@pytest.mark.parametrize("text, expected", [
    ("88 90 89", [88.0, 90.0, 89.0]), ("88; 90", [88.0, 90.0]), ("88,5 90", [88.5, 90.0]), ("88, 90", [88.0, 90.0]),
    ("  60  ", [60.0]), ("-5", [-5.0]), ("0", [0.0]), ("", []), ("   ", []), (";;", []), ("12.25", [12.25]),
])
def test_what_is_typed_becomes_readings(text, expected):
    assert ref.parse_values(text) == expected


@pytest.mark.parametrize("text, why", [("abc", "not a number"), ("88 x", '"x" is not a number'), ("inf", "not an angle"),
                                       ("nan", "not an angle"), ("400", "not an angle"), ("-200", "not an angle"),
                                       ("1 2 3 4 5 6", "at most 5")])
def test_what_cannot_be_a_reading_is_refused_in_plain_words(text, why):
    with pytest.raises(ref.ReadingError) as e:
        ref.parse_values(text)
    assert why in str(e.value)


def test_a_typed_zero_is_a_reading_and_blank_is_not():
    r = ref.ReferenceReadings()
    r.set_values("knee", ref.parse_values("0"))
    assert r.has_data() and r.to_meta()["readings"]["knee"]["values_deg"] == [0.0]
    r.set_values("knee", ref.parse_values(""))
    assert not r.has_data() and r.to_meta() is None


def test_format_gives_back_what_parse_reads():
    assert ref.format_values([88.0, 90.5, 89.0]) == "88 90.5 89" and ref.parse_values(ref.format_values([88.0, 90.5, 89.0])) == [88.0, 90.5, 89.0]


# -------------------------------------------------------------------------------------------------------------------- what is saved
def test_the_saved_form_keeps_every_reading_the_mean_the_instrument_and_the_moment():
    r = ref.ReferenceReadings()
    r.instrument, r.resolution_deg = "inclinometer", 0.5
    r.set_values("knee", [60.0, 61.0, 62.0])
    r.stamp = {"taken_at": "2026-10-20T12:03:10", "app_deg": {"knee": 59.8123456, "hip": None}}
    r.position = ("knee_extension", 60.0, "P01")
    m = r.to_meta(titles={"knee": "Knee flexion"})
    assert m["readings"]["knee"] == {"values_deg": [60.0, 61.0, 62.0], "mean_deg": 61.0}
    assert m["instrument"] == "inclinometer" and m["resolution_deg"] == 0.5 and m["stamped_at"] == "2026-10-20T12:03:10"
    assert m["app_at_reading_deg"] == {"knee": 59.812, "hip": None} and m["app_basis"] == ref.APP_BASIS_STAMP
    assert m["entered_for"] == {"test_key": "knee_extension", "target_deg": 60.0, "participant": "P01"} and m["titles"] == {"knee": "Knee flexion"}
    assert "Never a measurement of this program" in m["note"]
    json.dumps(m)                                                             # nothing that cannot be written to meta.json


# --------------------------------------------------------------------------------------------------------------------- the side file
def meta_with(readings=None, stamp=None, app=None, angles=("knee", "hip"), titles=None):
    m = {"participant": "P01", "test_key": "knee_extension", "target_deg": 60.0, "angles": [{"name": n, "label": n.title()} for n in angles],
         "files": {"data.csv": "x_data.csv"}}
    if readings or stamp:
        m["reference_readings"] = {"instrument": "goniometer", "resolution_deg": 1.0, "titles": titles or {},
                                   "readings": {k: {"values_deg": v, "mean_deg": sum(v) / len(v)} for k, v in (readings or {}).items()},
                                   "stamped_at": stamp, "app_at_reading_deg": app}
    return m


def test_a_side_file_is_written_whole_and_next_to_the_trial(tmp_path):
    meta_path = tmp_path / "P01_knee_extension_60deg_20261020-120000_meta.json"
    meta_path.write_text("{}")
    path = ref.write_sidecar(meta_path, "inclinometer", 0.1, {"knee": [59.0, 60.0], "hip": []})
    assert path == tmp_path / "P01_knee_extension_60deg_20261020-120000_reference.json"
    assert not list(tmp_path.glob("*.tmp"))
    data = ref.read_sidecar(meta_path)
    assert data["readings"] == {"knee": {"values_deg": [59.0, 60.0], "mean_deg": 59.5}} and data["instrument"] == "inclinometer"
    assert meta_path.read_text() == "{}"                                      # the trial's own file is never touched


def test_the_side_file_keeps_only_what_differs_from_what_was_saved_with_the_trial(tmp_path):
    meta_path = tmp_path / "T1_meta.json"
    meta_path.write_text("{}")
    saved = meta_with({"knee": [60.0], "hip": [90.0], "ankle": [5.0]})["reference_readings"]
    ref.write_sidecar(meta_path, "goniometer", 1.0, {"knee": [60.0], "hip": [], "ankle": [6.0], "extra": []}, recorded=saved)
    assert ref.read_sidecar(meta_path)["readings"] == {"hip": {"values_deg": [], "mean_deg": None},          # removed
                                                       "ankle": {"values_deg": [6.0], "mean_deg": 6.0}}       # changed; the knee is the same
    ref.write_sidecar(meta_path, "inclinometer", 1.0, {"knee": [60.0]}, recorded=saved)
    assert ref.read_sidecar(meta_path)["readings"] == {"knee": {"values_deg": [60.0], "mean_deg": 60.0}}     # another instrument is a change


def test_a_missing_or_broken_side_file_is_no_side_file(tmp_path):
    meta_path = tmp_path / "t_meta.json"
    assert ref.read_sidecar(meta_path) == {}
    (tmp_path / "t_reference.json").write_text("{not json", encoding="utf-8")
    assert ref.read_sidecar(meta_path) == {}
    (tmp_path / "t_reference.json").write_text("[1, 2]", encoding="utf-8")
    assert ref.read_sidecar(meta_path) == {}


def test_the_side_file_wins_key_by_key_over_what_was_saved_with_the_trial():
    meta = meta_with({"knee": [60.0], "hip": [90.0]})
    side = {"instrument": "inclinometer", "resolution_deg": 0.1, "readings": {"knee": {"values_deg": [58.0, 59.0], "mean_deg": 58.5}}}
    merged = ref.merged_readings(meta, side)
    assert merged["knee"]["mean_deg"] == 58.5 and merged["knee"]["source"] == "added later in Review" and merged["knee"]["instrument"] == "inclinometer"
    assert merged["hip"]["mean_deg"] == 90.0 and merged["hip"]["source"] == "recorded with the trial"


# ------------------------------------------------------------------------------------------------------------------ the comparison
def write_data_csv(path, knee_first_second=(40.0, 42.0), later=70.0):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "knee_deg", "hip_deg"])
        for i, v in enumerate(knee_first_second):
            w.writerow([i, i * 0.4, v, 90.0])
        w.writerow([9, 2.0, later, 90.0])                                     # after the first second: not used for the fallback


def test_the_app_angle_is_the_one_stamped_at_the_moment_of_the_reading():
    meta = meta_with({"knee": [60.0, 62.0]}, stamp="2026-10-20T12:03:10", app={"knee": 58.5, "hip": 91.0})
    row = next(r for r in ref.comparison(meta) if r["key"] == "knee")
    assert row["reference_deg"] == 61.0 and row["n_readings"] == 2 and row["app_deg"] == 58.5
    assert row["difference_deg"] == pytest.approx(-2.5)                       # app minus reference
    assert row["app_basis"] == ref.APP_BASIS_STAMP and row["title"] == "Knee"


def test_without_a_stamped_moment_the_first_second_of_the_recording_stands_in_and_says_so(tmp_path):
    write_data_csv(tmp_path / "x_data.csv")
    meta = meta_with({"knee": [45.0]})
    row = ref.comparison(meta, None, tmp_path / "x_data.csv")[0]
    assert row["app_deg"] == pytest.approx(41.0) and row["difference_deg"] == pytest.approx(-4.0) and row["app_basis"] == ref.APP_BASIS_FIRST_SECOND
    assert ref.comparison(meta)[0]["app_deg"] is None and ref.comparison(meta)[0]["difference_deg"] is None     # no CSV given: not made up


def test_an_extra_reading_has_no_app_angle_and_keeps_its_title():
    meta = meta_with({"hip_flexion": [2.0]}, titles={"hip_flexion": "Hip flexion (typed)"}, angles=("abd_goniometer",))
    row = ref.comparison(meta)[0]
    assert row["title"] == "Hip flexion (typed)" and row["app_deg"] is None and row["difference_deg"] is None and row["reference_deg"] == 2.0


# --------------------------------------------------------------------------------------------------------------------- agreement
def pair(key, reference, app, pid):
    return {"key": key, "title": key.title(), "reference_deg": reference, "app_deg": app, "difference_deg": app - reference,
            "instrument": "goniometer", "pair_id": pid}


def test_agreement_bias_spread_and_limits():
    rows = [pair("knee", r, r + d, f"p{i}") for i, (r, d) in enumerate([(30, 1.0), (60, -1.0), (90, 0.0), (120, 2.0)])]
    s = ref.agreement(rows)["knee"]
    assert s["n"] == 4 and s["bias_deg"] == pytest.approx(0.5) and s["sd_deg"] == pytest.approx(1.2910, abs=1e-3)
    assert s["limits_deg"] == pytest.approx((0.5 - 1.96 * 1.2910, 0.5 + 1.96 * 1.2910), abs=1e-3)
    assert s["max_abs_deg"] == 2.0 and s["mean_abs_deg"] == pytest.approx(1.0) and s["acceptance_met"] is False
    assert s["pairs"][0] == (30, 31.0)


def test_a_reading_shared_by_two_trials_of_one_position_counts_once():
    rows = [pair("knee", 60, 61.0, "same"), pair("knee", 60, 61.0, "same"), pair("knee", 90, 91.0, "other")]
    assert ref.agreement(rows)["knee"]["n"] == 2


def test_one_pair_gives_a_bias_but_no_spread_and_no_verdict():
    s = ref.agreement([pair("knee", 60, 60.2, "a")])["knee"]
    assert s["n"] == 1 and s["bias_deg"] == pytest.approx(0.2) and math.isnan(s["sd_deg"]) and s["acceptance_met"] is None


def test_the_proposed_acceptance_can_be_met():
    rows = [pair("knee", 60.0 + i, 60.0 + i + d, f"p{i}") for i, d in enumerate([0.1, -0.2, 0.15, -0.05, 0.0])]
    assert ref.agreement(rows)["knee"]["acceptance_met"] is True


def test_rows_without_both_numbers_do_not_count():
    rows = [pair("knee", 60, 61.0, "a"), {"key": "knee", "title": "Knee", "reference_deg": 60.0, "app_deg": None, "difference_deg": None,
                                         "instrument": "goniometer", "pair_id": "b"}]
    assert ref.agreement(rows)["knee"]["n"] == 1


def test_the_agreement_text_says_what_a_handheld_goniometer_cannot_show():
    text = ref.agreement_text(ref.agreement([pair("knee", 60.0 + i, 61.0 + i, f"p{i}") for i in range(3)]))
    assert "Knee: n = 3" in text and "+1.0 deg" in text and "cannot show agreement to +-1 deg" in text and "not met" in text
    assert "No trial has a reference reading" in ref.agreement_text({})


# --------------------------------------------------------------------------------------------- a folder of trials, and the table
def save_trial(folder, name, meta, csv_rows=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    meta["files"] = {"data.csv": f"{name}_data.csv"}
    (folder / f"{name}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if csv_rows:
        write_data_csv(folder / f"{name}_data.csv")


def test_a_folder_gives_one_row_per_reading_and_two_trials_of_one_position_are_one_pair(tmp_path):
    stamp = "2026-10-20T12:03:10"
    for n in ("T1", "T2"):                                                    # two trials of one position, one reading, one stamp
        save_trial(tmp_path / "P01", n, meta_with({"knee": [60.0]}, stamp=stamp, app={"knee": 59.0, "hip": 90.5}))
    save_trial(tmp_path / "P01", "T3", meta_with(None))                       # no reading: skipped
    meta4 = meta_with({"knee": [30.0]}, stamp="2026-10-20T12:30:00", app={"knee": 31.0, "hip": 90.0})
    meta4["target_deg"] = 30.0
    save_trial(tmp_path / "P02", "T4", meta4)
    rows = ref.collect_rows(tmp_path)
    assert sorted(r["trial"] for r in rows) == ["T1", "T2", "T4"]
    assert len({r["pair_id"] for r in rows}) == 2
    s = ref.agreement(rows)["knee"]
    assert s["n"] == 2 and s["bias_deg"] == pytest.approx((-1.0 + 1.0) / 2)


def test_readings_added_later_are_found_and_use_the_first_second_when_nothing_was_stamped(tmp_path):
    save_trial(tmp_path / "P01", "T1", meta_with(None))
    ref.write_sidecar(tmp_path / "P01" / "T1_meta.json", "goniometer", 1.0, {"knee": [43.0]})
    rows = ref.collect_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["source"] == "added later in Review" and rows[0]["app_basis"] == ref.APP_BASIS_FIRST_SECOND
    assert rows[0]["difference_deg"] == pytest.approx(41.0 - 43.0)


def test_the_table_of_a_folder_is_written_with_the_repeat_marked(tmp_path):
    for n in ("T1", "T2"):
        save_trial(tmp_path / "P01", n, meta_with({"knee": [60.0]}, stamp="2026-10-20T12:03:10", app={"knee": 59.0, "hip": 90.5}))
    out = ref.write_csv(ref.collect_rows(tmp_path), tmp_path / "reference_validation.csv")
    rows = list(csv.DictReader(open(out, newline="")))
    assert list(rows[0]) == ref.COLUMNS and len(rows) == 2
    assert [r["same_pair_as_another_trial"] for r in rows] == ["0", "1"]
    assert rows[0]["difference_app_minus_reference_deg"] == "-1.0" and rows[0]["instrument"] == "goniometer"


# --------------------------------------------------------------------------------------------------------------- in the engine
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)


def knee_engine(tmp_path, frames=8, pts=PTS, target=60.0):
    """A knee_extension engine at `target` with every dot of `pts` marked (also used by test_reference_review.py)."""
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = target
    eng.process(render(pts, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(pts[n] + 2))
    for i in range(frames):
        eng.process(render(pts, seed=i), 0.1 + i / 30)
    return eng


def test_the_moment_of_a_reading_keeps_the_apps_angles_of_that_moment(tmp_path):
    eng = knee_engine(tmp_path)
    app = eng.take_reference_stamp()
    assert app["knee"] == pytest.approx(60.0, abs=0.7) and app["hip"] == pytest.approx(90.0, abs=0.7)
    assert eng.reference.stamp["taken_at"] and eng.reference.position == ("knee_extension", 60.0, None)   # no participant code yet


def test_typed_readings_are_saved_with_the_trial_and_never_change_an_angle(tmp_path):
    eng = knee_engine(tmp_path)
    before = [a.raw_value for a in eng._last_angles]
    eng.take_reference_stamp()
    eng.set_reference({"knee": [61.0, 62.0], "hip": [89.0]}, instrument="inclinometer", resolution_deg=0.5)
    assert [a.raw_value for a in eng._last_angles] == before
    paths = eng.start_trial("P01")
    for i in range(5):
        eng.process(render(PTS, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    r = meta["reference_readings"]
    assert r["instrument"] == "inclinometer" and r["resolution_deg"] == 0.5
    assert r["readings"]["knee"] == {"values_deg": [61.0, 62.0], "mean_deg": 61.5} and r["readings"]["hip"]["mean_deg"] == 89.0
    assert r["app_at_reading_deg"]["knee"] == pytest.approx(60.0, abs=0.7) and r["entered_for"] == {"test_key": "knee_extension", "target_deg": 60.0, "participant": "P01"}
    assert r["titles"]["knee"] == "Knee flexion" and "changed_during_recording" not in meta


def test_a_trial_without_readings_says_so_by_leaving_the_field_null(tmp_path):
    eng = knee_engine(tmp_path)
    paths = eng.start_trial("P01")
    eng.process(render(PTS, seed=1), 1.0)
    eng.stop_trial()
    assert json.loads(paths["meta.json"].read_text())["reference_readings"] is None


def test_the_readings_cannot_change_while_recording_and_stay_for_the_second_trial_of_the_position(tmp_path):
    eng = knee_engine(tmp_path)
    eng.set_reference({"knee": [60.0]})
    eng.start_trial("P01")
    with pytest.raises(RuntimeError):
        eng.set_reference({"knee": [70.0]})
    with pytest.raises(RuntimeError):
        eng.take_reference_stamp()
    with pytest.raises(RuntimeError):
        eng.clear_reference()
    eng.process(render(PTS, seed=1), 1.0)
    eng.stop_trial()
    assert eng.reference.values == {"knee": [60.0]}                           # the same posture: the second trial gets the same reading
    paths = eng.start_trial("P01")
    eng.process(render(PTS, seed=2), 2.0)
    eng.stop_trial()
    assert json.loads(paths["meta.json"].read_text())["reference_readings"]["readings"]["knee"]["values_deg"] == [60.0]


def test_a_reading_typed_for_another_position_is_never_saved_with_this_trial(tmp_path):
    eng = knee_engine(tmp_path)
    eng.set_reference({"knee": [60.0]})
    assert eng.reference_for_this_position()
    eng.target = 70.0                                                         # the participant is moved to the next position
    assert not eng.reference_for_this_position()
    paths = eng.start_trial("P01")
    eng.process(render(PTS, seed=1), 1.0)
    eng.stop_trial()
    r = json.loads(paths["meta.json"].read_text())["reference_readings"]
    assert "readings" not in r and r["not_used"]["entered_for"] == {"test_key": "knee_extension", "target_deg": 60.0, "participant": None}
    assert r["not_used"]["this_trial"] == {"test_key": "knee_extension", "target_deg": 70.0, "participant": "P01"}


def test_typing_a_new_reading_at_a_new_position_starts_afresh(tmp_path):
    eng = knee_engine(tmp_path)
    eng.set_reference({"knee": [60.0], "hip": [90.0]})
    eng.target = 70.0
    eng.set_reference({"knee": [70.0]})
    assert eng.reference.values == {"knee": [70.0]} and eng.reference.position == ("knee_extension", 70.0, None)


def record(eng, participant, t0=1.0):
    paths = eng.start_trial(participant)
    for i in range(4):
        eng.process(render(PTS, seed=i), t0 + i / 30)
    eng.stop_trial()
    return json.loads(paths["meta.json"].read_text())


def test_readings_belong_to_the_participant_they_were_typed_for(tmp_path):
    eng = knee_engine(tmp_path)
    eng.participant_code = "P01"                                              # what the window says
    eng.take_reference_stamp()
    eng.set_reference({"knee": [61.0]})
    assert eng.reference.position == ("knee_extension", 60.0, "P01")
    first = record(eng, "P01")["reference_readings"]
    assert first["readings"]["knee"]["values_deg"] == [61.0] and first["entered_for"]["participant"] == "P01"
    eng.participant_code = "P02"                                              # the next participant, at the same position
    assert not eng.reference_for_this_position()
    nxt = record(eng, "P02", t0=2.0)["reference_readings"]
    assert "readings" not in nxt and nxt["not_used"]["entered_for"]["participant"] == "P01"
    assert nxt["not_used"]["this_trial"] == {"test_key": "knee_extension", "target_deg": 60.0, "participant": "P02"}


def test_the_trials_own_participant_decides_even_if_the_window_says_otherwise(tmp_path):
    eng = knee_engine(tmp_path)
    eng.participant_code = "P01"
    eng.set_reference({"knee": [61.0]})
    r = record(eng, "P02")["reference_readings"]                              # recorded for P02 (the engine is told so by start_trial)
    assert "readings" not in r and r["not_used"]["entered_for"]["participant"] == "P01"


def test_readings_typed_before_any_code_belong_to_the_first_participant_recorded(tmp_path):
    eng = knee_engine(tmp_path)
    eng.set_reference({"knee": [61.0]})                                       # no participant code in the window yet
    assert eng.reference.position == ("knee_extension", 60.0, None) and eng.reference_for_this_position()
    first = record(eng, "P01")
    assert first["reference_readings"]["readings"]["knee"]["values_deg"] == [61.0]
    assert eng.reference.position == ("knee_extension", 60.0, "P01") and "changed_during_recording" not in first
    eng.participant_code = "P01"
    second = record(eng, "P01", t0=2.0)["reference_readings"]                 # the second trial of the position: the same reading
    assert second["readings"]["knee"]["values_deg"] == [61.0]
    third = record(eng, "P02", t0=3.0)["reference_readings"]                  # someone else: never
    assert "readings" not in third


def test_only_what_the_test_takes_a_reading_for_is_accepted(tmp_path):
    eng = knee_engine(tmp_path)
    with pytest.raises(ValueError):
        eng.set_reference({"hip_flexion": [0.0]})                             # an extra of the frontal tests, not of this one
    with pytest.raises(ValueError):
        eng.set_reference(instrument="ruler")
    with pytest.raises(ValueError):
        eng.set_reference(resolution_deg=0.0)
    assert [k for k, _ in eng.reference_keys()] == ["knee", "hip"]          # (0.24.0: no ankle in the knee test)
    squat = Engine(TESTS, recordings_dir=tmp_path / "squat")
    squat.set_test("belt_squat")                                              # the tests with the foot still take an ankle reading
    assert [k for k, _ in squat.reference_keys()] == ["knee", "hip", "ankle"]


def test_the_stamp_needs_angles_to_keep_and_no_break(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    with pytest.raises(ValueError):
        eng.take_reference_stamp()                                            # nothing marked yet
    eng = knee_engine(tmp_path / "b")
    assert eng.pause() is None
    with pytest.raises(RuntimeError):
        eng.take_reference_stamp()


def test_a_reading_carried_to_the_second_trial_without_a_stamp_is_one_pair(tmp_path):
    r = ref.ReferenceReadings()
    r.set_values("knee", [61.0])
    first = r.typed_at
    r.set_values("knee", [61.0])                                              # the same numbers sent again: still the same reading
    assert first and r.typed_at == first
    eng = knee_engine(tmp_path)
    eng.participant_code = "P01"
    eng.set_reference({"knee": [61.0]})                                       # typed once, no 'reading taken now'
    record(eng, "P01")
    record(eng, "P01", t0=2.0)                                                # the second trial of the position keeps it
    assert ref.agreement(ref.collect_rows(tmp_path))["knee"]["n"] == 1
    eng.set_reference({"knee": [62.0]})                                       # read again: a new reading, a new pair
    record(eng, "P01", t0=3.0)
    assert ref.agreement(ref.collect_rows(tmp_path))["knee"]["n"] == 2


def test_the_stamp_needs_a_fresh_picture(tmp_path):
    eng = knee_engine(tmp_path)
    eng._frame_stamp -= 5.0                                                   # no picture for 5 s: the camera stopped or dropped out
    with pytest.raises(ValueError, match="No new picture has come in for 5 s"):
        eng.take_reference_stamp()
    assert eng.reference.stamp is None
    eng.process(render(PTS, seed=9), 1.0)                                     # the picture is back
    assert eng.take_reference_stamp()["knee"] == pytest.approx(60.0, abs=0.7)


def test_marking_the_dots_again_forgets_the_angles_kept_for_the_stamp(tmp_path):
    eng = knee_engine(tmp_path)
    eng.reset_dots()                                                          # no new picture since: nothing of the old dots may be kept
    with pytest.raises(ValueError, match="mark the dots first"):
        eng.take_reference_stamp()


def test_a_frontal_test_takes_the_typed_hip_flexion_and_knee_too(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("hip_abduction")
    assert [k for k, _ in eng.reference_keys()] == ["abd_goniometer", "abd_hip_centre", "hip_flexion", "knee_flexion"]
    p = dict(frontal_points(5.0))
    p["ruler_a"], p["ruler_b"] = [120.0, 690.0], [270.0, 690.0]
    import numpy as np
    p["ruler_a"], p["ruler_b"] = np.array(p["ruler_a"]), np.array(p["ruler_b"])
    eng.process(render(p, dot_radius=11.0, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    for i in range(6):
        eng.process(render(p, dot_radius=11.0, seed=i), 0.1 + i / 30)
    eng.take_reference_stamp()
    eng.set_reference({"abd_goniometer": [0.0], "hip_flexion": [1.0, 2.0], "knee_flexion": [0.0]})
    paths = eng.start_trial("P01")
    for i in range(4):
        eng.process(render(p, dot_radius=11.0, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    r = json.loads(paths["meta.json"].read_text())["reference_readings"]
    assert r["readings"]["hip_flexion"]["mean_deg"] == 1.5 and r["titles"]["hip_flexion"].startswith("Hip flexion")
    assert r["app_at_reading_deg"]["abd_goniometer"] == pytest.approx(0.0, abs=0.7)
    row = {x["key"]: x for x in ref.comparison(json.loads(paths["meta.json"].read_text()))}
    assert row["abd_goniometer"]["difference_deg"] == pytest.approx(0.0, abs=0.7) and row["hip_flexion"]["app_deg"] is None
