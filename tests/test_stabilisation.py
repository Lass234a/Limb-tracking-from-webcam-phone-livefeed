"""0.34.0 (F5): the stabilisation measure. From the dot tracks, how far the pelvis dots moved and how far the pelvis line and the neighbouring segment
that must stay still turned between the rest level (the resting force's stretch) and the peak force. Pure rules on made-up tracks, then three real
recordings (a moving pelvis with a ruler, a still one, a moving one without a ruler) read back through Review, the checks list and the export."""
import csv
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import analysis, stabilisation as st  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import SimulatedForce  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
P = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)        # the crest is 95 px above the trochanter, the thigh is 190 px long
RULER_PX, RULER_CM = 200.0, 40.0                             # 5 px per cm


# ------------------------------------------------------------------------------------------------ the pure part, on made-up tracks
def tracks_for(t, crest_move=(0.0, 0.0), troch_move=(0.0, 0.0), epi_move=(0.0, 0.0), move_from=6.0):
    """Static dots until `move_from` s, then each dot is displaced by its move (a step), as seen at the peak force at 7 s."""
    after = t >= move_from

    def track(base, move):
        return (np.where(after, base[0] + move[0], base[0]), np.where(after, base[1] + move[1], base[1]))
    return {"iliac_crest": track((500.0, 75.0), crest_move), "trochanter": track((500.0, 170.0), troch_move),
            "epicondyle": track((690.0, 170.0), epi_move)}


T = np.arange(0.0, 10.0, 1 / 30)


def measure(tracks, rule, **kw):
    kw.setdefault("px_per_cm", 5.0)
    return st.measure(T, tracks, 1.0, 3.5, 7.0, rule=rule, **kw)


def test_the_shift_in_cm_and_the_turn_of_the_pelvis_line_from_rest_to_the_peak():
    d = 95 * math.tan(math.radians(4.0))                     # the crest swings 4 degrees about the trochanter (clockwise on screen)
    res = measure(tracks_for(T, crest_move=(d, 0.0)), st.RULES["hip_flexion"])
    assert res["available"] and res["unit"] == "cm"
    crest, troch = res["dots"]
    assert (crest["name"], troch["name"]) == ("iliac_crest", "trochanter")
    assert crest["shift"] == pytest.approx(d / 5.0, abs=1e-6) and crest["dx"] == pytest.approx(d / 5.0) and crest["dy"] == pytest.approx(0.0)
    assert troch["shift"] == pytest.approx(0.0, abs=1e-9)
    assert res["pelvis_rot_deg"] == pytest.approx(4.0, abs=1e-6)                     # the same numbers worked out by hand
    assert res["still_rot_deg"] is None and res["still_segment"] == ""               # a hip test has no second segment
    assert res["rest_pictures"] > 50 and res["peak_pictures"] >= 5


def test_a_flag_is_raised_above_the_proposed_limits_only():
    assert st.PROPOSED_DISPLACEMENT_LIMIT_CM == 1.0 and st.PROPOSED_ROTATION_LIMIT_DEG == 3.0
    d = 95 * math.tan(math.radians(4.0))                     # 1.33 cm and 4 deg: both above
    both = measure(tracks_for(T, crest_move=(d, 0.0)), st.RULES["hip_flexion"])
    assert len(both["flags"]) == 2 and "iliac crest dot moved 1.3 cm" in both["flags"][0] and "pelvis line turned 4.0 deg" in both["flags"][1]
    small = 95 * math.tan(math.radians(2.0))                 # 0.66 cm and 2 deg: both below
    ok = measure(tracks_for(T, crest_move=(small, 0.0)), st.RULES["hip_flexion"])
    assert ok["flags"] == [] and ok["pelvis_rot_deg"] == pytest.approx(2.0, abs=1e-6)
    edge = measure(tracks_for(T, troch_move=(0.0, 5.0 * st.PROPOSED_DISPLACEMENT_LIMIT_CM)), st.RULES["hip_flexion"])      # exactly at the limit: not above it
    assert not any("dot moved" in f for f in edge["flags"])


def test_the_sign_is_clockwise_for_a_participant_facing_right_and_the_same_movement_for_one_facing_left():
    d = 95 * math.tan(math.radians(4.0))
    right = measure(tracks_for(T, crest_move=(d, 0.0)), st.RULES["hip_flexion"], facing_right=True)
    left = measure(tracks_for(T, crest_move=(d, 0.0)), st.RULES["hip_flexion"], facing_right=False)
    assert right["pelvis_rot_deg"] == pytest.approx(4.0) and left["pelvis_rot_deg"] == pytest.approx(-4.0)
    frontal = measure(tracks_for(T, crest_move=(d, 0.0)), st.RULES["hip_flexion"], facing_right=False, frontal=True)         # a frontal picture is not mirrored
    assert frontal["pelvis_rot_deg"] == pytest.approx(4.0)


def test_the_turn_wraps_through_180_not_around_it():
    far = (np.full(T.size, 600.0), np.where(T >= 6.0, 82.0, 75.0))                      # the line crest -> trochanter points left (180 deg), then 4 deg off it
    near = (np.full(T.size, 500.0), np.full(T.size, 75.0))
    res = measure({"iliac_crest": far, "trochanter": near}, st.RULES["hip_flexion"])
    assert res["pelvis_rot_deg"] == pytest.approx(math.degrees(math.atan2(7.0, 100.0)), abs=1e-9)    # +4.0, not -356


def test_a_knee_test_also_judges_the_thigh_and_an_ankle_test_the_shank():
    tracks = tracks_for(T)
    rot = math.radians(5.0)                                   # the epicondyle swings 5 degrees about the trochanter: the thigh turns
    tracks["epicondyle"] = (np.where(T >= 6.0, 500 + 190 * math.cos(rot), 690.0), np.where(T >= 6.0, 170 + 190 * math.sin(rot), 170.0))
    knee = measure(tracks, st.RULES["knee_extension"])
    assert knee["still_segment"] == "thigh" and knee["still_rot_deg"] == pytest.approx(5.0, abs=1e-6) and knee["pelvis_rot_deg"] == pytest.approx(0.0, abs=1e-6)
    assert [f for f in knee["flags"]] == ["the thigh turned 5.0 deg (proposed limit 3 deg)"]
    tracks.update(fibular_head=(np.full(T.size, 700.0), np.full(T.size, 200.0)),
                  malleolus=(np.where(T >= 6.0, 700.0 + 4.0, 700.0), np.full(T.size, 380.0)))
    ankle = measure(tracks, st.RULES["ankle_dorsiflexion"])
    assert ankle["still_segment"] == "shank" and ankle["still_rot_deg"] == pytest.approx(math.degrees(math.atan2(180, 4)) - 90.0, abs=1e-6)


def test_the_rule_of_each_test_is_written_down_and_covers_the_whole_protocol():
    assert set(st.RULES) == set(TESTS)                                        # a new test in protocol.json must be given a rule on purpose
    assert {k: (r.still_name, r.judged) for k, r in st.RULES.items()} == {
        "knee_extension": ("thigh", True), "knee_flexion": ("thigh", True), "hip_extension_k0": ("", True), "hip_extension_k90": ("", True),
        "hip_flexion": ("", True), "ankle_plantarflexion": ("shank", True), "ankle_dorsiflexion": ("shank", True), "hip_abduction": ("", True),
        "hip_adduction": ("", True), "belt_squat": ("", False)}
    assert st.RULES["hip_abduction"].pelvis_dots == ("asis_opposite", "asis") and st.RULES["hip_abduction"].shown_dots == ("asis", "asis_opposite")
    assert st.rule_for("a_test_from_the_future", "knee") is st.RULES["knee_extension"]                 # by the main joint
    assert st.rule_for("a_test_from_the_future", "ankle") is st.RULES["ankle_dorsiflexion"]
    assert st.rule_for("a_test_from_the_future", None) is st.RULES["hip_flexion"] and st.rule_for(None, "x", frontal=True) is st.RULES["hip_abduction"]


def test_the_belt_squat_is_measured_but_never_flagged():
    d = 95 * math.tan(math.radians(10.0))
    res = measure(tracks_for(T, crest_move=(d, 30.0)), st.RULES["belt_squat"])
    assert res["available"] and res["judged"] is False and res["flags"] == [] and res["pelvis_rot_deg"] > 8
    level, text, title = st.check_item(res)
    assert level == "ok" and "not judged in this test" in text and "closed chain" in text and title == "Stabilisation: not judged"


def test_a_frontal_pelvis_uses_the_two_asis():
    tracks = {"asis": (np.full(T.size, 760.0), np.where(T >= 6.0, 230.0, 200.0)), "asis_opposite": (np.full(T.size, 520.0), np.full(T.size, 200.0))}
    res = measure(tracks, st.RULES["hip_abduction"], frontal=True)
    assert [d["name"] for d in res["dots"]] == ["asis", "asis_opposite"] and res["dots"][0]["shift"] == pytest.approx(6.0)
    assert res["pelvis_rot_deg"] == pytest.approx(math.degrees(math.atan2(30, 240)), abs=1e-6)        # the line asis_opposite -> asis tipped down on the right: clockwise


def test_without_a_ruler_the_shifts_are_pixels_and_not_judged_but_the_turns_are_still_flagged():
    d = 95 * math.tan(math.radians(4.0))
    res = measure(tracks_for(T, crest_move=(d, 0.0)), st.RULES["hip_flexion"], px_per_cm=None)
    assert res["unit"] == "px" and res["dots"][0]["shift"] == pytest.approx(d)
    assert res["flags"] == ["the pelvis line turned 4.0 deg (proposed limit 3 deg)"]
    ok = measure(tracks_for(T, crest_move=(10.0, 0.0)), st.RULES["hip_flexion"], px_per_cm=None)           # 10 px would be 2 cm: no way to say
    assert not any("dot moved" in f for f in ok["flags"])


def test_a_dot_that_is_lost_leaves_blanks_not_invented_numbers():
    tracks = tracks_for(T, crest_move=(8.0, 0.0))
    x, y = tracks["iliac_crest"]
    lost_at_peak = (np.where(np.abs(T - 7.0) <= 0.35, np.nan, x), np.where(np.abs(T - 7.0) <= 0.35, np.nan, y))
    res = measure({**tracks, "iliac_crest": lost_at_peak}, st.RULES["knee_extension"])
    assert res["available"] and res["dots"][0]["shift"] is None and res["pelvis_rot_deg"] is None            # the crest has no peak position
    assert res["dots"][1]["shift"] == pytest.approx(0.0, abs=1e-9)
    cells = st.export_cells(res)
    assert cells["stab_dot1_name"] == "iliac_crest" and cells["stab_dot1_shift"] == "" and cells["stab_pelvis_rot_deg"] == ""
    assert cells["stab_still_segment"] == "thigh" and cells["stab_unit"] == "cm"
    gone = {k: v for k, v in tracks.items() if k != "trochanter"}                                           # a dot that does not exist in the trial
    assert measure(gone, st.RULES["hip_flexion"])["pelvis_rot_deg"] is None
    nothing = measure({}, st.RULES["hip_flexion"])
    assert nothing["available"] is False and "not tracked" in nothing["why"]


def test_a_rest_stretch_with_too_few_pictures_gives_nothing():
    res = st.measure(T, tracks_for(T), 12.0, 14.0, 7.0, 5.0, rule=st.RULES["hip_flexion"])
    assert res == {"available": False, "why": "there are too few pictures in the resting stretch before the push"}
    assert st.export_cells(res) == {c: "" for c in st.COLUMNS} and st.check_item(res) is None and st.check_item(None) is None


def test_the_peak_pictures_are_the_ones_of_the_angle_at_the_peak_and_the_picture_delay_moves_them():
    tracks = tracks_for(T, crest_move=(10.0, 0.0), move_from=7.25)                     # the crest moves at 7.25 s
    now = measure(tracks, st.RULES["hip_flexion"])
    late = measure(tracks, st.RULES["hip_flexion"], picture_delay=0.5)                 # the pictures 0.5 s later are the peak's
    assert now["dots"][0]["shift"] == pytest.approx(0.0, abs=1e-9) and late["dots"][0]["shift"] == pytest.approx(2.0)
    near = np.abs(T - 7.0) <= 0.1
    assert now["peak_pictures"] == int(near.sum())
    between = st.measure(np.array([0.0, 1.5, 2.0, 2.5, 3.0, 9.0]), {"iliac_crest": (np.arange(6.0) * 0 + 5, np.arange(6.0) * 0 + 5)}, 1.0, 3.5, 7.0,
                         rule=st.RULES["hip_flexion"])                                  # no picture within 0.1 s of the peak: the nearest one
    assert between["peak_pictures"] == 1


def test_the_check_lines_say_what_moved_and_that_the_limits_are_a_proposal():
    d = 95 * math.tan(math.radians(4.0))
    bad = st.check_item(measure(tracks_for(T, crest_move=(d, 0.0)), st.RULES["knee_extension"]))
    assert bad[0] == "check" and bad[2] == "Pelvis or segment moved under load" and len(bad[2]) <= analysis.CHECK_TITLE_MAX
    assert "the group has not confirmed one" in bad[1] and "iliac crest dot moved 1.3 cm" in bad[1] and "the thigh turned" in bad[1]
    good = st.check_item(measure(tracks_for(T), st.RULES["knee_extension"]))
    assert good[0] == "ok" and "Within the proposed limits (1 cm, 3 deg)" in good[1] and good[2] == "Stabilisation within limits"
    px = st.check_item(measure(tracks_for(T), st.RULES["knee_extension"], px_per_cm=None))
    assert px[0] == "ok" and "no ruler: the shifts are in pixels and not judged" in px[1]


# ------------------------------------------------------------------------------------------------ real recordings
def moving_trial(root, crest_px=0.0, ruler=True, test="knee_extension", thigh_deg=0.0):
    """A real recording in real time with a simulated load cell: the crest dot (and, with `thigh_deg`, the thigh) moves in step with the force, so
    at the peak the crest is `crest_px` to the right of where it rested. Returns the trial's meta.json path."""
    eng = Engine(TESTS, recordings_dir=root)
    eng.set_test(test)
    eng.target = 70.0
    eng.process(render(P, seed=0), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    if ruler:
        eng.scale.length_cm = RULER_CM
        eng.scale.ref_px = RULER_PX
    force = SimulatedForce(rest_s=3.9, ramp_s=0.6, hold_s=1.2, peak_n=400.0)
    force.start()
    eng.force = force
    time.sleep(0.3)
    eng.participant_code = "P01"
    paths = eng.start_trial("P01")
    rot = math.radians(thigh_deg)
    while True:
        now = time.perf_counter()
        el = now - force._t_start
        if el > 6.6:
            break
        f = float(force.newtons_at(el)) / 400.0                                     # 0 at rest, 1 in the hold
        pts = {k: v.copy() for k, v in P.items()}
        pts["iliac_crest"] = pts["iliac_crest"] + np.array([crest_px * f, 0.0])
        if thigh_deg:
            tro = pts["trochanter"]
            v = pts["epicondyle"] - tro
            a = rot * f
            pts["epicondyle"] = tro + np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])
        eng.process(render(pts, seed=int(el * 30) % 7), now)
        time.sleep(1 / 40)
    eng.stop_trial()
    force.stop()
    return paths["meta.json"]


@pytest.fixture(scope="module")
def trials(tmp_path_factory):
    root = tmp_path_factory.mktemp("stab")
    return {"moving": moving_trial(root / "a", crest_px=12.0, thigh_deg=5.0),
            "still": moving_trial(root / "b"),
            "noruler": moving_trial(root / "c", crest_px=12.0, ruler=False)}


def window(tr):
    return tr.force_window() or (float(tr.t[0]), float(tr.t[-1]))


def test_a_pelvis_that_moved_in_a_real_recording_is_measured_and_flagged(trials):
    tr = Trial(trials["moving"], open_video=False)
    try:
        assert tr.px_per_cm == pytest.approx(5.0)
        res = st.of_trial(tr, *window(tr))
        assert res["available"] and res["unit"] == "cm"
        assert res["dots"][0]["name"] == "iliac_crest" and res["dots"][0]["shift"] == pytest.approx(12.0 / 5.0, abs=0.35)          # 2.4 cm, within tracking
        assert res["dots"][1]["shift"] < 0.3                                          # the trochanter stayed
        assert res["pelvis_rot_deg"] == pytest.approx(math.degrees(math.atan2(12.0, 95.0)), abs=1.5)                       # about 7.2 deg
        assert res["still_segment"] == "thigh" and res["still_rot_deg"] == pytest.approx(5.0, abs=1.5)
        assert len(res["flags"]) == 3
        again = st.of_trial(tr, *window(tr), fs=tr.force_summary(*window(tr)))
        assert again == res                                                           # a summary handed in gives the same answer
        ri = tr.force_rest()
        assert res["rest_pictures"] == int(((tr.t >= ri["t_from"]) & (tr.t < ri["t_to"])).sum())              # the resting force's stretch
    finally:
        tr.close()


def test_a_still_pelvis_is_within_the_limits(trials):
    tr = Trial(trials["still"], open_video=False)
    try:
        res = st.of_trial(tr, *window(tr))
        assert res["available"] and res["flags"] == []
        assert all(d["shift"] < 0.3 for d in res["dots"]) and abs(res["pelvis_rot_deg"]) < 1.0 and abs(res["still_rot_deg"]) < 1.0
    finally:
        tr.close()


def test_no_ruler_in_a_real_recording_gives_pixels(trials):
    tr = Trial(trials["noruler"], open_video=False)
    try:
        res = st.of_trial(tr, *window(tr))
        assert tr.px_per_cm is None and res["unit"] == "px" and res["dots"][0]["shift"] == pytest.approx(12.0, abs=1.5)
        assert not any("dot moved" in f for f in res["flags"]) and any("pelvis line turned" in f for f in res["flags"])
    finally:
        tr.close()


def test_the_checks_list_has_the_line_and_the_still_trial_has_a_green_one(trials):
    tr = Trial(trials["moving"], open_video=False)
    try:
        items = analysis.trial_check_items(tr, *window(tr))
        got = [i for i in items if i["title"] == "Pelvis or segment moved under load"]
        assert len(got) == 1 and got[0]["level"] == "check" and got[0]["area"] == "position" and "proposed limit" in got[0]["text"]
        assert [(i["level"], i["text"]) for i in items] == analysis.trial_checks(tr, *window(tr))
    finally:
        tr.close()
    tr = Trial(trials["still"], open_video=False)
    try:
        ok = [i for i in analysis.trial_check_items(tr, *window(tr)) if i["title"].startswith("Stabilisation")]
        assert len(ok) == 1 and ok[0]["level"] == "ok"
    finally:
        tr.close()


def test_the_export_row_has_the_new_columns_at_the_end_and_the_old_ones_in_the_same_order(trials):
    tr = Trial(trials["moving"], open_video=False)
    try:
        cols = analysis.SESSION_COLUMNS
        assert cols[-len(st.COLUMNS) - 11 - 2: -11 - 2] == st.COLUMNS and cols[:3] == ["trial", "position_number", "trial_number"]
        assert cols.index("reference_app_basis") == 65 and cols[66] == "stab_unit"         # appended after the 66 that were there
        assert len(cols) == len(set(cols)) == 66 + len(st.COLUMNS) + 11 + 2         # (+ the two check weight columns of 0.35.0, appended after the standing ones)
        row = analysis.session_row(tr)
        res = st.of_trial(tr, *window(tr))
        assert row["stab_unit"] == "cm" and row["stab_dot1_name"] == "iliac_crest" and row["stab_dot1_shift"] == pytest.approx(res["dots"][0]["shift"])
        assert row["stab_dot2_name"] == "trochanter" and row["stab_pelvis_rot_deg"] == pytest.approx(res["pelvis_rot_deg"])
        assert row["stab_still_segment"] == "thigh" and row["stab_still_rot_deg"] == pytest.approx(res["still_rot_deg"])
        assert row["stab_flags"].count(";") == 2 and "proposed limit" in row["stab_flags"]
        out = analysis.write_session_csv([row], Path(trials["moving"]).parent / "summary_stab.csv")             # a real file, read back
        back = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
        assert list(back[0]) == cols and back[0]["stab_unit"] == "cm" and back[0]["stab_dot1_name"] == "iliac_crest"
        assert float(back[0]["stab_dot1_shift"]) == pytest.approx(res["dots"][0]["shift"], abs=1e-3) and back[0]["stab_still_segment"] == "thigh"
    finally:
        tr.close()


def test_the_numbers_that_were_there_do_not_change(trials, monkeypatch):
    """The same trial with the new measure switched off gives the same value in every one of the 66 old columns, except that the text column
    of checks to look at has one more line when the pelvis moved."""
    tr = Trial(trials["moving"], open_video=False)
    try:
        row = analysis.session_row(tr)
        monkeypatch.setattr(st, "of_trial", lambda *a, **k: {"available": False, "why": "off"})
        monkeypatch.setattr(analysis.standing_reference, "export_cells", lambda meta: {c: "" for c in analysis.standing_reference.COLUMNS})
        old = analysis.session_row(tr)
        assert row["checks_to_look_at"].startswith(old["checks_to_look_at"])          # the list of checks to look at gains the stabilisation line, nothing else
        assert row["checks_to_look_at"][len(old["checks_to_look_at"]):].startswith(" | Stabilisation: ")
        for c in (c for c in analysis.SESSION_COLUMNS[:66] if c != "checks_to_look_at"):
            assert row[c] == old[c] or (isinstance(row[c], float) and math.isnan(row[c]) and math.isnan(old[c])), c
        assert row["stab_unit"] == "cm" and old["stab_unit"] == ""
    finally:
        tr.close()


def test_a_trial_with_the_pelvis_dots_missing_exports_blanks(trials, tmp_path):
    """A copy of the trial whose crest dot was never tracked (data.csv: lost in every picture): the stabilisation cells are blank, nothing invented,
    and the row is still written."""
    src = Path(trials["moving"]).parent
    dst = tmp_path / "P01"
    shutil.copytree(src, dst)
    meta = next(dst.glob("*_meta.json"))
    data = next(dst.glob("*_data.csv"))
    rows = list(csv.DictReader(open(data, newline="", encoding="utf-8")))
    for r in rows:
        r["iliac_crest_x"], r["iliac_crest_y"], r["iliac_crest_lost"] = "", "", "1"
    with open(data, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tr = Trial(meta, open_video=False)
    try:
        row = analysis.session_row(tr)
        assert row["stab_dot1_name"] == "iliac_crest" and row["stab_dot1_shift"] == "" and row["stab_pelvis_rot_deg"] == ""
        assert row["stab_dot2_shift"] != "" and row["stab_still_rot_deg"] != ""           # the trochanter and the thigh could still be measured
        analysis.write_session_csv([row], tmp_path / "x.csv")
    finally:
        tr.close()


def test_a_trial_without_force_or_recorded_before_this_exports_blank(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(P, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(10):
        eng.process(render(P, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    tr = Trial(paths["meta.json"], open_video=False)
    try:
        assert st.of_trial(tr, 0.0, 1.0) == {"available": False, "why": "this trial has no force"}
        row = analysis.session_row(tr)
        assert all(row[c] == "" for c in st.COLUMNS) and all(not str(row[c]) for c in ("standing_taken_at", "standing_hip_offset_deg"))
        assert not any(i["title"].startswith(("Stabilisation", "Pelvis or")) for i in analysis.trial_check_items(tr))        # no line, not a made-up one
    finally:
        tr.close()


def test_review_shows_the_line_in_its_checks_and_the_overview_cache_is_a_new_format(trials):
    app = QApplication.instance() or QApplication([])
    root = Path(trials["moving"]).parent.parent
    tab = ReviewTab(root)
    tab.load(str(trials["moving"]))
    try:
        text = tab.checks.text(everything=True)
        assert "Stabilisation: From rest to the peak force the iliac crest dot moved" in text and "proposed limit" in text, text
    finally:
        tab.trial.close()
        tab.close()
    from limbtrack import review_layer
    assert review_layer.OVERVIEW_CACHE_FORMAT >= 3         # a cache of 0.33.0 has rows without the new columns: it is worked out again (4 since 0.35.0: the check weight columns)
    assert app is not None
