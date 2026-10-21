"""Hip abduction / adduction with the camera in front of the participant (0.15.0): the angle of Tabel 3 and the angle about the computed hip
centre, live, in the CSV, and in Review."""
import csv
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.geometry import hip_centre_frontal  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.results import LOST  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import frontal_points, leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
DOT_R = 11.0                       # 2.2 cm discs at 10 px/cm


def scene(deg=0.0, leg="right", roll=0.0):
    p = dict(frontal_points(deg, tested_leg=leg, roll_deg=roll))
    p["ruler_a"] = np.array([120.0, 690.0])                    # a 15 cm ruler at 10 px/cm (1 px per mm, like the pelvis)
    p["ruler_b"] = np.array([270.0, 690.0])
    return p


def pic(p, i=0):
    return render(p, dot_radius=DOT_R, seed=i)


def engine(tmp_path, test="hip_abduction", deg=0.0, leg="right", roll=0.0, ruler=True):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.tested_leg = leg
    p = scene(deg, leg, roll)
    eng.process(pic(p), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    if ruler:
        assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    return eng, p


def run(eng, p, frames=6):
    res = None
    for i in range(frames):
        res = eng.process(pic(p, i), 0.1 + i / 30)
    return res


def angle(res, name):
    return next(a for a in res.angles if a.name == name)


def goniometer_by_hand(p, sign=1):
    """Tabel 3 worked out independently of the program: the angle at the tested ASIS between the line to the other ASIS and the thigh, minus 90."""
    a, b = p["asis_opposite"] - p["asis"], p["patella"] - p["asis"]
    interior = math.degrees(math.acos(float(np.dot(a, b)) / (np.hypot(*a) * np.hypot(*b))))
    return sign * (interior - 90.0)


@pytest.mark.parametrize("deg", [0.0, 10.0, 20.0])
def test_abduction_both_angles_are_read_live(tmp_path, deg):
    eng, p = engine(tmp_path, "hip_abduction", deg)
    res = run(eng, p)
    gon, hjc = angle(res, "abd_goniometer"), angle(res, "abd_hip_centre")
    assert gon.raw_value == pytest.approx(goniometer_by_hand(p), abs=0.3)
    assert hjc.raw_value == pytest.approx(deg, abs=0.3)                      # the thigh was turned by exactly this much about the hip centre
    assert set(res.derived) == {"hip_centre"}
    assert res.derived["hip_centre"] == pytest.approx(hip_centre_frontal(p["asis"], p["asis_opposite"], 1.0), abs=1.0)


def test_the_two_angles_do_not_share_a_zero(tmp_path):
    """The goniometer reads 0 when the line from the ASIS to the patella is square to the pelvis. The hip centre lies about 35 mm medial to the
    ASIS (Harrington, PW 250 mm), so for a 400 mm thigh that line is square when the thigh itself leans out by about 5 deg about the hip
    centre: the two angles are 4 to 8 deg apart over the 0 to 20 deg range of the protocol, and the gap grows with the angle."""
    gaps = []
    for deg in (0.0, 5.0, 10.0, 20.0):
        eng, p = engine(tmp_path / f"d{deg:g}", "hip_abduction", deg)
        res = run(eng, p)
        gon, hjc = angle(res, "abd_goniometer").raw_value, angle(res, "abd_hip_centre").raw_value
        gaps.append(hjc - gon)
        if deg == 0.0:
            assert gon < -3.0                                                  # square thigh about the hip centre: the goniometer reads adduction
        if deg == 5.0:
            assert gon == pytest.approx(0.0, abs=0.7)                          # the goniometer's neutral
    assert all(3.0 < g < 9.0 for g in gaps) and gaps == sorted(gaps)


@pytest.mark.parametrize("deg", [10.0, 20.0])
def test_adduction_reads_positive_when_the_knee_moves_towards_the_midline(tmp_path, deg):
    eng, p = engine(tmp_path, "hip_adduction", -deg)
    res = run(eng, p)
    assert angle(res, "add_hip_centre").raw_value == pytest.approx(deg, abs=0.3)
    assert angle(res, "add_goniometer").raw_value == pytest.approx(goniometer_by_hand(p, sign=-1), abs=0.3)
    assert angle(res, "add_goniometer").raw_value > 0


@pytest.mark.parametrize("leg, roll", [("left", 0.0), ("right", 15.0), ("left", -20.0)])
def test_either_leg_and_a_rolled_picture_give_the_same_numbers(tmp_path, leg, roll):
    eng, p = engine(tmp_path, "hip_abduction", 12.0, leg=leg, roll=roll)
    res = run(eng, p)
    assert angle(res, "abd_hip_centre").raw_value == pytest.approx(12.0, abs=0.3)
    assert angle(res, "abd_goniometer").raw_value == pytest.approx(goniometer_by_hand(p), abs=0.3)
    assert eng.marking_problems() == []


def test_the_target_is_judged_on_the_goniometer_angle(tmp_path):
    eng, p = engine(tmp_path / "a", "hip_abduction", 5.0)                    # a thigh 5 deg out about the hip centre: the goniometer reads 0
    eng.target = 0.0
    res = run(eng, p)
    gon, hjc = angle(res, "abd_goniometer"), angle(res, "abd_hip_centre")
    assert gon.primary and gon.reference == 0.0 and abs(gon.raw_value) < 0.7 and gon.raw_status == "ok"
    assert not hjc.primary and hjc.reference is None                          # shown and saved, not judged against the typed target
    assert eng.lock_position()
    for i in range(21):                                                       # the thigh moves, 1 deg a picture, far from where it was locked
        res = eng.process(pic(scene(5.0 + i), i), 1.0 + i / 30)
    hjc = angle(res, "abd_hip_centre")
    assert hjc.raw_value == pytest.approx(25.0, abs=0.5)
    assert hjc.reference is None and hjc.status == "info" and hjc.raw_status == "info"          # never judged, not even against the locked value
    assert angle(res, "abd_goniometer").raw_status == "out"                   # the main joint is
    eng, p = engine(tmp_path / "b", "hip_abduction", 0.0)                    # a thigh square to the pelvis about the hip centre: the goniometer reads -4
    eng.target = 0.0
    gon = angle(run(eng, p), "abd_goniometer")
    assert gon.raw_status == "out" and gon.raw_deviation == pytest.approx(-4.1, abs=0.5)


def test_without_the_ruler_the_hip_centre_angle_is_blank_and_the_goniometer_angle_is_not(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 10.0, ruler=False)
    res = run(eng, p)
    assert angle(res, "abd_hip_centre").raw_status == LOST and math.isnan(angle(res, "abd_hip_centre").raw_value)
    assert angle(res, "abd_goniometer").raw_value == pytest.approx(goniometer_by_hand(p), abs=0.3)
    assert res.derived == {}
    assert any("HIP CENTRE needs the ruler" in h for h in res.header)


def test_without_the_ruler_the_position_can_still_be_locked_and_the_pictures_are_ok(tmp_path):
    """The hip-centre angle is shown and saved but never judged: blank without the ruler, it must not stop Lock or make 'all_ok' 0."""
    eng, p = engine(tmp_path, "hip_abduction", 5.0, ruler=False)              # the goniometer reads about 0 here
    eng.target = 0.0
    run(eng, p)
    assert eng.lock_position(), eng.lock_problem()
    paths = eng.start_trial("P01")
    for i in range(5):
        eng.process(pic(p, i), 1.0 + i / 30)
    eng.stop_trial()
    with open(paths["data.csv"], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(r["abd_hip_centre_deg"] == "" for r in rows) and all(r["abd_goniometer_ok"] == "1" for r in rows)
    assert all(r["all_ok"] == "1" for r in rows)
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["position_locked"] and "abd_hip_centre" not in meta["locked_references_deg"]


def test_a_lost_asis_takes_the_hip_centre_and_both_angles_with_it(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 10.0)
    run(eng, p)
    hidden = render(p, dot_radius=DOT_R, seed=9, hide=("asis_opposite",))
    for i in range(30):
        res = eng.process(hidden, 1.0 + i / 30)
    assert res.derived == {} and angle(res, "abd_hip_centre").raw_status == LOST and angle(res, "abd_goniometer").raw_status == LOST


def test_the_participant_faces_check_does_not_apply_to_a_frontal_test(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 10.0)
    eng.facing_right = False
    run(eng, p, 12)
    assert eng.facing_detected is None and not eng.facing_mismatch


def test_swapped_asis_dots_are_caught_before_they_turn_abduction_into_adduction(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 10.0, leg="right")
    eng.tested_leg = "left"                                                   # the right leg is on the picture's left: this says the opposite
    run(eng, p)
    problems = eng.marking_problems()
    assert problems and "tested ASIS should be on the right" in problems[0]
    assert not eng.lock_position()
    eng.tested_leg = "right"
    run(eng, p)
    assert eng.marking_problems() == [] and eng.lock_position()


def test_the_swapped_asis_check_follows_the_dot_names_of_the_protocol(tmp_path):
    text = (ROOT / "protocol.json").read_text(encoding="utf-8")
    text = text.replace('"asis_opposite"', '"asis_l"').replace('"asis"', '"asis_r"').replace('"patella"', '"knee_cap"')
    (tmp_path / "protocol.json").write_text(text, encoding="utf-8")
    eng = Engine(load_protocol(tmp_path / "protocol.json"), recordings_dir=tmp_path)
    eng.set_test("hip_abduction")
    assert eng._frontal_dots() == ("asis_r", "asis_l", "knee_cap")
    p = scene(10.0, "right")
    q = {"asis_r": p["asis"], "asis_l": p["asis_opposite"], "knee_cap": p["patella"]}
    eng.process(pic(p), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(q[n] + 2))
    eng.tested_leg = "left"                                                   # says the opposite of the picture
    run(eng, p)
    assert eng.marking_problems() and "tested ASIS should be on the right" in eng.marking_problems()[0]
    eng.tested_leg = "right"
    run(eng, p)
    assert eng.marking_problems() == []


def test_the_mirror_setting_turns_the_expected_side_round(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 10.0, leg="right")
    eng.mirror = True                                                         # a mirrored picture: the right leg is on the picture's right
    run(eng, p)
    assert eng.marking_problems()
    eng.tested_leg = "left"
    run(eng, p)
    assert eng.marking_problems() == []


def test_patella_and_asis_mixed_up_are_caught(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("hip_abduction")
    p = scene(10.0)
    eng.process(pic(p), 0.0)
    swapped = {"asis": p["patella"], "asis_opposite": p["asis_opposite"], "patella": p["asis"]}
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(swapped[n] + 2))
    run(eng, p)
    assert any("patella centre should be below the ASIS" in t for t in eng.marking_problems())


def test_the_tape_measure_checks_the_ruler_scale(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 10.0)
    res = run(eng, p)
    assert eng.pelvic_width_in_picture_mm() == pytest.approx(250.0, abs=2.0)
    assert eng.pelvic_width_difference_pct() is None and not any("CHECK SCALE" in h for h in res.header)       # no tape measure typed
    eng.tape_pelvic_width_mm = 250.0
    res = run(eng, p)
    assert abs(eng.pelvic_width_difference_pct()) < 1.5 and not any("CHECK SCALE" in h for h in res.header)
    eng.tape_pelvic_width_mm = 200.0                                          # the picture says 250 mm: 25 % more than the tape
    res = run(eng, p)
    assert eng.pelvic_width_difference_pct() == pytest.approx(25.0, abs=2.0)
    assert any("CHECK SCALE" in h for h in res.header)


def test_the_recording_holds_the_hip_centre_and_both_angles_and_says_how_they_were_made(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 20.0)
    eng.target = 20.0
    eng.tape_pelvic_width_mm = 248.0
    eng.wall_posture = True
    paths = eng.start_trial("P01")
    for i in range(8):
        eng.process(pic(p, i), 0.1 + i / 30)
    with pytest.raises(RuntimeError):
        eng.tested_leg = "left"                                              # frozen while recording
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    for col in ("hip_centre_x", "hip_centre_y", "abd_goniometer_deg", "abd_hip_centre_deg", "abd_goniometer_ref", "abd_hip_centre_ok"):
        assert col in rows[0], col
    assert float(rows[3]["abd_goniometer_ref"]) == 20.0 and rows[3]["abd_hip_centre_ref"] == "" and rows[3]["abd_hip_centre_ok"] == ""   # judged / not judged
    assert "hip_centre_lost" not in rows[0]                                  # a computed point has a position only
    x, y = hip_centre_frontal(p["asis"], p["asis_opposite"], 1.0)
    assert (float(rows[3]["hip_centre_x"]), float(rows[3]["hip_centre_y"])) == pytest.approx((x, y), abs=1.0)
    assert float(rows[3]["abd_hip_centre_deg"]) == pytest.approx(20.0, abs=0.3)
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["view"] == "frontal" and meta["tested_leg"] == "right" and meta["tape_pelvic_width_mm"] == 248.0
    assert meta["derived_points"]["definitions"]["hip_centre"]["kind"] == "hip_centre_frontal"
    assert "Harrington" in meta["derived_points"]["hip_centre_frontal"]["equation"] and "not checked" in meta["derived_points"]["hip_centre_frontal"]["status"]
    assert meta["hip_centre_scale"]["source"] == "ruler" and meta["hip_centre_scale"]["pelvic_width_tape_mm"] == 248.0
    assert meta["hip_centre_scale"]["pelvic_width_in_last_picture_mm"] == pytest.approx(250.0, abs=2.0)
    assert meta["sagittal_posture"] == {"controlled_by": "standing against a wall: heel, buttocks and back touching", "measured": False,
                                        "operator_ticked": True}
    assert meta["facing_not_used"] and "changed_during_recording" not in meta
    by_name = {a["name"]: a for a in meta["angles"]}
    assert by_name["abd_hip_centre"]["needs"] == ["asis", "asis_opposite", "patella"]
    assert by_name["abd_hip_centre"]["judged"] is False and by_name["abd_goniometer"]["judged"] is True
    assert meta["angles"][0]["name"] == "abd_goniometer" and meta["angles"][0]["primary"] is True


def test_review_opens_a_frontal_trial_and_draws_the_hip_centre(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 15.0)
    paths = eng.start_trial("P01")
    for i in range(8):
        eng.process(pic(p, i), 0.1 + i / 30)
    eng.stop_trial()
    tr = Trial(paths["meta.json"])
    assert tr.derived_names == ["hip_centre"]
    assert tr.deg["abd_hip_centre"][4] == pytest.approx(15.0, abs=0.3)
    plain = tr.drawn(4)
    assert plain is not None and plain.shape[:2] == (720, 1280)
    x, y = (int(round(v)) for v in hip_centre_frontal(p["asis"], p["asis_opposite"], 1.0))
    cyan = np.all(plain[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13] == (255, 255, 0), axis=2)
    assert cyan.any()                                                         # the computed point is drawn (a cyan diamond) where it was


def test_a_sagittal_trial_has_no_computed_columns_and_opens_as_before(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    q = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(q, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(q[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(5):
        eng.process(render(q, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["view"] == "sagittal" and meta["derived_points"] is None and meta["hip_centre_scale"] is None
    assert meta["tested_leg"] is None and meta["sagittal_posture"] is None
    assert Trial(paths["meta.json"]).derived_names == []
    assert "hip_centre_x" not in open(paths["data.csv"]).readline()


def test_the_live_tab_asks_which_leg_instead_of_which_way_the_participant_faces():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as d:
        win = MainWindow(TESTS, d)
        live = win.live
        live.test_combo.setCurrentIndex(live.test_combo.findData("knee_extension"))
        assert not live.facing_combo.isHidden() and live.leg_combo.isHidden() and live.tape_spin.isHidden() and live.chk_wall.isHidden()
        live.test_combo.setCurrentIndex(live.test_combo.findData("hip_abduction"))
        assert live.facing_combo.isHidden() and not live.leg_combo.isHidden() and not live.tape_spin.isHidden() and not live.chk_wall.isHidden()
        live.leg_combo.setCurrentIndex(live.leg_combo.findData("left"))
        assert win.engine.tested_leg == "left" and live.session_state()["tested_leg"] == "left"
        live.tape_spin.setValue(251.0)
        assert win.engine.tape_pelvic_width_mm == 251.0
        live.tape_spin.setValue(live.tape_spin.minimum())
        assert win.engine.tape_pelvic_width_mm is None                        # "not measured"
        live.chk_wall.setChecked(True)
        assert win.engine.wall_posture is True
        assert [r for r in live.plan_rows if r[0] in ("hip_abduction", "hip_adduction")] == [
            ("hip_abduction", 0), ("hip_abduction", 20), ("hip_adduction", 0), ("hip_adduction", 20)]
        live.test_combo.setCurrentIndex(live.test_combo.findData("hip_adduction"))
        assert win.engine.test.view == "frontal" and not live.leg_combo.isHidden()
        win.close()


def test_the_engine_clears_the_tape_measure_and_the_wall_tick_for_another_participant(tmp_path):
    eng, p = engine(tmp_path, "hip_abduction", 10.0)
    eng.participant_code = "P01"
    eng.tape_pelvic_width_mm, eng.wall_posture = 240.0, True
    eng.participant_code = "P01"                                              # the same code again: nothing happens
    assert eng.tape_pelvic_width_mm == 240.0 and eng.pelvis_facts_cleared_for is None
    run(eng, p)
    eng.start_trial("P01")
    eng.participant_code = "P02"                                              # typed for the next participant during the recording
    assert eng.tape_pelvic_width_mm == 240.0 and eng.wall_posture             # the trial keeps what it started with ...
    eng.process(pic(p, 1), 1.0)
    eng.stop_trial()
    assert eng.tape_pelvic_width_mm is None and not eng.wall_posture and eng.pelvis_facts_cleared_for == "P01"   # ... cleared when it stops


def test_the_tape_measure_and_the_wall_tick_belong_to_one_participant():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as d:
        win = MainWindow(TESTS, d)
        try:
            live, eng = win.live, win.engine
            live.test_combo.setCurrentIndex(live.test_combo.findData("hip_abduction"))
            live.participant.setText("P01")
            live.tape_spin.setValue(240.0)
            live.chk_wall.setChecked(True)
            live.participant.setText("P01")                                   # the same code again: nothing happens
            assert eng.tape_pelvic_width_mm == 240.0 and eng.wall_posture
            live.participant.setText("P02")                                   # the next participant
            assert eng.tape_pelvic_width_mm is None and not eng.wall_posture
            assert live.tape_spin.value() == live.tape_spin.minimum() and not live.chk_wall.isChecked()
            live.participant.setText("")
            live.tape_spin.setValue(250.0)                                    # measured before the code is typed ...
            live.participant.setText("P03")                                   # ... stays for the code typed after it
            assert eng.tape_pelvic_width_mm == 250.0
        finally:
            win.close()
