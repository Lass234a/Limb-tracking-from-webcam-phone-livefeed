"""0.34.0 (F7): the session check, what is still missing before the participant leaves (limbtrack/completeness.py, the Live tab's Session check
button, the Review overview's button and line). Pure rules on made-up facts, then real recordings read back through the real windows, and the P02
case of the plan (a position that was never recorded) on the pilot trials when the folder is there (read only)."""
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import completeness as cm  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol, plan_positions  # noqa: E402
from limbtrack.review import ReviewTab, trial_row  # noqa: E402

from tests.test_anybody_export import session  # noqa: E402,F401  (the real recordings, made once for this file too)

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PLAN = plan_positions(TESTS)
TITLES = cm.titles_of(TESTS)
FULL_MEASURES = {"height_cm": 180.0, "mass_kg": 80.0, "fat_pct": 15.0, "thigh_length_cm": 44.0, "shank_length_cm": 41.0}
GOOD_SETUP = {"strap_landmark": "lateral_malleolus", "strap_from_landmark_cm": 4.0}


def fact(key, target, peak=400.0, verdict="valid", rise=True, reading=True, setup=GOOD_SETUP, leg="right", measures=FULL_MEASURES, standing=True,
         row=True, base=None):
    r = ({"trial": base or f"{key}{target}", "participant": "P01", "test_key": key, "target_deg": target, "force_unit": "N", "force_peak": peak,
          "force_peak_above_rest": peak - 20.0 if rise else 0.0, "verdict": verdict} if row else None)
    return {"base": base or f"{key}{target}", "participant": "P01", "test_key": key, "target_deg": target, "verdict": verdict, "row": r,
            "tested_leg": leg, "measures": measures, "setup": setup, "standing": {"angles": {"hip": {}}} if standing else None,
            "main_angle_name": "knee", "main_angle_title": "Knee flexion", "has_main_reading": reading}


def kinds(res, number=None):
    return sorted(i["kind"] for i in res["items"] if number is None or i["position"] == number)


def full_session():
    out = []
    for key, tgt in PLAN:
        out += [fact(key, tgt, 400.0, base=f"{key}{tgt}a"), fact(key, tgt, 410.0, base=f"{key}{tgt}b")]
    return out


# ------------------------------------------------------------------------------------------------ the pure rules
def test_a_complete_session_has_nothing_missing():
    res = cm.check(PLAN, full_session(), "P01", TITLES)
    assert res["complete"] and res["items"] == [] and res["counts"] == {"todo": 0, "gap": 0, "info": 0}
    assert cm.summary_line(res) == "P01: nothing missing."
    assert cm.lines(res)[1][0] == "ok"


def test_a_position_never_recorded_is_the_first_thing_on_the_list():
    facts = [f for f in full_session() if not (f["test_key"] == "hip_flexion" and f["target_deg"] == 0)]       # the P02 case of the plan
    res = cm.check(PLAN, facts, "P01", TITLES)
    assert kinds(res) == ["never_recorded"] and not res["complete"]
    item = res["items"][0]
    assert item["position"] == 10 and item["title"].startswith("10  Hip flexion, knee 90") and "0 deg" in item["title"]
    assert cm.lines(res)[1] == ("heading", "To do (record or redo)") and "Never recorded" in cm.lines(res)[2][1]
    assert cm.summary_line(res) == "P01: 1 to do."


def test_a_trial_recorded_at_a_target_outside_the_plan_is_mentioned_under_the_position_it_may_belong_to():
    facts = [f for f in full_session() if not (f["test_key"] == "ankle_plantarflexion" and f["target_deg"] == 0)]
    facts.append(fact("ankle_plantarflexion", None, base="old"))                               # the pilot's ankle trials of 28 Oct have no target
    res = cm.check(PLAN, facts, "P01", TITLES)
    text = next(i["text"] for i in res["items"] if i["kind"] == "never_recorded")
    assert "Never recorded." in text and "no target" in text
    assert len(res["positions"]) == len(PLAN) + 1                                              # (and the odd one is checked on its own too)


def test_compromised_only_and_no_force_rise_mean_nothing_counts():
    facts = [f for f in full_session() if f["test_key"] != "knee_flexion"]
    facts += [fact("knee_flexion", 20, verdict="compromised"), fact("knee_flexion", 20, verdict="compromised", base="b"),
              fact("knee_flexion", 90, rise=False), fact("knee_flexion", 90, rise=False, base="b")]
    res = cm.check(PLAN, facts, "P01", TITLES)
    assert kinds(res, 4) == ["compromised_only"] and kinds(res, 5) == ["no_force"]
    assert "Record it again" in next(i["text"] for i in res["items"] if i["position"] == 4)


def test_a_third_trial_is_called_for_when_two_counted_peaks_differ_by_more_than_ten_percent():
    facts = [f for f in full_session() if f["test_key"] != "knee_extension" or f["target_deg"] != 20]
    facts += [fact("knee_extension", 20, 300.0), fact("knee_extension", 20, 400.0, base="b")]
    res = cm.check(PLAN, facts, "P01", TITLES)
    assert kinds(res, 1) == ["third_trial"] and "SOP 10.6" in res["items"][0]["text"]
    facts[-1] = fact("knee_extension", 20, 310.0, base="b")                                  # 3 % apart: fine
    assert cm.check(PLAN, facts, "P01", TITLES)["complete"]


def test_a_compromised_trial_does_not_count_towards_the_pair_and_one_counted_trial_is_not_enough():
    facts = [f for f in full_session() if f["test_key"] != "knee_extension" or f["target_deg"] != 20]
    facts += [fact("knee_extension", 20, 300.0), fact("knee_extension", 20, 900.0, verdict="compromised", base="b")]
    res = cm.check(PLAN, facts, "P01", TITLES)
    assert kinds(res, 1) == ["one_trial"]


def test_trials_not_judged_are_a_gap_for_the_export_not_a_job_to_redo():
    facts = [f for f in full_session() if f["test_key"] != "hip_flexion" or f["target_deg"] != 90]
    facts += [fact("hip_flexion", 90, verdict=None), fact("hip_flexion", 90, verdict=None, base="b")]
    res = cm.check(PLAN, facts, "P01", TITLES)
    assert kinds(res, 11) == ["not_judged"] and res["items"][0]["level"] == "gap" and "AnyBody export" in res["items"][0]["text"]
    assert res["counts"] == {"todo": 0, "gap": 1, "info": 0}


def test_missing_reference_readings_and_strap_places_are_gaps_per_position_and_the_belt_squat_needs_no_strap_place():
    facts = []
    for f in full_session():
        if f["test_key"] == "ankle_dorsiflexion" and f["target_deg"] == 0:
            f = {**f, "has_main_reading": False}
        if f["test_key"] == "knee_flexion" and f["target_deg"] == 20:
            f = {**f, "setup": None}
        if f["test_key"] == "belt_squat":
            f = {**f, "setup": None}
        facts.append(f)
    res = cm.check(PLAN, facts, "P01", TITLES)
    assert kinds(res, 18) == ["reference_missing"] and kinds(res, 4) == ["setup_missing"]
    assert kinds(res, 20) == [] and kinds(res, 21) == []
    assert "Knee flexion" in next(i["text"] for i in res["items"] if i["kind"] == "reference_missing")


def test_a_strap_place_without_a_distance_is_not_a_strap_place():
    facts = [{**f, "setup": {"strap_landmark": "trochanter", "strap_from_landmark_cm": None}} if f["test_key"] == "knee_flexion" else f
             for f in full_session()]
    assert kinds(cm.check(PLAN, facts, "P01", TITLES), 4) == ["setup_missing"]


def test_the_participant_items_measures_leg_and_standing_reference():
    facts = [{**f, "measures": {"height_cm": 180.0, "mass_kg": None}, "standing": None} for f in full_session()]
    res = cm.check(PLAN, facts, "P01", TITLES)
    assert kinds(res) == ["measures_missing", "standing_missing"]
    text = next(i["text"] for i in res["participant_items"] if i["kind"] == "measures_missing")
    assert "mass" in text and "body fat" in text and "thigh length" in text and "shank length" in text and "height" not in text
    nolegs = [{**f, "tested_leg": ""} for f in full_session()]
    assert kinds(cm.check(PLAN, nolegs, "P01", TITLES)) == ["leg_missing"]
    mixed = full_session()
    mixed[3] = {**mixed[3], "tested_leg": "left"}
    r = cm.check(PLAN, mixed, "P01", TITLES)
    assert kinds(r) == ["leg_mixed"] and "left" in r["items"][0]["text"] and "right" in r["items"][0]["text"]


def test_a_measure_typed_in_a_later_trial_counts_for_the_participant():
    facts = full_session()
    facts[0] = {**facts[0], "measures": None}                                                 # the first trial was recorded before the typing
    assert cm.check(PLAN, facts, "P01", TITLES)["complete"]


def test_positions_outside_the_plan_are_checked_too_and_other_participants_are_left_out():
    facts = full_session() + [fact("knee_extension", 45.0)]
    res = cm.check(PLAN, facts, "P01", TITLES)
    assert len(res["positions"]) == len(PLAN) + 1 and kinds(res, len(PLAN) + 1) == ["one_trial"]
    other = full_session()
    other[0] = {**other[0], "participant": "P09"}
    assert len(cm.check(PLAN, other, "P01", TITLES)["positions"]) == len(PLAN)
    nothing = cm.check(PLAN, [], "P03", TITLES)
    assert nothing["trials"] == 0 and cm.summary_line(nothing) == "P03: no trials yet." and len(cm.lines(nothing)) == 1


def test_a_trial_without_numbers_counts_unless_compromised_and_unreadable_ones_are_named(tmp_path):
    facts = [f for f in full_session() if f["test_key"] != "belt_squat"]
    facts += [fact("belt_squat", 90, row=False), fact("belt_squat", 60, row=False)]
    assert kinds(cm.check(PLAN, facts, "P01", TITLES), 20) == ["one_trial"]
    bad = tmp_path / "P01_x_20260101-000000_meta.json"
    bad.write_text("{not json", encoding="utf-8")
    got, unreadable = cm.gather([(bad, None)])
    assert got == [] and unreadable == ["P01_x_20260101-000000"]
    res = cm.check(PLAN, [], "P01", TITLES, unreadable=unreadable)
    assert res["participant_items"][0]["kind"] == "unreadable"


# ------------------------------------------------------------------------------------------------ real recordings, read back
def test_the_check_on_real_trials_reads_meta_verdicts_and_numbers(session):
    res = cm.check_folder(session["folder"], PLAN, TITLES, row_fn=lambda p: trial_row(p)[0])
    assert res["participant"] == "P01" and res["tested_leg"] == "right" and res["trials"] == 4
    assert kinds(res, 1) == ["not_judged", "one_trial", "reference_missing"]                 # knee extension 20: one trial, never judged, no reading
    assert [k for k in kinds(res, 2) if k != "third_trial"] == ["reference_missing"]         # knee extension 70: two valid, the third is compromised
    assert sum(1 for i in res["items"] if i["kind"] == "never_recorded") == len(PLAN) - 2
    assert not any(i["kind"] in ("measures_missing", "leg_missing", "setup_missing") for i in res["items"])        # typed in the window: found in meta.json
    assert any(i["kind"] == "standing_missing" for i in res["participant_items"])
    assert res["positions"][1]["trials"] == 3 and res["positions"][1]["counted"] == 2 and res["positions"][1]["valid"] == 2


def _app():
    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_the_live_tab_button_opens_the_list_beside_the_program_and_blocks_nothing(session, tmp_path):
    app = _app()
    shutil.copytree(session["root"], tmp_path / "recordings")
    win = MainWindow(TESTS, tmp_path)
    try:
        live = win.live
        assert live.btn_session_check.parent() is not None and live.btn_session_check.toolTip()
        live.participant.setText("")
        live.open_session_check()                                                            # no code yet: it says so and opens nothing
        assert live._session_dialog is None
        live.participant.setText("P01")
        live.open_session_check()
        dlg = live._session_dialog
        assert dlg is not None and not dlg.isModal()
        assert _wait(app, lambda: dlg.result is not None or dlg.error)
        assert dlg.error == "" and dlg.result["participant"] == "P01"
        html = dlg.view.toPlainText()
        assert "Never recorded" in html and "TO DO" in html and "MISSING" in html and "Hip flexion" in html
        assert dlg.status.text() == cm.summary_line(dlg.result)
        dlg.close()
    finally:
        win.close()


def test_the_review_overview_has_the_button_and_a_line_per_participant(session, tmp_path):
    app = _app()
    root = tmp_path / "recordings"
    shutil.copytree(session["root"], root)
    tab = ReviewTab(root, tests=TESTS)
    try:
        assert not tab.overview.session_button.isEnabled()
        tab.set_participant(root / "P01")
        tab.show_view("overview")
        assert tab.overview.session_button.isEnabled()
        assert tab.overview.wait_until_done(60000)
        assert _wait(app, lambda: tab.overview.session_label.text().startswith("Session check:"))
        assert "P01:" in tab.overview.session_label.text() and "to do" in tab.overview.session_label.text()
        tab.overview.session_button.click()
        dlg = tab._session_dialog
        assert dlg is not None and dlg.result is not None and dlg.result["participant"] == "P01"
        assert any(i["kind"] == "never_recorded" for i in dlg.result["items"])
        dlg.close()
    finally:
        if tab.trial is not None:
            tab.trial.close()
        tab.overview.shutdown()


def test_a_session_check_that_cannot_be_made_says_so_in_the_window_and_not_an_empty_list(session):
    """The error path of the window: compute() raises, on the calling thread and on the helper thread, and for a real folder whose row source fails."""
    from limbtrack.session_check_dialog import SessionCheckDialog
    app = _app()

    def broken():
        raise ValueError("damaged trial")

    dlg = SessionCheckDialog(None, "t", broken, threaded=False)
    assert dlg.result is None and dlg.error == "ValueError: damaged trial"
    assert "could not be made" in dlg.view.toPlainText() and "damaged trial" in dlg.view.toPlainText()
    assert dlg.status.text() == "Could not be made."
    dlg.close()
    dlg = SessionCheckDialog(None, "t", broken, threaded=True)
    assert _wait(app, lambda: dlg.error)
    assert dlg.result is None and dlg.error == "ValueError: damaged trial" and dlg.status.text() == "Could not be made." and dlg.btn_again.isEnabled()
    dlg.close()

    def bad_rows(meta_path):
        raise RuntimeError(f"cannot read {Path(meta_path).name}")
    dlg = SessionCheckDialog(None, "t", lambda: cm.check_folder(session["folder"], PLAN, TITLES, row_fn=bad_rows), threaded=False)
    assert dlg.result is None and dlg.error.startswith("RuntimeError: cannot read ") and "could not be made" in dlg.view.toPlainText()
    dlg.close()


# ------------------------------------------------------------------------------------------------ the pilot (read only)
PILOT = ROOT / "Pilot"


@pytest.mark.skipif(not (PILOT / "P02").is_dir(), reason="the Pilot/ folder is not here")
def test_the_p02_case_hip_flexion_0_was_never_recorded_and_the_check_says_so():
    def snap():
        return sorted((str(p.relative_to(PILOT)), p.stat().st_size, p.stat().st_mtime_ns) for p in PILOT.rglob("*") if p.is_file())
    before = snap()
    res = cm.check_folder(PILOT / "P02", PLAN, TITLES, row_fn=lambda p: trial_row(p)[0])
    never = [i for i in res["items"] if i["kind"] == "never_recorded"]
    assert [i["position"] for i in never] == [10] and "Hip flexion" in never[0]["title"] and "0 deg" in never[0]["title"]
    assert res["tested_leg"] and not res["complete"]
    assert snap() == before
