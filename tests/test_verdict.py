"""0.28.0: the operator's verdict on a trial, on site (Live tab) and later (Review): valid, or compromised with reasons (wrong dots, wrong
position, bad force, other) and a note. Kept next to the trial in `<trial>_verdict.json` with its history; a compromised trial is not
counted (session plan, 10 % check, the angle check of the position) and the export says why."""
import json
import math
import time

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import analysis, verdict  # noqa: E402
from limbtrack.review import ReviewTab, Trial, after_trial_check  # noqa: E402

from tests.test_analysis import torque_trial  # noqa: E402
from tests.test_pilot_checks import recording_at  # noqa: E402
from tests.test_rest_and_ten_percent import rows  # noqa: E402
from tests.test_rest_timer import real, record  # noqa: E402,F401  (the fixture: a real window recording a video file)


# ------------------------------------------------------------------------------------------------ the side file
def test_a_verdict_is_saved_next_to_the_trial_with_its_history(tmp_path):
    meta = tmp_path / "P01_02_knee_extension_70deg_T1_20261030-100000_meta.json"
    meta.write_text("{}")
    assert verdict.read_verdict(meta) is None and verdict.describe(None) == "not judged"
    v = verdict.write_verdict(meta, verdict.COMPROMISED, ["bad_force", "wrong_position"], " strap slipped ")
    assert verdict.verdict_path(meta).name == "P01_02_knee_extension_70deg_T1_20261030-100000_verdict.json"
    assert v["reasons"] == ["wrong_position", "bad_force"] and v["note"] == "strap slipped" and v["judged_in"] == verdict.ON_SITE
    assert verdict.is_compromised(meta) and verdict.describe(v) == "compromised: wrong position, bad force (strap slipped)"
    v2 = verdict.write_verdict(meta, verdict.VALID, where=verdict.IN_REVIEW)
    assert v2["verdict"] == "valid" and v2["judged_in"] == "Review" and not verdict.is_compromised(meta)
    assert [h["verdict"] for h in v2["history"]] == ["compromised"] and v2["history"][0]["note"] == "strap slipped"   # nothing is lost
    v3 = verdict.write_verdict(meta, None)
    assert verdict.read_verdict(meta)["verdict"] is None and len(v3["history"]) == 2
    assert meta.read_text() == "{}" and not list(tmp_path.glob("*.tmp"))       # the trial's own file untouched, no half-written file


@pytest.mark.parametrize("v, reasons, word", [("valid", ["bad_force"], "only a compromised"), ("compromised", [], "needs a reason"),
                                               ("maybe", [], "unknown verdict"), ("compromised", ["sneezed"], "unknown reason")])
def test_nonsense_is_refused(tmp_path, v, reasons, word):
    meta = tmp_path / "x_meta.json"
    with pytest.raises(ValueError, match=word):
        verdict.write_verdict(meta, v, reasons)
    assert not verdict.verdict_path(meta).exists()


# ------------------------------------------------------------------------------------------------ not counted
def test_the_10_percent_check_leaves_compromised_trials_out():
    r = rows(400.0, 250.0, 380.0)
    r[1]["verdict"] = "compromised"
    chk = analysis.position_check(r)
    assert chk["state"] == "ok" and "trial 1 400 N, trial 3 380 N" in chk["text"]
    assert "Not counted: trial 2 (marked compromised)" in chk["text"]
    only = analysis.position_check([dict(x, verdict="compromised") for x in rows(400.0, 380.0)])
    assert only["state"] == "none" and "trial 1 and 2 are marked compromised" in only["text"]
    g = [{"test_key": "k", "target_deg": 70, **x} for x in r]
    analysis.repeatability(g)
    assert "repeatability_pct_same_position" not in g[1] and g[0]["repeatability_pct_same_position"] == pytest.approx(5.13, abs=0.01)


def test_the_export_and_the_check_list_show_the_verdict(tmp_path):
    tr = torque_trial(tmp_path)
    row = analysis.session_row(tr)
    assert row["verdict"] == "" and {"verdict", "verdict_reasons", "verdict_note"} <= set(analysis.SESSION_COLUMNS)
    verdict.write_verdict(tr.meta_path, verdict.COMPROMISED, ["wrong_dots"], "malleolus dot on the sock")
    row = analysis.session_row(tr)
    assert (row["verdict"], row["verdict_reasons"], row["verdict_note"]) == ("compromised", "wrong dots", "malleolus dot on the sock")
    assert any("Marked compromised (on site (Live tab)): wrong dots (malleolus dot on the sock)" in t for lvl, t in analysis.trial_checks(tr)
               if lvl == "check")
    verdict.write_verdict(tr.meta_path, verdict.VALID)
    assert any(t.startswith("Marked valid") for lvl, t in analysis.trial_checks(tr) if lvl == "ok")


def test_a_compromised_trial_does_not_set_the_angle_of_its_position(tmp_path):
    root = tmp_path / "recordings"
    a = recording_at(root, 70.0)
    time.sleep(1.1)
    b = recording_at(root, 20.0)                     # '20' in the program, the leg did not move: reads like 70
    assert "previous position" in after_trial_check(b["meta.json"])["position_note"]
    verdict.write_verdict(a["meta.json"], verdict.COMPROMISED, ["wrong_position"])
    assert after_trial_check(b["meta.json"])["position_note"] == ""          # the 70 trial no longer counts as 'the previous position'


# ------------------------------------------------------------------------------------------------ in the Live tab
def test_after_a_trial_it_can_be_judged_on_site_and_the_plan_counts_only_the_good_ones(real):
    window, live, pump, tmp, clock, beeps = real
    bar = live.verdict_bar
    assert bar.isHidden()                                                  # nothing to judge before the first trial
    record(window, live, pump)
    meta = sorted((tmp / "recordings").rglob("*_meta.json"))[-1]
    assert not bar.isHidden() and "T1" in bar.label.text() and "not judged" in bar.state.text()
    assert "Next: trial 2 of 2" in live.trial_label.text()
    bar.reason_buttons["wrong_position"].click()                           # a click saves it at once
    v = verdict.read_verdict(meta)
    assert v["verdict"] == "compromised" and v["reasons"] == ["wrong_position"] and v["judged_in"] == verdict.ON_SITE
    assert "COMPROMISED" in bar.state.text() and "Next: trial 1 of 2" in live.trial_label.text() and "1 not counted" in live.trial_label.text()
    bar.reason_buttons["bad_force"].click()
    assert verdict.read_verdict(meta)["reasons"] == ["wrong_position", "bad_force"]
    bar.note.setText("participant still sitting")
    bar.note.editingFinished.emit()
    assert verdict.read_verdict(meta)["note"] == "participant still sitting"
    bar.btn_valid.click()                                                  # valid clears the reasons
    v = verdict.read_verdict(meta)
    assert v["verdict"] == "valid" and v["reasons"] == [] and not any(b.isChecked() for b in bar.reason_buttons.values())
    assert len(v["history"]) == 3 and "Next: trial 2 of 2" in live.trial_label.text()
    log = (tmp / "recordings" / "session_log.csv").read_text(encoding="utf-8")
    assert "trial judged" in log and "compromised: wrong position" in log and ": valid" in log
    live.btn_rec.click()                                                   # while recording the bar is away
    assert window.engine.rec is not None and bar.isHidden()
    assert pump(lambda: window.engine.rec.frames > 5)
    live.btn_rec.click()
    assert not bar.isHidden() and "T2" in bar.label.text() and "not judged" in bar.state.text()


# ------------------------------------------------------------------------------------------------ in Review
def test_review_shows_and_changes_the_verdict(tmp_path):
    QApplication.instance() or QApplication([])
    root = tmp_path / "recordings"
    a = recording_at(root, 70.0, seconds=2.0)
    verdict.write_verdict(a["meta.json"], verdict.COMPROMISED, ["wrong_dots"], "epicondyle jumped")
    tab = ReviewTab(root)
    tab.load(a["meta.json"])
    bar = tab.verdict_bar
    assert bar.reason_buttons["wrong_dots"].isChecked() and bar.note.text() == "epicondyle jumped" and "COMPROMISED" in bar.state.text()
    assert "Marked compromised" in tab.check_text.toPlainText()
    bar.btn_valid.click()
    v = verdict.read_verdict(a["meta.json"])
    assert v["verdict"] == "valid" and v["judged_in"] == verdict.IN_REVIEW and v["history"][0]["verdict"] == "compromised"
    assert "Marked valid" in tab.check_text.toPlainText()
    tab.trial.cap.release()
