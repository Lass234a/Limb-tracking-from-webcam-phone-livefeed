"""Reference readings in Review (0.16.0): the tab, adding / changing / removing readings (a side file, never meta.json), the agreement
dialog over a folder of trials, and the columns of the participant summary."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis, reference  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReferenceDialog, ReferenceValidationDialog, ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import frontal_points, leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=60, ankle_df=5)


def knee_trial(tmp_path, typed=None, stamp=True, participant="P01", target=60.0, knee=60.0):
    """A short recorded knee trial; `typed`: the readings typed before recording (key -> list), `stamp`: press 'reading taken now' first."""
    pts = leg_points(hip_flex=90, knee_flex=knee, ankle_df=5)
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = target
    eng.process(render(pts, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(pts[n] + 2))
    for i in range(8):
        eng.process(render(pts, seed=i), 0.1 + i / 30)
    if stamp:
        eng.take_reference_stamp()
    if typed:
        eng.set_reference(typed)
    paths = eng.start_trial(participant)
    for i in range(12):
        eng.process(render(pts, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    return paths["meta.json"]


@pytest.fixture
def quiet(monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(("info", a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warning", a[2])))
    return shown


def test_the_tab_is_there_between_force_and_calibration(tmp_path):
    app = QApplication.instance() or QApplication([])
    tab = ReviewTab(tmp_path)
    assert [tab.tabs.tabText(i) for i in range(tab.tabs.count())] == ["Check", "Angles", "Force", "Reference", "Calibration", "Dots and pull"]


def test_a_trial_without_readings_says_how_to_add_one(tmp_path):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, stamp=False)
    tab = ReviewTab(tmp_path)
    tab.load(str(meta))
    assert "No reading for this trial" in tab.ref_text.text() and "Live tab" in tab.ref_text.text()


def test_the_tab_shows_the_typed_reading_beside_the_app_at_the_same_moment(tmp_path):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, typed={"knee": [63.0, 64.0], "hip": [90.0]})
    tab = ReviewTab(tmp_path)
    tab.load(str(meta))
    text = tab.ref_text.text()
    assert "Knee flexion" in text and "63.5" in text and "recorded with the trial" in text and "Goniometer (read to 1 deg)" in text
    assert reference.APP_BASIS_STAMP in text and "Moment of the reading:" in text and "error of its own" in text
    rows = {r["key"]: r for r in Trial(meta).reference_rows()}
    assert rows["knee"]["difference_deg"] == pytest.approx(60.0 - 63.5, abs=0.7) and rows["hip"]["difference_deg"] == pytest.approx(0.0, abs=0.7)


def test_without_a_stamped_moment_the_text_says_the_first_second_stands_in(tmp_path):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, typed={"knee": [61.0]}, stamp=False)
    tab = ReviewTab(tmp_path)
    tab.load(str(meta))
    assert reference.APP_BASIS_FIRST_SECOND in tab.ref_text.text() and "No moment was kept" in tab.ref_text.text()


def test_readings_added_in_review_go_to_a_side_file_and_the_trial_files_are_untouched(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, stamp=False)
    csv_path = meta.parent / json.loads(meta.read_text())["files"]["data.csv"]
    before = (meta.read_bytes(), csv_path.read_bytes())
    tab = ReviewTab(tmp_path)
    tab.load(str(meta))
    dlg = ReferenceDialog(tab.trial, tab)
    assert [k for k in dlg.boxes] == ["knee", "hip", "ankle"]
    dlg.boxes["knee"].setText("58,5 59")
    dlg.instrument.setCurrentIndex(dlg.instrument.findData("inclinometer"))
    dlg.resolution.setCurrentIndex(dlg.resolution.findData(0.1))
    dlg._save()
    side = reference.sidecar_path(meta)
    assert side.exists() and json.loads(side.read_text())["readings"]["knee"]["values_deg"] == [58.5, 59.0]
    assert (meta.read_bytes(), csv_path.read_bytes()) == before
    tab.update_summary()
    assert "added later in Review" in tab.ref_text.text() and "58.8" in tab.ref_text.text() and "Digital inclinometer" in tab.ref_text.text()
    assert Trial(meta).reference_rows()[0]["reference_deg"] == pytest.approx(58.75)               # a new load sees it


def test_a_reading_that_cannot_be_read_is_refused_and_nothing_is_written(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, stamp=False)
    tab = ReviewTab(tmp_path)
    tab.load(str(meta))
    dlg = ReferenceDialog(tab.trial, tab)
    dlg.boxes["knee"].setText("sixty")
    dlg._save()
    assert quiet and quiet[-1][0] == "warning" and "not a number" in quiet[-1][1]
    assert not reference.sidecar_path(meta).exists()


def test_a_reading_saved_with_the_trial_can_be_changed_and_removed_in_review(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, typed={"knee": [63.0], "hip": [90.0]})
    tab = ReviewTab(tmp_path)
    tab.load(str(meta))
    dlg = ReferenceDialog(tab.trial, tab)
    assert dlg.boxes["knee"].text() == "63" and dlg.boxes["hip"].text() == "90"
    dlg.boxes["knee"].setText("61")                                         # changed
    dlg.boxes["hip"].setText("")                                            # removed
    dlg._save()
    rows = {r["key"]: r for r in Trial(meta).reference_rows()}
    assert rows["knee"]["reference_deg"] == 61.0 and rows["knee"]["source"] == "added later in Review" and "hip" not in rows
    assert json.loads(meta.read_text())["reference_readings"]["readings"]["hip"]["mean_deg"] == 90.0       # what was saved with the trial is still there


def two_trials_of_one_position(tmp_path, typed):
    """Two recorded knee trials of one position that share one typed reading and one 'reading taken now' (one pair, not two)."""
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(8):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    eng.participant_code = "P01"
    eng.take_reference_stamp()
    eng.set_reference(typed)
    metas = []
    for k in range(2):
        paths = eng.start_trial("P01")
        for i in range(12):
            eng.process(render(PTS, seed=i), 1.0 + k + i / 30)
        eng.stop_trial()
        metas.append(paths["meta.json"])
    return metas


def test_saving_in_review_without_a_change_leaves_the_recorded_readings_alone(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    first, _ = two_trials_of_one_position(tmp_path, {"knee": [61.0], "hip": [90.0]})
    assert reference.agreement(reference.collect_rows(tmp_path))["knee"]["n"] == 1
    tab = ReviewTab(tmp_path)
    tab.load(str(first))
    ReferenceDialog(tab.trial, tab)._save()                                 # opened and saved, nothing changed
    assert reference.read_sidecar(first)["readings"] == {}
    assert {r["source"] for r in Trial(first).reference_rows()} == {"recorded with the trial"}
    stats = reference.agreement(reference.collect_rows(tmp_path))
    assert stats["knee"]["n"] == 1 and stats["hip"]["n"] == 1             # still one reading of one position, not two


def test_adding_a_reading_in_review_puts_only_that_reading_in_the_side_file(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    first, _ = two_trials_of_one_position(tmp_path, {"knee": [61.0]})
    tab = ReviewTab(tmp_path)
    tab.load(str(first))
    dlg = ReferenceDialog(tab.trial, tab)
    dlg.boxes["ankle"].setText("4")
    dlg._save()
    assert list(reference.read_sidecar(first)["readings"]) == ["ankle"]
    rows = {r["key"]: r for r in Trial(first).reference_rows()}
    assert rows["knee"]["source"] == "recorded with the trial" and rows["ankle"]["source"] == "added later in Review"
    assert reference.agreement(reference.collect_rows(tmp_path))["knee"]["n"] == 1


def test_another_instrument_chosen_in_review_changes_every_reading(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    first, _ = two_trials_of_one_position(tmp_path, {"knee": [61.0]})
    tab = ReviewTab(tmp_path)
    tab.load(str(first))
    dlg = ReferenceDialog(tab.trial, tab)
    dlg.instrument.setCurrentIndex(dlg.instrument.findData("inclinometer"))
    dlg._save()
    row = Trial(first).reference_rows()[0]
    assert row["source"] == "added later in Review" and row["instrument"] == "inclinometer"


def test_a_frontal_trial_offers_the_typed_hip_flexion_and_knee(tmp_path):
    app = QApplication.instance() or QApplication([])
    import numpy as np
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("hip_abduction")
    p = dict(frontal_points(5.0))
    p["ruler_a"], p["ruler_b"] = np.array([120.0, 690.0]), np.array([270.0, 690.0])
    eng.process(render(p, dot_radius=11.0, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    for i in range(6):
        eng.process(render(p, dot_radius=11.0, seed=i), 0.1 + i / 30)
    paths = eng.start_trial("P01")
    for i in range(6):
        eng.process(render(p, dot_radius=11.0, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    tr = Trial(paths["meta.json"])
    assert [k for k, _ in tr.reference_edit_keys()] == ["abd_goniometer", "abd_hip_centre", "hip_flexion", "knee_flexion"]
    dlg = ReferenceDialog(tr)
    dlg.boxes["hip_flexion"].setText("1 2")
    dlg._save()
    row = next(r for r in Trial(paths["meta.json"]).reference_rows() if r["key"] == "hip_flexion")
    assert row["reference_deg"] == 1.5 and row["app_deg"] is None and row["title"].startswith("Hip flexion")


def test_the_agreement_dialog_lists_every_angle_plots_the_pairs_and_saves_the_table(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    for i, (target, knee, typed) in enumerate([(30.0, 30.0, 31.0), (60.0, 60.0, 59.0), (90.0, 90.0, 92.0)]):
        knee_trial(tmp_path, typed={"knee": [typed]}, participant="P01", target=target, knee=knee)
    tab = ReviewTab(tmp_path)
    rows = reference.collect_rows(tmp_path)
    assert len(rows) == 3
    dlg = ReferenceValidationDialog(tmp_path, rows, tab)
    assert dlg.combo.count() == 1 and "Knee flexion" in dlg.combo.itemText(0) and "n = 3" in dlg.combo.itemText(0)
    assert "Agreement of the app" in dlg.text.text() and "cannot show agreement to +-1 deg" in dlg.text.text()
    items = dlg.plot.plotItem.items
    assert any(getattr(i, "xData", None) is not None and len(i.xData) == 3 for i in items)            # the three pairs
    dlg.save_table()
    out = tmp_path / "reference_validation.csv"
    assert out.exists() and len(list(csv.DictReader(open(out, newline="")))) == 3 and quiet[-1][0] == "info"


def test_a_table_that_cannot_be_written_says_so(tmp_path, quiet, monkeypatch):
    app = QApplication.instance() or QApplication([])
    knee_trial(tmp_path, typed={"knee": [61.0]})
    tab = ReviewTab(tmp_path)
    dlg = ReferenceValidationDialog(tmp_path, reference.collect_rows(tmp_path), tab)

    def locked(rows, path):                                                 # Windows, with the file open in Excel
        raise PermissionError(13, "Permission denied", str(path))
    monkeypatch.setattr(reference, "write_csv", locked)
    dlg.save_table()
    assert quiet and quiet[-1][0] == "warning" and "open in Excel" in quiet[-1][1] and "Permission denied" in quiet[-1][1]


def test_readings_that_cannot_be_saved_say_so_and_the_dialog_stays_open(tmp_path, quiet, monkeypatch):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, stamp=False)
    tab = ReviewTab(tmp_path)
    tab.load(str(meta))
    dlg = ReferenceDialog(tab.trial, tab)
    dlg.boxes["knee"].setText("58")

    def locked(src, dst):
        raise PermissionError(13, "Permission denied", str(dst))
    monkeypatch.setattr(reference.os, "replace", locked)
    dlg._save()
    assert quiet and quiet[-1][0] == "warning" and "could not be saved" in quiet[-1][1]
    assert dlg.result() == 0 and dlg.boxes["knee"].text() == "58"                    # not closed, what was typed is still there
    assert not reference.sidecar_path(meta).exists() and not list(meta.parent.glob("*.tmp"))


def test_a_box_that_cannot_be_read_is_named_in_the_message(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, stamp=False)
    tab = ReviewTab(tmp_path)
    tab.load(str(meta))
    dlg = ReferenceDialog(tab.trial, tab)
    dlg.boxes["hip"].setText("ninety")
    dlg._save()
    assert quiet[-1][1].startswith(tab.trial.reference_edit_keys()[1][1] + ":")


def test_the_agreement_report_says_so_when_there_are_no_readings(tmp_path, quiet):
    app = QApplication.instance() or QApplication([])
    knee_trial(tmp_path, stamp=False)
    tab = ReviewTab(tmp_path)
    tab.show_reference_validation()
    assert quiet and quiet[-1][0] == "info" and "No trial" in quiet[-1][1]


def test_the_participant_summary_has_the_reference_of_the_main_joint(tmp_path):
    app = QApplication.instance() or QApplication([])
    meta = knee_trial(tmp_path, typed={"knee": [63.0], "hip": [90.0]})
    row = analysis.session_row(Trial(meta))
    assert row["reference_main_deg"] == 63.0 and row["reference_instrument"] == "goniometer"
    assert row["app_main_at_reading_deg"] == pytest.approx(60.0, abs=0.7) and row["app_minus_reference_main_deg"] == pytest.approx(-3.0, abs=0.7)
    assert row["reference_app_basis"] == reference.APP_BASIS_STAMP
    plain = analysis.session_row(Trial(knee_trial(tmp_path / "other", stamp=False)))
    assert plain["reference_main_deg"] == "" and plain["app_minus_reference_main_deg"] == ""
    out = analysis.write_session_csv([row, plain], tmp_path / "s.csv")
    back = list(csv.DictReader(open(out, newline="")))
    assert list(back[0]) == analysis.SESSION_COLUMNS and back[0]["reference_main_deg"] == "63.0" and back[1]["reference_main_deg"] == ""
