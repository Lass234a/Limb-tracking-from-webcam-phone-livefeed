"""The Review tab of 0.29.0 on real recordings: a participant and then a trial from a drop-down, previous / next / next not judged, the
verdict shown in the list, the participant overview (worked out in the background), the exports, and what is remembered."""
import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import review_nav as nav  # noqa: E402
from limbtrack import review_overview  # noqa: E402
from limbtrack import review_layer as rl  # noqa: E402
from limbtrack import verdict  # noqa: E402
from limbtrack.review import ReviewTab  # noqa: E402

from tests.test_pilot_checks import recording_at  # noqa: E402


@pytest.fixture(scope="module")
def trio(tmp_path_factory):
    """recordings/P01 with three real trials of one position (a simulated force and a recorded video), and a copy as recordings/P02."""
    root = tmp_path_factory.mktemp("recordings")
    for _ in range(3):
        recording_at(root, 70.0, seconds=1.5)
    shutil.copytree(root / "P01", root / "P02")
    return root


@pytest.fixture()
def root(trio, tmp_path):
    """A fresh copy for each test (verdicts and exports are written into it)."""
    dest = tmp_path / "recordings"
    shutil.copytree(trio, dest)
    return dest


def _app():
    return QApplication.instance() or QApplication([])


def _tab(root):
    _app()
    tab = ReviewTab(root)
    return tab


def _finish(tab):
    if tab.trial is not None:
        tab.trial.close()
    tab.overview.shutdown()


def test_the_participant_dropdown_lists_the_folders_with_trials_and_the_trial_dropdown_follows(root):
    tab = _tab(root)
    names = [tab.participant_combo.itemText(i) for i in range(tab.participant_combo.count())]
    assert names[0].startswith("Choose") and any(n.startswith("P01") and "3 trials" in n for n in names) and any(n.startswith("P02") for n in names)
    assert tab.trial is None and tab.btn_prev.isEnabled() is False
    tab.set_participant(root / "P01")
    assert tab.trial_combo.count() == 3 and tab.trial is not None                # the first trial is open, not a blank page
    assert tab._index == 0 and tab.trial.base == tab._entries[0].base and tab.trial_combo.currentIndex() == 0
    assert tab.participant_combo.currentText().startswith("P01")
    assert "not judged" in tab.trial_combo.itemText(0)
    _finish(tab)


def test_previous_next_and_the_dropdown_open_trials_and_let_go_of_the_old_video(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    first = tab.trial
    assert not tab.btn_prev.isEnabled() and tab.btn_next.isEnabled()
    tab.next_trial()
    assert tab._index == 1 and tab.trial.base == tab._entries[1].base and tab.trial is not first
    assert not first.cap.isOpened()                                            # the video of the trial left behind was released
    tab._trial_chosen(2)
    assert tab._index == 2 and tab.trial.base == tab._entries[2].base and not tab.btn_next.isEnabled() and tab.btn_prev.isEnabled()
    tab.prev_trial()
    assert tab._index == 1
    _finish(tab)


def test_judging_shows_in_the_dropdown_and_next_not_judged_walks_through_the_rest(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    assert tab._index == 0 and tab.btn_next_unjudged.isEnabled()
    tab.verdict_bar.btn_valid.click()
    assert "✓ valid" in tab.trial_combo.itemText(0)
    tab.next_unjudged()
    assert tab._index == 1
    tab.verdict_bar.reason_buttons["bad_force"].click()
    assert "✗ compromised" in tab.trial_combo.itemText(1)
    tab.next_unjudged()
    assert tab._index == 2
    tab.verdict_bar.btn_valid.click()
    assert not tab.btn_next_unjudged.isEnabled() and "Every readable trial" in tab.btn_next_unjudged.toolTip()
    on_disk = {nav.read_entry(e.meta_path).verdict for e in tab._entries}
    assert on_disk == {"valid", "compromised"} and verdict.read_verdict(tab._entries[1].meta_path)["judged_in"] == verdict.IN_REVIEW
    _finish(tab)


def test_opening_the_first_trial_without_a_verdict_when_a_participant_is_chosen(root):
    for e in nav.list_trials(root / "P01")[:2]:
        verdict.write_verdict(e.meta_path, verdict.VALID)
    tab = _tab(root)
    tab.set_participant(root / "P01")
    assert tab._index == 2                                                     # the first two are judged: it opens where the work stops
    _finish(tab)


def test_open_folder_takes_a_participant_or_a_folder_of_participants_and_says_when_there_is_nothing(root, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    outside = tmp_path / "Pilot"
    shutil.copytree(root / "P01", outside / "P07")
    tab = _tab(root)
    tab.open_folder(outside / "P07")                                           # a participant outside the recordings folder
    assert tab._folder == outside / "P07" and any(tab.participant_combo.itemText(i).startswith("P07") for i in range(tab.participant_combo.count()))
    tab.open_folder(root)                                                      # a folder of participants: all are added, the first opens
    assert tab._folder in (root / "P01", root) or tab._folder.name == "P01"
    tab.open_folder(tmp_path / "nothing_here")
    assert shown and "No trial" in shown[-1]
    _finish(tab)


def test_a_long_text_in_the_two_tables_never_widens_the_window(tmp_path):
    """The first cards asked for 1146 px (1904 with a long line) and made the window wider than a laptop screen: the texts now give way."""
    tab = _tab(tmp_path)
    before = tab.minimumSizeHint().width()
    for tree in (tab.angle_tree, tab.force_tree):
        tree.set_none("a very long line of explanation " * 30)
    assert tab.minimumSizeHint().width() == before


def test_the_picture_is_drawn_again_when_the_video_area_changes_size(root):
    import time
    app = _app()
    tab = _tab(root)
    tab.resize(2400, 900)                                                  # (wide enough for the video area to grow whatever the fonts of the machine)
    tab.show()
    tab.set_participant(root / "P01")

    def settle():
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.02)
    from PySide6.QtCore import QSize

    def fits():
        """The picture on the label is the picture scaled to the label (as large as fits, whole): the redraw after a resize happened."""
        w, h = tab.video.picture_size
        want, got = QSize(w, h).scaled(tab.video.size(), Qt.KeepAspectRatio), tab.video.pixmap().size()
        return abs(got.width() - want.width()) <= 1 and abs(got.height() - want.height()) <= 1
    tab.top_splitter.setSizes([420, 980])
    settle()
    narrow = tab.video.width()
    assert fits()
    tab.top_splitter.setSizes([900, 500])                       # the video area dragged wide: the picture is drawn again for it, not left at the old size
    settle()
    assert tab.video.width() > narrow + 100 and fits()
    tab.top_splitter.setSizes([420, 980])                       # and back: it does not stay cut off at the large size
    settle()
    assert tab.video.width() < tab.top_splitter.width() * 0.5 and fits()
    tab.close()
    _finish(tab)


def test_changing_the_recordings_folder_refreshes_the_participants(root, tmp_path):
    tab = _tab(root)
    other = tmp_path / "other"
    shutil.copytree(root / "P01", other / "P55")
    tab.dir = other
    names = [tab.participant_combo.itemText(i) for i in range(tab.participant_combo.count())]
    assert any(n.startswith("P55") for n in names) and not any(n.startswith("P02") for n in names)
    _finish(tab)


def test_the_participant_and_trial_are_remembered_and_put_back_when_the_tab_is_first_shown(root, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"recordings_dir": str(root)}))
    a = _tab(root)
    a.attach_memory(settings)
    a.set_participant(root / "P02")
    a.next_trial()
    base = a._entries[a._index].base
    saved = json.loads(settings.read_text())
    assert saved["recordings_dir"] == str(root) and saved["review_nav"]["trial"] == base                    # the rest of the file is kept
    _finish(a)
    b = _tab(root)
    b.attach_memory(settings)
    assert b.trial is None                                                      # nothing is opened at start-up
    b.show()
    _app().processEvents()
    assert b.trial is not None and b.trial.base == base and b._folder.name == "P02"
    b.close()
    _finish(b)


def test_a_missing_remembered_folder_is_ignored(root, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"review_nav": {"participant_dir": str(tmp_path / "gone"), "trial": "x", "extra_folders": [str(tmp_path / "gone2")]}}))
    tab = _tab(root)
    tab.attach_memory(settings)
    tab.show()
    _app().processEvents()
    assert tab.trial is None and tab._pending_restore is None
    tab.close()
    _finish(tab)


def test_the_overview_works_out_every_trial_in_the_background_and_shows_verdicts(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    verdict.write_verdict(tab._entries[0].meta_path, verdict.COMPROMISED, ["bad_force"], "slipped")
    tab.overview.set_participant("P01", nav.list_trials(root / "P01"))
    tab.show_view("overview")
    assert tab.stack.currentIndex() == 1 and tab.btn_view_overview.isChecked()
    assert tab.overview.wait_until_done(120000)
    ov = tab.overview
    assert len(ov._cache) == 3 and all(v[1] is not None for v in ov._cache.values())
    ov.render()
    assert ov.table.rowCount() == 3
    first = [ov.table.item(r, 0).text() for r in range(3)]
    assert sorted(first) == ["○ not judged", "○ not judged", "✗ compromised"]
    assert "3 trials" in ov.summary.text() and "1 compromised" in ov.summary.text() and "2 not judged" in ov.summary.text()
    row = ov.row_for(tab._entries[1].base)
    assert row["force_window_peak"] != "" and row["force_time_to_peak_s"] != ""
    ov.filter.setCurrentIndex(ov.filter.findData("compromised"))
    assert ov.table.rowCount() == 1
    ov.filter.setCurrentIndex(ov.filter.findData("unjudged"))
    assert ov.table.rowCount() == 2
    ov.filter.setCurrentIndex(0)
    _finish(tab)


def test_a_double_click_in_the_overview_opens_the_trial_on_the_trial_page(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    tab.show_view("overview")
    tab.overview.wait_until_done(120000)
    tab.overview.render()
    base = tab.overview.table.item(2, 0).data(Qt.UserRole + 1)
    tab.overview._open_row(2)
    assert tab.stack.currentIndex() == 0 and tab.trial.base == base and tab._entries[tab._index].base == base
    _finish(tab)


def test_a_verdict_given_in_review_updates_the_overview_line_and_the_old_numbers_stay_until_the_new_are_in(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    tab.show_view("overview")
    tab.overview.wait_until_done(120000)
    tab.show_view("trial")
    tab.verdict_bar.btn_valid.click()
    ov = tab.overview
    base = tab._entries[0].base
    assert ov.row_for(base) is None and ov.row_for(base, allow_stale=True)["verdict"] == "valid"      # patched at once, marked out of date
    tab.show_view("overview")                                                  # showing the overview works the out-of-date line again
    assert ov.wait_until_done(120000) and ov.row_for(base) is not None
    assert "Marked valid" in (ov.row_for(base).get("checks_to_look_at") or "Marked valid") or ov.row_for(base)["verdict"] == "valid"
    _finish(tab)


def test_the_sliding_window_setting_makes_the_overview_numbers_out_of_date(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    tab.show_view("overview")
    tab.overview.wait_until_done(120000)
    assert not tab.overview.stale_entries()
    tab.show_view("trial")
    tab.window_peak_spin.setValue(1000)
    assert tab.trial.window_peak_s == 1.0
    tab.show_view("overview")
    assert len(tab.overview.stale_entries()) == 3 or tab.overview._running
    assert tab.overview.wait_until_done(120000) and not tab.overview.stale_entries()
    assert float(tab.overview.row_for(tab._entries[0].base)["force_window_peak_ms"]) == pytest.approx(1000.0, abs=0.01)
    _finish(tab)


def test_export_all_writes_every_trial_with_its_verdict_and_says_how_many_are_not_judged(root, monkeypatch):
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a[2]))
    tab = _tab(root)
    tab.set_participant(root / "P01")
    verdict.write_verdict(tab._entries[0].meta_path, verdict.VALID)
    verdict.write_verdict(tab._entries[1].meta_path, verdict.COMPROMISED, ["wrong_dots"])
    out = tab.export_session()
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert out.name == "P01_session_summary_original.csv" and len(rows) == 3                                                  # (no fixes: the original data)
    assert out.parent == root / "review" / "P01" / "exports" and not list((root / "P01").glob("*session_summary*"))           # the review folder, not beside the trials
    assert [r["verdict"] for r in rows] == ["valid", "compromised", ""] and rows[1]["verdict_reasons"] == "wrong dots"
    for c in ("force_window_peak", "force_window_peak_ms", "force_time_to_peak_s", "force_time_to_window_peak_s"):
        assert c in rows[0] and rows[0][c] != ""
    assert float(rows[0]["force_window_peak_ms"]) == pytest.approx(500.0, abs=0.01) and float(rows[0]["force_window_peak"]) <= float(rows[0]["force_peak"]) + 1e-6
    assert infos[-1].startswith("Saved P01_session_summary_original.csv in the review folder (exports): 3 trials (1 valid, 1 compromised, 1 not judged yet)")
    assert "no verdict yet" in infos[-1]
    _finish(tab)


def test_the_overview_and_the_exports_follow_the_data_switch_and_mark_edited_trials(root, monkeypatch):
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a[2]))
    tab = _tab(root)
    tab.set_participant(root / "P01")
    t = tab.trial.t
    assert tab.commit_dot_edit("epicondyle", "hold", from_s=float(t[2]), to_s=float(t[10]), x=400.0, y=300.0)
    first, others = tab._entries[0], tab._entries[1:]
    assert first.edited and not any(e.edited for e in others) and "✎ edited" in tab.trial_combo.itemText(0)
    tab.show_view("overview")
    assert tab.overview.wait_until_done(120000)
    ov = tab.overview
    ov.render()
    assert ov.row_for(first.base)["data_source"] == "reviewed" and all(ov.row_for(e.base)["data_source"] == "original" for e in others)
    edits_col = review_overview.COLUMNS.index("Edits")
    assert [ov.table.item(r, edits_col).text() for r in range(3)] == ["✎", "", ""]
    ov.filter.setCurrentIndex(ov.filter.findData("edited"))
    assert ov.table.rowCount() == 1
    ov.filter.setCurrentIndex(0)
    out = tab.export_session()
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert out.name == "P01_session_summary_reviewed.csv" and [r["data_source"] for r in rows] == ["reviewed", "original", "original"]
    assert "dot epicondyle: held at (400.0, 300.0)" in rows[0]["review_edits"] and int(rows[0]["dots_corrected_pictures"]) == 9 and rows[1]["review_edits"] == ""
    tab.set_source("original")                                            # the same trials as recorded
    assert tab.trial.source == "original" and tab.overview.source == "original"
    out_o = tab.export_session()
    rows_o = list(csv.DictReader(open(out_o, newline="", encoding="utf-8")))
    assert out_o.name == "P01_session_summary_original.csv" and {r["data_source"] for r in rows_o} == {"original"} and not any(r["review_edits"] for r in rows_o)
    assert out.is_file() and out_o.is_file()                              # both versions side by side
    tab.set_source("reviewed")
    tab.export()
    d = rl.review_dir(tab.trial.meta_path)
    assert (d / "reviewed_data.csv").is_file() and (d / "reviewed_force.csv").is_file() and (d / f"{tab.trial.base}_summary_reviewed.csv").is_file()
    rd = list(csv.DictReader(open(d / "reviewed_data.csv", newline="", encoding="utf-8")))
    assert [r["dots_edited"] for r in rd[:12]].count("epicondyle") == 9 and rd[0]["dots_edited"] == ""
    assert float(rd[5]["epicondyle_x"]) == 400.0 and rd[5]["epicondyle_q"] == ""
    _finish(tab)


def test_export_valid_only_has_just_the_valid_trials_and_a_separate_name(root, monkeypatch):
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: infos.append(a[2]))
    tab = _tab(root)
    tab.set_participant(root / "P01")
    tab.export_session(valid_only=True)
    assert "No trial of this participant is marked valid" in infos[-1] and not list(root.rglob("*_valid.csv"))
    verdict.write_verdict(tab._entries[2].meta_path, verdict.VALID)
    out = tab.export_session(valid_only=True)
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert out.name == "P01_session_summary_original_valid.csv" and len(rows) == 1 and rows[0]["verdict"] == "valid" and rows[0]["trial"] == tab._entries[2].base
    assert not list(root.rglob("P01_session_summary_original.csv"))                                    # (only the valid file was asked for)
    _finish(tab)


def test_a_trial_that_cannot_be_read_is_named_in_the_overview_and_the_export_not_dropped(root, monkeypatch):
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a[2]))
    bad = nav.list_trials(root / "P01")[1]
    (bad.meta_path.parent / json.loads(bad.meta_path.read_text())["files"]["data.csv"]).unlink()            # its numbers file is gone
    tab = _tab(root)
    tab.set_participant(root / "P01")
    tab.show_view("overview")
    tab.overview.wait_until_done(120000)
    tab.overview.render()
    err = tab.overview._cache[bad.base][2]
    assert err and tab.overview._cache[bad.base][1] is None
    peak_col = review_overview.COLUMNS.index("Peak")
    texts = [tab.overview.table.item(r, peak_col).text() for r in range(tab.overview.table.rowCount())]
    assert "could not be read" in texts
    out = tab.export_session()
    assert len(list(csv.DictReader(open(out, newline="", encoding="utf-8")))) == 2 and bad.base in infos[-1] and "Not readable" in infos[-1]
    _finish(tab)
