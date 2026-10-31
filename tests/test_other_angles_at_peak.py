"""The other joints of the test at the peak force (0.33.0): the angle at the peak was only the main joint's. Now every angle of the trial has
one (the hip in a knee test, the knee in a hip test), in the summary, in the angle table, in the force table's note, in the export row and
in the participant overview."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import analysis, review_overview  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402

from tests.test_pilot_checks import recording_at  # noqa: E402


def _app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def one(tmp_path_factory):
    root = tmp_path_factory.mktemp("recordings")
    recording_at(root, 70.0, seconds=1.5)
    return root


def _meta(root):
    return next((root / "P01").glob("*_meta.json"))


def _window(tr):
    return tr.force_window() or (float(tr.t[0]), float(tr.t[-1]))


def test_every_angle_of_the_trial_has_an_angle_at_the_peak_and_the_main_one_is_as_before(one):
    tr = Trial(_meta(one), open_video=False)
    try:
        fs = tr.force_summary(*_window(tr), 0.0)
        at = fs["angles_at_peak_deg"]
        assert list(at) == [a.title for a in tr.angle_defs] and len(at) >= 2
        assert at[fs["main_angle"]] == fs["main_angle_at_peak_deg"]
        near = np.abs(tr.t - fs["t_peak_s"]) <= 0.1                              # the same rule for every angle: the pictures within 0.1 s of the peak
        for a in tr.angle_defs:
            vals = tr.deg[a.name][near]
            assert at[a.title] == pytest.approx(float(np.nanmean(vals[~np.isnan(vals)])))
    finally:
        tr.close()


def test_the_picture_delay_moves_every_angle_to_the_same_pictures(one):
    tr = Trial(_meta(one), open_video=False)
    try:
        win = _window(tr)
        a, b = tr.force_summary(*win, 0.0), tr.force_summary(*win, 0.4)
        t_pic = b["t_peak_s"] + 0.4
        for ad in tr.angle_defs:
            assert b["angles_at_peak_deg"][ad.title] == pytest.approx(tr.angle_near(ad.name, t_pic))
        assert a["angles_at_peak_deg"].keys() == b["angles_at_peak_deg"].keys()
    finally:
        tr.close()


def test_the_export_row_has_the_other_angles_in_columns_after_the_main_one(one):
    tr = Trial(_meta(one), open_video=False)
    try:
        row = analysis.session_row(tr)
        fs = tr.force_summary(*_window(tr), 0.0)
        others = [t for t in fs["angles_at_peak_deg"] if t != fs["main_angle"]]
        assert row["angle2_name"] == others[0] and row["angle2_at_peak_force_deg"] == pytest.approx(fs["angles_at_peak_deg"][others[0]])
        assert row["angle3_name"] == "" == row["angle3_at_peak_force_deg"] if len(others) == 1 else row["angle3_name"] == others[1]
        cols = analysis.SESSION_COLUMNS
        assert cols[cols.index("angle_at_peak_force_deg") + 1: cols.index("angle_at_peak_force_deg") + 8] == [
            "angle_at_peak_force_from_reference_deg", "angle2_name", "angle2_at_peak_force_deg", "angle2_at_peak_from_reference_deg",
            "angle3_name", "angle3_at_peak_force_deg", "angle3_at_peak_from_reference_deg"]
        assert row["angle2_at_peak_from_reference_deg"] == "" and row["angle_at_peak_force_from_reference_deg"] == ""         # no reference: blank
        assert row["angle_at_peak_force_deg"] == fs["main_angle_at_peak_deg"]                  # the old column is as it was
    finally:
        tr.close()


def test_a_trial_without_force_has_blank_other_angle_columns(one):
    tr = Trial(_meta(one), open_video=False)
    try:
        assert analysis.other_angles_at_peak({"main_angle": "A", "angles_at_peak_deg": {"A": 1.0}}) == {}
        assert analysis.other_angles_at_peak({"main_angle": "A"}) == {}
        got = analysis.other_angles_at_peak({"main_angle": "A", "angles_at_peak_deg": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}})
        assert got == {"angle2_name": "B", "angle2_at_peak_force_deg": 2.0, "angle3_name": "C", "angle3_at_peak_force_deg": 3.0}      # (at most two others)
    finally:
        tr.close()


def test_the_angle_table_gives_every_angle_an_at_peak_and_the_force_table_names_them(one):
    _app()
    tab = ReviewTab(one)
    tab.load(str(_meta(one)))
    try:
        text = tab.angle_tree.text()
        n = len(tab.trial.angle_defs)
        assert text.count("at the peak force") == n, text                                       # a line per angle: the hip too, not only the knee
        for a in tab.trial.angle_defs:
            line = next(l for l in text.splitlines() if l.startswith(a.title))
            assert "at the peak force" in line
        others = [a.title for a in tab.trial.angle_defs if not a.primary]
        note = tab.force_tree.text(everything=True)
        assert all(f"; {t} " in note for t in others), note
    finally:
        tab.trial.close()
        tab.close()


def test_the_overview_has_a_column_for_the_other_angle_at_the_peak(one, tmp_path):
    import shutil
    _app()
    dest = tmp_path / "recordings"
    shutil.copytree(one, dest)
    tab = ReviewTab(dest)
    tab.set_participant(dest / "P01")
    tab.show_view("overview")
    assert tab.overview.wait_until_done(120000)
    tab.overview.render()
    col = review_overview.COLUMNS.index("Other angle at peak")
    assert review_overview.COLUMNS.index("Angle at peak (deg)") + 1 == col                     # right beside the main joint's
    cell = tab.overview.table.item(0, col)
    assert cell is not None and cell.text().endswith("°") and "Hip" in cell.text(), cell.text()
    assert "°" in cell.toolTip()
    tab.trial.close() if tab.trial is not None else None
    tab.overview.shutdown()
