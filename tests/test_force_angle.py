"""Force against angle in Review (0.17.0): how far the joint moved while the force rose. Display only: it uses the angles and the force that
are already there, paired by the picture delay, and changes no number."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.force import ForceCalibration  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402

from tests import test_force as T  # noqa: E402  (its helpers record a trial with force)


def ramp_trial(tmp_path, delay_free=True):
    """A real recorded trial whose arrays are replaced by a known ones: pictures every 1/30 s for 15 s, the knee angle = 2 deg per second, a
    force of 10 V per second (raw, no smoothing), so every pairing can be worked out by hand."""
    eng, paths = T.recording(tmp_path)
    tr = Trial(paths["meta.json"])
    set_ramp(tr)
    return tr, paths


def set_ramp(tr):
    n = len(tr.t)
    tr.t = np.linspace(0.0, 15.0, n)
    main = next(a for a in tr.angle_defs if a.primary)
    tr.deg[main.name] = 2.0 * tr.t
    tr.force_t = np.arange(-1.0, 16.0, 0.001)
    tr.force_v = 10.0 * tr.force_t
    tr.force_n = np.full(tr.force_t.size, np.nan)
    tr.force_smooth_s = 0.0
    return main


# -------------------------------------------------------------------------------------------------------------- the helper
def test_each_picture_is_paired_with_the_force_of_the_moment_it_shows(tmp_path):
    tr, _ = ramp_trial(tmp_path)
    p = analysis.force_angle_path(tr, 0.0)
    assert p["unit"] == "V" and p["angle_name"] == "Knee flexion" and p["ok"].all()
    assert p["angle"] == pytest.approx(2.0 * tr.t) and p["force"] == pytest.approx(10.0 * tr.t, abs=1e-6)
    late = analysis.force_angle_path(tr, 0.3)                                  # pictures stamped 0.3 s after they were taken: the picture at t shows t - 0.3
    assert late["force"] == pytest.approx(10.0 * (tr.t - 0.3), abs=1e-6) and late["angle"] == pytest.approx(2.0 * tr.t)
    early = analysis.force_angle_path(tr, -0.2)
    assert early["force"] == pytest.approx(10.0 * (tr.t + 0.2), abs=1e-6)


def test_no_force_is_made_up_outside_what_was_recorded(tmp_path):
    tr, _ = ramp_trial(tmp_path)
    p = analysis.force_angle_path(tr, -3.0)                                    # the last pictures would need force from after the recording
    assert not p["ok"][-1] and p["ok"][0] and np.isnan(p["force"][-1])
    assert not analysis.force_angle_path(tr, 17.0)["ok"].any()                 # nothing of the force reaches the pictures at all


def test_pictures_without_an_angle_are_left_out_and_not_filled_in(tmp_path):
    tr, _ = ramp_trial(tmp_path)
    main = next(a for a in tr.angle_defs if a.primary)
    tr.deg[main.name][10:14] = np.nan                                         # the dots were lost
    p = analysis.force_angle_path(tr, 0.0)
    assert not p["ok"][10:14].any() and p["ok"][:10].all() and p["ok"][14:].all()


def test_the_unit_is_newtons_when_the_force_is_calibrated(tmp_path):
    tr, _ = ramp_trial(tmp_path)
    cal = ForceCalibration()
    cal.set_zero(0.0)
    cal.n_per_v = 100.0
    tr.apply_calibration(cal, "test")
    p = analysis.force_angle_path(tr, 0.0)
    assert p["unit"] == "N" and p["force"] == pytest.approx(1000.0 * tr.t, abs=1e-3)


def test_a_trial_without_force_or_without_angles_has_no_path(tmp_path):
    tr, _ = ramp_trial(tmp_path)
    tr.angle_defs = []
    assert analysis.force_angle_path(tr) is None
    tr, _ = ramp_trial(tmp_path / "b")
    tr.force_t = None
    assert analysis.force_angle_path(tr) is None


def test_the_path_is_coloured_from_purple_to_yellow():
    cols = analysis.time_colours(5)
    assert cols[0] == (68, 1, 84) and cols[-1] == (253, 231, 37) and len(cols) == 5 and len(set(cols)) == 5
    assert analysis.time_colours(1) == [(68, 1, 84)] and analysis.time_colours(0) == []
    assert all(0 <= v <= 255 for c in analysis.time_colours(300) for v in c)


# ------------------------------------------------------------------------------------------------------------------ in Review
def review_with_ramp(tmp_path):
    app = QApplication.instance() or QApplication([])
    eng, paths = T.recording(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    main = set_ramp(tab.trial)
    tab.window.setRegion((5.0, 10.0))
    tab.update_summary()
    return tab, main


def items(tab):
    return tab.fa_plot.plotItem.items


def scatters(tab):
    return [i for i in items(tab) if isinstance(i, pg.ScatterPlotItem) and i is not tab._fa_cursor]


def test_the_default_is_the_time_graphs_and_the_switch_needs_force(tmp_path):
    tab, _ = review_with_ramp(tmp_path)
    assert tab.plot_view.currentData() == "time" and tab.plot_view.isEnabled()
    assert not tab.plot.isHidden() and not tab.force_plot.isHidden() and tab.fa_plot.isHidden()
    tab.trial.force_t = None                                                   # (a trial without force)
    tab._apply_plot_view()
    tab.plot_view.setCurrentIndex(1)
    assert not tab.plot_view.isEnabled() and not tab.plot.isHidden() and tab.fa_plot.isHidden() and tab.force_plot.isHidden()


def test_switching_shows_the_force_against_angle_and_hides_the_time_graphs(tmp_path):
    tab, main = review_with_ramp(tmp_path)
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("fa"))
    assert tab.plot.isHidden() and tab.force_plot.isHidden() and not tab.fa_plot.isHidden()
    assert "Knee flexion (deg)" in tab.fa_plot.getAxis("bottom").labelText and "Force (V)" in tab.fa_plot.getAxis("left").labelText
    n = len(tab.trial.t)
    colored, big, star = scatters(tab)[0], scatters(tab)[1], scatters(tab)[2]
    assert len(colored.data) == n                                              # a dot per picture
    inside = np.sum((tab.trial.t >= 5.0) & (tab.trial.t <= 10.0))
    assert len(big.data) == inside                                             # larger dots in the analysis window
    line = next(i for i in items(tab) if isinstance(i, pg.PlotDataItem))
    x, y = line.getData()
    assert x == pytest.approx(2.0 * tab.trial.t) and y == pytest.approx(10.0 * tab.trial.t, abs=1e-6)
    fs = tab.trial.force_summary(5.0, 10.0, 0.0)                               # the star is the peak the Force tab reports
    sx, sy = star.getData()
    assert sx[0] == pytest.approx(fs["main_angle_at_peak_deg"]) and sy[0] == pytest.approx(fs["peak"]) and sy[0] == pytest.approx(100.0, abs=0.1)
    tab.plot_view.setCurrentIndex(0)
    assert not tab.plot.isHidden() and not tab.force_plot.isHidden() and tab.fa_plot.isHidden()


def test_the_ring_follows_the_video(tmp_path):
    tab, _ = review_with_ramp(tmp_path)
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("fa"))
    for i in (0, 20, len(tab.trial.t) - 1):
        tab.slider.setValue(i)
        tab.show_frame(i)
        x, y = tab._fa_cursor.getData()
        assert x[0] == pytest.approx(2.0 * tab.trial.t[i]) and y[0] == pytest.approx(10.0 * tab.trial.t[i], abs=1e-6)
    tab.plot_view.setCurrentIndex(0)
    tab.show_frame(5)
    assert len(tab._fa_cursor.getData()[0]) == 0                               # no ring when the view is not on


def test_the_picture_delay_the_window_and_the_smoothing_move_the_path(tmp_path):
    tab, _ = review_with_ramp(tmp_path)
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("fa"))
    tab.picture_delay.setValue(0.5)
    line = next(i for i in items(tab) if isinstance(i, pg.PlotDataItem))
    ok = ~np.isnan(tab._fa_path["force"])
    assert line.getData()[1] == pytest.approx(10.0 * (tab.trial.t[ok] - 0.5), abs=1e-6)
    before = len(scatters(tab)[1].data)
    tab.window.setRegion((2.0, 12.0))                                          # a wider window: more large dots
    assert len(scatters(tab)[1].data) > before


def test_a_trial_that_has_no_picture_with_both_numbers_says_so(tmp_path):
    tab, _ = review_with_ramp(tmp_path)
    main = next(a for a in tab.trial.angle_defs if a.primary)
    tab.trial.deg[main.name][:] = np.nan
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("fa"))
    assert "No picture has both" in tab.fa_plot.plotItem.titleLabel.text


def test_showing_it_changes_no_number(tmp_path):
    tab, _ = review_with_ramp(tmp_path)
    before = (tab.stats.text(), tab.force_text.text(), tab.check_text.toPlainText())
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("fa"))
    assert (tab.stats.text(), tab.force_text.text(), tab.check_text.toPlainText()) == before


def test_the_large_dots_are_the_pictures_whose_force_lies_in_the_window(tmp_path):
    """The window is in force time (as in the Force tab): with a picture delay the picture stamped t shows the force of t - delay."""
    tab, _ = review_with_ramp(tmp_path)
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("fa"))
    tab.picture_delay.setValue(0.5)
    tab.window.setRegion((5.0, 10.0))
    ft = tab.trial.t - 0.5
    inside = (ft >= 5.0) & (ft <= 10.0)
    x, y = scatters(tab)[1].getData()
    assert len(y) == inside.sum() and y == pytest.approx(10.0 * ft[inside], abs=1e-6)          # forces 50 to 100: the window's own force
    assert min(y) >= 50.0 - 1e-6 and max(y) <= 100.0 + 1e-6


def test_dragging_the_window_redraws_only_its_dots_and_the_star(tmp_path):
    tab, _ = review_with_ramp(tmp_path)
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("fa"))
    path_dots = scatters(tab)[0]
    tab.window.setRegion((2.0, 12.0))
    tab.window.setRegion((3.0, 11.0))
    assert scatters(tab)[0] is path_dots and len(scatters(tab)) == 3            # the same path; one set of large dots and one star
    tab.picture_delay.setValue(0.2)                                            # the pairing of force and picture changed: drawn afresh
    assert scatters(tab)[0] is not path_dots and len(scatters(tab)) == 3


def test_a_trial_opened_after_the_smoothing_was_changed_shows_that_smoothing(tmp_path):
    app = QApplication.instance() or QApplication([])
    eng, paths = T.recording(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.force_smooth.setValue(100.0)                                           # set before any trial is open
    tab.load(str(paths["meta.json"]))
    assert tab.trial.force_smooth_s == pytest.approx(0.1)
    curve = [i for i in tab.force_plot.plotItem.items if isinstance(i, pg.PlotDataItem)][-1]
    assert curve.yData == pytest.approx(tab.trial.force_signal()[1], nan_ok=True)                 # the strip shows what the numbers use
