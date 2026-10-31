"""Force and angle on one graph with two vertical axes (0.31.0): the Live tab's trace mode and Review's view. Only what is drawn: the force is
the smoothed force the other graphs already show, and nothing here touches what is saved. Also the fixed 'peak (20 ms)'."""
import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.review import PEAK_20MS_S, ReviewTab  # noqa: E402
from limbtrack.twoaxis import TwoAxisPlot  # noqa: E402

from tests import test_force as T  # noqa: E402
from tests.test_force_gui import connect_simulated, setup  # noqa: E402,F401  (the fixture: a real window with the simulated force)


def _app():
    return QApplication.instance() or QApplication([])


# ------------------------------------------------------------------------------------------------ the plot itself
def test_the_second_axis_is_hidden_until_it_is_asked_for_and_shares_the_time_axis():
    _app()
    pl = TwoAxisPlot()
    pl.resize(600, 300)
    pl.show()
    _app().processEvents()
    assert not pl._right_on and not pl.right.isVisible()
    curve = pg.PlotCurveItem([0, 1, 2], [10, 20, 30])
    pl.use_right(True)
    pl.add_right(curve)
    pl.set_right_range(0, 100)
    pl.setXRange(-15, 0, padding=0)
    _app().processEvents()
    assert pl._right_on and pl.right.isVisible() and curve in pl.right.addedItems
    assert pl.right.viewRange()[0] == pytest.approx(pl.getPlotItem().vb.viewRange()[0])               # the same time range
    assert pl.right.viewRange()[1] == pytest.approx([0, 100])
    pl.remove_right(curve)
    pl.remove_right(curve)                                                                              # twice is fine
    assert curve not in pl.right.addedItems
    pl.use_right(False)
    assert not pl._right_on
    pl.close()


# ------------------------------------------------------------------------------------------------ the Live tab
def test_the_live_trace_can_show_the_angle_and_the_smoothed_force_on_two_axes(setup):
    app, win, live, warnings, pump, tmp = setup
    live.trace_mode.setCurrentIndex(live.trace_mode.findData("both"))
    live.update_trace()                                                    # no force connected: nothing breaks, the axis says so
    assert live.trace._right_on and live._trace_right_items == []
    connect_simulated(live)
    assert pump(lambda: live.force_source.average_volts(1.0)[1] >= 500, timeout=10)
    live.update_trace()
    assert len(live._trace_right_items) == 1
    x, y = live._trace_right_items[0].getData()
    assert 100 < len(x) <= 1500 and x[0] < x[-1] <= 0.5                                                 # the same thinned 'seconds ago' series as the Force mode
    other = live.trace_mode
    other.setCurrentIndex(other.findData("force"))                         # the other modes are as they were
    live.update_trace()
    assert not live.trace._right_on and live._trace_right_items == [] and live._trace_items
    other.setCurrentIndex(other.findData("main"))
    live.update_trace()
    assert not live.trace._right_on and live._trace_right_items == []
    live.btn_force.click()                                                  # disconnect while the two-axis mode is on
    other.setCurrentIndex(other.findData("both"))
    live.update_trace()
    assert live._trace_right_items == []


def test_the_two_axis_mode_is_added_last_so_the_older_modes_keep_their_places(setup):
    app, win, live, warnings, pump, tmp = setup
    assert [live.trace_mode.itemData(i) for i in range(live.trace_mode.count())] == ["main", "drift", "force", "force_v", "both"]


# ------------------------------------------------------------------------------------------------ Review
@pytest.fixture(scope="module")
def force_trial(tmp_path_factory):
    root = tmp_path_factory.mktemp("two_axis")
    eng, paths = T.recording(root)
    return root, paths["meta.json"]


@pytest.fixture()
def tab(force_trial, tmp_path):
    _app()
    dest = tmp_path / "recordings"
    shutil.copytree(force_trial[0], dest)
    t = ReviewTab(dest)
    t.resize(1500, 900)
    t.show()
    t.load(dest / "P01" / force_trial[1].name)
    yield t
    t.trial.close()
    t.close()


def test_review_has_a_view_with_the_angle_on_the_left_axis_and_the_force_on_the_right(tab):
    assert tab.plot_view.isEnabled() and tab.plot_view.findData("both") >= 0
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("both"))
    _app().processEvents()
    assert tab.both_plot.isVisible() and not tab.plot.isVisible() and not tab.force_plot.isVisible() and not tab.fa_plot.isVisible()
    assert len(tab._both_items) == len(tab.trial.angle_defs) and len(tab._both_right_items) == 1
    unit, ys, _ = tab.trial.force_signal()
    curve = tab._both_right_items[0]
    assert curve in tab.both_plot.right.addedItems and curve.xData.size == tab.trial.force_t.size and np.allclose(curve.yData, ys, equal_nan=True)
    assert "Force (N, smoothed 100 ms)" in tab.both_plot.getAxis("right").labelText
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("time"))
    _app().processEvents()
    assert tab.plot.isVisible() and tab.force_plot.isVisible() and not tab.both_plot.isVisible()


def test_the_two_axis_view_follows_the_picture_the_window_and_the_smoothing(tab):
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("both"))
    tab.slider.setValue(7)
    assert tab.both_cursor.value() == pytest.approx(float(tab.trial.t[7]))
    tab.window.setRegion((1.0, 2.0))
    assert tuple(tab.window_both.getRegion()) == pytest.approx((1.0, 2.0))
    tab.window_both.setRegion((1.2, 2.4))                                    # dragged on the two-axis graph: it moves the analysis window
    assert tuple(tab.window.getRegion()) == pytest.approx((1.2, 2.4)) and tuple(tab.force_window_mirror.getRegion()) == pytest.approx((1.2, 2.4))
    tab.force_smooth.setValue(0)
    assert np.allclose(tab._both_right_items[0].yData, tab.trial.force_n, equal_nan=True) and "raw" in tab.both_plot.getAxis("right").labelText
    tab.force_smooth.setValue(200)
    assert "smoothed 200 ms" in tab.both_plot.getAxis("right").labelText


def test_a_trial_without_force_has_no_two_axis_view(tmp_path):
    from limbtrack.engine import Engine
    from limbtrack.synthetic import render
    _app()
    eng = Engine(T.TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.process(render(T.PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(T.PTS[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(6):
        eng.process(render(T.PTS, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    t = ReviewTab(tmp_path)
    t.load(paths["meta.json"])
    t.plot_view.setCurrentIndex(t.plot_view.findData("both"))
    assert not t.plot_view.isEnabled() and t._both_items == [] and t._both_right_items == [] and not t.both_plot.isVisibleTo(t)
    t.trial.close()


# ------------------------------------------------------------------------------------------------ peak (20 ms)
def test_the_peak_of_20_ms_is_the_highest_20_ms_average_and_does_not_depend_on_the_smoothing_box(tmp_path):
    tr, clean = T.humming_trial(tmp_path, hum=0.09)                          # 90 mV of 50 Hz hum on a hold at 1.0 V
    win = (8.0, 13.5)
    tr.force_smooth_s = 0.1
    a = tr.force_summary(*win)
    tr.force_smooth_s = 0.5
    b = tr.force_summary(*win)
    tr.force_smooth_s = 0.0
    c = tr.force_summary(*win)
    assert PEAK_20MS_S == 0.02
    assert a["peak_20ms"] == b["peak_20ms"] == c["peak_20ms"]                # the same at any Smoothing
    assert a["peak_20ms"] == pytest.approx(1.0, abs=0.02)                    # the hum is gone
    assert c["raw_peak"] > 1.08 and c["raw_peak"] - a["peak_20ms"] > 0.07   # a raw single sample stands on the crest of the hum
    assert a["peak_20ms"] <= a["raw_peak"] and abs(a["peak_20ms"] - a["peak"]) < 0.02
    row = analysis.session_row(tr)
    assert row["force_peak_20ms"] == pytest.approx(1.0, abs=0.02) and "force_peak_20ms" in analysis.SESSION_COLUMNS
