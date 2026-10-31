"""Review tab layout (0.32.0; before that 0.14.0): the trial on the left and the signal with its checks on the right, the number panels under the video,
divider positions that are remembered."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import LAYOUT_VERSION, ReviewTab, _CalibrationPane  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def _app():
    return QApplication.instance() or QApplication([])


def _trial_with_markers(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    pts = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(pts, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(pts[n] + 2))
    eng.process(render(pts, seed=0), 0.01)
    assert eng.lock_position()
    eng.start_trial("P01")
    for i in range(40):
        if i == 10:
            assert eng.add_event("MVIC start")
        if i == 30:
            assert eng.add_event("MVIC end")
        eng.process(render(leg_points(hip_flex=70, knee_flex=60, ankle_df=5), seed=i), i / 30.0)
    return eng.stop_trial()


def _luminance(c):
    def lin(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(c.red()) + 0.7152 * lin(c.green()) + 0.0722 * lin(c.blue())


def _contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_position_line_and_marker_lines_can_be_seen_on_the_plot_background(tmp_path):
    _app()
    paths = _trial_with_markers(tmp_path)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    bg = tab.plot.backgroundBrush().color()
    assert _contrast(tab.cursor.pen.color(), bg) >= 4.5          # it was black on black until 0.14.0
    assert _contrast(tab.force_cursor.pen.color(), tab.force_plot.backgroundBrush().color()) >= 4.5
    marks = [i for i in tab.plot.plotItem.items if isinstance(i, pg.InfiniteLine) and i is not tab.cursor and i.label is not None]
    assert len(marks) == 2                                       # MVIC start and MVIC end
    for line in marks:
        assert _contrast(line.pen.color(), bg) >= 4.5


def test_the_trial_is_on_the_left_and_the_signal_on_the_right(tmp_path):
    _app()
    tab = ReviewTab(tmp_path)
    top = tab.top_splitter
    assert top.orientation().name == "Horizontal" and tab.left_split.orientation().name == "Vertical"
    assert top.widget(0) is tab.left_split and top.widget(1) is tab.right_stack
    assert tab.left_split.widget(1) is tab.angle_tree and tab.left_split.widget(0).isAncestorOf(tab.video)    # video and its controls; the table of angles and their dots under them
    assert tab.left_split.widget(0).isAncestorOf(tab.btn_fix) and not tab.left_split.widget(0).isAncestorOf(tab.plot)
    signal = tab.right_stack.widget(0)
    assert signal.isAncestorOf(tab.plot) and signal.isAncestorOf(tab.force_plot) and signal.isAncestorOf(tab.calib) and signal.isAncestorOf(tab.force_tree)
    assert not signal.isAncestorOf(tab.checks) and not tab.right_stack.widget(1).isAncestorOf(tab.plot)
    assert tab.right_stack.widget(1).isAncestorOf(tab.fix_dots)                 # the dot workspace takes the right side while the dots are fixed
    for gone in ("tabs", "cards", "angle_cards", "kpi_widget", "force_box", "angle_panel", "force_panel", "dots_panel"):
        assert not hasattr(tab, gone), gone                                       # (the cards, the tabs and the panels are the two tables now)


def test_the_checks_are_a_chip_in_the_bottom_bar_below_the_graphs(tmp_path):
    _app()
    tab = ReviewTab(tmp_path)
    tab.resize(1400, 820)
    tab.show()
    _app().processEvents()
    assert tab.checks.isVisible() and tab.plot.isVisible()
    assert tab.checks.mapTo(tab, tab.checks.rect().topLeft()).y() > tab.force_plot.mapTo(tab, tab.force_plot.rect().bottomLeft()).y() or tab.force_plot.isHidden()
    assert tab.plot.mapTo(tab, tab.plot.rect().topLeft()).y() < tab.checks.mapTo(tab, tab.checks.rect().topLeft()).y()
    tab.close()


def test_the_graphs_and_the_video_get_the_room_not_the_tables(tmp_path):
    app = _app()
    tab = ReviewTab(tmp_path)
    tab.resize(1400, 820)
    tab.show()
    app.processEvents()
    left, right = tab.top_splitter.sizes()
    assert left > right * 1.25                                                   # 0.33.0: the video side is the wider one (630 : 370), the graph side the narrower
    assert right >= 460                                                          # ... but never narrower than the graphs need
    video_block, table = tab.left_split.sizes()
    assert video_block > table >= tab.angle_tree.minimumSizeHint().height()
    tab.close()


def test_the_calibration_pane_is_a_graph_view_with_its_plots_over_its_words():
    _app()
    pane = _CalibrationPane()
    assert isinstance(pane.layout(), QVBoxLayout)
    assert pane.text.maximumHeight() < 250


def test_export_and_more_are_not_in_the_header_but_in_the_overview(tmp_path):
    _app()
    tab = ReviewTab(tmp_path)
    assert not hasattr(tab, "btn_export") and not hasattr(tab, "btn_more")
    assert not tab.header.isAncestorOf(tab.overview) and tab.overview.isAncestorOf(tab.overview.export_button) and tab.overview.isAncestorOf(tab.overview.more_button)
    assert tab.overview.export_button.menu() is tab.export_menu and tab.overview.more_button.menu() is tab.more_menu
    assert [a.text() for a in tab.export_menu.actions() if not a.isSeparator()] == [tab.act_export_all.text(), tab.act_export_valid.text(),
                                                                                   tab.act_export_trial.text(), tab.act_export_video.text()]


def test_video_label_does_not_dictate_its_size():
    _app()
    tab = ReviewTab(".")
    pol = tab.video.sizePolicy()
    assert pol.horizontalPolicy() == QSizePolicy.Ignored and pol.verticalPolicy() == QSizePolicy.Ignored


def test_divider_positions_are_remembered_and_the_rest_of_the_file_is_kept(tmp_path):
    app = _app()
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"recordings_dir": "D:/data", "session": {"format": 1, "test": "hip_flexion"}}), encoding="utf-8")
    a = ReviewTab(tmp_path)
    a.attach_memory(settings)
    a.resize(2300, 900)                                          # wide enough that the panes' minimum widths leave room to move the divider
    a.show()
    app.processEvents()
    a.top_splitter.setSizes([800, 1400])
    a.left_split.setSizes([400, 400])
    app.processEvents()
    a._save_layout()
    a.close()
    saved = json.loads(settings.read_text(encoding="utf-8"))
    assert saved["recordings_dir"] == "D:/data" and saved["session"]["test"] == "hip_flexion"
    assert set(saved["review_ui"]) == {"v", "top", "main", "fix"} and saved["review_ui"]["v"] == LAYOUT_VERSION

    b = ReviewTab(tmp_path)
    b.attach_memory(settings)
    b.resize(2300, 900)
    b.show()
    app.processEvents()
    left, graphs = b.top_splitter.sizes()
    assert left / (left + graphs) == pytest.approx(800 / 2200, abs=0.04)
    upper, lower = b.left_split.sizes()
    assert upper / (upper + lower) == pytest.approx(0.5, abs=0.05)
    b.close()


def test_a_layout_saved_before_0_33_0_is_not_put_back_but_a_new_one_is(tmp_path):
    app = _app()
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"review_ui": {"top": [516, 872], "main": [651, 400], "fix": [487, 901]}}), encoding="utf-8")   # the user's, from 0.32.0
    old = ReviewTab(tmp_path)
    old.attach_memory(settings)
    old.resize(1400, 820)
    old.show()
    app.processEvents()
    left, right = old.top_splitter.sizes()
    assert left > right                                                          # the new proportions, not 516 : 872
    old.close()
    settings.write_text(json.dumps({"review_ui": {"v": LAYOUT_VERSION, "top": [500, 900], "main": [300, 300], "fix": [800, 600]}}), encoding="utf-8")
    new = ReviewTab(tmp_path)
    new.attach_memory(settings)
    new.resize(1400, 820)
    new.show()
    app.processEvents()
    left, right = new.top_splitter.sizes()
    assert left < right                                                          # a layout of this version is kept
    new.close()


def test_the_table_hidden_while_dots_are_fixed_does_not_save_a_zero_share(tmp_path):
    app = _app()
    paths = _trial_with_markers(tmp_path)
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    tab = ReviewTab(tmp_path)
    tab.attach_memory(settings)
    tab.resize(1700, 950)
    tab.show()
    tab.load(str(paths["meta.json"]))
    app.processEvents()
    normal_main = list(tab.layout_state()["main"])
    tab.set_fix_mode(True)
    app.processEvents()
    assert tab.angle_tree.isHidden() and 0 in tab.left_split.sizes()          # (the table is hidden: its share is 0 now)
    tab.top_splitter.setSizes([900, 400])
    tab._top_moved()
    tab._save_layout()
    saved = json.loads(settings.read_text(encoding="utf-8"))["review_ui"]
    assert saved["main"] == normal_main and all(s > 0 for s in saved["main"] + saved["top"] + saved["fix"])
    tab.set_fix_mode(False)
    tab.trial.close()
    tab.close()


def test_a_trial_opens_on_one_graph_with_two_axes_and_a_trial_without_force_on_the_angle_graph(tmp_path):
    from tests import test_force as T
    app = _app()
    eng, paths = T.recording(tmp_path / "with")
    tab = ReviewTab(tmp_path / "with")
    tab.resize(1500, 900)
    tab.show()
    assert tab.plot_view.currentData() == "time" and not tab.plot_view.isEnabled()      # nothing open: the angle graph, the chooser is off
    tab.load(paths["meta.json"])
    app.processEvents()
    assert tab.plot_view.isEnabled() and tab.plot_view.currentData() == "both"
    assert tab.both_plot.isVisible() and not tab.plot.isVisible() and not tab.force_plot.isVisible()
    tab.plot_view.setCurrentIndex(tab.plot_view.findData("fa"))                          # the choice is kept for the next trial of the tab
    assert tab._plot_choice == "fa"
    tab.trial.close()
    tab.close()


@pytest.mark.parametrize("bad", [None, [], "x", {"top": "x"}, {"top": [1], "main": [1, 2]}, {"top": [0, 5], "main": [5, 5]},
                                 {"top": ["a", "b"], "main": [3, 4]}, {"v": LAYOUT_VERSION, "top": [0, 5], "main": [5, 0]},
                                 {"v": LAYOUT_VERSION, "top": ["a", "b"], "main": [3, 4]}])
def test_unusable_remembered_positions_are_ignored(tmp_path, bad):
    app = _app()
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"review_ui": bad}), encoding="utf-8")
    tab = ReviewTab(tmp_path)
    tab.attach_memory(settings)                                  # must not raise
    tab.resize(1400, 820)
    tab.show()
    app.processEvents()
    assert all(s > 0 for s in tab.top_splitter.sizes() + tab.left_split.sizes())
    tab.close()


def test_without_a_settings_file_nothing_is_written(tmp_path):
    _app()
    tab = ReviewTab(tmp_path)
    tab._save_layout()                                           # no memory attached: harmless
    assert not (tmp_path / "settings.json").exists()
