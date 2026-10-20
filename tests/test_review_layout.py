"""Review tab layout (0.14.0): a visible position line, the graphs top right, the number tabs in a wide strip below,
divider positions that are remembered."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QHBoxLayout, QSizePolicy  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReviewTab, _CalibrationPane  # noqa: E402
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


def test_graphs_are_top_right_and_the_number_tabs_are_below(tmp_path):
    _app()
    tab = ReviewTab(tmp_path)
    top = tab.top_splitter
    assert top.orientation().name == "Horizontal" and tab.splitter.orientation().name == "Vertical"
    assert tab.splitter.widget(0) is top and tab.splitter.widget(1) is tab.tabs
    assert top.widget(0).isAncestorOf(tab.video)                 # video and its controls on the left
    assert top.widget(1).isAncestorOf(tab.plot) and top.widget(1).isAncestorOf(tab.force_plot)
    assert not top.isAncestorOf(tab.tabs)


def test_graphs_get_the_larger_share_of_the_top_row(tmp_path):
    app = _app()
    tab = ReviewTab(tmp_path)
    tab.resize(1400, 820)
    tab.show()
    app.processEvents()
    left, graphs = tab.top_splitter.sizes()
    assert graphs > left
    upper, lower = tab.splitter.sizes()
    assert upper > lower >= tab.tabs.minimumHeight()
    tab.close()


def test_calibration_pane_lies_side_by_side_to_fit_the_strip():
    _app()
    assert isinstance(_CalibrationPane().layout(), QHBoxLayout)


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
    a.splitter.setSizes([400, 400])
    app.processEvents()
    a._save_layout()
    a.close()
    saved = json.loads(settings.read_text(encoding="utf-8"))
    assert saved["recordings_dir"] == "D:/data" and saved["session"]["test"] == "hip_flexion"
    assert set(saved["review_ui"]) == {"top", "main"}

    b = ReviewTab(tmp_path)
    b.attach_memory(settings)
    b.resize(2300, 900)
    b.show()
    app.processEvents()
    left, graphs = b.top_splitter.sizes()
    assert left / (left + graphs) == pytest.approx(800 / 2200, abs=0.04)
    upper, lower = b.splitter.sizes()
    assert upper / (upper + lower) == pytest.approx(0.5, abs=0.05)
    b.close()


@pytest.mark.parametrize("bad", [None, [], "x", {"top": "x"}, {"top": [1], "main": [1, 2]}, {"top": [0, 5], "main": [5, 5]},
                                 {"top": ["a", "b"], "main": [3, 4]}])
def test_unusable_remembered_positions_are_ignored(tmp_path, bad):
    app = _app()
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"review_ui": bad}), encoding="utf-8")
    tab = ReviewTab(tmp_path)
    tab.attach_memory(settings)                                  # must not raise
    tab.resize(1400, 820)
    tab.show()
    app.processEvents()
    assert all(s > 0 for s in tab.top_splitter.sizes() + tab.splitter.sizes())
    tab.close()


def test_without_a_settings_file_nothing_is_written(tmp_path):
    _app()
    tab = ReviewTab(tmp_path)
    tab._save_layout()                                           # no memory attached: harmless
    assert not (tmp_path / "settings.json").exists()
