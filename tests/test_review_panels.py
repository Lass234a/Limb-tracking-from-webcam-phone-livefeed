"""The number panels of the Review tab, 0.32.0 (limbtrack/review_panels.py): the angle panel (the angle at the peak force as the headline, the
table of every angle, the reference readings), the force panel (Peak, Level, Timing, Torque and Load cell as sections that re-flow with the
width) and the FactGrid both are made of. Until 0.31.0 these were monospace text in tabs, tested by looking for words in a string; the
panels have `text()` for the same purpose, and these tests also look at what is drawn (colours, columns, positions) because a panel can
read right as text and still be cut off, overlap, or widen the window.

The numbers come from one real recorded trial (a calibrated 200 N hold with a ruler scale and the anchor dot, tests.test_analysis.torque_trial);
what a test needs to be exact (an angle at the peak of 77.5, a hum of 10 %) is put into copies of the dicts the panels are given, which
is also what Review does: the panels only show what they are handed."""
import copy
import math
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QScrollArea, QVBoxLayout, QWidget  # noqa: E402

from limbtrack import analysis  # noqa: E402
from limbtrack.loadcell import COVERAGE, GRAVITY, fit_points  # noqa: E402
from limbtrack.review_panels import UNKNOWN, AnglePanel, FactGrid, ForcePanel, num, strong_colour  # noqa: E402
from limbtrack.review_style import GOOD, STATE_COLOURS, WARN  # noqa: E402

from tests.test_analysis import torque_trial  # noqa: E402

WINDOW = (8.5, 13.0)                 # the hold of torque_trial (the force is at 200 N from about 8.6 to 13 s)
MORE_ROWS = ["Mean", "SD", "Min", "Max", "Start to end", "Max deviation", "Median", "Rest", "Rest to hold"]


# ------------------------------------------------------------------------------------------------ set-up
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def base(app, tmp_path_factory):
    """One recorded trial with a calibrated force, a ruler scale and the anchor dot (it takes some seconds to make); `trial` hands out a fresh
    state of it to each test."""
    tr = torque_trial(tmp_path_factory.mktemp("panels_trial"))
    yield tr
    tr.close()


HIDDEN = []                          # errors raised inside an event handler or a slot (see `no_hidden_errors`)


@pytest.fixture(autouse=True)
def no_hidden_errors(monkeypatch, request):
    """Qt prints an exception raised inside an event handler (a resize, a palette change, a click) and carries on, so a test would pass straight
    over it. They are written down here instead, and the test fails."""
    HIDDEN.clear()
    monkeypatch.setattr(sys, "excepthook", lambda kind, err, tb: HIDDEN.append(f"{kind.__name__}: {err}"))
    yield
    if HIDDEN and not request.node.get_closest_marker("xfail"):
        pytest.fail("an error was raised inside a Qt handler and hidden by Qt: " + "; ".join(HIDDEN))


_KEPT = ("meta", "tracks", "px_per_cm", "force_v", "force_n", "regression", "force_cal", "force_cal_note", "_reference_rows", "reference_side",
         "events", "force_smooth_s", "window_peak_s")


@pytest.fixture
def trial(base):
    """The recorded trial in a known state (tolerance 5 deg, no markers, no reference reading); whatever a test changes is put back afterwards."""
    saved = {k: (copy.deepcopy(getattr(base, k)) if k in ("meta", "force_cal", "reference_side") else
                 dict(base.tracks) if k == "tracks" else getattr(base, k)) for k in _KEPT}
    base.meta["tolerance_deg"] = 5.0
    base.meta.pop("reference_readings", None)
    base._reference_rows, base.reference_side, base.events = None, {}, []
    yield base
    for k, v in saved.items():
        setattr(base, k, v)
    base.__dict__.pop("_rest_cache", None)


@pytest.fixture
def palette(app):
    old = app.palette()
    yield app
    app.setPalette(old)


def dark(app):
    p = QPalette()
    p.setColor(QPalette.Window, QColor(32, 32, 32))
    p.setColor(QPalette.WindowText, QColor(230, 230, 230))
    p.setColor(QPalette.Base, QColor(24, 24, 24))
    p.setColor(QPalette.Text, QColor(230, 230, 230))
    p.setColor(QPalette.Button, QColor(48, 48, 48))
    p.setColor(QPalette.ButtonText, QColor(230, 230, 230))
    app.setPalette(p)


def numbers(tr, window=WINDOW, delay=0.0):
    """(rows, fs, ft, fp): what Review hands the panels for a window (ReviewTab.update_summary)."""
    t0, t1 = window
    rows = tr.summary(t0, t1)
    fs = tr.force_summary(t0, t1, delay)
    ft = analysis.force_timing(tr, t0, t1) if fs is not None else None
    return rows, fs, ft, analysis.force_peaks(tr, t0, t1, timing=ft)


def force_panel(tr, fs=None, ft="real", fp="real", window=WINDOW, delay=0.0, width=None):
    """A ForcePanel filled for `tr`; `fs`/`ft`/`fp` replace the real numbers (pass None for a missing one)."""
    rows, real_fs, real_ft, real_fp = numbers(tr, window, delay)
    panel = ForcePanel()
    panel.set_data(tr, window[0], window[1], real_fs if fs is None else fs, real_ft if ft == "real" else ft, real_fp if fp == "real" else fp, delay)
    if width:
        panel.resize(width, 700)
    return panel


def angle_panel(tr, rows=None, fs="real", window=WINDOW):
    r, real_fs, _ft, _fp = numbers(tr, window)
    panel = AnglePanel()
    panel.set_data(tr, r if rows is None else rows, real_fs if fs == "real" else fs, window)
    return panel


def colour_of(widget):
    """The colour of a style sheet with one 'color: #rrggbb;' in it, or None."""
    m = re.search(r"color:\s*(#[0-9a-fA-F]{6})", widget.styleSheet())
    return m.group(1).lower() if m else None


def luminance(c):
    return 0.2126 * c.red() + 0.7152 * c.green() + 0.0722 * c.blue()


def show(widget, width, height=800):
    widget.resize(width, height)
    widget.show()
    QApplication.processEvents()
    return widget


# ------------------------------------------------------------------------------------------------ the small helpers
def test_num_never_shows_minus_zero_or_nan():
    assert num(-0.04) == "0.0" and num(float("nan")) == UNKNOWN == "-" and num(None) == "-" and num("x") == "-"
    assert num(-0.04, signed=True) == "+0.0" and num(2.5, signed=True) == "+2.5" and num(-2.5, signed=True) == "-2.5" and num(1.23456, d=3) == "1.235"


def test_the_strong_colour_is_darker_in_a_light_theme_and_lighter_in_a_dark_one(app):
    light, night = QPalette(), QPalette()
    light.setColor(QPalette.WindowText, QColor(20, 20, 20))
    night.setColor(QPalette.WindowText, QColor(235, 235, 235))
    a, b = QColor(strong_colour(GOOD, light)), QColor(strong_colour(GOOD, night))
    assert luminance(a) < luminance(QColor(GOOD)) < luminance(b)              # text on a pale background needs more than the plain green


# ------------------------------------------------------------------------------------------------ FactGrid
def test_a_fact_grid_is_rows_of_label_value_and_note_and_gives_them_back_as_lines(app):
    grid = FactGrid()
    grid.set_rows([("Peak", "400.0 N", "at 0.55 s", "good"), ("Rest", "-")])                     # a row may be short: the rest is filled in
    assert grid.rows == [("Peak", "400.0 N", "at 0.55 s", "good"), ("Rest", "-", None, None)]
    assert (grid.label_text(0), grid.value_text(0), grid.note_text(0)) == ("Peak", "400.0 N", "at 0.55 s")
    assert grid.text() == "Peak: 400.0 N  at 0.55 s\nRest: -"                                    # no double space and no 'None' for a row without a note
    grid.set_rows([])
    assert grid.text() == "" and grid.heightForWidth(300) == 0 and grid.sizeHint().height() == 0


def test_a_note_stands_beside_the_value_when_it_fits_and_under_the_row_when_it_does_not(app):
    grid = FactGrid()
    note = "highest mean over 500 ms of the raw samples, 11.13 to 11.63 s, and some more words so that it cannot fit beside"
    grid.set_rows([("Peak", "400.0 N", "at 0.55 s", None), ("Sliding-window peak", "399.9 N", note, None), ("Rest", "1.0 N", "", None)])
    show(grid, 600)
    labels, values, short, long_, none = grid._labels, grid._values, grid._notes[0], grid._notes[1], grid._notes[2]
    assert short.isVisible() and short.y() == values[0].y() and short.x() >= values[0].x() + values[0].width()        # beside the value
    wide = grid.heightForWidth(grid.width())
    show(grid, 220)
    assert long_.isVisible() and long_.y() > values[1].y() and long_.x() == 0 and long_.width() == grid.width()       # under the row, wrapped to the width
    assert none.isHidden()                                                                                         # no note, no room taken
    assert grid.heightForWidth(220) > wide                                                                         # (the wrapped note makes the grid taller)
    assert labels[0].x() == 0 and labels[0].y() == values[0].y()


def test_a_very_long_label_or_value_is_cut_with_dots_and_never_widens_the_grid(app):
    grid = FactGrid()
    long_label, long_value = "A label that is much too long for any grid " * 4, "123456789012345678901234567890 N " * 4
    grid.set_rows([(long_label, long_value, "", None)])
    show(grid, 260)
    from PySide6.QtWidgets import QLabel
    shown_label, shown_value = QLabel.text(grid._labels[0]), QLabel.text(grid._values[0])
    assert shown_label.endswith("…") and len(shown_label) < len(long_label) and shown_value.endswith("…") and len(shown_value) < len(long_value)
    assert grid.label_text(0) == long_label and grid.value_text(0) == long_value                  # the whole text is still there for text() ...
    assert long_label in grid._labels[0].toolTip() and long_value in grid._values[0].toolTip()   # ... and in the tooltip
    assert grid.minimumSizeHint().width() == 100 and grid.minimumWidth() == 100
    assert grid._values[0].geometry().right() < grid.width() and grid._labels[0].geometry().right() < grid.width()


def test_a_state_colours_the_value_and_fewer_rows_hide_the_rest(app):
    grid = FactGrid()
    grid.set_rows([("a", "1", "", "good"), ("b", "2", "", "warn"), ("c", "3", "", "bad"), ("d", "4", "", None)])
    pal = QApplication.palette()
    assert [colour_of(w) for w in grid._values] == [strong_colour(STATE_COLOURS[s], pal) for s in ("good", "warn", "bad")] + [None]
    grid.set_rows([("a", "1", "n", None)])
    assert colour_of(grid._values[0]) is None                                                   # the colour left with the state
    assert all(w.isHidden() for i in (1, 2, 3) for w in (grid._labels[i], grid._values[i], grid._notes[i]))
    before = len(grid.findChildren(QWidget))
    grid.set_rows([("a", "1", "n", None)])                                                      # the same rows again: nothing is made again
    grid.set_rows([("a", "1", "n", None), ("b", "2", "", None)])                                # and a row that was hidden comes back, not a new one
    assert len(grid.findChildren(QWidget)) == before and not grid._labels[1].isHidden()


# ------------------------------------------------------------------------------------------------ AnglePanel: the headline
def test_the_headline_is_the_angle_at_the_peak_force_with_its_name_and_target(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    panel = angle_panel(trial)
    assert panel.headline_caption.text() == "Knee flexion at the peak force"
    assert panel.headline_value.text() == f"{fs['main_angle_at_peak_deg']:.1f} deg" == "70.0 deg"
    assert panel.headline_target.text() == "Target 70 deg, tolerance 5" and panel.headline_off.text() == "On the target"
    assert panel._headline_state == "good"
    assert colour_of(panel.headline_value) == strong_colour(GOOD, QApplication.palette(), 0.8) == colour_of(panel.headline_off)
    first = panel.text().splitlines()[0]
    assert first == "Knee flexion at the peak force: 70.0 deg  Target 70 deg, tolerance 5  On the target"


@pytest.mark.parametrize("at_peak, tol, state, off", [
    (77.5, 5.0, "warn", "7.5 deg above the target"),
    (62.0, 5.0, "warn", "8.0 deg below the target"),
    (75.0, 5.0, "good", "5.0 deg above the target"),             # exactly the tolerance is still good
    (75.2, 5.0, "warn", "5.2 deg above the target"),
    (65.0, 5.0, "good", "5.0 deg below the target"),
    (73.0, 2.0, "warn", "3.0 deg above the target"),             # the tolerance is the trial's own
    (73.0, 3.0, "good", "3.0 deg above the target"),
    (70.02, 5.0, "good", "On the target"),                       # less than 0.05 deg off reads as on the target
])
def test_the_headline_is_green_within_the_tolerance_and_amber_outside_it(trial, app, at_peak, tol, state, off):
    trial.meta["tolerance_deg"] = tol
    rows, fs, _ft, _fp = numbers(trial)
    panel = angle_panel(trial, fs=dict(fs, main_angle_at_peak_deg=at_peak))
    assert panel.headline_value.text() == f"{at_peak:.1f} deg" and panel.headline_off.text() == off
    assert panel._headline_state == state and panel.headline_target.text() == f"Target 70 deg, tolerance {tol:g}"
    assert colour_of(panel.headline_value) == strong_colour(STATE_COLOURS[state], QApplication.palette(), 0.8)


def test_the_headline_names_the_main_angle_from_the_rows_when_the_force_summary_has_none(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    panel = angle_panel(trial, fs=dict(fs, main_angle=""))
    assert panel.headline_caption.text() == "Knee flexion at the peak force"                       # the row's name stands in
    panel = angle_panel(trial, fs=dict(fs, main_angle="Knee extension"))
    assert panel.headline_caption.text() == "Knee extension at the peak force"


def test_the_picture_delay_is_in_the_headlines_tooltip(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    assert "after the force peak" not in angle_panel(trial, fs=fs).headline.toolTip()
    panel = angle_panel(trial, fs=dict(fs, picture_delay_s=0.04))
    assert "+0.040 s after the force peak" in panel.headline.toolTip()


def test_without_force_the_headline_is_the_hold_angle_and_the_facts_start_with_the_target(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    panel = angle_panel(trial, fs=None)
    assert panel.headline_caption.text() == "Knee flexion during the hold" and panel.headline_value.text() == "70.0 deg"
    assert "no peak force" in panel.headline.toolTip()
    assert [r[0] for r in panel.facts.rows] == ["Target", "Rest to hold", "Pictures lost", "In tolerance"]
    assert panel.facts.rows[0][:3] == ("Target", "70 deg", "tolerance 5 deg")
    with_force = angle_panel(trial, fs=fs)
    assert [r[0] for r in with_force.facts.rows][0] == "Hold angle"                              # with force the hold angle is a fact beside the headline
    # the hold angle off the target colours the headline the same way
    off = [dict(r, plateau_mean_deg=r["plateau_mean_deg"] + 8.0) if r["role"] == "main" else r for r in rows]
    panel = angle_panel(trial, rows=off, fs=None)
    assert panel.headline_value.text() == "78.0 deg" and panel._headline_state == "warn" and panel.headline_off.text() == "8.0 deg above the target"


def test_a_peak_with_no_angle_falls_back_to_the_hold_angle(trial, app):
    """The angle at the peak is NaN when no picture is near the peak: the headline then says it is the hold angle, not an empty '-'."""
    rows, fs, _ft, _fp = numbers(trial)
    panel = angle_panel(trial, fs=dict(fs, main_angle_at_peak_deg=math.nan))
    assert panel.headline_caption.text() == "Knee flexion during the hold" and panel.headline_value.text() == "70.0 deg"


def test_no_rows_and_no_main_row_do_not_break_the_panel(trial, app):
    panel = AnglePanel()
    panel.set_data(trial, [], None)                                                              # a trial without an angle
    assert panel.headline_caption.text() == "No angle" and panel.headline_value.text() == "-" and panel.headline_target.text() == "No target for this angle"
    assert panel.facts.rows == [] and panel.table.rowCount() == 0 and not panel.table.isVisibleTo(panel)
    assert "No angle: -" in panel.text()
    rows, fs, _ft, _fp = numbers(trial)
    neighbours = [dict(r, role="neighbour") for r in rows]                                       # nothing is marked main: the first row is
    panel = angle_panel(trial, rows=neighbours, fs=None)
    assert panel.headline_caption.text() == "Knee flexion during the hold" and panel.headline_value.text() == "70.0 deg"
    assert panel._main_row == [False, False] and "(main joint)" not in panel.text()


def test_the_facts_beside_the_headline(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    panel = angle_panel(trial, fs=fs)
    hold, shift, lost, tol = panel.facts.rows
    assert hold == ("Hold angle", "70.0 deg", "0.0 deg above the target 70", "good")
    assert shift[0] == "Rest to hold" and shift[1].endswith("deg") and shift[2].startswith("at rest ")
    assert lost == ("Pictures lost", f"0 of {rows[0]['frames']}", "", None)
    assert tol == ("In tolerance", "100 %", "of the pictures in the window", None)
    # a hold angle off the target, a few lost pictures and a missing rest
    main = dict(rows[0], plateau_mean_deg=77.0, frames=100, frames_lost=int(analysis.LOST_WARN_PCT), pct_in_tolerance=40.0,
                rest_mean_deg=math.nan, rest_to_hold_shift_deg=math.nan)
    panel = angle_panel(trial, rows=[main] + rows[1:], fs=fs)
    hold, shift, lost, tol = panel.facts.rows
    assert hold == ("Hold angle", "77.0 deg", "7.0 deg above the target 70", "warn") and shift[1:3] == ("-", "")
    assert lost == ("Pictures lost", "10 of 100", "no angle in these", None)                      # exactly the limit is not yet a warning ...
    assert tol[1] == "40 %"
    main["frames_lost"] = int(analysis.LOST_WARN_PCT) + 1
    panel = angle_panel(trial, rows=[main] + rows[1:], fs=fs)
    assert panel.facts.rows[2][3] == "warn"                                                      # ... one more is
    main["plateau_mean_deg"] = math.nan                                                          # no hold angle at all
    panel = angle_panel(trial, rows=[main] + rows[1:], fs=fs)
    assert panel.facts.rows[0][1] == "-" and panel.facts.rows[0][3] is None


@pytest.mark.xfail(strict=True, reason="BUG review_panels.py:619-620: a missing HOLD angle is annotated 'no target' although the target (70) is known and shown in the headline")
def test_a_missing_hold_angle_is_not_blamed_on_a_missing_target(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    gone = [dict(r, plateau_mean_deg=math.nan) if r["role"] == "main" else r for r in rows]          # fewer than the needed pictures in the middle of the window
    panel = angle_panel(trial, rows=gone, fs=fs)
    assert panel.headline_target.text() == "Target 70 deg, tolerance 5"                                   # the target is known ...
    assert panel.facts.rows[0][2] != "no target"                                                         # ... so the Hold angle row must not say it is not


def test_set_none_shows_a_quiet_message_instead_and_set_data_brings_the_numbers_back(trial, app):
    panel = AnglePanel()
    assert panel.text() == "No trial is open." and not panel.content.isVisibleTo(panel) and panel.empty.isVisibleTo(panel)
    rows, fs, _ft, _fp = numbers(trial)
    panel.set_data(trial, rows, fs, WINDOW)
    assert panel.content.isVisibleTo(panel) and not panel.empty.isVisibleTo(panel) and "Knee flexion at the peak force" in panel.text()
    panel.set_none("No angle data in this trial.")
    assert panel.text() == "No angle data in this trial." and not panel.content.isVisibleTo(panel) and panel._last is None
    panel.set_data(trial, rows, fs, WINDOW)                                                      # the same numbers again: the tables are filled again too
    assert panel.table.rowCount() == 2 and panel.table.item(0, 0).text() == "Knee flexion" and "Knee flexion at the peak force" in panel.text()


# ------------------------------------------------------------------------------------------------ AnglePanel: the tables
def test_the_angle_table_has_its_default_columns_and_the_main_joint_in_bold(trial, app):
    panel = angle_panel(trial)
    t = panel.table
    assert [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())] == ["Angle", "Hold", "Target", "Off target", "In tolerance", "Lost"]
    assert t.rowCount() == 2 and [t.item(r, 0).text() for r in range(2)] == ["Knee flexion", "Hip flex (+) / ext (-)"]
    assert [t.item(0, c).text() for c in range(1, 6)] == ["70.0", "70.0", "+0.0", "100 %", "0"]
    assert t.item(0, 0).font().bold() and not t.item(1, 0).font().bold() and "(the main joint)" in t.item(0, 0).toolTip()
    assert t.item(0, 1).textAlignment() & Qt.AlignRight and t.item(0, 0).textAlignment() & Qt.AlignLeft
    assert not panel.more_check.isChecked() and not panel.more_table.isVisibleTo(panel)           # the minor numbers are behind the switch
    text = panel.text()
    assert "Knee flexion (main joint): Hold 70.0 deg, Target 70.0 deg, Off target +0.0 deg, In tolerance 100 %, Lost 0" in text
    assert "More numbers" not in text and "Window 8.50 to 13.00 s" in text


def test_the_off_target_cell_is_green_within_the_tolerance_and_amber_outside_it(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    off = [dict(rows[0], plateau_mean_deg=78.0), dict(rows[1], plateau_mean_deg=math.nan)]
    panel = angle_panel(trial, rows=off)
    pal = QApplication.palette()
    assert panel.table.item(0, 3).text() == "+8.0" and panel.table.item(0, 3).foreground().color().name() == strong_colour(WARN, pal)
    assert panel.table.item(1, 1).text() == "-" and panel.table.item(1, 3).text() == "-"                              # a NaN is a dash, not 'nan'
    good = angle_panel(trial)
    assert good.table.item(0, 3).foreground().color().name() == strong_colour(GOOD, pal)
    assert "Hold -" in panel.text() and "Hold - deg" not in panel.text()                                              # no unit after a dash


def test_the_more_numbers_switch_adds_a_second_table_with_a_column_per_angle(trial, app):
    panel = angle_panel(trial)
    main_headers = [panel.table.horizontalHeaderItem(c).text() for c in range(panel.table.columnCount())]
    panel.more_check.setChecked(True)
    m = panel.more_table
    assert m.isVisibleTo(panel) and m.columnCount() == 1 + 2 and m.rowCount() == len(MORE_ROWS)
    assert [m.item(i, 0).text() for i in range(m.rowCount())] == MORE_ROWS
    assert [m.horizontalHeaderItem(k).text() for k in (1, 2)] == ["Knee flexion", "Hip flex (+) / ext (-)"]
    assert [panel.table.horizontalHeaderItem(c).text() for c in range(panel.table.columnCount())] == main_headers       # the main table keeps its columns
    assert m.item(0, 1).text() == "70.0" and m.item(1, 1).text() == "0.01"                                           # the mean, and the SD with two decimals
    assert m.item(7, 1).text() == "70.0" and m.item(8, 1).text() == "+0.0"                                           # the rest angle and the shift from rest to hold
    text = panel.text()
    assert "More numbers" in text and "Mean: Knee flexion 70.0 deg, Hip flex (+) / ext (-) 90.0 deg" in text
    assert "SD: Knee flexion 0.01 deg" in text and "Rest to hold: Knee flexion +0.0 deg" in text
    panel.more_check.setChecked(False)
    assert not m.isVisibleTo(panel) and "More numbers" not in panel.text()


def test_everything_has_the_more_numbers_even_with_the_switch_off(trial, app):
    panel = angle_panel(trial)
    assert not panel.more_check.isChecked()
    everything = panel.text(everything=True)
    assert "More numbers" in everything and "Median: Knee flexion 70.0 deg" in everything and "Start to end: Knee flexion +0.0 deg" in everything
    assert "More numbers" not in panel.text() and not panel.more_table.isVisibleTo(panel)               # (and the switch is still off)


def test_the_text_has_the_numbers_with_their_units_and_the_window_and_the_note(trial, app):
    panel = angle_panel(trial)
    text = panel.text(everything=True)
    assert "Hold angle: 70.0 deg  0.0 deg above the target 70" in text and "In tolerance: 100 %  of the pictures in the window" in text
    assert "Pictures lost: 0 of" in text and "Window 8.50 to 13.00 s" in text
    assert "Hold is the angle during the contraction (the middle 60 % of the window)" in text and "No MVIC start/end markers in this trial" in text
    trial.events = [(0, 9.0, "MVIC start"), (1, 12.0, "MVIC end")]                                      # markers: the numbers are not 'as set'
    assert "No MVIC start/end markers" not in angle_panel(trial).text()


def test_set_data_twice_with_the_same_or_fewer_rows_leaves_no_leftovers(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    panel = AnglePanel()
    panel.set_data(trial, rows, fs, WINDOW)
    first, children = panel.text(everything=True), len(panel.findChildren(QWidget))
    panel.set_data(trial, rows, fs, WINDOW)
    assert panel.text(everything=True) == first and len(panel.findChildren(QWidget)) == children
    panel.set_data(trial, rows[:1], fs, WINDOW)                                                         # the hip row is gone from both tables
    assert panel.table.rowCount() == 1 and panel.more_table.columnCount() == 2 and "Hip flex" not in panel.text(everything=True)
    panel.set_data(trial, rows, fs, WINDOW)
    assert panel.table.rowCount() == 2 and panel.more_table.columnCount() == 3


# ------------------------------------------------------------------------------------------------ AnglePanel: the reference block
def test_without_a_reference_reading_the_block_says_how_to_add_one(trial, app):
    panel = angle_panel(trial)
    ref = panel.reference
    assert ref.n_rows == 0 and not ref.grid_host.isVisibleTo(ref) and ref.notice.isVisibleTo(ref)
    assert ref.notice.text().startswith("No reading for this trial. Type it in the Live tab") and ref.button.text() == "Add the reference readings..."
    assert not ref.basis.isVisibleTo(ref)
    assert ref.text().splitlines() == ["Reference readings", "Typed from a goniometer or an inclinometer, not measured by this program.", ref.notice.text(),
                                       "[Add the reference readings...]"]


def test_a_reference_reading_shows_the_difference_to_the_app_and_the_instrument(trial, app):
    trial.meta["reference_readings"] = {"instrument": "goniometer", "resolution_deg": 1.0,
                                        "readings": {"knee": {"values_deg": [68.0, 69.0], "mean_deg": 68.5}}}
    trial._reference_rows = None
    panel = angle_panel(trial)
    ref = panel.reference
    app_deg = trial.reference_rows()[0]["app_deg"]
    assert ref.n_rows == 1 and ref.grid_host.isVisibleTo(ref) and not ref.notice.isVisibleTo(ref) and ref.button.text() == "Add or change the reference readings..."
    assert abs(app_deg - 70.0) < 0.1
    text = ref.text()
    assert f"Knee flexion: reference 68.5 deg (mean of 2), app {app_deg:.1f} deg, difference +{app_deg - 68.5:.1f} deg  Goniometer (read to 1 deg), recorded with the trial" in text
    assert "No moment was kept" in ref.basis.text() and "first second of the recording" in ref.basis.text()          # no stamp: said so
    assert "error of its own of a few degrees" in ref.basis.text()
    assert "reference 68.5 deg" in panel.text() and "[Add or change the reference readings...]" in panel.text()
    cells = [w.text() for w in ref.line_widgets if w.objectName() == "refCell"]
    assert cells[0] == "Knee flexion" and cells[1] == "68.5 deg (mean of 2)" and cells[3].startswith("+")
    trial.meta["reference_readings"]["stamped_at"] = "2026-10-30T10:00:00"                               # a moment kept: no warning about the first second
    trial._reference_rows = None
    panel = angle_panel(trial)
    assert "Moment of the reading: 2026-10-30T10:00:00." in panel.reference.basis.text() and "No moment was kept" not in panel.reference.basis.text()


def test_readings_with_different_instruments_each_name_theirs(trial, app):
    trial.meta["reference_readings"] = {"instrument": "goniometer", "resolution_deg": 1.0, "readings": {"knee": {"values_deg": [68.0], "mean_deg": 68.0}}}
    trial.reference_side = {"instrument": "inclinometer", "resolution_deg": 0.5, "readings": {"hip": {"values_deg": [88.0], "mean_deg": 88.0}}}
    trial._reference_rows = None
    panel = angle_panel(trial)
    ref = panel.reference
    assert ref.n_rows == 2
    lines = ref._lines
    assert "Goniometer (read to 1 deg), recorded with the trial" in lines[0] and "Digital inclinometer (read to 0.5 deg), added later in Review" in lines[1]
    assert len([w for w in ref.line_widgets if w.objectName() == "refInstrument"]) == 2                  # one instrument line under each reading
    one = {**trial.meta["reference_readings"], "readings": {"knee": {"values_deg": [68.0], "mean_deg": 68.0}, "hip": {"values_deg": [88.0], "mean_deg": 88.0}}}
    trial.meta["reference_readings"], trial.reference_side, trial._reference_rows = one, {}, None
    panel = angle_panel(trial)
    ref = panel.reference
    assert len([w for w in ref.line_widgets if w.objectName() == "refInstrument"]) == 1                  # the same instrument is said once, under them


def test_readings_typed_for_another_position_are_said_to_be_not_saved(trial, app):
    trial.meta["reference_readings"] = {"not_used": {"entered_for": {"test_key": "knee_extension", "target_deg": 60, "participant": "P07"}}}
    trial._reference_rows = None
    panel = angle_panel(trial)
    ref = panel.reference
    assert ref.notice.text().startswith("Readings typed for knee_extension 60 (participant P07) were NOT saved with this trial")
    assert "No reading for this trial" in ref.notice.text()                                               # and it says there is none


def test_the_edit_button_asks_for_the_reference_dialog(trial, app):
    panel = angle_panel(trial)
    asked = []
    panel.editReferenceRequested.connect(lambda: asked.append(1))
    panel.reference.button.click()
    assert asked == [1]
    trial.meta["reference_readings"] = {"instrument": "goniometer", "resolution_deg": 1.0, "readings": {"knee": {"values_deg": [68.0], "mean_deg": 68.0}}}
    trial._reference_rows = None
    panel = angle_panel(trial)
    panel.editReferenceRequested.connect(lambda: asked.append(2))
    panel.reference.button.click()
    assert asked == [1, 2]


def test_setting_the_reference_again_replaces_the_lines_instead_of_adding(trial, app):
    trial.meta["reference_readings"] = {"instrument": "goniometer", "resolution_deg": 1.0, "readings": {"knee": {"values_deg": [68.0], "mean_deg": 68.0}}}
    trial._reference_rows = None
    panel = angle_panel(trial)
    n = len(panel.reference.line_widgets)
    rows, fs, _ft, _fp = numbers(trial)
    panel.set_data(trial, rows, fs, WINDOW)
    panel.set_data(trial, rows, fs, WINDOW)
    assert len(panel.reference.line_widgets) == n and len(panel.reference._lines) == 1
    trial.meta.pop("reference_readings")
    trial._reference_rows = None
    panel.set_data(trial, rows, fs, WINDOW)
    assert panel.reference.line_widgets == [] and panel.reference.n_rows == 0


# ------------------------------------------------------------------------------------------------ AnglePanel: looks
def test_the_angle_panel_has_no_scroll_area_and_never_a_scroll_bar(trial, app):
    panel = angle_panel(trial)
    assert not panel.findChildren(QScrollArea)
    for table in (panel.table, panel.more_table):
        assert table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff and table.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    show(panel, 700)
    panel.more_check.setChecked(True)
    QApplication.processEvents()
    for table in (panel.table, panel.more_table):                     # every row is in view: the table is as tall as its rows
        assert table.height() >= table.horizontalHeader().height() + table.rowCount() * table.verticalHeader().defaultSectionSize()
    panel.close()


def test_the_headline_goes_above_the_facts_when_the_panel_is_narrow(trial, app):
    """The handler is called with the sizes themselves: on a machine whose fonts need more room the panel cannot really be made 400 px wide."""
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent
    from PySide6.QtWidgets import QBoxLayout
    panel = show(angle_panel(trial), 900)
    assert panel.top.direction() == QBoxLayout.LeftToRight
    panel.resizeEvent(QResizeEvent(QSize(400, 700), QSize(900, 700)))
    assert panel.top.direction() == QBoxLayout.TopToBottom
    panel.resizeEvent(QResizeEvent(QSize(430, 700), QSize(400, 700)))
    assert panel.top.direction() == QBoxLayout.LeftToRight
    panel.close()


def test_light_and_dark_palettes_do_not_break_the_panels_and_the_colours_follow(trial, palette):
    rows, fs, _ft, _fp = numbers(trial)
    off = [dict(r, plateau_mean_deg=78.0) if r["role"] == "main" else r for r in rows]
    amber = angle_panel(trial, rows=off, fs=dict(fs, main_angle_at_peak_deg=80.0))
    force = force_panel(trial)
    before_text, before_force = amber.text(everything=True), force.text()
    before_colour, before_cell = colour_of(amber.headline_value), amber.table.item(0, 3).foreground().color().name()
    dark(palette)
    QApplication.processEvents()
    assert amber.text(everything=True) == before_text and force.text() == before_force                  # the numbers are not affected
    after_colour = colour_of(amber.headline_value)
    assert after_colour != before_colour and after_colour == strong_colour(WARN, QApplication.palette(), 0.8)    # the amber is lighter on a dark window
    assert luminance(QColor(after_colour)) > luminance(QColor(before_colour))
    cell = amber.table.item(0, 3)
    assert cell.text() == "+8.0" and cell.foreground().color().name() == strong_colour(WARN, QApplication.palette()) != before_cell      # the table follows too
    fresh = angle_panel(trial, rows=off, fs=dict(fs, main_angle_at_peak_deg=80.0))                      # a panel made in the dark theme reads the same
    assert fresh.text(everything=True) == before_text and colour_of(fresh.headline_value) == after_colour
    assert force_panel(trial, width=500).text() == before_force


def test_long_texts_never_make_the_angle_panel_wider(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    short = angle_panel(trial)
    before = short.minimumSizeHint().width()
    name = "A joint angle with a name that goes on and on and on " * 6
    long_rows = [dict(r, angle=name + r["angle"]) for r in rows]
    panel = angle_panel(trial, rows=long_rows, fs=dict(fs, main_angle=name))
    cap = 300 + 12 + 100 + 16                                       # the headline card (at most 300 px) beside the facts (at least 100 px), and the margins
    assert panel.minimumSizeHint().width() <= max(before, cap)
    assert panel.headline.width() <= 300 and panel.headline.minimumWidth() == panel.headline.maximumWidth() <= 300
    from PySide6.QtWidgets import QLabel
    assert QLabel.text(panel.headline_caption) != name and panel.headline_caption.text().startswith(name)   # shown cut, the whole text kept
    show(panel, 450)
    for w in (panel.headline_caption, panel.headline_target, panel.headline_off):
        assert w.geometry().right() <= panel.headline.width()
    panel.close()


# ------------------------------------------------------------------------------------------------ ForcePanel: the sections
def test_the_force_panel_has_five_sections_with_the_numbers_of_the_window(trial, app):
    rows, fs, ft, fp = numbers(trial)
    panel = force_panel(trial)
    assert list(panel.sections) == ["Peak", "Level", "Timing", "Torque", "Load cell"] and panel._shown == list(panel.sections.values())
    assert panel.caption.text() == "Force in the window: N, 100 ms moving average"
    text = panel.text()
    peak = dict(panel.sections["Peak"].grid.rows[i][:2] for i in range(len(panel.sections["Peak"].grid.rows)))
    assert list(peak) == ["Peak", "Peak (20 ms)", "Sliding-window peak", "Raw single sample", "Angle at the peak"]
    assert peak["Peak"] == f"{fs['peak']:.1f} N" and peak["Peak (20 ms)"] == f"{fs['peak_20ms']:.1f} N"
    assert peak["Sliding-window peak"] == f"{fp['window_peak']:.1f} N" and peak["Raw single sample"] == f"{fs['raw_peak']:.1f} N"
    assert peak["Angle at the peak"] == "70.0 deg"
    grid = panel.sections["Peak"].grid
    assert grid.rows[0][2] == f"at {fs['t_peak_s']:.2f} s (100 ms moving average)"
    assert "highest 20 ms average of the raw force" in grid.rows[1][2]
    assert grid.rows[2][2] == f"highest mean over 500 ms of the raw samples, {fp['window_peak_start_s']:.2f} to {fp['window_peak_end_s']:.2f} s"
    assert grid.rows[4][2] == "Knee flexion"
    assert text.splitlines()[0] == panel.caption.text() and "Warning" not in text
    for title in ("Peak", "Level", "Timing", "Torque", "Load cell"):
        assert title in text.splitlines()


def test_the_raw_single_sample_is_shown_only_when_there_is_smoothing(trial, app):
    trial.force_smooth_s = 0.0
    rows, fs, _ft, _fp = numbers(trial)
    panel = force_panel(trial)
    assert fs["smoothing_s"] == 0 and panel.caption.text() == "Force in the window: N, raw samples"
    assert [r[0] for r in panel.sections["Peak"].grid.rows] == ["Peak", "Peak (20 ms)", "Sliding-window peak", "Angle at the peak"]
    assert panel.sections["Peak"].grid.rows[0][2].endswith("(raw samples)")
    trial.force_smooth_s = 0.1
    assert "Raw single sample" in force_panel(trial).text()


def test_a_window_shorter_than_the_sliding_window_says_so_instead_of_a_number(trial, app):
    window = (10.0, 10.3)                                             # 300 ms against a 500 ms sliding window
    rows, fs, ft, fp = numbers(trial, window)
    assert math.isnan(fp["window_peak"])
    panel = force_panel(trial, window=window)
    sliding = next(r for r in panel.sections["Peak"].grid.rows if r[0] == "Sliding-window peak")
    assert sliding == ("Sliding-window peak", "-", "window shorter than 500 ms, or gaps in the force", "warn")
    assert colour_of(panel.sections["Peak"].grid._values[2]) == strong_colour(WARN, QApplication.palette())
    trial.window_peak_s = 0.2                                         # a shorter sliding window fits
    assert force_panel(trial, window=window).sections["Peak"].grid.rows[2][1].endswith(" N")


def test_the_level_section_has_the_rest_the_rise_and_the_plateau(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    grid = force_panel(trial).sections["Level"].grid
    assert [r[0] for r in grid.rows] == ["Rest before", "Peak above rest", "Plateau mean"]
    assert grid.rows[0] == ("Rest before", f"{fs['rest_mean']:.1f} N", f"{fs['rest_basis']}, {fs['rest_seconds']:.1f} s of force", None)
    assert grid.rows[1][1] == f"{fs['peak_minus_rest']:.1f} N" and grid.rows[2] == ("Plateau mean", f"{fs['plateau_mean']:.1f} N", "middle 60 % of the window", None)
    shaky = force_panel(trial, fs=dict(fs, rest_note="The force was not steady before the push."))        # a doubtful rest is amber
    assert shaky.sections["Level"].grid.rows[0][3] == "warn"


def test_zero_2_rows_appear_only_when_there_is_a_relaxed_level_or_a_reason_for_none(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    assert [r[0] for r in force_panel(trial).sections["Level"].grid.rows] == ["Rest before", "Peak above rest", "Plateau mean"]        # none, no word about it
    relaxed = dict(fs, relaxed=5.0, peak_minus_relaxed=fs["peak"] - 5.0, rest_minus_relaxed=fs["rest_mean"] - 5.0)
    grid = force_panel(trial, fs=relaxed).sections["Level"].grid
    assert [r[0] for r in grid.rows][3:] == ["Relaxed (Zero 2)", "Peak above relaxed"]
    assert grid.rows[3][1] == "5.0 N" and "the rest before the push was 5.0 N lower than that" in grid.rows[3][2] and grid.rows[4][1] == f"{fs['peak'] - 5.0:.1f} N"
    higher = force_panel(trial, fs=dict(relaxed, rest_minus_relaxed=12.0)).sections["Level"].grid.rows[3][2]
    assert "the rest before the push was 12.0 N higher than that" in higher
    unused = force_panel(trial, fs=dict(fs, relaxed_note="A relaxed level was taken for another position (60), so it is not used for this trial.")).sections["Level"].grid
    assert unused.rows[3] == ("Relaxed (Zero 2)", "not used", "A relaxed level was taken for another position (60), so it is not used for this trial.", None)


def test_the_timing_section_and_its_missing_push(trial, app):
    rows, fs, ft, fp = numbers(trial)
    grid = force_panel(trial).sections["Timing"].grid
    assert [r[0] for r in grid.rows] == ["Onset", "Time to peak", "10 to 90 % of the rise", "Steepest rise", "Held above 90 %", "Steadiness"]
    assert grid.rows[0][1] == f"{ft['onset_s']:.2f} s" and grid.rows[1][1] == f"{ft['time_to_peak_s']:.2f} s"
    assert grid.rows[1][2] == f"from the onset to the peak; to the middle of the sliding-window peak {fp['time_to_window_peak_s']:.2f} s"
    assert grid.rows[3][1] == f"{ft['rise_rate_per_s']:.1f} N/s" and grid.rows[4][1] == f"{ft['hold_above_90pct_s']:.2f} s" and grid.rows[5][1].endswith("%")
    no_window_peak = force_panel(trial, fp=dict(fp, time_to_window_peak_s=math.nan)).sections["Timing"].grid.rows[1][2]
    assert no_window_peak == "from the onset to the peak"
    none = force_panel(trial, ft=None).sections["Timing"].grid                                          # no clear push
    assert none.rows == [("Timing", "-", "no clear push in the window", "warn")]
    redone = force_panel(trial, ft=None, fp=None)                                                       # without fp the panel works both out itself
    assert redone.sections["Timing"].grid.rows[0][0] == "Onset"


def test_the_torque_section_with_a_moment_arm_and_with_a_stand_in_dot(trial, app):
    tq = analysis.torque_summary(trial, *WINDOW)
    grid = force_panel(trial).sections["Torque"].grid
    assert [r[0] for r in grid.rows] == ["Peak", "At the peak force", "Plateau", "Moment arm"]
    assert grid.rows[0] == ("Peak", f"{tq['peak_nm']:.1f} N.m", f"at {tq['t_peak_s']:.2f} s", None)
    assert grid.rows[3] == ("Moment arm", "30.0 cm", "at the peak force, mean 30.0 cm", None)
    del trial.tracks["strap"]                                                                           # the strap dot was never seen: the ankle dot stands in
    stand_in = force_panel(trial).sections["Torque"].grid
    last = stand_in.rows[-1]
    assert last[0] == "Stand-in dot" and last[3] == "warn" and "the strap dot was not seen in" in last[2] and "the ankle dot stood in for it" in last[2]
    assert last[1].startswith(str(analysis.torque_summary(trial, *WINDOW)["pictures"]) + " of ")


@pytest.mark.parametrize("change, reason", [
    ("uncalibrated", "the force is not calibrated"),
    ("no_ruler", "there is no ruler scale"),
    ("no_anchor", "the anchor dot was not used"),
    ("no_arm", "no moment arm could be measured"),
])
def test_torque_is_not_available_and_says_why(trial, app, change, reason):
    window = WINDOW
    if change == "uncalibrated":
        trial.force_n = np.full_like(trial.force_n, np.nan)
    elif change == "no_ruler":
        trial.px_per_cm = None
    elif change == "no_anchor":
        del trial.tracks["anchor"]
    else:
        window = (10.0, 10.05)                                    # no picture in the window, so no moment arm
    panel = force_panel(trial, window=window)
    assert panel.sections["Torque"].grid.rows == [("Torque", "not available", reason, None)]
    assert "Torque: not available  " + reason in panel.text()


def test_the_load_cell_section_says_which_calibration_and_that_there_is_no_precision_without_a_regression(trial, app):
    grid = force_panel(trial).sections["Load cell"].grid
    assert grid.rows[0] == ("Calibration", "recorded", trial.force_cal_note, None)
    assert grid.rows[1][:2] == ("Regression", "none stored") and "N per volt: no precision can be given" in grid.rows[1][2]
    assert len(grid.rows) == 2 and not any("plus or minus" in r[0] for r in grid.rows)
    trial.force_cal = None                                                                              # nothing recorded about the calibration
    assert "unknown calibration, 0.00 N per volt" in force_panel(trial).sections["Load cell"].grid.rows[1][2]


def regression(top_kg=100.0):
    """A load cell regression of 222.2 N per volt (the slope of torque_trial) with a little scatter, calibrated up to `top_kg`."""
    kg = np.linspace(0.0, top_kg, 6)
    noise = np.array([0.001, -0.001, 0.0005, -0.0005, 0.0008, -0.0008])
    return fit_points(kg, 0.1 + kg * GRAVITY / (200.0 / 0.9) + noise, id="R-test", date="2026-10-30")


def test_the_load_cell_section_gives_the_precision_with_a_plus_or_minus(trial, app):
    reg = regression(100.0)
    trial.regression, trial.force_cal = reg, {"n_per_v": 200.0 / 0.9, "offset_v": 0.1, "method": "test"}
    rows, fs, _ft, _fp = numbers(trial)
    grid = force_panel(trial).sections["Load cell"].grid
    assert grid.rows[1] == ("Regression", "R-test", f"{200.0 / 0.9:.2f} N per volt, zero +0.1000 V", None)
    labels = [r[0] for r in grid.rows]
    assert labels == ["Calibration", "Regression", "Peak plus or minus", "Plateau mean plus or minus"]
    for row, key in ((grid.rows[2], "peak"), (grid.rows[3], "plateau_mean")):
        f = fs[key]
        kg = f / GRAVITY
        sd = COVERAGE * math.sqrt(reg.rmse_kg ** 2 + (kg * reg.slope_rel_se) ** 2) * GRAVITY             # 2 standard deviations of one reading at this load
        assert row[1] == f"{sd:.1f} N" and row[2] == f"{100 * sd / abs(f):.2f} % at about 95 % ({kg:.1f} kg)" and row[3] is None
    assert "Peak plus or minus: " in force_panel(trial).text()


def test_a_load_above_the_heaviest_calibration_load_is_called_out(trial, app):
    reg = regression(15.0)                                                                              # 15 kg is 147 N: the 200 N peak is beyond the line
    trial.regression, trial.force_cal = reg, {"n_per_v": 200.0 / 0.9, "offset_v": 0.1, "method": "test"}
    rows, fs, _ft, _fp = numbers(trial)
    grid = force_panel(trial).sections["Load cell"].grid
    above = [r for r in grid.rows if r[0].endswith("above the line")]
    assert [r[0] for r in above] == ["Peak above the line", "Plateau mean above the line"]
    assert above[0][1] == f"{fs['peak'] / GRAVITY:.0f} kg" and above[0][3] == "warn"
    assert above[0][2] == "the heaviest calibration load is 15 kg: the line is extended there, not measured"
    # a NaN plateau is left out, not shown as 'nan'
    nan_plateau = force_panel(trial, fs=dict(fs, plateau_mean=math.nan)).sections["Load cell"].grid
    assert not any(r[0].startswith("Plateau mean") for r in nan_plateau.rows)


def test_the_calibration_button_asks_for_the_calibration_graph(trial, app):
    panel = force_panel(trial)
    asked = []
    panel.calibrationRequested.connect(lambda: asked.append(1))
    assert panel.button_calibration.text() == "Show the calibration graph" and panel.sections["Load cell"].extra is panel.button_calibration
    panel.button_calibration.click()
    assert asked == [1] and "[Show the calibration graph]" in panel.text()


# ------------------------------------------------------------------------------------------------ ForcePanel: warnings
def test_no_warning_without_a_reason(trial, app):
    panel = force_panel(trial)
    assert panel.warnings == [] and "Warning" not in panel.text()
    rows, fs, _ft, _fp = numbers(trial)
    at_limit = force_panel(trial, fs=dict(fs, mains_hum_v=0.027, mains_hum_pct_of_peak=analysis.HUM_WARN_PCT))        # exactly the limit is not over it
    assert at_limit.warnings == []


def test_a_50_hz_hum_over_the_limit_is_a_warning_with_its_numbers(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    panel = force_panel(trial, fs=dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0))
    assert len(panel.warnings) == 1 and panel.warnings[0].text.startswith("50 Hz mains interference: 90 mV = 10 % of the rise above rest ")
    assert "(check shielding and grounding; a smoothing window that is a multiple of 20 ms removes it)." in panel.warnings[0].text
    assert panel.text().splitlines()[1] == "Warning: " + panel.warnings[0].text                           # the whole sentence is in the text, not the cut one


def test_a_hum_in_the_recorded_voltage_is_found_by_the_real_numbers_too(trial, app):
    t = trial.force_t
    trial.force_v = trial.force_v + 0.09 * np.sin(2 * np.pi * 50.0 * t)                                   # the raw voltage hums by 90 mV
    rows, fs, _ft, _fp = numbers(trial)
    assert 8 < fs["mains_hum_pct_of_peak"] < 12
    panel = force_panel(trial)
    assert len(panel.warnings) == 1 and panel.warnings[0].text.startswith("50 Hz mains interference: 90 mV")


def test_a_rest_note_and_a_zero_2_mismatch_are_warnings_in_that_order(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    note = "The force was not steady before the push (it varied by 12 N, SD): the strap was loaded or moving, so the resting force is doubtful."
    bad = dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0, rest_note=note, relaxed=100.0, peak_minus_relaxed=fs["peak"] - 100.0,
               rest_minus_relaxed=fs["rest_mean"] - 100.0)
    panel = force_panel(trial, fs=bad)
    mismatch = analysis.relaxed_mismatch("N", 100.0, fs["rest_mean"], fs["peak_minus_rest"])
    assert mismatch.startswith("Zero 2 (relaxed) was 100 N, but the force just before this push was 0 N (-100 N)")
    assert [w.text for w in panel.warnings][1:] == [note, mismatch] and panel.warnings[0].text.startswith("50 Hz")
    lines = panel.text().splitlines()
    assert lines[1:4] == ["Warning: " + w.text for w in panel.warnings]
    only_mismatch = force_panel(trial, fs=dict(bad, mains_hum_pct_of_peak=0.1, rest_note=""))
    assert [w.text for w in only_mismatch.warnings] == [mismatch]
    close = force_panel(trial, fs=dict(fs, relaxed=fs["rest_mean"] + 10.0))                              # 10 N apart is within the limit
    assert close.warnings == []


def test_warnings_are_kept_when_they_do_not_change_and_cleared_when_they_go(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    hum = dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0)
    panel = force_panel(trial, fs=hum)
    first = panel.warnings[0]
    panel.set_data(trial, *WINDOW, hum, *numbers(trial)[2:], 0.0)
    assert panel.warnings[0] is first                                                                    # the same warning: not made again (it may be open)
    panel.set_data(trial, *WINDOW, fs, *numbers(trial)[2:], 0.0)
    assert panel.warnings == [] and "Warning" not in panel.text()
    panel.set_data(trial, *WINDOW, hum, *numbers(trial)[2:], 0.0)
    assert len(panel.warnings) == 1
    panel.set_none("No force was recorded in this trial.")
    assert panel.warnings == []


def test_a_warning_is_one_line_until_it_is_clicked_and_then_all_of_it(trial, app):
    rows, fs, _ft, _fp = numbers(trial)
    mismatch_fs = dict(fs, relaxed=100.0, peak_minus_relaxed=100.0, rest_minus_relaxed=-100.0)
    panel = show(force_panel(trial, fs=mismatch_fs), 400)
    w = panel.warnings[0]
    height = panel.heightForWidth(400)
    assert not w.expanded and w.height() == w.heightForWidth(400) and "…" in w.label.text() and w.toolTip() == w.text
    QTest.mouseClick(w, Qt.LeftButton)
    QApplication.processEvents()
    assert w.expanded and w.height() > w.heightForWidth(400) // 2 and w.label.text() == w.text and panel.heightForWidth(400) > height
    assert all(s.geometry().top() >= w.geometry().bottom() for s in panel._shown)                          # the sections moved down: nothing under the warning
    QTest.mouseClick(w, Qt.LeftButton)
    QApplication.processEvents()
    assert not w.expanded and panel.heightForWidth(400) == height
    panel.close()


# ------------------------------------------------------------------------------------------------ ForcePanel: none, units
def test_set_none_shows_the_reason_and_hides_everything_else(trial, app):
    panel = ForcePanel()
    assert panel.text() == "No trial is open." and panel._shown == [] and all(s.isHidden() for s in panel.sections.values())
    rows, fs, ft, fp = numbers(trial)
    panel.set_data(trial, *WINDOW, fs, ft, fp, 0.0)
    assert panel.text().startswith("Force in the window") and all(not s.isHidden() for s in panel.sections.values()) and not panel.caption.isHidden()
    panel.set_none("No force was recorded in this trial.")
    assert panel.text() == "No force was recorded in this trial." and panel._shown == [] and panel.caption.isHidden()
    assert all(s.isHidden() for s in panel.sections.values()) and panel._last is None
    panel.set_data(trial, *WINDOW, None, None, None, 0.0)                                                 # a window without force samples
    assert panel.text() == "Force was recorded but there are no samples in this window." and all(s.isHidden() for s in panel.sections.values())
    panel.set_data(trial, *WINDOW, fs, ft, fp, 0.0)
    assert panel.text().startswith("Force in the window") and len(panel._shown) == 5


def test_a_force_in_volts_says_so_and_shows_three_decimals(trial, app):
    trial.force_n = np.full_like(trial.force_n, np.nan)                                                   # recorded before any calibration
    rows, fs, ft, fp = numbers(trial)
    assert fs["unit"] == "V"
    panel = force_panel(trial)
    assert panel.caption.text() == "Force in the window: V, not calibrated, 100 ms moving average"
    peak = panel.sections["Peak"].grid.rows
    assert peak[0][1] == f"{fs['peak']:.3f} V" and peak[1][1] == f"{fs['peak_20ms']:.3f} V" and peak[2][1] == f"{fp['window_peak']:.3f} V"
    assert re.fullmatch(r"\d\.\d{3} V", peak[0][1])
    assert panel.sections["Timing"].grid.rows[3][1] == f"{ft['rise_rate_per_s']:.3f} V/s"
    load = panel.sections["Load cell"].grid.rows
    assert load[0][:2] == ("Calibration", "none") and load[0][3] == "warn"
    assert load[1] == ("Newtons", "not yet", "no zero was taken: use Apply zero to get newtons", None) and len(load) == 2
    assert panel.sections["Torque"].grid.rows[0][1:3] == ("not available", "the force is not calibrated")


def test_a_force_in_newtons_shows_one_decimal(trial, app):
    panel = force_panel(trial)
    assert re.fullmatch(r"\d+\.\d N", panel.sections["Peak"].grid.rows[0][1]) and "not calibrated" not in panel.caption.text()
    assert panel.sections["Timing"].grid.rows[3][1].endswith(" N/s")


# ------------------------------------------------------------------------------------------------ ForcePanel: the flow
def drawn_columns(panel):
    return len({s.geometry().x() for s in panel._shown})


@pytest.mark.parametrize("width", [260, 460, 700, 1000, 1500])
def test_the_sections_flow_into_the_columns_the_width_allows_without_overlap_or_cutting(trial, app, width):
    panel = show(force_panel(trial), width)
    cols = panel.columns_for(width)
    cap = max(1, min(4, 5, (width - 2 * 8 + 8) // (214 + 8)))                                            # as many as fit at 214 px each, and no more than four
    assert 1 <= cols <= cap and drawn_columns(panel) == cols
    shown = panel._shown
    geoms = [s.geometry() for s in shown]
    assert all(g.left() >= 0 and g.right() < width and g.width() >= ForcePanel.SECTION_MIN_WIDTH for g in geoms)       # inside the panel, none squeezed below its minimum
    assert not any(a.intersects(b) for i, a in enumerate(geoms) for b in geoms[i + 1:])
    by_place = sorted(shown, key=lambda s: (s.geometry().x(), s.geometry().y()))
    assert by_place == shown                                                                             # the sections keep their order: down a column, then the next
    assert max(g.bottom() for g in geoms) + 1 <= panel.heightForWidth(width)
    panel.close()


def test_the_columns_are_one_to_four_by_width_and_the_sections_keep_their_order(trial, app, monkeypatch):
    """Heights of the five sections are set here (100, 200, 200, 100, 100 px), so which split is best does not depend on this machine's fonts:
    the columns are as many as fit at 214 px each (at most four) and as the split pays off (the tallest column gets shorter)."""
    panel = force_panel(trial)
    for section, h in zip(panel._shown, (100, 200, 200, 100, 100)):
        monkeypatch.setattr(section, "heightForWidth", lambda w, h=h: h, raising=False)
    assert [panel.columns_for(w) for w in (300, 451, 500, 673, 700, 895, 1000, 2400)] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert panel._plan(1000)[1] == [[0], [1], [2], [3, 4]] and panel._plan(300)[1] == [[0, 1, 2, 3, 4]]
    for w in (300, 500, 700, 1000):
        groups = panel._plan(w)[1]
        assert [i for g in groups for i in g] == [0, 1, 2, 3, 4]                                                     # in order, none twice, none missing
    tall = panel._plan(1000)
    assert tall[2] == [100, 200, 200, 208]                                                                           # the height of each column (with a gap between sections)


def test_more_width_gives_more_columns_and_less_height(trial, app):
    panel = force_panel(trial)
    widths = [300, 500, 700, 1400]
    columns = [panel.columns_for(w) for w in widths]
    heights = [panel.heightForWidth(w) for w in widths]
    assert columns[0] == 1 and columns[1] == 2 and columns == sorted(columns) and columns[-1] >= 3
    assert heights == sorted(heights, reverse=True) and heights[0] > heights[-1]
    assert panel.columns_for(1) == 1 and panel.columns_for(0) == 1                                       # a width that is no width at all
    assert panel.sizeHint().height() == panel.heightForWidth(panel.sizeHint().width())


def test_the_force_panel_has_no_scroll_area_and_the_height_follows_the_width(trial, app):
    panel = show(force_panel(trial), 1000)
    assert not panel.findChildren(QScrollArea) and not panel.findChildren(QAbstractScrollArea) and panel.hasHeightForWidth()
    wide = panel.heightForWidth(1000)
    panel.resize(300, 700)
    QApplication.processEvents()
    assert panel.heightForWidth(300) > wide and drawn_columns(panel) == 1
    bottom = max(s.geometry().bottom() for s in panel._shown)
    assert bottom + 1 <= panel.heightForWidth(300)                                                       # everything is in the height the panel asks for
    panel.close()


def test_the_panel_re_flows_in_a_layout_when_the_window_is_resized(trial, app):
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    panel = force_panel(trial)
    lay.addWidget(panel)
    show(host, 1000)
    assert panel.width() == 1000 and drawn_columns(panel) == panel.columns_for(1000) >= 3
    host.resize(500, 800)
    QApplication.processEvents()
    assert panel.width() == 500 and drawn_columns(panel) == panel.columns_for(500) == 2
    host.close()


def test_calling_set_data_again_changes_nothing_that_can_be_seen(trial, app):
    rows, fs, ft, fp = numbers(trial)
    panel = show(force_panel(trial), 700)
    first, children = panel.text(), len(panel.findChildren(QWidget))
    geoms = [s.geometry() for s in panel._shown]
    panel.set_data(trial, *WINDOW, fs, ft, fp, 0.0)
    panel.set_data(trial, *WINDOW, fs, ft, fp, 0.0)
    QApplication.processEvents()
    assert panel.text() == first and len(panel.findChildren(QWidget)) == children and [s.geometry() for s in panel._shown] == geoms
    panel.close()


def test_light_and_dark_palettes_do_not_break_the_force_panel(trial, palette):
    rows, fs, ft, fp = numbers(trial)
    hum = dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0)
    panel = show(force_panel(trial, fs=hum), 800)
    css, text = panel.styleSheet(), panel.text()
    dark(palette)
    QApplication.processEvents()
    assert panel.text() == text and panel.styleSheet() != css and drawn_columns(panel) == panel.columns_for(800)       # the sheet follows the palette
    fresh = show(force_panel(trial, fs=hum), 800)
    assert fresh.text() == text
    fresh.close()
    panel.close()


def test_long_texts_never_make_the_force_panel_wider(trial, app):
    rows, fs, ft, fp = numbers(trial)
    short = force_panel(trial)
    before = short.minimumSizeHint()
    trial.force_cal_note = "a calibration note that goes on and on " * 20
    long_note = "The force was not steady before the push, and then some more words about it. " * 8
    huge = dict(fs, main_angle="A joint angle with a name that goes on and on " * 8, rest_note=long_note, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0,
                relaxed=100.0, peak_minus_relaxed=100.0, rest_minus_relaxed=-100.0, relaxed_note="x " * 200)
    panel = force_panel(trial, fs=huge)
    assert panel.minimumSizeHint() == before and before.width() == ForcePanel.SECTION_MIN_WIDTH
    host = QWidget()                                                 # in a layout, the window's minimum width does not grow either
    QVBoxLayout(host).addWidget(panel)
    host.layout().setContentsMargins(0, 0, 0, 0)
    assert host.minimumSizeHint().width() <= ForcePanel.SECTION_MIN_WIDTH
    show(host, 250)
    assert host.width() == 250 or host.width() >= 250 and host.minimumSizeHint().width() <= 250
    for s in panel._shown:                                           # every row of every section stays inside its section
        assert s.geometry().right() < panel.width()
        grid = s.grid
        for w in grid._labels + grid._values + grid._notes:
            if not w.isHidden():
                assert w.geometry().right() < grid.width() and w.geometry().left() >= 0, (s.title.text(), w.text())
    for w in panel.warnings:
        assert w.geometry().right() < panel.width() and w.height() == w.heightForWidth(w.width())             # (one cut line each)
    host.close()
