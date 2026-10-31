"""The force tree of the Review tab, 0.32.0 (limbtrack/review_force.py): five bold numbers (peak force, above rest, plateau mean, peak torque, load
cell) that fold open to what they are made of, with the sliding window, smoothing, picture delay and zero as real controls inside the rows.

It replaces ForcePanel (the cards, the block of force settings and the Force tab), so these tests check three things: that every fact the old
panel showed is still there (the equivalence test builds three different trials and looks for each number of the old text in the new one),
that the controls the Review tab passes in are really moved into the rows and keep working (and never keep the keyboard), and that the tree
looks right (folds, colours in both palettes, never wider than its parent, a scroll bar of its own only when it is too short).

The tree is built standalone from a Trial and stand-in controls: no ReviewTab is made here."""
import copy
import math
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QLabel, QPushButton, QStyle, QStyleOptionSpinBox, QTreeWidgetItem,  # noqa: E402
                               QVBoxLayout, QWidget)

from limbtrack import analysis  # noqa: E402
from limbtrack.loadcell import COVERAGE, GRAVITY, fit_points  # noqa: E402
from limbtrack.review_force import CONTROLS, METRICS, TITLE, ZERO_FROM, ZERO_WORDS, ForceTree  # noqa: E402
from limbtrack.review_style import muted_colour  # noqa: E402

try:
    from limbtrack.review_panels import ForcePanel  # noqa: E402 - the old panel, until the lead deletes it
except ImportError:  # pragma: no cover
    ForcePanel = None

from tests.test_analysis import torque_trial  # noqa: E402
from tests.test_force import humming_trial  # noqa: E402

WINDOW = (8.5, 13.0)                 # the hold of torque_trial and humming_trial (the force is at its top from about 8.6 to 13 s)
HIDDEN = []                          # errors raised inside an event handler or a slot (see `no_hidden_errors`)


# ------------------------------------------------------------------------------------------------ set-up
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_hidden_errors(monkeypatch, request):
    """Qt prints an exception raised inside an event handler (a resize, a click, a palette change) and carries on, so a test would pass straight
    over it. They are written down here instead, and the test fails."""
    HIDDEN.clear()
    monkeypatch.setattr(sys, "excepthook", lambda kind, err, tb: HIDDEN.append(f"{kind.__name__}: {err}"))
    yield
    if HIDDEN and not request.node.get_closest_marker("xfail"):
        pytest.fail("an error was raised inside a Qt handler and hidden by Qt: " + "; ".join(HIDDEN))


_KEPT = ("meta", "tracks", "px_per_cm", "force_v", "force_n", "regression", "force_cal", "force_cal_note", "force_smooth_s", "window_peak_s",
         "zero_edit")


def _restoring(base):
    """A trial that is put back as it was after each test (a test may change its force, regression or dots)."""
    saved = {k: (copy.deepcopy(getattr(base, k)) if k in ("meta", "force_cal", "zero_edit") else dict(base.tracks) if k == "tracks" else getattr(base, k))
             for k in _KEPT}
    yield base
    for k, v in saved.items():
        setattr(base, k, v)
    base.__dict__.pop("_rest_cache", None)


def relaxed_trial(tmp_path, this_trial=True, relaxed_v=0.2):
    """A volts-only recording made into newtons (500 N per volt, zero 0.05 V) with Zero 2 taken at 0.2 V (as tests/test_force_relaxed.py)."""
    tr, _ = humming_trial(tmp_path, hum=0.0)
    tr.force_cal = {"offset_v": 0.05, "n_per_v": 500.0, "zero_set": True, "zero_v": 0.05}
    tr.force_n = (tr.force_v - 0.05) * 500.0
    tr.meta.setdefault("force", {})["relaxed"] = {"volts": relaxed_v, "this_trial": this_trial, "taken_for": "P01 | knee_extension | 80"}
    return tr


@pytest.fixture(scope="module")
def base_calibrated(app, tmp_path_factory):
    tr = torque_trial(tmp_path_factory.mktemp("tree_calibrated"))
    yield tr
    tr.close()


@pytest.fixture(scope="module")
def base_volts(app, tmp_path_factory):
    tr, _ = humming_trial(tmp_path_factory.mktemp("tree_volts"), hum=0.0)
    yield tr
    tr.close()


@pytest.fixture(scope="module")
def base_relaxed(app, tmp_path_factory):
    tr = relaxed_trial(tmp_path_factory.mktemp("tree_relaxed"))
    yield tr
    tr.close()


@pytest.fixture
def calibrated(base_calibrated):
    """200 N hold, a ruler scale and the anchor dot (so a torque), no regression stored."""
    yield from _restoring(base_calibrated)


@pytest.fixture
def volts(base_volts):
    yield from _restoring(base_volts)


@pytest.fixture
def relaxed(base_relaxed):
    yield from _restoring(base_relaxed)


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


def regression(top_kg=100.0):
    """A load cell regression of about 222 N per volt with a little scatter, calibrated up to `top_kg`."""
    kg = np.linspace(0.0, top_kg, 6)
    noise = np.array([0.001, -0.001, 0.0005, -0.0005, 0.0008, -0.0008])
    return fit_points(kg, 0.1 + kg * GRAVITY / (200.0 / 0.9) + noise, id="R-test", date="2026-10-30")


def controls():
    """Stand-ins for the widgets the Review tab owns and passes in (it connects them; here they only have to exist)."""
    def spin(lo, hi, decimals, step, suffix, value):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setSuffix(suffix)
        s.setValue(value)
        return s
    c = {"sliding": spin(0, 2000, 0, 20, " ms", 500), "smoothing": spin(0, 1000, 0, 20, " ms", 100), "delay": spin(-1, 2, 3, 0.01, " s", 0.0),
         "zero": spin(-10, 10, 4, 0.001, " V", 0.1), "apply": QPushButton("Apply zero"), "zero_stretch": QPushButton("Zero = grey stretch"),
         "zero_rest": QPushButton("Zero = rest before the push"), "zero_reset": QPushButton("Recorded zero"),
         "zero_label": QLabel("the zero taken when this trial was recorded")}
    c["zero_label"].setObjectName("hint")
    c["zero_stretch"].setToolTip("Drag the grey stretch on the force graph, then press this.")
    c["clicks"] = []
    for key in ("apply", "zero_stretch", "zero_rest", "zero_reset"):
        c[key].clicked.connect(lambda _=False, k=key: c["clicks"].append(k))
    return c


def numbers(tr, window=WINDOW, delay=0.0):
    """(fs, ft, fp): what Review hands the panels for a window (ReviewTab.update_summary)."""
    t0, t1 = window
    fs = tr.force_summary(t0, t1, delay)
    ft = analysis.force_timing(tr, t0, t1) if fs is not None else None
    return fs, ft, analysis.force_peaks(tr, t0, t1, timing=ft)


def make(tr, fs=None, ft="real", fp="real", window=WINDOW, delay=0.0, width=None, ctl=None):
    """A ForceTree filled for `tr`; `fs`/`ft`/`fp` replace the real numbers (pass None for a missing one)."""
    ctl = ctl or controls()
    real_fs, real_ft, real_fp = numbers(tr, window, delay)
    tree = ForceTree(ctl)
    tree.set_data(tr, window[0], window[1], real_fs if fs is None else fs, real_ft if ft == "real" else ft, real_fp if fp == "real" else fp, delay)
    if width:
        tree.resize(width, 700)
    tree.ctl = ctl
    return tree


def show(widget, width, height=700):
    widget.resize(width, height)
    widget.show()
    QApplication.processEvents()
    QApplication.processEvents()
    return widget


def row(tree, key):
    """(name, value, note) of a row as it is written in its cells."""
    item = tree.items[key]
    return item.text(0) or key, item.text(1), item.text(2)


def colour(item, column):
    return item.foreground(column).color().name()


# ------------------------------------------------------------------------------------------------ what it is
def test_it_has_the_five_bold_numbers_and_only_the_first_is_open(calibrated, app):
    tree = make(calibrated)
    assert list(tree.rows) == list(METRICS) == ["Peak force", "Above rest", "Plateau mean", "Peak torque", "Load cell"]
    assert [tree.tree.topLevelItem(i).text(0) for i in range(tree.tree.topLevelItemCount())] == list(METRICS)
    assert [tree.rows[m].isExpanded() for m in METRICS] == [True, False, False, False, False]
    assert tree.title.text() == TITLE == "Force numbers and what they depend on" and tree.tree.headerItem().text(1) == "Value"
    assert [tree.tree.headerItem().text(c) for c in (0, 1, 2)] == ["", "Value", "How"]
    for m in METRICS:                                                                  # the number is large and bold, the name bold
        item = tree.rows[m]
        assert item.font(1).bold() and item.font(1).pointSizeF() >= 14 and item.font(0).bold()
    assert tree.open_all_link.text() == "Open all" and tree.warnings == []


def test_the_peak_force_is_the_sliding_window_peak_with_what_it_depends_on_under_it(calibrated, app):
    fs, ft, fp = numbers(calibrated)
    tree = make(calibrated)
    peak = tree.rows["Peak force"]
    assert [peak.child(i).data(0, Qt.UserRole + 1) for i in range(peak.childCount())] == [
        "Sliding window", "Smoothing", "Picture delay", "Smoothed peak", "Time to peak"]
    assert row(tree, "Peak force") == ("Peak force", f"{fp['window_peak']:.1f} N", "500 ms sliding-window average")
    assert f"{fp['window_peak_start_s']:.2f} to {fp['window_peak_end_s']:.2f} s" in tree.items["Sliding window"].text(2)      # where the stretch is (defect a)
    assert f"{fp['window_peak_start_s']:.2f} to {fp['window_peak_end_s']:.2f} s" in tree.rows["Peak force"].toolTip(2)
    smoothed = row(tree, "Smoothed peak")
    assert smoothed[1] == f"{fs['peak']:.1f} N" and f"peak 20 ms {fs['peak_20ms']:.1f} N" in smoothed[2] and f"raw single sample {fs['raw_peak']:.1f} N" in smoothed[2]
    assert f"at {fs['t_peak_s']:.2f} s" in smoothed[2]
    assert row(tree, "Time to peak")[1:] == (f"{ft['time_to_peak_s']:.2f} s", "from the onset to the peak")
    ttp = tree.items["Time to peak"]
    assert [ttp.child(i).data(0, Qt.UserRole + 1) for i in range(ttp.childCount())] == ["Onset", "10 to 90 % of the rise", "Steepest rise", "To the sliding peak"]
    assert row(tree, "Onset")[1] == f"{ft['onset_s']:.2f} s" and row(tree, "10 to 90 % of the rise")[1] == f"{ft['time_to_90pct_s']:.2f} s"
    assert row(tree, "Steepest rise")[1:] == (f"{ft['rise_rate_per_s']:.1f} N/s", "over 100 ms")
    assert row(tree, "To the sliding peak")[1] == f"{fp['time_to_window_peak_s']:.2f} s"


def test_the_raw_single_sample_is_named_only_when_there_is_smoothing(calibrated, app, monkeypatch):
    monkeypatch.setattr(calibrated, "force_smooth_s", 0.0)
    fs, _ft, _fp = numbers(calibrated)
    tree = make(calibrated)
    assert fs["smoothing_s"] == 0 and tree.caption.text() == "Force in the window: N, raw samples"
    assert "raw single sample" not in tree.text(everything=True) and "peak 20 ms" in tree.items["Smoothed peak"].text(2)
    monkeypatch.setattr(calibrated, "force_smooth_s", 0.1)
    assert "raw single sample" in make(calibrated).text(everything=True)


def test_a_window_shorter_than_the_sliding_window_says_so_instead_of_a_number(calibrated, app, monkeypatch):
    window = (10.0, 10.3)                                             # 300 ms against a 500 ms sliding window
    fs, ft, fp = numbers(calibrated, window)
    assert math.isnan(fp["window_peak"])
    tree = make(calibrated, window=window)
    assert row(tree, "Peak force")[1:] == ("-", "window shorter than 500 ms, or gaps in the force")
    warn = colour(tree.rows["Peak force"], 2)
    assert warn != colour(tree.items["Smoothed peak"], 2)             # the note is amber, the others quiet grey
    assert colour(tree.rows["Peak force"], 1) == warn                 # and so is the value
    assert "no stretch" in tree.items["Sliding window"].text(2)
    monkeypatch.setattr(calibrated, "window_peak_s", 0.2)             # a shorter sliding window fits
    assert make(calibrated, window=window).rows["Peak force"].text(1).endswith(" N")


def test_above_rest_has_the_rest_the_zero_and_where_it_came_from(calibrated, app, monkeypatch):
    fs, _ft, _fp = numbers(calibrated)
    tree = make(calibrated)
    zero, where = calibrated.current_zero_v()
    assert row(tree, "Above rest")[1:] == (f"{fs['peak_minus_rest']:.1f} N", f"rest {fs['rest_mean']:.1f} N - zero {zero:.4f} V ({ZERO_WORDS[calibrated.zero_source()]})")
    assert row(tree, "Rest before the push")[1:] == (f"{fs['rest_mean']:.1f} N", f"{fs['rest_basis']}, {fs['rest_seconds']:.1f} s of force")
    monkeypatch.setattr(calibrated, "zero_edit", {"zero_v": 0.0123, "source": "stretch"})                 # a zero set in Review from the grey stretch
    assert row(make(calibrated), "Above rest")[2].endswith("- zero 0.0123 V (grey stretch)")
    monkeypatch.setattr(calibrated, "zero_edit", {"zero_v": 0.0123, "source": "rest"})
    assert row(make(calibrated), "Above rest")[2].endswith("(rest before the push)")
    unknown = make(calibrated, fs=dict(fs, rest_mean=math.nan, peak_minus_rest=math.nan))                  # no rest could be found
    assert row(unknown, "Above rest")[1] == "-" and "rest not known" in row(unknown, "Above rest")[2] and "nan" not in unknown.text(everything=True)
    shaky = make(calibrated, fs=dict(fs, rest_note="The force was not steady before the push."))           # a doubtful rest is amber
    assert colour(shaky.items["Rest before the push"], 1) != colour(tree.items["Rest before the push"], 1)
    assert len(shaky.warnings) == 1 and shaky.warnings[0].text == "The force was not steady before the push."


def test_zero_2_rows_appear_only_when_there_is_a_relaxed_level_or_a_reason_for_none(calibrated, relaxed, app):
    fs, _ft, _fp = numbers(calibrated)
    none = make(calibrated)
    assert none.items["Relaxed (Zero 2)"].isHidden() and none.items["Peak above relaxed"].isHidden()          # none: no word about it
    assert "Zero 2" not in none.text(everything=True)
    window = relaxed.force_window()
    fs_r, _ft_r, _fp_r = numbers(relaxed, window)
    tree = make(relaxed, window=window)
    assert not tree.items["Relaxed (Zero 2)"].isHidden() and not tree.items["Peak above relaxed"].isHidden()
    assert row(tree, "Relaxed (Zero 2)")[1] == f"{fs_r['relaxed']:.1f} N" and "lower than that" in row(tree, "Relaxed (Zero 2)")[2]
    assert row(tree, "Peak above relaxed")[1] == f"{fs_r['peak_minus_relaxed']:.1f} N"
    higher = make(relaxed, fs=dict(fs_r, rest_minus_relaxed=12.0), window=window)
    assert "the rest before the push was 12.0 N higher than that" in row(higher, "Relaxed (Zero 2)")[2]
    unused = make(relaxed, fs=dict(fs_r, relaxed=math.nan, peak_minus_relaxed=math.nan, rest_minus_relaxed=math.nan,
                                   relaxed_note="A relaxed level was taken for another position (60), so it is not used for this trial."), window=window)
    assert row(unused, "Relaxed (Zero 2)")[1:] == ("not used", "A relaxed level was taken for another position (60), so it is not used for this trial.")
    assert unused.items["Peak above relaxed"].isHidden()


def test_the_plateau_mean_and_how_steadily_it_was_held(calibrated, app):
    fs, ft, _fp = numbers(calibrated)
    tree = make(calibrated)
    assert row(tree, "Plateau mean")[1:] == (f"{fs['plateau_mean']:.1f} N", "middle 60 % of the window")
    assert row(tree, "Held above 90 %")[1:] == (f"{ft['hold_above_90pct_s']:.2f} s", "of the peak") and row(tree, "Steadiness")[1].endswith(" %")
    none = make(calibrated, ft=None, fp=analysis.force_peaks(calibrated, *WINDOW, timing=None))             # no clear push in the window
    assert none.items["Held above 90 %"].isHidden() and none.items["Steadiness"].isHidden() and none.items["Onset"].isHidden()
    assert row(none, "Time to peak")[1:] == ("-", "no clear push in the window") and colour(none.items["Time to peak"], 2) != colour(tree.items["Time to peak"], 2)


def test_the_peak_torque_with_a_moment_arm_and_with_a_stand_in_dot(calibrated, app, monkeypatch):
    tq = analysis.torque_summary(calibrated, *WINDOW)
    tree = make(calibrated)
    assert row(tree, "Peak torque")[1] == f"{tq['peak_nm']:.1f} N.m" and f"force x arm {tq['arm_at_peak_force_cm']:.1f} cm" in row(tree, "Peak torque")[2]
    assert f"at {tq['t_peak_s']:.2f} s" in row(tree, "Peak torque")[2]
    assert row(tree, "At the peak force")[1] == f"{tq['at_peak_force_nm']:.1f} N.m" and row(tree, "Plateau")[1] == f"{tq['plateau_nm']:.1f} N.m"
    assert row(tree, "Moment arm")[1:] == ("30.0 cm", "at the peak force, mean 30.0 cm")
    assert tree.items["Stand-in dot"].isHidden() and not tree.items["Strap and anchor dots: see the angle table"].isHidden()
    assert "strap and anchor dots: see the angle table" in tree.text(everything=True).lower()
    monkeypatch.delitem(calibrated.tracks, "strap")                                                      # the strap dot was never seen: the ankle dot stands in
    stand_in = make(calibrated)
    name, value, note = row(stand_in, "Stand-in dot")
    pictures = analysis.torque_summary(calibrated, *WINDOW)["pictures"]
    assert not stand_in.items["Stand-in dot"].isHidden() and "the strap dot was not seen in" in note and "the ankle dot stood in for it" in note
    assert value == f"{pictures} of {pictures}" and colour(stand_in.items["Stand-in dot"], 2) != colour(stand_in.items["Moment arm"], 2)


@pytest.mark.parametrize("change, reason", [
    ("uncalibrated", "the force is not calibrated"),
    ("no_ruler", "there is no ruler scale"),
    ("no_anchor", "the anchor dot was not used"),
    ("no_arm", "no moment arm could be measured"),
])
def test_torque_is_not_available_and_says_why_in_muted_text(calibrated, app, monkeypatch, change, reason):
    window = WINDOW
    if change == "uncalibrated":
        monkeypatch.setattr(calibrated, "force_n", np.full_like(calibrated.force_n, np.nan))
    elif change == "no_ruler":
        monkeypatch.setattr(calibrated, "px_per_cm", None)
    elif change == "no_anchor":
        monkeypatch.delitem(calibrated.tracks, "anchor")
    else:
        window = (10.0, 10.05)                                    # no picture in the window, so no moment arm
    calibrated.__dict__.pop("_rest_cache", None)
    tree = make(calibrated, window=window)
    item = tree.rows["Peak torque"]
    assert row(tree, "Peak torque")[1:] == ("not available", reason)
    assert item.childIndicatorPolicy() == QTreeWidgetItem.DontShowIndicator                              # no little arrow: there is nothing under it
    assert all(item.child(i).isHidden() for i in range(item.childCount())) and not item.isExpanded()
    assert colour(item, 1) == muted_colour(QApplication.palette()) and colour(item, 2) == muted_colour(QApplication.palette())    # muted, not bold black
    assert not item.font(1).bold()                                # (the words are not a large number)
    assert "Peak torque: not available  " + reason in tree.text(everything=True)


def test_the_load_cell_gives_the_precision_with_a_plus_or_minus_and_which_calibration(calibrated, app, monkeypatch):
    reg = regression(100.0)
    monkeypatch.setattr(calibrated, "regression", reg)
    monkeypatch.setattr(calibrated, "force_cal", {"n_per_v": 200.0 / 0.9, "offset_v": 0.1, "method": "test"})
    fs, _ft, _fp = numbers(calibrated)
    tree = make(calibrated)
    sd = COVERAGE * float(reg.sd_newton(fs["peak"]))
    assert row(tree, "Load cell")[1:] == (f"± {sd:.1f} N", "calibration R-test, 2026-10-30")
    assert row(tree, "Calibration")[1:] == ("recorded", calibrated.force_cal_note) and row(tree, "Regression")[1] == "R-test"
    assert row(tree, "Regression")[2] == f"{200.0 / 0.9:.2f} N per volt, zero +0.1000 V"
    for key, f in (("Peak plus or minus", fs["peak"]), ("Plateau plus or minus", fs["plateau_mean"])):
        kg = f / GRAVITY
        sd = COVERAGE * math.sqrt(reg.rmse_kg ** 2 + (kg * reg.slope_rel_se) ** 2) * GRAVITY             # 2 standard deviations of one reading at this load
        assert row(tree, key)[1:] == (f"{sd:.1f} N", f"{100 * sd / abs(f):.2f} % at about 95 % ({kg:.1f} kg)")
    assert tree.items["Peak above the line"].isHidden() and tree.items["Newtons"].isHidden()


def test_a_load_above_the_heaviest_calibration_load_is_called_out(calibrated, app, monkeypatch):
    monkeypatch.setattr(calibrated, "regression", regression(15.0))                                      # 15 kg is 147 N: the 200 N peak is beyond the line
    monkeypatch.setattr(calibrated, "force_cal", {"n_per_v": 200.0 / 0.9, "offset_v": 0.1, "method": "test"})
    fs, _ft, _fp = numbers(calibrated)
    tree = make(calibrated)
    assert row(tree, "Peak above the line")[1:] == (f"{fs['peak'] / GRAVITY:.0f} kg", "the heaviest calibration load is 15 kg: the line is extended there, not measured")
    assert not tree.items["Peak above the line"].isHidden() and colour(tree.items["Peak above the line"], 2) != colour(tree.items["Regression"], 2)
    nan_plateau = make(calibrated, fs=dict(fs, plateau_mean=math.nan))                                   # a NaN plateau is left out, not shown as 'nan'
    assert nan_plateau.items["Plateau plus or minus"].isHidden() and "nan" not in nan_plateau.text(everything=True)


def test_without_a_regression_there_is_no_precision_and_it_says_so(calibrated, app, monkeypatch):
    tree = make(calibrated)                                                                              # the trial has a calibration but no regression
    assert row(tree, "Load cell")[1:] == ("-", "no regression stored: no precision can be given")
    assert row(tree, "Regression")[1] == "none stored" and "N per volt: no precision can be given" in row(tree, "Regression")[2]
    assert tree.items["Peak plus or minus"].isHidden()
    monkeypatch.setattr(calibrated, "force_cal", None)                                                   # nothing recorded about the calibration
    assert "unknown calibration, 0.00 N per volt" in row(make(calibrated), "Regression")[2]


def test_the_calibration_button_asks_for_the_calibration_graph(calibrated, app):
    tree = make(calibrated)
    asked = []
    tree.calibrationRequested.connect(lambda: asked.append(1))
    assert tree.button_calibration.text() == "Show the calibration graph" and tree.tree.itemWidget(tree.items["Show the calibration graph"], 0) is not None
    tree.button_calibration.click()
    assert asked == [1] and "[Show the calibration graph]" in tree.text(everything=True)


# ------------------------------------------------------------------------------------------------ units
def test_a_force_in_volts_says_so_and_shows_three_decimals(volts, app):
    fs, ft, fp = numbers(volts)
    assert fs["unit"] == "V"
    tree = make(volts)
    assert tree.caption.text() == "Force in the window: V, not calibrated, 100 ms moving average"
    assert row(tree, "Peak force")[1] == f"{fp['window_peak']:.3f} V" and re.fullmatch(r"\d\.\d{3} V", row(tree, "Peak force")[1])
    assert row(tree, "Smoothed peak")[1] == f"{fs['peak']:.3f} V" and f"peak 20 ms {fs['peak_20ms']:.3f} V" in row(tree, "Smoothed peak")[2]
    assert row(tree, "Steepest rise")[1] == f"{ft['rise_rate_per_s']:.3f} V/s"
    assert row(tree, "Peak torque")[1:] == ("not available", "the force is not calibrated")
    assert row(tree, "Load cell")[1] == "-" and "not calibrated" in row(tree, "Load cell")[2]
    assert row(tree, "Calibration")[1] == "none" and row(tree, "Newtons")[1:] == ("not yet", "no zero was taken: use Apply zero to get newtons")
    assert not tree.items["Newtons"].isHidden() and tree.items["Regression"].isHidden()
    assert "volts only" in row(tree, "Load cell")[2] and "volts only" in tree.text(everything=True)


def test_a_force_in_newtons_shows_one_decimal(calibrated, app):
    tree = make(calibrated)
    assert re.fullmatch(r"\d+\.\d N", row(tree, "Peak force")[1]) and "not calibrated" not in tree.caption.text()
    assert row(tree, "Steepest rise")[1].endswith(" N/s")


# ------------------------------------------------------------------------------------------------ none
def test_set_none_shows_the_text_centred_and_hides_the_tree(calibrated, app):
    tree = ForceTree(controls())
    assert tree.text() == "No trial is open." and tree.tree.isHidden() and tree.caption.isHidden() and tree.open_all_link.isHidden()
    fs, ft, fp = numbers(calibrated)
    tree.set_data(calibrated, *WINDOW, fs, ft, fp, 0.0)
    assert tree.text().startswith(TITLE) and not tree.tree.isHidden() and not tree.empty.isVisibleTo(tree)
    tree.set_none("No force was recorded in this trial.")
    assert tree.text() == "No force was recorded in this trial." and tree.tree.isHidden() and tree.empty.alignment() == Qt.AlignCenter and tree.empty.wordWrap()
    assert tree.warnings == [] and tree._last is None
    tree.set_data(calibrated, *WINDOW, None, None, None, 0.0)                                            # a window without force samples
    assert tree.text() == "Force was recorded but there are no samples in this window." and tree.tree.isHidden()
    tree.set_data(calibrated, *WINDOW, fs, ft, fp, 0.0)
    assert tree.text().startswith(TITLE) and not tree.tree.isHidden()


def test_a_missing_control_is_named_in_plain_words(app):
    ctl = controls()
    del ctl["zero_label"]
    with pytest.raises(KeyError, match="zero_label"):
        ForceTree(ctl)
    assert set(CONTROLS) <= set(controls())


# ------------------------------------------------------------------------------------------------ warnings
def test_no_warning_without_a_reason(calibrated, app):
    tree = make(calibrated)
    assert tree.warnings == [] and "Warning" not in tree.text(everything=True)
    fs, _ft, _fp = numbers(calibrated)
    at_limit = make(calibrated, fs=dict(fs, mains_hum_v=0.027, mains_hum_pct_of_peak=analysis.HUM_WARN_PCT))        # exactly the limit is not over it
    assert at_limit.warnings == []


def test_a_50_hz_hum_over_the_limit_is_a_warning_with_its_numbers(calibrated, app):
    fs, _ft, _fp = numbers(calibrated)
    tree = make(calibrated, fs=dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0))
    assert len(tree.warnings) == 1 and tree.warnings[0].text.startswith("50 Hz mains interference: 90 mV = 10 % of the rise above rest ")
    assert "(check shielding and grounding; a smoothing window that is a multiple of 20 ms removes it)." in tree.warnings[0].text
    assert tree.text().splitlines()[2] == "Warning: " + tree.warnings[0].text                           # the whole sentence is in the text, not the cut one


def test_a_hum_in_the_recorded_voltage_is_found_by_the_real_numbers_too(volts, app, monkeypatch):
    monkeypatch.setattr(volts, "force_v", volts.force_v + 0.09 * np.sin(2 * np.pi * 50.0 * volts.force_t))     # the raw voltage hums by 90 mV
    volts.__dict__.pop("_rest_cache", None)
    fs, _ft, _fp = numbers(volts)
    assert 8 < fs["mains_hum_pct_of_peak"] < 12
    tree = make(volts)
    assert len(tree.warnings) == 1 and tree.warnings[0].text.startswith("50 Hz mains interference: 90 mV")


def test_a_rest_note_and_a_zero_2_mismatch_are_warnings_in_that_order(calibrated, app):
    fs, _ft, _fp = numbers(calibrated)
    note = "The force was not steady before the push (it varied by 12 N, SD): the strap was loaded or moving, so the resting force is doubtful."
    bad = dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0, rest_note=note, relaxed=100.0, peak_minus_relaxed=fs["peak"] - 100.0,
               rest_minus_relaxed=fs["rest_mean"] - 100.0)
    tree = make(calibrated, fs=bad)
    mismatch = analysis.relaxed_mismatch("N", 100.0, fs["rest_mean"], fs["peak_minus_rest"])
    assert mismatch.startswith("Zero 2 (relaxed) was 100 N, but the force just before this push was 0 N (-100 N)")
    assert [w.text for w in tree.warnings][1:] == [note, mismatch] and tree.warnings[0].text.startswith("50 Hz")
    lines = tree.text().splitlines()
    assert lines[2:5] == ["Warning: " + w.text for w in tree.warnings]
    assert [tree.tree.topLevelItem(i).text(0) for i in range(3, 8)] == list(METRICS)                      # the amber rows are the first rows of the tree
    only_mismatch = make(calibrated, fs=dict(bad, mains_hum_pct_of_peak=0.1, rest_note=""))
    assert [w.text for w in only_mismatch.warnings] == [mismatch]
    close = make(calibrated, fs=dict(fs, relaxed=fs["rest_mean"] + 10.0))                                 # 10 N apart is within the limit
    assert close.warnings == []


def test_warnings_are_kept_when_they_do_not_change_and_cleared_when_they_go(calibrated, app):
    fs, ft, fp = numbers(calibrated)
    hum = dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0)
    tree = make(calibrated, fs=hum)
    first = tree.warnings[0]
    tree.set_data(calibrated, *WINDOW, hum, ft, fp, 0.0)
    assert tree.warnings[0] is first                                                                     # the same warning: not made again (it may be open)
    tree.set_data(calibrated, *WINDOW, fs, ft, fp, 0.0)
    assert tree.warnings == [] and "Warning" not in tree.text() and tree.tree.topLevelItemCount() == 5
    tree.set_data(calibrated, *WINDOW, hum, ft, fp, 0.0)
    assert len(tree.warnings) == 1 and tree.tree.topLevelItemCount() == 6
    tree.set_none("No force was recorded in this trial.")
    assert tree.warnings == [] and tree.tree.topLevelItemCount() == 5


def test_a_warning_is_one_line_until_it_is_clicked_and_then_all_of_it(calibrated, app):
    fs, _ft, _fp = numbers(calibrated)
    tree = show(make(calibrated, fs=dict(fs, relaxed=100.0, peak_minus_relaxed=100.0, rest_minus_relaxed=-100.0)), 420)
    w = tree.warnings[0]
    item = tree._warning_items[0]
    height = tree.tree.visualItemRect(item).height()
    assert not w.expanded and "…" in w.label.text() and w.toolTip() == w.text
    QTest.mouseClick(w, Qt.LeftButton)
    QApplication.processEvents()
    assert w.expanded and w.label.text() == w.text and tree.tree.visualItemRect(item).height() > height * 1.5     # the row grew to hold the sentence
    below = tree.tree.visualItemRect(tree.rows["Peak force"])
    assert below.top() >= tree.tree.visualItemRect(item).bottom()                                        # the numbers moved down: nothing under the warning
    QTest.mouseClick(w, Qt.LeftButton)
    QApplication.processEvents()
    assert not w.expanded and tree.tree.visualItemRect(item).height() == height
    tree.close()


# ------------------------------------------------------------------------------------------------ the controls in the rows
def test_the_rows_are_built_once_and_the_controls_stay_alive_in_them(calibrated, volts, app):
    ctl = controls()
    tree = ForceTree(ctl)
    items = dict(tree.items)
    for k in CONTROLS:
        assert tree.isAncestorOf(ctl[k]), k
    for k in ("sliding", "smoothing", "delay", "zero", "apply"):                                         # in the cells of the tree itself
        assert tree.tree.viewport().isAncestorOf(ctl[k]), k
    assert ctl["zero_label"].parent() is not None and tree.tree.viewport().isAncestorOf(ctl["zero_label"])
    assert all(ctl[k].isHidden() and ctl[k].parent() is tree for k in ("zero_stretch", "zero_rest", "zero_reset"))     # alive, hidden, children
    children = len(tree.findChildren(QWidget))
    for tr in (calibrated, volts, calibrated):
        fs, ft, fp = numbers(tr)
        tree.set_data(tr, *WINDOW, fs, ft, fp, 0.0)
        tree.set_data(tr, *WINDOW, fs, ft, fp, 0.0)
    assert tree.items == items and all(tree.items[k] is items[k] for k in items)                         # the same rows
    assert len(tree.findChildren(QWidget)) == children                                                   # nothing made again
    ctl["sliding"].setValue(300)                                                                         # the controls still work (they are the Review tab's)
    assert ctl["sliding"].value() == 300 and "300 ms" in tree.text(everything=True)


def test_the_controls_show_in_their_cells_when_the_rows_are_open(calibrated, app):
    ctl = controls()
    tree = show(make(calibrated, ctl=ctl), 600, 800)
    view = tree.tree.viewport()
    for k in ("sliding", "smoothing", "delay"):                                                          # (Peak force is open to begin with)
        assert ctl[k].isVisible() and view.rect().contains(ctl[k].mapTo(view, QPoint(0, 0))), k
    assert not ctl["zero"].isVisible() and not ctl["apply"].isVisible() and not ctl["zero_label"].isVisible()   # folded away with 'Above rest'
    tree.set_fold("Above rest", True)
    tree.tree.settle()
    QApplication.processEvents()
    for k in ("zero", "apply", "zero_label"):
        assert ctl[k].isVisible(), k
    rect = tree.tree.visualItemRect(tree.items["Zero"])
    assert rect.height() >= ctl["zero"].height() and view.rect().contains(rect.topLeft())
    assert tree.zero_from.isVisible()
    tree.close()


def test_the_zero_from_menu_presses_the_three_buttons_the_review_tab_handles(calibrated, app):
    ctl = controls()
    tree = make(calibrated, ctl=ctl)
    assert tree.zero_from.text() == "Zero from" and [a.text() for a in tree.zero_menu.actions()] == ["grey stretch", "rest before the push", "recorded zero"]
    assert [k for k, _t in ZERO_FROM] == ["zero_stretch", "zero_rest", "zero_reset"] and tree.zero_from.menu() is tree.zero_menu
    for key in ("zero_stretch", "zero_rest", "zero_reset"):
        tree.zero_actions[key].trigger()
    ctl["apply"].click()
    assert ctl["clicks"] == ["zero_stretch", "zero_rest", "zero_reset", "apply"]
    assert tree.zero_actions["zero_stretch"].toolTip() == ctl["zero_stretch"].toolTip()


def test_the_menu_entries_are_as_enabled_as_the_buttons(calibrated, app):
    ctl = controls()
    tree = make(calibrated, ctl=ctl)
    assert all(a.isEnabled() for a in tree.zero_actions.values()) and tree.zero_from.isEnabled()
    ctl["zero_reset"].setEnabled(False)                                                                  # no zero was set in Review: nothing to go back to
    QApplication.processEvents()
    assert not tree.zero_actions["zero_reset"].isEnabled() and tree.zero_actions["zero_rest"].isEnabled() and tree.zero_from.isEnabled()
    tree.zero_actions["zero_reset"].trigger()                                                            # (a disabled entry does nothing; neither does its button)
    assert "zero_reset" not in ctl["clicks"]
    ctl["zero_reset"].setEnabled(True)
    ctl["zero_stretch"].setEnabled(False)
    ctl["zero_rest"].setEnabled(False)
    ctl["zero_reset"].setEnabled(False)
    QApplication.processEvents()
    assert not tree.zero_from.isEnabled()                                                                # all three off: the drop-down is off
    ctl["zero_rest"].setEnabled(True)
    QApplication.processEvents()
    assert tree.zero_from.isEnabled() and tree.zero_actions["zero_rest"].isEnabled()


def test_open_zero_unfolds_above_rest_and_gives_the_zero_box_the_keyboard(calibrated, app):
    ctl = controls()
    win = QWidget()
    lay = QVBoxLayout(win)
    tree = make(calibrated, ctl=ctl)
    lay.addWidget(tree)
    show(win, 560, 420)
    assert not tree.rows["Above rest"].isExpanded() and not ctl["zero"].isVisible()
    win.activateWindow()
    assert QTest.qWaitForWindowActive(win, 5000)
    tree.open_zero()
    QApplication.processEvents()
    assert tree.rows["Above rest"].isExpanded() and ctl["zero"].isVisible() and QApplication.focusWidget() is ctl["zero"]
    assert tree.tree.visualItemRect(tree.items["Zero"]).bottom() <= tree.tree.viewport().height()          # scrolled into view, whatever the height
    win.close()


def test_a_spin_box_gives_the_keyboard_back_after_a_number_was_typed(calibrated, app):
    ctl = controls()
    win = QWidget()
    lay = QVBoxLayout(win)
    tree = make(calibrated, ctl=ctl)
    lay.addWidget(tree)
    show(win, 560, 420)
    win.activateWindow()
    assert QTest.qWaitForWindowActive(win, 5000)
    assert all(ctl[k].focusPolicy() == Qt.ClickFocus for k in ("sliding", "smoothing", "delay", "zero")) and ctl["apply"].focusPolicy() == Qt.NoFocus
    assert tree.tree.focusPolicy() == Qt.NoFocus and tree.zero_from.focusPolicy() == Qt.NoFocus and tree.open_all_link.focusPolicy() == Qt.NoFocus
    spin = ctl["smoothing"]
    spin.setFocus()
    assert QApplication.focusWidget() is spin
    spin.selectAll()
    QTest.keyClicks(spin, "60")
    assert QApplication.focusWidget() is spin                                                            # (while it is typed in, it has the keyboard)
    QTest.keyClick(spin, Qt.Key_Return)                                                                  # the number is accepted
    assert spin.value() == 60 and QApplication.focusWidget() is not spin
    win.close()


def test_the_little_arrows_of_a_spin_box_also_give_the_keyboard_back(calibrated, app):
    ctl = controls()
    win = QWidget()
    lay = QVBoxLayout(win)
    tree = make(calibrated, ctl=ctl)
    lay.addWidget(tree)
    show(win, 560, 420)
    win.activateWindow()
    assert QTest.qWaitForWindowActive(win, 5000)
    spin = ctl["sliding"]
    opt = QStyleOptionSpinBox()
    opt.initFrom(spin)
    opt.rect = spin.rect()
    opt.subControls = QStyle.SC_All
    up = spin.style().subControlRect(QStyle.CC_SpinBox, opt, QStyle.SC_SpinBoxUp, spin).center()
    before = spin.value()
    QTest.mouseClick(spin, Qt.LeftButton, Qt.NoModifier, up)
    QTest.qWait(30)
    assert spin.value() == before + 20 and QApplication.focusWidget() is not spin                        # (the arrow worked, and let go of the keyboard)
    QTest.mouseClick(spin, Qt.LeftButton, Qt.NoModifier, QPoint(8, spin.height() // 2))                  # a click in the number is for typing: it keeps it
    assert QApplication.focusWidget() is spin
    win.close()


# ------------------------------------------------------------------------------------------------ the zero label
def test_the_line_about_the_zero_wraps_and_follows_the_text(calibrated, app):
    ctl = controls()
    tree = show(make(calibrated, ctl=ctl), 520, 800)
    tree.set_fold("Above rest", True)
    tree.tree.settle()
    QApplication.processEvents()
    note = tree.items["Zero comes from"]
    one_line = tree.tree.visualItemRect(note).height()
    assert not note.isHidden() and "the zero taken when this trial was recorded" in tree.text()
    long_text = ctl["zero_label"].text() + "  (the stretch is 1050 ms: not a multiple of 20 ms, so mains hum does not cancel exactly) and some more words " * 3
    ctl["zero_label"].setText(long_text)                                                                 # the Review tab writes to its own label
    QTest.qWait(60)
    QApplication.processEvents()
    assert tree.tree.visualItemRect(note).height() > 2 * one_line and long_text in tree.text()
    assert ctl["zero_label"].geometry().right() <= tree.tree.viewport().width()
    ctl["zero_label"].setText("")
    QTest.qWait(60)
    QApplication.processEvents()
    assert note.isHidden() and long_text not in tree.text(everything=True)
    tree.close()


# ------------------------------------------------------------------------------------------------ folds
def test_the_folds_are_remembered_over_set_data_and_over_other_trials(calibrated, volts, app):
    tree = make(calibrated)
    tree.set_fold("Above rest", True)
    tree.set_fold("Peak force", False)
    tree.set_fold("Time to peak", True)
    expected = {"Peak force": False, "Above rest": True, "Time to peak": True, "Plateau mean": False, "Peak torque": False, "Load cell": False}
    for tr in (calibrated, volts, calibrated):
        fs, ft, fp = numbers(tr)
        tree.set_data(tr, *WINDOW, fs, ft, fp, 0.0)
        assert {k: tree.items[k].isExpanded() for k in expected} == expected
    tree.set_none("No force was recorded in this trial.")
    fs, ft, fp = numbers(calibrated)
    tree.set_data(calibrated, *WINDOW, fs, ft, fp, 0.0)
    assert {k: tree.items[k].isExpanded() for k in expected} == expected


def test_a_fold_the_torque_cannot_show_is_remembered_for_the_next_trial_that_has_one(calibrated, volts, app):
    tree = make(calibrated)
    tree.set_fold("Peak torque", True)
    assert tree.rows["Peak torque"].isExpanded()
    fs, ft, fp = numbers(volts)                                                                          # volts only: no torque, no little arrow
    tree.set_data(volts, *WINDOW, fs, ft, fp, 0.0)
    assert not tree.rows["Peak torque"].isExpanded() and tree.rows["Peak torque"].childIndicatorPolicy() != tree.rows["Plateau mean"].childIndicatorPolicy()
    fs, ft, fp = numbers(calibrated)
    tree.set_data(calibrated, *WINDOW, fs, ft, fp, 0.0)
    assert tree.rows["Peak torque"].isExpanded()


def test_a_click_on_a_row_folds_it_and_the_fold_is_remembered(calibrated, app):
    tree = show(make(calibrated), 600, 600)
    item = tree.rows["Plateau mean"]
    rect = tree.tree.visualItemRect(item)
    QTest.mouseClick(tree.tree.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(300, rect.center().y()))
    assert item.isExpanded() and tree._open["Plateau mean"] is True
    rect = tree.tree.visualItemRect(item)
    QTest.mouseClick(tree.tree.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(300, rect.center().y()))
    assert not item.isExpanded() and tree._open["Plateau mean"] is False
    tree.close()


def test_open_all_and_close_all(calibrated, volts, app):
    tree = make(calibrated)
    assert tree.open_all_link.text() == "Open all"
    tree.open_all_link.click()
    foldable = [k for k, kind in tree._kind.items() if kind == "fold"]
    assert tree.open_all_link.text() == "Close all" and all(tree.items[k].isExpanded() for k in foldable) and "Time to peak" in foldable
    assert "Onset: " in tree.text() and "Moment arm: " in tree.text()                                    # everything is on the screen now
    tree.open_all_link.click()
    assert tree.open_all_link.text() == "Open all" and not any(tree.items[k].isExpanded() for k in foldable)
    tree.set_fold("Load cell", True)
    tree.open_all_link.click()                                                                           # one is open, not all: the link opens the rest
    assert tree.open_all_link.text() == "Close all"
    tree.set_fold("Plateau mean", False)                                                                 # the text follows every fold
    assert tree.open_all_link.text() == "Open all"
    fs, ft, fp = numbers(volts)                                                                          # a trial without torque: 'all' means all that there is
    tree.open_all_link.click()
    tree.set_data(volts, *WINDOW, fs, ft, fp, 0.0)
    assert tree.open_all_link.text() == "Close all" and not tree.rows["Peak torque"].isExpanded()
    with pytest.raises(KeyError):
        tree.set_fold("Onset", True)                                                                     # a row with nothing under it does not fold
    with pytest.raises(KeyError):
        tree.set_fold("Nothing like it", True)


def test_the_text_without_everything_leaves_out_what_is_folded(calibrated, app):
    tree = make(calibrated)
    short, full = tree.text(), tree.text(everything=True)
    assert "Sliding window: 500 ms" in short and "Moment arm" not in short and "Regression" not in short and "Onset" not in short       # (folded)
    assert "Moment arm" in full and "Regression" in full and "Onset" in full and "Rest before the push" in full and len(full) > len(short)
    assert short.splitlines()[0] == f"{TITLE}  [Open all]" and short.splitlines()[1] == tree.caption.text()
    assert all(line in full.splitlines() for line in short.splitlines())


# ------------------------------------------------------------------------------------------------ every fact of the old panel
def numeric_tokens(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def scenario_calibrated(tr, mp):
    return tr, WINDOW, None


def scenario_precision(tr, mp):
    mp.setattr(tr, "regression", regression(15.0))
    mp.setattr(tr, "force_cal", {"n_per_v": 200.0 / 0.9, "offset_v": 0.1, "method": "test"})
    return tr, WINDOW, None


def scenario_stand_in(tr, mp):
    mp.delitem(tr.tracks, "strap")
    return tr, WINDOW, None


def scenario_hum_and_rest_note(tr, mp):
    return tr, WINDOW, lambda fs: dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0, rest_note="The force was not steady before the push (it varied "
                                                                                                    "by 12 N, SD): the resting force is doubtful.")


def scenario_raw(tr, mp):
    mp.setattr(tr, "force_smooth_s", 0.0)
    return tr, WINDOW, None


def scenario_short_window(tr, mp):
    return tr, (10.0, 10.3), None


def scenario_no_push(tr, mp):
    return tr, WINDOW, "no_push"


def scenario_relaxed(tr, mp):
    return tr, tr.force_window(), None


def scenario_relaxed_mismatch(tr, mp):
    return tr, tr.force_window(), lambda fs: dict(fs, relaxed=100.0, peak_minus_relaxed=fs["peak"] - 100.0, rest_minus_relaxed=fs["rest_mean"] - 100.0,
                                                  rest_note="The force was not steady before the push.")


def scenario_relaxed_other_position(tr, mp):
    return tr, tr.force_window(), lambda fs: dict(fs, relaxed=math.nan, peak_minus_relaxed=math.nan, rest_minus_relaxed=math.nan,
                                                  relaxed_note="A relaxed level was taken for another position (60), so it is not used for this trial.")


def scenario_volts(tr, mp):
    return tr, WINDOW, None


SCENARIOS = [("calibrated", scenario_calibrated), ("calibrated", scenario_precision), ("calibrated", scenario_stand_in),
             ("calibrated", scenario_hum_and_rest_note), ("calibrated", scenario_raw), ("calibrated", scenario_short_window),
             ("calibrated", scenario_no_push), ("volts", scenario_volts), ("volts", scenario_raw), ("volts", scenario_hum_and_rest_note),
             ("relaxed", scenario_relaxed), ("relaxed", scenario_relaxed_mismatch), ("relaxed", scenario_relaxed_other_position)]


@pytest.mark.skipif(ForcePanel is None, reason="the old ForcePanel has been deleted: nothing to compare with")
@pytest.mark.parametrize("which, scenario", SCENARIOS, ids=[f"{w}-{s.__name__[9:]}" for w, s in SCENARIOS])
def test_every_fact_of_the_old_force_panel_is_in_the_new_tree(request, app, monkeypatch, which, scenario):
    """Three different trials (calibrated with a torque, volts only, with a relaxed Zero 2) and a few variations of each: every number with its
    unit, every row value and every warning and note of ForcePanel.text(everything=True) is in ForceTree.text(everything=True)."""
    tr = request.getfixturevalue(which)
    tr, window, change = scenario(tr, monkeypatch)
    tr.__dict__.pop("_rest_cache", None)
    fs, ft, fp = numbers(tr, window)
    ft_arg = fp_arg = "real"
    if change == "no_push":
        ft_arg, fp_arg = None, analysis.force_peaks(tr, *window, timing=None)
    elif change is not None:
        fs = change(fs)
    old = ForcePanel()
    old.set_data(tr, window[0], window[1], fs, ft if ft_arg == "real" else None, fp if fp_arg == "real" else fp_arg, 0.0)
    tree = ForceTree(controls())
    tree.set_data(tr, window[0], window[1], fs, ft if ft_arg == "real" else None, fp if fp_arg == "real" else fp_arg, 0.0)
    old_text, new_text = old.text(everything=True), tree.text(everything=True)
    missing = numeric_tokens(old_text) - numeric_tokens(new_text)
    assert not missing, f"numbers of the old text missing from the new: {sorted(missing)}\nOLD:\n{old_text}\nNEW:\n{new_text}"
    new_lines = new_text.splitlines()
    for line in old_text.splitlines():                                                                   # the caption and the warnings, word for word
        if line.startswith("Warning: ") or line.startswith("Force in the window"):
            assert line in new_lines, line
    for section in old.sections.values():                                                                # every value the old rows showed
        for label, value, note, _state in section.grid.rows:
            if value not in ("-", "not available", "not used") and not value.endswith(" deg"):
                assert value in new_text, f"{label}: {value}"
            if label in ("Calibration", "Regression", "Newtons", "Rest before", "Relaxed (Zero 2)", "Moment arm", "Stand-in dot", "Torque") or "above the line" in label:
                assert note in new_text or numeric_tokens(note) <= numeric_tokens(new_text), f"{label}: {note}"
    if "Torque: not available" in old_text:                                                             # the torque's reason, in the same words
        reason = [r for r in old.sections["Torque"].grid.rows][0][2]
        assert f"not available  {reason}" in new_text
    for angle in re.findall(r"-?\d+\.\d deg", old_text):                                                 # the angle at the peak, read at the picture delay
        assert angle in new_text, angle
    for phrase in ("the force is not calibrated", "no zero was taken: use Apply zero to get newtons", "no precision can be given", "above the line",
                   "Relaxed (Zero 2)", "Peak above relaxed", "lower than that", "higher than that", "another position"):
        if phrase in old_text:
            assert phrase in new_text, phrase
    assert "no target" not in new_text.lower()
    old.close()


def test_the_wording_is_plain_in_every_case(request, app, monkeypatch):
    """No word of the angle table ('no target'), no 'nan' or 'None' that leaked out of a missing number, no double spaces inside a row's words."""
    for which, scenario in SCENARIOS:
        with monkeypatch.context() as mp:
            tr = request.getfixturevalue(which)
            tr, window, change = scenario(tr, mp)
            tr.__dict__.pop("_rest_cache", None)
            fs, ft, fp = numbers(tr, window)
            if callable(change):
                fs = change(fs)
            tree = ForceTree(controls())
            tree.set_data(tr, window[0], window[1], fs, None if change == "no_push" else ft, fp, 0.0)
            text = tree.text(everything=True)
            assert not re.search(r"\b(nan|NaN|None|inf)\b", text) and "no target" not in text.lower() and "—" not in text, (which, scenario.__name__)
            for item_text in re.findall(r"^\s*[^:\n]+: ", text, re.M):
                assert "  " not in item_text.strip()


# ------------------------------------------------------------------------------------------------ the look
def test_light_and_dark_palettes_both_work_and_the_colours_follow_the_palette(calibrated, palette):
    fs, _ft, _fp = numbers(calibrated)
    tree = show(make(calibrated, fs=dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0)), 600, 600)
    css, text = tree.styleSheet(), tree.text(everything=True)
    note, band = colour(tree.items["Smoothed peak"], 2), tree.rows["Peak force"].background(0).color().name()
    assert note == muted_colour(QApplication.palette())
    dark(palette)
    QApplication.processEvents()
    assert tree.text(everything=True) == text and tree.styleSheet() != css                                  # the sheet follows the palette
    assert colour(tree.items["Smoothed peak"], 2) == muted_colour(QApplication.palette()) != note
    assert tree.rows["Peak force"].background(0).color().name() != band                                     # the tint of the bold rows too
    assert tree.rows["Peak torque"].foreground(1).style() == Qt.NoBrush and tree.warnings                  # (a plain value takes the palette's own text colour)
    fresh = show(make(calibrated, fs=dict(fs, mains_hum_v=0.09, mains_hum_pct_of_peak=10.0)), 600, 600)
    assert fresh.text(everything=True) == text and fresh.styleSheet() == tree.styleSheet()
    fresh.close()
    tree.close()


def test_dark_text_is_readable_against_its_background(calibrated, palette):
    dark(palette)
    tree = show(make(calibrated), 600, 600)
    base = QColor(24, 24, 24)
    for key in ("Smoothed peak", "Time to peak"):
        for col in (1, 2):
            c = QColor(colour(tree.items[key], col)) if tree.items[key].foreground(col).style() != Qt.NoBrush else QColor(230, 230, 230)
            assert abs(c.lightness() - base.lightness()) > 80, (key, col)
    tree.close()


@pytest.mark.parametrize("width", [520, 700, 900])
def test_the_columns_fill_the_width_with_no_sideways_scroll(calibrated, app, width):
    tree = make(calibrated)
    tree.open_all_link.click()
    show(tree, width, 600)
    view = tree.tree
    head = view.header()
    assert sum(head.sectionSize(c) for c in range(3)) <= view.viewport().width() and head.sectionSize(2) >= 200
    assert view.horizontalScrollBar().maximum() == 0 and not view.horizontalScrollBar().isVisible()
    assert tree.width() == width and view.width() <= width
    tree.close()


def test_it_is_usable_at_520_and_never_asks_for_more_room(calibrated, app):
    fs, _ft, _fp = numbers(calibrated)
    short = make(calibrated)
    before = short.minimumSizeHint()
    assert before.width() <= 520 and before.height() <= 200 and short.sizeHint().width() <= 520
    calibrated.force_cal_note = "a calibration note that goes on and on " * 20
    huge = dict(fs, main_angle="A joint angle with a name that goes on and on " * 8, rest_note="The force was not steady before the push, and more. " * 8,
                mains_hum_v=0.09, mains_hum_pct_of_peak=10.0, relaxed=100.0, peak_minus_relaxed=100.0, rest_minus_relaxed=-100.0, relaxed_note="x " * 200)
    ctl = controls()
    ctl["zero_label"].setText("a very long sentence about where the zero came from " * 12)
    tree = make(calibrated, fs=huge, ctl=ctl)
    tree.open_all_link.click()
    assert tree.minimumSizeHint() == before
    host = QWidget()                                                                                     # in a layout, the window's minimum width does not grow either
    QVBoxLayout(host).addWidget(tree)
    host.layout().setContentsMargins(0, 0, 0, 0)
    assert host.minimumSizeHint().width() <= 520
    show(host, 520, 600)
    assert host.width() == 520 and tree.width() == 520
    view = tree.tree.viewport()
    for key in ("sliding", "smoothing", "delay"):
        w = ctl[key]
        assert w.isVisible() and w.mapTo(view, QPoint(w.width(), 0)).x() <= view.width()
    assert tree.tree.horizontalScrollBar().maximum() == 0
    assert all(w.geometry().right() < view.width() for w in tree.warnings)
    host.close()


def test_a_scroll_bar_of_its_own_comes_only_when_the_room_is_too_short(calibrated, app):
    tree = make(calibrated)
    tree.open_all_link.click()
    show(tree, 520, 240)
    bar = tree.tree.verticalScrollBar()
    assert bar.maximum() > 0 and bar.isVisible() and tree.tree.horizontalScrollBar().maximum() == 0
    tree.resize(520, 1500)
    QApplication.processEvents()
    QApplication.processEvents()
    assert bar.maximum() == 0 and not bar.isVisible()
    tree.close()


def test_the_tree_fills_the_height_it_is_given(calibrated, app):
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    tree = make(calibrated)
    lay.addWidget(tree)
    show(host, 600, 500)
    assert tree.height() == 500 and tree.tree.geometry().bottom() >= 490
    host.resize(600, 300)
    QApplication.processEvents()
    assert tree.height() == 300 and tree.tree.geometry().bottom() >= 290
    host.close()


def test_calling_set_data_again_changes_nothing_that_can_be_seen(calibrated, app):
    fs, ft, fp = numbers(calibrated)
    tree = show(make(calibrated), 700, 500)
    tree.open_all_link.click()
    first, children = tree.text(everything=True), len(tree.findChildren(QWidget))
    rects = {k: tree.tree.visualItemRect(i) for k, i in tree.items.items() if not i.isHidden() and i.parent() is None or i.parent() is not None and i.parent().isExpanded()}
    tree.set_data(calibrated, *WINDOW, fs, ft, fp, 0.0)
    tree.set_data(calibrated, *WINDOW, fs, ft, fp, 0.0)
    QApplication.processEvents()
    assert tree.text(everything=True) == first and len(tree.findChildren(QWidget)) == children
    assert all(tree.tree.visualItemRect(tree.items[k]) == r for k, r in rects.items())
    tree.close()


def test_the_picture_delay_is_in_the_note_of_its_row(calibrated, app):
    main = next(a for a in calibrated.angle_defs if a.primary)
    fs, _ft, _fp = numbers(calibrated)
    tree = make(calibrated)
    assert f"angle at the peak {fs['main_angle_at_peak_deg']:.1f} deg ({main.title})" in row(tree, "Picture delay")[2] and "pictures read" not in row(tree, "Picture delay")[2]
    fs, _ft, _fp = numbers(calibrated, delay=0.05)
    delayed = make(calibrated, delay=0.05)
    assert "pictures read +0.050 s after the peak" in row(delayed, "Picture delay")[2]
    unknown = make(calibrated, fs=dict(fs, main_angle_at_peak_deg=math.nan, main_angle=""))
    assert "angle at the peak not known" in row(unknown, "Picture delay")[2]
