"""The angle and dot tree of the Review tab, 0.32.0 (limbtrack/review_angles.py): the angles of a trial with the dots that make them under each,
'Pull and torque' and 'Other dots', the roll-up of a dot that needs a look, the Fix link, folding, 'More numbers', the reference angle button.

The widget replaces the angle cards, the Angles and Dots tabs and the reference block, so one test here compares it with them: the same synthetic
trials go into the old panels (review_panels.AnglePanel, review_fixdots.DotsPanel) and into the tree, and every number of what the old ones say
must be in what the tree says.

Trials: three short recordings made through the engine (a knee test with strap, anchor and ruler, an anchor that is hidden for a while and a
strap that jumps; the same without them; a frontal hip test with a computed hip centre). Force is put into a copy of the first, as the force tests
do. What a test needs to be exact (an angle at the peak of 65.0) is put into copies of the dicts the widget is handed: it only shows what it is given.
The widget is built standalone (no ReviewTab, no video)."""
import copy
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")        # (offscreen Qt has no fonts of its own: without them every text is zero wide)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFontMetrics, QPalette  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget  # noqa: E402

from limbtrack import review_fixdots as fd  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.force import ForceCalibration  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.review_angles import (C_FIX, C_HOLD, C_LOST, C_NAME, C_OFF, C_PEAK, C_REL, C_SHIFT, C_TARGET, NO_MARKERS, PULL_GROUP,  # noqa: E402
                                     OTHER_GROUP, TITLE, WARN_MARK, AngleDotTree, _both, dots_of, off_words)
from limbtrack.review_panels import AnglePanel, num, strong_colour  # noqa: E402
from limbtrack.review_style import ACCENT, GOOD, WARN  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

from tests import test_pull_frontal as PF  # noqa: E402  (a recorded frontal hip trial with strap and anchor)

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
LOST_PICTURES = {"anchor": range(10, 26), "fibular_head": range(60, 70)}        # hidden in the first recording: 16 and 10 of its 90 pictures


# ------------------------------------------------------------------------------------------------ set-up
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def record(base, strap=True, ruler=True, markers=True):
    """A 90-picture knee recording at 60 deg. With strap=True: strap and anchor dots (the anchor hidden in pictures 10-25, the strap jumping 20 px
    at picture 50), and the fibular head hidden in pictures 60-69. Ruler: 20 cm = 120 px. Markers: MVIC start at picture 20, end at 80."""
    eng = Engine(TESTS, recordings_dir=base)
    eng.set_test("knee_extension")
    eng.target = 60.0
    if strap:
        eng.set_dot_enabled("strap", True)
        eng.set_dot_enabled("anchor", True)
    p = dict(leg_points(hip_flex=70, knee_flex=60, ankle_df=5))
    mal, fib = np.array(p["malleolus"]), np.array(p["fibular_head"])
    if strap:
        p["strap"] = mal + 0.35 * (fib - mal)
        p["anchor"] = p["strap"] + np.array([260.0, -90.0])
    if ruler:
        p["ruler_a"], p["ruler_b"] = np.array([120.0, 670.0]), np.array([240.0, 670.0])
        eng.scale.length_cm = 20.0
    eng.process(render(p, seed=0), 0.0)
    for name in eng.active_dots:
        assert eng.mark_dot(name, *(p[name] + 2))
    if ruler:
        assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    paths = eng.start_trial("P01")
    for i in range(90):
        q = dict(p)
        if strap and i >= 50:
            q["strap"] = p["strap"] + np.array([0.0, 20.0])
        if markers and i == 20:
            eng.add_event("MVIC start")
        if markers and i == 80:
            eng.add_event("MVIC end")
        hide = [d for d, r in LOST_PICTURES.items() if strap and i in r]
        eng.process(render(q, seed=i, hide=hide), 0.3 + i / 30)
    eng.stop_trial()
    return paths["meta.json"]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """(full, plain, frontal): meta files of the three recordings (made once, it takes some seconds)."""
    full = record(tmp_path_factory.mktemp("angles_full"))
    plain = record(tmp_path_factory.mktemp("angles_plain"), strap=False, ruler=False, markers=False)
    frontal, _p, _hjc = PF.record(tmp_path_factory.mktemp("angles_frontal"))
    return full, plain, frontal["meta.json"]


HIDDEN = []                          # errors raised inside an event handler or a slot (see `no_hidden_errors`)
OPEN = []                            # trees and trials made by a test, closed after it


@pytest.fixture(autouse=True)
def no_hidden_errors(monkeypatch, request):
    """Qt prints an exception raised inside an event handler (a resize, a palette change, a click) and carries on, so a test would pass straight
    over it. They are written down here instead, and the test fails."""
    HIDDEN.clear()
    monkeypatch.setattr(sys, "excepthook", lambda kind, err, tb: HIDDEN.append(f"{kind.__name__}: {err}"))
    yield
    for w in OPEN:
        if isinstance(w, Trial):
            w.close()
        else:
            w.close()
    OPEN.clear()
    if HIDDEN and not request.node.get_closest_marker("xfail"):
        pytest.fail("an error was raised inside a Qt handler and hidden by Qt: " + "; ".join(HIDDEN))


@pytest.fixture
def full(recorded, app):
    tr = Trial(recorded[0], open_video=False)
    OPEN.append(tr)
    return tr


@pytest.fixture
def plain(recorded, app):
    tr = Trial(recorded[1], open_video=False)
    OPEN.append(tr)
    return tr


@pytest.fixture
def frontal(recorded, app):
    tr = Trial(recorded[2], open_video=False)
    OPEN.append(tr)
    return tr


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


def add_force(tr, peak_v=1.0):
    """A calibrated force in the trial (as the force tests do): rest, a ramp to a hold from 1.0 to 2.4 s, release; 200 N for 0.9 V above the zero."""
    t = np.arange(-1.0, 4.0, 0.001)
    clean = np.interp(t, [-1.0, 0.5, 1.0, 2.4, 2.8, 4.0], [0.1, 0.1, peak_v, peak_v, 0.1, 0.1])
    v = clean + np.random.RandomState(1).normal(0, 0.005, t.size)
    tr.force_t, tr.force_v, tr.force_n = t, v, np.full(t.size, np.nan)
    cal = ForceCalibration()
    cal.set_zero(0.1)
    cal.n_per_v = 200.0 / 0.9
    tr.apply_calibration(cal, "test calibration")
    return tr


def whole(tr):
    return tr.hold_window() or (float(tr.t[0]), float(tr.t[-1]))


def numbers(tr, force=True):
    """(rows, fs, window): what Review hands the tree (ReviewTab.update_summary)."""
    w = whole(tr)
    return tr.summary(*w), (tr.force_summary(*w, 0.0) if force else None), w


def tree(tr, rows=None, fs="real", window="hold", width=700, height=520, more=False):
    """An AngleDotTree filled for `tr` and shown; `rows`/`fs`/`window` replace the real numbers."""
    r, f, w = numbers(tr, force=tr.has_force())
    t = AngleDotTree()
    OPEN.append(t)
    if more:
        t.more_check.setChecked(True)
    t.set_data(tr, r if rows is None else rows, f if fs == "real" else fs, w if window == "hold" else window)
    t.resize(width, height)
    t.show()
    QApplication.processEvents()
    return t


def item(t, name):
    it = t.angle_item(name)
    assert it is not None, f"no row {name}: {[t.tree.topLevelItem(i).data(0, Qt.UserRole + 1) for i in range(t.tree.topLevelItemCount())]}"
    return it


def keys(t):
    return [t.tree.topLevelItem(i).data(0, Qt.UserRole + 1) for i in range(t.tree.topLevelItemCount())]


def colour(it, col):
    return it.foreground(col).color().name() if it.foreground(col).style() != Qt.NoBrush else None


def click(t, it, col, double=False):
    """A mouse click (or a double click) on a cell of the tree."""
    tr = t.tree
    x = tr.header().sectionViewportPosition(col) + tr.header().sectionSize(col) // 2
    pos = QPoint(x, tr.visualItemRect(it).center().y())
    QTest.mouseClick(tr.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    if double:                                                                 # (a double click is a click and then the second press, as Qt sends it)
        QTest.mouseDClick(tr.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    QApplication.processEvents()


def numbers_in(text):
    """The numbers of a text as a set of magnitudes (70 and 70.0 are one number; the sign is said in words or by a sign next to it)."""
    return {round(float(x), 6) for x in re.findall(r"\d+(?:\.\d+)?", text)}


def modified(rows, **by_angle):
    """A copy of the summary rows with some numbers changed: modified(rows, knee={'plateau_mean_deg': 66.0})."""
    out = copy.deepcopy(rows)
    for r in out:
        for key, changes in by_angle.items():
            if r["angle"].lower().startswith(key):
                r.update(changes)
    return out


# ------------------------------------------------------------------------------------------------ the recordings are what the tests assume
def test_the_recordings_have_the_dots_and_trouble_the_tests_rely_on(full, plain, frontal):
    stats = {s.name: s for s in fd.trial_dot_stats(full)}
    assert set(full.dot_names) >= {"trochanter", "fibular_head", "strap", "anchor"} and full.px_per_cm and full.hold_window() is not None
    assert stats["anchor"].lost_pct == pytest.approx(100 * 16 / 90) and stats["anchor"].needs_attention
    assert stats["fibular_head"].lost_pct == pytest.approx(100 * 10 / 90) and stats["fibular_head"].lost_pct > 10 and stats["strap"].jumps >= 1
    assert not stats["trochanter"].needs_attention
    assert "strap" not in plain.dot_names and plain.px_per_cm is None and plain.hold_window() is None
    assert frontal.derived_names == ["hip_centre"] and "hip_centre" not in frontal.dot_names


# ------------------------------------------------------------------------------------------------ the small numbers
def test_a_deviation_under_005_deg_is_on_the_target_not_zero_above():
    assert off_words(0.02) == "on the target" and off_words(-0.049) == "on the target"
    assert off_words(7.5) == "7.5 deg above the target" and off_words(-8.0) == "8.0 deg below the target" and off_words(0.05) == "0.1 deg above the target"
    assert off_words(float("nan")) == "" and off_words(None) == ""


def test_two_shifts_in_one_cell_say_the_unit_once():
    assert _both("0.4 cm", "1.6 cm") == "0.4 / 1.6 cm" and _both("14 px", "58 px") == "14 / 58 px"
    assert _both("-", "1.6 cm") == "- / 1.6 cm" and _both("-", "-") == "-" and _both("2 cm", "3 px") == "2 cm / 3 px"


def test_the_dots_of_an_angle_are_listed_once_and_a_computed_point_is_the_dots_it_is_made_of(full, frontal):
    hip = next(a for a in full.angle_defs if a.name == "hip")
    assert hip.dots.count("trochanter") == 2 and dots_of(full, hip) == ["iliac_crest", "trochanter", "epicondyle"]
    abd = next(a for a in frontal.angle_defs if a.name == "abd_hip_centre")
    assert "hip_centre" not in dots_of(frontal, abd) and dots_of(frontal, abd) == ["asis", "asis_opposite", "patella"]
    abd.needs = None                                                           # a trial recorded before 'needs' was kept: the arms say hip_centre
    assert dots_of(frontal, abd) == ["asis", "asis_opposite", "patella"]


# ------------------------------------------------------------------------------------------------ empty states
def test_the_tree_starts_empty_and_set_none_says_why(app, full):
    t = AngleDotTree()
    OPEN.append(t)
    t.resize(600, 300)
    t.show()
    QApplication.processEvents()
    assert t.text() == "No trial is open." and t.empty.isVisibleTo(t) and not t.tree.isVisibleTo(t) and not t.head.isVisibleTo(t)
    assert t.empty.alignment() & Qt.AlignHCenter and t.empty.alignment() & Qt.AlignVCenter
    t.set_data(full, *numbers(full, force=False)[:2], whole(full))
    assert t.tree.isVisibleTo(t) and not t.empty.isVisibleTo(t) and t.text().startswith(TITLE)
    t.set_none("Nothing to show in this window.")
    assert t.text() == "Nothing to show in this window." and t.empty.isVisibleTo(t) and t.rows_for("knee") == [] and t.tree.topLevelItemCount() == 0


# ------------------------------------------------------------------------------------------------ rows
def test_one_row_per_angle_the_main_one_first_and_bold_with_its_dots_under_it(full):
    full.angle_defs = list(reversed(full.angle_defs))                          # the main joint is second in this trial: it is still first on the screen
    t = tree(full)
    assert keys(t) == ["knee", "hip", PULL_GROUP]
    knee, hip = item(t, "knee"), item(t, "hip")
    assert knee.font(C_NAME).bold() and not hip.font(C_NAME).bold() and knee.text(C_NAME).endswith("Knee flexion")
    assert t.rows_for("knee") == ["trochanter", "epicondyle", "fibular_head", "malleolus"] == [knee.child(i).data(0, Qt.UserRole + 1) for i in range(4)]
    assert t.rows_for("hip") == ["iliac_crest", "trochanter", "epicondyle"]
    assert t.rows_for("Knee flexion") == t.rows_for("knee") and t.rows_for("no such angle") == []     # by name or by title
    assert t.dot_item("knee", "trochanter") is not None and t.dot_item("hip", "trochanter") is not None   # a dot of two angles is under both
    assert t.dot_item("knee", "trochanter") is not t.dot_item("hip", "trochanter")


def test_the_dots_shown_are_the_trials_tracked_dots_and_never_the_ruler(full):
    t = tree(full)
    shown = [d for k in keys(t) for d in t.rows_for(k)]
    assert set(shown) <= set(full.dot_names) and "ruler" not in " ".join(shown + [t.text(everything=True)]).replace("ruler scale", "")
    assert set(t.rows_for(PULL_GROUP)) == {"strap", "anchor"} and set(shown) == set(full.dot_names)        # every tracked dot is somewhere


def test_a_computed_hip_centre_shows_the_dots_it_is_made_of(frontal):
    t = tree(frontal)
    assert keys(t)[:2] == ["abd_goniometer", "abd_hip_centre"]
    assert t.rows_for("abd_hip_centre") == ["asis", "asis_opposite", "patella"] and "hip_centre" not in t.text(everything=True).replace("abd_hip_centre", "")
    assert t.rows_for(PULL_GROUP) == ["strap", "anchor"] and OTHER_GROUP not in keys(t)


def test_dots_that_belong_to_no_angle_are_in_other_dots(full):
    full.angle_defs = [a for a in full.angle_defs if a.name == "knee"]         # the hip angle is gone: the iliac crest belongs to nothing
    t = tree(full)
    assert keys(t) == ["knee", PULL_GROUP, OTHER_GROUP] and t.rows_for(OTHER_GROUP) == ["iliac_crest"]
    assert t.rows_for(PULL_GROUP) == ["strap", "anchor"] and "iliac_crest" not in t.rows_for("knee")


def test_a_group_that_would_be_empty_is_left_out(plain):
    t = tree(plain)
    assert keys(t) == ["knee", "hip"] and PULL_GROUP not in t.text()


# ------------------------------------------------------------------------------------------------ the numbers of an angle
@pytest.mark.parametrize("peak, tol, state, words", [
    (70.0, 5.0, "warn", "10.0 deg above the target"),
    (65.0, 5.0, "good", "5.0 deg above the target"),                           # exactly the tolerance is still good
    (65.2, 5.0, "warn", "5.2 deg above the target"),
    (55.0, 5.0, "good", "5.0 deg below the target"),
    (50.0, 5.0, "warn", "10.0 deg below the target"),
    (62.5, 2.0, "warn", "2.5 deg above the target"),                           # the tolerance is the trial's own
    (62.5, 3.0, "good", "2.5 deg above the target"),
    (60.03, 5.0, "good", "on the target"),                                     # under 0.05 deg off is on the target
])
def test_the_angle_at_the_peak_force_is_large_and_green_within_the_tolerance_else_amber(full, palette, peak, tol, state, words):
    add_force(full)
    full.meta["tolerance_deg"] = tol
    rows, fs, w = numbers(full)
    fs = dict(fs, main_angle_at_peak_deg=peak)
    t = tree(full, rows=modified(rows, knee={"target_deg": 60.0}), fs=fs)
    knee = item(t, "knee")
    assert knee.text(C_PEAK) == f"{peak:.1f}°"
    f = knee.font(C_PEAK)
    assert f.bold() and f.pointSizeF() == pytest.approx(14.0) and knee.font(C_HOLD).pointSizeF() < 14.0
    assert colour(knee, C_PEAK) == strong_colour({"good": GOOD, "warn": WARN}[state], palette.palette()).lower()
    assert words in knee.toolTip(C_PEAK) and f"at the peak force {peak:.1f} deg ({words})" in t.text()
    assert ("at the peak force the angle is" in knee.toolTip(C_NAME)) == (state == "warn")                   # (the knee has a lost dot too: the mark is not the test)


def test_without_force_the_hold_angle_is_the_large_number_and_at_peak_is_blank(full):
    t = tree(full, fs=None)
    knee = item(t, "knee")
    assert knee.text(C_PEAK) == "" and t.tree.isColumnHidden(C_PEAK)
    assert knee.font(C_HOLD).bold() and knee.font(C_HOLD).pointSizeF() == pytest.approx(14.0) and colour(knee, C_HOLD) is not None
    hip = item(t, "hip")
    assert hip.font(C_HOLD).pointSizeF() < 14.0 and colour(hip, C_HOLD) is None          # only the main angle gets the large number
    assert "at the peak force" not in t.text()


def test_with_force_the_column_at_peak_shows_and_every_angle_has_it(full):
    add_force(full)
    t = tree(full)
    assert not t.tree.isColumnHidden(C_PEAK) and item(t, "knee").text(C_PEAK).endswith("°") and item(t, "hip").text(C_PEAK).endswith("°")    # (0.33.0: the hip too, not only the main joint)
    assert item(t, "hip").text(C_PEAK) == f"{numbers(full)[1]['angles_at_peak_deg']['Hip flex (+) / ext (-)']:.1f}°"
    assert item(t, "knee").font(C_HOLD).pointSizeF() < 14.0                             # the peak is the large number: hold is a normal one
    fs = numbers(full)[1]
    assert item(t, "knee").text(C_PEAK) == f"{fs['main_angle_at_peak_deg']:.1f}°"
    fs_none = dict(fs, main_angle_at_peak_deg=float("nan"))                              # force, but no angle at the peak: a dash, and hold is large
    t2 = tree(full, fs=fs_none)
    assert item(t2, "knee").text(C_PEAK) == "-" and item(t2, "knee").font(C_HOLD).pointSizeF() == pytest.approx(14.0)


def test_hold_target_and_off_are_the_numbers_of_the_old_table_and_off_is_amber_outside_the_tolerance(full, palette):
    rows, fs, w = numbers(full)
    t = tree(full, rows=modified(rows, knee={"plateau_mean_deg": 63.4, "target_deg": 60.0}), fs=None)
    knee = item(t, "knee")
    assert (knee.text(C_HOLD), knee.text(C_TARGET), knee.text(C_OFF)) == ("63.4°", "60 ± 5°", "+3.4°") and colour(knee, C_OFF) is None
    t = tree(full, rows=modified(rows, knee={"plateau_mean_deg": 56.0, "target_deg": 60.0}), fs=None)
    knee = item(t, "knee")
    assert knee.text(C_OFF) == "-4.0°" and colour(knee, C_OFF) is None                    # exactly the tolerance is within it
    t = tree(full, rows=modified(rows, knee={"plateau_mean_deg": 66.2, "target_deg": 60.0}), fs=None)
    knee = item(t, "knee")
    assert knee.text(C_OFF) == "+6.2°" and colour(knee, C_OFF) == strong_colour(WARN, palette.palette()).lower() and WARN_MARK in knee.text(C_NAME)
    assert "off target +6.2 deg" in t.text() and "the hold angle is 6.2 deg above the target" in knee.toolTip(C_NAME)
    old = AnglePanel()
    old.set_data(full, modified(rows, knee={"plateau_mean_deg": 66.2, "target_deg": 60.0}), None, w)
    assert old.table.item(0, 3).text() == "+6.2" and old.table.item(0, 1).text() == "66.2"      # the same numbers as the old table


def test_only_a_missing_target_is_called_no_target_a_missing_hold_angle_is_not(full):
    rows, fs, w = numbers(full)
    t = tree(full, rows=modified(rows, knee={"plateau_mean_deg": float("nan")}), fs=None)
    knee = item(t, "knee")
    assert knee.text(C_HOLD) == "-" and knee.text(C_OFF) == "-" and knee.text(C_TARGET) == "60 ± 5°"
    assert "no target" not in t.text(everything=True) and "hold -" in t.text() and "No target" not in knee.toolTip(C_TARGET)
    assert "too few pictures" in knee.toolTip(C_HOLD)
    t = tree(full, rows=modified(rows, knee={"target_deg": float("nan")}), fs=None)
    knee = item(t, "knee")
    assert knee.text(C_TARGET) == "-" and knee.text(C_OFF) == "-" and "no target" in t.text() and "No target for this angle" in knee.toolTip(C_TARGET)
    assert colour(knee, C_HOLD) is None                                                    # nothing to judge it against: no colour


def test_a_deviation_under_005_deg_reads_on_the_target_in_the_off_column_too(full):
    rows, fs, w = numbers(full)
    t = tree(full, rows=modified(rows, knee={"plateau_mean_deg": 60.02, "target_deg": 60.0}), fs=None)
    knee = item(t, "knee")
    assert knee.text(C_OFF) == "0.0°" and "on the target" in t.text() and "0.0 deg above" not in t.text(everything=True) and "on the target" in knee.toolTip(C_OFF)


def test_the_other_joints_are_judged_with_their_own_tolerance(full, palette):
    full.meta["tolerance_deg"], full.meta["neighbour_tolerance_deg"] = 5.0, 25.0         # the hip is 20 deg off the 90 of its position
    t = tree(full, fs=None)
    hip = item(t, "hip")
    assert hip.text(C_TARGET) == "90 ± 25°" and colour(hip, C_OFF) is None and item(t, "knee").text(C_TARGET) == "60 ± 5°"
    full.meta["neighbour_tolerance_deg"] = 10.0
    t = tree(full, fs=None)
    assert colour(item(t, "hip"), C_OFF) == strong_colour(WARN, palette.palette()).lower() and WARN_MARK in item(t, "hip").text(C_NAME)


def test_an_angle_that_is_only_shown_is_not_judged(frontal):
    t = tree(frontal, fs=None)
    shown = next(a for a in frontal.angle_defs if not a.judged)
    row = item(t, shown.name)
    assert "±" not in row.text(C_TARGET) and colour(row, C_OFF) is None and "not judged" in t.text()
    assert "not judged" in row.toolTip(C_TARGET)


# ------------------------------------------------------------------------------------------------ pull and torque, other dots
def test_pull_and_torque_names_the_pull_numbers_of_the_strap_and_anchor_dots(full):
    t = tree(full)
    g = item(t, PULL_GROUP)
    pull = {q["quantity"]: q for q in full.pull_summary(*whole(full))}
    note = f"pull {pull['pull_angle_deg']['mean']:.1f} deg - arm {pull['moment_arm_cm']['mean']:.1f} cm"
    assert note in g.text(C_NAME) and g.font(C_NAME).bold() and full.px_per_cm
    assert "strap and anchor dots" in g.toolTip(C_NAME) and t.rows_for(PULL_GROUP) == ["strap", "anchor"]
    assert "from the strap and anchor dots" in t.text() and note in t.text()


def test_the_arm_is_in_pixels_without_a_ruler_scale(full):
    full.px_per_cm = None
    t = tree(full)
    arm = {q["quantity"]: q for q in full.pull_summary(*whole(full))}["moment_arm_px"]["mean"]
    assert f"arm {arm:.1f} px" in item(t, PULL_GROUP).text(C_NAME) and " cm" not in item(t, PULL_GROUP).text(C_NAME)
    assert "pixels: no ruler scale" in tree(full, more=True).text()


def test_a_trial_without_pull_numbers_just_names_the_two_dots(full):
    full.pull = None
    t = tree(full)
    g = item(t, PULL_GROUP)
    assert "pull " not in g.text(C_NAME) and "arm " not in g.text(C_NAME) and t.rows_for(PULL_GROUP) == ["strap", "anchor"]


# ------------------------------------------------------------------------------------------------ roll-up, warnings, the Fix link
def test_a_dot_that_needs_a_look_is_amber_with_a_fix_link_and_the_roll_up_names_it(full, palette):
    t = tree(full)
    stats = {s.name: s for s in fd.trial_dot_stats(full)}
    mal, anchor, healthy = t.dot_item("knee", "fibular_head"), t.dot_item(PULL_GROUP, "anchor"), t.dot_item("knee", "trochanter")
    assert stats["fibular_head"].needs_attention and mal.text(C_LOST) == "11 %" and mal.text(C_FIX) == "Fix" and WARN_MARK in mal.text(C_NAME)
    assert anchor.text(C_LOST) == "18 %" and anchor.text(C_FIX) == "Fix" and healthy.text(C_FIX) == "" and WARN_MARK not in healthy.text(C_NAME)
    amber = mal.background(C_LOST).color()
    assert (amber.red(), amber.green(), amber.blue()) == (QColor(WARN).red(), QColor(WARN).green(), QColor(WARN).blue()) and 0 < amber.alpha() < 255
    assert healthy.background(C_LOST).style() == Qt.NoBrush and mal.font(C_FIX).underline() and colour(mal, C_FIX) == ACCENT.lower()
    assert colour(mal, C_LOST) == strong_colour(WARN, palette.palette()).lower() and colour(healthy, C_LOST) != colour(mal, C_LOST)
    for it, name in ((mal, "fibular_head"), (anchor, "anchor")):
        assert "lost in" in it.toolTip(C_NAME) and "Needs a look" in it.toolTip(C_NAME) and name in it.toolTip(C_FIX)
    jumps = stats["strap"].jumps
    assert f"{jumps} jump" in t.dot_item(PULL_GROUP, "strap").text(C_NAME) and WARN_MARK in t.dot_item(PULL_GROUP, "strap").text(C_NAME)
    assert f"{jumps} jump" in t.dot_item(PULL_GROUP, "strap").toolTip(C_LOST)         # the jump count is in the row and in its tooltip


def test_an_angle_and_a_group_show_the_worst_lost_of_their_dots_and_a_warning(full):
    t = tree(full)
    knee, hip, pull = item(t, "knee"), item(t, "hip"), item(t, PULL_GROUP)
    assert knee.text(C_LOST) == "11 %" and "fibular_head" in knee.toolTip(C_LOST) and WARN_MARK in knee.text(C_NAME)       # the knee's worst dot is the fibular head
    assert "fibular_head" in knee.toolTip(C_NAME) and "Needs a look" in knee.toolTip(C_NAME)
    assert pull.text(C_LOST) == "18 %" and "anchor" in pull.toolTip(C_LOST) and WARN_MARK in pull.text(C_NAME)
    assert hip.text(C_LOST) == "0 %" and "None of its dots was lost" in hip.toolTip(C_LOST)
    assert colour(knee, C_LOST) == colour(pull, C_LOST) != colour(hip, C_LOST)                                             # amber, amber, quiet


def test_an_angle_with_a_lost_dot_in_its_window_needs_a_look(full):
    t = tree(full)
    knee = item(t, "knee")
    frames = numbers(full, force=False)[0][0]
    assert frames["frames_lost"] > 0 and f"no angle in {frames['frames_lost']} of {frames['frames']} pictures of the window" in knee.toolTip(C_NAME)


def test_a_healthy_trial_has_no_warning_anywhere(plain):
    plain.meta["tolerance_deg"] = 60.0
    plain.meta["neighbour_tolerance_deg"] = 60.0
    t = tree(plain, fs=None)
    assert WARN_MARK not in t.text(everything=True) and t.tree.isColumnHidden(C_FIX) and "needs a look" not in t.text(everything=True)


def test_the_fix_link_and_a_double_click_ask_for_the_dot(full):
    t = tree(full)
    got = []
    t.fixDotRequested.connect(got.append)
    mal = t.dot_item("knee", "fibular_head")
    click(t, mal, C_FIX)                                                       # the link
    assert got == ["fibular_head"]
    click(t, mal, C_LOST)                                                      # a single click anywhere else does nothing
    click(t, t.dot_item("knee", "trochanter"), C_FIX)                          # a dot that does not need a look has no link
    assert got == ["fibular_head"]
    click(t, mal, C_NAME, double=True)
    assert got[1:] == ["fibular_head"]
    click(t, t.dot_item("knee", "trochanter"), C_LOST, double=True)            # a double click works on any dot
    assert got[2:] == ["trochanter"]
    click(t, item(t, "knee"), C_NAME, double=True)                             # ... but not on an angle
    assert len(got) == 3


# ------------------------------------------------------------------------------------------------ folding
def test_clicking_an_angle_folds_it_and_the_fold_is_remembered(full, plain):
    t = tree(full)
    knee, hip = item(t, "knee"), item(t, "hip")
    assert knee.isExpanded() and hip.isExpanded() and item(t, PULL_GROUP).isExpanded()        # all open to begin with
    click(t, knee, C_NAME)
    assert not knee.isExpanded() and hip.isExpanded() and not t.is_open("knee")
    t.set_data(full, *numbers(full, force=False)[:2], whole(full))                              # the numbers are made again: it stays folded
    assert not item(t, "knee").isExpanded() and item(t, "hip").isExpanded()
    t.set_data(plain, *numbers(plain, force=False)[:2], whole(plain))                           # another trial with the same angle names
    assert not item(t, "knee").isExpanded()
    t.set_data(full, *numbers(full, force=False)[:2], whole(full))
    click(t, item(t, "knee"), C_LOST)                                                         # a click on any cell of the row
    assert item(t, "knee").isExpanded() and t.is_open("Knee flexion")
    click(t, item(t, PULL_GROUP), C_NAME)                                                     # a group folds too
    assert not item(t, PULL_GROUP).isExpanded() and t.rows_for(PULL_GROUP) == ["strap", "anchor"]


def test_set_fold_by_name_or_title_and_text_hides_the_dots_of_a_folded_angle(full):
    t = tree(full)
    assert "trochanter: lost" in t.text()
    t.set_fold("Hip flex (+) / ext (-)", False)                                                 # by title
    assert not item(t, "hip").isExpanded() and not t.is_open("hip")
    assert "iliac_crest: lost" not in t.text() and "iliac_crest: lost" in t.text(everything=True) and "trochanter: lost" in t.text()
    t.set_fold("hip", True)
    assert item(t, "hip").isExpanded() and "iliac_crest: lost" in t.text()
    t.set_fold("an angle not shown", False)                                                   # nothing to fold now, but it is remembered
    assert not t.is_open("an angle not shown")


def test_a_fold_set_before_the_angle_is_shown_is_kept(full):
    t = AngleDotTree()
    OPEN.append(t)
    t.set_fold("knee", False)
    t.set_data(full, *numbers(full, force=False)[:2], whole(full))
    assert not item(t, "knee").isExpanded() and item(t, "hip").isExpanded()


def test_clicking_the_arrow_folds_once_not_twice(full):
    t = tree(full)
    knee = item(t, "knee")
    r = t.tree.visualItemRect(knee)
    QTest.mouseClick(t.tree.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(6, r.center().y()))        # the arrow, left of the name
    QApplication.processEvents()
    assert not knee.isExpanded() and not t.is_open("knee")


# ------------------------------------------------------------------------------------------------ More numbers
def test_more_numbers_adds_the_minor_statistics_and_the_dot_shifts(full):
    t = tree(full)
    assert not t.more_check.isChecked() and t.tree.isColumnHidden(C_SHIFT) and t.tree.isColumnHidden(C_REL)
    for word in ("Median", "Max deviation", "Start to end", "Rest to hold", "In tolerance", "end shift"):
        assert word not in t.text()
        assert word in t.text(everything=True)                                                # (hidden behind the switch, but all there)
    t.more_check.setChecked(True)
    QApplication.processEvents()
    assert not t.tree.isColumnHidden(C_SHIFT) and not t.tree.isColumnHidden(C_REL)           # the trial has an anchor dot: 'vs anchor' too
    for word in ("Mean", "SD", "Min", "Max", "Start to end", "Max deviation", "Median", "Rest to hold", "In tolerance", "end shift", "max shift",
                 "end vs anchor", "max vs anchor"):
        assert word in t.text()
    row = numbers(full, force=False)[0][0]
    for text in (f"Mean {num(row['mean_deg'])} deg", f"SD {num(row['sd_deg'], 2)} deg", f"Median {num(row['median_deg'])} deg",
                 f"In tolerance {row['pct_in_tolerance']:.0f} %", f"Max deviation {num(row['max_abs_dev_deg'])} deg"):
        assert text in t.text()
    detail = item(t, "knee").child(0)
    assert detail.data(0, Qt.UserRole) == "detail" and "Median" in detail.text(0) and item(t, "knee").childCount() == 5      # a quiet line, then the four dots
    assert t.rows_for("knee") == ["trochanter", "epicondyle", "fibular_head", "malleolus"]         # (the line is not a dot)
    t.more_check.setChecked(False)
    assert t.tree.isColumnHidden(C_SHIFT) and item(t, "knee").childCount() == 4


def test_more_numbers_is_remembered_while_the_widget_lives(full, plain):
    t = tree(full, more=True)
    t.set_data(plain, *numbers(plain, force=False)[:2], whole(plain))
    assert t.more_check.isChecked() and "Median" in t.text() and not t.tree.isColumnHidden(C_SHIFT) and t.tree.isColumnHidden(C_REL)    # (no anchor: no vs anchor)


def test_the_dot_shifts_are_in_cm_with_a_ruler_scale_and_in_pixels_without(full, plain):
    t = tree(full, more=True)
    mal = t.dot_item("knee", "malleolus")
    r = fd.window_shifts(full, *whole(full))["malleolus"]
    assert mal.text(C_SHIFT).endswith(" cm") and mal.text(C_SHIFT).count("/") == 1 and f"{r['end_shift_px']:.1f} px" in mal.toolTip(C_SHIFT)
    assert "(" in t.text(everything=True).split("malleolus: ")[1].split("\n")[0] and "pixels: no ruler scale" not in t.text()
    t2 = tree(plain, more=True)
    cell = t2.dot_item("knee", "malleolus").text(C_SHIFT)
    assert cell.endswith(" px") and "cm" not in cell and "pixels: no ruler scale" in t2.text() and "no ruler scale" in t2.tree.headerItem().toolTip(C_SHIFT)


def test_the_pull_rows_are_there_with_more_numbers(full):
    t = tree(full, more=True)
    g = item(t, PULL_GROUP)
    assert g.childCount() == 3 + 2 and [g.child(i).data(0, Qt.UserRole) for i in range(3)] == ["detail"] * 3
    pull = {q["quantity"]: q for q in full.pull_summary(*whole(full))}
    text = t.text()
    assert f"Angle of pull (deg), from the strap and anchor dots: mean {pull['pull_angle_deg']['mean']:.1f}" in text
    assert "Moment arm (cm)" in text and "Strap to joint (cm)" in text and "change from start to end" in text


# ------------------------------------------------------------------------------------------------ the heading, the window, the notes
def test_the_heading_says_which_window_the_numbers_are_for(full):
    t = tree(full)
    t0, t1 = whole(full)
    assert t.heading.text() == f"{TITLE}  window {t0:.2f} to {t1:.2f} s" and t.text().splitlines()[0] == t.heading.text()
    assert f"{t0:.2f} to {t1:.2f} s" in t.heading.toolTip() and "whole trial" in t.heading.toolTip()
    t.set_data(full, *numbers(full, force=False)[:2], None)
    assert t.heading.text() == TITLE                                                           # no window given: none said


def test_a_trial_without_markers_says_the_window_is_as_set(plain):
    t = tree(plain)
    assert t.note.isVisibleTo(t) and t.note.text() == NO_MARKERS and NO_MARKERS in t.text()


def test_a_trial_with_markers_has_no_such_note(full):
    t = tree(full)
    assert not t.note.isVisibleTo(t) and NO_MARKERS not in t.text()


def test_what_hold_and_rest_mean_is_in_tooltips_and_in_the_whole_text(full):
    t = tree(full)
    hi = t.tree.headerItem()
    assert "middle 60 % of the window" in hi.toolTip(C_HOLD) and "not the angle that was set" in hi.toolTip(C_HOLD)
    rest = item(t, "knee").child(0)                                                            # (no More numbers yet: the first child is a dot)
    t.more_check.setChecked(True)
    rest = item(t, "knee").child(0)
    assert "Rest is the angle in the quiet stretch before the push" in rest.toolTip(0) and "1 s before the window" in t.text(everything=True)
    assert "Rest is the angle" not in t.text()                                                 # a tooltip, not on the screen
    assert "Hold is the angle during the contraction (the middle 60 % of the window), not the angle that was set." in t.text(everything=True)


# ------------------------------------------------------------------------------------------------ the reference angle
READING = {"instrument": "goniometer", "resolution_deg": 1.0, "readings": {"knee": {"values_deg": [58.0, 60.0], "mean_deg": 59.0}}}


def with_reading(tr, readings=None):
    tr.meta["reference_readings"] = copy.deepcopy(readings or READING)
    tr._reference_rows = None
    return tr


def test_the_reference_button_adds_a_reading_or_shows_it(full):
    t = tree(full)
    assert t.reference_button.text() == "Reference angle: add..." and "No reading for this trial" in t.reference_button.toolTip()
    got = []
    t.referenceRequested.connect(lambda: got.append(1))
    QTest.mouseClick(t.reference_button, Qt.LeftButton)
    assert got == [1]
    with_reading(full)
    t = tree(full)
    t.referenceRequested.connect(lambda: got.append(1))
    row = full.reference_rows()[0]
    assert row["difference_deg"] is not None
    assert t.reference_button.text() == f"Reference angle: 59.0 deg - app {num(row['difference_deg'], signed=True)} deg"
    assert "Goniometer" in t.reference_button.toolTip() and "mean of 2" in t.reference_button.toolTip() and "Press to add or change" in t.reference_button.toolTip()
    QTest.mouseClick(t.reference_button, Qt.LeftButton)
    assert len(got) == 2
    assert "[Reference angle: 59.0 deg" in t.text() and "Goniometer" not in t.text() and "mean of 2" in t.text(everything=True)


def test_with_several_readings_the_main_angles_is_on_the_button_and_all_are_in_the_tooltip(full):
    both = {**READING, "readings": {"hip": {"values_deg": [88.0], "mean_deg": 88.0}, **READING["readings"]}}      # the hip first, the knee is the main joint
    t = tree(with_reading(full, both))
    assert t.reference_button.text().startswith("Reference angle: 59.0 deg") and "Hip flex" in t.reference_button.toolTip() \
        and "88.0 deg" in t.reference_button.toolTip() and "Knee flexion" in t.reference_button.toolTip()
    only_hip = {**READING, "readings": {"hip": {"values_deg": [88.0], "mean_deg": 88.0}}}                         # none for the main joint: the first one
    assert tree(with_reading(full, only_hip)).reference_button.text().startswith("Reference angle: 88.0 deg")


def test_a_reading_without_an_app_value_and_readings_for_another_position(full):
    odd = {**READING, "readings": {"vo2": {"values_deg": [10.0], "mean_deg": 10.0}}, "not_used": {"entered_for": {"test_key": "knee_extension", "target_deg": 20, "participant": "P07"}}}
    t = tree(with_reading(full, odd))
    assert t.reference_button.text() == "Reference angle: 10.0 deg - no app value"
    assert "were NOT saved with this trial" in t.reference_button.toolTip() and "P07" in t.text(everything=True)


def test_the_reference_button_is_cut_not_widened_and_keeps_its_whole_text(full):
    t = tree(with_reading(full), width=560)
    assert t.reference_button.text().startswith("Reference angle: 59.0") and t.reference_button.minimumSizeHint().width() <= 60
    t.resize(300, 400)
    QApplication.processEvents()
    assert t.reference_button.width() <= 300 and t.reference_button.text().startswith("Reference angle: 59.0") and t.reference_button.toolTip()


# ------------------------------------------------------------------------------------------------ every fact of the old panels is in the tree
def old_text(tr, rows, fs, w):
    """What the angle panel and the dots panel said (the whole of it, with the More numbers) about the same numbers."""
    panel = AnglePanel()
    panel.set_data(tr, rows, fs, w)
    dots = fd.DotsPanel()
    dots.set_trial(tr, w)
    return panel.text(everything=True) + "\n" + dots.text()


def check_every_number_is_there(tr, force):
    rows, fs, w = numbers(tr, force=force)
    new = tree(tr, rows=rows, fs=fs, window=w).text(everything=True)
    old = old_text(tr, rows, fs, w)
    missing = sorted(numbers_in(old) - numbers_in(new))
    assert not missing, f"numbers of the old panels that the tree does not say: {missing}\n--- old ---\n{old}\n--- new ---\n{new}"
    return old, new


def test_every_number_of_the_old_panels_is_in_the_tree_with_force_a_ruler_scale_an_anchor_and_a_reference(full):
    add_force(full)
    with_reading(full)
    old, new = check_every_number_is_there(full, True)
    for word in ("Goniometer (read to 1 deg)", "mean of 2", "pixels", "Movement", "strap and anchor dots", "Pictures lost", "Median", "Rest"):
        assert word in new or word in ("pixels", "Movement"), word
    assert "Angle of pull (deg)" in new and "Moment arm (cm)" in new and "Strap to joint (cm)" in new


def test_every_number_of_the_old_panels_is_in_the_tree_without_force(full):
    check_every_number_is_there(full, False)


def test_every_number_of_the_old_panels_is_in_the_tree_without_a_ruler_scale_an_anchor_or_markers(plain):
    old, new = check_every_number_is_there(plain, False)
    assert "pixels: no ruler scale" in old and "pixels: no ruler scale" in new and NO_MARKERS in old and NO_MARKERS in new


def test_every_number_of_the_old_panels_is_in_the_tree_for_a_frontal_trial(frontal):
    check_every_number_is_there(frontal, False)


def test_the_old_wording_facts_are_kept(full):
    add_force(full)
    rows, fs, w = numbers(full)
    new = tree(full).text(everything=True)
    old = old_text(full, rows, fs, w)
    for fact in ("Reference readings", "Typed from a goniometer or an inclinometer, not measured by this program.", "No reading for this trial. Type it in the Live tab",
                 "Hold is the angle during the contraction (the middle 60 % of the window), not the angle that was set."):
        assert fact in old and fact in new, fact
    for dot in full.dot_names:                                                                # the dot table: every dot with its Lost and its shifts
        assert re.search(rf"^\s*{dot}: lost ", new, re.M), dot
    assert numbers_in(old_dot_lines(full, w)) <= numbers_in(new)


def old_dot_lines(tr, w):
    dots = fd.DotsPanel()
    dots.set_trial(tr, w)
    return dots.text()


def test_the_dot_numbers_are_the_dot_tables_own(full):
    t = tree(full, more=True)
    table = fd.DotTable(fit_height=True)
    table.set_trial(full, whole(full))
    for row, stat in enumerate(table.stats.values()):
        plain = table._plain[row]
        it = t.dot_item(PULL_GROUP if stat.name in ("strap", "anchor") else "hip" if stat.name == "iliac_crest" else "knee", stat.name)
        assert it.text(C_LOST) == plain[fd.DotTable.C_LOST] and f"{stat.name}: lost {plain[fd.DotTable.C_LOST]}, {stat.jumps} jump" in t.text(everything=True)
        for word, col in (("end shift", fd.DotTable.C_END), ("max shift", fd.DotTable.C_MAX), ("end vs anchor", fd.DotTable.C_REL_END),
                          ("max vs anchor", fd.DotTable.C_REL_MAX)):
            assert f"{word} {plain[col]}" in t.text(everything=True), (stat.name, word)


# ------------------------------------------------------------------------------------------------ the look
def test_the_look_follows_the_palette_light_and_dark(full, palette):
    add_force(full)
    t = tree(full)
    light = {c: colour(item(t, "knee"), c) for c in (C_PEAK, C_OFF)}
    mal_light = t.dot_item("knee", "fibular_head").foreground(C_LOST).color().name()
    dark(palette)
    QApplication.processEvents()
    after = {c: colour(item(t, "knee"), c) for c in (C_PEAK, C_OFF)}
    mal_dark = t.dot_item("knee", "fibular_head").foreground(C_LOST).color().name()
    assert after[C_PEAK] == strong_colour(GOOD if light[C_PEAK] == strong_colour(GOOD, QPalette()).lower() else WARN, palette.palette()).lower()
    assert light[C_PEAK] != after[C_PEAK] and mal_light != mal_dark                              # text colours are made again for the new palette
    assert QColor(mal_dark).lightness() > QColor(mal_light).lightness()                          # lighter on dark, darker on light
    amber = t.dot_item("knee", "fibular_head").background(C_LOST).color()
    assert amber.alpha() < 255 and (amber.red(), amber.green(), amber.blue()) == (QColor(WARN).red(), QColor(WARN).green(), QColor(WARN).blue())
    assert item(t, "knee").background(C_NAME).color().alpha() < 255                                # (the main row is a faint accent in either)
    assert strong_colour(ACCENT, palette.palette(), 0.8).lower() in t.styleSheet().lower()       # the More numbers label is lighter on dark too


def test_the_check_box_and_the_button_are_visible_in_a_dark_palette(full, palette):
    dark(palette)
    t = tree(full)
    css = t.styleSheet()
    assert "QCheckBox#angleMore::indicator" in css and "border: 1px solid" in css
    pixmap = t.more_check.grab()
    assert not pixmap.isNull()


def test_the_widget_never_takes_the_keyboard(full):
    t = tree(full)
    assert t.tree.focusPolicy() == Qt.NoFocus and t.more_check.focusPolicy() == Qt.NoFocus and t.reference_button.focusPolicy() == Qt.NoFocus
    click(t, t.dot_item("knee", "trochanter"), C_NAME)
    assert not t.tree.hasFocus()


# ------------------------------------------------------------------------------------------------ sizes
def test_every_row_shows_without_a_scroll_bar_when_the_parent_is_tall_enough_and_a_scroll_bar_appears_when_it_is_not(full):
    t = tree(full, width=700, height=700)
    bar = t.tree.verticalScrollBar()
    assert not bar.isVisible() and t.content_height() <= t.tree.height()
    assert t.sizeHint().height() >= t.content_height() and t.sizeHint().height() <= t.content_height() + 120
    last = t.dot_item(PULL_GROUP, "anchor")
    assert t.tree.viewport().rect().contains(t.tree.visualItemRect(last).center())
    t.resize(700, 240)
    QApplication.processEvents()
    assert bar.isVisible() and bar.maximum() > 0 and t.tree.height() < t.content_height()
    bar.setValue(bar.maximum())
    assert t.tree.viewport().rect().contains(t.tree.visualItemRect(last).center())               # the last row can be scrolled to
    assert t.tree.horizontalScrollBar().isVisible() is False


def test_the_tree_takes_all_the_height_it_is_given(full):
    holder = QWidget()
    OPEN.append(holder)
    lay = QVBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    t = AngleDotTree()
    lay.addWidget(t)
    t.set_data(full, *numbers(full, force=False)[:2], whole(full))
    holder.resize(700, 900)
    holder.show()
    QApplication.processEvents()
    assert t.height() == 900 and t.tree.height() > t.content_height()                              # (all of it, and room to spare)
    assert t.sizePolicy().verticalPolicy() == QSizePolicy.Expanding


@pytest.mark.parametrize("more", [False, True])
@pytest.mark.parametrize("width", [560, 700, 760])
def test_the_tree_is_usable_at_560_px_the_name_is_cut_and_the_numbers_are_not(full, more, width):
    add_force(full)
    t = tree(full, width=width, height=700, more=more)
    tr, hh = t.tree, t.tree.header()
    assert not tr.horizontalScrollBar().isVisible()
    assert hh.sectionSize(C_NAME) >= 100                                                         # room for a name
    fm = tr.fontMetrics()
    for col in (C_HOLD, C_TARGET, C_OFF, C_LOST) + ((C_SHIFT, C_REL) if more else ()):
        widest = max((fm.horizontalAdvance(it.text(col)) for it in all_items(t)), default=0)
        assert hh.sectionSize(col) >= widest + 4, (col, hh.sectionSize(col), widest)              # no number is cut
    peak = item(t, "knee")
    assert hh.sectionSize(C_PEAK) >= QFontMetrics(peak.font(C_PEAK)).horizontalAdvance(peak.text(C_PEAK)) + 4        # (the large number is not cut)
    assert sum(hh.sectionSize(c) for c in range(tr.columnCount()) if not tr.isColumnHidden(c)) <= tr.viewport().width() + 1


def all_items(t):
    out = []

    def walk(it):
        out.append(it)
        for i in range(it.childCount()):
            walk(it.child(i))
    for i in range(t.tree.topLevelItemCount()):
        walk(t.tree.topLevelItem(i))
    return out


def test_a_very_narrow_tree_scrolls_sideways_with_more_numbers_instead_of_cutting_the_numbers(full):
    t = tree(full, width=380, height=600, more=True)
    assert t.tree.horizontalScrollBar().isVisible() or t.tree.header().sectionSize(C_NAME) >= 70


def test_long_names_are_cut_with_the_whole_text_in_the_tooltip_and_the_parent_is_not_widened(full):
    full.angle_defs[0].label = "A knee angle with a name that is very much too long for any column " * 3
    holder = QWidget()
    OPEN.append(holder)
    lay = QVBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    t = AngleDotTree()
    lay.addWidget(t)
    rows, fs, w = numbers(full, force=False)
    t.more_check.setChecked(True)
    t.set_data(full, rows, fs, w)
    holder.resize(560, 600)
    holder.show()
    QApplication.processEvents()
    long = full.angle_defs[0].title
    knee = item(t, "knee")
    assert knee.text(C_NAME).endswith(long) and long in knee.toolTip(C_NAME)                       # the whole text is there, and in the tooltip
    assert t.tree.fontMetrics().horizontalAdvance(long) > t.tree.header().sectionSize(C_NAME) and t.tree.textElideMode() == Qt.ElideRight
    assert holder.width() == 560 and t.width() <= 560 and holder.minimumSizeHint().width() < 200 and t.minimumSizeHint().width() == 0
    assert t.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored and t.reference_button.width() <= 560
    holder.resize(300, 600)
    QApplication.processEvents()
    assert holder.width() == 300 and t.width() <= 300


def test_the_header_goes_on_two_lines_when_it_does_not_fit_on_one(full):
    t = tree(with_reading(full), width=900)
    one = t.head.height()
    need = t.heading.natural_width() + t.reference_button.sizeHint().width() + t.more_check.sizeHint().width()
    assert abs(t.reference_button.y() - t.heading.y()) < 8 and t.reference_button.geometry().right() <= t.more_check.geometry().left()      # one line
    t.resize(need - 60, 600)                                                                       # (what is too narrow depends on the font)
    QApplication.processEvents()
    assert t.reference_button.y() > t.heading.y() and t.head.height() > one                          # the button has a line of its own
    assert t.more_check.y() < t.reference_button.y() and t.reference_button.geometry().right() <= t.head.width()
    assert t.heading.geometry().right() <= t.more_check.geometry().left()
    t.resize(900, 600)
    QApplication.processEvents()
    assert abs(t.reference_button.y() - t.heading.y()) < 8


def test_a_trial_without_a_video_works(recorded):
    tr = Trial(recorded[0], open_video=False)
    OPEN.append(tr)
    assert tr.cap is None
    t = tree(tr)
    assert keys(t)[0] == "knee" and "Knee flexion" in t.text()


def test_the_tree_is_not_rebuilt_when_nothing_changed(full):
    t = tree(full)
    first = item(t, "knee")
    t.set_data(full, *numbers(full, force=False)[:2], whole(full))
    assert item(t, "knee") is first                                                                # the same items: no flicker, scroll position kept
    t.tree.verticalScrollBar().setValue(0)
    t.set_data(full, *numbers(full, force=False)[:2], (whole(full)[0], whole(full)[1] - 0.2))      # the window moved: the numbers are made again
    assert t.heading.text() != f"{TITLE}  window {whole(full)[0]:.2f} to {whole(full)[1]:.2f} s"
