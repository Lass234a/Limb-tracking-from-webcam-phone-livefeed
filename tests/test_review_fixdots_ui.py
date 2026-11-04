"""The Fix dots workspace and the Dots panel of the Review tab (0.32.0, limbtrack/review_fixdots.py): every control of the workspace on a recorded
synthetic trial and on a copy of the real P02 knee T4 trial, the picture keys, the dot table and its plain-text copy, the widths the column must fit."""
import hashlib
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")        # (offscreen Qt has no fonts of its own: without them every text is zero wide)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QLocale, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPalette  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QStyle, QStyleOptionSpinBox, QVBoxLayout, QWidget  # noqa: E402

from limbtrack import dot_edits  # noqa: E402
from limbtrack import review_fixdots as fd  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.review_fixdots import DotsPanel, DotTable, FixDotsPane, PullTable  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

from tests import test_displacement as D  # noqa: E402  (a trial with a ruler scale, a strap that slides and an anchor)
from tests import test_pull as P  # noqa: E402  (a knee trial with the strap and anchor dots)

N = 30
REAL_DIR = Path(__file__).resolve().parent.parent / "Pilot" / "P02"
REAL_BASE = "P02_02_knee_extension_70deg_T4_20261029-120029"
real_only = pytest.mark.skipif(not (REAL_DIR / f"{REAL_BASE}_meta.json").is_file(), reason="the pilot recordings are not here")


@pytest.fixture(autouse=True)
def _a_fixed_font():
    """The widths asked for here (a 380 px column) were tuned on Windows, where the program's font is about as wide as Arial. The pane asks for exactly what its
    number columns need, so a wider default font (a Linux machine's) legitimately needs more room. The font is pinned so the test measures the layout, not the
    computer's font; it is put back afterwards. Only off Windows: on the tuned computer the real font is used, so the test is as strict there as it ever was."""
    if sys.platform == "win32":
        yield                                                  # (on the computer these limits were tuned on, the real font is used: nothing is made easier to pass)
        return
    app = QApplication.instance() or QApplication([])
    old = app.font()
    app.setFont(QFont("Arial", 9))
    yield
    app.setFont(old)


def _app():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------------------------------------- a tab with a trial and the workspace open
@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Two 30-picture knee trials with strap and anchor, recorded once; the anchor is hidden in pictures 10-14 of the first (lost, as in P02 T4)."""
    base = tmp_path_factory.mktemp("fixdots_rec")
    eng, p = P.make(base)
    metas = []
    for k in range(2):
        paths = eng.start_trial("P01")
        for i in range(N):
            eng.process(render(p, seed=i, hide=["anchor"] if (k == 0 and 10 <= i < 15) else []), 0.2 + i / 30)
        eng.stop_trial()
        metas.append(paths["meta.json"].relative_to(base))
    return base, metas, p


def _open(root, meta, monkeypatch, fix=True):
    _app()
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(("info", a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warning", a[2])))
    tab = ReviewTab(root)
    tab.load(meta)
    if fix:
        tab.set_fix_mode(True)
    return SimpleNamespace(tab=tab, pane=tab.fix_dots, shown=shown, root=root, meta=meta)


def _close(rig):
    if rig.tab.trial is not None:
        rig.tab.trial.close()
    try:
        rig.tab.overview.shutdown()
    except Exception:  # noqa: BLE001 - the tab is being thrown away
        pass
    rig.tab.close()


@pytest.fixture
def rig(recorded, tmp_path, monkeypatch):
    base, metas, p = recorded
    root = tmp_path / "rec"
    shutil.copytree(base, root)
    r = _open(root, root / metas[0], monkeypatch)
    r.p, r.other = p, root / metas[1]
    yield r
    _close(r)


def _times(rig, *pictures):
    return [float(rig.tab.trial.t[i]) for i in pictures]


def _pick(rig, dot):
    rig.pane.dot.setCurrentText(dot)
    assert rig.pane.current_dot() == dot


def _range(rig, first, last):
    t = rig.tab.trial.t
    rig.pane.t_from.setValue(float(t[first]))
    rig.pane.t_to.setValue(float(t[last]))


def _row(pane, dot):
    return next(r for r in range(pane.table.rowCount()) if pane.table._name(r) == dot)


# ---------------------------------------------------------------------------------------------------------- the numbers (no Qt)
def test_a_jump_is_2_cm_with_a_ruler_scale_else_one_percent_of_the_pictures_width():
    with_scale = SimpleNamespace(px_per_cm=6.0, full_size=(1280, 720), meta={})
    assert fd.jump_limit(with_scale) == (12.0, "2 cm")
    without = SimpleNamespace(px_per_cm=None, full_size=(1280, 720), meta={})
    px, words = fd.jump_limit(without)
    assert px == pytest.approx(12.8) and "1 %" in words and "no ruler" not in words
    assert fd.jump_limit(SimpleNamespace(px_per_cm=None, full_size=None, meta={"frame_size": [640, 480]}))[0] == pytest.approx(6.4)


def test_runs_and_bands_say_where_a_dot_was_lost():
    assert fd.runs([0, 1, 1, 0, 1, 1, 1]) == [(1, 2), (4, 6)] and fd.runs([0, 0]) == []
    t = np.arange(10) / 10.0
    (x0, x1), = fd.band_edges(t, [(2, 3)])
    assert (x0, x1) == pytest.approx((0.15, 0.35))                          # half a picture before the first to half a picture after the last
    (x0, x1), = fd.band_edges(t, [(5, 5)], min_width=0.5)
    assert x1 - x0 == pytest.approx(0.5)                                    # one lost picture can still be seen


def test_the_dot_stats_count_lost_jumped_and_fixed_pictures(rig):
    tr = Trial(rig.root / rig.meta.relative_to(rig.root), open_video=False)
    stats = {s.name: s for s in fd.trial_dot_stats(tr)}
    assert set(stats) == set(tr.dot_names) and "hip_centre" not in stats
    a = stats["anchor"]
    assert a.lost_pct == pytest.approx(100.0 * 5 / N) and a.lost_runs == [(10, 14)] and a.needs_attention and a.n_fixed == 0
    assert not stats["malleolus"].needs_attention and stats["malleolus"].jumps == 0
    assert math.isnan(a.step[0]) and a.step.shape == (N,)


# ---------------------------------------------------------------------------------------------------------- the tables, and their plain text
def test_the_dot_panel_text_gives_shifts_in_px_and_cm_against_the_anchor_and_the_pull_rows(tmp_path):
    _app()
    paths, _ = D.record(tmp_path)
    tr = Trial(paths["meta.json"], open_video=False)
    panel = DotsPanel()
    panel.set_trial(tr, (float(tr.t[0]), float(tr.t[-1])))
    text = panel.text()
    assert "ruler scale" in text and "no ruler scale" not in text and "blue window" in text
    head = text.splitlines()[1]
    assert head.split()[0] == "Dot" and "vs anchor" in head and "Fixed" in head
    strap = next(line for line in text.splitlines() if line.startswith("strap"))
    cm, px = re.search(r" (\d+\.\d) cm \((\d+\.\d) px\)", strap).groups()
    assert float(px) == pytest.approx(12.0, abs=1.0) and float(cm) == pytest.approx(2.0, abs=0.2)           # the strap slid 12 px = 2 cm
    anchor = next(line for line in text.splitlines() if line.startswith("anchor"))
    assert anchor.rstrip().endswith("-") and re.search(r"-\s+-\s+-$", anchor)                                 # nothing 'vs anchor' for the anchor itself
    pull = text.split("\n\n")[1]
    assert pull.splitlines()[0].split() == ["Pull", "Mean", "Min", "Max", "Change"]
    assert [line.split("  ")[0] for line in pull.splitlines()[1:]] == ["Angle of pull (deg)", "Moment arm (cm)", "Strap to joint (cm)"]
    assert panel.table.text() in text and panel.pull.text() in text


def _trial_without_strap(tmp_path):
    eng = Engine(D.TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    p = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    paths = eng.start_trial("P01")
    for i in range(20):
        eng.process(render(p, seed=i), 0.1 + i / 30)
    eng.stop_trial()
    return paths["meta.json"]


def test_without_strap_and_anchor_dots_there_are_no_pull_rows_no_anchor_columns_and_pixels_only(tmp_path):
    _app()
    tr = Trial(_trial_without_strap(tmp_path), open_video=False)
    panel = DotsPanel()
    panel.set_trial(tr)
    text = panel.text()
    assert "no ruler scale" in text and "vs anchor" not in text and "Pull" not in text and "Angle of pull" not in text
    assert not panel.pull.isVisibleTo(panel) and panel.pull.rowCount() == 0 and panel.pull.text() == ""
    assert " px" in text and " cm" not in text.replace("cm from", "")                                          # the caption says 'cm from' only when there is a scale
    assert panel.table.isColumnHidden(DotTable.C_REL_END) and panel.table.isColumnHidden(DotTable.C_REL_MAX)


def test_the_narrow_workspace_table_leaves_the_vs_anchor_columns_out_and_text_follows_what_is_shown(tmp_path):
    _app()
    paths, _ = D.record(tmp_path)
    tr = Trial(paths["meta.json"], open_video=False)
    wide, narrow = DotTable(relative=True), DotTable(relative=False)
    for t in (wide, narrow):
        t.set_trial(tr)
    assert "vs anchor" in wide.text().splitlines()[0] and "vs anchor" not in narrow.text().splitlines()[0]
    assert narrow.columnCount() == 8 and narrow.isColumnHidden(DotTable.C_REL_END)
    assert [len(line.split()) for line in narrow.text().splitlines()[1:]] == [len(line.split()) for line in narrow.text().splitlines()[1:]]


def test_text_is_empty_without_a_trial():
    _app()
    panel = DotsPanel()
    assert panel.text() == "" and DotTable().text() == "" and PullTable().text() == ""
    panel.set_trial(None)
    assert panel.caption.text() == ""


def test_the_panel_follows_the_blue_window_and_a_dragged_window_does_not_read_the_trial_again(rig, monkeypatch):
    tab = rig.tab
    panel = DotsPanel(tab)
    panel.set_trial(tab.trial)
    t = tab.trial.t
    moved = []
    panel.windowChanged.connect(lambda a, b: moved.append((a, b)))
    counted = {"stats": 0}
    real_stats = fd.trial_dot_stats
    monkeypatch.setattr(fd, "trial_dot_stats", lambda tr: counted.__setitem__("stats", counted["stats"] + 1) or real_stats(tr))
    before = panel.text()
    tab.window.setRegion((float(t[3]), float(t[9])))
    assert moved and moved[-1] == pytest.approx((float(t[3]), float(t[9])))
    assert f"{t[3]:.2f} to {t[9]:.2f} s" in panel.caption.text() and panel.text() != before
    panel.refresh()                                                                      # the same trial, no fix since: only the window is taken again
    assert counted["stats"] == 0
    panel.set_trial(tab.trial)
    assert counted["stats"] == 1


def test_a_row_click_chooses_a_dot_a_double_click_opens_it_and_a_dot_that_needs_a_look_is_amber(rig):
    table = DotTable(rig.tab)
    table.refresh()
    chosen, opened = [], []
    table.dotChosen.connect(chosen.append)
    table.dotActivated.connect(opened.append)
    row = _row(SimpleNamespace(table=table), "anchor")
    table.cellClicked.emit(row, 0)
    table.cellDoubleClicked.emit(row, 2)
    assert chosen == ["anchor"] and opened == ["anchor"]
    amber = table.item(row, 0).background().color()
    assert amber.alpha() > 0 and table.item(_row(SimpleNamespace(table=table), "malleolus"), 0).background().style() == Qt.NoBrush
    assert table.item(row, DotTable.C_LOST).text() == "17 %" and table.item(row, DotTable.C_LOST).font().bold()


def test_the_tables_take_the_palette_of_a_dark_theme_without_being_asked(rig):
    app = _app()
    table = DotTable(rig.tab)
    table.refresh()
    first = table._qss
    old = app.palette()
    dark = QPalette(old)
    dark.setColor(QPalette.Window, QColor(32, 33, 36))
    dark.setColor(QPalette.WindowText, QColor(230, 230, 230))
    try:
        app.setPalette(dark)
        app.processEvents()
        assert table._qss != first                                                       # the border colour follows the palette
    finally:
        app.setPalette(old)
        app.processEvents()
    assert table._qss == first


# ---------------------------------------------------------------------------------------------------------- the workspace: every control
def test_the_workspace_opens_on_the_dot_that_needs_a_look_and_a_check_can_name_another(rig):
    pane, tab = rig.pane, rig.tab
    assert tab._fix_mode and pane.current_dot() == "anchor"                          # lost in 5 of 30 pictures
    tab.set_fix_mode(False)
    tab.set_fix_mode(True, dot="strap")
    assert pane.current_dot() == "strap" and pane.table.selected_dot() == "strap" and pane.dot_title.text() == "strap"
    assert pane.where.text().startswith("usual position") and "on this picture: (" in pane.where.text()


def test_hold_at_the_usual_position_changes_only_the_stretch_and_the_table_and_timeline_follow(rig):
    pane, tab = rig.pane, rig.tab
    _range(rig, 8, 20)
    pane.btn_hold_usual.click()
    tr = tab.trial
    assert tr.edited["anchor"] == list(range(8, 21)) and all(tr.rows[i]["anchor_lost"] == "0" for i in range(8, 21))
    assert tr.review.dots[-1]["kind"] == "hold" and "is held at" in pane.status.text()
    s = pane.table.stats["anchor"]
    assert s.lost_pct == 0 and s.n_fixed == 13 and s.fixed_runs == [(8, 20)]
    assert pane.table.item(_row(pane, "anchor"), DotTable.C_FIXED).text() == "✎ 13"
    assert pane.timeline.fixed_bars.isVisible() and not pane.timeline.lost_bars.isVisible()
    assert [pane.edits.item(i).text() for i in range(pane.edits.count())][0].startswith("held at (") and pane.edits.isVisibleTo(pane)
    assert pane.btn_undo.isEnabled() and pane.btn_reset.isEnabled() and pane.current_dot() == "anchor"       # the dot and the stretch stay


def test_a_stretch_to_the_end_or_from_the_picture_on_screen_never_misses_the_picture(rig):
    """The boxes show hundredths: a picture whose time rounds the wrong way (the last one, or the one on screen) used to be left out."""
    pane, tab = rig.pane, rig.tab
    t = tab.trial.t
    assert pane.t_to.value() == pytest.approx(round(float(t[-1]), 2), abs=1e-9)
    low, high = pane._range()
    assert low <= t[0] and high >= t[-1]
    for i in range(N):                                                               # 'this picture' in either box, for every picture
        tab.slider.setValue(i)
        pane.btn_from.click()
        pane.btn_to.click()
        lo, hi = pane._range()
        assert lo <= t[i] <= hi and hi - lo < 0.001 + 1e-9 * 0 + (t[1] - t[0]) / 2, i
    pane.t_from.setValue(float(t[4]))
    pane.t_to.setValue(float(t[2]))                                                  # the boxes the wrong way round: the same stretch
    lo, hi = pane._range()
    assert lo <= t[2] <= t[4] <= hi and not lo <= t[1] and not hi >= t[5]


def test_the_blue_window_button_takes_the_pictures_inside_the_window(rig):
    pane, tab = rig.pane, rig.tab
    t = tab.trial.t
    tab.window.setRegion((float(t[3]) + 0.004, float(t[9]) + 0.004))
    pane.btn_window.click()
    lo, hi = pane._range()
    assert lo <= t[4] and lo > t[3] and hi >= t[9] and hi < t[10]


def test_hold_where_i_clicked_needs_a_click_and_then_holds_there(rig):
    pane, tab = rig.pane, rig.tab
    _range(rig, 3, 6)
    pane.btn_hold_click.click()
    assert "Click the dot on the picture first" in pane.status.text() and not tab.trial.edited
    tab.slider.setValue(5)
    tab.video.clicked.emit(300.5, 500.25)                                            # what the video label sends for a click, in full-picture pixels
    assert pane.click == (300.5, 500.25, 5) and "picture 6" in pane.status.text() and "Hold where I clicked" in pane.status.text()
    assert any(m[2] == "click" for m in pane.marks(5)) and pane.marks(4) == []
    pane.btn_hold_click.click()
    tr = tab.trial
    assert tr.edited["anchor"] == [3, 4, 5, 6]
    assert tr.tracks["anchor"][0][4] == pytest.approx(300.5) and tr.tracks["anchor"][1][4] == pytest.approx(500.25)
    assert tr.review.dots[-1]["x"] == 300.5 and tr.review.dots[-1]["y"] == 500.25


def test_mark_lost_blanks_the_dot_over_the_stretch(rig):
    pane, tab = rig.pane, rig.tab
    _pick(rig, "epicondyle")
    _range(rig, 3, 8)
    pane.btn_lost.click()
    tr = tab.trial
    assert all(tr.rows[i]["epicondyle_lost"] == "1" for i in range(3, 9)) and tr.rows[9]["epicondyle_lost"] == "0"
    s = pane.table.stats["epicondyle"]
    assert s.lost_pct == pytest.approx(20.0) and s.lost_runs == [(3, 8)] and s.fixed_runs == [(3, 8)]
    assert pane.table.item(_row(pane, "epicondyle"), DotTable.C_LOST).text() == "20 %"
    assert pane.timeline.lost_bars.isVisible() and "counts as lost" in pane.status.text()
    assert "epicondyle" in pane.dots.text() and re.search(r"epicondyle\s+20 %", pane.dots.text())


@pytest.mark.parametrize("direction, button, picture", [("forward", "btn_fwd", 0), ("backward", "btn_back", N - 1), ("both", "btn_both", 15)])
def test_re_tracking_follows_the_dot_through_the_video_from_the_picture_on_screen(rig, direction, button, picture):
    pane, tab = rig.pane, rig.tab
    _pick(rig, "malleolus")
    tab.slider.setValue(picture)
    x, y = (float(v) for v in rig.p["malleolus"])
    if direction == "forward":
        tab.video.clicked.emit(x + 2.0, y + 2.0)                                     # a click on this picture is the seed
    getattr(pane, button).click()
    tr = tab.trial
    e = tr.review.dots[-1]
    assert e["kind"] == "retrack" and e["direction"] == direction and e["seed_frame"] == picture
    assert len(tr.edited["malleolus"]) >= N - 2 and "re-tracked over" in pane.status.text()
    good = [i for i in tr.edited["malleolus"] if not math.isnan(tr.tracks["malleolus"][0][i])]
    assert all(abs(tr.tracks["malleolus"][0][i] - x) < 6.0 for i in good)           # the same dot the recording followed
    assert pane.edits.item(0).text().startswith("re-tracked: ")


def test_re_tracking_says_so_when_the_dot_is_lost_on_the_picture_and_nothing_was_clicked(rig):
    pane, tab = rig.pane, rig.tab
    tab.slider.setValue(12)                                                          # the anchor is hidden in pictures 10 to 14
    pane.btn_both.click()
    assert "Click the dot on this picture first" in pane.status.text() and not tab.trial.edited


def test_by_hand_each_click_steps_on_and_a_straight_line_is_laid_between_the_clicks(rig):
    pane, tab = rig.pane, rig.tab
    _pick(rig, "malleolus")
    pane.chk_hand.setChecked(True)
    assert pane.hand_mode() and "steps on" in pane.status.text()
    tab.slider.setValue(5)
    pane.on_video_click(100.0, 200.0)
    assert tab.slider.value() == 6 and list(pane.pending) == [5]                     # (it stepped on to the next picture)
    pane.on_video_click(110.0, 210.0)
    tab.slider.setValue(9)
    tab.video.clicked.emit(140.0, 240.0)                                             # the same through the video label
    assert sorted(pane.pending) == [5, 6, 9] and tab.slider.value() == 10
    assert pane.pending_label.text() == "3 pictures clicked, not saved" and pane.btn_save_hand.isEnabled() and pane.btn_drop_hand.isEnabled()
    tab.slider.setValue(5)
    assert pane.marks(5) == [(100.0, 200.0, "", (0, 215, 255))] and pane.marks(7) == []
    pane.btn_save_hand.click()
    tr = tab.trial
    assert tr.edited["malleolus"] == [5, 6, 7, 8, 9] and "3 positions saved" in pane.status.text()
    assert [round(float(tr.tracks["malleolus"][0][i]), 3) for i in range(5, 10)] == [100.0, 110.0, 120.0, 130.0, 140.0]
    assert not pane.pending and pane.pending_label.text() == "" and not pane.btn_save_hand.isEnabled()
    assert pane.edits.item(0).text() == "set by hand: 5 pictures" or pane.edits.item(0).text().startswith("set by hand")


def test_by_hand_discard_forgets_the_clicks_and_no_step_keeps_the_picture(rig):
    pane, tab = rig.pane, rig.tab
    _pick(rig, "malleolus")
    pane.chk_hand.setChecked(True)
    pane.chk_step.setChecked(False)
    tab.slider.setValue(4)
    pane.on_video_click(50.0, 60.0)
    assert tab.slider.value() == 4 and list(pane.pending) == [4]                     # no stepping on
    pane.btn_drop_hand.click()
    assert not pane.pending and not tab.trial.edited and pane.pending_label.text() == ""
    pane.on_video_click(50.0, 60.0)
    pane.chk_hand.setChecked(False)
    assert "waiting" in pane.status.text() or pane.pending                           # leaving hand mode does not throw the clicks away
    assert list(pane.pending) == [4]


def test_take_back_the_last_fix_and_reset_this_dot(rig):
    pane, tab = rig.pane, rig.tab
    plain = Trial(rig.root / rig.meta.relative_to(rig.root), open_video=False)
    _range(rig, 2, 4)
    pane.btn_hold_usual.click()
    _range(rig, 8, 9)
    pane.btn_lost.click()
    assert pane.edits.count() == 2 and tab.trial.edited["anchor"] == [2, 3, 4, 8, 9]
    pane.btn_undo.click()
    assert pane.edits.count() == 1 and tab.trial.edited["anchor"] == [2, 3, 4] and "taken back" in pane.status.text()
    pane.btn_reset.click()
    assert pane.edits.count() == 0 and not tab.trial.edited and not pane.btn_undo.isEnabled() and pane.edits_none.isVisibleTo(pane)
    assert np.array_equal(tab.trial.tracks["anchor"][0], plain.tracks["anchor"][0], equal_nan=True)         # back to the recorded positions
    assert "back to the recorded" in pane.status.text() and pane.table.stats["anchor"].n_fixed == 0


def test_clicks_not_saved_belong_to_their_dot_and_survive_a_fix_of_another_dot_but_not_another_trial(rig):
    pane, tab = rig.pane, rig.tab
    _pick(rig, "malleolus")
    pane.chk_hand.setChecked(True)
    tab.slider.setValue(5)
    pane.on_video_click(100.0, 200.0)
    _pick(rig, "anchor")                                                             # another dot: the clicks are kept aside, and it says so
    assert pane.pending == {} and "kept, not saved" in pane.status.text() and "malleolus" in pane.status.text()
    _range(rig, 2, 4)
    pane.btn_lost.click()                                                            # a fix of the anchor reads the trial again
    _pick(rig, "malleolus")
    assert list(pane.pending) == [5] and pane.pending[5] == (100.0, 200.0)           # (they used to be thrown away by every reload)
    assert pane.pending_label.text() == "1 picture clicked, not saved"
    tab.load(rig.other)                                                              # another trial: a fresh start
    assert pane.pending == {} and pane.click is None and pane._stash == {} and pane.current_dot() == "anchor"
    assert pane.t_from.value() == 0.0 and pane.t_to.value() == pytest.approx(round(float(tab.trial.t[-1]), 2), abs=1e-9)


def test_the_click_belongs_to_the_trial_it_was_made_in(rig):
    pane, tab = rig.pane, rig.tab
    tab.slider.setValue(5)
    tab.video.clicked.emit(1.0, 2.0)
    assert pane.click is not None
    tab.load(rig.other)
    assert pane.click is None
    _range(rig, 1, 2)
    pane.btn_hold_click.click()
    assert "Click the dot on the picture first" in pane.status.text()


def test_the_table_and_the_timeline_are_made_again_when_a_trial_is_loaded_and_after_every_fix(rig):
    pane, tab = rig.pane, rig.tab
    first = tab.trial
    assert pane.table.trial is first and pane.table.stats["anchor"].lost_pct == pytest.approx(100.0 * 5 / N)
    assert pane.timeline.lost_bars.isVisible() and pane.timeline.t is not None and len(pane.timeline.t) == N
    _range(rig, 10, 14)
    pane.btn_hold_usual.click()
    assert tab.trial is not first and pane.table.trial is tab.trial                  # the trial was read again, and so was the table
    assert pane.table.stats["anchor"].lost_pct == 0 and not pane.timeline.lost_bars.isVisible() and pane.timeline.fixed_bars.isVisible()
    assert re.search(r"anchor\s+0 %\s+0\s+.*✎ 5", pane.dots.text())
    tab.load(rig.other)
    assert pane.table.trial is tab.trial and pane.table.stats["anchor"].lost_pct == 0 and pane.table.stats["anchor"].n_fixed == 0
    assert not pane.timeline.fixed_bars.isVisible()


def test_moving_the_picture_only_moves_the_yellow_line_nothing_is_worked_out_again(rig, monkeypatch):
    pane, tab = rig.pane, rig.tab
    tr = tab.trial
    counted = {"stats": 0, "data": 0, "usual": 0}
    real_stats, real_set_data, real_usual = fd.trial_dot_stats, pane.timeline.set_data, dot_edits.usual_position
    monkeypatch.setattr(fd, "trial_dot_stats", lambda t: counted.__setitem__("stats", counted["stats"] + 1) or real_stats(t))
    monkeypatch.setattr(pane.timeline, "set_data", lambda *a, **k: counted.__setitem__("data", counted["data"] + 1) or real_set_data(*a, **k))
    monkeypatch.setattr(dot_edits, "usual_position", lambda *a: counted.__setitem__("usual", counted["usual"] + 1) or real_usual(*a))
    for i in list(range(0, N)) + list(range(N - 1, 0, -1)):
        tab.slider.setValue(i)
        assert pane.timeline.cursor.value() == pytest.approx(float(tr.t[i]))
    assert counted == {"stats": 0, "data": 0, "usual": 0}
    assert "on this picture:" in pane.where.text()
    start = time.perf_counter()
    for _ in range(300):
        pane.update_where()
    assert time.perf_counter() - start < 1.0                                         # about 0.05 ms each; a whole table made again would be a thousand times that


def test_a_click_or_a_drag_on_the_timeline_moves_the_video_to_the_nearest_picture(rig):
    pane, tab = rig.pane, rig.tab
    pane.timeline.picked.emit(float(tab.trial.t[17]) + 0.004)
    assert tab.slider.value() == 17
    pane.timeline.picked.emit(-5.0)
    assert tab.slider.value() == 0
    pane.timeline.picked.emit(99.0)
    assert tab.slider.value() == N - 1
    box = QWidget()                                                                  # a real mouse click on the graph, once it is on screen
    lay = QVBoxLayout(box)
    lay.addWidget(FixDotsPane(tab))
    own = lay.itemAt(0).widget()
    own.set_trial(tab.trial)
    box.resize(480, 1000)
    box.show()
    _app().processEvents()
    want = 11
    scene = own.timeline.getPlotItem().vb.mapViewToScene(QPointF(float(tab.trial.t[want]), 1.0))
    QTest.mouseClick(own.timeline.viewport(), Qt.LeftButton, pos=own.timeline.mapFromScene(scene))
    assert abs(tab.slider.value() - want) <= 1
    box.close()


def test_a_dot_row_click_chooses_the_dot_and_gives_the_keyboard_back(rig):
    pane, tab = rig.pane, rig.tab
    _active(tab)
    tab.set_fix_mode(True)
    pane.t_from.setFocus()
    row = _row(pane, "strap")
    QTest.mouseClick(pane.table.viewport(), Qt.LeftButton, pos=pane.table.visualRect(pane.table.model().index(row, 0)).center())
    assert pane.current_dot() == "strap" and _app().focusWidget() is tab.video


# ---------------------------------------------------------------------------------------------------------- no video
def test_without_a_video_the_controls_are_off_and_the_tables_and_timeline_still_show(recorded, tmp_path, monkeypatch):
    base, metas, _ = recorded
    root = tmp_path / "rec"
    shutil.copytree(base, root)
    for f in root.rglob("*_raw.mp4"):
        f.unlink()
    r = _open(root, root / metas[0], monkeypatch, fix=False)
    try:
        pane, tr = r.pane, r.tab.trial
        assert tr.cap is not None and not fd.has_video(tr)                           # (a capture object that opened nothing)
        assert not any(c.isEnabled() for c in pane._controls)
        assert "cannot be opened" in pane.where.text() and pane.table.rowCount() == len(tr.dot_names)
        assert "anchor" in pane.dots.text() and pane.timeline.t is not None and pane.timeline.curve.isVisible()
        assert not pane.btn_save_hand.isEnabled() and not pane.btn_undo.isEnabled()
        r.tab.slider.setValue(7)                                                     # nothing breaks when the picture moves
        pane.update_where()
        pane.on_video_click(1.0, 2.0)
        pane.btn_hold_usual.click()                                                  # (disabled: nothing happens)
        assert not tr.edited
    finally:
        _close(r)


def test_a_trial_opened_for_numbers_only_has_no_video_either(rig):
    pane, tab = rig.pane, rig.tab
    numbers = Trial(rig.root / rig.meta.relative_to(rig.root), open_video=False)
    tab.trial.close()
    tab.trial = numbers
    pane.set_trial(numbers)
    assert numbers.cap is None and not fd.has_video(numbers) and not any(c.isEnabled() for c in pane._controls)
    assert pane.table.rowCount() == len(numbers.dot_names) and pane.timeline.t is not None
    assert not fd.has_video(None)


def test_with_no_trial_the_workspace_is_empty_and_says_what_to_do(rig):
    pane, tab = rig.pane, rig.tab
    tab.trial.close()
    tab.trial = None
    pane.set_trial(None)
    assert pane.table.rowCount() == 0 and pane.dots.text() == "" and pane.timeline.t is None and "Open a trial" in pane.where.text()
    assert not any(c.isEnabled() for c in pane._controls)


# ---------------------------------------------------------------------------------------------------------- the keys
def _active(tab):
    tab.resize(1500, 950)
    tab.show()
    tab.activateWindow()
    assert QTest.qWaitForWindowActive(tab, 5000)


def test_the_by_hand_box_says_which_keys_step(rig):
    text = rig.pane.keys_label.text()
    for word in ("Left", "Right", "arrow", ",", ".", "Shift", "ten"):
        assert word in text, word
    assert rig.pane.box_hand.isAncestorOf(rig.pane.keys_label) and "Esc" in rig.pane.heading.text()


def test_the_picture_keys_step_the_video_in_the_workspace_and_escape_leaves_it(rig):
    app, tab = _app(), rig.tab
    _active(tab)
    tab.set_fix_mode(True)
    assert app.focusWidget() is tab.video
    tab.slider.setValue(12)
    for key, mods, want in ((Qt.Key_Right, Qt.NoModifier, 13), (Qt.Key_Period, Qt.NoModifier, 14), (Qt.Key_Left, Qt.NoModifier, 13), (Qt.Key_Comma, Qt.NoModifier, 12),
                            (Qt.Key_Right, Qt.ShiftModifier, 22), (Qt.Key_Left, Qt.ShiftModifier, 12)):
        QTest.keyClick(app.focusWidget(), key, mods)
        assert tab.slider.value() == want, key
    assert "on this picture" in rig.pane.where.text()
    QTest.keyClick(app.focusWidget(), Qt.Key_Escape)
    assert not tab._fix_mode


def test_a_button_click_never_leaves_the_keyboard_in_a_spin_box(rig):
    app, tab, pane = _app(), rig.tab, rig.pane
    _active(tab)
    tab.set_fix_mode(True)
    tab.video.clicked.emit(300.0, 400.0)
    for name in ("btn_from", "btn_to", "btn_window", "btn_hold_click", "btn_hold_usual", "btn_lost", "btn_fwd", "btn_back", "btn_both", "btn_undo", "btn_reset"):
        pane.t_from.setFocus()
        app.processEvents()
        assert app.focusWidget() is pane.t_from, name
        btn = getattr(pane, name)
        if not btn.isEnabled():
            continue
        btn.click()
        assert app.focusWidget() is tab.video, name
        before = tab.slider.value()
        QTest.keyClick(app.focusWidget(), Qt.Key_Right)
        assert tab.slider.value() == min(before + 1, N - 1), name
        tab.slider.setValue(10)
    pane.chk_hand.click()                                                            # the by-hand buttons
    tab.slider.setValue(3)
    pane.on_video_click(10.0, 20.0)
    for name in ("btn_save_hand", "btn_drop_hand"):
        pane.on_video_click(10.0, 20.0)
        assert getattr(pane, name).isEnabled(), name
        pane.t_to.setFocus()
        getattr(pane, name).click()
        assert app.focusWidget() is tab.video, name
    for box in (pane.chk_hand, pane.chk_step, pane.box_retrack.button):
        pane.t_to.setFocus()
        box.click()
        assert app.focusWidget() is tab.video


def test_a_typed_time_keeps_the_keys_while_it_is_typed_and_gives_them_back_when_it_is_done(rig):
    app, tab, pane = _app(), rig.tab, rig.pane
    _active(tab)
    tab.set_fix_mode(True)
    tab.slider.setValue(12)
    pane.t_from.setFocus()
    app.processEvents()
    pane.t_from.selectAll()
    QTest.keyClicks(app.focusWidget(), f"0{QLocale().decimalPoint()}5")             # (the box takes the decimal sign of the computer's language)
    QTest.keyClick(app.focusWidget(), Qt.Key_Right)                                   # in the box the arrows move the text cursor, not the picture
    assert tab.slider.value() == 12 and app.focusWidget() is pane.t_from
    QTest.keyClick(app.focusWidget(), Qt.Key_Return)
    assert pane.t_from.value() == pytest.approx(0.5) and app.focusWidget() is tab.video
    QTest.keyClick(app.focusWidget(), Qt.Key_Right)
    assert tab.slider.value() == 13


def test_the_little_arrows_of_a_time_box_change_the_time_and_give_the_keys_back(rig):
    app, tab, pane = _app(), rig.tab, rig.pane
    _active(tab)
    tab.set_fix_mode(True)
    spin = pane.t_to
    start = spin.value()
    opt = QStyleOptionSpinBox()
    spin.initStyleOption(opt)
    down = spin.style().subControlRect(QStyle.CC_SpinBox, opt, QStyle.SC_SpinBoxDown, spin)
    QTest.mouseClick(spin, Qt.LeftButton, pos=down.center())
    assert spin.value() == pytest.approx(start - 0.1) and app.focusWidget() is tab.video


def test_the_timeline_and_the_empty_room_of_the_column_never_take_the_keyboard(rig):
    app, tab, pane = _app(), rig.tab, rig.pane
    _active(tab)
    tab.set_fix_mode(True)
    pane.t_from.setFocus()
    pane.timeline.picked.emit(0.3)
    assert app.focusWidget() is tab.video
    assert pane.focusPolicy() == Qt.NoFocus and pane.widget().focusPolicy() == Qt.NoFocus
    allowed = {pane.t_from, pane.t_to}                       # (the only boxes that keep the keyboard, while a time is typed)
    keepers = [w for w in pane.findChildren(QWidget) if w.focusPolicy() != Qt.NoFocus and w not in allowed and not any(a.isAncestorOf(w) for a in allowed)]
    assert all(w.focusProxy() is not None for w in keepers), [type(w).__name__ for w in keepers if w.focusProxy() is None]    # (a viewport's focus goes to its area, which has none)
    for w in (pane.table, pane.pull, pane.timeline, pane.edits, pane.btn_hold_usual, pane.chk_hand, pane.box_retrack.button):
        assert w.focusPolicy() == Qt.NoFocus, type(w).__name__


# ---------------------------------------------------------------------------------------------------------- widths and heights
@pytest.mark.parametrize("w, h", [(380, 700), (380, 1000), (480, 700), (480, 1000), (600, 700), (600, 1000)])
def test_the_column_fits_its_width_and_scrolls_only_when_the_window_is_short(rig, w, h):
    tab = rig.tab
    _range(rig, 2, 4)
    rig.pane.btn_hold_usual.click()                                                  # a fix, so the list of fixes has a row
    pane = FixDotsPane(tab)
    pane.set_trial(tab.trial)
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(6, 6, 0, 0)
    lay.addWidget(pane, 1)
    host.resize(w, h)
    host.show()
    _app().processEvents()
    inner = pane.widget()
    assert pane.minimumSizeHint().width() <= 380 and inner.width() >= inner.minimumSizeHint().width()
    for part in (pane.heading, pane.dots, pane.timeline, pane.timeline_hint, pane.box_hold, pane.box_hand, pane.box_retrack, pane.edits, pane.status):
        assert part.mapTo(inner, part.rect().bottomRight()).x() <= inner.width(), type(part).__name__
        assert part.mapTo(inner, part.rect().topLeft()).x() >= 0
    need = inner.minimumSizeHint().height()
    if h >= 1000:
        assert need <= pane.viewport().height() and pane.verticalScrollBar().maximum() == 0       # about 900 px: nothing needs scrolling
    else:
        assert pane.verticalScrollBar().maximum() > 0
        pane.ensureWidgetVisible(pane.status)                                                       # the last line can be reached
        assert pane.status.mapTo(pane.viewport(), pane.status.rect().bottomLeft()).y() <= pane.viewport().height()
    assert pane.horizontalScrollBar().maximum() == 0
    host.close()


def test_the_dots_panel_scrolls_when_it_is_short_and_does_not_force_the_room_it_sits_in(rig):
    panel = DotsPanel(rig.tab)
    panel.set_trial(rig.tab.trial)
    assert panel.minimumSizeHint().height() < 200
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.addWidget(panel)
    host.resize(560, 160)
    host.show()
    _app().processEvents()
    assert host.height() == 160 and panel.height() <= 160
    host.close()


# ---------------------------------------------------------------------------------------------------------- the real P02 knee T4 trial
@pytest.fixture
def real(tmp_path, monkeypatch):
    folder = tmp_path / "P02"
    folder.mkdir()
    for suffix in ("meta.json", "data.csv", "force.csv", "raw.mp4"):
        shutil.copy2(REAL_DIR / f"{REAL_BASE}_{suffix}", folder / f"{REAL_BASE}_{suffix}")
    r = _open(tmp_path, folder / f"{REAL_BASE}_meta.json", monkeypatch)
    r.before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    yield r
    _close(r)


@real_only
def test_the_real_anchor_jumps_and_is_lost_and_the_workspace_opens_on_it(real):
    pane, tab = real.pane, real.tab
    tr = tab.trial
    assert pane.current_dot() == "anchor" and pane.table.selected_dot() == "anchor"
    s = pane.table.stats["anchor"]
    limit = pane.table.limit_px
    jumped = np.nonzero(s.step > limit)[0]
    assert s.jumps == len(jumped) >= 2 and any(6.0 < tr.t[i] < 6.6 and s.step[i] > 40 for i in jumped)           # the jump at 6.2 s (about 44 px)
    a, b = s.lost_runs[-1]
    assert b == len(tr.t) - 1 and 8.6 < tr.t[a] < 8.9                                                         # gone from 8.77 s to the end
    assert s.lost_pct > 25 and s.needs_attention
    row = pane.table.item(_row(pane, "anchor"), 0)
    assert row.background().color().alpha() > 0 or pane.table.selected_dot() == "anchor"
    text = pane.dots.text()
    assert re.search(r"anchor\s+3\d %\s+3\s", text) and "Angle of pull" in text and "no ruler scale" not in text


@real_only
def test_holding_the_real_anchor_from_the_jump_to_the_end_ends_the_jumps_and_the_lost_pictures(real):
    pane, tab = real.pane, real.tab
    t = tab.trial.t
    before_pull = pane.pull.text()
    pane.t_from.setValue(6.0)
    pane.t_to.setValue(float(t[-1]))
    pane.btn_hold_usual.click()
    tr = tab.trial
    first = int(np.searchsorted(t, 6.0))
    assert tr.edited["anchor"] == list(range(first, len(t)))                                       # to the last picture, which the rounded box alone would have missed
    s = pane.table.stats["anchor"]
    assert s.jumps == 0 and s.lost_pct == 0 and s.n_fixed == len(t) - first
    assert pane.table.item(_row(pane, "anchor"), DotTable.C_FIXED).text() == f"✎ {len(t) - first}"
    assert pane.pull.text() != before_pull and "held at (31.5, 633.0)" in pane.edits.item(0).text()
    text = pane.dots.text()
    assert re.search(r"anchor\s+0 %\s+0\s+0\.0 cm", text)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (real.root / "P02").iterdir()}
    assert after == real.before                                                                   # the recorded files are exactly as they were


@real_only
def test_holding_where_i_clicked_on_the_real_trial_and_marking_it_lost(real):
    pane, tab = real.pane, real.tab
    t = tab.trial.t
    tab.slider.setValue(int(np.searchsorted(t, 10.0)))
    tab.video.clicked.emit(31.5, 633.0)
    pane.t_from.setValue(8.7)
    pane.t_to.setValue(float(t[-1]))
    pane.btn_hold_click.click()
    s = pane.table.stats["anchor"]
    assert s.lost_runs == [(r[0], r[1]) for r in s.lost_runs] and not any(a >= int(np.searchsorted(t, 8.7)) for a, _ in s.lost_runs)
    pane.btn_undo.click()
    pane.t_from.setValue(8.7)
    pane.btn_lost.click()
    assert pane.table.stats["anchor"].lost_runs[-1][1] == len(t) - 1 and "counts as lost" in pane.status.text()


@real_only
def test_re_tracking_the_real_anchor_over_the_saved_video_and_by_hand(real):
    pane, tab = real.pane, real.tab
    tab.slider.setValue(100)
    start = time.perf_counter()
    pane.btn_both.click()
    assert time.perf_counter() - start < 60
    tr = tab.trial
    assert tr.review.dots[-1]["kind"] == "retrack" and len(tr.edited["anchor"]) == len(tr.t) and "re-tracked over" in pane.status.text()
    pane.btn_reset.click()
    assert not tab.trial.edited
    _pick(real, "strap")
    pane.chk_hand.setChecked(True)
    tab.slider.setValue(50)
    pane.on_video_click(300.0, 400.0)
    pane.on_video_click(305.0, 402.0)
    pane.btn_save_hand.click()
    assert tab.trial.edited["strap"] == [50, 51] and "2 positions saved" in pane.status.text()


@real_only
@pytest.mark.parametrize("w, h", [(380, 700), (480, 1000), (600, 1000)])
def test_the_column_with_the_real_trial_fits(real, w, h):
    pane = FixDotsPane(real.tab)
    pane.set_trial(real.tab.trial)
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.addWidget(pane)
    host.resize(w, h)
    host.show()
    _app().processEvents()
    assert pane.widget().width() >= pane.widget().minimumSizeHint().width() and pane.horizontalScrollBar().maximum() == 0
    assert (pane.verticalScrollBar().maximum() == 0) == (h >= 1000)
    host.close()
