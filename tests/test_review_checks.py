"""The checks of a trial as items with a headline, an area and an action (analysis.trial_check_items) and as the panel that is always in view
(review_checks.ChecksPanel), 0.32.0."""
import copy
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea, QWidget  # noqa: E402

from limbtrack import analysis, verdict  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.review_checks import COMPACT_BELOW, MAX_HEIGHT, ChecksPanel, action_label  # noqa: E402

from tests.test_analysis import torque_trial  # noqa: E402

REAL = Path(os.environ.get("LIMBTRACK_REAL_P02", r"C:\Users\llind\AppData\Local\Temp\claude\C--Users-llind--claude-spares-spare-3DeC6G"
                                                  r"\12f365b2-ac58-4b76-99af-bc9d1318be9a\scratchpad\lead_data\P02"))
WINDOW = (8.5, 13.0)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """One recorded trial with a calibrated force (it takes a few seconds to make); `trial` hands out a fresh state of it to each test."""
    tr = torque_trial(tmp_path_factory.mktemp("checks_trial"))
    yield tr
    tr.close()


@pytest.fixture
def trial(base):
    """The recorded trial in a known good state (29 pictures per second, nothing lost); what a test changes is put back afterwards."""
    saved = (copy.deepcopy(base.meta), dict(base.tracks), list(base.rows), list(base.events), list(base.dot_names), list(getattr(base, "review_notes", [])),
             base.force_n, base.force_v)
    base.meta.update(measured_fps=29.0, pipeline_timing={}, dot_size_check=None, partial_dot_frames={}, gaps={"count": 0}, target_deg=70.0, tolerance_deg=5.0)
    yield base
    (base.meta, base.tracks, base.rows, base.events, base.dot_names, base.review_notes, base.force_n, base.force_v) = saved
    for path in (verdict.verdict_path(base.meta_path), verdict.review_verdict_path(base.meta_path)):
        Path(path).unlink(missing_ok=True)


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


def item(level, title, text, area="force", action=None, arg=None):
    return {"level": level, "text": text, "title": title, "area": area, "action": action, "arg": arg}


def by_title(items, start):
    found = [i for i in items if i["title"].startswith(start)]
    assert found, f"no item with a title starting {start!r} in {[i['title'] for i in items]}"
    return found[0]


# ------------------------------------------------------------------------------------------------ analysis.trial_check_items
def test_every_item_has_the_fields_and_they_are_the_checks_of_trial_checks(trial):
    """Everything that can go wrong at once: the items are the old checks (same order, level and text) with a headline, an area and an action."""
    trial.meta.update(measured_fps=11.1, pipeline_timing={"process_share_of_interval": 0.93, "process_ms": {"mean": 84.0}, "frame_interval_ms": {"median": 68.0}},
                      dot_size_check={"below_minimum": ["malleolus", "mt5_head"]}, partial_dot_frames={"mt5_head": 8}, gaps={"count": 2, "max_s": 1.4},
                      rest_before={"short": True, "seconds": 20.0, "required_s": 60, "why": "before the next trial at this position"},
                      checks={"dots_apart_frames": {"knee-hip": {"frames": 4, "first_t_s": 9.0, "last_t_s": 9.4},
                                                    "epicondyle-malleolus": {"frames": 2, "first_t_s": 9.0, "last_t_s": 9.1},
                                                    "epicondyle-fibular_head": {"frames": 2, "first_t_s": 9.0, "last_t_s": 9.1}},
                              "label_checks": {"swap": "the knee and hip dots may be swapped"},
                              "label_check_frames": {"swap": {"frames": 3, "first_t_s": 10.0, "last_t_s": 10.2}}})
    trial.review_notes = ["WARNING: the original file(s) changed after this review was made: a_data.csv", "force zero typed: +0.1000 V",
                          "dot anchor: held at (1.0, 2.0) 1.00-2.00 s", "dot ghost: not a dot of this trial, the edit is ignored"]
    n = len(trial.t)
    trial.tracks["strap"] = (np.where(np.arange(n) < 5, 100.0, np.nan), np.where(np.arange(n) < 5, 250.0, np.nan))
    ax, ay = trial.tracks["anchor"]
    trial.tracks["anchor"] = (ax + np.sin(np.arange(n) / 3.0) * 9.0, ay)
    items = analysis.trial_check_items(trial, *WINDOW)
    assert items and [(i["level"], i["text"]) for i in items] == analysis.trial_checks(trial, *WINDOW)
    for i in items:
        assert set(i) == {"level", "text", "title", "area", "action", "arg"}
        assert i["level"] in ("ok", "check", "info") and i["text"]
        assert i["title"] and len(i["title"]) <= analysis.CHECK_TITLE_MAX and "\n" not in i["title"]
        assert i["area"] in analysis.CHECK_AREAS
        assert i["action"] is None or i["action"] in analysis.CHECK_ACTIONS
        assert i["arg"] is None or (isinstance(i["arg"], str) and i["arg"])
        if i["action"] != "fix_dots":
            assert i["arg"] is None
    # the headlines that stand for the sentences
    fps = by_title(items, "Pictures per second low")
    assert fps["title"] == "Pictures per second low (11.1)" and fps["area"] == "recording" and fps["action"] is None
    assert by_title(items, "The computer was the limit")["area"] == "recording"
    gaps = by_title(items, "2 gaps")
    assert gaps["title"] == "2 gaps with no pictures" and gaps["area"] == "recording"
    assert by_title(items, "Dots seen only partly")["level"] == "info"
    apart = by_title(items, "Dots knee and hip moved apart")
    assert apart["action"] == "fix_dots" and apart["arg"] == "knee" and apart["area"] == "dots" and "Dots knee and hip moved apart in 4 pictures" in apart["text"]
    assert by_title(items, "epicondyle and malleolus moved apart")["arg"] == "epicondyle"            # (a longer pair: 'Dots' goes first)
    assert by_title(items, "Two dots moved apart")["arg"] == "epicondyle"                            # (a pair too long for that: the sentence has the names)
    labels = by_title(items, "Dot labels may be swapped")
    assert labels["action"] == "fix_dots" and labels["arg"] is None and labels["text"].startswith("CHECK LABELS in 3 pictures")
    rest = by_title(items, "Only 0:20 of rest")
    assert rest["area"] == "recording" and "the SOP asks for 1:00" in rest["text"]
    small = by_title(items, "Dots under 2 cm")
    assert small["action"] == "fix_dots" and small["arg"] == "malleolus"
    assert by_title(items, "Strap dot seen in only")["arg"] == "strap"
    anchor = by_title(items, "Anchor wandered")
    assert anchor["action"] == "fix_dots" and anchor["arg"] == "anchor"
    # what Review changed
    warning = by_title(items, "Originals changed after review")
    assert warning["level"] == "check" and warning["area"] == "review" and warning["text"].startswith("WARNING: the original file(s) changed")
    zero = by_title(items, "Force zero set in Review")
    assert zero["level"] == "info" and zero["text"] == "Reviewed: force zero typed: +0.1000 V"
    assert by_title(items, "Dot anchor fixed in Review")["area"] == "review" and by_title(items, "Dot ghost edit ignored")["level"] == "info"


def test_the_force_checks_say_what_to_do(trial):
    trial.events = [(0, 3.0, "MVIC start"), (1, 14.0, "MVIC end")]                # far from where the force rose
    trial.meta["force"] = dict(trial.meta.get("force") or {}, samples_at_range_limit=12, frames_without_samples_over_0_5s=3, read_error="board unplugged")
    items = analysis.trial_check_items(trial)
    early = by_title(items, "Start marker")
    assert early["action"] == "window" and early["area"] == "force" and "MVIC start marker was pressed" in early["text"]
    assert by_title(items, "Force hit the input limit")["action"] is None
    assert by_title(items, "Force reading stopped")["area"] == "recording"
    assert by_title(items, "Force signal had gaps")["area"] == "force"
    low = {i["title"] for i in analysis.trial_check_items(trial)}
    assert "MVIC marker matches the force" not in low


def test_a_position_off_the_target_and_a_joint_that_moved_are_called_out(trial):
    trial.meta["target_deg"] = 40.0                                               # the knee is held at 70: 30 deg off
    pos = next(i for i in analysis.trial_check_items(trial, *WINDOW) if i["area"] == "position")
    assert pos["level"] == "check" and pos["title"].startswith("Knee ") and pos["title"].endswith("deg, target 40") and "from the target 40" in pos["text"]
    trial.meta["target_deg"] = 70.0
    ok = next(i for i in analysis.trial_check_items(trial, *WINDOW) if i["area"] == "position")
    assert ok["level"] == "ok" and ok["title"].endswith("deg, target 70")


def test_the_verdict_shows_as_a_check_or_as_fine(trial):
    verdict.write_verdict(trial.meta_path, verdict.COMPROMISED, ["wrong_dots"], "malleolus dot on the sock")
    bad = by_title(analysis.trial_check_items(trial, *WINDOW), "Marked compromised")
    assert bad["level"] == "check" and bad["area"] == "verdict" and "malleolus dot on the sock" in bad["text"] and bad["action"] is None
    verdict.write_verdict(trial.meta_path, verdict.VALID)
    good = by_title(analysis.trial_check_items(trial, *WINDOW), "Marked valid")
    assert good["level"] == "ok" and good["area"] == "verdict"


def test_a_long_dot_name_never_makes_a_headline_longer_than_the_panel_can_show(trial):
    name = "a_very_long_name_of_a_dot_that_a_person_typed"
    trial.dot_names = [name]
    trial.rows = [{**r, f"{name}_lost": "1"} for r in trial.rows]
    lost = by_title(analysis.trial_check_items(trial, *WINDOW), "Dot")
    assert len(lost["title"]) <= analysis.CHECK_TITLE_MAX and lost["action"] == "fix_dots" and lost["arg"] == name
    assert f"Dot '{name}' was lost in 100 % of the pictures" in lost["text"]


def test_the_force_in_volts_offers_the_calibration_and_the_zero_checks_the_zero(trial, monkeypatch):
    trial.force_n = np.full(trial.force_t.size, np.nan)
    trial.meta["force"] = dict(trial.meta.get("force") or {}, calibrated=False)
    volts = by_title(analysis.trial_check_items(trial, *WINDOW), "Force is in volts")
    assert volts["level"] == "info" and volts["action"] == "calibration"
    monkeypatch.setattr(analysis, "zero_drift", lambda tr, t0, t1: {"unit": "N", "before": 2.0, "after": 40.0, "drift": 38.0})
    trial.force_n = trial.force_v * 100.0
    zero = by_title(analysis.trial_check_items(trial, *WINDOW), "Zero moved during the trial")
    assert zero["action"] == "zero" and "the zero moved during the trial" in zero["text"]


def test_trial_checks_is_the_list_of_level_and_text_of_the_items(trial):
    for window in ((), WINDOW):
        assert analysis.trial_checks(trial, *window) == [(i["level"], i["text"]) for i in analysis.trial_check_items(trial, *window)]


# ------------------------------------------------------------------------------------------------ the panel
def panel_with(items, width=900, show=True):
    p = ChecksPanel()
    p.resize(width, 400)
    p.set_items(items)
    if show:
        p.show()
        QApplication.processEvents()
        p.resize(width, p.sizeHint().height())
        QApplication.processEvents()
    return p


FOUR = [item("check", "Dot anchor lost 32 %", "Dot 'anchor' was lost in 32 % of the pictures.", "dots", "fix_dots", "anchor"),
        item("check", "Zero moved during the trial", "The resting force was 123 N before the push and 160 N at the end (+37 N): the zero moved during the trial. "
                                                     "Set the zero in Review if you know where it should be.", "force", "zero"),
        item("check", "Peak above calibration (73 kg)", "The peak force (719 N = 73 kg) is above the heaviest calibration load (50 kg): the newtons there are the calibration "
                                                        "line extended, not measured.", "force", "calibration"),
        item("check", "Pictures per second low (14.2)", "Pictures: 14.2 per second (below 20: fast movements are sampled coarsely)", "recording"),
        item("check", "Start marker 1.0 s early", "The MVIC start marker was pressed 1.0 s before the force rose. Use 'Window = force above half of peak' for the hold numbers.",
             "force", "window"),
        item("ok", "Anchor steady (0.4 px)", "Anchor dot wandered 0.4 px (steady)", "dots"),
        item("ok", "Marked valid", "Marked valid (on site (Live tab))", "verdict"),
        item("info", "No ruler scale", "No ruler scale: the moment arm is in pixels only, so no torque can be given.", "recording")]
# (five things to look at, four with a button and one without; two fine ones; one note)
MANY =[item("check", f"Check number {i} of the trial", f"The long sentence of check number {i}. " * 4, "force", [None, "zero", "window", "calibration"][i % 4]) for i in range(15)]


def test_the_pills_count_what_is_fine_what_to_look_at_and_what_is_a_note(app):
    p = panel_with(FOUR)
    assert p.counts() == {"ok": 2, "check": 5, "info": 1}
    assert (p.pill_check.text(), p.pill_ok.text(), p.pill_info.text()) == ("5 to look at", "2 fine", "1 note")
    assert len(p.rows) == 5 and p.extra_rows == []                                 # only the things to look at have a line; the rest is behind a pill
    p.set_items(FOUR + [item("info", "Another note", "Another note.", "recording")])
    assert p.pill_info.text() == "2 notes"
    p.set_items([FOUR[0]])
    assert p.pill_check.text() == "1 to look at" and not p.pill_ok.isVisibleTo(p) and not p.pill_info.isVisibleTo(p)


def test_nothing_to_look_at_is_one_calm_line_and_no_items_is_a_quiet_one(app):
    p = panel_with([x for x in FOUR if x["level"] != "check"])
    assert p.rows == [] and "Nothing to look at" in p.text()
    assert p.pill_check.isHidden() and p.pill_ok.isVisibleTo(p)
    quiet = panel_with([])
    assert quiet.text() == "No checks to show" and quiet.counts() == {"ok": 0, "check": 0, "info": 0}
    assert quiet.pill_check.isHidden() and quiet.pill_ok.isHidden() and quiet.pill_info.isHidden()
    assert quiet.height() < 60 and p.height() < 70


def test_a_button_sends_what_to_do_and_for_which_dot(app):
    p = panel_with(FOUR)
    got = []
    p.actionRequested.connect(lambda action, arg: got.append((action, arg)))
    buttons = {r.item["title"]: r.button for r in p.rows}
    assert {t: (b.text() if b else None) for t, b in buttons.items()} == {
        "Dot anchor lost 32 %": "Show dot", "Zero moved during the trial": "Set zero", "Peak above calibration (73 kg)": "Calibration",
        "Pictures per second low (14.2)": None, "Start marker 1.0 s early": "Set window"}
    for title in ("Dot anchor lost 32 %", "Zero moved during the trial", "Start marker 1.0 s early", "Peak above calibration (73 kg)"):
        buttons[title].click()
    assert got == [("fix_dots", "anchor"), ("zero", ""), ("window", ""), ("calibration", "")]
    assert action_label(item("check", "x", "y", action="fix_dots")) == "Show dots" and action_label(item("check", "x", "y")) == ""
    assert got and all(b.focusPolicy().name == "NoFocus" for b in buttons.values() if b)


def test_a_click_on_a_line_shows_the_whole_sentence_and_a_click_on_the_pill_all_of_them(app):
    p = panel_with(FOUR, width=520)
    zero = next(r for r in p.rows if r.item["title"].startswith("Zero moved"))
    assert not zero.detail.isVisibleTo(p) and zero.sentence.isVisibleTo(p) and zero.sentence.toolTip() == zero.item["text"]
    before = p.scroll.height()
    zero.set_expanded(True)
    assert zero.detail.isVisibleTo(p) and zero.detail.text() == zero.item["text"] and p.scroll.height() >= before
    zero.set_expanded(False)
    p.pill_check.click()
    assert all(r.expanded for r in p.rows) and p.pill_check.isChecked()
    p.pill_check.click()
    assert not any(r.expanded for r in p.rows)


def test_the_fine_ones_and_the_notes_are_behind_their_number(app):
    p = panel_with(FOUR)
    assert p.extra_rows == [] and "Anchor steady" not in p.text() and "Fine" not in p.text()
    p.pill_ok.click()
    assert [r.item["title"] for r in p.extra_rows] == ["Anchor steady (0.4 px)", "Marked valid"] and p.pill_ok.isChecked()
    assert "Fine\n  Anchor steady (0.4 px): Anchor dot wandered 0.4 px (steady)" in p.text() and "No ruler scale" not in p.text()
    p.pill_info.click()                                                              # one list at a time
    assert [r.item["title"] for r in p.extra_rows] == ["No ruler scale"] and not p.pill_ok.isChecked()
    p.pill_info.click()
    assert p.extra_rows == []
    assert "No ruler scale: No ruler scale" in p.text(everything=True) and "Anchor steady (0.4 px)" in p.text(everything=True)


def test_text_is_everything_a_person_can_read_with_the_whole_sentences_and_the_button_labels(app):
    p = panel_with(FOUR, width=380)                                                   # (a narrow panel cuts the sentences on screen: text() has them whole)
    t = p.text()
    assert t.splitlines()[0] == "Checks: 5 to look at, 2 fine, 1 note" and "To look at" in t
    for i in FOUR[:5]:
        assert f"  {i['title']}: {i['text']}" in t
    assert "  Dot anchor lost 32 %: Dot 'anchor' was lost in 32 % of the pictures.  [Show dot]" in t
    assert "[Set zero]" in t and "[Set window]" in t and "[Calibration]" in t
    assert "  Pictures per second low (14.2): Pictures: 14.2 per second (below 20: fast movements are sampled coarsely)\n" in t + "\n"     # (no button: no brackets)
    assert "Anchor steady" not in t and "No ruler scale" not in t                  # behind their pills
    assert len(t.splitlines()) == 1 + 1 + 5


def test_the_same_list_again_changes_nothing_and_a_new_list_replaces_the_lines(app):
    p = panel_with(FOUR)
    rows = list(p.rows)
    p.rows[0].set_expanded(True)
    p.set_items([dict(i) for i in FOUR])
    assert p.rows == rows and p.rows[0].expanded                                       # (called on every move of the analysis window: no flicker, nothing closes)
    p.set_items(FOUR[:2])
    assert len(p.rows) == 2 and p.rows[0].expanded                                     # the line that is still there stays open
    p.set_items([])
    assert p.rows == []


@pytest.mark.parametrize("width", [380, 520, 900])
def test_four_things_take_about_130_px_and_fifteen_never_more_than_40_percent_of_a_700_px_column(app, width):
    p = panel_with(FOUR[:4], width)
    assert p.height() <= (140 if width >= COMPACT_BELOW else 200), p.height()
    assert not p.scroll.verticalScrollBar().isVisible()
    many = panel_with(MANY, width)
    assert many.height() <= MAX_HEIGHT <= 0.4 * 700 and many.scroll.verticalScrollBar().isVisibleTo(many)
    assert not many.scroll.horizontalScrollBar().isVisible()
    assert many.minimumSizeHint().width() <= 380                                       # (it never asks the window to be wider)
    for panel in (p, many):
        assert all(w.mapTo(panel, w.rect().topRight()).x() <= panel.width() + 1 for w in panel.findChildren(QWidget) if w.isVisibleTo(panel) and w.width() > 0)
    many.pill_ok.click()
    many.pill_info.click()


def test_a_narrow_panel_puts_the_sentence_under_the_headline_a_wide_one_beside_it(app):
    narrow = panel_with(FOUR[:3], width=380)
    row = narrow.rows[0]
    assert row.compact and row.sentence.y() > row.title_label.y() + 8
    wide = panel_with(FOUR[:3], width=900)
    row = wide.rows[0]
    assert not row.compact and abs(row.sentence.y() - row.title_label.y()) <= 3
    assert narrow.scroll.height() > wide.scroll.height()
    narrow.resize(900, narrow.height())                                                 # it follows the width of the panel, both ways
    QApplication.processEvents()
    assert not narrow.rows[0].compact and narrow.scroll.height() == wide.scroll.height()
    narrow.resize(380, narrow.height())
    QApplication.processEvents()
    assert narrow.rows[0].compact


def test_the_panel_follows_a_dark_palette(palette):
    p = panel_with(FOUR)
    light_css = p.styleSheet()
    dark(palette)
    QApplication.processEvents()
    assert p.styleSheet() != light_css
    p.set_items(FOUR[:2])
    assert len(p.rows) == 2 and p.grab().width() > 0


# ------------------------------------------------------------------------------------------------ real data
@pytest.mark.skipif(not REAL.exists(), reason="the copy of the real P02 trials is not on this computer")
@pytest.mark.parametrize("name", ["knee_extension_70deg", "hip_flexion_90deg", "ankle_plantarflexion"])
def test_the_checks_of_a_real_trial_fill_the_panel(app, name):
    meta = next(REAL.glob(f"*{name}*_meta.json"))
    tr = Trial(meta, open_video=False)
    try:
        w = tr.force_window() or tr.hold_window() or (float(tr.t[0]), float(tr.t[-1]))
        items = analysis.trial_check_items(tr, *w)
        assert [(i["level"], i["text"]) for i in items] == analysis.trial_checks(tr, *w)
        assert all(len(i["title"]) <= analysis.CHECK_TITLE_MAX and i["area"] in analysis.CHECK_AREAS for i in items)
        p = panel_with(items, 600)
        c = p.counts()
        assert len(p.rows) == c["check"] and c["ok"] + c["check"] + c["info"] == len(items)
        assert p.height() <= MAX_HEIGHT
        for i in (x for x in items if x["level"] == "check"):
            assert i["text"] in p.text()
    finally:
        tr.close()


def test_the_checks_panel_has_no_scroll_area_but_the_one_for_the_lines(app):
    p = panel_with(FOUR)
    assert len(p.findChildren(QScrollArea)) == 1 and p.findChildren(QScrollArea)[0] is p.scroll
