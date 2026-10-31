"""The Review tab's header, bottom bar and Fix-dots mode of the 0.32.0 layout: where in the list the trial is, the Data switch in the header, the checks chip with its
pop-up, the verdict as Valid and a Compromised menu (same interface as the Live tab's bar), the keys 1 to 5 and F, the Fix dots button by Show trails, the zoom and the
divider that can go far to the right. (The tables of angles and force have their own test files.)"""
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeySequence, QMouseEvent, QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import verdict  # noqa: E402
from limbtrack.review_checks import ChecksChip  # noqa: E402
from limbtrack.review_decide import VerdictControl  # noqa: E402
from tests.test_review_overview import _finish, _tab, root, trio  # noqa: E402,F401  (the fixtures: three real trials of one position, a fresh copy for each test)


def _app():
    return QApplication.instance() or QApplication([])


def _shortcut(tab, key):
    return next(sc for sc in tab.trial_page.findChildren(QShortcut) if sc.key() == QKeySequence(key))


ITEMS = [
    {"level": "check", "text": "Dot 'anchor' was lost in 32 % of the pictures.", "title": "Dot anchor lost 32 %", "area": "dots", "action": "fix_dots", "arg": "anchor"},
    {"level": "check", "text": "The MVIC start marker was pressed 1.0 s before the force rose.", "title": "Start marker 1.0 s early", "area": "force", "action": "window", "arg": ""},
    {"level": "ok", "text": "Pictures: 30.2 per second", "title": "Pictures per second", "area": "recording", "action": None, "arg": ""},
    {"level": "info", "text": "No ruler scale.", "title": "No ruler scale", "area": "recording", "action": None, "arg": ""},
]


# ------------------------------------------------------------------------------------------------ the checks chip
def test_the_chip_says_how_many_things_need_a_look_and_previews_them():
    _app()
    chip = ChecksChip()
    assert "No checks" in chip.chip.text() and chip.chip.property("kind") == "none"
    chip.set_items(ITEMS)
    assert chip.counts() == {"ok": 1, "check": 2, "info": 1} and "2 to look at" in chip.chip.text() and chip.chip.property("kind") == "check"
    assert chip.preview.text() == "Dot anchor lost 32 % · Start marker 1.0 s early" or chip.preview.toolTip().startswith("Dot anchor lost 32 %")
    assert "Dot anchor lost 32 %" in chip.text() and "Pictures per second" not in chip.text() and "Pictures per second" in chip.text(everything=True)
    chip.set_items([i for i in ITEMS if i["level"] != "check"])
    assert "Nothing to look at" in chip.chip.text() and chip.chip.property("kind") == "ok" and chip.preview.text() == ""


def test_the_pop_up_opens_above_the_chip_and_a_button_in_it_acts_and_closes_it():
    app = _app()
    chip = ChecksChip()
    chip.resize(500, 30)
    chip.move(100, 500)                                                                # (low on the screen: there is room above it)
    chip.show()
    chip.set_items(ITEMS)
    got = []
    chip.actionRequested.connect(lambda a, arg: got.append((a, arg)))
    chip.show_popup()
    app.processEvents()
    assert chip.popup.isVisible() and chip.popup.width() == ChecksChip.POPUP_WIDTH
    assert chip.popup.mapToGlobal(chip.popup.rect().bottomLeft()).y() <= chip.chip.mapToGlobal(chip.chip.rect().topLeft()).y() + 1       # (above it)
    assert len(chip.rows) == 2
    chip.rows[0].button.click()
    assert got == [("fix_dots", "anchor")] and not chip.popup.isVisible()
    chip.popup.close()
    chip.close()


# ------------------------------------------------------------------------------------------------ the verdict control
def test_the_verdict_control_works_like_the_live_tabs_bar():
    _app()
    bar = VerdictControl()
    seen = []
    bar.changed.connect(lambda v, reasons, note: seen.append((v, list(reasons), note)))
    assert bar.current() == (None, [], "") and "not judged" in bar.state.text()
    bar.btn_valid.click()
    assert seen[-1] == (verdict.VALID, [], "") and "valid" in bar.state.text() and bar.btn_valid.isChecked()
    bar.reason_buttons["bad_force"].click()                                            # a reason takes Valid away
    assert seen[-1][0] == verdict.COMPROMISED and seen[-1][1] == ["bad_force"] and not bar.btn_valid.isChecked()
    assert "compromised" in bar.state.text() and bar.btn_comp.text() == "Compromised · 1" and bar.reason_actions["bad_force"].isChecked()
    bar.reason_actions["wrong_dots"].setChecked(True)                                  # the menu does the same as the hidden button
    assert seen[-1][1] == ["wrong_dots", "bad_force"] and bar.reason_buttons["wrong_dots"].isChecked() and bar.btn_comp.text() == "Compromised · 2"
    bar.btn_valid.click()                                                              # Valid takes the reasons away, in the menu too
    assert seen[-1] == (verdict.VALID, [], "") and not any(a.isChecked() for a in bar.reason_actions.values())
    bar.note.setText("epicondyle jumped")
    bar.note.editingFinished.emit()
    assert seen[-1][2] == "epicondyle jumped" and bar.btn_note.text() == "Note ✎"


def test_showing_a_saved_verdict_does_not_report_a_change():
    _app()
    bar = VerdictControl()
    seen = []
    bar.changed.connect(lambda *a: seen.append(a))
    bar.set_verdict({"verdict": "compromised", "reasons": ["wrong_dots", "other"], "note": "see me"})
    assert not seen and bar.current() == (verdict.COMPROMISED, ["wrong_dots", "other"], "see me")
    assert bar.reason_actions["other"].isChecked() and bar.btn_comp.text() == "Compromised · 2" and not bar.btn_valid.isChecked()
    bar.set_verdict(None)
    assert bar.current() == (None, [], "") and not seen and bar.btn_comp.text() == "Compromised"


# ------------------------------------------------------------------------------------------------ the header and the bottom bar of the tab
def test_the_header_says_where_in_the_list_the_trial_is_and_holds_the_data_switch(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    assert tab.nav_label.text() == "1 of 3 · 3 left"
    for w in (tab.nav_label, tab.btn_data_original, tab.btn_data_reviewed, tab.showing_label, tab.edits_label, tab.btn_view_overview):
        assert tab.header.isAncestorOf(w)
    for w in (tab.btn_next_unjudged, tab.btn_fix, tab.verdict_bar):
        assert not tab.header.isAncestorOf(w)
    tab.verdict_bar.btn_valid.click()
    assert tab.nav_label.text() == "1 of 3 · 2 left"
    tab.next_trial()
    assert tab.nav_label.text() == "2 of 3 · 2 left"
    _finish(tab)


def test_the_bottom_bar_holds_the_checks_the_verdict_and_next_and_changes_only_in_fix_mode(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    bar, fixbar = tab.bottom_stack.widget(0), tab.bottom_stack.widget(1)
    assert tab.bottom_stack.currentIndex() == 0 and not tab.trial_page.isAncestorOf(tab.header)
    for w in (tab.checks, tab.verdict_bar, tab.btn_next_unjudged):
        assert bar.isAncestorOf(w)
    assert fixbar.isAncestorOf(tab.btn_fix_done) and tab.trial_page.isAncestorOf(tab.bottom_stack)
    assert tab.btn_fix.isEnabled() and tab.left_split.widget(0).isAncestorOf(tab.btn_fix) and tab.left_split.widget(0).isAncestorOf(tab.chk_trails)
    dot = tab.trial.dot_names[-1]
    tab.set_fix_mode(True, dot=dot)
    assert tab.bottom_stack.currentIndex() == 1 and f"Fixing {dot}" in tab.fix_bar_label.text() and "arrow keys" in tab.fix_bar_label.text()
    tab.fix_dots.on_video_click(100.0, 100.0)
    tab.set_fix_mode(False)
    assert tab.bottom_stack.currentIndex() == 0
    _finish(tab)


def test_the_picture_counter_shows_the_time_out_of_the_whole_and_the_facts_are_in_the_tooltip(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    tab.slider.setValue(3)
    assert re.fullmatch(r"4 / \d+  ·  \d+\.\d\d / \d+\.\d\d s", tab.frame_label.text()), tab.frame_label.text()
    tip = tab.video.toolTip()
    assert "frames" in tip and "P01" in tip and ("ruler" in tip)
    assert not tab.title.isVisibleTo(tab.trial_page)                                  # (nothing UNFINISHED or STOPPED: no line at all)
    _finish(tab)


def test_the_keys_one_to_five_judge_the_trial_and_f_opens_fix_dots(root):
    tab = _tab(root)
    tab.set_participant(root / "P01")
    meta = tab.trial.meta_path
    _shortcut(tab, "1").activated.emit()
    assert verdict.read_verdict(meta)["verdict"] == verdict.VALID
    _shortcut(tab, "3").activated.emit()                                              # 3 = position: compromised, and Valid is taken away
    v = verdict.read_verdict(meta)
    assert v["verdict"] == verdict.COMPROMISED and v["reasons"] == ["wrong_position"]
    _shortcut(tab, "3").activated.emit()                                              # the same key again takes the reason back
    assert verdict.read_verdict(meta).get("verdict") in (None, "")
    _shortcut(tab, "F").activated.emit()
    assert tab._fix_mode and tab.btn_fix.isChecked()
    _shortcut(tab, "1").activated.emit()                                              # (no judging while dots are fixed)
    assert verdict.read_verdict(meta).get("verdict") in (None, "")
    _shortcut(tab, "F").activated.emit()
    assert not tab._fix_mode
    _finish(tab)


# ------------------------------------------------------------------------------------------------ fix mode: the zoom and the divider
def _click_at(label, x, y):
    ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), QPointF(x, y), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    label.mousePressEvent(ev)


def test_zoom_shows_a_part_of_the_picture_around_the_dot_and_a_click_still_gives_picture_pixels(root):
    app = _app()
    tab = _tab(root)
    tab.resize(1500, 900)
    tab.show()
    tab.set_participant(root / "P01")
    tab.set_fix_mode(True, dot=tab.trial.dot_names[0])
    app.processEvents()
    w, h = tab.video.picture_size
    assert tab.video.view_rect is None                                                 # Fit: all of it
    tab.zoom_combo.setCurrentIndex(tab.zoom_combo.findData(3.0))
    app.processEvents()
    x0, y0, cw, ch = tab.video.view_rect
    assert (cw, ch) == (round(w / 3), round(h / 3)) and 0 <= x0 <= w - cw and 0 <= y0 <= h - ch
    got = []
    tab.video.clicked.connect(lambda x, y: got.append((x, y)))
    _click_at(tab.video, tab.video.width() / 2.0, tab.video.height() / 2.0)
    assert got and abs(got[0][0] - (x0 + cw / 2.0)) < 2 and abs(got[0][1] - (y0 + ch / 2.0)) < 2     # the middle of the view is the middle of the cut-out
    tab.fix_dots.on_video_click(*got[0])                                               # a click is a place in the whole picture: the pane takes it as such
    assert tab.fix_dots.click[:2] == pytest.approx(got[0])
    tab.set_fix_mode(False)
    assert tab.video.view_rect is None and not tab.zoom_combo.isVisibleTo(tab) and tab.zoom_combo.currentData() == 3.0
    _finish(tab)


def test_zoom_follows_the_last_click_and_never_leaves_the_picture(root):
    app = _app()
    tab = _tab(root)
    tab.resize(1500, 900)
    tab.show()
    tab.set_participant(root / "P01")
    tab.set_fix_mode(True, dot=tab.trial.dot_names[0])
    tab.zoom_combo.setCurrentIndex(tab.zoom_combo.findData(4.0))
    w, h = tab.video.picture_size
    tab.fix_dots.on_video_click(5.0, 5.0)                                              # in the very corner: the cut-out stays inside the picture
    app.processEvents()
    x0, y0, cw, ch = tab.video.view_rect
    assert (x0, y0) == (0, 0)
    tab.fix_dots.on_video_click(w - 3.0, h - 3.0)
    app.processEvents()
    x0, y0, cw, ch = tab.video.view_rect
    assert x0 + cw == w and y0 + ch == h
    tab.fix_dots.on_video_click(w / 2.0, h / 2.0)
    app.processEvents()
    x0, y0, cw, ch = tab.video.view_rect
    assert abs(x0 + cw / 2.0 - w / 2.0) <= 1 and abs(y0 + ch / 2.0 - h / 2.0) <= 1
    _finish(tab)


def test_the_divider_can_be_pulled_far_to_the_right_while_dots_are_fixed_and_comes_back(root):
    app = _app()
    tab = _tab(root)
    tab.resize(1700, 950)
    tab.show()
    tab.set_participant(root / "P01")
    app.processEvents()
    normal = tab.top_splitter.sizes()
    tab.set_fix_mode(True)
    app.processEvents()
    left, right = tab.top_splitter.sizes()
    assert left > right * 1.8                                                           # about 70 % for the video by default
    tab.top_splitter.setSizes([1500, 100])
    app.processEvents()
    left, right = tab.top_splitter.sizes()
    assert right >= 320 - 1 and left > 1250
    tab.set_fix_mode(False)
    app.processEvents()
    assert abs(tab.top_splitter.sizes()[0] - normal[0]) <= 3                            # the graphs side is as it was
    _finish(tab)
