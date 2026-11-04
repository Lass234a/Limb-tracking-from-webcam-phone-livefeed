"""The keys of the Review tab (2026-11-04): a key for every control that never takes the keyboard, so that nothing in Review needs the mouse
except the dots of Fix dots, which are clicked on the picture by nature.

These tests press real keys through Qt (the window is made active first, so Qt looks for a shortcut exactly as it does for a person at the
keyboard). A test that only called a function would not notice a wrong key, a key that Qt refuses because two shortcuts share it, or a key that a
box swallowed. Three things matter most, because a wrong key here could write a wrong verdict:
  - while the list of checks is open it owns the keyboard (the key 1 is "valid" on the page, and "the first button" in the list);
  - typing in a box (the zero, the note) never fires a key of the page;
  - nothing acts without the question a click would ask (Approve review asks first).
"""
import hashlib
import os
import re
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QShortcut  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from limbtrack import review_layer, verdict  # noqa: E402
from limbtrack.review import ReviewTab  # noqa: E402
from limbtrack.review_checks import MAX_KEYED_BUTTONS, ChecksChip  # noqa: E402

from tests.test_pilot_checks import recording_at  # noqa: E402

# The table of keys, written out here on purpose: a key taken away, moved or renamed fails a test, and a new key has to be added here and said
# on its control. (scope, key, where the control says it, what it says)
KEYS = [
    ("head", "Alt+1", "btn_view_trial", "Key: Alt+1."),
    ("head", "Alt+2", "btn_view_overview", "Key: Alt+2."),
    ("head", "Alt+O", "btn_data_original", "Key: Alt+O."),
    ("head", "Alt+R", "btn_data_reviewed", "Key: Alt+R."),
    ("trial", "C", "checks.chip", "Key: C"),
    ("trial", "N", "verdict_bar.btn_note", "Key: N."),
    ("trial", "A", "angle_tree.reference_button", "(key A)"),
    ("trial", "M", "angle_tree.more_check", "Key: M."),
    ("trial", "U", "force_tree.open_all_link", "Key: U."),
    ("trial", "Z", "zero_spin", "Key: Z"),
    ("trial", "Shift+Z", "force_tree.zero_from", "Key: Shift+Z."),
    ("overview", "A", "overview.approve_button", "Key: A."),
    ("overview", "S", "overview.session_button", "Key: S."),
    ("overview", "E", "overview.export_button", "Key: E."),
    ("overview", "M", "overview.more_button", "Key: M."),
]


def _attr(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _scope(tab, name):
    return {"head": tab, "trial": tab.trial_page, "overview": tab.overview}[name]


def _shortcuts(tab, key, scope):
    return [sc for sc in tab.findChildren(QShortcut) if sc.key().toString() == key and sc.parent() is _scope(tab, scope)]


def _said(w):
    """What a widget says to a person: its tooltip and its text (a few widgets have a `text` that is not a method)."""
    text = getattr(w, "text", None)
    return w.toolTip() + " " + (text() if callable(text) else "")


def press(widget, key, mods=Qt.NoModifier):
    """A key pressed the way a person does: Qt first looks for a shortcut that the focused part does not claim, then delivers the key."""
    QTest.keyClick(widget, key, mods)
    QApplication.processEvents()


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    """recordings/P01 with two real trials (a recorded video and a simulated force): enough to open a trial, the overview and the next trial."""
    root = tmp_path_factory.mktemp("key_recordings")
    for _ in range(2):
        recording_at(root, 70.0, seconds=1.5)
    return root


@pytest.fixture()
def tab(recordings, tmp_path):
    """A Review tab on a fresh copy, its window active, a trial open and the picture holding the keyboard (as after a click on the picture)."""
    dest = tmp_path / "recordings"
    shutil.copytree(recordings, dest)
    app = QApplication.instance() or QApplication([])
    t = ReviewTab(dest)
    t.resize(1500, 900)
    t.show()
    t.activateWindow()
    assert QTest.qWaitForWindowActive(t, 3000), "the window did not become active, so Qt would not look for shortcuts"
    t.set_participant(dest / "P01")
    app.processEvents()
    t.video.setFocus(Qt.OtherFocusReason)
    focus = QApplication.focusWidget()
    assert focus is not None and t.trial_page.isAncestorOf(focus)
    yield t
    if t.trial is not None:
        t.trial.close()
    t.overview.shutdown()
    t.close()


@pytest.fixture()
def overview_tab(tab):
    """The same tab on the overview page, the rows worked out, the table holding the keyboard."""
    tab.show_view("overview")
    assert tab.overview.wait_until_done(120000)
    QApplication.processEvents()
    tab.overview.table.setFocus(Qt.OtherFocusReason)
    focus = QApplication.focusWidget()
    assert focus is not None and tab.overview.isAncestorOf(focus)
    return tab


def verdict_of(tab):
    return (verdict.read_verdict(tab.trial.meta_path) or {}).get("verdict")


# ---------------------------------------------------------------------------------------------------------------- the table of keys
def test_the_table_of_keys_is_bound_exactly_and_each_control_says_its_key(tab):
    names = {id(tab): "head", id(tab.trial_page): "trial", id(tab.overview): "overview"}
    assert {(names[id(scope)], seq) for scope, seq, _control in tab.keyed} == {(scope, key) for scope, key, _where, _says in KEYS}
    assert len(tab.keyed) == len(KEYS)
    for scope, key, where, says in KEYS:
        bound = _shortcuts(tab, key, scope)
        assert len(bound) == 1, f"{key} on {scope}: {len(bound)} shortcuts"
        assert bound[0].context() == Qt.WidgetWithChildrenShortcut, key
        said = _attr(tab, where).toolTip()
        assert says in said, f"{where} should say {says!r}: {said!r}"


def test_no_two_keys_can_be_active_together_because_qt_then_fires_neither(tab):
    """Two shortcuts with one key, where the focus can be inside both, make Qt treat the key as ambiguous and do nothing. The trial page and the
    overview are separate pages, so the same letter on both (A and M) is allowed; the tab itself holds both."""
    shortcuts = tab.findChildren(QShortcut)
    assert len(shortcuts) > 30
    clashes = []
    for i, a in enumerate(shortcuts):
        for b in shortcuts[i + 1:]:
            if a.key().toString() != b.key().toString():
                continue
            pa, pb = a.parent(), b.parent()
            if pa is pb or pa.isAncestorOf(pb) or pb.isAncestorOf(pa):
                clashes.append((a.key().toString(), type(pa).__name__, type(pb).__name__))
    assert clashes == []


def test_every_key_the_screen_promises_is_bound(tab):
    """A hint such as 'Key: M.' or '(key A)' on the trial page, the overview or the bottom bar must name a key that really is bound: a hint that
    outlives its key would send a person to a key that does nothing."""
    bound = {sc.key().toString() for sc in tab.findChildren(QShortcut)}
    named = []
    for w in tab.findChildren(QWidget):
        text = _said(w)
        named += re.findall(r"\bKey: ([^\s.;]+)", text) + re.findall(r"\(key ([^)\s]+)\)", text)
    assert len(named) >= len(KEYS)                                                       # (it did look at the hints)
    assert [k for k in named if k not in bound] == []


# ---------------------------------------------------------------------------------------------------------------- the header: both pages
def test_alt_1_and_alt_2_switch_between_the_trial_and_the_overview_from_either_page(tab):
    press(tab.video, Qt.Key_2, Qt.AltModifier)
    assert tab.stack.currentIndex() == 1 and tab.btn_view_overview.isChecked() and not tab.btn_view_trial.isChecked()
    press(tab, Qt.Key_2, Qt.AltModifier)                                                 # (again on the overview: stays there)
    assert tab.stack.currentIndex() == 1
    press(tab, Qt.Key_1, Qt.AltModifier)
    assert tab.stack.currentIndex() == 0 and tab.btn_view_trial.isChecked() and not tab.btn_view_overview.isChecked()


def test_alt_o_and_alt_r_switch_between_the_original_and_the_reviewed_data(tab):
    assert tab._source == "reviewed"
    press(tab.video, Qt.Key_O, Qt.AltModifier)
    assert tab._source == "original" and tab.btn_data_original.isChecked() and not tab.btn_data_reviewed.isChecked()
    press(tab.video, Qt.Key_R, Qt.AltModifier)
    assert tab._source == "reviewed" and tab.btn_data_reviewed.isChecked() and not tab.btn_data_original.isChecked()


def test_the_alt_keys_work_while_a_number_is_being_typed(tab):
    """Alt+2 is not a character, so a box does not claim it: the page can be left from the zero box."""
    tab.force_tree.open_zero()
    focus = QApplication.focusWidget()
    assert focus is tab.zero_spin or tab.zero_spin.isAncestorOf(focus)
    press(focus, Qt.Key_2, Qt.AltModifier)
    assert tab.stack.currentIndex() == 1


def test_from_a_drop_down_the_alt_keys_act_and_the_letter_keys_wait_for_the_page(tab):
    """What the README tells a person: right after a trial is chosen the keyboard is on the drop-down; there the Alt keys work and a letter does
    not (a stray letter in a drop-down must not act), until the picture is clicked or Tab moves into the page."""
    tab.trial_combo.setFocus(Qt.TabFocusReason)
    assert QApplication.focusWidget() is tab.trial_combo
    index = tab.trial_combo.currentIndex()
    press(tab.trial_combo, Qt.Key_C)
    press(tab.trial_combo, Qt.Key_1)
    assert not tab.checks.popup.isVisible() and verdict_of(tab) is None and tab.trial_combo.currentIndex() == index
    press(tab.trial_combo, Qt.Key_2, Qt.AltModifier)
    assert tab.stack.currentIndex() == 1
    press(tab.trial_combo, Qt.Key_1, Qt.AltModifier)
    assert tab.stack.currentIndex() == 0


# ---------------------------------------------------------------------------------------------------------------- the trial page
def test_c_opens_the_list_of_checks_and_c_closes_it(tab):
    chip = tab.checks
    assert not chip.popup.isVisible()
    press(tab.video, Qt.Key_C)
    assert chip.popup.isVisible()
    press(chip.popup, Qt.Key_C)                                                          # (a pop-up has the keyboard: the key goes to it)
    assert not chip.popup.isVisible()
    press(tab.video, Qt.Key_C)                                                           # (the same guard as for the click: a list that has just closed does not open again)
    assert not chip.popup.isVisible()
    QTest.qWait(300)
    press(tab.video, Qt.Key_C)
    assert chip.popup.isVisible()
    press(chip.popup, Qt.Key_Escape)
    assert not chip.popup.isVisible()
    QTest.qWait(300)
    tab.set_fix_mode(True)                                                               # (the chip is not on screen in Fix dots: the key has nothing to open)
    press(tab.video, Qt.Key_C)
    assert not chip.popup.isVisible()


def test_while_the_list_of_checks_is_open_no_key_of_the_page_acts_through_it(tab, monkeypatch):
    """The key 1 is 'valid' on the page. In the open list it means 'the first button'. A person who presses 1 to press that button must not
    write a verdict (this was found in a probe before the fix: the verdict was written and the button was not pressed)."""
    asked = []
    monkeypatch.setattr(tab, "open_privacy", lambda: asked.append("privacy"))           # (the first button of this trial's list is Privacy)
    assert verdict_of(tab) is None
    press(tab.video, Qt.Key_C)
    assert tab.checks.popup.isVisible()
    first = tab.checks.keyed_buttons()[0]
    assert first.text() == "Privacy"
    press(tab.checks.popup, Qt.Key_1)
    assert asked == ["privacy"] and verdict_of(tab) is None
    for key in (Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_F, Qt.Key_P, Qt.Key_N, Qt.Key_Z):     # keys of the page that have no meaning in the list
        tab.checks.show_popup()
        press(tab.checks.popup, key)
        assert verdict_of(tab) is None and not tab._fix_mode and asked == ["privacy"], key


def _items(n, actions=("zero", "window", "calibration", "fix_dots", "privacy")):
    return [{"level": "check", "title": f"Check {i + 1}", "text": f"Sentence {i + 1} " * 12, "action": actions[i % len(actions)], "arg": ""} for i in range(n)]


@pytest.fixture()
def chip():
    QApplication.instance() or QApplication([])
    c = ChecksChip()
    c.resize(400, 30)
    c.show()
    yield c
    c.popup.hide()
    c.close()


def test_in_the_open_list_the_keys_1_to_9_press_the_buttons_in_order_and_there_is_no_tenth(chip):
    asked = []
    chip.actionRequested.connect(lambda action, arg: asked.append(action))
    chip.set_items(_items(11))
    buttons = chip.keyed_buttons()
    assert len(buttons) == MAX_KEYED_BUTTONS == 9
    chip.show_popup()
    assert [b.toolTip().rsplit("(key ", 1)[1] for b in buttons] == [f"{n})" for n in range(1, 10)]
    for n, key in enumerate((Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9)):
        chip.show_popup()                                                                # (a button closes the list)
        asked.clear()
        press(chip.popup, key)
        assert asked == [_items(11)[n]["action"]], (n + 1, asked)
    chip.show_popup()
    asked.clear()
    press(chip.popup, Qt.Key_0)                                                          # (0 is no button)
    assert asked == [] and chip.popup.isVisible()


def test_a_key_for_a_line_with_no_button_does_nothing_and_leaves_the_list_open(chip):
    asked = []
    chip.actionRequested.connect(lambda action, arg: asked.append(action))
    chip.set_items(_items(2) + [{"level": "check", "title": "Only a note to read", "text": "No button here.", "action": None}])
    chip.show_popup()
    press(chip.popup, Qt.Key_3)
    assert asked == [] and chip.popup.isVisible()
    press(chip.popup, Qt.Key_2)
    assert asked == ["window"] and not chip.popup.isVisible()


def test_the_key_hints_in_the_list_do_not_pile_up_when_it_is_opened_again(chip):
    chip.set_items(_items(3))
    for _ in range(3):
        chip.show_popup()
        chip.popup.hide()
    assert [b.toolTip().count("(key ") for b in chip.keyed_buttons()] == [1, 1, 1]
    assert chip.keyed_buttons()[0].toolTip().endswith("(key 1)")


def test_return_in_the_open_list_reads_every_sentence_and_again_folds_them(chip):
    chip.set_items(_items(3))
    chip.show_popup()
    pill = chip.panel.pill_check
    assert pill.isVisibleTo(chip.popup) and not pill.isChecked()
    press(chip.popup, Qt.Key_Return)
    assert pill.isChecked() and all(r.expanded for r in chip.panel.rows)
    press(chip.popup, Qt.Key_Enter)
    assert not pill.isChecked() and not any(r.expanded for r in chip.panel.rows)
    assert chip.popup.isVisible()


def test_n_opens_the_note_and_does_so_only_with_a_trial_open_and_not_while_dots_are_fixed(tab, monkeypatch):
    shown = []
    monkeypatch.setattr(tab.verdict_bar.btn_note, "showMenu", lambda: shown.append("note"))
    press(tab.video, Qt.Key_N)
    assert shown == ["note"]
    tab.set_fix_mode(True)
    press(tab.video, Qt.Key_N)
    assert shown == ["note"]                                                              # (the same as the click: the bar is the Fix dots bar then)
    tab.set_fix_mode(False)
    tab.verdict_bar.setEnabled(False)
    press(tab.video, Qt.Key_N)
    assert shown == ["note"]


def test_a_opens_the_reference_readings_and_only_while_the_button_is_there(tab):
    asked = []
    tab.angle_tree.referenceRequested.disconnect()                                        # (the real one opens a dialog and waits for a person)
    tab.angle_tree.referenceRequested.connect(lambda: asked.append("reference"))
    press(tab.video, Qt.Key_A)
    assert asked == ["reference"]
    tab.set_fix_mode(True)                                                                # (the angle numbers are not on screen in Fix dots)
    press(tab.video, Qt.Key_A)
    assert asked == ["reference"]


def test_m_ticks_more_numbers_and_u_opens_and_closes_every_number(tab):
    more = tab.angle_tree.more_check
    assert not more.isChecked()
    press(tab.video, Qt.Key_M)
    assert more.isChecked()
    press(tab.video, Qt.Key_M)
    assert not more.isChecked()
    link = tab.force_tree.open_all_link
    assert link.isVisible() and link.text() == "Open all"
    press(tab.video, Qt.Key_U)
    assert link.text() == "Close all"
    press(tab.video, Qt.Key_U)
    assert link.text() == "Open all"


def test_z_gives_the_keyboard_to_the_zero_box_and_shift_z_opens_the_zero_menu(tab, monkeypatch):
    assert tab.trial.has_force()
    press(tab.video, Qt.Key_Z)
    focus = QApplication.focusWidget()
    assert focus is tab.zero_spin or tab.zero_spin.isAncestorOf(focus), focus
    assert tab.force_tree._open["Above rest"]                                            # (the row that holds the box was opened)
    shown = []
    monkeypatch.setattr(tab.force_tree.zero_from, "showMenu", lambda: shown.append("zero menu"))
    tab.video.setFocus(Qt.OtherFocusReason)
    press(tab.video, Qt.Key_Z, Qt.ShiftModifier)
    assert shown == ["zero menu"]
    tab.force_tree.set_fold("Above rest", False)
    monkeypatch.setattr(tab.trial, "has_force", lambda: False)                           # (a trial with no force has no zero box to open)
    press(tab.video, Qt.Key_Z)
    assert tab.force_tree._open["Above rest"] is False
    monkeypatch.undo()
    tab.set_fix_mode(True)                                                               # (the force numbers are not on screen in Fix dots)
    tab.force_tree.set_fold("Above rest", False)
    press(tab.video, Qt.Key_Z)
    assert tab.force_tree._open["Above rest"] is False


def test_return_in_the_zero_box_applies_the_zero_once_like_the_button(tab, monkeypatch):
    """Once: Qt 6.11 has a `returnPressed` signal of its own on spin boxes, and a second one under that name applied the zero twice."""
    saves = []
    original = ReviewTab._save_zero
    monkeypatch.setattr(ReviewTab, "_save_zero", lambda self, *a, **k: (saves.append(a), original(self, *a, **k))[1])
    spin = tab.zero_spin
    start = spin.value()
    tab.force_tree.open_zero()
    spin.setValue(start + 0.01)
    target = spin.value()                                                                # (the box rounds to its own number of decimals)
    assert target != start
    assert review_layer.Review.load(tab.trial.meta_path).zero is None and saves == []    # (nothing is saved by typing alone)
    press(spin.lineEdit(), Qt.Key_Return)
    saved = review_layer.Review.load(tab.trial.meta_path).zero
    assert saved is not None and saved["source"] == "typed" and saved["zero_v"] == pytest.approx(target)
    assert saves == [(target, "typed")], saves


def test_typing_in_the_zero_box_never_fires_a_key_of_the_page(tab):
    """The letters and digits of the page keys are typed into the box (which refuses the ones it cannot use), and nothing else happens."""
    tab.force_tree.open_zero()
    focus = QApplication.focusWidget()
    assert focus is tab.zero_spin or tab.zero_spin.isAncestorOf(focus)
    more_before, text_before = tab.angle_tree.more_check.isChecked(), tab.force_tree.open_all_link.text()
    for ch in "CNAMUZFP12345":
        QTest.keyClicks(focus, ch)
    QApplication.processEvents()
    assert not tab.checks.popup.isVisible() and not tab._fix_mode
    assert verdict_of(tab) is None
    assert tab.angle_tree.more_check.isChecked() == more_before and tab.force_tree.open_all_link.text() == text_before
    assert tab.stack.currentIndex() == 0


def test_the_keys_do_nothing_and_break_nothing_with_no_trial_open(recordings, tmp_path):
    dest = tmp_path / "recordings"
    shutil.copytree(recordings, dest)
    t = ReviewTab(dest)
    t.resize(1500, 900)
    t.show()
    t.activateWindow()
    assert QTest.qWaitForWindowActive(t, 3000)
    t.video.setFocus(Qt.OtherFocusReason)
    assert t.trial is None
    for key, mods in ((Qt.Key_N, Qt.NoModifier), (Qt.Key_A, Qt.NoModifier), (Qt.Key_M, Qt.NoModifier), (Qt.Key_U, Qt.NoModifier),
                      (Qt.Key_Z, Qt.NoModifier), (Qt.Key_Z, Qt.ShiftModifier), (Qt.Key_1, Qt.NoModifier), (Qt.Key_C, Qt.NoModifier)):
        press(t.video, key, mods)
    t.checks.popup.hide()                                                                # (C opens the list, as a click on the chip does; it holds no checks)
    assert t.trial is None and t.stack.currentIndex() == 0 and not t._fix_mode
    assert not (dest / "review").exists()                                                # (no verdict, no zero: nothing was written)
    t.overview.shutdown()
    t.close()


def test_a_person_with_only_a_keyboard_can_reach_the_page_keys_by_tab(tab):
    """The letter keys act while the keyboard is on the page (so a stray letter elsewhere, such as in a drop-down, does not act). From the trial
    drop-down, one press of Tab leads into the page (the test allows three), and the keys then work."""
    tab.trial_combo.setFocus(Qt.TabFocusReason)
    for presses in range(1, 13):
        press(QApplication.focusWidget(), Qt.Key_Tab)
        if tab.trial_page.isAncestorOf(QApplication.focusWidget()):
            break
    else:
        pytest.fail("Tab did not lead from the trial drop-down into the page in 12 presses")
    assert presses <= 3, presses                                                         # (it is 1 today)
    press(QApplication.focusWidget(), Qt.Key_C)
    assert tab.checks.popup.isVisible()
    press(tab.checks.popup, Qt.Key_Escape)


# ---------------------------------------------------------------------------------------------------------------- the overview page
def test_s_e_and_m_on_the_overview_do_what_their_buttons_do(overview_tab, monkeypatch):
    ov = overview_tab.overview
    seen = []
    ov.sessionCheckRequested.disconnect()
    ov.sessionCheckRequested.connect(lambda *a: seen.append("session"))
    monkeypatch.setattr(ov.export_button, "showMenu", lambda: seen.append("export menu"))
    monkeypatch.setattr(ov.more_button, "showMenu", lambda: seen.append("more menu"))
    for key in (Qt.Key_S, Qt.Key_E, Qt.Key_M):
        press(ov.table, key)
    assert seen == ["session", "export menu", "more menu"]


def test_a_on_the_overview_asks_first_and_approves_only_when_the_answer_is_yes(overview_tab):
    ov = overview_tab.overview
    assert ov.approve_button.isEnabled()
    asked = []
    ov.confirm = lambda *a: asked.append(a[1]) or False
    press(ov.table, Qt.Key_A)
    assert len(asked) == 1 and not list(overview_tab.dir.rglob("approval.json"))         # (asked, and the answer no left no approval)
    ov.confirm = lambda *a: asked.append(a[1]) or True
    press(ov.table, Qt.Key_A)
    assert len(asked) == 2 and len(list(overview_tab.dir.rglob("approval.json"))) == 1


def test_the_overview_keys_do_not_act_on_the_trial_page_and_the_trial_keys_not_on_the_overview(overview_tab, monkeypatch):
    """A and M mean one thing on the overview and another on the trial page; the other letters belong to one page only."""
    tab, ov, seen = overview_tab, overview_tab.overview, []
    ov.confirm = lambda *a: seen.append("approve asked") or False
    ov.sessionCheckRequested.disconnect()
    ov.sessionCheckRequested.connect(lambda *a: seen.append("session"))
    tab.angle_tree.referenceRequested.disconnect()
    tab.angle_tree.referenceRequested.connect(lambda: seen.append("reference"))
    monkeypatch.setattr(ov.export_button, "showMenu", lambda: seen.append("export menu"))
    monkeypatch.setattr(ov.more_button, "showMenu", lambda: seen.append("more menu"))
    more = tab.angle_tree.more_check
    for key in (Qt.Key_A, Qt.Key_M, Qt.Key_C, Qt.Key_U, Qt.Key_Z, Qt.Key_N):             # on the overview: A asks to approve, M opens More, the rest are for the trial page
        press(ov.table, key)
    assert seen == ["approve asked", "more menu"] and not more.isChecked() and not tab.checks.popup.isVisible()
    press(tab, Qt.Key_1, Qt.AltModifier)
    assert tab.stack.currentIndex() == 0
    tab.video.setFocus(Qt.OtherFocusReason)
    seen.clear()
    for key in (Qt.Key_S, Qt.Key_E):                                                      # on the trial page: S and E are for the overview
        press(tab.video, key)
    assert seen == []
    press(tab.video, Qt.Key_A)                                                            # A is the reference readings here, M ticks More numbers
    press(tab.video, Qt.Key_M)
    assert seen == ["reference"] and more.isChecked()


def test_a_key_changes_no_recorded_file(tab):
    """Every key above only looks, moves or asks. The recorded files of the trial stay as they were, byte for byte."""
    folder = tab.trial.meta_path.parent
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}
    for key, mods in ((Qt.Key_C, Qt.NoModifier), (Qt.Key_Escape, Qt.NoModifier), (Qt.Key_M, Qt.NoModifier), (Qt.Key_U, Qt.NoModifier),
                      (Qt.Key_Z, Qt.NoModifier), (Qt.Key_O, Qt.AltModifier), (Qt.Key_R, Qt.AltModifier), (Qt.Key_2, Qt.AltModifier), (Qt.Key_1, Qt.AltModifier)):
        tab.video.setFocus(Qt.OtherFocusReason)
        press(tab.video, key, mods)
        if tab.checks.popup.isVisible():
            tab.checks.popup.hide()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}
    assert after == before
