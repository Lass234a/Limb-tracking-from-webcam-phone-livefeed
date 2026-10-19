"""The panel building blocks: sections that open one at a time, and the strip of step chips."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QVBoxLayout  # noqa: E402

from limbtrack.panels import Accordion, Section, StepStrip  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_a_section_opens_and_closes_and_shows_its_summary_only_while_closed(app):
    sec = Section("camera", "Camera", [QLabel("body")], number="1")
    assert not sec.is_open() and sec.body.isHidden() and "▸" in sec.header.text() and "1  Camera" in sec.header.text()
    sec.set_summary("DroidCam Video")
    assert "DroidCam Video" in sec.header.text()                 # a closed section says what it is set to
    sec.set_open(True)
    assert sec.is_open() and not sec.body.isHidden() and "▾" in sec.header.text()
    assert "DroidCam Video" not in sec.header.text()             # an open one shows its content instead
    sec.set_open(False)
    assert "DroidCam Video" in sec.header.text() and sec.body.isHidden()


def test_a_group_box_loses_its_own_title_and_frame_inside_a_section(app):
    box = QGroupBox("Video source (set once per session)")
    QVBoxLayout(box).addWidget(QLabel("x"))
    sec = Section("k", "Camera", [box])                          # (kept, or Qt deletes the box with it)
    assert box.title() == "" and box.isFlat() and sec.body.isAncestorOf(box)


def test_opening_a_section_closes_the_others_and_all_can_be_closed(app):
    acc = Accordion()
    for k in ("a", "b", "c"):
        acc.add(k, k.upper(), [QLabel(k)])
    seen = []
    acc.opened.connect(seen.append)
    acc.open("a")
    assert acc.current() == "a"
    acc.open("b")
    assert acc.current() == "b" and not acc.sections["a"].is_open()
    assert [s.is_open() for s in acc.sections.values()] == [False, True, False]
    assert seen == ["a", "b"]
    acc.sections["b"].set_open(False)                              # clicking the open header closes it
    assert acc.current() is None
    acc.open("nonsense")                                           # an unknown key is ignored
    assert acc.current() is None


def test_summaries_are_kept_per_section(app):
    acc = Accordion()
    acc.add("a", "A", [QLabel("a")])
    acc.add("b", "B", [QLabel("b")])
    acc.set_summary("a", "one")
    acc.set_summary("b", "two")
    acc.set_summary("zzz", "ignored")
    assert "one" in acc.sections["a"].header.text() and "two" in acc.sections["b"].header.text()


def test_the_step_chips_show_their_state_and_report_a_click(app):
    strip = StepStrip([("camera", "Camera"), ("dots", "Dots"), ("record", "Record")])
    clicked = []
    strip.activated.connect(clicked.append)
    assert strip.state_of("camera") == "off"
    strip.set_state("camera", "done", "Camera", "tip")
    strip.set_state("dots", "next", "Dots 3/7", "Click each dot")
    strip.set_state("record", "todo")
    assert strip.chips["camera"].text().startswith("✓") and strip.chips["camera"].toolTip() == "tip"
    assert strip.chips["dots"].text().startswith("▶") and "3/7" in strip.chips["dots"].text()
    assert strip.chips["record"].text() == "Record"
    assert strip.state_of("dots") == "next" and "fbbf24" in strip.chips["dots"].styleSheet()
    strip.chips["dots"].click()
    strip.chips["camera"].click()
    assert clicked == ["dots", "camera"]


def test_setting_the_same_state_again_changes_nothing(app):
    strip = StepStrip([("camera", "Camera")])
    strip.set_state("camera", "done", "Camera", "t")
    sheet = strip.chips["camera"].styleSheet()
    strip.chips["camera"].setStyleSheet("")                        # if the chip were restyled it would get its sheet back
    strip.set_state("camera", "done", "Camera", "t")
    assert strip.chips["camera"].styleSheet() == "" and sheet
