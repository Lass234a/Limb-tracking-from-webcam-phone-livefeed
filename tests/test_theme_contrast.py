"""The step chips and the dot buttons of the Live tab are readable on a light window and on a dark one (WCAG AA: 4.5:1 for text), and the
dark window gets exactly the styles it always had. The contrast is worked out from the colours in the styles, blended over the window's own colour."""
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack import gui, panels  # noqa: E402

LIGHT_WINDOWS = ("#ffffff", "#f0f0f0")
DARK_WINDOWS = ("#202020", "#2b2b2b")
WCAG_AA = 4.5


def palette(window):
    pal = QPalette()
    win = QColor(window)
    pal.setColor(QPalette.Window, win)
    pal.setColor(QPalette.WindowText, QColor("#101010") if win.lightness() > 128 else QColor("#e6e6e6"))
    return pal


def _lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def parse(css, prop):
    """The colour of `prop` ('color' or 'background') in a style as (r, g, b, alpha); None for 'transparent' or when it is not there."""
    m = re.search(rf"(?<![-\w]){prop}:\s*(#[0-9a-fA-F]{{3,6}}|rgba?\([^)]*\)|transparent)", css)
    if not m or m.group(1) == "transparent":
        return None
    v = m.group(1)
    if v.startswith("#"):
        c = QColor(v)
        return c.red(), c.green(), c.blue(), 1.0
    parts = [float(x) for x in v[v.index("(") + 1:-1].split(",")]
    return parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else 1.0


def text_contrast(css, window):
    """The text colour of a style against its own background laid over the window colour."""
    win = QColor(window)
    base = (win.red(), win.green(), win.blue())
    bg = parse(css, "background")
    if bg is not None:
        a = bg[3]
        base = tuple(round(bg[i] * a + base[i] * (1 - a)) for i in range(3))
    fg = parse(css, "color")
    return contrast(fg[:3], base)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


STATES = ("done", "todo", "next", "off", "check")
DOT_KEYS = ("ok", "lost", "unmarked", "armed")


@pytest.mark.parametrize("window", LIGHT_WINDOWS + DARK_WINDOWS)
def test_every_step_chip_and_dot_button_is_readable_on_the_window_it_is_on(app, window):
    pal = palette(window)
    for state in STATES:
        assert text_contrast(panels.chip_style(state, pal), window) >= WCAG_AA, (state, window)
    for key in DOT_KEYS:
        for pending in (False, True):
            assert text_contrast(gui.dot_chip_style(key, pending, pal), window) >= WCAG_AA, (key, pending, window)


@pytest.mark.parametrize("window", DARK_WINDOWS)
def test_a_dark_window_gets_exactly_the_styles_it_always_had(app, window):
    pal = palette(window)
    assert all(panels.chip_style(s, pal) == panels.CHIP_STYLES[s] for s in STATES)
    assert all(gui.dot_chip_style(k, False, pal) == gui.DOT_CHIP_STYLE[k] and gui.dot_chip_style(k, True, pal) == gui.DOT_CHIP_PENDING[k] for k in DOT_KEYS)


def test_only_the_text_colour_changes_on_a_light_window_never_the_meaning(app):
    """Same border, same background, same weight; the 'next' chip and the 'armed' dot are the same on both."""
    def without_colour(css):
        return re.sub(r"(?<![-\w])color:\s*#[0-9a-fA-F]+;\s*", "", css)
    for s in STATES:
        assert without_colour(panels.CHIP_STYLES_LIGHT[s]) == without_colour(panels.CHIP_STYLES[s]), s
    for k in DOT_KEYS:
        assert without_colour(gui.DOT_CHIP_STYLE_LIGHT[k]) == without_colour(gui.DOT_CHIP_STYLE[k]), k
    assert panels.CHIP_STYLES_LIGHT["next"] == panels.CHIP_STYLES["next"] and gui.DOT_CHIP_STYLE_LIGHT["armed"] == gui.DOT_CHIP_STYLE["armed"]


def test_the_strip_restyles_its_chips_when_the_theme_changes(app):
    old = app.palette()
    try:
        app.setPalette(palette("#ffffff"))
        strip = panels.StepStrip([("camera", "Camera"), ("force", "Force")])
        strip.set_state("camera", "done")
        light = strip.chips["camera"].styleSheet()
        assert panels.CHIP_STYLES_LIGHT["done"] in light and panels.CHIP_STYLES["done"] not in light
        app.setPalette(palette("#202020"))
        QApplication.processEvents()
        assert panels.CHIP_STYLES["done"] in strip.chips["camera"].styleSheet()
    finally:
        app.setPalette(old)


def test_the_measurement_itself_is_right():
    """Black on white is 21:1 and the old light-window chips were really unreadable (the reason for this change)."""
    assert contrast((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0)
    assert text_contrast(panels.CHIP_STYLES["done"], "#ffffff") < 2.0 and text_contrast(panels.CHIP_STYLES["todo"], "#f0f0f0") < 2.0
