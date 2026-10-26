"""Shared test set-up."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _answer_the_tested_leg_question(monkeypatch):
    """Record asks for the tested leg when none was chosen for the participant (0.20.0), in a dialog that would wait for a click.
    Window tests that are not about that question answer it 'Right leg', the leg the program assumed for everyone before 0.20.0.
    tests/test_tested_leg.py asks the real question."""
    from limbtrack import gui

    def answer(self):
        self.leg_combo.setCurrentIndex(self.leg_combo.findData("right"))
        return self.engine.tested_leg == "right"
    monkeypatch.setattr(gui.LiveTab, "_ask_tested_leg", answer)
