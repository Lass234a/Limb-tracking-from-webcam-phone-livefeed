"""Shared test set-up."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _close_the_cue_threads(monkeypatch):
    """Every Engine makes a CuePlayer whose thread is started by the first cue and was never closed, so a full run piled up
    100 and more idle 'audio-cues' threads (and a crash dump, which lists at most 100 threads, never reached the main thread).
    Every player made during a test is closed, and its thread joined, when the test ends."""
    from limbtrack import cues

    made = []
    original = cues.CuePlayer.__init__

    def remember(self, *a, **k):
        original(self, *a, **k)
        made.append(self)
    monkeypatch.setattr(cues.CuePlayer, "__init__", remember)
    yield
    for player in made:
        player.close()
    for player in made:
        t = player._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)


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
