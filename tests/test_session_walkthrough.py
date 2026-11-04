"""A whole session through the real window, the way a first-time operator would do it: they do only what the screen tells them.

At every step the highlighted chip and the one-line instruction must say what to do next (a cognitive walkthrough, automated), the action is
done through the real widgets, and the actions are counted. The count is a keystroke-level estimate of the work, and a friction budget: if a
later change makes a session need more actions than this, the test says so and the change has to be justified.

After the trial: its files are saved, the independent check (tools/check_session.py) finds nothing wrong, Review opens it, a verdict is given,
the export is written, and not one byte of the recorded files has changed.
"""
import hashlib
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.gui import STEPS, MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import check_session  # noqa: E402

# Keystroke-level model (Card, Moran and Newell 1980; the usual values): a mouse action is pointing and pressing, a key is one keystroke, and
# each step starts with a moment of thinking.
KLM_SECONDS = {"click": 1.1 + 0.1, "key": 0.28, "think": 1.35}
# What a first trial needs today, measured on this walkthrough, plus about a fifth of room. It is a budget, not a goal: fewer is better.
MAX_ACTIONS_TO_FIRST_TRIAL = 25                                                # (21 measured on 2026-11-04)
NOTICE_SECONDS = 6.0                                                           # how long the program shows a timed notice (gui.py: say(..., seconds=6.0))
PLAN = {"camera": "camera", "force": "force", "markers": "beeps", "participant": "participant", "privacy": "face", "ruler": "ruler",
        "dots": "dot", "lock": "lock", "record": "record"}                    # a word the instruction line must contain while that step is the next one


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("walkthrough_clip") / "leg.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(900):                                                         # (30 s: the recording and its 2 s tail fit in it)
        w.write(render(leg_points(hip_flex=90, knee_flex=70 + 1.5 * math.sin(i / 25), ankle_df=5), seed=i))
    w.release()
    return path


class Operator:
    """Does what the instruction line says and counts every click and key."""

    def __init__(self, win, app, video):
        self.win, self.live, self.app, self.video = win, win.live, app, video
        self.actions = []                                                          # (step, kind, what)
        self.said = []                                                             # (step, the instruction the operator was reading)
        self.latency = []                                                          # seconds the screen took to move on after each step
        self.notice_seconds = []                                                   # how long a timed notice covered the instruction line
        self.dialogs = []

    def pump(self, cond=lambda: False, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    def do(self, step, kind, what, fn=None):
        self.actions.append((step, kind, what))
        if fn is not None:
            fn()
        self.app.processEvents()

    def next_step(self):
        return next((k for k, (state, _label, _tip) in self.live.strip.states.items() if state == "next"), None)

    def perform(self, step):
        live, e = self.live, self.win.engine
        if step == "camera":
            self.do(step, "click", "choose the camera")
            self.do(step, "click", "Start camera", lambda: live.start_source(str(self.video)))
            assert self.pump(lambda: e._last_gray is not None), "no picture arrived"
        elif step == "force":                                                       # (a video file cannot be recorded with force: the program says so; the live camera path is in test_force.py)
            self.do(step, "click", "No force today", live.chk_force_skip.click)
        elif step == "markers":
            self.do(step, "click", "Test the beeps", live.btn_cue_test.click)
            assert self.pump(lambda: live._beeps_tested, 8), "the beeps test never finished"
        elif step == "participant":
            for ch in "P01":
                self.do(step, "key", ch)
            self.do(step, "key", "Enter", lambda: live.participant.setText("P01"))
            self.do(step, "click", "choose the leg")
            self.do(step, "click", "Right leg", lambda: live.leg_combo.setCurrentIndex(live.leg_combo.findData("right")))
        elif step == "privacy":
            self.do(step, "click", "I looked: nothing identifies the participant", live.chk_priv_ok.click)
        elif step == "ruler":
            self.do(step, "click", "No ruler today", live.chk_ruler_skip.click)
        elif step == "dots":
            p0 = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
            for name in e.test.dots:
                if e.dot_status().get(name) == "off":
                    continue
                self.do(step, "click", f"click the {name} dot", lambda n=name: live.view.clicked.emit(float(p0[n][0]) + 2, float(p0[n][1]) + 2))
            self.pump(timeout=0.5)
        elif step == "lock":
            self.do(step, "key", "Space (lock)", live.btn_lock.click)
        elif step == "record":
            self.do(step, "key", "R (record)", live.btn_rec.click)
            assert self.pump(lambda: e.rec is not None, 5), f"Record did not start; the screen says: {live.instruction.text()!r} / {live.rec_label.text()!r}; dialogs: {self.dialogs}"
            self.pump(timeout=2.5)
            self.do(step, "key", "S (MVIC start)", live.btn_start_mark.click)
            self.pump(timeout=1.5)
            self.do(step, "key", "E (MVIC end)", live.btn_end_mark.click)
            assert self.pump(lambda: e.rec is None, 25), "the recording did not stop by itself after the MVIC end"
        else:
            pytest.fail(f"no action known for the step '{step}'")

    def follow_the_screen(self, until="record"):
        """Do what the screen asks, step by step, until the record step has been done. Fails when the screen does not say what to do."""
        for _ in range(len(STEPS) + 3):
            self.pump(lambda: self.next_step() is not None, 3.0)                   # (the chips appear a moment after the window opens)
            step = self.next_step()
            assert step is not None, f"no step is marked as the next one; the instruction says: {self.live.instruction.text()!r}"
            text = self.live.instruction.text()
            if PLAN[step] not in text.lower():                                       # a timed notice (such as the beeps test) may cover the instruction for a few seconds,
                covered = time.time()                                                # the chip still shows the step; the instruction must be back within the notice's time
                self.pump(lambda: PLAN[step] in self.live.instruction.text().lower(), NOTICE_SECONDS + 1.5)
                text = self.live.instruction.text()
                self.notice_seconds.append(time.time() - covered)
            self.said.append((step, text))
            assert text.strip() and PLAN[step] in text.lower(), f"at the step '{step}' the instruction line says {text!r}, which does not tell what to do"
            self.perform(step)
            if step == until:
                return
            started = time.time()
            assert self.pump(lambda: self.next_step() != step, 3.0), f"after the step '{step}' was done the screen still asks for it: {self.live.instruction.text()!r}"
            self.latency.append(time.time() - started)
        pytest.fail("the steps did not lead to a recording")

    def klm_seconds(self):
        steps = {s for s, _k, _w in self.actions}
        return sum(KLM_SECONDS[k] for _s, k, _w in self.actions) + KLM_SECONDS["think"] * len(steps)


def manifest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).iterdir()) if p.is_file() and not p.name.startswith(("session_log", "settings"))}


@pytest.fixture
def session(tmp_path, video, monkeypatch):
    app = QApplication.instance() or QApplication([])
    dialogs = []                                                                  # what the program said in a dialog (a refusal says why, here)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: dialogs.append(("warning", a[2])))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: dialogs.append(("information", a[2])))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (dialogs.append(("question", a[2])), QMessageBox.Yes)[1])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    win.resize(1500, 950)
    win.show()
    app.processEvents()
    op = Operator(win, app, video)
    op.dialogs = dialogs
    yield op, tmp_path
    if win.engine.rec is not None:
        win.live.btn_rec.click()
    if win.live.force_source is not None:
        win.live.btn_force.click()
    win.live.worker = None
    win.close()


def test_a_first_time_operator_can_record_a_trial_by_following_the_screen(session):
    op, tmp = session
    op.follow_the_screen()
    steps_done = [s for s, _ in op.said]
    assert steps_done == ["camera", "force", "markers", "participant", "privacy", "ruler", "dots", "lock", "record"]      # the order of a session, every step asked once
    assert all(op.said[i][0] != op.said[i + 1][0] for i in range(len(op.said) - 1))
    count = len(op.actions)
    print(f"\nactions to the first trial: {count} ({sum(1 for a in op.actions if a[1] == 'click')} clicks, {sum(1 for a in op.actions if a[1] == 'key')} keys); "
          f"keystroke-level estimate {op.klm_seconds():.0f} s of operator work (equipment waits not included)")
    assert count <= MAX_ACTIONS_TO_FIRST_TRIAL, f"{count} actions: " + "; ".join(f"{s}: {w}" for s, _k, w in op.actions)
    assert max(op.latency) < 2.0, op.latency                                                                  # the screen answers an action within two seconds
    # the screen goes back to asking for the next trial
    assert op.pump(lambda: "next trial" in op.live.instruction.text().lower(), 5), op.live.instruction.text()


def test_the_trial_it_saved_passes_the_independent_check_opens_in_review_and_stays_untouched(session):
    op, tmp = session
    op.follow_the_screen()
    metas = sorted(tmp.rglob("*_meta.json"))
    assert len(metas) == 1, sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*"))[:20]
    meta = metas[0]
    folder = meta.parent
    base = str(meta)[: -len("_meta.json")]
    assert all(Path(base + suffix).is_file() for suffix in ("_data.csv", "_raw.mp4", "_overlay.mp4")), sorted(p.name for p in folder.iterdir())
    res = check_session.check_trial(meta)
    assert not [x for x in res["issues"] if x[0] == check_session.FAIL], res["issues"]                       # an independent second opinion on the numbers
    before = manifest(folder)
    review = op.win.review
    op.do("review", "click", "open the Review tab")
    op.do("review", "click", "choose the trial", lambda: review.load(str(meta)))
    assert review.trial is not None and review.trial.base == meta.name[: -len("_meta.json")]
    op.do("review", "click", "Valid", review.verdict_bar.btn_valid.click)
    op.do("review", "click", "Export", review.export_session)
    exports = list((meta.parent.parent / "review" / meta.parent.name / "exports").glob("*session_summary*.csv"))
    assert exports, "the export was not written in the review folder: " + str(sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if "review" in p.parts))[:600]
    review.trial.close()
    assert manifest(folder) == before, "Review changed a recorded file"                                      # byte for byte
    assert list(folder.glob("*_verdict.json")) or list((meta.parent.parent / "review").rglob("verdict*.json")), "the verdict was not saved"
