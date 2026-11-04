"""The window of the rig check (0.35.0, F9): the list of limbtrack/rig_check.py, opened from the Live tab (next to the Session check). Run it with the
rig empty before the participant arrives.

The equipment is read on a helper thread (the picture is never held up; "Working out..." until done), and the operator's own parts are in the window:
*Play the beeps* plays the real cues through limbtrack/cues.py and the tick says they were heard, and the lamp tick. Changing a tick re-judges the list
at once (nothing is read again). *Save this check* writes one line to the session log and keeps the block for the `meta.json` of the following
trials. It only reads, never moves a zero or a setting, never blocks Record or anything else, and stays open beside the program (not modal).
"""
import html
import threading
from datetime import datetime

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout

from . import rig_check

COLOURS = {"green": "#15803d", "amber": "#b45309", "grey": "#555555"}
MARK = {"green": "READY", "amber": "LOOK AT", "grey": "NOT CHECKED"}


def to_html(result):
    """The list as small HTML: one paragraph per item, its state in words and colour (never colour alone), then the reason."""
    out = [f"<p><b>{html.escape(rig_check.summary_line(result))}</b> Run this with the rig empty, before the participant arrives.</p>"]
    for _key, title, state, reason in rig_check.rows(result):
        out.append(f"<p style='margin:3px 0 3px 8px'><b style='color:{COLOURS[state]}'>{MARK[state]}</b> <b>{html.escape(title)}</b><br>"
                   f"<span style='margin-left:6px'>{html.escape(reason)}</span></p>")
    return "".join(out)


class RigCheckDialog(QDialog):
    ready = Signal(object)                    # the facts (or an error text) from the helper thread

    def __init__(self, parent, title, gather, play_beeps, beep_result, save, context=None, threaded=True):
        """gather(context) -> the facts of rig_check.gather (run on a helper thread); context() -> what the window's widgets say (read on the
        window's thread before the helper starts); play_beeps() -> seconds the beeps take, or None when they cannot be played now;
        beep_result() -> (names the sound system took, error or None); save(result) -> None or a sentence about a problem (the caller keeps the block)."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(640, 520)
        self.gather, self.play_beeps, self.beep_result, self.save_fn, self.threaded = gather, play_beeps, beep_result, save, threaded
        self.context = context or (lambda: {})
        self.facts = None
        self.result = None                    # the list as judged now (a dict), for tests and the caller
        self.saved = None                     # the block that was saved last (a dict), or None
        self.error = ""
        self.manual = {"played": False, "names": [], "error": None, "heard": False, "lamp": False}
        self._busy = False
        self._playing = False                 # the beeps are sounding (the result is read when they are over)
        self.view = QTextBrowser()
        self.view.setObjectName("rigCheckView")
        self.btn_beeps = QPushButton("Play the beeps")
        self.btn_beeps.setToolTip("Plays three of the real cues (record, ramp, end) through the computer's sound system, also when the beeps are "
                                  "switched off for recordings.")
        self.btn_beeps.clicked.connect(self.start_beeps)
        self.chk_heard = QCheckBox("I heard them")
        self.chk_heard.setEnabled(False)
        self.chk_heard.toggled.connect(self._ticked)
        self.chk_lamp = QCheckBox("The lamp is on and the light is even")
        self.chk_lamp.toggled.connect(self._ticked)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        self.btn_again = QPushButton("Check again")
        self.btn_again.clicked.connect(self.run)
        self.btn_save = QPushButton("Save this check")
        self.btn_save.setToolTip("One line in the session log with the time and the result of every item, and the result goes into the meta.json of "
                                 "the following trials.")
        self.btn_save.clicked.connect(self.save_now)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        ticks = QHBoxLayout()
        ticks.addWidget(self.btn_beeps)
        ticks.addWidget(self.chk_heard)
        ticks.addSpacing(12)
        ticks.addWidget(self.chk_lamp, 1)
        row = QHBoxLayout()
        row.addWidget(self.status, 1)
        row.addWidget(self.btn_again)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(ticks)
        lay.addLayout(row)
        self.ready.connect(self._facts_ready)
        self.run()

    # ------------------------------------------------------------------------------------------ reading the equipment
    def run(self):
        if self._busy:
            return
        ctx = self.context()
        if not self.threaded:
            self._facts_ready(self._call(ctx))
            return
        self._busy = True
        self.btn_again.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.view.setHtml("<p>Working out... reading the camera, the force and the check weight.</p>")

        def work():
            res = self._call(ctx)
            try:
                self.ready.emit(res)
            except RuntimeError:              # the window was closed meanwhile: nobody left to tell
                pass
        threading.Thread(target=work, name="rig-check", daemon=True).start()

    def _call(self, ctx):
        try:
            return self.gather(ctx)
        except Exception as e:  # noqa: BLE001 - said in the window, never hidden
            return f"{type(e).__name__}: {e}"

    def _facts_ready(self, res):
        self._busy = False
        self.btn_again.setEnabled(True)
        self.btn_save.setEnabled(True)
        if isinstance(res, str):
            self.facts, self.result, self.error = None, None, res
            self.view.setHtml(f"<p><b>The rig check could not be made:</b> {html.escape(res)}</p>")
            self.status.setText("Could not be made.")
            return
        self.facts, self.error = res, ""
        self._render()

    # ------------------------------------------------------------------------------------------ the operator's parts
    def start_beeps(self):
        seconds = self.play_beeps()
        if seconds is None:
            self.status.setText("The beeps cannot be played now (a recording is running).")
            return
        self.manual.update(played=True, names=[], error=None, heard=False)
        self.chk_heard.blockSignals(True)
        self.chk_heard.setChecked(False)
        self.chk_heard.blockSignals(False)
        self.chk_heard.setEnabled(False)
        self.btn_beeps.setEnabled(False)
        self._playing = True
        self.status.setText("Playing the beeps...")
        QTimer.singleShot(int(seconds * 1000), self._beeps_done)
        self._render()

    def _beeps_done(self):
        names, err = self.beep_result()
        self._playing = False
        self.manual.update(names=list(names), error=err)
        self.btn_beeps.setEnabled(True)
        self.chk_heard.setEnabled(True)
        self._render()

    def _ticked(self, _=None):
        self.manual.update(heard=self.chk_heard.isChecked(), lamp=self.chk_lamp.isChecked())
        self._render()

    def _render(self):
        if self.facts is None:
            return
        self.result = rig_check.evaluate(self.facts, self.manual)
        self.view.setHtml(to_html(self.result))
        saved = (" Not saved yet." if not self.saved else f" Saved at {self.saved['saved_at'][11:16]}"
                 + ("." if self.saved["items"] == self.result["items"] else "; the list has changed since: save again."))
        if not self._playing:
            self.status.setText(rig_check.summary_line(self.result) + saved)

    def save_now(self):
        """Judge the list as it stands, keep it (the caller writes the log line and holds the block for the trials)."""
        if self.facts is None:
            return
        self._render()
        block = dict(self.result, saved_at=datetime.now().isoformat(timespec="seconds"))     # (the list's own time is when the equipment was read)
        problem = self.save_fn(block)
        if problem:
            self.status.setText(problem)
            return
        self.saved = block
        self._render()
