"""The window of the session check (0.34.0, F7): a readable list of what is still missing for one participant (limbtrack/completeness.py), opened
from the Live tab (next to the session plan) and from Review's participant overview. It only reads; it never blocks Record, an export or anything
else, and it stays open beside the program (not modal) until closed.

`compute()` returns the result of completeness.check. In the Live tab it reads the trials (about 0.1-0.3 s each), so it runs on a helper thread
and the window says "Working out..." until it is done; in Review the rows are the overview's own and the call is quick (`threaded=False`).
"""
import html
import threading

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout

from . import completeness

COLOURS = {"todo": "#c2410c", "gap": "#b45309", "info": "#555555", "ok": "#15803d"}
MARK = {"todo": "TO DO", "gap": "MISSING", "info": "NOTE", "ok": "OK"}


def to_html(res):
    """The list of completeness.lines as small HTML (headings, then one paragraph per item with its level in colour)."""
    out = []
    for level, text in completeness.lines(res):
        t = html.escape(text)
        if level == "heading":
            out.append(f"<h4 style='margin:10px 0 2px 0'>{t}</h4>")
        else:
            out.append(f"<p style='margin:2px 0 2px 8px'><b style='color:{COLOURS.get(level, '#000')}'>{MARK.get(level, '')}</b> {t}</p>")
    return "".join(out)


class SessionCheckDialog(QDialog):
    ready = Signal(object)                    # the result (or an error text) from the helper thread

    def __init__(self, parent, title, compute, threaded=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(640, 560)
        self.compute, self.threaded = compute, threaded
        self.result = None                    # the latest result (a dict), for tests and the caller
        self.error = ""
        self._busy = False
        self.view = QTextBrowser()
        self.view.setObjectName("sessionCheckView")
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.btn_again = QPushButton("Check again")
        self.btn_again.clicked.connect(self.run)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addWidget(self.status, 1)
        row.addWidget(self.btn_again)
        row.addWidget(self.btn_close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(row)
        self.ready.connect(self._show)
        self.run()

    def run(self):
        if self._busy:
            return
        if not self.threaded:
            self._finish(self._call())
            return
        self._busy = True
        self.btn_again.setEnabled(False)
        self.view.setHtml("<p>Working out... reading the saved trials of this participant.</p>")

        def work():
            res = self._call()
            try:
                self.ready.emit(res)
            except RuntimeError:              # the window was closed meanwhile: nobody left to tell
                pass
        threading.Thread(target=work, name="session-check", daemon=True).start()

    def _call(self):
        try:
            return self.compute()
        except Exception as e:  # noqa: BLE001 - said in the window, never hidden
            return f"{type(e).__name__}: {e}"

    def _show(self, res):
        self._busy = False
        self.btn_again.setEnabled(True)
        self._finish(res)

    def _finish(self, res):
        if isinstance(res, str):
            self.result, self.error = None, res
            self.view.setHtml(f"<p><b>The session check could not be made:</b> {html.escape(res)}</p>")
            self.status.setText("Could not be made.")
            return
        self.result, self.error = res, ""
        self.view.setHtml(to_html(res))
        self.status.setText(completeness.summary_line(res))
