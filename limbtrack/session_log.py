"""A plain record of what the operator did and when: one line per action in `session_log.csv` in the recordings folder.

Why: the trials of a session show how each recording went, but not where the time between recordings went (marking dots, zeroing,
re-positioning the participant ...). On 2026-10-19 the P11 session of 2026-10-15 could not answer "how long does it take to get from
one measurement to the next, and how many clicks does marking the dots take". With this file the next review can measure it: the time
from one Stop to the next Record, the clicks per dot, the clicks that found no dot.

The program never reads it back and nothing measured is in it (no picture, angle or force): only the time, the action, and names
(the participant CODE as in the file names, dots, positions). A line that cannot be written is reported, never silently lost.
The log never creates the recordings folder: until the folder exists (it is made by the first recording) the lines are kept and
written with the next line after that.
"""
import csv
import time
from pathlib import Path

FILE_NAME = "session_log.csv"
COLUMNS = ("time", "seconds_since_start", "event", "detail")


class SessionLog:
    def __init__(self, folder):
        self.folder = Path(folder)
        self._t0 = time.perf_counter()
        self._waiting = []           # lines kept until the recordings folder exists (the log never creates it: the operator may
                                     # still choose another folder, and an empty one must not be left behind)
        self.problem = None          # why the last line could not be written, in plain English (None when it was)

    @property
    def path(self):
        return self.folder / FILE_NAME

    def write(self, event, detail=""):
        """Append one line (or keep it until the folder exists). Returns False only when writing failed."""
        self._waiting.append([time.strftime("%Y-%m-%dT%H:%M:%S"), f"{time.perf_counter() - self._t0:.1f}", event, detail])
        if not self.folder.is_dir():
            return True
        try:
            new = not self.path.exists()
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(COLUMNS)
                w.writerows(self._waiting)
            self._waiting = []
            self.problem = None
            return True
        except OSError as e:
            self.problem = f"The session log ({self.path.name}) could not be written: {e}"
            return False
