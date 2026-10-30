"""The rest between contractions (0.27.0), timed by the program instead of a stopwatch (SOP v0.18, 1.7).

SOP v0.18 (KVT3-PROJ-01, 2026-10-21): 60 s between the trials of one position (10.5, 10.6), 120 s before the next position (10.8),
120 s between the belt squat trials (11.7), and 30 s after the submaximal contraction before the first MVIC trial (9.11).

A rest starts by itself when a recording stops, and by hand (the *Submax rest* button, T) after the submaximal contraction, which is not
recorded. It is counted from the moment the recording stopped (with the automatic stop that is 2 s after the MVIC end marker). How long
it must be follows from where the operator is now: 60 s while the next trial is at the same position; 120 s once the position is finished
(the 10 % check needs no further trial there), when another position is chosen, or in the belt squat. A rest that is still running when
Record is pressed is a reminder, never a refusal, and the rest taken is saved with the trial (`meta.json` `rest_before`).

In the pilot sessions (2026-10-28/29) 33 of the 89 rests between two trials of one position were shorter than 60 s (the shortest 5 s),
and 8 of the 37 changes of position took less than 120 s.
"""
import time

REST_SAME_POSITION_S = 60.0       # SOP 10.5 / 10.6: between the trials of one position
REST_NEXT_POSITION_S = 120.0      # SOP 10.8: before the next position (the set-up is changed in it)
REST_AFTER_SUBMAX_S = 30.0        # SOP 9.11: after the submaximal contraction, before the first MVIC trial
LONG_REST_TESTS = {"belt_squat"}  # SOP 11.7: 120 s between all belt squat trials


def clock_text(seconds):
    """0:42, 2:00, 12:05."""
    s = max(int(seconds), 0)
    return f"{s // 60}:{s % 60:02d}"


class RestTimer:
    """One rest at a time. `position` is any value that tells two positions apart (the window uses (participant, test, target));
    `clock` is in seconds (time.monotonic: not changed by the PC's clock)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.started = None          # clock time the rest began, None when there is no rest
        self.kind = None             # "trial" (a recording stopped) or "submax" (the button)
        self.position = None         # the position of the trial the rest follows
        self.test_key = None
        self._beeped_at = None       # the length the rest had when its 'over' beep was played (a longer one, after a change of position, beeps again)

    def start(self, kind, position=None, test_key=None, at=None):
        self.started = self._clock() if at is None else at
        self.kind, self.position, self.test_key, self._beeped_at = kind, position, test_key, None

    def stop(self):
        self.started = None

    @property
    def running(self):
        return self.started is not None

    def elapsed(self):
        return None if self.started is None else self._clock() - self.started

    def required(self, position=None, position_done=False):
        """(seconds, why) the rest should last for what comes next. `position`: the one chosen now; `position_done`: the 10 % check
        needs no further trial at the position the rest follows."""
        if self.kind == "submax":
            return REST_AFTER_SUBMAX_S, "after the submaximal contraction"
        if self.test_key in LONG_REST_TESTS:
            return REST_NEXT_POSITION_S, "belt squat"
        if position_done or (position is not None and position != self.position):
            return REST_NEXT_POSITION_S, "before the next position"
        return REST_SAME_POSITION_S, "before the next trial at this position"

    def status(self, position=None, position_done=False):
        """None without a rest, else a dict: elapsed, required, why, done, short (still running), text (one line for the window)."""
        if self.started is None:
            return None
        e = self.elapsed()
        need, why = self.required(position, position_done)
        done = e >= need
        text = (f"Rest {clock_text(e)}: done (the SOP's {clock_text(need)} {why})" if done
                else f"REST {clock_text(e)} of {clock_text(need)} ({why})")
        return {"elapsed": e, "required": need, "why": why, "done": done, "short": not done, "text": text}

    def beep_due(self, position=None, position_done=False):
        """True once, the first time the rest is found to be over (for the beep); once more if the rest became longer afterwards (60 s were
        over, then another position was chosen: 120 s) and that is over too."""
        st = self.status(position, position_done)
        if st is None or not st["done"] or (self._beeped_at is not None and st["required"] <= self._beeped_at):
            return False
        self._beeped_at = st["required"]
        return True

    def meta(self, position=None, position_done=False):
        """What is saved with the next trial (`meta.json` `rest_before`), or None when no rest was timed (the first trial of a run)."""
        st = self.status(position, position_done)
        if st is None:
            return None
        return {"seconds": round(st["elapsed"], 1), "required_s": st["required"], "after": "submaximal contraction" if self.kind == "submax" else "trial",
                "why": st["why"], "short": st["short"],
                "meaning": "seconds from the end of the previous recording (or the Submax rest button) to this Record press, in this run of "
                           "the program; SOP v0.18: 60 s between the trials of a position, 120 s before the next position and between belt "
                           "squat trials, 30 s after the submaximal contraction (0.27.0)"}

    def reminder(self, position=None, position_done=False):
        """The soft reminder when Record is pressed during a rest, or ''."""
        st = self.status(position, position_done)
        if st is None or st["done"]:
            return ""
        return (f"Rest: only {clock_text(st['elapsed'])} {'since the submaximal contraction' if self.kind == 'submax' else 'since the last trial'} "
                f"(the SOP asks for {clock_text(st['required'])} {st['why']}).")
