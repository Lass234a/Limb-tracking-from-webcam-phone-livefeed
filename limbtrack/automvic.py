"""Automatic MVIC start and end markers from the force.

The protocol asks for every contraction in a 1-5-1 tempo: ramp up for 1 s, hold the maximum for 5 s, relax over 1 s, 7 s in all.
So once the person starts the ramp the end can be placed 7 s later, and the start is the moment the force clearly leaves its rest
level. This module only decides WHEN the markers go in; it never touches the force data (the volts are saved exactly as measured)
and it never changes an angle or a dot position. The markers it asks for are the same "MVIC start" / "MVIC end" markers the
operator can press by hand (S / E), so Review, the checks and the exports read them the same way.

How the start is found (all on the force, 1000 samples per second):
  * the force is averaged over the last 20 ms (one 50 Hz cycle, so mains hum cancels exactly);
  * the REST LEVEL is the median of that signal from 3.0 s to 0.5 s before now, so a slow drift (a participant settling into the
    strap) is followed, and the ramp itself is never part of the reference (it has only just begun);
  * a start is called when the force differs from the rest level by at least the THRESHOLD, and has moved by at least 40 % of the
    threshold within the last 0.25 s (a slow pre-tension is not a ramp), and both stay true for 70 ms (a knock of up to about
    50 ms is not a ramp);
  * the threshold is raised to 6 times the noise of the rest level if the rest was unsteady;
  * the instant the force really began to rise is estimated by walking back to where it was last at the rest level; it is saved in
    meta.json, the marker itself goes in when the start is called (typically 0.1 to 0.2 s after the force began to rise).
The end marker is placed LENGTH seconds after the start marker (default 7 s).
Either direction of force counts (a load cell that reads negative works the same).
"""
import numpy as np

DETECT_SMOOTH_S = 0.02          # the force is averaged over this long (one 50 Hz cycle: the hum cancels exactly)
REFERENCE_FROM_S = 3.0          # rest level = median of the force from this long ago ...
REFERENCE_TO_S = 0.5            # ... to this long ago
MIN_REFERENCE_S = 0.5           # less rest signal than this: not ready yet
RATE_WINDOW_S = 0.25            # the force must have moved by RATE_FRACTION x threshold within this time
RATE_FRACTION = 0.4
SUSTAIN_S = 0.07                # both conditions must hold this long (a knock of up to about 50 ms is shorter)
NOISE_FACTOR = 6.0              # the threshold is at least this many times the noise of the rest level
ONSET_FRACTION = 0.2            # the onset is where the force was last within this fraction of the threshold of the rest level
KEEP_S = REFERENCE_FROM_S + 1.5
DEFAULT_THRESHOLD_N = 10.0
DEFAULT_THRESHOLD_V = 0.02      # used when there is no load cell slope (volts only)
DEFAULT_LENGTH_S = 7.0          # the protocol's 1-5-1 tempo
AUTO_STOP_AFTER_END_S = 2.0     # the window stops the recording this long after the MVIC end marker (0 = the operator stops it)

WAITING, TRIGGERED, RUNNING, DONE, MANUAL = "waiting", "triggered", "running", "done", "manual"


class AutoMvic:
    """One recording's worth of automatic marking. Feed it force samples; ask it for the markers.

    `values` are the force in any unit that is proportional to the load (newtons computed without the zero, or volts): only
    differences from the rest level are used. `threshold` is in the same unit.
    """

    def __init__(self, threshold, length_s=DEFAULT_LENGTH_S, rate=1000.0, unit="N"):
        self.threshold, self.length_s, self.rate, self.unit = float(threshold), float(length_s), float(rate), unit
        self.state = WAITING
        self._t = np.empty(0)
        self._v = np.empty(0)
        self._run = 0                 # consecutive samples for which the start conditions hold
        self._run_first_t = None
        self.trigger = None           # what the detection saw when it called the start (see _call_start)
        self.start_marker_t = None    # picture time at which the start marker was written
        self.end_marker_t = None
        self.end_due_t = None
        self.reason = ""              # why nothing was marked (not enough rest signal, ...)

    # ------------------------------------------------------------------ feeding
    def prime(self, times, values):
        """Hand over the force from just before arming (the rest level). Nothing is looked for in it: a ramp that began
        before Record was pressed is not this recording's MVIC."""
        self._t = np.asarray(times, dtype=np.float64).ravel()[-int(KEEP_S * self.rate):]
        self._v = np.asarray(values, dtype=np.float64).ravel()[-int(KEEP_S * self.rate):]

    def feed(self, times, values):
        """Add force samples (times increasing). While waiting for the ramp, look for it; returns True when the start was called."""
        times = np.asarray(times, dtype=np.float64).ravel()
        values = np.asarray(values, dtype=np.float64).ravel()
        if times.size == 0:
            return False
        first_new = self._t.size
        self._t = np.concatenate([self._t, times])
        self._v = np.concatenate([self._v, values])
        keep = int(KEEP_S * self.rate)
        if self._t.size > keep:
            cut = self._t.size - keep
            self._t, self._v = self._t[cut:], self._v[cut:]
            first_new = max(first_new - cut, 0)
        if self.state != WAITING:
            return False
        return self._scan(first_new)

    def _scan(self, first_new):
        t, v = self._t, self._v
        n = max(int(round(DETECT_SMOOTH_S * self.rate)), 1)
        if v.size < n + 1 or first_new >= v.size:
            return False
        c = np.cumsum(np.insert(v, 0, 0.0))
        s = np.full(v.size, np.nan)
        s[n - 1:] = (c[n:] - c[:-n]) / n                                  # trailing mean: s[i] covers the n samples up to i
        t_block = t[first_new]
        in_ref = (t >= t_block - REFERENCE_FROM_S) & (t < t_block - REFERENCE_TO_S) & ~np.isnan(s)
        if in_ref.sum() < MIN_REFERENCE_S * self.rate:
            self.reason = "collecting the rest level"
            return False
        self.reason = ""
        ref = float(np.median(s[in_ref]))
        noise = float(np.std(s[in_ref]))
        thr = max(self.threshold, NOISE_FACTOR * noise)
        lag = int(round(RATE_WINDOW_S * self.rate))
        need = max(int(round(SUSTAIN_S * self.rate)), 1)
        for i in range(max(first_new, n - 1 + lag), v.size):
            d = s[i] - ref
            sign = 1.0 if d >= 0 else -1.0
            ok = sign * d >= thr and sign * (s[i] - s[i - lag]) >= RATE_FRACTION * thr
            if ok:
                if self._run == 0:
                    self._run_first_t = float(t[i])
                self._run += 1
                if self._run >= need:
                    self._call_start(i, ref, noise, thr, sign)
                    return True
            else:
                self._run = 0
        return False

    def _call_start(self, i, ref, noise, thr, sign):
        t, v = self._t, self._v
        n = max(int(round(DETECT_SMOOTH_S * self.rate)), 1)
        c = np.cumsum(np.insert(v, 0, 0.0))
        s = np.full(v.size, np.nan)
        s[n - 1:] = (c[n:] - c[:-n]) / n
        j = int(np.searchsorted(t, self._run_first_t))
        floor = max(ONSET_FRACTION * thr, NOISE_FACTOR / 2 * noise)
        while j > n - 1 and sign * (s[j] - ref) > floor:
            j -= 1
        onset = float(t[j]) - (n - 1) / 2.0 / self.rate                   # the trailing mean runs about half its length late
        self.trigger = {"trigger_t": float(t[i]), "rise_began_t": self._run_first_t, "onset_t": onset, "rest_level": ref,
                        "rest_noise_sd": noise, "threshold_used": thr, "rise_at_trigger": float(sign * (s[i] - ref)),
                        "direction": "up" if sign > 0 else "down"}
        self.state = TRIGGERED

    # ------------------------------------------------------------------ what the engine does with it
    def start_marked(self, t):
        """The start marker went into the recording at picture time t: the end is due `length_s` later."""
        self.start_marker_t = float(t)
        self.end_due_t = float(t) + self.length_s
        self.state = RUNNING

    def end_is_due(self, t):
        return self.state == RUNNING and t >= self.end_due_t

    def end_marked(self, t):
        self.end_marker_t = float(t)
        self.state = DONE

    def manual(self):
        """The operator pressed a marker by hand: from here on the markers are theirs."""
        if self.state in (WAITING, TRIGGERED, RUNNING):
            self.state = MANUAL

    # ------------------------------------------------------------------ reporting
    def status_text(self, t):
        """One plain line for the picture, the window and the overlay video."""
        thr = f"{self.threshold:g} {self.unit}"
        if self.state in (WAITING, TRIGGERED):
            return (f"AUTO MVIC: armed, waiting for the ramp (force up by {thr})" if not self.reason
                    else f"AUTO MVIC: armed, {self.reason}")
        if self.state == RUNNING:
            return f"AUTO MVIC: {max(t - self.start_marker_t, 0.0):.1f} / {self.length_s:g} s"
        if self.state == DONE:
            return f"AUTO MVIC: start and end marked ({self.length_s:g} s)"
        return "AUTO MVIC: off (a marker was pressed by hand)"

    def result(self, t0=None):
        """What happened, for meta.json; times are seconds after `t0` (the first picture) when t0 is given."""
        def rel(x):
            return None if x is None else round(x - (t0 or 0.0), 4)
        tr = self.trigger or {}
        return {
            "state": self.state,
            "threshold": round(self.threshold, 4), "threshold_used": None if not tr else round(tr["threshold_used"], 4), "unit": self.unit,
            "length_s": self.length_s,
            "start_marker_t_s": rel(self.start_marker_t), "end_marker_t_s": rel(self.end_marker_t),
            "force_rise_began_t_s": rel(tr.get("rise_began_t")), "force_onset_estimate_t_s": rel(tr.get("onset_t")),
            "start_marker_after_onset_s": None if not tr or self.start_marker_t is None else round(self.start_marker_t - tr["onset_t"], 3),
            "rest_level": None if not tr else round(tr["rest_level"], 4), "rest_noise_sd": None if not tr else round(tr["rest_noise_sd"], 5),
            "rise_at_trigger": None if not tr else round(tr["rise_at_trigger"], 4), "direction": tr.get("direction"),
            "note": "markers are written like the S / E keys (labels 'MVIC start' and 'MVIC end'); the end is `length_s` after the start "
                    "marker; the start marker is typically 0.1 to 0.2 s after the force began to rise (see start_marker_after_onset_s)",
        }
