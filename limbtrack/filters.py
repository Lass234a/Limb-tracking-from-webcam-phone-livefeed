"""Angle smoothing.

A One-Euro filter: heavy smoothing while the value is nearly still (so tracking jitter
disappears), light smoothing while it moves fast (so real movement is not delayed).
"""
import math

# name -> filter settings. min_cutoff (Hz): smoothing when still (lower = calmer).
# beta: how quickly smoothing relaxes when the angle moves (higher = less lag).
LEVELS = {
    "off": None,
    "light": {"min_cutoff": 1.5, "beta": 0.5},
    "medium": {"min_cutoff": 0.8, "beta": 0.3},
}


class OneEuroFilter:
    def __init__(self, min_cutoff=2.0, beta=0.08, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._t = None
        self._x = 0.0
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        """Filter value `x` measured at time `t` (seconds). Starts afresh after a gap or if time goes backwards."""
        if self._t is None or t <= self._t or t - self._t > 0.5:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x


def make_filter(level):
    """A filter for 'off' (returns None), 'light' or 'medium'."""
    cfg = LEVELS[level]
    return None if cfg is None else OneEuroFilter(**cfg)
