"""Display smoothing of angles.

A plain moving average over the last 0.2 s. It is used ONLY for what the operator sees
(numbers, colours, live trace, overlay video). The angles saved in the CSV are never smoothed.
"""
from collections import deque

SMOOTHING_WINDOW_S = 0.2


class MovingAverage:
    """Mean of all values whose time lies within the last `window` seconds (including the newest).

    The window is in seconds, not frames, so it does not depend on the camera's frame rate.
    It is trailing (it only looks back), so a steadily changing angle is shown about window/2 late.
    Starts afresh if time does not move forward (frame stepping backwards).
    """

    def __init__(self, window=SMOOTHING_WINDOW_S):
        self.window = window
        self._items = deque()

    def reset(self):
        self._items.clear()

    def __call__(self, x, t):
        if self._items and t <= self._items[-1][0]:
            self._items.clear()
        self._items.append((t, x))
        while self._items and t - self._items[0][0] >= self.window:
            self._items.popleft()
        return sum(v for _, v in self._items) / len(self._items)
