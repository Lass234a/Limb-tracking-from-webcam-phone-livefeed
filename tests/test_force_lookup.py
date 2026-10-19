"""Looking up force samples by time (ForceSource.samples_between) gives exactly what a full scan gives, and its cost no longer grows
with how long the force has been connected.

Found on 2026-10-19 in the P11 trials of 2026-10-15: every trial with the automatic MVIC markers ran at 4 to 11 pictures per second
(camera: 30). With the force connected for 10 minutes or more, each look-up walked through all 60,000 stored chunks (13 ms), and the
recording, the automatic markers and the force trace each look up several times a second. Replaying a P11 video through the whole
window: 17 to 19 pictures per second before, 29.5 after (the camera's own rate)."""
import time

import numpy as np
import pytest

from limbtrack.force import ForceSource


def full_scan(src, t0, t1):
    """The way it was done before: look at every chunk."""
    with src._lock:
        parts = [c for c in src._chunks if c[0][-1] >= t0 and c[0][0] < t1]
    if not parts:
        return np.empty(0), np.empty(0)
    t = np.concatenate([p[0] for p in parts])
    v = np.concatenate([p[1] for p in parts])
    m = (t >= t0) & (t < t1)
    return t[m], v[m]


def filled(seconds, rng=None, uneven=False):
    """A source that has been reading for `seconds` (10 ms chunks; with `uneven`, chunks of 1 to 40 samples and arrival jitter)."""
    src = ForceSource()
    now = time.perf_counter()
    t, k = now - seconds, 0
    while t < now:
        n = int(rng.integers(1, 41)) if uneven else 10
        t += n / 1000.0 + (rng.normal(0, 0.002) if uneven else 0.0)
        src._push(np.arange(k, k + n, dtype=float), t)
        k += n
    return src


def test_the_same_samples_as_a_full_scan_for_any_window():
    rng = np.random.default_rng(1)
    src = filled(30.0, rng, uneven=True)
    first, last = src._chunks[0][0][0], src._chunks[-1][0][-1]
    windows = [(first - 1, last + 1), (last - 0.1, last), (last - 4.5, last), (first, first + 0.05), (last + 1, last + 2),
               (last - 10, last - 9.99), (first - 5, first - 1)]
    windows += [tuple(sorted(rng.uniform(first - 1, last + 1, 2))) for _ in range(300)]
    for t0, t1 in windows:
        a_t, a_v = src.samples_between(t0, t1)
        b_t, b_v = full_scan(src, t0, t1)
        assert np.array_equal(a_t, b_t) and np.array_equal(a_v, b_v), (t0, t1)


def test_the_chunks_are_kept_in_time_order_which_the_search_relies_on():
    rng = np.random.default_rng(2)
    src = filled(20.0, rng, uneven=True)
    ends = [c[0][-1] for c in src._chunks]
    starts = [c[0][0] for c in src._chunks]
    assert all(b >= a for a, b in zip(ends, ends[1:]))
    assert all(s >= e for e, s in zip(ends, starts[1:]))


def test_a_recent_window_costs_the_same_after_ten_minutes_as_after_ten_seconds():
    """Compared on the same machine in the same test (no fixed time limit, so a busy computer cannot fail it)."""
    long = filled(600.0)                           # the 10 minutes the source keeps
    assert len(long._chunks) >= 59000
    now = long._chunks[-1][0][-1]

    def per_call(fn, n=50):
        t = time.perf_counter()
        for _ in range(n):
            fn(now - 0.1, now)
        return (time.perf_counter() - t) / n

    new, old = per_call(long.samples_between), per_call(lambda a, b: full_scan(long, a, b), n=5)
    assert new < old / 20, (new, old)


@pytest.mark.parametrize("seconds", [0.1, 4.5, 15.0])
def test_the_windows_the_program_asks_for_are_complete(seconds):
    """A picture's samples (0.1 s), the automatic markers' rest level (4.5 s) and the force trace (15 s) after 10 minutes."""
    src = filled(600.0)
    now = src._chunks[-1][0][-1]
    t, v = src.samples_between(now - seconds, now + 1.0)
    assert t.size == pytest.approx(seconds * 1000, abs=11) and np.all(np.diff(v) == 1.0)
