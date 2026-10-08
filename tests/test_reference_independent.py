"""DOC-02: the neighbour reference stored at Lock does not depend on the display-smoothing switch."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def locked_refs(tmp_path, smoothing):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.set_smoothing(smoothing)
    p = leg_points(hip_flex=90, knee_flex=70, ankle_df=5, pelvic_tilt=3)
    eng.process(render(p, seed=0, noise=6.0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    for i in range(12):
        eng.process(render(p, seed=i + 1, noise=6.0), 0.1 + i / 30)       # noisy frames: a single frame differs from an average
    assert eng.lock_position()
    return dict(eng._refs)


def test_the_saved_reference_is_the_same_with_smoothing_on_and_off(tmp_path):
    on, off = locked_refs(tmp_path / "on", True), locked_refs(tmp_path / "off", False)
    assert set(on) == set(off) == {"hip", "ankle", "pelvic_tilt"}
    for k in on:
        assert on[k] == pytest.approx(off[k], abs=1e-9), k
