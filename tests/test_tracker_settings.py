"""Tracker settings are written to meta.json, and tracking output is pinned to a recorded reference.

The reference (golden_tracking.npz) was recorded from the tracker before its numbers were turned into named
settings. If a deliberate change to tracking makes this test fail, regenerate it with `python tests/test_tracker_settings.py`
and say why in the commit message: the numbers in people's recordings will differ.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from limbtrack import tracker  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden_tracking.npz"
TESTS = load_protocol(ROOT / "protocol.json")


def run(tmp):
    """150 frames: swinging knee, one dot hidden for 30 frames, another for 10."""
    eng = Engine(TESTS, recordings_dir=tmp)
    eng.set_test("knee_extension")
    eng.target = 40.0
    p = leg_points(hip_flex=70, knee_flex=40, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    out = []
    for i in range(150):
        pose = leg_points(hip_flex=70, knee_flex=40 + 25 * np.sin(2 * np.pi * i / 60), ankle_df=5)
        hide = ["epicondyle"] if 40 <= i < 70 else (["malleolus"] if 100 <= i < 110 else [])
        res = eng.process(render(pose, seed=i, hide=hide), 0.1 + i / 30)
        for n in sorted(res.dots):
            d = res.dots[n]
            out += [d.x, d.y, float(d.lost), d.quality]
        out += [a.raw_value if a.raw_value is not None else np.nan for a in res.angles]
    return np.array(out, dtype=np.float64)


def test_tracking_output_is_unchanged_from_the_recorded_reference(tmp_path):
    ref = np.load(GOLDEN)["synthetic"]
    now = run(tmp_path)
    assert now.shape == ref.shape
    assert np.allclose(now, ref, atol=1e-6, equal_nan=True)


def test_every_tracking_number_is_listed_in_settings():
    s = tracker.settings()
    assert set(s) == {"image_preparation", "tracking", "lost_dot"}
    assert s["tracking"]["min_contrast"] == tracker.MIN_CONTRAST == 8.0
    assert s["tracking"]["area_range"] == [0.3, 3.5]
    assert s["lost_dot"]["search_radius_diam_factor"] == 2.0 and s["lost_dot"]["resume_min_quality"] == 0.6
    assert s["image_preparation"]["kernel_diam_factor"] == 2.5
    json.dumps(s)                                                     # must be writable to meta.json


def test_meta_json_records_the_settings(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 40.0
    p = leg_points(hip_flex=70, knee_flex=40, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    paths = eng.start_trial("P01")
    eng.process(render(p, seed=1), 0.1)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["tracker_settings"] == tracker.settings()
    assert meta["smoothing"]["window_s"] == 0.2                       # filter settings were already recorded


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        np.savez_compressed(GOLDEN, synthetic=run(d))
    print("wrote", GOLDEN)
