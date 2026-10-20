"""Optional strap and anchor dots: tracked and recorded like landmarks, off by default, never part of an angle."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from limbtrack import overlay  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402
from limbtrack.tracker import DotState  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def strap_scene(strap_shift=0.0):
    p = dict(leg_points(hip_flex=70, knee_flex=60, ankle_df=5))
    mal, epi = np.array(p["malleolus"]), np.array(p["epicondyle"])
    p["strap"] = mal + 0.6 * (epi - mal) + np.array([0.0, strap_shift])        # on the shank, well away from the other dots
    p["anchor"] = np.array([380.0, 520.0])
    return p


def engine(tmp_path, with_strap=True):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    if with_strap:
        eng.set_dot_enabled("strap", True)
        eng.set_dot_enabled("anchor", True)
    p = strap_scene()
    eng.process(render(p, seed=0), 0.0)
    return eng, p


def test_the_protocol_offers_the_two_optional_dots_in_every_sagittal_test():
    # (the frontal hip abduction/adduction tests have none: the pull geometry and the torque are sagittal, see protocol.json)
    assert all(t.optional_dots == [] for t in TESTS.values() if t.view == "frontal")
    for t in (t for t in TESTS.values() if t.view == "sagittal"):
        assert t.optional_dots == ["strap", "anchor"]
        assert t.all_dots == t.dots + ["strap", "anchor"]
        assert "strap" not in t.dots and "anchor" not in t.dots               # not part of any angle
        assert "strap" in t.dot_title("strap").lower() and "anchor" in t.dot_title("anchor").lower()


def test_optional_dots_start_switched_off_and_are_not_asked_for(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    assert {"strap", "anchor"} <= eng.disabled_dots
    assert "strap" not in eng.active_dots and "anchor" not in eng.active_dots
    p = strap_scene()
    eng.process(render(p, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    assert eng.next_dot_to_mark() is None and eng.trackers.is_ready()          # nothing more to mark
    assert eng.mark_dot("strap", *(p["strap"] + 2)) is False                    # switched off: not markable


def test_switching_them_on_asks_for_them_and_they_are_tracked_without_changing_any_angle(tmp_path):
    out = {}
    for with_strap in (False, True):
        eng, p = engine(tmp_path / str(with_strap), with_strap)
        for n in eng.test.dots:
            assert eng.mark_dot(n, *(p[n] + 2))
        if with_strap:
            assert eng.next_dot_to_mark() == "strap"
            assert eng.mark_dot("strap", *(p["strap"] + 2)) and eng.mark_dot("anchor", *(p["anchor"] + 2))
        assert eng.next_dot_to_mark() is None
        for i in range(25):
            res = eng.process(render(p, seed=i), 0.1 + i / 30)
        out[with_strap] = (res, eng)
    assert "strap" in out[True][0].dots and "strap" not in out[False][0].dots
    assert np.allclose(out[True][0].dots["strap"].x, strap_scene()["strap"][0], atol=0.7)
    a_with, a_without = ([a.raw_value for a in out[k][0].angles] for k in (True, False))
    assert np.allclose(a_with, a_without, atol=0.2)                              # the extra dots changed no angle
    assert [a.name for a in out[True][0].angles] == [a.name for a in out[False][0].angles]


def test_lock_and_record_work_and_are_not_blocked_by_a_lost_strap(tmp_path):
    eng, p = engine(tmp_path)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    for i in range(5):
        eng.process(render(p, seed=i), 0.1 + i / 30)
    assert eng.lock_position()
    paths = eng.start_trial("P01")
    for i in range(30):
        eng.process(render(p, seed=i, hide=["strap"] if 10 <= i < 20 else []), 0.3 + i / 30)
    eng.stop_trial()
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert rows[0]["strap_x"] != "" and rows[12]["strap_x"] == "" and rows[12]["strap_lost"] == "1"
    assert all(r["knee_deg"] != "" for r in rows)                                # the knee angle never noticed
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["optional_dots_used"] == ["strap", "anchor"] and meta["dots_disabled"] == []
    assert "strap" in meta["dots_used"] and "anchor" in meta["dots_used"]


def test_meta_lists_optional_dots_as_not_used_when_off(tmp_path):
    eng, p = engine(tmp_path, with_strap=False)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    paths = eng.start_trial("P01")
    eng.process(render(p, seed=1), 0.1)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["optional_dots_used"] == [] and meta["dots_disabled"] == [] and "strap" not in meta["dots_used"]
    assert "strap_x" not in open(paths["data.csv"]).readline()


def test_the_choice_survives_switching_test(tmp_path):
    eng, _ = engine(tmp_path)
    eng.set_test("hip_extension_k90")
    assert "strap" in eng.active_dots and "anchor" in eng.active_dots
    eng.set_dot_enabled("anchor", False)
    eng.set_test("knee_extension")
    assert "strap" in eng.active_dots and "anchor" not in eng.active_dots


def test_the_overlay_draws_the_rope_between_the_two_dots_only_when_both_are_found():
    img = np.full((300, 400, 3), 90, np.uint8)
    base = overlay.draw(img, {}, [])
    both = {"strap": DotState("strap", 50, 100, False), "anchor": DotState("anchor", 300, 220, False)}
    assert np.abs(overlay.draw(img, both, []).astype(int) - base.astype(int)).sum() > 0
    both["anchor"] = DotState("anchor", 300, 220, True)
    drawn = overlay.draw(img, both, [])
    mid = drawn[160, 175]
    assert tuple(mid) == tuple(base[160, 175])                                   # no rope line through the middle
