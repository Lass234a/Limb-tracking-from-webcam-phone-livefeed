"""Ruler scale: a ruler with a dot at each end gives pixels per centimetre, tracked live; it never changes an angle."""
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PX_PER_CM = 6.0
RULER_CM = 20.0


def scene(a=(120.0, 670.0), length_px=PX_PER_CM * RULER_CM, **leg):
    p = leg_points(hip_flex=70, knee_flex=60, ankle_df=5, **leg)
    p = dict(p)
    p["ruler_a"] = np.array(a)
    p["ruler_b"] = np.array([a[0] + length_px, a[1]])
    return p


def engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.scale.length_cm = RULER_CM
    p = scene()
    eng.process(render(p, seed=0), 0.0)
    return eng, p


def mark_ruler(eng, p):
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2))
    assert eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    eng.process(render(p, seed=1), 0.05)


def test_scale_comes_from_the_two_marked_ruler_dots(tmp_path):
    eng, p = engine(tmp_path)
    assert not eng.scale.configured and eng.scale.px_per_cm is None
    mark_ruler(eng, p)
    assert eng.scale.configured
    assert eng.scale.px_per_cm == pytest.approx(PX_PER_CM, rel=0.005)          # 120 px for a 20 cm ruler
    assert eng.scale.live_px_per_cm() == pytest.approx(PX_PER_CM, rel=0.005)
    assert eng.scale.problems() == []
    eng.scale.length_cm = 15.0                                                  # a different typed length rescales
    assert eng.scale.px_per_cm == pytest.approx(PX_PER_CM * 20 / 15, rel=0.005)


def test_the_second_click_on_the_same_dot_is_refused(tmp_path):
    eng, p = engine(tmp_path)
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2))
    assert eng.mark_scale_dot(1, *(p["ruler_a"] + 3)) is False
    assert "same dot" in eng.mark_problem
    assert not eng.scale.configured
    assert eng.mark_scale_dot(1, *(p["ruler_b"] + 2))                           # the right dot still works afterwards


def test_a_short_ruler_a_lost_dot_and_a_moved_ruler_are_flagged(tmp_path):
    eng, p = engine(tmp_path)
    short = scene(length_px=40.0)
    eng2 = Engine(TESTS, recordings_dir=tmp_path)
    eng2.set_test("knee_extension")
    eng2.process(render(short, seed=0), 0.0)
    assert eng2.mark_scale_dot(0, *(short["ruler_a"] + 2)) and eng2.mark_scale_dot(1, *(short["ruler_b"] + 2))
    assert any("coarse" in t for t in eng2.scale.problems())

    mark_ruler(eng, p)
    for i in range(4):                                                          # ruler dot covered
        eng.process(render(p, seed=2 + i, hide=["ruler_b"]), 0.1 + i / 30)
    assert eng.scale.live_px_per_cm() is None
    assert any("lost" in t for t in eng.scale.problems())
    assert eng.scale.configured                                                 # the marked scale itself is kept

    eng3, p3 = engine(tmp_path)
    mark_ruler(eng3, p3)
    moved = dict(p3)
    moved["ruler_b"] = p3["ruler_b"] + np.array([-14.0, 0.0])                   # the far end slides in by 14 px (about 12 %)
    for i in range(6):
        eng3.process(render(moved, seed=3 + i), 0.1 + i / 30)
    assert any("changed" in t for t in eng3.scale.problems())


def test_the_ruler_does_not_change_any_leg_dot_or_angle(tmp_path):
    p_leg = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    out = {}
    for with_ruler in (False, True):
        eng = Engine(TESTS, recordings_dir=tmp_path / str(with_ruler))
        eng.set_test("knee_extension")
        eng.target = 60.0
        p = scene() if with_ruler else p_leg
        eng.process(render(p, seed=0), 0.0)
        for n in eng.test.dots:
            assert eng.mark_dot(n, *(p[n] + 2))
        if with_ruler:
            assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
        for i in range(20):
            res = eng.process(render(p, seed=i), 0.1 + i / 30)
        out[with_ruler] = ([(d.x, d.y) for d in res.dots.values()], [a.raw_value for a in res.angles])
    assert np.allclose(np.array(out[True][0]), np.array(out[False][0]), atol=0.2)
    assert np.allclose(out[True][1], out[False][1], atol=0.3)


def test_recording_stores_the_scale_in_meta_and_csv(tmp_path):
    eng, p = engine(tmp_path)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    mark_ruler(eng, p)
    paths = eng.start_trial("P01")
    for i in range(30):
        eng.process(render(p, seed=i, hide=["ruler_b"] if 10 <= i < 14 else []), 0.2 + i / 30)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())["scale"]
    assert meta["ruler_length_cm"] == RULER_CM
    assert meta["px_per_cm_at_marking"] == pytest.approx(PX_PER_CM, rel=0.005)
    assert meta["during_recording"]["frames"] == 30 and 20 <= meta["during_recording"]["frames_with_scale"] < 30
    assert meta["during_recording"]["px_per_cm_mean"] == pytest.approx(PX_PER_CM, rel=0.01)
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    vals = [r["scale_px_per_cm"] for r in rows]
    assert vals[0] != "" and float(vals[0]) == pytest.approx(PX_PER_CM, rel=0.01)
    assert "" in vals                                                           # blank while a ruler dot was covered


def test_no_ruler_means_a_blank_scale_column_and_no_scale_in_meta(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    p = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    paths = eng.start_trial("P01")
    eng.process(render(p, seed=1), 0.1)
    eng.stop_trial()
    assert json.loads(paths["meta.json"].read_text())["scale"] is None
    assert [r["scale_px_per_cm"] for r in csv.DictReader(open(paths["data.csv"], newline=""))] == [""]


def test_scale_cannot_be_changed_while_recording_and_is_cleared_by_the_colour_setting(tmp_path):
    eng, p = engine(tmp_path)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    mark_ruler(eng, p)
    eng.start_trial("P01")
    eng.clear_scale()
    assert eng.scale.configured                                                 # ignored while recording
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) is False
    eng.stop_trial()
    eng.set_dot_kind("white")
    assert not eng.scale.configured                                             # new colour setting: mark the ruler again


def test_overlay_draws_the_ruler():
    from limbtrack import overlay
    img = np.full((300, 400, 3), 90, np.uint8)
    base = overlay.draw(img, {}, [])
    drawn = overlay.draw(img, {}, [], scale={"points": [(50, 200, False), (170, 200, False)], "text": "ruler 20 cm", "warn": False})
    assert np.abs(drawn.astype(int) - base.astype(int)).sum() > 0


def test_the_window_flow_marks_the_ruler_with_two_clicks(tmp_path):
    app = QApplication.instance() or QApplication([])
    p = scene()
    video = tmp_path / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(120):
        w.write(render(p, seed=i))
    w.release()
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live

    def pump(cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    assert "No scale" in live.scale_label.text()
    live.start_source(str(video))
    assert pump(lambda: win.engine._last_gray is not None)
    live.ruler_len.setValue(RULER_CM)
    assert win.engine.scale.length_cm == RULER_CM
    live.btn_ruler.click()
    live.view.clicked.emit(*(float(v) + 2 for v in p["ruler_a"]))
    assert win.engine.scale.next_to_mark() == 1 and "other end" in live.instruction.text()
    live.view.clicked.emit(*(float(v) + 2 for v in p["ruler_b"]))
    assert win.engine.scale.configured and "Scale set" in live.instruction.text()
    assert pump(lambda: "px/cm" in live.scale_label.text() and "now" in live.scale_label.text())
    assert win.engine.next_dot_to_mark() is not None                              # leg dots are still to be marked: the ruler did not take them
    live.stop_camera()
    live.start_source(str(video))                                                 # a new source forgets the scale
    assert not win.engine.scale.configured
    live.stop_camera()
    win.close()


# ---- dot-size check: each dot's approximate width in cm, warned below the 2 cm minimum
def marked_with_radius(tmp_path, radius):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.scale.length_cm = RULER_CM
    p = scene()
    eng.process(render(p, seed=0, dot_radius=radius), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    assert eng.mark_scale_dot(0, *(p["ruler_a"] + 2)) and eng.mark_scale_dot(1, *(p["ruler_b"] + 2))
    eng.process(render(p, seed=1, dot_radius=radius), 0.05)
    return eng, p


def test_dot_sizes_in_cm_are_close_to_the_truth_and_fine_dots_pass(tmp_path):
    eng, _ = marked_with_radius(tmp_path, 7.0)                   # 14 px at 6 px/cm = 2.33 cm
    sizes = eng.dot_sizes_cm()
    assert set(sizes) == set(eng.active_dots)
    assert all(2.1 < v < 2.6 for v in sizes.values()), sizes
    assert eng.small_dots() == {}


def test_dots_under_the_minimum_are_listed(tmp_path):
    eng, _ = marked_with_radius(tmp_path, 4.5)                   # 9 px at 6 px/cm = 1.5 cm
    small = eng.small_dots()
    assert set(small) == set(eng.active_dots) and all(1.2 < v < 1.8 for v in small.values()), small


def test_no_scale_means_no_size_check(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    p = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    assert eng.dot_sizes_cm() == {} and eng.small_dots() == {}


def test_the_size_check_is_saved_in_meta_and_shown_in_the_window(tmp_path):
    eng, p = marked_with_radius(tmp_path / "e", 4.5)
    paths = eng.start_trial("P01")
    eng.process(render(p, seed=2, dot_radius=4.5), 0.2)
    eng.stop_trial()
    check = json.loads(paths["meta.json"].read_text())["dot_size_check"]
    assert check["min_dot_diameter_cm"] == 2.0 and len(check["below_minimum"]) == len(check["dots_cm_estimate"])

    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path / "w")
    win.engine.scale.length_cm = RULER_CM
    pt = scene()
    win.engine.process(render(pt, seed=0, dot_radius=4.5), 0.0)
    for n in win.engine.test.dots:
        win.engine.mark_dot(n, *(pt[n] + 2))
    win.engine.mark_scale_dot(0, *(pt["ruler_a"] + 2))
    win.engine.mark_scale_dot(1, *(pt["ruler_b"] + 2))
    win.live.update_scale_label()
    assert "TOO SMALL" in win.live.scale_label.text() and "minimum 2 cm" in win.live.scale_label.text()
    win.close()
