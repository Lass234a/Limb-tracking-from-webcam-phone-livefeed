"""REP-05: privacy crop and masks change only the saved video, never the tracking or the CSV."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack import privacy  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)


def make(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    return eng


def bounds(eng):
    xs = [st.x for st in eng._last_states.values()]
    ys = [st.y for st in eng._last_states.values()]
    return min(xs), min(ys), max(xs), max(ys)


def record(eng, n=20):
    paths = eng.start_trial("P01")
    for i in range(n):
        eng.process(render(PTS, seed=i % 5), 0.1 + i / 30)
    eng.stop_trial()
    return paths


def frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            return out
        out.append(img)


# ---------------------------------------------------------------- helpers
def test_snap_rect_makes_even_clamped_rectangles():
    assert privacy.snap_rect(100, 50, 901, 601, 1280, 720) == (100, 50, 800, 550)
    assert privacy.snap_rect(900, 601, 100, 50, 1280, 720) == (100, 50, 800, 550)        # corners in any order
    assert privacy.snap_rect(-20, -20, 5000, 5000, 1280, 720) == (0, 0, 1280, 720)
    assert privacy.snap_rect(10, 10, 20, 20, 1280, 720) is None                           # a stray click


def test_apply_never_changes_its_input_and_blacks_out_masks():
    img = np.full((100, 200, 3), 200, np.uint8)
    out = privacy.apply(img, (20, 10, 100, 60), [(40, 20, 20, 20)])
    assert out.shape == (60, 100, 3)
    assert (out[10:30, 20:40] == 0).all() and out[0, 0, 0] == 200
    assert (img == 200).all()
    assert privacy.apply(img, None, []) is img


def test_pad_to_full_puts_the_crop_back_where_it_was():
    crop = (20, 10, 100, 60)
    small = np.full((60, 100, 3), 7, np.uint8)
    full = privacy.pad_to_full(small, crop, (200, 100))
    assert full.shape == (100, 200, 3) and (full[10:70, 20:120] == 7).all() and full[0, 0, 0] == 0


def test_participant_id_and_notes_checks():
    assert privacy.check_participant_id("P01") is None and privacy.check_participant_id("P123") is None
    for bad in ("", "p01", "Anna", "P1", "P0001", "P01 "):
        assert privacy.check_participant_id(bad)
    assert privacy.notes_problem("strap loosened, rest 2 min") is None
    assert privacy.notes_problem("call me at a@b.dk")
    assert privacy.notes_problem("010203-1234")


# ------------------------------------------------------------- recording
def test_cropped_recording_has_the_same_numbers_and_a_smaller_video(tmp_path):
    plain = record(make(tmp_path / "a"))
    eng = make(tmp_path / "b")
    x0, y0, x1, y1 = bounds(eng)
    ok, msg = eng.set_privacy_rect("crop", x0 - 60, y0 - 60, x1 + 61, y1 + 61)
    assert ok, msg
    cropped = record(eng)
    a = list(csv.DictReader(open(plain["data.csv"], newline="")))
    b = list(csv.DictReader(open(cropped["data.csv"], newline="")))
    assert len(a) == len(b)
    cols = [c for c in a[0] if c.endswith(("_x", "_y", "_deg"))]
    for ra, rb in zip(a, b):
        for c in cols:
            assert ra[c] == rb[c], c                            # identical numbers, full-picture pixel positions
    meta = json.loads(cropped["meta.json"].read_text())
    cx, cy, cw, ch = meta["privacy"]["crop_xywh"]
    assert cw % 2 == 0 and ch % 2 == 0 and meta["frame_size"] == [cw, ch]
    for p in ("raw.mp4", "overlay.mp4"):
        fr = frames(cropped[p])
        assert len(fr) == 20 and fr[0].shape[:2] == (ch, cw)
    assert meta["privacy"]["full_picture_size"] == [1280, 720]


def test_review_puts_the_cropped_video_back_at_full_size(tmp_path):
    plain = Trial(record(make(tmp_path / "plain"))["meta.json"])
    eng = make(tmp_path / "cropped")
    x0, y0, x1, y1 = bounds(eng)
    assert eng.set_privacy_rect("crop", x0 - 60, y0 - 60, x1 + 61, y1 + 61)[0]
    paths = record(eng)
    tr = Trial(paths["meta.json"])
    img = tr.frame(0)
    w, h = tr.full_size
    assert img.shape[:2] == (h, w)
    cx, cy, cw, ch = tr.crop
    assert img[:cy].max() == 0 and img[:, :cx].max() == 0                      # black outside the crop
    ref = plain.frame(0)[cy:cy + ch, cx:cx + cw].astype(float)           # the same part of the uncropped recording
    assert abs(img[cy:cy + ch, cx:cx + cw].astype(float) - ref).mean() < 6       # (video compression only)


def test_masks_are_black_in_the_saved_video_but_tracking_still_sees_the_picture(tmp_path):
    eng = make(tmp_path)
    x0, y0, x1, y1 = bounds(eng)
    mask_y = int(y1) + 80                                   # below the leg, nothing marked there
    assert eng.set_privacy_rect("mask", 0, mask_y, 400, mask_y + 100)[0]
    paths = record(eng)
    fr = frames(paths["raw.mp4"])
    assert fr[0].shape[:2] == (720, 1280)
    assert fr[0][mask_y + 2:mask_y + 98, 4:396].max() <= 12      # black (video compression leaves tiny values)
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["privacy"]["masks_xywh"] and meta["privacy"]["crop_xywh"] is None


def test_recording_refuses_to_start_with_a_dot_inside_a_mask_or_at_the_crop_edge(tmp_path):
    eng = make(tmp_path)
    x0, y0, x1, y1 = bounds(eng)
    assert eng.set_privacy_rect("mask", x0 - 40, y0 - 40, x0 + 40, y0 + 40)[0]
    with pytest.raises(RuntimeError, match="inside a black mask"):
        eng.start_trial("P01")
    assert eng.rec is None
    eng.clear_privacy()
    assert eng.set_privacy_rect("crop", x0 + 5, y0 - 80, x1 + 80, y1 + 80)[0]     # the left-most dot is 5 px from the edge
    with pytest.raises(RuntimeError, match="edge of the privacy crop"):
        eng.start_trial("P01")


def test_crop_cannot_be_changed_while_recording_and_mirror_change_is_caught(tmp_path):
    eng = make(tmp_path)
    x0, y0, x1, y1 = bounds(eng)
    assert eng.set_privacy_rect("crop", x0 - 60, y0 - 60, x1 + 61, y1 + 61)[0]
    eng.start_trial("P01")
    with pytest.raises(RuntimeError):
        eng.set_privacy_rect("mask", 0, 0, 100, 100)
    with pytest.raises(RuntimeError):
        eng.clear_privacy()
    eng.stop_trial()
    eng.mirror = True
    with pytest.raises(RuntimeError, match="mirror"):
        eng.start_trial("P01")


def test_warning_when_a_dot_drifts_to_the_crop_edge_while_recording(tmp_path):
    eng = make(tmp_path)
    x0, y0, x1, y1 = bounds(eng)
    assert eng.set_privacy_rect("crop", x0 - 40, y0 - 40, x1 + 40, y1 + 40)[0]
    eng.start_trial("P01")
    moved = {k: v + np.array([30.0, 0.0]) for k, v in PTS.items()}            # whole leg slides 30 px to the right
    heads = []
    for i in range(10):
        res = eng.process(render(moved, seed=i), 0.1 + i / 30)
        heads.append(res.header)
    eng.stop_trial()
    assert any("PRIVACY" in " ".join(h) for h in heads)
    meta = json.loads(next(Path(tmp_path).rglob("*_meta.json")).read_text())
    assert sum(meta["privacy_dot_warnings"]["frames_near_crop_edge_or_in_mask"].values()) > 0


def test_a_changed_picture_size_clears_the_crop(tmp_path):
    eng = make(tmp_path)
    x0, y0, x1, y1 = bounds(eng)
    assert eng.set_privacy_rect("crop", x0 - 60, y0 - 60, x1 + 61, y1 + 61)[0]
    eng.process(np.zeros((360, 640, 3), np.uint8), 1.0)
    assert eng.privacy_crop is None and eng.privacy_masks == []
