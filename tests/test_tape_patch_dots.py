"""Tape dots: a hand-drawn black disc on a square of white sports tape (see MARKERS.md).

The planned stickers are not sticker dots but a white tape square with a black disc drawn on it. These tests pin what
MARKERS.md promises: the tracker follows the disc, not the tape, as well as it follows a bare dot; a disc seen side-on (ASIS and
PSIS from the side) is still followed; the dot-size reading is close to the true width; and the dot colour *Black* keeps the
tracker on the ink even when the circle is only an outline on tape it is not centred on.
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.scale import MIN_DOT_DIAMETER_CM, estimate_diameter_cm  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402
from limbtrack.tracker import DotTracker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PXCM = 6.5          # pixels per cm: about what a phone 2 to 3 m away gives
W, H = 640, 360


def flat_scene(cx, cy, patch_cm=4.0, disc_cm=2.0, outline_cm=None, squash=1.0, offset=(0.0, 0.0), blur=0.9, noise=3.0, seed=0):
    """Skin-coloured picture with a white tape square centred on (cx, cy) and a black disc (or, with outline_cm, a ring) on it.
    Coordinates are pixel-centre coordinates, like the tracker's. The disc is `offset` away from the middle of the tape."""
    big = 8

    def cover(draw):
        m = np.zeros((H * big, W * big), np.uint8)
        draw(m)
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    def pt(x, y):
        return int(round((x + 0.5) * big)), int(round((y + 0.5) * big))

    half = patch_cm * PXCM / 2
    tape = cover(lambda m: cv2.rectangle(m, pt(cx - half, cy - half), pt(cx + half, cy + half), 255, -1))
    r = disc_cm * PXCM / 2
    axes = (int(round(r * squash * big)), int(round(r * big)))
    thick = -1 if outline_cm is None else int(round(outline_cm * PXCM * big))
    ink = cover(lambda m: cv2.ellipse(m, pt(cx + offset[0], cy + offset[1]), axes, 0, 0, 360, 255, thick))
    img = np.full((H, W), 150.0, np.float32)
    img = img * (1 - tape) + 232.0 * tape
    img = img * (1 - ink) + 35.0 * ink
    img = cv2.GaussianBlur(img, (0, 0), blur)
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def wander(i):
    return 300 + 40 * np.sin(i / 19) + (0.37 * i) % 5, 170 + 25 * np.sin(i / 9.5)


def follow(kind, frames=60, **scene_kw):
    """Mark at frame 0, follow the moving dot, return (kind chosen, lost frames, mean and max centre error in px)."""
    offset = scene_kw.get("offset", (0.0, 0.0))
    t = DotTracker("d", kind)
    x0, y0 = wander(0)
    assert t.init_at(flat_scene(x0, y0, **scene_kw), x0 + 1.5, y0 - 1.0)
    errs, lost = [], 0
    for i in range(1, frames):
        x, y = wander(i)
        s = t.update(flat_scene(x, y, seed=i, **scene_kw))
        if s.lost:
            lost += 1
            continue
        errs.append(np.hypot(s.x - (x + offset[0]), s.y - (y + offset[1])))
    return t.kind, lost, float(np.mean(errs)), float(np.max(errs))


@pytest.mark.parametrize("patch_cm", [3.0, 4.0, 6.0])
def test_auto_marks_the_black_disc_not_the_white_tape(patch_cm):
    cx, cy = 320.0, 180.0
    t = DotTracker("d", "auto")
    assert t.init_at(flat_scene(cx, cy, patch_cm=patch_cm), cx + 1.5, cy - 1.0)
    assert t.kind == "black"
    assert abs(t.marked_diam / PXCM - 2.0) < 0.15, t.marked_diam            # the disc's width, not the tape's
    assert np.hypot(t.x - cx, t.y - cy) < 0.3


@pytest.mark.parametrize("squash", [1.0, 0.7, 0.5])
def test_a_taped_disc_is_followed_without_loss_even_when_seen_side_on(squash):
    kind, lost, mean_err, max_err = follow("auto", patch_cm=4.0, squash=squash)
    assert kind == "black" and lost == 0
    assert max_err < 0.35, (mean_err, max_err)


def test_a_taped_disc_is_followed_as_well_as_a_bare_one():
    bare = follow("black", patch_cm=0.0)          # no tape at all (zero-size square)
    taped = follow("black", patch_cm=4.0)
    assert taped[1] == bare[1] == 0
    assert taped[3] < bare[3] + 0.1


def test_the_black_setting_stays_on_the_ink_of_an_outline_circle_that_is_off_the_middle_of_the_tape():
    # The worst case for the tape: a thin outline circle (not filled), a slightly soft picture, circle 8 mm off the tape's middle.
    # With Dot colour = Auto this made the tracker follow the white tape (about 11 mm off, no warning); Black does not.
    kind, lost, mean_err, max_err = follow("black", patch_cm=5.0, outline_cm=0.25, blur=1.5, offset=(5.2, 2.6))
    assert kind == "black" and lost == 0 and max_err < 0.5, (mean_err, max_err)


@pytest.mark.parametrize("disc_cm", [2.0, 2.2, 2.5])
def test_the_size_reading_of_a_taped_disc_is_close_to_its_true_width(disc_cm):
    cx, cy = 320.0, 180.0
    t = DotTracker("d", "black")
    assert t.init_at(flat_scene(cx, cy, disc_cm=disc_cm), cx + 1.0, cy)
    est = estimate_diameter_cm(t.marked_diam, PXCM)
    assert abs(est - disc_cm) < 0.12, (est, disc_cm)


def test_a_disc_drawn_a_little_over_the_minimum_passes_the_two_centimetre_check():
    cx, cy = 320.0, 180.0
    for blur in (0.9, 1.5, 2.2):                 # sharp to rather soft picture
        t = DotTracker("d", "black")
        assert t.init_at(flat_scene(cx, cy, disc_cm=2.2, blur=blur), cx + 1.0, cy)
        assert estimate_diameter_cm(t.marked_diam, PXCM) >= MIN_DOT_DIAMETER_CM, blur


@pytest.mark.parametrize("hip_flex", [-20.0, 10.0, 60.0])
def test_angles_from_taped_dots_equal_angles_from_bare_dots(tmp_path, hip_flex):
    """The full leg: eight black dots, each on a white tape square, against the same eight dots bare. Dot colour = Black."""
    def angles(patch):
        eng = Engine(TESTS, recordings_dir=tmp_path)
        eng.set_test("hip_extension_k0")
        eng.set_dot_kind("black")
        eng.facing_right = True
        p = leg_points(hip_flex=hip_flex, knee_flex=0.0, ankle_df=5.0, pelvic_tilt=8.0)
        eng.process(render(p, kind="black", patch=patch, seed=0), 0.0)
        for n in eng.active_dots:
            assert eng.mark_dot(n, *(p[n] + 2)), n
        res = eng.process(render(p, kind="black", patch=patch, seed=1), 0.1)
        assert not any(d.lost for d in res.dots.values())
        return {a.name: a.raw_value for a in res.angles}, {n: (d.x, d.y) for n, d in res.dots.items()}

    bare_angles, bare_xy = angles(None)
    taped_angles, taped_xy = angles(21.0)                 # a 42 px square round 14 px dots (about 3 dot-widths)
    assert bare_angles["hip"] == pytest.approx(hip_flex, abs=0.1)
    assert taped_angles["hip"] == pytest.approx(hip_flex, abs=0.1)         # both within 0.1 deg of the true angle
    for name, value in bare_angles.items():
        assert taped_angles[name] == pytest.approx(value, abs=0.1), name   # measured differences: 0.07 deg at most
    for n, (x, y) in bare_xy.items():
        assert np.hypot(taped_xy[n][0] - x, taped_xy[n][1] - y) < 0.3, n
