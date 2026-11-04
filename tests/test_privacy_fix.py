"""Fixing the privacy crop and masks in Review (audit of 2026-11-04, limbtrack/privacy_fix.py): protected copies of a recorded trial are made in the review
layer, the trial's own files are never written, and no measured number changes."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack import privacy, privacy_fix as pf, review_layer  # noqa: E402
from limbtrack.review import Trial  # noqa: E402

from tests.test_privacy import bounds, frames, make, record  # noqa: E402


def tree_hash(folder):
    """Name, size and SHA-256 of every file in a folder: what must not change."""
    return {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(Path(folder).iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    """A trial recorded with neither crop nor mask (what the recorder did when it forgot)."""
    paths = record(make(tmp_path_factory.mktemp("plain")), n=12)
    meta = paths["meta.json"]
    yield meta, json.loads(meta.read_text())


@pytest.fixture(scope="module")
def cropped(tmp_path_factory):
    eng = make(tmp_path_factory.mktemp("cropped"))
    x0, y0, x1, y1 = bounds(eng)
    assert eng.set_privacy_rect("crop", x0 - 60, y0 - 60, x1 + 61, y1 + 61)[0]
    meta = record(eng, n=12)["meta.json"]
    yield meta, json.loads(meta.read_text())


BLACK = 40          # (the video codec leaves faint ringing round a black rectangle, up to about 20 of 255; the bright dot under it is above 100)


def first_dot_at(meta):
    """Where the first dot is in the first picture, in full-picture pixels (a bright disc on the synthetic leg)."""
    tr = Trial(meta)
    try:
        d = tr.dot_names[0]
        return int(float(tr.rows[0][f"{d}_x"])), int(float(tr.rows[0][f"{d}_y"]))
    finally:
        tr.close()


def a_crop_and_a_mask(meta, meta_dict):
    """A crop round the leg and a mask over the first dot (so the mask hides something bright), both in full-picture pixels."""
    x, y = first_dot_at(meta)
    crop = (max(0, x - 300) // 2 * 2, max(0, y - 200) // 2 * 2, 600, 400)
    return crop, [(x - 20, y - 20, 40, 40)]


# ---------------------------------------------------------------------------------------------- the state of a trial
def test_a_trial_with_neither_crop_nor_mask_is_unprotected_until_someone_says_otherwise(plain):
    meta, d = plain
    s = pf.status(meta, d)
    assert s["state"] == pf.UNPROTECTED and "No crop and no mask" in s["text"] and "Privacy" in s["text"]
    said = json.loads(json.dumps(d))
    said["privacy"]["checked"] = {"how": pf.LOOKED_NOTHING}
    assert pf.status(meta, said)["state"] == pf.NOTHING_TO_HIDE                       # the operator looked: nothing to hide
    old = {k: v for k, v in d.items() if k != "privacy"}
    assert pf.status(meta, old)["state"] == pf.UNKNOWN                                # recorded before the program kept a privacy record


def test_a_trial_recorded_with_a_crop_is_recorded_as_protected(cropped):
    meta, d = cropped
    assert pf.status(meta, d)["state"] == pf.RECORDED and pf.recorded(d)[0] == tuple(d["privacy"]["crop_xywh"])


# ---------------------------------------------------------------------------------------------- the copies
def test_protected_copies_have_the_crop_and_masks_and_the_originals_are_untouched(tmp_path, plain):
    src_meta, d = plain
    folder = tmp_path / "recordings" / "P01"
    folder.mkdir(parents=True)
    for p in src_meta.parent.iterdir():
        (folder / p.name).write_bytes(p.read_bytes())
    meta = folder / src_meta.name
    before = tree_hash(folder)
    crop, masks = a_crop_and_a_mask(meta, d)
    seen = []
    out = pf.make_copies(meta, d, crop, masks, progress=lambda k, done, total: seen.append((k, done)))
    assert out == {"raw": 12, "overlay": 12} and seen[0][0] == "raw" and seen[-1] == ("overlay", 12)
    assert tree_hash(folder) == before                                                # not one byte of the trial's own files changed
    review = review_layer.review_dir(meta)
    assert sorted(p.name for p in review.iterdir()) == sorted([pf.FILE, pf.copy_path(meta, "raw").name, pf.copy_path(meta, "overlay").name])
    mx, my = masks[0][0] - crop[0], masks[0][1] - crop[1]                              # the mask in the crop's own pixels
    for kind in pf.KINDS:
        fr = frames(pf.copy_path(meta, kind))
        assert len(fr) == 12 and fr[0].shape[:2] == (crop[3], crop[2])               # cropped to the drawn rectangle
        assert fr[0][my:my + 40, mx:mx + 40].max() < BLACK                              # the dot under the mask is gone
    full = frames(folder / d["files"]["raw.mp4"])[0]
    cx, cy, cw, ch = crop
    assert full[masks[0][1]:masks[0][1] + 40, masks[0][0]:masks[0][0] + 40].max() > 100       # (and it was there in the original: the test can tell)
    got = frames(pf.copy_path(meta, "raw"))[0]
    away = np.ones(got.shape[:2], bool)
    away[max(my - 4, 0):my + 44, max(mx - 4, 0):mx + 44] = False
    assert abs(got[away].astype(float) - full[cy:cy + ch, cx:cx + cw][away].astype(float)).mean() < 6     # elsewhere it is the same picture (video compression only)
    assert pf.status(meta, d)["state"] == pf.FIXED and pf.read(meta)["crop_xywh"] == crop


def test_a_mask_alone_keeps_the_size_and_blacks_the_rectangle(tmp_path, plain):
    src_meta, d = plain
    folder = tmp_path / "recordings" / "P01"
    folder.mkdir(parents=True)
    for p in src_meta.parent.iterdir():
        (folder / p.name).write_bytes(p.read_bytes())
    meta = folder / src_meta.name
    w, h = d["privacy"]["full_picture_size"]
    x, y = first_dot_at(meta)
    pf.make_copies(meta, d, None, [(x - 20, y - 20, 40, 40)])
    fr = frames(pf.copy_path(meta, "raw"))
    assert fr[0].shape[:2] == (h, w) and fr[0][y - 20:y + 20, x - 20:x + 20].max() < BLACK
    assert frames(folder / d["files"]["raw.mp4"])[0][y - 20:y + 20, x - 20:x + 20].max() > 100


def test_a_trial_recorded_with_a_crop_is_put_back_at_full_size_before_a_new_mask_is_applied(tmp_path, cropped):
    src_meta, d = cropped
    folder = tmp_path / "recordings" / "P01"
    folder.mkdir(parents=True)
    for p in src_meta.parent.iterdir():
        (folder / p.name).write_bytes(p.read_bytes())
    meta = folder / src_meta.name
    rcrop = tuple(d["privacy"]["crop_xywh"])
    x, y = first_dot_at(meta)
    mask = (x - 20, y - 20, 40, 40)                                                    # in FULL-picture pixels
    pf.make_copies(meta, d, None, [mask])
    got = frames(pf.copy_path(meta, "raw"))[0]
    assert got.shape[:2] == (rcrop[3], rcrop[2])                                       # no new crop: the video stays as it was recorded
    assert got[mask[1] - rcrop[1]:mask[1] - rcrop[1] + 40, mask[0] - rcrop[0]:mask[0] - rcrop[0] + 40].max() < BLACK       # the mask, moved into the crop's own pixels
    assert frames(meta.parent / d["files"]["raw.mp4"])[0][mask[1] - rcrop[1]:mask[1] - rcrop[1] + 40, mask[0] - rcrop[0]:mask[0] - rcrop[0] + 40].max() > 100
    assert pf.outside_recorded_crop((0, 0, 1280, 720), rcrop) and not pf.outside_recorded_crop(rcrop, rcrop)


def test_drawing_again_makes_new_copies_and_moves_the_old_ones_aside_and_nothing_is_ever_deleted(tmp_path, plain):
    src_meta, d = plain
    folder = tmp_path / "recordings" / "P01"
    folder.mkdir(parents=True)
    for p in src_meta.parent.iterdir():
        (folder / p.name).write_bytes(p.read_bytes())
    meta = folder / src_meta.name
    before = tree_hash(folder)
    crop, masks = a_crop_and_a_mask(meta, d)
    pf.make_copies(meta, d, crop, masks)
    stamp = pf.copy_path(meta, "raw").stat().st_mtime_ns
    pf.make_copies(meta, d, crop, masks)                                               # the same drawing: the copies are kept
    assert pf.copy_path(meta, "raw").stat().st_mtime_ns == stamp
    first_raw = frames(pf.copy_path(meta, "raw"))[0].shape[:2]
    pf.make_copies(meta, d, None, [(0, 0, 64, 64)])                                    # another drawing: new copies, the old ones moved aside
    assert frames(pf.copy_path(meta, "raw"))[0].shape[:2] == tuple(d["privacy"]["full_picture_size"][::-1])
    aside = sorted((review_layer.review_dir(meta) / "superseded").iterdir())
    assert len(aside) == 2 and all(p.name.startswith(tuple(f"{review_layer.base_of(meta)}_{k}_protected_" for k in pf.KINDS)) for p in aside)
    assert frames([p for p in aside if "_raw_" in p.name][0])[0].shape[:2] == first_raw             # the earlier copy is whole and still readable
    assert [h["action"] for h in pf.read(meta)["history"]][0] == "drawn"
    assert not hasattr(pf, "clear")                                                    # there is no way to delete a copy from the program
    pf.make_copies(meta, d, crop, masks)                                               # back to the first drawing: the second pair is set aside as well, none overwritten
    assert len(list((review_layer.review_dir(meta) / "superseded").iterdir())) == 4
    assert tree_hash(folder) == before


def test_a_cancelled_copy_leaves_no_half_video_and_says_what_is_still_to_make(tmp_path, plain):
    src_meta, d = plain
    folder = tmp_path / "recordings" / "P01"
    folder.mkdir(parents=True)
    for p in src_meta.parent.iterdir():
        (folder / p.name).write_bytes(p.read_bytes())
    meta = folder / src_meta.name
    crop, masks = a_crop_and_a_mask(meta, d)
    pf.make_copies(meta, d, crop, masks, progress=lambda k, done, total: False if done >= 3 else None)
    assert not list(review_layer.review_dir(meta).glob("*.mp4")) and not list(review_layer.review_dir(meta).glob("*.part*"))
    s = pf.status(meta, d)
    assert s["state"] == pf.PENDING and s["missing"] == ["raw", "overlay"] and "not made yet" in s["text"]


def test_a_fix_cannot_be_written_beside_the_trial(plain):
    meta, d = plain
    with pytest.raises(review_layer.OriginalsProtected):
        pf.protect_video(meta.parent / d["files"]["raw.mp4"], meta.parent / "oops_protected.mp4", None, [(0, 0, 64, 64)])


def test_the_numbers_of_a_trial_do_not_change_when_copies_are_made(tmp_path, plain):
    src_meta, d = plain
    folder = tmp_path / "recordings" / "P01"
    folder.mkdir(parents=True)
    for p in src_meta.parent.iterdir():
        (folder / p.name).write_bytes(p.read_bytes())
    meta = folder / src_meta.name
    a = Trial(meta)
    rows_before, deg_before = list(a.rows), {k: v.copy() for k, v in a.deg.items()}
    a.close()
    crop, masks = a_crop_and_a_mask(meta, d)
    pf.make_copies(meta, d, crop, masks)
    b = Trial(meta)
    assert b.rows == rows_before and all(np.array_equal(b.deg[k], deg_before[k], equal_nan=True) for k in deg_before)
    b.close()


# ---------------------------------------------------------------------------------------------- the dots the drawing would hide
def test_a_dot_inside_a_mask_or_cut_off_by_the_crop_is_reported_with_the_number_of_pictures():
    rows = [{"knee_x": "100", "knee_y": "100", "knee_lost": "0", "hip_x": "485", "hip_y": "300", "hip_lost": "0", "ankle_x": "530", "ankle_y": "330", "ankle_lost": "0"},
            {"knee_x": "105", "knee_y": "105", "knee_lost": "0", "hip_x": "", "hip_y": "", "hip_lost": "1", "ankle_x": "531", "ankle_y": "331", "ankle_lost": "0"},
            {"knee_x": "900", "knee_y": "900", "knee_lost": "1", "hip_x": "486", "hip_y": "301", "hip_lost": "0", "ankle_x": "532", "ankle_y": "332", "ankle_lost": "0"}]
    got = pf.dot_problems(rows, ["knee", "hip", "ankle"], (480, 280, 100, 100), [(90, 90, 40, 40)])
    assert got == {"knee": {"frames": 2, "inside_mask": 2, "at_edge": 2},            # in a mask, and (so far from it) outside the crop
                   "hip": {"frames": 2, "inside_mask": 0, "at_edge": 2}}             # 5 px from the crop edge twice; a lost picture is not counted
    assert pf.dot_problems(rows, ["knee", "hip", "ankle"], (0, 0, 1000, 1000), []) == {}


def test_fit_snaps_rectangles_like_the_live_tab_and_drops_stray_ones():
    assert pf.fit([(10, 10, 301, 201), (0, 0, 5, 5)], (1280, 720)) == [(10, 10, 300, 200)]
    assert privacy.MIN_SIDE_PX == 32


# ---------------------------------------------------------------------------------------------- Review: the check, the button, the window
def _review_with_a_copy_of(src_meta, tmp_path):
    from PySide6.QtWidgets import QApplication
    from limbtrack.review import ReviewTab
    QApplication.instance() or QApplication([])
    folder = tmp_path / "recordings" / "P01"
    folder.mkdir(parents=True)
    for p in src_meta.parent.iterdir():
        (folder / p.name).write_bytes(p.read_bytes())
    meta = folder / src_meta.name
    tab = ReviewTab(tmp_path / "recordings")
    tab.resize(1500, 900)
    tab.show()
    tab.load(str(meta))
    return tab, meta


def _drag(dlg, a, b):
    """Drag from picture pixel `a` to `b` on the Privacy window's picture, with the mouse events a person makes."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    pic = dlg.picture
    for kind, pt, btn in ((QEvent.MouseButtonPress, a, Qt.LeftButton), (QEvent.MouseMove, b, Qt.NoButton), (QEvent.MouseButtonRelease, b, Qt.LeftButton)):
        pos = pic.to_widget(*pt)
        ev = QMouseEvent(kind, QPointF(pos), QPointF(pic.mapToGlobal(pos.toPoint())), btn, Qt.LeftButton if kind != QEvent.MouseButtonRelease else Qt.NoButton, Qt.NoModifier)
        {QEvent.MouseButtonPress: pic.mousePressEvent, QEvent.MouseMove: pic.mouseMoveEvent, QEvent.MouseButtonRelease: pic.mouseReleaseEvent}[kind](ev)


def test_review_flags_an_unprotected_trial_and_the_privacy_window_makes_the_copies(tmp_path, plain, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    src_meta, d = plain
    tab, meta = _review_with_a_copy_of(src_meta, tmp_path)
    before = tree_hash(meta.parent)
    try:
        assert "No crop or mask on the video" in tab.checks.text(everything=True) and tab.btn_privacy.isEnabled()
        tab._check_action("privacy", "")                                                  # the button on the check
        dlg = tab._privacy_dialog
        assert dlg is not None and dlg.isVisible() and not dlg.btn_make.isEnabled()      # nothing drawn: nothing to make
        x, y = first_dot_at(meta)
        dlg.mode_crop.setChecked(True)
        _drag(dlg, (x - 300, y - 200), (x + 300, y + 200))
        w, h = d["privacy"]["full_picture_size"]
        want = privacy.snap_rect(x - 300, y - 200, x + 300, y + 200, w, h)                # (clamped to the picture: the dot is near its top)
        assert dlg.crop == want
        dlg.mode_mask.setChecked(True)
        _drag(dlg, (x - 20, y - 20), (x + 20, y + 20))
        assert len(dlg.masks) == 1 and "dot is under a mask" in dlg.problem.text() and dlg.btn_make.isEnabled()      # the dot it hides is named
        _drag(dlg, (5, 5), (10, 10))                                                      # a stray click makes nothing
        assert len(dlg.masks) == 1
        dlg.btn_make.click()
        assert pf.status(meta, d)["state"] == pf.FIXED and tree_hash(meta.parent) == before       # the originals are byte for byte what they were
        text = tab.checks.text(everything=True)
        assert "Protected copies made" in text and "No crop or mask on the video" not in text
        assert frames(pf.copy_path(meta, "raw"))[0].shape[:2] == (want[3], want[2])
        assert not hasattr(dlg, "btn_forget")                                            # the window cannot delete a copy
        dlg.close()
    finally:
        tab.trial.close()
        tab.close()


def test_a_trial_the_operator_cleared_has_no_privacy_check_and_the_button_is_off_without_a_video(tmp_path, plain):
    src_meta, d = plain
    tab, meta = _review_with_a_copy_of(src_meta, tmp_path)
    try:
        assert tab.btn_privacy.isEnabled()
        tab.trial.meta.setdefault("privacy", {})["checked"] = {"how": pf.LOOKED_NOTHING}
        tab.update_summary()
        assert "No crop or mask on the video" not in tab.checks.text(everything=True)
        tab.show_view("overview")
        assert not tab.btn_privacy.isVisible()
    finally:
        tab.trial.close()
        tab.close()


def test_the_privacy_window_belongs_to_one_trial_and_closes_when_another_is_opened(tmp_path, plain, cropped):
    tab, meta = _review_with_a_copy_of(plain[0], tmp_path)
    other_folder = tmp_path / "recordings" / "P02"
    other_folder.mkdir(parents=True)
    for p in cropped[0].parent.iterdir():
        (other_folder / p.name).write_bytes(p.read_bytes())
    other = other_folder / cropped[0].name
    try:
        dlg = tab.open_privacy()
        assert dlg.isVisible()
        tab.load(str(meta))                                                               # the same trial again: the window stays
        assert dlg.isVisible()
        tab.load(str(other))
        assert not dlg.isVisible() and tab._privacy_dialog is None
        assert "No crop or mask on the video" not in tab.checks.text(everything=True)    # (this one was recorded with a crop)
        again = tab.open_privacy()
        assert again is not dlg and again.isVisible()
        again.close()
    finally:
        tab.trial.close()
        tab.close()
