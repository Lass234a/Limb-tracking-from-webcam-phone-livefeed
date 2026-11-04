"""Fixing the privacy crop and masks after the recording (0.34.0, the item "Fix the privacy crop and masks in Review" of PARKED.md).

In the Live tab the crop and the black masks are set before recording, and `privacy.apply` changes only the saved copies. When the
recorder forgot (no crop, no mask, a face outside what was covered), the saved `raw.mp4` and `overlay.mp4` show the unprotected picture.
Review never writes a trial's own files (`review_layer`), so the fix is made the same way as every other review edit:

    recordings/review/P02/<trial base>/privacy_fix.json            the crop and masks drawn in Review (full-picture pixels) and the copies made
    recordings/review/P02/<trial base>/<base>_raw_protected.mp4    the saved raw video with the crop and masks applied
    recordings/review/P02/<trial base>/<base>_overlay_protected.mp4

The unprotected originals stay where they are, untouched; what happens to them (delete, move, encrypt) is a decision for the people
responsible for the data. Nothing here touches tracking or the CSV: no measured number changes. This module is pure (no Qt).
"""
from datetime import datetime
from pathlib import Path

import cv2

from . import privacy, review_layer

FILE = "privacy_fix.json"
FORMAT = 1
KINDS = ("raw", "overlay")
LOOKED_NOTHING = "looked: nothing identifies the participant"          # the operator's own word in the Live tab (gui.py): nothing to hide

# the states of a trial, in the order they are decided
RECORDED = "recorded"            # the recorder applied a crop or masks
FIXED = "fixed"                  # no crop or mask when recorded, but Review made protected copies
PENDING = "pending"              # a crop or masks were drawn in Review, the protected copies are not (all) made yet
NOTHING_TO_HIDE = "nothing"      # the operator looked and said nothing identifies the participant
UNPROTECTED = "unprotected"      # no crop, no mask, and nobody said that nothing needs hiding
UNKNOWN = "unknown"              # the trial has no privacy record at all (recorded before it existed)


# ------------------------------------------------------------------------------------------------ where things live
def fix_path(meta_path):
    return review_layer.review_file(meta_path, FILE)


def copy_path(meta_path, kind):
    """Where the protected copy of 'raw' or 'overlay' goes: in the trial's review folder, named like the original."""
    return review_layer.review_file(meta_path, f"{review_layer.base_of(meta_path)}_{kind}_protected.mp4")


def original_path(meta_path, meta, kind):
    """The saved video of this trial ('raw' or 'overlay') or None when the trial has none."""
    name = (meta.get("files") or {}).get(f"{kind}.mp4")
    p = Path(meta_path).parent / name if name else None
    return p if p is not None and p.is_file() else None


# ------------------------------------------------------------------------------------------------ the record
def recorded(meta):
    """(crop, masks) the recorder applied: crop (x, y, w, h) or None, masks a list of (x, y, w, h). Both empty: nothing was applied."""
    pv = (meta or {}).get("privacy") or {}
    crop = tuple(pv["crop_xywh"]) if pv.get("crop_xywh") else None
    masks = [tuple(m) for m in (pv.get("masks_xywh") or [])]
    return crop, masks


def read(meta_path):
    """The saved fix as {'crop_xywh': [x, y, w, h] or None, 'masks_xywh': [...], 'copies': {...}, ...}; {} when there is none."""
    d = review_layer.read_json(fix_path(meta_path))
    if d.get("format") != FORMAT:
        return {}
    crop = d.get("crop_xywh")
    d["crop_xywh"] = tuple(int(v) for v in crop) if isinstance(crop, (list, tuple)) and len(crop) == 4 else None
    d["masks_xywh"] = [tuple(int(v) for v in m) for m in (d.get("masks_xywh") or []) if isinstance(m, (list, tuple)) and len(m) == 4]
    d["copies"] = d.get("copies") if isinstance(d.get("copies"), dict) else {}
    return d


def drawn(fix):
    """(crop, masks) of a fix that was read; (None, []) for no fix."""
    return (fix.get("crop_xywh"), list(fix.get("masks_xywh") or [])) if fix else (None, [])


def write(meta_path, crop, masks, copies=None, history=None):
    """Save the crop and masks drawn in Review (full-picture pixels) and what has been made from them. Only inside the review layer."""
    old = read(meta_path)
    data = {"format": FORMAT, "trial": review_layer.base_of(meta_path),
            "crop_xywh": list(crop) if crop else None, "masks_xywh": [list(m) for m in masks],
            "copies": copies if copies is not None else {},
            "meaning": "Drawn in Review on the full picture. The protected copies in this folder have the crop and the masks applied; the trial's own "
                       "raw.mp4 and overlay.mp4 are not changed. No tracked position and no number changes.",
            "history": list(old.get("history") or []) + [{"at": datetime.now().isoformat(timespec="seconds"), **(history or {"action": "saved"})}]}
    review_layer.write_json(fix_path(meta_path), data)
    return data


def clear(meta_path):
    """Forget the fix and delete the protected copies this module made (never an original). Returns the names of the files removed."""
    gone = []
    for p in [fix_path(meta_path)] + [copy_path(meta_path, k) for k in KINDS]:
        try:
            review_layer.guard(p).unlink()
            gone.append(p.name)
        except FileNotFoundError:
            pass
    return gone


def copies_made(meta_path, fix=None):
    """{'raw': True/False, 'overlay': ...} for the copies recorded in the fix that still exist and were made from this crop and these masks."""
    fix = fix if fix is not None else read(meta_path)
    out = {}
    for k in KINDS:
        c = (fix.get("copies") or {}).get(k) or {}
        same = (c.get("crop_xywh") and tuple(c["crop_xywh"]) or None) == fix.get("crop_xywh") and \
            [tuple(m) for m in c.get("masks_xywh") or []] == list(fix.get("masks_xywh") or [])
        out[k] = bool(c) and same and copy_path(meta_path, k).is_file()
    return out


# ------------------------------------------------------------------------------------------------ the state of a trial
def status(meta_path, meta):
    """{'state': one of the constants above, 'text': a plain sentence, 'missing': ['raw', ...] the protected copies still to make}."""
    crop, masks = recorded(meta)
    fix = read(meta_path)
    fcrop, fmasks = drawn(fix)
    if fcrop or fmasks:
        made = copies_made(meta_path, fix)
        missing = [k for k in KINDS if original_path(meta_path, meta, k) is not None and not made[k]]
        if missing:
            return {"state": PENDING, "missing": missing,
                    "text": "A crop or masks were drawn in Review, but the protected copy of the " + " and ".join(missing) + " video is not made yet."}
        return {"state": FIXED, "missing": [],
                "text": "Protected copies made in Review (crop and masks applied). The unprotected originals are still on the disk."}
    if crop or masks:
        return {"state": RECORDED, "missing": [], "text": "A crop or masks were applied when this trial was recorded."}
    pv = meta.get("privacy")
    if pv is None:
        return {"state": UNKNOWN, "missing": [],
                "text": "This trial has no privacy record (it was recorded before the program kept one): the saved video may show the unprotected picture."}
    if ((pv.get("checked") or {}).get("how") or "") == LOOKED_NOTHING:
        return {"state": NOTHING_TO_HIDE, "missing": [], "text": "The operator looked at the picture and said nothing identifies the participant."}
    return {"state": UNPROTECTED, "missing": [],
            "text": "No crop and no mask were applied to the saved video, and nobody said that nothing needs hiding: a face or a name may be in "
                    "raw.mp4 and overlay.mp4. Draw a crop or masks in Review (Privacy) to make protected copies."}


# ------------------------------------------------------------------------------------------------ geometry
def fit(rects, full_size):
    """Clamp the rectangles drawn by hand to the picture and to even sizes (`privacy.snap_rect`); rectangles that are too small are dropped."""
    w, h = full_size
    out = []
    for x, y, rw, rh in rects:
        r = privacy.snap_rect(x, y, x + rw, y + rh, w, h)
        if r is not None:
            out.append(r)
    return out


def effective_crop(new_crop, recorded_crop):
    """The crop of a protected copy: the one drawn in Review, else the one the video was recorded with (the video holds nothing outside it)."""
    return tuple(new_crop) if new_crop else (tuple(recorded_crop) if recorded_crop else None)


def outside_recorded_crop(new_crop, recorded_crop):
    """True when a crop drawn in Review reaches outside the crop the video was recorded with: those pixels were never saved, they come out black."""
    if not new_crop or not recorded_crop:
        return False
    x, y, w, h = new_crop
    rx, ry, rw, rh = recorded_crop
    return x < rx or y < ry or x + w > rx + rw or y + h > ry + rh


def dot_problems(rows, dot_names, crop, masks):
    """{dot: {'frames': n, 'inside_mask': n, 'at_edge': n}} for the dots that the crop or masks drawn would cut off or cover: a tracked dot
    inside a mask, outside the crop or closer to its edge than `privacy.DOT_WARN_MARGIN_PX` in a picture. `rows`: data.csv rows as dicts.
    A hidden dot cannot be re-checked by eye in the protected video, so the operator is told before the copies are made."""
    out = {}
    for d in dot_names:
        inside = edge = frames = 0
        for r in rows:
            if r.get(f"{d}_x", "") == "" or r.get(f"{d}_lost") == "1":
                continue
            try:
                x, y = float(r[f"{d}_x"]), float(r[f"{d}_y"])
            except (KeyError, ValueError, TypeError):
                continue
            hit_mask = any(privacy.mask_distance(m, x, y) <= 0 for m in masks)
            hit_edge = bool(crop) and privacy.edge_distance(crop, x, y) < privacy.DOT_WARN_MARGIN_PX
            if hit_mask:
                inside += 1
            if hit_edge:
                edge += 1
            frames += hit_mask or hit_edge
        if frames:
            out[d] = {"frames": frames, "inside_mask": inside, "at_edge": edge}
    return out


# ------------------------------------------------------------------------------------------------ making the copies
def protect_video(src, dst, crop, masks, recorded_crop=None, full_size=None, fps=None, progress=None):
    """Write `dst` (a path inside the review layer) as `src` with the crop and masks applied, and return the number of pictures written.

    `crop`, `masks`: drawn in Review, in full-picture pixels. `recorded_crop`: the crop `src` was recorded with (its pictures are only that
    part, so they are put back at their place on a black full picture first, as Review does). `full_size`: (w, h) of the full picture; the
    size of `src` itself when it was not cropped. `progress(done, total)` may return False to stop: nothing is put in place then (returns
    the number written, and `dst` is not created). The temporary file is removed when anything goes wrong; `src` is only read."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise OSError(f"Could not open {Path(src).name} to read it.")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not full_size:
            full_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_crop = effective_crop(crop, recorded_crop)
        ow, oh = (out_crop[2], out_crop[3]) if out_crop else tuple(full_size)
        part = review_layer.temp_video_path(dst)
        writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        if not writer.isOpened():
            raise OSError("Could not open the protected video file for writing.")
        done, finished = 0, False
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    finished = True
                    break
                full = privacy.pad_to_full(img, recorded_crop, full_size) if recorded_crop else img
                writer.write(privacy.apply(full, out_crop, masks))
                done += 1
                if progress is not None and progress(done, total) is False:
                    break
        finally:
            writer.release()
        if finished and done:
            review_layer.finish_video(part, dst)
        else:
            part.unlink(missing_ok=True)
        return done
    finally:
        cap.release()


def make_copies(meta_path, meta, crop, masks, progress=None):
    """Save the fix and make the protected copy of every saved video the trial has. `progress(kind, done, total)` may return False to cancel
    (what was already made stays, the rest is reported as still to make). Returns {'raw': pictures written or None, ...}."""
    rcrop, _rmasks = recorded(meta)
    full_size = tuple((meta.get("privacy") or {}).get("full_picture_size") or ()) or None
    fps = meta.get("video_fps_written") or meta.get("measured_fps")
    rec = {"crop_xywh": list(crop) if crop else None, "masks_xywh": [list(m) for m in masks]}
    copies, result = {}, {}
    old = read(meta_path)
    same = (old.get("crop_xywh") == (tuple(crop) if crop else None)) and list(old.get("masks_xywh") or []) == [tuple(m) for m in masks]
    if same:
        copies = dict(old.get("copies") or {})                          # (the copies already made from this very crop and masks stay)
    else:
        for k in KINDS:                                                  # a new crop or masks: the old copies no longer match, they are removed first
            try:
                review_layer.guard(copy_path(meta_path, k)).unlink()
            except FileNotFoundError:
                pass
    write(meta_path, crop, masks, copies, history={"action": "drawn", "crop_xywh": rec["crop_xywh"], "masks_xywh": rec["masks_xywh"]})
    for kind in KINDS:
        src = original_path(meta_path, meta, kind)
        if src is None:
            result[kind] = None
            continue
        if same and copies_made(meta_path, read(meta_path)).get(kind):
            result[kind] = (copies.get(kind) or {}).get("frames")
            continue
        n = protect_video(src, copy_path(meta_path, kind), crop, masks, recorded_crop=rcrop, full_size=full_size, fps=fps,
                          progress=(lambda d, t, k=kind: progress(k, d, t)) if progress else None)
        result[kind] = n
        if (copy_path(meta_path, kind)).is_file():
            copies[kind] = {**rec, "frames": n, "made_at": datetime.now().isoformat(timespec="seconds"), "file": copy_path(meta_path, kind).name}
            write(meta_path, crop, masks, copies, history={"action": f"{kind} copy made", "frames": n})
        else:
            break                                                        # (cancelled: no more copies; the state says what is still to make)
    return result
