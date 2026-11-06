"""Screenshots of saved trials: one contact sheet per trial, to look at the positions from the picture.

For every trial in the recordings folder (`..._meta.json` + video) a picture is made with six frames side by side: the start, the
frame just before the MVIC start marker, the MVIC start, the middle of the hold, the MVIC end and the last frame. (Where a trial
has no markers, six evenly spaced frames are used.) Each frame is labelled with its number and time. The sheet is written as a
PNG next to nothing else: into `snapshots\\` in the project folder.

THE PICTURES SHOW THE PARTICIPANT. They stay on this computer: `snapshots\\` is in .gitignore, and the sheets must not be sent,
uploaded or committed. (A privacy crop and masks set during the recording are already in the saved video.)

Run from the project folder:
    .venv\\Scripts\\python.exe tools\\trial_snapshots.py                          (every trial in recordings\\)
    .venv\\Scripts\\python.exe tools\\trial_snapshots.py --participant P01 P02    (only these)
    .venv\\Scripts\\python.exe tools\\trial_snapshots.py D:\\other\\recordings --video overlay
    .venv\\Scripts\\python.exe tools\\trial_snapshots.py --each                    (also one PNG per frame, full size)

--video raw (default) is the untouched picture; --video overlay is the picture with the angles, trails and marker banners drawn on.
--each also writes every chosen frame as its own full-size PNG, for a closer look at a dot.
Nothing is written into the recordings folder, and the videos are only read.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = ("start", "before MVIC", "MVIC start", "hold", "MVIC end", "last")


def events_of(meta):
    """{label: first frame} from meta.json (the marker list the program saved)."""
    out = {}
    for e in meta.get("events") or []:
        lab, fr = e.get("label"), e.get("frame")
        if isinstance(lab, str) and isinstance(fr, int) and lab not in out:
            out[lab] = fr
    return out


def pick_frames(meta, n_frames, fps):
    """Six (label, frame) pairs. Frames come from the MVIC markers where they exist, otherwise evenly spaced."""
    ev = events_of(meta)
    last = max(n_frames - 1, 0)
    s, e = ev.get("MVIC start"), ev.get("MVIC end")
    if s is not None and e is not None and 0 <= s < e <= last:
        picks = [("start", 0), ("before MVIC", max(s - int(round(fps)), 0)), ("MVIC start", s),
                 ("hold", (s + e) // 2), ("MVIC end", e), ("last", last)]
    else:
        picks = [(f"{round(100 * k / 5)} %", int(round(last * k / 5))) for k in range(6)]
    return picks


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    return img if ok else None


def pad_crop(img, meta):
    """A privacy-cropped video is shown as it is (the saved copy); nothing is put back here."""
    return img


def label_frame(img, text, scale):
    out = img.copy()
    h = max(int(26 * scale), 18)
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def sheet(frames, cols=3, tile_w=640):
    tiles = []
    for img in frames:
        s = tile_w / img.shape[1]
        tiles.append(cv2.resize(img, (tile_w, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA))
    th = max(t.shape[0] for t in tiles)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        row += [np.zeros((th, tile_w, 3), np.uint8)] * (cols - len(row))
        row = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in row]
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="?", default=str(HERE.parent / "recordings"))
    ap.add_argument("--participant", nargs="+", help="only these participant codes (P01 P02 ...)")
    ap.add_argument("--video", choices=("raw", "overlay"), default="raw")
    ap.add_argument("--out", default=str(HERE.parent / "snapshots"), help="where the PNG files go (default: snapshots\\ in the project)")
    ap.add_argument("--each", action="store_true", help="also write every chosen frame as its own full-size PNG")
    args = ap.parse_args()

    wanted = {p.upper() for p in args.participant} if args.participant else None
    metas = sorted(Path(args.recordings).rglob("*_meta.json"))
    if not metas:
        print(f"No *_meta.json found under {args.recordings}")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    for mp in metas:
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            print(f"skipped {mp.name}: {ex}")
            continue
        code = str(meta.get("participant") or mp.parent.name)
        if wanted and code.upper() not in wanted:
            continue
        name = (meta.get("files") or {}).get(f"{args.video}.mp4")
        video = mp.parent / name if name else None
        if not video or not video.is_file():
            print(f"skipped {mp.name}: no {args.video} video")
            continue
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("video_fps_written") or 30.0)
        if n < 2:
            print(f"skipped {mp.name}: video has {n} frames")
            cap.release()
            continue
        tiles = []
        stem = re.sub(r"_meta$", "", mp.stem)
        for lab, idx in pick_frames(meta, n, fps):
            img = read_frame(cap, min(idx, n - 1))
            if img is None:
                continue
            tiles.append(label_frame(img, f"{lab}  (frame {idx}, {idx / fps:.1f} s)", img.shape[1] / 640.0))
            if args.each:
                cv2.imwrite(str(out / f"{stem}_{lab.replace(' ', '_').replace('%', 'pct')}_f{idx}.png"), img)
        cap.release()
        if not tiles:
            print(f"skipped {mp.name}: no frame could be read")
            continue
        head = f"{code}  {meta.get('test')}  target {meta.get('target_deg', meta.get('target'))}  ({args.video})"
        sh = sheet(tiles)
        bar = np.zeros((34, sh.shape[1], 3), np.uint8)
        cv2.putText(bar, head, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}_sheet.png"), np.vstack([bar, sh]))
        made += 1
        print(f"{code} {meta.get('test')}: {stem}_sheet.png")
    print(f"\n{made} sheet(s) in {out}  (they show the participant: keep them on this computer)")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
