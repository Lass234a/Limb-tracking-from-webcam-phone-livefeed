"""Keep the recorded files safe (0.30.0). Review never writes a trial's own files; this tool lets you PROVE it later, and stops an accidental edit.

    python tools/protect_originals.py manifest <folder>   record the SHA-256 of every recorded file
    python tools/protect_originals.py verify   <folder>   compare the files with that record: changed, missing, new
    python tools/protect_originals.py readonly <folder>   mark the recorded files read-only (add --writable to undo it)

<folder> is a recordings folder (it holds the participant folders: P01, P02, ...) or one participant folder. A participant folder is one with
`*_meta.json` files in it. The record is `review/<participant>/MANIFEST.sha256` next to the participant folders, never inside them, so the
recorded folders stay exactly as the recorder left them. Nothing here changes the content of a file.

Also keep an ordinary backup of the recordings: this protects against a mistake in the program or a stray edit, not against a failed disk.
"""
import argparse
import hashlib
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from limbtrack import review_layer  # noqa: E402

MANIFEST = "MANIFEST.sha256"


def participant_folders(folder):
    """The participant folders under `folder` (or `folder` itself when it holds trials)."""
    folder = Path(folder)
    if any(folder.glob("*_meta.json")):
        return [folder]
    return sorted(p for p in folder.iterdir() if p.is_dir() and p.name.lower() != review_layer.REVIEW_FOLDER and any(p.glob("*_meta.json")))


def recorded_files(participant):
    return sorted(p for p in Path(participant).iterdir() if p.is_file())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(participant):
    return review_layer.participant_review_dir(Path(participant) / "x_meta.json") / MANIFEST


def write_manifest(participant):
    lines = [f"# SHA-256 of the recorded files of {Path(participant).name}, made {datetime.now().isoformat(timespec='seconds')}"]
    for p in recorded_files(participant):
        lines.append(f"{sha256(p)}  {p.stat().st_size}  {p.name}")
    out = manifest_path(participant)
    with review_layer.atomic(out) as fh:
        fh.write("\n".join(lines) + "\n")
    return out, len(lines) - 1


def read_manifest(participant):
    out = {}
    try:
        text = manifest_path(participant).read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line and not line.startswith("#"):
            digest, size, name = line.split(None, 2)
            out[name] = (digest, int(size))
    return out


def verify(participant):
    """(changed, missing, new) file names against the manifest, or None when there is no manifest."""
    recorded = read_manifest(participant)
    if recorded is None:
        return None
    now = {p.name: p for p in recorded_files(participant)}
    changed = sorted(n for n, (digest, size) in recorded.items() if n in now and (now[n].stat().st_size != size or sha256(now[n]) != digest))
    return changed, sorted(n for n in recorded if n not in now), sorted(n for n in now if n not in recorded)


def set_readonly(participant, readonly=True):
    n = 0
    for p in recorded_files(participant):
        os.chmod(p, stat.S_IREAD if readonly else stat.S_IREAD | stat.S_IWRITE)
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["manifest", "verify", "readonly"])
    ap.add_argument("folder")
    ap.add_argument("--writable", action="store_true", help="with readonly: make the files writable again")
    args = ap.parse_args(argv)
    folders = participant_folders(args.folder)
    if not folders:
        print(f"No participant folder (with *_meta.json files) found in {args.folder}")
        return 2
    bad = 0
    for p in folders:
        if args.command == "manifest":
            out, n = write_manifest(p)
            print(f"{p.name}: {n} files recorded in {out}")
        elif args.command == "verify":
            res = verify(p)
            if res is None:
                print(f"{p.name}: no manifest yet (run: manifest)")
                bad += 1
                continue
            changed, missing, new = res
            if changed or missing:
                bad += 1
            print(f"{p.name}: " + ("unchanged" if not (changed or missing or new) else
                                    f"CHANGED {changed or '-'}; MISSING {missing or '-'}; new files {new or '-'}"))
        else:
            n = set_readonly(p, readonly=not args.writable)
            print(f"{p.name}: {n} files are now {'writable' if args.writable else 'read-only'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
