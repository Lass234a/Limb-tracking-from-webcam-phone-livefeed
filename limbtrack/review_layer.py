"""The review layer (0.30.0): everything Review changes or adds about a trial is kept here, never in the trial's own files.

    recordings/P02/                         ORIGINALS: written once by the recorder (meta, data, force, videos), never by Review
    recordings/review/P02/<trial base>/     the review layer of one trial
        review.json                         the edit list (the source of truth): force zero, filter, dot edits, fingerprints of the
                                            originals it was made from, an append-only history
        dots_edit.csv                       per-picture positions of dots fixed by re-tracking or by hand
        verdict.json, reference.json        what was judged or typed in Review (the on-site verdict stays with the trial)
        ...exports of this trial
    recordings/review/P02/exports/          the participant's exports

A reviewed trial is the original files read through its edit list: nothing is copied (the videos stay where they are), any edit can
be reset, and "original" and "reviewed" are two readings of the same trial. This module is the only code that writes in the review
layer, and `guard` refuses every path that is not under a folder called `review` or that lies next to a `*_meta.json`, so a mistake
elsewhere cannot make Review write into a participant's folder.
"""
import contextlib
import csv
import hashlib
import json
import shutil
import math
import os
from datetime import datetime
from pathlib import Path

REVIEW_FOLDER = "review"
FORMAT = 1
META_SUFFIX = "_meta.json"
DOT_COLUMNS = ["edit", "frame", "t_s", "dot", "x", "y", "source"]
HASHED = ("meta.json", "data.csv", "force.csv")            # fingerprinted with SHA-256 (the videos by size and time only: they are large)


class OriginalsProtected(OSError):
    """A write was asked for outside the review layer, or next to a trial's own files."""


# ------------------------------------------------------------------------------------------------ where things live
def base_of(meta_path):
    p = Path(meta_path)
    return p.name[: -len(META_SUFFIX)] if p.name.endswith(META_SUFFIX) else p.stem


def participant_review_dir(meta_path):
    """`<folder that holds the participant folders>/review/<participant folder name>`: a sibling tree of the originals."""
    folder = Path(meta_path).parent
    return folder.parent / REVIEW_FOLDER / folder.name


def review_dir(meta_path):
    return participant_review_dir(meta_path) / base_of(meta_path)


def review_json_path(meta_path):
    return review_dir(meta_path) / "review.json"


def dots_csv_path(meta_path):
    return review_dir(meta_path) / "dots_edit.csv"


def exports_dir(meta_path):
    return participant_review_dir(meta_path) / "exports"


def review_file(meta_path, name):
    """A file of this trial's review layer (`verdict.json`, `reference.json`, an export name)."""
    return review_dir(meta_path) / name


def stamp(meta_path):
    """Modification times of this trial's review files: a change means every number derived from the review is out of date."""
    out = []
    for p in (review_json_path(meta_path), dots_csv_path(meta_path), review_file(meta_path, "verdict.json"), review_file(meta_path, "reference.json")):
        try:
            out.append(os.stat(p).st_mtime_ns)
        except OSError:
            out.append(0)
    return tuple(out)


# ------------------------------------------------------------------------------------------------ the only way to write
def guard(path):
    """Raise OriginalsProtected unless `path` is inside a folder called `review` and not beside a trial's own files."""
    p = Path(path).resolve()
    if REVIEW_FOLDER not in [part.lower() for part in p.parts[:-1]]:
        raise OriginalsProtected(f"{p} is not in a review folder: Review only writes there")
    try:
        beside_a_trial = p.parent.is_dir() and next(p.parent.glob("*" + META_SUFFIX), None) is not None
    except OSError:
        beside_a_trial = False
    if beside_a_trial:
        raise OriginalsProtected(f"{p.parent} holds a trial's own files: Review does not write there")
    return p


@contextlib.contextmanager
def atomic(path, mode="w", **kwargs):
    """Open `path` (in the review layer) for writing; the file appears only when the block ends without an error. The folder is made."""
    p = guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    kwargs.setdefault("encoding", None if "b" in mode else "utf-8")
    try:
        with open(tmp, mode, **kwargs) as fh:
            yield fh
        os.replace(tmp, p)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_json(path, data):
    with atomic(path) as fh:
        fh.write(json.dumps(data, indent=2))


def read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def temp_video_path(path):
    """A guarded temporary name for a video being written (cv2 picks the codec from the extension); `finish_video` puts it in place."""
    p = guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.with_name(p.stem + ".part" + p.suffix)


def finish_video(tmp, path):
    os.replace(guard(tmp), guard(path))


# ------------------------------------------------------------------------------------------------ fingerprints of the originals
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(meta_path, meta=None):
    """{file key: {size, mtime_ns, sha256 (not for videos)}} of the trial's own files, read only."""
    meta_path = Path(meta_path)
    if meta is None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = {"meta.json": meta_path.name, **(meta.get("files") or {})}
    out = {}
    for key, name in files.items():
        p = meta_path.parent / name
        if not p.is_file():
            continue
        st = p.stat()
        entry = {"file": name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        if key in HASHED:
            entry["sha256"] = _sha256(p)
        out[key] = entry
    return out


def changed_originals(meta_path, recorded):
    """Names of the trial's files that differ from the fingerprints a review was made from (size or time first, then the hash). []: unchanged."""
    meta_path = Path(meta_path)
    changed = []
    for key, old in (recorded or {}).items():
        p = meta_path.parent / old.get("file", "")
        try:
            st = p.stat()
        except OSError:
            changed.append(old.get("file", key) + " (missing)")
            continue
        if st.st_size == old.get("size") and st.st_mtime_ns == old.get("mtime_ns"):
            continue
        if "sha256" in old and st.st_size == old.get("size") and _sha256(p) == old["sha256"]:
            continue                                                                  # touched, not changed
        changed.append(old.get("file", key))
    return changed


# ------------------------------------------------------------------------------------------------ the edit list
class Review:
    """The edit list of one trial. `Review.load` never fails (a missing or unreadable file is an empty review); nothing is written until
    `save`, and `save` writes only in the review layer. Every change adds a line to `history`, so what was done, when, is never lost."""

    def __init__(self, meta_path, data=None, dot_rows=None):
        self.meta_path = Path(meta_path)
        d = data or {}
        self.data = {"format": FORMAT, "trial": base_of(meta_path), "participant": d.get("participant", ""),
                     "software_version": d.get("software_version", ""), "original": d.get("original") or {},
                     "force": d.get("force") or {"zero": None, "filter": None}, "window": d.get("window"),
                     "dots": list(d.get("dots") or []), "created_at": d.get("created_at", ""), "edited_at": d.get("edited_at", ""),
                     "history": list(d.get("history") or [])}
        self.data["force"].setdefault("zero", None)
        self.data["force"].setdefault("filter", None)
        self.dot_rows = list(dot_rows or [])              # [{edit, frame, t_s, dot, x, y, source}] of the frame-by-frame edits

    # ---- loading and saving
    @classmethod
    def load(cls, meta_path):
        data = read_json(review_json_path(meta_path))
        rows = []
        try:
            with open(dots_csv_path(meta_path), newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        rows.append({"edit": int(r["edit"]), "frame": int(r["frame"]), "t_s": float(r["t_s"]), "dot": r["dot"],
                                     "x": float(r["x"]), "y": float(r["y"]), "source": r["source"]})
                    except (KeyError, ValueError, TypeError):
                        continue
        except OSError:
            pass
        return cls(meta_path, data, rows)

    def exists(self):
        return review_json_path(self.meta_path).is_file()

    def save(self, software_version=""):
        """Write the edit list (and the per-picture positions) in the review layer. The first save also records the fingerprints of the
        originals. Raises OriginalsProtected if the review folder would be somewhere it must not be."""
        now = datetime.now().isoformat(timespec="seconds")
        if not self.data["original"]:
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            self.data["original"] = {"folder": str(self.meta_path.parent), "files": fingerprint(self.meta_path, meta) if meta else {}}
            self.data["participant"] = meta.get("participant") or ""
            self.data["created_at"] = now
        self.data["edited_at"] = now
        if software_version:
            self.data["software_version"] = software_version
        if self.dot_rows:
            with atomic(dots_csv_path(self.meta_path), newline="") as fh:
                w = csv.writer(fh)
                w.writerow(DOT_COLUMNS)
                for r in self.dot_rows:
                    w.writerow([r["edit"], r["frame"], f"{r['t_s']:.4f}", r["dot"], f"{r['x']:.2f}", f"{r['y']:.2f}", r["source"]])
        else:
            with contextlib.suppress(OSError):
                guard(dots_csv_path(self.meta_path)).unlink()
        write_json(review_json_path(self.meta_path), self.data)

    def log(self, action, detail=""):
        self.data["history"].append({"at": datetime.now().isoformat(timespec="seconds"), "action": action, "detail": detail})

    # ---- what is in it
    @property
    def zero(self):
        return self.data["force"].get("zero")

    @property
    def dots(self):
        return self.data["dots"]

    def has_edits(self):
        return bool(self.zero or self.data["force"].get("filter") or self.dots)

    def edited_dot_names(self):
        return sorted({e["dot"] for e in self.dots})

    def originals_changed(self):
        return changed_originals(self.meta_path, (self.data.get("original") or {}).get("files"))

    # ---- zero
    def set_zero(self, zero_v, source, slope, region=None):
        """`source`: 'typed', 'stretch' or 'rest'. `slope`: {'n_per_v', 'regression_id', 'from': 'recorded'|'active'}; kept so the newtons of
        this edit do not change if the active regression is replaced later."""
        self.data["force"]["zero"] = {"zero_v": float(zero_v), "source": source, "region_s": list(region) if region else None, **slope}
        self.log("zero", f"{source}: {zero_v:+.5f} V" + (f" from {region[0]:.2f}-{region[1]:.2f} s" if region else ""))

    def clear_zero(self):
        if self.zero:
            self.data["force"]["zero"] = None
            self.log("zero reset", "back to the zero recorded with the trial")

    # ---- dots
    def next_edit_id(self):
        return 1 + max([e.get("id", 0) for e in self.dots] + [0])

    def add_dot_edit(self, dot, kind, rows=None, **params):
        """Add an edit of `dot`. `kind`: 'hold' (from_s, to_s, x, y), 'lost' (from_s, to_s), 'retrack' or 'manual' (rows: the per-picture
        positions, each {frame, t_s, x, y}). Later edits win over earlier ones where they overlap. Returns the edit's id."""
        eid = self.next_edit_id()
        entry = {"id": eid, "dot": dot, "kind": kind, **params}
        self.data["dots"].append(entry)
        for r in rows or []:
            self.dot_rows.append({"edit": eid, "frame": int(r["frame"]), "t_s": float(r["t_s"]), "dot": dot, "x": float(r["x"]), "y": float(r["y"]),
                                  "source": kind})
        self.log(f"dot {dot}: {kind}", ", ".join(f"{k} {v}" for k, v in params.items() if not isinstance(v, (list, dict))))
        return eid

    def undo_dot_edit(self, dot):
        """Take back the latest edit of `dot` (the one with the highest id). False when it has none."""
        mine = [e for e in self.dots if e["dot"] == dot]
        if not mine:
            return False
        last = max(mine, key=lambda e: e["id"])
        self.data["dots"] = [e for e in self.dots if e is not last]
        self.dot_rows = [r for r in self.dot_rows if r["edit"] != last["id"]]
        self.log(f"dot {dot}: {last['kind']} taken back")
        return True

    def reset_dot(self, dot):
        ids = {e["id"] for e in self.dots if e["dot"] == dot}
        if ids:
            self.data["dots"] = [e for e in self.dots if e["dot"] != dot]
            self.dot_rows = [r for r in self.dot_rows if r["edit"] not in ids]
            self.log(f"dot {dot} reset", "back to the positions recorded with the trial")

    def reset_all(self):
        if self.has_edits():
            self.data["force"] = {"zero": None, "filter": None}
            self.data["dots"], self.dot_rows = [], []
            self.log("reset", "everything back to the recorded trial")

    def summary_lines(self, n_pictures=None):
        """Plain lines for the Check tab and the exports' `review_edits` column."""
        out = []
        z = self.zero
        if z:
            how = {"typed": "typed", "stretch": "taken from a stretch", "rest": "taken from the rest before the push"}.get(z.get("source"), z.get("source", ""))
            out.append(f"force zero {how}: {z['zero_v']:+.4f} V" + (f" ({z['region_s'][0]:.2f}-{z['region_s'][1]:.2f} s)" if z.get("region_s") else ""))
        f = self.data["force"].get("filter")
        if f:
            out.append("force filter: " + ", ".join(f"{k} {v}" for k, v in f.items() if v not in (None, False)))
        for dot in self.edited_dot_names():
            parts = []
            for e in (x for x in self.dots if x["dot"] == dot):
                if e["kind"] == "hold":
                    parts.append(f"held at ({e['x']:.1f}, {e['y']:.1f}) {e['from_s']:.2f}-{e['to_s']:.2f} s")
                elif e["kind"] == "lost":
                    parts.append(f"marked lost {e['from_s']:.2f}-{e['to_s']:.2f} s")
                else:
                    k = sum(1 for r in self.dot_rows if r["edit"] == e["id"])
                    parts.append(f"{'re-tracked' if e['kind'] == 'retrack' else 'set by hand'} in {k} picture{'' if k == 1 else 's'}")
            out.append(f"dot {dot}: " + "; ".join(parts))
        return out


# ------------------------------------------------------------------------------------------------ participant-level files (0.32.0)
#   recordings/review/P02/overview_cache.json   the participant overview's finished rows, so a second load needs no work
#   recordings/review/P02/approval.json         "this participant has been looked through" and which state of the review was accepted
# Both live in the participant's review folder (beside the per-trial folders), are written only through `atomic` (so only inside a review
# folder, never beside a trial's own files), and neither changes a number or a recorded file.
OVERVIEW_CACHE = "overview_cache.json"
OVERVIEW_CACHE_FORMAT = 4        # 0.35.0: the rows also have the check weight columns; 0.34.0 (3): the stabilisation and standing-reference columns (an older cache is worked out again; 2 had the other angles at the peak, 0.33.0)
OVERVIEW_CACHE_VARIANTS = 4        # per trial: the latest keys kept (original / reviewed, a few smoothing choices), so switching back is instant
APPROVAL = "approval.json"
APPROVAL_FORMAT = 1
APPROVALS_IN_FULL = 5              # the latest approvals keep their list of trials; older ones are cut to their counts (a short history)
APPROVALS_KEPT = 50


def _json_default(o):
    """json's `default`: numpy numbers and arrays, paths and anything else a row might hold, as plain JSON."""
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def participant_dir_of(folder_or_meta):
    """`review/<participant>/` for a participant folder (`recordings/P02`) or for the `*_meta.json` of one of its trials."""
    p = Path(folder_or_meta)
    if p.name.endswith(META_SUFFIX) or p.is_file():
        p = p.parent
    return p.parent / REVIEW_FOLDER / p.name


def overview_cache_path(folder_or_meta):
    return participant_dir_of(folder_or_meta) / OVERVIEW_CACHE


def approval_path(folder_or_meta):
    return participant_dir_of(folder_or_meta) / APPROVAL


def read_overview_cache(folder_or_meta):
    """The saved overview rows as {'format', 'trials': {base: [{'key', 'row'}, ...]}}, or {} (none, unreadable, or written by another format)."""
    d = read_json(overview_cache_path(folder_or_meta))
    if d.get("format") != OVERVIEW_CACHE_FORMAT or not isinstance(d.get("trials"), dict):
        return {}
    clean = {}                                   # (a file edited by hand or half written: only entries of the right shape are used, the rest is dropped)
    for base, variants in d["trials"].items():
        good = [v for v in variants if isinstance(v, dict) and isinstance(v.get("key"), list) and isinstance(v.get("row"), dict)] if isinstance(variants, list) else []
        if good:
            clean[base] = good
    return {**d, "trials": clean}


def write_overview_cache(folder_or_meta, trials):
    """Save `trials` ({base: [{'key': [...], 'row': {...}}, ...], newest first}) as the participant's overview cache: one compact file, written
    whole or not at all. Raises OriginalsProtected if the file would not be inside a review folder."""
    kept = {b: list(v)[:OVERVIEW_CACHE_VARIANTS] for b, v in trials.items() if v}
    data = {"format": OVERVIEW_CACHE_FORMAT, "written_at": datetime.now().isoformat(timespec="seconds"), "trials": kept}
    with atomic(overview_cache_path(folder_or_meta)) as fh:
        fh.write(json.dumps(data, separators=(",", ":"), default=_json_default))        # (NaN is written as NaN: only this program reads the file)


def trial_fingerprint(meta_path):
    """A short hash of what the review of one trial consists of: its edit list and dot positions, the verdict (judged in Review and the one
    judged on site), the reference readings (review layer and the older file beside the trial). Content, not times, so a file that is saved
    again with the same content is the same review, and any real change shows."""
    meta_path = Path(meta_path)
    base = base_of(meta_path)
    h = hashlib.sha256()
    for label, p in (("review", review_json_path(meta_path)), ("dots", dots_csv_path(meta_path)),
                     ("verdict", review_file(meta_path, "verdict.json")), ("reference", review_file(meta_path, "reference.json")),
                     ("site_verdict", meta_path.with_name(base + "_verdict.json")), ("site_reference", meta_path.with_name(base + "_reference.json"))):
        h.update(label.encode("ascii") + b"\0")
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"-")
        h.update(b"\0")
    return h.hexdigest()[:16]


def read_approvals(folder_or_meta):
    """Every approval of this participant, oldest first (an empty list: never approved, or the file cannot be read)."""
    d = read_json(approval_path(folder_or_meta))
    items = d.get("approvals")
    return [a for a in items if isinstance(a, dict)] if isinstance(items, list) else []


def read_approval(folder_or_meta):
    """The latest approval record, or {} when the participant was never approved."""
    items = read_approvals(folder_or_meta)
    return items[-1] if items else {}


def _short(record):
    """An old approval cut to what the history needs: when, by which version, how many trials of which kind (not the list of trials)."""
    out = {k: v for k, v in record.items() if k != "trials"}
    if "trials" in record or "trials_left_out" not in out:      # (an approval that was cut short already says how many it left out: keep that)
        out["trials_left_out"] = len(record.get("trials") or [])
    return out


def write_approval(folder_or_meta, record):
    """Add `record` as the participant's newest approval and return it as saved. `record`: approved_at (local time, ISO), software_version,
    participant, counts {trials, valid, compromised, not_judged, unreadable, edited}, trials [{base, verdict, edited, fingerprint}]
    (see review_overview.OverviewPage.commit_approval). The earlier approvals stay in the file (append-only: none is changed); to keep it
    short, all but the latest APPROVALS_IN_FULL lose their list of trials and keep their counts, and at most APPROVALS_KEPT are kept.
    Written whole or not at all, and only inside a review folder."""
    saved = dict(record)
    saved.setdefault("approved_at", datetime.now().isoformat(timespec="seconds"))
    path = approval_path(folder_or_meta)
    if path.is_file() and path.stat().st_size and not read_approvals(folder_or_meta):
        shutil.copy2(path, guard(path.with_name(path.name + ".bak")))         # (an unreadable file is kept, not silently replaced: its history may be recoverable)
    approvals = read_approvals(folder_or_meta) + [saved]
    approvals = [a if i >= len(approvals) - APPROVALS_IN_FULL else _short(a) for i, a in enumerate(approvals)][-APPROVALS_KEPT:]
    data = {"format": APPROVAL_FORMAT, "participant": saved.get("participant") or participant_dir_of(folder_or_meta).name,
            "meaning": "Approving records that this participant's review was looked through, and which state of each trial's review was accepted "
                       "(fingerprint = a hash of the trial's review files). It changes no number and no recorded file. Older approvals are kept.",
            "approvals": approvals}
    with atomic(approval_path(folder_or_meta)) as fh:
        fh.write(json.dumps(data, indent=2, default=_json_default))
    return saved
