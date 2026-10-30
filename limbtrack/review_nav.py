"""Finding a participant and the trials of that participant for the Review tab (0.29.0). No Qt and no video here: a trial is listed from
its small `*_meta.json` and `*_verdict.json` only, so a participant with 64 trials lists in a fraction of a second.

A participant is a folder that holds `*_meta.json` files (`recordings/P01`, `Pilot/P02`). A trial is named by the recorder
(`P01_03_knee_extension_20deg_T2_20261028-114202_meta.json`, older: `P11_hip_extension_k0_0deg_20261015-120009_meta.json`).
A trial that cannot be read (a crashed recording) is still listed, with the reason, so it is not silently missing from a participant.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import verdict as verdicts

META_SUFFIX = "_meta.json"
_STAMP = re.compile(r"(?:_T(\d+))?_(\d{8}-\d{6})(?:-(\d+))?" + re.escape(META_SUFFIX) + r"$")    # same form as recorder._STARTED
GLYPHS = {verdicts.VALID: "✓", verdicts.COMPROMISED: "✗", None: "○"}
WORDS = {verdicts.VALID: "valid", verdicts.COMPROMISED: "compromised", None: "not judged"}


def _natural(name):
    """'P2' before 'P10': digits compare as numbers."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def _short_test(label):
    """'Knee extension (MVIC)' -> 'Knee extension' (every test here is an MVIC; the suffix only makes the drop-down wider)."""
    return re.sub(r"\s*\(MVIC\)\s*$", "", label or "").strip()


@dataclass
class TrialEntry:
    """One line of the trial drop-down."""
    meta_path: Path
    base: str                       # the file name without `_meta.json`
    participant: str = ""
    test_key: str = ""
    test_label: str = ""
    target_deg: object = None
    position_number: object = None
    trial_number: object = None
    started: str = ""               # 'YYYY-MM-DDTHH:MM:SS' (from the file name, so also for a trial whose meta.json cannot be read)
    status: str = ""
    readable: bool = True
    error: str = ""
    verdict: object = None          # verdict.VALID, verdict.COMPROMISED or None (not judged)

    @property
    def unfinished(self):
        return self.status == "recording"

    @property
    def clock(self):
        return self.started[11:16]

    def refresh_verdict(self):
        v = verdicts.read_verdict(self.meta_path)
        self.verdict = (v or {}).get("verdict") or None

    @property
    def label(self):
        """'01 · Knee extension 20° · T2 · 12:27 · ✓ valid'. A trial that cannot be read says so instead."""
        if not self.readable:
            return f"{self.base} · cannot be read"
        parts = []
        if self.position_number not in (None, ""):
            parts.append(f"{int(self.position_number):02d}")
        parts.append(_short_test(self.test_label) or self.test_key)
        if self.target_deg not in (None, ""):
            parts.append(f"{float(self.target_deg):g}°")                     # (apart: the test's own name may end in a number, 'hip 90')
        if self.trial_number not in (None, ""):
            parts.append(f"T{self.trial_number}")
        if self.clock:
            parts.append(self.clock)
        parts.append(f"{GLYPHS[self.verdict]} {WORDS[self.verdict]}" + (" · unfinished" if self.unfinished else ""))
        return " · ".join(parts)


def _started_from_name(name):
    m = _STAMP.search(name)
    if not m:
        return ""
    try:
        return datetime.strptime(m.group(2), "%Y%m%d-%H%M%S").isoformat(timespec="seconds")
    except ValueError:
        return ""


def read_entry(meta_path):
    """The drop-down line of one trial. Never raises: a file that cannot be read gives an entry with `readable` False and the reason."""
    p = Path(meta_path)
    base = p.name[: -len(META_SUFFIX)] if p.name.endswith(META_SUFFIX) else p.stem
    e = TrialEntry(meta_path=p, base=base, started=_started_from_name(p.name))
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("meta.json is not a trial description")
    except (OSError, ValueError) as err:
        e.readable, e.error = False, str(err)
        return e
    e.participant = meta.get("participant") or ""
    e.test_key, e.test_label = meta.get("test_key") or "", meta.get("test_label") or ""
    e.target_deg, e.position_number, e.trial_number = meta.get("target_deg"), meta.get("position_number"), meta.get("trial_number")
    e.status = meta.get("status") or ""
    e.started = _started_from_name(p.name) or str(meta.get("started_at") or "")[:19]
    e.refresh_verdict()
    return e


def trial_paths(folder):
    """The `*_meta.json` files of a participant folder, oldest first (the time in the name, then the trial number, then -2, -3, then the name)."""
    def key(p):
        m = _STAMP.search(p.name)
        return (m.group(2), int(m.group(1) or 0), int(m.group(3) or 0), p.name) if m else ("", 0, 0, p.name)
    return sorted(Path(folder).glob("*" + META_SUFFIX), key=key)


def list_trials(folder):
    """[TrialEntry] of a participant folder, in the order they were recorded. [] when the folder is missing or holds none."""
    try:
        return [read_entry(p) for p in trial_paths(folder)]
    except OSError:
        return []


def has_trials(folder):
    try:
        return next(Path(folder).glob("*" + META_SUFFIX), None) is not None
    except OSError:
        return False


def participant_folders(root):
    """[(name, path, number of trials)] of the sub-folders of `root` that hold trials, 'P2' before 'P10'."""
    out = []
    try:
        subs = [d for d in Path(root).iterdir() if d.is_dir()]
    except OSError:
        return out
    for d in subs:
        try:
            n = sum(1 for _ in d.glob("*" + META_SUFFIX))
        except OSError:
            continue
        if n:
            out.append((d.name, d, n))
    return sorted(out, key=lambda x: _natural(x[0]))


def resolve_folder(folder):
    """What 'Open folder...' picked: ('participant', [folder]) when it holds trials itself (`Pilot/P01`); ('participants', [sub-folders])
    when its sub-folders do (`Pilot/`); ('none', []) otherwise."""
    folder = Path(folder)
    if has_trials(folder):
        return "participant", [folder]
    subs = [p for _, p, _ in participant_folders(folder)]
    return ("participants", subs) if subs else ("none", [])


def next_unjudged(entries, current):
    """Index of the next readable trial after `current` that has no verdict (looking on from the current one, then from the start; never the
    current one itself), or None when there is none. `current` may be -1 (nothing open yet)."""
    n = len(entries)
    for step in range(1, n + 1 if current < 0 else n):
        i = (current + step) % n
        if entries[i].readable and entries[i].verdict is None:
            return i
    return None


def counts(entries):
    """{'valid': n, 'compromised': n, 'not judged': n, 'unreadable': n}."""
    out = {"valid": 0, "compromised": 0, "not judged": 0, "unreadable": 0}
    for e in entries:
        if not e.readable:
            out["unreadable"] += 1
        else:
            out[WORDS[e.verdict]] += 1
    return out
