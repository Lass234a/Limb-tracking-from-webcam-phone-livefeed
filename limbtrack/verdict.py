"""The operator's verdict on a trial (0.28.0): judged on site right after it (the Live tab) or later (Review).

A trial is **valid** (completely well done, deemed valid on site) or **compromised**, with one or more reasons:

  wrong dots      a dot placed or marked in the wrong place, jumped, swapped or lost when it mattered
  wrong position  the participant was not in the position chosen in the program (or the wrong one was chosen)
  bad force       the strap was not pulling, slipped or jerked, the signal was cut off, the zero was off
  other           what happened is in the note

Not judged until something is marked. A note can be added to any verdict.

The verdict is kept in a small file next to the trial, `<trial>_verdict.json`, because `meta.json` is never rewritten after a trial
(like the reference readings' side file). Every change keeps the one before it in `history`, so a verdict changed later in Review still
shows what was decided on site. A compromised trial is not counted: the session plan and the 10 % check leave it out, and the export says
why. Its files stay where they are; nothing is renamed or deleted.

In the pilot sessions (2026-10-28/29) six trials were saved under the wrong position and two had no force at all; nothing in the files
said so, and finding them took a review of the whole session afterwards.
"""
import json
import os
from datetime import datetime
from pathlib import Path

VALID, COMPROMISED = "valid", "compromised"
REASONS = {                                  # key -> what the window and the files say
    "wrong_dots": "wrong dots",
    "wrong_position": "wrong position",
    "bad_force": "bad force",
    "other": "other",
}
ON_SITE, IN_REVIEW = "on site (Live tab)", "Review"


def verdict_path(meta_path):
    """`<trial>_verdict.json` next to `<trial>_meta.json`."""
    p = Path(meta_path)
    return p.with_name(p.name[: -len("_meta.json")] + "_verdict.json")


def read_verdict(meta_path):
    """The verdict of a trial as a dict (verdict, reasons, note, judged_at, judged_in, history), or None when it was never judged or the
    file cannot be read."""
    try:
        data = json.loads(verdict_path(meta_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check(verdict, reasons):
    """None when the pair makes sense, else why not (a programming error: the window only offers sensible pairs)."""
    if verdict not in (VALID, COMPROMISED, None):
        return f"unknown verdict {verdict!r}"
    unknown = [r for r in reasons if r not in REASONS]
    if unknown:
        return f"unknown reason {unknown[0]!r}"
    if verdict == COMPROMISED and not reasons:
        return "a compromised trial needs a reason"
    if verdict != COMPROMISED and reasons:
        return "only a compromised trial has reasons"
    return None


def write_verdict(meta_path, verdict, reasons=(), note="", where=ON_SITE):
    """Save the verdict next to the trial (atomically) and return what was saved. `verdict`: VALID, COMPROMISED or None (not judged: a
    verdict taken back). The verdict before it, if any, goes into `history`. The trial's own files are never touched."""
    problem = check(verdict, list(reasons))                              # (before sorting: an unknown reason must not just vanish)
    if problem:
        raise ValueError(problem)
    reasons = [r for r in REASONS if r in set(reasons)]                  # in a fixed order
    old = read_verdict(meta_path)
    history = list((old or {}).get("history") or [])
    if old is not None:
        history.append({k: old.get(k) for k in ("verdict", "reasons", "note", "judged_at", "judged_in")})
    data = {
        "verdict": verdict, "reasons": reasons, "note": (note or "").strip(),
        "judged_at": datetime.now().isoformat(timespec="seconds"), "judged_in": where,
        "history": history,
        "meaning": "valid: the trial was completely well done (judged by the operator); compromised: not counted (session plan, 10 % "
                   "check), with the reasons; null: not judged. Earlier verdicts are in history (0.28.0).",
    }
    path = verdict_path(meta_path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)                # no half-written file is left behind; the verdict is as it was
        except OSError:
            pass
        raise
    return data


def is_compromised(meta_path):
    v = read_verdict(meta_path)
    return v is not None and v.get("verdict") == COMPROMISED


def describe(v):
    """One line: 'valid', 'compromised: wrong position, bad force (strap slipped)', 'not judged'."""
    if not v or not v.get("verdict"):
        return "not judged" + (f" ({v['note']})" if v and v.get("note") else "")
    text = v["verdict"]
    if v["verdict"] == COMPROMISED:
        text += ": " + ", ".join(REASONS.get(r, r) for r in v.get("reasons") or [])
    return text + (f" ({v['note']})" if v.get("note") else "")
