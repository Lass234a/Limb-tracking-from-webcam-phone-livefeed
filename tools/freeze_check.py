#!/usr/bin/env python3
"""The version-freeze check (0.35.0, plan feature F13 / V6): prints what to paste into the study log before a participant is measured, so the
program that recorded a trial can be found again. It only READS: it never tags, commits, writes or changes a file.

usage (from the project folder, with the project's Python):
    .venv/Scripts/python.exe tools/freeze_check.py [--strict]

It prints the git commit and branch, whether the working tree is clean (no changed or new file that git does not know about: a commit only
reproduces the program when there are none), the tags at the commit, the program's version, the hash of the program's own source files and of
protocol.json (the same two hashes every trial's meta.json carries), the study's frozen settings of protocol.json, and for the numbers that are
fixed in the program's code whether they are still the frozen ones. The settings the operator can change (tolerances, dot colour, markers ...)
are compared by the running program itself (amber line in the Live tab, `meta.json` `study_settings`).
--strict: the exit status is 1 when the tree is not clean or a fixed number differs (for a script); without it the status is 0.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from limbtrack import provenance, study_settings                  # noqa: E402
from limbtrack.engine import SOFTWARE_VERSION                       # noqa: E402
from limbtrack.protocol import load_study_settings                  # noqa: E402

LIST_MAX = 12          # files named in the 'not clean' lines


def git(*args, cwd=ROOT, raw=False):
    """The output of a git command, or None when git is missing or the command failed (never raises, never writes). `raw` keeps the leading
    spaces of the first line (git status --porcelain starts a line with a space)."""
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.rstrip("\n") if raw else out.stdout.strip()


def git_state(cwd=ROOT):
    """{commit, branch, tags, changed (tracked files), untracked (new files git ignores nothing of)}; every value None where git cannot say."""
    status = git("status", "--porcelain", cwd=cwd, raw=True)
    lines = None if status is None else [ln for ln in status.splitlines() if ln.strip()]
    tags = git("tag", "--points-at", "HEAD", cwd=cwd)
    return {"commit": git("rev-parse", "HEAD", cwd=cwd), "branch": git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd),
            "tags": None if tags is None else tags.split(),
            "changed": None if lines is None else [ln[3:] for ln in lines if not ln.startswith("??")],
            "untracked": None if lines is None else [ln[3:] for ln in lines if ln.startswith("??")]}


def fixed_numbers(block):
    """The study_settings comparison of the numbers fixed in the program's code ('same' / 'differs' rows). Reads the code, not a running engine."""
    from limbtrack import gui                                       # (the window's own constants; importing the module opens no window)
    engine = SimpleNamespace(study_extras=dict(gui.STUDY_EXTRAS))
    values, unread = study_settings.read_values(engine, code_only=True)
    result = study_settings.compare(block, values, unread)
    code_keys = {s.key for s in study_settings.SPECS if not s.live}
    return [r for r in result["rows"] if r["key"] in code_keys]


def report(git_info, version, code_hash, protocol_hash, block, fixed, now=None):
    """The lines to paste into the study log. Pure: everything it prints is an argument."""
    lines = [f"LIMBTRACK FREEZE CHECK  {time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}",
             f"Program version      : {version}",
             f"Git commit           : {git_info['commit'] or 'unknown (git could not say)'}"
             + (f"  (branch {git_info['branch']})" if git_info["branch"] else ""),
             f"Tags at this commit  : {'unknown' if git_info['tags'] is None else ', '.join(git_info['tags']) or 'none'}"]
    ch, un = git_info["changed"], git_info["untracked"]
    if ch is None or un is None:
        lines.append("Working tree         : UNKNOWN (git could not say)")
    elif not ch and not un:
        lines.append("Working tree         : CLEAN (the commit is the program)")
    else:
        lines.append(f"Working tree         : NOT CLEAN - {len(ch)} changed file(s), {len(un)} new file(s) that are not in the commit")
        for title, names in (("changed", ch), ("new", un)):
            for n in names[:LIST_MAX]:
                lines.append(f"                       {title}: {n}")
            if len(names) > LIST_MAX:
                lines.append(f"                       ... and {len(names) - LIST_MAX} more {title}")
    lines += [f"Code hash (sha256)   : {code_hash} (limbtrack/*.py, line endings ignored; needs no git)",
              f"Protocol hash        : {protocol_hash or 'unknown'} (protocol.json as JSON; carries the study settings below)"]
    if block is None:
        lines.append("Study settings       : none (protocol.json has no study_settings block)")
        return lines
    n = len(block["settings"])
    lines.append(f"Study settings       : {n} frozen, {len(block['proposals'])} of them proposals the group has not confirmed (marked *)")
    for spec in study_settings.SPECS:
        if spec.key in block["settings"]:
            lines.append(f"  {spec.key} = {study_settings.say(spec, block['settings'][spec.key])}{' *' if spec.key in block['proposals'] else ''}")
    bad = [r for r in fixed if r["state"] != "same"]
    if bad:
        lines.append(f"Numbers fixed in the program: {len(bad)} of {len(fixed)} are NOT the frozen ones - the code is not the frozen code:")
        lines += [f"  {r['text']}" for r in bad]
    else:
        lines.append(f"Numbers fixed in the program: all {len(fixed)} are the frozen ones.")
    lines.append("The settings you can change in the window are compared by the running program (amber line in the Live tab, session_log.csv, "
                 "meta.json study_settings).")
    return lines


def problems(git_info, fixed):
    """What makes --strict fail: a tree that is not clean (or unknown) and fixed numbers that differ."""
    out = []
    if git_info["changed"] is None or git_info["untracked"] is None or git_info["changed"] or git_info["untracked"]:
        out.append("the working tree is not clean")
    out += [f"{r['key']} is not the frozen value" for r in fixed if r["state"] == "differs"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the version-freeze facts to paste into the study log (reads only).")
    ap.add_argument("--strict", action="store_true", help="exit status 1 when the tree is not clean or a fixed number differs")
    args = ap.parse_args(argv)
    protocol = ROOT / "protocol.json"
    block = load_study_settings(protocol)
    info = git_state()
    fixed = fixed_numbers(block) if block is not None else []
    print("\n".join(report(info, SOFTWARE_VERSION, provenance.code_sha256(), provenance.protocol_sha256(protocol), block, fixed)))
    bad = problems(info, fixed)
    if bad:
        print("\nNot ready to freeze: " + "; ".join(bad) + ".")
    return 1 if args.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
