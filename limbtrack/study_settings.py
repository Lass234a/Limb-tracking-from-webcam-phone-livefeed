"""The study's frozen settings (0.35.0, plan feature F13, validity item V6 'code freeze'; SOP_vs_APP A7).

The program remembers its set-up per computer (`settings.json`), so a value can differ from what the study decided without anyone noticing: on
2026-10-14 the dot colour was left on Auto and the tracking "seemed worse", and the live behaviour changed between the pilot participants P01 and
P02. `protocol.json` now has a block `study_settings` that lists the settings the study freezes, with the value of each. This module compares the
program's present values with that block and says in plain English which differ. It is a reminder and a record, never a rule: nothing here stops
Record, moves a setting or changes a measured number.

What is in the list (see SPECS; the block may freeze any subset):
  * settings the operator can change and that are remembered between launches (tolerances, dot colour, markers, cues, ruler length, the
    wanted camera picture, force smoothing, the load cell calibration in use): SOP_limbtrack.tex Table C1 lists most of them;
  * numbers that are fixed in the program's code and that decide a result or a procedure (the tracker's settings as one hash, the sliding-window
    peak of 500 ms, the force smoothing of the analysis, the rest times, the 10 % rule, the resting-force span). These can only change when the
    code does, so a difference there means the code is not the frozen code.
A value that cannot be read at this moment (the force is not connected, no camera is running) is "not checked", never a difference and never a
guess; the saved record lists what was not checked.

The values in protocol.json are the SOP's where the SOP states one (Table C1), else the program's value on the day the block was written; the
block's `proposals` names the values the group has not confirmed. Nothing was tried on the real rig.
"""
import difflib
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable

BLOCK_KEYS = ("settings", "proposals")
COLUMNS = ["study_settings_differences"]      # appended to the AnyBody export only (a count; blank when the trial has no block)
NOTE = ("settings_frozen: the names the study freezes; differences: what differed when the recording started (key, label, expected, actual, text), "
        "an empty list when nothing differed; not_checked: frozen settings that could not be read then (the force was not connected, no camera "
        "picture ...); proposals: frozen values the group has not confirmed. A reminder and a record: the recording was made either way.")
CHECK_TITLE = "Settings differ from the study"


class Unread(Exception):
    """A setting cannot be read at this moment; the message says why (it ends up as 'not checked')."""


@dataclass(frozen=True)
class Spec:
    key: str
    label: str                 # plain English, as the operator knows it
    kind: str                  # "bool", "number", "text"
    where: str                 # where the operator finds it, or "fixed in the program"
    read: Callable             # read(engine) -> the present value, or raises Unread
    unit: str = ""
    lo: float = None
    hi: float = None
    live: bool = True          # False: a number in the program's code (it changes only with the code)


# ------------------------------------------------------------------------------------------------ reading the program
def _force(e):
    if e.force is None:
        raise Unread("the force is not connected")
    return e.force


def _auto(e):
    _force(e)
    if not e.auto_mvic:
        raise Unread("the automatic MVIC markers are off")
    return e


def _camera_wanted(e):
    info = e.camera_info or {}
    r = info.get("requested")
    if not r:
        raise Unread("no camera is running (or the picture is a file or a stream)")
    return f"{int(r['width'])}x{int(r['height'])} at {r['fps']:g} fps"


def _tracker_hash(_e=None):
    from .tracker import settings
    return hashlib.sha256(json.dumps(settings(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _constant(module, name, factor=1.0):
    def read(_e):
        import importlib
        v = getattr(importlib.import_module(f"limbtrack.{module}"), name)
        return round(v * factor, 6)
    return read


def _extra(name):
    def read(e):
        extras = getattr(e, "study_extras", None) or {}
        if name not in extras:
            raise Unread("the window has not said (this number lives in the window's code)")
        return round(extras[name], 6)
    return read


SPECS = [
    # --- the operator's settings
    Spec("tolerance_deg", "Tolerance, main joint", "number", "Position section", lambda e: float(e.tolerance), " deg", 0.5, 30),
    Spec("neighbour_tolerance_deg", "Tolerance, other joints", "number", "Position section", lambda e: float(e.neighbour_tolerance), " deg", 0.5, 45),
    Spec("dot_kind", "Dot colour", "text", "Dots section", lambda e: e.kind),
    Spec("display_smoothing", "Smooth the displayed angles", "bool", "Position section", lambda e: bool(e.smoothing)),
    Spec("recording_first", "Recording first (less on screen while recording)", "bool", "Record section", lambda e: bool(e.recording_first)),
    Spec("save_overlay", "Also save the overlay video", "bool", "Record section", lambda e: bool(e.save_overlay)),
    Spec("camera_wanted", "Camera picture asked for", "text", "Camera section", _camera_wanted),
    Spec("ruler_length_cm", "Ruler length", "number", "Ruler section", lambda e: float(e.scale.length_cm), " cm", 1, 100),
    Spec("auto_mvic", "Mark the MVIC automatically", "bool", "Markers section", lambda e: (_force(e), bool(e.auto_mvic))[1]),
    Spec("auto_mvic_threshold_n", "Start when the force rises by", "number", "Markers section", lambda e: float(_auto(e).auto_mvic_threshold), " N", 1, 500),
    Spec("auto_mvic_length_s", "MVIC length", "number", "Markers section", lambda e: float(_auto(e).auto_mvic_seconds), " s", 1, 30),
    Spec("auto_stop_s", "Stop by itself after the MVIC end", "number", "Markers section", lambda e: float(e.auto_stop_s), " s", 0, 30),
    Spec("audio_cues", "Sound cues", "bool", "Markers section", lambda e: bool(e.audio_cues)),
    Spec("cue_push_every_s", "Tick during the hold every", "number", "Markers section", lambda e: float(e.cue_push_every_s), " s", 0, 10),
    Spec("force_smoothing_ms", "Force smoothing on screen", "number", "Force section", lambda e: round(_force(e).smooth_s * 1000.0, 6), " ms", 20, 1000),
    Spec("force_calibration_id", "Load cell calibration in use", "text", "Force section", lambda e: _force(e).calibration.regression_id),
    # --- numbers fixed in the program's code
    Spec("tracker_settings_sha256", "Tracking program version", "text", "fixed in the program", _tracker_hash, live=False),
    Spec("display_smoothing_window_s", "Window of the displayed-angle smoothing", "number", "fixed in the program",
         _constant("filters", "SMOOTHING_WINDOW_S"), " s", 0.01, 5, live=False),
    Spec("peak_average_ms", "Peak force: averaging against hum", "number", "fixed in the program",
         _constant("engine", "FORCE_PEAK_WINDOW_S", 1000.0), " ms", 1, 1000, live=False),
    Spec("window_peak_ms", "Sliding-window peak: window", "number", "fixed in the program (adjustable in Review)",
         _constant("force", "WINDOW_PEAK_DEFAULT_S", 1000.0), " ms", 20, 5000, live=False),
    Spec("analysis_force_smoothing_ms", "Force smoothing in Review", "number", "fixed in the program (adjustable in Review)",
         _constant("force", "SMOOTH_DEFAULT_S", 1000.0), " ms", 20, 1000, live=False),
    Spec("rest_same_position_s", "Rest between trials of one position", "number", "fixed in the program",
         _constant("rest", "REST_SAME_POSITION_S"), " s", 0, 600, live=False),
    Spec("rest_next_position_s", "Rest before the next position", "number", "fixed in the program",
         _constant("rest", "REST_NEXT_POSITION_S"), " s", 0, 600, live=False),
    Spec("rest_after_submax_s", "Rest after the submaximal contraction", "number", "fixed in the program",
         _constant("rest", "REST_AFTER_SUBMAX_S"), " s", 0, 600, live=False),
    Spec("repeat_check_pct", "Two trials of a position must agree within", "number", "fixed in the program",
         _constant("analysis", "REPEAT_WARN_PCT"), " %", 1, 100, live=False),
    Spec("max_trials_per_position", "Most trials at one position", "number", "fixed in the program",
         _constant("analysis", "MAX_TRIALS_PER_POSITION"), "", 1, 20, live=False),
    Spec("trials_per_position_planned", "Trials planned at each position", "number", "fixed in the program", _extra("trials_per_position"),
         "", 1, 20, live=False),
    Spec("rest_force_from_s", "Resting force: taken from this long before the push", "number", "fixed in the program",
         _extra("rest_force_from_s"), " s", 0.5, 30, live=False),
    Spec("rest_force_to_s", "Resting force: ... to this long before the push", "number", "fixed in the program",
         _extra("rest_force_to_s"), " s", 0, 30, live=False),
]
BY_KEY = {s.key: s for s in SPECS}


# ------------------------------------------------------------------------------------------------ the block in protocol.json
def _near(word, options):
    m = difflib.get_close_matches(str(word), [str(o) for o in options], n=1)
    return f' (did you mean "{m[0]}"?)' if m else ""


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(block, path="study_settings"):
    """Every mistake in the block as plain-English lines starting with their place in the file (empty when the block is fine)."""
    errs = []
    if not isinstance(block, dict):
        return [f"{path}: must be an object with the parts {', '.join(BLOCK_KEYS)}"]
    for k in block:
        if k not in BLOCK_KEYS and not str(k).startswith("_"):
            errs.append(f'{path}: unknown field "{k}"{_near(k, BLOCK_KEYS)}')
    settings = block.get("settings")
    if not isinstance(settings, dict) or not settings:
        errs.append(f"{path}.settings: needs at least one setting, as {{name: value}}")
        settings = {}
    for key, value in settings.items():
        if str(key).startswith("_"):
            continue
        spec = BY_KEY.get(key)
        where = f"{path}.settings.{key}"
        if spec is None:
            errs.append(f'{where}: unknown setting "{key}"{_near(key, BY_KEY)}')
        elif spec.kind == "bool":
            if not isinstance(value, bool):
                errs.append(f"{where}: must be true or false, not {value!r}")
        elif spec.kind == "number":
            if not _is_num(value):
                errs.append(f"{where}: must be a number, not {value!r}")
            elif not spec.lo <= value <= spec.hi:
                errs.append(f"{where}: must be between {spec.lo:g} and {spec.hi:g}{spec.unit}, not {value!r}")
        elif not isinstance(value, str):
            errs.append(f"{where}: must be text, not {value!r}")
        elif spec.key == "dot_kind" and value not in ("auto", "white", "black"):
            errs.append(f'{where}: "{value}" is not a dot colour; use auto, white or black{_near(value, ("auto", "white", "black"))}')
        elif spec.key == "tracker_settings_sha256" and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            errs.append(f"{where}: must be the 64 characters of a SHA-256 hash (python tools/freeze_check.py prints the present one)")
    proposals = block.get("proposals", [])
    if not (isinstance(proposals, list) and all(isinstance(p, str) for p in proposals)):
        errs.append(f"{path}.proposals: must be a list of setting names")
    else:
        errs += [f'{path}.proposals: "{p}" is not one of the frozen settings{_near(p, settings)}' for p in proposals if p not in settings]
    return errs


def clean(block):
    """The block in the form the program uses ({'settings': {...}, 'proposals': [...]}), or None for no block. Raises ValueError on mistakes."""
    if block is None:
        return None
    errs = validate(block)
    if errs:
        raise ValueError("; ".join(errs))
    return {"settings": {k: v for k, v in block["settings"].items() if not str(k).startswith("_")}, "proposals": list(block.get("proposals", []))}


# ------------------------------------------------------------------------------------------------ comparing
def say(spec, value):
    """A value as the operator would say it."""
    if spec.kind == "bool":
        return "on" if value else "off"
    if spec.kind == "number":
        return f"{value:g}{spec.unit}"
    if spec.key == "tracker_settings_sha256":
        return f"{value[:12]}..."
    if spec.key == "force_calibration_id" and value == "":
        return "none (a known weight, or no calibration)"
    return f'"{value}"' if value != "" else "empty"


def same(spec, expected, actual):
    if spec.kind == "number":
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-6)
    return expected == actual


def read_values(engine, code_only=False):
    """({key: present value}, {key: why it could not be read}) for every setting of SPECS (code_only: only the numbers fixed in the program's
    code, which need no running engine: `engine` then only has to carry `study_extras`)."""
    values, unread = {}, {}
    for spec in SPECS:
        if code_only and spec.live:
            continue
        try:
            values[spec.key] = spec.read(engine)
        except Unread as e:
            unread[spec.key] = str(e)
    return values, unread


def compare(block, values, unread=None):
    """Compare the (cleaned) block with the present values. Returns None without a block, else a dict:
    rows (one per frozen setting, in the order of SPECS: key, label, where, state 'same' / 'differs' / 'not_checked', expected, actual, text,
    proposal), differences (the 'differs' rows), not_checked (the 'not_checked' rows), names, proposals. Pure."""
    if block is None:
        return None
    unread = unread or {}
    rows = []
    for spec in SPECS:
        if spec.key not in block["settings"]:
            continue
        exp = block["settings"][spec.key]
        row = {"key": spec.key, "label": spec.label, "where": spec.where, "expected": exp, "actual": None,
               "proposal": spec.key in block["proposals"]}
        if spec.key in values:
            act = values[spec.key]
            row["actual"] = act
            if same(spec, exp, act):
                row.update(state="same", text=f"{spec.label}: {say(spec, act)}, as the study says.")
            elif spec.key == "tracker_settings_sha256":              # (the two hashes stay in the row and in meta.json; the operator is told what to do)
                row.update(state="differs", text=f"{spec.label}: the tracking program is not the version the study was frozen with. Recording is still "
                                                 "possible, but tell whoever maintains the program before using these trials.")
            else:
                row.update(state="differs", text=f"{spec.label}: {say(spec, act)} now; the study says {say(spec, exp)}. ({spec.where})")
        else:
            row.update(state="not_checked", text=f"{spec.label}: not checked ({unread.get(spec.key, 'not read')}).")
        rows.append(row)
    names = [r["key"] for r in rows]
    return {"names": names, "rows": rows, "proposals": [r["key"] for r in rows if r["proposal"]],
            "differences": [r for r in rows if r["state"] == "differs"], "not_checked": [r for r in rows if r["state"] == "not_checked"]}


def evaluate(block, engine):
    """The comparison for an engine (reads the program, see read_values); None without a block."""
    if block is None:
        return None
    values, unread = read_values(engine)
    return compare(block, values, unread)


def signature(result):
    """What changes when the list of differences changes (used to write the session log only when it did)."""
    return None if result is None else tuple((d["key"], d["actual"]) for d in result["differences"])


def reminder(result, max_names=3):
    """The one-line amber sentence for the Live tab, or '' when nothing differs (or there is no block)."""
    if not result or not result["differences"]:
        return ""
    labels = [d["label"] for d in result["differences"]]
    n = len(labels)
    shown = "; ".join(labels[:max_names]) + (f"; and {n - max_names} more" if n > max_names else "")
    return f"Study settings: {n} differ{'s' if n == 1 else ''} from the frozen list ({shown})"


def list_text(result):
    """What the 'Show the list' button shows: the differences first, then what could not be checked, then a count of the rest."""
    if result is None:
        return "protocol.json has no study_settings block, so nothing is compared."
    diff, nc = result["differences"], result["not_checked"]
    lines = []
    if diff:
        lines.append(f"{len(diff)} of the {len(result['rows'])} frozen settings differ from the study:")
        lines += [f"  - {d['text']}" + ("  [the study's value is a proposal]" if d["proposal"] else "") for d in diff]
    else:
        lines.append(f"All {len(result['rows']) - len(nc)} settings that can be read now are as the study says.")
    if nc:
        lines += ["", f"{len(nc)} could not be checked now:"] + [f"  - {r['text']}" for r in nc]
    lines += ["", "This is a reminder only. Nothing is stopped and nothing measured changes; every recording saves this list in its meta.json "
                  "(study_settings)."]
    return "\n".join(lines)


def log_detail(result):
    """The session log line: counts, then each difference as key=present (study expected)."""
    if result is None:
        return "no study_settings block in protocol.json"
    diff = result["differences"]
    parts = [f"checked={len(result['rows']) - len(result['not_checked'])}", f"not_checked={len(result['not_checked'])}", f"differs={len(diff)}"]
    parts += [f"{d['key']}={d['actual']} (study {d['expected']})" for d in diff]
    return "; ".join(parts)


def meta_block(result, block_hash=None):
    """What goes into a trial's meta.json (`study_settings`): None when protocol.json has no block, else the frozen names, the differences (an empty
    list when none), what was not checked, the proposals and a note."""
    if result is None:
        return None
    return {"settings_frozen": list(result["names"]),
            "differences": [{k: d[k] for k in ("key", "label", "expected", "actual", "text")} for d in result["differences"]],
            "not_checked": [{"key": r["key"], "reason": r["text"]} for r in result["not_checked"]],
            "proposals": list(result["proposals"]), "block_sha256": block_hash, "note": NOTE}


def block_hash(block):
    """A hash of the frozen values (so two trials can be seen to have been recorded against the same list), or None."""
    if block is None:
        return None
    return hashlib.sha256(json.dumps(block["settings"], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def check_item(meta):
    """What Review's checks list says about a trial's saved block: ('info', text, title) when the trial was recorded with differences, else None
    (no block, nothing differed, or a damaged block). Level 'info' on purpose: it says the set-up was not the study's, not that a number is wrong."""
    if not isinstance(meta, dict) or not isinstance(meta.get("differences"), list) or not meta["differences"]:
        return None
    texts = [str(d.get("text") or d.get("key")) for d in meta["differences"] if isinstance(d, dict)]
    if not texts:
        return None
    n = len(texts)
    return ("info", f"Recorded with {n} setting{'' if n == 1 else 's'} that differ{'s' if n == 1 else ''} from the study's frozen list: " + " ".join(texts)
            + " The numbers are what was measured; this only says the set-up was not the study's.", f"{CHECK_TITLE} ({n})")


def export_cells(meta):
    """The AnyBody export cell (COLUMNS) from a trial's meta.json: the number of differences, or blank without a block (an earlier trial) or with a
    damaged one."""
    out = {k: "" for k in COLUMNS}
    block = meta.get("study_settings")
    if isinstance(block, dict) and isinstance(block.get("differences"), list):
        out["study_settings_differences"] = len(block["differences"])
    return out
