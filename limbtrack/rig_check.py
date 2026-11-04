"""The rig check (0.35.0, plan feature F9, workflow item W1): one list, run with the rig EMPTY before the participant arrives, that says in a line each
whether the equipment is ready. It separates the time the equipment takes from the time the participant takes (the pilot's first trial came 1.0 to
1.6 hours after the start, camera trouble on 29 October).

The items, each green, amber or grey ("not checked"), with a one-line reason:
  camera        a live camera opened by NAME, pictures arriving, and the picture rate the live loop really runs at (the engine's own loop)
  force         the force source connected, samples arriving, and the sample rate
  zero_noise    the spread of the zeroed signal over 2 s with the 50 Hz hum taken out (the hum amplitude is given beside it)
  check_weight  a check weight of the last hours, green, with the calibration in use (limbtrack/check_weight.py, F8)
  beeps         the real cues played through limbtrack/cues.py, and the operator's tick that they were heard
  ruler         the scale is marked
  lamp          the operator's tick that the lamp is on and even (nothing in the program can see that)

It only READS: it moves no zero, changes no setting and writes nothing but the line the operator asks to save (the session log) and the saved
block (`meta.json` `rig_check` of the following trials). It never blocks Record or anything else. The thresholds are a PROPOSAL for the group to
confirm. No measured number depends on it.
"""
import math
import re
import time
from datetime import datetime

from . import check_weight as check_mod
from . import cues, force as force_mod

MIN_CAMERA_FPS = 25.0              # the same limit as camera.health(): fewer pictures a second than this is amber
FRAME_SILENCE_S = 1.0              # no picture for this long: the camera has stopped
FORCE_SILENCE_S = 0.5              # no force sample for this long: the same limit as the program's own 'FORCE LOST'
FORCE_RATE_SHARE = 0.9             # fewer samples than this share of the nominal rate in the last second: amber
ZERO_NOISE_SECONDS = 2.0           # the zero noise is read over this long (a whole number of mains cycles)
ZERO_NOISE_AMBER_V = 0.010         # PROPOSAL (the group must confirm): above this spread, hum not counted, is amber; a third of the 30 mV at which
                                   # a zero is refused (force.MAX_STEADY_SD_V); about 5 N with the 2026-10-13 regression (0.0186 V per kg)
HUM_AMBER_V = force_mod.HUM_NOTE_V # 50 Hz hum at least this big (10 mV): amber, with the charger and outlet routine in the reason
PLACEHOLDER_NAME = re.compile(r"^Camera \d+$")   # the name camera.list_cameras gives when the real names cannot be read: no name at all
BEEP_NAMES = ("record", "ramp", "end")     # the three real cues the check plays (engine.beep_check)
ITEMS = (("camera", "Camera"), ("force", "Force connected"), ("zero_noise", "Zero noise"), ("check_weight", "Check weight"),
         ("beeps", "Beeps"), ("ruler", "Ruler"), ("lamp", "Lamp"))
STATES = ("green", "amber", "grey")
NOTE = ("green = ready, amber = look at it, grey = not checked; the thresholds are a proposal; the list only reads the equipment. Run it with the rig "
        "empty, before the participant arrives.")


# ------------------------------------------------------------------------------------------------ reading the equipment (a helper thread)
def zero_noise(src, seconds=ZERO_NOISE_SECONDS):
    """The spread of the signal over the last `seconds` with the mains hum taken out, and the hum amplitude, in volts and (with a slope) newtons.
    {'why': sentence} when there are too few samples. Reads only."""
    v = check_mod.read_window(src, seconds)
    if v is None:
        return {"why": f"Not enough force samples in the last {seconds:g} s."}
    hum = force_mod.mains_amplitude(v, src.rate)
    hum = hum if math.isfinite(hum) else 0.0
    noise = math.sqrt(max(float(v.std()) ** 2 - hum ** 2 / 2.0, 0.0))
    slope = src.calibration.n_per_v if src.calibration.has_slope else None
    return {"noise_v": noise, "hum_v": hum, "noise_n": None if slope is None else noise * slope, "hum_n": None if slope is None else hum * slope,
            "samples": int(v.size), "seconds": float(seconds)}


def gather(engine, camera_name="", force_skipped=False, now=None):
    """What the equipment says right now, as plain numbers (no judgement): the engine's camera and ruler, the force source, the check weight. Reads
    only; the window runs it on a helper thread."""
    facts = {"time": (now or datetime.now()).isoformat(timespec="seconds")}
    info = engine.camera_info or {}
    # the name: the one the driver gave (a camera opened through Media Foundation), else the one the window had when the camera was started
    facts["camera"] = {"started": bool(engine.camera_desc), "description": engine.camera_desc or "",
                       "name": info.get("capture_device_name") or camera_name or "", "matched_by_name": info.get("capture_matched_by_name"),
                       "kind": "file" if info.get("is_file") else info.get("source_kind") or "", "fps": engine.live_fps(),
                       "silent_s": engine.seconds_since_last_frame()}
    src = engine.force
    f = {"connected": src is not None, "skipped": bool(force_skipped)}
    if src is not None:
        t1 = time.perf_counter()
        _, v = src.samples_between(t1 - 1.0, t1 + 1.0)
        cal = src.calibration
        f.update(kind=src.kind, error=src.error, silent_s=src.seconds_since_last_sample(), samples_last_s=int(v.size), nominal_hz=float(src.rate),
                 zero_set=bool(cal.zero_set), has_slope=bool(cal.has_slope))
        facts["zero_noise"] = zero_noise(src) if cal.zero_set else {}
    facts["force"] = f
    cal = src.calibration if src is not None else None
    check = engine.check_weight
    state, age = check_mod.standing(check, cal, now)
    facts["check_weight"] = {"check": check, "state": state, "age_h": age}
    sc = engine.scale
    facts["ruler"] = {"configured": bool(sc.configured), "px_per_cm": sc.px_per_cm if sc.configured else None,
                      "length_cm": sc.length_cm if sc.configured else None, "problems": list(sc.problems()) if sc.configured else []}
    facts["beeps_silent"] = bool(cues.silent())
    return facts


# ------------------------------------------------------------------------------------------------ judging it (pure)
def _item(key, state, reason):
    return {"key": key, "title": dict(ITEMS)[key], "state": state, "reason": reason}


def _camera(c):
    if not c.get("started"):
        return "grey", "No camera is started."
    if c.get("kind") == "file":
        return "amber", "A video file is playing, not a live camera."
    if math.isinf(c.get("silent_s", 0)):
        return "amber", "No picture has arrived yet from the camera."
    if c.get("silent_s", 0) > FRAME_SILENCE_S:
        return "amber", f"No picture for {c['silent_s']:.1f} s: the camera has stopped."
    if c.get("kind") == "webcam" and (not c.get("name") or c.get("matched_by_name") is False or PLACEHOLDER_NAME.match(c["name"])):
        return "amber", "A camera is open, but its name is not known (it was opened by number, which can change between starts)."
    fps = c.get("fps")
    if fps is None:
        return "grey", "The picture rate is not measured yet: wait a few seconds."
    name = c.get("name") or c.get("description") or "camera"
    if fps < MIN_CAMERA_FPS:
        return "amber", f"{name}: only {fps:.1f} pictures a second (at least {MIN_CAMERA_FPS:g} wanted)."
    return "green", f"{name}: {fps:.1f} pictures a second."


def _force(f):
    if f.get("skipped") and not f.get("connected"):
        return "grey", "Marked 'No force today' (angles only)."
    if not f.get("connected"):
        return "amber", "The force is not connected (Force box: choose the source and press Connect)."
    if f.get("error"):
        return "amber", f["error"]
    if math.isinf(f.get("silent_s", 0)):
        return "amber", "No force sample has arrived yet."
    if f.get("silent_s", 0) > FORCE_SILENCE_S:
        return "amber", f"No force sample for {f['silent_s']:.1f} s."
    n, nominal = f.get("samples_last_s", 0), f.get("nominal_hz", 0)
    if n < FORCE_RATE_SHARE * nominal:
        return "amber", f"Only {n} samples in the last second (about {nominal:g} expected)."
    return "green", f"{f.get('kind', 'force')}: {n} samples in the last second."


def _zero_noise(f, z):
    if not f.get("connected") or f.get("error"):
        return "grey", "Needs the force connected and reading."
    if not f.get("zero_set"):
        return "grey", "The rig is not zeroed: hang nothing but the hook or strap, press Zero (Z), then check again."
    if "why" in z:
        return "grey", z["why"]
    n = z.get("noise_n")
    shown = f"{z['noise_v'] * 1000:.1f} mV" + ("" if n is None else f" ({n:.2f} N)")
    hum = f"{z['hum_v'] * 1000:.1f} mV" + ("" if z.get("hum_n") is None else f" ({z['hum_n']:.2f} N)")
    if z["hum_v"] >= HUM_AMBER_V:
        return "amber", (f"50 Hz hum {hum}, noise {shown} over {z['seconds']:g} s. The hum cancels out of a whole-cycle average but is still on the "
                         "signal: charge the laptop from the isolated outlet.")
    if z["noise_v"] > ZERO_NOISE_AMBER_V:
        return "amber", f"Noise {shown} over {z['seconds']:g} s (hum {hum}), above the {ZERO_NOISE_AMBER_V * 1000:g} mV proposed. Is something touching the rig?"
    return "green", f"Noise {shown} over {z['seconds']:g} s, hum {hum}."


def _check_weight(c):
    check, state = c.get("check"), c.get("state")
    if state == "none":
        return "grey", "No check weight this run: hang the known weight and press Check weight in the Force box."
    when = str(check.get("time", ""))[:16].replace("T", " ")
    what = f"{check['mass_kg']:g} kg at {when}: {check['error_pct']:+.2f} %"
    if state == "stale":
        return "amber", f"The last check is over {check_mod.VALID_FOR_HOURS:g} h old ({what})."
    if state == "calibration":
        return "amber", f"The last check ({what}) was made with another calibration than the one in use."
    if state == "zero":
        return "amber", f"The zero was set again after the last check ({what}): hang the known weight again."
    if state == "level":
        return "amber", f"The last check was {check['level'].upper()} ({what})."
    return "green", f"{what} (green)."


def _beeps(manual, silent):
    if silent:
        return "grey", "Sound is switched off on this computer (no screen, or LIMBTRACK_SILENT): the beeps cannot be checked here."
    if not manual.get("played"):
        return "grey", "Not played yet: press Play the beeps, then tick that you heard them."
    if manual.get("error"):
        return "amber", f"The sound system could not play a beep ({manual['error']}). Check the sound output and volume of this computer."
    names = list(manual.get("names") or [])
    if len(names) < len(BEEP_NAMES):
        return "amber", f"The sound system took only {len(names)} of the {len(BEEP_NAMES)} beeps."
    if not manual.get("heard"):
        return "amber", "The beeps played without error, but you have not ticked that you heard them."
    return "green", "Played without error and heard."


def _ruler(r):
    if not r.get("configured"):
        return "grey", "Not marked: mark the ruler in the picture (C)."
    if r.get("problems"):
        return "amber", f"{r['px_per_cm']:.2f} px/cm. {r['problems'][0]}"
    return "green", f"{r['px_per_cm']:.2f} px per cm ({r['length_cm']:g} cm ruler). The scale is fixed: mark it again if the camera or ruler moves."


def evaluate(facts, manual=None):
    """The list from the facts of `gather` and the operator's ticks `manual` ({'played', 'names', 'error', 'heard', 'lamp'}). Returns the saved
    block: time, overall state, counts, the items (state and reason each, in the order of ITEMS), the thresholds and a note. Pure."""
    manual = manual or {}
    f = facts.get("force", {})
    judged = {
        "camera": _camera(facts.get("camera", {})),
        "force": _force(f),
        "zero_noise": _zero_noise(f, facts.get("zero_noise") or {}),
        "check_weight": _check_weight(facts.get("check_weight", {"state": "none"})),
        "beeps": _beeps(manual, facts.get("beeps_silent", False)),
        "ruler": _ruler(facts.get("ruler", {})),
        "lamp": ("green", "Ticked: the lamp is on and the light is even.") if manual.get("lamp") else
                ("grey", "Not ticked: switch the lamp on, look at the picture, then tick."),
    }
    items = {k: {"state": s, "reason": r} for k, (s, r) in judged.items()}
    counts = {s: sum(1 for v in items.values() if v["state"] == s) for s in STATES}
    overall = "green" if counts["amber"] == 0 and counts["grey"] == 0 else "amber" if counts["amber"] else "grey"
    return {"time": facts.get("time") or datetime.now().isoformat(timespec="seconds"), "overall": overall, "all_green": overall == "green",
            "counts": counts, "items": items,
            "thresholds": {"min_camera_fps": MIN_CAMERA_FPS, "zero_noise_amber_v": ZERO_NOISE_AMBER_V, "hum_amber_v": HUM_AMBER_V,
                           "check_weight_amber_above_pct": check_mod.AMBER_ABOVE_PCT, "check_weight_red_above_pct": check_mod.RED_ABOVE_PCT,
                           "check_weight_valid_for_hours": check_mod.VALID_FOR_HOURS},
            "note": NOTE}


def rows(result):
    """[(key, title, state, reason)] of a result in the order of ITEMS."""
    return [(k, t, result["items"][k]["state"], result["items"][k]["reason"]) for k, t in ITEMS if k in result.get("items", {})]


def summary_line(result):
    c = result["counts"]
    parts = [f"{c['green']} of {len(ITEMS)} ready"]
    if c["amber"]:
        parts.append(f"{c['amber']} to look at")
    if c["grey"]:
        parts.append(f"{c['grey']} not checked")
    return ", ".join(parts) + "."


# ------------------------------------------------------------------------------------------------ the saved forms
def log_detail(result):
    """The session log line: the overall state, then every item as `key=state: reason`, joined by ' | '."""
    return f"overall={result['overall']} | " + " | ".join(f"{k}={s}: {r}" for k, _, s, r in rows(result))


def parse_log_detail(text):
    """The detail of a rig check line back as {key: (state, reason)} with 'overall' as (state, '') (for tests and a later look at the log)."""
    out = {}
    for part in str(text).split(" | "):
        if "=" not in part:
            continue
        k, rest = part.split("=", 1)
        state, _, reason = rest.partition(": ")
        out[k.strip()] = (state.strip(), reason.strip())
    return out
