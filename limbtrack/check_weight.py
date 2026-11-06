"""The check weight (0.35.0, plan feature F8, validity item V4): a known mass (20 to 50 kg) is hung on the rig before the session and the program
compares the force it reads with the force the mass must give (mass x 9.80665 N), so a drifting amplifier, a wrong slope or a loose connection is
seen BEFORE the participant arrives instead of in the review.

How it is used: Zero the rig with nothing loaded (the hook or strap that stays on the cell hanging, so only the mass counts), hang the mass, press
*Check weight* in the Force box. The program averages the zeroed force over 3 seconds and shows the error in percent as green, amber or red. The result
goes to the session log and into the `meta.json` of every following trial (`check_weight`), and a soft reminder (never a refusal) comes up when no check was
made in the last VALID_FOR_HOURS hours while the force is connected, or when the last check was amber or red.

Nothing here changes the force, the zero or the calibration, and no measured number depends on it: the check only READS the samples the force source
already has. The averaging window is a whole number of 50 Hz mains cycles (3.000 s = 150 cycles at 1000 samples a second), so the hum cancels out of the
mean exactly (the rule of force.py: a multiple of 20 ms). It is NOT a single sample.

The tolerances below are a PROPOSAL; the group has to confirm them. The plan says the load cell regression is good to about 1 % inside 0-250 kg; the
single-run scatter around the regression of 2026-10-13 is 0.15 kg, i.e. 0.8 % at 20 kg and 0.3 % at 50 kg, so a heavier mass is the fairer check.

A check is only an independent check of the regression when the slope in use came from the regression. If the operator calibrated with a known weight in
this session (force.py `calibrate`), the slope was made from a known mass and a check with the same mass agrees by construction; the result says so.
"""
import math
import time
from datetime import datetime

from . import force as force_mod
from .loadcell import GRAVITY

MASS_RANGE_KG = (20.0, 50.0)       # the known masses the study uses (a plate or a calibrated weight that one person can hang)
AVERAGE_S = 3.0                    # the force is averaged over this long (a whole number of 50 Hz cycles: the hum averages out exactly)
MIN_SAMPLE_SHARE = 0.9             # fewer than this share of the expected samples in the window: wait and try again
AMBER_ABOVE_PCT = 1.0              # PROPOSAL (the group must confirm): an error above this is amber ...
RED_ABOVE_PCT = 2.0                # ... and above this red
VALID_FOR_HOURS = 12.0             # PROPOSAL: a check older than this is not enough for today (the same idea as the participant code window)
DECIMALS = 3
ZERO_TOLERANCE_V = 2e-6            # the stored zero_v is rounded to 6 decimals: a zero closer than this to it is the same zero
COLUMNS = ["check_weight_error_pct", "check_weight_time"]       # appended to the session summary and to the AnyBody export (blank without a check)
LEVELS = ("green", "amber", "red")
NOTE = ("error_pct = (measured - expected) / expected x 100, with expected = mass_kg x 9.80665 N and measured = the mean of the zeroed force over "
        "seconds s (a whole number of 50 Hz cycles); level: green up to amber_above_pct, amber up to red_above_pct, red above; ok is true for green "
        "only; zero_v is the zero in use. A mass typed by the operator: the program cannot weigh it.")


class CheckWeightError(ValueError):
    """The check cannot be made; the message says why in plain English."""


def parse_mass(text):
    """What was typed -> (kg or None, problem or None). Empty is fine (None, None). A comma may be the decimal mark and 'kg' may follow the number.
    A number outside MASS_RANGE_KG or something that is not a number gives (None, sentence): it is not kept."""
    s = str(text or "").strip().lower().replace(",", ".")
    if s.endswith("kg"):
        s = s[:-2].strip()
    if not s:
        return None, None
    try:
        v = float(s)
    except ValueError:
        return None, f"Check weight: '{str(text).strip()}' is not a number."
    problem = mass_problem(v)
    return (None, problem) if problem else (round(v, 2), None)


def mass_problem(kg):
    """None when `kg` is a usable known mass, else the sentence that says why not."""
    lo, hi = MASS_RANGE_KG
    if isinstance(kg, bool) or not isinstance(kg, (int, float)) or not math.isfinite(kg):
        return f"Check weight: {kg!r} is not a number."
    if not lo <= kg <= hi:
        return f"Check weight: {kg:g} kg is outside the {lo:g} to {hi:g} kg the check is made for. Typed the wrong unit?"
    return None


def expected_n(mass_kg):
    """The force a hanging mass gives, in newtons (standard gravity)."""
    return float(mass_kg) * GRAVITY


def level_of(error_pct):
    """'green', 'amber' or 'red' for an error in percent (the sign does not matter)."""
    e = abs(float(error_pct))
    return "green" if e <= AMBER_ABOVE_PCT else "amber" if e <= RED_ABOVE_PCT else "red"


def read_window(src, seconds=AVERAGE_S):
    """The newest samples of `src` over `seconds`, cut to a whole number of mains cycles (newest kept): the volts, or None when fewer than
    MIN_SAMPLE_SHARE of the expected samples are there. Reads only."""
    now = time.perf_counter()
    _, v = src.samples_between(now - seconds, now + 1.0)
    if v.size < MIN_SAMPLE_SHARE * seconds * src.rate or v.size < 4:
        return None
    return force_mod.whole_mains_cycles(v, src.rate)


def measure(src, mass_kg, seconds=AVERAGE_S, now=None):
    """Make the check. Returns (result dict, None) or (None, why): why is a plain-English reason and nothing was recorded.

    Needs a source that delivers samples, a slope (the load cell's calibration) and a zero; the signal must be steady (the same drift and noise
    limits as the zero of force.py: a swinging weight is refused, not averaged)."""
    problem = mass_problem(mass_kg)
    if problem:
        return None, problem
    if src is None:
        return None, "The force is not connected: connect it (Force box), zero the rig, hang the weight, then check."
    if src.error:
        return None, src.error
    if src.frozen:
        return None, "Stop recording first: the check weight is made between recordings."
    cal = src.calibration
    if not cal.has_slope:
        return None, "There is no load cell calibration (newtons per volt) to check: only volts are read. Choose a calibration first."
    if not cal.zero_set:
        return None, "The rig is not zeroed. Hang nothing but the hook or strap, press Zero (Z), then hang the weight and check."
    v = read_window(src, seconds)
    if v is None:
        return None, f"Not enough force samples in the last {seconds:g} s yet. Wait a moment and try again."
    hum = force_mod.mains_amplitude(v, src.rate)
    hum = hum if math.isfinite(hum) else 0.0
    noise = math.sqrt(max(float(v.std()) ** 2 - hum ** 2 / 2.0, 0.0))
    half = v.size // 2
    drift = float(v[half:].mean() - v[:half].mean())
    if abs(drift) > force_mod.MAX_DRIFT_V:
        return None, (f"The signal is not steady: it moved {drift * 1000:+.0f} mV during the {seconds:g} s (limit {force_mod.MAX_DRIFT_V * 1000:.0f} mV). "
                      "Let the weight stop swinging and check again.")
    if noise > force_mod.MAX_STEADY_SD_V:
        return None, (f"The signal is not steady enough: it wobbles {noise * 1000:.0f} mV (limit {force_mod.MAX_STEADY_SD_V * 1000:.0f} mV, the 50 Hz hum "
                      "not counted). Let the weight hang still and check again.")
    expected = expected_n(mass_kg)
    measured = float(cal.to_newton(float(v.mean())))               # (the mean of the volts: the conversion is linear)
    error = 100.0 * (measured - expected) / expected
    level = level_of(error)
    note = ""
    if not cal.regression_id:
        note = (f"The slope in use ({cal.method}) did not come from the load cell's regression, so this only shows that the weight and the slope agree; "
                "it does not check the regression.")
    return {"mass_kg": round(float(mass_kg), 2), "expected_n": round(expected, DECIMALS), "measured_n": round(measured, DECIMALS),
            "error_pct": round(error, DECIMALS), "level": level, "ok": level == "green",
            "time": (now or datetime.now()).isoformat(timespec="seconds"),
            "zero_v": round(float(cal.offset_v), 6), "zero_set_at": cal.when, "calibration_id": cal.regression_id, "calibration_method": cal.method,
            "n_per_v": float(cal.n_per_v), "seconds": float(seconds), "samples": int(v.size), "noise_v": round(noise, 6), "hum_v": round(hum, 6),
            "drift_v": round(drift, 6), "amber_above_pct": AMBER_ABOVE_PCT, "red_above_pct": RED_ABOVE_PCT, "note": note}, None


def clean(result):
    """A check result as the engine keeps it (a copy), or ValueError for something that is not one. None clears."""
    if result is None:
        return None
    if not isinstance(result, dict):
        raise ValueError("A check weight result must be a dict.")
    for k in ("mass_kg", "expected_n", "measured_n", "error_pct"):
        v = result.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"A check weight result needs a number for {k}.")
    if result.get("level") not in LEVELS:
        raise ValueError("A check weight result needs a level (green, amber or red).")
    if not isinstance(result.get("time"), str):
        raise ValueError("A check weight result needs its time.")
    return dict(result)


def same_calibration(check, cal):
    """Whether the calibration `cal` (a ForceCalibration, or None) is the one the check was made with (same regression, same slope)."""
    if not isinstance(check, dict) or cal is None or not cal.has_slope:
        return False
    old = check.get("n_per_v")
    return (check.get("calibration_id") or "") == (cal.regression_id or "") and isinstance(old, (int, float)) and math.isclose(old, cal.n_per_v, rel_tol=1e-9)


def same_zero(check, cal):
    """Whether the zero in use (`cal`) is the one the check was made against: True / False, or None when it cannot be told (no calibration, or a
    check without its zero). A re-zero, or a reconnect that has not been zeroed again, makes it False: the newtons then rest on another zero."""
    if not isinstance(check, dict) or cal is None:
        return None
    old = check.get("zero_v")
    if isinstance(old, bool) or not isinstance(old, (int, float)):
        return None
    return bool(cal.zero_set) and abs(cal.offset_v - old) <= ZERO_TOLERANCE_V


def meta_block(check, cal):
    """What meta.json holds as `check_weight`: the latest check of this program run, with whether the calibration is still the one it was made with
    (None: the trial has no force) and, from 0.35.0, whether the zero is still the one it was made against (`same_zero`). None when no check was made."""
    if not isinstance(check, dict):
        return None
    block = dict(check)
    block["same_calibration"] = None if cal is None else same_calibration(check, cal)
    block["note_on_values"] = NOTE
    block["same_zero"] = same_zero(check, cal)                       # (last: the keys before it keep their place)
    return block


def age_hours(check, now=None):
    """Hours since the check was made, or None when its time cannot be read."""
    try:
        made = datetime.fromisoformat(str(check.get("time")))
    except (ValueError, AttributeError, TypeError):
        return None
    return (((now or datetime.now()) - made).total_seconds()) / 3600.0


def standing(check, cal, now=None):
    """('ok' | 'none' | 'stale' | 'level' | 'calibration' | 'zero', age in hours or None) for the latest check against the calibration and the zero in
    use. 'ok' = made within VALID_FOR_HOURS, green, with the same calibration and the same zero."""
    if not isinstance(check, dict):
        return "none", None
    age = age_hours(check, now)
    if age is None or age < -0.05 or age > VALID_FOR_HOURS:
        return "stale", age
    if not same_calibration(check, cal):
        return "calibration", age
    if same_zero(check, cal) is False:
        return "zero", age
    if check.get("level") != "green":
        return "level", age
    return "ok", age


def reminder(check, cal, connected, now=None):
    """The one-line soft reminder (never a refusal) or '' : only while the force is connected; no check in the last VALID_FOR_HOURS hours, one
    made with another calibration, or the last one amber or red. The sentences carry no changing numbers, so a 'Noted' answer holds until the
    reason changes."""
    if not connected:
        return ""
    state, _ = standing(check, cal, now)
    if state == "none":
        return (f"Check weight: none made in the last {VALID_FOR_HOURS:g} h. Zero the rig, hang the known weight and press Check weight in the Force box.")
    if state == "stale":
        return f"Check weight: the last one is over {VALID_FOR_HOURS:g} h old. Hang the known weight again and press Check weight."
    if state == "calibration":
        return "Check weight: the last one was made with another calibration than the one in use. Hang the known weight again and check."
    if state == "zero":
        return "Check weight: the zero was set again since the last check. Hang the known weight again and check."
    if state == "level":
        e = check["error_pct"]
        word = "AMBER" if check.get("level") == "amber" else "RED"
        return (f"Check weight: the last check was {word} ({e:+.1f} % against {check['mass_kg']:g} kg). Hang the weight again and check; "
                "if it stays so, the newtons of today may be off.")
    return ""


def summary_text(check, cal=None, now=None):
    """A sentence for the Force box about the latest check ('' when none was made)."""
    if not isinstance(check, dict):
        return ""
    when = str(check.get("time", ""))[11:16] or "?"
    text = (f"Check weight {check['mass_kg']:g} kg at {when}: expected {check['expected_n']:.1f} N, read {check['measured_n']:.1f} N, "
            f"error {check['error_pct']:+.2f} % ({check['level'].upper()}).")
    if cal is not None and not same_calibration(check, cal):
        text += " The calibration changed since: check again."
    elif cal is not None and same_zero(check, cal) is False:
        text += " The zero was set again since: check again."
    if check.get("note"):
        text += " " + check["note"]
    return text


# ------------------------------------------------------------------------------------------------ the session log line
LOG_KEYS = ("mass_kg", "expected_n", "measured_n", "error_pct", "level", "ok", "zero_v", "calibration_id", "calibration_method", "time")


def log_detail(check):
    """The text of the session log line: `key=value` pairs joined by '; ' (the log's own time column is the time of the line)."""
    return "; ".join(f"{k}={'yes' if check[k] is True else 'no' if check[k] is False else check[k]}" for k in LOG_KEYS if k in check)


def parse_log_detail(text):
    """The `key=value` pairs of a log line's detail back as a dict of strings (for tests and for a later look at the session log)."""
    out = {}
    for part in str(text).split("; "):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ------------------------------------------------------------------------------------------------ the export columns
def export_cells(meta):
    """The export row's cells (COLUMNS) from a trial's meta.json; both blank for a trial without a check (an older trial, or none made that run)
    or with a damaged block (never a made-up number)."""
    out = {k: "" for k in COLUMNS}
    block = meta.get("check_weight")
    if not isinstance(block, dict):
        return out
    e = block.get("error_pct")
    if isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e):
        out["check_weight_error_pct"] = e
    out["check_weight_time"] = str(block.get("time") or "")
    return out
