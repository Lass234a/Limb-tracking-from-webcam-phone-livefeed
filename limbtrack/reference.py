"""Reference readings: what a person read off a goniometer or an inclinometer, typed in next to the app's own angle (0.16.0).

The app's angle comes from dots on skin. The only way to know how far it is from the angle a physio would read is to read it too,
at the same moment, and to keep the two numbers side by side. This module is the pure part: parsing what is typed, what is stored,
how a reading is compared with the app's angle, and the agreement statistics over many trials. Nothing here changes a measured number.

* A typed reading is a number a person wrote down, never a measurement of this program. Blank means "not measured"; a typed 0 is a
  reading of 0 degrees.
* The comparison has to be like for like. The goniometer can only be read at rest, so the app's angle is taken at the moment the
  operator says "reading taken now" (the unsmoothed angle averaged over the 0.2 s before), not at the hold of the contraction (the
  knee moved about 5.8 deg between rest and hold in the first force trials). If no moment was stamped, the first second of the
  recording stands in, and the table says so.
* What is saved with a trial (`meta.json` -> `reference_readings`) is never rewritten. Readings added or changed later, in Review,
  go to a small file next to the trial, `<trial>_reference.json`, which wins over the saved ones for the keys it has.
"""
import csv
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

from .session_memory import read_all as _read_json_dict

INSTRUMENTS = {"goniometer": "Goniometer", "inclinometer": "Digital inclinometer", "other": "Other instrument"}
RESOLUTIONS_DEG = (0.1, 0.5, 1.0, 2.0, 5.0)          # what the instrument can be read to
DEFAULT_INSTRUMENT, DEFAULT_RESOLUTION_DEG = "goniometer", 1.0
VALUE_RANGE_DEG = (-180.0, 360.0)
MAX_READINGS = 5                                     # readings of one angle in one posture (repeat readings are averaged)
APP_BASIS_STAMP = "the app's unsmoothed angle, averaged over the 0.2 s before 'reading taken now'"
APP_BASIS_FIRST_SECOND = "the app's unsmoothed angle in the first second of the recording (no moment was stamped)"
ACCEPT_BIAS_DEG = 0.5                                # proposed acceptance (SETUP_AND_VALIDATION.md 4A; the user's decision): bias within this
ACCEPT_LIMIT_DEG = 1.0                               # ... and the limits of agreement within this
LIMITS_Z = 1.96                                      # 95 % limits of agreement: bias +- 1.96 SD of the differences
REST_SECONDS = 1.0


class ReadingError(ValueError):
    """What was typed cannot be used as readings; the message says why in plain English."""


def parse_values(text):
    """The readings typed in one box, in degrees. Separate several readings with a space or a semicolon; a comma is the decimal mark
    ("88,5 90" is 88.5 and 90; a comma left at the end of a number is ignored). Empty text is no reading."""
    out = []
    for tok in re.split(r"[\s;]+", str(text).strip()):
        tok = tok.strip(",")
        if not tok:
            continue
        try:
            v = float(tok.replace(",", "."))
        except ValueError:
            raise ReadingError(f'"{tok}" is not a number (readings are in degrees; separate several with a space)') from None
        if not math.isfinite(v) or not VALUE_RANGE_DEG[0] <= v <= VALUE_RANGE_DEG[1]:
            raise ReadingError(f"{tok} is not an angle (between {VALUE_RANGE_DEG[0]:g} and {VALUE_RANGE_DEG[1]:g} degrees)")
        out.append(v)
    if len(out) > MAX_READINGS:
        raise ReadingError(f"at most {MAX_READINGS} readings of one angle (these are {len(out)})")
    return out


def format_values(values):
    """Readings as they are typed back into a box: '88 90 89.5'."""
    return " ".join(f"{v:g}" for v in values)


class ReferenceReadings:
    """The readings of the posture the participant is in now, before they are saved with a trial."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.instrument = DEFAULT_INSTRUMENT
        self.resolution_deg = DEFAULT_RESOLUTION_DEG
        self.values = {}                  # key -> list of readings (key: an angle name, or an extra such as 'hip_flexion')
        self.stamp = None                 # {'taken_at': iso time, 'app_deg': {angle name: value or None}}
        self.position = None              # (test key, target, participant code) these were entered for; participant None = typed before a code
        self.typed_at = None              # when the typed numbers last changed (iso, microseconds): the readings' identity for the agreement

    def has_data(self):
        return bool(self.values) or self.stamp is not None

    def entered_for(self):
        """The position (and participant) the readings were entered for, as it is written into meta.json, or None."""
        if self.position is None:
            return None
        key, target, who = self.position
        return {"test_key": key, "target_deg": target, "participant": who}

    def mismatch(self, here):
        """Why these readings do not belong to `here` (test key, target, participant code): None when they do (or there are none),
        'participant' when only the participant differs, 'position' otherwise. Readings typed while there was no participant code
        (participant None) belong to whoever is recorded first."""
        if self.position is None:
            return None
        key, target, who = self.position
        if (key, target) != tuple(here[:2]):
            return "position"
        return None if who is None or who == here[2] else "participant"

    def claim(self, here, bind=False):
        """The readings now belong to `here`. They take its participant code once there is one (an earlier code is kept), and with
        `bind` (a recording starts) whatever it is: from then on they are that participant's only."""
        kept = self.position[2] if self.position is not None else None
        self.position = (here[0], here[1], here[2] if bind else (here[2] or kept))

    def set_values(self, key, values):
        new = [float(v) for v in values or []]
        if new == self.values.get(key, []):
            return                                        # the same numbers: still the same readings (the same pair in the agreement)
        if new:
            self.values[key] = new
        else:
            self.values.pop(key, None)
        self.typed_at = datetime.now().isoformat(timespec="microseconds")

    def to_meta(self, titles=None):
        """What goes into meta.json, or None when there is nothing. `titles`: key -> the name shown for it (so Review can label an extra
        such as the typed hip flexion of a frontal trial, which is not one of the trial's angles)."""
        if not self.has_data():
            return None
        return {
            "instrument": self.instrument,
            "resolution_deg": self.resolution_deg,
            "titles": dict(titles or {}),
            "readings": {k: _reading(v) for k, v in self.values.items()},
            "stamped_at": (self.stamp or {}).get("taken_at"),
            "typed_at": self.typed_at,
            "app_at_reading_deg": ({k: (None if v is None else round(float(v), 3)) for k, v in self.stamp["app_deg"].items()}
                                   if self.stamp else None),
            "app_basis": APP_BASIS_STAMP if self.stamp else None,
            "entered_for": self.entered_for(),
            "note": "typed by the operator from a hand instrument; blank = not measured. Never a measurement of this program.",
        }


def _reading(values):
    """One key's readings as they are saved: every value and the mean."""
    return {"values_deg": [float(x) for x in values], "mean_deg": round(statistics.fmean(values), 4)}


# ---------------------------------------------------------------------------------------------------------------- the side file
def sidecar_path(meta_path):
    """`<trial>_reference.json` next to `<trial>_meta.json`."""
    p = Path(meta_path)
    return p.with_name(p.name[: -len("_meta.json")] + "_reference.json")


def read_sidecar(meta_path):
    """The readings added or changed later in Review, or {} (none, or unreadable)."""
    return _read_json_dict(sidecar_path(meta_path))


def write_sidecar(meta_path, instrument, resolution_deg, values_by_key, note="", recorded=None):
    """Save readings next to the trial (atomically). `values_by_key`: key -> list of readings; an empty list is 'not measured'.
    `recorded`: what was saved with the trial (meta.json -> reference_readings), or None. Only what differs from it goes into the side
    file: a reading saved with the trial that is still the same (same numbers, same instrument and resolution) is left out, so it stays
    'recorded with the trial' and still counts once for the two trials of its position; an empty list for one of them is written as an
    explicit 'removed in Review', so that the reading saved with the trial no longer counts. Returns the path. The trial's own
    meta.json is never touched."""
    path = sidecar_path(meta_path)
    saved = recorded or {}
    saved_readings = saved.get("readings") or {}
    same_instrument = (instrument, resolution_deg) == (saved.get("instrument"), saved.get("resolution_deg"))
    readings = {}
    for k, v in values_by_key.items():
        was = (saved_readings.get(k) or {}).get("values_deg")
        if v and not (same_instrument and was and format_values(v) == format_values(was)):
            readings[k] = _reading(v)
        elif not v and k in saved_readings:
            readings[k] = {"values_deg": [], "mean_deg": None}
    data = {
        "instrument": instrument, "resolution_deg": resolution_deg,
        "readings": readings,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "note": note or "readings added or changed in Review; they win over the ones saved with the trial",
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)                # no half-written file is left behind; the side file is as it was
        except OSError:
            pass
        raise
    return path


# ---------------------------------------------------------------------------------------------------------------- comparing
def merged_readings(meta, sidecar=None):
    """key -> {values_deg, mean_deg, instrument, resolution_deg, source} for one trial. The side file wins, key by key."""
    out = {}
    saved = meta.get("reference_readings") or {}
    for k, r in (saved.get("readings") or {}).items():
        out[k] = {**r, "instrument": saved.get("instrument"), "resolution_deg": saved.get("resolution_deg"), "source": "recorded with the trial"}
    side = sidecar or {}
    for k, r in (side.get("readings") or {}).items():
        if not r.get("values_deg"):
            out.pop(k, None)                                  # removed in Review
            continue
        out[k] = {**r, "instrument": side.get("instrument"), "resolution_deg": side.get("resolution_deg"), "source": "added later in Review"}
    return out


def first_second_means(data_csv, names, seconds=REST_SECONDS):
    """name -> the mean unsmoothed angle of the first `seconds` of a recording (None when there is none). Reads the CSV only."""
    sums = {n: [] for n in names}
    try:
        with open(data_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    if float(row["t_s"]) > seconds:
                        break
                except (KeyError, ValueError):
                    continue
                for n in names:
                    try:
                        sums[n].append(float(row[f"{n}_deg"]))
                    except (KeyError, ValueError):
                        pass
    except OSError:
        pass
    return {n: (statistics.fmean(v) if v else None) for n, v in sums.items()}


def comparison(meta, sidecar=None, data_csv=None, first_second=None):
    """One row per reading of a trial: the typed reference, the app's angle at the same moment, and the difference (app - reference).

    The app's angle is the one stamped at 'reading taken now' when there is one; otherwise the first second of the recording:
    `first_second` (angle name -> mean or None) when the caller already has the trial's angles (Review), else read from `data_csv`.
    Rows for an extra such as the typed hip flexion of a frontal trial have no app angle."""
    titles = {a["name"]: a.get("label") or a["name"] for a in meta.get("angles", [])}
    saved = meta.get("reference_readings") or {}
    extras = {**(meta.get("reference_extras") or {}), **(saved.get("titles") or {})}       # names of readings that are not angles
    readings = merged_readings(meta, sidecar)
    stamped = saved.get("app_at_reading_deg") or {}
    fallback = None
    rows = []
    for key, r in readings.items():
        app, basis = None, None
        if key in titles:
            if stamped.get(key) is not None:
                app, basis = stamped[key], APP_BASIS_STAMP
            elif first_second is not None or data_csv is not None:
                if fallback is None:
                    fallback = first_second if first_second is not None else first_second_means(data_csv, list(titles))
                app, basis = fallback.get(key), (APP_BASIS_FIRST_SECOND if fallback.get(key) is not None else None)
        mean = r.get("mean_deg")
        rows.append({
            "key": key, "title": titles.get(key) or extras.get(key) or key, "reference_deg": mean, "n_readings": len(r.get("values_deg") or []),
            "instrument": r.get("instrument"), "resolution_deg": r.get("resolution_deg"), "source": r.get("source"),
            "app_deg": app, "app_basis": basis,
            "difference_deg": (app - mean) if app is not None and mean is not None else None,
        })
    return rows


def collect_rows(folder):
    """Every reading of every trial under `folder`, with the trial's identity: a list of dicts (see `comparison`) plus 'trial',
    'participant', 'test_key', 'target_deg', 'pair_id'. Reads only meta.json, the side file and, for the fallback, the CSV."""
    rows = []
    for meta_path in sorted(Path(folder).rglob("*_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sidecar = read_sidecar(meta_path)
        if not ((meta.get("reference_readings") or {}).get("readings") or sidecar.get("readings")):
            continue
        csv_name = (meta.get("files") or {}).get("data.csv")
        data_csv = meta_path.parent / csv_name if csv_name else None
        saved = meta.get("reference_readings") or {}
        # which typing (and which 'reading taken now') the readings come from; trials before 0.17.1 have no typed_at, and without a
        # stamp each of them is then a pair of its own, as before
        stamp, typed = saved.get("stamped_at"), saved.get("typed_at")
        same = f"{stamp}|{typed}" if (stamp or typed) else meta_path.name
        for r in comparison(meta, sidecar, data_csv):
            r.update(trial=meta_path.name[: -len("_meta.json")], participant=meta.get("participant"), test_key=meta.get("test_key"),
                     target_deg=meta.get("target_deg"),
                     # two trials of one position share one typed reading (and one stamp, if any): they are ONE pair, not two
                     pair_id=f"{meta.get('participant')}|{meta.get('test_key')}|{meta.get('target_deg')}|{same}|{r['key']}"
                     if r["source"] == "recorded with the trial" else f"{meta_path.name}|{r['key']}")
            rows.append(r)
    return rows


def agreement(rows):
    """Per angle: how far the app is from the hand instrument. `rows` as from `collect_rows`; only rows with both numbers count, and
    a reading shared by two trials of one position is counted once. Returns {key: stats}; stats has n, bias_deg (mean of app -
    reference), sd_deg, limits_deg (bias -+ 1.96 SD), mean_abs_deg, max_abs_deg, and whether the proposed acceptance is met."""
    by_key = {}
    seen = set()
    for r in rows:
        if r.get("difference_deg") is None or r.get("reference_deg") is None or r.get("app_deg") is None:
            continue
        if r["pair_id"] in seen:
            continue
        seen.add(r["pair_id"])
        by_key.setdefault(r["key"], []).append(r)
    out = {}
    for key, rs in by_key.items():
        d = [r["difference_deg"] for r in rs]
        bias = statistics.fmean(d)
        sd = statistics.stdev(d) if len(d) > 1 else math.nan
        lim = (bias - LIMITS_Z * sd, bias + LIMITS_Z * sd) if len(d) > 1 else (math.nan, math.nan)
        out[key] = {
            "key": key, "title": rs[0]["title"], "n": len(d), "bias_deg": bias, "sd_deg": sd, "limits_deg": lim,
            "mean_abs_deg": statistics.fmean(abs(x) for x in d), "max_abs_deg": max(abs(x) for x in d),
            "instruments": sorted({str(r["instrument"]) for r in rs}),
            "acceptance_met": (abs(bias) <= ACCEPT_BIAS_DEG and max(abs(lim[0]), abs(lim[1])) <= ACCEPT_LIMIT_DEG) if len(d) > 1 else None,
            "pairs": [(r["reference_deg"], r["app_deg"]) for r in rs],
        }
    return out


def agreement_text(stats):
    """The agreement table in plain English, for the Review window."""
    if not stats:
        return "No trial has a reference reading yet (type the goniometer or inclinometer reading in the Live tab, or add it in Review)."
    lines = ["Agreement of the app's angle with the hand instrument (app minus reference), one line per angle:", ""]
    for s in stats.values():
        lo, hi = s["limits_deg"]
        lines.append(f"{s['title']}: n = {s['n']}, average difference {s['bias_deg']:+.1f} deg, spread (SD) "
                     + ("n/a" if math.isnan(s["sd_deg"]) else f"{s['sd_deg']:.1f} deg")
                     + ("" if math.isnan(lo) else f", 95 % of differences between {lo:+.1f} and {hi:+.1f} deg")
                     + f", largest {s['max_abs_deg']:.1f} deg  [{', '.join(s['instruments'])}]")
    lines += ["",
              f"Proposed acceptance (SETUP_AND_VALIDATION.md 4A, your decision): average difference within {ACCEPT_BIAS_DEG:g} deg and 95 % of "
              f"differences within +-{ACCEPT_LIMIT_DEG:g} deg. This is the pair of instruments, not the app alone: a handheld goniometer is read "
              "to about a degree and has an error of its own of a few degrees, so it cannot show agreement to +-1 deg; use a jig with a digital "
              "inclinometer for that claim."]
    for s in stats.values():
        verdict = {True: "met", False: "not met", None: "cannot be told (one pair only)"}[s["acceptance_met"]]
        lines.append(f"  {s['title']}: {verdict}")
    return "\n".join(lines)


COLUMNS = ["trial", "participant", "test_key", "target_deg", "angle", "reference_deg", "n_readings", "instrument", "resolution_deg",
           "reference_source", "app_deg", "app_basis", "difference_app_minus_reference_deg", "same_pair_as_another_trial"]


def write_csv(rows, path):
    """All reference rows of a folder as a table (one row per reading). Returns the path."""
    def f(v, nd=3):
        return "" if v is None else round(v, nd) if isinstance(v, float) else v
    firsts = set()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            dup = r["pair_id"] in firsts
            firsts.add(r["pair_id"])
            w.writerow([r["trial"], r["participant"], r["test_key"], f(r["target_deg"]), r["title"], f(r["reference_deg"]), r["n_readings"],
                        r["instrument"], f(r["resolution_deg"]), r["source"], f(r["app_deg"]), r["app_basis"] or "",
                        f(r["difference_deg"]), 1 if dup else 0])
    return Path(path)
