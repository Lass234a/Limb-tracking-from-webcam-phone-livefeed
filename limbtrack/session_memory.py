"""What the program remembers between launches, so the set-up is not done again: a plain-text file next to the program
(`settings.json`, local to this computer and not in the repository).

Remembered: the camera (by NAME, because the numbers change between restarts), the test and target, tolerances, dot colour, which way
the participant faces, mirror, landmarks, force settings, the automatic MVIC and sound settings, the camera model, the
ruler length, which panel section was open, the window shape, and the participant code if the program was used in the last 12 hours.

NOT remembered, on purpose: the marked dots and the lock (the picture is different), the force zero (the rig hangs differently), the
ruler marks, the privacy crop and masks (a rectangle drawn for yesterday's picture could silently fail to cover a face today) and the
camera set-up TICKS (focus locked, zoom 1x ...: each trial's meta.json states them as facts about the camera, and a phone app that was
restarted has lost its focus lock; the ticks are made again every session).
A file that cannot be read is ignored: the program then starts with its defaults.
"""
import json
import os
import time
from pathlib import Path

FORMAT = 1
PARTICIPANT_VALID_S = 12 * 3600       # a participant code is offered again only if the program was used within this time


def read_all(path):
    """The whole settings file as a dict ({} when it is missing or unreadable)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_session(path):
    """The remembered session set-up ({} when there is none or it is from an unknown format)."""
    s = read_all(path).get("session")
    return s if isinstance(s, dict) and s.get("format") == FORMAT else {}


def write_part(path, **parts):
    """Add or replace top-level parts of the settings file (the others, such as recordings_dir, are kept). Never raises."""
    path = Path(path)
    data = read_all(path)
    data.update(parts)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def write_session(path, session):
    session = {**session, "format": FORMAT, "saved_at": time.time()}
    return write_part(path, session=session)


def participant_if_recent(session, now=None):
    """The remembered participant code, but only when the program was used within PARTICIPANT_VALID_S (a new day starts blank)."""
    saved = session.get("saved_at")
    code = session.get("participant")
    if not isinstance(code, str) or not isinstance(saved, (int, float)):
        return ""
    return code if 0 <= (now if now is not None else time.time()) - saved <= PARTICIPANT_VALID_S else ""


def autostart_allowed():
    """No camera or DAQ is opened by itself on a machine without a screen (the automatic tests) or when asked not to."""
    return not (os.environ.get("QT_QPA_PLATFORM") == "offscreen" or os.environ.get("LIMBTRACK_NO_AUTOSTART"))
