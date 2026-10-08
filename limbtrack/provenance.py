"""What produced a recording: code, libraries, computer and protocol file, for the methods section.

Everything here is looked up once per program run and never raises: a missing git, a ZIP download without
a .git folder or an unreadable file just gives None for that item. The code is also identified by a content
hash of limbtrack/*.py (line endings ignored), which does not need git.
"""
import functools
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=2.0,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - git missing, slow, or not a repository
        return None


def _version(name):
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=1)
def code_sha256():
    """Hash of the program's own source files (sorted, line endings normalised), independent of git."""
    h = hashlib.sha256()
    for p in sorted(PACKAGE.glob("*.py")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).digest())
    return h.hexdigest()


def protocol_sha256(path):
    """Hash of the protocol file's content as JSON (so line endings and spacing do not matter), or None."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=8)
def collect(protocol_path=None):
    """The provenance record for meta.json."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "code_sha256": code_sha256(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "opencv": _version("opencv-python") or _version("opencv-python-headless"),
        "pyside6": _version("PySide6"),
        "pyqtgraph": _version("pyqtgraph"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timezone": time.strftime("%z"),
        "protocol_sha256_canonical": protocol_sha256(protocol_path) if protocol_path else None,
    }
