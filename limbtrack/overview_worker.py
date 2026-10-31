"""A worker process of the participant overview (0.32.0): works out the overview rows of trials, one trial at a time, in a process of its
own, so a participant with sixty trials is read by several processes at once and a crash while reading one trial cannot take the window
down. Run as `python -m limbtrack.overview_worker --row-fn limbtrack.review:trial_row`; the window (review_overview.OverviewPage) starts and
stops it.

The talk is lines of JSON (ASCII only) over stdin / stdout, one line each way per trial:

    worker -> window   {"ready": true, "pid": 123, "memory_mb": 91}                      once, when the row function is imported
                       {"ready": false, "error": "..."}                                  it could not be imported: the worker exits
    window -> worker   {"id": 7, "path": "...\\P02_..._meta.json", "smooth": 0.1, "window": 0.5, "source": "original"}
    worker -> window   {"id": 7, "row": {...} or null, "error": "text"}                   the answer of `row_fn(path, smooth, window, source)`
    window -> worker   {"cmd": "quit"}                                                  (closing its stdin does the same: a window that dies
                       {"bye": true, "peak_mb": 104}                                     takes no worker with it)

It does not use multiprocessing: on Windows that starts a new interpreter that imports the launching script again, and the program's
launcher (Start.bat -> pythonw -m limbtrack) has no way to tell a worker from a second window. A plain `python -m` command has no such problem.
The module imports nothing heavy itself; the row function brings in what it needs (limbtrack.review: Qt, OpenCV, numpy: about two seconds).
"""
import importlib
import json
import os
import sys

DEFAULT_ROW_FN = "limbtrack.review:trial_row"
ENV_WORKERS = "LIMBTRACK_OVERVIEW_WORKERS"        # a whole number: the most worker processes to use; 0 = none (the numbers are worked out in the window)
MODULE = "limbtrack.overview_worker"


def spec_for(row_fn):
    """'module:name' of a module-level function a worker can import by itself, or None (a lambda, a method, a function made on the fly, or one
    in a script run as __main__: those cannot be found again in another process)."""
    mod, name = getattr(row_fn, "__module__", None), getattr(row_fn, "__qualname__", None)
    if not mod or not name or mod == "__main__" or "<" in name or "." in name:
        return None
    try:
        return f"{mod}:{name}" if getattr(importlib.import_module(mod), name, None) is row_fn else None
    except Exception:  # noqa: BLE001 - a module that cannot be imported here cannot be found there either
        return None


def configured_workers(default):
    """The most worker processes to use: `default`, unless LIMBTRACK_OVERVIEW_WORKERS says otherwise (0 or a bad value for 'none' / 'default')."""
    raw = os.environ.get(ENV_WORKERS, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def python_command():
    """The interpreter that starts a worker: the one running the window. Tried from pythonw.exe (how Start.bat starts the program): the
    pipes work and no console window opens; python.exe started from there would open a black console window for every worker."""
    return sys.executable


def processes_allowed():
    """False in a bundled program (sys.executable would start the whole program again, not a worker) or when the user switched them off."""
    return not getattr(sys, "frozen", False) and configured_workers(1) > 0


def resolve(spec):
    """The function a 'module:name' spec names."""
    mod, _, name = spec.partition(":")
    fn = getattr(importlib.import_module(mod), name)
    if not callable(fn):
        raise TypeError(f"{spec} is not a function")
    return fn


def _memory_mb(peak=False):
    """This process's memory in MB (working set, or its peak), 0 when Windows does not say (only used to report)."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(Counters), wt.DWORD]
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return 0
        return round((c.PeakWorkingSetSize if peak else c.WorkingSetSize) / 1e6)
    except Exception:  # noqa: BLE001
        return 0


def _plain(o):
    """json's `default`: numpy numbers and arrays, paths and anything else a row might hold, as plain JSON."""
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def dumps(message):
    """One protocol line (ASCII, no newline inside)."""
    return json.dumps(message, default=_plain, ensure_ascii=True, separators=(",", ":"))


def serve(row_fn, lines, send):
    """Answer every job line of `lines` with `send(message)` until a quit line or the end of the lines. A trial whose row function raises
    is answered with the error text, not left unanswered: one bad trial never stops the others."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        if msg.get("cmd") == "quit":
            break
        reply = {"id": msg.get("id")}
        try:
            row, err = row_fn(msg["path"], float(msg["smooth"]), float(msg["window"]), msg.get("source") or "original")
            reply.update(row=row, error=err or "")
        except Exception as e:  # noqa: BLE001 - named on the trial's line in the overview
            reply.update(row=None, error=f"{type(e).__name__}: {e}")
        send(reply)


def lower_priority():
    """Below normal: a worker reading sixty trials must not slow the window (or a recording in it) down. Windows only; silently nothing elsewhere."""
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)           # BELOW_NORMAL_PRIORITY_CLASS
    except Exception:  # noqa: BLE001 - only a courtesy
        pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    spec = DEFAULT_ROW_FN
    if "--row-fn" in argv:
        i = argv.index("--row-fn")
        spec = argv[i + 1] if i + 1 < len(argv) else spec
    lower_priority()
    # the protocol goes through a private copy of stdout; the real fd 1 now points at stderr, so a stray print() or a library writing to
    # the console cannot end up in the middle of a message. stdin is read from fd 0 itself: pythonw has no sys.stdin / sys.stdout of its own.
    proto = os.fdopen(os.dup(1), "w", encoding="ascii", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    lines = os.fdopen(os.dup(0), "r", encoding="ascii", errors="replace")

    def send(message):
        proto.write(dumps(message) + "\n")
        proto.flush()
    try:
        row_fn = resolve(spec)
    except Exception as e:  # noqa: BLE001 - told to the window, which then runs the trials itself
        send({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return 2
    try:
        send({"ready": True, "pid": os.getpid(), "memory_mb": _memory_mb()})
        serve(row_fn, lines, send)
        send({"bye": True, "peak_mb": _memory_mb(peak=True)})
    except OSError:                       # the window is gone (its end of the pipe is closed): nobody is waiting for an answer
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
