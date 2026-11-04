"""The worker processes of the participant overview (0.32.0, limbtrack/overview_worker.py and the window's side in review_overview.py):
the lines of JSON they speak, rows from a worker equal to rows worked out in the window, a worker that crashes, is killed or does not
answer losing only its own trial, no hang when the window is closed, and the thread taking over when no worker can start.

This module is also the home of the small helpers the overview tests share: `make_trials` writes a participant of made-up trials (a meta
file each, nothing else) and `fake_row` is a row function a worker can import (`tests.test_overview_worker:fake_row`), quick and the same
in a thread and in a process. Its top-level imports are light on purpose: a worker imports this module to find `fake_row`."""
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from limbtrack import overview_worker as ow  # noqa: E402
from limbtrack import review_layer  # noqa: E402


# ------------------------------------------------------------------------------------------------ shared helpers
def make_trials(root, participant="P09", plan=((1, 70, 2), (2, 90, 2), (3, 110, 1)), minutes_from=0):
    """`root/<participant>/` with a meta file for every trial of `plan` [(position number, target, how many trials)], nothing else (the
    overview's listing needs no more; `fake_row` reads only the meta). Returns the meta paths in recording order."""
    folder = Path(root) / participant
    folder.mkdir(parents=True, exist_ok=True)
    out, n = [], minutes_from
    for pos, deg, trials in plan:
        for t in range(1, trials + 1):
            base = f"{participant}_{pos:02d}_knee_extension_{deg}deg_T{t}_20261030-{10 + n // 60:02d}{n % 60:02d}00"
            meta = {"participant": participant, "test_key": "knee_extension", "test_label": "Knee extension (MVIC)", "target_deg": deg,
                    "position_number": pos, "trial_number": t, "status": "complete", "started_at": f"2026-10-30T10:{n % 60:02d}:00",
                    "files": {"data.csv": base + "_data.csv"}}
            p = folder / (base + "_meta.json")
            p.write_text(json.dumps(meta), encoding="utf-8")
            out.append(p)
            n += 1
    return out


def fake_row(path, smooth, window, source):
    """A row like analysis.session_row makes it, from the meta file only. 'crash' in the file name ends the process, 'slow' waits half a
    minute, 'boom' raises: for the tests of a worker that dies or does not answer. A file that is not there raises like the real one."""
    name = os.path.basename(path)
    if "crash" in name:
        os._exit(3)
    if "slow" in name:
        time.sleep(30)
    if "boom" in name:
        raise RuntimeError("boom")
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    from limbtrack import verdict as verdicts
    i, pos = int(meta.get("trial_number") or 1), int(meta.get("position_number") or 1)
    v = verdicts.read_verdict(path) or {}
    peak = 300.0 + 25.0 * i + 10.0 * pos + (7 * (i * 5 + pos * 3) % 13)
    row = {"trial": name[: -len("_meta.json")], "position_number": pos, "trial_number": i, "participant": meta.get("participant"),
           "test_key": meta.get("test_key"), "target_deg": meta.get("target_deg"), "force_unit": "N", "force_peak": peak,
           "force_peak_above_rest": peak - 20.0, "force_window_peak": peak - 9.0, "force_window_peak_ms": window * 1000.0,
           "force_time_to_peak_s": 1.5 + 0.1 * i + (pos % 3) * 0.2, "angle_at_peak_force_deg": float(meta.get("target_deg") or 0) + 1.5 - i,
           "main_angle": "Knee", "force_zero_drift_n": float("nan"), "smooth": smooth, "data_source": source, "review_edits": "",
           "checks_to_look_at": "the zero drifted | the strap slipped" if (i + pos) % 3 == 0 else "",
           "verdict": v.get("verdict") or "", "verdict_reasons": "", "verdict_note": v.get("note") or "",
           "repeatability_pct_same_position": "", "third_trial_needed": ""}
    return row, ""


def same(a, b):
    """Two rows (or values) equal, NaN equal to NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def pid_alive(pid):
    """Whether a process of this id is still running (the id may be a few seconds old). Windows: the exit code; elsewhere: a signal 0 and not a zombie."""
    if os.name != "nt":
        if pid <= 0:
            return False                                                # (os.kill(0, 0) would signal the whole process group)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                return fh.read().rsplit(")", 1)[1].split()[0] != "Z"      # (a killed child that nobody has waited for yet is a zombie: gone)
        except OSError:
            return True
    import ctypes
    k32 = ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = k32.OpenProcess(0x1000, 0, pid)               # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_uint32()
    ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
    k32.CloseHandle(handle)
    return bool(ok) and code.value == 259                  # STILL_ACTIVE


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _page(folder, procs=2, use_thread=False, row_fn=fake_row, parallel=True, **knobs):
    """An overview page that sends every trial of `folder` to `procs` worker processes (the thread helps only with use_thread)."""
    from limbtrack import review_nav as nav
    from limbtrack.review_overview import OverviewPage
    _app()
    ov = OverviewPage(row_fn, parallel=parallel)
    ov.process_min_todo, ov.max_processes, ov.trials_per_process, ov.use_thread = 1, procs, 1, use_thread
    for k, v in knobs.items():
        setattr(ov, k, v)
    ov.set_participant(Path(folder).name, nav.list_trials(folder))
    ov.set_settings(0.1, 0.5, "original")
    return ov


def _manifest(folder):
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------------------------ the talk
def test_a_job_line_is_answered_with_a_row_or_the_reason_and_a_bad_line_or_a_bad_trial_stops_nothing():
    sent = []
    lines = [json.dumps({"id": 1, "path": "a_meta.json", "smooth": 0.1, "window": 0.5, "source": "original"}) + "\n",
             "\n", "this is not json\n", "[1, 2]\n",
             json.dumps({"id": 2, "path": "b_meta.json", "smooth": 0.1, "window": 0.5}) + "\n",
             json.dumps({"id": 3, "smooth": 0.1}) + "\n",                                           # no path: answered with the error
             json.dumps({"cmd": "quit"}) + "\n",
             json.dumps({"id": 4, "path": "never_meta.json", "smooth": 0.1, "window": 0.5}) + "\n"]    # after quit: not read

    def row_fn(path, smooth, window, source):
        if path == "b_meta.json":
            raise ValueError("no good")
        return {"p": path, "source": source, "x": float("nan")}, ""
    ow.serve(row_fn, lines, sent.append)
    assert [m["id"] for m in sent] == [1, 2, 3]
    assert sent[0]["row"]["source"] == "original" and sent[0]["error"] == ""
    assert sent[1] == {"id": 2, "row": None, "error": "ValueError: no good"} and sent[2]["row"] is None and "KeyError" in sent[2]["error"]


def test_every_line_is_plain_ascii_json_with_numpy_numbers_nan_and_odd_text_in_it():
    import numpy as np
    line = ow.dumps({"id": 1, "row": {"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(True), "d": float("nan"), "e": "Ångström – ✎",
                                      "f": np.array([1.0, 2.0]), "g": Path("C:/x/y")}})
    assert line.isascii() and "\n" not in line
    back = json.loads(line)["row"]
    assert back["a"] == 1.5 and back["b"] == 3 and back["c"] is True and back["d"] != back["d"] and back["e"] == "Ångström – ✎" and back["f"] == [1.0, 2.0]
    from limbtrack.review_overview import plain_row
    assert plain_row({"a": np.float64(2.0), "b": np.int64(4), "c": np.bool_(False)}) == {"a": 2.0, "b": 4, "c": False}
    assert plain_row(None) is None


def test_only_functions_a_worker_can_find_again_are_sent_to_workers():
    assert ow.spec_for(fake_row) == "tests.test_overview_worker:fake_row"
    assert ow.spec_for(lambda *a: None) is None
    assert ow.spec_for(Path.exists) is None                       # a method
    assert ow.spec_for(ow.spec_for) == "limbtrack.overview_worker:spec_for"
    assert ow.resolve("tests.test_overview_worker:fake_row") is fake_row


def test_the_number_of_workers_can_be_set_or_switched_off(monkeypatch):
    monkeypatch.delenv(ow.ENV_WORKERS, raising=False)
    assert ow.configured_workers(7) == 7 and ow.processes_allowed()
    monkeypatch.setenv(ow.ENV_WORKERS, "3")
    assert ow.configured_workers(7) == 3
    monkeypatch.setenv(ow.ENV_WORKERS, "0")
    assert ow.configured_workers(7) == 0 and not ow.processes_allowed()
    monkeypatch.setenv(ow.ENV_WORKERS, "many")
    assert ow.configured_workers(7) == 7
    monkeypatch.delenv(ow.ENV_WORKERS)
    monkeypatch.setattr(sys, "frozen", True, raising=False)       # a bundled program would start itself again, not a worker
    assert not ow.processes_allowed()


def test_a_worker_process_says_ready_answers_a_job_and_leaves_when_told_to():
    proc = subprocess.Popen([sys.executable, "-m", "limbtrack.overview_worker", "--row-fn", "tests.test_overview_worker:fake_row"], cwd=str(ROOT),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready["ready"] is True and ready["pid"] > 0
        assert not pid_alive(0)
        proc.stdin.write('{"id": 7, "path": "nowhere_meta.json", "smooth": 0.1, "window": 0.5, "source": "original"}\n')
        proc.stdin.flush()
        answer = json.loads(proc.stdout.readline())
        assert answer["id"] == 7 and answer["row"] is None and answer["error"].startswith("FileNotFoundError")
        proc.stdin.write('{"cmd": "quit"}\n')
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline())["bye"] is True
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_worker_that_cannot_find_the_row_function_says_so_and_ends():
    out = subprocess.run([sys.executable, "-m", "limbtrack.overview_worker", "--row-fn", "tests.nothing_here:row"], cwd=str(ROOT), input="",
                         capture_output=True, text=True, timeout=60)
    msg = json.loads(out.stdout.splitlines()[0])
    assert msg["ready"] is False and "ModuleNotFoundError" in msg["error"] and out.returncode == 2


def test_a_worker_whose_window_has_gone_ends_by_itself():
    proc = subprocess.Popen([sys.executable, "-m", "limbtrack.overview_worker", "--row-fn", "tests.test_overview_worker:fake_row"], cwd=str(ROOT),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert json.loads(proc.stdout.readline())["ready"] is True
        proc.stdin.close()                                         # (what the end of the window does to its pipe)
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


# ------------------------------------------------------------------------------------------------ the window's side
def test_rows_from_worker_processes_equal_rows_worked_out_in_the_window_and_the_participants_folder_is_untouched(tmp_path):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 3), (2, 90, 2), (3, 110, 3)))
    before = _manifest(tmp_path / "recordings")
    ov = _page(tmp_path / "recordings" / "P09", procs=2)
    ov.ensure_rows()
    assert ov.wait_until_done(60000)
    assert ov.last_run["mode"] == "thread and processes" and ov.last_run["by"] == {"process 1": ov.last_run["rows"][0], "process 2": ov.last_run["rows"][1]} \
        or sum(ov.last_run["rows"]) == len(metas)
    assert sum(ov.last_run["rows"]) == len(metas) and ov.last_run["ready"] == 2 and not ov.last_run["reasons"] and "thread" not in ov.last_run["by"]
    assert not ov.stale_entries()
    from limbtrack.review_overview import plain_row
    for p in metas:
        want, err = fake_row(str(p), 0.1, 0.5, "original")
        got = ov.row_for(p.name[: -len("_meta.json")])
        assert err == "" and same(plain_row(want), got), p.name
    ov.shutdown()
    after = _manifest(tmp_path / "recordings")
    assert {k: v for k, v in after.items() if not k.startswith("review")} == before          # nothing beside the trials changed
    assert [k for k in after if k not in before] == [str(Path("review") / "P09" / "overview_cache.json")]


def test_the_thread_helps_while_the_workers_start_and_every_trial_is_worked_out_once(tmp_path):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 3), (2, 90, 3), (3, 110, 3), (4, 130, 3)))
    ov = _page(tmp_path / "recordings" / "P09", procs=2, use_thread=True)
    ov.ensure_rows()
    assert ov.wait_until_done(60000)
    assert sum(ov.last_run["by"].values()) == len(metas) and ov.last_run["by"].get("thread", 0) > 0          # (the workers need a moment to start)
    assert all(ov.row_for(p.name[: -len("_meta.json")]) is not None for p in metas)
    ov.shutdown()


def test_a_worker_that_crashes_loses_only_its_own_trial_and_says_why(tmp_path):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 3), (2, 90, 3), (3, 110, 3)))
    crash = metas[4]
    renamed = crash.with_name(crash.name.replace("knee_extension", "crash_knee_extension"))
    crash.rename(renamed)
    ov = _page(tmp_path / "recordings" / "P09", procs=2)
    ov.ensure_rows()
    assert ov.wait_until_done(60000)
    bad = ov._cache[renamed.name[: -len("_meta.json")]]
    assert bad[1] is None and "worker process" in bad[2] and "exit code 3" in bad[2]
    others = [b for b in ov._cache if b != renamed.name[: -len("_meta.json")]]
    assert len(others) == len(metas) - 1 and all(ov._cache[b][1] is not None for b in others)                # every other trial has its numbers
    ov.flush()
    peak_col = [ov.table.horizontalHeaderItem(c).text() for c in range(ov.table.columnCount())].index("Peak")
    assert [ov.table.item(r, peak_col).text() for r in range(ov.table.rowCount())].count("could not be read") == 1
    ov.shutdown()
    saved = json.loads((tmp_path / "recordings" / "review" / "P09" / "overview_cache.json").read_text())["trials"]
    assert renamed.name[: -len("_meta.json")] not in saved and len(saved) == len(metas) - 1                  # a failure is not kept for next time


def test_a_worker_that_is_killed_from_outside_loses_only_the_trial_it_was_reading(tmp_path):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 3), (2, 90, 3)))
    slow = metas[2]
    renamed = slow.with_name(slow.name.replace("knee_extension", "slow_knee_extension"))
    slow.rename(renamed)
    ov = _page(tmp_path / "recordings" / "P09", procs=2)
    ov.ensure_rows()
    end = time.monotonic() + 60
    victim = None
    while time.monotonic() < end and victim is None:                # wait until a worker is reading the slow trial
        _app().processEvents()
        victim = next((w for w in ov._procs.workers if w.job and w.job[1] == renamed.name[: -len("_meta.json")]), None)
        time.sleep(0.01)
    assert victim is not None and victim.pid
    os.kill(victim.pid, 15)                                         # (the interpreter itself, as a crash or a task manager would)
    assert ov.wait_until_done(60000)
    bad = ov._cache[renamed.name[: -len("_meta.json")]]
    assert bad[1] is None and bad[2].startswith("Could not be read: the worker process")
    assert all(ov._cache[b][1] is not None for b in ov._cache if b != renamed.name[: -len("_meta.json")]) and len(ov._cache) == len(metas)
    assert not pid_alive(victim.pid)
    ov.shutdown()


def forget_cache(folder):
    """Remove the rows a page kept on disk in the review layer, so the next page has to work them out."""
    review_layer.overview_cache_path(folder).unlink(missing_ok=True)


def test_a_worker_that_does_not_answer_in_time_is_ended_and_its_trial_marked(tmp_path):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 2), (2, 90, 2)))
    slow = metas[1]
    renamed = slow.with_name(slow.name.replace("knee_extension", "slow_knee_extension"))
    slow.rename(renamed)
    ov = _page(tmp_path / "recordings" / "P09", procs=2, job_timeout_ms=2500)
    t0 = time.monotonic()
    ov.ensure_rows()
    pids = set()                                                     # (the workers' process numbers, taken while they exist: the run retires them when it ends)
    while ov._running and time.monotonic() - t0 < 60:
        _app().processEvents()
        pids.update(w.pid for w in (ov._procs.workers if ov._procs is not None else []) if w.pid)
        time.sleep(0.005)
    assert ov.wait_until_done(60000)
    assert time.monotonic() - t0 < 25                                # not the 30 s the trial would have taken
    bad = ov._cache[renamed.name[: -len("_meta.json")]]
    assert bad[1] is None and "did not answer within 2 s" in bad[2]
    assert sum(1 for h in ov._cache.values() if h[1] is not None) == len(metas) - 1
    ov.shutdown()
    assert pids and not any(pid_alive(p) for p in pids)


def test_closing_the_window_in_the_middle_of_a_run_ends_every_worker_without_a_hang(tmp_path):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 3), (2, 90, 3), (3, 110, 3)))
    for m in metas[3:]:                                               # six trials that take half a minute each
        m.rename(m.with_name(m.name.replace("knee_extension", "slow_knee_extension")))
    ov = _page(tmp_path / "recordings" / "P09", procs=3)
    ov.ensure_rows()
    end = time.monotonic() + 60
    while time.monotonic() < end and not (ov._procs.busy() >= 2):
        _app().processEvents()
        time.sleep(0.01)
    pids = [w.pid for w in ov._procs.workers if w.pid]
    assert len(pids) >= 2 and all(pid_alive(p) for p in pids)
    t0 = time.monotonic()
    ov.shutdown()
    assert time.monotonic() - t0 < 20, "the shutdown waited for the slow trials"
    time.sleep(0.5)
    assert not any(pid_alive(p) for p in pids)
    assert not ov._running or ov.wait_until_done(10000)


def test_cancel_lets_the_trials_being_read_finish_and_starts_no_more(tmp_path):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 3), (2, 90, 3), (3, 110, 3), (4, 130, 3)))
    ov = _page(tmp_path / "recordings" / "P09", procs=1)
    ov.ensure_rows()
    end = time.monotonic() + 60
    while time.monotonic() < end and ov._done < 1:
        _app().processEvents()
        time.sleep(0.01)
    ov.cancel()
    assert ov.wait_until_done(30000)
    assert 1 <= ov._done <= len(metas) and ov.last_run["cancelled"] is True
    ov.shutdown()


def test_when_no_worker_can_start_the_thread_works_out_every_trial(tmp_path):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 3), (2, 90, 3)))
    ov = _page(tmp_path / "recordings" / "P09", procs=2, worker_python=str(tmp_path / "no_such_python.exe"))      # (only the workers: no thread at first)
    ov.ensure_rows()
    assert ov.wait_until_done(60000)
    assert ov.last_run.get("fell_back_to_thread") is True and ov.last_run["by"] == {"thread": len(metas)} and not ov.stale_entries()
    assert any("could not be started" in r for r in ov.last_run["reasons"])
    ov.shutdown()


def test_a_worker_that_cannot_load_the_row_function_is_given_up_and_the_thread_goes_on(tmp_path, monkeypatch):
    metas = make_trials(tmp_path / "recordings", plan=((1, 70, 3), (2, 90, 2)))
    monkeypatch.setattr(ow, "spec_for", lambda fn: "tests.nothing_here:row")
    ov = _page(tmp_path / "recordings" / "P09", procs=2)
    ov.ensure_rows()
    assert ov.wait_until_done(60000)
    assert ov.last_run.get("fell_back_to_thread") is True and sum(ov.last_run["by"].values()) == len(metas)
    assert any("could not load the row function" in r for r in ov.last_run["reasons"])
    ov.shutdown()


def test_processes_are_not_used_when_switched_off_for_a_function_they_cannot_find_or_for_a_small_job(tmp_path, monkeypatch):
    make_trials(tmp_path / "recordings", plan=((1, 70, 2), (2, 90, 2)))
    folder = tmp_path / "recordings" / "P09"
    monkeypatch.setenv(ow.ENV_WORKERS, "0")
    ov = _page(folder)
    ov.ensure_rows()
    assert ov.wait_until_done(60000) and ov.last_run["mode"] == "thread" and "switched off" in ov.last_run["why_not_processes"]
    ov.shutdown()
    monkeypatch.delenv(ow.ENV_WORKERS)
    forget_cache(folder)                                                       # (the first page kept its rows on disk: this one must work them out)
    ov = _page(folder, row_fn=lambda *a: fake_row(*a))
    ov.ensure_rows()
    assert ov.wait_until_done(60000) and ov.last_run["mode"] == "thread" and "cannot be imported" in ov.last_run["why_not_processes"]
    ov.shutdown()
    forget_cache(folder)
    ov = _page(folder, process_min_todo=40)
    ov.ensure_rows()
    assert ov.wait_until_done(60000) and ov.last_run["mode"] == "thread" and "quicker" in ov.last_run["why_not_processes"]
    ov.shutdown()
    forget_cache(folder)
    ov = _page(folder, parallel=False)
    ov.ensure_rows()
    assert ov.wait_until_done(60000) and ov.last_run["mode"] == "thread" and ov.last_run["why_not_processes"] == "processes not asked for"
    ov.shutdown()


# ------------------------------------------------------------------------------------------------ the real row function
@pytest.fixture(scope="module")
def real_trials(tmp_path_factory):
    """Trials with numbers to read: four real trials of the pilot (P02: knee extension at 20, 70 degrees and a hip test), copied without their
    videos, or, where the pilot folder is not on this machine, simulated recordings of three trials."""
    root = tmp_path_factory.mktemp("parity")
    pilot = ROOT / "Pilot" / "P02"
    if pilot.is_dir():
        dest = root / "P02"
        dest.mkdir()
        metas = sorted(pilot.glob("*_meta.json"))
        chosen = [m for m in metas if "_T1_" in m.name][:4]
        for m in chosen:
            for f in pilot.glob(m.name[: -len("_meta.json")] + "_*"):
                if f.suffix in (".json", ".csv"):
                    (dest / f.name).write_bytes(f.read_bytes())
        return dest, True
    from tests.test_pilot_checks import recording_at
    for _ in range(3):
        recording_at(root, 70.0, seconds=1.5)
    return root / "P01", False


def test_rows_from_worker_processes_equal_the_real_rows_of_recorded_trials(real_trials):
    from limbtrack import review, review_nav as nav
    from limbtrack.review_overview import OverviewPage, plain_row
    folder, from_pilot = real_trials
    entries = nav.list_trials(folder)
    assert len(entries) >= 3
    _app()
    ov = OverviewPage(review.trial_row, parallel=True)
    ov.process_min_todo, ov.max_processes, ov.trials_per_process, ov.use_thread = 1, 2, 1, False        # (the workers alone: their rows are compared)
    ov.set_participant(folder.name, entries)
    ov.set_settings(0.1, 0.5, "original")
    ov.ensure_rows()
    assert ov.wait_until_done(120000)
    assert ov.last_run["mode"] == "thread and processes" and sum(ov.last_run["rows"]) == len(entries) and not ov.last_run["reasons"], ov.last_run
    for e in entries:
        row, err = review.trial_row(str(e.meta_path), 0.1, 0.5, "original")
        assert err == "" and row is not None
        got = ov.row_for(e.base)
        assert got is not None and same(plain_row(row), got), (e.base, [k for k in got if not same(plain_row(row).get(k), got[k])])
    ov.shutdown()
    # and the same trials again at another smoothing and read through the review layer: still equal
    ov2 = OverviewPage(review.trial_row, parallel=True)
    ov2.process_min_todo, ov2.max_processes, ov2.trials_per_process, ov2.use_thread = 1, 1, 1, False
    ov2.set_participant(folder.name, entries)
    ov2.set_settings(0.2, 1.0, "reviewed")
    ov2.ensure_rows()
    assert ov2.wait_until_done(120000)
    for e in entries:
        row, _ = review.trial_row(str(e.meta_path), 0.2, 1.0, "reviewed")
        assert same(plain_row(row), ov2.row_for(e.base))
    ov2.shutdown()


def test_the_worker_starts_from_pythonw_the_way_the_program_is_launched(real_trials):
    """Start.bat runs `pythonw -m limbtrack`: a worker started by pythonw has no console and no sys.stdin of its own, yet talks over its pipes."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        pytest.skip("no pythonw.exe beside this interpreter")
    proc = subprocess.Popen([str(pythonw), "-m", "limbtrack.overview_worker", "--row-fn", "tests.test_overview_worker:fake_row"], cwd=str(ROOT),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        ready = json.loads(proc.stdout.readline().decode("ascii"))
        assert ready["ready"] is True
        proc.stdin.write(b'{"id": 1, "path": "nowhere_meta.json", "smooth": 0.1, "window": 0.5, "source": "original"}\n')
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline().decode("ascii"))["error"].startswith("FileNotFoundError")
        proc.stdin.write(b'{"cmd": "quit"}\n')
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline().decode("ascii"))["bye"] is True
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
