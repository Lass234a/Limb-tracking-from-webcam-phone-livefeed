"""Approving a participant and the saved overview rows, 0.32.0 (limbtrack/review_layer.py: write_approval, read_approval(s), approval_path,
trial_fingerprint, overview_cache_path, read/write_overview_cache; limbtrack/review_overview.py: OverviewPage.request_approval, commit_approval,
approval_state, the 'Changed since approved' filter, the cache on disk and the worker processes).

Two promises matter more than the rest and have the most tests: an approval says which state of every trial's review was accepted and
notices when a trial differs from it later (and never blocks an export), and the saved rows and approvals are written ONLY in the
participant's review folder, whole or not at all, so that nothing beside a trial's own files is ever touched and a corrupt file only costs
the work of a first load.

The trials are made up (tests.test_overview_worker.make_trials: a meta file each, and `fake_row`, a row function a worker process can
import), so a participant of seven trials is worked out in a few milliseconds; the tests that need real trials (the Review tab and its
exports) record two. NEVER open a real dialog here: the question before approving is `OverviewPage.confirm` and the complaint `OverviewPage.warn`,
both replaced in every page made here, and a fixture makes any real QMessageBox/QDialog.exec raise instead of waiting for a click."""
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from limbtrack import overview_worker, reference, verdict  # noqa: E402
from limbtrack import review_layer as rl  # noqa: E402
from limbtrack import review_nav as nav  # noqa: E402
from limbtrack import review_overview as ro  # noqa: E402
from limbtrack.engine import SOFTWARE_VERSION  # noqa: E402
from limbtrack.review_overview import OverviewPage, plain_row  # noqa: E402
from limbtrack.review_style import GOOD, WARN  # noqa: E402

from tests.test_overview_worker import fake_row, make_trials, same  # noqa: E402

PLAN = ((1, 70, 3), (2, 90, 2), (3, 110, 2))                    # seven trials in three positions
CALLS = []                                                       # the trials `counting_row` was asked to work out
HIDDEN = []                                                      # errors raised inside a Qt slot (see `no_hidden_errors`)
BASE_ROLE = Qt.UserRole + 1


def counting_row(path, smooth, window, source):
    """`fake_row` that writes down every call: a cache hit is a call that does not happen."""
    CALLS.append((Path(path).name[: -len("_meta.json")], round(smooth, 4), round(window, 4), source))
    return fake_row(path, smooth, window, source)


# ------------------------------------------------------------------------------------------------ set-up
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_real_dialogs(monkeypatch, app):
    """A dialog that is really opened waits for a click for ever. Any that a test did not replace fails the test at once instead."""
    def boom(*a, **k):
        raise AssertionError("a real dialog was asked for in a test: replace OverviewPage.confirm / .warn")
    for cls in (QMessageBox, QDialog):
        monkeypatch.setattr(cls, "exec", boom, raising=False)
    for name in ("warning", "information", "question", "critical", "about"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(boom))
    CALLS.clear()


@pytest.fixture(autouse=True)
def no_hidden_errors(monkeypatch, request):
    """Qt prints an exception raised inside a slot and carries on, so a test would pass straight over it (the overview works its rows out in
    slots). They are written down here instead, and the test fails (a test that is marked xfail for such a bug fails in its own body)."""
    HIDDEN.clear()
    monkeypatch.setattr(sys, "excepthook", lambda kind, err, tb: HIDDEN.append(f"{kind.__name__}: {err}"))
    yield
    if HIDDEN and not request.node.get_closest_marker("xfail"):
        pytest.fail("an error was raised inside a Qt slot and hidden by Qt: " + "; ".join(HIDDEN))


@pytest.fixture
def part(tmp_path):
    """recordings/P09 with seven made-up trials. `.folder` is the participant's folder, `.review` the participant's review folder."""
    root = tmp_path / "recordings"
    metas = make_trials(root, plan=PLAN)
    return SimpleNamespace(root=root, folder=root / "P09", review=root / "review" / "P09", metas=metas, bases=[base_of(m) for m in metas])


@pytest.fixture
def page(app):
    """Makes overview pages (confirm and warn replaced, nothing real can open) and shuts them all down afterwards."""
    made = []

    def make(folder, row_fn=counting_row, answer=True, parallel=False, **knobs):
        ov = OverviewPage(row_fn, parallel=parallel)
        ov.asked, ov.warned, ov.events = [], [], []

        def confirm(parent, title, text, detail=""):
            ov.asked.append((title, text, detail))
            return answer
        ov.confirm = confirm
        ov.warn = lambda parent, title, text: ov.warned.append((title, text))
        ov.approvalChanged.connect(lambda: ov.events.append("approvalChanged"))
        for k, v in knobs.items():
            setattr(ov, k, v)
        ov.set_participant(Path(folder).name, nav.list_trials(folder))
        ov.set_settings(0.1, 0.5, "original")
        made.append(ov)
        return ov
    yield make
    for ov in made:
        ov.shutdown()


def base_of(meta):
    return Path(meta).name[: -len("_meta.json")]


def work(ov):
    ov.ensure_rows()
    assert ov.wait_until_done(60000)
    assert not HIDDEN, HIDDEN


def judge(meta, v, reasons=(), where=verdict.IN_REVIEW):
    return verdict.write_verdict(meta, v, reasons, where=where)


def edit(meta):
    """A Review edit saved in the review layer (a typed force zero)."""
    rv = rl.Review.load(meta)
    rv.set_zero(0.0123, "typed", {"n_per_v": 100.0, "regression_id": "", "from": "recorded"})
    rv.save(software_version="test")


def bump(path, seconds=7):
    """Move a file's modification time on: a file changed twice within the clock's tick would otherwise look as it did (a person cannot do that)."""
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 10 ** 9))


def manifest(folder):
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def label_colour(ov):
    return ov.approval_label.styleSheet()


def approved_page(part, page, **kw):
    """Two trials valid, one compromised, one edited in Review; a page on them; approved."""
    judge(part.metas[0], verdict.VALID)
    judge(part.metas[1], verdict.VALID)
    judge(part.metas[2], verdict.COMPROMISED, ["bad_force"])
    edit(part.metas[3])
    ov = page(part.folder, **kw)
    rec = ov.commit_approval()
    ov.events.clear()
    return ov, rec


# ------------------------------------------------------------------------------------------------ where the files live
def test_the_approval_and_the_cache_live_in_the_participants_review_folder(part):
    assert rl.approval_path(part.folder) == part.review / "approval.json" == rl.approval_path(part.metas[0])        # for the folder or for a trial's meta file
    assert rl.overview_cache_path(part.folder) == part.review / "overview_cache.json" == rl.overview_cache_path(part.metas[4])
    assert rl.participant_dir_of(part.folder) == part.review == rl.participant_dir_of(part.metas[0])
    assert rl.REVIEW_FOLDER in rl.approval_path(part.folder).parts
    assert rl.approval_path(part.root / "P55") == part.root / "review" / "P55" / "approval.json"                       # (a participant that is not there yet)
    assert not (part.root / "review").exists()                                                                       # looking writes nothing


# ------------------------------------------------------------------------------------------------ the approval file
def test_a_participant_never_approved_reads_as_nothing_whatever_the_file_holds(part):
    path = rl.approval_path(part.folder)
    assert rl.read_approval(part.folder) == {} and rl.read_approvals(part.folder) == [] and not path.exists()         # no file: nothing, and nothing made
    path.parent.mkdir(parents=True)
    for content in (b"", b"{ not json", b"[1, 2, 3]", b'"text"', b'{"approvals": "x"}', b'{"approvals": null}', b'{"nothing": 1}'):
        path.write_bytes(content)
        assert rl.read_approval(part.folder) == {} and rl.read_approvals(part.folder) == [], content
    path.write_text(json.dumps({"approvals": [1, {"a": 1}, "x", None, {"b": 2}]}), encoding="utf-8")
    assert rl.read_approvals(part.folder) == [{"a": 1}, {"b": 2}] and rl.read_approval(part.folder) == {"b": 2}       # only records are records


def test_an_approval_is_saved_whole_and_read_back_as_it_was(part):
    record = {"software_version": "9.9.9", "participant": "P09", "counts": {"trials": 7, "valid": 2, "compromised": 1, "not_judged": 4, "unreadable": 0, "edited": 1},
              "trials": [{"base": b, "verdict": None, "edited": False, "fingerprint": "0123456789abcdef"} for b in part.bases[:2]]}
    before, copy = datetime.now().replace(microsecond=0), json.loads(json.dumps(record))
    saved = rl.write_approval(part.folder, record)
    after = datetime.now()
    assert record == copy                                                                                            # the caller's record is not changed
    assert before <= datetime.fromisoformat(saved["approved_at"]) <= after and "." not in saved["approved_at"]       # local time, to the second, added when missing
    assert {k: v for k, v in saved.items() if k != "approved_at"} == record
    path = rl.approval_path(part.folder)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["format"] == rl.APPROVAL_FORMAT == 1 and on_disk["participant"] == "P09" and on_disk["approvals"] == [saved]
    assert "changes no number and no recorded file" in on_disk["meaning"] and "\n  " in path.read_text(encoding="utf-8")       # (a person can read it)
    assert rl.read_approval(part.folder) == saved == rl.read_approval(part.metas[0]) and rl.read_approvals(part.folder) == [saved]
    assert not list(part.review.glob("*.tmp")) and [p.name for p in part.review.iterdir()] == ["approval.json"]
    kept = rl.write_approval(part.folder, {"approved_at": "2026-01-02T03:04:05", "note": "x"})                       # a given time is kept, and the folder name stands in for a name
    assert kept["approved_at"] == "2026-01-02T03:04:05" and json.loads(path.read_text(encoding="utf-8"))["participant"] == "P09"


def test_a_write_that_fails_leaves_the_earlier_approval_as_it_was(part):
    rl.write_approval(part.folder, {"approved_at": "2026-01-01T10:00:00", "n": 1})
    path = rl.approval_path(part.folder)
    first = path.read_bytes()
    loop = {}
    loop["me"] = loop                                                                                                # cannot be written as JSON
    with pytest.raises(ValueError):
        rl.write_approval(part.folder, {"approved_at": "2026-01-02T10:00:00", "loop": loop})
    assert path.read_bytes() == first and not list(part.review.glob("*.tmp"))                                       # whole or not at all
    assert [a["n"] for a in rl.read_approvals(part.folder)] == [1]


def test_an_approval_is_refused_anywhere_but_in_a_review_folder_and_beside_no_trial(part, tmp_path, monkeypatch):
    assert issubclass(rl.OriginalsProtected, OSError)                                                                # (so every handler of a failed save shows it)
    before = manifest(part.root)
    with pytest.raises(rl.OriginalsProtected):                                                                       # the participant's own folder
        with rl.atomic(part.folder / "approval.json") as fh:
            fh.write("{}")
    elsewhere = tmp_path / "elsewhere" / "P09" / "approval.json"
    real_approval_path, real_cache_path = rl.approval_path, rl.overview_cache_path
    with pytest.raises(rl.OriginalsProtected):                                                                       # a folder that is not a review folder
        rl.write_json(elsewhere, {"approvals": []})
    monkeypatch.setattr(rl, "approval_path", lambda f: elsewhere)                                                    # even if the path logic went wrong
    with pytest.raises(rl.OriginalsProtected):
        rl.write_approval(part.folder, {"approved_at": "2026-01-01T10:00:00"})
    monkeypatch.setattr(rl, "overview_cache_path", lambda f: elsewhere.with_name("overview_cache.json"))
    with pytest.raises(rl.OriginalsProtected):
        rl.write_overview_cache(part.folder, {part.bases[0]: [{"key": [1], "row": {"a": 1}}]})
    assert not (tmp_path / "elsewhere").exists() and manifest(part.root) == before                                   # nothing was made, nothing changed
    monkeypatch.setattr(rl, "approval_path", real_approval_path)
    monkeypatch.setattr(rl, "overview_cache_path", real_cache_path)
    beside = tmp_path / "rec2" / "review" / "P01"                                                                     # a review folder that holds a trial's own files
    beside.mkdir(parents=True)
    (beside / "P01_x_meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(rl.OriginalsProtected):
        rl.write_approval(tmp_path / "rec2" / "P01", {"approved_at": "2026-01-01T10:00:00"})
    assert [p.name for p in beside.iterdir()] == ["P01_x_meta.json"]


def test_the_earlier_approvals_stay_and_are_listed_oldest_first(part):
    first = rl.write_approval(part.folder, {"approved_at": "2026-01-01T10:00:00", "mark": 1, "trials": [{"base": "a", "fingerprint": "x"}]})
    snapshot = rl.read_approvals(part.folder)
    rl.write_approval(part.folder, {"approved_at": "2026-01-02T10:00:00", "mark": 2})
    rl.write_approval(part.folder, {"approved_at": "2026-01-03T10:00:00", "mark": 3})
    approvals = rl.read_approvals(part.folder)
    assert [a["mark"] for a in approvals] == [1, 2, 3] and rl.read_approval(part.folder)["mark"] == 3
    assert approvals[0] == first == snapshot[0]                                                                      # none of the earlier ones is changed
    assert json.loads(rl.approval_path(part.folder).read_text(encoding="utf-8"))["approvals"] == approvals


def test_older_approvals_lose_their_list_of_trials_but_keep_their_counts(part):
    for i in range(rl.APPROVALS_IN_FULL + 1):
        rl.write_approval(part.folder, {"approved_at": f"2026-01-0{i + 1}T10:00:00", "mark": i, "software_version": "1.2.3", "counts": {"trials": 3, "valid": i},
                                        "trials": [{"base": f"b{k}", "fingerprint": "f"} for k in range(3)]})
    approvals = rl.read_approvals(part.folder)
    assert len(approvals) == 6 and all("trials" in a for a in approvals[1:]) and all(len(a["trials"]) == 3 for a in approvals[1:])        # the latest five are in full
    cut = approvals[0]
    assert "trials" not in cut and cut["trials_left_out"] == 3                                                       # the oldest is short: how many were left out
    assert (cut["mark"], cut["software_version"], cut["counts"], cut["approved_at"]) == (0, "1.2.3", {"trials": 3, "valid": 0}, "2026-01-01T10:00:00")


def test_an_approval_that_was_cut_short_keeps_saying_how_many_trials_it_left_out(part):
    for i in range(rl.APPROVALS_IN_FULL + 3):
        rl.write_approval(part.folder, {"approved_at": f"2026-01-0{i + 1}T10:00:00", "mark": i, "trials": [{"base": f"b{k}", "fingerprint": "f"} for k in range(3)]})
    cut = [a for a in rl.read_approvals(part.folder) if "trials" not in a]
    assert len(cut) == 3 and [a["trials_left_out"] for a in cut] == [3, 3, 3]


def test_at_most_fifty_approvals_are_kept_the_newest(part):
    for i in range(rl.APPROVALS_KEPT + 3):
        rl.write_approval(part.folder, {"approved_at": "2026-01-01T10:00:00", "mark": i})
    approvals = rl.read_approvals(part.folder)
    assert len(approvals) == rl.APPROVALS_KEPT == 50 and [a["mark"] for a in approvals] == list(range(3, 53))


def test_an_unreadable_approval_file_does_not_stop_the_next_approval(part):
    path = rl.approval_path(part.folder)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{ half a file")
    saved = rl.write_approval(part.folder, {"approved_at": "2026-01-01T10:00:00", "mark": 1})
    assert rl.read_approvals(part.folder) == [saved]                                                                 # (the unreadable history cannot be kept)
    assert not list(part.review.glob("*.tmp"))


# ------------------------------------------------------------------------------------------------ the fingerprint of a trial's review
def test_the_fingerprint_is_short_stable_and_changes_with_anything_the_review_consists_of(part):
    meta = part.metas[0]
    fp = rl.trial_fingerprint(meta)
    assert len(fp) == 16 and int(fp, 16) >= 0 and rl.trial_fingerprint(meta) == fp == rl.trial_fingerprint(str(meta))
    seen = {fp}

    def changed(what):
        new = rl.trial_fingerprint(meta)
        assert new not in seen, what
        seen.add(new)
    judge(meta, verdict.VALID)                                                                                       # a verdict made in Review
    changed("verdict in Review")
    verdict.write_verdict(meta, verdict.COMPROMISED, ["wrong_dots"], where=verdict.ON_SITE)                          # the verdict judged on site (beside the trial)
    changed("verdict on site")
    edit(meta)                                                                                                       # an edit: review.json
    changed("zero edit")
    rv = rl.Review.load(meta)
    rv.add_dot_edit("anchor", "manual", rows=[{"frame": 3, "t_s": 0.1, "x": 10.0, "y": 20.0}])                       # a dot fix: dots_edit.csv
    rv.save()
    changed("dot fix")
    reference.write_sidecar(meta, "goniometer", 1.0, {"knee": [68.0]})                                               # a reference reading typed in Review
    changed("reference reading")
    meta.with_name(base_of(meta) + "_reference.json").write_text('{"instrument": "goniometer"}', encoding="utf-8")   # the older file beside the trial
    changed("older reference file")


def test_the_fingerprint_is_the_content_not_the_time_and_not_the_trials_own_files(part):
    meta = part.metas[0]
    judge(meta, verdict.VALID)
    fp = rl.trial_fingerprint(meta)
    for p in rl.review_dir(meta).iterdir():                                                                          # touched, not changed
        bump(p)
    judge_file = rl.review_file(meta, "verdict.json")
    judge_file.write_bytes(judge_file.read_bytes())                                                                  # saved again with the same content
    assert rl.trial_fingerprint(meta) == fp
    meta.write_text(json.dumps({**json.loads(meta.read_text(encoding="utf-8")), "target_deg": 55}), encoding="utf-8")   # the trial itself: not part of the review
    assert rl.trial_fingerprint(meta) == fp
    assert rl.trial_fingerprint(part.metas[1]) == rl.trial_fingerprint(part.metas[2])                                # (no review at all is the same for every trial: the base name ties it to one)
    a, b = part.metas[3], part.metas[4]
    rl.write_json(rl.review_file(a, "review.json"), {"x": 1})
    rl.write_json(rl.review_file(b, "verdict.json"), {"x": 1})
    assert rl.trial_fingerprint(a) != rl.trial_fingerprint(b)                                                        # the same bytes in another file are another review


# ------------------------------------------------------------------------------------------------ approving
def test_committing_an_approval_records_the_state_of_every_trial(part, page):
    judge(part.metas[0], verdict.VALID)
    judge(part.metas[1], verdict.VALID)
    judge(part.metas[2], verdict.COMPROMISED, ["bad_force"])
    edit(part.metas[3])
    part.metas[6].write_text("{ not a trial", encoding="utf-8")                                                      # cannot be read
    ov = page(part.folder)
    before = datetime.now().replace(microsecond=0)
    rec = ov.commit_approval()
    after = datetime.now()
    assert before <= datetime.fromisoformat(rec["approved_at"]) <= after
    assert rec["software_version"] == SOFTWARE_VERSION != "" and rec["participant"] == "P09"
    assert rec["counts"] == {"trials": 7, "valid": 2, "compromised": 1, "not_judged": 3, "unreadable": 1, "edited": 1}
    assert [t["base"] for t in rec["trials"]] == part.bases                                                           # in the order they were recorded
    assert [t["verdict"] for t in rec["trials"]] == ["valid", "valid", "compromised", None, None, None, None]         # (an unreadable trial has none)
    assert [t["edited"] for t in rec["trials"]] == [False, False, False, True, False, False, False]
    for t, meta in zip(rec["trials"], part.metas):
        assert set(t) == {"base", "verdict", "edited", "fingerprint"} and t["fingerprint"] == rl.trial_fingerprint(meta)
    assert rl.read_approval(part.folder) == rec and rl.read_approvals(part.folder) == [rec]                          # and that is what is on disk
    assert ov.events == ["approvalChanged"]
    assert [p.name for p in part.review.iterdir() if p.is_file()] == ["approval.json"]


def test_before_and_after_an_approval_the_state_and_the_label(part, page):
    ov = page(part.folder)
    s = ov.approval_state()
    assert s == {"approved": False, "approved_at": "", "label": "Not approved yet", "changed": [], "removed": [], "counts": {}, "approvals": 0,
                 "now": {"trials": 7, "valid": 0, "compromised": 0, "not_judged": 7, "unreadable": 0, "edited": 0}}
    assert ov.approval_label.text() == "Not approved yet" and label_colour(ov) == "" and ov.approve_button.isEnabled()
    rec = ov.commit_approval()
    s = ov.approval_state()
    assert s["approved"] and s["approved_at"] == rec["approved_at"] and s["approvals"] == 1 and s["changed"] == [] == s["removed"] and s["counts"] == rec["counts"]
    want = f"Approved {ro.when_text(rec['approved_at'])} - 0 valid, 0 compromised, 7 not judged"
    assert s["label"] == want == ov.approval_label.text() and f"color: {GOOD}" in label_colour(ov)
    again = page(part.folder)                                                                                        # another page (or the next start of the program) finds it
    assert again.approval_state() == s and again.approval_label.text() == want
    judge(part.metas[0], verdict.VALID)
    assert page(part.folder).approval_state()["changed"] == [part.bases[0]]                                          # ... also for a change made while the program was shut


def test_the_label_names_valid_compromised_and_not_judged_as_they_were_approved(part, page):
    ov, rec = approved_page(part, page)
    assert ov.approval_label.text() == f"Approved {ro.when_text(rec['approved_at'])} - 2 valid, 1 compromised, 4 not judged"
    for m in part.metas[3:]:
        judge(m, verdict.VALID)
    ov2 = page(part.folder)
    ov2.commit_approval()
    assert ov2.approval_label.text().endswith("- 6 valid, 1 compromised") and "not judged" not in ov2.approval_label.text()          # nothing left out to say


def test_when_text_reads_a_time_for_people():
    assert ro.when_text("2026-10-30T17:12:04") == "30 Oct 17:12" and ro.when_text("2026-01-02T03:04:05") == "2 Jan 03:04"
    assert ro.when_text("last Tuesday") == "last Tuesday" and ro.when_text(None) == "" and ro.when_text("") == ""


def test_the_button_is_off_without_trials_and_asks_nothing(app):
    ov = OverviewPage(counting_row)
    asked = []
    ov.confirm = lambda *a: asked.append(a) or True
    assert not ov.approve_button.isEnabled() and ov.approval_label.text() == "Not approved yet"
    assert ov.request_approval() is None and asked == []
    with pytest.raises(ValueError):
        ov.commit_approval()
    ov.shutdown()


# ------------------------------------------------------------------------------------------------ a trial that changes after the approval
def test_a_verdict_changed_after_the_approval_is_noticed_at_once(part, page):
    ov, rec = approved_page(part, page)
    assert ov.approval_state()["changed"] == [] and ov.events == []
    judge(part.metas[4], verdict.VALID)
    ov.update_verdict(part.bases[4])                                                                                 # what the Review tab does when the verdict button is pressed
    s = ov.approval_state()
    assert s["changed"] == [part.bases[4]] and s["removed"] == [] and s["approved"]
    assert ov.approval_label.text() == f"Approved {ro.when_text(rec['approved_at'])} - 1 trial changed since" and f"color: {WARN}" in label_colour(ov)
    assert ov.events == ["approvalChanged"]
    judge(part.metas[5], verdict.COMPROMISED, ["other"])                                                             # and another
    ov.update_verdict(part.bases[5])
    assert ov.approval_label.text().endswith("- 2 trials changed since") and sorted(ov.approval_state()["changed"]) == sorted(part.bases[4:6])
    assert ov.events == ["approvalChanged"] * 2
    assert rl.read_approval(part.folder) == rec                                                                      # the approval itself is not touched by a change


def test_an_edit_saved_after_the_approval_is_noticed_when_the_overview_looks_again(part, page):
    ov, rec = approved_page(part, page)
    edit(part.metas[5])
    entry = ov.entries[5]
    entry.refresh_edited()                                                                                           # what the tab does after an edit
    assert ov.approval_state()["changed"] == [part.bases[5]] and "changed since" not in ov.approval_label.text()      # (the label waits until the page looks again)
    ov.refresh_approval()
    assert ov.approval_state()["changed"] == [part.bases[5]] and ov.approval_label.text().endswith("- 1 trial changed since")
    assert ov.approval_state()["now"]["edited"] == 2 and ov.approval_state()["counts"]["edited"] == 1                 # now against as approved
    assert ov.events == ["approvalChanged"]
    ov.refresh_approval()                                                                                            # looking again at the same state says nothing more
    assert ov.events == ["approvalChanged"]


def test_the_label_catches_up_when_the_rows_are_asked_for_or_the_page_is_shown(part, page):
    ov, rec = approved_page(part, page)
    judge(part.metas[4], verdict.VALID)                                                                              # changed behind the page's back
    assert "changed since" not in ov.approval_label.text()                                                           # (the page has not looked yet)
    work(ov)                                                                                                         # asking for the rows looks at the trials again
    assert ov.approval_label.text().endswith("- 1 trial changed since") and ov.events == ["approvalChanged"]
    judge(part.metas[5], verdict.VALID)
    ov.show()
    QApplication.processEvents()
    assert ov.approval_label.text().endswith("- 2 trials changed since")
    ov.close()


def test_touching_a_review_file_without_changing_it_is_not_a_change(part, page):
    ov, rec = approved_page(part, page)
    for p in (part.review / part.bases[0] / "verdict.json", part.review / part.bases[3] / "review.json"):
        p.write_bytes(p.read_bytes())                                                                                # saved again, the same content
        bump(p)
    ov.refresh_approval()
    assert ov.approval_state()["changed"] == [] and "changed since" not in ov.approval_label.text() and ov.events == []


def test_a_trial_that_is_new_or_gone_counts_as_changed(part, page):
    ov, rec = approved_page(part, page)
    new = make_trials(part.root, plan=((4, 130, 1),), minutes_from=40)                                              # recorded after the approval
    ov.set_participant("P09", nav.list_trials(part.folder))                                                          # (the tab lists the participant again when it is shown)
    assert ov.approval_state()["changed"] == [base_of(new[0])] and ov.approval_label.text().endswith("- 1 trial changed since")
    new[0].unlink()
    part.metas[1].unlink()                                                                                           # a trial taken away
    ov.set_participant("P09", nav.list_trials(part.folder))
    s = ov.approval_state()
    assert s["removed"] == [part.bases[1]] and s["changed"] == [] and ov.approval_label.text().endswith("- 1 trial changed since") and f"color: {WARN}" in label_colour(ov)
    assert s["now"]["trials"] == 6 and s["counts"]["trials"] == 7


def test_approving_again_accepts_what_there_is_now_and_keeps_the_first_approval(part, page):
    ov, first = approved_page(part, page)
    judge(part.metas[4], verdict.VALID)
    ov.update_verdict(part.bases[4])
    assert ov.approval_state()["changed"]
    second = ov.commit_approval()
    s = ov.approval_state()
    assert s["changed"] == [] and s["approvals"] == 2 and "changed since" not in s["label"] and s["counts"]["valid"] == 3
    assert f"color: {GOOD}" in label_colour(ov)
    approvals = rl.read_approvals(part.folder)
    assert approvals == [first, second] and "trials" in approvals[0] and approvals[0]["counts"]["valid"] == 2        # the first is still there, as it was
    assert ov.events == ["approvalChanged"] * 2


def test_changes_are_found_against_the_latest_approval_only(part, page):
    ov, first = approved_page(part, page)
    judge(part.metas[4], verdict.VALID)
    ov.update_verdict(part.bases[4])
    ov.commit_approval()
    judge(part.metas[5], verdict.VALID)
    ov.update_verdict(part.bases[5])
    assert ov.approval_state()["changed"] == [part.bases[5]]                                                         # (the first change was accepted by the second approval)


# ------------------------------------------------------------------------------------------------ the filter
def shown_bases(ov):
    return [ov.table.item(r, 0).data(BASE_ROLE) for r in range(ov.table.rowCount())]


def test_the_changed_since_approved_filter_lists_just_the_changed_trials(part, page):
    ov, rec = approved_page(part, page)
    assert ("changed", "Changed since approved") in ro.FILTERS and ov.filter.itemText(ov.filter.findData("changed")) == "Changed since approved"
    ov.filter.setCurrentIndex(ov.filter.findData("changed"))
    assert ov.table.rowCount() == 0                                                                                  # nothing changed yet
    judge(part.metas[2], verdict.VALID)                                                                              # the compromised one is now valid
    bump(rl.review_file(part.metas[2], "verdict.json"))                                                              # (a file written again within the same tick of the clock)
    ov.update_verdict(part.bases[2])
    assert shown_bases(ov) == [part.bases[2]]                                                                        # the list follows by itself while the filter is on
    edit(part.metas[6])
    ov.entries[6].refresh_edited()
    ov.refresh_approval()
    assert sorted(shown_bases(ov)) == sorted([part.bases[2], part.bases[6]])
    assert "7 trials" in ov.summary.text()                                                                           # the summary is about the participant, not the filter
    ov.filter.setCurrentIndex(ov.filter.findData("all"))
    assert ov.table.rowCount() == 7
    ov.filter.setCurrentIndex(ov.filter.findData("changed"))
    assert ov.table.rowCount() == 2


def test_the_changed_since_approved_filter_is_empty_for_a_participant_never_approved(part, page):
    ov = page(part.folder)
    ov.filter.setCurrentIndex(ov.filter.findData("changed"))
    assert ov.table.rowCount() == 0
    judge(part.metas[0], verdict.VALID)
    ov.update_verdict(part.bases[0])
    assert ov.table.rowCount() == 0 and ov.approval_state()["changed"] == []                                         # nothing was approved, so nothing differs from it
    ov.filter.setCurrentIndex(ov.filter.findData("all"))
    assert ov.table.rowCount() == 7


# ------------------------------------------------------------------------------------------------ the question before approving
def test_the_button_asks_first_and_approves_only_on_a_yes(part, page):
    judge(part.metas[0], verdict.VALID)
    ov = page(part.folder, answer=False)
    ov.approve_button.click()
    assert len(ov.asked) == 1 and ov.asked[0][0] == "Approve the review of P09?"
    assert not rl.approval_path(part.folder).exists() and ov.approval_label.text() == "Not approved yet" and ov.events == []          # "no" changes nothing
    assert ov.request_approval() is None and len(ov.asked) == 2
    ov.confirm = lambda parent, title, text, detail="": ov.asked.append((title, text, detail)) or True
    ov.approve_button.click()
    rec = rl.read_approval(part.folder)
    assert len(ov.asked) == 3 and rec["counts"]["valid"] == 1 and ov.approval_label.text().startswith("Approved ") and ov.events == ["approvalChanged"]
    assert ov.request_approval()["approved_at"] and len(rl.read_approvals(part.folder)) == 2                         # (the return value is the record)


def test_the_question_says_how_much_is_judged_and_names_what_is_not(part, page):
    ov = page(part.folder)
    s = ov.approval_summary()
    assert s["title"] == "Approve the review of P09?" and s["judged"] == 0 and s["trials"] == 7
    assert s["text"].splitlines() == ["0 of 7 trials are judged (0 valid, 0 compromised).",
                                      "7 trials are NOT judged yet. You can approve anyway; they stay marked not judged."]
    assert "changes no number and no recorded file" in s["detail"] and "you can still export" in s["detail"]
    ov.request_approval()
    assert ov.asked[-1] == (s["title"], s["text"], s["detail"])                                                      # the dialog gets exactly this


def test_the_question_counts_in_the_singular_and_the_plural(part, page):
    for m in part.metas[:5]:
        judge(m, verdict.VALID)
    judge(part.metas[5], verdict.COMPROMISED, ["wrong_dots"])
    edit(part.metas[0])
    part.metas[6].write_text("[", encoding="utf-8")
    text = page(part.folder).approval_summary()["text"].splitlines()
    assert text == ["6 of 7 trials are judged (5 valid, 1 compromised).", "1 trial cannot be read.", "1 trial was edited in Review (a force zero or dot fixes)."]
    part.metas[4].write_text("[", encoding="utf-8")
    judge(part.metas[3], None)                                                                                       # taken back: not judged
    edit(part.metas[1])
    text = page(part.folder).approval_summary()["text"].splitlines()
    assert text[0] == "4 of 7 trials are judged (3 valid, 1 compromised)."
    assert text[1:] == ["1 trial is NOT judged yet. You can approve anyway; it stays marked not judged.", "2 trials cannot be read.",
                        "2 trials were edited in Review (a force zero or dot fixes)."]


def test_the_question_has_no_line_for_what_does_not_apply(part, page):
    for m in part.metas:
        judge(m, verdict.VALID)
    assert page(part.folder).approval_summary()["text"] == "7 of 7 trials are judged (7 valid, 0 compromised)."


def test_a_save_that_fails_is_told_and_nothing_is_approved(part, page, monkeypatch):
    ov = page(part.folder)

    def refuse(folder, record):
        raise rl.OriginalsProtected("the review folder is read-only")
    monkeypatch.setattr(rl, "write_approval", refuse)
    assert ov.request_approval() is None
    assert len(ov.warned) == 1 and ov.warned[0][0] == "Not approved" and "the review folder is read-only" in ov.warned[0][1]
    assert ov.approval_label.text() == "Not approved yet" and ov.events == [] and not rl.approval_path(part.folder).exists()
    with pytest.raises(OSError):                                                                                     # the direct call raises, for a caller that handles it
        ov.commit_approval()


class FakeBox:
    """A QMessageBox that answers by itself: the real `_confirm_box` is run without a window."""
    Question, AcceptRole, RejectRole = QMessageBox.Question, QMessageBox.AcceptRole, QMessageBox.RejectRole
    press = "Approve"
    made = []

    def __init__(self, parent=None):
        self.parent, self.buttons, self.said, self.default, self.clicked = parent, {}, {}, None, None
        FakeBox.made.append(self)

    def setIcon(self, icon):
        self.said["icon"] = icon

    def setWindowTitle(self, text):
        self.said["title"] = text

    def setText(self, text):
        self.said["text"] = text

    def setInformativeText(self, text):
        self.said["detail"] = text

    def addButton(self, text, role):
        self.buttons[text] = (text, role)
        return self.buttons[text]

    def setDefaultButton(self, button):
        self.default = button

    def exec(self):
        self.clicked = self.buttons[FakeBox.press]
        return 0

    def clickedButton(self):
        return self.clicked


def test_the_real_question_box_has_approve_and_cancel_and_returns_the_answer(monkeypatch):
    monkeypatch.setattr(ro, "QMessageBox", FakeBox)
    FakeBox.made.clear()
    FakeBox.press = "Approve"
    assert ro._confirm_box(None, "Approve the review of P09?", "7 of 7 trials are judged.", "It changes no number.") is True
    box = FakeBox.made[-1]
    assert box.said == {"icon": FakeBox.Question, "title": "Approve the review of P09?", "text": "7 of 7 trials are judged.", "detail": "It changes no number."}
    assert sorted(box.buttons) == ["Approve", "Cancel"] and box.buttons["Approve"][1] == FakeBox.AcceptRole and box.buttons["Cancel"][1] == FakeBox.RejectRole
    assert box.default == box.buttons["Approve"]
    FakeBox.press = "Cancel"
    assert ro._confirm_box(None, "t", "x") is False


def test_the_default_page_asks_through_the_real_question_box_not_through_a_hidden_yes(part, monkeypatch):
    """Without the test replacements, the page's own `confirm` is the box above (and the autouse fixture would stop a real one)."""
    ov = OverviewPage(counting_row)
    assert ov.confirm is ro._confirm_box and ov.warn is ro._warn_box
    monkeypatch.setattr(ro, "QMessageBox", FakeBox)
    FakeBox.press = "Cancel"
    ov.set_participant("P09", nav.list_trials(part.folder))
    assert ov.request_approval() is None and not rl.approval_path(part.folder).exists()
    FakeBox.press = "Approve"
    assert ov.request_approval()["participant"] == "P09" and rl.approval_path(part.folder).is_file()
    ov.shutdown()


# ------------------------------------------------------------------------------------------------ approval and exports
def test_approving_does_not_stop_an_export_before_or_after_a_change(part, page):
    ov, rec = approved_page(part, page)
    asked = []
    ov.exportRequested.connect(asked.append)
    ov.request_export(False)                                                                                         # approved, nothing changed
    judge(part.metas[4], verdict.VALID)
    ov.update_verdict(part.bases[4])
    ov.request_export(True)                                                                                          # approved, a trial changed since
    assert asked == [False, True] and ov.warned == [] and ov.asked == []                                             # no question, no complaint
    assert ov.approve_button.isEnabled()


@pytest.fixture(scope="module")
def real_pair(tmp_path_factory):
    """recordings/P01 with two real trials (a simulated force and a recorded video), for the Review tab's exports."""
    from tests.test_pilot_checks import recording_at
    root = tmp_path_factory.mktemp("approval_real")
    for _ in range(2):
        recording_at(root, 70.0, seconds=1.0)
    return root


def test_the_review_tab_approves_notices_a_judgement_and_still_exports(real_pair, tmp_path, monkeypatch):
    from limbtrack.review import ReviewTab
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a[2]))
    root = tmp_path / "recordings"
    shutil.copytree(real_pair, root)
    tab = ReviewTab(root)
    try:
        tab.set_participant(root / "P01")
        ov = tab.overview
        ov.confirm = lambda *a, **k: True
        ov.warn = lambda *a, **k: infos.append(a[-1])
        assert ov.approve_button.isEnabled() and ov.approval_label.text() == "Not approved yet"
        ov.approve_button.click()
        approval = rl.approval_path(root / "P01")
        assert approval == root / "review" / "P01" / "approval.json" and approval.is_file() and ov.approval_label.text().startswith("Approved ")
        first = approval.read_bytes()
        out = tab.export_session()                                                                                   # approved: the export works
        assert out is not None and out.is_file() and out.parent == root / "review" / "P01" / "exports" and approval.read_bytes() == first
        tab.verdict_bar.btn_valid.click()                                                                            # a trial is judged in Review after the approval
        assert ov.approval_label.text().endswith("- 1 trial changed since") and ov.approval_state()["changed"] == [tab._entries[tab._index].base]
        ov.filter.setCurrentIndex(ov.filter.findData("changed"))
        assert shown_bases(ov) == [tab._entries[tab._index].base]
        out2 = tab.export_session()                                                                                  # changed since: the export still works, and says nothing about it
        assert out2 is not None and out2.is_file() and approval.read_bytes() == first
        with open(out2, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 2 and not any("approv" in name.lower() for name in reader.fieldnames)                   # (an export does not carry the approval)
        assert not infos or all("approv" not in str(i).lower() for i in infos)
        assert not list((root / "P01").glob("*approv*")) and not list((root / "P01").glob("*overview_cache*"))      # nothing beside the trials
    finally:
        if tab.trial is not None:
            tab.trial.close()
        tab.overview.shutdown()


# ------------------------------------------------------------------------------------------------ the cache on disk
def read_cache(part):
    return rl.read_overview_cache(part.folder)


def test_a_first_load_works_every_trial_out_once_and_saves_the_rows(part, page):
    ov = page(part.folder)
    assert len(ov.stale_entries()) == 7 and not rl.overview_cache_path(part.folder).exists()
    work(ov)
    assert sorted(c[0] for c in CALLS) == sorted(part.bases) and ov.cache_hits == 0 and ov.last_run["trials"] == 7 and ov.last_run["mode"] == "thread"
    cache = read_cache(part)
    assert cache["format"] == FMT and cache["written_at"] and sorted(cache["trials"]) == sorted(part.bases)
    for meta in part.metas:
        (variant,) = cache["trials"][base_of(meta)]
        assert variant["key"] == list(ro.file_stamp(meta) + ro.settings_stamp(0.1, 0.5, "original"))                 # keyed by the files, the settings and the program
        assert same(variant["row"], plain_row(fake_row(str(meta), 0.1, 0.5, "original")[0]))
    assert not list(part.review.glob("*.tmp")) and ov.cache_error == ""
    assert ov.table.rowCount() == 7


def test_a_second_load_takes_every_row_from_the_cache_and_works_nothing_out(part, page):
    first = page(part.folder)
    work(first)
    rows = {b: first.row_for(b) for b in part.bases}
    first.shutdown()
    CALLS.clear()
    ov = page(part.folder)                                                                                           # a new page: the next start of the program
    assert ov.stale_entries() == [] and CALLS == [] and ov.cache_hits == 7
    work(ov)
    assert CALLS == [] and ov._running == 0 and ov.last_run == {} and ov.cache_hits == 7                              # nothing was started
    for b in part.bases:
        assert same(ov.row_for(b), rows[b]), b                                                                       # (NaN equal to NaN)
    assert ov.row_for(part.bases[0])["force_zero_drift_n"] != ov.row_for(part.bases[0])["force_zero_drift_n"]        # (the NaN in the row came back as NaN)
    ov.render()
    assert ov.table.rowCount() == 7 and "…" not in [ov.table.item(r, ro.COLUMNS.index("Peak")).text() for r in range(7)]    # the table shows numbers, not '…'


@pytest.mark.parametrize("kind", ["meta", "verdict_in_review", "verdict_on_site", "review_edit", "reference_in_review", "reference_beside_trial"])
def test_a_changed_file_works_out_only_its_own_trial_again(part, page, kind):
    first = page(part.folder)
    work(first)
    first.shutdown()
    meta = part.metas[3]
    if kind == "meta":
        meta.write_text(json.dumps({**json.loads(meta.read_text(encoding="utf-8")), "target_deg": 55}), encoding="utf-8")
        bump(meta)
    elif kind == "verdict_in_review":
        judge(meta, verdict.VALID)
    elif kind == "verdict_on_site":
        judge(meta, verdict.COMPROMISED, ["bad_force"], where=verdict.ON_SITE)
    elif kind == "review_edit":
        edit(meta)
    elif kind == "reference_in_review":
        reference.write_sidecar(meta, "goniometer", 1.0, {"knee": [68.0]})
    else:
        meta.with_name(base_of(meta) + "_reference.json").write_text('{"instrument": "goniometer"}', encoding="utf-8")
    CALLS.clear()
    ov = page(part.folder)
    assert [e.base for e in ov.stale_entries()] == [part.bases[3]]                                                    # the six others are as they were
    assert ov.cache_hits == 6
    work(ov)
    assert [c[0] for c in CALLS] == [part.bases[3]]
    ov.shutdown()
    CALLS.clear()
    again = page(part.folder)
    assert again.stale_entries() == [] and CALLS == []                                                               # and the new row was kept, too
    variants = read_cache(part)["trials"]
    assert len(variants[part.bases[3]]) == 2 and variants[part.bases[3]][0]["key"] != variants[part.bases[3]][1]["key"]         # (the old one stays behind it)
    assert all(len(variants[b]) == 1 for b in part.bases if b != part.bases[3])


def test_a_verdict_given_now_shows_at_once_but_its_numbers_wait_to_be_worked_out_again(part, page):
    ov = page(part.folder)
    work(ov)
    base = part.bases[0]
    judge(part.metas[0], verdict.VALID)
    ov.update_verdict(base)
    assert ov.row_for(base) is None and ov.row_for(base, allow_stale=True)["verdict"] == "valid"                     # the old numbers with the new verdict, marked out of date
    assert [e.base for e in ov.stale_entries()] == [base]
    CALLS.clear()
    work(ov)
    assert [c[0] for c in CALLS] == [base] and ov.row_for(base)["verdict"] == "valid"


def test_new_settings_work_every_trial_out_again_and_switching_back_costs_nothing(part, page):
    ov = page(part.folder)
    work(ov)
    CALLS.clear()
    ov.set_settings(0.2, 0.5)                                                                                        # smoothing
    assert len(ov.stale_entries()) == 7
    work(ov)
    assert sorted(CALLS) == sorted((b, 0.2, 0.5, "original") for b in part.bases)
    for settings in ((0.2, 1.0, None), (0.2, 1.0, "reviewed")):                                                      # the sliding window, original or reviewed
        CALLS.clear()
        ov.set_settings(*settings)
        assert len(ov.stale_entries()) == 7
        work(ov)
        assert len(CALLS) == 7 and {c[1:] for c in CALLS} == {(0.2, 1.0, settings[2] or "original")}
    assert ov.source == "reviewed"
    ov.set_settings(0.2, 1.0, "nonsense")                                                                            # an unknown source is not taken
    assert ov.source == "reviewed"
    ov.set_settings(0.1, 0.5, "original")                                                                            # back to the first: all four are on disk
    CALLS.clear()
    assert ov.stale_entries() == [] and ov.row_for(part.bases[0])["data_source"] == "original" and ov.row_for(part.bases[0])["smooth"] == 0.1
    work(ov)
    assert CALLS == []
    ov.shutdown()
    fresh = page(part.folder)                                                                                        # and in a new session too
    for settings in ((0.2, 0.5, "original"), (0.2, 1.0, "reviewed")):
        fresh.set_settings(*settings)
        assert fresh.stale_entries() == []
    assert CALLS == []


def test_only_the_latest_variants_of_a_trial_are_kept(part, page):
    ov = page(part.folder)
    steps = [round(0.1 * k, 1) for k in range(1, 7)]                                                                 # six smoothings: 0.1 ... 0.6
    for s in steps:
        ov.set_settings(s, 0.5)
        work(ov)
    variants = read_cache(part)["trials"][part.bases[0]]
    assert len(variants) == rl.OVERVIEW_CACHE_VARIANTS == 4
    smoothing_at = len(ro.file_stamp(part.metas[0]))                                                                 # the key is the files' times, then the settings
    assert [v["key"][smoothing_at] for v in variants] == [0.6, 0.5, 0.4, 0.3]                                       # the newest first
    ov.set_settings(0.1, 0.5)
    assert len(ov.stale_entries()) == 7                                                                              # the oldest was dropped: worked out again


def test_a_new_program_version_works_everything_out_again(part, page, monkeypatch):
    first = page(part.folder)
    work(first)
    first.shutdown()
    assert ro.settings_stamp(0.1, 0.5, "original")[3] == SOFTWARE_VERSION
    monkeypatch.setattr(ro, "SOFTWARE_VERSION", "99.0.0")                                                            # a new version may work the numbers out differently
    CALLS.clear()
    ov = page(part.folder)
    assert len(ov.stale_entries()) == 7
    work(ov)
    assert len(CALLS) == 7
    ov.shutdown()
    monkeypatch.setattr(ro, "SOFTWARE_VERSION", SOFTWARE_VERSION)
    CALLS.clear()
    back = page(part.folder)
    assert back.stale_entries() == [] and CALLS == []                                                                # (the rows of the other version are still kept)


def test_a_changed_load_cell_calibration_file_works_everything_out_again(part, page, monkeypatch, tmp_path):
    registry = tmp_path / "load_cell_calibrations.json"
    monkeypatch.setattr(ro, "REGISTRY_FILE", registry)
    assert ro.settings_stamp(0.1, 0.5, "original")[4] == 0                                                           # no registry: a number, not an error
    registry.write_text("{}", encoding="utf-8")
    first = page(part.folder)
    work(first)
    first.shutdown()
    assert page(part.folder).stale_entries() == []
    bump(registry)                                                                                                   # a calibration was added: the newtons come from it
    ov = page(part.folder)
    assert len(ov.stale_entries()) == 7


FMT = rl.OVERVIEW_CACHE_FORMAT


@pytest.mark.parametrize("content", [
    b"{ not json", b"", b"[1, 2, 3]", b'"text"', b"null", b"\x00\xff\xfe",
    json.dumps({"format": FMT - 1, "trials": {"T": [{"key": [1], "row": {"a": 1}}]}}).encode(),                        # another format (the one before 0.33.0 too)
    json.dumps({"format": FMT, "trials": [1, 2]}).encode(),                                                          # no trials
    json.dumps({"format": FMT}).encode(),
    json.dumps({"format": FMT, "trials": {"T": [{"key": [1], "row": "not a row"}, {"row": {"a": 1}}, {"key": "x", "row": {"a": 1}}, {}]}}).encode(),
])
def test_a_cache_file_that_cannot_be_used_is_ignored_and_replaced(part, page, content):
    path = rl.overview_cache_path(part.folder)
    path.parent.mkdir(parents=True)
    path.write_bytes(content.replace(b'"T"', json.dumps(part.bases[0]).encode()))
    ov = page(part.folder)
    assert len(ov.stale_entries()) == 7 and ov.cache_hits == 0
    work(ov)
    assert len(CALLS) == 7 and all(ov.row_for(b) is not None for b in part.bases) and ov._done == 7
    cache = read_cache(part)
    assert cache["format"] == FMT and sorted(cache["trials"]) == sorted(part.bases)                                  # the file is good again
    for b, meta in zip(part.bases, part.metas):
        variants = cache["trials"][b]
        assert same(variants[0]["row"], plain_row(fake_row(str(meta), 0.1, 0.5, "original")[0])) and len(variants) <= rl.OVERVIEW_CACHE_VARIANTS   # (the newest first)
    assert not list(part.review.glob("*.tmp"))


@pytest.mark.parametrize("entry", [5, None, "text", {"key": [1]}, [None, 5], ["a"]])
def test_a_cache_with_a_trial_entry_of_the_wrong_shape_is_ignored_too(part, page, entry):
    """Valid JSON in the right format, but an entry the program never wrote (a hand-edited file): it must cost a first load, not the participant."""
    path = rl.overview_cache_path(part.folder)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"format": FMT, "trials": {part.bases[0]: entry}}), encoding="utf-8")
    ov = page(part.folder)
    assert len(ov.stale_entries()) == 7
    work(ov)
    assert len(CALLS) == 7 and ov._done == 7
    (variant,) = read_cache(part)["trials"][part.bases[0]]
    assert isinstance(variant, dict) and variant["row"]["trial"] == part.bases[0]


def test_reading_a_cache_that_is_not_there_or_not_right_gives_nothing(part):
    assert rl.read_overview_cache(part.folder) == {} and rl.read_overview_cache(part.root / "P55") == {}
    path = rl.overview_cache_path(part.folder)
    path.parent.mkdir(parents=True)
    for content in (b"{", b"[]", b'{"format": 9, "trials": {}}', b'{"format": 1, "trials": {}}', ('{"format": %d, "trials": []}' % FMT).encode()):
        path.write_bytes(content)                                                                                   # (format 1: the cache of 0.32.0, without the other angles)
        assert rl.read_overview_cache(part.folder) == {}
    path.write_text('{"format": %d, "trials": {}}' % FMT, encoding="utf-8")
    assert rl.read_overview_cache(part.folder) == {"format": FMT, "trials": {}}


def test_the_cache_keeps_the_newest_variants_numbers_nan_and_plain_json(part):
    import numpy as np
    variants = [{"key": [k], "row": {"v": k, "nan": float("nan"), "n": np.float64(1.5), "arr": np.array([1, 2]), "path": Path("a/b")}} for k in range(6)]
    rl.write_overview_cache(part.folder, {part.bases[0]: variants, part.bases[1]: [], part.bases[2]: variants[:1]})
    path = rl.overview_cache_path(part.folder)
    raw = path.read_text(encoding="utf-8")
    assert ": " not in raw and ", " not in raw                                                                        # one compact line
    cache = rl.read_overview_cache(part.folder)
    assert sorted(cache["trials"]) == sorted([part.bases[0], part.bases[2]])                                         # a trial with no variant is not written
    kept = cache["trials"][part.bases[0]]
    assert [v["key"] for v in kept] == [[0], [1], [2], [3]]                                                          # the first four: the list is newest first
    row = kept[0]["row"]
    assert row["v"] == 0 and row["nan"] != row["nan"] and row["n"] == 1.5 and row["arr"] == [1, 2] and row["path"] == str(Path("a/b"))
    assert not list(part.review.glob("*.tmp"))


def test_the_cache_holds_only_the_trials_of_the_participant_as_it_is_now(part, page):
    first = page(part.folder)
    work(first)
    first.shutdown()
    gone = part.metas[1]
    gone.unlink()
    edit(part.metas[0])                                                                                              # something to work out again, so the cache is written
    ov = page(part.folder)
    work(ov)
    ov.shutdown()
    assert sorted(read_cache(part)["trials"]) == sorted(part.bases[:1] + part.bases[2:])


def test_a_cache_that_cannot_be_written_is_said_and_the_numbers_are_still_there(part, page, tmp_path, monkeypatch):
    outside = tmp_path / "elsewhere" / "P09" / "overview_cache.json"
    real_cache_path = rl.overview_cache_path
    monkeypatch.setattr(rl, "overview_cache_path", lambda f: outside)
    ov = page(part.folder)
    work(ov)
    assert ov.cache_error.startswith("OriginalsProtected") and "not in a review folder" in ov.cache_error
    assert "The numbers could not be kept for next time" in ov.hint.text() and ov.cache_error in ov.hint.text()
    assert all(ov.row_for(b) is not None for b in part.bases) and not (tmp_path / "elsewhere").exists()
    monkeypatch.setattr(rl, "overview_cache_path", real_cache_path)                                                  # the folder is fine again: the next save works, and the words go
    ov.set_settings(0.2, 0.5)
    work(ov)
    assert ov.cache_error == "" and "could not be kept" not in ov.hint.text() and rl.overview_cache_path(part.folder).is_file()


def test_nothing_is_written_beside_the_trials_only_in_the_review_folder(part, page):
    before = manifest(part.root)
    assert not any(k.startswith("review") for k in before)
    judge(part.metas[0], verdict.VALID)
    edit(part.metas[1])
    ov = page(part.folder)
    work(ov)
    ov.commit_approval()
    judge(part.metas[2], verdict.COMPROMISED, ["other"])
    ov.update_verdict(part.bases[2])
    work(ov)
    ov.request_approval()
    ov.shutdown()
    after = manifest(part.root)
    kept = {k: v for k, v in after.items() if not k.startswith("review")}
    assert kept == before                                                                                            # the participant's folder: the same files, sizes and times
    new = sorted(k for k in after if k not in before)
    assert new and all(Path(k).parts[:2] == ("review", "P09") for k in new), new
    names = {Path(k).name for k in new}
    assert {"approval.json", "overview_cache.json", "verdict.json", "review.json"} <= names and not any(n.endswith(".tmp") for n in names)
    assert names <= {"approval.json", "overview_cache.json", "verdict.json", "review.json"}
    assert sorted(p.name for p in part.folder.iterdir()) == sorted(p.name for p in part.metas)                       # nothing else beside the trials


# ------------------------------------------------------------------------------------------------ worker processes
def tree(tmp_path, name):
    root = tmp_path / name
    make_trials(root, plan=((1, 70, 3), (2, 90, 3), (3, 110, 3), (4, 130, 3)))
    return root / "P09"


def test_worker_processes_give_the_rows_the_window_gives_and_the_cache_takes_them(tmp_path, page):
    if not overview_worker.processes_allowed():
        pytest.skip("worker processes are switched off here (LIMBTRACK_OVERVIEW_WORKERS=0 or a bundled program): nothing to compare")
    inside, outside = tree(tmp_path, "inside"), tree(tmp_path, "outside")
    a = page(inside, row_fn=fake_row, parallel=False)
    b = page(outside, row_fn=fake_row, parallel=True, process_min_todo=1, max_processes=2, trials_per_process=1, use_thread=False)
    work(a)
    work(b)
    assert a.last_run["mode"] == "thread" and a.last_run["why_not_processes"] == "processes not asked for"
    if (os.cpu_count() or 1) > 1:
        assert b.last_run["mode"] == "thread and processes" and sum(b.last_run["rows"]) == 12 and "thread" not in b.last_run["by"] and not b.last_run["reasons"]
    bases = [e.base for e in a.entries]
    assert len(bases) == 12 and bases == [e.base for e in b.entries]
    for base in bases:
        assert a.row_for(base) is not None and same(a.row_for(base), b.row_for(base)), base                          # equal, NaN and all
    assert b.row_for(bases[0])["force_zero_drift_n"] != b.row_for(bases[0])["force_zero_drift_n"]                     # (the NaN really is in the rows)
    a.shutdown()
    b.shutdown()
    on_a, on_b = rl.read_overview_cache(inside)["trials"], rl.read_overview_cache(outside)["trials"]
    assert sorted(on_a) == sorted(on_b) == sorted(bases)
    for base in bases:
        assert same(on_a[base][0]["row"], on_b[base][0]["row"])                                                      # the same on disk, too
    again = page(outside, row_fn=fake_row, parallel=True, process_min_todo=1)                                         # a second load of the processes' rows needs no process
    assert again.stale_entries() == [] and again.cache_hits == 12
    work(again)
    assert again.last_run == {}
