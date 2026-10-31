"""The review layer (0.30.0, limbtrack/review_layer.py): where Review's changes live, that nothing is ever written beside a trial's own files,
and the edit list: save, load, history, reset, fingerprints of the originals."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from limbtrack import review_layer as rl  # noqa: E402


def _trial(root, participant="P01", base="P01_02_knee_extension_70deg_T1_20261028-113014", data="a,b\n1,2\n"):
    """The trial's own files (a small meta, data.csv, force.csv) and nothing else."""
    folder = root / participant
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / f"{base}_meta.json"
    meta.write_text(json.dumps({"participant": participant, "files": {"data.csv": f"{base}_data.csv", "force.csv": f"{base}_force.csv",
                                                                      "raw.mp4": f"{base}_raw.mp4"}}), encoding="utf-8")
    (folder / f"{base}_data.csv").write_text(data, encoding="utf-8")
    (folder / f"{base}_force.csv").write_text("t_s,clock_s,volts,newtons\n0,0,0.1,\n", encoding="utf-8")
    (folder / f"{base}_raw.mp4").write_bytes(b"not really a video")
    return meta


def _listing(folder):
    return sorted((p.name, p.stat().st_size) for p in Path(folder).iterdir())


def test_the_review_layer_is_a_sibling_tree_keyed_by_participant_and_trial(tmp_path):
    meta = _trial(tmp_path / "recordings")
    base = "P01_02_knee_extension_70deg_T1_20261028-113014"
    assert rl.review_dir(meta) == tmp_path / "recordings" / "review" / "P01" / base
    assert rl.review_json_path(meta).name == "review.json" and rl.dots_csv_path(meta).name == "dots_edit.csv"
    assert rl.exports_dir(meta) == tmp_path / "recordings" / "review" / "P01" / "exports"
    assert rl.review_file(meta, "verdict.json") == rl.review_dir(meta) / "verdict.json"


def test_nothing_can_be_written_outside_a_review_folder_or_beside_a_trials_files(tmp_path):
    meta = _trial(tmp_path / "recordings")
    before = _listing(meta.parent)
    with pytest.raises(rl.OriginalsProtected):                     # next to the trial
        with rl.atomic(meta.parent / "summary.csv") as fh:
            fh.write("x")
    with pytest.raises(rl.OriginalsProtected):                     # not under a review folder at all
        rl.write_json(tmp_path / "recordings" / "elsewhere.json", {"a": 1})
    with pytest.raises(rl.OriginalsProtected):                     # a folder called review that holds a trial's own files
        _trial(tmp_path / "x" / "review")
        rl.write_json(tmp_path / "x" / "review" / "P01" / "oops.json", {})
    assert issubclass(rl.OriginalsProtected, OSError)               # so every handler of a failed save shows it
    assert _listing(meta.parent) == before


def test_a_write_appears_whole_or_not_at_all(tmp_path):
    meta = _trial(tmp_path / "recordings")
    target = rl.review_file(meta, "x.json")
    rl.write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    with pytest.raises(RuntimeError):
        with rl.atomic(target) as fh:
            fh.write("half a file")
            raise RuntimeError("the disk went away")
    assert json.loads(target.read_text()) == {"a": 1} and not list(target.parent.glob("*.tmp"))        # the earlier file is as it was


def test_the_fingerprints_say_when_an_original_was_changed_afterwards(tmp_path):
    meta = _trial(tmp_path / "recordings")
    fp = rl.fingerprint(meta)
    assert set(fp) == {"meta.json", "data.csv", "force.csv", "raw.mp4"} and "sha256" in fp["data.csv"] and "sha256" not in fp["raw.mp4"]
    assert rl.changed_originals(meta, fp) == []
    data = meta.parent / "P01_02_knee_extension_70deg_T1_20261028-113014_data.csv"
    st = data.stat()
    os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))                          # only touched: same bytes
    assert rl.changed_originals(meta, fp) == []
    data.write_text("a,b\n9,9\n", encoding="utf-8")                                              # really changed
    assert rl.changed_originals(meta, fp) == [data.name]
    data.unlink()
    assert rl.changed_originals(meta, fp) == [data.name + " (missing)"]


def test_a_new_review_is_empty_and_writes_nothing_until_it_is_saved(tmp_path):
    meta = _trial(tmp_path / "recordings")
    rv = rl.Review.load(meta)
    assert not rv.exists() and not rv.has_edits() and rv.summary_lines() == []
    rv.log("looked", "nothing yet")
    assert not (tmp_path / "recordings" / "review").exists()


def test_zero_and_dot_edits_survive_a_save_and_a_load_with_their_history(tmp_path):
    meta = _trial(tmp_path / "recordings")
    rv = rl.Review.load(meta)
    rv.set_zero(0.0123, "typed", {"n_per_v": 250.0, "regression_id": "R1", "from": "recorded"})
    rv.add_dot_edit("anchor", "hold", from_s=6.0, to_s=12.2, x=31.5, y=633.0)
    eid = rv.add_dot_edit("anchor", "manual", rows=[{"frame": 3, "t_s": 0.1, "x": 10.0, "y": 20.0}, {"frame": 9, "t_s": 0.3, "x": 16.0, "y": 26.0}])
    rv.save(software_version="9.9.9")
    again = rl.Review.load(meta)
    assert again.has_edits() and again.zero["zero_v"] == 0.0123 and again.zero["source"] == "typed" and again.zero["n_per_v"] == 250.0
    assert [e["kind"] for e in again.dots] == ["hold", "manual"] and again.dots[0]["x"] == 31.5
    assert [(r["edit"], r["frame"], r["x"]) for r in again.dot_rows] == [(eid, 3, 10.0), (eid, 9, 16.0)]
    assert [h["action"] for h in again.data["history"]] == ["zero", "dot anchor: hold", "dot anchor: manual"]
    assert again.data["software_version"] == "9.9.9" and again.data["participant"] == "P01" and again.data["created_at"]
    assert set(again.data["original"]["files"]) == {"meta.json", "data.csv", "force.csv", "raw.mp4"}      # the originals it was made from
    lines = again.summary_lines()
    assert "force zero typed: +0.0123 V" in lines[0] and "dot anchor: held at (31.5, 633.0) 6.00-12.20 s; set by hand in 2 pictures" in lines[1]


def test_reset_and_undo_take_edits_back_and_keep_the_history(tmp_path):
    meta = _trial(tmp_path / "recordings")
    rv = rl.Review.load(meta)
    rv.add_dot_edit("anchor", "hold", from_s=1.0, to_s=2.0, x=1.0, y=2.0)
    rv.add_dot_edit("anchor", "lost", from_s=3.0, to_s=4.0)
    rv.add_dot_edit("strap", "lost", from_s=3.0, to_s=4.0)
    assert rv.undo_dot_edit("anchor") and [e["kind"] for e in rv.dots if e["dot"] == "anchor"] == ["hold"]
    assert not rv.undo_dot_edit("epicondyle")
    rv.reset_dot("anchor")
    assert rv.edited_dot_names() == ["strap"]
    rv.set_zero(0.01, "rest", {"n_per_v": 1.0, "regression_id": "", "from": "active"}, region=(1.0, 2.0))
    rv.clear_zero()
    assert rv.zero is None
    rv.reset_all()
    assert not rv.has_edits() and rv.dot_rows == []
    rv.save()
    saved = rl.Review.load(meta)
    assert not saved.has_edits() and [h["action"] for h in saved.data["history"]][-1] == "reset" and len(saved.data["history"]) >= 7


def test_saving_an_empty_edit_list_removes_the_per_picture_file(tmp_path):
    meta = _trial(tmp_path / "recordings")
    rv = rl.Review.load(meta)
    rv.add_dot_edit("anchor", "manual", rows=[{"frame": 1, "t_s": 0.0, "x": 1.0, "y": 2.0}])
    rv.save()
    assert rl.dots_csv_path(meta).is_file()
    rv.reset_all()
    rv.save()
    assert not rl.dots_csv_path(meta).exists() and rl.review_json_path(meta).is_file()


def test_saving_a_review_never_changes_the_trials_own_files(tmp_path):
    meta = _trial(tmp_path / "recordings")
    before = _listing(meta.parent)
    contents = {p.name: p.read_bytes() for p in meta.parent.iterdir()}
    rv = rl.Review.load(meta)
    rv.set_zero(0.02, "typed", {"n_per_v": 100.0, "regression_id": "", "from": "recorded"})
    rv.add_dot_edit("anchor", "hold", from_s=0.0, to_s=1.0, x=1.0, y=1.0)
    rv.save()
    assert _listing(meta.parent) == before and {p.name: p.read_bytes() for p in meta.parent.iterdir()} == contents


def test_the_verdict_made_in_review_goes_to_the_review_layer_and_the_one_on_site_stays_with_the_trial(tmp_path):
    from limbtrack import verdict
    meta = _trial(tmp_path / "recordings")
    verdict.write_verdict(meta, verdict.COMPROMISED, ["wrong_dots"], "on site", where=verdict.ON_SITE)
    site_file = verdict.verdict_path(meta)
    site_bytes = site_file.read_bytes()
    v = verdict.write_verdict(meta, verdict.VALID, where=verdict.IN_REVIEW)
    assert verdict.read_verdict(meta)["verdict"] == "valid" and v["history"][0]["verdict"] == "compromised"
    assert site_file.read_bytes() == site_bytes                                                 # the on-site file is exactly as it was
    assert verdict.review_verdict_path(meta).is_file() and rl.review_dir(meta) in verdict.review_verdict_path(meta).parents
    verdict.write_verdict(meta, None, where=verdict.IN_REVIEW)                                  # taken back in Review
    assert verdict.read_verdict(meta)["verdict"] is None and len(verdict.read_verdict(meta)["history"]) == 2
