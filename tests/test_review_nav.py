"""Finding a participant's trials for the Review tab (limbtrack/review_nav.py, 0.29.0): from meta.json and verdict.json only, no video."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from limbtrack import review_nav as nav  # noqa: E402
from limbtrack import verdict as verdicts  # noqa: E402


def _trial(folder, name, **meta):
    """A trial's meta.json only (all the drop-down needs)."""
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}_meta.json"
    base = {"participant": folder.name, "test_key": "knee_extension", "test_label": "Knee extension (MVIC)", "target_deg": 20,
            "position_number": 1, "trial_number": 1, "status": "complete"}
    base.update(meta)
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_trials_come_in_the_order_they_were_recorded_even_across_positions_and_with_a_same_second_suffix(tmp_path):
    f = tmp_path / "P01"
    _trial(f, "P01_02_knee_extension_40deg_T1_20261028-114500")
    _trial(f, "P01_01_knee_extension_20deg_T2_20261028-114202-2", trial_number=2)
    _trial(f, "P01_01_knee_extension_20deg_T2_20261028-114202", trial_number=2)
    _trial(f, "P01_01_knee_extension_20deg_T1_20261028-113000", trial_number=1)
    assert [e.base for e in nav.list_trials(f)] == [
        "P01_01_knee_extension_20deg_T1_20261028-113000", "P01_01_knee_extension_20deg_T2_20261028-114202",
        "P01_01_knee_extension_20deg_T2_20261028-114202-2", "P01_02_knee_extension_40deg_T1_20261028-114500"]


def test_the_label_names_position_test_angle_trial_time_and_verdict(tmp_path):
    p = _trial(tmp_path / "P01", "P01_03_knee_extension_20deg_T2_20261028-122701", position_number=3, trial_number=2)
    e = nav.read_entry(p)
    assert e.label == "03 · Knee extension · 20° · T2 · 12:27 · ○ not judged"
    verdicts.write_verdict(p, verdicts.VALID)
    e.refresh_verdict()
    assert e.label.endswith("12:27 · ✓ valid")
    verdicts.write_verdict(p, verdicts.COMPROMISED, ["bad_force"])
    assert nav.read_entry(p).label.endswith("✗ compromised")


def test_a_test_without_a_target_and_an_old_file_name_still_get_a_sensible_label(tmp_path):
    p = _trial(tmp_path / "P11", "P11_hip_extension_k0_0deg_20261015-120009", test_label="Hip extension", target_deg=None,
               position_number=None, trial_number=None)
    e = nav.read_entry(p)
    assert e.label == "Hip extension · 12:00 · ○ not judged"
    assert e.started == "2026-10-15T12:00:09"


def test_an_unreadable_trial_is_listed_with_the_reason_not_dropped(tmp_path):
    f = tmp_path / "P01"
    _trial(f, "P01_01_knee_extension_20deg_T1_20261028-113000")
    (f / "P01_01_knee_extension_20deg_T2_20261028-114000_meta.json").write_text("{not json", encoding="utf-8")
    entries = nav.list_trials(f)
    assert [e.readable for e in entries] == [True, False]
    assert entries[1].error and "cannot be read" in entries[1].label and entries[1].base.endswith("20261028-114000")
    assert nav.counts(entries) == {"valid": 0, "compromised": 0, "not judged": 1, "unreadable": 1}


def test_an_unfinished_trial_is_named(tmp_path):
    p = _trial(tmp_path / "P01", "P01_01_knee_extension_20deg_T1_20261028-113000", status="recording")
    e = nav.read_entry(p)
    assert e.unfinished and e.label.endswith("unfinished")


def test_participants_are_the_subfolders_with_trials_in_natural_order(tmp_path):
    for name in ("P10", "P2", "P01"):
        _trial(tmp_path / name, f"{name}_a_20261028-113000")
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    _trial(tmp_path / "P2", "P2_b_20261028-114000")
    got = nav.participant_folders(tmp_path)
    assert [(n, c) for n, _, c in got] == [("P01", 1), ("P2", 2), ("P10", 1)]
    assert nav.participant_folders(tmp_path / "missing") == []


def test_open_folder_accepts_a_participant_or_a_folder_of_participants(tmp_path):
    _trial(tmp_path / "Pilot" / "P01", "P01_a_20261028-113000")
    _trial(tmp_path / "Pilot" / "P02", "P02_a_20261028-113000")
    assert nav.resolve_folder(tmp_path / "Pilot" / "P01")[0] == "participant"
    kind, folders = nav.resolve_folder(tmp_path / "Pilot")
    assert kind == "participants" and [Path(p).name for p in folders] == ["P01", "P02"]
    assert nav.resolve_folder(tmp_path / "nothing") == ("none", [])
    (tmp_path / "blank").mkdir()
    assert nav.resolve_folder(tmp_path / "blank") == ("none", [])


def test_next_not_judged_skips_judged_and_unreadable_trials_and_wraps(tmp_path):
    f = tmp_path / "P01"
    paths = [_trial(f, f"P01_a_T{i}_20261028-1130{i:02d}", trial_number=i) for i in range(1, 6)]
    (f / "P01_a_T6_20261028-113006_meta.json").write_text("nope", encoding="utf-8")
    verdicts.write_verdict(paths[1], verdicts.VALID)
    verdicts.write_verdict(paths[2], verdicts.COMPROMISED, ["other"])
    entries = nav.list_trials(f)                                  # T1 open, T2 valid, T3 compromised, T4 open, T5 open, T6 unreadable
    assert nav.next_unjudged(entries, 0) == 3
    assert nav.next_unjudged(entries, 3) == 4
    assert nav.next_unjudged(entries, 4) == 0                     # wraps over the unreadable one
    assert nav.next_unjudged(entries, -1) == 0
    for p in (paths[0], paths[3], paths[4]):
        verdicts.write_verdict(p, verdicts.VALID)
    entries = nav.list_trials(f)
    assert nav.next_unjudged(entries, 0) is None
    assert nav.next_unjudged([], -1) is None
    assert nav.counts(entries) == {"valid": 4, "compromised": 1, "not judged": 0, "unreadable": 1}
