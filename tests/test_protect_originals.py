"""tools/protect_originals.py (0.30.0): a record of the recorded files outside their folders, a check against it, and read-only."""
import importlib.util
import json
import os
import stat
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("protect_originals", ROOT / "tools" / "protect_originals.py")
po = importlib.util.module_from_spec(spec)
spec.loader.exec_module(po)


def _recordings(tmp_path):
    root = tmp_path / "recordings"
    for pid in ("P01", "P02"):
        folder = root / pid
        folder.mkdir(parents=True)
        base = f"{pid}_02_knee_extension_70deg_T1_20261028-113014"
        (folder / f"{base}_meta.json").write_text(json.dumps({"participant": pid}))
        (folder / f"{base}_data.csv").write_text("a,b\n1,2\n")
        (folder / f"{base}_raw.mp4").write_bytes(b"video" * 100)
    return root


def _listing(folder):
    return {p.name: (p.stat().st_size, p.read_bytes()) for p in Path(folder).iterdir()}


def test_the_record_is_made_outside_the_participant_folders_and_leaves_them_as_they_were(tmp_path, capsys):
    root = _recordings(tmp_path)
    before = {pid: _listing(root / pid) for pid in ("P01", "P02")}
    assert po.main(["manifest", str(root)]) == 0
    assert {pid: _listing(root / pid) for pid in ("P01", "P02")} == before                      # not a file added or changed in them
    m = root / "review" / "P01" / "MANIFEST.sha256"
    assert m.is_file() and sum(1 for line in m.read_text().splitlines() if not line.startswith("#")) == 3
    assert "P01: 3 files recorded" in capsys.readouterr().out


def test_verify_says_unchanged_and_then_names_a_changed_a_missing_and_a_new_file(tmp_path, capsys):
    root = _recordings(tmp_path)
    po.main(["manifest", str(root)])
    capsys.readouterr()
    assert po.main(["verify", str(root)]) == 0 and capsys.readouterr().out.count("unchanged") == 2
    (root / "P01" / "P01_02_knee_extension_70deg_T1_20261028-113014_data.csv").write_text("a,b\n9,9\n")      # same size, other bytes
    (root / "P01" / "P01_02_knee_extension_70deg_T1_20261028-113014_raw.mp4").unlink()
    (root / "P01" / "stray.csv").write_text("x")
    assert po.main(["verify", str(root)]) == 1
    out = capsys.readouterr().out
    assert "CHANGED ['P01_02_knee_extension_70deg_T1_20261028-113014_data.csv']" in out and "MISSING ['P01_02_knee_extension_70deg_T1_20261028-113014_raw.mp4']" in out
    assert "new files ['stray.csv']" in out and "P02: unchanged" in out


def test_verify_without_a_record_says_so_and_a_single_participant_folder_works(tmp_path, capsys):
    root = _recordings(tmp_path)
    assert po.main(["verify", str(root / "P02")]) == 1 and "no manifest yet" in capsys.readouterr().out
    assert po.main(["manifest", str(root / "P02")]) == 0
    assert (root / "review" / "P02" / "MANIFEST.sha256").is_file() and not (root / "review" / "P01").exists()
    assert po.main(["verify", str(tmp_path / "recordings" / "review")]) == 2                    # no participant folder there


def test_readonly_marks_the_recorded_files_and_can_be_undone(tmp_path):
    root = _recordings(tmp_path)
    f = root / "P01" / "P01_02_knee_extension_70deg_T1_20261028-113014_data.csv"
    assert po.main(["readonly", str(root / "P01")]) == 0
    assert not os.access(f, os.W_OK) and not (f.stat().st_mode & stat.S_IWRITE)
    assert po.main(["readonly", str(root / "P01"), "--writable"]) == 0
    assert os.access(f, os.W_OK)
