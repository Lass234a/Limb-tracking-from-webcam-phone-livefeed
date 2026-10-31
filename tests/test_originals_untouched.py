"""The promise of 0.30.0: whatever is done in Review, the recorded files of a trial stay byte-identical and nothing new appears beside them.
A scripted Review session on a real recording (zero, dot fixes of every kind, verdict, reference readings, every export, the overlay video,
the overview) is compared, file by file, with the folder as it was before."""
import hashlib
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import dot_edits, reference  # noqa: E402
from limbtrack import review_layer as rl  # noqa: E402
from limbtrack.review import ReferenceDialog, ReferenceValidationDialog, ReviewTab  # noqa: E402

from tests import test_force as T  # noqa: E402


def _manifest(folder):
    """{relative path: (size, mtime_ns, sha256)} of every file under `folder`."""
    out = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(folder))] = (st.st_size, st.st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def test_a_full_review_session_leaves_the_recorded_files_untouched(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(("info", a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warning", a[2])))
    opened = []
    monkeypatch.setattr(os, "startfile", lambda p: opened.append(p), raising=False)
    eng, paths = T.recording(tmp_path)
    meta = paths["meta.json"]
    folder = meta.parent
    before = _manifest(folder)
    assert len(before) >= 4 and any(k.endswith("_raw.mp4") for k in before)

    tab = ReviewTab(tmp_path)
    tab.load(meta)
    tr = tab.trial
    # force zero: typed, from a stretch, from the rest, and taken back
    tab.zero_spin.setValue(0.05)
    tab.apply_force_calibration()
    tab.zero_region.setRegion((float(tab.trial.force_t[0]), float(tab.trial.force_t[0]) + 0.2))
    tab.zero_from_stretch()
    tab.zero_from_rest()
    tab.zero_reset()
    # dots: hold, mark lost, by hand, re-track, then take one back
    dot = "epicondyle"
    t = tab.trial.t
    tab.commit_dot_edit(dot, "hold", from_s=float(t[2]), to_s=float(t[6]), x=500.0, y=300.0)
    tab.commit_dot_edit(dot, "lost", from_s=float(t[8]), to_s=float(t[9]))
    tab.commit_dot_edit(dot, "manual", rows=[{"frame": 11, "t_s": float(t[11]), "x": 520.0, "y": 310.0}, {"frame": 14, "t_s": float(t[14]), "x": 530.0, "y": 320.0}])
    seed = (float(tab.trial.tracks[dot][0][0]), float(tab.trial.tracks[dot][1][0]))
    rows, why = dot_edits.retrack(tab.trial, "malleolus", 0, (float(tab.trial.tracks["malleolus"][0][0]), float(tab.trial.tracks["malleolus"][1][0])), "forward")
    tab.rewind_video()
    assert rows and not why
    tab.commit_dot_edit("malleolus", "retrack", rows=rows, seed_frame=0, seed_xy=[1.0, 2.0], direction="forward")
    tab.undo_dot_edit(dot)
    # what a person types and judges in Review
    dlg = ReferenceDialog(tab.trial, tab)
    dlg.boxes["knee"].setText("58 59")
    dlg._save()
    tab.verdict_bar.btn_valid.click()
    # the other places Review looks at and writes
    tab.set_source("original")
    tab.set_source("reviewed")
    tab.export()
    tab.export_session()
    tab.export_session(valid_only=True)
    tab.export_overlay_video()
    tab.show_view("overview")
    assert tab.overview.wait_until_done(120000)
    tab.show_view("trial")
    tab.open_review_folder()
    ReferenceValidationDialog(tmp_path, reference.collect_rows(tmp_path), tab).save_table()           # (the dialog itself is modal: its save button is pressed in code)
    tab.trial.close()
    tab.overview.shutdown()

    assert not [m for kind, m in shown if kind == "warning"], shown                                      # nothing went wrong on the way
    after = _manifest(folder)
    assert after == before, f"changed: {[k for k in set(before) | set(after) if before.get(k) != after.get(k)]}"       # byte for byte, no new file
    review_root = tmp_path / "review"
    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file() and p.is_relative_to(review_root))
    base = tab.trial.base
    for must in (f"review/P01/{base}/review.json", f"review/P01/{base}/dots_edit.csv", f"review/P01/{base}/verdict.json", f"review/P01/{base}/reference.json",
                 f"review/P01/{base}/reviewed_data.csv", f"review/P01/{base}/reviewed_force.csv", f"review/P01/{base}/{base}_overlay_reviewed.mp4"):
        assert any(w.replace("\\", "/") == must for w in written), (must, written)
    assert any("exports/P01_session_summary_reviewed.csv" in w.replace("\\", "/") for w in written)
    outside = [str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file() and not p.is_relative_to(review_root) and not p.is_relative_to(folder)]
    assert outside == []                                                                                 # and nothing anywhere else in the recordings folder
    assert opened and Path(opened[-1]) == review_root / "P01"


def test_the_review_code_has_no_other_way_to_write_a_trials_folder(tmp_path):
    """Every file a Review function writes goes through limbtrack.review_layer: a path beside a trial's own files is refused, whoever asks."""
    eng, paths = T.recording(tmp_path)
    meta = paths["meta.json"]
    before = _manifest(meta.parent)
    for attempt in (lambda: rl.write_json(meta.parent / "x.json", {}), lambda: rl.write_json(meta.with_name("y_verdict.json"), {}),
                    lambda: reference.write_sidecar(meta.parent / "nothing_meta.json", "goniometer", 1.0, {"knee": [1.0]}) and None):
        try:
            attempt()
        except rl.OriginalsProtected:
            pass
    assert _manifest(meta.parent) == before
    with pytest.raises(rl.OriginalsProtected):
        rl.guard(meta.parent / "z.csv")
