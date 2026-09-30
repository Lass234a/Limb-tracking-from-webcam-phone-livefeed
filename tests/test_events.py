import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import ReviewTab, Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def _engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    pts = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)
    eng.process(render(pts, seed=0), 0.0)
    for n in eng.test.dots:
        assert eng.mark_dot(n, *(pts[n] + 2))
    eng.process(render(pts, seed=0), 0.01)
    assert eng.lock_position()
    return eng


def _record_hold_trial(eng):
    """3 s trial at 30 fps: knee 60 deg, drifts to 66 deg between 1.0 and 2.0 s, then holds.
    Markers: MVIC start at frame 30, marker at frame 45, MVIC end at frame 85."""
    eng.start_trial("P01")
    headers = []
    for i in range(90):
        knee = 60 + 6 * min(max((i - 30) / 30, 0), 1)
        if i == 30:
            assert eng.add_event("MVIC start")
        if i == 45:
            assert eng.add_event()
        if i == 85:
            assert eng.add_event("MVIC end")
        res = eng.process(render(leg_points(hip_flex=70, knee_flex=knee, ankle_df=5), seed=i), i / 30.0)
        headers.append(res.header)
    return eng.stop_trial(), headers


def test_events_are_ignored_when_not_recording(tmp_path):
    eng = _engine(tmp_path)
    assert eng.add_event("MVIC start") is False


def test_markers_land_in_csv_meta_and_banner(tmp_path):
    eng = _engine(tmp_path)
    paths, headers = _record_hold_trial(eng)
    rows = list(csv.DictReader(open(paths["data.csv"], newline="")))
    assert [(i, r["event"]) for i, r in enumerate(rows) if r["event"]] == [
        (30, "MVIC start"), (45, "marker 1"), (85, "MVIC end")]
    meta = json.loads(Path(paths["meta.json"]).read_text())
    assert [(e["frame"], e["label"]) for e in meta["events"]] == [(30, "MVIC start"), (45, "marker 1"), (85, "MVIC end")]
    assert meta["events"][0]["t_s"] == pytest.approx(1.0, abs=0.01)
    # banner shows for ~1.5 s after each marker, not before or long after
    assert not any("MARK" in " ".join(h) for h in headers[:30])
    assert "MARK: MVIC start" in " ".join(headers[35])
    assert "MARK: marker 1" in " ".join(headers[50])
    assert not any("MARK: MVIC start" in " ".join(h) for h in headers[80:])


def test_new_trial_restarts_marker_numbering(tmp_path):
    eng = _engine(tmp_path)
    _record_hold_trial(eng)
    paths, _ = _record_hold_trial(eng)
    meta = json.loads(Path(paths["meta.json"]).read_text())
    assert [e["label"] for e in meta["events"]] == ["MVIC start", "marker 1", "MVIC end"]


def test_review_hold_window_and_start_to_end_change(tmp_path):
    eng = _engine(tmp_path)
    paths, _ = _record_hold_trial(eng)
    tr = Trial(paths["meta.json"])
    assert [lab for _, _, lab in tr.events] == ["MVIC start", "marker 1", "MVIC end"]
    t0, t1 = tr.hold_window()
    assert t0 == pytest.approx(1.0, abs=0.01) and t1 == pytest.approx(85 / 30, abs=0.01)

    whole = {r["angle"]: r for r in tr.summary(0, 3)}
    knee = whole["Knee flexion"]
    assert knee["change_start_to_end_deg"] == pytest.approx(6.0, abs=0.7)   # 60 -> 66 deg
    hip = whole["Hip flex (+) / ext (-)"]
    assert abs(hip["change_start_to_end_deg"]) < 0.7                        # untouched joint

    hold = {r["angle"]: r for r in tr.summary(t0, t1)}
    assert hold["Knee flexion"]["change_start_to_end_deg"] == pytest.approx(6.0, abs=0.7)


def test_review_tab_markers_ui(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    eng = _engine(tmp_path)
    paths, _ = _record_hold_trial(eng)
    tab = ReviewTab(tmp_path)
    tab.load(str(paths["meta.json"]))
    assert tab.event_combo.count() == 3 and tab.btn_hold.isEnabled()
    tab.event_combo.setCurrentIndex(2)
    tab.jump_to_event(2)
    assert tab.slider.value() == 85
    tab.snap_window()
    lo, hi = tab.window.getRegion()
    assert lo == pytest.approx(1.0, abs=0.01) and hi == pytest.approx(85 / 30, abs=0.01)
    assert "start>end" in tab.stats.text()
    tab.export()
    out = list(Path(paths["meta.json"]).parent.glob("*_summary.csv"))
    assert len(out) == 1
    head = open(out[0], newline="").readline()
    assert "change_start_to_end_deg" in head and "window_start_s" in head


def test_trial_without_markers_still_opens(tmp_path):
    eng = _engine(tmp_path)
    eng.start_trial("P01")
    for i in range(10):
        eng.process(render(leg_points(hip_flex=70, knee_flex=60, ankle_df=5), seed=i), i / 30.0)
    paths = eng.stop_trial()
    tr = Trial(paths["meta.json"])
    assert tr.events == [] and tr.hold_window() is None
