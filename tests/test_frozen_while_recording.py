"""CR-01 (+GUI-02, COD-02, DAT-02, TST-03): nothing that changes the measurement may change during a recording."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
PTS = leg_points(hip_flex=70, knee_flex=60, ankle_df=5)


def recording_engine(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 60.0
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(5):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    assert eng.lock_position()
    paths = eng.start_trial("P01")
    eng.process(render(PTS, seed=9), 0.5)
    return eng, paths


@pytest.mark.parametrize("attr, value", [("target", 70.0), ("tolerance", 0.5), ("neighbour_tolerance", 9.0),
                                         ("facing_right", False), ("mirror", True)])
def test_settings_cannot_be_set_while_recording(tmp_path, attr, value):
    eng, _ = recording_engine(tmp_path)
    before = getattr(eng, attr)
    with pytest.raises(RuntimeError, match="Stop recording"):
        setattr(eng, attr, value)
    assert getattr(eng, attr) == before and eng.rec is not None
    eng.stop_trial()
    setattr(eng, attr, value)                                          # allowed again afterwards
    assert getattr(eng, attr) == value


def test_changing_test_colour_dots_or_the_lock_is_refused_while_recording(tmp_path):
    eng, _ = recording_engine(tmp_path)
    for call in (lambda: eng.set_test("knee_flexion"), lambda: eng.set_dot_kind("white"), eng.reset_dots, eng.lock_position):
        with pytest.raises(RuntimeError, match="Stop recording"):
            call()
    assert eng.test.key == "knee_extension" and eng.kind == "auto" and eng.locked and eng.rec is not None
    assert eng.trackers.is_ready()                                     # the marked dots survived
    assert eng.mark_dot("trochanter", *(PTS["trochanter"] + 2)) is False
    assert "Stop recording" in eng.mark_problem
    eng.stop_trial()


def test_meta_reports_the_settings_at_the_start_and_lists_any_change(tmp_path):
    eng, paths = recording_engine(tmp_path / "a")
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["target_deg"] == 60.0 and meta["facing"] == "right" and meta["position_locked"] is True
    assert "changed_during_recording" not in meta and meta["test_key"] == "knee_extension"

    eng, paths = recording_engine(tmp_path / "b")
    eng._target, eng._facing_right = 99.0, False                        # a change that slipped past the guards
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["target_deg"] == 60.0 and meta["facing"] == "right"      # the start values are what is reported
    assert meta["changed_during_recording"]["target_deg"] == [60.0, 99.0]
    assert meta["changed_during_recording"]["facing"] == ["right", "left"]


# ---- the window
def wheel(widget):
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)


def test_the_mouse_wheel_does_not_change_the_settings_without_focus(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    before = (live.test_combo.currentIndex(), live.kind_combo.currentIndex(), live.facing_combo.currentIndex(),
              live.target_spin.value(), live.tol_spin.value(), live.ntol_spin.value())
    for w in (live.test_combo, live.kind_combo, live.facing_combo, live.target_spin, live.tol_spin, live.ntol_spin):
        wheel(w)
    after = (live.test_combo.currentIndex(), live.kind_combo.currentIndex(), live.facing_combo.currentIndex(),
             live.target_spin.value(), live.tol_spin.value(), live.ntol_spin.value())
    assert before == after
    win.close()


def test_the_controls_are_disabled_while_recording_and_a_refused_change_is_put_back(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    live = win.live
    eng = win.engine
    eng.process(render(PTS, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(PTS[n] + 2))
    for i in range(5):
        eng.process(render(PTS, seed=i), 0.1 + i / 30)
    eng.start_trial("P01")
    live._rec_started = time.perf_counter()
    live.refresh_status()
    controls = (live.test_combo, live.kind_combo, live.facing_combo, live.target_spin, live.tol_spin, live.ntol_spin,
                live.suggest, live.btn_remark, live.btn_lock)
    assert not any(w.isEnabled() for w in controls)

    # even if a change got through to the engine (keyboard on a combo that was somehow enabled), it is refused and undone
    for w in controls[:6]:
        w.setEnabled(True)
    test_before, facing_before, target_before = live.test_combo.currentIndex(), live.facing_combo.currentIndex(), eng.target
    live.test_combo.setCurrentIndex(2)
    live.facing_combo.setCurrentIndex(1)
    live.target_spin.setValue(33.0)
    assert live.test_combo.currentIndex() == test_before and live.facing_combo.currentIndex() == facing_before
    assert eng.test.key == "knee_extension" and eng.facing_right is True and eng.target == target_before
    assert live.target_spin.value() == target_before and "Stop recording" in live.instruction.text()

    eng.stop_trial()
    live._rec_started = None
    live.refresh_status()
    assert all(w.isEnabled() for w in (live.test_combo, live.kind_combo, live.facing_combo, live.target_spin, live.btn_remark))
    win.close()
