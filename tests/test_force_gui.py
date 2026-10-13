"""The Force box in the window: connect, read-out, zero and calibration, errors in plain English, release on close."""
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from limbtrack import gui as gui_mod  # noqa: E402
from limbtrack.force import ForceError, NiDaqForce  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P0 = leg_points(hip_flex=80, knee_flex=20, ankle_df=5)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)

    def pump(cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    yield app, win, win.live, warnings, pump, tmp_path
    win.close()


def connect_simulated(live):
    live.force_kind.setCurrentIndex(live.force_kind.findData("simulated"))
    live.btn_force.click()


def test_connect_shows_a_live_reading_and_disconnect_releases_it(setup):
    app, win, live, warnings, pump, tmp = setup
    assert "not connected" in live.force_label.text() and not live.btn_force_zero.isEnabled()
    connect_simulated(live)
    assert win.engine.force is live.force_source and live.btn_force.text() == "Disconnect"
    assert pump(lambda: "N" in live.force_label.text() or "V" in live.force_label.text())
    assert live.btn_force_zero.isEnabled() and not live.force_kind.isEnabled()
    src = live.force_source
    live.btn_force.click()
    assert live.force_source is None and win.engine.force is None and not src.running
    assert live.btn_force.text() == "Connect" and live.force_kind.isEnabled()


def test_zero_with_nothing_loaded_and_a_refused_calibration_say_why(setup, monkeypatch):
    app, win, live, warnings, pump, tmp = setup
    connect_simulated(live)
    live.force_source.calibration.n_per_v = None                          # start uncalibrated
    live.force_source.calibration.zero_v = None
    live.calibrate_force()
    assert "Zero" in live.force_info.text() and "first" in live.force_info.text()
    assert pump(lambda: live.force_source.average_volts(2.0)[1] >= 1100, timeout=10)       # 2 s of rest
    live.zero_force()
    assert "Zero set" in live.force_info.text() and live.force_source.calibration.zero_v is not None
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (5.0, True))
    live.calibrate_force()                                                # the signal did not move from zero: refused
    assert "moved only" in live.force_info.text() and not live.force_source.calibration.calibrated


def test_a_busy_or_missing_device_is_explained_and_nothing_is_connected(setup, monkeypatch):
    app, win, live, warnings, pump, tmp = setup

    def fail(self):
        raise ForceError("Dev1 is in use by another program (MATLAB). Close it, then connect again.")
    monkeypatch.setattr(NiDaqForce, "start", fail)
    live.force_kind.setCurrentIndex(live.force_kind.findData("ni"))
    live.btn_force.click()
    assert live.force_source is None and win.engine.force is None
    assert "in use by another program" in live.force_info.text() and live.force_info.styleSheet()
    live.force_kind.setCurrentIndex(live.force_kind.findData("off"))
    live.btn_force.click()
    assert "Choose a source" in live.force_info.text()


def test_connecting_uses_the_regression_slope_and_a_zero_is_not_kept_between_sessions(setup, monkeypatch):
    app, win, live, warnings, pump, tmp = setup
    from limbtrack.force import ForceCalibration, SimulatedForce
    assert not hasattr(live, "force_cal_path") and not list(tmp.glob("force_calibration*"))

    class Fake(SimulatedForce):
        kind = "NI DAQmx"
    monkeypatch.setattr(gui_mod, "NiDaqForce", lambda **kw: Fake(rest_s=30.0, offset_v=0.087))
    live.force_kind.setCurrentIndex(live.force_kind.findData("ni"))

    def connect():
        live.btn_force.click()
        live.force_source.calibration = ForceCalibration.load_cell()         # what connecting to the NI DAQ gives
        assert pump(lambda: live.force_source.average_volts(1.0)[1] >= 500, timeout=10)
        live.update_force_box()
    connect()
    src = live.force_source
    assert "Press Zero" in live.force_info.text() and src.latest_newtons() is None and "V" in live.force_label.text()
    live.btn_force_zero.click()                                              # the Z shortcut presses this button
    assert src.calibration.calibrated and abs(src.latest_newtons()) < 1.0
    assert "Zero set" in live.force_info.text() and "about 95 %" in live.force_info.text() and not live.force_info.styleSheet()
    live.btn_force_zero.click()                                              # zero again and again is fine, and says how far it moved
    assert "moved" in live.force_info.text()
    live.btn_force.click()                                                   # close and open again: the zero is gone, the slope is back
    assert live.force_source is None
    connect()
    assert not live.force_source.calibration.calibrated and live.force_source.calibration.has_slope
    assert "Press Zero" in live.force_info.text()
    assert not list(tmp.glob("force_calibration*"))                         # nothing was written to disk


def test_the_force_trace_can_be_shown_and_changing_the_mode_never_fails(setup):
    app, win, live, warnings, pump, tmp = setup
    live.trace_mode.setCurrentIndex(live.trace_mode.findData("force"))
    live.update_trace()                                                    # nothing connected: nothing drawn
    connect_simulated(live)
    assert pump(lambda: live.force_source.average_volts(1.0)[1] >= 500, timeout=10)
    live.update_trace()
    assert live._trace_items
    live.trace_mode.setCurrentIndex(live.trace_mode.findData("main"))
    live.update_trace()


def test_recording_with_force_is_refused_for_a_video_file_with_advice(setup):
    app, win, live, warnings, pump, tmp = setup
    video = tmp / "demo.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(60):
        w.write(render(P0, seed=i))
    w.release()
    live.start_source(str(video))
    assert pump(lambda: win.engine._last_gray is not None)
    for name in win.engine.test.dots:
        live.view.clicked.emit(float(P0[name][0]) + 2, float(P0[name][1]) + 2)
    assert win.engine.next_dot_to_mark() is None
    connect_simulated(live)
    live.btn_lock.click()
    live.participant.setText("P01")
    live.btn_rec.click()
    assert win.engine.rec is None and not live.btn_rec.isChecked()
    assert warnings and "live camera" in warnings[-1]
    live.stop_camera()


def test_closing_the_window_releases_the_daq(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow(load_protocol(ROOT / "protocol.json"), tmp_path)
    connect_simulated(win.live)
    src = win.live.force_source
    assert src.running
    win.close()
    assert not src.running and win.live.force_source is None


def test_the_read_out_shows_newtons_and_volts_together(setup):
    app, win, live, warnings, pump, tmp = setup
    connect_simulated(live)
    assert pump(lambda: " N " in live.force_label.text() and " V)" in live.force_label.text(), timeout=10)
    live.trace_mode.setCurrentIndex(live.trace_mode.findData("force_v"))
    assert pump(lambda: live.force_source.average_volts(1.0)[1] >= 500, timeout=10)
    live.update_trace()
    assert live._trace_items
