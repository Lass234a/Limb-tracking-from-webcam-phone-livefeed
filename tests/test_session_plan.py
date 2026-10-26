"""Session plan: every protocol position as a row, trial counts from the saved files, a click sets test and target."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from limbtrack.gui import TRIALS_PER_POSITION, MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.recorder import count_trials  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


@pytest.fixture
def win(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(TESTS, tmp_path)
    yield w
    w.close()


def row_of(live, key, tgt):
    return live.plan_rows.index((key, tgt))


def save_fake_trial(root, pid, key, tgt, stamp):
    folder = Path(root) / pid
    folder.mkdir(parents=True, exist_ok=True)
    t = "" if tgt is None else f"_{tgt:g}deg"
    (folder / f"{pid}_{key}{t}_{stamp}_meta.json").write_text("{}")


def test_one_row_per_position(win):
    n = sum(len(t.targets) or 1 for t in TESTS.values())
    assert win.live.plan.count() == len(win.live.plan_rows) == n


def test_count_trials_counts_only_that_position_and_participant(tmp_path):
    save_fake_trial(tmp_path, "P01", "knee_extension", 20, "20260101-000000")
    save_fake_trial(tmp_path, "P01", "knee_extension", 20, "20260101-000100")
    save_fake_trial(tmp_path, "P01", "knee_extension", 70, "20260101-000200")
    save_fake_trial(tmp_path, "P02", "knee_extension", 20, "20260101-000300")
    save_fake_trial(tmp_path, "P01", "belt_squat", None, "20260101-000400")
    save_fake_trial(tmp_path, "P01", "hip_extension_k90", 0, "20260101-000500")
    assert count_trials(tmp_path, "P01", "knee_extension", 20) == 2
    assert count_trials(tmp_path, "P01", "knee_extension", 70) == 1
    assert count_trials(tmp_path, "P02", "knee_extension", 20) == 1
    assert count_trials(tmp_path, "P01", "belt_squat", None) == 1
    assert count_trials(tmp_path, "P01", "hip_extension_k0", 0) == 0          # k0 is not a prefix problem for k90
    assert count_trials(tmp_path, "P09", "knee_extension", 20) == 0


def test_rows_show_counts_for_the_typed_participant(win, tmp_path):
    live = win.live
    assert "participant code" in live.trial_label.text()
    save_fake_trial(win.engine.recordings_dir, "P01", "knee_extension", 20, "20260101-000000")
    live.participant.setText("P01")
    first = live.plan.item(row_of(live, "knee_extension", 20)).text()
    other = live.plan.item(row_of(live, "knee_extension", 70)).text()
    assert first.strip().startswith(f"1/{TRIALS_PER_POSITION}") and other.strip().startswith(f"0/{TRIALS_PER_POSITION}")
    save_fake_trial(win.engine.recordings_dir, "P01", "knee_extension", 20, "20260101-000100")
    live.refresh_plan()
    assert live.plan.item(row_of(live, "knee_extension", 20)).text().strip().startswith("done")


def test_clicking_a_row_sets_test_target_and_the_next_trial_caption(win):
    live = win.live
    live.participant.setText("P01")
    save_fake_trial(win.engine.recordings_dir, "P01", "knee_flexion", 90, "20260101-000000")
    live.on_plan_clicked(live.plan.item(row_of(live, "knee_flexion", 90)))
    assert win.engine.test.key == "knee_flexion" and win.engine.target == 90
    assert live.plan.currentRow() == row_of(live, "knee_flexion", 90)
    assert f"trial 2 of {TRIALS_PER_POSITION}" in live.trial_label.text()
    live.on_plan_clicked(live.plan.item(row_of(live, "hip_extension_k90", 0)))
    assert win.engine.test.key == "hip_extension_k90" and win.engine.target == 0
    live.on_plan_clicked(live.plan.item(row_of(live, "belt_squat", 60)))     # two belt squat positions since 0.20.0 (90 first)
    assert win.engine.test.key == "belt_squat" and win.engine.target == 60
    assert row_of(live, "belt_squat", 90) == row_of(live, "belt_squat", 60) - 1 == len(live.plan_rows) - 2


def test_a_typed_target_outside_the_plan_is_not_counted(win):
    live = win.live
    live.participant.setText("P01")
    live.target_spin.setValue(33.0)
    assert "not a position of the plan" in live.trial_label.text()


def test_advanced_settings_are_folded_away_until_asked_for(win):
    live = win.live
    assert not live.adv_body.isVisibleTo(win) and live.btn_adv.text().startswith("Show")
    live.btn_adv.click()
    assert live.adv_body.isVisibleTo(win) and live.btn_adv.text() == "Hide advanced"
