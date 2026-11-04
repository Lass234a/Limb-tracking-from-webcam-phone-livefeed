"""0.34.0 (F2): participant measures typed in the Participant section (height, mass, body fat, thigh and shank length by tape, ASIS width),
kept per participant code like the tested leg, saved in every trial's meta.json as `participant_measures`, and the thigh and shank checked
against the picture when a ruler is marked (a quiet reminder above 10 %, never a reason to refuse Record).

Everything here writes and reads real files (meta.json, settings.json) or uses the real widgets offscreen."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import participant_measures as pm  # noqa: E402
from limbtrack import session_memory  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
P = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)          # thigh 190 px, shank (fibular head to malleolus) 135 px
RULER_PX, RULER_CM = 200.0, 40.0                              # 5 px per cm: the thigh is 38.0 cm in the picture, the shank 27.0 cm


def engine(tmp_path, ruler=True):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.process(render(P, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    if ruler:
        eng.scale.length_cm = RULER_CM
        eng.scale.ref_px = RULER_PX                            # both ruler dots marked (as tests/test_reminders.py does)
    for i in range(5):
        eng.process(render(P, seed=i), 0.1 + i / 30)
    return eng


def record(eng, code="P01", frames=6):
    eng.participant_code = code
    paths = eng.start_trial(code)
    for i in range(frames):
        eng.process(render(P, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    return paths


# ------------------------------------------------------------------------------------------------ the pure part
def test_typing_accepts_a_comma_and_a_unit_and_empty_and_refuses_what_is_not_plausible():
    assert pm.parse("height_cm", "182") == (182.0, None)
    assert pm.parse("height_cm", " 182,5 cm ") == (182.5, None)
    assert pm.parse("mass_kg", "78 kg") == (78.0, None)
    assert pm.parse("fat_pct", "18.2 %") == (18.2, None)
    assert pm.parse("thigh_length_cm", "") == (None, None) and pm.parse("thigh_length_cm", "   ") == (None, None)
    v, why = pm.parse("height_cm", "1.82")                       # metres typed into a centimetre box
    assert v is None and "outside the plausible" in why and "wrong unit" in why
    v, why = pm.parse("mass_kg", "heavy")
    assert v is None and "not a number" in why
    for bad in ("nan", "inf", "-inf"):
        assert pm.parse("mass_kg", bad)[0] is None and pm.parse("mass_kg", bad)[1]
    assert pm.parse("fat_pct", "-3")[0] is None and pm.parse("fat_pct", "75")[0] is None
    lo, hi = pm.RANGE["asis_width_cm"]
    assert pm.parse("asis_width_cm", str(lo))[0] == lo and pm.parse("asis_width_cm", str(hi))[0] == hi          # the limits themselves are fine


def test_clean_names_every_measure_and_refuses_unknown_or_implausible_ones():
    assert pm.clean({}) == {k: None for k in pm.KEYS} and pm.entered(pm.clean({})) == {}
    c = pm.clean({"height_cm": 182.04, "mass_kg": None})
    assert c["height_cm"] == 182.0 and c["mass_kg"] is None and list(c) == list(pm.KEYS)
    with pytest.raises(ValueError):
        pm.clean({"height_cm": 18.0})
    with pytest.raises(ValueError):
        pm.clean({"shoe_size": 44})
    with pytest.raises(ValueError):
        pm.clean({"height_cm": True})
    with pytest.raises(ValueError):
        pm.clean({"height_cm": float("nan")})


def test_the_picture_check_gives_the_difference_and_warns_only_above_ten_percent():
    xy = {"trochanter": (0.0, 0.0), "epicondyle": (190.0, 0.0), "fibular_head": (0.0, 10.0), "malleolus": (0.0, 145.0)}   # thigh 190 px, shank 135 px
    out = pm.picture_check({"thigh_length_cm": 38.0, "shank_length_cm": 30.0}, xy, 5.0)
    assert out["thigh"]["picture_cm"] == 38.0 and out["thigh"]["difference_pct"] == 0.0 and out["thigh"]["warn"] is False
    assert out["shank"]["picture_cm"] == 27.0 and out["shank"]["difference_pct"] == -10.0 and out["shank"]["warn"] is False      # exactly 10 %: no warning
    out = pm.picture_check({"thigh_length_cm": 33.0, "shank_length_cm": 29.9}, xy, 5.0)
    assert out["thigh"]["difference_pct"] == pytest.approx(15.2, abs=0.05) and out["thigh"]["warn"]
    assert out["shank"]["difference_pct"] == pytest.approx(-9.7, abs=0.05) and out["shank"]["warn"] is False
    text = pm.warning_text(out)
    assert text.startswith("Measures: thigh 33.0 cm by tape, 38.0 cm in the picture (+15 %).") and "shank" not in text
    assert pm.warning_text(pm.picture_check({"thigh_length_cm": 38.0}, xy, 5.0)) == ""
    assert pm.picture_check({"thigh_length_cm": 38.0}, xy, None) == {} and pm.picture_check({"thigh_length_cm": 38.0}, xy, 0) == {}   # no ruler
    assert pm.picture_check({}, xy, 5.0) == {}                                                                                  # nothing typed
    assert pm.picture_check({"thigh_length_cm": 38.0}, {"trochanter": (0.0, 0.0)}, 5.0) == {}                                  # a dot missing


def test_the_meta_block_is_none_when_nothing_was_typed():
    assert pm.meta_block({}, None) is None and pm.meta_block({"height_cm": None}) is None
    b = pm.meta_block({"height_cm": 182.0}, {"thigh": {"typed_cm": 1}})
    assert b["height_cm"] == 182.0 and b["mass_kg"] is None and b["picture_check"] == {"thigh": {"typed_cm": 1}} and b["note"]


# ------------------------------------------------------------------------------------------------ the engine
def test_the_measures_belong_to_the_participant_code(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.participant_code = "P01"
    assert eng.participant_measures == {}
    eng.set_participant_measure("height_cm", 182.0)
    eng.set_participant_measure("mass_kg", 78.5)
    eng.participant_code = "P02"
    assert eng.participant_measures == {}                          # the next participant does not get P01's numbers
    eng.set_participant_measure("height_cm", 165.0)
    eng.participant_code = "P0"                                    # a typing slip on the way back ...
    eng.participant_code = "P01"
    assert eng.participant_measures == {"height_cm": 182.0, "mass_kg": 78.5}       # ... and P01's numbers are there again
    assert eng.participant_measures_all == {"P01": {"height_cm": 182.0, "mass_kg": 78.5}, "P02": {"height_cm": 165.0}}
    eng.set_participant_measure("mass_kg", None)                   # clearing one
    assert eng.participant_measures == {"height_cm": 182.0}
    eng.set_participant_measure("height_cm", None)
    assert eng.participant_measures == {} and "P01" not in eng.participant_measures_all
    with pytest.raises(ValueError):
        eng.set_participant_measure("height_cm", 1.82)
    with pytest.raises(ValueError):
        eng.set_participant_measure("shoe_size", 44.0)
    assert eng.participant_measures == {}                          # a refused value changes nothing


def test_what_is_remembered_survives_a_bad_entry(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.participant_measures_all = {"P01": {"height_cm": 182.0}, "P02": {"height_cm": 18.0}, "P03": "x", "": {"height_cm": 170.0}, "P04": {}}
    assert eng.participant_measures_all == {"P01": {"height_cm": 182.0}}       # the bad, the empty and the code-less entries are dropped


def test_a_trial_saves_the_measures_and_the_picture_check_in_meta_json(tmp_path):
    eng = engine(tmp_path)
    eng.participant_code = "P01"
    eng.set_participant_measure("height_cm", 182.0)
    eng.set_participant_measure("thigh_length_cm", 33.0)           # the picture says 38.0 cm: +15 %
    eng.set_participant_measure("shank_length_cm", 27.5)           # the picture says 27.0 cm: -2 %
    paths = record(eng)
    meta = json.loads(paths["meta.json"].read_text())              # the real file
    block = meta["participant_measures"]
    assert block["height_cm"] == 182.0 and block["thigh_length_cm"] == 33.0 and block["shank_length_cm"] == 27.5
    assert block["mass_kg"] is None and block["fat_pct"] is None and block["asis_width_cm"] is None          # not typed = null, not invented
    chk = block["picture_check"]
    assert chk["thigh"]["picture_cm"] == pytest.approx(38.0, abs=0.2) and chk["thigh"]["warn"] is True       # (the tracked dots are a fraction of a pixel off)
    assert chk["thigh"]["difference_pct"] == pytest.approx(15.2, abs=0.8)
    assert chk["shank"]["warn"] is False and chk["shank"]["dots"] == ["fibular_head", "malleolus"]
    assert block["picture_check_warn_pct"] == 10.0 and "tape" in block["note"]
    assert "changed_during_recording" not in meta                  # the snapshot at the start is what the file reports
    assert meta["tested_leg"] is None and meta["setup"] is None    # (the neighbours keep their meaning)


def test_a_trial_without_measures_has_none_and_one_without_a_ruler_has_no_check(tmp_path):
    eng = engine(tmp_path / "a")
    meta = json.loads(record(eng)["meta.json"].read_text())
    assert "participant_measures" in meta and meta["participant_measures"] is None
    eng = engine(tmp_path / "b", ruler=False)
    eng.participant_code = "P01"
    eng.set_participant_measure("thigh_length_cm", 33.0)
    meta = json.loads(record(eng)["meta.json"].read_text())
    assert meta["participant_measures"]["thigh_length_cm"] == 33.0 and meta["participant_measures"]["picture_check"] is None


def test_the_trial_keeps_the_measures_it_started_with(tmp_path):
    eng = engine(tmp_path)
    eng.participant_code = "P01"
    eng.set_participant_measure("height_cm", 182.0)
    paths = eng.start_trial("P01")
    with pytest.raises(RuntimeError):
        eng.set_participant_measure("height_cm", 190.0)            # not while recording
    eng.participant_code = "P02"                                   # the next code typed during the recording
    assert eng.participant_measures == {"height_cm": 182.0}
    for i in range(4):
        eng.process(render(P, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["participant_measures"]["height_cm"] == 182.0 and "changed_during_recording" not in meta
    assert eng.participant_measures == {}                          # P02 has none


def test_an_old_trial_without_the_key_still_opens_in_review(tmp_path):
    eng = engine(tmp_path)
    paths = record(eng)
    meta = json.loads(paths["meta.json"].read_text())
    for key in ("participant_measures", "setup"):
        del meta[key]                                              # a trial recorded by 0.33.0 or earlier
    paths["meta.json"].write_text(json.dumps(meta), encoding="utf-8")
    tr = Trial(paths["meta.json"])
    assert tr.meta.get("participant_measures") is None and len(tr.t) > 0


# ------------------------------------------------------------------------------------------------ the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow(TESTS, tmp_path)
    w.app = app
    yield w
    w.close()                                                      # (its closeEvent stops a running video source; dropping the worker first left the thread running)


def typed(edit, text):
    """What typing does: setText does not emit textEdited, so emit it as the keyboard would."""
    edit.setText(text)
    edit.textEdited.emit(text)


def test_the_boxes_are_in_the_participant_section_and_wait_for_a_participant_code(win):
    live = win.live
    box = live.measures_box
    assert live.accordion.sections["participant"].body.isAncestorOf(box) and not live.accordion.sections["position"].body.isAncestorOf(box)
    assert list(box.edits) == list(pm.KEYS)
    live.participant.setText("")
    live.refresh_status()
    assert not any(e.isEnabled() for e in box.edits.values())      # no code: nothing to keep them for
    live.participant.setText("P01")
    live.refresh_status()
    assert all(e.isEnabled() for e in box.edits.values())


def test_typing_keeps_the_value_for_the_participant_and_changing_the_code_swaps_the_boxes(win):
    live = win.live
    box, e = live.measures_box, win.engine
    live.participant.setText("P01")
    live.refresh_status()
    typed(box.edits["height_cm"], "182,5")
    typed(box.edits["mass_kg"], "78 kg")
    assert e.participant_measures == {"height_cm": 182.5, "mass_kg": 78.0}
    live.participant.setText("P02")
    live.refresh_status()
    assert box.edits["height_cm"].text() == "" and box.edits["mass_kg"].text() == "" and e.participant_measures == {}
    typed(box.edits["height_cm"], "165")
    live.participant.setText("P01")
    live.refresh_status()
    assert box.edits["height_cm"].text() == "182.5" and box.edits["mass_kg"].text() == "78"     # P01's numbers are back
    typed(box.edits["mass_kg"], "")                                # emptied = not measured
    assert e.participant_measures == {"height_cm": 182.5}


def test_an_implausible_entry_is_not_kept_and_the_box_says_why_and_record_is_not_blocked(win):
    live = win.live
    box, e = live.measures_box, win.engine
    live.participant.setText("P01")
    live.refresh_status()
    typed(box.edits["height_cm"], "182")
    typed(box.edits["height_cm"], "1.82")                          # metres in a centimetre box
    assert e.participant_measures == {}                            # the old valid number is gone too: what is shown is what is kept
    assert "outside the plausible" in box.status.text() and not box.status.isHidden()
    assert "fcd34d" in box.edits["height_cm"].styleSheet()
    typed(box.edits["mass_kg"], "abc")
    assert "not a number" in box.status.text()
    typed(box.edits["height_cm"], "182")
    typed(box.edits["mass_kg"], "78")
    assert box.status.isHidden() and box.edits["height_cm"].styleSheet() == "" and e.participant_measures == {"height_cm": 182.0, "mass_kg": 78.0}
    # nothing typed, or typed wrongly, is a reason to refuse Record: the only measures-related reminders are quiet rows
    typed(box.edits["height_cm"], "1.82")
    assert not any("easure" in text for text, _ in live._open_reminders())


def test_the_boxes_are_switched_off_while_recording(win):
    live = win.live
    live.participant.setText("P01")
    live.refresh_status()
    live.measures_box.refresh(True)
    assert not any(e.isEnabled() for e in live.measures_box.edits.values())
    live.measures_box.refresh(False)
    assert all(e.isEnabled() for e in live.measures_box.edits.values())


class RunningCamera:
    is_file = False
    paused = False
    at_end = False
    source = 0
    camera = None
    status_note = ""

    def stop(self):
        pass

    def wait(self, ms=0):
        return True

    def isRunning(self):
        return True


@pytest.fixture
def ready(win):
    """A window with the synthetic leg marked, the 5 px per cm ruler set and P01 chosen: what the reminders need."""
    live, e = win.live, win.engine
    live.worker = RunningCamera()
    e.set_test("knee_extension")
    e.process(render(P, seed=0), 0.0)
    for n in e.active_dots:
        assert e.mark_dot(n, *(P[n] + 2))
    e.scale.length_cm = RULER_CM
    e.scale.ref_px = RULER_PX
    for i in range(4):
        e.process(render(P, seed=i), 0.1 + i / 30)
    live.participant.setText("P01")
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live.chk_priv_ok.setChecked(True)
    live.refresh_status()
    return win


def test_a_difference_above_ten_percent_is_a_quiet_row_that_noted_silences(ready):
    live, e = ready.live, ready.engine
    typed(live.measures_box.edits["thigh_length_cm"], "38")        # the picture says 38.0 cm
    live.update_steps()
    assert "measures" not in live.reminders.open_keys()
    typed(live.measures_box.edits["thigh_length_cm"], "33")        # +15 %
    live.update_steps()
    assert "measures" in live.reminders.open_keys() and not live.reminders.isHidden()
    label = live.reminders.rows["measures"][1].text()
    assert "thigh 33.0 cm by tape, 38.0 cm in the picture (+15 %)" in label and "Ruler in the leg's plane" in label
    assert any(t.startswith("Measures:") and in_file is False for t, in_file in live._open_reminders())     # it goes to the session log and meta.json
    assert live.strip.states["participant"][0] != "check" and live._first_todo != "measures"                    # not a step, not 'next'
    live.btn_rem_measures.click()                                  # Noted
    assert "measures" not in live.reminders.open_keys() and not any(t.startswith("Measures:") for t, _ in live._open_reminders())
    typed(live.measures_box.edits["thigh_length_cm"], "34")        # another number: asked again
    live.update_steps()
    assert "measures" in live.reminders.open_keys()
    typed(live.measures_box.edits["thigh_length_cm"], "37")        # within 10 % of 38.0: gone
    live.update_steps()
    assert "measures" not in live.reminders.open_keys()


def test_noted_belongs_to_the_position_and_the_row_shows_only_when_record_is_possible(ready):
    live, e = ready.live, ready.engine
    typed(live.measures_box.edits["thigh_length_cm"], "33")        # +15 % against the picture
    live.update_steps()
    assert "measures" in live.reminders.open_keys()
    here = e.target
    live.btn_rem_measures.click()                                  # Noted at this position
    assert "measures" not in live.reminders.open_keys()
    e.target = 70.0 if here != 70.0 else 20.0                      # another position of the same test: asked again
    live.update_steps()
    assert "measures" in live.reminders.open_keys()
    e.target = here                                                # back where Noted was pressed: still quiet
    live.update_steps()
    assert "measures" not in live.reminders.open_keys()
    live._measures_dismissed = None                                # (start again for the second half)
    live.update_steps()
    assert "measures" in live.reminders.open_keys()
    dot = e.trackers.trackers[next(iter(e.trackers.trackers))]
    dot.initialised = False                                        # a dot not marked: Record is not possible yet, so no reminder about it
    assert e.measures_picture_check()["thigh"]["warn"]            # (the check itself still says so; only the row waits)
    live.update_steps()
    assert "measures" not in live.reminders.open_keys()
    dot.initialised = True
    live.worker = None                                             # no camera running either
    live.update_steps()
    assert "measures" not in live.reminders.open_keys()
    live.worker = RunningCamera()
    live.update_steps()
    assert "measures" in live.reminders.open_keys()


def test_no_row_without_a_ruler_or_a_typed_length(ready):
    live, e = ready.live, ready.engine
    typed(live.measures_box.edits["height_cm"], "182")
    live.update_steps()
    assert "measures" not in live.reminders.open_keys()            # nothing to compare
    typed(live.measures_box.edits["thigh_length_cm"], "33")
    e.scale.ref_px = None                                          # ruler taken away
    live.update_steps()
    assert "measures" not in live.reminders.open_keys()


@pytest.fixture
def clip(tmp_path):
    video = tmp_path / "leg.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(120):
        w.write(render(P, seed=i))
    w.release()
    return video


def pump(app, cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_a_real_recording_from_the_window_saves_the_measures_even_when_the_tape_and_picture_disagree(win, clip):
    """Record is pressed with a thigh length 15 % off the picture: the trial is recorded, the reminder is in the session log and
    meta.json (reminders_open_at_start) and the check is in participant_measures."""
    live, e, app = win.live, win.engine, win.app
    live.start_source(str(clip))
    assert pump(app, lambda: e._last_gray is not None)
    for name in e.test.dots:
        live.view.clicked.emit(float(P[name][0]) + 2, float(P[name][1]) + 2)
    assert pump(app, lambda: e.frame_index > 0)
    e.scale.length_cm = RULER_CM
    e.scale.ref_px = RULER_PX
    live.participant.setText("P01")
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live.refresh_status()
    typed(live.measures_box.edits["thigh_length_cm"], "33")
    typed(live.measures_box.edits["height_cm"], "182")
    live.btn_lock.click()
    live.btn_rec.click()
    assert e.rec is not None                                       # Record was not refused
    assert pump(app, lambda: e.rec is not None and e.rec.frames > 5)
    live.btn_rec.click()
    metas = sorted((live.recordings_dir / "P01").glob("*_meta.json"))
    meta = json.loads(metas[-1].read_text())
    assert meta["participant_measures"]["thigh_length_cm"] == 33.0 and meta["participant_measures"]["height_cm"] == 182.0
    assert meta["participant_measures"]["picture_check"]["thigh"]["warn"] is True
    assert any(t.startswith("Measures:") for t in meta["reminders_open_at_start"])
    log = (live.recordings_dir / "session_log.csv")
    assert log.exists() and "Measures:" in log.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ remembered like the tested leg
def test_the_measures_are_remembered_with_a_recent_session_only(win):
    live = win.live
    recent = {"participant": "P07", "participant_measures": {"P07": {"height_cm": 175.0, "thigh_length_cm": 41.5}, "P08": {"mass_kg": 70.0}},
              "saved_at": time.time() - 60, "format": session_memory.FORMAT}
    live.restore_session(recent)
    assert live.participant.text() == "P07" and win.engine.participant_measures == {"height_cm": 175.0, "thigh_length_cm": 41.5}
    live.refresh_status()
    assert live.measures_box.edits["height_cm"].text() == "175" and live.measures_box.edits["thigh_length_cm"].text() == "41.5"
    old = {"participant": "P09", "participant_measures": {"P09": {"height_cm": 160.0}}, "saved_at": time.time() - 3 * 24 * 3600,
           "format": session_memory.FORMAT}
    win.engine.participant_measures_all = {}
    live.restore_session(old)
    assert win.engine.participant_measures_all == {}               # a new day starts blank, like the code and the leg


def test_a_full_round_trip_through_settings_json_and_a_second_window(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    w1 = MainWindow(TESTS, tmp_path)
    live = w1.live
    live.participant.setText("P05")
    live.refresh_status()
    typed(live.measures_box.edits["height_cm"], "181")
    typed(live.measures_box.edits["asis_width_cm"], "27,5")
    live.save_session()
    w1.close()
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["session"]            # the real file
    assert saved["participant_measures"] == {"P05": {"height_cm": 181.0, "asis_width_cm": 27.5}}
    w2 = MainWindow(TESTS, tmp_path)
    try:
        assert w2.live.participant.text() == "P05"
        assert w2.engine.participant_measures == {"height_cm": 181.0, "asis_width_cm": 27.5}
        w2.live.refresh_status()
        assert w2.live.measures_box.edits["asis_width_cm"].text() == "27.5"
    finally:
        w2.close()
