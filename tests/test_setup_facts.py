"""0.34.0 (F3): set-up facts of a position (strap distance from a named landmark, which supports held the participant, the seat/back angle),
typed in the Position section, saved in every trial's meta.json as `setup`, prefilled for the next trial of the same participant, test
and target, never a reason to refuse Record.

Everything here writes and reads real files (meta.json, settings.json) or uses the real widgets offscreen."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import session_memory  # noqa: E402
from limbtrack import setup_facts as sf  # noqa: E402
from limbtrack.engine import Engine  # noqa: E402
from limbtrack.gui import MainWindow  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")
P = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)

FACTS = {"strap_landmark": "lateral_malleolus", "strap_from_landmark_cm": 4.5, "supports": ["thigh_strap", "hands_hold"], "seat_back_angle_deg": 90.0}


def engine(tmp_path, code="P01", test="knee_extension", target=70.0):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.set_test(test)
    eng.target = target
    eng.participant_code = code
    eng.process(render(P, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))
    for i in range(5):
        eng.process(render(P, seed=i), 0.1 + i / 30)
    return eng


def record(eng, frames=5):
    paths = eng.start_trial(eng.participant_code)
    for i in range(frames):
        eng.process(render(P, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    return paths


def meta_of(paths):
    return json.loads(paths["meta.json"].read_text())


# ------------------------------------------------------------------------------------------------ the pure part
def test_the_numbers_accept_a_comma_and_a_unit_and_empty_and_refuse_what_is_not_plausible():
    assert sf.parse_distance("4,5 cm") == (4.5, None) and sf.parse_distance("") == (None, None) and sf.parse_distance("0") == (0.0, None)
    v, why = sf.parse_distance("450")                              # millimetres typed into a centimetre box
    assert v is None and "outside the plausible" in why
    assert sf.parse_distance("-1")[0] is None and sf.parse_distance("near the ankle")[0] is None and sf.parse_distance("inf")[0] is None
    assert sf.parse_angle("90") == (90.0, None) and sf.parse_angle("45 deg") == (45.0, None) and sf.parse_angle("45°") == (45.0, None)
    assert sf.parse_angle("200")[0] is None and sf.parse_angle("-5")[0] is None


def test_the_landmarks_include_the_four_of_the_plan_and_the_two_the_protocol_names():
    assert {"lateral_epicondyle", "lateral_malleolus", "trochanter", "other"} <= set(sf.LANDMARK_KEYS)
    assert {"patellar_border", "metatarsal_heads"} <= set(sf.LANDMARK_KEYS)                      # protocol 2.6: strap locations relative to these
    assert sf.LANDMARK_KEYS[-1] == "other"


def test_clean_keeps_the_fixed_shape_and_refuses_unknown_things():
    c = sf.clean({"strap_landmark": "other", "strap_landmark_other": "  tibial tuberosity ", "strap_from_landmark_cm": 12.34,
                  "supports": ["hands_hold", "pelvic_belt", "pelvic_belt"]})
    assert c == {"strap_landmark": "other", "strap_landmark_other": "tibial tuberosity", "strap_from_landmark_cm": 12.3,
                 "supports": ["pelvic_belt", "hands_hold"], "seat_back_angle_deg": None}               # one each, in the fixed order
    assert sf.clean({"strap_landmark": "trochanter", "strap_landmark_other": "left over"})["strap_landmark_other"] is None   # a word for 'other' means nothing now
    assert sf.clean({}) == sf.blank() and sf.is_empty({}) and sf.is_empty(sf.blank())
    for bad in ({"strap_landmark": "knee"}, {"supports": ["belt"]}, {"strap_from_landmark_cm": 500}, {"seat_back_angle_deg": -3},
                {"strap_from_landmark_cm": True}, {"colour": "red"}, {"seat_back_angle_deg": float("nan")}):
        with pytest.raises(ValueError):
            sf.clean(bad)


def test_the_free_landmark_text_follows_the_notes_rule_about_personal_data():
    for bad in ("jane@example.com", "12345678901", "010203-1234"):
        with pytest.raises(ValueError, match="identify"):
            sf.clean({"strap_landmark": "other", "strap_landmark_other": bad})
        with pytest.raises(ValueError):
            sf.meta_block({"strap_landmark": "other", "strap_landmark_other": bad})
    with pytest.raises(ValueError):
        sf.clean({"strap_landmark": "other", "strap_landmark_other": "a@b.c"})
    ok = sf.clean({"strap_landmark": "other", "strap_landmark_other": "tibial tuberosity 12"})
    assert ok["strap_landmark_other"] == "tibial tuberosity 12"


def test_the_meta_block_and_the_summary():
    assert sf.meta_block(None) is None and sf.meta_block(sf.blank()) is None
    b = sf.meta_block(FACTS, typed_at="2026-11-02T10:00:00", prefilled=True)
    assert b["strap_landmark"] == "lateral_malleolus" and b["prefilled"] is True and b["typed_at"] == "2026-11-02T10:00:00" and b["note"]
    assert sf.summary(None) == "not noted"
    text = sf.summary(FACTS)
    assert "strap 4.5 cm from lateral malleolus" in text and "thigh fixation strap" in text and "back 90 deg" in text


# ------------------------------------------------------------------------------------------------ the engine
def test_the_facts_belong_to_participant_test_and_target(tmp_path):
    eng = engine(tmp_path)
    assert eng.setup_facts is None and eng.setup_state is None
    eng.setup_facts = FACTS
    assert eng.setup_facts == sf.clean(FACTS)
    eng.target = 20.0                                              # another position of the same test
    assert eng.setup_facts is None
    eng.target = 70.0
    assert eng.setup_facts == sf.clean(FACTS)                      # back: still there
    eng.set_test("knee_flexion")                                   # another test
    eng.target = 20.0
    assert eng.setup_facts is None
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.participant_code = "P02"                                   # another participant
    assert eng.setup_facts is None
    eng.participant_code = "P01"
    assert eng.setup_facts == sf.clean(FACTS)
    eng.setup_facts = sf.blank()                                   # emptied = nothing noted
    assert eng.setup_facts is None and eng.setup_state is None
    with pytest.raises(ValueError):
        eng.setup_facts = {"strap_from_landmark_cm": 900}
    assert eng.setup_facts is None                                 # a refused entry changes nothing


def test_the_same_values_typed_again_do_not_count_as_a_change(tmp_path):
    eng = engine(tmp_path)
    eng.setup_facts = FACTS
    first = eng.setup_state
    time.sleep(1.1)
    eng.setup_facts = FACTS
    assert eng.setup_state == first
    eng.setup_facts = {**FACTS, "strap_from_landmark_cm": 5.0}
    assert eng.setup_state[0] != first[0] and eng.setup_state[1] is False


def test_a_trial_saves_the_facts_and_the_next_one_at_the_position_says_they_are_carried_over(tmp_path):
    eng = engine(tmp_path)
    eng.setup_facts = FACTS
    m1 = meta_of(record(eng))                                      # the real file
    s1 = m1["setup"]
    assert s1["strap_landmark"] == "lateral_malleolus" and s1["strap_from_landmark_cm"] == 4.5 and s1["seat_back_angle_deg"] == 90.0
    assert s1["supports"] == ["thigh_strap", "hands_hold"] and s1["prefilled"] is False and s1["typed_at"] and s1["note"]
    assert "changed_during_recording" not in m1
    assert eng.setup_facts == sf.clean(FACTS)                      # still there for the next trial
    m2 = meta_of(record(eng))
    assert m2["setup"]["prefilled"] is True and m2["setup"]["typed_at"] == s1["typed_at"]       # nobody typed anything for the second trial
    assert {k: v for k, v in m2["setup"].items() if k != "prefilled"} == {k: v for k, v in s1.items() if k != "prefilled"}
    eng.setup_facts = {**FACTS, "strap_from_landmark_cm": 6.0}     # the strap was moved: a change
    m3 = meta_of(record(eng))
    assert m3["setup"]["strap_from_landmark_cm"] == 6.0 and m3["setup"]["prefilled"] is False
    m4 = meta_of(record(eng))
    assert m4["setup"]["strap_from_landmark_cm"] == 6.0 and m4["setup"]["prefilled"] is True


def test_a_trial_without_facts_has_none_and_other_meta_keys_keep_their_meaning(tmp_path):
    eng = engine(tmp_path)
    meta = meta_of(record(eng))
    assert "setup" in meta and meta["setup"] is None
    assert meta["tested_leg"] is None and meta["participant"] == "P01" and meta["test_key"] == "knee_extension" and meta["target_deg"] == 70.0


def test_a_trial_keeps_the_facts_it_started_with_and_nothing_changes_while_recording(tmp_path):
    eng = engine(tmp_path)
    eng.setup_facts = FACTS
    paths = eng.start_trial("P01")
    with pytest.raises(RuntimeError):
        eng.setup_facts = {**FACTS, "strap_from_landmark_cm": 9.0}
    eng.participant_code = "P02"                                   # the next code typed during the recording
    assert eng.setup_facts == sf.clean(FACTS)
    for i in range(4):
        eng.process(render(P, seed=i), 1.0 + i / 30)
    eng.stop_trial()
    meta = meta_of(paths)
    assert meta["setup"]["strap_from_landmark_cm"] == 4.5 and "changed_during_recording" not in meta
    eng.participant_code = "P01"
    assert eng.setup_state[1] is True                              # P01's facts went into that trial


def test_a_recorder_that_fails_does_not_use_up_the_facts(tmp_path, monkeypatch):
    eng = engine(tmp_path)
    eng.setup_facts = FACTS
    import limbtrack.engine as engine_module

    def broken(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(engine_module, "TrialRecorder", broken)
    with pytest.raises(OSError):
        eng.start_trial("P01")
    assert eng.setup_state[1] is False                             # no trial was saved: the next one is not 'carried over'


def test_an_old_trial_without_the_key_still_opens_in_review(tmp_path):
    eng = engine(tmp_path)
    eng.setup_facts = FACTS
    paths = record(eng)
    meta = meta_of(paths)
    for key in ("setup", "participant_measures"):
        del meta[key]                                              # a trial recorded by 0.33.0 or earlier
    paths["meta.json"].write_text(json.dumps(meta), encoding="utf-8")
    tr = Trial(paths["meta.json"])
    assert tr.meta.get("setup") is None and len(tr.t) > 0


def test_the_remembered_list_survives_a_bad_entry(tmp_path):
    eng = engine(tmp_path)
    eng.setup_facts = FACTS
    eng.target = 20.0
    eng.setup_facts = {"seat_back_angle_deg": 80.0}
    good = eng.setups_all
    assert len(good) == 2 and {e["target"] for e in good} == {70.0, 20.0}
    eng.setups_all = good + [{"participant": "P09"}, {"participant": "P09", "test": "knee_extension", "target": 1.0, "facts": {"strap_landmark": "knee"}},
                            "x", {"participant": "", "test": "knee_extension", "target": 5.0, "facts": FACTS}]
    assert len(eng.setups_all) == 2                                # the bad and the code-less entries are dropped, the good ones stay
    eng.target = 70.0
    assert eng.setup_facts == sf.clean(FACTS)


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
    edit.setText(text)
    edit.textEdited.emit(text)


def fill(box, facts):
    """Do what a person does in the box: choose the landmark, type the numbers, tick the supports."""
    box.landmark.setCurrentIndex(box.landmark.findData(facts["strap_landmark"]))
    typed(box.distance, str(facts["strap_from_landmark_cm"]))
    typed(box.angle, str(facts["seat_back_angle_deg"]))
    for k, cb in box.checks.items():
        cb.setChecked(k in facts["supports"])


def test_the_box_is_in_the_position_section_and_waits_for_a_participant_code(win):
    live = win.live
    box = live.setup_box
    assert live.accordion.sections["position"].body.isAncestorOf(box) and not live.accordion.sections["participant"].body.isAncestorOf(box)
    live.participant.setText("")
    live.refresh_status()
    assert not box.distance.isEnabled() and not box.landmark.isEnabled()
    live.participant.setText("P01")
    live.refresh_status()
    assert box.distance.isEnabled() and box.landmark.isEnabled() and all(cb.isEnabled() for cb in box.checks.values())


def test_typing_is_kept_for_the_position_and_prefills_the_next_visit(win):
    live, e = win.live, win.engine
    box = live.setup_box
    live.participant.setText("P01")
    live._pick(live.test_combo, "knee_extension")
    live.target_spin.setValue(70.0)
    live.refresh_status()
    assert "Not noted" in box.status.text()
    fill(box, FACTS)
    assert e.setup_facts == sf.clean(FACTS)
    assert "Typed" in box.status.text()
    live.target_spin.setValue(20.0)                                # another position: empty
    live.refresh_status()
    assert e.setup_facts is None and box.distance.text() == "" and box.landmark.currentData() is None and not any(cb.isChecked() for cb in box.checks.values())
    live.target_spin.setValue(70.0)                                # back: what was typed is there
    live.refresh_status()
    assert box.distance.text() == "4.5" and box.angle.text() == "90" and box.landmark.currentData() == "lateral_malleolus"
    assert [k for k, cb in box.checks.items() if cb.isChecked()] == ["thigh_strap", "hands_hold"]
    live.participant.setText("P02")                                # another participant: empty
    live.refresh_status()
    assert box.distance.text() == "" and e.setup_facts is None


def test_other_asks_for_a_word_and_a_wrong_number_is_not_kept(win):
    live, e = win.live, win.engine
    box = live.setup_box
    live.participant.setText("P01")
    live.refresh_status()
    assert box.other.isHidden()
    box.landmark.setCurrentIndex(box.landmark.findData("other"))
    assert not box.other.isHidden()
    typed(box.other, "tibial tuberosity")
    typed(box.distance, "12")
    assert e.setup_facts["strap_landmark"] == "other" and e.setup_facts["strap_landmark_other"] == "tibial tuberosity"
    typed(box.distance, "120")                                     # centimetres? it is longer than a thigh
    assert e.setup_facts["strap_from_landmark_cm"] is None and "outside the plausible" in box.status.text()
    assert "fcd34d" in box.distance.styleSheet()
    typed(box.angle, "abc")
    assert "not a number" in box.status.text()
    typed(box.distance, "12")
    typed(box.angle, "")
    assert e.setup_facts["strap_from_landmark_cm"] == 12.0 and box.distance.styleSheet() == ""
    box.landmark.setCurrentIndex(0)                                # 'not noted' again: the word for 'other' goes
    assert e.setup_facts["strap_landmark"] is None and e.setup_facts["strap_landmark_other"] is None and box.other.isHidden()


def test_personal_data_typed_as_the_landmark_is_not_kept_and_the_box_says_why(win):
    live, e = win.live, win.engine
    box = live.setup_box
    live.participant.setText("P01")
    live.refresh_status()
    box.landmark.setCurrentIndex(box.landmark.findData("other"))
    typed(box.other, "tibial tuberosity")
    assert e.setup_facts["strap_landmark_other"] == "tibial tuberosity"
    typed(box.other, "jane@example.com")
    assert e.setup_facts["strap_landmark_other"] == "tibial tuberosity"        # the engine refused it: the earlier word stays
    assert "identify" in box.status.text()


def test_the_box_is_switched_off_while_recording(win):
    live = win.live
    live.participant.setText("P01")
    live.refresh_status()
    live.setup_box.refresh(True)
    assert not live.setup_box.distance.isEnabled() and not any(cb.isEnabled() for cb in live.setup_box.checks.values())
    live.setup_box.refresh(False)
    assert live.setup_box.distance.isEnabled()


@pytest.fixture
def clip(tmp_path):
    video = tmp_path / "leg.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(150):
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


def test_two_real_recordings_from_the_window_second_one_starts_with_the_first_ones_facts(win, clip):
    live, e, app = win.live, win.engine, win.app
    live.start_source(str(clip))
    assert pump(app, lambda: e._last_gray is not None)
    for name in e.test.dots:
        live.view.clicked.emit(float(P[name][0]) + 2, float(P[name][1]) + 2)
    assert pump(app, lambda: e.frame_index > 0)
    live.participant.setText("P01")
    live.leg_combo.setCurrentIndex(live.leg_combo.findData("right"))
    live._pick(live.test_combo, "knee_extension")
    live.target_spin.setValue(70.0)
    live.refresh_status()
    # an empty box never stops Record
    live.btn_lock.click()
    live.btn_rec.click()
    assert e.rec is not None
    assert pump(app, lambda: e.rec is not None and e.rec.frames > 5)
    live.btn_rec.click()
    first = sorted((live.recordings_dir / "P01").glob("*_meta.json"))
    assert len(first) == 1 and json.loads(first[0].read_text())["setup"] is None
    # now type the facts; the second trial of the position carries them, and so does the third without anyone typing again
    fill(live.setup_box, FACTS)
    time.sleep(1.1)                                                # (the file names carry the second)
    live.btn_rec.click()
    assert e.rec is not None
    assert pump(app, lambda: e.rec is not None and e.rec.frames > 5)
    live.btn_rec.click()
    time.sleep(1.1)
    live.refresh_status()
    live.btn_rec.click()
    assert e.rec is not None
    assert pump(app, lambda: e.rec is not None and e.rec.frames > 5)
    live.btn_rec.click()
    metas = [json.loads(p.read_text()) for p in sorted((live.recordings_dir / "P01").glob("*_meta.json"))]
    assert [m["setup"] is None for m in metas] == [True, False, False]
    assert metas[1]["setup"]["strap_from_landmark_cm"] == 4.5 and metas[1]["setup"]["prefilled"] is False
    assert metas[2]["setup"]["strap_from_landmark_cm"] == 4.5 and metas[2]["setup"]["prefilled"] is True


# ------------------------------------------------------------------------------------------------ remembered
def test_the_facts_are_remembered_with_a_recent_session_only(win):
    live = win.live
    entry = {"participant": "P07", "test": "knee_extension", "target": 70.0, "facts": sf.clean(FACTS), "typed_at": "2026-11-02T09:00:00", "used": True}
    recent = {"participant": "P07", "setups": [entry], "saved_at": time.time() - 60, "format": session_memory.FORMAT}
    live.restore_session(recent)
    win.engine.set_test("knee_extension")
    win.engine.target = 70.0
    assert live.participant.text() == "P07" and win.engine.setup_facts == sf.clean(FACTS) and win.engine.setup_state == ("2026-11-02T09:00:00", True)
    live.refresh_status()
    assert live.setup_box.distance.text() == "4.5" and "Carried over" in live.setup_box.status.text()
    old = {"participant": "P09", "setups": [{**entry, "participant": "P09"}], "saved_at": time.time() - 3 * 24 * 3600, "format": session_memory.FORMAT}
    win.engine.setups_all = []
    live.restore_session(old)
    assert win.engine.setups_all == []                             # a new day starts blank


def test_a_full_round_trip_through_settings_json_and_a_second_window(tmp_path):
    QApplication.instance() or QApplication([])
    w1 = MainWindow(TESTS, tmp_path)
    live = w1.live
    live.participant.setText("P05")
    live._pick(live.test_combo, "knee_flexion")
    live.target_spin.setValue(90.0)
    live.refresh_status()
    fill(live.setup_box, FACTS)
    live.save_session()
    w1.close()
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["session"]            # the real file
    assert len(saved["setups"]) == 1 and saved["setups"][0]["participant"] == "P05" and saved["setups"][0]["test"] == "knee_flexion"
    assert saved["setups"][0]["target"] == 90.0 and saved["setups"][0]["facts"]["strap_from_landmark_cm"] == 4.5
    w2 = MainWindow(TESTS, tmp_path)
    try:
        assert w2.live.participant.text() == "P05" and w2.engine.test.key == "knee_flexion" and w2.engine.target == 90.0
        assert w2.engine.setup_facts == sf.clean(FACTS)
        w2.live.refresh_status()
        assert w2.live.setup_box.distance.text() == "4.5"
    finally:
        w2.close()
