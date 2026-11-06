"""0.35.0 (F13): the study's frozen settings. `protocol.json` has a block `study_settings`; the program compares its own settings with it and says in
plain English which differ: an amber line in the Live tab (never stops Record), a line in session_log.csv at start-up and when the differences change,
`study_settings` in every trial's meta.json (an empty list when nothing differs, null without a block), an informational line in Review's checks, a
count column at the end of the AnyBody export, and `tools/freeze_check.py` for the study log.

Files are written and read back (protocol.json copies, meta.json, session_log.csv, the export CSV); the widgets are real, offscreen; the comparison reads
the real engine, a real simulated load cell, the real constants of the program (a changed constant is noticed), and the freeze tool runs git in a
real repository."""
import csv
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIMBTRACK_SILENT", "1")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from limbtrack import analysis, anybody, rest, study_settings as ss  # noqa: E402
from limbtrack import verdict as verdicts  # noqa: E402
from limbtrack.engine import SOFTWARE_VERSION, Engine  # noqa: E402
from limbtrack.force import SimulatedForce  # noqa: E402
from limbtrack.gui import STUDY_EXTRAS, MainWindow  # noqa: E402
from limbtrack.protocol import ProtocolError, load_protocol, load_study_settings, plan_positions, validate_protocol  # noqa: E402
from limbtrack.review import Trial, trial_row  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "protocol.json"
TESTS = load_protocol(PROTOCOL)
PLAN = plan_positions(TESTS)
BLOCK = load_study_settings(PROTOCOL)
P = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
sys.path.insert(0, str(ROOT / "tools"))
import freeze_check  # noqa: E402


def block_with(**settings):
    """The shipped block with some values changed (a copy)."""
    b = copy.deepcopy(BLOCK)
    b["settings"].update(settings)
    return b


def engine_like_the_study(tmp_path):
    """An engine whose settings are the study's (the ones that need no camera and no force)."""
    eng = Engine(TESTS, recordings_dir=tmp_path / "recordings")
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.set_dot_kind("black")
    eng.scale.length_cm = 100.0
    eng.study_block = BLOCK
    eng.study_extras = dict(STUDY_EXTRAS)
    return eng


def keys(items):
    return [d["key"] for d in items]


# ------------------------------------------------------------------------------------------------ the block in protocol.json
def test_the_shipped_block_is_valid_known_and_every_proposal_is_one_of_its_settings():
    assert validate_protocol(json.loads(PROTOCOL.read_text(encoding="utf-8"))) == []
    assert BLOCK is not None and len(BLOCK["settings"]) >= 20
    assert set(BLOCK["settings"]) <= set(ss.BY_KEY)
    assert set(BLOCK["proposals"]) <= set(BLOCK["settings"]) and BLOCK["proposals"]
    assert BLOCK["settings"]["dot_kind"] == "black" and BLOCK["settings"]["tolerance_deg"] == 5.0          # (the SOP's Table C1)
    assert BLOCK["settings"]["window_peak_ms"] == 500.0 and BLOCK["settings"]["force_smoothing_ms"] == 100.0
    assert load_protocol(PROTOCOL)                                  # the file still loads as a whole


def test_the_values_the_group_has_not_confirmed_are_named_and_the_sops_own_are_not():
    assert "auto_mvic_threshold_n" in BLOCK["proposals"]            # SOP Table C1: 25 N is marked 'confirm'
    assert "dot_kind" not in BLOCK["proposals"] and "rest_same_position_s" not in BLOCK["proposals"]


def test_a_file_without_the_block_loads_and_has_no_study_settings(tmp_path):
    data = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    del data["study_settings"]
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert load_study_settings(p) is None and load_protocol(p)


@pytest.mark.parametrize("change, expect", [
    (lambda b: b.update(extra=1), 'unknown field "extra"'),
    (lambda b: b.update(setings=b.pop("settings")), 'unknown field "setings"'),
    (lambda b: b["settings"].update(tolerence_deg=5.0), 'unknown setting "tolerence_deg" (did you mean "tolerance_deg"?)'),
    (lambda b: b["settings"].update(tolerance_deg="five"), "study_settings.settings.tolerance_deg: must be a number"),
    (lambda b: b["settings"].update(tolerance_deg=True), "must be a number"),
    (lambda b: b["settings"].update(tolerance_deg=99), "must be between 0.5 and 30 deg"),
    (lambda b: b["settings"].update(audio_cues="yes"), "audio_cues: must be true or false"),
    (lambda b: b["settings"].update(dot_kind="blak"), 'is not a dot colour; use auto, white or black (did you mean "black"?)'),
    (lambda b: b["settings"].update(tracker_settings_sha256="abc"), "64 characters of a SHA-256 hash"),
    (lambda b: b["settings"].update(camera_wanted=720), "camera_wanted: must be text"),
    (lambda b: b.update(proposals=["nothing_like_it"]), '"nothing_like_it" is not one of the frozen settings'),
    (lambda b: b.update(proposals="dot_kind"), "proposals: must be a list"),
    (lambda b: b.update(settings={}), "needs at least one setting"),
])
def test_the_validator_refuses_a_mistake_in_the_block_with_its_place(change, expect):
    data = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    change(data["study_settings"])
    errs = validate_protocol(data)
    assert any(expect in e for e in errs), errs
    assert all(e.startswith("study_settings") for e in errs)


def test_a_block_that_is_not_an_object_and_a_file_with_a_mistake_are_refused(tmp_path):
    data = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    data["study_settings"] = [1, 2]
    assert any("study_settings: must be an object" in e for e in validate_protocol(data))
    data["study_settings"] = {"settings": {"dot_kind": "grey"}}
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProtocolError, match="not a dot colour"):
        load_study_settings(p)
    with pytest.raises(ProtocolError, match="not a dot colour"):
        load_protocol(p)


def test_underscore_lines_are_comments_in_the_block(tmp_path):
    data = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    data["study_settings"]["_why"] = "a comment"
    data["study_settings"]["settings"]["_x"] = "also a comment"
    assert validate_protocol(data) == []
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert "_x" not in load_study_settings(p)["settings"]


def test_the_protocol_check_command_accepts_the_shipped_file():
    out = subprocess.run([sys.executable, "-m", "limbtrack.protocol"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, out.stdout + out.stderr


# ------------------------------------------------------------------------------------------------ comparing (pure)
def test_equal_numbers_are_the_same_and_the_wording_is_plain():
    spec = ss.BY_KEY["tolerance_deg"]
    assert ss.same(spec, 5.0, 5) and ss.same(spec, 5.0, 5.0000000001) and not ss.same(spec, 5.0, 5.5)
    assert ss.say(spec, 5.0) == "5 deg" and ss.say(ss.BY_KEY["audio_cues"], True) == "on" and ss.say(ss.BY_KEY["audio_cues"], False) == "off"
    assert ss.say(ss.BY_KEY["dot_kind"], "black") == '"black"'
    assert ss.say(ss.BY_KEY["force_calibration_id"], "").startswith("none")


def test_compare_names_what_differs_what_is_the_same_and_what_could_not_be_read():
    block = {"settings": {"tolerance_deg": 5.0, "dot_kind": "black", "audio_cues": True, "force_smoothing_ms": 100.0}, "proposals": ["tolerance_deg"]}
    res = ss.compare(block, {"tolerance_deg": 10.0, "dot_kind": "black", "audio_cues": False}, {"force_smoothing_ms": "the force is not connected"})
    assert keys(res["differences"]) == ["tolerance_deg", "audio_cues"] and keys(res["not_checked"]) == ["force_smoothing_ms"]
    order = [sp.key for sp in ss.SPECS if sp.key in block["settings"]]
    assert [r["key"] for r in res["rows"]] == order                                           # the rows follow the order of SPECS
    assert {r["key"]: r["state"] for r in res["rows"]} == {"tolerance_deg": "differs", "dot_kind": "same", "audio_cues": "differs",
                                                           "force_smoothing_ms": "not_checked"}
    d = res["differences"][0]
    assert d["text"] == "Tolerance, main joint: 10 deg now; the study says 5 deg. (Position section)" and d["proposal"] is True
    assert "Sound cues: off now; the study says on" in res["differences"][1]["text"]
    assert res["not_checked"][0]["text"] == "Force smoothing on screen: not checked (the force is not connected)."
    assert ss.compare(None, {}, {}) is None and ss.signature(None) is None and ss.reminder(None) == ""


def test_the_reminder_names_the_first_three_and_counts_the_rest():
    block = {"settings": {"tolerance_deg": 5.0, "dot_kind": "black", "audio_cues": True, "save_overlay": True, "recording_first": True}, "proposals": []}
    res = ss.compare(block, {"tolerance_deg": 9.0, "dot_kind": "auto", "audio_cues": False, "save_overlay": False, "recording_first": False}, {})
    text = ss.reminder(res)
    assert text.startswith("Study settings: 5 differ from the frozen list (") and "and 2 more" in text and text.count(";") == 3
    one = ss.compare(block, {"tolerance_deg": 9.0, "dot_kind": "black", "audio_cues": True, "save_overlay": True, "recording_first": True}, {})
    assert ss.reminder(one) == "Study settings: 1 differs from the frozen list (Tolerance, main joint)"
    none = ss.compare(block, {"tolerance_deg": 5.0, "dot_kind": "black", "audio_cues": True, "save_overlay": True, "recording_first": True}, {})
    assert ss.reminder(none) == "" and "are as the study says" in ss.list_text(none)


def test_the_list_text_puts_the_differences_first_marks_proposals_and_says_it_is_a_reminder():
    block = {"settings": {"auto_mvic_threshold_n": 25.0, "dot_kind": "black"}, "proposals": ["auto_mvic_threshold_n"]}
    res = ss.compare(block, {"auto_mvic_threshold_n": 10.0}, {"dot_kind": "no camera"})
    text = ss.list_text(res)
    assert text.index("Start when the force rises by: 10 N now") < text.index("could not be checked now")
    assert "[the study's value is a proposal]" in text and "Nothing is stopped" in text and "no camera" in text
    assert "no study_settings block" in ss.list_text(None)


def test_the_log_detail_and_the_meta_block_say_expected_and_actual():
    block = {"settings": {"tolerance_deg": 5.0, "dot_kind": "black"}, "proposals": []}
    res = ss.compare(block, {"tolerance_deg": 10.0}, {"dot_kind": "x"})
    assert ss.log_detail(res) == "checked=1; not_checked=1; differs=1; tolerance_deg=10.0 (study 5.0)"
    meta = ss.meta_block(res, ss.block_hash(block))
    assert meta["settings_frozen"] == ["tolerance_deg", "dot_kind"]
    assert meta["differences"] == [{"key": "tolerance_deg", "label": "Tolerance, main joint", "expected": 5.0, "actual": 10.0,
                                    "text": res["differences"][0]["text"]}]
    assert meta["not_checked"][0]["key"] == "dot_kind" and len(meta["block_sha256"]) == 64 and meta["note"] == ss.NOTE
    assert ss.meta_block(None) is None and ss.block_hash(None) is None
    clean = ss.meta_block(ss.compare(block, {"tolerance_deg": 5.0, "dot_kind": "black"}, {}))
    assert clean["differences"] == []                                  # an empty list when nothing differs


def test_the_signature_changes_only_with_the_differences():
    block = {"settings": {"tolerance_deg": 5.0, "dot_kind": "black"}, "proposals": []}
    a = ss.signature(ss.compare(block, {"tolerance_deg": 10.0, "dot_kind": "black"}, {}))
    b = ss.signature(ss.compare(block, {"tolerance_deg": 10.0}, {"dot_kind": "x"}))
    c = ss.signature(ss.compare(block, {"tolerance_deg": 12.0, "dot_kind": "black"}, {}))
    assert a == b and a != c and ss.signature(ss.compare(block, {"tolerance_deg": 5.0, "dot_kind": "black"}, {})) == ()


def test_the_review_item_is_informational_and_only_for_a_trial_with_differences():
    block = {"settings": {"tolerance_deg": 5.0, "dot_kind": "black"}, "proposals": []}
    with_diff = ss.meta_block(ss.compare(block, {"tolerance_deg": 10.0, "dot_kind": "auto"}, {}))
    level, text, title = ss.check_item(with_diff)
    assert level == "info" and "2 settings that differ" in text and "Tolerance, main joint" in text and "only says the set-up was not the study's" in text
    assert title == "Settings differ from the study (2)" and len(title) <= analysis.CHECK_TITLE_MAX
    assert ss.check_item(ss.meta_block(ss.compare(block, {"tolerance_deg": 5.0, "dot_kind": "black"}, {}))) is None
    for nothing in (None, {}, {"differences": "x"}, {"differences": []}, {"differences": [3]}, 5):
        assert ss.check_item(nothing) is None


def test_the_export_cell_is_a_count_and_blank_for_no_block_or_a_damaged_one():
    assert ss.export_cells({}) == {"study_settings_differences": ""} and ss.export_cells({"study_settings": None}) == {"study_settings_differences": ""}
    assert ss.export_cells({"study_settings": {"differences": "x"}}) == {"study_settings_differences": ""}
    assert ss.export_cells({"study_settings": {"differences": []}}) == {"study_settings_differences": 0}
    assert ss.export_cells({"study_settings": {"differences": [{}, {}]}}) == {"study_settings_differences": 2}


# ------------------------------------------------------------------------------------------------ the program's own values
def test_an_engine_set_like_the_study_has_no_difference_and_the_unreadable_are_not_checked(tmp_path):
    eng = engine_like_the_study(tmp_path)
    res = eng.study_settings_state()
    assert res["differences"] == [], [d["text"] for d in res["differences"]]
    not_read = {r["key"]: r["text"] for r in res["not_checked"]}
    assert set(not_read) == {"camera_wanted", "auto_mvic", "auto_mvic_threshold_n", "auto_mvic_length_s", "force_smoothing_ms", "force_calibration_id"}
    assert "the force is not connected" in not_read["auto_mvic"] and "no camera is running" in not_read["camera_wanted"]
    assert "automatic MVIC markers are off" not in not_read["auto_mvic"]                       # (it says the force first)


def test_a_fresh_program_differs_in_the_dot_colour_and_the_ruler_and_says_so(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path / "recordings")
    eng.study_block, eng.study_extras = BLOCK, dict(STUDY_EXTRAS)
    res = eng.study_settings_state()
    assert keys(res["differences"]) == ["dot_kind", "ruler_length_cm"]
    assert res["differences"][0]["actual"] == "auto" and res["differences"][0]["expected"] == "black"
    eng.set_dot_kind("black")
    assert keys(eng.study_settings_state()["differences"]) == ["ruler_length_cm"]


def test_each_setting_the_operator_can_change_is_noticed_when_it_changes(tmp_path):
    eng = engine_like_the_study(tmp_path)
    eng.tolerance, eng.neighbour_tolerance = 10.0, 7.5
    eng.smoothing, eng.recording_first, eng.save_overlay = False, False, False
    eng.audio_cues, eng.cue_push_every_s, eng.auto_stop_s = False, 2.0, 0.0
    eng.set_dot_kind("white")
    res = eng.study_settings_state()
    assert set(keys(res["differences"])) == {"tolerance_deg", "neighbour_tolerance_deg", "display_smoothing", "recording_first", "save_overlay",
                                            "audio_cues", "cue_push_every_s", "auto_stop_s", "dot_kind"}
    act = {d["key"]: d["actual"] for d in res["differences"]}
    assert act["tolerance_deg"] == 10.0 and act["auto_stop_s"] == 0.0 and act["audio_cues"] is False and act["dot_kind"] == "white"


def test_the_force_settings_are_read_from_a_real_connected_source_and_the_markers_only_when_on(tmp_path):
    eng = engine_like_the_study(tmp_path)
    src = SimulatedForce()                                 # (not started: the settings are read, no sample is needed)
    eng.force = src
    res = eng.study_settings_state()
    diff = {d["key"]: d for d in res["differences"]}
    assert set(diff) == {"auto_mvic", "force_calibration_id"}, list(diff)                            # no calibration regression; markers off
    assert diff["auto_mvic"]["actual"] is False and diff["force_calibration_id"]["actual"] == ""
    assert "none (a known weight, or no calibration)" in diff["force_calibration_id"]["text"]
    assert {"auto_mvic_threshold_n", "auto_mvic_length_s"} <= set(keys(res["not_checked"]))          # irrelevant while the markers are off
    eng.auto_mvic = True
    eng.auto_mvic_threshold = 10.0
    src.smooth_s = 0.2
    src.calibration.regression_id = "2026-10-13"
    res = eng.study_settings_state()
    diff = {d["key"]: d for d in res["differences"]}
    assert set(diff) == {"auto_mvic_threshold_n", "force_smoothing_ms"}, list(diff)
    assert diff["auto_mvic_threshold_n"]["actual"] == 10.0 and diff["force_smoothing_ms"]["actual"] == 200.0
    assert diff["auto_mvic_threshold_n"]["proposal"] is True and diff["force_smoothing_ms"]["proposal"] is False
    eng.auto_mvic_threshold, src.smooth_s = 25.0, 0.1
    assert eng.study_settings_state()["differences"] == []


def test_the_wanted_camera_picture_is_read_from_what_the_video_thread_reported(tmp_path):
    eng = engine_like_the_study(tmp_path)
    eng.camera_info = {"requested": {"width": 1280, "height": 720, "fps": 30}, "granted_by_driver": {}}
    assert eng.study_settings_state()["differences"] == []
    eng.camera_info = {"requested": {"width": 1920, "height": 1080, "fps": 30}}
    d = eng.study_settings_state()["differences"]
    assert keys(d) == ["camera_wanted"] and d[0]["actual"] == "1920x1080 at 30 fps"
    eng.camera_info = {"requested": None, "is_file": True}                      # a video file: nothing was asked of a camera
    assert "camera_wanted" in keys(eng.study_settings_state()["not_checked"])


def test_the_numbers_fixed_in_the_program_are_the_frozen_ones_today_and_a_changed_constant_is_noticed(tmp_path, monkeypatch):
    eng = engine_like_the_study(tmp_path)
    fixed = [s.key for s in ss.SPECS if not s.live]
    res = eng.study_settings_state()
    assert not [r for r in res["rows"] if r["key"] in fixed and r["state"] != "same"], [r["text"] for r in res["rows"] if r["state"] != "same"]
    assert len([r for r in res["rows"] if r["key"] in fixed]) == len(fixed) == 13
    monkeypatch.setattr(rest, "REST_SAME_POSITION_S", 45.0)                      # someone edits the rest time in the code
    res = eng.study_settings_state()
    assert keys(res["differences"]) == ["rest_same_position_s"] and res["differences"][0]["actual"] == 45.0
    assert "45 s now; the study says 60 s" in res["differences"][0]["text"]


def test_a_changed_tracker_number_changes_the_hash_and_is_noticed(tmp_path, monkeypatch):
    from limbtrack import tracker
    eng = engine_like_the_study(tmp_path)
    assert ss.BY_KEY["tracker_settings_sha256"].read(eng) == BLOCK["settings"]["tracker_settings_sha256"]
    monkeypatch.setattr(tracker, "VELOCITY_GAIN", 0.6)
    res = eng.study_settings_state()
    assert keys(res["differences"]) == ["tracker_settings_sha256"]
    d = res["differences"][0]
    assert d["label"] == "Tracking program version" and "not the version the study was frozen with" in d["text"] and "tell whoever maintains" in d["text"]
    assert d["actual"] != d["expected"] and len(d["actual"]) == 64                            # the hashes stay in the row (and meta.json), not in the sentence
    assert d["actual"][:12] not in d["text"] and "hash" not in d["text"] and "tracker.py" not in d["text"]
    assert "hash" not in ss.reminder(res) and "tracker.py" not in ss.list_text(res)
    assert d["actual"] in ss.log_detail(res)


def test_no_block_means_no_comparison_at_all(tmp_path):
    eng = engine_like_the_study(tmp_path)
    eng.study_block = None
    assert eng.study_settings_state() is None and eng._study_settings_meta() is None


# ------------------------------------------------------------------------------------------------ a real recording
def picture(eng, seed=0):
    """A synthetic picture of the leg in the colour the engine looks for (black dots when it is set to black, else white)."""
    return render(P, kind="black" if eng.kind == "black" else "white", seed=seed)


def record(eng, seconds=0.6, then=None):
    """One real recording of `seconds` (synthetic pictures); `then` is called half way (a setting touched while recording). Returns meta.json."""
    paths = eng.start_trial("P01")
    end = time.perf_counter() + seconds
    done = False
    while time.perf_counter() < end:
        eng.process(picture(eng, int(time.perf_counter() * 30) % 5), time.perf_counter())
        if then and not done and time.perf_counter() > end - seconds / 2:
            then()
            done = True
        time.sleep(1 / 30)
    eng.stop_trial()
    time.sleep(1.1)                                       # (trial file names carry the second)
    return paths["meta.json"]


def mark(eng):
    """Mark every dot on a synthetic picture (a change of dot colour takes the marks away: they are made again)."""
    eng.process(picture(eng), time.perf_counter())
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(P[n] + 2))


@pytest.fixture(scope="module")
def trials(tmp_path_factory):
    root = tmp_path_factory.mktemp("study") / "recordings"
    eng = Engine(TESTS, recordings_dir=root)
    eng.set_test("knee_extension")
    eng.target = 70.0
    eng.participant_code = "P01"
    eng.tested_leg = "right"
    mark(eng)
    out = {"eng": eng, "folder": root / "P01"}
    out["no_block"] = record(eng)                                     # protocol.json had no block
    eng.study_block, eng.study_extras = BLOCK, dict(STUDY_EXTRAS)
    out["differing"] = record(eng)                                    # the engine is still on Auto and a 15 cm ruler
    eng.set_dot_kind("black")
    eng.scale.length_cm = 100.0
    mark(eng)
    out["matching"] = record(eng)
    eng.set_dot_kind("white")
    mark(eng)
    out["touched"] = record(eng, then=lambda: setattr(eng.scale, "length_cm", 80.0))       # (the ruler length may be touched while recording)
    out["ruler_after_touch"] = eng.scale.length_cm
    eng.set_dot_kind("black")
    eng.scale.length_cm = 100.0
    mark(eng)
    src = SimulatedForce(peak_n=400.0, rest_s=1e6, ramp_s=0.6, hold_s=1.2, period_s=1e7, n_per_v=527.0, offset_v=0.0123)     # a real-time load cell
    src.start()
    eng.force = src
    try:
        time.sleep(1.4)
        assert src.zero() is None
        src.rest_s = src._next / src.rate + 0.7                       # the push begins 0.7 s into the next recording
        out["with_force"] = record(eng, 3.5)
        out["force_state"] = eng.study_settings_state()
    finally:
        src.stop()
        eng.force = None
    return out


def meta_of(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_trial_without_a_block_says_null_a_differing_one_lists_it_and_a_matching_one_an_empty_list(trials):
    assert "study_settings" in meta_of(trials["no_block"]) and meta_of(trials["no_block"])["study_settings"] is None
    got = meta_of(trials["differing"])["study_settings"]
    assert got["settings_frozen"] == list(k for k in ss.BY_KEY if k in BLOCK["settings"])
    assert [d["key"] for d in got["differences"]] == ["dot_kind", "ruler_length_cm"]
    d = got["differences"][0]
    assert d["expected"] == "black" and d["actual"] == "auto" and "Dot colour" in d["text"] and d["label"] == "Dot colour"
    assert got["block_sha256"] == ss.block_hash(BLOCK) and set(got["proposals"]) == set(BLOCK["proposals"])
    assert "camera_wanted" in [r["key"] for r in got["not_checked"]]
    ok = meta_of(trials["matching"])["study_settings"]
    assert ok["differences"] == [] and ok["settings_frozen"] == got["settings_frozen"]


def test_the_trial_keeps_the_list_as_it_was_at_the_start_and_a_touched_display_setting_is_not_a_change(trials):
    m = meta_of(trials["touched"])
    assert [d["key"] for d in m["study_settings"]["differences"]] == ["dot_kind"]                 # white at the start; the ruler was 100 cm then
    assert "changed_during_recording" not in m and trials["ruler_after_touch"] == 80.0       # (it was changed while recording)
    assert m["software_version"] == SOFTWARE_VERSION


def test_nothing_measured_changes_between_a_trial_with_and_without_the_list(trials):
    a, b = meta_of(trials["no_block"]), meta_of(trials["matching"])
    for k in ("test_key", "target_deg", "tolerance_deg", "neighbour_tolerance_deg", "dots_used", "tracker_settings", "angles", "smoothing",
              "force_setup", "privacy", "mirrored_image", "facing"):
        assert a[k] == b[k], k
    assert set(b) - set(a) == set() and set(a) - set(b) == set()                       # the same keys; only the value of study_settings differs


def test_review_says_so_in_an_info_line_and_the_checks_to_look_at_do_not_change(trials):
    tr = Trial(trials["differing"], open_video=False)
    items = [i for i in analysis.trial_check_items(tr) if i["title"].startswith(ss.CHECK_TITLE)]
    tr.close()
    assert len(items) == 1 and items[0]["level"] == "info" and items[0]["area"] == "recording" and items[0]["title"] == "Settings differ from the study (2)"
    assert "Dot colour" in items[0]["text"]
    for name in ("matching", "no_block"):
        t2 = Trial(trials[name], open_video=False)
        assert not [i for i in analysis.trial_check_items(t2) if i["title"].startswith(ss.CHECK_TITLE)]
        t2.close()
    row, err = trial_row(trials["differing"], 0.1, 0.5)
    assert row is not None, err
    assert "Recorded with" not in row["checks_to_look_at"] and "frozen list" not in row["checks_to_look_at"]
    assert list(analysis.SESSION_COLUMNS).count("study_settings_differences") == 0           # not a column of the session summary


def test_the_anybody_export_gets_one_count_column_at_the_end_and_a_note_when_blank(trials, tmp_path):
    from limbtrack import review_nav
    assert anybody.COLUMNS[-1] == "study_settings_differences" and anybody.COLUMNS[-3:-1] == ["check_weight_error_pct", "check_weight_time"]
    assert anybody.COLUMNS[-4] == "notes" and anybody.COLUMNS[0] == "participant"
    want = len(meta_of(trials["with_force"])["study_settings"]["differences"])
    assert want == len(trials["force_state"]["differences"]) and want >= 1                    # (the markers are off and the simulated cell has no regression)
    verdicts.write_verdict(trials["with_force"], verdicts.VALID)
    rows, paths = [], {}
    for e in review_nav.list_trials(trials["folder"]):
        r, err = trial_row(e.meta_path, 0.1, 0.5)
        assert r is not None, err
        rows.append(r)
        paths[r["trial"]] = e.meta_path
    res = analysis.anybody_rows(rows, lambda r: paths[r["trial"]], lambda p: Trial(p, open_video=False), plan=PLAN, judged_only=True,
                                export_version=SOFTWARE_VERSION)
    assert len(res["rows"]) == 1
    row = res["rows"][0]
    assert row["study_settings_differences"] == want and "no frozen study settings" not in row["notes"]
    out = tmp_path / "anybody_inputs.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        anybody.write_csv(fh, res["rows"])
    back = list(csv.DictReader(open(out, encoding="utf-8")))
    assert list(back[0])[-1] == "study_settings_differences" and back[0]["study_settings_differences"] == str(want)
    # the same trial exported as one of an earlier version (the key removed): blank, and the note says why
    old = tmp_path / "old_export"
    old.mkdir()
    for p in trials["folder"].glob(trials["with_force"].name[: -len("_meta.json")] + "*"):
        (old / p.name).write_bytes(p.read_bytes())
    path = old / trials["with_force"].name
    meta = meta_of(path)
    del meta["study_settings"]
    path.write_text(json.dumps(meta), encoding="utf-8")
    verdicts.write_verdict(path, verdicts.VALID)
    r2, err = trial_row(path, 0.1, 0.5)
    assert r2 is not None, err
    res2 = analysis.anybody_rows([r2], lambda r: path, lambda p: Trial(p, open_video=False), plan=PLAN, judged_only=True, export_version=SOFTWARE_VERSION)
    assert len(res2["rows"]) == 1 and res2["rows"][0]["study_settings_differences"] == "" and "study_settings_differences: no frozen study settings" in res2["rows"][0]["notes"]


def test_a_trial_of_an_earlier_version_without_the_key_opens_and_exports_with_a_blank(trials, tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    for p in trials["folder"].glob(trials["matching"].name[: -len("_meta.json")] + "*"):
        (old / p.name).write_bytes(p.read_bytes())
    path = old / trials["matching"].name
    meta = meta_of(path)
    del meta["study_settings"]
    meta["software_version"] = "0.34.0"
    path.write_text(json.dumps(meta), encoding="utf-8")
    row, err = trial_row(path, 0.1, 0.5)
    assert row is not None, err
    tr = Trial(path, open_video=False)
    assert len(tr.t) > 0 and not [i for i in analysis.trial_check_items(tr) if i["title"].startswith(ss.CHECK_TITLE)]
    tr.close()
    assert ss.export_cells(meta) == {"study_settings_differences": ""}


# ------------------------------------------------------------------------------------------------ the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2] if len(a) > 2 else k.get("text")))
    (tmp_path / "recordings").mkdir()                      # (the session log never creates the folder)
    shutil.copy(PROTOCOL, tmp_path / "protocol.json")
    w = MainWindow(TESTS, tmp_path)
    w.shown = shown
    yield w
    w.live.worker = None
    w.close()


def log_lines(win):
    path = win.live.recordings_dir / "session_log.csv"
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def study_lines(win):
    return [r for r in log_lines(win) if r["event"] == "study settings"]


def test_the_window_loads_the_block_from_the_protocol_file_next_to_it(win, tmp_path):
    assert win.engine.study_block == BLOCK and win.engine.study_extras == STUDY_EXTRAS
    assert STUDY_EXTRAS == {"trials_per_position": 2, "rest_force_from_s": 3.0, "rest_force_to_s": 0.5}


def test_the_amber_row_shows_only_with_differences_names_them_and_one_click_shows_the_list(win):
    live, e = win.live, win.engine
    live.update_steps()
    assert "studysettings" in live.reminders.open_keys()                                   # a fresh program: Auto colour, 15 cm ruler
    text = live.reminders.rows["studysettings"][1].text()
    assert text == "Study settings: 2 differ from the frozen list (Dot colour; Ruler length)"
    live.btn_rem_study_list.click()
    assert len(win.shown) == 1 and "2 of the " in win.shown[0] and 'Dot colour: "auto" now; the study says "black"' in win.shown[0]
    assert "could not be checked now" in win.shown[0] and "Nothing is stopped" in win.shown[0]
    live._pick(live.kind_combo, "black")
    live.ruler_len.setValue(100.0)
    live.update_steps()
    assert "studysettings" not in live.reminders.open_keys()


def test_noted_silences_the_row_until_the_differences_change_and_is_written_down(win):
    live = win.live
    live.update_steps()
    live.btn_rem_study.click()
    assert "studysettings" not in live.reminders.open_keys()
    assert [r for r in log_lines(win) if r["event"] == "study settings reminder dismissed"][0]["detail"].startswith("Study settings: 2 differ")
    live.tol_spin.setValue(10.0)
    live.update_steps()
    assert "studysettings" in live.reminders.open_keys() and "3 differ" in live.reminders.rows["studysettings"][1].text()


def test_record_is_never_refused_and_the_open_reminder_is_logged_but_the_file_holds_the_list(win):
    live = win.live
    reminders = live._open_reminders()
    mine = [(t, in_file) for t, in_file in reminders if t.startswith("Study settings:")]
    assert mine and mine[0][1] is False                                                    # a reminder; the meta.json holds the list itself
    live.update_steps()
    assert "studysettings" in live.reminders.open_keys()                                   # (the real recordings above were made with the list differing)


def test_the_session_log_gets_a_line_at_start_up_and_one_per_change_not_one_per_tick(win, monkeypatch):
    live = win.live
    monkeypatch.setattr("limbtrack.gui.STUDY_LOG_SETTLE_S", 0.0)
    live._study_log()
    first = study_lines(win)
    assert len(first) == 1 and first[0]["detail"].startswith("checked=") and "differs=2" in first[0]["detail"]
    assert "dot_kind=auto (study black)" in first[0]["detail"] and "ruler_length_cm=15.0 (study 100.0)" in first[0]["detail"]
    for _ in range(5):
        live._study_log()
    assert len(study_lines(win)) == 1                                                      # the same differences: nothing more
    live.tol_spin.setValue(10.0)
    live._study_log()
    live._study_log()
    lines = study_lines(win)
    assert len(lines) == 2 and "differs=3" in lines[1]["detail"] and "tolerance_deg=10.0 (study 5.0)" in lines[1]["detail"]
    live.tol_spin.setValue(5.0)
    live._pick(live.kind_combo, "black")
    live.ruler_len.setValue(100.0)
    live._study_log()
    assert len(study_lines(win)) == 3 and "differs=0" in study_lines(win)[2]["detail"]


def test_a_change_must_stay_for_a_moment_before_it_is_written(win, monkeypatch):
    live = win.live
    monkeypatch.setattr("limbtrack.gui.STUDY_LOG_SETTLE_S", 0.3)
    live._study_log()
    assert study_lines(win) == []                                                          # seen, but not yet settled
    time.sleep(0.35)
    live._study_log()
    assert len(study_lines(win)) == 1


def test_nothing_is_written_while_recording_and_without_a_block_one_line_says_so(win, monkeypatch):
    live = win.live
    monkeypatch.setattr("limbtrack.gui.STUDY_LOG_SETTLE_S", 0.0)
    live._rec_started = time.perf_counter()
    live._study_log()
    assert study_lines(win) == []
    live._rec_started = None
    win.engine.study_block = None
    live._study_log()
    live._study_log()
    lines = study_lines(win)
    assert len(lines) == 1 and lines[0]["detail"] == "no study_settings block in protocol.json"
    live.update_steps()
    assert "studysettings" not in live.reminders.open_keys()


def test_the_row_is_not_shown_while_recording(win):
    live = win.live
    live._rec_started = time.perf_counter()
    live.update_steps()
    assert "studysettings" not in live.reminders.open_keys()
    live._rec_started = None


def test_a_window_without_a_protocol_file_next_to_it_has_no_block(tmp_path):
    QApplication.instance() or QApplication([])
    w = MainWindow(TESTS, tmp_path)
    try:
        assert w.engine.study_block is None
    finally:
        w.live.worker = None
        w.close()


# ------------------------------------------------------------------------------------------------ tools/freeze_check.py
def git_repo(path):
    def run(*a):
        subprocess.run(["git", *a], cwd=path, check=True, capture_output=True, text=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (path / "a.txt").write_text("one\n", encoding="utf-8")
    run("add", "a.txt")
    run("commit", "-q", "-m", "first")
    return run


def test_the_git_state_of_a_real_repository_is_clean_then_not_clean_and_lists_the_files(tmp_path):
    run = git_repo(tmp_path)
    s = freeze_check.git_state(cwd=tmp_path)
    assert len(s["commit"]) == 40 and s["changed"] == [] and s["untracked"] == [] and s["tags"] == []
    assert freeze_check.problems(s, []) == []
    run("tag", "v-test")
    (tmp_path / "a.txt").write_text("two\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")
    s = freeze_check.git_state(cwd=tmp_path)
    assert s["tags"] == ["v-test"] and s["changed"] == ["a.txt"] and s["untracked"] == ["new.txt"]
    assert freeze_check.problems(s, []) == ["the working tree is not clean"]


def test_the_report_says_clean_or_not_clean_and_shows_the_frozen_values(tmp_path):
    git_repo(tmp_path)
    clean = freeze_check.git_state(cwd=tmp_path)
    fixed = [{"key": "rest_same_position_s", "state": "same", "text": ""}]
    lines = freeze_check.report(clean, "9.9.9", "c" * 64, "d" * 64, BLOCK, fixed, now=0)
    text = "\n".join(lines)
    assert "Program version      : 9.9.9" in text and clean["commit"] in text and "Working tree         : CLEAN" in text
    assert "c" * 64 in text and "d" * 64 in text and "dot_kind = \"black\"" in text and "auto_mvic_threshold_n = 25 N *" in text
    assert "Numbers fixed in the program: all 1 are the frozen ones." in text
    dirty = dict(clean, changed=[f"f{i}.py" for i in range(20)], untracked=["u.py"])
    t2 = "\n".join(freeze_check.report(dirty, "1", "c", "d", BLOCK, [{"key": "k", "state": "differs", "text": "Rest: 45 s now; the study says 60 s."}], now=0))
    assert "NOT CLEAN - 20 changed file(s), 1 new file(s)" in t2 and "and 8 more changed" in t2 and "are NOT the frozen ones" in t2
    t3 = "\n".join(freeze_check.report(dict(clean, changed=None), "1", "c", "d", None, [], now=0))
    assert "UNKNOWN" in t3 and "none (protocol.json has no study_settings block)" in t3


def test_the_tool_runs_on_this_project_prints_the_facts_and_changes_nothing(capsys):
    def tree():
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return out, PROTOCOL.stat().st_mtime_ns, subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    before = tree()
    code = freeze_check.main([])
    printed = capsys.readouterr().out
    assert code == 0
    assert tree() == before                                                                # read-only: not a file, not a commit, not a tag
    assert f"Program version      : {SOFTWARE_VERSION}" in printed and "Git commit           : " in printed
    assert "Numbers fixed in the program: all 13 are the frozen ones." in printed
    from limbtrack import provenance
    assert provenance.protocol_sha256(PROTOCOL) in printed and provenance.code_sha256() in printed
    assert printed.count(" = ") >= len(BLOCK["settings"])
    code = freeze_check.main(["--strict"])
    capsys.readouterr()
    assert code in (0, 1)                                                                  # (1 while the tree has uncommitted work; main's use of it is next)


def test_strict_exits_1_for_a_dirty_tree_or_a_changed_fixed_number_and_0_otherwise(monkeypatch, capsys):
    """T5: the exit status of main(), with the git facts and the fixed numbers given."""
    clean = {"commit": "a" * 40, "branch": "main", "tags": [], "changed": [], "untracked": []}
    dirty = dict(clean, changed=["a.py"])
    same_rows = [{"key": "rest_same_position_s", "state": "same", "text": ""}]
    differs = [{"key": "rest_same_position_s", "state": "differs", "text": "Rest: 45 s now; the study says 60 s."}]

    def run(info, rows, *argv):
        monkeypatch.setattr(freeze_check, "git_state", lambda *a, **k: info)
        monkeypatch.setattr(freeze_check, "fixed_numbers", lambda block: rows)
        code = freeze_check.main(list(argv))
        return code, capsys.readouterr().out
    assert run(dirty, same_rows, "--strict")[0] == 1
    assert run(clean, differs, "--strict")[0] == 1
    assert run(dict(clean, changed=None), same_rows, "--strict")[0] == 1                      # git could not say: not proven clean
    code, out = run(clean, same_rows, "--strict")
    assert code == 0 and "Not ready to freeze" not in out
    code, out = run(dirty, differs)                                                           # without --strict it only reports
    assert code == 0 and "Not ready to freeze: the working tree is not clean; rest_same_position_s is not the frozen value." in out
