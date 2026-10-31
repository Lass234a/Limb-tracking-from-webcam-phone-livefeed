"""Fixing dots after the trial (0.30.0, limbtrack/dot_edits.py): hold, mark lost, by hand, re-track. Only the edited pictures change, and
what is worked out from them is worked out with the recording's own functions (so an edit that puts a dot where it was changes nothing)."""
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack import analysis, dot_edits  # noqa: E402
from limbtrack import review_layer as rl  # noqa: E402
from limbtrack.angles import measure  # noqa: E402
from limbtrack.geometry import hip_centre_frontal  # noqa: E402
from limbtrack.review import Trial  # noqa: E402
from limbtrack.synthetic import render  # noqa: E402

from tests import test_pull as P  # noqa: E402  (a knee trial with the strap and anchor dots)
from tests import test_pull_frontal as PF  # noqa: E402  (a frontal trial: the hip centre is a computed point and the pull joint)

N = 30


@pytest.fixture(scope="module")
def knee(tmp_path_factory):
    """A 30-picture knee trial with strap and anchor; the anchor is hidden in pictures 10-14 (lost, as in P02 T4)."""
    eng, p = P.make(tmp_path_factory.mktemp("knee"))
    paths = eng.start_trial("P01")
    for i in range(N):
        eng.process(render(p, seed=i, hide=["anchor"] if 10 <= i < 15 else []), 0.2 + i / 30)
    eng.stop_trial()
    return paths["meta.json"], p


@pytest.fixture(scope="module")
def frontal(tmp_path_factory):
    eng, p, hjc = PF.make(tmp_path_factory.mktemp("frontal"))
    paths = eng.start_trial("P01")
    for i in range(N):
        eng.process(PF.F.pic(p, i), 0.2 + i / 30)
    eng.stop_trial()
    return paths["meta.json"], p


def _review(meta, edits=()):
    """A Review in memory (nothing is written): `edits` = [(dot, kind, rows, params)]."""
    rv = rl.Review(meta)
    for dot, kind, rows, params in edits:
        rv.add_dot_edit(dot, kind, rows=rows, **params)
    return rv


def _identity_rows(tr, dot):
    x, y = tr.tracks[dot]
    return [{"frame": i, "t_s": float(tr.t[i]), "x": float(x[i]), "y": float(y[i])} for i in range(len(tr.t)) if not (math.isnan(x[i]) or math.isnan(y[i]))]


def test_an_edit_that_puts_the_dots_where_they_were_changes_no_number(knee):
    meta, _ = knee
    plain = Trial(meta, open_video=False)
    tr = Trial(meta, open_video=False)
    dots = ("epicondyle", "fibular_head", "malleolus", "strap", "anchor")
    res = dot_edits.apply(tr, _review(meta, [(d, "retrack", _identity_rows(plain, d), {}) for d in dots]))       # (re-track rows: no line is laid across a gap)
    assert set(res["edited"]) == set(dots) and not res["skipped"]
    for a in plain.angle_defs:
        assert np.allclose(tr.deg[a.name], plain.deg[a.name], atol=0.01, equal_nan=True), a.name
        assert np.allclose(tr.dev[a.name], plain.dev[a.name], atol=0.01, equal_nan=True), a.name
        assert tr.ok[a.name] == plain.ok[a.name], a.name
    for c, v in plain.pull.items():
        assert np.allclose(tr.pull[c], v, atol=0.06, equal_nan=True), c                         # (the recording rounds positions to 0.01 px)
    assert [r["all_ok"] for r in tr.rows] == [r["all_ok"] for r in plain.rows]
    assert np.array_equal(np.isnan(tr.tracks["anchor"][0]), np.isnan(plain.tracks["anchor"][0]))      # the hidden pictures stay lost: no position was made up


def test_holding_a_dot_changes_only_the_pictures_in_the_range_and_the_numbers_made_from_it(knee):
    meta, p = knee
    plain, tr = Trial(meta, open_video=False), Trial(meta, open_video=False)
    t0, t1 = float(plain.t[8]), float(plain.t[20])
    dot_edits.apply(tr, _review(meta, [("anchor", "hold", None, {"from_s": t0, "to_s": t1, "x": 300.0, "y": 500.0})]))
    inside = np.zeros(N, bool)
    inside[8:21] = True
    assert tr.edited["anchor"] == list(range(8, 21)) and all(tr.rows[i]["anchor_lost"] == "0" for i in range(8, 21))      # (the hidden ones are found again: held)
    assert np.allclose(tr.tracks["anchor"][0][inside], 300.0) and np.array_equal(tr.tracks["anchor"][0][~inside], plain.tracks["anchor"][0][~inside], equal_nan=True)
    pa = tr.pull["pull_angle_deg"]
    assert np.array_equal(pa[~inside], plain.pull["pull_angle_deg"][~inside], equal_nan=True)      # other pictures: exactly what was recorded
    assert not np.isnan(pa[inside]).any() and not np.allclose(pa[inside], plain.pull["pull_angle_deg"][inside], equal_nan=True)
    assert all(np.array_equal(tr.deg[a.name], plain.deg[a.name], equal_nan=True) for a in plain.angle_defs)       # the anchor is in no angle
    assert [tr.rows[i]["pull_angle_deg"] != "" for i in (12, 13)] == [True, True]                                  # pictures 10-14 had no pull
    assert "q" not in tr.rows[12]["anchor_q"] if isinstance(tr.rows[12]["anchor_q"], str) else True


def test_an_angle_of_an_edited_picture_is_what_the_recordings_own_function_gives(knee):
    meta, _ = knee
    plain, tr = Trial(meta, open_video=False), Trial(meta, open_video=False)
    main = tr.primary_angle
    x0, y0 = plain.tracks["epicondyle"][0][12], plain.tracks["epicondyle"][1][12]
    dot_edits.apply(tr, _review(meta, [("epicondyle", "hold", None, {"from_s": float(plain.t[12]), "to_s": float(plain.t[12]), "x": x0 + 60.0, "y": y0 - 40.0})]))
    xy = {d: (tr.tracks[d][0][12], tr.tracks[d][1][12]) for d in tr.dot_names if not math.isnan(tr.tracks[d][0][12])}
    want = measure(main, xy, tr.facing_right)
    assert tr.deg[main.name][12] == pytest.approx(want.value, abs=1e-6) and tr.deg[main.name][12] != pytest.approx(plain.deg[main.name][12], abs=1.0)
    assert tr.dev[main.name][12] == pytest.approx(tr.deg[main.name][12] - tr.ref[main.name][12], abs=1e-6)
    assert tr.ok[main.name][12] in ("0", "1") and tr.rows[12][f"{main.name}_deg"] != ""
    assert [tr.deg[main.name][i] for i in (11, 13)] == [plain.deg[main.name][11], plain.deg[main.name][13]]       # the neighbours are untouched


def test_marking_a_dot_lost_blanks_it_and_what_needs_it(knee):
    meta, _ = knee
    plain, tr = Trial(meta, open_video=False), Trial(meta, open_video=False)
    dot_edits.apply(tr, _review(meta, [("epicondyle", "lost", None, {"from_s": float(plain.t[3]), "to_s": float(plain.t[8])})]))
    main = tr.primary_angle
    assert all(tr.rows[i]["epicondyle_lost"] == "1" and tr.rows[i]["epicondyle_x"] == "" for i in range(3, 9))
    assert all(math.isnan(tr.deg[main.name][i]) for i in range(3, 9)) and all(tr.rows[i]["all_ok"] == "0" for i in range(3, 9))
    assert all(math.isnan(tr.pull["moment_arm_px"][i]) for i in range(3, 9))                 # the moment arm needs the joint (the epicondyle)
    assert not math.isnan(tr.deg[main.name][9]) and not math.isnan(tr.deg[main.name][2])
    assert all(not ((idx >= 3) & (idx <= 8)).any() for idx, _ in tr.trail_segments["epicondyle"]) and any(idx[0] == 9 for idx, _ in tr.trail_segments["epicondyle"])
    checks = " ".join(t for _, t in analysis.trial_checks(tr))
    assert "'epicondyle' was lost in 20 %" in checks                                             # 6 of 30 pictures: the Check tab counts it


def test_by_hand_a_straight_line_is_laid_between_two_clicks_and_nothing_outside_them_changes(knee):
    meta, _ = knee
    plain, tr = Trial(meta, open_video=False), Trial(meta, open_video=False)
    rows = [{"frame": 5, "t_s": float(plain.t[5]), "x": 100.0, "y": 200.0}, {"frame": 9, "t_s": float(plain.t[9]), "x": 140.0, "y": 240.0}]
    dot_edits.apply(tr, _review(meta, [("malleolus", "manual", rows, {})]))
    x, y = tr.tracks["malleolus"]
    assert [round(float(x[i]), 3) for i in range(5, 10)] == [100.0, 110.0, 120.0, 130.0, 140.0]
    assert [round(float(y[i]), 3) for i in range(5, 10)] == [200.0, 210.0, 220.0, 230.0, 240.0]
    assert tr.edited["malleolus"] == [5, 6, 7, 8, 9]
    assert np.array_equal(x[:5], plain.tracks["malleolus"][0][:5]) and np.array_equal(x[10:], plain.tracks["malleolus"][0][10:])


def test_a_later_edit_wins_where_two_overlap(knee):
    meta, _ = knee
    plain, tr = Trial(meta, open_video=False), Trial(meta, open_video=False)
    t = plain.t
    dot_edits.apply(tr, _review(meta, [("anchor", "hold", None, {"from_s": float(t[0]), "to_s": float(t[N - 1]), "x": 11.0, "y": 22.0}),
                                       ("anchor", "manual", [{"frame": 3, "t_s": float(t[3]), "x": 77.0, "y": 88.0}], {})]))
    x, y = tr.tracks["anchor"]
    assert (float(x[3]), float(y[3])) == (77.0, 88.0) and float(x[2]) == 11.0 and float(x[4]) == 11.0


def test_dots_on_one_bone_that_end_up_far_apart_are_flagged_again_for_the_edited_pictures(knee):
    meta, _ = knee
    plain, tr = Trial(meta, open_video=False), Trial(meta, open_video=False)
    assert not any(r["dots_apart"] for r in plain.rows)
    x, y = plain.tracks["trochanter"][0][6], plain.tracks["trochanter"][1][6]
    dot_edits.apply(tr, _review(meta, [("epicondyle", "hold", None, {"from_s": float(plain.t[6]), "to_s": float(plain.t[7]), "x": float(x) + 400.0, "y": float(y)})]))
    assert "trochanter-epicondyle" in tr.rows[6]["dots_apart"] and tr.rows[6]["all_ok"] == "0" and tr.rows[5]["dots_apart"] == ""
    found = tr.meta["checks"]["dots_apart_frames"]
    assert found["trochanter-epicondyle"]["frames"] == 2
    assert any("moved apart" in t for lvl, t in analysis.trial_checks(tr))


def test_an_edit_of_a_dot_the_trial_does_not_have_is_skipped_and_said(knee):
    meta, _ = knee
    tr = Trial(meta, open_video=False)
    res = dot_edits.apply(tr, _review(meta, [("no_such_dot", "lost", None, {"from_s": 0.0, "to_s": 1.0})]))
    assert res["skipped"] == ["no_such_dot"] and res["edited"] == {} and tr.edited == {}
    tr2 = Trial(meta, open_video=False)
    tr2.apply_review(_review(meta, [("no_such_dot", "lost", None, {"from_s": 0.0, "to_s": 1.0})]))
    assert any("not a dot of this trial" in n for n in tr2.review_notes)


def test_the_usual_position_is_where_the_dot_was_seen_well(knee):
    meta, _ = knee
    tr = Trial(meta, open_video=False)
    ux, uy = dot_edits.usual_position(tr, "anchor")
    assert ux == pytest.approx(float(np.nanmedian(tr.tracks["anchor"][0])), abs=1.0) and uy == pytest.approx(float(np.nanmedian(tr.tracks["anchor"][1])), abs=1.0)
    assert dot_edits.usual_position(tr, "no_such_dot") is None


def test_re_tracking_follows_a_dot_through_the_real_video_and_says_when_it_cannot_start(knee):
    meta, p = knee
    tr = Trial(meta)
    rows, why = dot_edits.retrack(tr, "malleolus", 0, (float(p["malleolus"][0]) + 3.0, float(p["malleolus"][1]) + 3.0), "forward")
    assert not why and [r["frame"] for r in rows] == list(range(N))
    good = [r for r in rows if not math.isnan(r["x"])]
    assert len(good) >= N - 2 and all(abs(r["x"] - tr.tracks["malleolus"][0][r["frame"]]) < 4.0 for r in good)        # the same dot the recording followed
    back, why = dot_edits.retrack(tr, "malleolus", N - 1, (float(tr.tracks["malleolus"][0][N - 1]), float(tr.tracks["malleolus"][1][N - 1])), "backward")
    assert not why and [r["frame"] for r in back] == list(range(N))
    stopped, _ = dot_edits.retrack(tr, "malleolus", 0, (float(p["malleolus"][0]), float(p["malleolus"][1])), "forward", stop_at={10})
    assert [r["frame"] for r in stopped] == list(range(10))                                                           # it stops before a picture fixed by hand
    nothing, why = dot_edits.retrack(tr, "malleolus", 0, (5.0, 5.0), "forward")
    assert nothing == [] and "No dot was found" in why
    tr.close()
    assert dot_edits.retrack(Trial(meta, open_video=False), "malleolus", 0, (1.0, 1.0))[1] == "The video is not open."


def test_the_computed_hip_centre_and_the_pull_about_it_follow_an_edited_asis(frontal):
    meta, p = frontal
    plain, tr = Trial(meta, open_video=False), Trial(meta, open_video=False)
    names = tr.derived_names
    assert names, "the frontal trial has a computed hip centre"
    name = names[0]
    ident = [(d, "retrack", _identity_rows(plain, d), {}) for d in ("asis", "asis_opposite", "patella", "strap", "anchor") if d in plain.tracks]
    dot_edits.apply(tr, _review(meta, ident))
    for c, v in plain.pull.items():
        assert np.allclose(tr.pull[c], v, atol=0.06, equal_nan=True), c
    assert np.allclose(tr.derived_tracks[name][0], plain.derived_tracks[name][0], atol=0.02, equal_nan=True)        # the same hip centre
    moved = Trial(meta, open_video=False)
    ax, ay = plain.tracks["asis"][0][10], plain.tracks["asis"][1][10]
    dot_edits.apply(moved, _review(meta, [("asis", "hold", None, {"from_s": float(plain.t[10]), "to_s": float(plain.t[10]), "x": float(ax) + 30.0, "y": float(ay)})]))
    ppm = float(moved.rows[10]["scale_px_per_cm"]) / 10.0
    spec = next(iter(moved.meta["derived_points"]["definitions"].values()))
    want = hip_centre_frontal((moved.tracks["asis"][0][10], moved.tracks["asis"][1][10]), (moved.tracks["asis_opposite"][0][10], moved.tracks["asis_opposite"][1][10]), ppm,
                              knee_xy=(moved.tracks[spec["knee"]][0][10], moved.tracks[spec["knee"]][1][10]) if spec.get("knee") in moved.tracks else None)
    assert (moved.derived_tracks[name][0][10], moved.derived_tracks[name][1][10]) == pytest.approx(want, abs=0.02)
    assert moved.derived_tracks[name][0][9] == plain.derived_tracks[name][0][9] and moved.derived_tracks[name][0][11] == plain.derived_tracks[name][0][11]
