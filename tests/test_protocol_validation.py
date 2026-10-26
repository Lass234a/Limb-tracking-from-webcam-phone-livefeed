"""COD-06: mistakes in protocol.json are rejected when the program starts, with the place and a hint."""
import copy
import json
from pathlib import Path

import pytest

from limbtrack.protocol import ProtocolError, load_protocol, validate_protocol

ROOT = Path(__file__).resolve().parent.parent
GOOD = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))


def mutated(fn):
    d = copy.deepcopy(GOOD)
    fn(d)
    return d


def test_the_shipped_protocol_is_valid_and_loads():
    assert validate_protocol(GOOD) == []
    assert load_protocol(ROOT / "protocol.json")


def test_the_old_kind_points_spelling_still_loads(tmp_path):
    d = mutated(lambda d: d["tests"]["knee_extension"].update(
        angles=[{"name": "legacy_knee", "kind": "joint", "points": ["trochanter", "epicondyle", "malleolus"]}], primary="legacy_knee",
        neighbour_targets={}))
    assert validate_protocol(d) == []


# (description, change, text that the message must contain)
CASES = [
    ("misspelt arm direction", lambda d: d["angle_library"]["hip"].update(vectors=["Forward", ["trochanter", "epicondyle"]]), 'angle_library.hip.vectors[0]'),
    ("unknown dot in an arm", lambda d: d["angle_library"]["knee"].update(vectors=[["trochanter", "epicondyle"], ["fibula_head", "malleolus"]]), 'fibula_head'),
    ("arm with three dots", lambda d: d["angle_library"]["knee"].update(vectors=[["trochanter", "epicondyle", "x"], ["fibular_head", "malleolus"]]), "exactly two dot names"),
    ("perp arm with one dot", lambda d: d["angle_library"]["hip"].update(vectors=[{"perp": ["iliac_crest"]}, ["trochanter", "epicondyle"]]), "exactly two dot names"),
    ("same dot twice", lambda d: d["angle_library"]["knee"].update(vectors=[["trochanter", "trochanter"], ["fibular_head", "malleolus"]]), "both dots are"),
    ("sign as text", lambda d: d["angle_library"]["hip"].update(sign="-1"), "angle_library.hip.sign"),
    ("offset as text", lambda d: d["angle_library"]["ankle"].update(offset="90"), "angle_library.ankle.offset"),
    ("signed as text", lambda d: d["angle_library"]["hip"].update(signed="yes"), "angle_library.hip.signed"),
    ("unknown field in an angle", lambda d: d["angle_library"]["knee"].update(fulcrm="epicondyle"), 'unknown field "fulcrm"'),
    ("misspelt fulcrum", lambda d: d["angle_library"]["knee"].update(fulcrum="epicondle"), "angle_library.knee.fulcrum"),
    ("misspelt primary", lambda d: d["tests"]["knee_extension"].update(primary="Knee"), 'tests.knee_extension.primary'),
    ("angle not in the library", lambda d: d["tests"]["knee_extension"].update(angles=["knee", "hip_flexion"]), '"hip_flexion" is not in angle_library'),
    ("no angles", lambda d: d["tests"]["knee_extension"].update(angles=[]), "tests.knee_extension.angles"),
    ("target as text", lambda d: d["tests"]["knee_extension"].update(targets=["70"]), "tests.knee_extension.targets"),
    ("target out of range", lambda d: d["tests"]["knee_extension"].update(targets=[270]), "tests.knee_extension.targets"),
    ("negative tolerance", lambda d: d["tests"]["knee_extension"].update(tolerance_deg=-3), "tests.knee_extension.tolerance_deg"),
    ("tolerance as text", lambda d: d.update(tolerance_deg="3"), "tolerance_deg"),
    ("tolerance above what the box allows", lambda d: d["tests"]["knee_extension"].update(neighbour_tolerance_deg=60), "neighbour_tolerance_deg"),
    ("unknown field in a test", lambda d: d["tests"]["knee_extension"].update(targts=[20]), 'unknown field "targts"'),
    ("unknown top-level key", lambda d: d.update(tests_=1), 'unknown top-level key "tests_"'),
    ("missing tests", lambda d: d.pop("tests"), "tests: needs at least one test"),
    ("duplicate dot_order", lambda d: d.update(dot_order=["iliac_crest", "iliac_crest"]), "dot_order"),
    ("unknown optional dot", lambda d: d["tests"]["knee_extension"].update(optional_dots=["strapp"]), "strapp"),
    ("two main joints", lambda d: d["angle_library"]["hip"].update(primary=True) or d["angle_library"]["knee"].update(primary=True) or
     d["tests"]["knee_extension"].pop("primary"), "more than one main joint"),
    ("neighbour target on an unknown angle", lambda d: d["tests"]["knee_extension"].update(neighbour_targets={"hipp": 90}), "neighbour_targets"),
    ("neighbour target on the main joint", lambda d: d["tests"]["knee_extension"].update(neighbour_targets={"knee": 90}), "main joint"),
    # computed points and the view of a test (0.15.0)
    ("unknown kind of computed point", lambda d: d["derived_points"]["hip_centre"].update(kind="hip_center_frontal"),
     'derived_points.hip_centre.kind: "hip_center_frontal" is not a kind of computed point'),
    ("computed point from an unknown dot", lambda d: d["derived_points"]["hip_centre"].update(tested_asis="asiss"), 'unknown dot "asiss"'),
    ("computed point from the same dot twice", lambda d: d["derived_points"]["hip_centre"].update(other_asis="asis"), "the dots must be different"),
    ("computed point named like a dot", lambda d: d["derived_points"].update(asis=dict(d["derived_points"]["hip_centre"])), "already the name of a dot"),
    ("computed point with an unknown field", lambda d: d["derived_points"]["hip_centre"].update(extra=1), 'unknown field "extra"'),
    ("computed point is not an object", lambda d: d["derived_points"].update(hip_centre="asis"), "derived_points.hip_centre: must be an object"),
    ("derived_points is not an object", lambda d: d.update(derived_points=[]), "derived_points: must be an object"),
    ("computed point listed as a dot to mark", lambda d: d["tests"]["hip_abduction"].update(dots=["asis", "hip_centre"]),
     '"hip_centre" is a computed point, not a dot to mark'),
    ("arm ends at an unknown computed point", lambda d: d["angle_library"]["abd_hip_centre"].update(
        vectors=[["asis", "asis_opposite"], ["hip_centre_x", "patella"]]), 'unknown dot "hip_centre_x"'),
    ("judged as text", lambda d: d["angle_library"]["abd_hip_centre"].update(judged="no"), "angle_library.abd_hip_centre.judged"),
    ("unknown view", lambda d: d["tests"]["hip_abduction"].update(view="sideways"), "tests.hip_abduction.view"),
    ("misspelt view", lambda d: d["tests"]["hip_abduction"].update(view="frontl"), 'did you mean "frontal"?'),
    # angles that would silently not do what the protocol says (0.17.1)
    ("signed angle in a frontal test", lambda d: d["angle_library"]["abd_goniometer"].update(signed=True),
     'tests.hip_abduction.angles: "abd_goniometer" is a signed angle'),
    ("main joint marked as not judged", lambda d: d["angle_library"]["abd_goniometer"].update(judged=False),
     '"abd_goniometer" is the main joint, which is always judged'),
    ("required neighbour angle marked as not judged", lambda d: d["angle_library"]["hip"].update(judged=False),
     'tests.knee_extension.neighbour_targets: "hip" has "judged": false'),
]


@pytest.mark.parametrize("desc, change, expect", CASES, ids=[c[0] for c in CASES])
def test_each_mistake_is_rejected_with_its_place(tmp_path, desc, change, expect):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(mutated(change)), encoding="utf-8")
    with pytest.raises(ProtocolError) as e:
        load_protocol(path)
    assert expect in str(e.value), str(e.value)


def test_the_message_suggests_the_right_spelling_and_lists_every_problem(tmp_path):
    d = mutated(lambda d: (d["tests"]["knee_extension"].update(primary="Knee", targts=[1]),
                           d["angle_library"]["hip"].update(vectors=["Forward", ["trochanter", "epicondyle"]])))
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ProtocolError) as e:
        load_protocol(path)
    text = str(e.value)
    assert "3 problem(s)" in text and 'did you mean "knee"?' in text and 'did you mean "targets"?' in text and 'did you mean "forward"?' in text


def test_a_computed_point_counts_as_the_dots_it_is_made_of():
    tests = load_protocol(ROOT / "protocol.json")
    t = tests["hip_abduction"]
    angles = {a.name: a for a in t.angles}
    assert angles["abd_hip_centre"].needs == ["asis", "asis_opposite", "patella"] == angles["abd_hip_centre"].dots
    assert angles["abd_goniometer"].needs is None                          # only dots in its arms
    assert t.derived == {"hip_centre": {"kind": "hip_centre_frontal", "tested_asis": "asis", "other_asis": "asis_opposite", "knee": "patella"}}
    assert t.dots == ["asis", "asis_opposite", "patella"] and "hip_centre" not in t.all_dots     # the hip centre is never clicked
    assert t.view == "frontal" and tests["knee_extension"].view == "sagittal" and tests["knee_extension"].derived == {}
    assert tests["hip_abduction"].all_dots == tests["hip_adduction"].all_dots      # the same dots: switching between them keeps them marked


def test_old_protocol_files_without_computed_points_still_load(tmp_path):
    d = copy.deepcopy(GOOD)
    d.pop("derived_points")
    for k in ("hip_abduction", "hip_adduction"):
        d["tests"].pop(k)
    for k in ("abd_goniometer", "add_goniometer", "abd_hip_centre", "add_hip_centre"):    # (the add_ pair is gone since 0.20.0)
        d["angle_library"].pop(k, None)
    assert validate_protocol(d) == []
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    assert all(t.view == "sagittal" and t.derived == {} for t in load_protocol(path).values())
