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
        angles=[{"name": "legacy_knee", "kind": "joint", "points": ["trochanter", "epicondyle", "malleolus"]}], primary="legacy_knee"))
    assert validate_protocol(d) == []


# (description, change, text that the message must contain)
CASES = [
    ("misspelt arm direction", lambda d: d["angle_library"]["pelvic_tilt"].update(vectors=["Forward", ["psis", "asis"]]), 'angle_library.pelvic_tilt.vectors[0]'),
    ("unknown dot in an arm", lambda d: d["angle_library"]["knee"].update(vectors=[["trochanter", "epicondyle"], ["fibula_head", "malleolus"]]), 'fibula_head'),
    ("arm with three dots", lambda d: d["angle_library"]["knee"].update(vectors=[["trochanter", "epicondyle", "x"], ["fibular_head", "malleolus"]]), "exactly two dot names"),
    ("perp arm with one dot", lambda d: d["angle_library"]["hip"].update(vectors=[{"perp": ["asis"]}, ["trochanter", "epicondyle"]]), "exactly two dot names"),
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
    ("duplicate dot_order", lambda d: d.update(dot_order=["asis", "asis"]), "dot_order"),
    ("unknown optional dot", lambda d: d["tests"]["knee_extension"].update(optional_dots=["strapp"]), "strapp"),
    ("two main joints", lambda d: d["angle_library"]["hip"].update(primary=True) or d["angle_library"]["knee"].update(primary=True) or
     d["tests"]["knee_extension"].pop("primary"), "more than one main joint"),
    ("neighbour target on an unknown angle", lambda d: d["tests"]["knee_extension"].update(neighbour_targets={"hipp": 90}), "neighbour_targets"),
    ("neighbour target on the main joint", lambda d: d["tests"]["knee_extension"].update(neighbour_targets={"knee": 90}), "main joint"),
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
                           d["angle_library"]["pelvic_tilt"].update(vectors=["Forward", ["psis", "asis"]])))
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ProtocolError) as e:
        load_protocol(path)
    text = str(e.value)
    assert "3 problem(s)" in text and 'did you mean "knee"?' in text and 'did you mean "targets"?' in text and 'did you mean "forward"?' in text
