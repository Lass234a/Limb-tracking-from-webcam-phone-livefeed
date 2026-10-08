"""Test definitions, read from protocol.json so they can be edited without touching code."""
import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import spec_dots


@dataclass
class AngleDef:
    name: str
    vectors: list                 # two arms, see geometry.py
    label: str = ""
    signed: bool = False
    sign: float = 1.0
    offset: float = 0.0
    primary: bool = False
    fulcrum: str = None           # dot where the read-out is drawn

    @property
    def title(self):
        return self.label or self.name

    @property
    def dots(self):
        return [d for v in self.vectors for d in spec_dots(v)]


@dataclass
class TestDef:
    key: str
    label: str
    dots: list
    angles: list
    targets: list = field(default_factory=list)   # suggested primary-angle positions
    tolerance_deg: float = 3.0
    neighbour_tolerance_deg: float = 5.0
    dot_labels: dict = field(default_factory=dict)
    optional_dots: list = field(default_factory=list)   # extra tracked dots (strap, anchor): off until switched on, never part of an angle

    @property
    def all_dots(self):
        """The angle landmarks followed by the optional dots."""
        return list(self.dots) + [d for d in self.optional_dots if d not in self.dots]

    @property
    def primary(self):
        for a in self.angles:
            if a.primary:
                return a
        return self.angles[0]

    def dot_title(self, name):
        return self.dot_labels.get(name, name)


def _legacy(d):
    """Convert the v0.1 'kind'/'points' spelling into arms."""
    d = dict(d)
    kind, pts = d.pop("kind", None), d.pop("points", None)
    if kind == "joint":
        a, b, c = pts
        d["vectors"] = [[b, a], [b, c]]
        d.setdefault("fulcrum", b)
    elif kind == "vertical":
        p, q = pts
        d["vectors"] = [[p, q], "up"]
        d.setdefault("fulcrum", p)
    elif kind is not None:
        raise ValueError(f"unknown angle kind '{kind}'")
    return d



class ProtocolError(ValueError):
    """protocol.json has mistakes; the message lists all of them with their place in the file."""


_TOP_KEYS = {"tolerance_deg", "neighbour_tolerance_deg", "dot_order", "dot_labels", "optional_dots", "angle_library", "tests"}
_TEST_KEYS = {"label", "primary", "targets", "angles", "dots", "tolerance_deg", "neighbour_tolerance_deg", "optional_dots",
              "neighbour_targets"}
_ANGLE_KEYS = {"name", "vectors", "label", "signed", "sign", "offset", "primary", "fulcrum", "kind", "points"}
_FIXED_ARMS = ("up", "down", "forward")
_MAX_TOLERANCE = (30.0, 45.0)        # the ranges of the tolerance boxes in the window (main joint, other joints)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _near(word, options):
    m = difflib.get_close_matches(str(word), [str(o) for o in options], n=1)
    return f' (did you mean "{m[0]}"?)' if m else ""


def _check_arm(arm, path, known, errs):
    if isinstance(arm, str):
        if arm not in _FIXED_ARMS:
            errs.append(f'{path}: "{arm}" is not a direction; use one of {", ".join(_FIXED_ARMS)}{_near(arm, _FIXED_ARMS)}')
        return
    names = arm.get("perp") if isinstance(arm, dict) else arm
    if isinstance(arm, dict) and set(arm) != {"perp"}:
        errs.append(f'{path}: a dict arm must be exactly {{"perp": [dotA, dotB]}}')
        return
    if not (isinstance(names, list) and len(names) == 2 and all(isinstance(n, str) for n in names)):
        errs.append(f"{path}: an arm needs exactly two dot names, a direction ({', '.join(_FIXED_ARMS)}) or {{\"perp\": [A, B]}}")
        return
    if names[0] == names[1]:
        errs.append(f'{path}: both dots are "{names[0]}"')
    for n in names:
        if n not in known:
            errs.append(f'{path}: unknown dot "{n}"{_near(n, known)}')


def _check_angle(d, path, known, errs):
    if not isinstance(d, dict):
        errs.append(f"{path}: must be an object")
        return
    try:
        d = _legacy(d)
    except (ValueError, TypeError, KeyError) as e:
        errs.append(f"{path}: {e}")
        return
    for k in d:
        if k not in _ANGLE_KEYS:
            errs.append(f'{path}: unknown field "{k}"{_near(k, _ANGLE_KEYS)}')
    vectors = d.get("vectors")
    if not isinstance(vectors, list) or len(vectors) != 2:
        errs.append(f"{path}.vectors: an angle needs exactly two arms")
    else:
        for i, arm in enumerate(vectors):
            _check_arm(arm, f"{path}.vectors[{i}]", known, errs)
        arm_dots = []
        for arm in vectors:
            names = arm.get("perp") if isinstance(arm, dict) else arm
            if isinstance(names, list):
                arm_dots += [n for n in names if isinstance(n, str)]
        f = d.get("fulcrum")
        if f is not None and f not in arm_dots:
            errs.append(f'{path}.fulcrum: "{f}" is not one of the angle\'s dots {sorted(set(arm_dots))}{_near(f, arm_dots)}')
    for key in ("sign", "offset"):
        if key in d and not _num(d[key]):
            errs.append(f"{path}.{key}: must be a number, not {d[key]!r}")
    for key in ("signed", "primary"):
        if key in d and not isinstance(d[key], bool):
            errs.append(f"{path}.{key}: must be true or false, not {d[key]!r}")


def _angle_name(item):
    return item if isinstance(item, str) else (item.get("name") if isinstance(item, dict) else None)


def validate_protocol(data):
    """Every mistake found in the protocol data, as plain-English lines starting with their place in the file."""
    errs = []
    if not isinstance(data, dict):
        return ["protocol.json: the top level must be an object"]
    for k in data:
        if k not in _TOP_KEYS and not str(k).startswith("_"):
            errs.append(f'protocol.json: unknown top-level key "{k}"{_near(k, _TOP_KEYS)}')
    for key, mx in (("tolerance_deg", _MAX_TOLERANCE[0]), ("neighbour_tolerance_deg", _MAX_TOLERANCE[1])):
        if key in data and not (_num(data[key]) and 0 < data[key] <= mx):
            errs.append(f"{key}: must be a number above 0 and at most {mx:g}, not {data[key]!r}")
    order = data.get("dot_order", [])
    if not (isinstance(order, list) and all(isinstance(n, str) for n in order)) or len(set(order)) != len(order):
        errs.append("dot_order: must be a list of different dot names")
        order = [n for n in order if isinstance(n, str)] if isinstance(order, list) else []
    labels = data.get("dot_labels", {})
    optional = data.get("optional_dots", [])
    if not isinstance(labels, dict):
        errs.append("dot_labels: must be an object")
        labels = {}
    if not isinstance(optional, list):
        errs.append("optional_dots: must be a list of dot names")
        optional = []
    known = list(dict.fromkeys(list(order) + list(labels) + [n for n in optional if isinstance(n, str)]))
    library = data.get("angle_library", {})
    if not isinstance(library, dict):
        errs.append("angle_library: must be an object")
        library = {}
    for name, d in library.items():
        _check_angle(d, f"angle_library.{name}", known, errs)
    tests = data.get("tests")
    if not isinstance(tests, dict) or not tests:
        errs.append("tests: needs at least one test")
        return errs
    for key, t in tests.items():
        path = f"tests.{key}"
        if not isinstance(t, dict):
            errs.append(f"{path}: must be an object")
            continue
        for k in t:
            if k not in _TEST_KEYS and not str(k).startswith("_"):
                errs.append(f'{path}: unknown field "{k}"{_near(k, _TEST_KEYS)}')
        items = t.get("angles")
        if not isinstance(items, list) or not items:
            errs.append(f"{path}.angles: needs at least one angle")
            items = []
        names, primaries = [], 0
        for i, item in enumerate(items):
            if isinstance(item, str):
                if item not in library:
                    errs.append(f'{path}.angles[{i}]: "{item}" is not in angle_library{_near(item, library)}')
                elif isinstance(library[item], dict) and library[item].get("primary") is True:
                    primaries += 1
            else:
                _check_angle(item, f"{path}.angles[{i}]", known, errs)
                if isinstance(item, dict) and item.get("primary") is True:
                    primaries += 1
            names.append(_angle_name(item))
        primary = t.get("primary")
        if primary is not None and primary not in names:
            errs.append(f'{path}.primary: "{primary}" is not one of this test\'s angles {[n for n in names if n]}{_near(primary, [n for n in names if n])}')
        if primaries > 1 or (primaries == 1 and primary is not None and primary in names
                             and not (isinstance(library.get(primary), dict) and library[primary].get("primary") is True)):
            errs.append(f"{path}: more than one main joint (primary) is defined")
        targets = t.get("targets", [])
        if not isinstance(targets, list) or not all(_num(v) and -180 <= v <= 180 for v in targets):
            errs.append(f"{path}.targets: must be a list of numbers between -180 and 180, not {targets!r}")
        for key2, mx in (("tolerance_deg", _MAX_TOLERANCE[0]), ("neighbour_tolerance_deg", _MAX_TOLERANCE[1])):
            if key2 in t and not (_num(t[key2]) and 0 < t[key2] <= mx):
                errs.append(f"{path}.{key2}: must be a number above 0 and at most {mx:g}, not {t[key2]!r}")
        for key2 in ("dots", "optional_dots"):
            if key2 in t:
                v = t[key2]
                if not isinstance(v, list) or not all(isinstance(n, str) for n in v):
                    errs.append(f"{path}.{key2}: must be a list of dot names")
                else:
                    errs += [f'{path}.{key2}: unknown dot "{n}"{_near(n, known)}' for n in v if n not in known]
        nt = t.get("neighbour_targets")
        if nt is not None:
            if not isinstance(nt, dict):
                errs.append(f"{path}.neighbour_targets: must be an object {{angle name: required angle}}")
            else:
                for n, v in nt.items():
                    if n not in names:
                        errs.append(f'{path}.neighbour_targets: "{n}" is not one of this test\'s angles{_near(n, [x for x in names if x])}')
                    elif n == primary or (isinstance(library.get(n), dict) and library[n].get("primary") is True):
                        errs.append(f'{path}.neighbour_targets: "{n}" is the main joint; its target is set with "targets"')
                    if not (_num(v) and -360 <= v <= 360):
                        errs.append(f"{path}.neighbour_targets.{n}: must be a number, not {v!r}")
    return errs


def load_protocol(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_protocol(data)
    if errors:
        raise ProtocolError(f"{Path(path).name} has {len(errors)} problem(s):\n  - " + "\n  - ".join(errors))
    library = data.get("angle_library", {})
    order = data.get("dot_order", [])
    labels = data.get("dot_labels", {})
    tests = {}
    for key, t in data["tests"].items():
        angles = []
        for item in t["angles"]:
            if isinstance(item, str):
                d = dict(library[item])
                d.setdefault("name", item)
            else:
                d = _legacy(item)
            a = AngleDef(**d)
            a.primary = a.primary or a.name == t.get("primary")
            if len(a.vectors) != 2:
                raise ValueError(f"{key}/{a.name}: an angle needs exactly two arms")
            angles.append(a)
        used = []
        for a in angles:
            for dname in a.dots:
                if dname not in used:
                    used.append(dname)
        dots = t.get("dots") or sorted(used, key=lambda n: order.index(n) if n in order else len(order))
        for a in angles:
            missing = [d for d in a.dots if d not in dots]
            if missing:
                raise ValueError(f"{key}/{a.name}: uses dots not in the test's dot list: {missing}")
        tests[key] = TestDef(
            key=key,
            label=t.get("label", key),
            dots=dots,
            angles=angles,
            targets=t.get("targets", []),
            tolerance_deg=t.get("tolerance_deg", data.get("tolerance_deg", 3.0)),
            neighbour_tolerance_deg=t.get("neighbour_tolerance_deg", data.get("neighbour_tolerance_deg", 5.0)),
            dot_labels=labels,
            optional_dots=list(t.get("optional_dots", data.get("optional_dots", []))),
        )
    return tests


if __name__ == "__main__":      # python -m limbtrack.protocol check [file]
    import sys

    target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "protocol.json"
    try:
        load_protocol(target)
    except (ProtocolError, ValueError, OSError) as e:
        print(e)
        sys.exit(1)
    print(f"{target.name}: OK")
