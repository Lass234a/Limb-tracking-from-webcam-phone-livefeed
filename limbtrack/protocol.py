"""Test definitions, read from protocol.json so they can be edited without touching code."""
import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import DERIVED_KINDS, derived_input_dots, spec_dots

VIEWS = ("sagittal", "frontal")      # which way the camera looks at the movement; a frontal test has no 'Participant faces' (0.15.0)


@dataclass
class AngleDef:
    name: str
    vectors: list                 # two arms, see geometry.py (an arm's end may be a computed point, see geometry.compute_derived)
    label: str = ""
    signed: bool = False
    sign: float = 1.0
    offset: float = 0.0
    primary: bool = False
    fulcrum: str = None           # dot (or computed point) where the read-out is drawn
    needs: list = None            # the dots clicked for this angle, when an arm uses a computed point (filled by load_protocol; saved in meta.json)
    judged: bool = True           # False: shown and saved, but never judged green/red (not against a target, a required neighbour angle or the locked value)

    @property
    def title(self):
        return self.label or self.name

    @property
    def dots(self):
        """The dots that must be marked and tracked for this angle (a computed point counts as the dots it is made of)."""
        if self.needs is not None:
            return list(self.needs)
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
    neighbour_targets: dict = field(default_factory=dict)   # angle name -> required angle of that neighbouring joint (Tabel 2: H 90, K 0, K 90)
    optional_dots: list = field(default_factory=list)   # extra tracked dots (strap, anchor): off until switched on, never part of an angle
    view: str = "sagittal"                        # "frontal": the camera is in front of the participant (hip abduction/adduction)
    derived: dict = field(default_factory=dict)   # computed points this test's angles use: name -> definition (geometry.DERIVED_KINDS)
    reference_extras: dict = field(default_factory=dict)   # readings the operator may type that are not one of the angles (key -> name), e.g. the hip flexion of a frontal test

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


_TOP_KEYS = {"tolerance_deg", "neighbour_tolerance_deg", "dot_order", "dot_labels", "optional_dots", "derived_points", "angle_library", "tests"}
_TEST_KEYS = {"label", "primary", "targets", "angles", "dots", "tolerance_deg", "neighbour_tolerance_deg", "optional_dots",
              "neighbour_targets", "view", "reference_extras"}
_ANGLE_KEYS = {"name", "vectors", "label", "signed", "sign", "offset", "primary", "fulcrum", "kind", "points", "judged"}
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
    for key in ("signed", "primary", "judged"):
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
    derived = data.get("derived_points", {})
    if not isinstance(derived, dict):
        errs.append("derived_points: must be an object {name: definition}")
        derived = {}
    derived = {k: v for k, v in derived.items() if not str(k).startswith("_")}
    for dname, spec in derived.items():                   # computed points: made from dots, never clicked
        path = f"derived_points.{dname}"
        if dname in known:
            errs.append(f'{path}: "{dname}" is already the name of a dot; a computed point needs a name of its own')
        if not isinstance(spec, dict):
            errs.append(f"{path}: must be an object")
            continue
        kind = spec.get("kind")
        if kind not in DERIVED_KINDS:
            errs.append(f'{path}.kind: "{kind}" is not a kind of computed point{_near(kind, DERIVED_KINDS)}; use one of '
                        + ", ".join(DERIVED_KINDS))
            continue
        keys = DERIVED_KINDS[kind]
        for k in spec:
            if k != "kind" and k not in keys and not str(k).startswith("_"):
                errs.append(f'{path}: unknown field "{k}"{_near(k, keys)}')
        for k in keys:
            v = spec.get(k)
            if not isinstance(v, str):
                errs.append(f"{path}.{k}: needs the name of a dot")
            elif v not in known:
                errs.append(f'{path}.{k}: unknown dot "{v}"{_near(v, known)}')
        names = [spec.get(k) for k in keys]
        if all(isinstance(n, str) for n in names) and len(set(names)) < len(names):
            errs.append(f"{path}: the dots must be different")
    known_arms = known + [d for d in derived if d not in known]       # an arm may end at a dot or at a computed point
    library = data.get("angle_library", {})
    if not isinstance(library, dict):
        errs.append("angle_library: must be an object")
        library = {}
    for name, d in library.items():
        _check_angle(d, f"angle_library.{name}", known_arms, errs)
    tests = data.get("tests")
    if not isinstance(tests, dict) or not tests:
        errs.append("tests: needs at least one test")
        return errs
    for key, t in tests.items():
        if str(key).startswith("_"):          # comment lines
            continue
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
                _check_angle(item, f"{path}.angles[{i}]", known_arms, errs)
                if isinstance(item, dict) and item.get("primary") is True:
                    primaries += 1
            names.append(_angle_name(item))
        if t.get("view") is not None and t["view"] not in VIEWS:
            errs.append(f'{path}.view: "{t["view"]}" is not a view; use one of {", ".join(VIEWS)}{_near(t["view"], VIEWS)}')
        extras = t.get("reference_extras")
        if extras is not None:
            if not (isinstance(extras, dict) and all(isinstance(k, str) and isinstance(v, str) and k and v for k, v in extras.items())):
                errs.append(f"{path}.reference_extras: must be an object {{key: name}} of text, for example "
                            '{"hip_flexion": "Hip flexion (inclinometer)"}')
            else:
                errs += [f'{path}.reference_extras: "{k}" is also the name of an angle of this test; use another key'
                         for k in extras if k in [n for n in names if n]]
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
                    errs += [f'{path}.{key2}: "{n}" is a computed point, not a dot to mark' if n in derived
                             else f'{path}.{key2}: unknown dot "{n}"{_near(n, known)}' for n in v if n not in known]
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
    derived_defs = {k: v for k, v in data.get("derived_points", {}).items() if not str(k).startswith("_")}
    tests = {}
    for key, t in data["tests"].items():
        if key.startswith("_"):
            continue
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
            a.needs = _clicked_dots(a, derived_defs)
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
            neighbour_targets=dict(t.get("neighbour_targets", {})),
            view=t.get("view", "sagittal"),
            reference_extras=dict(t.get("reference_extras", {})),
            derived={n: dict(derived_defs[n]) for a in angles for v in a.vectors for n in spec_dots(v) if n in derived_defs},
        )
    return tests


def _clicked_dots(angle, derived_defs):
    """None when the angle's arms use only dots. Otherwise the dots to click for it: each computed point replaced by the dots it is made of."""
    names = [n for v in angle.vectors for n in spec_dots(v)]
    if not any(n in derived_defs for n in names):
        return None
    out = []
    for n in names:
        for m in (derived_input_dots(derived_defs[n]) if n in derived_defs else [n]):
            if m not in out:
                out.append(m)
    return out


if __name__ == "__main__":      # python -m limbtrack.protocol check [file]
    import sys

    target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "protocol.json"
    try:
        load_protocol(target)
    except (ProtocolError, ValueError, OSError) as e:
        print(e)
        sys.exit(1)
    print(f"{target.name}: OK")
