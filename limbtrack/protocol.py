"""Test definitions, read from protocol.json so they can be edited without touching code."""
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


def load_protocol(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
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
        )
    return tests
