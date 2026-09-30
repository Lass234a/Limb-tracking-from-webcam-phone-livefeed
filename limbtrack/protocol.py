"""Test definitions, read from protocol.json so they can be edited without touching code."""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AngleDef:
    name: str
    kind: str                # "joint" (3 dots, angle at the middle one) or "vertical" (2 dots vs vertical)
    points: list             # dot names
    sign: float = 1.0
    offset: float = 0.0
    primary: bool = False
    label: str = ""

    @property
    def title(self):
        return self.label or self.name


@dataclass
class TestDef:
    key: str
    label: str
    dots: list
    angles: list
    targets: list = field(default_factory=list)   # suggested primary-angle positions
    tolerance_deg: float = 3.0
    neighbour_tolerance_deg: float = 5.0

    @property
    def primary(self):
        for a in self.angles:
            if a.primary:
                return a
        return self.angles[0]


def load_protocol(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tests = {}
    for key, t in data["tests"].items():
        angles = [AngleDef(**a) for a in t["angles"]]
        used = []
        for a in angles:
            for p in a.points:
                if p not in used:
                    used.append(p)
        dots = t.get("dots") or used
        for a in angles:
            need = 3 if a.kind == "joint" else 2
            if len(a.points) != need or any(p not in dots for p in a.points):
                raise ValueError(f"{key}/{a.name}: needs {need} points that are all listed in 'dots'")
        tests[key] = TestDef(
            key=key,
            label=t.get("label", key),
            dots=dots,
            angles=angles,
            targets=t.get("targets", []),
            tolerance_deg=t.get("tolerance_deg", data.get("tolerance_deg", 3.0)),
            neighbour_tolerance_deg=t.get("neighbour_tolerance_deg", data.get("neighbour_tolerance_deg", 5.0)),
        )
    return tests
