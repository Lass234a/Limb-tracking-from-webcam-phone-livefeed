from dataclasses import dataclass, field

OK, OUT, LOST, INFO = "ok", "out", "lost", "info"


@dataclass
class AngleResult:
    name: str
    title: str
    kind: str                  # "joint" or "vertical"
    points: list               # dot names, vertex is points[1] for joints
    primary: bool
    value: float = float("nan")      # in the protocol's convention (NaN if a dot is lost)
    reference: float = None          # target (primary) or locked value (neighbours)
    deviation: float = float("nan")  # value - reference
    tolerance: float = None
    status: str = INFO               # ok / out / lost / info (no reference yet)


@dataclass
class FrameResult:
    frame_index: int
    t: float                                   # seconds since the camera started
    overlay: object = None                     # BGR image with drawings
    angles: list = field(default_factory=list)
    dots: dict = field(default_factory=dict)   # name -> DotState
