from dataclasses import dataclass, field

OK, OUT, LOST, INFO = "ok", "out", "lost", "info"


@dataclass
class AngleResult:
    name: str
    title: str
    primary: bool
    segments: list = field(default_factory=list)   # (dotA, dotB) pairs to draw as limb lines
    rays: list = field(default_factory=list)       # unit vectors of arms that are not dot-to-dot (drawn from the fulcrum)
    fulcrum: tuple = None                          # image position of the read-out
    dir1: tuple = None                             # unit vectors of the two arms (image coords), for the arc
    dir2: tuple = None
    value: float = float("nan")      # in the protocol's convention, smoothed (NaN if a dot is lost)
    raw_value: float = float("nan")  # same, before smoothing
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
    header: list = field(default_factory=list)  # text lines drawn at the top of the picture
    ghost: dict = None                         # locked pose (name -> (x, y)) if one is being shown
