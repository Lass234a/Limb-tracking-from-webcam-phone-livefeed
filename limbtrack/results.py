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
    # What the operator SEES (smoothed over 0.2 s if smoothing is on). Used by the screen, trace and overlay video.
    value: float = float("nan")      # in the protocol's convention (NaN if a dot is lost)
    deviation: float = float("nan")  # value - reference
    status: str = INFO               # ok / out / lost / info (no reference yet)
    # What is RECORDED in the CSV: the unsmoothed angle and the verdicts computed from it.
    raw_value: float = float("nan")
    raw_deviation: float = float("nan")
    raw_status: str = INFO
    reference: float = None          # target (main joint) or the value shown when the position was locked (others)
    tolerance: float = None


@dataclass
class FrameResult:
    frame_index: int
    t: float                                   # seconds since the camera started
    overlay: object = None                     # BGR image with drawings
    angles: list = field(default_factory=list)
    dots: dict = field(default_factory=dict)   # name -> DotState
    header: list = field(default_factory=list)  # text lines drawn at the top of the picture
    # PARKED-GHOST: ghost: dict = None                         # locked pose (name -> (x, y)) if one is being shown
