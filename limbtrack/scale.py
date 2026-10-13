"""Pixel-to-centimetre scale from a ruler with a dot at each end.

The operator puts a ruler of known length (10-20 cm) at the same distance from the camera as the joint,
with a dot sticker centred on each end mark, marks both dots in the picture and types the ruler length.
The two ruler dots are FIXED points: a click snaps to the dot's centre once and from then on the two positions
never change (they are not tracked, so a shadow or a hand cannot lose them). If the camera or the ruler is moved
after marking, the scale is wrong and nothing notices: mark the ruler again.

The scale is only right for things at the ruler's distance from the camera, in the plane square to the
camera. Lens distortion is not corrected. Nothing here changes any angle: it only provides a conversion
from pixels to centimetres.
"""
import copy
import math

from .tracker import DotState, TrackerSet

NAMES = ("ruler_a", "ruler_b")
MIN_LENGTH_PX = 60        # a ruler shorter than this in the picture gives a coarse scale
MIN_DOT_DIAMETER_CM = 2.0  # protocol requirement: stickers at least this wide
EDGE_BLUR_PX = 0.0        # allowance taken off the size read from the picture. Was 1.3 until 0.7.2, measured on synthetic dots that
                          # cv2.circle draws about 1 px wider than asked. On discs of exactly known size the reading is within
                          # +0.3 px (sharp) to -0.7 px (soft) of the truth, so nothing is taken off. Verify with real stickers.
MIN_SEPARATION_PX = 20    # the two ruler dots must be this far apart (otherwise the same dot was clicked twice)


def estimate_diameter_cm(diameter_px, px_per_cm):
    """Approximate real dot width. The tracker measures the blob above half its peak brightness; on exactly sized discs that
    is within +0.3 px (sharp) to -0.7 px (soft picture) of the true width, so EDGE_BLUR_PX is 0 (no correction). A real
    sticker or hand-drawn disc (ink spreading, lens blur) can differ: verify with a disc of known size."""
    return max(diameter_px - EDGE_BLUR_PX, 0.0) / px_per_cm


class ScaleReference:
    def __init__(self, kind="auto"):
        self.length_cm = 15.0
        self.reset(kind)

    def reset(self, kind=None):
        """Forget the scale (new source, mirror switched, colour setting changed)."""
        if kind is not None:
            self.kind = kind
        self.trackers = TrackerSet(NAMES, self.kind)
        self.states = {}
        self.ref_px = None            # distance between the two dots when they were marked

    @property
    def configured(self):
        return self.ref_px is not None

    def next_to_mark(self):
        for i, n in enumerate(NAMES):
            if not self.trackers.trackers[n].initialised:
                return i
        return None

    # ------------------------------------------------------------------ marking
    def mark(self, index, gray, x, y):
        """Mark ruler dot `index` (0 or 1). Returns (ok, problem); problem is None for a plain 'no dot found'."""
        name, other = NAMES[index], self.trackers.trackers[NAMES[1 - index]]
        old = copy.deepcopy(self.trackers.trackers[name])
        if not self.trackers.init_dot(name, gray, x, y):
            return False, None
        t = self.trackers.trackers[name]
        if other.initialised and math.hypot(t.x - other.x, t.y - other.y) < MIN_SEPARATION_PX:
            self.trackers.trackers[name] = old
            return False, "That is the same dot as the other ruler dot. Click the dot at the other end of the ruler."
        self.states[name] = DotState(name, t.x, t.y, False, 0)
        a, b = (self.trackers.trackers[n] for n in NAMES)
        self.ref_px = math.hypot(a.x - b.x, a.y - b.y) if a.initialised and b.initialised else None
        return True, None

    # ----------------------------------------------------------------- per frame
    def update(self, gray):
        """Nothing to follow: the ruler dots are fixed points (kept so the engine's per-frame call stays the same)."""
        return self.states

    def live_px(self):
        """The ruler's length in pixels (fixed: the length at marking), or None before both dots are marked."""
        return self.ref_px

    @property
    def px_per_cm(self):
        """The scale as marked (pixels per centimetre), or None."""
        return self.ref_px / self.length_cm if self.ref_px and self.length_cm > 0 else None

    def live_px_per_cm(self):
        """The scale for the current frame: the marked scale (it does not change), or None when no scale is set."""
        px = self.live_px()
        return px / self.length_cm if px is not None and self.configured and self.length_cm > 0 else None

    def problems(self):
        """Plain-English warnings about the scale (empty when all is well or no scale is set)."""
        if not self.configured:
            return []
        out = []
        if self.ref_px < MIN_LENGTH_PX:
            out.append(f"ruler is only {self.ref_px:.0f} px long in the picture: the scale is coarse "
                       f"(use a longer ruler or move the camera closer)")
        return out

    # ------------------------------------------------------------------- output
    def overlay_info(self):
        """What to draw on the picture, or None when nothing is marked."""
        pts = [(s.x, s.y, s.lost) for n in NAMES if (s := self.states.get(n)) is not None]
        if not pts:
            return None
        text = ""
        if self.configured:
            text = f"ruler {self.length_cm:g} cm = {self.ref_px:.0f} px  ({self.px_per_cm:.2f} px/cm)"
        return {"points": pts, "text": text, "warn": bool(self.problems())}

    def meta(self, live_values):
        """The scale for meta.json. `live_values`: per-frame px/cm during the recording (None where unavailable)."""
        if not self.configured:
            return None
        valid = [v for v in live_values if v is not None]
        a, b = (self.trackers.trackers[n] for n in NAMES)
        return {
            "ruler_length_cm": self.length_cm,
            "px_per_cm_at_marking": round(self.px_per_cm, 4),
            "ruler_length_px_at_marking": round(self.ref_px, 2),
            "ruler_dots_px_at_marking": [[round(a.x, 2), round(a.y, 2)], [round(b.x, 2), round(b.y, 2)]],
            "ruler_dots_are_fixed_points": True,
            "during_recording": {
                "frames": len(live_values), "frames_with_scale": len(valid),
                "px_per_cm_mean": round(sum(valid) / len(valid), 4) if valid else None,
                "px_per_cm_min": round(min(valid), 4) if valid else None,
                "px_per_cm_max": round(max(valid), 4) if valid else None,
            },
            "valid_for": "things at the ruler's distance from the camera, in the plane square to the camera; "
                         "lens distortion is not corrected; the ruler dots are fixed points, so a camera or ruler moved after "
                         "marking is not noticed",
        }
