"""Pixel-to-centimetre scale from a ruler with a dot at each end.

The operator puts a ruler of known length (10-20 cm) at the same distance from the camera as the joint,
with a dot sticker centred on each end mark, marks both dots in the picture and types the ruler length.
The two dots are then tracked like any other dot, so the app sees if the scale changes during a trial
(camera or ruler moved).

The scale is only right for things at the ruler's distance from the camera, in the plane square to the
camera. Lens distortion is not corrected. Nothing here changes any angle: it only provides a conversion
from pixels to centimetres.
"""
import copy
import math

from .tracker import DotState, TrackerSet

NAMES = ("ruler_a", "ruler_b")
MIN_LENGTH_PX = 60        # a ruler shorter than this in the picture gives a coarse scale
DRIFT_WARN = 0.03         # live scale differing from the marked one by more than 3 % is flagged
MIN_DOT_DIAMETER_CM = 2.0  # protocol requirement: stickers at least this wide
EDGE_BLUR_PX = 1.3        # the size read from the picture runs this many pixels too large (soft edges); measured on synthetic dots
MIN_SEPARATION_PX = 20    # the two ruler dots must be this far apart (otherwise the same dot was clicked twice)


def estimate_diameter_cm(diameter_px, px_per_cm):
    """Approximate real dot width. The tracker measures the blob above half its peak brightness; with soft edges this
    reads about EDGE_BLUR_PX too large, which is removed here. Verify with real stickers of known size."""
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
        if any(t.initialised for t in self.trackers.trackers.values()):
            self.states = self.trackers.update(gray)
        return self.states

    def live_px(self):
        a, b = (self.states.get(n) for n in NAMES)
        if a is None or b is None or a.lost or b.lost:
            return None
        return math.hypot(a.x - b.x, a.y - b.y)

    @property
    def px_per_cm(self):
        """The scale as marked (pixels per centimetre), or None."""
        return self.ref_px / self.length_cm if self.ref_px and self.length_cm > 0 else None

    def live_px_per_cm(self):
        """The scale measured in the current frame, or None when a ruler dot is lost or no scale is set."""
        px = self.live_px()
        return px / self.length_cm if px is not None and self.configured and self.length_cm > 0 else None

    def drift(self):
        """Relative change of the live scale from the marked one (0.02 = 2 % larger), or None."""
        live = self.live_px_per_cm()
        return live / self.px_per_cm - 1.0 if live is not None and self.px_per_cm else None

    def problems(self):
        """Plain-English warnings about the scale (empty when all is well or no scale is set)."""
        if not self.configured:
            return []
        out = []
        if self.ref_px < MIN_LENGTH_PX:
            out.append(f"ruler is only {self.ref_px:.0f} px long in the picture: the scale is coarse "
                       f"(use a longer ruler or move the camera closer)")
        lost = [n for n in NAMES if n in self.states and self.states[n].lost]
        if lost:
            out.append("ruler dot lost: the live scale check is off")
        d = self.drift()
        if d is not None and abs(d) > DRIFT_WARN:
            out.append(f"ruler scale has changed by {100 * d:+.1f} %: has the camera or the ruler moved?")
        return out

    # ------------------------------------------------------------------- output
    def overlay_info(self):
        """What to draw on the picture, or None when nothing is marked."""
        pts = [(s.x, s.y, s.lost) for n in NAMES if (s := self.states.get(n)) is not None]
        if not pts:
            return None
        text = ""
        if self.configured:
            live = self.live_px_per_cm()
            text = f"ruler {self.length_cm:g} cm = {self.ref_px:.0f} px  ({self.px_per_cm:.2f} px/cm)"
            if live is None:
                text += "  LOST"
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
            "during_recording": {
                "frames": len(live_values), "frames_with_scale": len(valid),
                "px_per_cm_mean": round(sum(valid) / len(valid), 4) if valid else None,
                "px_per_cm_min": round(min(valid), 4) if valid else None,
                "px_per_cm_max": round(max(valid), 4) if valid else None,
            },
            "valid_for": "things at the ruler's distance from the camera, in the plane square to the camera; "
                         "lens distortion is not corrected",
        }
