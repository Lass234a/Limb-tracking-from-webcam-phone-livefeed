"""How far each dot moved during a chosen stretch of a trial (Review tab and export).

Everything is computed after the fact from the recorded CSV positions (pixels in the picture). A dot's
displacement is measured from its starting position in the window, taken as the mean of its first few valid
positions, to its later positions:

* end shift: mean of the last few valid positions minus the starting position (a vector and its length);
* max shift: the largest distance from the starting position at any frame (a single bad tracking frame would
  show up here; look at the trails).

If the anchor marker is present (the fixed end of the strap, which should not move), the same numbers are also
given for each dot's position relative to it. That removes movement of the whole picture (a camera that was
bumped); it does not separate the person shifting from a strap sliding on the skin. For the strap, look at the
pull numbers (strap-to-joint distance and moment arm) instead.

Centimetres use the ruler scale as marked and are only right at the ruler's distance from the camera.
"""
import math

import numpy as np

AVERAGE_N = 5          # positions averaged at the start and at the end of the window


def _stats(x, y, ppc):
    """Displacement of one track (x, y arrays with NaN where unknown) from its first valid positions."""
    ok = ~(np.isnan(x) | np.isnan(y))
    n_valid = int(ok.sum())
    nan = math.nan
    out = {"frames": int(len(x)), "frames_missing": int(len(x) - n_valid),
           "end_dx_px": nan, "end_dy_px": nan, "end_shift_px": nan, "max_shift_px": nan}
    if n_valid < 2:
        return _add_cm(out, ppc)
    gx, gy = x[ok], y[ok]
    x0, y0 = float(np.mean(gx[:AVERAGE_N])), float(np.mean(gy[:AVERAGE_N]))
    dx, dy = float(np.mean(gx[-AVERAGE_N:])) - x0, float(np.mean(gy[-AVERAGE_N:])) - y0
    out.update(end_dx_px=dx, end_dy_px=dy, end_shift_px=math.hypot(dx, dy),
               max_shift_px=float(np.max(np.hypot(gx - x0, gy - y0))))
    return _add_cm(out, ppc)


def _add_cm(out, ppc):
    for k in ("end_dx", "end_dy", "end_shift", "max_shift"):
        out[f"{k}_cm"] = out[f"{k}_px"] / ppc if ppc else math.nan
    return out


def window_displacements(t, tracks, t0, t1, px_per_cm=None, reference=None):
    """Displacement of every dot within [t0, t1] seconds.

    `tracks`: dot name -> (x array, y array) over the whole trial, NaN where the dot is lost.
    `reference`: name of a dot that should not move (the anchor), or None. When given, each other dot also gets
    `rel_` numbers: its position relative to the reference dot, in the frames where both are known.
    Returns a list of dicts, one per dot, in the order of `tracks`.
    """
    m = (t >= t0) & (t <= t1)
    rows = []
    for name, (x, y) in tracks.items():
        row = {"dot": name}
        row.update(_stats(x[m], y[m], px_per_cm))
        if reference is not None and name != reference and reference in tracks:
            rx, ry = tracks[reference]
            rel = _stats((x - rx)[m], (y - ry)[m], px_per_cm)
            row.update({f"rel_{k}": v for k, v in rel.items() if k.startswith(("end_", "max_"))})
        rows.append(row)
    return rows
