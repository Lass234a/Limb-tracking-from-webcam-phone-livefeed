"""Angle of pull and moment arm from the strap marker and the anchor marker.

The strap marker sits on the strap where it meets the person, the anchor marker at the fixed end of the
strap/rope/chain. The line between them is the line of pull; the pull points from the strap marker
towards the anchor marker.

* pull angle: angle between the pull direction and the segment the strap acts on (the second arm of the test's
  main joint: shank in the knee tests, thigh in the hip tests, foot in the ankle tests), measured along the
  segment from its proximal dot to its distal dot. 90 deg means the pull is square to the segment; below 90
  the pull has a component towards the feet, above 90 towards the body.
* moment arm: shortest distance from the joint (the main joint's fulcrum dot) to the line of pull. In the frontal
  tests (hip abduction / adduction) the joint is the computed hip centre and the segment runs from it to the patella
  centre (protocol.json "pull_joint", 0.25.0); the ASIS, the fulcrum of the Tabel 3 angle, is not the joint.
* strap-joint distance: straight distance from the joint to the strap marker.

All of this is the 2D picture: a pull toward or away from the camera is invisible. Distances in cm use the ruler
scale (limbtrack/scale.py) and are only right at the ruler's distance from the camera. Nothing here is judged.
"""
import math

COLUMNS = ["pull_angle_deg", "pull_off90_deg", "moment_arm_px", "moment_arm_cm", "strap_joint_px", "strap_joint_cm"]


def compute_pull(strap, anchor, seg_start, seg_end, fulcrum, px_per_cm=None):
    """All arguments are (x, y). Returns a dict with the COLUMNS keys (cm values None without a scale), or None if
    the strap and anchor coincide or the segment has no length."""
    ux, uy = anchor[0] - strap[0], anchor[1] - strap[1]
    sx, sy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]
    nu, ns = math.hypot(ux, uy), math.hypot(sx, sy)
    if nu < 1e-6 or ns < 1e-6:
        return None
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (ux * sx + uy * sy) / (nu * ns)))))
    fx, fy = fulcrum[0] - strap[0], fulcrum[1] - strap[1]
    arm_px = abs(ux * fy - uy * fx) / nu
    joint_px = math.hypot(fx, fy)
    cm = (lambda px: px / px_per_cm) if px_per_cm else (lambda px: None)
    return {"pull_angle_deg": angle, "pull_off90_deg": angle - 90.0, "moment_arm_px": arm_px,
            "moment_arm_cm": cm(arm_px), "strap_joint_px": joint_px, "strap_joint_cm": cm(joint_px)}


def describe(p):
    """One line for the picture and the operator."""
    arm = f"{p['moment_arm_cm']:.1f} cm" if p["moment_arm_cm"] is not None else f"{p['moment_arm_px']:.0f} px (no scale)"
    return f"PULL {p['pull_angle_deg']:.1f} deg to the segment ({p['pull_off90_deg']:+.1f} from perpendicular); moment arm {arm}"
