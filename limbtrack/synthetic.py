"""Synthetic leg videos with dots at exactly known angles.

Used by the tests, and handy for trying the app without a camera:
    python -m limbtrack.synthetic demo.mp4
then use "Open video file" in the app.
"""
import math
import sys

import cv2
import numpy as np


def _rot(v, deg):
    """Rotate v by deg in image coordinates (y down): positive = clockwise on screen."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def leg_points(hip_flex=0.0, knee_flex=0.0, ankle_df=0.0, pelvic_tilt=0.0, facing="right",
               trochanter_xy=(560.0, 170.0), size=(1280, 720), thigh=190.0, shank=180.0, foot=110.0):
    """Dot positions (pixels) whose angles, in protocol.json's convention, are exactly the ones given.

    hip_flex: + flexion, - extension (0 = femur along the pelvis midline, i.e. the line from the iliac crest dot to the trochanter)
    knee_flex: 0 = straight
    ankle_df: + dorsiflexion, 0 = fibula 90 deg to the 5th metatarsal
    pelvic_tilt: the whole pelvis tipped forward by this many degrees (the iliac crest dot swings round the trochanter;
        no angle follows it, the hip angle is measured against the pelvis and does not change)
    facing: 'right' or 'left' way the participant faces in the image
    """
    T = np.array(trochanter_xy, float)
    u = np.array([math.cos(math.radians(pelvic_tilt)), math.sin(math.radians(pelvic_tilt))])  # the pelvis's own forward direction
    down = np.array([-u[1], u[0]])                        # pelvis midline, towards the feet
    crest = T - 95.0 * down                               # iliac crest dot (halfway between ASIS and PSIS), above the trochanter
    femur = _rot(down, -hip_flex)                         # trochanter -> epicondyle
    epi = T + thigh * femur
    fib = _rot(femur, knee_flex)                          # direction fibular head -> malleolus
    side = np.array([-fib[1], fib[0]]) * 14.0             # arm parallel to the femur line but not on it
    fibular_head = epi + 45 * fib + side
    malleolus = epi + shank * fib + side
    up_shank = -fib                                       # malleolus -> fibular head
    foot_dir = _rot(up_shank, 90.0 - ankle_df)            # 5th metatarsal base -> head
    mt5_base = malleolus + 18 * foot_dir + 34 * fib
    mt5_head = mt5_base + foot * foot_dir
    pts = {"iliac_crest": crest, "trochanter": T, "epicondyle": epi, "fibular_head": fibular_head,
           "malleolus": malleolus, "mt5_base": mt5_base, "mt5_head": mt5_head}
    if facing == "left":
        pts = {k: np.array([size[0] - v[0], v[1]]) for k, v in pts.items()}
    return pts


def frontal_points(abduction_deg=0.0, tested_leg="right", roll_deg=0.0, pelvic_width_mm=250.0, thigh_mm=400.0, px_per_mm=1.0,
                   centre=(640.0, 150.0)):
    """Dot positions (pixels) of a person facing the camera (hip abduction / adduction tests, 0.15.0).

    Returns the tested ASIS ('asis'), the other ASIS ('asis_opposite') and the centre of the patella ('patella'). The thigh is turned about
    the hip joint centre (the point geometry.hip_centre_frontal gives for these two ASIS) by `abduction_deg`, + = abduction (the knee moves
    away from the other leg), - = adduction, measured from the pelvis' own 'down' (square to the line between the ASIS). So the angle about
    the hip centre is exactly `abduction_deg`.
    tested_leg: 'right' = the tested ASIS is on the picture's left (the person faces the camera), 'left' = on the right.
    roll_deg: the whole picture rolled (a tilted camera or a tilted pelvis); angles do not change.
    px_per_mm: the picture's scale at the pelvis; centre: the midpoint of the two ASIS.
    """
    from .geometry import HARRINGTON_DOWN, HARRINGTON_SIDEWAYS
    out = -1.0 if tested_leg == "right" else 1.0                    # picture direction of "towards the tested side"
    u = _rot((out, 0.0), roll_deg)
    v = _rot((0.0, 1.0), roll_deg)
    m = np.array(centre, float)
    half = 0.5 * pelvic_width_mm * px_per_mm
    tested, other = m + half * u, m - half * u
    side = (HARRINGTON_SIDEWAYS[0] * pelvic_width_mm + HARRINGTON_SIDEWAYS[1]) * px_per_mm
    down = (HARRINGTON_DOWN[0] * pelvic_width_mm + HARRINGTON_DOWN[1]) * px_per_mm
    hjc = m + side * u + down * v
    a = math.radians(abduction_deg)
    patella = hjc + thigh_mm * px_per_mm * (math.sin(a) * u + math.cos(a) * v)
    return {"asis": tested, "asis_opposite": other, "patella": patella}


def render(points, size=(1280, 720), kind="white", dot_radius=7.0, noise=3.0, seed=0, distractors=True,
           dot_gain=1.0, hide=(), patch=None):
    """`patch`: half the side (pixels) of a square 'tape patch' drawn behind every dot, in the opposite shade to the dot
    (white tape behind black dots, dark behind white ones). None = bare dots (the default; the pictures are unchanged)."""
    w, h = size
    rng = np.random.default_rng(seed)
    bg = 90 if kind == "white" else 170
    img = np.full((h, w, 3), bg, np.uint8)
    limb_col = (95, 110, 140) if kind == "white" else (120, 140, 175)
    names = list(points)
    for a, b in [("iliac_crest", "trochanter"), ("trochanter", "epicondyle"), ("fibular_head", "malleolus"),
                 ("epicondyle", "fibular_head"), ("mt5_base", "mt5_head"), ("malleolus", "mt5_base"),
                 ("asis_opposite", "asis"), ("asis", "patella")]:             # (the last two: a frontal picture, see frontal_points)
        if a in points and b in points:
            cv2.line(img, tuple(np.int32(points[a])), tuple(np.int32(points[b])), limb_col, 26, cv2.LINE_AA)
    if distractors:  # something bright and something dark that is NOT a dot
        cv2.rectangle(img, (w - 260, h - 140), (w - 90, h - 80), (240, 240, 240), -1)
        cv2.rectangle(img, (60, h - 140), (230, h - 80), (20, 20, 20), -1)
    full = 245 if kind == "white" else 15
    level = int(round(bg + dot_gain * (full - bg)))     # dot_gain < 1 = faded / low-contrast dots
    col = (level, level, level)
    if patch:
        tape = 235 if kind == "black" else 25
        for n in names:
            if n not in hide:
                x, y = points[n]
                cv2.rectangle(img, (int(round(x - patch)), int(round(y - patch))), (int(round(x + patch)), int(round(y + patch))),
                              (tape, tape, tape), -1)
    shift = 4
    for n in names:
        if n in hide:                                      # dot covered (e.g. by a hand), limb still drawn
            continue
        x, y = points[n]
        cv2.circle(img, (int(round(x * (1 << shift))), int(round(y * (1 << shift)))),
                   int(round(dot_radius * (1 << shift))), col, -1, cv2.LINE_AA, shift)
    if noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape[:2] + (1,)).astype(np.float32), 0, 255).astype(np.uint8)
    return img


def write_demo(path, seconds=8, fps=30, kind="white"):
    """Knee flexion sweeping 20 -> 65 -> 20 deg with the other joints held still."""
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    n = int(seconds * fps)
    for i in range(n):
        knee = 42.5 - 22.5 * math.cos(2 * math.pi * i / n)
        w.write(render(leg_points(hip_flex=80, knee_flex=knee, ankle_df=5, pelvic_tilt=4), kind=kind, seed=i))
    w.release()


if __name__ == "__main__":
    write_demo(sys.argv[1] if len(sys.argv) > 1 else "demo.mp4")
    print("written")
