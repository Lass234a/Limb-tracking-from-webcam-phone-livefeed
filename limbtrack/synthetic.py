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

    hip_flex: + flexion, - extension (0 = femur along the pelvis midline)
    knee_flex: 0 = straight
    ankle_df: + dorsiflexion, 0 = fibula 90 deg to the 5th metatarsal
    pelvic_tilt: + anterior tilt (ASIS lower than PSIS)
    facing: 'right' or 'left' way the participant faces in the image
    """
    T = np.array(trochanter_xy, float)
    u = np.array([math.cos(math.radians(pelvic_tilt)), math.sin(math.radians(pelvic_tilt))])  # PSIS -> ASIS
    mid = T + np.array([0.0, -95.0])
    asis, psis = mid + 60 * u, mid - 60 * u
    down = np.array([-u[1], u[0]])                        # pelvis midline, towards the feet
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
    pts = {"asis": asis, "psis": psis, "trochanter": T, "epicondyle": epi, "fibular_head": fibular_head,
           "malleolus": malleolus, "mt5_base": mt5_base, "mt5_head": mt5_head}
    if facing == "left":
        pts = {k: np.array([size[0] - v[0], v[1]]) for k, v in pts.items()}
    return pts


def render(points, size=(1280, 720), kind="white", dot_radius=7.0, noise=3.0, seed=0, distractors=True):
    w, h = size
    rng = np.random.default_rng(seed)
    bg = 90 if kind == "white" else 170
    img = np.full((h, w, 3), bg, np.uint8)
    limb_col = (95, 110, 140) if kind == "white" else (120, 140, 175)
    names = list(points)
    for a, b in [("asis", "psis"), ("trochanter", "epicondyle"), ("fibular_head", "malleolus"),
                 ("epicondyle", "fibular_head"), ("mt5_base", "mt5_head"), ("malleolus", "mt5_base")]:
        cv2.line(img, tuple(np.int32(points[a])), tuple(np.int32(points[b])), limb_col, 26, cv2.LINE_AA)
    if distractors:  # something bright and something dark that is NOT a dot
        cv2.rectangle(img, (w - 260, h - 140), (w - 90, h - 80), (240, 240, 240), -1)
        cv2.rectangle(img, (60, h - 140), (230, h - 80), (20, 20, 20), -1)
    col = (245, 245, 245) if kind == "white" else (15, 15, 15)
    shift = 4
    for n in names:
        x, y = points[n]
        cv2.circle(img, (int(round(x * (1 << shift))), int(round(y * (1 << shift)))),
                   int(round(dot_radius * (1 << shift))), col, -1, cv2.LINE_AA, shift)
    if noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
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
