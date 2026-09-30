"""Synthetic leg videos with dots at exactly known angles.

Used by the tests, and handy for trying the app without a camera:
    python -m limbtrack.synthetic demo.mp4
then use "Open video file" in the app.
"""
import math
import sys

import cv2
import numpy as np


def _dir(deg_from_down):
    """Unit vector rotated `deg_from_down` clockwise (on screen) from straight down."""
    a = math.radians(deg_from_down)
    return np.array([math.sin(a), math.cos(a)])


def _rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def leg_points(hip_flex=0.0, knee_flex=0.0, ankle_df=0.0, trunk_lean=0.0,
               hip_xy=(500.0, 330.0), trunk=200.0, thigh=190.0, shank=180.0, foot=90.0):
    """Dot positions (pixels) whose angles, in protocol.json's convention, are exactly the ones given.

    hip_flex, knee_flex: 0 = straight.  ankle_df: + dorsiflexion, 0 = shank 90 deg to foot.
    trunk_lean: degrees from vertical.
    """
    hip = np.array(hip_xy, float)
    up = _dir(180.0 - trunk_lean)                       # hip -> shoulder
    shoulder = hip + trunk * up
    thigh_dir = _rot(-up, hip_flex)                     # hip -> knee
    knee = hip + thigh * thigh_dir
    shank_dir = _rot(thigh_dir, -knee_flex)             # knee -> ankle
    ankle = knee + shank * shank_dir
    to_knee = -shank_dir                                # ankle -> knee
    toe_dir = _rot(to_knee, -(90.0 - ankle_df))         # ankle -> toe, interior angle = 90 - df
    toe = ankle + foot * toe_dir
    return {"shoulder": shoulder, "hip": hip, "knee": knee, "ankle": ankle, "toe": toe}


def render(points, size=(1280, 720), kind="white", dot_radius=7.0, noise=3.0, seed=0, distractors=True):
    w, h = size
    rng = np.random.default_rng(seed)
    bg = 90 if kind == "white" else 170
    img = np.full((h, w, 3), bg, np.uint8)
    limb_col = (95, 110, 140) if kind == "white" else (120, 140, 175)
    names = list(points)
    for a, b in [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"), ("ankle", "toe")]:
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
        w.write(render(leg_points(hip_flex=80, knee_flex=knee, ankle_df=5, trunk_lean=3), kind=kind, seed=i))
    w.release()


if __name__ == "__main__":
    write_demo(sys.argv[1] if len(sys.argv) > 1 else "demo.mp4")
    print("written")
