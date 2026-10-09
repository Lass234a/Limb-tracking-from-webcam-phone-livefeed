"""A dot that touches a shadow, a finger or a strap merges with it into one big blob. The tracker must keep the dot's own
centre (found on the board recordings of 2026-10-09: the centre was dragged toward the finger and the dot was then lost)."""
import cv2
import numpy as np

from limbtrack import tracker as T
from limbtrack.tracker import TrackerSet, to_gray

DOT = (300, 200)
R = 11


def frame(touch_px=0, dot=DOT, finger_value=95):
    """Grey board, a black dot, and (touch_px > 0) a fainter dark finger-like ellipse touching the dot's right edge."""
    img = np.full((400, 600, 3), 150, np.uint8)
    cv2.circle(img, dot, R, (20, 20, 20), -1, cv2.LINE_AA)
    if touch_px:
        cx = dot[0] + R + touch_px // 2 - 2
        cv2.ellipse(img, (cx, dot[1] + 2), (max(touch_px // 2, 6), 9), 0, 0, 360, (finger_value,) * 3, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


def run(frames, dot=DOT):
    ts = TrackerSet(["a"], "auto")
    assert ts.init_dot("a", to_gray(frame()), *dot)
    out = []
    for f in frames:
        out.append(ts.update(to_gray(f))["a"])
    return out


def test_the_centre_stays_on_the_dot_while_a_finger_touches_it():
    states = run([frame(0)] * 3 + [frame(k) for k in (10, 20, 30, 40, 50, 60)] + [frame(60)] * 5)
    for s in states:
        assert not s.lost
        assert np.hypot(s.x - DOT[0], s.y - DOT[1]) < 1.5, (s.x, s.y)


def test_a_lost_dot_found_again_while_the_finger_still_touches_it():
    hide = np.full((400, 600, 3), 150, np.uint8)               # the dot is completely hidden for a while
    states = run([frame(0)] * 2 + [hide] * 6 + [frame(40)] * 6)
    assert states[3].lost and states[7].lost
    assert not states[-1].lost and np.hypot(states[-1].x - DOT[0], states[-1].y - DOT[1]) < 1.5


def test_the_dots_learned_size_does_not_grow_with_what_is_attached_to_it():
    ts = TrackerSet(["a"], "auto")
    assert ts.init_dot("a", to_gray(frame()), *DOT)
    t = ts.trackers["a"]
    area0 = t.area
    for _ in range(40):
        ts.update(to_gray(frame(50)))
    assert t.area < 1.3 * area0


def test_a_small_fragment_is_not_taken_for_a_hidden_dot():
    """The dot disappears (covered); a small dark speck 20 px away must not become the dot."""
    ts = TrackerSet(["a"], "auto")
    assert ts.init_dot("a", to_gray(frame()), *DOT)
    ts.update(to_gray(frame()))
    covered = np.full((400, 600, 3), 150, np.uint8)
    cv2.circle(covered, (DOT[0] + 22, DOT[1] + 10), 5, (30, 30, 30), -1, cv2.LINE_AA)       # a fragment of about a fifth of the dot
    s = ts.update(to_gray(cv2.GaussianBlur(covered, (3, 3), 0)))["a"]
    assert s.lost


def test_blob_measuring_gives_the_same_numbers_as_measuring_one_by_one():
    """Speed-up of 2026-10-09: all blobs are measured at once. The result must equal the one-by-one reference."""
    from limbtrack.tracker import DotTracker

    def one_by_one(enh, thr, min_px=3):
        _, mask = cv2.threshold(enh, thr, 255, cv2.THRESH_BINARY)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_px:
                continue
            ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
            wts = enh[y:y + h, x:x + w][ys, xs].astype(np.float64)
            out.append((x + float((xs * wts).sum() / wts.sum()), y + float((ys * wts).sum() / wts.sum()), float(area), int(w), int(h)))
        return out

    rng = np.random.RandomState(3)
    tracker = DotTracker("a")
    for k in range(30):
        img = cv2.normalize(cv2.GaussianBlur((rng.rand(160, 200) * 255).astype(np.uint8), (0, 0), 2.0), None, 0, 255, cv2.NORM_MINMAX)
        ref, now = one_by_one(img, 140 + k), tracker._blobs(img, 140 + k)
        assert len(ref) == len(now)
        for a, b in zip(ref, now):
            assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 and a[2:] == b[2:]
