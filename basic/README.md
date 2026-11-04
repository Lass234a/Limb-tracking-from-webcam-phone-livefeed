# Basic limb tracker

The smallest version of the idea, in one file ([`basic_tracker.py`](basic_tracker.py), about 350 lines, no window toolkit, nothing to
install beyond what the project already has). It is separate from the full program in `../limbtrack` and does not touch it.

**What it does:** shows a camera or video, you click three dots once, it follows them, and it shows the angle at the middle dot.
It can write the positions and angle to a CSV file.

## Run it

| What | How |
|---|---|
| Camera 0, black dots | double-click `Run_basic.bat` |
| A video file | drag the video file onto `Run_basic.bat` |
| Another camera | `..\.venv\Scripts\python.exe basic_tracker.py 1` (try 0, 1, 2 ...) |
| White dots | add `--colour white` |
| Other dot names | add `--dots hip,knee,ankle` (the angle is always at the second one) |

## Use it

1. Click the three dots in the order the top of the picture asks for (default: trochanter, epicondyle, malleolus). Click near the
   middle of the dot. If no dot is found you are told.
2. The dots turn into green rings and follow the limb. The top line shows the angle at the middle dot: **180 = straight leg**,
   and the flexion (0 = straight) next to it.
3. **r** starts and stops a recording (a CSV in `basic/recordings/`, which git ignores). **m** marks the dots again. **space** pauses
   (you can mark on a paused picture). **q** quits.

A dot that cannot be found turns red and stays where it was last seen; no angle is shown while a dot is lost, and the CSV is blank
for it. Nothing is ever guessed or filled in.

## CSV columns

`t_s`, then `<dot>_x` and `<dot>_y` for each dot (pixels, blank while the dot is lost), then `angle_deg` (the interior angle) and
`flexion_deg` (= 180 - angle). Written one row per picture and flushed at once, so a crash loses nothing.

## How the tracking works (the same method as the full program)

1. After your click, a black-hat filter (top-hat for white dots) removes the background and keeps only small dark (bright) features.
   The dot's size and contrast are remembered.
2. In each new picture only a small window around where the dot should be is searched.
3. Patches above a threshold are found; those of about the right size and roundness count, and the nearest one to the expected
   position wins. Its brightness-weighted centre is the position.
4. If none fits, the dot is lost: it freezes, searches only a small fixed circle, and comes back only on a convincing dot.

The numbers at the top of the file are the same as in `../limbtrack/tracker.py`, so the two can be compared on the same video.
Tested on 120 synthetic pictures with known dots (black and white): position error 0.04 px on average, angle error under 0.1 deg.
On the real P09 video it follows the same positions as the full program (within 0.2 px) for the three default dots.

## What is not in it (all of this is in the full program)

- Targets, tolerances, the protocol, Lock, judging the other joints
- The **Auto** dot colour (here you choose black or white yourself)
- Extra rules against a dot touching a shadow or finger, a neighbour dot being taken, or a half-covered dot (flagged)
- Weak-dot and dot-size warnings, the ruler scale, strap and anchor dots, pull angle
- Force from the load cell, automatic MVIC markers, the force trace
- Saving video, privacy crop and masks, `meta.json` with the settings used, Review and its checks
- Choosing a phone camera by name (here a camera is just a number; try the next number if the wrong one opens)
- Mirror, "participant faces", three-point angle choices
