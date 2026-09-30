# Limb angle tracker (MVIC testing)

Tracks white or black dots on a live camera feed, shows the joint angles on screen, warns the operator when someone drifts from the set position, and saves the video + angles for each trial.

**Status: version 0.1 (webcam, sagittal view).** It passes its automatic tests on synthetic video. It has **not yet been validated on real footage or against a goniometer** - do that before collecting real data (see "Validation" below).

## Setup (once, on a new computer)
Needs Windows and Python 3.12. In this folder:
```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Start it
Double-click **Start.bat**. (First try it without a camera: Live tab -> *Open video file (practice)* -> `demo/demo_leg_white_dots.mp4`.)

## Using it (Live tab, top to bottom)
1. **Video source**: pick the camera and press *Start camera*. If it isn't listed, press *Scan cameras*.
   For video files only: **Pause / Play** (**P**) and single-frame steps back and forward (**Left** / **Right** arrows). You can mark dots, lock, and change targets while paused and the picture updates. At the end of a file it waits on the last frame so you can step back. Stepping is switched off while recording.
2. **Test and target**: choose the test, type the target angle for the main joint (or pick a suggested one), and set tolerances. Pick the dot colour, or leave on Auto.
3. **Mark dots**: click each dot on the video when asked. The eight landmarks follow your goniometer table: ASIS, PSIS, greater trochanter, lateral femoral epicondyle, fibular head, lateral malleolus, 5th metatarsal base and head. Click the centre. To fix one dot later, click it again. Then position the participant. The main joint turns **green** inside target +/- tolerance, **red** outside.
4. **Lock position** (button or **Space**) once they are in position. The other joints' current angles become their references, and from then on they go red if they drift more than the "other joints" tolerance. A dot that can't be seen is shown grey with a cross, and its angles show "lost" (never guessed).
5. **Record** (button or **R**): enter a participant ID first. Each trial is saved in `recordings/<participant>/`:
   - `..._raw.mp4` - untouched video (opens in Kinovea)
   - `..._overlay.mp4` - video with the angles drawn on
   - `..._data.csv` - one row per frame: time, every dot position, every angle, deviation from reference, in/out of tolerance
   - `..._meta.json` - participant, test, target, tolerances, camera, notes

## Review tab
*Open trial...* -> pick a `..._meta.json`. Scrub or play the video with its overlay, see angle-vs-time with the green tolerance band, drag the blue window on the plot to the hold phase of the MVIC, and read mean / SD / min / max / max deviation / % time in tolerance. *Export summary CSV* saves those numbers.

## Changing tests, angles and conventions
Edit `protocol.json` (plain text). Each test lists the dots, the angles (three dots = joint angle at the middle one; two dots = segment vs vertical), and the convention: shown = `sign` x raw + `offset`. Landmarks and arms follow Tabel 3 (goniometer alignment): hip = pelvis midline (perpendicular to ASIS-PSIS) vs femur, knee = femur vs fibula, ankle = fibula vs 5th metatarsal, plus pelvic tilt (ASIS-PSIS vs horizontal). Sign conventions: hip flexion +, extension -; ankle dorsiflexion +; anterior pelvic tilt +. Target angles come from `ChenData/StrengthCurve/plot_proposal.m`. Choose which way the participant faces ("Participant faces") so signed angles come out right. **Check these against your final protocol.**

## Where the numbers can go wrong (read this)
- The angle is the **2D angle in the image**. It is only the true joint angle if the limb moves in a plane facing the camera. Put the camera level, at joint height, square to the sagittal plane, and don't let the participant rotate.
- Dots on skin are not the joint centres. Place them the same way every time (same person, palpated landmarks). The hip is the hardest.
- Lens distortion is not corrected yet. Keep the participant near the centre of the frame.
- Dot tracking assumes the dots are clearly brighter (white) or darker (black) than what's around them. Good even lighting and matte dots help. Glare on shiny dots or clothing seams near a dot can confuse it.

## Validation (do once, and again if the camera setup changes)
1. **Goniometer check**: set the limb to 5+ angles per joint (e.g. 30/60/90/120/150 deg), hold still ~5 s each while recording, then in Review select each hold and compare the mean with the goniometer. Aim for agreement within about 2-3 deg.
2. **Kinovea check**: open a `..._raw.mp4` in Kinovea, measure the angle with its angle tool at a few moments, and compare with the CSV at the same times.
3. **Noise floor**: record a still participant (or a mannequin) for 60 s; the SD in Review is the tracking noise. It should be well under your tolerance.
4. **Stress it**: cover a dot with a hand, change the lighting, and confirm the dot goes grey ("lost") and comes back correctly.

## Not built yet (planned, in this order)
Lens calibration -> phones (Android/iPhone, via Iriun/Camo/DroidCam or a network stream) -> load cell/ADC recorded on the same clock and overlaid -> two cameras. The CSV already has a `clock_s` column (camera clock) so force data can be merged later.

## Developer notes
- Python 3.12 in `.venv`. Run tests: `.venv\Scripts\python.exe -m pytest -q`
- `limbtrack/`: `geometry` (angle maths), `tracker` (dots), `engine` (pipeline, no GUI), `recorder`, `overlay`, `camera`, `gui`, `review`, `synthetic` (test video generator).
