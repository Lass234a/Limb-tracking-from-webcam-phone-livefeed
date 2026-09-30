# Limb angle tracker (MVIC testing)

Tracks white or black dots on a live camera feed, shows the joint angles on screen, warns the operator when someone drifts from the set position, and saves the video + angles for each trial.

**Status: version 0.2 (webcam / video file, sagittal view).** It passes its automatic tests on synthetic video. It has **not yet been validated on real footage or against a goniometer** - do that before collecting real data (see "Validation" below).

## Setup (once, on a new computer)
Needs Windows and Python 3.12. In this folder:
```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Start it
Double-click **Start.bat**. (First try it without a camera: Live tab -> *Open video file (practice)* -> `demo/demo_leg_white_dots.mp4`.)

## Using it (Live tab)
1. **Video source**: pick the camera and press *Start camera*. If it isn't listed, press *Scan cameras*.
   For video files only: **Pause / Play** (**P**) and single-frame steps back and forward (**Left** / **Right** arrows). You can mark dots, lock, and change targets while paused and the picture updates. At the end of a file it waits on the last frame so you can step back. Stepping is switched off while recording.
2. **Test and target**: choose the test, type the target angle for the main joint (or pick a suggested one), and set tolerances. Pick the dot colour (or leave on Auto), which way the participant faces in the image, and the angle smoothing.
   **Landmarks to use (pilot)**: tick or untick which of the 8 landmarks you will put dots on, or use a preset such as "only Knee flexion". Angles that need an unticked landmark disappear (the panel lists which are available and what the others need); the rest keep working. If the main joint's landmarks are unticked, no target is judged and every remaining angle is checked against the locked position instead. Your choice stays while you switch tests, is saved with each trial (`dots_disabled` in `meta.json`, and the CSV only has the chosen dots and angles), and is forgotten when the program is closed. It cannot be changed during a recording; re-enabling a landmark means marking it again.
3. **Mark dots**: click each dot on the video when asked. The eight landmarks follow your goniometer table: ASIS, PSIS, greater trochanter, lateral femoral epicondyle, fibular head, lateral malleolus, 5th metatarsal base and head. Click the centre. To fix one dot later, click it again. Then position the participant. The main joint turns **green** inside target +/- tolerance, **red** outside.
4. **Lock position** (button or **Space**) once they are in position. The other joints' current angles become their references, and from then on they go red if they drift more than the "other joints" tolerance. A faint dashed **ghost** of the locked pose stays on screen so drift is visible as a gap between the live limb and the ghost. By default the ghost follows the main joint, so it shows angle drift only; untick that to see whole-body movement too.
5. **Record** (button or **R**): enter a participant ID first. During the trial press **S** (MVIC start), **E** (MVIC end) or **M** (numbered marker). Each trial is saved in `recordings/<participant>/`:
   - `..._raw.mp4` - untouched video (opens in Kinovea)
   - `..._overlay.mp4` - video with the angles, ghost and marker banners drawn on
   - `..._data.csv` - one row per frame (columns below)
   - `..._meta.json` - participant, test, target, tolerances, facing, smoothing, camera, notes, markers, locked pose

While the video runs, the **live trace** under it shows the last 15 s: the main angle with its tolerance band, or (switch at the top) the drift of all joints from their set position, which is the best view for spotting slow creep.

### What the screen colours mean
| Where | Meaning |
|---|---|
| Angle text / limb lines green | inside tolerance |
| Angle text / limb lines red | outside tolerance |
| Angle text white | no reference yet (not locked) |
| Dot ring yellow | dot strong |
| Dot ring orange + "weak dot (fading)" | dot contrast has dropped below 60 % of when it was marked: fix lighting/glare before it is lost |
| Dot grey with a cross, angle "lost" | dot not found; nothing is guessed |

### CSV columns
`frame, t_s` (seconds from trial start), `clock_s` (camera clock, for merging force data later), `locked`, then per dot `<dot>_x, <dot>_y, <dot>_lost, <dot>_q` (quality 0-1), then per angle `<angle>_deg` (smoothed, what is judged), `<angle>_raw_deg`, `<angle>_ref`, `<angle>_dev`, `<angle>_ok` (1 in tolerance, 0 out, blank = no reference / lost), then `all_ok` and `event` (marker label on the frame where it was pressed).

## Review tab
*Open trial...* -> pick a `..._meta.json`. Scrub or play the video with its overlay (including the ghost), see angle-vs-time with the green tolerance band and the markers as dashed vertical lines. Pick a marker in the list to jump to it. **Window = MVIC start to end** sets the analysis window to your markers; otherwise drag the blue window on the plot. The numbers below use the window: mean, SD, min, max, **start>end** (how far the angle moved between the start and the end of the window, the slack take-up number), max deviation, % time in tolerance, lost frames. *Export summary CSV* saves them.

## Changing tests, angles and conventions
Edit `protocol.json` (plain text). An angle is the angle between two arms: an arm is a vector between two dots, "perpendicular to a line between two dots", or a fixed direction ("up", "down", "forward"). Shown value = `sign` x angle + `offset`; `signed` angles can be negative. Landmarks and arms follow Tabel 3 (goniometer alignment):
- hip = pelvis midline (perpendicular to ASIS-PSIS) vs femur (trochanter -> epicondyle); flexion +, extension -
- knee = femur vs fibula (fibular head -> malleolus); 0 = straight
- ankle = fibula vs 5th metatarsal (base -> head); dorsiflexion +, 90 deg between the arms = 0
- pelvic tilt = ASIS-PSIS line vs horizontal; anterior tilt +

Target angles come from `ChenData/StrengthCurve/plot_proposal.m`. Signed angles assume which way the participant faces, so set "Participant faces" correctly. **Check all of this against your final protocol.**

## Where the numbers can go wrong (read this)
- The angle is the **2D angle in the image**. It is only the true joint angle if the limb moves in a plane facing the camera. Put the camera level, at joint height, square to the sagittal plane, and don't let the participant rotate.
- Dots on skin are not the joint centres. Place them the same way every time (same person, palpated landmarks). The hip is the hardest.
- Dots on straps can slide over the skin without the joint moving (and the reverse). Validate strap-mounted dots separately; prefer dots on skin or tape in the gaps where you can.
- Smoothing delays fast movement slightly (under 1 deg at about 50 deg/s on Light), and none for slow drift. Raw values are always saved next to the smoothed ones.
- Lens distortion is not corrected yet. Keep the participant near the centre of the frame.
- Dot tracking assumes the dots are clearly brighter (white) or darker (black) than what's around them. Good even lighting and matte dots help. Glare on shiny dots or clothing seams near a dot can confuse it. Dots closer together than about 3 dot-widths can be confused.

## Validation (do once, and again if the camera setup changes)
1. **Goniometer check**: set the limb to 5+ angles per joint (e.g. 30/60/90/120/150 deg), hold still ~5 s each while recording (use S / E markers), then in Review select each hold and compare the mean with the goniometer. Aim for agreement within about 2-3 deg.
2. **Kinovea check**: open a `..._raw.mp4` in Kinovea, measure the angle with its angle tool at a few moments, and compare with the CSV at the same times.
3. **Noise floor**: record a still participant (or a mannequin) for 60 s; the SD in Review is the tracking noise. It should be well under your tolerance.
4. **Stress it**: cover a dot with a hand, change the lighting, and confirm the dot goes orange/grey and comes back correctly.

## Not built yet (planned, in this order)
Lens calibration -> phones (Android/iPhone, via Iriun/Camo/DroidCam or a network stream) -> load cell/ADC recorded on the same clock and overlaid -> front view for hip abduction/adduction -> two cameras. The CSV already has a `clock_s` column (camera clock) so force data can be merged later.

## Developer notes
- Python 3.12 in `.venv`. Run tests: `.venv\Scripts\python.exe -m pytest -q` (about 2 minutes).
- `limbtrack/`: `geometry` (angle maths), `angles` (measure an angle definition), `protocol` (protocol.json), `tracker` (dots), `filters` (smoothing), `engine` (pipeline, no GUI), `recorder`, `overlay`, `camera`, `gui`, `review`, `synthetic` (test video generator with exact known angles).
