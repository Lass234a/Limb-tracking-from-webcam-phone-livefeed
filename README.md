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
   A line under the buttons shows what the camera really delivers (resolution and measured frame rate) and turns orange with a warning if that differs from the 1280x720, 30 fps the app asks for. **Lock exposure and white balance** (webcams only) stops the camera from changing brightness or colour by itself; do it once the lighting is final. The app says in words whether the camera accepted it (many webcams ignore the request), and the result is saved with each trial. **Mirror the image** flips the picture left-right before tracking and recording (saved video and pixel positions are flipped too, `mirrored_image` in `meta.json`); it clears the marked dots, cannot be changed while recording, and after switching it check *Participant faces* against what you now see.
   If something goes wrong in the video thread, the video stops with a message in the instruction line, any recording is saved up to that point, and the details go to `recordings/last_error.log`.
   For video files only: **Pause / Play** (**P**) and single-frame steps back and forward (**Left** / **Right** arrows). You can mark dots, lock, and change targets while paused and the picture updates. At the end of a file it waits on the last frame so you can step back. Stepping is switched off while recording.
2. **Test and target**: choose the test, type the target angle for the main joint (or pick a suggested one), and set tolerances. Pick the dot colour (or leave on Auto) and which way the participant faces in the image. The smoothing tick box averages the *displayed* angles over 0.2 s; it never touches what is saved.
   **Landmarks to use (pilot)**: tick or untick which of the 8 landmarks you will put dots on, or use a preset such as "only Knee flexion". Angles that need an unticked landmark disappear (the panel lists which are available and what the others need); the rest keep working. If the main joint's landmarks are unticked, no target is judged and every remaining angle is checked against the locked position instead. Your choice stays while you switch tests, is saved with each trial (`dots_disabled` in `meta.json`, and the CSV only has the chosen dots and angles), and is forgotten when the program is closed. It cannot be changed during a recording; re-enabling a landmark means marking it again.
   **Three-point angle**: tick *Use a three-point angle* and choose any three landmarks; the app shows the unsigned angle at the middle one (a straight limb reads 180 deg) and switches those three landmarks on. By default it becomes the main joint, so the target above applies to it (the test's own angles are then checked against the locked position instead); untick *Use it as the main joint* to show it alongside the others. This works with as few as three dots, for example trochanter, epicondyle and malleolus when no fibular-head or foot dots are available.
3. **Mark dots**: click each dot on the video when asked. The eight landmarks follow your goniometer table: ASIS, PSIS, greater trochanter, lateral femoral epicondyle, fibular head, lateral malleolus, 5th metatarsal base and head. Click the centre. To fix one dot later, click it again. Then position the participant. The main joint turns **green** inside target +/- tolerance, **red** outside.
4. **Lock position** (button or **Space**) once they are in position. The other joints' current angles become their references, and from then on they go red if they drift more than the "other joints" tolerance. (The *ghost* of the locked pose that earlier versions drew is switched off in this version; see `PARKED.md`.)
   **Trails**: while you record, each dot draws a thin coloured path of where it has been (a gap where it was lost). The trails start when you press Start recording, stay on screen after you stop, and start afresh at the next recording. They appear in the live view, the saved overlay video and the Review tab (tick box *Show trails*, which only changes what is drawn; no number is affected). A dot adds a trail point only after moving at least 0.5 px.
5. **Record** (button or **R**): enter a participant ID first. During the trial press **S** (MVIC start), **E** (MVIC end) or **M** (numbered marker). Each trial is saved in `recordings/<participant>/`:
   - `..._raw.mp4` - untouched video (opens in Kinovea)
   - `..._overlay.mp4` - video with the angles, trails and marker banners drawn on
   - `..._data.csv` - one row per frame (columns below)
   - `..._meta.json` - participant, test, target, tolerances, facing, mirror, smoothing, camera and its settings, every tracker number (`tracker_settings`), notes, markers

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
`frame, t_s` (seconds from trial start), `clock_s` (camera clock, for merging force data later), `locked`, then per dot `<dot>_x, <dot>_y, <dot>_lost, <dot>_q` (quality 0-1; x and y are blank while the dot is lost), then per angle `<angle>_deg` (the unsmoothed angle), `<angle>_ref`, `<angle>_dev`, `<angle>_ok` (1 in tolerance, 0 out, blank = no reference / lost; all computed from the unsmoothed angle), then `all_ok` and `event` (marker label on the frame where it was pressed).

## Review tab
*Open trial...* -> pick a `..._meta.json`. Scrub or play the video with its overlay (including the trails), see angle-vs-time (unsmoothed, as saved) with the green tolerance band and the markers as dashed vertical lines. Pick a marker in the list to jump to it. **Window = MVIC start to end** sets the analysis window to your markers; otherwise drag the blue window on the plot. The numbers below use the window and the unsmoothed saved angles (so the SD includes tracking noise): mean, SD, min, max, **start>end** (how far the angle moved between the start and the end of the window, the slack take-up number), max deviation, % time in tolerance, lost frames. *Export summary CSV* saves them.

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
- The display smoothing is a trailing average over 0.2 s, so a steadily changing angle is shown about 0.1 s late (about 5 deg at 50 deg/s, 0.1 deg at 1 deg/s). Untick it to see the raw angle. The saved CSV is never smoothed, so the green/red on screen can differ slightly from the `_ok` column.
- Lens distortion is not corrected yet. Keep the participant near the centre of the frame.
- A dot that is hidden (hand, strap) is frozen where it was last seen and its angles go blank; nothing is guessed. It picks up again only if a round dot of about the right size and brightness reappears within about two dot-widths of that spot; otherwise click it again. The video it was tuned on had dots smaller than planned, so test this with your real dots (planned size: at least 2 cm).
- Dot tracking assumes the dots are clearly brighter (white) or darker (black) than what's around them. Good even lighting and matte dots help. Glare on shiny dots or clothing seams near a dot can confuse it. Dots closer together than about 3 dot-widths can be confused.

## Validation (do once, and again if the camera setup changes)
1. **Goniometer check**: set the limb to 5+ angles per joint (e.g. 30/60/90/120/150 deg), hold still ~5 s each while recording (use S / E markers), then in Review select each hold and compare the mean with the goniometer. Aim for agreement within about 2-3 deg.
2. **Kinovea check**: open a `..._raw.mp4` in Kinovea, measure the angle with its angle tool at a few moments, and compare with the CSV at the same times.
3. **Noise floor**: record a still participant (or a mannequin) for 60 s; the SD in Review is the tracking noise. It should be well under your tolerance.
4. **Stress it**: cover a dot with a hand, change the lighting, and confirm the dot goes orange/grey and comes back correctly.

## Not built yet (planned, in this order)
Ruler/scale in the app and a minimum-dot-size check (waiting for your go-ahead) -> lens calibration -> phones (Android/iPhone, via Iriun/Camo/DroidCam or a network stream) -> load cell/ADC recorded on the same clock and overlaid -> front view for hip abduction/adduction -> two cameras. The CSV already has a `clock_s` column (camera clock) so force data can be merged later.

## Developer notes
- Python 3.12 in `.venv`. Run tests: `.venv\Scripts\python.exe -m pytest -q` (about 2 to 4 minutes; 126 pass, 1 is skipped because the ghost pose is parked).
- `FEATURES.md` lists every feature, whether it can change a measured number, and what is unreliable; `PARKED.md` lists what was deliberately set aside.
- Tracking numbers are named constants at the top of `limbtrack/tracker.py` and are saved in every `meta.json`. `tests/golden_tracking.npz` pins the tracking output: if a deliberate change makes `tests/test_tracker_settings.py` fail, regenerate it with `python tests/test_tracker_settings.py` and say why in the commit message.
- `limbtrack/`: `geometry` (angle maths), `angles` (measure an angle definition), `protocol` (protocol.json), `tracker` (dots), `filters` (display smoothing), `engine` (pipeline, no GUI), `recorder`, `overlay`, `camera`, `gui`, `review`, `synthetic` (test video generator with exact known angles).
