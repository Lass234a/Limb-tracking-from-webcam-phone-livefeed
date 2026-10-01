# Feature inventory

Version 0.2.0. Written 2026-10-01. Line numbers refer to commit `4b186fd`; function names stay valid even if lines move.

**Purpose of the app:** show a live view of joint positions and angles from marker dots on bone landmarks, so each participant (maximal isometric contractions, strapped into a harness with an S-type load cell) can be placed in the correct position quickly. A webcam is the current proof of concept; two cameras (frontal and sagittal) are planned.

**Status:** everything below is tested on synthetic video only (74 automatic tests). Nothing has been validated on real footage or against a goniometer.

**What the app does NOT do to the data:** no interpolation of missing dots or angles, no drift correction, no outlier rejection (other than the "lost dot" rule), no frame-rate resampling, no lens-distortion correction, no camera-tilt correction.

**Legend**
- **Who:** *You* = you asked for it. *Approved* = I proposed it and you approved. *Added* = I added it without discussing it.
- **Need:** *Essential* = needed to place a participant live. *Helpful* = optional but useful live. *Extra* = data logging or review; requested for the study, but not needed for live positioning.

---

## A. Features that alter or define the measured numbers

These are the ones to justify in a methods section.

### A1. Marker snapping at marking
- **What:** clicking a dot moves the marker to the nearest real dot instead of using the click point.
- **Problem / who:** gives an exact starting centre from an imprecise click. *Added.*
- **Alters numbers:** yes, the starting position and the dot's reference size and contrast.
  - An 81x81 px window around the click is blurred (3x3) and top-hat filtered with a 41 px square kernel.
  - The strongest response within +/-8 px of the click must be at least 8 grey levels, otherwise marking fails.
  - Blobs above 50 % of that peak are taken; the one whose centre is nearest the click wins.
  - It is rejected if that centre is more than 15 px from the click.
  - The blob area sets the dot's reference area (diameter = 2*sqrt(area/pi), minimum 4 px) and its peak sets the reference contrast.
- **Need:** Essential.
- **Code:** `limbtrack/tracker.py:89` `DotTracker.init_at`

### A2. Frame-to-frame snapping (the tracker)
- **What:** every frame, the tracker searches near where the dot should be and snaps to the best-matching blob.
- **Problem / who:** follows the dots without re-clicking. *You* (live tracking); the specific method was *Added*.
- **Alters numbers:** yes, it defines the position of every dot in every frame.
  - Predicted position = last position + velocity; velocity = 0.5 x (new - old position).
  - Search radius `reach = min(3 x diam + 12 + 12 x frames_lost, 160)` px; the window extends `diam` beyond that.
  - Detection threshold = `max(0.4 x contrast, 8)` grey levels, where `contrast` is a running average (0.95 old + 0.05 new).
  - A blob is accepted only if its area is 0.3 to 3.5 times the running area (running area = 0.9 old + 0.1 new) and its longer side is at most `2.5 x shorter side + 2`.
  - Of the accepted blobs, the one nearest the predicted position is used.
- **Need:** Essential.
- **Code:** `limbtrack/tracker.py:122` `DotTracker.update`

### A3. Image preparation for detection
- **What:** the frame is converted to grey (OpenCV weighted average), blurred 3x3, then a top-hat (white dots) or black-hat (dark dots) removes the background.
- **Problem / who:** large bright or dark areas (shoes, clothes) are ignored. *Added.*
- **Alters numbers:** yes, slightly: the blur softens the image before the position is measured. The kernel is a square of side `odd(max(2.5 x diam, 9))` px; at marking `diam` is assumed to be 16 px.
- **Need:** Essential.
- **Code:** `limbtrack/tracker.py:56` `_enhance`; `limbtrack/tracker.py:169` `to_gray`

### A4. Sub-pixel dot centre
- **What:** the dot position is the intensity-weighted mean of the blob's pixels, weighted by their top-hat value.
- **Alters numbers:** yes; this is the measurement itself. Only pixels above the A2 threshold count, so uneven lighting across a dot can bias the centre slightly. On synthetic dots the error was under 0.4 px.
- **Need:** Essential.
- **Code:** `limbtrack/tracker.py:70` `_blobs`

### A5. Neighbour exclusion
- **What:** a tracker ignores blobs within `0.8 x diam` of where another tracked dot was last frame. A second rule declares the later tracker "lost" if two land within `0.5 x diam` of each other.
- **Problem / who:** when a dot is hidden, its tracker could jump onto a neighbouring dot (this happened in a test). *Added* (bug fix).
- **Alters numbers:** yes; it changes which blob a dot is allowed to take.
- **Need:** Helpful, important when dots sit close together.
- **Code:** `limbtrack/tracker.py:197` `TrackerSet.update`

### A6. Lost-dot rule
- **What:** if no acceptable blob is found, the dot is flagged lost and its last position is kept.
- **Alters numbers:**
  - Any angle using that dot becomes blank ("lost"); it is never guessed or interpolated.
  - The velocity is halved each lost frame and the search radius grows by 12 px per lost frame (cap 160 px).
  - The CSV still writes the held x/y with `lost = 1`. **Ignore x/y whenever `lost = 1`.**
- **Who:** *You* (no wrong angles); the exact rule was *Added*.
- **Need:** Essential.
- **Code:** `limbtrack/tracker.py:150`; `limbtrack/engine.py:197` `_compute_angles`

### A7. Angle smoothing (One-Euro filter)
- **What:** filters each angle over time: calm when the angle is still, responsive when it moves fast.
- **Alters numbers:** yes.
  - The tolerance colours and the CSV `_deg` column use the smoothed value. `_raw_deg` keeps the unfiltered value.
  - **Light (default):** min cutoff 1.5 Hz, beta 0.5. **Medium:** 0.8 Hz, beta 0.3. Derivative cutoff 1.0 Hz for both. **Off:** raw values.
  - The filter restarts after a lost dot, a gap longer than 0.5 s, or time going backwards.
  - It uses each frame's timestamp, so uneven webcam timing affects it slightly.
  - Measured on synthetic data (Light): about 2x less jitter, about 0.8 deg lag at a 50 deg/s sweep, about 0.06 deg lag on a 1 deg/s creep.
- **Who:** *Approved.*
- **Need:** Helpful, optional.
- **Code:** `limbtrack/filters.py:10` `LEVELS`; `limbtrack/filters.py:32` `OneEuroFilter.__call__`; `limbtrack/engine.py:189` `_smooth`

### A8. Angle definition and conventions
- **What:** an angle is measured between two "arms". An arm is a vector between two dots, "perpendicular to the line between two dots pointing toward the feet" (the pelvis midline), or a fixed direction (up, down, forward).
- **Alters numbers:** yes.
  - Shown value = `sign x angle + offset`. Hip: signed, sign -1 (flexion +, extension -). Knee: unsigned. Ankle: unsigned, sign -1, offset +90 (dorsiflexion +). Pelvic tilt: signed, sign +1 (anterior +).
  - If "participant faces left" is chosen, the x-direction of dot-based arms is mirrored so signs mean the same either way. A wrong setting flips the sign of hip and pelvic tilt without any warning; knee and ankle are unaffected.
- **Who:** *You* (landmarks and ASIS-PSIS perpendicular); the facing setting and sign/offset mechanism were *Added*.
- **Need:** Essential.
- **Code:** `limbtrack/geometry.py:69` `resolve_arm`; `limbtrack/geometry.py:85` `vector_angle`; `limbtrack/geometry.py:34` `to_display`; `limbtrack/angles.py:6` `measure`; `protocol.json`

### A9. Reference, deviation and tolerance
- **What:** decides whether an angle is "in position".
- **Alters numbers:** the verdicts and deviation columns.
  - Main joint: `deviation = value - target`, where the target is what the operator types.
  - Other joints: after Lock, each is compared with its value in the single frame at which Lock was pressed (smoothed value, not averaged).
  - In tolerance when `|deviation| <= tolerance` (inclusive, no hysteresis). Defaults: 3 deg main joint, 5 deg other joints; both adjustable.
- **Who:** *You* (track deviations from the set position); the design was *Added*.
- **Need:** Helpful.
- **Code:** `limbtrack/engine.py:169` `lock_position`; `limbtrack/engine.py:197` `_compute_angles`

### A10. Timestamps and video speed
- **What:** time base for the CSV and videos.
- **Alters numbers:** timing only.
  - Live camera: a frame's time is the PC clock when it was read (not the camera's own clock). `t_s` counts from the first recorded frame; `clock_s` is the raw clock.
  - Video file: time = frame number / the file's frame rate.
  - The raw and overlay videos are written at an estimated frame rate (from the last 30 frame intervals; 30 if fewer than 5), so they may not play at exactly real-time speed. **The CSV `t_s` is the authoritative time.**
- **Need:** Helpful; essential if force is added later.
- **Code:** `limbtrack/camera.py:37` `CameraSource.read`; `limbtrack/engine.py:271` `_fps`

### A11. Review-tab statistics
- **What:** the summary numbers for a chosen time window.
- **Alters numbers:** computed, not measured. All use the smoothed `_deg` values.
  - Mean, SD (sample SD, ddof = 1), min, max, max |deviation|, % of frames in tolerance (among frames that have a verdict), lost frames.
  - `start>end` = mean of the last 5 valid values minus mean of the first 5 valid values in the window.
  - Raw values are not used, so the SD understates tracking noise when smoothing is on.
- **Who:** *Approved.*
- **Need:** Extra.
- **Code:** `limbtrack/review.py:98` `Trial.summary`

### A12. Ghost pose (display only; does not alter the numbers)
- **What:** a dashed copy of the pose at the moment of Lock, drawn behind the live limb.
- **Detail:** by default the ghost is shifted by (live minus locked position of the main joint's fulcrum dot), so it shows angle change rather than body shift. Unticking that shows absolute position. The locked dot positions are saved in `meta.json`.
- **Who:** *Approved.*
- **Need:** Helpful.
- **Code:** `limbtrack/geometry.py:123` `anchored_ghost`; `limbtrack/overlay.py:48` `_draw_ghost`; `limbtrack/engine.py:264` `_ghost_anchor`

---

## B. Detection

| Feature | Who | Alters numbers? | Need | Code |
|---|---|---|---|---|
| White / black / Auto dot colour. Auto compares a 5x5 patch at the click with the median of the 81x81 window | Added | Indirectly (selects top-hat or black-hat) | Essential | `tracker.py:96` |
| 8 sagittal landmarks from your table (ASIS, PSIS, greater trochanter, lateral femoral epicondyle, fibular head, lateral malleolus, 5th metatarsal base and head) | You | No | Essential | `protocol.json` |
| Choose which landmarks to use, with presets such as "only Knee flexion" | You | Changes which angles exist; nothing else | Helpful (pilot) | `engine.py:107` `set_dot_enabled`; `gui.py:385` |
| Re-mark one dot by clicking it (nearest dot within 60 px) | Added | No | Helpful | `engine.py:159` `nearest_dot`; `gui.py:587` |
| Dot quality: current contrast / contrast at marking; "weak" below 0.6 | Approved | No (diagnostic) | Helpful | `tracker.py:166`; `engine.py:277` |

## C. Tracking

The numeric behaviour is in A2 to A6. Other items:

| Feature | Who | Need | Code |
|---|---|---|---|
| Re-finds a dot after it reappears (growing search window) | Added | Essential | `tracker.py:131` |
| Switching a landmark off then on needs it marked again | Added | Helpful | `tracker.py:183` |

## D. Angle calculation

The method is in A7 to A9. Other items:

| Feature | Who | Alters numbers? | Need | Code |
|---|---|---|---|---|
| 7 tests with main joint and suggested target angles (from your `plot_proposal.m`): knee extension/flexion, hip extension/flexion, ankle plantar/dorsiflexion, belt squat | You | No | Essential | `protocol.json`; `protocol.py` `load_protocol` |
| Pelvic tilt tracked as a neighbour in every test (ASIS-PSIS vs horizontal) | Added | Assumes a level camera | Helpful | `protocol.json` |
| Angles drop out automatically when a needed landmark is switched off | You | No | Helpful | `engine.py:92` `active_angles` |
| Editable `protocol.json` (dots, arms, signs, offsets, targets, tolerances) | Added | Defines the angles | Essential | `protocol.py` |

## E. Display

| Feature | Who | Need | Code |
|---|---|---|---|
| Live video with limb lines, arcs and angle numbers; green = in tolerance, red = out, white = no reference yet, grey = lost | You | Essential | `overlay.py:62` `draw` |
| Ghost pose | Approved | Helpful | see A12 |
| Live trace: last 15 s, main angle with tolerance band, or drift of all joints | Approved | Helpful | `gui.py:510` `update_trace`; `engine.py:236` `history` |
| Header messages (locked, dot lost, weak dot, marker banner) | Added | Helpful | `engine.py:277` `process` |
| Pause and frame-by-frame stepping (video files only) | You | Extra for live use (practice) | `gui.py:111` `VideoWorker.run`; `gui.py:554` |
| Keyboard keys: Space lock, R record, P pause, Left/Right step, S/E/M markers | Added | Helpful | `gui.py:163` |
| Scrollable control panel for small screens | Added | Helpful | `gui.py:163` |

## F. Calibration

**Nothing is implemented:** no lens correction, no pixel scale, no camera-tilt correction.

Things that look like calibration but are not: the dot size and contrast learned when you click a dot (A1), the dot-colour auto-detect (B), and the "participant faces" setting (A8).

## G. Everything else

| Feature | Who | Need | Code |
|---|---|---|---|
| Webcam input. Requests MJPG, 1280x720, 30 fps, buffer size 1. The camera may not grant these and the app does not check | You | Essential | `camera.py:11` |
| Video-file input, practice video, synthetic test-video generator | Added (file input approved) | Extra | `camera.py`; `synthetic.py` |
| Recording: raw video, overlay video, per-frame CSV, trial details JSON | You | Extra for live use | `recorder.py:53` `write`; `engine.py:322` `start_trial` |
| Event markers (MVIC start, MVIC end, numbered marker) | Approved | Extra | `engine.py:217` `add_event` |
| Review tab: scrub, plot, markers, summary export | You | Extra | `review.py` |
| `Start.bat`, README, 74 automatic tests, git history | Added | Extra | project root; `tests/` |

### CSV columns
`frame, t_s, clock_s, locked`, then per dot `<dot>_x, <dot>_y, <dot>_lost, <dot>_q`, then per angle `<angle>_deg` (smoothed), `<angle>_raw_deg`, `<angle>_ref`, `<angle>_dev`, `<angle>_ok`, then `all_ok, event`. Only the chosen landmarks and the angles that can be computed from them are written.

---

## H. Unfinished or unreliable

1. **Never run on real footage.** Thresholds were not tuned for real skin, lighting, straps or glare.
2. **Silent freeze risk.** The video thread has no error handling around its main loop; an unexpected error would stop the picture with no message (`gui.py:111`).
3. **The "raw" video is not lossless.** Frames are re-encoded with the `mp4v` codec. It is also not exactly real-time speed (A10).
4. **Review statistics hide noise.** They use smoothed values (A11).
5. **The main joint is judged against the typed target, not the locked pose.** If you lock at 63 deg with a target of 65 deg, the deviation shown is relative to 65. There is no "change since lock" number for the main joint in the CSV or live trace; the review's `start>end` partly covers it.
6. **The locked reference is a single smoothed frame**, so its noise carries into every neighbour deviation.
7. **A safety rule leaves a tracker on the wrong blob.** The "two trackers on one dot" rule marks the later tracker lost but does not move it back (`tracker.py:204`). It rarely triggers now and has no direct test.
8. **Held x/y are written while a dot is lost.** Easy to misread unless filtered on `lost`.
9. **Webcam specifics are unchecked:** auto-exposure and white balance are not locked, the granted resolution and frame rate are not verified, dropped frames are not counted, and timestamps jitter.
10. **Camera tilt biases pelvic tilt**, which is measured against image-horizontal. The hip angle is not affected by rotation.
11. **A wrong "participant faces" setting** flips hip and pelvic-tilt signs with no warning. There is no mirror toggle for mirrored webcam images.
12. **False precision.** Angles display to 0.1 deg, but real accuracy is unknown and likely worse. On synthetic data the typical error was about 0.5 deg or less.
13. **Stepping backwards** keeps the tracker state, so it can mis-track after a jump. Files only, and not recorded.
14. **Marker timing:** a marker attaches to the next processed frame (up to one frame late) plus the operator's reaction time.
15. **Green/red flicker** at the tolerance edge (no hysteresis).
16. **Methods reproducibility:** `meta.json` stores the smoothing level, angle definitions, target and tolerances, but not the tracker thresholds, filter parameters, or a code version beyond "0.2.0".
17. **Not built:** network-stream entry (the code path exists, no box in the window), two cameras, frontal view, lens calibration, force/ADC, non-Windows support.

---

## I. Roadmap (not built)

### No hardware needed
| Feature | Why |
|---|---|
| Real-footage validation helper | Goniometer vs app readings in one table, with agreement in degrees |
| Camera controls | Choose resolution and frame rate, lock exposure and focus, rotate or mirror |
| Camera alignment aid | Mark a plumb line once, show camera tilt, optionally correct it |
| "Change since lock" for the main joint | Closes unreliable item 5 |
| Review statistics on raw values (or both) | Closes unreliable item 4 |
| Error handling in the video thread, dropped-frame counter, storing tracker parameters in `meta.json` | Closes unreliable items 2, 9, 16 |
| Stream-address box, remembered settings, session checklist, batch summary export, compare trials | Workflow |
| Custom dots and angles | Pilot flexibility (you chose not to need this yet) |

### Needs a phone or hardware
| Feature | Needs |
|---|---|
| Lens calibration | Checkerboard print, once per camera; needed for phones and wide lenses |
| Phones as cameras (Android, iPhone or mixed) | Phone and a streaming app (Iriun, Camo, DroidCam) or a network stream |
| ADC and load cell recorded on the same clock; live force trace and overlay; sync check | ADC; TAS501C with SIC-A2 wiring |
| Frontal view (hip abduction/adduction) and two cameras at once | Second camera position or two phones |
| True 3D angles | Calibrated camera pair; a separate decision |

### Suggested order
1. Real-world check of what exists (webcam test, then goniometer comparison).
2. The no-hardware fixes above, mainly the unreliable-list items.
3. Camera controls and alignment aid.
4. Phones and lens calibration.
5. Force, once the ADC arrives.
6. Frontal view and two cameras.
