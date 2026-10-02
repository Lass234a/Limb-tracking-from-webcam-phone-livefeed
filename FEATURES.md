# Feature inventory

Version 0.2.0 (the version string in `meta.json` has not been bumped; see H12). Rewritten 2026-10-02 to match the code at commit `77c4d7d`. Function names are given instead of line numbers because lines move.

**Purpose of the app:** show a live view of joint positions and angles from marker dots on bone landmarks, so each participant (maximal isometric contractions, strapped into a harness with an S-type load cell) can be placed in the correct position quickly. A webcam is the current proof of concept; two cameras (frontal and sagittal) are planned and parked.

**Status:** everything below is tested on synthetic video, plus one real practice video (`Bundepus.mp4`, which is not in the repository) for the lost-dot rule. There are 126 automatic tests (1 skipped: the parked ghost). Nothing has been validated against a goniometer, and the dots in the practice video are smaller than the planned 2 cm.

**What the app does NOT do to the data:** no interpolation of missing dots or angles, no drift correction, no outlier rejection (other than the lost-dot rule), no frame-rate resampling, no lens-distortion correction, no camera-tilt correction, no pixel-to-centimetre scale.

**What changed since the first inventory (2026-10-01), following your decisions:** the growing search and the old lost-dot rule were replaced (A6); the One-Euro filter was replaced by a 0.2 s moving average that only changes the screen (A7); the ghost pose was commented out (A13); a three-point angle mode, trails, error handling, camera checks, exposure lock, a mirror toggle and stored tracker settings were added (sections B to G).

**Legend**
- **Who:** *You* = you asked for it or decided it. *Approved* = I proposed it and you approved. *Added* = I added it without discussing it.
- **Need:** *Essential* = needed to place a participant live. *Helpful* = optional but useful live. *Extra* = data logging or review; requested for the study, but not needed for live positioning.

---

## A. Features that alter or define the measured numbers

These are the ones to justify in a methods section. **Every number in A1 to A5 is also written into each trial's `meta.json` under `tracker_settings`**, and a test (`tests/test_tracker_settings.py`) fails if tracking output changes, so a recording can be traced to the exact settings that made it. Constants live at the top of `limbtrack/tracker.py`.

### A1. Marker snapping at marking
- **What:** clicking a dot moves the marker to the nearest real dot instead of using the click point.
- **Problem / who:** gives an exact starting centre from an imprecise click. *Added; you chose to keep it.*
- **Alters numbers:** yes, the starting position and the dot's reference size and contrast.
  - An 81x81 px window around the click is blurred (3x3) and top-hat filtered with a 41 px square kernel (dot diameter assumed 16 px).
  - The strongest response within 8 px of the click must be at least 8 grey levels, otherwise marking fails.
  - Blobs above 50 % of that peak are taken; the one whose centre is nearest the click wins.
  - It is rejected if that centre is more than 15 px from the click.
  - The blob area sets the dot's reference area (diameter = 2*sqrt(area/pi), minimum 4 px) and its peak sets the reference contrast.
- **Need:** Essential.
- **Code:** `tracker.py` `DotTracker.init_at`; settings `image_preparation.marking_*`.

### A2. Frame-to-frame snapping (the tracker)
- **What:** every frame, the tracker searches near where the dot should be and snaps to the best-matching blob.
- **Alters numbers:** yes, it defines the position of every dot in every frame.
  - Predicted position = last position + velocity; velocity = 0.5 x (new - old position).
  - Search reach = `min(3 x diam + 12, 160)` px; the window extends `diam` beyond that. The reach no longer grows while a dot is lost (see A6).
  - Detection threshold = `max(0.4 x contrast, 8)` grey levels; `contrast` is a running average (0.95 old + 0.05 new).
  - A blob is accepted only if its area is 0.3 to 3.5 times the running area (running area = 0.9 old + 0.1 new) and its longer side is at most `2.5 x shorter side + 2`.
  - Of the accepted blobs, the one nearest the predicted position is used.
- **Who:** *You* (live tracking); the method was *Added*.
- **Need:** Essential.
- **Code:** `tracker.py` `DotTracker.update`; settings `tracking.*`.

### A3. Image preparation for detection
- **What:** the frame is converted to grey (OpenCV weighted average), blurred 3x3, then a top-hat (white dots) or black-hat (dark dots) removes the background.
- **Problem / who:** large bright or dark areas (shoes, clothes) are ignored. *Added; you chose to keep it.*
- **Alters numbers:** yes, slightly: the blur softens the image before the position is measured. The kernel is a square of side `odd(max(2.5 x diam, 9))` px.
- **Need:** Essential.
- **Code:** `tracker.py` `_enhance`, `to_gray`; settings `image_preparation.*`.

### A4. Sub-pixel dot centre
- **What:** the dot position is the intensity-weighted mean of the blob's pixels, weighted by their top-hat value.
- **Alters numbers:** yes; this is the measurement itself. Only pixels above the A2 threshold count, so uneven lighting across a dot can bias the centre slightly. On synthetic dots the error was under 0.4 px.
- **Need:** Essential.
- **Code:** `tracker.py` `_blobs`.

### A5. Neighbour exclusion
- **What:** a tracker ignores blobs within `0.8 x diam` of another tracked dot (one that is currently found). A second rule declares the later tracker lost if two end up within `0.5 x diam` of each other; that tracker is then put back at its previous position (`undo_update`).
- **Problem / who:** when a dot is hidden, its tracker could jump onto a neighbouring dot (this happened in a test). *Added (bug fix); you chose to keep it.*
- **Alters numbers:** yes; it changes which blob a dot is allowed to take.
- **Need:** Helpful, important when dots sit close together.
- **Code:** `tracker.py` `TrackerSet.update`; settings `tracking.neighbour_exclusion_factor`, `duplicate_factor`.

### A6. Lost-dot rule (replaces the old rule and the growing search)
- **What:** if no acceptable blob is found, the dot is flagged lost and **frozen** at its last good position. There is no velocity prediction while it is lost. Only a **fixed circle** of 2.0 x the dot's diameter (measured when it was lost) is searched, and it never grows.
- **Resuming** needs a convincing dot inside that circle, not just anything dark or bright:
  - contrast at least 0.6 x the contrast at marking,
  - area between 0.5 and 2.0 x the learned area,
  - roughly round: longer side at most `1.6 x shorter side + 1`.
  - A dot that moved further than the circle while hidden stays lost until you click it again.
- **Alters numbers:**
  - Any angle using a lost dot is blank ("lost"); it is never guessed or interpolated.
  - **The CSV writes blank x and y while a dot is lost** (the frozen position is not written). `_lost = 1` marks it.
- **Evidence:** on the real practice video the old rule locked onto a heel shadow 60 px away for 70 frames. The new rule with only a brightness check locked onto a dark table-leg sliver; the size and shape checks fixed that. The circle size was tried at several values on that video. At the default 2.0 diameters, a dot that reappears further away stays lost (safe, but you must click it again). At 8 to 12 diameters (46 to 70 px) the dot resumed correctly; at 16 or more (93 px and up) the tracker locked onto other dark objects. The default was kept at 2.0 until a proper test video exists.
- **Who:** *You* (your decision on 2026-10-01).
- **Need:** Essential.
- **Code:** `tracker.py` `DotTracker.update`; `engine.py` `_compute_angles`; settings `lost_dot.*`; tests `tests/test_lost_rule.py`.

### A7. Display smoothing (replaces the One-Euro filter)
- **What:** one fixed trailing moving average over 0.2 s, with an on/off tick box (default on).
- **Alters numbers: on the screen only.** The displayed angle, the colours, the live trace and the overlay video use the smoothed value. **The CSV (`_deg`, `_ref`, `_dev`, `_ok`, `all_ok`) always uses the unsmoothed angle**, so what you see can differ slightly from the CSV verdict.
  - A steadily changing angle is shown about 0.1 s late (about 5 deg at 50 deg/s, 0.1 deg at 1 deg/s).
  - The average restarts after a gap where an angle could not be computed (lost dot) and if time stops increasing (stepping backwards).
  - The setting is written to `meta.json` (`smoothing`).
- **Who:** *You.*
- **Need:** Helpful, optional.
- **Code:** `filters.py` `MovingAverage`; `engine.py` `_compute_angles`, `set_smoothing`.

### A8. Angle definition and conventions
- **What:** an angle is measured between two "arms". An arm is a vector between two dots, "perpendicular to the line between two dots pointing toward the feet" (the pelvis midline), or a fixed direction (up, down, forward).
- **Alters numbers:** yes.
  - Shown value = `sign x angle + offset`. Hip: signed, sign -1 (flexion +, extension -). Knee: unsigned. Ankle: unsigned, sign -1, offset +90 (dorsiflexion +). Pelvic tilt: signed, sign +1 (anterior +).
  - If "participant faces left" is chosen, the x-direction of dot-based arms is mirrored so signs mean the same either way. A wrong setting flips the sign of hip and pelvic tilt without any warning; knee and ankle are unaffected. "Faces" refers to the picture on screen, so after switching the mirror (E3) it must be set again.
- **Who:** *You* (landmarks and ASIS-PSIS perpendicular; you chose to keep the definitions); the facing setting and sign/offset mechanism were *Added*.
- **Need:** Essential.
- **Code:** `geometry.py` `resolve_arm`, `vector_angle`, `to_display`; `angles.py` `measure`; `protocol.json`.

### A9. Three-point angle
- **What:** choose any three landmarks and the app shows the unsigned interior angle at the middle one (a straight limb reads 180 deg). It switches those three landmarks on, and by default becomes the main joint so the typed target applies to it; the test's own angles are then checked against the locked position instead. A tick box shows it alongside the other angles instead.
- **Problem / who:** only three dots are available in some pilot set-ups; the knee angle needs four. *You.*
- **Alters numbers:** it adds one angle; the other definitions are unchanged. Saved as `three_point` in `meta.json` and as an extra angle column group in the CSV.
- **Need:** Helpful (pilot).
- **Code:** `geometry.py` `interior_angle`; `engine.py` `set_three_point`, `clear_three_point`; `gui.py` three-point panel.

### A10. Reference, deviation and tolerance
- **What:** decides whether an angle is "in position".
- **Alters numbers:** the verdicts and deviation columns.
  - Main joint: `deviation = value - target`, where the target is what the operator types.
  - Other joints: after Lock, each is compared with the value that was **shown** at the moment of locking (a single smoothed frame, not averaged).
  - In tolerance when `|deviation| <= tolerance` (inclusive, no hysteresis). Defaults: 3 deg main joint, 5 deg other joints; both adjustable.
  - The CSV `_dev` and `_ok` are computed from the unsmoothed angle against these references.
- **Who:** *You* (you chose to keep it); the design was *Added*.
- **Need:** Helpful.
- **Code:** `engine.py` `lock_position`, `_compute_angles`.

### A11. Timestamps and video speed
- **What:** time base for the CSV and videos.
- **Alters numbers:** timing only.
  - Live camera: a frame's time is the PC clock when it was read (not the camera's own clock). `t_s` counts from the first recorded frame; `clock_s` is the raw clock.
  - Video file: time = frame number / the file's frame rate.
  - The raw and overlay videos are written at an estimated frame rate (from the last 30 frame intervals; 30 if fewer than 5), so they may not play at exactly real-time speed. **The CSV `t_s` is the authoritative time.**
- **Need:** Helpful; essential if force is added later.
- **Code:** `camera.py` `CameraSource.read`; `engine.py` `_fps`.

### A12. Review-tab statistics
- **What:** the summary numbers for a chosen time window.
- **Alters numbers:** computed, not measured. **They are now computed from the CSV, so they use the unsmoothed angles.**
  - Mean, SD (sample SD, ddof = 1), min, max, max |deviation|, % of frames in tolerance (among frames that have a verdict), lost frames.
  - `start>end` = mean of the last 5 valid values minus mean of the first 5 valid values in the window.
  - The SD therefore includes tracking noise; the smoothing no longer hides it.
- **Who:** *You* (review tab kept).
- **Need:** Extra.
- **Code:** `review.py` `Trial.summary`.

### A13. Ghost pose: parked, not active
- The dashed copy of the locked pose is commented out (code and checkboxes), marked `# PARKED-GHOST:`; its tests are skipped. See `PARKED.md`. It never altered numbers.

### A14. Mirror toggle (changes which picture the numbers come from)
- **What:** tick box "Mirror the image". The frame is flipped left-right **before** tracking, recording and drawing.
- **Alters numbers:** pixel positions in the CSV and both saved videos belong to the flipped picture, and `meta.json` says `mirrored_image: true`. Joint angles are the same if "Participant faces" is set to match what is shown; otherwise signed angles flip sign. Changing it clears the marked dots and the lock, and it cannot be changed while recording.
- **Who:** *You.*
- **Need:** Helpful (mirrored webcam images).
- **Code:** `gui.py` `VideoWorker._show`, `on_mirror_toggled`; `engine.py` `mirror`.

---

## B. Detection

| Feature | Who | Alters numbers? | Need | Code |
|---|---|---|---|---|
| White / black / Auto dot colour. Auto compares a 5x5 patch at the click with the median of the 81x81 window, once per dot, so white and black dots can in principle share one video (white on dark shorts, black on skin). **Not tested together yet.** Re-marking a dot keeps the colour it was first given | Added | Indirectly (selects top-hat or black-hat) | Essential | `tracker.py` `init_at` |
| 8 sagittal landmarks from your table (ASIS, PSIS, greater trochanter, lateral femoral epicondyle, fibular head, lateral malleolus, 5th metatarsal base and head) | You | No | Essential | `protocol.json` |
| Choose which landmarks to use, with presets such as "only Knee flexion"; angles that need a switched-off landmark disappear | You | Changes which angles exist; nothing else | Helpful (pilot) | `engine.py` `set_dot_enabled`, `angle_availability` |
| Message when Lock is refused, listing which angle needs which dot (replaces the confusing "every dot must be marked") | You | No | Helpful | `engine.py` `lock_problem` |
| Re-mark one dot by clicking it (nearest dot within 60 px) | Added | No | Helpful | `engine.py` `nearest_dot` |
| Dot quality: current contrast / contrast at marking; "weak" below 0.6, shown as an orange ring and a header warning | Approved | No (diagnostic) | Helpful | `tracker.py` `DotState.weak` |

## C. Tracking

The numeric behaviour is in A2 to A6. Other items:

| Feature | Who | Need | Code |
|---|---|---|---|
| Switching a landmark off then on needs it marked again | Added | Helpful | `tracker.py` `TrackerSet.set_enabled` |
| Trails: every dot draws its path over one recording (a gap where it was lost; a point is added only after a 0.5 px move). They start at Start recording, stay after Stop, reset at the next recording, and appear in the live view, the overlay video and the Review tab. A tick box turns them off. Display only; **no number is affected** | You | Helpful | `engine.py` `_update_trails`; `overlay.py` `_draw_trails`; `review.py` |

## D. Angle calculation

The method is in A7 to A10. Other items:

| Feature | Who | Alters numbers? | Need | Code |
|---|---|---|---|---|
| 7 tests with main joint and suggested target angles (from your `plot_proposal.m`): knee extension/flexion, hip extension/flexion, ankle plantar/dorsiflexion, belt squat | You | No | Essential | `protocol.json`; `protocol.py` `load_protocol` |
| Pelvic tilt tracked as a neighbour in every test (ASIS-PSIS vs horizontal; parked, left as built) | Added | Assumes a level camera | Helpful | `protocol.json` |
| Angles drop out automatically when a needed landmark is switched off | You | No | Helpful | `engine.py` `active_angles` |
| Editable `protocol.json` (dots, arms, signs, offsets, targets, tolerances) | Added | Defines the angles | Essential | `protocol.py` |

## E. Display

| Feature | Who | Need | Code |
|---|---|---|---|
| Live video with limb lines, arcs and angle numbers; green = in tolerance, red = out, white = no reference yet, grey = lost | You | Essential | `overlay.py` `draw` |
| Live trace: last 15 s, main angle with tolerance band, or drift of all joints | You | Helpful | `gui.py` `update_trace`; `engine.py` `history` |
| Header messages (locked, dot lost, weak dot, marker banner) | Added | Helpful | `engine.py` `process` |
| Pause and frame-by-frame stepping (video files only) | You | Extra for live use (practice) | `gui.py` `VideoWorker` |
| Keyboard keys: Space lock, R record, P pause, Left/Right step, S/E/M markers | Added | Helpful | `gui.py` |
| Scrollable control panel for small screens | Added | Helpful | `gui.py` |
| Line under "Video source" showing what the camera really delivers (resolution, measured frame rate, orange warning when it differs from the request) | You | Helpful | `camera.py` `granted`, `measured_fps`, `settings_problems`; `gui.py` `update_camera_info` |

## F. Calibration

**Nothing is implemented:** no lens correction, no pixel scale, no camera-tilt correction. A ruler in the trials and an in-app scale are planned (see I). The minimum dot size of about 2 cm is a protocol requirement, not checked by the app; the app only compares each dot with the size learned when you clicked it.

Things that look like calibration but are not: the dot size and contrast learned when you click a dot (A1), the dot-colour auto-detect (B), the "participant faces" setting (A8), and the camera lock below.

## G. Everything else

| Feature | Who | Need | Code |
|---|---|---|---|
| Webcam input. Requests MJPG, 1280x720, 30 fps, buffer size 1. The granted values and measured rate are now shown and saved (E, `meta.json` `camera_settings`) | You | Essential | `camera.py` |
| "Lock exposure and white balance" button (webcams only). It switches automatic exposure and white balance off and holds the current values, then reports in words whether the camera accepted it and stores the raw read-back in `meta.json` (`camera_settings.exposure_wb_lock`). **Not tried on a real webcam**; many webcams accept the request and ignore it | You | Helpful | `camera.py` `lock_exposure_wb`, `lock_summary` |
| Errors in the video thread no longer freeze the picture silently: the video stops, a message appears on the instruction line, any recording is closed and saved, and the traceback goes to `recordings/last_error.log` | You | Essential | `gui.py` `VideoWorker.run`, `_report_error` |
| Video-file input, practice video, synthetic test-video generator | Added (file input approved) | Extra | `camera.py`; `synthetic.py` |
| Recording: raw video, overlay video, per-frame CSV, trial details JSON | You | Extra for live use | `recorder.py`; `engine.py` `start_trial`, `stop_trial` |
| Event markers (MVIC start, MVIC end, numbered marker) | Approved | Extra | `engine.py` `add_event` |
| Review tab: scrub, plot, markers, trails tick box, summary export | You | Extra | `review.py` |
| `Start.bat`, README, `PARKED.md`, 126 automatic tests (one skipped), git history | Added | Extra | project root; `tests/` |

### CSV columns
`frame, t_s, clock_s, locked`, then per dot `<dot>_x, <dot>_y` (both **blank while the dot is lost**), `<dot>_lost, <dot>_q`, then per angle `<angle>_deg, <angle>_ref, <angle>_dev, <angle>_ok` (all from the **unsmoothed** angle; blank = lost or no reference), then `all_ok, event`. Only the chosen landmarks and the angles that can be computed from them are written. There is no longer a separate `_raw_deg` column because `_deg` is already raw.

### `meta.json` contents
`software_version`, participant, notes, test, target and tolerances (main and neighbour), `dots_used`, `dots_disabled`, `locked_references_deg`, `dot_kind`, `camera` and `camera_settings` (requested, driver-reported, exposure/white-balance lock result), `mirrored_image`, `facing`, `smoothing`, `trails`, `reference_basis`, `three_point`, `tracker_settings` (every number in A1 to A6), `angles` (full definitions), the events list, frame count, written and measured frame rate. The locked dot positions that the ghost pose used are no longer saved.

---

## H. Unfinished or unreliable

1. **Mostly synthetic evidence.** The only real footage so far is one practice video with undersized dots. Thresholds are not tuned for real skin, lighting, straps or glare, and the lost-dot circle (2.0 x diameter) should be re-tested on a proper video with spec-sized dots, a hand over a dot, and white dots on dark clothing.
2. **The "raw" video is not lossless.** Frames are re-encoded with the `mp4v` codec. It is also not exactly real-time speed (A11). With the mirror on it is the flipped picture, not the camera's original.
3. **The main joint is judged against the typed target, not the locked pose.** If you lock at 63 deg with a target of 65 deg, the deviation shown is relative to 65. There is no "change since lock" number for the main joint (parked); the review's `start>end` partly covers it.
4. **The locked reference is a single smoothed frame**, so its noise carries into every neighbour deviation.
5. **Screen and CSV can disagree.** The screen colours use the smoothed angle, the CSV flags use the unsmoothed one, so near the tolerance edge the two can differ for a frame or two.
6. **Webcam specifics are only partly handled:** the exposure/white-balance lock is untested on real hardware, dropped frames are not counted, and timestamps jitter (PC clock, not camera clock).
7. **Camera tilt biases pelvic tilt**, which is measured against image-horizontal. The hip angle is not affected by rotation.
8. **A wrong "participant faces" setting** flips hip and pelvic-tilt signs with no warning (also after switching the mirror).
9. **False precision.** Angles display to 0.1 deg, but real accuracy is unknown and likely worse. On synthetic data the typical error was about 0.5 deg or less.
10. **Stepping backwards** keeps the tracker state, so it can mis-track after a jump. Files only, and not recorded.
11. **Marker timing:** a marker attaches to the next processed frame (up to one frame late) plus the operator's reaction time.
12. **Methods reproducibility:** `meta.json` now stores every tracker number, the smoothing, camera settings and angle definitions, but `software_version` is still "0.2.0" although the code has changed, and the git commit is not recorded.
13. **Green/red flicker** at the tolerance edge (no hysteresis).
14. **Dots of different colours in one video** (A, B) have not been tested together, and a dot's colour cannot be set by hand per dot.
15. **No minimum dot size check, no scale.** See F and PARKED.md.
16. **Not built:** network-stream entry (the code path exists, no box in the window), two cameras, frontal view, lens calibration, force/ADC, non-Windows support.

Closed since the first inventory: silent video-thread errors, unverified camera resolution and frame rate, tracker thresholds missing from `meta.json`, no mirror toggle, the "two trackers on one dot" rule leaving a tracker on the wrong blob, Review statistics hiding noise, held x/y written while a dot is lost.

---

## I. Roadmap (not built)

### Asked for, waiting for a go-ahead
| Feature | Note |
|---|---|
| Ruler in the trials, with a scale in the app: click both ends of the ruler, type its length, store pixels per centimetre in `meta.json`, show the minimum dot size in cm | Only valid at the ruler's distance from the camera; lens distortion is not corrected. A pixel-based fallback slider was proposed |
| Minimum dot size (about 2 cm) and brightness check in the protocol | Needs the scale above or a camera distance; the pixel floor is listed under "Proposed, not added" in `PARKED.md` |
| Test of white and black dots in the same video | Synthetic test first; a per-dot colour override only if Auto fails |

### Parked (see `PARKED.md`)
Pelvic tilt as a neighbour (stays as built), frontal view, two cameras, lens calibration, camera alignment aid, phones as cameras, ADC and load cell (an Arduino with a 16-bit ADC or an HX711 is a candidate; needs the SIC-A2 output spec), change-since-lock for the main joint, custom dots and angles, ghost pose.

### Other ideas
| Feature | Why |
|---|---|
| Real-footage validation helper | Goniometer vs app readings in one table, with agreement in degrees |
| Dropped-frame counter, git commit and version bump stored in `meta.json` | Closes H6 and H12 |
| Stream-address box, remembered settings, session checklist, batch summary export, compare trials | Workflow |
| True 3D angles | Calibrated camera pair; a separate decision |

### Suggested order
1. Real-world check of what exists (webcam test, then goniometer comparison, then a proper lost-dot video).
2. Ruler/scale and dot-size check; white/black dot test.
3. Phones and lens calibration, when the cameras are known.
4. Force, once the ADC is chosen.
5. Frontal view and two cameras.
