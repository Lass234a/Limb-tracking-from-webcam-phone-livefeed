# Feature inventory

Version 0.5.2 (`meta.json` now stores the version, a `meta_schema`, and a `provenance` block: code hash, git commit, library versions, computer, protocol-file hash). First written 2026-10-02 at commit `77c4d7d`; many sections updated since, see CHANGELOG.md. Function names are given instead of line numbers because lines move.

**Purpose of the app:** show a live view of joint positions and angles from marker dots on bone landmarks, so each participant (maximal isometric contractions, strapped into a harness with an S-type load cell) can be placed in the correct position quickly. A webcam is the current proof of concept; two cameras (frontal and sagittal) are planned and parked.

**Status:** everything below is tested on synthetic video, plus one real practice video (`Bundepus.mp4`, which is not in the repository) for the lost-dot rule and the partly-covered-dot flag, and stand-ins for cameras, phones and a network stream. There are 378 automatic tests (1 skipped: the parked ghost). Nothing has been validated against a goniometer or inclinometer (see `SETUP_AND_VALIDATION.md`), no phone has been tried, and the dots in the practice video are smaller than the planned 2 cm.

**What the app does NOT do to the data:** no interpolation of missing dots or angles, no drift correction, no outlier rejection (other than the lost-dot rule), no frame-rate resampling, no lens-distortion correction, no camera-tilt correction. A pixel-to-centimetre scale exists only if you mark a ruler (section F).

**What changed since the first inventory (2026-10-01), following your decisions:** the growing search and the old lost-dot rule were replaced (A6); the One-Euro filter was replaced by a 0.2 s moving average that only changes the screen (A7); the ghost pose was commented out (A13); then a three-point angle mode, trails, a ruler scale, strap and anchor dots with the angle of pull and moment arm, dot displacement, the hold-angle report, Tabel 2 protocol positions, camera handling for webcams, phones and streams, recording safety and many checks were added. `CHANGELOG.md` lists them in one place.

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
  - A blob is accepted only if its area is 0.40 to 3.5 times the running area (running area = 0.9 old + 0.1 new; the lower limit was 0.3 until 2026-10-09, when small fragments of a dot hidden by a hand were accepted as the dot) and its longer side is at most `2.5 x shorter side + 2`.
  - Of the accepted blobs, the one nearest the predicted position is used.
  - **Dot merged with a shadow, finger or strap (2026-10-09):** if no blob is accepted, or the best one is bigger than 1.5 x the area when marked, or longer than `1.6 x shorter side + 1`, the search is repeated at 0.6, 0.75 and 0.9 x the dot's contrast and the first acceptable blob that is no longer merged is used. The running dot area is no longer updated from a blob bigger than 1.5 x the marked area. Found on the board recordings: before this, the position was dragged toward the finger and the dot was then lost until the finger left. On the Bundepus reference video this moves positions by at most 0.27 px (angles by 0.02 deg) in 40 of 312 frames; the synthetic reference is identical.
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
- **What:** a tracker ignores blobs within `0.8 x diam` of where another tracked dot is **expected** this frame (its last position plus its step, with the circle widened by the length of that step), so a fast-moving neighbour cannot be taken by a covered dot (audit TRK-01). A second rule handles two trackers ending up within `0.5 x diam` of each other: the one that fits its own expectation worse (distance from its predicted position, or from where it froze if it was lost) gives the dot up, is put back at its previous position (`undo_update`) and declared lost; on a tie the later one. Normal tracking output is unchanged (checked against the saved reference run and `Bundepus.mp4`); a dot moving faster than about 360 deg/s may now be lost instead of mis-tracked.
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

### A8a. Participant-faces check (new)
- **What:** the program works out which side of the body the marked dots show, from the turning direction of the pelvis dots (ASIS, PSIS, trochanter), or of the foot dots (malleolus, fibular head, metatarsal base and head) if the pelvis is not marked, and compares it with the *Participant faces* setting. This works for either leg, with the mirror on or off, and for upright, seated or lying postures. If they disagree for 5 frames in a row, the header says "CHECK 'Participant faces' ...", the setting's box turns amber, and **Lock is refused** with the reason (the hip and pelvic-tilt signs would be wrong). `meta.json` records `facing` and `facing_detected`. With only three dots (three-point mode) there is nothing to check. Record is not blocked, but Lock is the step before it.
- **Code:** `geometry.py` `view_handedness`; `engine.py` `_check_facing`, `facing_mismatch`, `lock_problem`.

### A8. Angle definition and conventions
- **What:** an angle is measured between two "arms". An arm is a vector between two dots, "perpendicular to the line between two dots pointing toward the feet" (the pelvis midline), or a fixed direction (up, down, forward).
- **Alters numbers:** yes.
  - Shown value = `sign x angle + offset`. Hip: signed, sign -1 (flexion +, extension -). Knee: signed (flexion +, hyperextension -). Ankle: unsigned, sign -1, offset +90 (dorsiflexion +). Pelvic tilt: signed, sign +1 (anterior +).
  - If "participant faces left" is chosen, the x-direction of dot-based arms is mirrored so signs mean the same either way. A wrong setting flips the sign of hip and pelvic tilt without any warning; knee and ankle are unaffected. "Faces" refers to the picture on screen, so after switching the mirror (E3) it must be set again.
- **Who:** *You* (landmarks and ASIS-PSIS perpendicular; you chose to keep the definitions); the facing setting and sign/offset mechanism were *Added*.
- **Need:** Essential.
- **Code:** `geometry.py` `resolve_arm`, `vector_angle`, `to_display`; `angles.py` `measure`; `protocol.json`.

### A9. Three-point angle
- **Two conventions (audit ANG-05):** *interior angle* (default; a straight limb reads 180, a knee flexed 70 reads 110) or, with the tick box *Show it as flexion*, **flexion = 180 minus the interior angle** (a straight limb reads 0, like the protocol's flexion angles and targets). The convention is in the angle's name on screen and in `meta.json` (`three_point_convention`). The typed target never silently changes meaning: switching the convention converts it (110 interior becomes 70 flexion), and in the interior convention the test's own flexion positions are not offered or carried over (the target starts as *no target*). Because the protocol targets are flexion, the flexion convention is the one to use when the protocol positions apply. The three-point knee (trochanter, epicondyle, malleolus) is a few degrees different from the protocol knee (which uses the fibular head), by how far the fibular head sits off the epicondyle-malleolus line.
- **What:** choose any three landmarks and the app shows the unsigned interior angle at the middle one (a straight limb reads 180 deg). It switches those three landmarks on, and by default becomes the main joint so the typed target applies to it; the test's own angles are then checked against the locked position instead. A tick box shows it alongside the other angles instead.
- **Problem / who:** only three dots are available in some pilot set-ups; the knee angle needs four. *You.*
- **Alters numbers:** it adds one angle; the other definitions are unchanged. Saved as `three_point` in `meta.json` and as an extra angle column group in the CSV.
- **Need:** Helpful (pilot).
- **Code:** `geometry.py` `interior_angle`; `engine.py` `set_three_point`, `clear_three_point`; `gui.py` three-point panel.

### A10. Reference, deviation and tolerance
- **What:** decides whether an angle is "in position".
- **Alters numbers:** the verdicts and deviation columns.
  - Main joint: `deviation = value - target`, where the target is what the operator types.
  - Neighbour joints with a required angle in the protocol (Tabel 2: hip 90, knee 0 or knee 90): that angle, from the start. Other joints: after Lock, each is compared with the **unsmoothed angle averaged over the 0.2 s before Lock**. This is the same whether the display smoothing is on or off (audit DOC-02), so the saved `_ref` and `locked_references_deg` do not depend on that switch.
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
  - The raw and overlay videos are tagged with the **source's own frame rate**: a video file's rate, or a camera's measured arrival rate (falling back to the requested rate, then to an estimate from the last frame times, always between 1 and 240 fps). Redrawing the same frame, stepping back and switching source no longer distort it (audit CR-06, CAM-03). If the computer cannot keep up while recording, the video can still play faster than real time. **The CSV `t_s` is the authoritative time.**
  - CSV `frame` is the 0-based row number, which equals the frame index in `raw.mp4` and `overlay.mp4`; the new last column `source_frame` is the frame number in the source (the file's frame, or the camera's read count) (audit CR-08).
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

### A12a. The angle during the hold (audit VAL-01)
- **Why:** the joint angle changes between positioning and the contraction (in the literature by about 19 deg for the knee against a rigid arm), so the typed target is not the angle at which the moment was produced. The angle to report is the one measured **during the hold**.
- **What the Review tab and the exported summary now give per angle** (all from the unsmoothed CSV angle, for the analysis window, which opens on the MVIC start-to-end markers): `plateau_mean_deg` (mean over the middle 60 % of the window: the first and last 20 % are left out as ramps), `median_deg`, `rest_mean_deg` (mean over the 1 s before the window, at least 5 valid frames), `rest_to_hold_shift_deg` (plateau minus rest), `target_deg` (the typed target for the main joint, the reference for the others) and `mean_minus_target_deg`. A window too short for a plateau or rest gives a blank (NaN), never a guess.
- **The export** now starts with `participant, test_key, trial, tolerance_deg, software_version, window_source` so the summaries of a whole study can be put in one table; `window_source` says whether the window was the MVIC markers or set by hand. Without markers the screen says so.
- **Which number goes to AnyBody is your decision.** Both are now reported side by side: the plateau mean angle, and (with force recorded) the main joint angle at the peak force (mean of the video frames within 0.1 s of the peak; section A16).

### A12b. Dot displacement in Review (new)
- **What:** for the chosen analysis window, the Review tab shows how far every recorded dot moved, and the export writes `<trial>_displacement.csv` next to the summary CSV.
  - **end shift:** mean of the dot's last 5 valid positions minus the mean of its first 5 valid positions in the window (a vector dx, dy and its length), in pixels and, with a ruler scale, in cm.
  - **max shift:** the largest distance from the starting position at any frame in the window. A single bad tracking frame shows up here; check the trails.
  - **vs anchor:** the same numbers for the dot's position relative to the anchor marker (when it was used). This removes movement of the whole picture (a camera that was bumped). It does **not** separate the person shifting from a strap sliding on the skin; for the strap, use the pull numbers.
  - **Pull section** (only when strap and anchor dots were used): mean, min, max and start>end for the pull angle, moment arm and strap-to-joint distance (cm with a scale).
- **Start of the hold:** the window now **opens on the MVIC start-to-end window when both markers were pressed**, otherwise on the whole trial, so the displacement (and all Review statistics) start at the hold by default. Drag the blue window or press *Window = MVIC start to end* to change it. (This changes the default window of the existing Review statistics too.)
- **Alters numbers:** none are measured; all are computed afterwards from the CSV positions (and the scale in `meta.json`). Lost frames are skipped, never filled in. Image pixels are the picture's, so displacement in cm is only right at the ruler's distance from the camera.
- **Code:** `limbtrack/displacement.py`; `review.py` `Trial.displacement`, `pull_summary`, `ReviewTab._displacement_text`, `export`.

### A13. Ghost pose: parked, not active
- The dashed copy of the locked pose is commented out (code and checkboxes), marked `# PARKED-GHOST:`; its tests are skipped. See `PARKED.md`. It never altered numbers.

### A14. Mirror toggle (changes which picture the numbers come from)
- **What:** tick box "Mirror the image". The frame is flipped left-right **before** tracking, recording and drawing.
- **Alters numbers:** pixel positions in the CSV and both saved videos belong to the flipped picture, and `meta.json` says `mirrored_image: true`. Joint angles are the same if "Participant faces" is set to match what is shown; otherwise signed angles flip sign. Changing it clears the marked dots and the lock, and it cannot be changed while recording.
- **Who:** *You.*
- **Need:** Helpful (mirrored webcam images).
- **Code:** `gui.py` `VideoWorker._show`, `on_mirror_toggled`; `engine.py` `mirror`.

### A15. Angle of pull and moment arm (strap and anchor dots)
- **What:** when both the strap marker and the anchor marker are switched on, the app shows and records, every frame:
  - **pull angle** (`pull_angle_deg`): the angle between the direction of pull (from the strap marker towards the anchor marker) and the segment the strap acts on, measured along the segment from its proximal to its distal dot. 90 deg = square to the segment; below 90 the pull has a component towards the feet, above 90 towards the body. `pull_off90_deg` is the same minus 90.
  - **moment arm** (`moment_arm_px`, `moment_arm_cm`): shortest distance from the joint dot to the line of pull.
  - **strap-joint distance** (`strap_joint_px`, `strap_joint_cm`): straight distance from the joint dot to the strap marker.
- **Which segment and joint:** the second arm and the fulcrum of the test's main joint: knee tests use the shank (fibular head to malleolus) and the epicondyle, hip tests the thigh (trochanter to epicondyle) and the trochanter, ankle tests the foot (5th metatarsal base to head) and the malleolus.
- **Alters numbers:** it adds columns only; no angle or dot position changes (checked by a test). Nothing is judged and nothing blocks Lock or Record. The pull is shown as a line in the picture header ("PULL 111.8 deg to the segment (+21.8 from perpendicular); moment arm 10.2 cm") and in the saved overlay video. Values are unsmoothed.
- **Blank, not guessed,** whenever the strap, anchor or a needed segment dot is lost or switched off.
- **cm values** need the ruler scale and use the scale as marked; they are only right at the ruler's distance from the camera. Without a scale the header shows pixels ("no scale").
- **Limit:** this is the 2D picture. A pull toward or away from the camera, or a rope that is not in the leg's plane, is invisible, and the real pull angle then differs.
- **Saved:** the six CSV columns (only when both dots are used) and `meta.json` `pull` (segment, joint dot, definitions; null otherwise).
- **Code:** `limbtrack/pull.py` `compute_pull`; `engine.py` `_compute_pull`, `pull_segment`.

---

## B. Detection

| Feature | Who | Alters numbers? | Need | Code |
|---|---|---|---|---|
| White / black / Auto dot colour. Auto applies both the white-dot and the black-dot filter to the 81x81 window and takes the one that responds more strongly near the click, **on every click**; the choice is kept only if the marking succeeds, so a failed click can no longer leave a dot stuck on the wrong colour (audit TRK-03). White and black dots can in principle share one video (white on dark shorts, black on skin) because each dot decides for itself. Tested together on synthetic pictures (white dots on dark cloth, black dots on light skin, one picture); not yet on real clothing | Added | Indirectly (selects top-hat or black-hat) | Essential | `tracker.py` `init_at` |
| 8 sagittal landmarks from your table (ASIS, PSIS, greater trochanter, lateral femoral epicondyle, fibular head, lateral malleolus, 5th metatarsal base and head) | You | No | Essential | `protocol.json` |
| **Strap and anchor dots (new, optional):** two extra tracked dots, "strap marker" (on the person, where the strap attaches) and "anchor marker" (fixed end of the strap, rope or chain; put it on the rope itself if the anchor is out of the picture). Both start switched off; tick them in the Landmarks box (or the preset "all landmarks + strap and anchor") and they must then be marked like any landmark. They are tracked, shown (a magenta line is drawn between them) and written to the CSV like any other dot, with trails; they belong to no angle, so they cannot change an angle, be judged, or block Lock. A lost strap or anchor dot does not affect any angle. Defined by `optional_dots` in `protocol.json`; `meta.json` lists them under `optional_dots_used` (and `dots_disabled` now only lists landmarks you switched off). Place the strap dot at least about three dot-widths from the malleolus dot, or the trackers can be confused | You | No | Helpful | `protocol.py` `TestDef.all_dots`; `engine.py` `active_dots`; `overlay.py` |
| Choose which landmarks to use, with presets such as "only Knee flexion"; angles that need a switched-off landmark disappear | You | Changes which angles exist; nothing else | Helpful (pilot) | `engine.py` `set_dot_enabled`, `angle_availability` |
| **Marking mistakes are caught** (audit GUI-14, TRK-07): a dot cannot be marked on a sticker that another landmark already holds ("That is the ... dot, which is already marked"), also at the first marking. Four distance checks that hold in any posture and for either leg flag labels that look swapped or clicked in the wrong order: the 5th metatarsal head farther from the malleolus than the base, the shank longer than the foot, the thigh longer than the pelvis line, the malleolus farther from the epicondyle than from the fibular head. A failing check puts "CHECK LABELS" in the header and refuses Lock with the reason; *Re-mark all dots* clears it. (Epicondyle/fibular head swaps are only caught when the knee is not strongly bent; an ASIS/PSIS swap is caught by the Participant-faces check.) | You | No | Helpful | `engine.py` `marking_problems`, `mark_dot` |
| Message when Lock is refused, listing which angle needs which dot (replaces the confusing "every dot must be marked") | You | No | Helpful | `engine.py` `lock_problem` |
| Re-mark one dot by clicking it. A lost dot is chosen first (nearest lost dot within 150 px of where it froze); a tracked dot only if the click is within twice its diameter (at least 15 px). A re-mark onto a dot another tracker holds is refused and the old marker is kept. The result is announced ("Re-marked: ..."). **Clicks on the video are ignored while recording** (audit GUI-01, DOC-10) | Added | No | Helpful | `engine.py` `nearest_dot`, `mark_dot`; `gui.py` `on_click` |
| **Partly covered or cut-off dots are flagged** (audit TRK-02): a tracked dot is marked *partial* when its visible area drops below 70 % of its normal area (the normal area never falls below 80 % of the area at marking, so a covered dot cannot teach the tracker a small size), when the blob is at least twice as long as wide, or when its centre is within half a diameter of the image edge. A flagged dot gets an orange ring and a header line ("dot partly covered or cut off (its centre may be shifted)"), a `<dot>_partial` column at the end of the CSV (1 = flagged) and a per-dot frame count in `meta.json` (`partial_dot_frames`). **It is a warning only**: the dot keeps being tracked and no verdict changes. Checked: no flags on clean, noisy (sd 6) or blurred synthetic dots, none on the three dots of `Bundepus.mp4` (0 of 862 tracked dot-frames), and a dot half covered on one side is flagged in every frame; a dot only about a quarter covered (centre shift about 0.1 of its width) is not flagged, and one covered on a diagonal is flagged only part of the time. Thresholds are in `tracker.py` and `meta.json` `tracker_settings`; re-check them on a video with your real stickers | You | No (diagnostic) | Helpful | `tracker.py` `DotState.partial` |
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
| **Changing the test keeps the marked dots** (audit GUI-06): when the new test uses the same landmarks (all Tabel 2 tests do), the dots stay marked and the operator's own tolerances stay, so the 120 s changeover is not spent re-clicking eight dots. The lock is dropped (the neighbour angles and their required values belong to the test), the new test's first position and required neighbour angle apply at once. A test with other landmarks starts afresh | You | No | Essential | `engine.py` `set_test` |
| 8 tests from Tabel 2 of the protocol draft (2026-09-28), with suggested positions and the required angle of the neighbouring joint: knee extension (hip 90) 20/70/110; knee flexion (hip 90) 20/90; hip extension knee 0: 0/45; hip extension knee 90: 0/90; hip flexion (knee 90) 0/90; ankle plantarflexion and dorsiflexion (knee 0) 0/-20; plus belt squat (no positions). The required neighbour angle (`neighbour_targets` in `protocol.json`) is judged from the start with the neighbour tolerance (5 deg), is the `_ref` in the CSV and is saved in `meta.json` (`neighbour_targets_deg`); other neighbours still use their value at Lock. The 4 frontal hip abduction/adduction positions are not in this app. The ankle row follows Tabel 2 (0/-20), not section 2.7 (-20/0/20), because only Tabel 2 gives the stated 19 positions | You | Changes the verdicts of the neighbour joints (A10) | Essential | `protocol.json`; `protocol.py` `load_protocol`; `engine.py` `_compute_angles` |
| protocol.json is checked at start-up: every mistake is listed with its place and a did-you-mean hint (audit COD-06); `python -m limbtrack.protocol check` checks without starting | Added | No | Essential | `protocol.py` `validate_protocol` |
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
| Keyboard keys: Space lock, R record, P pause, Left/Right step, S/E/M markers. Letter keys are typed into the Participant/Notes boxes while those have focus, so **F5 record, F6 MVIC start, F7 MVIC end, F8 marker, F9 lock** always work; Enter or Esc leaves a text box; an orange hint shows while a box has focus; each marker is confirmed on screen with a running count (audit GUI-13) | Added | Helpful | `gui.py` `TextField`, `mark` |
| Scrollable control panel for small screens | Added | Helpful | `gui.py` |
| "Target" can be set to *no target* (lowest value of the box): the main joint is shown but not judged, with no target in the file name and `meta.json`. Tests without positions (belt squat) start that way instead of with a forced 0 deg (audit GUI-05) | You | Changes the verdicts of that test only | Helpful | `gui.py` `_set_setting` |
| Line under "Video source" showing what the camera really delivers (resolution, measured frame rate, orange warning when it differs from the request) | You | Helpful | `camera.py` `granted`, `measured_fps`, `settings_problems`; `gui.py` `update_camera_info` |

## F. Calibration

**Pixel scale (new):** a ruler of 10 to 20 cm with a dot sticker centred on each end, placed at the same distance from the camera as the joint and square to it. In the "Scale (ruler)" box of the Live tab you type the length between the dot centres, press *Mark the two ruler dots* and click both. The app stores pixels per centimetre. The two ruler dots are **fixed points** (a click snaps to the dot's centre once; they are not tracked afterwards, so a shadow or hand cannot lose them). The app therefore does NOT notice a camera or ruler that moves after marking: mark the ruler again. It shows the ruler (blue rings and a line) in the live picture, in the saved overlay video and in Review.
- **Alters numbers: none.** It does not touch any angle or dot position; it only provides the conversion from pixels to centimetres for later features (pull angle, moment arm, displacement in cm).
- **Warnings:** ruler shorter than 60 px in the picture (coarse scale). (Until 2026-10-09 the ruler dots were tracked and a lost dot or a 3 % change was flagged; that was dropped because shadows and hands lost the dots.)
- **Saved:** CSV column `scale_px_per_cm` (the same fixed value in every row; blank if no ruler was marked) and `meta.json` `scale` (ruler length, pixels per cm at marking, the two fixed dot positions, `ruler_dots_are_fixed_points`, and the validity note). Review draws the ruler from these positions and shows `ruler 15 cm = 5.80 px/cm` in its title line. `scale` is null when no ruler was marked.
- **Limits:** correct only for things at the ruler's distance from the camera; an object 5 cm nearer or farther at 2.5 m is off by about 2 %. Lens distortion is not corrected, so keep the ruler near the joint and near the middle of the picture. The accuracy also depends on how exactly the dot centres sit on the ruler marks (1 mm off on a 15 cm ruler is 0.7 %).
- It cannot be changed while recording, and it is cleared when a new source is opened, the mirror is switched, or the dot colour setting is changed.
- **Code:** `limbtrack/scale.py` `ScaleReference`; `engine.py` `mark_scale_dot`, `clear_scale`; `overlay.py` `_draw_scale`; `gui.py` scale box.

**Dot-size check (new):** with a scale set, the scale box lists the approximate width of the smallest dot in cm and says **TOO SMALL** for any dot under the protocol minimum of 2.0 cm. The width comes from the dot's blob size when it was marked, minus 1.3 px for soft edges (the unadjusted blob reads about 1.3 px too large on synthetic dots), divided by the scale. It is an estimate: check it once against stickers of known size under your real lighting and focus. Saved in `meta.json` as `dot_size_check` (limit, each dot's estimate, which are below). No effect on tracking or angles. Code: `engine.py` `dot_sizes_cm`, `small_dots`; `scale.py` `estimate_diameter_cm`.

**Still not implemented:** lens correction, camera-tilt correction.

Things that look like calibration but are not: the dot size and contrast learned when you click a dot (A1), the dot-colour auto-detect (B), the "participant faces" setting (A8), and the camera lock below.

## G. Everything else

| Feature | Who | Need | Code |
|---|---|---|---|
| **Cameras are listed by name** ("1: Iriun Webcam"), read from Windows without opening any camera (the small `pygrabber` package; without it the indexes are tried one by one and called "Camera N"). A phone used as a webcam through an app such as Iriun, DroidCam or Camo (Android, including Samsung Galaxy S24/S25/S26, and iPhone) appears there once the app runs on the phone and the PC; press *Scan cameras* after starting it. The program then treats it like any webcam (format, frame rate and gaps are checked and recorded the same way) | You | Essential | `camera.py` `camera_names`, `list_cameras` |
| **Stream address** (audit CAM-08): a box *stream address* and *Connect to stream* open a network stream published by a phone app (http, https, rtsp, rtmp, udp, tcp, srt, rtp). One rule (`classify_source`) tells webcam, stream and file apart everywhere. A stream is opened with a 5 s open and 1 s read time-out and read by a background reader that always hands over the newest frame, so a slow computer skips frames (counted as `frames_discarded` in `meta.json`) instead of showing a picture that lags behind more and more. Tested against a local stream server; **not tried with a real phone app**. A stream behaves like a live camera in every other way (gaps, reconnect, recording) | You | Helpful | `camera.py` `classify_source`, `_LatestFrame`; `gui.py` `connect_stream` |
| **Camera set-up checklist and camera check** (audit ARC-06, ARC-08): tick boxes for stabilisation off, zoom 1x, focus locked, exposure/white balance locked in the camera app, orientation locked, USB connected, plus the camera/phone model and app, are saved with each trial (`camera_setup_checklist`). *Check the camera (5 s)* measures the live feed (frame rate, longest wait for a frame, format) and gives a plain verdict ("Camera OK: 1280x720, 29.9 fps, longest wait 0.04 s" or "NOT good enough: only 12.0 fps ..."); the last result is saved as `camera_check`. Both cannot be changed while recording | You | Helpful | `camera.py` `health`; `gui.py` `check_camera` |
| Webcam input. Requests 1280x720 and 30 fps first and the compressed MJPG format last (the other order made some DirectShow webcams fall back to uncompressed YUY2 at about 10 fps; audit CAM-04), buffer size 1. The granted values, the video format the camera reports and the measured rate are shown and saved (E, `meta.json` `camera_settings`). A slow rate with a non-MJPG format is explained as such; otherwise the warning names dim light or a computer that cannot keep up | You | Essential | `camera.py` |
| "Lock exposure and white balance" button: **switched off** (audit CAM-05; on some webcams the lock blacks out the picture and reports the opposite). A webcam is now opened on automatic exposure and white balance so settings left over from an earlier session cannot black it out; use steady lighting or lock exposure in the phone/camera app. The old code is still in `camera.py`. Original description: it switches automatic exposure and white balance off and holds the current values, then reports in words whether the camera accepted it and stores the raw read-back in `meta.json` (`camera_settings.exposure_wb_lock`). **Not tried on a real webcam**; many webcams accept the request and ignore it | You | Helpful | `camera.py` `lock_exposure_wb`, `lock_summary` |
| **A live camera that goes quiet does not end the session** (audit CR-05, CAM-02): the app waits (the window says "No frames from the camera for x s"), after 2 s it closes and opens the same camera again, and only after 6 s without a frame does it end the session (the recording is saved up to there). Network streams are opened with a 5 s open and 1 s read time-out instead of the default 30 s freeze. The number of failed reads and reconnects goes into `meta.json` (`camera_settings`). A webcam comes back on automatic exposure, so an exposure lock has to be pressed again. While recording, any stretch longer than 3 normal frame intervals (and 0.15 s) with no frame is listed in `meta.json` `gaps` (count, total, longest, where) and the Review title shows GAPS; the angles during a gap are unknown and nothing is interpolated | You | Essential | `gui.py` `VideoWorker._loop`; `camera.py` `reopen`; `recorder.py` gaps |
| **Video encoding runs on its own thread** (audit CAM-07, COD-08, DAT-10): the raw and overlay videos are encoded in the background while tracking carries on, so recording is lighter on the tracking thread. Frames are never dropped: the queue holds 90 frames and, if the disk is slower than that, the program waits rather than letting video and CSV drift apart; a failing disk is reported ("Saving the video failed"). `meta.json` `video_writer` lists frames encoded and the deepest the queue got. "Also save the overlay video" (on by default) can be switched off: only `raw.mp4`, the CSV and `meta.json` are written, which roughly halves the encoding work, and the Review tab's *Export overlay video* draws the same picture (angles, trails) for every frame afterwards as `..._overlay_export.mp4` | You | Essential | `recorder.py` `_VideoThread`; `review.py` `export_overlay_video` |
| **A recording survives trouble** (audit DAT-01 low-risk part, DAT-06, COD-04): every CSV row is written to disk as it is recorded; a complete `meta.json` (status `recording`) exists from the first frame and is replaced in one piece at the end (status `complete`), so a trial cut short by a crash or power loss still opens in Review with its numbers (the video may not open, and Review says UNFINISHED). If the camera changes picture size mid-recording (a phone that rotates), the recording is stopped and saved up to that frame, `meta.json` gets `stopped_because`, the dots are cleared and the header asks you to mark them again, so CSV rows and video frames always match. A failure while saving still clears the recording state and writes `meta.json` with `close_errors`; the window says NOT SAVED PROPERLY. The video files are still MP4 (a hard crash can leave them unreadable; an AVI container is a later option) | You | Essential | `recorder.py` `write_start_meta`, `FrameSizeChanged`, `close`; `engine.py` `_picture_size_changed`, `stop_trial` |
| **Settings are frozen while recording** (audit CR-01, GUI-02, COD-02, DAT-02): the test, target, both tolerances, dot colour, "Participant faces", mirror, landmarks, three-point angle, ruler scale, Re-mark all, the Lock button and the exposure-lock button are greyed out during a recording, and the engine itself refuses the changes ("Stop recording before changing ..."; a combo that was changed anyway snaps back with that message). Drop-downs and number boxes ignore the mouse wheel unless they have focus, so scrolling the side panel can no longer change them. **Lock the position before you press Record**: the lock cannot be changed during a recording. `meta.json` reports the settings as they were at the start; `changed_during_recording` appears only if something slipped through anyway | You | Essential | `engine.py` `_FrozenWhileRecording`, `_forbid_while_recording`, `_trial_meta`; `gui.py` `NoWheelComboBox`, `refresh_status` |
| Errors in the video thread no longer freeze the picture silently: the video stops, a message appears on the instruction line, any recording is closed and saved, and the traceback goes to `recordings/last_error.log` | You | Essential | `gui.py` `VideoWorker.run`, `_report_error` |
| Video-file input, practice video, synthetic test-video generator | Added (file input approved) | Extra | `camera.py`; `synthetic.py` |
| Recording: raw video, overlay video, per-frame CSV, trial details JSON. **Record is refused** when no video is coming in (source stopped or dead) or a video file has already ended. While recording, the label turns red with "NO VIDEO - nothing is being recorded" if frames stop arriving for more than 0.5 s. After Stop the label shows the frame count and duration, calls an empty trial "NOTHING RECORDED" and a trial under 30 frames "VERY SHORT RECORDING"; Review explains an empty trial instead of failing (audit COD-01) | You | Extra for live use | `recorder.py`; `engine.py` `start_trial`, `stop_trial`, `seconds_since_last_frame`; `gui.py` `toggle_record` |
| Event markers (MVIC start, MVIC end, numbered marker) | Approved | Extra | `engine.py` `add_event` |
| Review tab: scrub, plot, markers, trails tick box, summary export, dot-displacement table and export (A12b) | You | Extra | `review.py` |
| `Start.bat`, README, `PARKED.md`, 126 automatic tests (one skipped), git history | Added | Extra | project root; `tests/` |

### Privacy crop and masks (audit REP-05)
Does not alter any number. The tracker always sees the whole picture and the CSV holds whole-picture pixel positions. The crop (`x, y, w, h`, even sizes, at least 32 px) and up to 3 masks are applied only to the copies of the frames written to `raw.mp4` and `overlay.mp4`. Recording is refused when a marked dot is within 30 px of the crop edge or inside a mask; while recording, a dot within 10 px of the crop edge or inside a mask is flagged and counted. The crop and masks are cleared when the mirror or the picture size changes and are fixed while recording. Review pads a cropped video back to the full size (black outside), so everything lines up. Also: participant code format `P01`..`P999`, notes with an `@` or a 6+ digit run are refused, and the recordings folder can be changed (kept in `settings.json`). Code: `limbtrack/privacy.py`, `engine.py` (`set_privacy_rect`, `privacy_problem`, `_privacy_check`), `recorder.py`, `review.py`, `gui.py`. Tests: `tests/test_privacy.py`, `tests/test_privacy_gui.py`.

### A16. Force from the load cell (new, 2026-10-09; `limbtrack/force.py`)
- **What:** a load cell (S-type, with an amplifier) on one analog input of an NI DAQ (written for the USB-6003, 16 bit, 100 kS/s shared, +-10 V). The DAQ is sampled continuously at 1000 samples per second on its own thread; the newest 10 minutes are kept in memory. During a recording every sample is written to `<trial>_force.csv` as it arrives.
- **Alters numbers:** none of the angle numbers. It adds force numbers: volts are always saved; newtons = (volts - zero) x newtons per volt, from the calibration (zero with nothing on the sensor, then a known mass: newtons per volt = mass x 9.80665 / (load volts - zero volts)). A calibration is refused if the signal is not steady (spread above 20 mV over 2 s) or moves less than 5 mV, and cannot change during a recording.
- **Time alignment:** pictures and force samples both use `time.perf_counter()`. A DAQ chunk reaches the program a little after its last sample was taken, so the time of sample 0 is estimated as the smallest (arrival time - samples/rate) seen over the last 20 s; the error left is the smallest USB/driver delay, a few milliseconds, which is small against a 33 ms picture interval. The clock of the DAQ itself and the PC clock can differ by tens of ppm (a few milliseconds over a minute); the estimate follows that drift. The MVIC start marker gives an independent check.
- **Two channels:** the **voltage** is the primary record and is always saved, exactly as the DAQ measured it (input range +-10 V; a sample within 1 % of the edge is counted as `samples_at_range_limit` and flagged in the picture, because the peak may be cut off there). **Newtons** are computed from the voltage with a calibration and are blank until one exists. A calibration made later can be applied to old recordings in Review (*Apply the saved force calibration*): it computes newtons from the saved volts and never changes the volts; the export records which calibration was used.
- **Per picture:** `force_v` and `force_n` in `data.csv` = means of the samples between the previous and this picture (newtons blank if uncalibrated; both blank if no sample). The complete signal is in `force.csv`; use it for the peak (the peak of a contraction is short and lies between pictures).
- **How force and pictures are tied, and where that can go wrong:** the force is NOT sampled per picture. It runs on its own clock at 1000 samples per second and only the time stamps tie it to the pictures, so a slow or uneven frame rate does not lose force. The weak points are all in the time stamps of the PICTURES: (1) a picture is stamped when the program receives it, not when the camera took it. The phone, its video encoding, the USB or Wi-Fi link, DroidCam and Windows add a fixed delay, typically 0.1 to 0.3 s, and unknown until measured (tap test in `SETUP_AND_VALIDATION.md`). It shifts which picture belongs to the peak force by that delay. Review has a *Picture delay* box for it (default 0; saved in the export). (2) Frames arrive in bursts (the board recordings had 24 to 77 ms between pictures at a nominal 33 ms), which is +-15 to 30 ms of jitter in each stamp; the angle at the peak force therefore uses all pictures within 0.1 s of the (delayed) peak. (3) At 17 to 26 fps there are only 2 to 3 pictures in that 0.2 s window; in a steady hold the angle changes little, so this matters for fast movement, not for an isometric hold. (4) A dropped or frozen picture stream is flagged as gaps in `meta.json`; the force is unaffected. (5) The force clock estimate (see above) is good to a few milliseconds.
- **Not recorded** with a video file as the source (its clock does not match the DAQ).
- **Review:** the force on a right-hand axis; for the window: peak and when, plateau mean (middle 60 %), rest level (1 s before), peak above rest, and the main joint angle at the peak force. In `_summary.csv`: `force_unit, force_peak, force_t_peak_s, force_plateau_mean, force_rest_mean, main_angle_at_peak_force_deg`.
- **Limits:** the sensor's own accuracy and linearity are not checked by the program; a load applied off the sensor axis, or the strap pulling at an angle, changes what the sensor reads (the pull angle is recorded separately when the strap and anchor dots are used); the amplifier must give a voltage within +-10 V (a current output, 4-20 mA, needs a shunt resistor, the USB-6003 has no current input).
- **Code:** `force.py`; `engine.py` `_take_force`, `_force_header`, `_force_meta`; `recorder.py` `write_force`; `review.py` `force_summary`; `gui.py` Force box.

### CSV columns
`frame, t_s, clock_s, locked`, then per dot `<dot>_x, <dot>_y` (both **blank while the dot is lost**), `<dot>_lost, <dot>_q`, then per angle `<angle>_deg, <angle>_ref, <angle>_dev, <angle>_ok` (all from the **unsmoothed** angle; blank = lost or no reference), then `all_ok, event, scale_px_per_cm` (the live ruler scale; blank when no ruler is marked or a ruler dot is lost), `force_n` (only when a force source was connected: mean newtons since the previous picture), and, only when the strap and anchor dots are both used, `pull_angle_deg, pull_off90_deg, moment_arm_px, moment_arm_cm, strap_joint_px, strap_joint_cm` (blank while a needed dot is lost), `source_frame`, and last one `<dot>_partial` per dot (1 = the dot looked partly covered or cut off in that frame). Only the chosen landmarks and the angles that can be computed from them are written. There is no longer a separate `_raw_deg` column because `_deg` is already raw.

### `meta.json` contents
`software_version`, participant, notes, test, target and tolerances (main and neighbour), `dots_used`, `dots_disabled`, `locked_references_deg`, `dot_kind`, `camera` and `camera_settings` (requested, driver-reported, exposure/white-balance lock result), `mirrored_image`, `scale` (ruler length, pixels per cm, null if no ruler), `dot_size_check`, `pull` (definitions, null if the strap and anchor dots were not used), `facing`, `smoothing`, `trails`, `privacy` (crop, masks, full picture size) and `privacy_dot_warnings`, `pipeline_timing` (work per frame against time between frames: shows whether the computer or the camera limits the frame rate), `reference_basis`, `three_point`, `tracker_settings` (every number in A1 to A6), `angles` (full definitions), the events list, frame count, written and measured frame rate. The locked dot positions that the ghost pose used are no longer saved.

---

## H. Unfinished or unreliable

1. **Mostly synthetic evidence.** The only real footage so far is one practice video with undersized dots (and one earlier recording). Thresholds are not tuned for real skin, lighting, straps or glare. Re-test, on a video with your real stickers: the lost-dot circle (2.0 x diameter), the partly-covered-dot thresholds, the dot-size estimate (1.3 px edge allowance), and white and black dots together on real clothing and skin.
2. **No validation against a reference yet.** See `SETUP_AND_VALIDATION.md` (jig and inclinometer check, sticker placement repeatability). Until it is done, the angles are positioning data, not validated measurements. Sticker placement (a few mm) is the largest error source.
3. **The "raw" video is not lossless and not crash-proof.** Frames are re-encoded with the `mp4v` codec into MP4 files, which can be unreadable after a hard crash or power cut (the CSV and `meta.json` are safe; Review then shows UNFINISHED). With the mirror on it is the flipped picture. An AVI container with a repair step is a possible later change.
4. **The main joint is judged against the typed target, not the locked pose.** There is no "change since lock" number for the main joint (parked). The Review hold numbers (plateau, rest-to-hold shift) cover the analysis.
5. **Screen and CSV can disagree.** The screen colours use the smoothed angle, the CSV flags use the unsmoothed one, so near the tolerance edge the two can differ for a frame or two. No hysteresis, so the colour can flicker at the edge.
6. **Timestamps are the PC clock when a frame is read**, not the camera's own clock, so they jitter by a few milliseconds. Gaps are measured and flagged; dropped frames inside the camera are not counted.
7. **2D only.** Out-of-plane rotation and camera tilt add error (camera tilt biases pelvic tilt, which is measured against the image horizontal); a pull toward or away from the camera is invisible; the ruler scale is only valid at the ruler's distance; lens distortion is not corrected.
8. **False precision.** Angles display to 0.1 deg, but real accuracy is set by placement and set-up (item 2).
9. **Stepping a practice video backwards** keeps the tracker state, so it can mis-track after a jump. Files only; not recorded.
10. **Marker timing:** a marker attaches to the next processed frame (up to one frame late) plus the operator's reaction time.
11. **Little tried on real hardware:** one phone-as-webcam app has been tried (DroidCam Client with an Android phone on 2026-10-08: it opens through Media Foundation, 1280x720, about 30 fps measured, no warnings). Not tried: Samsung S24/S25/S26 specifically, iPhone, Iriun, Camo, network streams on a real phone, the camera check on a real camera. The rest is tested against stand-ins (a fake camera, a local stream server).
12. **A half-covered dot is flagged, not corrected**, and a dot only about a quarter covered is not flagged; epicondyle/fibular-head swaps are only caught when the knee is not strongly bent.
13. **Force is untried on the real load cell.** It was built and tested with a simulated source; the real DAQ was held by MATLAB when it was written. To check in a first session: wiring, the amplifier's output range, the noise level, the calibration against known weights, and the time alignment (press the MVIC start marker at the start of a push and compare with the force onset).
14. **Not built:** frontal view (hip abduction/adduction), two cameras, lens calibration, numbers-only recording (see PARKED.md), non-Windows support.

Closed in the 2026-10 work (see CHANGELOG.md): silent video-thread errors, unverified camera format and frame rate, tracker thresholds and provenance missing from `meta.json`, no mirror toggle, trackers swapping onto fast neighbours, the Auto colour sticking after a failed click, a stray click re-marking a dot during recording, controls changeable during recording, the keyboard shortcuts dead while typing, empty and half-saved trials, camera dropouts ending the session, a wrong "Participant faces" setting, the mirror-image three-point target, swapped landmark labels, the saved reference depending on the smoothing switch, targets and tests not matching the protocol table, the exposure lock blacking out a webcam.

---

## I. Roadmap (not built)

### Needs you (decisions or hardware)
| Item | Note |
|---|---|
| Real-world check with the planned camera and your stickers | The remaining items in H1 and H11 can only be settled there |
| Validation (SETUP_AND_VALIDATION.md section 4) | Jig and inclinometer; placement repeatability |
| Which number goes to AnyBody | Plateau mean angle or angle at peak force (both reported in Review) |
| Real-world check of the force | Wiring, amplifier range (the SIC-A2 datasheet was not found online: check output type and range), calibration, time alignment; a shared hardware trigger or an LED in view would remove the remaining few milliseconds of uncertainty |

### Parked (see `PARKED.md`)
Pelvic tilt as a neighbour (stays as built), frontal view, two cameras, lens calibration, camera alignment aid, change-since-lock for the main joint, custom dots and angles beyond strap and anchor, ghost pose, the larger audit items (plug-in camera architecture, session-layout redesign, privacy masks, upper-body reuse).

### Other ideas
| Feature | Why |
|---|---|
| AVI container with a repair step | Video that survives a hard crash |
| Per-dot colour override | Only if Auto proves unreliable on real clothing |
| Compare trials, batch summary export | Workflow |
| True 3D angles | Calibrated camera pair; a separate decision |
