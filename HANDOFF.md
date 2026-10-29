# Handoff: limb angle tracker with force (limbtrack)

Written 2026-10-13 for whoever continues this project (a person or another Claude session). It collects what the project is, where it
stands, what was learned from real recordings, what must not be broken, and what is open. It was written from the full session history
(the chat was compacted several times) and from the repository. **Facts here were checked on 2026-10-13; if something looks different in
the code, trust the code and `CHANGELOG.md`.**

Repository: https://github.com/Lass234a/Limb-tracking-from-webcam-phone-livefeed (**public**). Local folder:
`C:\Users\llind\Desktop\Studie\KVT3\Live feed limb tracking project` (Windows 11, Python 3.12 venv in `.venv`).

---

## 1. Read this first (the short version)

1. **What it is:** a desktop app (PySide6 + OpenCV + pyqtgraph) that tracks black or white sticker dots on bone landmarks in a live camera
   picture, shows joint angles and drift alerts to the operator, records video + CSV + `meta.json`, reads a load cell through an NI DAQ at
   1000 samples per second, and has a Review tab that turns a recording into numbers (angle during the hold, peak force, torque, checks).
2. **Why:** maximal isometric contractions (MVIC) at set joint angles for a strength-scaling study; the angles and forces feed AnyBody
   muscle models, so every number must be defensible to an external partner (AnyBody Technology).
3. **State:** version **0.13.0** (2026-10-19: review of the first real session P11; the live picture was 4-11 pictures/s because every force look-up scanned 10 minutes of chunks, fixed; the recording stops by itself 2 s after the MVIC end; Record asks when the force is not zeroed; 'Participant faces' also checked from the knee; readable click prompt; `session_log.csv`; see section 5 'First real session' and CHANGELOG. Full suite 679 passed, 1 skipped, 1 failed (fixed, its file re-run green). **0.9.0 to 0.13.0 were committed locally on 2026-10-19 as one commit, `cae2553` (their changes share files and were never saved apart), pushed later (on `origin/main` with `17cc109` by 2026-10-20); `basic/` stays untracked.** Before: 0.12.0 (ten-step list with the participant first and privacy, ruler, force and markers as steps, redo one dot, try the markers without recording: 2026-10-15, see CHANGELOG; 0.11.0 was the redesigned Live tab and remembered set-up; pause with ghost, sound cues for the MVIC, zero tolerance: 2026-10-15; `SOFTWARE_VERSION` in `limbtrack/engine.py`; tape dots 0.7.2, automatic MVIC markers 0.8.0, camera opened by name 0.8.1, **one iliac crest dot instead of ASIS + PSIS 0.9.0 (2026-10-15)**). 0.7.2 and 0.8.0 are on `origin/main` (pushed 2026-10-14, head `8678bc7`); 0.8.1 is on `origin/main` too (on 2026-10-19 the branch was 1 commit ahead: `cae2553`). Tags `v0.3.0` to `v0.7.0` are pushed; 0.7.1 and later have no tag.
   Full test runs: 0.7.1: 481 passed, 1 skipped; 0.7.2: 496; 0.8.0: 517 (on an idle machine; the 0.7.x runs were green only because the computer was busy, see section 9); 0.8.1: 523 passed, 1 skipped (20 min while a benchmark ran alongside); 0.9.0: 528 passed, 1 skipped, 1 failed (an old trace-curve count, fixed); **0.11.0: 618 passed, 1 skipped, 0 failed** (24.5 min). **0.12.0: 649 passed, 1 skipped, 0 failed** (on the final code; a first full run had one failure, a new test of mine that raced the status timer, fixed).
3b0000000. **Update 2026-10-29 (evening): 0.26.0 = fixes and after-trial checks from the pilot sessions P01 (28 Oct) and P02 (29 Oct) (`FEATURES.md` A36; review `audit/PILOT_P01_P02_review_2026-10-29.md` + `.html`, scripts in `audit/evidence/PILOT_2026-10-29/`).** The pilot data is in `Pilot/` (not `recordings/`; now in `.gitignore`). Fixed: Review/10 % check/export crash on a trial with the pull segment dots off; the 10 % check passing trials without force (negative mean); no warning when the automatic marker never fired; a "not steady (0 N)" note; no after-trial word on a peak cut off at the input limit or beyond the calibration. New after each trial: the angle during the push with the shift from rest and "set about X at rest"; an angle that does not fit the position (more than 15 deg from the earlier trials, or the first trial reading like the previous position); Zero 2 not fitting the force before the push. No measured number changes. New `tests/test_pilot_checks.py` (21). Full suite 1067 passed, 1 skipped, 2 failed (two old tests that pinned the old behaviour and the version string; updated, their files re-run green, 95 passed). Committed and pushed to `origin/main` on the user's word (2026-10-29) as `7c6fba1`. **Open from the review, the user's/group's decisions:** load cell capacity and rig rating (loads to 5,231 N, clipped at 10 V); is the target the angle during the push (knee moves about 19 deg, ankle dorsiflexion 20 deg under load; 36 of 116 trials within 5 deg); the seated hip with the crest line (now three participants); the ruler not in the leg's plane (thigh 35-49 cm across set-ups: torque); a fixed-point anchor; a "do not count" mark for trials; six pilot trials saved under the wrong position (listed in the review).
3b000000. **Update 2026-10-29 (later): 0.25.0 = strap and anchor dots in hip abduction and adduction (`FEATURES.md` A35).** Asked in one line ("we need an anchor and strap marker for abduction and adduction"). The two frontal tests no longer have an empty `optional_dots`, so the Landmarks list offers the two dots (off until ticked). My decision: the pull is measured about the computed hip centre (`"pull_joint": "hip_centre"` in `protocol.json`, new optional test key; `"asis"` switches it), because the ASIS, the fulcrum of the Tabel 3 angle, is not the joint; the SOP has the pivot as undecided (9.9) and both are in every CSV. Review reads the hip centre as a joint (`Trial.derived_tracks`); no stand-in for a missing strap dot in a frontal picture. Side effect fixed: a visit to a frontal test no longer switched the strap/anchor dots on in the next test. New `tests/test_pull_frontal.py`. Not tried on a person; where the strap sits in the front view is not in the SOP. Committed and pushed to `origin/main` the same day (2026-10-29), on the user's word, as `09e2ddb`, before the full suite had finished; the full suite on that code then ran green: 1048 passed, 1 skipped, 0 failed.
3b00000. **Update 2026-10-29: 0.24.0 = new step order, soft reminders, ruler button, foot dots only in the ankle tests (`FEATURES.md` A34).** Order Camera, Force, Markers, Participant, Position, Privacy, Ruler, Dots, Lock, Record; *Participant faces* is in the Participant section; after a position change rows with buttons for privacy (No face / Mask / Crop + a banner on the live picture), ruler (Fine / Redo, key C, a pinned *Ruler (C)* button) and the resting force (Zero 2, X) - none of them refuses Record; the privacy refusal of 0.23.0 and the Yes/No questions for lock, main angle and zero are gone (written to the session log and `meta.json` `reminders_open_at_start`; the file-changing ones repeat in the trial warnings); the SOP's "3 s at rest" notes are gone from Review and the trial warnings; knee and hip tests have 5 dots (`protocol.json`, `ankle` out of `angles`). Foot and belt-squat analysis in post: no change needed. Committed and pushed as `0f14cef` (2026-10-29, full suite 1027 passed, 1 skipped); not tried on the real rig, the real load cell or real beeps.
3b0000. **Update 2026-10-28 (later): 0.23.0 = Redo all and the dot keys remove the old dots; a privacy check at every change of position (`FEATURES.md` A33).** D removes every dot and the lock at once (reverses the 0.18.1 choice); a dot's button/number key removes that dot at once (lock kept); a new test or target clears the 'I looked' tick, shows *Privacy: recheck* and refuses recording until the picture is looked at again; `meta.json` `privacy.checked_for_this_position`. Not committed; not tried on the rig.
3b000. **Update 2026-10-28: 0.22.0 = a second zero level (`FEATURES.md` A32).** Button *Zero 2: relaxed in the strap (X)*: 2 s on top of the zero, for one participant/test/target, cleared by a new zero; the newtons stay the total force from the first zero; *above relaxed* live, `meta.json` `force.relaxed`, Review lines and two new export columns. Not committed (the user commits on request); not tried with the real load cell. Open: when in the SOP to press it (proposal: after the participant is in position, before Record).
3b00. **Update 2026-10-27: 0.21.0 = 'recording first' (on by default).** A lower load cell rate would not help the frame rate (force work about 2 ms per picture at 1000 Hz). The window's trace, status panel and picture share Python with the tracking (25 ms of work per picture without the window, 42-61 ms with it, P11 replay on battery). While recording the screen now shows at most 10 pictures a second and the trace and panel refresh once a second; every picture is still tracked and saved. P11 replay: 23 -> 28 pictures/s on battery, 13 -> 27 with 16 busy programs alongside. Windows priority above normal was tried and left out (no gain). `meta.json` `pipeline_timing.recording_first`. Replay tools (real window + stand-in phone stream): `audit/evidence/SPEED_2026-10-27/`. Not tried on the real DroidCam + DAQ: read `pipeline_timing` of the next real trial. Not committed (the user commits on request).
3b0. **Update 2026-10-26: 0.20.0 = compatibility with the protocol draft 2026-10-20 / SOP v0.18 (review `audit/SOP_vs_APP_2026-10-26.md`).** Resting force = median 3.0-0.5 s before the push (`Trial.force_rest`; the old 1 s before the half-peak window held half the ramp: peak above rest 6-16 % low on P11), with notes for a short or unsteady rest; the 10 % check after each trial on a helper thread (`review.after_trial_check`, `analysis.position_check`, total peak, plan `3rd!`); the tested leg per participant code (`Engine.tested_leg`/`tested_legs`, Participant section, Record asks, saved in every meta.json); protocol.json in the SOP order, tolerance 5, hip adduction judged on abduction at 0/20, belt squat 90/60; posture tick "Hip against the bench". `FEATURES.md` A31. SOP text and technical description for the SOP: `SOP_limbtrack.tex`. Group decisions on 2026-10-26: adduction at 20 deg abduction, belt squat 90/60, 10 % rule on the total peak. Resting force vs AnyBody: subtract the rest XOR model the segment's gravity, never both (SOP_limbtrack.tex, force section). File names now carry the position number (SOP test order 01-21, `protocol.plan_positions`) and the trial number: `P01_03_knee_extension_110deg_T2_<time>_raw.mp4`; old names are still counted (`recorder.trial_metas`). Not committed (the user commits on request).
3a. **Update 2026-10-23 (later): 0.19.0 = checks from the review of the P20 session (`audit/P20_session_review_2026-10-23.md`, 22 trials of 2026-10-22): dots on one bone that move apart (`protocol.json` `rigid_pairs`, 25 %) mark the angle CHECK DOTS, refuse Lock and make `all_ok` 0; `label_check` / `dots_apart` CSV columns and `meta.json` `checks`; non-blocking *Check this trial* message and session-log line after a trial; force `peak_n` from a 20 ms average (`peak_raw_n` kept) and `force.rise_n` / `no_rise_warning` (60 N); side-lying frontal tests (hip centre "down" towards the knee, rotation-proof side check); Record asks when the main angle cannot be measured; keys do not repeat while held (except Left/Right). `FEATURES.md` A30. Full suite 943 tests, none failed (in two parts, 2026-10-23/26). The foot plate for the ankle (review option 1) is NOT built: not decided. Committed locally (2026-10-26) as its own commit, after 0.18.2 (`bf8cc78`); not pushed (main is 4 commits ahead of origin: 0.18.0, 0.18.1, 0.18.2, 0.19.0).** 3a''. **Update 2026-10-23: 0.18.2 = review of the 0.18.x tests and speed: ruler/dot click hand-over fixed (only one thing waits for a click), tests made sharper by breaking the code on purpose (25 damages, all noticed), `tests/test_mark_again_live.py` (real video, real recording), speed measured against 0.17.2 (no measurable cost; CHANGELOG 0.18.2); local commit `bf8cc78` (2026-10-26, from the patch kept before 0.19.0 was started), not pushed.** 3a'. **Update 2026-10-22 (later): 0.18.1 = Redo all marks every dot again without clearing first (old dots keep tracking until replaced, lock released at the first dot really re-marked, skip/jump, `Engine.start_remarking`, `mark_dot(replacing=...)`; `FEATURES.md` A29; local, not pushed).** Before it, 0.18.0 = pinned bar (position dropdown with previous/next, dot buttons, Redo all) and the keys D/F4, 1-9, N/F3, B/F2, in `gui.py` (`quick_bar`); UI only, no measured number changes; `FEATURES.md` A28; local commit, not pushed.**
3b. **Update 2026-10-20/21: versions 0.14.0 to 0.17.2 (six commits after `17cc109`), pushed to `origin/main` on 2026-10-21 (head `6326238`); `basic/` still untracked.** Details in `CHANGELOG.md` and `FEATURES.md` A24-A27.
   - **0.14.0 Review layout:** graphs top right, number tabs in a wide strip below, controls that act on the graphs moved under them, export buttons in the top bar, both dividers draggable and remembered (`settings.json` key `review_ui`); the yellow position line (it was black on black) and light-grey marker lines.
   - **0.15.0 Frontal hip abduction/adduction (camera moved to the front, no second camera):** tests `hip_abduction` / `hip_adduction` (0 and 20 deg; the plan is now the full 19 positions), dots `asis`, `asis_opposite`, `patella`; *Tested leg* instead of *Participant faces*; two angles per test: the Tabel 3 **goniometer angle** (the main joint, judged green/red) and the **angle about a computed hip centre** (shown and saved, `"judged": false`). The hip centre is a **computed point** (`derived_points` in `protocol.json`; `geometry.hip_centre_frontal`; saved as `hip_centre_x/_y`): Harrington et al. 2007 set 1 from the two ASIS and the ruler scale (needs the ruler). **The constants were typed from a secondary source (a biomechanics wiki via a search summary) and the original table was never read: check it before any number is reported.** **The two angles do not share a zero** (synthetic: the goniometer reads -4.1, 0, 4.1, 12.4 deg where the thigh is 0, 5, 10, 20 deg out about the hip centre). Which one AnyBody wants, and where AnyBody puts zero hip abduction, is an open question for AnyBody Technology. Sagittal posture in these tests is controlled with a wall (heel, buttocks, back) and not measured (a tick is saved). No torque from the front, no strap/anchor dots in these tests. **Never tried on a person.**
   - **0.16.0 Reference readings:** the Live tab section *Reference reading* (G = reading taken now) keeps the app's angles at the moment a goniometer/inclinometer is read; the typed readings go into `meta.json` `reference_readings` (never for another position); Review has a *Reference* tab, a side file `<trial>_reference.json` for later edits (`meta.json` never rewritten), and *Reference validation...* (agreement over the recordings folder, plot, CSV). This is the tool for SETUP_AND_VALIDATION 4C. Nothing has been read from a real instrument yet.
   - **0.17.0 Force against angle** in Review (switch above the graphs).
   - Full suite on the final code of 0.17.0: **829 passed, 1 skipped, 0 failed** (17.4 min). Looked at in real windows on synthetic trials only.
   - **0.17.1 (2026-10-21, commit `a38aaa6`): fixes from an Opus code review of 0.14.0-0.17.0, all 15 findings.** Reference readings now follow the position and the participant (`engine.participant_code`; `entered_for.participant`; typed before a code they go with the first participant recorded), Record hands over what is typed in the boxes (F5), Review's side file holds only real changes, `typed_at` keeps one typed reading one pair; unjudged angles never block Lock or make `all_ok` 0; tape width and wall tick are cleared for another participant; the stamp needs a picture under 1 s old; Review save errors are shown; frontal label checks take dot names from the protocol; more protocol mistakes refused; force-angle window dots in force time; less redraw work. Found on the way and fixed: Review's force strip used the default smoothing on load; two trials started in the same second overwrote each other (now `-2`, `-3`). Details in CHANGELOG 0.17.1. **Full suite on the final code: 858 passed, 1 skipped, 0 failed** (26.2 min). Not looked at in a real window.
   - **0.17.2 (2026-10-21, commit `6326238`): code clean-up of 0.14.0-0.17.1** (reuse, simplification, efficiency, altitude; no measured number changes). The reading-position rule lives in `ReferenceReadings.mismatch/claim`, the pelvis tape/wall ownership in the engine (`_PelvisFact`, `participant_code` property), one per-angle reset, `geometry.pelvis_axes`, `TestDef.frontal`, `Trial.main_angle`; the force-against-angle view keeps its window items and uses a 64-brush palette. Details in CHANGELOG 0.17.2. **Full suite: 859 passed, 1 skipped, 0 failed** (18.1 min).
4. **Not yet validated:** the angles have never been compared with a goniometer/inclinometer; no real session with the final camera,
   lighting and stickers has been run. Everything so far is synthetic tests plus a few bench recordings (board with black magnets, one leg
   session P12). Treat angles as positioning data until `SETUP_AND_VALIDATION.md` section 4 is done.
5. **The user is not a programmer.** Explain in plain English, say whether any measured number can change, and keep a technical trail
   (CHANGELOG, FEATURES) separate from the plain explanation.
6. **Never commit video of a person, `recordings/`, `audit/`, PDFs, `settings.json`.** The repository is public (`.gitignore` blocks these).
7. **Raw data is never smoothed or corrected.** Volts and dot positions are saved exactly as measured; smoothing and calibration are
   applied afterwards, in Review, and recorded.
8. **Biggest practical risks right now:** (a) strap dot not visible on the strap, (b) mains hum on the force unless the laptop charges from
   the isolated outlet, (c) live frame rate (17-20 fps, camera can do 25-30), (d) MVIC markers pressed 4-5 s before the force really rises (answered in 0.8.0 by the automatic markers, which are untried on the real load cell),
   (e) only one program can hold the DAQ, and a phone stream allows only one viewer.

---

## 2. Study context (from the early sessions)

- Study: strength scaling of AnyBody muscle models from measured MVIC at specific joint positions (knee ext/flex, hip ext/flex, ankle
  plantar/dorsiflexion, belt squat). The protocol draft is `KVT3_protokol.pdf` (Tabel 2 = positions and required neighbour angles,
  Tabel 3 = goniometer landmarks). The SOP skeleton is `C:\Users\llind\Desktop\Studie\ChenData\SOP\sop_strength_scaling.tex`; strength
  curve scripts are in `...\ChenData\StrengthCurve\*.m` (`plot_proposal.m` was the source of the first default target angles).
- Design discussions (Danish/English) about number of trials, hold length, active insufficiency and validation exercises happened in an
  earlier session (id `94442a43`); they are background, not requirements for the code. Check the SOP for what was actually decided.
- An AnyBody seated knee-extension MVIC model (206 cm / 106 kg) exists at `C:\Users\llind\Documents\AnyBody.8.2.x\SeatedKneeExtMVIC`
  (session `55bb8a41`; see its README). Open: which number goes into AnyBody, the **plateau mean angle** or the **angle at peak force**
  (Review reports both; the decision is the user's).
- Protocol in the app (`protocol.json`, checked at start-up): knee_extension [20, 70, 110] with hip 90; knee_flexion [20, 90] hip 90;
  hip_extension_k0 [0, 45] knee 0; hip_extension_k90 [0, 90] knee 90; hip_flexion [0, 90] knee 90; ankle_plantarflexion and
  ankle_dorsiflexion [0, -20] knee 0 (Tabel 2 used over the 3-position text in section 2.7); belt_squat has no targets yet. The session
  plan shows 15 joint positions + one belt-squat row; the protocol text counts 19 positions (belt-squat positions are not entered; **check**).
- Operators: one person at the laptop, one positioning the participant. Participant codes `P01`..`P999`; consent for video is covered in
  the informed consent sheet; storage/retention of raw video is **undecided** (AAU's GDPR contact should confirm).

---

## 3. How to run, test, and find things

| Task | How |
|---|---|
| Start the app | double-click `Start.bat` (or `.venv\Scripts\python.exe -m limbtrack`). Frontal tests: pick *Hip abduction* or *Hip adduction* (they need the ruler) |
| Full test suite (8-17 min) | double-click `Run_tests.bat` (writes `tests_last_run.txt`) or `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` |
| Check a phone camera | double-click `Check_phone_camera.bat` (reads only; writes `phone_camera_check.txt`) |
| Try without hardware | Live tab: *Open video file* `demo/demo_leg_white_dots.mp4`; Force box: source *Simulated* |
| Add a load cell calibration | `LoadCellCalibration/logs_to_points.py FOLDER` then `add_calibration.py POINTS.csv --id NAME --activate` |
| Re-check the protocol file | `.venv\Scripts\python.exe -m limbtrack.protocol check` |

Documents (all in the project root, written for the user as well): `README.md` (usage), `FEATURES.md` (every function, what it does to
the numbers, CSV/meta contents, section H = what is unreliable, section I = roadmap), `CHANGELOG.md` (per version, with measured
effects), `PARKED.md` (decided-not-to-build list), `SETUP_AND_VALIDATION.md` (set-up rules, validation plan, force/hum/strap notes), `MARKERS.md` (how to make and place the tape dots, what was tested), `PELVIS_LANDMARK.md` (the argument, with PubMed-checked sources, for the iliac crest dot as the hip angle's pelvis landmark, its limits, and three checks).
`audit/` (not in git) holds the 2026-10 Opus audit (`AUDIT.md` technical, `AUDIT_overview.html` plain, `findings.json`).

### Code map (`limbtrack/`)
`tracker.py` (dot detection/tracking, all constants + `settings()` into meta) | `engine.py` (pipeline, judging against targets, lock,
recording, privacy, force hand-off) | `recorder.py` (CSV row by row, two encoder threads, atomic `meta.json`) | `camera.py` (webcam via
DSHOW then MSMF, streams, 20 s open time-out) | `gui.py` (Live tab) | `review.py` (Review tab, `Trial`) | `analysis.py` (checks, force
timing, torque, per-participant summary) | `force.py` (DAQ source, calibration, smoothing, hum) | `loadcell.py` + `load_cell_calibrations.json`
(regression registry, precision) | `scale.py` (ruler) | `pull.py` (strap/anchor pull angle, moment arm) | `privacy.py` | `protocol.py` |
`automvic.py` (automatic MVIC start/end markers from the force) | `cues.py` (sound cues for an MVIC) | `panels.py` (strip of step chips, sections that open one at a time) | `session_memory.py` (what `settings.json` remembers) | `reference.py` (typed goniometer/inclinometer readings: parsing, side file, comparison, agreement; 0.16.0) | `geometry.py` (also the computed hip centre, 0.15.0), `angles.py`, `filters.py`, `displacement.py`, `provenance.py`, `diagnose.py`, `overlay.py`, `synthetic.py`.
Tests: 52 test files in `tests/` (49 before the 2026-10-13 work) (synthetic leg generator `synthetic.py` makes exact-angle pictures).

---

## 4. Invariants (do not break; each has tests)

- **Tracking numbers are pinned.** All tracker numbers are named constants in `tracker.py` and are copied into every `meta.json`.
  `tests/golden_tracking.npz` pins the synthetic output; regenerate only for a deliberate change with `python tests/test_tracker_settings.py`
  and say why in the commit (the numbers in people's recordings change).
- **Angles** are two-arm goniometer definitions in `protocol.json` (sign, offset, signed knee: hyperextension negative). The display
  smoothing (0.2 s moving average) changes only what is shown; CSV angles and flags are unsmoothed. Neighbour angles are judged from
  the start when Tabel 2 gives a required value, otherwise against the unsmoothed 0.2 s average taken at Lock.
- **Lost-dot rule:** a lost dot freezes at its last good position, searches only a fixed circle (2.0 x diameter), and resumes only on a
  round, right-sized, contrasty blob. Angles are never computed from a lost dot. No interpolation anywhere.
- **Settings are frozen while recording** (`_FrozenWhileRecording` descriptors) and a start-of-recording snapshot goes into `meta.json`
  (`changed_during_recording` should always be absent). Clicks on the video are ignored while recording.
- **Recording safety:** CSV flushed per row, a complete `meta.json` written at the start (status `recording`) and replaced atomically at
  the end; video encoding on its own threads, frames never dropped (bounded queue, caller waits); picture size change stops and saves.
- **Force:** the **volts are the permanent record** (`force.csv` has every sample; `force_v` per picture). Newtons are a second channel,
  computed with the active calibration and the session zero. Smoothing is display/analysis only (default 100 ms; a multiple of 20 ms
  removes 50 Hz exactly). Force and pictures share `time.perf_counter()`; the DAQ clock is aligned with `ClockAligner`.
- **Privacy crop/masks** only change the saved video copies; tracking and CSV use the whole picture (CSV in full-picture pixels; Review
  pads a cropped video back).
- **Ruler dots are fixed points** (not tracked); a moved camera/ruler after marking is not noticed: mark again.
- **Provenance** in every trial: code hash, git commit, library versions, protocol hash, tracker settings, camera settings, force
  setup, privacy, pipeline timing.

---

## 5. Findings from real recordings (what we learned)

### Tracking (board tests with black magnets, 2026-10-09; recordings `P01`-`P10`)
- Dots were lost when a **finger or its shadow touched/merged with the dot**: the blob became big/elongated, the centre was dragged toward
  the finger, the learned dot area ballooned (383 -> 821 px), and the dot was then lost. Fixed in 0.5.0: search again at higher
  contrast thresholds when no blob fits or the blob is merged; never learn area from a merged blob; area floor 0.40 (small fragments of a
  hidden dot are not accepted; a half-covered dot at about 0.5 is still tracked and flagged). Checked on pictures of the real videos;
  Bundepus reference moves at most 0.27 px / 0.02 deg in 40 of 312 frames; synthetic reference identical.
- Genuine occlusion by a hand is still lost (correct). **Bright, even light** is the real remedy for shadows (a lamp is planned).
- Dots on P12: malleolus 1.6 cm and trochanter 1.98 cm were below the 2 cm minimum (the dot-size estimate uses a 1.3 px edge allowance; verify with real stickers of known size).
- **Strap dot** (second force trial) was lost in 95 % of pictures: it sat on the ankle buckle/sock where no dot is visible. Use a bright
  sticker (>= 2 cm) on the flat, camera-facing part of the strap, a few cm from the buckle. The **anchor dot** was steady (0.3 px).

### Markers: tape dots (decided 2026-10-13, synthetic tests only; see `MARKERS.md`)
- **Why:** black sticker dots could not be sourced. The planned dot is a **solid black disc (about 2.2 cm) drawn by hand on a square of white matt sports tape**, with *Dot colour = Black dots*. The user also floated: tape strip from ASIS to PSIS, a black line along a taped strap, tape patches at the strap/anchor. Decision after testing: dots (a strip with a disc at each end is fine), **no line tracker** (it would not improve accuracy: tracker scatter 0.2 mm is under 0.1 deg on a 15 cm ASIS-PSIS line, a 5 mm placement error is about 2 deg), revisit only if the first real session loses the ASIS/PSIS dots from the side.
- **(Superseded by 0.9.0, see the next bullet.) Side view of ASIS/PSIS:** these bone points face forward/backward, so a disc on them is seen edge-on. The camera only needs the point's position in the picture, which does not change when the disc is moved sideways towards the camera: put the disc on the outer surface of the hip level with the bone point (palpate, hold a ruler horizontally at the point towards the camera, mark where it meets the skin). Side-on discs (50-70 % width) tracked as well as round ones.
- **Silent failures found (all avoided by the rules in `MARKERS.md`):** a thin *outline* circle on tape it is not centred on with Auto colour in a soft picture was tracked on the white tape, 11 mm off, nothing lost, no warning; an outline with a gap shifts the centre 1.3-3 mm; a thick (4 mm) marker line joining two discs merges them into one blob (marked up to 1.7 cm off, or lost). Ideas to harden this in the program (not built): a roundness/size check at marking (refuse or warn on an elongated or much larger blob), a smarter Auto that prefers a dark disc inside a white blob.
- **Dot-size check was wrong by about 1 px** (the 1.3 px allowance came from OpenCV drawing the synthetic dots about 1 px too big); corrected in 0.7.2. Needs one check with a real disc of known size.
- **Pelvis decision (2026-10-15, version 0.9.0, the user's):** ASIS and PSIS are replaced by **one dot on the iliac crest halfway between them** (`iliac_crest`). **The hip angle's definition changed:** the pelvis arm is now the vector iliac crest dot -> trochanter (the goniometer's lateral midline of the pelvis) instead of the perpendicular to ASIS-PSIS; hip angles from 0.9.0 on are not comparable with earlier ones, and the supervisor and the AnyBody side should be told. The **pelvic tilt angle was removed** (needs two dots). The Participant-faces check now reads only the foot dots. `tracker.py` is unchanged. Old trials (with `asis`, `psis`, `pelvic_tilt`) still open in Review; checked with a trial recorded by 0.8.1. All verification is synthetic; the crest dot has never been tried on a person.
- **Not known:** whether the iliac crest dot (before 0.9.0: ASIS/PSIS) stays visible in the real hip-extension posture (the protocol draft says "(lying??)" for the extensors; face down would hide the ASIS), real glare/creasing/sweat, tape and ink sliding on skin.

### First real session: P11, 2026-10-15 (11 trials, hip extension and flexion; reviewed 2026-10-19, version 0.13.0)
- **Frame rate:** the 9 trials with the automatic markers ran at 4.5-11.3 pictures/s (77-200 ms of own work per picture), the two without at 20 and 28. Cause: `ForceSource.samples_between` scanned all 60,000 stored chunks (13 ms per call after 10 minutes connected), several calls per picture. Fixed in 0.13.0; a window replay (P11 video as a live MJPEG stream, simulated force with 600 s history, auto markers, recording) went from 17-19 to 29.5 pictures/s. The real session was slower than the old-code replay, so check `pipeline_timing` of the next real trial. The saved mp4s of those trials are tagged 22-30 fps but hold 4-11 pictures per second (they play too fast outside Review).
- **Hip flexion 90 deg sitting reads 37-42 deg** with the crest -> trochanter pelvis line (the pelvis tips back when sitting); all three trials 0 % in target. Affects every *hip 90* neighbour requirement too (knee extension/flexion). **Needs a decision** (target in pelvis terms, or another reference): not changed in code. Hip extension knee 0 at 45 deg read 37.7.
- **Facing:** three trials were recorded with the wrong *Participant faces*: the two face-down hip extension knee-90 trials (13:38, 13:41; knee -55 / -78 deg) and the standing hip extension knee-0 trial at 12:00 (facing left, knee -12, hip -11). Their knee and hip values have the wrong sign (the offsets are 0, so a sign change gives the right value). The knee check of 0.13.0 catches the first two, not the third (nearly straight knee); only the foot dots or the operator's eye can.
- **No newtons in 4 trials** (12:00 and the three hip flexion 0 deg at 14:47, after a restart): not zeroed. Their volts are complete; a zero from the same rig (14:14 zero: 0.03185 V) could give newtons afterwards, if the rig did not change (the user decides).
- **Dots:** colour *Auto* in 3 trials (12:00, 12:50, 13:38). Real dot losses: malleolus when the free shank swung (hip flexion 0, 90 of 144 pictures), epicondyle on and off next to the strap dot (hip flexion 90, 137 of 305), most others after the MVIC end (participant moving or walking away: hence the auto stop). Knee 50-72 deg in the hip flexion 0 trials (required 90). Ruler not marked in 2 trials. Distances along the leg are steady across positions (thigh 201-231 px, shank 176-237, crest-trochanter 66-82).
- **Automatic re-finding of dots tried and dropped:** see CHANGELOG 0.13.0 (1 in 6 wrong; 1 in 20 with two dots clicked). The module and its evaluation are not in the repo.
- **Click tolerance measured:** a click marks a real tape dot from up to about 12 px (image pixels) away (`MARKING_MAX_SNAP_PX` 15).

### Cameras
- DroidCam: the virtual camera "DroidCam Video" cannot be opened by index through DirectShow; it works through Media Foundation (the
  app falls back automatically: 1280x720, ~30 fps). Windows **camera index order changes between restarts** (select by name).
- The PC client and the app cannot both read the phone stream: **the phone allows one viewer**. What worked on 2026-10-09: close the DroidCam
  Client and let the app read `http://PHONE-IP:4747/video` (a bare address without `/video` is a web page; the app adds `/video`).
  The phone was on the Windows hotspot network (192.168.137.x). The newer DroidCam Client looks OBS-like and needs its virtual camera started.
- **New DroidCam Client over USB (2026-10-14, Samsung SM-S942B):** add the phone in the client (Add > DroidCam, Refresh Device List, accept USB debugging on the phone; the new client has no separate 'start virtual camera' step; an earlier note saying so was wrong). **DirectShow and Media Foundation number the cameras in opposite order** (DirectShow `DroidCam Video, Integrated Camera`; Media Foundation `Integrated Camera, DroidCam Video`). Until 0.8.1 the fallback opened Media Foundation under the same number and so opened the laptop webcam (slowly: the 20 s time-out was blamed on the phone). Fixed in 0.8.1 by matching the camera by name (`msmf_camera_names`). Lesson: when a camera 'hangs', check WHICH camera was opened before tuning OpenCV settings. A hung camera open also stays inside the program (the helper thread cannot be cancelled): close the program completely and start it again.
- A camera that never answers is now reported after 20 s with advice (a hang happened with the client running but not connected).
- **1080p (benchmark 2026-10-14, this laptop, 16 logical cores, synthetic pictures, `scratchpad` scripts not kept):** tracking + drawing 9-11 ms per picture at 720p and 16-18 ms at 1080p (fine for 30 pictures per second, tracker accuracy unchanged: 0.07 px, constants are in pixels but work for 21 px dots). **The two mp4v video encoders (one core each) are the limit: about 60 pictures/s at 720p, 21-25 at 1080p**, so at 30 pictures/s the 90-picture queue fills in 15 s and the program (which never drops pictures) starts to wait (late by up to 2 s); video files are 2.3x bigger and the queues can hold 1.1 GB. Real video is noisier than synthetic and will encode slower. The program always asks the camera for 1280x720 (`CameraSource` default). Options if 1080p is wanted: save the video at 720p while tracking at 1080p (Review must then scale the positions), a faster codec (MJPG), or both; none built.
- **Frame rate:** first real stream recordings ran at 11 and 8.5 fps with the program's own work at 84-112 ms per picture (`pipeline_timing`
  in `meta.json` shows this; `process_share_of_interval` near 1 = the PC was the limit). Reproduced locally (stream + 7 dots + force + window)
  and cut to 17 fps (20 without the live trace) in 0.6.0. Camera is 25-30 fps. Next lever if needed: track on a down-scaled picture.
  Earlier webcam/DroidCam-client tests ran at 17-26 fps while the camera delivered 30 (dim light also lowers a phone's rate).

### Force (first two P12 trials, uncalibrated, then the calibration sessions)
- **50 Hz mains hum**: ~90 mV on a ~1 V signal (9-18 % of the rise); the raw single-sample peak was 8-11 % above the 100 ms-smoothed peak.
  On 13 Oct the hum was **removed by powering through the lab's isolated outlet** and then shown to depend on the **laptop (DAQ ground
  reference) being on its charger from that same outlet**: A/B at 75 kg, charger on 1.3 mV vs charger off 53 mV (~40x). Standard condition:
  laptop charging from the isolated outlet. Amplifier 0 V to AI GND wiring is **still unconfirmed**.
- **MVIC markers were pressed 4-5 s before the force rose** in both trials; a window from the markers gave a plateau of 0.75 V instead of
  0.96 V. Use *Window = force above half of peak* (the Check tab flags it).
- Two trials of the same position (65 deg) differed by **30 %** in peak (1.013 V vs 0.751 V): the protocol's 10 % rule means a third trial.
  The knee fell ~5.8 deg from rest to hold in both (strap slack/stretch): holds landed at 58 and 61 deg against a target of 65.
  Knee angle followed force with correlation 0.93-0.97, but the best-fit lag was broad (0.1-0.3 s) and differed between trials, so it
  cannot replace the **tap test** for the picture delay.
- **Picture delay** (how late a picture is stamped compared with when it was taken) is unknown; typical 0.1-0.3 s. Review has a *Picture
  delay* box (default 0, saved in exports). Measure with a sharp tap on the sensor/strap in view (`SETUP_AND_VALIDATION.md`).
- **Load cell calibration** (separate session `9c32a976`; data in `C:\Users\llind\Desktop\Studie\KVT3\Kalibrerings data\`, analysis in
  `LoadCellCalibration/`, report HTML there): TAS501 cell, amplifier SIC-A2 powered from 24 V (cell excitation from the amplifier), output
  in volts (~4.7 V at 250 kg), NI USB-6003 differential `ai0`. **Final fit 13 Oct (0-250 kg up and down, 24 runs): V = 0.08703 + 18.610 mV/kg
  x plates -> 526.95 N/V (53.734 kg/V), R2 0.999996, linear to 250 kg.** Descending runs read +0.29 kg above ascending; chain + strap
  weigh ~1.1 kg (zero with them hanging before every trial). Day-1 (12 Oct, 0-60 kg, 528.35 N/V, different rig) is not pooled. Three data files are mislabelled
  (see the memory note `project_force-calibration.md`). The regression is stored as id `2026-10-13` in `limbtrack/load_cell_calibrations.json` and is active.
- **Open calibration items:** weigh the plates and chain (real masses), **the cell's rated capacity (the TAS501 comes as 200 kg but 250 kg
  was hung: confirm)**, safety rating of the ceiling rail/carabiner/strap/chain, a 100-250 kg check with an independent weight, a long no-load drift run,
  a repeat zero at the end of a session, room temperature logging.

---

## 6. Force workflow as built (0.7.1, automatic markers added in 0.8.0)

1. Close MATLAB/NI MAX/anything holding the DAQ (the first attempt failed with "resource reserved" because MATLAB held `Dev1`).
2. Live tab -> Force box -> source *NI DAQ*, `Dev1`, `ai0`, differential -> *Connect*.
3. **Zero** (button or Z) with the rig hanging and nothing loaded; repeat whenever the set-up changes. **The zero is not kept when the program
   closes** (a new start shows volts only until zeroed); the zero used is saved in each trial. The active regression gives newtons.
4. Record. **Optionally tick *Mark the MVIC automatically from the force* first (Markers and sound section; *Try it without recording* tests the chain without a trial): Record arms it, the start marker is written when the force leaves its rest level (about 0.15-0.2 s after the ramp began) and the end marker 7 s later; S / E by hand take over.** Force appears in the picture header (N and V) and in `force.csv` / `force_v`, `force_n`. A DAQ that stops, a signal at the +-10 V
   limit, or no samples for 0.5 s is flagged in the picture and `meta.json`. Force needs a **live camera** (not a video file).
5. Review: Check / Angles / Force / Calibration / Dots-and-pull tabs, force strip under the angle plot, *Window = force above half of
   peak*, smoothing box, hum warning, *Recalculate newtons*, precision of the peak from the regression (warns beyond the heaviest calibration load),
   *Export all trials of this participant* (one row per trial + the 10 % repeatability flag).

Torque (N.m) = smoothed force x moment arm: needs calibrated force + ruler scale + anchor dot; when the strap dot is not seen the ankle dot
stands in (reported). It has only been exercised on synthetic geometry so far.

---

## 7. Open items, in a suggested order

**Before real data collection (user/hardware):**
1. Real-world session with the final camera, bright lamp, **tape dots made as in `MARKERS.md`** (filled disc about 2.2 cm, Dot colour = Black), ruler and anchor/strap dots; run the Check tab on it. First things to look at: does the iliac crest disc stay tracked when the hip moves, and does the size read-out agree with a disc of known size.
2. Validation against a jig + inclinometer, and sticker placement repeatability (`SETUP_AND_VALIDATION.md` section 4).
3. Confirm the amplifier-to-DAQ ground wiring and the isolated-outlet/charger routine; confirm the cell's rated capacity and the rig's safety rating.
4. Tap test to measure the picture delay for the camera chain that will be used (repeat if phone/app/cable/Wi-Fi changes).
5. Decide: plateau mean angle vs angle at peak force for AnyBody; trials per position (`TRIALS_PER_POSITION = 2` in `gui.py`; the session
   plan counts against it); storage/retention of raw video; whether numbers-only recording is needed if video is not permitted.

**From the P11 review (2026-10-19):** decide the hip 90 target with the crest pelvis line (see section 5); look at `session_log.csv` after the next session to see where set-up time goes; check `pipeline_timing.loop_fps` of the next real trial (should be near 30); the four unzeroed P11 trials and the two mirrored face-down trials need a decision before analysis.
**Software ideas the user was offered (none started):** force-versus-angle plot (slack/stiffness); *(automatic MVIC markers from the force: done in 0.8.0)*;
an automatic picture-delay estimator from a tap; a faster live loop (down-scaled tracking); a check-weight step in the calibration flow.
**From the tape-dot work (2026-10-13, none started):** a roundness/size check when a dot is marked (an elongated or much larger blob is
refused or warned: catches a merged line, the tape instead of the disc); a smarter *Auto* colour that prefers a dark disc inside a white blob;
a line (stripe-on-tape) marker, only if the first real session loses the iliac crest dot from the side.
**Never done:** a final multi-agent Opus re-audit of 0.3-0.7 (TRK-01 neighbour protection and the recording rework were not Opus-reviewed).
**Open since 2026-10-20 (decisions for the user and for AnyBody Technology, none blocks the software):** (1) check the hip-centre constants and the coordinate origin in the original Harrington paper; (2) ask AnyBody what zero and definition they use for hip abduction and which pelvis frame (this decides how the goniometer angle and the hip-centre angle map onto the model); (3) the hip 90 target with the crest pelvis line (P11 read 37-42 deg at a nominal 90) is still open: one reading with the new reference section would show whether the dot placement, the definition or the set-up is off; (4) tape or calipers for the ASIS-to-ASIS width at the sessions (a cross-check of the ruler, optional); (5) torque for the frontal tests needs a moment-arm decision with AnyBody.
**Parked (see `PARKED.md`):** two cameras at once, lens calibration, camera alignment aid, change-since-lock for the main
joint, ghost pose (commented out, `# PARKED-GHOST:`), numbers-only mode, plug-in camera architecture, AVI container with repair.

---

## 8. Working agreements with the user (observed over many sessions)

- Commit locally after each finished step; **trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`**. The user pushes
  manually or asks ("push" on 2026-10-12 pushed 40 commits + tags). Do not push on your own initiative. The user prefers a deep version history over destructive edits.
- No `CLAUDE.md` exists; the user said to ignore its absence.
- Plain-English reports; always say what could change a measured number and what could not be tested. Typical prompts are "continue" / "next".
- **Ask before probing the user's network, processes or devices.** On 2026-10-09 the user rejected a PowerShell probe of processes/network;
  a self-run script (`Check_phone_camera.bat`) was the accepted alternative. Camera-open probes were tolerated.
- For audits/reviews the user wants two outputs: a technical `.md` (for a model to implement fixes) and a separate plain-language HTML.
- Heavy workflows hit the usage limit in the audit; keep work in small batches.

---

## 9. Notes for the next agent (methods and quirks)

- **Tests that depend on how fast the computer is:** the helper `humming_trial` (`tests/test_force.py`) used to keep the picture times of a real 3 s recording (about 7.5 s long on an idle machine, longer on a busy one) while the torque and picture-delay tests in `tests/test_analysis.py` look at 8 to 13 s, so they failed on an idle machine and passed on a busy one. Fixed in 0.8.0 (the pictures now get a time axis that spans the injected signal). If a test passes only 'when the machine is busy', look for a real-time recording whose length is assumed. Run the full suite on an idle machine at least once before trusting a green result.
- **Automatic MVIC markers (0.8.0):** pure detection in `limbtrack/automvic.py` (unit tests on made-up traces, `tests/test_automvic.py`); the engine glue is `_arm_auto_mvic` / `_auto_mvic_step` / `add_event` in `engine.py` and is tested in real time with `SimulatedForce` (`tests/test_automvic_engine.py`). Starting a recording takes a variable time (video writers), so those tests read the true onset from the saved `force.csv` instead of assuming when the ramp comes. The markers use the same labels as S / E so Review needs no change. Parameters worth tuning on the real load cell: the 10 N threshold, the 70 ms sustain, the 20 ms averaging (constants at the top of `automvic.py`). One MVIC per recording by design.

- **Test status of 0.7.1:** committed by another session. On 2026-10-13 I re-ran `test_loadcell`, `test_force`, `test_force_gui`, `test_analysis` and
  `test_provenance`: 55 passed and 1 failed, the failure being `test_provenance` still expecting the literal `0.7.0` after the bump to `0.7.1`. I changed that
  test to check that the version is a version number equal to `SOFTWARE_VERSION` (no edit needed at the next bump); it passes. **The rest of the suite was not
  re-run on 0.7.1: run `Run_tests.bat` first thing.** Some force tests are timing-sensitive under load (they use a simulated source in real time).
- **Run long tests in the background** writing to a file (`> file 2>&1` with `run_in_background`), not as `cmd &` in a subshell (the monitor then sees an early exit).
- **Patching:** write a patch script with the Write tool and run it (multiple heredocs in one Git-Bash command fail; `\n` inside non-raw
  patch strings can create real newlines: run `python -c "import limbtrack.gui"` after every patch).
- **Screenshots:** the offscreen Qt platform has no fonts; for real screenshots run with `QT_QPA_PLATFORM=` and `win.grab().save(...)`
  (a window appears for a few seconds). GUI tests use `QT_QPA_PLATFORM=offscreen` and monkeypatch `QMessageBox`.
- **Replay method for tracking questions:** read `recordings/.../_raw.mp4`, create `TrackerSet`/`Engine`, mark dots at the frame-0 positions
  in the CSV, step frames, and compare lost frames / draw circles (a strict "truth" detector did not work well; looking at pictures did).
  The live loop can be reproduced by streaming a recorded video from a separate process as MJPEG to `http://127.0.0.1:PORT/video`.
  These helper scripts live only in the session scratchpad (`...\scratchpad`), not in the repo.
- Performance facts: `engine.process` ~5-12 ms for 3-5 dots, ~36 ms with 7 dots + strap lost + window; `_blobs` and the GUI trace redraw are the
  main costs; the force reader thread costs ~2 ms.
- OpenCV warnings (DSHOW "unknown C++ exception") on the DroidCam index are expected; `OPENCV_LOG_LEVEL=ERROR` silences them.
- Other sessions' notes live in `C:\Users\llind\.claude\projects\C--Users-llind-Desktop-Studie-ChenData\memory\` (`project_limb-angle-tracker.md`,
  `project_force-calibration.md`, `project_anybody-seated-mvic.md`, `feedback_two-audience-deliverables.md`).

## 10. Things I could not verify

- Any real-camera/stickers/inclinometer accuracy (nothing validated); torque on real data; the picture delay; behaviour of the final
  camera chain; the real DAQ in my own tests (it was held by MATLAB; the later calibration session used it with FlexLogger and the user ran the app on it).
- The cell's rated capacity vs the 250 kg test load, and the amplifier wiring.
- Whether the full suite is green on 0.7.1 (only the force-related subset was re-run; see section 9).
