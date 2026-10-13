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
3. **State:** version **0.7.1** (`SOFTWARE_VERSION` in `limbtrack/engine.py`), on `origin/main` (commit `e321ff0`). Tags `v0.3.0` to `v0.7.0` are pushed; **0.7.1 has no tag**.
   Last full test run (0.7.0): **472 passed, 1 skipped**. 0.7.1 was made in a separate session (load cell regression); I re-ran the force/load cell/analysis/provenance tests on it (56 tests; one stale version check was fixed, see section 9) but **not the full suite**.
4. **Not yet validated:** the angles have never been compared with a goniometer/inclinometer; no real session with the final camera,
   lighting and stickers has been run. Everything so far is synthetic tests plus a few bench recordings (board with black magnets, one leg
   session P12). Treat angles as positioning data until `SETUP_AND_VALIDATION.md` section 4 is done.
5. **The user is not a programmer.** Explain in plain English, say whether any measured number can change, and keep a technical trail
   (CHANGELOG, FEATURES) separate from the plain explanation.
6. **Never commit video of a person, `recordings/`, `audit/`, PDFs, `settings.json`.** The repository is public (`.gitignore` blocks these).
7. **Raw data is never smoothed or corrected.** Volts and dot positions are saved exactly as measured; smoothing and calibration are
   applied afterwards, in Review, and recorded.
8. **Biggest practical risks right now:** (a) strap dot not visible on the strap, (b) mains hum on the force unless the laptop charges from
   the isolated outlet, (c) live frame rate (17-20 fps, camera can do 25-30), (d) MVIC markers pressed 4-5 s before the force really rises,
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
| Start the app | double-click `Start.bat` (or `.venv\Scripts\python.exe -m limbtrack`) |
| Full test suite (8-17 min) | double-click `Run_tests.bat` (writes `tests_last_run.txt`) or `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` |
| Check a phone camera | double-click `Check_phone_camera.bat` (reads only; writes `phone_camera_check.txt`) |
| Try without hardware | Live tab: *Open video file* `demo/demo_leg_white_dots.mp4`; Force box: source *Simulated* |
| Add a load cell calibration | `LoadCellCalibration/logs_to_points.py FOLDER` then `add_calibration.py POINTS.csv --id NAME --activate` |
| Re-check the protocol file | `.venv\Scripts\python.exe -m limbtrack.protocol check` |

Documents (all in the project root, written for the user as well): `README.md` (usage), `FEATURES.md` (every function, what it does to
the numbers, CSV/meta contents, section H = what is unreliable, section I = roadmap), `CHANGELOG.md` (per version, with measured
effects), `PARKED.md` (decided-not-to-build list), `SETUP_AND_VALIDATION.md` (set-up rules, validation plan, force/hum/strap notes).
`audit/` (not in git) holds the 2026-10 Opus audit (`AUDIT.md` technical, `AUDIT_overview.html` plain, `findings.json`).

### Code map (`limbtrack/`)
`tracker.py` (dot detection/tracking, all constants + `settings()` into meta) | `engine.py` (pipeline, judging against targets, lock,
recording, privacy, force hand-off) | `recorder.py` (CSV row by row, two encoder threads, atomic `meta.json`) | `camera.py` (webcam via
DSHOW then MSMF, streams, 20 s open time-out) | `gui.py` (Live tab) | `review.py` (Review tab, `Trial`) | `analysis.py` (checks, force
timing, torque, per-participant summary) | `force.py` (DAQ source, calibration, smoothing, hum) | `loadcell.py` + `load_cell_calibrations.json`
(regression registry, precision) | `scale.py` (ruler) | `pull.py` (strap/anchor pull angle, moment arm) | `privacy.py` | `protocol.py` |
`geometry.py`, `angles.py`, `filters.py`, `displacement.py`, `provenance.py`, `diagnose.py`, `overlay.py`, `synthetic.py`.
Tests: 52 files in `tests/` (synthetic leg generator `synthetic.py` makes exact-angle pictures).

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

### Cameras
- DroidCam: the virtual camera "DroidCam Video" cannot be opened by index through DirectShow; it works through Media Foundation (the
  app falls back automatically: 1280x720, ~30 fps). Windows **camera index order changes between restarts** (select by name).
- The PC client and the app cannot both read the phone stream: **the phone allows one viewer**. What worked on 2026-10-09: close the DroidCam
  Client and let the app read `http://PHONE-IP:4747/video` (a bare address without `/video` is a web page; the app adds `/video`).
  The phone was on the Windows hotspot network (192.168.137.x). The newer DroidCam Client looks OBS-like and needs its virtual camera started.
- A camera that never answers is now reported after 20 s with advice (a hang happened with the client running but not connected).
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

## 6. Force workflow as built (0.7.1)

1. Close MATLAB/NI MAX/anything holding the DAQ (the first attempt failed with "resource reserved" because MATLAB held `Dev1`).
2. Live tab -> Force box -> source *NI DAQ*, `Dev1`, `ai0`, differential -> *Connect*.
3. **Zero** (button or Z) with the rig hanging and nothing loaded; repeat whenever the set-up changes. **The zero is not kept when the program
   closes** (a new start shows volts only until zeroed); the zero used is saved in each trial. The active regression gives newtons.
4. Record. Force appears in the picture header (N and V) and in `force.csv` / `force_v`, `force_n`. A DAQ that stops, a signal at the +-10 V
   limit, or no samples for 0.5 s is flagged in the picture and `meta.json`. Force needs a **live camera** (not a video file).
5. Review: Check / Angles / Force / Calibration / Dots-and-pull tabs, force strip under the angle plot, *Window = force above half of
   peak*, smoothing box, hum warning, *Recalculate newtons*, precision of the peak from the regression (warns beyond the heaviest calibration load),
   *Export all trials of this participant* (one row per trial + the 10 % repeatability flag).

Torque (N.m) = smoothed force x moment arm: needs calibrated force + ruler scale + anchor dot; when the strap dot is not seen the ankle dot
stands in (reported). It has only been exercised on synthetic geometry so far.

---

## 7. Open items, in a suggested order

**Before real data collection (user/hardware):**
1. Real-world session with the final camera, bright lamp, spec-size (>= 2 cm) stickers, ruler and anchor/strap dots; run the Check tab on it.
2. Validation against a jig + inclinometer, and sticker placement repeatability (`SETUP_AND_VALIDATION.md` section 4).
3. Confirm the amplifier-to-DAQ ground wiring and the isolated-outlet/charger routine; confirm the cell's rated capacity and the rig's safety rating.
4. Tap test to measure the picture delay for the camera chain that will be used (repeat if phone/app/cable/Wi-Fi changes).
5. Decide: plateau mean angle vs angle at peak force for AnyBody; trials per position (`TRIALS_PER_POSITION = 2` in `gui.py`; the session
   plan counts against it); storage/retention of raw video; whether numbers-only recording is needed if video is not permitted.

**Software ideas the user was offered (none started):** force-versus-angle plot (slack/stiffness); automatic MVIC markers from the force;
an automatic picture-delay estimator from a tap; a faster live loop (down-scaled tracking); a check-weight step in the calibration flow.
**Never done:** a final multi-agent Opus re-audit of 0.3-0.7 (TRK-01 neighbour protection and the recording rework were not Opus-reviewed).
**Parked (see `PARKED.md`):** frontal view/hip abduction, two cameras, lens calibration, camera alignment aid, change-since-lock for the main
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
