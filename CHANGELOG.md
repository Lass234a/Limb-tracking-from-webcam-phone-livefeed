# Changelog

Version numbers refer to `SOFTWARE_VERSION` in `limbtrack/engine.py`; every recording's `meta.json` stores the version, the
meta schema and a `provenance` block (code hash, git commit, library versions, protocol hash).

## 0.8.1 - 2026-10-14

### The DroidCam camera is opened by name: the program was opening the laptop webcam instead (bug fix)
- **Found with the new DroidCam Client over USB.** DirectShow lists the cameras as `DroidCam Video, Integrated Camera`; Windows Media Foundation lists them as `Integrated Camera, DroidCam Video`. DirectShow cannot open the DroidCam camera (it refuses at once, known since 0.4.1), so the program fell back to Media Foundation **under the same number, 0, which is the laptop webcam**. That camera is slow to open, so the program waited 20 s and then blamed the phone ("Opening the camera did not answer ... not connected to the phone"), although the client preview moved and the Windows Camera app worked. The DroidCam camera itself, as Media Foundation camera 1, opens in 0.3 s and delivers 1280x720 at about 28 fps.
- **Now** the fallback lists the Media Foundation cameras by name (`camera.msmf_camera_names`, which opens no camera) and opens the one with the same name as the camera chosen in the list. If the names are known and the camera is not in the Media Foundation list, it **refuses** (it will not open another camera under its number). If the names cannot be read at all, the same number is used as before. The window's camera line names the camera opened through Media Foundation, and `meta.json` `camera_settings` records `capture_device_name`, `capture_media_foundation_index` and `capture_matched_by_name`. With the real DroidCam: list number 0 opens in 0.7 s as 'DroidCam Video' (Media Foundation 1), 1280x720.
- **Heads-up for older recordings:** before this version a camera opened through the Media Foundation fallback could silently be a different camera than the one chosen if the two numberings differed. Look at the video of any trial recorded with `capture_backend: MSMF` (the first stream and phone tests) to be sure it shows what you expect.
- The advice texts ("did not answer within 20 s" and the phone check) no longer blame only a client that is not connected: they say how to add the phone in the new client (Add > DroidCam, Refresh Device List; the older client has Wi-Fi and USB tabs) and that a hung camera open stays inside the program until it is closed and started again.
- No tracking, angle, force or recording number changes. `tests/test_camera_by_name.py` (no camera hardware: OpenCV and the name lists are faked). **Not tried:** other phone apps (Iriun, Camo) and other virtual cameras. (An idea that turned out wrong on the way: that OpenCV's hardware-accelerated Media Foundation conversion made the DroidCam open stall; it only slowed down the laptop webcam that was being opened by mistake, and no such setting is made.)

## 0.8.0 - 2026-10-13

### Automatic MVIC start and end markers from the force
- **New tick box in the Record box: *Mark the MVIC automatically from the force*** (needs the force connected), with *Start when force rises by* (default 10 N) and *MVIC length* (default 7 s, the protocol's 1-5-1 tempo: ramp 1 s, hold 5 s, relax 1 s). **Pressing Record arms it.** When the force clearly leaves its rest level (the ramp begins) the program writes the **MVIC start** marker, and the **MVIC end** marker 7 s after it. They are the same markers the S / E keys write (same labels, CSV `event` column, `meta.json` events), so Review's window, the Check tab and the exports read them unchanged. Pressing S or E by hand takes the markers over. The picture header (and so the saved overlay video) and the Record box say *armed, waiting for the ramp* / *3.2 / 7 s* / *start and end marked*; the marker banner says *(automatic)*.
- **How the ramp is found** (`limbtrack/automvic.py`): the force (newtons without the zero, only the rise matters; volts with a 0.02 V threshold if there is no load cell slope) is averaged over 20 ms (one 50 Hz cycle: the hum cancels exactly). The rest level is the median of that signal from 3.0 s to 0.5 s before now, so a slow drift is followed and the ramp itself is never part of the reference; the force from just before Record is used, so a ramp that begins at once is found. The start is called when the force is at least the threshold away from the rest level, has moved at least 40 % of the threshold within the last 0.25 s (a slow pre-tension is not a ramp) and both hold for 70 ms (a knock is not a ramp). The threshold is raised to 6 x the noise of the rest level if the rest was unsteady. Either direction of force counts. The end marker is placed on the first picture 7.0 s after the start marker.
- **Measured on made-up force traces** (`tests/test_automvic.py`): for the protocol's ramp (400 N in 1 s) the start marker goes in **0.11 s after the force begins to rise** (100 N in 1 s: 0.18 s; 200 N in 3 s: 0.25 s; 1000 N in 0.5 s: 0.11 s), and the program's estimate of when the force really began is within 0.03 s. Ignored: 50 Hz hum of 47 N (90 mV), a pre-tension of 8 N per second, knocks of up to 50 ms, a rest that sways 6 N. A push of 15 N on a swaying rest is not called, a 60 ms knock is. In real-time recordings with the simulated force (3 runs, pictures rendered at about 20 per second) the start marker went onto the picture **0.14 to 0.21 s after the real start of the ramp** (read from the saved `force.csv`) and the end marker 2.53 to 2.62 s after the start for a set 2.5 s (the end goes on the first picture after it is due, so with 30 pictures per second it is at most 0.03 s late); the estimate of the ramp's start was within 0.001 s there (`tests/test_automvic_engine.py`).
- `meta.json` gets `auto_mvic_setup` (on or off, threshold, length; frozen while recording like every setting) and, when on, `auto_mvic`: what happened (state, the times of the markers, the force's estimated onset and how long after it the start marker came, the rest level and noise, the threshold used). Review's Check tab says the markers were automatic, or, if the force never rose by the threshold, that no start was marked (use *Window = force above half of peak*).
- **What can change a measured number: nothing.** Force volts and newtons, dot positions, angles and the CSV values are exactly as before; only the `event` column and the marker list can contain markers the operator did not press. The start marker is about 0.15 to 0.2 s later than the true start of the ramp (it is not moved back: the rows are already written); the true start estimate is saved. Markers sit on the picture time stamps like those pressed by hand, so the *picture delay* applies to them as to everything else. **Not tried:** the real load cell and DAQ (only the simulated source); check the first real session's marker against the force in Review.
- One MVIC per recording: after the end marker nothing more is marked automatically.
- **Test fix (no program change): three tests in `tests/test_analysis.py` (the two torque tests and the picture-delay test) failed on an idle computer, and on the committed 0.7.2 code too.** Their helper `humming_trial` replaces the force of a real 3 s recording with a known 17 s signal but kept the real picture times, which end at about 7.5 s on an idle machine (the recording is 90 pictures with a 33 ms wait each plus the time to render them); the tests look at 8 to 13 s. They had passed only while the computer was busy with other work (slower pictures, a longer recording), which is why the full runs of 0.7.1 and 0.7.2 were green. The helper now gives the pictures a time axis that spans the injected signal (`tests/test_force.py`), so the result no longer depends on the computer's speed.

## 0.7.2 - 2026-10-13

### Tape dots (a black disc drawn on a square of white sports tape), because black sticker dots could not be sourced
- New `MARKERS.md`: how to make and place the dots (white matt tape square about 3.5-4 cm, solid black disc about 2.2 cm, **Dot colour = Black dots**), where each landmark's dot goes, why the ASIS/PSIS discs go on the outer side of the hip level with the bone point (a point moved towards the camera keeps its position in a side picture), why no black line between or through dots, why dots rather than a line marker, and what was tested.
- **Tested on synthetic pictures with the real tracker code** (6.5 px/cm, a 2 cm disc on a white square, 120 moving pictures): a black disc on tape is followed as well as a bare black disc (no picture lost, centre about 0.1 px = 0.2 mm) for tape squares of 3 to 8 cm, tilted tape, a streaky hand-filled disc, more noise, a soft picture, 45 % light and a disc seen side-on at 70 % and 50 % of its width (ASIS/PSIS from the side). Full leg, eight dots on tape against eight bare dots: every angle within 0.07 deg of the bare dots and within 0.1 deg of the true angle.
- **Three hazards found** (the first two avoided by filling the disc and choosing Black dots; the third by not drawing lines between dots): (1) a thin *outline* circle on tape it is not centred on, with Dot colour *Auto* in a slightly soft picture, was followed on the white tape instead of the ink, **11 mm off, with no picture lost and no warning** (Dot colour *Black*: exact; a filled disc in Auto: exact); (2) an outline circle with a gap pulls the centre 1.3 mm (40 deg gap) to 3 mm (90 deg gap) towards the closed side; (3) two discs on one tape strip joined by a thick (4 mm) marker line merge into one long dark blob: the dot was marked up to 1.7 cm off the disc without any warning, or marking failed, or the dot was lost in every picture (a 2 mm line did no harm). The *Dot colour* box now has a tooltip about the first. Tracker code and `tests/golden_tracking.npz` are unchanged.
- **Dot-size check corrected** (a diagnostic: **no angle, dot position, force or tracking number changes**): `EDGE_BLUR_PX` 1.3 -> 0 in `scale.py`. The 1.3 px allowance had been measured on the synthetic test dots, which OpenCV draws about 1 px wider than their stated size (a dot "of 14 px" really covers 15.0 px: measured from its area). On discs of exactly known size the tracker reads +0.3 px (sharp) to -0.7 px (soft) from the truth. With the allowance, a perfect 2.0 cm disc read about 1.8 cm and a 2.2 cm disc 2.0 cm, so discs drawn to spec were flagged TOO SMALL. `dot_size_check` in `meta.json` is now larger by 1.3 px / scale (0.2 cm at 6.5 px/cm) than 0.7.1 would have given for the same dots. It is still an estimate: check it once with a disc of known size.
- `synthetic.render(..., patch=half_side_px)` draws a tape square behind every dot (default `None`: pictures unchanged). New `tests/test_tape_patch_dots.py`.
- **Not done / not known:** real glare on tape, creased tape, sweat, skin sliding, a real camera. A line (black stripe on tape) tracker was considered and not built (`MARKERS.md` section 4).

## 0.7.1 - 2026-10-13

### Force: the load cell's regression is built in, Zero is the only step, the regression is shown in Review
- **Regression kept as data** (`limbtrack/load_cell_calibrations.json`, code in `limbtrack/loadcell.py`): every calibration ever made is stored with its points, one is *active* and the program uses it from the start. The first entry is the calibration of 13 Oct 2026 (24 steady runs with 0 to 250 kg hung on the cell): **18.6102 mV/kg = 526.95 N per volt**, slope known to 0.04 %, scatter of single runs 0.15 kg, the cell reads about 0.3 kg higher unloading than loading, no significant curvature. A reading is good to about +-3 N (95 %) from 10 to 250 kg.
- **New calibrations need no code change**: `LoadCellCalibration/logs_to_points.py FOLDER` turns the LogFile CSVs of a session into a points file (and flags short, unsteady or wrongly named runs), `LoadCellCalibration/add_calibration.py POINTS.csv --id NAME --activate` fits it and stores it. Older recordings keep the id of the regression they were measured with.
- **Zero** (button or **Z**) is the only step before a session and can be repeated as often as the set-up changes: it takes 1 s of signal as 0 N and says how far it moved. **The zero is not kept when the program closes**: a new start shows and saves volts only until the first zero, so a wrong zero can never carry over. The zero used is saved in every trial's `meta.json`.
- **Volts are still the permanent record**: every sample is saved as measured (`force.csv`, `force_v`); newtons are the second channel.
- **Review**: new *Calibration* tab with the regression (points up and down, the line, how far each run is from it with the +-95 % band, this trial's peak and plateau marked) and in words how precise this trial's peak and plateau are (+- N and %), with a warning for forces beyond the heaviest calibration load. The *Force* box has a **Zero** box and *Recalculate newtons*: it applies the zero in the box and the active regression to the saved volts (to correct a wrong zero, or to use a newer calibration). The export has `force_regression_id` and `force_peak_95pct_n`.
- *Re-calibrate with a known weight* is kept but only for the present session (not saved).
- Removed: `force_calibration.json` (it kept the zero between sessions).

## 0.7.0 - 2026-10-09

### Review, easier to read and more to read from
- New layout: the angle plot is full width, the force has its own strip under it on the same time axis (it used to share the plot, with a second axis and a thick raw band that hid the angle curve), and the numbers are split over four tabs: *Check*, *Angles*, *Force*, *Dots and pull*. The force controls are only shown for a trial with force.
- **Check tab**: a go / no-go list for the trial in plain English (picture rate, whether the computer was the limit, lost dots, strap dot not seen, anchor wander, hold angle against target, joint drift in the hold, hum, range limit, calibration, MVIC marker against the force onset).
- **Force tab** additions: onset, 10-90 % rise time, steepest rise, how long above 90 % of the peak, steadiness, and **torque (N.m)** from the force and the moment arm when the force is calibrated and a ruler scale and the anchor dot exist (the ankle dot stands in where the strap dot was not seen, and that is reported).
- **Export all trials of this participant**: one summary CSV row per trial, with the 10 % repeatability check between trials of the same position.

## 0.6.0 - 2026-10-09

From the first force and strap tests (P12).

### Force
- **Smoothing**: a centred moving average (default 100 ms, a multiple of 20 ms so that 50 Hz mains hum is removed exactly) for the live read-out, the live trace, Review's curve and all Review force numbers. The saved files keep every raw sample. The first recordings had about 90 mV of 50 Hz hum on a 1 V signal; the raw single-sample maximum was 8 to 11 % above the smoothed peak.
- **Review**: smoothing box, the 50 Hz hum measured and warned about, the raw peak listed next to the smoothed one, and a button *Window = force above half of peak* (the MVIC markers were 4 to 5 s early in both trials). The export has the smoothing length, the hum share and the raw peak.

### Speed (live rate with a stream, 7 dots, the strap lost, force and the window on; replayed on this computer)
- 9 to 12 fps before, 17 fps now (20 fps with the live trace off). The program's own work per picture fell from 49-74 ms to 36 ms.
- Blob measuring only looks at the pixels that belong to blobs; extra higher-threshold searches are skipped when no blob near the dot is big enough to hold it (exact: a blob at a higher threshold always lies inside a blob at the base threshold); the force clock estimate no longer scans its whole window for every chunk; the live trace redraws 5 times a second instead of 10.

## 0.5.3 - 2026-10-09
- A camera that does not open within 20 s is now reported with a plain message instead of leaving the program waiting silently. Found with a DroidCam Client that was running but not connected to the phone: the camera was listed, DirectShow could not open it, and the Media Foundation open then hung. The message says the usual cause (the phone's PC program is not connected to the phone; its own preview must be moving) and a camera that finally opens late is released. `Check_phone_camera.bat` uses the same wait and no longer hangs; it hides the driver's warning lines.
- 'The camera stopped delivering frames' now adds the same advice for phone cameras.

## 0.5.2 - 2026-10-09
- `Check_phone_camera.bat` (`limbtrack/diagnose.py`): a step-by-step check of a phone camera (DroidCam, Iriun, Camo, IP Webcam): is the PC program running, do pictures arrive from the camera list, can the phone be reached over Wi-Fi, which stream address (/video or /mjpegfeed) works and can the program read it. It reads only, prints a plain-English verdict and saves it. The connection-failed message points to it.

## 0.5.1 - 2026-10-09

### Force: voltage as the permanent record
- The amplifier gives -10 to +10 V, which the DAQ reads directly. The **voltage** is logged as is (`volts` in `force.csv`, new `force_v` column per picture in `data.csv`); **newtons** are a second channel that stays blank until a calibration exists.
- A calibration made later can be applied to older recordings in Review (*Apply the saved force calibration*); the saved volts are never changed and the export lists the calibration used.
- A signal at the edge of the +-10 V range is flagged in the picture and counted in `meta.json` (`samples_at_range_limit`).
- Review: *Picture delay* box (how late pictures are time-stamped; default 0) shifts where the angle at peak force is read; saved in the export. The live read-out shows newtons and volts together, and the trace menu has a voltage view.
- Fixed before release: opening a trial with force in Review would have crashed (a plot item without the thinning method).

## 0.5.0 - 2026-10-09

### Force (load cell through an NI DAQ, new)
- New **Force** box in the Live tab: source (NI DAQ, or a simulated contraction to try the program), device, input, wiring (differential or single-ended), *Connect*, a live read-out in N (or V until calibrated), *Zero* and *Calibrate with a known weight*. The calibration is kept in `force_calibration.json` (not in the repository) and copied into every trial.
- The DAQ is sampled on its own thread at 1000 samples per second; every sample gets a time on the same clock as the video frames, so a force sample can be matched to the joint angle at that moment. The time alignment is good to a few milliseconds (see FEATURES.md); the MVIC start marker is an independent check.
- Recording saves `<trial>_force.csv` (every sample: `t_s`, `clock_s`, `volts`, `newtons`), a `force_n` column in `data.csv` (mean force since the previous picture), and `force` / `force_setup` blocks in `meta.json` (device, rate, calibration, peak and when, measured rate, gaps, read errors). The force and its peak are drawn in the picture header and the saved overlay video.
- Safe handling: the zero and the calibration cannot change during a recording; a recording is refused if the DAQ is not delivering (with the reason) or with a video file (its clock does not match the DAQ); a DAQ that stops during a recording is flagged in the picture and in `meta.json`; a DAQ held by another program (MATLAB, NI MAX) gives a plain message.
- Review draws the force on a right-hand axis and reports, for the window: peak force and when, plateau mean, the rest level before, and the **main joint angle at the peak force** (the second candidate number for a model input, next to the plateau mean angle). These are also in the exported `_summary.csv`.
- New dependency `nidaqmx` (and the NI-DAQmx driver, already installed on this computer).

From the first tests with a board and black markers (recordings of 2026-10-09).

### Tracking (tracking numbers can differ slightly: at most 0.27 px on the Bundepus reference video)
- A dot that touches a shadow, a finger or a strap is no longer dragged toward it: when no blob fits or the blob is too big or too long, the search is repeated at higher contrast thresholds, which cut the attachment away. This also finds a lost dot again while a finger still touches it.
- The learned dot size no longer follows a merged blob (it had grown to twice the dot size).
- Small fragments of a dot hidden by a hand are no longer accepted as the dot (lower area limit 0.3 to 0.40 x the dot area; a half-covered dot, about 0.5, is still tracked and flagged).
- On the board recordings the dots stay on the dots through finger contact (checked on pictures); the tracked dot of the first trials is no longer lost for 131 and 44 frames when replayed.

### Ruler
- The two ruler dots are fixed points, not tracked (nothing can lose them). The scale is constant; a camera or ruler moved after marking is not noticed, so mark it again.
- Review draws the ruler of the trial and shows `ruler N cm = X px/cm` (or `no ruler scale`) in the title line.

### Speed
- Measured on this computer (real window, real camera, recording with overlay): 30 fps live and 29.6 fps while recording, so the computer is not what limited the first board recordings (17-26 fps); the camera or phone delivering fewer frames (dim light) fits the frame spacing better.
- Blobs are measured all at once instead of one by one (same numbers to 1e-13 px; about 2.7 x less time per dot in a cluttered scene), and the extra high-threshold search for a dot merged with a shadow is skipped when nothing could be that bright or dark. Worst case (8 dots all lost in a busy picture): 27 ms per frame.
- Every recording's `meta.json` has `pipeline_timing`: the program's work per frame (mean, 95th percentile, max), the time between frames (median, 95th percentile, max), the loop frame rate and the share of the interval spent working. If work is much smaller than the interval, the camera or phone is the limit.

### Recording
- The raw and the overlay video are encoded by two threads in parallel (insurance for slower computers; on the development computer encoding was not the cause of the 17-26 fps seen in the board recordings: the writer queue never held more than 2 frames).

## 0.4.1 - 2026-10-08

### Cameras
- A webcam that DirectShow cannot open by index is opened through Windows Media Foundation instead. Found with a real DroidCam ("DroidCam Video"): DirectShow failed with "Could not open video source", Media Foundation delivers 1280x720 at about 30 fps. `meta.json` records `capture_backend`. The "uncompressed video" warning applies to DirectShow only.

## 0.4.0 - 2026-10-08

### Operator workflow
- **Session plan**: a list of every position of the protocol with the number of trials already saved for the typed participant (counted from the files on disk). Click a row to set the test and target; the Record box says "Next: trial 2 of 2". Positions outside the plan (a typed target) are not counted. The planned number of trials per position is `TRIALS_PER_POSITION` (2) in `limbtrack/gui.py`.
- The panel now starts with what is needed for every position (plan, mark/lock, record, test and target); set-once-per-session items (video source, camera checklist, ruler, saved files and privacy) come after; landmarks and the three-point angle are folded under *Show advanced*.

### Phone streams
- A stream address without a path (for example DroidCam's `http://192.168.0.100:4747`) gets `/video` added, because the bare address is a web page. A bare `192.168.0.100:4747` also works.
- *Connect to stream* stops a running camera first (not while recording). A failed connection now shows its reason, with a hint, under the video-source buttons instead of on a line that may be scrolled out of view.

### Privacy (GDPR) - audit REP-05
- Privacy crop: drag a rectangle round the leg; only that part of the picture is written to `raw.mp4` and `overlay.mp4` (about half the encoding work).
- Up to 3 black masks (for example over a face), black in the saved videos only.
- Tracking runs on the whole picture and the CSV keeps whole-picture pixel positions, so no angle or position changes (checked: identical numbers with and without a crop).
- Recording will not start if a marked dot is within 30 px of the crop edge or inside a mask; while recording, a dot within 10 px of the crop edge or inside a mask is flagged on screen and counted in `meta.json` (`privacy_dot_warnings`).
- The crop and masks are cleared when the mirror setting or the picture size changes, and cannot be changed while recording.
- Review puts a cropped video back at its place on a black full-size picture, so dots, trails and the overlay export line up as before.
- `meta.json` gained `privacy` (crop, masks, full picture size).
- Participant code must be like `P01` (2-3 digits); notes containing an `@` or a long number are refused (a name or personal ID number must not end up in the files).
- Recordings folder can be chosen in the window (for example an encrypted drive) and is remembered in `settings.json` (not in the repository).

## 0.3.0 - 2026-10 (pilot build)

Changes after the first audit and the simplification decisions. Details and reasons are in `FEATURES.md`.

### Measurement and tracking
- Lost-dot rule: frozen position, fixed search circle, strict resume checks (round, right size, bright enough).
- A covered dot can no longer take a fast-moving neighbour (neighbours are protected where they will be; duplicate trackers are resolved by fit to prediction).
- The Auto dot colour is decided on every click and kept only if the marking succeeds.
- Partly covered or cut-off dots are flagged (warning only).
- Re-marking prefers lost dots and never lands on a neighbour; clicks on the video are ignored while recording.
- Display smoothing is a 0.2 s moving average and never touches the saved numbers; the reference stored at Lock does not depend on it.

### Protocol
- `protocol.json` follows Tabel 2 of the 2026-09-28 draft (hip extension with knee 0 and knee 90, positions and required neighbour angles); validated at start-up.
- The knee is signed (hyperextension negative); a test without positions shows the main joint without judging it.
- A wrong "Participant faces" setting is detected from the marked dots and Lock is refused.
- Three-point angle as interior angle or as flexion, without the mirror-image target trap.

### Scale, strap and displacement
- Pixel scale from a ruler with a dot at each end; dot size in cm against the 2 cm minimum.
- Optional strap and anchor dots; angle of pull, moment arm and strap-to-joint distance.
- Review: dot displacement, pull table, hold-plateau angle, rest-to-hold shift, identifying columns in the exported summary.

### Recording
- CSV written row by row, complete `meta.json` from the start, safe closing, settings frozen while recording and snapshotted at the start.
- Video encoded on its own thread; the overlay video is optional (Review can export it later).
- A silent live camera is waited for and reopened; gaps, failed reads and reconnects are recorded; a picture-size change stops and saves the recording.
- Videos carry the source's own frame rate; CSV `frame` is the video frame index; new last columns `source_frame` and `<dot>_partial`.
- Exposure-lock button switched off; webcams open on automatic exposure.

### Input
- Webcam opened in the right order (MJPG last), granted format and measured frame rate shown and saved; mirror switch; error log for the video thread.

## 0.2.0 - 2026-09/10
First tested version (see git history): live view of dot positions and angles, recording, review.
