# Changelog

Version numbers refer to `SOFTWARE_VERSION` in `limbtrack/engine.py`; every recording's `meta.json` stores the version, the
meta schema and a `provenance` block (code hash, git commit, library versions, protocol hash).

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
