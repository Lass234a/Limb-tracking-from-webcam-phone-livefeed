# Changelog

Version numbers refer to `SOFTWARE_VERSION` in `limbtrack/engine.py`; every recording's `meta.json` stores the version, the
meta schema and a `provenance` block (code hash, git commit, library versions, protocol hash).

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
