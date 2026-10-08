# Parked

Decided on 2026-10-01 after reviewing FEATURES.md. "Parked" means: no further work for now, and nothing is removed. The reasons below are written from context; edit them if they are not what you meant.

| Item | Status in the code | Why parked |
|---|---|---|
| Pelvic tilt as a neighbour angle | Already built and active in every test; left exactly as it is | Not part of the first pilot's essential angles. Also assumes a level camera, which is not checked yet |
| Frontal view (hip abduction/adduction) | Not built | Needs the frontal camera and different landmarks; the webcam proof of concept is sagittal only |
| Two cameras (frontal + sagittal) | Not built | Webcam is a proof of concept; two cameras also need synchronising |
| Lens calibration | Not built | Not needed until cameras with noticeable distortion (phones, wide lenses) are in use; needs a checkerboard session |
| Camera alignment aid (plumb line, tilt check) | Not built | Useful once the real camera position is fixed; no real setup yet |
| Phones as cameras (Android, iPhone, mixed) | **Built 2026-10 as far as can be without a phone**: phone-as-webcam apps (Iriun, DroidCam, Camo) appear as named cameras, a stream-address box, a camera check and set-up checklist. Not tried on a real phone | Needs a phone and an app to test (see `FEATURES.md` H11) |
| ADC and load cell | Not built | Hardware not available yet; the CSV already carries a camera clock column (`clock_s`) so force can be merged later |
| Change-since-lock for the main joint | Not built | The main joint is judged against the typed target; a separate "movement since lock" number is deferred |
| Ghost pose (dashed copy of the locked pose) | Code and checkboxes commented out, marked `# PARKED-GHOST:` in `engine.py`, `overlay.py`, `geometry.py`, `review.py`, `gui.py`, `results.py`; its tests are skipped (`tests/test_ghost.py`) | Decided to simplify the app after reviewing FEATURES.md |
| Numbers-only mode (no video saved at all; audit REP-05 part c) | Not built. **Keep in mind as a fallback:** if video recording is not permitted (consent or ethics), record only the dot positions. The CSV and `meta.json` already hold everything needed for the angles; what is missing is a "Save video" switch (the recorder opens no video writers, Review must open a trial without a video). A numbers-only trial cannot be re-checked in Kinovea afterwards, so the validation subset should keep video. Decided 2026-10-08 to build it only if it becomes necessary |
| Custom dots and angles | Not built (apart from two optional dots, strap and anchor, added 2026-10-07; they belong to no angle) | The 8 landmarks plus the three-point angle mode cover the pilot |

## Proposed, not added
| Item | Note |
|---|---|
| Fixed pixel floor for dot size | **Replaced 2026-10-07** by a check in centimetres once a ruler scale is set (dots under 2 cm are flagged). A fixed pixel floor is still not added. Original note: stickers will be at least 2 cm. The app has no centimetre scale, so a floor can only be set in pixels, and the right number depends on camera distance and resolution. The existing check against the size learned at marking stays. In the 2026-09-30 test video the dots measured 9.5, 7.5 and 5.8 px across, and they were not spec-sized, so the floor cannot be chosen from that video |

## Sidelined audit items (2026-10-08)
These come from `audit/AUDIT.md` and were not done because they are large or speculative for the pilot; nothing is lost.
| Item (audit id) | Why sidelined |
|---|---|
| Plug-in camera architecture, per-source capture threads, two cameras on one PC, ADC clock map, lens-calibration hook (ARC-02..04, ARC-07, ARC-10..12) | Repeats parked items or needs hardware that is not chosen yet |
| New session layout with big readout and session plan (GUI-16), review window defaults and axes (GUI-08), notes captured at stop (GUI-03), colour-blind-safe cues and hysteresis (GUI-11) | Better designed after one real session |
| Storage and retention rules (REP-06: how long raw video is kept, encryption), licence (REP-07) | Ethics and administration decisions (the recordings folder can now be chosen in the window) |
| Upper-body reuse refactor (PRK-07), moment arm with a scale-free ratio method (ANG-11 part), AnyBody conversion (ANG-06), lying postures (ANG-03) | Only if those tests or that workflow are used |
| AVI container with a repair step (DAT-01 rest), 60 fps cameras (CAM-07 rest) | Possible later if crashes or frame rates become a problem |
| Test-suite polish: fast/slow split, tolerance tests (TST-05, 06, 09) | Convenience |
