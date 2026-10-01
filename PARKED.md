# Parked

Decided on 2026-10-01 after reviewing FEATURES.md. "Parked" means: no further work for now, and nothing is removed. The reasons below are written from context; edit them if they are not what you meant.

| Item | Status in the code | Why parked |
|---|---|---|
| Pelvic tilt as a neighbour angle | Already built and active in every test; left exactly as it is | Not part of the first pilot's essential angles. Also assumes a level camera, which is not checked yet |
| Frontal view (hip abduction/adduction) | Not built | Needs the frontal camera and different landmarks; the webcam proof of concept is sagittal only |
| Two cameras (frontal + sagittal) | Not built | Webcam is a proof of concept; two cameras also need synchronising |
| Lens calibration | Not built | Not needed until cameras with noticeable distortion (phones, wide lenses) are in use; needs a checkerboard session |
| Camera alignment aid (plumb line, tilt check) | Not built | Useful once the real camera position is fixed; no real setup yet |
| Phones as cameras (Android, iPhone, mixed) | Not built (the video-source code can already accept a network address, but there is no box for it in the window) | Webcam first; needs a phone and a streaming app |
| ADC and load cell | Not built | Hardware not available yet; the CSV already carries a camera clock column (`clock_s`) so force can be merged later |
| Change-since-lock for the main joint | Not built | The main joint is judged against the typed target; a separate "movement since lock" number is deferred |
| Custom dots and angles | Not built | The 8 landmarks plus the coming three-point angle mode cover the pilot |

## Proposed, not added
| Item | Note |
|---|---|
| Fixed pixel floor for dot size | Stickers will be at least 2 cm. The app has no centimetre scale, so a floor can only be set in pixels, and the right number depends on camera distance and resolution. The existing check against the size learned at marking stays. In the 2026-09-30 test video the dots measured 9.5, 7.5 and 5.8 px across, and they were not spec-sized, so the floor cannot be chosen from that video |
