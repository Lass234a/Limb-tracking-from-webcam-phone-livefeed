# Set-up rules and validation plan

Written for the pilot and for sharing with the supervisor. It turns the findings of the audit (`audit/AUDIT.md`, kept
outside the repository) into things to do. The software side is in `README.md` and `FEATURES.md`; this file is about the
set-up and about showing that the angles are right.

**What the app measures:** the 2D angle between marker dots in the picture. It is a positioning aid and a record of the
joint angle during each hold. It is not yet validated against a reference (see "Validation" below): until that is done,
treat the angles as positioning data and **do not give them to AnyBody as measured joint angles**.

## 1. Set-up for every session

**Camera**
- Phone or webcam about 2 to 3 m from the participant, level, at the height of the tested joint, looking square at the
  movement plane (a sagittal view for the lower-limb tests). Keep the tested joint in the middle third of the picture.
- Use 1280x720 or 1920x1080 at 30 fps. Start the camera and read the line under *Video source*: it should show the size you
  expect and at least 25 fps measured, without an orange warning.
- **Light the scene brightly and evenly** (a strong lamp near the camera, not only from the side or behind). Shadows are the main reason dots are lost: a hand, a strap or the limb itself casts a shadow beside a dot, and a dark dot and a dark shadow merge into one blob. A dim room also makes a phone camera lower its frame rate (the first board recordings ran at 17 to 26 fps although the camera can deliver 30). Matte boards and clothing, no glossy surfaces.
- Use steady lighting. Do not rely on the camera's automatic exposure changing mid-trial; in a phone app, lock exposure,
  white balance and focus, switch stabilisation off, set zoom to 1x and lock the orientation.
- USB is steadier than Wi-Fi for a phone. A phone that rotates or changes resolution mid-trial stops the recording (it is
  saved up to that frame).

**Dots**
- Stickers at least 2 cm wide, matt, clearly lighter (white) or darker (black) than their surroundings: white dots on dark
  clothing, black dots on skin. Keep neighbouring stickers at least one sticker-width apart; keep the strap marker at least
  three sticker-widths from the malleolus dot.
- **One person places all stickers**, by palpation, in the test posture. A 5 mm placement error already changes the angle by
  about 1.5 deg at the knee, 3 deg at the hip and 6 deg at the ankle (estimates from the audit's geometry), more than the
  tolerances; placement is the biggest error source, not the software.
- Set *Dot colour* to White or Black if the picture has only one kind. Auto decides per click and also works with both kinds
  in one picture, but a click well off the dot can still pick the wrong one.

**Ruler (scale)**
- A 10 to 20 cm ruler at the same distance from the camera as the joint, square to the camera, with a dot sticker centred on
  each end mark. Type the length between the two dot centres, mark both dots. Check that the scale box shows no warning and
  that the dot sizes are at least 2 cm.

**Strap and anchor**
- Strap marker on the strap where it meets the person, anchor marker at the fixed end (on the rope itself if the anchor is out
  of the picture). Both in frame. The header then shows the angle of pull and the moment arm. They are 2D values: a pull
  toward or away from the camera is not seen.

**Privacy (video of a person)**
- Best protection first: aim the camera at the leg so that the face is not in the picture at all.
- In the *Privacy* box draw a **crop** round the leg (everything outside it is never written to disk) and, if anything
  identifying is still inside it, add up to 3 **black masks**. Do this after the dots are marked, so the program can check
  that no dot is near the crop edge or inside a mask. Tracking and the CSV are not affected.
- Participant code only (`P01`, `P02`, ...). No names or personal numbers in the notes (they are refused).
- Choose the recordings folder (*Change recordings folder*) on an encrypted drive. Video of a person must never be
  committed to the repository (`.gitignore` blocks `*.mp4`, `recordings/`, `settings.json`).
- Keep video of the validation subset only, if that is what the consent and AAU's GDPR contact allow. The program can be
  changed to save no video at all (numbers only; see `PARKED.md`) if video turns out not to be permitted.
- This is technical help, not legal advice; the data-handling plan must be confirmed by the GDPR contact.

**Force (load cell and DAQ)**
- Wire the load cell's amplifier output to the DAQ: differential on `ai0` (+) and `ai4` (-) is best against noise; connect the amplifier's ground to an AI GND pin of the DAQ. The amplifier must give a voltage within +-10 V (a 4-20 mA output needs a shunt resistor). Check the amplifier's output range in its manual (the SIC-A2 datasheet was not found online).
- Only one program can hold the DAQ at a time: close MATLAB (`clear` or `release` its DAQ object), NI MAX test panels and other recordings before pressing *Connect*.
- Let the amplifier warm up as its manual says. The slope comes from the active regression (526.95 N per volt, 24 runs with 0 to 250 kg; `limbtrack/load_cell_calibrations.json`); with no pull on the strap press *Zero* (Z), as often as the set-up changes. As a check hang a known mass on the strap: the reading should be within the +-95 % shown in the Live box (about +-3 N; the cell itself reads about 0.3 kg higher unloading than loading). If the amplifier gain, the cell or the wiring changed, make a new calibration: hang 0 to the heaviest load up and down, record each step for at least 15 s, then `LoadCellCalibration/logs_to_points.py` and `add_calibration.py --activate`.
- Calibrate in the same direction the sensor is loaded in the experiment (an S-type cell reads tension and compression with opposite signs; if the reading is negative in your set-up, calibrate with the weight in the direction that gives the positive number, or accept that newtons per volt is negative).
- The strap pulls at an angle (see the pull angle): the load cell reads the force along its own axis.
- **Measure the picture delay** (how late the pictures are time-stamped; it is a property of the camera chain, so repeat it when the phone, app, cable or Wi-Fi changes): record a short trial with the camera seeing the sensor or the strap, and give the sensor one sharp tap (or pull sharply on the strap) while the participant is not involved. Open it in Review, find the first picture where the strap or hand visibly moves (step through with the slider) and the time of the force spike in the plot. Picture delay = the picture's time minus the force spike's time (normally +0.1 to +0.3 s). Type it in the *Picture delay* box in Review; the number is saved in the exported summary. Do this 3 times and use the median.
- The recorded **voltage** is the permanent record. You can calibrate or correct the zero later: in Review open the trial, change the *Zero* box if the zero was wrong, and press *Recalculate newtons*. Newtons then come from the saved volts, the zero in the box and the active regression, and the export lists what was used. Keep the amplifier gain and wiring unchanged between the recordings and the calibration, otherwise the volts of the older recordings do not match it.
- **Mains hum (50 Hz)**: the first recordings showed about 90 mV of 50 Hz on a signal of about 1 V. Review reports it. To reduce it at the source: run the laptop on its battery for a test (a charger is the usual source), use a shielded twisted-pair cable from the amplifier with the shield joined to an AI GND pin at the DAQ end only, keep that cable away from mains cables, chargers and motors, keep the DAQ's USB cable short, and use the differential input. Until then smoothing with a multiple of 20 ms removes it from the numbers.
- **Strap dot**: a dot on the buckle, behind the sock or under a fold is not seen. Put a bright sticker (white on the blue strap), at least 2 cm, on the flat part of the strap that faces the camera, a few cm from the buckle, and check in the picture that it stays visible while the strap tightens. The anchor dot was steady (0.3 px) in the first test; the strap dot was lost in 161 of 169 pictures because it was not visible.
- Check the time alignment once: record a short push, press **S** (MVIC start) as the force begins, and compare the marker time with the force onset in Review (they should agree within about 0.2 s, the reaction time; the program's own alignment is a few milliseconds).

## 2. During each trial
1. Click the next position in the session plan (or choose the test and target by hand). The required neighbour joint angle (for example hip 90) is shown green/red from
   the start.
2. Check *Participant faces*. The program warns and refuses Lock if the setting contradicts the marked dots.
3. Mark the dots, position the participant, **press Lock before Record** (it cannot be changed during a recording).
4. Record. Press **S** (or F6) at the start of the contraction and **E** (or F7) at its end for every trial: the hold
   analysis depends on these markers. After typing in Participant or Notes press Enter first, or use the F-keys.
5. After the trial look at the recording label (frames, duration, "VERY SHORT", "NOTHING RECORDED"), and at any gap or
   partly covered dot message. Repeat the trial if the camera lost frames during the hold.
6. Do not click the video, change settings or scroll over the side panel during a recording; the program ignores most of it
   but it is easier not to try.

## 3. Which number to report
- Report the angle **during the hold**: `plateau_mean_deg` in the exported summary (middle 60 % of the MVIC start-to-end
  window), together with `rest_to_hold_shift_deg`. Do not use the typed target or the file name as the measured angle.
- The saved CSV angles are unsmoothed; the picture on screen is a 0.2 s average and can lag a moving joint by up to a few
  degrees.
- The exported summary has participant, test, trial and software-version columns so trials from the whole study can be
  combined. Each trial's `meta.json` says exactly which code, libraries and protocol file produced it.

## 4. Validation (to do before the angles are used as measurements)
The tracker itself is very precise on its own (about 0.1 px, a few hundredths of a degree in tests), so the validation is
about the whole chain: stickers on skin, the camera set-up and the angle definitions.

**A. Bench check with a jig (about half a day).** A hinged jig with two rigid arms and dot stickers at the joint and along
the arms, a digital inclinometer on each arm. Set 5 or more angles (for example 30, 60, 90, 120, 150 deg), hold each for
5 s while recording with S/E markers, read the inclinometer, compare it with `plateau_mean_deg`. Repeat at the camera
distances and positions you will use, and once with the jig turned about 10 deg out of the camera plane to see the error
that out-of-plane rotation causes. *Proposed acceptance (your decision): bias within 0.5 deg and limits of agreement within
+/- 1 deg.*

**B. Sticker placement repeatability (a few participants).** The same person places the stickers twice in the same
posture on each of about 5 participants, with the stickers removed in between; compare the angles from the two placements
for each joint. This gives the real uncertainty of an angle in people, and it will be largest at the ankle and hip. For the
ankle, consider an inclinometer on the foot as the main measure, or an extra dot on the calcaneus.

**C. Comparison with the goniometer or inclinometer on people.** For the positions of Tabel 2, read the inclinometer on the
participant at the moment you press Lock and compare it with the app. Note differences by joint.

**D. Known limits to state in the methods.** 2D angle in the image (out-of-plane rotation and camera tilt add error);
lens distortion is not corrected (keep the joint central); dots on skin and straps can slide; the ruler scale is valid
only at the ruler's distance; the three-point knee differs a few degrees from the protocol knee (the fibular head is off the
epicondyle-malleolus line).

## 5. Not done / not covered
- Frontal view (hip abduction/adduction), two cameras, lens calibration, camera tilt aid: see `PARKED.md`. Force is built but not yet tried on the real load cell (`FEATURES.md` H13).
- Which number goes to AnyBody (plateau mean or the angle at peak force), and the conversion of skin-dot angles to the
  model's joint-angle conventions, are decisions for you and the AnyBody side (the AnyBody ankle sign is opposite to this
  app's convention).
