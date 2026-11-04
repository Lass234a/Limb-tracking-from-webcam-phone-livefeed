---
name: anybody-position-models
description: Makes AnyBody (AMMR 3.2, AnyBody 8.2) models of the limbtrack MVIC test positions from the AnyBody export (anybody_inputs.csv), one model per participant and position, with the joint angles and force at the peak force, and an easy way to change force, force direction and joint angles. Every position is run with the hip held ideally and with the hip free, to give a range for the strength factor. Use this whenever someone asks to make, build, update or adjust an AnyBody model, AnyScript or simulation of a measured position or MVIC trial (knee extension first), to turn the limbtrack measurements into AnyBody input, to change the force or angle in a position model, to run the position models, to scale muscle strength from the MVIC, or says things like "lav en anybody model", "anybody script til hver position", "what does the model say about this trial", even if the skill is not named. Also use it to add the next test (knee flexion, hip, ankle) to the models.
---

# anybody-position-models

Turns the measured peak force of each MVIC position into a small AnyBody model, so the strength scaling starts from one trusted model per
position instead of numbers typed into AnyBody by hand. The model is the group's own **SeatedKneeExtMVIC** (run and checked by the group) made
general: the subject, angles, force, pull direction and tested side now come from the limbtrack export. Built: **knee extension** (and knee
flexion, the same model with the strap on the other side, not looked at yet). The other tests are planned: `references/positions.md`.

## Read this first: what is known

- **The base has been run.** SeatedKneeExtMVIC loads and solves in AnyBody 8.2 (its README: posture matches the photo to 0.1 deg, strap at
  90 deg, max activity 0.99999 at the predicted load). Our port of it has **not** been run. The first job on the AnyBody computer is the
  regression case below: it must give the group's own numbers (2036 N hip ideal, 555 N hip free) to 2 %.
- **The data are not reviewed yet.** The pilot has no trial marked valid, so models come from `--include-unjudged` rows. Each
  `Position.any` says `NOT REVIEWED`. After the review, run the steps again: only the positions whose numbers changed are rewritten.
- The numbers are limbtrack's, read from the export, never measured here. A blank in the export becomes a stated default plus a warning.

## The steps

From the project folder (the tools use only the Python standard library; the export needs the project's `.venv`):

```
.venv\Scripts\python.exe tools\anybody_export.py --root Pilot --out Export --include-unjudged
python tools\anybody_models.py --regression-case --out AnyBody_models --ammr-libdef "../../libdef.any"
python tools\anybody_models_run.py AnyBody_models --only REG:1
python tools\anybody_models.py --inputs Export\anybody_inputs.csv --out AnyBody_models --ammr-libdef "../../libdef.any"
python tools\anybody_models_run.py AnyBody_models --only P01:2 --dry-run
```

1. **Export** (already in the project): one row per participant and position, SI units. Drop `--include-unjudged` once trials are valid.
2. **Regression case**: writes the group's own model's numbers (left leg, 2.06 m, 106 kg, 15 %, knee 72.9, hip 82.2, ankle 25 deg plantarflexed). Run it
   on the AnyBody computer (`pip install anypytools`). Its `status` must be `ok`. Its `MeasuredForce_N` (800) is a dummy.
3. **Models**: writes the folder below. `--list-tests` shows what is built. `--ammr-libdef "../../libdef.any"` is right when `AnyBody_models`
   sits in `Documents\AnyBody.8.2.x` next to the group's model; otherwise give the path of your AMMR `libdef.any`.
4. **Run** one real position first, then the rest (take away `--dry-run`). Or open one `Main.any` in AnyBody and run `Main.RunApplication`.

```
AnyBody_models/
  SeatedKnee/                 the shared model, the same for every knee position (tools/anybody_templates/SeatedKnee)
  P01/Participant.any         height, mass, fat percent
  P01/pos02_knee_extension_t70/
      Main.any                open this: the hip held ideally (pelvic belt and thigh strap hold it)
      Main_HipFree.any        open this: the hip muscles hold the hip
      Position.any           the numbers of the position: edit this to adjust it
  manifest.json / .csv        every model: inputs, where each number came from, warnings, and what AnyBody must reproduce
  results.csv / .json         one row per position and hip version;  results_bracket.csv: one row per position, the two side by side
```

## Why two hip versions

When the knee muscles pull, something has to hold the hip still. **Ideal:** an invisible perfect clamp (the thigh strap and pelvic belt); the hip
muscles do nothing; the group's model predicted 2036 N. **Free:** no clamp; the hip muscles must hold it, and the rectus femoris (it crosses
hip and knee) needs the hip extensors as counterweight; 555 N. A tight strap makes ideal plausible; a strap that gives makes free closer. Nobody
knows where the truth is, and the answer moves by a factor of 3.7, so every position gets both and the strength factor is reported as a range.

## Adjusting a position

Everything you may want to change is in the `Position.any` of that position, one line each, with the source beside it. Reload the model in AnyBody
after saving. The same can be done when writing the files: `--set force_n=450`, or scoped `--set P01:2:force_n=450 --set P01:knee_deg=72`.

| Variable (`--set` key) | Unit | Default from | Notes |
|---|---|---|---|
| `MeasuredForce_N` (`force_n`) | N | export `force_n`: total force from the load cell zero, 500 ms peak | the model is run at this force for the read-backs |
| `ProbeLoad_N` (`probe_load_n`, `--probe-load`) | N | 800 (the group's model) | MaxLoading starts here and uses twice this; keep above about 300 N |
| `PullAngle_deg` (`pull_angle_deg`) | deg | export `pull_angle_deg` (2D); else 90 | between the pull and the shank, read from knee to ankle; 90 = square |
| `StrapAboveMalleolus_m` (`strap_dist_m`) | m | typed strap place from the lateral malleolus; else 0.05 | other landmarks are not converted |
| `KneeFlexion_deg` (`knee_deg`, `--knee-offset-deg`) | deg | export: angle at the peak force from the standing reference; else dots; else the target | `--angles nominal` takes the target |
| `HipFlexion_deg` (`hip_deg`) | deg | 90 (protocol); the measured hip is not trusted | the group's photo gave 82.2 |
| `AnklePlantarFlexion_deg` (`ankle_deg`) | deg | 0 | `--set ankle_deg` is in limbtrack's sign (dorsiflexion +); the group estimated 25 plantarflexed |
| `HipAbduction_deg` (`hip_abduction_deg`) | deg | 0 | assumed |

A `Position.any` you edited by hand is **never overwritten** by a later run: the new text goes beside it as `Position.generated.any`
(`--force` replaces). Angles in the file are AnyBody's, not limbtrack's: the ankle sign is the opposite.

## What the results mean

- `max_load_n`: the strap force at which the model's highest muscle activity reaches 1, with the AMMR default strength (the group's number).
- **`scale_needed_line`: the strength factor the leg muscles need so that the model's capacity equals the measured force.** The two MaxLoad runs give a
  line `activity = a * load + b` (b is gravity and the like); the factor is `a * F_measured + b`. 1.0 = the default strength is right; 0.4 = the person's
  muscles are 40 % as strong as the default. `scale_needed_ratio` is the plain `F_measured / max_load_n`, without the gravity term.
- `max_muscle_activity` is the activity at the measured force itself. It is **not** the strength factor: the trunk's baseline activity (about 13 %)
  hides the legs at low loads. Cross-check only.
- It is one factor for the whole leg. The study's muscle groups are scaled later (`PRIMARY.tex`: the method is open until the data are in).
- `check_*` and `status`: what the model read back (knee angle, pull angle and direction, lever arm, moment) against what the manifest expects. If
  `status` says `checks differ`, do not use the numbers. `knee_zero_offset_deg` is the gap between limbtrack's dot-based knee angle and the model's
  (1.5 deg in the group's test); use it with `--knee-offset-deg` if it is more than a degree.

## What was kept, what was added, what is opt-in

Kept from the group's model, unchanged in substance: hard drivers; every driver with reactions on except the tested knee; the strap node in the ISB
shank frame; full body with trunk and arm muscles on (switching them off made the solver fail); relaxed other leg (90 deg knee, 10 deg ankle);
MR_MinMaxStrict; nStep 1; EvaluateMaxLoad starting at 800 N. Changed to be general: the tested side, and the posture-check read-backs.
Added: the pull angle from the data (the group's was fixed at 90), the measured force, both hip versions, the strength factor from the load line.
**Opt-in, off by default:** tape thigh and shank length (`--segment-lengths`): in the AMMR the fibre length follows the segment length and muscle
force is volume / fibre length, so these numbers change the strength being scaled. The knee zero offset (`--knee-offset-deg`, default 0).
Not copied: the table and floor graphics (their heights depend on the subject); the group's PostureVel had 10 deg/s for the shoulder and elbow,
probably a slip; this model uses 0 (it should not change the knee result, those joints are held by reactions; if the regression case misses 2036 N
by a little, put the group's values back first).

## Choices that are the group's

`references/first-run-checklist.md` lists them with how to test each: the range of the hip; the angle at the peak force or the plateau mean; the
total force with gravity in the model (the resting force is never subtracted as well); the force kept against the moment kept (the model's lever arm
follows its own scaled shank); the whole-leg factor against the group factors; the knee zero offset.

## Add the next test

1. Copy `tools/anybody_templates/SeatedKnee` to a new folder; change `TestedLeg.any`, `JointsAndDrivers.any` (what is held, where the strap is),
   `Mannequin.any` (posture). `references/positions.md` says what differs per test.
2. In `tools/anybody_models.py` add the test to `REGISTRY` with `status: built`, its `model` folder, `drives` and `effort_axis_z`.
3. Add its checks to `tests/test_anybody_models.py` (the drift guards there fail when the AnyScript and the Python disagree).
4. Run on one position in AnyBody and write down in the checklist what you had to change.

## Say these things when you report results

"Our port was not run in AnyBody" until the regression case has passed. "Data not reviewed" while any model says `NOT REVIEWED`. Report the strength
factor as a range (hip ideal to hip free), never as one number. The force direction is 2D (a pull towards or away from the camera is invisible). The
angles are skin-dot angles with a standing offset, not validated joint angles (`SETUP_AND_VALIDATION.md` section 4). A factor far from what is
plausible may point at the data or the set-up (strap place, pull angle, lost dots, the hip) before it points at weak muscles.

## Keep local, and the AMMR licence

`AnyBody_models/` and `Export/` hold participant numbers and are in `.gitignore`: never commit them; the repository is public. The templates are
the group's own model made general, and the group's model follows the AMMR human template; the AMMR is covered by AnyBody Technology's licence
agreement. Do not copy AMMR files into this repository, and ask AnyBody before the repository stays public with these templates.
