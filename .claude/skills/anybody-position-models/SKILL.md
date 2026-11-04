---
name: anybody-position-models
description: Makes AnyBody (AMMR 3.2, AnyBody 8.2) models of the limbtrack MVIC test positions from the AnyBody export (anybody_inputs.csv), one model per participant and position, with the joint angles and force at the peak force, and an easy way to change force, force direction and joint angles. Use this whenever someone asks to make, build, update or adjust an AnyBody model, AnyScript or simulation of a measured position or MVIC trial (knee extension first), to turn the limbtrack measurements into AnyBody input, to change the force or angle in a position model, to run the position models, or says things like "lav en anybody model", "anybody script til hver position", "scale the model from the MVIC", "what does the model say about this trial", even if the skill is not named. Also use it to add the next test (knee flexion, hip, ankle) to the models.
---

# anybody-position-models

Turns the measured peak-force moment of each MVIC position into a small AnyBody model, so the strength scaling can start from one
trusted model per position instead of typing numbers into AnyBody by hand. Built so far: **knee extension** (and knee flexion, the same
model with the effort turned round, not looked at yet). The other tests are planned: `references/positions.md`.

## Read this first: what is and is not known

- **The models have not been run in AnyBody.** There was no AnyBody here. Every AMMR name in the AnyScript was checked by text search
  against the AMMR 3.2.1 source (human template, mannequin, leg segments, `EvaluateMaxLoad`). The parts that are our own (handing the
  two folders over with `#path`, the point support on the thigh) were only checked for balanced brackets and existing includes. How the
  model behaves when loaded and solved is untested. Do the first run on **one** position and go through `references/first-run-checklist.md` before trusting any number.
- **The data are not reviewed yet.** The pilot has no trial marked valid, so models come from `--include-unjudged` rows. Each
  `Position.any` says `NOT REVIEWED`. After the review, run the steps again: only the positions whose numbers changed are rewritten.
- The numbers are limbtrack's, read from the export, never measured here. A blank in the export becomes a stated default plus a warning,
  never a silent guess.

## The three steps

From the project folder (the tools use only the Python standard library; the first one needs the project's `.venv`):

```
.venv\Scripts\python.exe tools\anybody_export.py --root Pilot --out Export --include-unjudged
python tools\anybody_models.py --inputs Export\anybody_inputs.csv --out AnyBody_models --ammr-libdef "C:\...\AMMR\libdef.any"
python tools\anybody_models_run.py AnyBody_models --only P01:2 --dry-run
```

1. **Export** (already in the project): one row per participant and position, SI units. Drop `--include-unjudged` once trials are valid.
2. **Models**: writes the folder below. `--list-tests` shows what is built. `--ammr-libdef` points the model at your AMMR (or edit
   `SeatedKnee/libdef.any`).
3. **Run** on the computer with AnyBody 8.2 and `pip install anypytools`: take away `--dry-run`. Or open one `Main.any` in AnyBody
   and run the operations by hand (`RunApplication`).

```
AnyBody_models/
  SeatedKnee/                 the shared model, the same for every knee position (tools/anybody_templates/SeatedKnee)
  P01/Participant.any         height, mass, fat, thigh and shank length
  P01/pos02_knee_extension_t70/
      Main.any                open this in AnyBody
      Position.any            the numbers of the position: edit this to adjust it
  manifest.json / .csv        every model: inputs, where each number came from, warnings, and what AnyBody must reproduce
  results.csv / .json         written by the run step
```

## Adjusting a position

Everything you may want to change is in the `Position.any` of that position, one line each, with the source of the number beside it.
Reload the model in AnyBody after saving. The same can be done when writing the files: `--set force_n=450`, or scoped
`--set P01:2:force_n=450 --set P01:knee_deg=72` (participant, position number, key).

| Variable (`--set` key) | Unit | Default from | Notes |
|---|---|---|---|
| `Force_N` (`force_n`) | N | export `force_n`: total force from the load cell zero, 500 ms peak | a design variable; the MaxLoad evaluation moves it |
| `PullAngle_deg` (`pull_angle_deg`) | deg | export `pull_angle_deg` (2D); else 90 | between the pull and the shank, read from knee to ankle; 90 = square |
| `StrapDist_m` (`strap_dist_m`) | m | typed strap place; else 0.05 above the ankle | `StrapFromKnee` = 1 when measured from the epicondyle |
| `KneeFlexion_deg` (`knee_deg`) | deg | export: angle at the peak force from the standing reference; else dots; else the target | `--angles nominal` takes the target |
| `HipFlexion_deg` (`hip_deg`) | deg | 90 (protocol); the measured hip is not trusted | the measured value is in the manifest |
| `AnklePlantarFlexion_deg` (`ankle_deg`) | deg | 0 | `--set ankle_deg` is in limbtrack's sign (dorsiflexion +) |
| `HipAbduction_deg` (`hip_abduction_deg`) | deg | 0 | assumed |
| `ThighFixationDist_m` (`thigh_fixation_dist_m`) | m | 0.10 | where the thigh strap holds the thigh; not recorded |

A `Position.any` you edited by hand is **never overwritten** by a later run: the new text goes beside it as `Position.generated.any`
(`--force` replaces). Angles in the file are AnyBody's, not limbtrack's: the ankle sign is the opposite.

## What the results mean

- `max_muscle_activity`: the highest muscle activity the model needs to hold the measured force, with the AMMR default strength. **1.0 = just
  strong enough; 1.4 = the leg muscles would need to be 40% stronger.** This is one factor for the whole leg. The study's muscle groups
  are scaled later (`PRIMARY.tex`: the method is open until the data are in).
- `max_load_n` and `force_over_max_load`: the AMMR tool `EvaluateMaxLoad` finds the force at which the activity would reach 1 from two
  runs. It assumes activity is a straight line in the load, which fails at low loads where gravity dominates
  ([AnyBody blog on estimating maximum model strength](https://anyscript.org/tools/estimating-maximum-model-strength/)).
- `check_*`: what the model read back (angles, force direction, lever arm, moment) against what the manifest expects. If any differs, the
  `status` says `checks differ` and the activity must not be used yet.

## Choices made that the group should confirm

Each is in `references/first-run-checklist.md` with its basis and how to test it. The main ones: the force enters as the **total** force
(the limb's weight comes from gravity in the model, the resting force is not subtracted too); the force is applied as a pure force at the
strap, in the sagittal plane, as limbtrack's 2D pull angle gives it; the thigh is held by a point support at the distal femur and the pelvis by
six reactions; muscles are the AMMR simple model, recruited with `MR_MinMaxStrict`; the tape thigh and shank lengths replace the
regression lengths; the force is kept and the lever arm follows the model (the alternative is to keep the moment).

## Add the next test

1. Copy `tools/anybody_templates/SeatedKnee` to a new folder; change `TestedLeg.any` (the nodes), `Reactions.any` (what holds the body),
   `Mannequin.any` (posture) and `Loads.any` (where and how the strap pulls). `references/positions.md` says what differs per test.
2. In `tools/anybody_models.py` add the test to `REGISTRY` with `status: built`, its `model` folder, `drives` and `effort_axis_z`.
3. Add its checks to `tests/test_anybody_models.py` (the drift guards there fail when the AnyScript and the Python disagree).
4. Run on one position in AnyBody and write down in the checklist what you had to change.

## Say these things when you report results

"Not run in AnyBody yet" until a first run has passed. "Data not reviewed" while any model says `NOT REVIEWED`. The force direction is 2D
(a pull towards or away from the camera is invisible). The angles are skin-dot angles with a standing offset, not validated joint angles
(`SETUP_AND_VALIDATION.md` section 4). A model that needs `max_muscle_activity` well above 1 may point at the data or the set-up (strap
place, pull angle, lost dots) before it points at weak muscles.

## Keep local, and the AMMR licence

`AnyBody_models/` and the export hold participant numbers and are in `.gitignore`: never commit them; the repository is public. The
templates are our own text but follow the structure of the AMMR human template (the pelvis reaction block is close to it), and the AMMR is
covered by AnyBody Technology's licence agreement. Do not copy AMMR files into this repository, and ask AnyBody before the repository stays public with these templates.
