# First run in AnyBody: what is proven, what is ours to test

The base is the group's SeatedKneeExtMVIC model, which has been run (AMMR 3.2.1, AnyBody 8.2). Our port makes it general, and that part was
written without AnyBody. Names were checked by text search in the AMMR 3.2.1 source and against the group's files; behaviour was not.

## Order of work

1. **Regression case.** `python tools/anybody_models.py --regression-case --out AnyBody_models --ammr-libdef "../../libdef.any"`, then
   `python tools/anybody_models_run.py AnyBody_models --only REG:1`. Expected: `hip_ideal` max load 2036 N (+/- 2 %), strap-to-knee 0.438 m,
   knee dots 71.4, thigh 7.8, shank 10.8 deg; `hip_free` 555 N. If it does not load, the message names the line: fix it in the shared
   `SeatedKnee/` files, or set `WITH_CHECKS` to 0 in `Position.any` if the line is in `Checks.any`. If it loads but the numbers differ, nothing else
   is trusted until they match. Compare with the group's own run, line by line, before changing anything.
2. **One real position** (`--only P01:2`). Open `Main.any`, run the inverse dynamics once and look at the model view: a blue arrow (the pull) on
   the shank, pointing backwards for knee extension. Then the script. Read `status` and the `check_*` columns.
3. The rest. 4. Write below what you had to change.

## Proven by the group's run (do not re-test, only keep)

Hard drivers and `BM_ARM_DETAILED_HAND OFF`; every default driver with reaction on except the tested knee; trunk and arm muscles on; the other leg
without muscles, held by actuators; `_SCALING_LENGTHMASSFAT_` with `Main.HumanModel.Anthropometrics` set before the human model loads; the strap
node (`KneeJointAnatomicalFrame` axes, malleolus tip plus 0.05 m) and `Flocal = {-F, 0, 0}`; `MR_MinMaxStrict`, `nStep = 1`; `EvaluateMaxLoad` at 800
and 1600 N; the left leg tested.

## Ours: assumed, to test

| # | Assumed | How to test | If it fails |
|---|---|---|---|
| B1 | The right leg works as the left does: the ISB shank frame has x anterior, y proximal, z right for both legs (the AMMR mirrors with a sign) | a right-tested position: `check_pull_dir_diff_deg` < 2 deg and the arrow points backwards for knee extension | the strap force sign for the right leg in `JointsAndDrivers.any` |
| B2 | A pull angle other than 90 deg: `Flocal = {PullSideX*sin(a)*F, -cos(a)*F, 0}` (y is proximal, so -cos is towards the feet) | `check_pull_angle_deg` equals `PullAngle_deg`; try `--set pull_angle_deg=75` | the sign of one component |
| B3 | `Main_HipFree.any` (a `#define` before the includes) switches off exactly the three tested-hip driver reactions | regression `hip_free` gives 555 N | the `#ifndef MVIC_HIP_FREE` blocks |
| B4 | `Main.Ops.UseMeasuredForce` / `UseProbeLoad` (AnyOperationSetValue, as in the AMMR tool) followed by `classoperation Main "Update Values"` set the strap load before each run | `check_moment_nm` uses the measured force; `max_load_n` does not depend on the measured force | the macro lines in `macro_for` |
| B5 | The load line can be dumped: `Main.MaxLoading.Calculations.{Load1, Load2, MaxMuscleActivity1, MaxMuscleActivity2}` | `scale_needed_line` is filled and `max_load_n` equals `(1 - intercept) / slope` | if the dumps fail, `scale_needed_ratio` is still there |
| B6 | `Checks.any`: `Itf.KneeFlexion.Pos[0]`, `FibularHead` node on the shank, `abs`, `atan2`, `acos(uDistal*PullDir')` | loads; `check_knee_flexion_deg` equals the asked angle | set `WITH_CHECKS` to 0 and tell us which line |
| B7 | Hip 90 and ankle 0 (nominal) instead of the group's photo values (82.2, 25 plantarflexed) | in the regression case compare `--set hip_deg=82.2 --set ankle_deg=-25` (the default) with 90 and 0: the change in `max_load_n` is the cost of the nominal values | use measured thigh orientation (not in the export yet) |
| B8 | `ProbeLoad_N` 800 fits every participant: above about 300 N, and not so far above the capacity that the line bends | `max_load_n` is not far from 800 to 2400; the line from two points is the same as from 1000 and 2000 | `--probe-load` |
| B9 | AnyPyTools: macro lines as strings; dumped values come back under their full names | `results.csv` has numbers | adapt `macro_for` and `_first` in `tools/anybody_models_run.py` |
| B10 | Tape thigh and shank length (`--segment-lengths`) are used for both legs | compare `Check.ShankLength_m` with the tape; run once with and once without and read the change in `max_load_n` | leave it off |

## Decisions that are the group's, not the code's

- **D1 range of the hip.** Both versions are run. The strength factor is a range. If the group later learns how stiff the strap and belt are, a
  third version between them is a small change in `JointsAndDrivers.any` (a weak reaction instead of on or off).
- **D2 force or moment kept.** The model keeps the force and the strap place; its lever arm follows its own scaled shank, so its moment is not
  exactly the app's. `check_moment_nm` shows the gap. To keep the moment instead, scale the force by (app lever arm / model lever arm).
- **D3 angle at peak force or plateau mean.** The export gives the angle at the peak force (HANDOFF: undecided). Type the plateau mean over with
  `--set knee_deg=`.
- **D4 total or net force.** Total, with gravity in the model (SOP 9.10). Never subtract the resting force and keep gravity.
- **D5 whole-leg factor or group factors.** The result is one factor for all leg muscles (`BM_LEG_STRENGTH_INDEX` scales them all and is a design
  variable). The group scaling of `PRIMARY.tex` needs the muscles of each group scaled alone; in the AMMR the force of every muscle element is
  built from volume, fibre length and `StrengthIndexLeg`, so per-group scaling needs a change in how the muscle models are built.
- **D6 knee zero offset.** `knee_zero_offset_deg` in the results is the gap between limbtrack's dot-based knee angle and the model's, per position.
  The group's test gave 1.5 deg. Use `--knee-offset-deg` if it is more than a degree and stable across positions.
- **D7 hip from the thigh.** The group's photo gave the hip from the thigh's angle below horizontal (7.8 deg). The export has no such angle; the hip angle
  of the app uses the pelvis line, which is not trusted. Adding the thigh's angle from horizontal to the export would let the models use it.

## What was changed at the first run

(empty: write the date, who, which assumption, what was changed)
