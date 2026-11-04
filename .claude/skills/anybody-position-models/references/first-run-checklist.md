# First run in AnyBody: what was assumed, and how to test it

The models were written without AnyBody. The names below were checked by text search in the AMMR 3.2.1 source (it needs AnyBody 8.2 or newer,
which is the version of `Documents\AnyBody.8.2.x` in HANDOFF.md). Behaviour was not tested. Do this on ONE position first
(`anybody_models_run.py AnyBody_models --only P01:2`), then on the rest.

## Order of work

1. `--dry-run`: read the macro. 2. Open `P01/pos02.../Main.any` in AnyBody. If it does not load, the message names the line: fix it in the
shared `SeatedKnee/` files (or set `WITH_CHECKS` to 0 in `Position.any` if the line is in `Checks.any`). 3. Run `InverseDynamics` once by
hand and look at the model view: a blue arrow (the pull) on the shank, the shank at the knee angle, the thigh horizontal. 4. Run the script and
read `status` and the `check_*` columns in `results.csv`. 5. Write below what you had to change.

## The assumptions

| # | Assumed | Basis in the AMMR 3.2.1 source | How to test | If it fails |
|---|---|---|---|---|
| A1 | The AMMR is found through `libdef.any` | `Tools/Templates/libdef.any` | model loads | `--ammr-libdef PATH` or edit `SeatedKnee/libdef.any` |
| A2 | Posture is set with `HumanModel.Mannequin.Posture.Right.{HipFlexion, HipAbduction, HipExternalRotation, KneeFlexion, AnklePlantarFlexion, SubTalarEversion}` in degrees; the left leg mirrors | `Tools/Templates/Human/Model/Mannequin.any`, `BodyModels/Include/MannequinDrivers.any` (`pi/180*`) | `check_knee_flexion_deg` equals the asked angle | if it differs by a constant, the AMMR zero differs from limbtrack's (A3) |
| A3 | limbtrack's knee angle (0 = straight, + = flexion, from the standing reference) equals AnyBody's `KneeFlexion` | both + flexion; the zero of the TLEM2 model may sit a few degrees from the anatomical zero | `check_knee_flexion_deg_diff` < 1 deg at knee 20, 70 and 110 | add a constant offset in `Mannequin.any` and write it here |
| A4 | Ankle: limbtrack + = dorsiflexion, AnyBody `AnklePlantarFlexion` + = plantarflexion, so the sign is turned (HANDOFF and SETUP_AND_VALIDATION say the same) | `Interface.any` of the leg | `check_ankle_plantar_flexion_deg` | change `sign` in `SEATED_KNEE_DRIVES` |
| A5 | Hip is 90 deg and the ankle 0 deg (nominal), not the measured values | the app's pelvis line reads far from the target (HANDOFF.md); the protocol gives no ankle angle for the knee tests | manifest `measured` shows what the app read | `--set hip_deg=...` or edit `Position.any` |
| A6 | Pelvis fixed and upright; six reactions on it | `Tools/Templates/Human/Model/Reactions.any` | loads and solves | add the reactions the solver names |
| A7 | The thigh strap and bench act as a point support (3 forces) at `ThighFixationDist_m` = 0.10 m above the knee | none: our choice. Without it the hip muscles carry the knee moment | `MaxMuscleActivity` and which muscle limits: it must be a knee extensor. If a hip muscle limits, the support is wrong | move the node, or use hip moment reactions |
| A8 | Left leg and both arms off; trunk (with muscles) stays | `BM_LEG_LEFT`, `BM_ARM_*` in `Docs/bm_config`; `BikeModel-LowerBody.main.any` | loads | switch the trunk muscles off (`BM_TRUNK_MUSCLES OFF`) if it is slow |
| A9 | Simple muscles, TLEM2 leg, `_SCALING_LENGTHMASSFAT_`; `Main.HumanModel.Anthropometrics` set before the human model loads | `BodyModel.defaults.any`, `Scaling/DefaultAnthropometrics.any` (`??=`), `ScalingLengthMassFat.any` | dump `Main.HumanModel.Anthropometrics.BodyMass` and the thigh length; compare `Check.ShankLength_m` with the tape shank | if AMMR ignores the segment lengths, remove the `SegmentDimensions` block and say so |
| A10 | The force is a pure force at a node on the shank, **total** from the load cell zero (not above rest) | SOP 9.10; `anybody.py` D4 | `check_moment_nm` against the app's `lever_arm_app_m` x force | see D1 below |
| A11 | The pull is in the sagittal plane, at the stated angle to the shank, on the side that resists the effort | `Loads.any`; tested in Python against the moment sign | `check_pull_dir_diff_deg` < 2 deg; the arrow points backward for knee extension | the model's neutral angles differ from the ideal ones used in Python: compare and decide |
| A12 | Strap is `StrapDist_m` above the ankle joint (lateral malleolus taken as the ankle axis) along the line to the knee joint | `Seg.Shank.KneeJoint`, `.AnkleJoint` nodes | `Check.StrapToKnee_m` against shank tape minus strap distance | adjust `StrapDist_m` |
| A13 | `MR_MinMaxStrict`, `nStep = 1`, `KinSolOverDeterminate` | the AMMR human template and `EvaluateMaxLoad` | solves | per the message |
| A14 | `EvaluateMaxLoad` with `LoadPtr = &Main.Position.Force_N` (a `DesignVar`) | `Application/Validation/EvaluateMaxExternalLoading/EvalMaxLoad.main.any` | `max_load_n` is sensible: `force_over_max_load` near 1 for a person of the model's strength | set `WITH_MAXLOAD` to 0 |
| A15 | AnyPyTools: macro lines as strings; dumped values come back under their full names | AnyPyTools user guide (anybody-research-group.github.io/AnyPyTools) | `results.csv` has numbers | adapt `macro_for` and `_first` in `tools/anybody_models_run.py` |

Names confirmed in the AMMR source and not assumed: `Main.HumanModel.BodyModel.Trunk.SegmentsLumbar.PelvisSeg`,
`...BodyModel.Right.Leg.Seg.{Thigh.HipJoint, Thigh.KneeJoint, Shank.KneeJoint, Shank.AnkleJoint}` (with `sRel`), `...BodyModel.Interface.Right.KneeFlexion`
(a measure, read with `.Pos[0]*180/pi` as AMMR does), `HumanModel.StrengthParameters.StrengthIndexLeg`, `AnyForce3D` with `AnyRefFrame &ref` and
`F`, `EvaluateMaxLoad`, `MR_MinMaxStrict`.

## Decisions that are the group's, not the code's

- **D1 force or moment kept.** The model keeps the force and the strap place; its lever arm follows its own scaled shank, so its moment is
  not exactly the app's. If the two differ by more than a few percent, either scale the force by (app lever arm / model lever arm) to keep the
  moment, or keep as is and report both. `check_moment_nm` shows the size of the gap.
- **D2 angle at peak force or plateau mean.** The export gives the angle at the peak force (HANDOFF: the group has not decided). The
  plateau mean is in the session summary and can be typed over with `--set knee_deg=`.
- **D3 total or net force.** Total, with gravity in the model (SOP 9.10 recommends it). Never subtract the resting force and keep gravity.
- **D4 whole-leg factor or group factors.** `max_muscle_activity` is one factor for all leg muscles. The group scaling of `PRIMARY.tex`
  needs the muscles of each group scaled alone. `Main.HumanModel.StrengthParameters.StrengthIndexLeg` scales all leg muscles and is a design
  variable (AnyPyTools `SetValue`), so a whole-leg search needs no reload. Per-group F0 needs a change in how the muscle models are built.

## What was changed at the first run

(empty: write the date, who, which assumption, what was changed)
