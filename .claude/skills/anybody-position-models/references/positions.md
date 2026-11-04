# The tests, and what each needs from the model

Source of the set-ups: `SOP.tex` (KVT3-protokol), "Test set-ups", and `protocol.json`. Axes in the model: x forward, y up, z to the
participant's right. `effort axis` = the sign of the rotation about z that the effort gives the segment the strap acts on (the code uses it to
put the strap's sideways force on the side that resists).

`python tools/anybody_models.py --list-tests` prints the same status from the code. The registry (`REGISTRY` in `tools/anybody_models.py`) is
checked against `protocol.json` by a test, so a new test in the protocol cannot be forgotten.

## Built: the seated knee model (`tools/anybody_templates/SeatedKnee`)

| Test | Positions | Posture | Strap and load cell | Effort axis |
|---|---|---|---|---|
| Knee extension | knee 20, 70, 110; hip 90 | seated on the bench edge, trunk upright, hands on the bench, thigh strap over the distal femur | strap on the distal shank just above the malleoli; chain to the bench frame, anchor moved so the pull is about square to the shank | +1 |
| Knee flexion | knee 20, 90; hip 90 | the same | the same strap; load cell on the power rack in front, high at knee 20, at ankle height at knee 90 | -1 (same model; not looked at yet) |

What holds the body: pelvis (six reactions), distal thigh (a point support). What drives the posture: the AMMR mannequin (hip, knee, ankle).

## Planned: what differs

| Test | Positions | Strap on | Effort axis | Differs from the seated knee model |
|---|---|---|---|---|
| Hip flexion, knee 90 | hip 90 (seated), 0 (supine) | distal thigh, load cell straight down | +1 | the strap moves from the shank to the thigh, and the thigh fixation goes (the pelvic belt holds the pelvis); supine at hip 0 turns the body, so gravity no longer points along the trunk |
| Hip extension, knee straight | hip 0, 45 | distal shank, chain horizontal | -1 | standing on the other leg: the stance leg and foot need support (foot reactions as in `Tools/Templates/Human Standing`); at hip 45 the trunk rests on the bench (trunk support) |
| Hip extension, knee 90 | hip 0, 90 | distal shank | -1 | as above with the knee bent; Figure 10 shows hip 0 standing but the SOP text says prone (TBD in `limbtrack/anybody.py`): decide first |
| Hip abduction | hip 0; abduction 0, 20 | distal thigh/shank, frontal | frontal plane (axis x) | the force and tangent are in the frontal plane; limbtrack's goniometer and hip-centre angles share no zero: agree with AnyBody which one is `HipAbduction` |
| Hip adduction | held at abduction 0, 20 | as abduction | frontal plane | as abduction, effort reversed |
| Plantarflexion, knee straight | ankle 0, -20 | under the foot, from a hip belt | -1 | standing on the tested leg: the whole body weight is on the foot; ankle sign is turned (A4) |
| Dorsiflexion | ankle 0, -20 | foot dorsum, padded block | +1 | the pilot used knee 90, the protocol knee 0: decide first |
| Belt squat (validation) | knee 90, 60 | hip belt, load through the chain | n/a | whole body, both feet; not used for scaling; forces to 3.8 kN; a different model |

## Order that costs least

1. Knee extension (done, to be run). 2. Knee flexion (same model). 3. Hip flexion at hip 90 (seated: the strap node moves to the thigh and the
thigh fixation goes). 4. Plantarflexion and dorsiflexion (foot). 5. The standing hip tests (a stance-leg support is new). 6. The frontal
tests (a new force geometry and a decision on the pivot). 7. The belt squat as a validation model, last.

## Scaling groups (from `PRIMARY.tex`)

Knee extension scales the knee extensors; knee flexion the hamstrings; hip extension the gluteus maximus (and with the knee straight the
hamstrings too); hip flexion the iliopsoas group; plantarflexion the triceps surae; dorsiflexion tibialis anterior; abduction gluteus medius;
adduction the adductors. The code records the group in each manifest entry (`scaling_group`); the scaling itself is not built.
