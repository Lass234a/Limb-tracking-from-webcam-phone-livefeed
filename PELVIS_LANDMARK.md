# Why the iliac crest dot is a valid pelvis landmark for the hip angle

Written 2026-10-15 for the supervisor and the AnyBody side, from the protocol draft of 2026-09-28 and the papers listed at the end.
Plain English first; the technical detail is in `CHANGELOG.md` (0.9.0) and `MARKERS.md`. **Every statement below that is my own
reasoning and not something a paper shows is marked as such.** Papers marked *(PubMed)* were looked up and their abstracts read;
the book (Norkin & White) was not available to me.

## The claim in one paragraph

The hip angle the program reports is the goniometric hip angle of the protocol's Tabel 3: the angle between the **lateral midline of
the pelvis** and the **lateral midline of the femur**, with the greater trochanter as the fulcrum. A line needs two points. One is the
greater trochanter. The other is **a dot on the iliac crest, halfway between the ASIS and the PSIS**, on the side of the pelvis facing
the camera. The line from that dot down to the trochanter is the lateral midline of the pelvis, drawn in the picture. This is a
reading of the protocol's own words, it does not need a new landmark that the protocol lacks, and it replaces a definition (the
perpendicular to the ASIS-PSIS line) that the protocol never asked for.

## 1. What the protocol says

Tabel 3, row *Hip flexion and extension*: "Fulcrum: greater trochanter reference. Proximal: lateral midline of the pelvis. Distal:
lateral midline of the femur, towards the lateral femoral epicondyle." Section 2.9: the same examiner palpates and marks all landmarks
before testing, to limit inter-examiner variation [6], and Tabel 3 "is based on Norkin and White [7]".

The ASIS and PSIS appear in Tabel 3 only for two other things: the fulcrum of hip abduction/adduction (ASIS) and the **pelvic tilt**
row (the ASIS-PSIS line against the horizontal [8]). They are **not** how the table defines the pelvis arm of the hip angle. The
program's earlier hip angle (pelvis arm = the perpendicular to the ASIS-PSIS line) was the software's approximation of "lateral
midline of the pelvis", not the protocol's definition.

## 2. The argument, step by step

**Step 1: the target is the protocol's goniometric angle.** The positions in Tabel 2 (hip 90 degrees in the knee tests, and so on) are
positions of that goniometric angle. A measurement is valid for the protocol if it estimates the same angle the examiner would read
on a goniometer set up as Tabel 3 describes.

**Step 2: the crest dot plus the trochanter dot is that goniometer's stationary arm.** A goniometer's stationary arm lies along the
lateral midline of the pelvis, through the trochanter. The iliac crest is the long bony ridge along the top of the pelvis; it is
under the skin and easy to feel. A dot at the middle of the crest, above the trochanter, and the trochanter dot define a line along
the side of the pelvis in the picture. *This step is my reading of the protocol's wording* (the protocol does not say "midpoint of
the crest"); the exact way Norkin & White place the arm should be checked in the book. Secondary sources I found describe the arm
as lying along the lateral midline of the pelvis, some as running along the side of the trunk.

**Step 3: the dot is placed from two landmarks the examiner already finds, and the sensitive direction is averaged.** The halfway
point is measured with a tape between two bone points the examiner palpates anyway. The crest runs front to back, so the placement
error that matters for the angle is a front-to-back shift of the dot (it rotates the near-vertical crest-to-trochanter line).
*My reasoning:* if the ASIS and the PSIS are each placed with an independent front-to-back error, the halfway point has about 0.7 of
that error, and the angle error is roughly the placement error divided by the distance between the crest dot and the trochanter
(5 mm of shift over 12 cm is about 2.4 degrees). The old definition depended instead on the *height difference* between ASIS and PSIS
over their distance, so both definitions are sensitive to palpation; neither is free of it. **The ASIS and PSIS still have to be
palpated once, to find the halfway point.** What changes is that they are no longer stickers that a side camera has to see edge-on.

**Step 4: the old reference was not free of person-to-person offset either.** According to PubMed, Preece et al. (2008) measured 30
cadaver pelves in a fixed anatomical position and found the ASIS-PSIS line at 0 to 23 degrees (mean 13, SD 5) to the horizontal
[Preece 2008, [DOI](https://doi.org/10.1179/106698108790818459)]. So "perpendicular to the ASIS-PSIS line" is not a fixed pelvic axis:
it carries a person-to-person offset of a few degrees relative to any axis fixed in the pelvis. The lateral-midline definition is a
different convention with its own offset, not a perfect one.

**Step 5: the measurement is a single-axis, sagittal flexion/extension angle in a held posture.** According to PubMed, Leardini et al.
(2005) review skin-marker artefact (the skin sliding over the bone) as "the most critical source of error" in marker-based movement
analysis, greatest at the thigh, task dependent, and conclude that in vivo only flexion/extension about the hip, knee and ankle
axes can be determined reliably [Leardini 2005, [DOI](https://doi.org/10.1016/j.gaitpost.2004.05.002)]. The program measures exactly
that component (the angle in the sagittal plane), and the protocol holds each position isometrically. *My reasoning:* a held posture
involves less skin movement than a swung limb. **How much the crest and trochanter dots slide during a maximal contraction has not
been measured here.**

**Step 6: the camera set-up is the validated one.** According to PubMed, Puig-Diví et al. (2019) found Kinovea angle and distance
measurements valid and reliable up to 5 m and with the camera at 90 to 45 degrees to the movement plane, best at 90
[Puig-Diví 2019, [DOI](https://doi.org/10.1371/journal.pone.0216448)]; the protocol's own set-up rule is a camera square to the
plane. *My reasoning:* the crest dot and the trochanter dot are both on the side of the body facing the camera, at nearly the same
distance, so they do not suffer the depth difference that a front dot (ASIS) and a back dot (PSIS) can have.

## 3. What this does not show (please read)

1. **Equivalence with the AnyBody pelvis frame is not shown.** The ISB recommendation for the hip joint coordinate system [Wu 2002,
   [DOI](https://doi.org/10.1016/s0021-9290(01)00222-6), PubMed] defines pelvis and femur axes for reporting hip motion; as far as I
   know the pelvis axes there are built from pelvic landmarks including the ASIS and PSIS (*from memory; the PubMed abstract does not
   say, and the AnyBody model's own pelvis frame must be checked with AnyBody*). The crest-to-trochanter line will differ from a
   model's pelvis axis by an offset that varies between people. It is unknown here, and it applies to the absolute hip angle, not to
   the difference between two positions of the same person.
2. **Palpation is not made easy.** According to PubMed, Moriguchi et al. found the greater trochanter, the ASIS, C7 and the femoral
   epicondyle showed the larger palpation discrepancies, the ASIS discrepancy was larger in the overweight group, and the angular
   variation of hip flexion and pelvic inclination followed the palpation discrepancy; they advise a single rater
   [Moriguchi 2008/2009, [DOI](https://doi.org/10.1016/j.math.2008.04.002)]. The protocol already uses one examiner. The abstract
   does not list the iliac crest midpoint among the landmarks, so **its repeatability is unknown here**.
3. **Nothing has been tried on a person yet.** All checks of the new hip definition are synthetic (the program reads the true hip
   angle of 40 made-up poses to within a millionth of a degree). There is no comparison with a goniometer or inclinometer.
4. **Pelvic tilt is no longer measured.** Tabel 3 lists it (ASIS-PSIS against the horizontal [8]) and needs two pelvis points. Either
   the protocol drops it, or it is measured another way (for example with the inclinometer on the ASIS and PSIS, as in [8]).
   *A decision for you and the supervisor.*
5. **Hip angles before 0.9.0 and after are different quantities** and must not be pooled.

## 4. Three checks that would turn the argument into evidence

1. **Placement repeatability:** the same examiner marks the crest dot on the same person on two days (or twice in a day after removing
   it); measure the distance between the two marks. Within 5 mm keeps the hip within about 2.4 degrees at 12 cm.
2. **Agreement with the reference instrument:** video hip angle against the digital inclinometer (protocol section 2.8) at three hip
   angles in about ten participants; report the mean difference and its spread.
3. **Offset to the AnyBody pelvis:** in the scaled model, the angle between the line (crest midpoint to trochanter) and the model's
   pelvis axis in the neutral pose, over the participants in the study. If it is nearly constant it can be reported once.

## 5. A paragraph that can go in the protocol (draft, adapt freely)

"Hip flexion and extension are measured in the sagittal plane as the angle between the lateral midline of the pelvis and the lateral
midline of the femur, with the greater trochanter as the fulcrum, following Norkin and White [7]. The lateral midline of the pelvis
is the line from the greater trochanter to a point on the iliac crest halfway between the anterior and the posterior superior iliac
spines, marked by the same examiner before testing [6]. The femoral line runs from the greater trochanter to the lateral femoral
epicondyle. Pelvic tilt is not derived from this angle."

## References

- Norkin CC, White DJ. *Measurement of Joint Motion: A Guide to Goniometry*. 5th ed. F. A. Davis; 2016. [7 in the protocol; not read for this note]
- Moriguchi CS et al. Reliability of intra- and inter-rater palpation discrepancy and estimation of its effects on joint angle
  measurements. *Man Ther* 2009;14(3):299-305. [DOI](https://doi.org/10.1016/j.math.2008.04.002) (PubMed 18555729) [6 in the protocol]
- Herrington L. Assessment of the degree of pelvic tilt within a normal asymptomatic population. *Man Ther* 2011;16(6):646-8.
  [DOI](https://doi.org/10.1016/j.math.2011.04.006) (PubMed 21658988) [8 in the protocol: pelvic angle as the ASIS-PSIS line from the horizontal]
- Preece SJ et al. Variation in pelvic morphology may prevent the identification of anterior pelvic tilt. *J Man Manip Ther*
  2008;16(2):113-7. [DOI](https://doi.org/10.1179/106698108790818459) (PubMed 19119397)
- Leardini A, Chiari L, Della Croce U, Cappozzo A. Human movement analysis using stereophotogrammetry. Part 3. Soft tissue artifact
  assessment and compensation. *Gait Posture* 2005;21(2):212-25. [DOI](https://doi.org/10.1016/j.gaitpost.2004.05.002) (PubMed 15639400)
- Puig-Diví A et al. Validity and reliability of the Kinovea program in obtaining angles and distances using coordinates in 4
  perspectives. *PLoS One* 2019;14(6):e0216448. [DOI](https://doi.org/10.1371/journal.pone.0216448) (PubMed 31166989)
- Wu G et al. ISB recommendation on definitions of joint coordinate system of various joints for the reporting of human joint
  motion, part I: ankle, hip, and spine. *J Biomech* 2002;35(4):543-8. [DOI](https://doi.org/10.1016/s0021-9290(01)00222-6)
  (PubMed 11934426)
