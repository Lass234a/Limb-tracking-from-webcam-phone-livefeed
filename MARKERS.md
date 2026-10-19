# Making and placing the dots (tape dots)

Written 2026-10-13 for the first real sessions, when black sticker dots could not be sourced. The plan: **a square of white sports
tape on the skin, with a solid black disc drawn on it by hand, centred on the bone point.** This page says how to make and place
them, what was tested (on synthetic pictures only, no real camera yet) and what is still unknown. The tracker itself is unchanged:
it looks for a dark round spot, and a white square behind it only makes that spot stand out more, whatever the skin or the shorts.

## 1. One tape dot

| Part | Rule | Why |
|---|---|---|
| Tape | White, matt sports tape (cotton or rigid athletic tape). Not shiny kinesiology or sports tape with a gloss | A lamp reflecting in a shiny disc makes a bright spot inside the black one |
| Square | About **3.5 to 4 cm** a side, the disc in the middle | Bigger adds nothing and runs into the next dot (at the knee the dots are only a few cm apart) |
| Disc | **Solid black, about 2.2 cm** (the protocol minimum is 2 cm; drawing a little over lets the size check pass). A Danish 2-krone coin is about 21.5 mm: check it with a ruler and draw round it, or cut a hole of that size in an old plastic card and use it as a stencil | The hole in the card also lets you see the bone point while you place it |
| Ink | Permanent marker, matt, **two coats**, let it dry before the participant moves | Streaks and thin spots are harmless; a shiny or wet disc is not |
| Fill | **Fill it in completely. Never use an outline circle** | See the tests in section 3: an outline can make the program follow the white tape instead of the ink, with no warning |
| Program setting | **Dot colour = Black dots** (not Auto) | Same reason: it removes the one way the tape can be mistaken for the dot |

Order that works: palpate the bone point, put the tape over it, palpate again through the tape, put the stencil's hole over the
point, draw round it and fill it in. Do all the tape dots in the test posture (the existing rule: one person places everything).

## 2. Where each dot goes

- **Trochanter, lateral epicondyle, fibular head, lateral malleolus**: on the outer side of the leg, so they already face a side
  camera. Because the epicondyle and the fibular head are only a few cm apart, use **one strip of tape with two discs** instead of two
  squares. Do the same along the outer edge of the foot for the 5th metatarsal base and head.
- **Iliac crest dot (the pelvis dot; since 0.9.0 it replaces ASIS and PSIS; the argument for it is in `PELVIS_LANDMARK.md`).** Place it on the side of the pelvis, **on the iliac crest, halfway between the ASIS and the PSIS**. A simple way: palpate both bone points and mark each with a pen dot on the skin, measure along the crest between the two marks with a flexible tape (or a string laid on the skin) and mark the middle; put the tape square and disc on that mark. The side of the crest faces a side camera more than the ASIS or PSIS themselves do, so the disc is less edge-on than it would be there.
  The hip angle is now measured from the line **iliac crest dot to trochanter dot** (the goniometer's lateral midline of the pelvis) against the thigh, so the angle depends on where this one dot is put: with about 12 cm between the two dots, 5 mm of sideways error is about 2.4 deg. One person places both dots, in the test posture, and the same person repeats it every session.
- **No strip or line between the pelvis dots any more.** There is only one pelvis dot. Still true for any two dots that are near each other (the epicondyle and fibular head on one strip): **do not draw a black line along the strip, and do not draw a line between any two dots.** A thick marker line (about 4 mm) joins two discs into one long dark shape: in the test the marked centre landed up to 1.7 cm away from the disc, with no warning, or the dot was lost. A thin pen line (2 mm) did no harm, but there is no gain in drawing one, and the margin depends on how sharp the picture is.
- **Strap dot:** a wrap of white tape round the strap, on the part that faces the camera, a few cm from the buckle, with the disc on
  it (the dot on the buckle or under the sock was the one that was lost). **Anchor dot:** a tape patch with a disc at the fixed
  end, or on the rope itself, as far from the strap dot as the picture allows, if the anchor is out of frame. A pulled strap or
  rope is straight, so two dots define it; a black line along it would not add anything.
- **Ruler:** the same tape dots at its two ends.

## 3. What was tested, and what it showed

Synthetic pictures (a skin-coloured background, 6.5 pixels per cm, a 2 cm disc on a white square), the real tracker code. The
tracked centre was compared with the true centre over a moving 120-picture sequence.

| Case | Result |
|---|---|
| Black disc on 3, 4, 5, 6 cm tape squares, tape turned 20 deg | No lost pictures, the centre as exact as for a bare black disc (about 0.1 px, 0.2 mm). Tape size does not matter to the tracker |
| Streaky, hand-filled disc | Same |
| More noise, a soft (blurred) picture, dim light (45 %) | Same, with and without tape |
| Disc seen side-on (70 % and 50 % as wide as tall), as a disc on the front or back of the hip would be | Tracked without loss, same accuracy. The size check will read it smaller (see below) |
| **Thin outline circle, Auto colour, soft picture, circle not exactly in the middle of the tape** | **The program followed the white tape, 11 mm away from the circle, with no picture lost and no warning.** With *Dot colour = Black* the same case was exact |
| Outline circle with a gap in it | Centre pulled 1.3 mm (40 deg gap) to 3 mm (90 deg gap) towards the closed side |
| Two discs on one tape strip joined by a black line, 3.5, 6 and 15 cm apart | 2 mm line: no effect. **4 mm line: marked up to 1.7 cm off the disc (nothing lost, no warning), or marking failed, or the dot was lost in every picture** |
| Filled disc, Auto colour, off-centre on the tape, sharp or soft | Correct (black chosen every time) |

Not tested: real glare on tape, creased or peeling tape, sweat, skin sliding over the bone, a real camera. Tape and ink can also
shift on the skin during a contraction; check the dots on the video after each trial (Review's trails and *Dots and pull* tab).

**The dot-size check (Live tab, with a ruler marked) was reading about 0.2 cm too small** for any dot, because it subtracted an
allowance that came from the test pictures being drawn about 1 px too large. It was corrected in 0.7.2. It is still an estimate: draw
a disc with the stencil, mark it, and see what the program says; if it reads far from what you drew, tell me. A disc seen side-on
reads narrower than 2 cm; that is the picture, not a wrong sticker (the tall side is the true width).

## 4. Why not a line instead of dots

- A line (a black stripe on a tape strip, tracked as a line) would only help if the **end points** could not be seen. It would not
  make the angle more accurate: the tracker's own scatter is about 0.2 mm, which on a 12 cm iliac crest to trochanter line is under 0.1 deg, whereas
  a 5 mm placement error at one end is about 2.4 deg. The error that matters is where the tape and disc are put on the body.
- A line detector would be new code and a new thing to validate, and the strap, anchor, angles, Review and lost-dot rule all work on points.
- If the first real session shows that the iliac crest dot is lost or off from the side, a line detector is the next thing to
  build. Until then: dots.
- **Done in 0.9.0 (2026-10-15, your decision):** the goniometer's own proximal arm in Tabel 3 is "the lateral midline of the pelvis", so
  one dot on the side of the iliac crest, halfway between ASIS and PSIS, replaced ASIS and PSIS in the hip angle (and the pelvic tilt
  angle, which needs two pelvis dots, was dropped). That changed the hip angle's definition: make sure the supervisor and the AnyBody
  side know, and note that hip angles from before 0.9.0 (measured against the perpendicular of the ASIS-PSIS line) are not directly
  comparable with hip angles from 0.9.0 on.

## 5. Quick check at the start of a session

1. Dot colour is **Black dots**; the ruler is marked; every dot reads at least 2.0 cm in the scale box.
2. After marking, every dot's ring is yellow (strong), none orange ("weak" or "partly covered").
3. Move the hip and the leg by hand: no dot goes grey. If the iliac crest dot is lost when the hip moves, move its disc further out on the side of the hip and mark it again.
4. Record a few seconds, open it in Review, look at the *Check* tab.
