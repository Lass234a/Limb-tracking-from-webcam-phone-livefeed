---
name: limbtrack-pilot-collection
description: Collects all correct (clean) MVIC measurements of the limbtrack pilot (participants P01 and P02, the Pilot/ folder) into one local collection with their data, positions and dots, plus an index of what was left out and why. Use this whenever someone asks for the correct, clean, good or usable pilot measurements, a collection or overview of the P01/P02 trials, which trials to use for further analysis (dynamometer or force trials, knee extension at a given angle, positions), the dots of those trials, or says things like "samle de korrekte målinger", "alle gode forsøg", "which trials can we use", "export the pilot trials", even if the skill is not named. Also use it to rebuild the collection when new pilot trials arrive or the criteria change.
---

# limbtrack pilot collection

Builds one folder with every technically clean MVIC measurement of the pilot, so further analysis starts from a short, trusted list
instead of 128 trial files. "Correct" is defined the same way every time, and every excluded trial is accounted for.

## Run it

From the project folder (the script is `tools/collect_pilot.py`; it uses only the Python standard library, so the project's `.venv` is fine):

```
.venv\Scripts\python.exe tools\collect_pilot.py
```

By default it reads `Pilot/` in the project folder (or in the current folder) and writes `Pilot_collection/` next to it; both are in
`.gitignore`. Options: `--pilot FOLDER`, `--out FOLDER`, `--participants P01 P02`, `--window 6 8`, `--min-rise 60`, `--no-copy`.
Check that the pilot folder holds all trials (`*_meta.json`, `*_data.csv`, `*_raw.mp4`) before trusting the counts: an incomplete copy
silently shrinks the collection. On the 128 pilot trials of P01 and P02 it finds 98 correct ones.

## What it writes

- `measurements.csv`: one row per correct measurement: participant, position number, test, target, trial, force (peak, rest, rise),
  tempo (rise 10-90 %, hold at 80 % of peak, class), main angle at the hold / at the peak / at rest, deviation from the target,
  dots used, dots lost, and the paths of the raw and overlay video.
- `dots.csv`: per dot per measurement: pictures in the window, lost pictures, median quality, partly covered pictures, position at the hold.
- `positions.csv`: per position: trials recorded and correct, which trials, mean peak force, mean hold angle and its spread between trials.
- `excluded.csv`: every trial left out, with the reason.
- `data/`: a copy of `meta.json`, `data.csv` and `force.csv` of each correct measurement. Videos are not copied.
- `README.md`: criteria, columns and caveats, regenerated each run.

## The criteria, and why

A measurement is correct when both MVIC markers exist, no landmark dot (strap and anchor excluded) was lost in the MVIC window, the
program flagged no CHECK DOTS or CHECK LABELS in it, and the force rises at least 60 N above its rest level. A lost landmark or a flagged
jump makes the hold angle untrustworthy; a trial without a force rise has no peak to analyse. The window length is checked too, but the
automatic end marker is always 7.0 s after the start, so it says nothing about the tempo: the tempo (1-5-1) is measured on the force curve.

## Say these things when you report results

"Correct" means clean and complete, not validated. The angles have been tested in several rounds but are not directly validated against a goniometer or inclinometer (SETUP_AND_VALIDATION.md,
section 4). A small spread inside a hold only shows that the tracking is steady. Ankle tests have no main angle in `data.csv`, and the hip
angles read far from their targets (the pelvis-line definition issue in HANDOFF.md). When asked whether the angles were right, compare
`hold_angle_deg` with `target_deg` and with the other trials of the same position, and say that absolute accuracy is unproven.

## Keep it local

The collection holds participant codes, dates and paths to participant video. The repository is public: never commit the collection, the
`Pilot/` folder or any video. The script refuses to write into a non-empty folder it did not create itself.
