"""The session check (0.34.0, plan feature F7, W2): before the participant leaves, what is still missing?

From the trials saved for one participant and the session plan (`protocol.plan_positions`) it lists, in plain words:
  * positions never recorded (the P02 case: hip flexion 0 was never recorded and nobody noticed until the review),
  * positions whose trials are all marked compromised, or have no force rise (nothing counts there),
  * positions that need a third trial (the two peaks differ by more than 10 %, SOP 10.6) or have only one counted trial,
  * positions with trials that are not marked valid yet (they will not go into the AnyBody export until they are),
  * missing reference readings (a goniometer or inclinometer reading of the main joint, SOP 9.3),
  * missing on-site measurements: the participant measures (F2: height, mass, body fat, thigh and shank length) and, per position, where the strap
    sat (F3),
  * a standing reference not taken (F4), and the tested leg: not recorded, or different in different trials.
It only READS files and numbers (rows of `analysis.session_row`, meta.json, the verdict files); it never blocks anything and never writes. The
window shows the result as a list the operator can read (Live: the *Session check* button next to the session plan; Review: the participant
overview). Levels: `todo` (record or redo something), `gap` (a number the export will lack), `info`.
"""
import json
from pathlib import Path

from . import analysis, participant_measures, reference, review_nav
from . import verdict as verdicts

REQUIRED_MEASURES = ("height_cm", "mass_kg", "fat_pct", "thigh_length_cm", "shank_length_cm")      # what the AnyBody scaling uses (plan M1)
NO_STRAP_PLACE = ("belt_squat",)                  # a belt and chain, not a strap at a named landmark: no strap place is asked for
LEVELS = ("todo", "gap", "info")


def default_tests(path=None):
    """The tests of the project's protocol.json (or `path`), or None when it cannot be read. Used where the window has no engine at hand
    (Review, the tool)."""
    from .protocol import load_protocol
    try:
        return load_protocol(Path(path) if path else Path(__file__).resolve().parent.parent / "protocol.json")
    except Exception:  # noqa: BLE001 - the check then runs on the positions found in the trials, and says nothing about the rest
        return None


def default_plan(path=None):
    """The session plan [(test key, target)] of the project's protocol.json, or None when it cannot be read."""
    from .protocol import plan_positions
    tests = default_tests(path)
    return plan_positions(tests) if tests else None


def titles_of(tests):
    """{test key: label} from a protocol's tests (for the lines of the list)."""
    return {k: t.label.replace(" (MVIC)", "") for k, t in tests.items()}


def _same(a, b):
    return analysis._num(a) is None and analysis._num(b) is None or (
        analysis._num(a) is not None and analysis._num(b) is not None and abs(float(a) - float(b)) < 1e-6)


def gather(items):
    """The facts of every trial from `items` = [(meta_path, export row or None)] (oldest first). A trial whose meta.json cannot be read is left
    out and named in the second result: (facts, [unreadable trial names])."""
    facts, bad = [], []
    for meta_path, row in items:
        p = Path(meta_path)
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("not a trial description")
        except (OSError, ValueError):
            bad.append(p.name[: -len("_meta.json")] if p.name.endswith("_meta.json") else p.name)
            continue
        v = verdicts.read_verdict(p)
        side = reference.read_sidecar(p)
        main = next((a.get("name") for a in meta.get("angles") or [] if a.get("primary")), None) or (
            (meta.get("angles") or [{}])[0].get("name"))
        readings = reference.merged_readings(meta, side)
        facts.append({"base": p.name[: -len("_meta.json")], "participant": meta.get("participant") or "", "test_key": meta.get("test_key"),
                      "target_deg": meta.get("target_deg"), "verdict": (v or {}).get("verdict") or None, "row": row,
                      "tested_leg": meta.get("tested_leg") or "", "measures": meta.get("participant_measures"), "setup": meta.get("setup"),
                      "standing": meta.get("standing_reference"), "main_angle_name": main, "main_angle_title": next(
                          (a.get("label") or a.get("name") for a in meta.get("angles") or [] if a.get("name") == main), main),
                      "has_main_reading": bool(main and (readings.get(main) or {}).get("mean_deg") is not None)})
    return facts, bad


def _counted(f):
    """A trial that can count: not marked compromised and, when its force numbers are known, a real force rise."""
    if f["verdict"] == verdicts.COMPROMISED:
        return False
    return analysis.contracted(f["row"]) if f["row"] is not None else True


def _item(kind, level, position, text, title=""):
    return {"kind": kind, "level": level, "position": position, "title": title, "text": text}


def check(plan, facts, participant="", titles=None, unreadable=()):
    """The session check. `plan`: [(test key, target)] (None: the positions found in the trials); `facts`: from `gather`; `titles`: {test key:
    name} (else the key). Returns {participant, tested_leg, positions [{number, test_key, target_deg, title, trials, counted, valid, items}],
    items (all, in the order of the plan, then the participant's own), counts {todo, gap, info}, complete (nothing to do and no gap), text}."""
    titles = titles or {}
    mine = [f for f in facts if not participant or f["participant"] == participant]
    who = participant or (mine[0]["participant"] if mine else "")
    plan = list(plan) if plan else []
    in_plan = list(plan)
    seen = [(f["test_key"], f["target_deg"]) for f in mine]
    for key, tgt in seen:                                           # positions that are not in the plan are checked too
        if not any(k == key and _same(t, tgt) for k, t in plan):
            plan.append((key, tgt))
    positions, items = [], []
    for number, (key, tgt) in enumerate(plan, 1):
        at = [f for f in mine if f["test_key"] == key and _same(f["target_deg"], tgt)]
        counted = [f for f in at if _counted(f)]
        valid = [f for f in counted if f["verdict"] == verdicts.VALID]
        name = f"{number:02d}  {titles.get(key, key)}" + (f", {float(tgt):g} deg" if tgt is not None else "")
        pos_items = []

        def add(kind, level, text, pos_items=pos_items, number=number, name=name):
            pos_items.append(_item(kind, level, number, text, name))
        if not at:
            planned = [t for k, t in in_plan if k == key]
            odd = sorted({"no target" if f["target_deg"] is None else f"{float(f['target_deg']):g} deg" for f in mine
                          if f["test_key"] == key and not any(_same(f["target_deg"], t) for t in planned)})
            add("never_recorded", "todo", "Never recorded." + (f" (Trials of this test exist at a target that is not a position of the plan: "
                                                              f"{', '.join(odd)}.)" if odd else ""))
        else:
            if all(f["verdict"] == verdicts.COMPROMISED for f in at):
                add("compromised_only", "todo", f"Only compromised trials ({len(at)}): nothing counts here. Record it again.")
            elif not counted:
                add("no_force", "todo", "No trial has a force rise (or no force was recorded): nothing counts here.")
            else:
                rows = [f["row"] for f in at if f["row"] is not None]
                st = analysis.position_check(rows) if rows else {"state": "none", "text": ""}
                if st["state"] == "third":
                    add("third_trial", "todo", "A third trial is needed: the two peaks differ by more than 10 % (SOP 10.6). " + st["text"])
                elif st["state"] == "deviation":
                    add("deviation", "info", "No two peaks are within 10 %: note it in the deviation log (SOP 10.6).")
                elif len(counted) == 1:
                    add("one_trial", "todo", "Only one counted trial; the SOP asks for two.")
                if not valid:
                    add("not_judged", "gap", f"{len(counted)} counted trial{'' if len(counted) == 1 else 's'}, none marked valid yet: it "
                                             "does not go into the AnyBody export until you judge it.")
                if not any(f["has_main_reading"] for f in counted):
                    add("reference_missing", "gap", f"No reference reading (goniometer or inclinometer) typed for the "
                                                    f"{counted[0]['main_angle_title'] or 'main angle'} at this position.")
                if key not in NO_STRAP_PLACE and not any(isinstance(f["setup"], dict) and f["setup"].get("strap_landmark")
                                                         and f["setup"].get("strap_from_landmark_cm") is not None for f in counted):
                    add("setup_missing", "gap", "Where the strap sat (landmark and distance) was not typed.")
        positions.append({"number": number, "test_key": key, "target_deg": tgt, "title": name, "trials": len(at), "counted": len(counted),
                          "valid": len(valid), "items": pos_items})
        items.extend(pos_items)
    me = []
    if mine:
        legs = sorted({f["tested_leg"] for f in mine if f["tested_leg"]})
        if not legs:
            me.append(_item("leg_missing", "gap", None, "The tested leg was not recorded in any trial.", "Participant"))
        elif len(legs) > 1 or any(not f["tested_leg"] for f in mine):
            me.append(_item("leg_mixed", "gap", None, "The tested leg is not the same in all trials (" + ", ".join(legs)
                            + (", and missing in some" if any(not f["tested_leg"] for f in mine) else "") + ").", "Participant"))
        have = {k: None for k in participant_measures.KEYS}
        for f in mine:
            if isinstance(f["measures"], dict):
                for k in participant_measures.KEYS:
                    if analysis._num(f["measures"].get(k)) is not None:
                        have[k] = f["measures"][k]
        missing = [participant_measures.LABEL[k].lower() for k in REQUIRED_MEASURES if have.get(k) is None]
        if missing:
            me.append(_item("measures_missing", "gap", None, "Participant measures not typed: " + ", ".join(missing) + ".", "Participant"))
        if not any(isinstance(f["standing"], dict) and f["standing"].get("angles") for f in mine):
            me.append(_item("standing_missing", "gap", None, "No standing reference was taken (hip, knee and ankle at the anatomical zero).",
                            "Participant"))
    if unreadable:
        me.append(_item("unreadable", "info", None, "Could not be read, so not checked: " + ", ".join(unreadable) + ".", "Participant"))
    items.extend(me)
    counts = {lv: sum(1 for i in items if i["level"] == lv) for lv in LEVELS}
    return {"participant": who, "tested_leg": next((f["tested_leg"] for f in reversed(mine) if f["tested_leg"]), ""), "positions": positions,
            "participant_items": me, "items": items, "counts": counts, "complete": not (counts["todo"] or counts["gap"]), "trials": len(mine)}


def summary_line(res):
    """One line for a label: 'P02: 2 to do, 5 gaps in the data' or 'P02: nothing missing'."""
    if not res["trials"]:
        return f"{res['participant'] or 'No participant'}: no trials yet."
    c = res["counts"]
    if res["complete"]:
        return f"{res['participant']}: nothing missing."
    parts = []
    if c["todo"]:
        parts.append(f"{c['todo']} to do")
    if c["gap"]:
        parts.append(f"{c['gap']} gap{'' if c['gap'] == 1 else 's'} in the data")
    return f"{res['participant']}: " + ", ".join(parts) + "."


def lines(res):
    """The readable list: [(level or 'heading', text)] -- the headline, what to do, the gaps in the data, then the participant's own."""
    out = [("heading", summary_line(res))]
    if not res["trials"]:
        return out
    todo = [i for i in res["items"] if i["level"] == "todo"]
    gaps = [i for i in res["items"] if i["level"] == "gap" and i["position"] is not None]
    info = [i for i in res["items"] if i["level"] == "info" and i["position"] is not None]
    if todo:
        out.append(("heading", "To do (record or redo)"))
        out += [("todo", f"{i['title']}: {i['text']}") for i in todo]
    if gaps:
        out.append(("heading", "Missing for the export"))
        out += [("gap", f"{i['title']}: {i['text']}") for i in gaps]
    if res["participant_items"]:
        out.append(("heading", "The participant"))
        out += [(i["level"], i["text"]) for i in res["participant_items"]]
    if info:
        out.append(("heading", "For your information"))
        out += [("info", f"{i['title']}: {i['text']}") for i in info]
    if res["complete"]:
        out.append(("ok", "Every position has its counted trials, judged, with a reference reading, the strap place and the participant "
                          "measures. Nothing is missing."))
    out.append(("info", "This list only reads what is saved. It never stops Record or an export."))
    return out


def check_folder(folder, plan, titles=None, row_fn=None, rows=None):
    """The session check of one participant folder: every trial listed (review_nav), its export row from `row_fn(meta_path) -> row or None`
    (the quick way is `lambda p: review.trial_row(p)[0]`) or from `rows` ({trial base: row}), then `check`. Reads files only. A trial that cannot
    be read is named in the result. Without a row source the force numbers are not looked at (a trial then counts unless it is compromised)."""
    items, bad = [], []
    for e in review_nav.list_trials(folder):
        if not e.readable:
            bad.append(e.base)
            continue
        row = (rows or {}).get(e.base) if rows is not None else (row_fn(e.meta_path) if row_fn else None)
        items.append((e.meta_path, row))
    facts, unreadable = gather(items)
    who = next((f["participant"] for f in facts if f["participant"]), Path(folder).name)
    return check(plan, facts, who, titles, unreadable=bad + unreadable)
