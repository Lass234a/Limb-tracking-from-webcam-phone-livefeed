"""The angles of an open trial and the dots that make them, as one tree under the video (0.32.0). It replaces the angle cards, the 'Angles' and
'Dots' tabs and the reference-readings block of the Review tab.

    AngleDotTree   header (title + the window, the reference angle, 'More numbers') over a tree: one row per angle, the dots it is made of under it

        Knee flexion        At peak   Hold    Target   Off    Lost
          trochanter                                           0 %
          anchor  (amber)                                     32 %   Fix

An angle row folds and unfolds when it is clicked. A dot that needs a look (lost in more than 10 % of the pictures, or it jumped) is amber and has
a 'Fix' link; the link, or a double click on the dot, sends `fixDotRequested(dot)`. 'Pull and torque' (the strap and anchor dots) and 'Other dots'
come after the angles. Everything is built from the numbers it is handed (`set_data(trial, rows, fs, window)`); nothing here reads a file or changes
a trial. `text()` is everything a person can read in it as plain lines, so a test can look for a fact without knowing where it sits on the screen.

The model (`build_model`, no Qt) is kept apart from the widget: the rows, their texts, their colour states and their tooltips are worked out first,
then drawn. The dot numbers (Lost, Jumps, the shifts) are the Fix dots workspace's (review_fixdots.py), the angle numbers the old panel's
(review_panels.py): a number reads the same here as there.
"""
import math
from dataclasses import dataclass, field

from PySide6.QtCore import QEvent, QRect, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QGridLayout, QHeaderView, QLabel, QPushButton, QSizePolicy, QStyle,
                               QStyledItemDelegate, QStyleOptionButton, QStylePainter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .analysis import LOST_WARN_PCT
from .geometry import derived_input_dots
from .review_fixdots import PULL_NAMES, _pct, _shift, jump_limit, trial_dot_stats, trial_signature, window_shifts
from .review_panels import (UNKNOWN, _bold, _nan, _off, _plateau_percent, _small, _ReferenceBlock, num, reference_cells, reference_notice,
                            strong_colour)
from .review_style import ACCENT, STATE_COLOURS, WARN, ElidedLabel, border_colour, muted_colour

TITLE = "Angles and the dots that make them"
PULL_GROUP, OTHER_GROUP = "Pull and torque", "Other dots"
PULL_DOTS = ("strap", "anchor")                    # the optional dots of the pull: they belong to no angle
WARN_MARK = "⚠"
FIX_WORD = "Fix"
NO_MARKERS = "No MVIC start/end markers in this trial: the numbers use the window as set."
REF_TITLE = "Reference readings"
REF_SUBTITLE = "Typed from a goniometer or an inclinometer, not measured by this program."
REF_ADD = "Reference angle: add..."
PEAK_POINTS = 14.0                                 # the angle at the peak force is the large number of the main row
SECOND_DOT = "  ·  "

# the columns; the two shift columns are the dots' 'More numbers' (the end shift and the max shift in one cell each), the Fix link is the last
C_NAME, C_PEAK, C_HOLD, C_TARGET, C_OFF, C_LOST, C_SHIFT, C_REL, C_FIX = range(9)
N_COLUMNS = 9
SHIFT_COLUMNS = (C_SHIFT, C_REL)
HEADS = ["Angle / dot", "At peak", "Hold", "Target", "Off", "Lost", "End / max\nshift", "End / max\nvs anchor", ""]
NAME_MIN = 110                                     # the name column keeps at least this much before the numbers are squeezed ...
NAME_LEAST = 70                                    # ... and never goes below this (then the tree gets a sideways scroll bar)
ROW_MIN = 22

ROLE_KIND, ROLE_KEY, ROLE_FIX, ROLE_TEXT = Qt.UserRole, Qt.UserRole + 1, Qt.UserRole + 2, Qt.UserRole + 3


# ---------------------------------------------------------------------------------------------------------- the numbers (no Qt)
def _g(v, d=1):
    """70.0 -> '70', 21.549 -> '21.5': a target or a tolerance without a pointless '.0'."""
    return f"{round(float(v), d) + 0.0:g}"


def _deg(v, d=1):
    return UNKNOWN if _nan(v) else f"{num(v, d)}°"


def _state(dev, tol, judged=True):
    """None for an angle that is not judged or has nothing to judge, else 'good' within the tolerance (the edge included) and 'warn' beyond it."""
    if not judged or _nan(dev):
        return None
    return "good" if abs(dev) <= tol else "warn"


def off_words(dev):
    """'on the target' (under 0.05 deg), '7.5 deg above the target', or '' when nothing is known."""
    if _nan(dev):
        return ""
    if abs(dev) < 0.05:
        return "on the target"
    return f"{abs(dev):.1f} deg {'above' if dev > 0 else 'below'} the target"


def dots_of(trial, angle):
    """The tracked dots an angle is made of, in order and without repeats; a computed point (the hip centre) counts as the dots it is made of."""
    defs = (trial.meta.get("derived_points") or {}).get("definitions") or {}
    out = []
    for n in angle.dots:
        for m in ([n] if n in trial.dot_names else derived_input_dots(defs[n]) if n in defs else []):
            if m in trial.dot_names and m not in out:
                out.append(m)
    return out


def _both(a, b):
    """'0.4 cm' and '1.6 cm' -> '0.4 / 1.6 cm' (the unit once when it is the same); '-' for two missing."""
    if a == b == UNKNOWN:
        return UNKNOWN
    ua, ub = a.rsplit(" ", 1), b.rsplit(" ", 1)
    return f"{ua[0]} / {b}" if len(ua) == 2 and len(ub) == 2 and ua[1] == ub[1] else f"{a} / {b}"


def dot_reasons(s):
    """Why a dot needs a look, in words ('' for a dot that does not)."""
    why = []
    if s.lost_pct > LOST_WARN_PCT:
        why.append(f"lost in {s.lost_pct:.0f} % of the pictures")
    if s.jumps:
        why.append(f"jumped {s.jumps} time{'' if s.jumps == 1 else 's'}")
    return " and ".join(why)


@dataclass
class Row:
    """One line of the tree. `kind`: 'angle', 'group' (Pull and torque, Other dots), 'dot', or 'detail' (a quiet line of More numbers)."""
    kind: str
    key: str                                   # the angle's name, the group's title or the dot's name
    label: str                                 # the plain name (the angle's title)
    name: str = ""                             # what the first column shows (the warning sign and the notes included)
    cells: dict = field(default_factory=dict)  # column -> text
    states: dict = field(default_factory=dict)  # column -> 'good' | 'warn'
    tips: dict = field(default_factory=dict)   # column -> tooltip
    tip: str = ""                              # the tooltip of the name (and of the cells that have none of their own)
    line: str = ""                             # the plain-text line (text())
    extra: str = ""                            # the part of that line that is More numbers (dots: the shifts)
    suffix: str = ""                           # the end of that line: ' - needs a look: ...'
    main: bool = False
    tinted: bool = False                       # amber background: a dot that needs a look
    fix: bool = False                          # shows the Fix link
    large: tuple = ()                          # columns whose number is large
    needs: list = field(default_factory=list)  # the reasons it needs a look
    details: list = field(default_factory=list)   # 'detail' rows, shown with More numbers
    always: bool = False                       # a 'detail' row that is shown without More numbers (the dots' numbers of an angle with a reference)
    children: list = field(default_factory=list)


@dataclass
class Model:
    rows: list
    window_text: str = ""
    window_tip: str = ""
    no_markers: bool = False
    notes: list = field(default_factory=list)  # the sentences that are tooltips on the screen: what Hold and Rest mean, what the Lost is for
    movement: str = ""                         # the line over the dots' shifts ('Movement of the dots in the window ...')
    has_rel: bool = False                      # the trial has an anchor dot: 'vs anchor' columns
    cm: bool = False                           # the trial has a ruler scale
    has_peak_column: bool = False
    reference_text: str = REF_ADD
    reference_tip: str = ""
    reference_lines: list = field(default_factory=list)


def _main_row(rows):
    return next((r for r in rows if r["role"] == "main"), rows[0] if rows else None)


def _row_of(rows, a, i):
    """The summary row of angle `a` (rows are in the order of the trial's angles; a row is found by its title if that order was changed)."""
    if i < len(rows) and rows[i].get("angle") == a.title:
        return rows[i]
    return next((r for r in rows if r.get("angle") == a.title), None)


def _angle_row(trial, a, r, fs, owner, tols, stats, shifts, cm, has_rel, limit_words):
    """The row of one angle: At peak (the main one, with force), Hold, Target, Off, Lost (the worst of its dots), the warning, and the dots under it."""
    is_main = (r["role"] == "main") if r is not None else bool(a.primary)
    tol = tols[0] if is_main else tols[1]
    judged = bool(getattr(a, "judged", True))
    dots = [d for d in dots_of(trial, a) if d in stats]
    hold = r["plateau_mean_deg"] if r is not None else math.nan
    tgt = r["target_deg"] if r is not None else math.nan
    off = _off(r) if r is not None else math.nan
    peak = math.nan                                                     # every angle has one since 0.33.0 (it was the main joint's only before)
    if fs is not None and r is not None:
        if r is owner:                                                  # the main joint's is the old number (the dots' angle), from the reference when it has one
            peak = fs["main_angle_at_peak_deg"] - r.get("offset_deg", 0.0)
        else:
            peak = (fs.get("angles_at_peak_deg") or {}).get(a.title, math.nan)
    has_peak = not _nan(peak)
    peak_dev = peak - tgt if has_peak and not _nan(tgt) else math.nan
    peak_state, hold_state = _state(peak_dev, tol, judged), _state(off, tol, judged)
    big_state = peak_state if has_peak else hold_state

    # ---- the cells
    row = Row("angle", a.name, a.title, main=is_main)
    if has_peak:
        row.cells[C_PEAK] = _deg(peak)
        row.large = (C_PEAK,)
        row.states[C_PEAK] = big_state
        delay = fs.get("picture_delay_s")
        row.tips[C_PEAK] = ("The joint angle in the pictures at the moment of the highest force."
                            + (f" The pictures are read {delay:+.3f} s after the force peak." if delay else "")
                            + (f" It is {off_words(peak_dev)} (tolerance {tol:g} deg)." if not _nan(peak_dev) else "")
                            + ("" if judged else " This angle is shown, not judged."))
    elif fs is not None and r is not None:
        row.cells[C_PEAK] = UNKNOWN
        row.tips[C_PEAK] = "The joint angle at the moment of the highest force: it could not be worked out."
    row.cells[C_HOLD] = _deg(hold)
    if is_main and not has_peak:                                        # without a peak force the hold angle is the large number
        row.large = (C_HOLD,)
        row.states[C_HOLD] = big_state
    row.tips[C_HOLD] = ("The angle during the contraction (the middle %.0f %% of the window), not the angle that was set." % _plateau_percent()
                        if not _nan(hold) else "No hold angle in this window: too few pictures with this angle.")
    if not judged:
        row.cells[C_TARGET] = UNKNOWN if _nan(tgt) else f"{_g(tgt)}°"
        row.tips[C_TARGET] = "This angle is shown, not judged: it has no tolerance."
    elif _nan(tgt):
        row.cells[C_TARGET], row.tips[C_TARGET] = UNKNOWN, "No target for this angle."
    else:
        row.cells[C_TARGET] = f"{_g(tgt)} ± {_g(tol)}°"
        row.tips[C_TARGET] = f"The angle that was asked for, {_g(tgt)} deg, and the tolerance, {_g(tol)} deg either way."
    row.cells[C_OFF] = UNKNOWN if _nan(off) else ("0.0°" if abs(off) < 0.05 else f"{num(off, signed=True)}°")
    row.states[C_OFF] = "warn" if hold_state == "warn" else None
    row.tips[C_OFF] = ("The hold angle minus the target" + (f": {off_words(off)}." if not _nan(off) else ".")
                       + (" Amber when it is outside the tolerance." if judged else ""))

    # ---- Lost: the worst of its dots; and why the angle needs a look
    worst = _lost_cells(row, dots, stats)
    needs = []
    if has_peak and peak_state == "warn":
        needs.append(f"at the peak force the angle is {off_words(peak_dev)}")
    if hold_state == "warn":
        needs.append(f"the hold angle is {off_words(off)}")
    frames, lost = (r["frames"], r["frames_lost"]) if r is not None else (0, 0)
    if frames and 100.0 * lost / frames > LOST_WARN_PCT:
        needs.append(f"no angle in {lost} of {frames} pictures of the window")
    needs += [f"{d} {dot_reasons(stats[d])}" for d in dots if stats[d].needs_attention]
    row.needs = needs
    row.name = (f"{WARN_MARK} " if needs else "") + a.title
    row.tip = (a.title + (" (the main joint)" if is_main else "") + (f"\nMade of the dots: {', '.join(dots)}." if dots else "")
               + (f"\nNeeds a look: {'; '.join(needs)}." if needs else ""))

    # ---- More numbers (a quiet line under the row) and the plain-text line
    if r is not None:
        def pair(label, v, d=1, signed=False):                      # (label, as on the screen, as plain text)
            return (label, UNKNOWN, UNKNOWN) if _nan(v) else (label, f"{num(v, d, signed)}°", f"{num(v, d, signed)} deg")
        pct = UNKNOWN if _nan(r["pct_in_tolerance"]) else f"{r['pct_in_tolerance']:.0f} %"
        pairs = [pair("Rest", r["rest_mean_deg"]), pair("Rest to hold", r["rest_to_hold_shift_deg"], signed=True), ("In tolerance", pct, pct),
                 ("Pictures lost", f"{lost} of {frames}", f"{lost} of {frames}"), pair("Mean", r["mean_deg"]), pair("SD", r["sd_deg"], 2),
                 pair("Min", r["min_deg"]), pair("Max", r["max_deg"]), pair("Start to end", r["change_start_to_end_deg"], signed=True),
                 pair("Max deviation", r["max_abs_dev_deg"]), pair("Median", r["median_deg"])]
        shown = " · ".join(f"{k} {s}" for k, s, _p in pairs)
        plain = ", ".join(f"{k} {p}" for k, _s, p in pairs)
        row.details = [Row("detail", a.name, a.title, name=shown, line="More numbers: " + plain, cells={C_NAME: shown},
                           tip="The numbers behind the row, for the window. Rest is the angle in the quiet stretch before the push; "
                               "Hold is the angle during the contraction.")]
    parts = []
    if has_peak:
        parts.append(f"at the peak force {num(peak)} deg" + (f" ({off_words(peak_dev)})" if not _nan(peak_dev) else ""))
    parts.append(f"hold {num(hold)} deg" if not _nan(hold) else "hold -")
    if not judged:
        parts.append("not judged" + ("" if _nan(tgt) else f", target {_g(tgt)} deg"))
    elif _nan(tgt):
        parts.append("no target")
    else:
        parts.append(f"target {_g(tgt)} ± {_g(tol)} deg")
    parts.append("off target -" if _nan(off) else "on the target" if abs(off) < 0.05 else f"off target {num(off, signed=True)} deg")
    parts.append(f"lost {row.cells[C_LOST]}" + (f" (worst dot: {worst})" if worst else ""))
    row.line = f"{a.title}{' (main joint)' if is_main else ''}: " + ", ".join(parts)
    row.suffix = f" - needs a look: {'; '.join(needs)}" if needs else ""

    off = r.get("offset_deg", 0.0) if r is not None else 0.0
    if off:                                                                 # 0.33.0: the numbers above are from the reference; the dots' own beside them
        peak_dots = (fs.get("angles_at_peak_dots_deg") or {}).get(a.title, math.nan) if fs is not None else math.nan
        hold_dots = r.get("plateau_mean_dots_deg", math.nan)
        taken = "taken when the trial was recorded" if abs(trial.recorded_offsets.get(a.name, 0.0) - off) < 1e-9 else "chosen in Review from a reference reading"
        shown = ("On the dots: " + (f"at the peak force {num(peak_dots)}°, " if has_peak else "") + f"hold {num(hold_dots)}°"
                 + ("" if _nan(tgt) else f", target {_g(tgt + off)}°") + f". The dots read {num(off, signed=True)}° more than the joint ({taken}).")
        plain = ("from the reference (the dots read " + num(off, signed=True) + " deg more than the joint; on the dots: "
                 + (f"at the peak force {num(peak_dots)} deg, " if has_peak else "") + f"hold {num(hold_dots)} deg)")
        row.details.insert(0, Row("detail", a.name, a.title, name=shown, line=plain, always=True, cells={C_NAME: shown},
                                  tip="The angle is shown from a reference: the joint was called a known angle while the dots read something else. "
                                      "The numbers above are from the reference; these are the dots' own."))
        row.tip += f"\nFrom the reference: the dots read {num(off, signed=True)} deg more than the joint."
        row.line += f" - {plain}"
    row.children = [_dot_row(d, stats[d], shifts, cm, has_rel, limit_words) for d in dots]
    return row


def _dot_row(d, s, shifts, cm, has_rel, limit_words):
    """One dot: Lost % of the whole trial, in the name the jumps and the fixes, and (More numbers) the shifts in the window."""
    why = dot_reasons(s) if s.needs_attention else ""
    bits = ([f"{s.jumps} jump{'' if s.jumps == 1 else 's'}"] if s.jumps else []) + ([f"✎ {s.n_fixed} fixed"] if s.n_fixed else [])
    row = Row("dot", d, d, name=(f"{WARN_MARK} " if why else "") + d + (SECOND_DOT + ", ".join(bits) if bits else ""), tinted=bool(why), fix=bool(why))
    row.cells[C_LOST] = _pct(s.lost_pct)
    row.states[C_LOST] = "warn" if s.lost_pct > LOST_WARN_PCT else None
    row.tip = (f"{d}: lost in {s.lost_pct:.0f} % of the pictures of the whole trial, {s.jumps} jump{'' if s.jumps == 1 else 's'}, "
               f"{s.n_fixed} picture{'' if s.n_fixed == 1 else 's'} fixed." + (" Needs a look: double-click it, or press Fix." if why else "")
               + f" A jump is a move of more than {limit_words} between two pictures.")
    row.tips[C_LOST] = row.tip
    row.needs = [why] if why else []
    r = shifts.get(d, {})
    plain = []
    for col, end, top, words in ((C_SHIFT, "end_shift", "max_shift", "shift"), (C_REL, "rel_end_shift", "rel_max_shift", "vs anchor")):
        if col == C_REL and not has_rel:
            continue
        (s_end, tip_end, p_end), (s_top, tip_top, p_top) = _shift(r, end, cm), _shift(r, top, cm)
        row.cells[col] = _both(s_end, s_top)
        row.tips[col] = f"{d}, {words}, in the window: end {tip_end or s_end}, max {tip_top or s_top}"
        plain += [f"end {words} {p_end}", f"max {words} {p_top}"]
    row.extra = ", ".join(plain)
    row.line = f"{d}: lost {_pct(s.lost_pct)}, {s.jumps} jump{'' if s.jumps == 1 else 's'}" + (f", ✎ {s.n_fixed} fixed" if s.n_fixed else "")
    row.suffix = f" - needs a look: {why}" if why else ""
    return row


def _lost_cells(row, dots, stats):
    """The Lost cell of an angle or a group: the worst of its dots. Returns the name of the dot lost most (None when none of them was lost)."""
    top = max(dots, key=lambda d: stats[d].lost_pct, default=None)
    row.cells[C_LOST] = _pct(stats[top].lost_pct) if top is not None else UNKNOWN
    row.states[C_LOST] = "warn" if top is not None and stats[top].lost_pct > LOST_WARN_PCT else None
    lost = top is not None and stats[top].lost_pct > 0
    row.tips[C_LOST] = (f"The worst of its dots over the whole trial: {top}, lost in {stats[top].lost_pct:.0f} % of the pictures." if lost
                        else "None of its dots was lost in the trial." if top is not None else "No tracked dot in the trial.")
    return top if lost else None


def _group_row(title, dots, stats, shifts, cm, has_rel, limit_words, note="", tip="", details=()):
    kids = [_dot_row(d, stats[d], shifts, cm, has_rel, limit_words) for d in dots]
    needs = [f"{d} {dot_reasons(stats[d])}" for d in dots if stats[d].needs_attention]
    row = Row("group", title, title, name=(f"{WARN_MARK} " if needs else "") + title + (SECOND_DOT + note if note else ""), children=kids, needs=needs)
    worst = _lost_cells(row, dots, stats)
    row.tip = tip or title
    row.details = list(details)
    what = "from the strap and anchor dots" if title == PULL_GROUP else "dots that belong to no angle"
    row.line = f"{title} ({what})" + (f": {note}" if note else "") + f", lost {row.cells[C_LOST]}" + (f" (worst dot: {worst})" if worst else "")
    row.suffix = f" - needs a look: {'; '.join(needs)}" if needs else ""
    return row


def _pull(trial, win):
    """(note, tip, detail rows) of the pull numbers of the strap and anchor dots in the window: ('', ..., []) when the trial has none."""
    rows = trial.pull_summary(*win)
    if not rows:
        return "", "", []
    by = {q["quantity"]: q for q in rows}
    arm = by.get("moment_arm_cm") or by.get("moment_arm_px")
    parts = []
    if "pull_angle_deg" in by and not _nan(by["pull_angle_deg"]["mean"]):
        parts.append(f"pull {by['pull_angle_deg']['mean']:.1f} deg")
    if arm is not None and not _nan(arm["mean"]):
        parts.append(f"arm {arm['mean']:.1f} {'cm' if arm['quantity'].endswith('cm') else 'px'}")
    note = " - ".join(parts) if parts else "no pull numbers in this window"
    tip = ("The pull numbers come from the strap and anchor dots, in the window: the angle between the line of pull (strap dot to anchor dot) and "
           "the lower leg segment, and the distances that make the torque.")
    details = []
    for q in rows:
        name = PULL_NAMES.get(q["quantity"], q["quantity"])
        v = lambda x, signed=False: UNKNOWN if _nan(x) else (f"{x:+.1f}" if signed else f"{x:.1f}")      # noqa: E731
        missing = f" ({q['frames_missing']} pictures of the window have no value)" if q["frames_missing"] else ""
        nums = [("mean", q["mean"], False), ("min", q["min"], False), ("max", q["max"], False), ("change from start to end", q["change_start_to_end"], True)]
        shown = f"{name}: " + " · ".join(f"{k} {v(x, sg)}" for k, x, sg in nums) + missing
        plain = f"{name}, from the strap and anchor dots: " + ", ".join(f"{k} {v(x, sg)}" for k, x, sg in nums) + missing
        details.append(Row("detail", q["quantity"], name, name=shown, line=plain, cells={C_NAME: shown},
                           tip="From the strap and anchor dots. Each end of the window is averaged over five pictures."))
    return note, tip, details


def build_model(trial, rows, fs, window, stats, shifts):
    """Everything the tree shows, as rows of texts and colour states. `rows` = trial.summary(t0, t1), `fs` = trial.force_summary(...) or None,
    `window` = (t0, t1) or None, `stats` = {dot: DotStat} (review_fixdots.trial_dot_stats), `shifts` = {dot: row of Trial.displacement}."""
    tol = float(trial.meta.get("tolerance_deg") or 5.0)
    tols = (tol, float(trial.meta.get("neighbour_tolerance_deg") or tol))          # the main joint's tolerance, the others'
    cm = bool(trial.px_per_cm)
    has_rel = any("rel_end_shift_px" in r for r in shifts.values())
    _px, limit_words = jump_limit(trial)
    win = tuple(window) if window is not None else (float(trial.t[0]), float(trial.t[-1]))
    main = _main_row(rows)
    owner = next((r for r in rows if fs is not None and r["angle"] == fs["main_angle"]), main)
    order = sorted(enumerate(trial.angle_defs), key=lambda p: not p[1].primary)    # the main joint first, the others as the trial has them
    out, used = [], set()
    for i, a in order:
        r = _row_of(rows, a, i)
        row = _angle_row(trial, a, r, fs, owner, tols, stats, shifts, cm, has_rel, limit_words)
        used.update(c.key for c in row.children)
        out.append(row)
    known = {a.title for a in trial.angle_defs}
    for r in rows:                                                                  # a row of the summary that no angle of the trial owns: shown anyway
        if r["angle"] not in known:
            out.append(Row("angle", r["angle"], r["angle"], name=r["angle"], cells={C_HOLD: _deg(r["plateau_mean_deg"])},
                           line=f"{r['angle']}: hold {num(r['plateau_mean_deg'])} deg"))
    pull_dots = [d for d in PULL_DOTS if d in stats]
    if pull_dots:
        note, tip, details = _pull(trial, win)
        out.append(_group_row(PULL_GROUP, pull_dots, stats, shifts, cm, has_rel, limit_words, note, tip or "The strap and anchor dots.", details))
    other = [d for d in trial.dot_names if d in stats and d not in used and d not in PULL_DOTS]
    if other:
        out.append(_group_row(OTHER_GROUP, other, stats, shifts, cm, has_rel, limit_words, tip="Tracked dots that belong to no angle."))
    m = Model(out, has_rel=has_rel, cm=cm, no_markers=trial.hold_window() is None)
    if window is not None:
        m.window_text = f"window {window[0]:.2f} to {window[1]:.2f} s"
        m.window_tip = (f"The numbers are for the blue window of the graphs, {window[0]:.2f} to {window[1]:.2f} s. "
                        "The Lost of the dots is for the whole trial.")
    m.movement = (f"Movement of the dots in the window, {win[0]:.2f} to {win[1]:.2f} s ({'cm from the ruler scale' if cm else 'pixels: no ruler scale'})")
    m.has_peak_column = any(C_PEAK in r.cells for r in out)
    m.notes = [f"Hold is the angle during the contraction (the middle {_plateau_percent():.0f} % of the window), not the angle that was set.",
               "Rest is the angle in the quiet stretch before the push (the stretch the resting force is taken from); without a clear push "
               "it is the 1 s before the window.",
               f"Lost: for a dot, the share of the pictures of the whole trial in which the program could not find it (more than {LOST_WARN_PCT:g} % "
               f"is amber); for an angle or a group, the worst of its dots. A jump is a move of more than {limit_words} between two pictures."]
    if any(r.kind == "group" and r.key == PULL_GROUP and r.details for r in out):
        m.notes.append("The pull numbers come from the strap and anchor dots.")
    _reference(trial, m)
    return m


def _reference(trial, m):
    """The words of the 'Reference angle' button, its tooltip and the plain-text lines of the readings."""
    rows = trial.reference_rows()
    saved = trial.meta.get("reference_readings") or {}
    notice = reference_notice(rows, saved)
    if not rows:
        m.reference_text = REF_ADD
        m.reference_tip = notice + "\nPress to type the readings in."
        m.reference_lines = [REF_TITLE, REF_SUBTITLE, notice]
        return
    cells = reference_cells(rows)
    main = trial.main_angle
    k = next((i for i, r in enumerate(rows) if main is not None and r["key"] == main.name), 0)
    r = rows[k]
    diff = r["difference_deg"]
    m.reference_text = f"Reference angle: {num(r['reference_deg'])} deg - " + (f"app {num(diff, signed=True)} deg" if diff is not None else "no app value")
    lines = [f"{c[0]}: reference {c[1]}, app {c[2]}, difference {c[3]}  {c[4]}" for c in cells]
    basis = _ReferenceBlock._explanation(trial, rows, saved)
    m.reference_lines = [REF_TITLE, REF_SUBTITLE] + ([notice] if notice else []) + lines + ([basis] if basis else [])
    m.reference_tip = "\n".join([REF_TITLE + " (" + REF_SUBTITLE[:-1].lower() + "):"] + ([notice] if notice else []) + lines + ([basis] if basis else [])
                                + ["Press to add or change them."])


# ---------------------------------------------------------------------------------------------------------- small widgets
class _Heading(QWidget):
    """The title in bold and, after it in quiet text, the window the numbers are for. The title keeps its room; the window text is cut first."""
    GAP = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._title, self._extra = TITLE, ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumWidth(40)

    def set_texts(self, title, extra, tip=""):
        self._title, self._extra = title, extra
        self.setToolTip(tip)
        self.updateGeometry()
        self.update()

    def text(self):
        return self._title + (f"  {self._extra}" if self._extra else "")

    def natural_width(self):
        w = QFontMetrics(_bold(self.font())).horizontalAdvance(self._title)
        return w + (self.GAP + QFontMetrics(self.font()).horizontalAdvance(self._extra) if self._extra else 0) + 2

    def sizeHint(self):
        return QSize(self.natural_width(), QFontMetrics(_bold(self.font())).height() + 6)

    def paintEvent(self, _e):
        p = QPainter(self)
        h, w = self.height(), self.width()
        title_font = _bold(self.font())
        fm = QFontMetrics(title_font)
        title_w = min(fm.horizontalAdvance(self._title) + 1, w)
        p.setFont(title_font)
        p.setPen(self.palette().color(QPalette.WindowText))
        p.drawText(QRect(0, 0, title_w, h), int(Qt.AlignLeft | Qt.AlignVCenter), fm.elidedText(self._title, Qt.ElideRight, title_w))
        room = w - title_w - self.GAP
        if self._extra and room >= 40:
            p.setFont(self.font())
            p.setPen(QColor(muted_colour(QApplication.palette())))
            p.drawText(QRect(title_w + self.GAP, 0, room, h), int(Qt.AlignLeft | Qt.AlignVCenter),
                       QFontMetrics(self.font()).elidedText(self._extra, Qt.ElideRight, room))


class _RefButton(QPushButton):
    """A button whose text is cut with '...' to the room it has (`text()` is always the whole text; the tooltip carries the rest)."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumWidth(60)
        self.setFocusPolicy(Qt.NoFocus)

    def sizeHint(self):
        s = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + 28, s.height())

    def minimumSizeHint(self):
        return QSize(60, super().minimumSizeHint().height())

    def paintEvent(self, _e):
        p = QStylePainter(self)
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        room = self.style().subElementRect(QStyle.SE_PushButtonContents, opt, self).width() - 4
        opt.text = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, max(room, 20))
        p.drawControl(QStyle.CE_PushButton, opt)


class _Delegate(QStyledItemDelegate):
    """The usual cells, and the quiet lines of More numbers: muted small text that wraps to the width of the row."""

    def paint(self, painter, option, index):
        if index.data(ROLE_KIND) != "detail":
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setFont(_small(option.font))
        painter.setPen(QColor(muted_colour(QApplication.palette())))
        painter.drawText(option.rect.adjusted(4, 2, -4, -1), int(Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap), str(index.data(ROLE_TEXT) or ""))
        painter.restore()


class _Tree(QTreeWidget):
    """A tree that shows a hand over what can be clicked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def mouseMoveEvent(self, e):
        item = self.itemAt(e.position().toPoint())
        col = self.columnAt(e.position().toPoint().x())
        kind = item.data(0, ROLE_KIND) if item is not None else None
        clickable = kind in ("angle", "group") or (kind == "dot" and col == C_FIX and bool(item.data(0, ROLE_FIX)))
        self.viewport().setCursor(Qt.PointingHandCursor if clickable else Qt.ArrowCursor)
        super().mouseMoveEvent(e)


# ---------------------------------------------------------------------------------------------------------- the widget
class AngleDotTree(QWidget):
    """The angles of the open trial with the dots that make them. `set_data(trial, rows, fs, window=None)`: `rows` = trial.summary(t0, t1), `fs` =
    trial.force_summary(t0, t1, picture_delay) (None for a trial without force: then Hold is the large number), `window` = (t0, t1) for the heading.
    `set_none(text)` shows a quiet message instead.

    Signals: `fixDotRequested(dot)` (the Fix link of a dot that needs a look, or a double click on any dot) and `referenceRequested()` (the Reference
    angle button). `set_fold(name, open)` folds or unfolds an angle (by its name or its title; a group by its title); the fold state is kept while the
    widget lives, across trials with the same angle names. `rows_for(name)` = the dot names under an angle or a group. `text(everything=False)` is
    everything a person can read as plain lines; `everything=True` adds the rows behind 'More numbers', the dots of folded angles and the sentences
    that are tooltips on the screen (what Hold and Rest mean, the reference readings in full). `.tree`, `.more_check`, `.reference_button`."""

    fixDotRequested = Signal(str)
    referenceRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.setMinimumWidth(0)
        self._last = None
        self._model = None
        self._open = {}                                   # angle name / group title -> unfolded (remembered while the widget lives)
        self._css = None
        self._sig = None
        self._stats_trial, self._stats, self._stats_sig = None, {}, None
        self._building = False
        self._fitting = False
        self._wide = None
        self._hint_h = None
        self._details = []                                # (item, text) of the lines of More numbers

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)
        self.empty = QLabel("")
        self.empty.setObjectName("anglesEmpty")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)
        outer.addWidget(self.empty, 1)
        self.body = QWidget()
        outer.addWidget(self.body, 1)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # the header: the title and the window, the reference angle, More numbers
        self.head = QWidget()
        self.head_grid = QGridLayout(self.head)
        self.head_grid.setContentsMargins(0, 0, 0, 0)
        self.head_grid.setHorizontalSpacing(10)
        self.head_grid.setVerticalSpacing(4)
        self.heading = _Heading()
        self.reference_button = _RefButton(REF_ADD)
        self.reference_button.setObjectName("angleRef")
        self.reference_button.clicked.connect(self.referenceRequested)
        self.more_check = QCheckBox("More numbers")
        self.more_check.setObjectName("angleMore")
        self.more_check.setFocusPolicy(Qt.NoFocus)
        self.more_check.setToolTip("Add the rest, the mean, the spread, the lowest and highest angle, the change from start to end, the largest deviation "
                                   "and the median of each angle, and how far each dot moved in the window.")
        self.more_check.toggled.connect(self._more_toggled)
        lay.addWidget(self.head)
        self.note = ElidedLabel("")
        self.note.setObjectName("anglesNote")
        lay.addWidget(self.note)

        # the tree
        self.tree = _Tree()
        self.tree.setObjectName("angleTree")
        self.tree.setAccessibleName("Angles and the dots that make them")
        self.tree.setColumnCount(N_COLUMNS)
        self.tree.setHeaderLabels(HEADS)
        self.tree.setIndentation(16)
        self.tree.setRootIsDecorated(True)
        self.tree.setItemsExpandable(True)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setAnimated(False)
        self.tree.setUniformRowHeights(False)
        self.tree.setSelectionMode(QAbstractItemView.NoSelection)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.setFocusPolicy(Qt.NoFocus)                           # (the picture keys stay with the video)
        self.tree.setTextElideMode(Qt.ElideRight)
        self.tree.setWordWrap(False)
        self.tree.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tree.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)    # (only if even the squeezed numbers do not fit: a very narrow tree with More numbers)
        self.tree.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.tree.setMinimumWidth(0)
        self.tree.setItemDelegate(_Delegate(self.tree))
        hh = self.tree.header()
        hh.setStretchLastSection(False)
        hh.setSectionsMovable(False)
        hh.setSectionsClickable(False)
        hh.setHighlightSections(False)
        hh.setMinimumSectionSize(20)
        hh.setSectionResizeMode(QHeaderView.Fixed)
        for c in range(N_COLUMNS):                                      # (the names of the columns sit over their numbers: the first to the left)
            self.tree.headerItem().setTextAlignment(c, (Qt.AlignLeft if c == C_NAME else Qt.AlignRight) | Qt.AlignVCenter)
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.itemDoubleClicked.connect(self._item_double_clicked)
        self.tree.itemExpanded.connect(self._folded)
        self.tree.itemCollapsed.connect(self._folded)
        self.tree.viewport().installEventFilter(self)                  # (the columns and the wrapped lines are fitted again when the room changes:
        lay.addWidget(self.tree, 1)                                    #  the tree is resized, or a scroll bar comes or goes)

        self._apply_style()
        self._arrange_header()
        self.set_none("No trial is open.")

    # ------------------------------------------------------------ what the lead calls
    def set_none(self, text):
        self._last = None
        self._model = None
        self._sig = None
        self._stats_trial, self._stats = None, {}
        self.tree.clear()
        self._details = []
        self.empty.setText(text)
        self.empty.setVisible(True)
        self.body.setVisible(False)
        self._invalidate()

    def set_data(self, trial, rows, fs, window=None):
        self._last = (trial, rows, fs, window)
        if self._stats_trial is not trial or self._stats_sig != trial_signature(trial):       # (the dot numbers are made again only after a fix)
            self._stats_trial, self._stats_sig = trial, trial_signature(trial)
            self._stats = {s.name: s for s in trial_dot_stats(trial)}
        win = tuple(window) if window is not None else (float(trial.t[0]), float(trial.t[-1]))
        self._model = build_model(trial, list(rows or []), fs, window, self._stats, window_shifts(trial, *win))
        m = self._model
        self.empty.setVisible(False)
        self.body.setVisible(True)
        self.heading.set_texts(TITLE, m.window_text, m.window_tip)
        self.reference_button.setText(m.reference_text)
        self.reference_button.setToolTip(m.reference_tip)
        self.note.setText(NO_MARKERS if m.no_markers else "")
        self.note.setVisible(m.no_markers)
        self._arrange_header(force=True)
        self._fill_tree()

    def set_fold(self, name, open):
        """Unfold (True) or fold (False) an angle by its name or its title, or a group by its title; remembered."""
        key = self._key_of(name)
        self._open[key] = bool(open)
        item = self.angle_item(key)
        if item is not None and item.isExpanded() != bool(open):
            item.setExpanded(bool(open))

    def is_open(self, name):
        return self._open.get(self._key_of(name), True)

    def rows_for(self, angle_name):
        """The dot names shown under an angle (or under 'Pull and torque' / 'Other dots'), in order; [] for a name that is not shown."""
        row = self._row(self._key_of(angle_name))
        return [c.key for c in row.children] if row is not None else []

    def angle_item(self, name):
        """The tree item of an angle or a group (by name or title), or None."""
        key = self._key_of(name)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, ROLE_KEY) == key:
                return item
        return None

    def dot_item(self, angle_name, dot):
        item = self.angle_item(angle_name)
        if item is not None:
            for i in range(item.childCount()):
                c = item.child(i)
                if c.data(0, ROLE_KIND) == "dot" and c.data(0, ROLE_KEY) == dot:
                    return c
        return None

    def content_height(self):
        """The height the tree needs to show every row without a scroll bar (the header row of the columns included)."""
        t = self.tree
        last = None
        for i in range(t.topLevelItemCount()):
            last = t.topLevelItem(i)
        while last is not None and last.isExpanded() and last.childCount():
            last = last.child(last.childCount() - 1)
        if last is None:
            return t.header().height() + 2 * t.frameWidth()
        r = t.visualItemRect(last)
        return t.header().height() + r.bottom() + 1 + t.verticalScrollBar().value() + 2 * t.frameWidth()

    # ------------------------------------------------------------ names
    def _row(self, key):
        if self._model is None:
            return None
        return next((r for r in self._model.rows if r.key == key), None)

    def _key_of(self, name):
        """The key of an angle or a group from its name or its title (case does not matter); the name itself if nothing is shown by that name."""
        if self._model is not None:
            for r in self._model.rows:
                if name in (r.key, r.label) or str(name).lower() in (r.key.lower(), r.label.lower()):
                    return r.key
        return name

    # ------------------------------------------------------------ the header
    def _arrange_header(self, force=False):
        """One line when the title, the reference button and More numbers fit side by side; else the button gets a line of its own."""
        gap = self.head_grid.horizontalSpacing()
        need = self.heading.natural_width() + self.reference_button.sizeHint().width() + self.more_check.sizeHint().width() + 2 * gap + 12
        m = self.layout().contentsMargins()
        wide = self.width() - m.left() - m.right() >= need                  # (the width of this widget: the header's own is not updated yet in a resize)
        if wide == self._wide and not force:
            return
        self._wide = wide
        g = self.head_grid
        for w in (self.heading, self.reference_button, self.more_check):
            g.removeWidget(w)
        for c in range(4):
            g.setColumnStretch(c, 0)
        g.setColumnStretch(0, 1)
        if wide:
            g.addWidget(self.heading, 0, 0)
            g.addWidget(self.reference_button, 0, 1)
            g.addWidget(self.more_check, 0, 2)
        else:
            g.addWidget(self.heading, 0, 0)
            g.addWidget(self.more_check, 0, 1)
            g.addWidget(self.reference_button, 1, 0, 1, 2, Qt.AlignLeft)
        self.head.updateGeometry()

    # ------------------------------------------------------------ the tree
    def _fill_tree(self):
        """Made again only when what it shows changed (the window moves many times a second); the scroll position and the folds are kept."""
        m, more = self._model, self.more_check.isChecked()
        pal = QApplication.palette()
        sig = (repr(m.rows), more, pal.color(QPalette.Window).name(), pal.color(QPalette.WindowText).name(), m.has_peak_column, m.has_rel)
        if sig == self._sig:
            return
        self._sig = sig
        t = self.tree
        scroll = t.verticalScrollBar().value()
        t.setUpdatesEnabled(False)
        self._building = True
        t.clear()
        self._details = []
        for row in m.rows:
            self._add(t.invisibleRootItem(), row, pal, more)
        self._building = False
        fix = any(self._any(r, lambda x: x.fix) for r in m.rows)
        for c in SHIFT_COLUMNS:
            t.setColumnHidden(c, not more or (c == C_REL and not m.has_rel))
        t.setColumnHidden(C_PEAK, not m.has_peak_column)
        t.setColumnHidden(C_FIX, not fix)
        shown_two_lines = any(not t.isColumnHidden(c) for c in SHIFT_COLUMNS)
        fm = QFontMetrics(_bold(t.font()))
        t.header().setFixedHeight(fm.height() * (2 if shown_two_lines else 1) + 8)
        self._head_tips(m)
        t.setUpdatesEnabled(True)
        self._fit_columns()
        self._reflow_details()
        t.verticalScrollBar().setValue(scroll)
        self._invalidate()

    @staticmethod
    def _any(row, test):
        return test(row) or any(AngleDotTree._any(c, test) for c in row.children)

    def _head_tips(self, m):
        hi = self.tree.headerItem()
        trim = _plateau_percent()
        tips = {C_NAME: "The angles of this trial (the main joint first, in bold) and, under each, the dots it is made of.",
                C_PEAK: "The joint angle at the moment of the highest force. Green is within the tolerance of the target, amber is outside it.",
                C_HOLD: f"The angle during the contraction (the middle {trim:.0f} % of the window), not the angle that was set.",
                C_TARGET: "The angle that was asked for, and the tolerance either way.",
                C_OFF: "The hold angle minus the target. Amber when it is outside the tolerance.",
                C_LOST: f"For a dot: the share of the pictures of the whole trial in which the program could not find it (more than {LOST_WARN_PCT:g} % is "
                        "amber). For an angle or a group: the worst of its dots.",
                C_SHIFT: "How far the dot ended from where it started, and the farthest it got from where it started, inside the window "
                         f"({'cm from the ruler scale' if m.cm else 'pixels: no ruler scale'}).",
                C_REL: "The same, compared with the anchor dot: movement the whole picture shares (a camera that was bumped) is taken away.",
                C_FIX: "Open the Fix dots workspace on the dot."}
        for c, tip in tips.items():
            hi.setToolTip(c, tip)

    def _add(self, parent, row, pal, more):
        """One item for `row` (and the items under it). A 'detail' line is a quiet wrapped text across the whole row."""
        if row.kind == "detail" and not (more or row.always):
            return
        item = QTreeWidgetItem()
        parent.addChild(item)
        item.setData(0, ROLE_KIND, row.kind)
        item.setData(0, ROLE_KEY, row.key)
        item.setData(0, ROLE_FIX, row.fix)
        if row.kind == "detail":
            item.setText(0, row.name)
            item.setData(0, ROLE_TEXT, row.name)
            item.setToolTip(0, row.tip)
            item.setFirstColumnSpanned(True)
            self._details.append((item, row.name))
            return
        base, bold, big = self._fonts()
        bold_row = row.main or row.kind == "group"
        height = ROW_MIN
        item.setText(C_NAME, row.name)
        item.setToolTip(C_NAME, row.tip)
        item.setFont(C_NAME, bold if bold_row else base)
        for col in range(1, N_COLUMNS):
            text = FIX_WORD if col == C_FIX and row.fix else row.cells.get(col, "")
            item.setText(col, text)
            item.setTextAlignment(col, (Qt.AlignLeft if col == C_FIX else Qt.AlignRight) | Qt.AlignVCenter)
            font = big if col in row.large else bold if bold_row else base
            if col == C_FIX:
                font = QFont(base)
                font.setUnderline(True)                                 # (a link)
            item.setFont(col, font)
            colour = STATE_COLOURS.get(row.states.get(col))
            if col == C_FIX and row.fix:
                item.setForeground(col, QBrush(QColor(ACCENT)))
                item.setToolTip(col, f"Open the Fix dots workspace on {row.key}.")
            elif colour:
                item.setForeground(col, QBrush(QColor(strong_colour(colour, pal))))
            elif col == C_LOST and text == "0 %":
                item.setForeground(col, QBrush(QColor(muted_colour(pal))))      # (a quiet zero)
            item.setToolTip(col, row.tips.get(col) or item.toolTip(col) or row.tip)
            if col in row.large:
                height = max(height, QFontMetrics(big).height() + 8)
        shade = QColor(WARN) if row.tinted else QColor(ACCENT) if row.main else None
        if shade is not None:
            shade.setAlpha(60 if row.tinted else 26)
            for col in range(N_COLUMNS):
                item.setBackground(col, QBrush(shade))
        item.setSizeHint(C_NAME, QSize(0, height))
        for d in row.details:
            self._add(item, d, pal, more)
        for c in row.children:
            self._add(item, c, pal, more)
        item.setExpanded(self._open.get(row.key, True) if row.kind in ("angle", "group") else False)

    # ------------------------------------------------------------ widths
    def _fonts(self):
        base = self.tree.font()
        big = _bold(base)
        big.setPointSizeF(PEAK_POINTS)
        return base, _bold(base), big

    def _natural(self, col):
        """(the width a column needs for its header and its widest cell with room round them, the least it can be squeezed to)."""
        base, bold, big = self._fonts()
        w = max(QFontMetrics(bold).horizontalAdvance(line) for line in HEADS[col].split("\n")) if HEADS[col] else 0
        for row in self._all_rows():
            if row.kind == "detail":
                continue
            text = FIX_WORD if col == C_FIX and row.fix else row.cells.get(col, "")
            if text:
                f = big if col in row.large else bold if (row.main or row.kind == "group") else base
                w = max(w, QFontMetrics(f).horizontalAdvance(text))
        return w + 2 * 5 + 6, w + 2 * 4 + 3

    def _all_rows(self):
        def walk(rows):
            for r in rows:
                yield r
                yield from walk(r.children)
        return list(walk(self._model.rows)) if self._model is not None else []

    def _fit_columns(self):
        """The numbers take what they need and the name takes the rest (cut with '...'). When the room is short the dot shifts are squeezed first,
        then the angle numbers (never below their text), before the name goes under NAME_MIN; below NAME_LEAST the tree scrolls sideways."""
        if self._model is None:
            return
        t = self.tree
        hh = t.header()
        W = t.viewport().width()
        if W <= 0:
            W = max(self.width() - 20, 300)
        cols = [c for c in range(1, N_COLUMNS) if not t.isColumnHidden(c)]
        nat, tight = {}, {}
        for c in cols:
            nat[c], tight[c] = self._natural(c)
        widths = dict(nat)
        need = sum(widths.values()) + NAME_MIN - W
        for group in ([c for c in cols if c in SHIFT_COLUMNS], [c for c in cols if c not in SHIFT_COLUMNS]):
            slack = sum(nat[c] - tight[c] for c in group)
            if need <= 0 or slack <= 0:
                continue
            take = min(need, slack)
            for c in group:
                widths[c] = nat[c] - (nat[c] - tight[c]) * take / slack
            need -= take
        for c in cols:
            hh.resizeSection(c, int(round(widths[c])))
        hh.resizeSection(C_NAME, max(NAME_LEAST, W - sum(int(round(widths[c])) for c in cols)))

    def _reflow_details(self):
        """The wrapped lines of More numbers are as tall as their text needs at the width the row has."""
        if not self._details:
            return
        t = self.tree
        fm = QFontMetrics(_small(t.font()))
        width = max(t.viewport().width() - 2 * t.indentation() - 8, 60)
        for item, text in self._details:
            h = fm.boundingRect(0, 0, width, 100000, int(Qt.AlignLeft | Qt.TextWordWrap), text).height() + 6
            if item.sizeHint(0).height() != h:
                item.setSizeHint(0, QSize(width, h))

    def eventFilter(self, obj, ev):
        if obj is self.tree.viewport() and ev.type() == QEvent.Resize and not self._fitting:
            self._fitting = True
            try:
                self._fit_columns()
                self._reflow_details()
            finally:
                self._fitting = False
            self._invalidate()
        return super().eventFilter(obj, ev)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._arrange_header()

    def _invalidate(self):
        self._hint_h = None
        self.updateGeometry()

    def sizeHint(self):
        if self._model is None:
            return QSize(600, 120)
        m = self.layout().contentsMargins()
        extra = self.head.sizeHint().height() + (self.note.sizeHint().height() + 4 if self.note.isVisibleTo(self) else 0) + 4
        return QSize(600, m.top() + m.bottom() + extra + self.content_height())

    def minimumSizeHint(self):
        return QSize(0, 150 if self._model is not None else 60)

    # ------------------------------------------------------------ clicks and folds
    def _item_clicked(self, item, col):
        kind = item.data(0, ROLE_KIND)
        if kind in ("angle", "group"):
            item.setExpanded(not item.isExpanded())
        elif kind == "dot" and col == C_FIX and item.data(0, ROLE_FIX):
            self.fixDotRequested.emit(str(item.data(0, ROLE_KEY)))

    def _item_double_clicked(self, item, col):
        if item.data(0, ROLE_KIND) == "dot" and col != C_FIX:        # (a double click on the link was already sent by its clicks)
            self.fixDotRequested.emit(str(item.data(0, ROLE_KEY)))

    def _folded(self, item):
        if self._building or item.data(0, ROLE_KIND) not in ("angle", "group"):
            return
        self._open[str(item.data(0, ROLE_KEY))] = item.isExpanded()
        self._invalidate()

    def _more_toggled(self, _on):
        if self._last is not None:
            self._fill_tree()
        self._arrange_header(force=True)

    # ------------------------------------------------------------ what a person can read
    def text(self, everything=False):
        """Plain lines of everything the tree shows: the heading with the window, the reference button, then each row with all its numbers and
        notes. The More numbers lines only while the switch is on, the dots of a folded angle only while it is open; `everything` adds them all,
        the explanatory sentences that are tooltips on the screen and the reference readings in full."""
        m = self._model
        if m is None:
            return self.empty.text()
        more = everything or self.more_check.isChecked()
        lines = [self.heading.text()]
        if m.no_markers:
            lines.append(NO_MARKERS)
        lines.append(f"[{m.reference_text}]")
        if everything:
            lines += m.reference_lines
        if more and any(r.kind == "dot" or r.children for r in m.rows):
            lines.append(m.movement)
        for r in m.rows:
            lines += self._lines(r, more, everything, "")
        if everything:
            lines += m.notes
        return "\n".join(lines)

    def _lines(self, row, more, everything, indent):
        out = [indent + row.line + (f", {row.extra}" if more and row.extra else "") + row.suffix]
        if more:
            out += [f"{indent}  {d.line}" for d in row.details]
        if row.children and (everything or self._open.get(row.key, True)):
            for c in row.children:
                out += self._lines(c, more, everything, indent + "  ")
        return out

    # ------------------------------------------------------------ the look
    def _apply_style(self):
        pal = QApplication.palette()
        muted, border = muted_colour(pal), border_colour(pal)
        css = f"""
QLabel#anglesEmpty {{ color: {muted}; }}
QLabel#anglesNote {{ color: {strong_colour(WARN, pal)}; }}
QTreeWidget#angleTree {{ border: 1px solid {border}; border-radius: 4px; }}
QTreeWidget#angleTree::item {{ padding: 1px 4px; }}
QTreeWidget#angleTree QHeaderView::section {{ padding: 2px 4px; border: none; border-bottom: 1px solid {border}; font-weight: 600; background: transparent; }}
QPushButton#angleRef {{ border: 1px solid {border}; border-radius: 4px; padding: 2px 10px; }}
QPushButton#angleRef:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QCheckBox#angleMore {{ color: {strong_colour(ACCENT, pal, 0.8)}; }}
QCheckBox#angleMore::indicator {{ width: 12px; height: 12px; border: 1px solid {muted}; border-radius: 3px; background: transparent; }}
QCheckBox#angleMore::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
"""
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_style()
            if self._last is not None:
                self._sig = None                                       # (the colours of the cells follow the palette)
                self._fill_tree()
            self.update()
