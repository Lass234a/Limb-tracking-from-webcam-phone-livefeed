"""The load cell's calibration regressions: kept, chosen, and turned into a precision for any force.

A calibration session hangs known masses on the cell and records the amplifier's voltage. A straight line through the
(mass, volts) points gives the slope (volts per kg, so newtons per volt) and, from the scatter of the points around the line, how
precise a reading is. Every regression ever made is kept in `load_cell_calibrations.json` next to this file, one is `active`,
and the program uses the active one from the start (the operator then only presses Zero, which sets the offset for the day).

A new calibration is added with `LoadCellCalibration/add_calibration.py` (no code change): it fits the points with `fit_points`
below and stores the result with the points, so any regression can be re-examined or re-fitted later. Trials record the id of the
regression they were measured with, so Review can show the regression and the precision of the forces it reports.

Precision model (all from the calibration, in kg of the hanging mass, 1 standard deviation):
    sd(L) = sqrt(rmse^2 + (L * slope_se / slope)^2)
`rmse` is the scatter of single runs around the line (it includes the cell reading higher when unloading than when loading), the second
term is how well the slope itself is known and grows with the load L. It is the precision of the calibration; the zero adds the
noise of the short average it was taken from, and the pull direction, mounting and temperature are not covered.
"""
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

GRAVITY = 9.80665                      # m/s^2: a mass of 1 kg weighs 9.80665 N
REGISTRY_FILE = Path(__file__).with_name("load_cell_calibrations.json")
COVERAGE = 2.0                         # "about 95 %" is this many standard deviations


@dataclass
class Regression:
    """One calibration of the load cell: volts = intercept_v + slope_v_per_kg * mass_kg."""
    id: str
    date: str = ""
    description: str = ""
    n_runs: int = 0
    range_kg: list = field(default_factory=lambda: [0.0, 0.0])
    slope_v_per_kg: float = math.nan
    slope_se_v_per_kg: float = math.nan
    intercept_v: float = math.nan
    rmse_kg: float = math.nan
    r2: float = math.nan
    hysteresis_kg: float = None              # reading when unloading minus loading, at the same mass (None without up and down runs)
    curvature_kg_at_max: float = None        # the squared term of a curved fit, as kg at the heaviest load (None when not fitted)
    curvature_se_kg: float = None
    points: list = field(default_factory=list)   # [{"kg", "volts", "phase", "run"}] of the runs used

    @property
    def n_per_v(self):
        return GRAVITY / self.slope_v_per_kg

    @property
    def n_per_v_se(self):
        return self.n_per_v * self.slope_se_v_per_kg / self.slope_v_per_kg

    @property
    def slope_rel_se(self):
        return self.slope_se_v_per_kg / self.slope_v_per_kg

    def sd_kg(self, kg):
        """Standard deviation (kg) of one reading at a load of `kg` (a number or an array)."""
        return np.sqrt(self.rmse_kg ** 2 + (np.abs(kg) * self.slope_rel_se) ** 2)

    def sd_newton(self, newtons):
        """Standard deviation (N) of one reading of `newtons`."""
        return self.sd_kg(np.asarray(newtons, dtype=np.float64) / GRAVITY) * GRAVITY

    def residuals_kg(self):
        """(mass, residual kg, phase) of every run used: how far each reading is from the line."""
        kg = np.array([p["kg"] for p in self.points], dtype=float)
        v = np.array([p["volts"] for p in self.points], dtype=float)
        return kg, (v - self.intercept_v - self.slope_v_per_kg * kg) / self.slope_v_per_kg, [p.get("phase", "") for p in self.points]

    def summary_lines(self):
        """Plain-English lines about the regression (Review, the Live read-out and the script print them)."""
        lo, hi = self.range_kg
        lines = [f"Regression {self.id}: {self.n_runs} runs, {lo:g} to {hi:g} kg" + (f", {self.description}" if self.description else ""),
                 f"  {self.slope_v_per_kg * 1000:.4f} mV per kg  =  {self.n_per_v:.2f} N per volt  "
                 f"(slope known to {100 * self.slope_rel_se:.3f} %, +-{self.n_per_v_se:.2f} N/V)",
                 f"  scatter of single runs around the line {self.rmse_kg:.3f} kg ({self.rmse_kg * GRAVITY:.2f} N), R2 {self.r2:.6f}"]
        if self.hysteresis_kg is not None:
            lines.append(f"  unloading reads {self.hysteresis_kg:+.2f} kg higher than loading at the same mass (hysteresis, part of the scatter)")
        if self.curvature_kg_at_max is not None:
            lines.append(f"  curvature at {hi:g} kg: {self.curvature_kg_at_max:+.2f} +- {self.curvature_se_kg:.2f} kg (a straight line is used)")
        for kg in (10, 50, 100, 250):
            if kg <= hi * 1.001:
                lines.append(f"  a reading of {kg:g} kg ({kg * GRAVITY:.0f} N) is good to +-{COVERAGE * float(self.sd_kg(kg)):.2f} kg "
                             f"(+-{COVERAGE * float(self.sd_newton(kg * GRAVITY)):.1f} N, {100 * COVERAGE * float(self.sd_kg(kg)) / kg:.2f} %), about 95 %")
        return lines

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ------------------------------------------------------------------------------------------- fitting
def _ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(r @ r) / dof
    return beta, s2 * np.linalg.inv(X.T @ X), r, math.sqrt(s2)


def fit_points(kg, volts, phase=None, id="", date="", description="", runs=None):
    """The regression of `volts` on `kg`. `phase` ('up' / 'down' / 'peak' per run) is optional and only used for the hysteresis.
    `runs` names each run (kept with the points). Needs at least 4 runs over a range of masses."""
    kg, volts = np.asarray(kg, dtype=float), np.asarray(volts, dtype=float)
    if kg.size < 4 or np.ptp(kg) <= 0:
        raise ValueError("At least 4 runs with different masses are needed for a regression.")
    phase = list(phase) if phase is not None else [""] * kg.size
    X = np.column_stack([np.ones_like(kg), kg])
    (a, b), cov, res, rmse_v = _ols(X, volts)
    if b <= 0:
        raise ValueError(f"The fitted slope is {b * 1000:.3f} mV/kg: the signal must rise with the mass (check the wiring or the points).")
    out = Regression(id=id, date=date, description=description, n_runs=int(kg.size), range_kg=[float(kg.min()), float(kg.max())],
                     slope_v_per_kg=float(b), slope_se_v_per_kg=float(math.sqrt(cov[1, 1])), intercept_v=float(a), rmse_kg=float(rmse_v / b),
                     r2=float(1 - (res @ res) / ((volts - volts.mean()) ** 2).sum()))
    down = np.array([p == "down" for p in phase], dtype=float)
    if down.any() and (down == 0).any() and kg.size >= 6:
        b3, _, _, _ = _ols(np.column_stack([X, down]), volts)
        out.hysteresis_kg = float(b3[2] / b3[1])
    if kg.size >= 6 and len(set(kg.tolist())) >= 4:
        b2, c2, _, _ = _ols(np.column_stack([X, kg ** 2]), volts)
        top = float(kg.max()) ** 2 / b2[1]
        out.curvature_kg_at_max, out.curvature_se_kg = float(b2[2] * top), float(math.sqrt(c2[2, 2]) * top)
    names = list(runs) if runs is not None else [""] * kg.size
    out.points = [{"kg": float(k), "volts": float(v), "phase": p, "run": n} for k, v, p, n in zip(kg, volts, phase, names)]
    return out


# ------------------------------------------------------------------------------------------- the registry
def load_registry(path=None):
    """(active id or None, {id: Regression}) from the registry file; empty when the file is missing or unreadable."""
    try:
        d = json.loads(Path(path or REGISTRY_FILE).read_text(encoding="utf-8"))
        regs = {r["id"]: Regression.from_dict(r) for r in d.get("calibrations", [])}
        active = d.get("active")
        return (active if active in regs else None), regs
    except (OSError, ValueError, KeyError, TypeError):
        return None, {}


def save_registry(active, regs, path=None):
    d = {"active": active, "calibrations": [r.to_dict() for r in regs.values()]}
    Path(path or REGISTRY_FILE).write_text(json.dumps(d, indent=1), encoding="utf-8")


def active_regression(path=None):
    """The regression the program uses, or None when there is none."""
    active, regs = load_registry(path)
    return regs.get(active)


def find_regression(reg_id, path=None):
    """The kept regression with this id (a trial records the id it was measured with), or None."""
    return load_registry(path)[1].get(reg_id) if reg_id else None


def add_regression(reg, activate=False, path=None):
    """Keep `reg` (a regression with the same id is replaced). It becomes the active one when asked or when there was none."""
    active, regs = load_registry(path)
    regs[reg.id] = reg
    save_registry(reg.id if activate or active is None else active, regs, path)
