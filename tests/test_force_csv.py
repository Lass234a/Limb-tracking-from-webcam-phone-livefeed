"""read_force_csv (0.29.0) gives the same numbers as the csv.DictReader it replaced, also for a file a crash left damaged."""
import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from limbtrack.review import _f, read_force_csv  # noqa: E402


def _old(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return tuple(np.array([_f(r[c]) for r in rows]) for c in ("t_s", "volts", "newtons"))


def _same(a, b):
    return all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))


def test_a_normal_file_is_read_exactly_as_before(tmp_path):
    p = tmp_path / "a_force.csv"
    t = np.arange(0, 2, 0.001)
    p.write_text("t_s,clock_s,volts,newtons\n" + "\n".join(f"{x:.5f},{100 + x:.5f},{0.4 + 0.1 * x:.6f},{200 * x:.3f}" for x in t) + "\n")
    got = read_force_csv(p)
    assert _same(got, _old(p)) and got[0].shape == (2000,) and got[1].flags["C_CONTIGUOUS"]


def test_an_uncalibrated_file_has_nan_newtons(tmp_path):
    p = tmp_path / "b_force.csv"
    p.write_text("t_s,clock_s,volts,newtons\n0.1,5,0.4,\n0.2,5.1,0.5,\n")
    t, v, n = read_force_csv(p)
    assert list(t) == [0.1, 0.2] and list(v) == [0.4, 0.5] and np.isnan(n).all()
    assert _same((t, v, n), _old(p))


def test_a_file_a_crash_cut_short_keeps_every_line_and_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "c_force.csv"
    p.write_text("t_s,clock_s,volts,newtons\n0.1,5,0.4,\n0.2,5.1,0.5,100\n\n0.3,5.2,abc,7\n0.4,5.3,0.6")
    got = read_force_csv(p)
    assert _same(got, _old(p))
    assert list(got[0]) == [0.1, 0.2, 0.3, 0.4] and np.isnan(got[1][2]) and np.isnan(got[2][3])


def test_a_file_without_samples_gives_none(tmp_path):
    p = tmp_path / "d_force.csv"
    p.write_text("t_s,clock_s,volts,newtons\n")
    assert read_force_csv(p) is None
    p.write_text("")
    assert read_force_csv(p) is None


def test_a_file_without_a_needed_column_is_an_error_not_a_guess(tmp_path):
    p = tmp_path / "e_force.csv"
    p.write_text("t_s,clock_s,volts\n0.1,5,0.4\n")
    with pytest.raises(ValueError):
        read_force_csv(p)
