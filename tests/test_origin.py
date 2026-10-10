"""Checks that the cleaned code reproduces the original research code.

Run from the repository root:  python -m pytest -q
"""

import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from origin import accelerated_origin, iterative_origin, quantization_errors  # noqa: E402
from vcr import ideal_shift, reconstruct, shifts_fixed, shifts_nonuniform, shifts_relocated  # noqa: E402


def original_errors(n, shift):
    """E(P) as computed by the first research script (prefix sums of the gaps of k * S).

    Rounding errors are symmetric, |(-j)S - round(-jS)| = |jS - round(jS)|, so
    E(P) only needs the gaps of the origin-0 positions.
    """
    s = Decimal(str(shift))
    ideal = [k * s for k in range(n)]
    gap = [abs(v - math.floor(v + Decimal("0.5"))) for v in ideal]
    return [sum(gap) - sum(gap[n - p:]) + sum(gap[1:1 + p]) for p in range(n)]


def test_errors_match_original():
    for n in (5, 7, 9, 10):
        for shift in (1.3, 2.5, 4.37, 7.81, 10.416666666666666, 12.0, 22.248, 29.664):
            assert quantization_errors(n, shift) == original_errors(n, shift)


def test_recorded_notebook_output():
    # 9 images, S = 300 * 50 * 0.002 / 24 / 0.285, output recorded in the research notebook
    errors = quantization_errors(9, 4.4642857142857135)
    best = min(errors)
    assert best == Decimal("2.1428571428571460")
    assert [p for p, e in enumerate(errors) if e == best] == [0, 2, 4, 6, 8]
    assert accelerated_origin(9, 4.4642857142857135)[0] == 0


def test_paper_example_1d():
    # ITC-CSCC 2025, Tables II-III: S = 1.3, positions 0..5 from origin 0 -> 0 1 3 4 5 7
    assert shifts_nonuniform(6, 1.3).tolist() == [0, 1, 3, 4, 5, 7]
    # origin moved by one image: ideal -1.3 0 1.3 2.6 3.9 5.2 -> -1 0 1 3 4 5, shifted to start at 0
    assert shifts_relocated(6, 1.3, 1).tolist() == [0, 1, 2, 4, 5, 6]
    # fixed VCR uses round(1.3) = 1 for every step
    assert shifts_fixed(6, 1.3).tolist() == [0, 1, 2, 3, 4, 5]


def test_shifts_match_original():
    # formulas of the final research notebook
    for n in (7, 9):
        for shift in (1.3, 4.37, 10.416666666666666, 29.664):
            assert np.array_equal(shifts_nonuniform(n, shift), np.floor(np.arange(n) * shift + 0.5).astype(np.int64))
            assert np.array_equal(shifts_fixed(n, shift), np.arange(n) * int(np.floor(shift + 0.5)))
            for p in range(n):
                coords = np.floor((np.arange(n) - p) * shift + 0.5).astype(np.int64)
                assert np.array_equal(shifts_relocated(n, shift, p), coords - coords.min())


def test_ideal_shift_paper_setting():
    # ITC-CSCC 2026 simulation: 1000 px, 30 mm pitch, f = 50 mm, 36 mm sensor, 4 m depth
    assert math.isclose(ideal_shift(1000, 30, 50, 36, 4000), 10.416666666666666)


def test_accelerated_origin_is_minimum():
    for shift in (1.3, 4.37, 7.81, 22.248):
        best, errors = accelerated_origin(9, shift)
        assert errors[best] == min(errors)


def test_reconstruct_constant_images():
    images = np.full((3, 3, 8, 8, 3), 100, dtype=np.uint8)
    out = reconstruct(images, np.array([0, 2, 4]), np.array([0, 3, 6]))
    assert out.shape == (14, 12, 3)
    assert np.allclose(out, 100.0)


def test_iterative_search_runs():
    rng = np.random.default_rng(0)
    images = rng.random((3, 3, 6, 6, 3))
    origin, _ = iterative_origin(images, 1.4, 1.4, score=lambda img: -float(img.std()))
    assert origin[0] in range(3) and origin[1] in range(3)
