"""Origin point optimization for nonuniform VCR.

accelerated_origin  -- ITC-CSCC 2026: pick the origin from camera parameters
                       only, by minimizing the total quantization error (Eq. (4)-(5)).
iterative_origin    -- the earlier image-based search: reconstruct for every
                       candidate origin and keep the best image-quality score.
"""

import math
from decimal import Decimal, getcontext

import numpy as np

from vcr import reconstruct_method

getcontext().prec = 28


def quantization_errors(n_images, shift):
    """Total quantization error E(P) for every candidate origin P = 0 .. n-1.

    E(P) = sum_k | (k - P) * S - round((k - P) * S) |
    Decimal arithmetic keeps exact .5 cases from flipping because of float error.
    """
    s = Decimal(str(shift))
    half = Decimal("0.5")
    errors = []
    for p in range(n_images):
        total = Decimal(0)
        for k in range(n_images):
            ideal = (k - p) * s
            total += abs(ideal - math.floor(ideal + half))
        errors.append(total)
    return errors


def accelerated_origin(n_images, shift):
    """Origin index with the smallest quantization error (image-independent).

    Returns (best_origin, errors). When several origins tie, the smallest index
    is returned; all tied indices give the same total error.
    """
    errors = quantization_errors(n_images, shift)
    best = min(range(n_images), key=lambda p: errors[p])
    return best, errors


def iterative_origin(images, shift_x, shift_y, score):
    """Image-based search over every (Px, Py): Lx * Ly full reconstructions.

    ``score`` maps a reconstructed image to a number where larger is better
    (for example PSNR against a reference). Returns ((Px, Py), best_score).
    """
    ly, lx = np.asarray(images).shape[:2]
    best_origin, best_score = None, -np.inf
    for py in range(ly):
        for px in range(lx):
            image = reconstruct_method(images, shift_x, shift_y, "relocated", (px, py))
            value = score(image)
            if value > best_score:
                best_origin, best_score = (px, py), value
    return best_origin, best_score
