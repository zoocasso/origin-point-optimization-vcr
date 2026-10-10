"""Volumetric computational reconstruction (VCR) for integral imaging.

Three ways to place the elemental images on the reconstruction plane:

* fixed VCR       -- every neighbouring pair is shifted by the same rounded step
* nonuniform VCR  -- each image is shifted by round(k * S) from the first image
* relocated VCR   -- each image is shifted by round((k - P) * S) from an origin
                     image P (the origin point optimized in origin.py)

S is the ideal, real-valued pixel shift between neighbouring elemental images
(Eq. (1) of the papers listed in README.md).
"""

import numpy as np


def ideal_shift(n_pixels, pitch, focal_length, sensor_size, depth):
    """Ideal pixel shift between neighbouring elemental images.

    ``pitch`` and ``depth`` must use the same unit, and so must
    ``focal_length`` and ``sensor_size``.
    """
    return n_pixels * pitch * focal_length / sensor_size / depth


def round_half_up(values):
    """Round to the nearest integer with halves rounded up (floor(x + 0.5))."""
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5).astype(np.int64)


def shifts_fixed(n_images, shift):
    """Integer offsets for fixed VCR: one rounded step reused for every image."""
    step = int(round_half_up(shift))
    return np.arange(n_images, dtype=np.int64) * step


def shifts_nonuniform(n_images, shift):
    """Integer offsets for nonuniform VCR: round(k * S) for k = 0 .. n-1."""
    return round_half_up(np.arange(n_images) * shift)


def shifts_relocated(n_images, shift, origin):
    """Integer offsets for nonuniform VCR with the origin at image ``origin``.

    The positions are rounded relative to the origin, round((k - P) * S), and
    then translated so that the smallest offset is 0 (the canvas starts at 0).
    """
    coords = round_half_up((np.arange(n_images) - origin) * shift)
    return coords - coords.min()


def reconstruct(images, shifts_x, shifts_y):
    """Average the elemental images placed at the given integer offsets.

    images   : array of shape (Ly, Lx, Ny, Nx, C), elemental image (row y, column x)
    shifts_x : integer offsets for the Lx columns
    shifts_y : integer offsets for the Ly rows
    Returns a float64 image of shape (Ny + max(shifts_y), Nx + max(shifts_x), C).
    """
    images = np.asarray(images)
    ly, lx, ny, nx, ch = images.shape
    height = ny + int(np.max(shifts_y))
    width = nx + int(np.max(shifts_x))
    canvas = np.zeros((height, width, ch), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)

    for y in range(ly):
        for x in range(lx):
            y0, x0 = int(shifts_y[y]), int(shifts_x[x])
            canvas[y0:y0 + ny, x0:x0 + nx, :] += images[y, x]
            count[y0:y0 + ny, x0:x0 + nx] += 1

    return canvas / np.maximum(count, 1)[..., None]


def reconstruct_method(images, shift_x, shift_y, method, origin=(0, 0)):
    """Reconstruct with ``method`` in {"fixed", "nonuniform", "relocated"}.

    ``origin`` is (Px, Py) and is only used by the relocated method.
    """
    ly, lx = np.asarray(images).shape[:2]
    if method == "fixed":
        sx, sy = shifts_fixed(lx, shift_x), shifts_fixed(ly, shift_y)
    elif method == "nonuniform":
        sx, sy = shifts_nonuniform(lx, shift_x), shifts_nonuniform(ly, shift_y)
    elif method == "relocated":
        sx = shifts_relocated(lx, shift_x, origin[0])
        sy = shifts_relocated(ly, shift_y, origin[1])
    else:
        raise ValueError(f"unknown method: {method}")
    return reconstruct(images, sx, sy)
