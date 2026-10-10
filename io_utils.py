"""Loading a grid of elemental images from disk."""

import os

import cv2
import numpy as np


def load_elemental_images(folder, lx, ly, pattern="DSC_{index:04d}.tif", start=1):
    """Load Lx x Ly 8-bit elemental images into a uint8 array of shape (Ly, Lx, Ny, Nx, 3).

    Images are numbered row by row: index = start + y * Lx + x.
    ``pattern`` is a format string with an ``index`` field.
    """
    rows = []
    for y in range(ly):
        row = []
        for x in range(lx):
            path = os.path.join(folder, pattern.format(index=start + y * lx + x))
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            row.append(image)
        rows.append(row)
    return np.asarray(rows, dtype=np.uint8)
