"""Image-quality helpers used for the PSNR / SSIM / MSE comparisons."""

import cv2
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def crop_black_border(image, tol=0):
    """Crop the black margin around a reconstruction (use only on black backgrounds)."""
    mask = image > tol
    if image.ndim == 3:
        mask = mask.all(axis=2)
    cols, rows = mask.any(axis=0), mask.any(axis=1)
    if not cols.any():
        return image
    c0, c1 = cols.argmax(), len(cols) - cols[::-1].argmax()
    r0, r1 = rows.argmax(), len(rows) - rows[::-1].argmax()
    return image[r0:r1, c0:c1]


def center_crop(image, size):
    """Center crop to a (height, width) region."""
    h, w = image.shape[:2]
    ch, cw = size
    top, left = (h - ch) // 2, (w - cw) // 2
    return image[top:top + ch, left:left + cw]


def compare(reference, image):
    """PSNR (color), SSIM and MSE (grayscale) of ``image`` against ``reference``.

    Both inputs are 8-bit BGR images; ``image`` is resized to the reference size.
    """
    image = cv2.resize(image, (reference.shape[1], reference.shape[0]))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "psnr": float(peak_signal_noise_ratio(reference, image, data_range=255)),
        "ssim": float(structural_similarity(ref_gray, img_gray, data_range=255)),
        "mse": float(mean_squared_error(ref_gray, img_gray)),
    }


def to_uint8(image):
    """Clip a float reconstruction to 0..255 and convert to uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)
