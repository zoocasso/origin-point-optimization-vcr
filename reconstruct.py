"""Command-line reconstruction with fixed, nonuniform, or proposed VCR.

Example (the 9 x 9 simulation setting of ITC-CSCC 2026, lengths in mm,
images named DSC_0001.tif .. DSC_0081.tif row by row):
    python reconstruct.py --input data/usaf_9x9 --lx 9 --ly 9 \
        --focal-length 50 --sensor-width 36 --sensor-height 36 \
        --pitch 30 --depth 4000 --method proposed --output proposed.png
"""

import argparse
import time

import cv2

from io_utils import load_elemental_images
from metrics import to_uint8
from origin import accelerated_origin
from vcr import ideal_shift, reconstruct_method


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="folder with the elemental images")
    parser.add_argument("--pattern", default="DSC_{index:04d}.tif", help="file name pattern with an {index} field")
    parser.add_argument("--start", type=int, default=1, help="index of the first image")
    parser.add_argument("--lx", type=int, required=True, help="number of elemental images per row")
    parser.add_argument("--ly", type=int, required=True, help="number of elemental images per column")
    parser.add_argument("--focal-length", type=float, required=True)
    parser.add_argument("--sensor-width", type=float, required=True, help="same unit as the focal length")
    parser.add_argument("--sensor-height", type=float, required=True, help="same unit as the focal length")
    parser.add_argument("--pitch", type=float, required=True, help="camera pitch, same unit as the depth")
    parser.add_argument("--depth", type=float, required=True, help="reconstruction depth")
    parser.add_argument("--method", choices=["fixed", "nonuniform", "proposed"], default="proposed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    images = load_elemental_images(args.input, args.lx, args.ly, args.pattern, args.start)
    ny, nx = images.shape[2:4]
    shift_x = ideal_shift(nx, args.pitch, args.focal_length, args.sensor_width, args.depth)
    shift_y = ideal_shift(ny, args.pitch, args.focal_length, args.sensor_height, args.depth)

    t0 = time.perf_counter()
    if args.method == "proposed":
        px, _ = accelerated_origin(args.lx, shift_x)
        py, _ = accelerated_origin(args.ly, shift_y)
        t1 = time.perf_counter()
        result = reconstruct_method(images, shift_x, shift_y, "relocated", (px, py))
        print(f"origin (Px, Py) = ({px}, {py}), optimization {t1 - t0:.6f} s")
    else:
        result = reconstruct_method(images, shift_x, shift_y, args.method)
    print(f"S = ({shift_x:.4f}, {shift_y:.4f}) px, total {time.perf_counter() - t0:.4f} s")

    cv2.imwrite(args.output, to_uint8(result))


if __name__ == "__main__":
    main()
