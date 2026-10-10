"""Timing of the iterative and the accelerated origin search (Tables III-IV, ITC-CSCC 2026).

The iterative method needs one full reconstruction per candidate origin
(Lx * Ly reconstructions); the accelerated method evaluates E(P) from the
camera parameters only and then reconstructs once.

Runs on random elemental images, so no dataset is needed. The defaults are the
simulation setting of the paper (9 x 9 images of 1000 x 1000 px, f = 50 mm,
36 mm sensor, 30 mm pitch, 4 m depth):
    python benchmark.py
    python benchmark.py --gpu        # CuPy, needs a CUDA GPU
"""

import argparse
import time

import numpy as np

from origin import accelerated_origin
from vcr import ideal_shift, reconstruct, shifts_relocated


def reconstruct_gpu(cp, images, sx, sy):
    """CuPy version of vcr.reconstruct (float32 on the GPU)."""
    ly, lx, ny, nx, ch = images.shape
    canvas = cp.zeros((ny + int(sy.max()), nx + int(sx.max()), ch), dtype=cp.float32)
    count = cp.zeros(canvas.shape[:2], dtype=cp.float32)
    for y in range(ly):
        for x in range(lx):
            y0, x0 = int(sy[y]), int(sx[x])
            canvas[y0:y0 + ny, x0:x0 + nx] += images[y, x]
            count[y0:y0 + ny, x0:x0 + nx] += 1
    return canvas / cp.maximum(count, 1)[..., None]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lx", type=int, default=9)
    parser.add_argument("--ly", type=int, default=9)
    parser.add_argument("--size", type=int, default=1000, help="elemental image width and height in pixels")
    parser.add_argument("--focal-length", type=float, default=50.0, help="mm")
    parser.add_argument("--sensor-size", type=float, default=36.0, help="mm")
    parser.add_argument("--pitch", type=float, default=30.0, help="mm")
    parser.add_argument("--depth", type=float, default=4000.0, help="mm")
    parser.add_argument("--repeat", type=int, default=3, help="timed reconstructions; the fastest is reported")
    parser.add_argument("--gpu", action="store_true", help="time the reconstruction with CuPy")
    args = parser.parse_args()

    shift = ideal_shift(args.size, args.pitch, args.focal_length, args.sensor_size, args.depth)
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (args.ly, args.lx, args.size, args.size, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    px, _ = accelerated_origin(args.lx, shift)
    py, _ = accelerated_origin(args.ly, shift)
    t_opt = time.perf_counter() - t0
    sx, sy = shifts_relocated(args.lx, shift, px), shifts_relocated(args.ly, shift, py)

    if args.gpu:
        import cupy as cp

        images = cp.asarray(images, dtype=cp.float32)
        sync = cp.cuda.Stream.null.synchronize

        def run():
            reconstruct_gpu(cp, images, sx, sy)

        run()  # warm-up: kernel compilation
    else:
        def sync():
            pass

        def run():
            reconstruct(images, sx, sy)

    times = []
    for _ in range(args.repeat):
        sync()
        t = time.perf_counter()
        run()
        sync()
        times.append(time.perf_counter() - t)
    t_rec = min(times)

    n = args.lx * args.ly
    print(f"S = {shift:.4f} px, origin (Px, Py) = ({px}, {py})")
    print(f"one reconstruction ({'GPU' if args.gpu else 'CPU'}): {t_rec:.4f} s")
    print(f"iterative search : {n} x {t_rec:.4f} s = {n * t_rec:.4f} s (one reconstruction per candidate origin)")
    print(f"accelerated      : {t_opt:.4f} s + {t_rec:.4f} s = {t_opt + t_rec:.4f} s")


if __name__ == "__main__":
    main()
