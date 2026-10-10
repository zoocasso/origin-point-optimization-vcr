# Origin Point Optimization for Volumetric Computational Reconstruction

Python implementation of two of my first-author conference papers on the quantization error of volumetric computational reconstruction (VCR) in integral imaging:

- *Image Enhancement of Volumetric Computational Reconstruction Using Origin Point Optimization* — ITC-CSCC 2025 (Seoul)
- *Accelerated Origin Point Optimization for Volumetric Computational Reconstruction* — ITC-CSCC 2026 (Bangkok)

## Idea

VCR reconstructs a depth plane by shifting each elemental image and averaging the overlaps. The ideal shift between neighbouring images is a real number

$$S = \frac{N \cdot p \cdot f}{c \cdot z}$$

($N$ pixels per elemental image, $p$ camera pitch, $f$ focal length, $c$ sensor size, $z$ reconstruction depth), but images can only be placed on integer pixels, so every position is rounded.

| Method | Integer offset of image $k$ |
|---|---|
| Fixed VCR | $k \cdot \mathrm{round}(S)$ — the rounding error grows with $k$ |
| Nonuniform VCR | $\mathrm{round}(k S)$ — measured from the first image |
| Origin point VCR | $\mathrm{round}((k - P) S)$ — measured from an origin image $P$ |

The total quantization error depends on which image is the origin:

$$E(P) = \sum_{k=0}^{L-1} \left| (k-P)S - \mathrm{round}\big((k-P)S\big) \right|, \qquad P^{*} = \arg\min_{P \in \{0,\dots,L-1\}} E(P)$$

- **2025 (iterative):** reconstruct with every candidate origin and keep the best image, which takes $L_x \times L_y$ full reconstructions.
- **2026 (accelerated):** $E(P)$ needs only the camera parameters, so $P^{*}$ is found per axis without touching the images. The result is the same origin and image quality, with one reconstruction.

## Results reported in the papers

The two papers use different scenes, image sizes and evaluation pipelines, so their numbers should not be compared with each other.

### ITC-CSCC 2025 — moving the origin improves image quality

Metrics are computed against the central elemental image. In the simulation, the optimized origin gives about 40 % lower MSE than nonuniform VCR (5.5078 → 3.3410).

Simulation with a 1951 USAF chart rendered in Unity: 9 × 9 elemental images of 3000 × 3000 px, object at 3,100 mm.

| Method | PSNR | SSIM | MSE |
|---|---:|---:|---:|
| Fixed VCR | 34.2666 | 0.9736 | 23.4585 |
| Nonuniform VCR | 40.5041 | 0.9884 | 5.5078 |
| Optimized origin point VCR | 42.6161 | 0.9908 | 3.3410 |

Real object captured with a camera: 9 × 9 elemental images of 4178 × 2752 px, object at 420 mm.

| Method | PSNR | SSIM | MSE |
|---|---:|---:|---:|
| Fixed VCR | 28.3032 | 0.8962 | 92.8568 |
| Nonuniform VCR | 33.1142 | 0.9692 | 28.1018 |
| Optimized origin point VCR | 34.6421 | 0.9763 | 19.1821 |

### ITC-CSCC 2026 — same quality with one reconstruction instead of 81

Simulation with a 1951 USAF chart: 9 × 9 elemental images of 1000 × 1000 px, f = 50 mm, pitch 30 mm, depth 4 m. Metrics are computed on a 180 × 180 center crop.

| Method | PSNR | SSIM | MSE |
|---|---:|---:|---:|
| Fixed VCR | 24.249 | 0.865 | 239.311 |
| Nonuniform VCR | 26.576 | 0.909 | 139.950 |
| Optimized origin point VCR (2025, iterative) | 27.816 | 0.916 | 104.785 |
| Proposed VCR (2026, accelerated) | 27.816 | 0.916 | 104.785 |

Time for 9 × 9 elemental images:

| Method | One reconstruction | Origin search | Total |
|---|---:|---:|---:|
| Iterative, CPU | 0.3142 s | 81 reconstructions | 25.4502 s |
| Accelerated, CPU | 0.3142 s | 0.0002 s | 0.3144 s |
| Iterative, GPU (CuPy, RTX 3090) | 0.0055 s | 81 reconstructions | 0.4455 s |
| Accelerated, GPU (CuPy, RTX 3090) | 0.0055 s | 0.0002 s | 0.0057 s |

The paper also checks object detection on a real toy-car scene (YOLOv8n). There, the accelerated method gives the same IoU (0.9780) and mAP (0.9950) as the iterative one. The detection pipeline is not part of this repository.

## Files

| File | Contents |
|---|---|
| `vcr.py` | ideal shift, integer offsets for fixed / nonuniform / origin-relocated VCR, reconstruction |
| `origin.py` | `quantization_errors` and `accelerated_origin` (2026), `iterative_origin` (2025) |
| `metrics.py` | PSNR / SSIM / MSE and cropping helpers |
| `io_utils.py` | loads an Lx × Ly grid of elemental images |
| `reconstruct.py` | command-line reconstruction |
| `benchmark.py` | iterative vs. accelerated timing on random images (CPU or CuPy) |
| `tests/` | checks against the original research scripts and the examples in the papers |

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q
python benchmark.py            # paper setting on random images; add --gpu for CuPy
```

To reconstruct your own elemental images, name them `DSC_0001.tif` … row by row (or pass `--pattern`). Lengths are in mm.

```bash
python reconstruct.py --input data/usaf_9x9 --lx 9 --ly 9 \
    --focal-length 50 --sensor-width 36 --sensor-height 36 \
    --pitch 30 --depth 4000 --method proposed --output proposed.png
```

From Python:

```python
from io_utils import load_elemental_images
from origin import accelerated_origin
from vcr import ideal_shift, reconstruct_method

images = load_elemental_images("data/usaf_9x9", lx=9, ly=9)   # (Ly, Lx, Ny, Nx, 3)
s = ideal_shift(1000, pitch=30, focal_length=50, sensor_size=36, depth=4000)
px, _ = accelerated_origin(9, s)
py, _ = accelerated_origin(9, s)
result = reconstruct_method(images, s, s, "relocated", origin=(px, py))
```

## Notes

- Rounding is half-up, `floor(x + 0.5)`, as in the papers. $E(P)$ is evaluated with `Decimal`, so exact .5 cases do not flip because of floating-point error.
- When several origins give the same minimum error, `accelerated_origin` returns the smallest index.
- In the papers, the iterative search time is the single reconstruction time × 81. `benchmark.py` prints the same estimate on your hardware.
- The elemental image datasets (simulation and lab captures) are not included.

## Tested

Python 3.13.9, numpy 2.5, opencv-python 5.0, scikit-image 0.26, pytest 9.1 on Windows 11: `pytest` passes 8 tests. The E(P) minimum and the set of minimizing origins match the original research script in 10,020 random cases.

## License

MIT © Hyeongju Lee
