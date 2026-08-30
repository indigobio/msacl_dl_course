#!/usr/bin/env python3
"""Build the peak-QC course slice from PeakOnly's annotated ROI data.

Source: Melnikov et al., Anal. Chem. 2020 (peakonly, github.com/arseha/peakonly).
The annotated data (5,365 ROI JSONs: serum HILIC/RPLC + MetaboLights studies)
lives on Yandex Disk; the app's own headless proxy URL works without login:
    https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/f6BiwqWYF4UVnA
License caveat: the code is MIT but the annotation zip carries no explicit
license — cite the paper in all course materials.

Each ROI JSON holds a variable-length 1D `intensity` array (median 47 points),
an ROI `label` (0 noise / 1 peak), per-peak quality sub-labels, and peak
`borders`. This script resamples every ROI to a fixed 256 points (as in the
paper), max-normalizes, and writes one compressed npz:

    X        (n, 256) float32  resampled, max-normalized intensity windows
    y        (n,)     uint8    0 = noise, 1 = contains real peak(s)
    quality  (n,)     uint8    first peak's quality sub-label where annotated
                               (1 good, 2 low-intensity, 3 lousy, 4 noisy;
                                0 = unlabeled or noise ROI) — stretch-goal labels
    source   (n,)     str      dataset folder the ROI came from

Usage:
    python data/prep/prepare_peakonly.py                  # downloads the zip
    python data/prep/prepare_peakonly.py --zip annotation.zip --out data/slices
"""

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/f6BiwqWYF4UVnA"
N_POINTS = 256


def resample(intensity, n_points=N_POINTS):
    arr = np.asarray(intensity, dtype=np.float32)
    if arr.size < 2 or not np.isfinite(arr).all():
        return None
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, n_points)
    out = np.interp(x_new, x_old, arr)
    peak = out.max()
    if peak <= 0:
        return None
    return (out / peak).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", type=Path, default=None,
                    help="existing annotation.zip (skips download)")
    ap.add_argument("--out", type=Path, default=Path("data/slices"))
    args = ap.parse_args()

    zip_path = args.zip
    if zip_path is None:
        zip_path = args.out / "annotation.zip"
        args.out.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            print(f"Downloading annotation data -> {zip_path}")
            urllib.request.urlretrieve(URL, zip_path)

    X, y, quality, source = [], [], [], []
    n_skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist()
                 if n.endswith(".json") and "__MACOSX" not in n]
        for name in sorted(names):
            try:
                roi = json.loads(zf.read(name))
            except (json.JSONDecodeError, UnicodeDecodeError):
                n_skipped += 1
                continue
            sig = resample(roi.get("intensity", []))
            label = roi.get("label")
            if sig is None or label not in (0, 1):
                n_skipped += 1
                continue
            peak_labels = roi.get("peaks' labels") or []
            X.append(sig)
            y.append(label)
            quality.append(int(peak_labels[0]) if peak_labels else 0)
            source.append(Path(name).parent.name)

    if not X:
        raise SystemExit("No ROIs parsed — is the zip the peakonly annotation archive?")

    X = np.stack(X)
    y = np.array(y, dtype=np.uint8)
    quality = np.array(quality, dtype=np.uint8)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "peakonly_roi_qc.npz"
    np.savez_compressed(dest, X=X, y=y, quality=quality, source=np.array(source))
    print(f"{len(y)} ROIs ({int(y.sum())} peak / {int((y == 0).sum())} noise), "
          f"skipped {n_skipped}")
    print("quality sub-labels among peak ROIs:",
          {int(k): int(v) for k, v in zip(*np.unique(quality[y == 1], return_counts=True))})
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
