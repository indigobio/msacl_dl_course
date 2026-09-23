#!/usr/bin/env python3
"""Build Lab 4's image slice: the FashionMNIST clothes and MNIST "impostor" digits.

Lab 4 used to call torchvision's downloaders at run time, which depend on
torchvision's mirrors staying up (MNIST's have gone down before). This script
does that download once, keeps exactly the images the lab uses, and writes one
compressed npz that we rehost with the other course data:

    fashion_train_x  (12000, 28, 28) uint8   first 12,000 FashionMNIST training images
    fashion_train_y  (12000,)        uint8   their clothing class, 0-9
    fashion_test_x   (2000, 28, 28)  uint8   first 2,000 FashionMNIST test images
    fashion_test_y   (2000,)         uint8
    mnist_test_x     (2000, 28, 28)  uint8   first 2,000 MNIST test digits (impostors)
    mnist_test_y     (2000,)         uint8

Raw 0-255 pixels, same order torchvision gives, so the lab's results are
unchanged. Sources and licences:
    FashionMNIST — Xiao, Rasul & Vollgraf 2017, Zalando Research, MIT licence
    MNIST        — LeCun, Cortes & Burges, CC BY-SA 3.0

Usage:
    python data/prep/prepare_fashion_mnist.py            # writes data/slices/
    python data/prep/prepare_fashion_mnist.py --cache torchvision_data --out data/slices
"""
import argparse
from pathlib import Path

import numpy as np
from torchvision import datasets

N_TRAIN, N_TEST, N_IMPOSTOR = 12000, 2000, 2000
ROOT = Path(__file__).resolve().parents[2]


def first(ds, n):
    return ds.data[:n].numpy().astype(np.uint8), ds.targets[:n].numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=str(ROOT / "torchvision_data"),
                    help="torchvision download cache (git-ignored)")
    ap.add_argument("--out", default=str(ROOT / "data" / "slices"))
    args = ap.parse_args()

    fx, fy = first(datasets.FashionMNIST(args.cache, train=True, download=True), N_TRAIN)
    tx, ty = first(datasets.FashionMNIST(args.cache, train=False, download=True), N_TEST)
    mx, my = first(datasets.MNIST(args.cache, train=False, download=True), N_IMPOSTOR)

    out = Path(args.out) / "fashion_mnist_vae.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fashion_train_x=fx, fashion_train_y=fy,
                        fashion_test_x=tx, fashion_test_y=ty,
                        mnist_test_x=mx, mnist_test_y=my)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB): "
          f"{len(fx)} train clothes, {len(tx)} test clothes, {len(mx)} impostor digits")


if __name__ == "__main__":
    main()
