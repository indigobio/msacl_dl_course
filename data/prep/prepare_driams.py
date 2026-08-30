#!/usr/bin/env python3
"""Build Colab-sized course slices from DRIAMS-A (one-time, instructor machine).

DRIAMS: MALDI-TOF spectra + antimicrobial-resistance labels, CC0.
Use the frozen Nov 2021 Zenodo release (published loader code expects it;
Dryad's Aug 2025 update renamed files): https://zenodo.org/records/5640517

The DRIAMS-A archive is 86 GB, but we only need `binned_6000/` (preprocessed
spectra as 6000-dim vectors) and `id/` (yearly CSVs with species + S/I/R labels).
Stream-extract just those to avoid ~200 GB of scratch disk:

    curl -L 'https://zenodo.org/records/5640517/files/DRIAMS_A.tar.gz?download=1' \
      | tar -xz -C /path/to/driams --wildcards '*/binned_6000/*' '*/id/*'

Then:

    python data/prep/prepare_driams.py --root /path/to/driams/DRIAMS-A \
        --task saureus_oxacillin --task ecoli_ceftriaxone --out data/slices

Each task produces `driams_a_<task>.npz` containing:
    X     (n, 6000) float16 binned spectra
    y     (n,)      uint8   0 = susceptible, 1 = resistant
    code  (n,)      str     DRIAMS sample codes (provenance)
    year  (n,)      str
Intermediate "I" labels are dropped by default (--merge-i-to-r to merge into R,
matching one common benchmark convention — state the choice in the lab text).

Rehost the .npz files as GitHub release assets for fast Colab download (CC0
permits redistribution). Expect ~15–45 MB per task at float16.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# task name -> (species string in id CSVs, antibiotic column name)
TASKS = {
    "saureus_oxacillin": ("Staphylococcus aureus", "Oxacillin"),
    "ecoli_ceftriaxone": ("Escherichia coli", "Ceftriaxone"),
    "kpneumoniae_ceftriaxone": ("Klebsiella pneumoniae", "Ceftriaxone"),
}


def iter_id_rows(root: Path):
    """Yield (year, row_dict) from every yearly id CSV under root/id/."""
    csv.field_size_limit(10_000_000)
    for csv_path in sorted(root.glob("id/*/*_clean.csv")):
        year = csv_path.parent.name
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                yield year, row
    # some releases keep plain <year>.csv without the _clean suffix
    for csv_path in sorted(root.glob("id/*/[0-9]*.csv")):
        if csv_path.name.endswith("_clean.csv"):
            continue
        year = csv_path.parent.name
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                yield year, row


def load_spectrum(root: Path, year: str, code: str):
    """Read one binned_6000 spectrum txt -> (6000,) float32, or None if absent."""
    path = root / "binned_6000" / year / f"{code}.txt"
    if not path.exists():
        return None
    arr = np.loadtxt(path, skiprows=1)
    intensities = arr[:, -1] if arr.ndim == 2 else arr
    if intensities.shape[0] != 6000:
        return None
    return intensities.astype(np.float32)


def build_task(root: Path, species: str, antibiotic: str, merge_i_to_r: bool):
    X, y, codes, years = [], [], [], []
    seen = set()
    n_dropped_i = n_missing = 0
    for year, row in iter_id_rows(root):
        if row.get("species", "").strip() != species:
            continue
        label = (row.get(antibiotic) or "").strip().upper()
        if label == "I" and not merge_i_to_r:
            n_dropped_i += 1
            continue
        if label not in ("S", "R", "I"):
            continue
        code = row.get("code", "").strip()
        if not code or (year, code) in seen:
            continue
        spec = load_spectrum(root, year, code)
        if spec is None:
            n_missing += 1
            continue
        seen.add((year, code))
        X.append(spec)
        y.append(0 if label == "S" else 1)
        codes.append(code)
        years.append(year)
    if not X:
        raise SystemExit(
            f"No spectra found for {species} / {antibiotic} under {root} — "
            "check --root points at DRIAMS-A and that id/ and binned_6000/ exist."
        )
    print(
        f"  {species} / {antibiotic}: {len(X)} spectra, "
        f"{sum(y)} resistant ({100 * sum(y) / len(y):.1f}%), "
        f"dropped I: {n_dropped_i}, missing spectrum files: {n_missing}"
    )
    return (
        np.stack(X).astype(np.float16),
        np.array(y, dtype=np.uint8),
        np.array(codes),
        np.array(years),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path, help="path to DRIAMS-A dir")
    ap.add_argument("--task", action="append", choices=sorted(TASKS), required=True)
    ap.add_argument("--out", type=Path, default=Path("data/slices"))
    ap.add_argument("--merge-i-to-r", action="store_true",
                    help="merge intermediate (I) labels into resistant instead of dropping")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    site = args.root.resolve().name.lower().replace("-", "_")   # e.g. DRIAMS-C -> driams_c
    for task in args.task:
        species, antibiotic = TASKS[task]
        print(f"Building task '{task}'...")
        X, y, codes, years = build_task(args.root, species, antibiotic, args.merge_i_to_r)
        dest = args.out / f"{site}_{task}.npz"
        np.savez_compressed(dest, X=X, y=y, code=codes, year=years)
        print(f"  wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
