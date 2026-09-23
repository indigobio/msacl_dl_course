#!/usr/bin/env python3
"""Generate the files student_pack/msacl.py reads, and stage the data upload.

Three outputs, all derived — never hand-edit them:

  student_pack/requirements-colab.txt
      The top-level lab packages, pinned to the versions in student_pack/uv.lock,
      with the lock's environment markers kept so the right pin applies on
      whatever Python version Colab is running. torch is deliberately absent:
      Colab keeps its own preinstalled GPU build.

  student_pack/datasets.json
      The data manifest: for each file, where to fetch it, its size, its SHA-256
      fingerprint, its licence, and which labs need it. msacl.setup("labNN")
      downloads exactly the files tagged with that lab.

  build/hf_upload/
      The files cleared for public rehosting, ready to push to the course's
      Hugging Face dataset repo. A file is staged ONLY if its entry below says
      publish=True; that flag is where a licence decision is recorded, and this
      script refuses to stage anything that has not been cleared.

    python3 tools/build_colab_bootstrap.py
    # then, once, with your own Hugging Face login (never commit a token):
    huggingface-cli upload indigobio/msacl-ds301 build/hf_upload . --repo-type dataset
"""
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "student_pack"
SLICES = ROOT / "data" / "slices"
STAGE = ROOT / "build" / "hf_upload"

HF_REPO = "indigobio/msacl-ds301"
HF = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
CIMCB = ("https://raw.githubusercontent.com/CIMCB/MetabComparisonBinaryML/"
         "master/notebooks/data/MTBLS90.xlsx")

# The course's data, and the licence status that decides whether we may rehost
# it publicly. Change `publish` only after the licence question is settled.
DATASETS = {
    "MTBLS90.xlsx": dict(
        urls=[CIMCB], labs=["lab01"], publish=False,
        license="Open access at EMBL-EBI MetaboLights; the CIMCB tidy copy has no "
                "LICENSE file, so it is fetched from source and NOT rehosted",
        source="MetaboLights MTBLS90 via CIMCB/MetabComparisonBinaryML "
               "(Mendez, Reinke & Broadhurst 2019)"),
    "driams_c_saureus_oxacillin.npz": dict(
        urls=[HF + "driams_c_saureus_oxacillin.npz"], labs=["lab02", "lab05"],
        publish=True, license="CC0-1.0",
        source="DRIAMS-C (Weis et al. 2022), Zenodo record 5640517"),
    "driams_c_ecoli_ceftriaxone.npz": dict(
        urls=[HF + "driams_c_ecoli_ceftriaxone.npz"], labs=["lab05"],
        publish=True, license="CC0-1.0",
        source="DRIAMS-C (Weis et al. 2022), Zenodo record 5640517"),
    "driams_c_kpneumoniae_ceftriaxone.npz": dict(
        urls=[HF + "driams_c_kpneumoniae_ceftriaxone.npz"], labs=[],
        publish=True, license="CC0-1.0",
        source="DRIAMS-C (Weis et al. 2022), Zenodo record 5640517 — "
               "prepared but not currently used by any lab"),
    "peakonly_roi_qc.npz": dict(
        urls=[HF + "peakonly_roi_qc.npz"], labs=["lab05"], publish=False,
        optional=True,           # Track B only — must not block Tracks A and C
        license="UNVERIFIED — peakonly's code is MIT but its annotation zip carries "
                "no licence; do not rehost publicly until that is cleared",
        source="peakonly annotated ROIs (Melnikov et al., Anal Chem 2020)"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def top_level_names() -> set[str]:
    project = tomllib.loads((PACK / "pyproject.toml").read_text())["project"]
    return {re.split(r"[\s<>=!~;\[]", d, maxsplit=1)[0].lower().replace("_", "-")
            for d in project["dependencies"]}


def build_requirements() -> list[str]:
    out = subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-default-groups",
         "--no-hashes", "--no-annotate", "--format", "requirements-txt"],
        cwd=PACK, check=True, capture_output=True, text=True).stdout
    keep = top_level_names()
    pins = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name = line.split("==", 1)[0].strip().lower().replace("_", "-")
        if name in keep:
            pins.append(line)
    if "torch" in keep or any(p.lower().startswith("torch") for p in pins):
        raise SystemExit("torch must not reach the Colab requirements — "
                         "Colab keeps its own GPU build")
    missing = keep - {p.split("==")[0].strip().lower().replace("_", "-") for p in pins}
    if missing:
        raise SystemExit(f"no locked version found for: {sorted(missing)} — "
                         "run `uv lock` in student_pack/")
    return pins


def build_manifest() -> dict:
    datasets = {}
    for name, meta in DATASETS.items():
        src = SLICES / name
        if not src.is_file():
            raise SystemExit(f"missing data/slices/{name} — run its data/prep script")
        datasets[name] = {"urls": meta["urls"], "bytes": src.stat().st_size,
                          "sha256": sha256(src), "labs": meta["labs"],
                          "license": meta["license"], "source": meta["source"],
                          "publish": meta["publish"],
                          "optional": meta.get("optional", False)}
    return {"schema": 1, "generated_by": "tools/build_colab_bootstrap.py",
            "hf_repo": HF_REPO, "datasets": datasets}


def stage(manifest: dict) -> list[str]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    staged = []
    for name, entry in manifest["datasets"].items():
        if entry["publish"]:
            shutil.copy2(SLICES / name, STAGE / name)
            staged.append(name)
    lines = ["# MSACL DS301 Deep Learning — course data", "",
             "Prepared, Colab-sized slices used by the MSACL DS301 labs. Every file "
             "here is licensed for redistribution; see `datasets.json` in the course "
             "repository for sources and SHA-256 fingerprints.", "",
             "| file | licence | source |", "|---|---|---|"]
    for name in staged:
        e = manifest["datasets"][name]
        lines.append(f"| `{name}` | {e['license']} | {e['source']} |")
    (STAGE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return staged


def main():
    pins = build_requirements()
    (PACK / "requirements-colab.txt").write_text(
        "# Generated by tools/build_colab_bootstrap.py from student_pack/uv.lock.\n"
        "# Top-level lab packages only; torch is deliberately absent so Colab keeps\n"
        "# its own GPU build. Do not hand-edit.\n" + "\n".join(pins) + "\n",
        encoding="utf-8")
    print(f"wrote student_pack/requirements-colab.txt  ({len(pins)} pinned lines)")

    manifest = build_manifest()
    (PACK / "datasets.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"wrote student_pack/datasets.json  ({len(manifest['datasets'])} datasets)")

    staged = stage(manifest)
    held = [n for n, e in manifest["datasets"].items() if not e["publish"]]
    print(f"staged {len(staged)} file(s) for Hugging Face in build/hf_upload/")
    for n in held:
        print(f"  held back (licence not cleared for rehosting): {n}")


if __name__ == "__main__":
    main()
