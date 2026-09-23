"""MSACL DS301 Deep Learning — one-cell setup for every lab.

Every lab notebook starts with the same setup cell. On Colab it installs uv and
fetches this file from the course repository; run locally it finds this file in
the course folder. Either way it ends with:

    import msacl
    DATA = msacl.setup("lab02")      # {file name: where it is on this machine}

`setup(lab)` does three things, so nobody downloads or copies anything by hand:

1. **Installs the course environment.** The packages the labs use, pinned to
   the versions in the course's uv lock (see `requirements-colab.txt`). On
   Colab they go into Colab's own Python with `uv pip install --system`, and
   Colab's preinstalled GPU build of torch is never touched.
2. **Downloads the data that lab needs**, listed in `datasets.json`, and checks
   every file's SHA-256 fingerprint, so a truncated or tampered download fails
   loudly instead of producing strange results an hour later.
3. **Caches it.** If you have mounted Google Drive, the data is kept there, so
   a large dataset downloads once per course rather than once per session.
   Otherwise it lives on the Colab machine until the session ends.

Standard library only: it has to run before anything else is installed.
Run locally from a repo checkout, it uses `data/slices/` directly and installs
nothing (your `uv sync` environment is already there).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW_BASE = ("https://raw.githubusercontent.com/indigobio/msacl_dl_course/"
            "main/student_pack/")
MANIFEST = "datasets.json"
REQUIREMENTS = "requirements-colab.txt"
DRIVE_ROOT = Path("/content/drive/MyDrive")
_UA = {"User-Agent": "msacl-ds301-labs/1.0"}
_MODULE = {"scikit-learn": "sklearn"}      # package name → import name, where they differ


# ----------------------------------------------------------------- where ----
def _in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").is_dir()


def _here() -> Path:
    return Path(__file__).resolve().parent


def _repo_slices() -> Path | None:
    """A repo checkout's data/slices, when running locally for development."""
    for base in (_here(), *_here().parents):
        cand = base / "data" / "slices"
        if cand.is_dir():
            return cand
    return None


def cache_dir() -> Path:
    """Drive if it is mounted (persists), else this machine (per session)."""
    if DRIVE_ROOT.is_dir():
        d = DRIVE_ROOT / "msacl_ds301_data"
    elif _in_colab():
        d = Path("/content/msacl_data")
    else:
        d = Path.home() / ".cache" / "msacl_ds301"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------- manifest ----
def _fetch_text(name: str) -> str:
    local = _here() / name
    if local.is_file():
        return local.read_text(encoding="utf-8")
    req = urllib.request.Request(RAW_BASE + name, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def manifest() -> dict:
    return json.loads(_fetch_text(MANIFEST))


# ------------------------------------------------------------- download ----
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _download(url: str, dest: Path, size: int | None, quiet: bool = False) -> None:
    """Stream to `dest.part`, resuming if a previous attempt was cut off."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(_UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:          # server ignored the Range header
            have = 0
        mode = "ab" if have else "wb"
        total = size or (have + int(r.headers.get("Content-Length") or 0)) or None
        got, last = have, 0.0
        with open(part, mode) as out:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                out.write(block)
                got += len(block)
                if not quiet and time.time() - last > 0.5:
                    pct = f"{100 * got / total:5.1f}%" if total else ""
                    print(f"\r    {pct} {_human(got)}", end="", flush=True)
                    last = time.time()
    if not quiet:
        print(f"\r    done  {_human(got)}" + " " * 12)
    part.replace(dest)


def data(name: str, *, quiet: bool = False) -> Path:
    """Return a local path to dataset `name`, downloading and verifying it once."""
    entry = manifest()["datasets"].get(name)
    if entry is None:
        raise KeyError(f"{name!r} is not in {MANIFEST}")
    want = entry.get("sha256")

    # Local development: use the repo's own slice if it is the right file.
    repo = _repo_slices()
    if repo and (repo / name).is_file() and (not want or _sha256(repo / name) == want):
        if not quiet:
            print(f"  ✓ {name}  (repo copy)")
        return repo / name

    dest = cache_dir() / name
    if dest.is_file() and (not want or _sha256(dest) == want):
        if not quiet:
            print(f"  ✓ {name}  (already cached)")
        return dest

    errors = []
    for url in entry["urls"]:
        if not quiet:
            size = entry.get("bytes")
            print(f"  ↓ {name}" + (f"  ({_human(size)})" if size else ""))
        try:
            _download(url, dest, entry.get("bytes"), quiet=quiet)
        except (urllib.error.URLError, OSError) as exc:
            errors.append(f"{url}\n      {exc}")
            continue
        if want and _sha256(dest) != want:
            dest.unlink(missing_ok=True)
            errors.append(f"{url}\n      downloaded, but the SHA-256 fingerprint "
                          "does not match — the file is incomplete or has changed")
            continue
        return dest
    raise RuntimeError(f"Could not get {name}. Tried:\n    " + "\n    ".join(errors)
                       + "\n  Check your internet connection and run the cell again.")


# ---------------------------------------------------------- environment ----
def _pins(text: str) -> dict[str, str] | None:
    """{package: pinned version} that apply to THIS Python, or None if unknown."""
    try:
        from packaging.markers import Marker     # noqa: PLC0415 - preinstalled on Colab
    except ImportError:
        return None
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        spec, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate():
            continue
        name, _, version = spec.partition("==")
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def _installed(name: str) -> str | None:
    from importlib import metadata               # noqa: PLC0415
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def install_env(force: bool = False) -> None:
    """Install the pinned course packages into this Python (Colab by default)."""
    if not (_in_colab() or force):
        print("  · local run — using your uv environment, installing nothing")
        return
    text = _fetch_text(REQUIREMENTS)
    pins = _pins(text)
    if pins is not None:
        stale = {n: v for n, v in pins.items() if _installed(n) != v}
        if not stale:
            print("  ✓ course packages already installed")
            return
    req = cache_dir() / REQUIREMENTS
    req.write_text(text, encoding="utf-8")
    print("  ↓ installing the course packages (about a minute the first time)…")
    cmd = ["uv", "pip", "install", "--system", "-q", "-r", str(req)]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # uv missing or refused: fall back to pip so the lab still runs.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                       check=True)
    # A package this kernel had ALREADY imported (Colab preloads some, e.g. pandas)
    # keeps running its old version until the session restarts. Mixing the two
    # gives baffling errors later, so stop here and say exactly what to do.
    names = stale if pins is not None else {
        l.split("==")[0].strip().lower() for l in text.splitlines() if "==" in l}
    changed = sorted(n for n in names if _MODULE.get(n, n.replace("-", "_")) in sys.modules)
    if changed:
        raise SystemExit(
            "\n  ✓ packages installed. One more step, needed once per session:\n"
            "    Runtime → Restart session, then run this setup cell again.\n"
            f"    (updated while already in use: {', '.join(changed)})")
    print("  ✓ packages ready")


# ---------------------------------------------------------------- setup ----
def setup(lab: str, *, install: bool | None = None, mount_drive: bool = False) -> dict:
    """Prepare everything lab `lab` needs. Returns {dataset name: local path}.

    mount_drive=True asks Google for permission to cache data in your Drive, so
    it survives the end of the session. Leave it off for small labs.
    """
    print(f"MSACL DS301 · setting up {lab}")
    if mount_drive and _in_colab() and not DRIVE_ROOT.is_dir():
        from google.colab import drive          # noqa: PLC0415 - Colab only
        drive.mount("/content/drive")
    if install is None:
        install = _in_colab()
    if install:
        install_env(force=True)

    paths = {}
    for name, entry in manifest()["datasets"].items():
        if lab not in entry.get("labs", []):
            continue
        try:
            paths[name] = data(name)
        except RuntimeError:
            if not entry.get("optional"):
                raise
            # An optional dataset (one track of a multi-track lab) must not stop
            # the rest of the lab from running.
            print(f"  ! {name} is not available yet — the parts of {lab} that "
                  "need it will say so; everything else works.")
    where = "in your Google Drive" if DRIVE_ROOT.is_dir() else (
        "for this Colab session" if _in_colab() else "on your machine")
    print(f"✓ {lab} is ready — {len(paths)} dataset(s), cached {where}.")
    return paths


if __name__ == "__main__":
    setup(sys.argv[1] if len(sys.argv) > 1 else "lab01")
