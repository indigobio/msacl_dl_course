# MSACL DS301 Deep Learning — Lab Pack

Everything you need to run the course labs, on your own laptop **or** on Google Colab.

You do **not** need to be a programmer. Pick the path that matches how you want to work
and follow the numbered steps.

```
student_pack/
├─ labs/                     the lab notebooks (open these)
│   └─ lab01_training_loop.ipynb
├─ pyproject.toml            the single source of truth for dependencies
├─ uv.lock                   exact pinned versions for local use (do not edit)
├─ .python-version           Python version for local use
└─ README.md                 this file
```

---

## Path A — Google Colab (easiest, nothing to install)

Colab is a free notebook that runs in your web browser, with a GPU included.

1. **Open the notebook in Colab.** Easiest: click the Colab link your instructor gave
   you — it opens the lab directly. Otherwise, go to
   https://colab.research.google.com, sign in with a Google account, choose
   **File → Upload notebook**, and pick a file from the `labs/` folder (start with
   `lab01_training_loop.ipynb`).
2. **Turn on the GPU** (once per notebook): **Runtime → Change runtime type →
   Hardware accelerator → T4 GPU → Save**. Lab 1 runs fine without it, but later labs
   need it.
3. **Run the setup cell first.** If the notebook already has a cell labeled
   "Colab setup", run it (Shift + Enter). Otherwise add a new code cell at the very top
   (**+ Code**), paste the **setup cell** below, and run it. It takes about a minute.
4. **Run the rest** top to bottom (**Runtime → Run all**), filling in the blanks marked
   **YOUR TURN ✏️** as you go.

### The Colab setup cell (copy–paste, run once at the top)

```python
# === MSACL DS301 · Colab setup — run me first ===
# Installs the course libraries into Colab WITHOUT reinstalling PyTorch:
# Colab already ships a GPU build of torch that matches its hardware, so we keep it.
# Safe to run more than once.
import os, urllib.request

# pyproject.toml is the single source of truth for the library list.
# Auto-download it if it isn't sitting next to the notebook.
# (Your instructor will confirm this URL points at the course repo.)
if not os.path.exists("pyproject.toml"):
    RAW_URL = "https://raw.githubusercontent.com/<ORG>/<REPO>/main/student_pack/pyproject.toml"
    urllib.request.urlretrieve(RAW_URL, "pyproject.toml")

get_ipython().system("pip install -q uv")
get_ipython().system("uv pip install --system -q -r pyproject.toml")

import torch
print("Setup complete. torch", torch.__version__, "| GPU available:", torch.cuda.is_available())
```

Why this works: `uv` is a very fast installer. We point it at Colab's own Python
(`--system`) and give it `pyproject.toml`, which lists only the base scientific and
deep-learning libraries. **PyTorch is deliberately left out of that list** (it lives in
a local-only group), so Colab's GPU PyTorch is never reinstalled — that would risk a
GPU/CUDA mismatch and waste minutes. The install is *additive*: it only adds what's
missing and never removes Colab's existing packages.

> Tip: if you uploaded the whole `student_pack` folder to Google Drive and opened the
> notebook from there, `pyproject.toml` may already be present and the download step is
> skipped automatically.

---

## Path B — Your own computer (with uv)

Use this if you'd rather run the notebooks locally. It installs a CPU-only build of
PyTorch, so it works on any Mac/Windows/Linux laptop with no GPU required.

1. **Install uv** (one time). Open a terminal and run:
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. In the terminal, go into this folder:
   `cd path/to/student_pack`
3. Build the environment (downloads all libraries, a few minutes the first time):
   `uv sync`
4. Launch Jupyter and open a lab:
   `uv run jupyter lab`
   Your browser opens; navigate into `labs/` and open a notebook.

That's it. `uv sync` reads `uv.lock`, so everyone gets the exact same versions.

### Optional: use this environment as a kernel in your own Jupyter/VS Code

```
uv run python -m ipykernel install --user --name msacl-ds301 --display-name "MSACL DS301"
```
Then pick the **MSACL DS301** kernel inside Jupyter or VS Code.

---

## For the instructor — maintaining the pack

- `pyproject.toml` (+ `uv.lock`) is the single source of truth. The env files here are
  committed; the lab notebooks are generated, so they are copied in rather than
  hand-edited:
  ```
  python tools/strip_solutions.py --all      # regenerate labs/student/*.ipynb
  python tools/build_student_pack.py          # copy them into student_pack/labs/
  ```
  Or do both in one step with `python tools/refresh_student_pack.py`.
- If you change dependencies, edit `pyproject.toml` and re-lock (that is all — there is
  no separate Colab requirements file to regenerate):
  ```
  cd student_pack
  uv lock
  ```
  Torch stays in the `local` group and Jupyter in the `dev` group, both listed under
  `[tool.uv] default-groups`, so a bare `uv sync` installs everything locally while the
  Colab command (`uv pip install -r pyproject.toml`, which reads only
  `[project.dependencies]`) skips both.
- Before sharing, set the real repo path in the Colab setup cell's `RAW_URL`
  (or tell students to upload `pyproject.toml` alongside the notebook).
- **Full publishing + live-session steps are in `course_plan/colab_setup_guide.md`.**
