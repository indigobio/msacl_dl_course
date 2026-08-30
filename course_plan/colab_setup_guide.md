# Colab setup — instructor guide

How to get the labs running in Google Colab, both for **yourself** (testing and the
live session) and for **students**. The student-facing version of the quick steps
lives in `student_pack/README.md`; this file adds the publishing and teaching bits
that students don't need to see.

Two facts that shape everything below:

- **Colab already ships a GPU build of PyTorch** that matches its hardware. We never
  reinstall torch on Colab — the setup cell installs only the *other* libraries by
  reading `student_pack/pyproject.toml`, in which torch lives in a local-only
  dependency-group and so is skipped. This keeps GPU support and saves minutes.
- **Students open one notebook at a time in Colab.** The "one shared folder"
  (`student_pack/`) is for local `uv` use and for keeping everything together in the
  repo; on Colab, the shared piece is the setup cell + `pyproject.toml` (the single
  source of truth for dependencies — there is no separate requirements file).

---

## Part 1 — one-time setup (do this once)

1. **Publish the repo to a public GitHub repo.** Colab can open any notebook from a
   public GitHub URL, and the setup cell fetches `pyproject.toml` over HTTPS —
   both need the repo to be public (or the file shared some other way; see §4).
   ```
   cd /Users/lixingsong/Teaching/MSACL_DL_Course
   git add -A && git commit -m "Course labs + student pack"
   git branch -M main
   git remote add origin https://github.com/<ORG>/<REPO>.git
   git push -u origin main
   ```
2. **Wire up `RAW_URL`.** In `student_pack/README.md` (the Colab setup cell) replace
   `<ORG>/<REPO>` in
   `https://raw.githubusercontent.com/<ORG>/<REPO>/main/student_pack/pyproject.toml`
   with your real path. The lab notebooks themselves contain this same cell — if you
   want it baked into the notebooks, put the URL in the solution notebook, re-run
   `python tools/refresh_student_pack.py`, and re-commit. (Leaving the placeholder is
   fine too — students can upload `pyproject.toml` next to the notebook instead.)
3. **Generate the "Open in Colab" links** you'll hand out:
   ```
   python tools/colab_links.py --org <ORG> --repo <REPO> --branch main
   ```
   This prints one `colab.research.google.com/github/...` link per lab notebook.

---

## Part 2 — set up Colab for yourself (test a lab)

1. Open a lab: click one of the links from `tools/colab_links.py`, **or** in Colab
   choose **File → Open notebook → GitHub**, paste your repo, and pick a notebook under
   `student_pack/labs/`. (Uploading the `.ipynb` by hand also works.)
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the **Colab setup** cell first (it prints the torch version and whether the GPU
   is visible), then **Runtime → Run all**.
4. Confirm the notebook finishes in well under 10 minutes and the `assert` self-checks
   pass. For Lab 1 you should see the loss fall and test accuracy ~0.74.

> Tip: to test the *student* experience, open the notebook from
> `student_pack/labs/` (the stripped, blanks version). To test that everything still
> computes, run `labs/solutions/` instead.

---

## Part 3 — set up Colab for students

Give students two things: the **Colab link** for each lab and the one-paragraph
instructions (already in `student_pack/README.md`, Path A). Their flow is:

1. Click the lab's Colab link (opens in Colab; sign in with a Google account).
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the **Colab setup** cell first (about a minute), then run the rest top to bottom,
   filling in the **YOUR TURN ✏️** blanks.

Because the setup cell installs the libraries listed in `pyproject.toml` on top of
Colab's base image, every student gets the same core stack, lab to lab. (The exact
pinned versions in `uv.lock` govern the *local* `uv` environment; on Colab, `uv pip
install` resolves compatible versions against Colab's preinstalled packages so it never
disturbs them — see Part 4.)

**No public repo?** Alternatives that skip GitHub:
- Zip `student_pack/` (exclude `.venv/`) and share it; students **File → Upload folder**
  to Colab or drop it in Google Drive, then open a notebook from there. When the notebook
  and `pyproject.toml` sit side by side, the setup cell skips the download.
- Or share a Google Drive folder of the notebooks + `pyproject.toml` and have
  students `File → Open notebook → Google Drive`.

---

## Part 4 — keeping it in sync

- After editing any lab (author in `labs/solutions/`), regenerate everything with:
  ```
  python tools/refresh_student_pack.py          # strip solutions + rebuild student_pack
  python tools/refresh_student_pack.py --relock  # also after dependency changes
  ```
  then commit and push. Committed notebooks mean the Colab links always point at the
  latest version.
- **Dependencies live only in `pyproject.toml`** (+ `uv.lock`); there is no exported
  requirements file to keep in step. Torch sits in the `local` group and Jupyter in the
  `dev` group, both under `[tool.uv] default-groups`, so:
  - **Local:** a bare `cd student_pack && uv sync` installs base deps + CPU torch +
    Jupyter, pinned exactly to `uv.lock`.
  - **Colab:** `uv pip install --system -r pyproject.toml` reads only
    `[project.dependencies]`, so it skips torch and Jupyter entirely. It is *additive*
    (never prunes) and leaves Colab's preinstalled GPU torch untouched. (We deliberately
    do **not** use `uv sync` on Colab: Colab's kernel is the system interpreter, not a
    uv-managed venv, and `uv sync` would try to realign dozens of Colab's system
    packages to the lock — too disruptive. `uv pip install` only adds what's missing.)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Setup cell 404s on `RAW_URL` | Repo isn't public yet, or `<ORG>/<REPO>` is still a placeholder. Make it public, or have students upload `pyproject.toml` beside the notebook. |
| `torch.cuda.is_available()` is `False` | GPU runtime not selected — **Runtime → Change runtime type → T4 GPU**, then re-run. |
| A later lab is slow / OOM | Confirm the T4 GPU is on; restart the runtime (**Runtime → Restart**) and re-run the setup cell. |
| "No module named transformers/datasets" (Lab 3) | The setup cell didn't run, or ran before GPU selection reset the runtime. Re-run the setup cell. |
| Want to reset everything | **Runtime → Disconnect and delete runtime**, reopen, select GPU, run setup cell. |
