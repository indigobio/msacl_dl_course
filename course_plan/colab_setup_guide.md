# Colab setup — instructor guide

How the labs run on Google Colab, for **yourself** (testing, the live session) and
for **students**. The student-facing quick steps are in `student_pack/README.md`
(Path A) and on the Lab 1 setup sheet (`labs/handouts/lab01_colab_setup.*`); this
file adds the publishing and maintenance side.

Three facts shape the design (DECISIONS.md, 2026-09-23):

- **A Colab link opens one notebook on a fresh, empty machine.** There is no way to
  hand students a "forked folder", so instead every lab's first cell rebuilds what
  the folder would have given them: the packages and the data.
- **Data size grows lab to lab**, so data does not travel with the notebook or the
  repo. It lives in a Hugging Face dataset repo and is fetched on demand, per lab,
  with a SHA-256 check, and optionally cached in the student's Google Drive.
- **Colab already ships a GPU build of PyTorch.** The setup never reinstalls torch.

---

## How the setup cell works

Every lab starts with the same cell. On Colab it installs `uv`, downloads
`student_pack/msacl.py` from the repo's `main` branch, and calls
`msacl.setup("labNN")`, which:

1. **installs `requirements-colab.txt`** — the 7 top-level lab packages pinned to the
   versions in `student_pack/uv.lock` (markers kept, so the right pin applies to
   Colab's Python), via `uv pip install --system`. Skipped when already satisfied.
   If the install upgrades a package the kernel had already imported (Colab
   preloads pandas), it stops and prints **Runtime → Restart session, then run this
   setup cell again** — the second run finds everything installed and moves on;
2. **downloads the data tagged with that lab** in `student_pack/datasets.json`,
   verifying each file's SHA-256; a mismatch deletes the file and fails loudly.
   Interrupted downloads resume. A dataset marked `optional` (one track of Lab 5)
   warns instead of failing;
3. **caches** in `MyDrive/msacl_ds301_data/` if Drive is mounted
   (`msacl.setup("labNN", mount_drive=True)`), else `/content/msacl_data/` for the
   session. Returns `DATA = {file name: local path}`.

Run locally, the same cell finds `msacl.py` in the course folder, installs nothing
(the `uv sync` environment is already there) and uses `data/slices/` directly.

---

## Part 1 — one-time publishing

1. **Hugging Face dataset repo.** Create `jaztsong88/msacl-ds301` (type: dataset,
   public) on huggingface.co. Then, with your own login (`huggingface-cli login`;
   never commit a token):
   ```
   python tools/build_colab_bootstrap.py
   huggingface-cli upload jaztsong88/msacl-ds301 build/hf_upload . --repo-type dataset
   ```
   Only files whose entry in `DATASETS` says `publish=True` are staged — that flag
   records the licence decision. Currently staged: the three DRIAMS-C slices (CC0).
   Held back: `MTBLS90.xlsx` (fetched from its CIMCB source, which has no LICENSE
   file) and `peakonly_roi_qc.npz` (annotation licence unverified; Lab 5 Track B).
2. **Merge to `main`.** The setup cell fetches `msacl.py`, `datasets.json` and
   `requirements-colab.txt` from `raw.githubusercontent.com/.../main/student_pack/`,
   so nothing works on Colab until they are on `main`.
3. **Colab links** for each lab:
   ```
   python tools/colab_links.py --org indigobio --repo msacl_dl_course --branch main
   ```

---

## Part 2 — rehearse (before every course)

Open each lab from `student_pack/labs/` via its link, **Runtime → Change runtime
type → T4 GPU**, run the setup cell, then **Runtime → Run all**. Check:

- the setup ends `✓ labNN is ready` (after at most one restart prompt);
- no import errors where Colab's preinstalled packages meet our pins (pandas 3,
  numpy 2.5). Colab's own `google-colab` package may pin an older pandas, so pip can
  print a dependency-conflict warning. That warning is expected; a real failure is
  not;
- each notebook finishes in under ~10 minutes and every `assert` passes.

To test that everything still *computes*, open `labs/solutions/` instead.

---

## Part 3 — keeping it in sync

- After editing a lab (author in `labs/solutions/`):
  `python tools/refresh_student_pack.py`, commit, push.
- After changing dependencies: edit `student_pack/pyproject.toml`, `uv lock`, then
  `python tools/build_colab_bootstrap.py` to regenerate the Colab pins.
- **Adding data to a lab:** prepare the slice into `data/slices/` (script in
  `data/prep/`, entry in `data/README.md`), add it to `DATASETS` in
  `tools/build_colab_bootstrap.py` with its `labs`, licence and `publish` flag,
  rerun the script, upload, commit `datasets.json`. In the notebook, read it as
  `DATA["file_name"]`. Large files are fine — Hugging Face serves them with
  resumable range requests, and `mount_drive=True` makes them download once.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Setup cell 404s fetching `msacl.py` | The branch isn't merged to `main`, or the repo isn't public. |
| `Could not get driams_….npz` (404) | The Hugging Face upload hasn't been done (Part 1, step 1). |
| "SHA-256 fingerprint does not match" | The uploaded file differs from `data/slices/`. Rerun `build_colab_bootstrap.py` and re-upload; commit the new `datasets.json`. |
| "Restart session, then run this setup cell again" | Expected once per session on Colab. Do exactly that. |
| `torch.cuda.is_available()` is `False` | **Runtime → Change runtime type → T4 GPU**, then rerun the setup cell. |
| Want to reset everything | **Runtime → Disconnect and delete runtime**, reopen, select GPU, run setup. |
