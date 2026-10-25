---
name: handwritten
description: Turn images of text into pages written in the user's own handwriting, and train or inspect the user's handwriting profile. Use when the user runs /handwritten, asks to train or update their handwriting, asks what their handwriting profile covers, or wants text written out in their handwriting.
argument-hint: "train <photos or folder> | template maths | identity train <photo> | status | <images> [--continuous] [--identity] [--pdf] [--clean] [--ink black] [--seed N]"
---

# /handwritten

Arguments: $ARGUMENTS

All paths below are relative to this skill's base directory. Before running any script, run
`bash scripts/setup_env.sh` (fast when the environment already exists), then run scripts with
`~/.handwritten/.venv/bin/python scripts/<name>.py`. Profiles live in `~/.handwritten/profiles/<name>/`
(default name `default`; pass `--profile NAME` through when the user names one).

## Choose the mode

| The user wants | Mode |
|---|---|
| `template maths`, "a template sheet", "a sheet to fill in" | **Template**: make the printable sheets, `reference/template.md` step 1 |
| `train …`, "learn my handwriting", or gives photos of their own writing to learn from | **Train**: run `scripts/read_template.py` on the photos first (`reference/template.md` step 2); photos it lists as not template pages follow `reference/train.md` exactly |
| `identity train <photo>`, or a photo of their serial number / name box to put on the first page | **Identity**: follow `reference/identity.md` exactly |
| `status`, "what does my profile cover" | **Status**: run `scripts/status.py`, show the report, and look at `preview.png` so you can describe it |
| `gaps` | Not built yet (Phase 4). Say which characters are missing from `status.py`, and offer the maths template (`template maths`) for symbols |
| "set line spacing to 9.6 mm", "use the measured spacing again" | **Settings**: `scripts/settings.py line_pitch_mm=9.6` (or `line_pitch_mm=` to go back to the measured value). Saved with the profile, kept when retraining. `--line-pitch MM` on `render.py` changes one run only |
| Anything else: images of text to write out | **Generate** (below) |

## Generate

Follow `reference/write.md` exactly. In short: check the profile with `scripts/status.py --require`
(no profile → tell the user to run `/handwritten train` and **stop**: never produce output without a
profile), transcribe each input image, render with `scripts/render.py`, check every page, deliver.

Options: per-image by default (1 input → its own page or pages), `--continuous` (all inputs flow as
one document), `--identity` (the trained serial number and name block in the top-left corner of the
first page of the output only), `--pdf` (also one A4 PDF of every page), `--clean` (flat scan instead
of the phone-photo look), `--ink blue|black|#hex`, `--seed N`.

## Rules

- Never invent text. Transcribe exactly what is written, and use `#` for any word you are not sure of.
- Training needs image **files** on disk. Images pasted into the chat can't be processed by the
  scripts unless the chat shows where the image was saved (`[Image: source: <path>]`); then use that
  path. Otherwise ask for a file path or folder instead.
- The profile reproduces the user's handwriting. Keep it local: never upload it, copy it elsewhere, or
  use it for signatures.
