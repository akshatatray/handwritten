# Troubleshooting

- [`/handwritten` isn't recognised](#handwritten-isnt-recognised)
- [Setup fails: no working Python](#setup-fails-no-working-python)
- [Setup fails while installing packages](#setup-fails-while-installing-packages)
- ["NO PROFILE" when generating](#no-profile-when-generating)
- [Training can't use a pasted image](#training-cant-use-a-pasted-image)
- [The page isn't found in a photo](#the-page-isnt-found-in-a-photo)
- [The writing is too big or too small](#the-writing-is-too-big-or-too-small)
- [Some letters look wrong every time](#some-letters-look-wrong-every-time)
- [It looks repetitive or font-like](#it-looks-repetitive-or-font-like)
- [Characters are missing or drawn another way](#characters-are-missing-or-drawn-another-way)
- [A template page is read wrongly](#a-template-page-is-read-wrongly)
- [Lines are too close together or too far apart](#lines-are-too-close-together-or-too-far-apart)
- [Text overflows onto extra pages](#text-overflows-onto-extra-pages)
- [Review sheets show boxes instead of labels (Linux)](#review-sheets-show-boxes-instead-of-labels-linux)

---

### `/handwritten` isn't recognised

- Check the skill is where Claude Code looks: `~/.claude/skills/handwritten/SKILL.md` (all
  projects) or `<project>/.claude/skills/handwritten/SKILL.md` (one project). The folder must
  contain `SKILL.md` directly, not inside another folder.
- Start a new Claude Code session after installing.

### Setup fails: no working Python

`setup_env.sh` needs Python 3.9 or newer. Check with `python3 --version`. On macOS, install the
Xcode command line tools (`xcode-select --install`) or Python from python.org or Homebrew; on
Linux, install `python3` and `python3-venv` from your package manager. Then run the command again.

### Setup fails while installing packages

The first run downloads NumPy, SciPy, OpenCV, scikit-image, Pillow and pillow-heif from PyPI, so it
needs internet access. If it fails part-way, delete `~/.handwritten/.venv` and run any
`/handwritten` command again to start fresh. Your profiles are not affected.

### "NO PROFILE" when generating

There's no trained profile with that name. Run `/handwritten train <photos>` first. If training
was started but not finished, `status` says so; run `train` again to finish it. If you trained
with `--profile NAME`, pass the same name when generating.

### Training can't use a pasted image

The training scripts need the photo as a file. Give a file path or folder instead (dragging the
file into the terminal pastes its path). Pasted images are fine for generating.

### The page isn't found in a photo

Messages like `page edges NOT found`, `page found (unknown)`, or a large deskew mean the
page's outline wasn't detected (often the desk was picked up instead). Claude usually fixes this by
finding the corners itself and re-running with `--corners`. To avoid it: photograph on a
contrasting background with all four edges in view and no other paper in the frame.

### The writing is too big or too small

The scale comes from the paper's edges. If the paper isn't A4 or US Letter (or the edges weren't
found), the scale is off. Check the x-height in `/handwritten status`: typical handwriting is
2.5–4 mm. Tell Claude your real paper size (*"these are 176 × 250 mm notebook pages"*) and it
re-processes those photos.

### Some letters look wrong every time

A bad variant made it into the profile (a wrong letter, two letters cut as one, a stray mark).
Open the `specimen_NN.png` sheets in your profile folder and tell Claude the numbers to reject, or
describe the problem (*"one of the `g`s has a piece of the next letter"*) and let Claude find it.
Rejections are kept across rebuilds.

### It looks repetitive or font-like

The profile has too few variants of some characters. `status` shows variant counts; characters
with only one or two variants repeat visibly. Train on more pages, especially ones that use the
thin characters. More pages also give more of your real joins.

### Characters are missing or drawn another way

The render output and `report.json` list characters drawn with a stand-in and characters not
drawn at all. Train on pages containing them, or use the
[maths template](training.md#the-maths-template) for maths symbols, scripts, roots, fractions and
matrix brackets.

### A template page is read wrongly

Check the page's `cells.png` (Claude points to it). If the whole page looks shifted or garbled,
the photo was blurry, curved or partly out of frame. Take a sharper, flatter photo with all four
corner squares visible, and ask Claude to read that page again (`--redo`). Make sure you printed
the template at 100 % (actual size), not "fit to page".

### Lines are too close together or too far apart

Line spacing is measured from your pages. Override it with *"set line spacing to 9.6 mm"* (saved
with the profile) or for one run only with *"use 9 mm line spacing for this one"*.

### Text overflows onto extra pages

In per-image mode, an input that doesn't fit on one A4 page continues onto `page_NNb`. This is
expected at your natural writing size. If several inputs are one document, use `--continuous` so
pages are filled. A tighter line spacing setting also fits more on a page.

### Review sheets show boxes instead of labels (Linux)

The labels on contact sheets, specimen sheets and templates need a Unicode font. Install DejaVu
(`sudo apt install fonts-dejavu` on Debian/Ubuntu).

Separately, stand-ins for missing characters that are redrawn from a font use macOS fonts
(Bradley Hand, Noteworthy, Arial Unicode). On Linux those fonts aren't there, so such characters are
left as gaps and reported as NOT DRAWN; the other stand-ins (enlarged lowercase letters, symbols
built from your own strokes) still work. Training on pages that contain those characters avoids
needing a stand-in at all.
