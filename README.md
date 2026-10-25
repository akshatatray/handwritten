# handwritten

A [Claude Code](https://claude.com/claude-code) skill that writes text out in **your own handwriting**.

Train it once on photos of pages you have written. After that, give it images of text (screenshots,
photos of printed pages, notes, maths) and it gives you back A4 pages written in your hand: your
letters, your joins, your slant and spacing, as a phone-photo-style PNG or a clean scan, plus an
optional PDF.

```
/handwritten train ~/Pictures/my-notes/          # once: learn your handwriting
/handwritten ~/Desktop/question.png --pdf        # any time: write this out by hand
```

- **It's really your writing.** Every letter on the page is one you wrote, cut from your photos.
  Nothing is a font (unless a character is missing, and then it tells you).
- **Not a copy-paste of one letter.** It keeps several versions of each character and each join
  (`th`, `ing`, `ou` …) and varies size, angle, baseline and spacing, so no two words look stamped.
- **Maths works.** Superscripts, subscripts, fractions, roots, sums with limits, matrices,
  Greek letters and set symbols, laid out the way you write them.
- **Runs locally.** No GPU and no model training. Claude reads the images and checks the work;
  bundled Python scripts do the image processing. Your handwriting profile stays on your machine.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Commands at a glance](#commands-at-a-glance)
- [Generation options at a glance](#generation-options-at-a-glance)
- [How it works](#how-it-works)
- [Where things are stored](#where-things-are-stored)
- [Privacy and responsible use](#privacy-and-responsible-use)
- [Limitations](#limitations)
- Detailed guides:
  - [How to train](docs/training.md): photos, the training steps, reviewing, the maths template,
    the identity block
  - [How to use](docs/usage.md): generating pages, modes, output, the transcript format
  - [Every argument](docs/options.md): every command, option and script flag, and when to use each
  - [Troubleshooting](docs/troubleshooting.md)

---

## Install

**You need:**

- [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) (CLI, desktop app or IDE extension)
- Python 3.9 or newer on your `PATH` (`python3 --version`)
- macOS (tested) or Linux. On Linux, install the DejaVu fonts (`fonts-dejavu` on Debian/Ubuntu)
  for the labels on review sheets; font-drawn stand-ins for missing characters need macOS fonts, so
  on Linux those characters are left as gaps (see
  [Troubleshooting](docs/troubleshooting.md#review-sheets-show-boxes-instead-of-labels-linux)).
  Windows hasn't been tested (the setup script is bash).

**Install for all your projects** (a personal skill):

```bash
git clone https://github.com/akshatatray/handwritten.git ~/.claude/skills/handwritten
```

**Or for one project only** (run from that project's root):

```bash
git clone https://github.com/akshatatray/handwritten.git .claude/skills/handwritten
```

Start a new Claude Code session (or restart the current one) and type `/handwritten status`.
It should say there is no profile yet: that means the skill is installed.

The first time the skill runs, it creates its own Python environment at `~/.handwritten/.venv` and
installs NumPy, SciPy, OpenCV, scikit-image and Pillow into it. That takes a minute or two once;
after that it starts instantly. Nothing is installed into your system Python.

**Update:**

```bash
git -C ~/.claude/skills/handwritten pull
```

Your profile lives outside the skill folder, so updating or reinstalling never touches it.

**Uninstall:** delete `~/.claude/skills/handwritten`. To also delete your handwriting profile and the
Python environment, delete `~/.handwritten`.

---

## Quick start

**1. Photograph 5–8 pages of your normal handwriting.** Notes, letters, anything you wrote
naturally. Whole page in frame, all four edges visible, flat, even light. Put them in a folder.
Don't have any? Write out the practice text in [docs/training.md](docs/training.md#no-notes-to-photograph).

**2. Train.**

```
/handwritten train ~/Pictures/handwriting/
```

Claude processes the photos, transcribes each page, labels the letters the scripts couldn't
label alone, and reviews every cut-out glyph, rejecting bad ones. This takes a while the first
time (it's careful work); you mostly watch. At the end you get a coverage report and a preview
sentence written in your hand.

**3. Check what it learnt.**

```
/handwritten status
```

**4. Write something.**

```
/handwritten ~/Desktop/page1.png ~/Desktop/page2.png --continuous --pdf
```

You can also paste images straight into the chat for this step, or just ask in plain English:
*"write these three screenshots out in my handwriting as one PDF, black pen"*.

Output goes to `./handwritten_out/<timestamp>/` in your current directory.

**5. (Optional) Add maths symbols.** Ordinary notes rarely contain every symbol. Print the maths
template, fill it in, photograph it, and train on the photos:

```
/handwritten template maths
/handwritten train ~/Pictures/template-photos/
```

---

## Commands at a glance

All of these also work as plain English ("train on the photos in ~/notes", "what does my profile
cover?"). Full details and when to use each: [docs/options.md](docs/options.md).

| Command | What it does |
|---|---|
| `/handwritten train <photos or folder>` | Learn your handwriting from photos of pages you wrote, or add more pages to an existing profile. Photos of the filled-in maths template are detected and read automatically. |
| `/handwritten template maths` | Make printable template sheets (a 6-page PDF, one box per maths symbol) to fill in and train on. |
| `/handwritten identity train <photo>` | Learn the serial-number/name block you write in the top-left corner of an answer sheet, so `--identity` can put it on page 1. |
| `/handwritten status` | Show what your profile covers: characters, variants, joins, maths pieces, measured style. |
| `/handwritten <images…> [options]` | Write the text in the images out in your handwriting. This is the default: anything that isn't one of the commands above. |
| "set line spacing to 9.6 mm" | Change a saved profile setting (line spacing is the only one so far). "use the measured spacing again" undoes it. |
| `/handwritten gaps` | Not built yet: lists the characters your profile is missing and points you at the maths template. |

Add `--profile NAME` to any command to use a profile other than `default` (for example, a second
person's handwriting).

## Generation options at a glance

| Option | Default | Use it when |
|---|---|---|
| *(none)* | per-image | Each input image becomes its own page (or pages, if it's long). |
| `--continuous` | off | The images are parts of one document and the text should flow from one into the next. |
| `--pdf` | off | You want one A4 PDF of all the pages as well as the PNGs. |
| `--clean` | off | You want a flat white scan instead of the default phone-photo look. |
| `--ink blue\|black\|#rrggbb` | `blue` | You write in another colour. |
| `--identity` | off | You want your trained serial number/name block on the first page. |
| `--seed N` | random | You want exactly the same variation as a previous run (the seed is printed every time). |
| `--profile NAME` | `default` | You keep more than one handwriting profile. |

Details, examples and the transcript format: [docs/usage.md](docs/usage.md).

---

## How it works

Claude does the parts that need understanding; Python does the pixels.

**Training**

1. `preprocess.py` finds the page in each photo, removes perspective, evens out lighting, removes
   ruled lines, deskews, and measures the physical scale from the page edges.
2. Claude reads each page and writes a transcript, row by row (skipping maths layouts, tables and
   crossed-out work).
3. `segment_freeform.py` splits each row into words and each word into pieces of ink. In
   semi-joined writing a piece is either one letter or a run you wrote without lifting the pen
   (`th`, `ing`): those runs become *ligatures*, so your real joins are reused.
4. Pieces that can't be matched to letters automatically go onto contact sheets; Claude labels
   them (`th|e`), and anything doubtful is skipped rather than guessed.
5. `build_profile.py` cuts out every labelled glyph, aligns them on a measured baseline, drops
   outliers, measures your style (x-height, slant, spacing, line pitch) and writes specimen
   sheets. Claude reviews every variant and rejects bad cuts.

**Generation**

1. Claude reads each input image and writes a transcript (plain text plus a small maths syntax).
2. `render.py` lays the text out on A4 at your measured size and spacing, picks a different
   variant of each character every time (using your joins where it has them), adds natural
   jitter, and inks it in ballpoint blue (or your colour).
3. Claude checks every rendered page against the transcript and re-renders if something is off.
4. `photo_effect.py` gives each page its own lighting, tilt and grain (unless `--clean`).

---

## Where things are stored

| What | Where |
|---|---|
| The skill (this repo) | `~/.claude/skills/handwritten/` |
| Python environment | `~/.handwritten/.venv/` |
| Your handwriting profile | `~/.handwritten/profiles/default/` (or `.../profiles/<NAME>/`) |
| Generated pages | `./handwritten_out/<YYYYMMDD-HHMMSS>/` in the directory you run Claude Code from |
| Printable templates | `./handwritten_templates/` in the directory you run Claude Code from |

Set the `HANDWRITTEN_HOME` environment variable to keep the environment and profiles somewhere
other than `~/.handwritten`.

A profile folder contains your training photos (copied), the cut-out glyphs, `profile.json`,
`coverage.txt`, `settings.json`, the review sheets and `preview.png`. Training rebuilds the profile
from all of its samples each time, so to start over, delete the profile folder.

---

## Privacy and responsible use

- **Your profile is a reproduction of your handwriting.** Keep it private. It never leaves your
  machine unless you copy it somewhere. This repository contains only code: no handwriting, no
  profile, no sample images (and `.gitignore` blocks images and profile folders as a safety net).
- **Claude sees your images.** To transcribe pages and check results, Claude reads the photos and
  rendered pages, the same as any image you share in a Claude Code session. The profile itself is
  built and stored locally by the scripts.
- **Don't use it to deceive.** Not for signatures, forging documents, or passing off generated
  pages as handwritten where that matters (exams, legal or official documents). The skill refuses
  signatures, and PDFs it makes carry metadata saying they were generated.

---

## Limitations

- English letters, digits, punctuation and maths notation. Other scripts aren't supported.
- Joins you never wrote in your samples are approximated by placing letters side by side. More
  training pages mean more real joins.
- Characters missing from your profile are drawn with a stand-in (an enlarged lowercase letter, a
  symbol built from others, or a font redrawn at your pen width) and reported. Characters with no
  stand-in are left as a gap and reported.
- Not supported yet: aligned multi-line equations, underbraces, wide hats, tables and drawings,
  paper styles other than plain white, page sizes other than A4.
- Quality depends on the training photos: sharp, flat, evenly lit whole pages work best.

---

## License

[MIT](LICENSE)
