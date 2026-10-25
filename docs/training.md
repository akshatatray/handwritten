# How to train

Training builds your **handwriting profile**: every character and join you wrote, cut out of your
photos, plus measurements of your style (letter size, slant, spacing, line pitch). Everything the
skill writes later comes from this profile.

You do two things: take good photos, and answer the occasional question. Claude does the rest:
it runs the scripts, transcribes your pages, labels letters the scripts can't label alone, and
reviews every cut-out glyph.

- [What to photograph](#what-to-photograph)
- [Run training](#run-training)
- [What happens during training](#what-happens-during-training)
- [Check the result](#check-the-result)
- [Improve the profile](#improve-the-profile)
- [The maths template](#the-maths-template)
- [Tall brackets and braces](#tall-brackets-and-braces)
- [The identity block](#the-identity-block)
- [Line spacing](#line-spacing)
- [Several profiles](#several-profiles)
- [Remove a page or start over](#remove-a-page-or-start-over)
- [No notes to photograph?](#no-notes-to-photograph)

---

## What to photograph

**How much:** 5–8 pages of your normal handwriting is a good first profile. More pages give more
variants of each letter and more of your real joins, so the output looks less repetitive. You can
always add pages later.

**What kind of writing:** whatever you naturally write: class notes, letters, journal pages. Write
the way you normally do, not neater: the small differences between your letters are what make the
output look like you. Pages full of maths are fine; Claude only uses the rows of ordinary writing
and skips stacked fractions, matrices and tables.

**How to take the photo:**

- The **whole page** in the frame, with **all four edges visible**. The page edges are how the
  skill works out the real size of your writing.
- Flat (not curling up at the spine), straight on, in **even light**, without your phone's shadow
  on the page.
- As sharp and high-resolution as your phone allows.
- Formats: `.jpg`, `.jpeg`, `.png`, `.heic`/`.heif` (iPhone photos), `.tif`/`.tiff`, `.webp`.

**Paper size:** A4 and US Letter are recognised automatically. For any other size (a B5 notebook,
say), tell Claude the size, for example: *"these are B5 notebook pages, 176 × 250 mm"*. A wrong
page size makes your writing come out too big or too small. Lined paper is fine: the lines are
removed.

**Files, not pasted images.** Training needs the photos as files on disk, because the scripts
process the pixels. Give a folder or file paths (dragging files into the terminal pastes their
paths). An image pasted into the chat only works if Claude Code shows where it saved it.

---

## Run training

```
/handwritten train ~/Pictures/handwriting/
/handwritten train ~/Pictures/page1.heic ~/Pictures/page2.heic
```

Or in plain English: *"learn my handwriting from the photos in ~/Pictures/handwriting"*.

The first run also creates the skill's Python environment (a minute or two, once). Training a
fresh set of pages takes a while, because Claude reads and checks every page and every glyph
sheet. Later runs only process the new photos: photos already in the profile are skipped.

---

## What happens during training

You'll see Claude go through these steps. Each one writes files into
`~/.handwritten/profiles/default/samples/<id>/` that you can open if you're curious.

| Step | Who | What happens |
|---|---|---|
| 1. Read templates | script | `read_template.py` takes any photos of the filled-in maths template and lists the rest as ordinary pages. |
| 2. Preprocess | script | `preprocess.py` finds the page, removes perspective, evens out the lighting, removes ruled lines, deskews, and measures the scale. It reports `page found (A4)` or warns when it had to guess. |
| 3. Transcribe | Claude | Claude reads each page and writes `transcript.txt`, one line per row of writing, exactly as you wrote it (your spelling too). Rows it can't use reliably (maths layouts, crossed-out work, rows that run into each other) are left out on purpose. |
| 4. Segment | script | `segment_freeform.py` splits each row into words and each word into pieces of ink, and writes `lines.png` so Claude can check the words lined up with the transcript. |
| 5. Label | Claude | Words whose pieces couldn't be matched to letters automatically go on contact sheets. Claude labels each piece (`th\|e` means piece 1 is "th", piece 2 is "e") and skips anything doubtful. |
| 6. Build | script | `build_profile.py` cuts out every labelled glyph and join, aligns them on a baseline, drops implausible ones, measures your style, and writes the coverage report, specimen sheets and `preview.png`. |
| 7. Review | Claude | Claude reads every specimen sheet and rejects bad variants (wrong letter, two letters cut as one, a fragment, a stray mark, sitting too high or low), then rebuilds. It also checks "suspects": glyphs whose shape looks like a different character. |
| 8. Report | Claude | Claude tells you how many pages were used, how well letters are covered, which characters are missing or thin, and your measured x-height. |

Claude never guesses: a word it can't read becomes `#` in the transcript, and a piece it isn't sure
about is skipped. Skipping costs a few glyphs; a wrong label would put a wrong letter in your
handwriting.

---

## Check the result

```
/handwritten status
```

This prints the coverage report, for example:

```
Profile "default": 1840 glyphs, 96 ligatures (410 variants), from 8 pages
  a–z   ✔ 26/26, 4–120 variants
  A–Z   ✘ 23/26, 1–14 variants; missing: Q X Z; fewer than 2: J K V
  0–9   ✔ 10/10, 3–40 variants
  punct ✘ 12/16, 1–200 variants; missing: ; @ & $
  math  ✘ 9/26, 1–30 variants; missing: ÷ ≤ ≥ ∫ ∞ θ …
  ligatures: th(31) he(22) in(19) er(17) …
  style: x-height 2.60 mm, slant +4.0 deg, stroke 0.35 mm, line pitch 8.4 mm
  dropped as implausible: 41, as outliers: 18, rejected in review: 23
```

(The exact lines depend on your profile.) Things to look at:

- **Missing or 1-variant characters.** Those will be drawn with a stand-in or look repetitive.
  More pages containing them fix it (see [Improve the profile](#improve-the-profile)).
- **x-height.** The height of a lowercase `x`. Typical handwriting is 2.5–4 mm. If yours is far
  outside that, the page size was probably detected wrongly: tell Claude the real paper size and
  retrain those photos.
- **`preview.png`** in your profile folder: a test sentence written with your profile. Open it.
- **`specimen_NN.png`**: every variant of every character, numbered. If you spot a bad one, tell
  Claude *"reject 57 and 103"* (numbers from the latest sheets: they change on every rebuild).

---

## Improve the profile

Run `train` again with more photos at any time:

```
/handwritten train ~/Pictures/more-pages/
```

New pages are added to the profile and it is rebuilt from all of them. Good things to add:

- pages that contain the characters `status` lists as missing or thin (capitals like J, Q, X, Z,
  digits, punctuation such as `; @ & %`);
- more pages in general, for more variety and more joins;
- the [maths template](#the-maths-template) for symbols ordinary notes rarely have.

---

## The maths template

Ordinary notes rarely contain every maths symbol, and they never show the skill exactly how you
write superscripts, roots or fractions. The maths template fixes both.

**1. Make the sheets:**

```
/handwritten template maths
```

This writes `./handwritten_templates/maths_template.pdf` (6 A4 pages, 487 boxes) and a PNG of each
page. The boxes cover:

| Kind | Examples | What it teaches |
|---|---|---|
| glyph | `∈ ∉ ⊂ ∩ ∪ ∀ ∃ → ⇒ λ θ Σ ≤ ≠ ∞ ⟨ ⟩` … | each symbol, several times |
| script | `Aᵀ`, `A⁻¹`, `xᵢ`, `uᵢᵀ`, `σᵢ²` | how small you write scripts and how high/low you put them |
| radical | `√` drawn round a dotted box | your root sign and its bar |
| tall | brackets and bars as tall as a 2-, 3- or 4-row matrix | matrix brackets and determinant bars |
| fraction | a stacked fraction | your fraction-bar length and spacing |
| bigop | a large `Σ` with limits, `max`/`min` with a limit below | how you lay out sums with limits |
| accent | `x̄`, `ᾱ`, `x̂` | bars and hats over letters |
| word | `span`, `rank`, `det`, `iff` … | operator names and abbreviations |

**2. Fill them in:**

- Print at **100 % / actual size**, not "fit to page". Colour or black and white both work.
- Write each box once, the way you write maths in your notes, in blue or black pen, at your
  normal size.
- Sit letters on the faint dotted line and stay inside the boxes. Leave a box empty if you're
  unsure: empty boxes are skipped.
- For the `√` boxes, draw the root around the dotted box with the bar just above it, and write
  nothing inside the box.

**3. Photograph each page** flat, with all four black corner squares in view.

**4. Train on the photos**, alone or mixed with ordinary pages:

```
/handwritten train ~/Pictures/template-photos/
```

Template pages are recognised by their corner markers and read box by box: no transcribing and
almost no labelling. Claude checks each page's `cells.png` (every box as it was cut out; red names
are empty boxes, orange ones ran over their outline and weren't used). You can send the pages at
different times: each page is read on its own. A page photographed again is only read again if you
ask (`--redo`).

After building, `status` shows extra lines: `scripts:`, `roots:`, `accents:`, `fractions:`,
`big sums:`, `tall:`.

---

## Tall brackets and braces

If your notes have brackets or braces drawn around matrices or column vectors (several lines tall),
ask Claude to capture them: *"capture the tall brackets on that page"*. Ordinary training skips
strokes that tall (they look like rules or box sides), so `tall_glyphs.py` finds them, numbers them
on a sheet, and Claude labels them (`(`, `)`, `[`, `]`, `{`, `}`, `|`). Matrices and tall braces
are then drawn with your own strokes. The maths template's `tall` boxes do the same thing.

---

## The identity block

Some answer sheets have a serial number and name written in the top-left corner, closed off by a
line underneath and a line down its right side. The skill can learn yours from one photo and
write it on the **first page** of an output with `--identity`.

```
/handwritten identity train ~/Pictures/first-page.jpg
```

- Use a photo of a whole page with the block on it (all four edges visible). Higher resolution
  gives crisper characters.
- Claude transcribes the block's rows exactly as written (for example `SERIAL NO. : 12-34-56-78`
  and `NAME : JANE DOE.`), trains it, checks every cut-out character, and shows you a preview.
- Every character in the block comes from that photo, with small differences each time it's
  written, as if you wrote it again.
- Training again with a new photo replaces the block.
- You don't need a full profile to train the block, but you do to use it.

---

## Line spacing

Line spacing is measured from your pages. If you'd like the output tighter or looser, tell Claude:

- *"set line spacing to 9.6 mm"*: saved with the profile and kept when you retrain
  (allowed range 6–15 mm);
- *"use the measured spacing again"*: removes the setting;
- *"use 8 mm line spacing for this one"*: changes a single generation run only.

---

## Several profiles

Add `--profile NAME` to any command, or name the profile in plain English:

```
/handwritten train ~/Pictures/alex/ --profile alex
/handwritten status --profile alex
/handwritten ~/Desktop/essay.png --profile alex
```

Without it, everything uses the profile called `default`. Each profile is a separate folder in
`~/.handwritten/profiles/`.

---

## Remove a page or start over

Each training photo becomes a sample folder in `~/.handwritten/profiles/<NAME>/samples/`. The
profile is rebuilt from all of its samples every time, so:

- **Remove one page:** delete its sample folder, then ask Claude to rebuild the profile
  (`build_profile.py`). The original photo is inside the folder as `original.<ext>`, so you can tell
  which is which.
- **Start over:** delete `~/.handwritten/profiles/<NAME>/` and train again.

---

## No notes to photograph?

Write this out on plain A4 in your normal handwriting (about two pages), then photograph it. Keep
clear gaps between words and a space on both sides of every maths symbol. Don't copy it neatly: write
at your usual speed.

```
The quick brown fox jumps over the lazy dog. Pack my box with five dozen
liquor jugs. Jackdaws love my big sphinx of quartz. How vexingly quick daft
zebras jump! Sixty zippers were quickly picked from the woven jute bag.

Alice, Bob, Carol, Dmitri, Emma, Felix, Grace, Hugo, Ingrid, Jack, Kofi,
Lena, Mateo, Nora, Oscar, Priya, Quentin, Rosa, Sven, Tara, Umar, Vera,
Will, Xavier, Yuki and Zane met in Oslo, Quebec, Zurich, Kyoto and Jaipur.

On 14/07 at 9:30, 26 people paid $185 (about 38%) for 17 tickets & 40 books.
Call 0123 456 789 or write to box 5; ask for Jo, Max or Liz: "Quick!" she said.
Is it 1, 2 or 3? It's 4-5 hours away, not 6/7 - maybe 8 or 9 @ 10% off.

then there these those thing nothing something morning evening during
around ground enough though through which while where when what
other mother brother under wonder before after between question answer
within without little letter better ought brought should would could

Let x , y ∈ R with x + y = 10 and x − y = 4 , so x = 7 and y = 3 .
If a ≠ 0 then a × b ÷ a = b . For all ε > 0 there is δ > 0 with | x | < δ .
A ⊂ B ⇒ A ∩ B = A and A ∪ B = B . α , β , γ , λ , μ , θ , π , σ , ∞ .
```

The last block gives the skill your maths symbols as ordinary words. For scripts, roots,
fractions, matrices and accents, use the [maths template](#the-maths-template).
