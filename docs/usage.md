# How to use

Once you have a profile (see [How to train](training.md)), give the skill images of text and it
writes them out in your handwriting.

- [The basics](#the-basics)
- [Per-image or continuous](#per-image-or-continuous)
- [Options](#options)
- [What happens when you run it](#what-happens-when-you-run-it)
- [The output folder](#the-output-folder)
- [Missing characters](#missing-characters)
- [Fixing or changing a result](#fixing-or-changing-a-result)
- [The transcript format](#the-transcript-format)
- [What's not supported yet](#whats-not-supported-yet)

---

## The basics

```
/handwritten ~/Desktop/question.png
/handwritten ~/Desktop/scans/                  # every image in the folder, sorted by file name
/handwritten a.png b.png c.png --continuous --pdf
```

Inputs can be:

- **image files**: screenshots, photos of printed or handwritten pages, scans
  (`.png`, `.jpg`, `.heic`, `.webp` …);
- **folders**: every image inside, in file-name order;
- **images pasted into the chat**: fine here, since only their text is needed.

The order you give is the order they're written in.

Plain English works just as well as flags:

> *write these three screenshots out in my handwriting as one continuous document*
>
> *make this a PDF in black pen, like a scan*
>
> *same as last time but with my serial number and name on the first page*

If you have no profile yet, the skill stops and tells you to train first: it never produces pages
without your handwriting.

---

## Per-image or continuous

**Per-image (default): one input, its own page.** Each image starts on a fresh page. Ten inputs give
ten pages, or more if some are long. An input that doesn't fit on one A4 page continues onto the
next, and those pages get letters: `page_03a.png`, `page_03b.png` mean input 3 needed two pages.

Use it when the inputs are separate things: different questions, different notes, different
documents.

**Continuous (`--continuous`): all inputs, one document.** The text of every input flows on from the
last one, filling each page before starting the next. Three inputs that add up to one and a half
pages give two pages, the second half-filled. Pages are named `page_01.png`, `page_02.png` …

Use it when the inputs are parts of one document: several screenshots of one long article, or the
pages of one set of notes. A paragraph that's split between two screenshots is joined back up.

---

## Options

| Option | Default | What it does | Use it when |
|---|---|---|---|
| `--continuous` | off | Flows all inputs into one document (above). | The images are parts of one document. |
| `--pdf` | off | Also writes `handwritten.pdf`: every page in order, each exactly A4. The PNGs are still written. | You need one file to print, upload or send. |
| `--clean` | off | A flat white "scan" instead of the default phone-photo look (paper grain, soft lighting, slight tilt, camera noise, different on every page). | You want to print it, or the photo look isn't wanted. |
| `--ink blue` / `black` / `#rrggbb` | `blue` | Ink colour. `blue` is ballpoint blue, `black` is a soft black, or any hex colour (`#1a237e`). | You write in black, or want to match a particular pen. |
| `--identity` | off | Writes your trained identity block (serial number and name) in the top-left corner of the **first page of the output only**. The text starts below it. | The output is an answer sheet that needs your details at the top. Train it first: [the identity block](training.md#the-identity-block). |
| `--seed N` | random | Fixes the random variation (which variant of each letter, jitter, photo effect), so the same transcript and seed give the same pages. | You liked a result and want to reproduce it, or change only the text and keep the look. The seed is printed every run and saved in `report.json`. |
| `--profile NAME` | `default` | Uses another handwriting profile. | You keep more than one profile. |

Line spacing for a single run: ask for it (*"use 8 mm line spacing for this one"*). To change it
permanently, see [Line spacing](training.md#line-spacing).

Every argument, including the script flags, is in [options.md](options.md).

---

## What happens when you run it

1. **Profile check.** `status.py --require` confirms you have a profile (and, with `--identity`,
   an identity block).
2. **Transcribe.** Claude reads each image and writes its text to `transcripts/input_01.md`,
   `input_02.md` … in the [transcript format](#the-transcript-format): paragraphs, headings and maths.
   It writes exactly what's there, leaves out app or browser chrome, page numbers and watermarks, and
   asks you if a word is unreadable rather than guessing.
3. **Render.** `render.py` lays the text out on A4 at your measured writing size and line spacing,
   picks a different variant of each character every time (using your real joins where it has
   them), and adds natural variation: size, angle, baseline drift and wave, spacing, and a left
   margin that shifts a little from line to line.
4. **Check.** Claude looks at every rendered page against the transcript: every line present,
   nothing cut off, every letter readable and in place, equations centred, scripts in the right
   place. If something is off it fixes the transcript or re-renders with a new seed, up to three
   times.
5. **Deliver.** You get the pages, which input went to which page, the folder, any stand-in or
   missing characters, and the seed.

---

## The output folder

Each run gets its own folder in the directory you started Claude Code from:

```
handwritten_out/20261025-143012/
├── page_01.png              # the finished pages (photo look unless --clean)
├── page_02.png
├── handwritten.pdf          # with --pdf
├── report.json              # seed, mode, pages per input, stand-in and missing characters
├── transcripts/
│   ├── input_01.md          # the text Claude read from each image
│   └── input_02.md
└── qa/
    ├── page_01_top.jpg      # clean renders, split in halves, used for checking
    └── page_01_bottom.jpg
```

Pages are 2480 × 3508 pixels: A4 at 300 DPI.

---

## Missing characters

If the text needs a character your profile doesn't have, the render output says so:

- **drawn another way**: a stand-in, for example an enlarged lowercase letter for a missing
  capital, `≠` built from your `=` and `/`, `⟨ ⟩` from your `<` and `>`, or a font redrawn at your
  pen thickness. Usually fine; Claude mentions which ones.
- **NOT DRAWN**: no stand-in exists, so a gap is left. Claude tells you which characters and where.

Either way, the fix is to train on pages that contain those characters, or the
[maths template](training.md#the-maths-template) for symbols.

---

## Fixing or changing a result

- **A word was read wrong:** tell Claude, or edit `transcripts/input_NN.md` yourself and ask it to
  render again into the same folder.
- **One letter looks odd or two overlap:** ask for a re-render; a new seed picks different variants.
- **The same letter looks wrong every time:** it's a bad variant in your profile. Ask Claude to show
  the specimen sheets and reject it (*"reject the second `k`"*), or reject it yourself by number.
- **Same look, different text:** pass the earlier run's `--seed`.
- **Writing too big or small:** your profile's scale is off; see
  [Troubleshooting](troubleshooting.md#the-writing-is-too-big-or-too-small).

---

## The transcript format

You don't need to know this to use the skill: Claude writes the transcripts. It's here if you want
to edit one, or to know what can be written.

**Text**

| Write | Result |
|---|---|
| one paragraph per line | each line starts a new line on the page; long lines wrap to the page width |
| a blank line | a gap, as between sections |
| `# Heading` | a heading, written like the text on its own line |
| `$$ … $$` alone on a line | a display equation, centred on its own line |
| `________` | an answer blank |

Bold, italics, links and code formatting are dropped; the words are kept.

**Maths**

| Write | Means |
|---|---|
| Unicode symbols: `− × ÷ ± ≠ ≤ ≥ ∈ ∉ ⊂ ∪ ∩ → ⇒ ∞ √ π θ α β λ Σ ℝ 𝔽 ℕ` | the symbol itself (`−` is a real minus) |
| `⟨x, y⟩` | an inner product |
| `⋯` / `…` | centred dots / low dots |
| `x^2`, `a^{−1}`, `u_i^T` | superscripts and subscripts (no spaces inside `{…}`) |
| `x_{1}`, `a_{ij}` | subscripts |
| `√2`, `√λ_i` | a root over the next number or letter (with its scripts) |
| `√{x^2+y^2}`, `√(…)`, `√⟨x,x⟩` | a root over the whole group (no spaces inside) |
| `a/b` | an inline fraction |
| `\frac{num}{den}` | a stacked fraction (display lines only) |
| `\sum_{i=1}^{n}`, `\max_{‖x‖=1}`, `\min_{z∈S}` | limits below and above (display lines); inline, `Σ_{i=1}^{n}` puts them beside it |
| `x̄`, `x̂` | a bar or hat: the letter followed by U+0304 or U+0302 |
| `\pmatrix{3 1 4; 2 α −1; 1 2 1}` | a matrix in round brackets: entries separated by spaces, rows by `;` (display lines only) |
| `\bmatrix{…}`, `\vmatrix{…}` | square brackets / determinant bars |
| `$$ \boxed α = 19. $$` | a box around a display line |
| `\quad` | a wide gap between side-by-side equations |

Notes: matrix entries can't contain spaces (`−1`, `a_{11}` and `x^2` are fine). A letter or word
written straight against a matrix (`a\pmatrix{…}`, `\pmatrix{…}.`) stays close to it. Braces on a
line with a matrix are drawn as tall as the matrix. ℝ, 𝔽, ℕ and similar are written as plain
capitals, the way most people write them by hand.

**Example**

```
# Question 3

Let A be a 3 × 3 matrix with det A ≠ 0. Show that A^{−1} exists and find it for

$$ A = \pmatrix{2 0 1; 1 3 2; 1 1 1} $$

The eigenvalues satisfy λ_1 + λ_2 + λ_3 = tr A = 6, and

$$ ‖x‖ = √{x_1^2+x_2^2+x_3^2} $$
```

---

## What's not supported yet

- Aligned multi-line equations, underbraces and wide hats: Claude writes them as plain lines and
  tells you.
- Tables and diagrams: only text and maths are written. Tables and drawings aren't drawn.
- Paper other than plain white, and page sizes other than A4.
- Joins you never wrote in your training pages are approximated by placing letters side by side.
