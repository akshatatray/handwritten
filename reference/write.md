# Generation procedure

Turns images of text into A4 pages in the user's handwriting. Claude reads the images and writes
transcripts; `scripts/render.py` does the writing; Claude checks every page before handing it over.

`PY=~/.handwritten/.venv/bin/python`; scripts are in `scripts/`. Add `--profile NAME` to every
command when the user named a profile.

## 1. Check the profile

```
$PY scripts/status.py --require
```

Exit code 2 means there is no profile: show the message, tell the user to run `/handwritten train`
with 5–8 photos of their normal handwriting, and stop.

With `--identity`, also run `$PY scripts/identity.py show`. Exit code 2 means no identity block has
been trained: ask for a photo of a page with the block on it and train it first (`reference/identity.md`).

## 2. Work out what the user wants

| The user says | Option |
|---|---|
| nothing special | per-image: 1 input → its own page(s) |
| `--continuous`, "these are the same document", "one after another" | `--continuous`: all inputs flow as one text |
| `--clean`, "like a scan", "no photo effect" | `--clean` |
| `--ink black`, "black pen" | `--ink black` (or `#rrggbb`) |
| `--seed N`, "the same as last time" | `--seed N` (the seed is in the previous run's `report.json`) |
| `--identity`, "put my serial number and name on it", "with my identity" | `--identity`: the trained block in the top-left corner of the **first page of the output only** (in per-image mode too: only `page_01`/`page_01a`) |
| `--pdf`, "as a PDF", "one file" | `--pdf`: also `handwritten.pdf`, every page in order, each exactly A4 (the PNGs are still written) |

Inputs are the images in the message (pasted images are fine here: only their text is needed) and any
file paths or folders given. Keep the order the user gave; for a folder, sort by file name.

Create the output folder in the current directory: `./handwritten_out/<YYYYMMDD-HHMMSS>/`, with a
`transcripts/` folder inside it.

## 3. Transcribe each input

Read each image and write `transcripts/input_NN.md` (NN = 01, 02… in input order). The renderer
re-wraps lines to fit the page, so the source's own line breaks don't matter. The format:

- **One paragraph per line.** Every line starts a new line on the page; long lines wrap.
- **A blank line** leaves a gap, as between sections. Use it where the source has a visible break.
- `# Heading` for a heading (written like the text, on its own line).
- `$$ … $$` alone on a line for a display equation (centred). Inline maths stays in the paragraph.
- **Maths:** use Unicode symbols (`−` minus, `≠ ≤ ≥ ∈ ∉ ⊂ ∪ ∩ → ⇒ ∞ × ÷ ± √ π θ α β λ Σ 𝔽 ℝ ℕ`, `⟨ ⟩` for
  inner products, `⋯` for centred dots and `…` for low dots),
  `^{...}` for superscripts and `_{...}` for subscripts (`a^{−1}`, `x_{1}`, `x^2`, `u_i^T`). No spaces
  inside `{...}`. Write ⩽ ⩾ as `≤ ≥` and a transpose ⊤ as `T`.
- **Roots:** `√2`, `√λ_i`, `√−1` put the bar over the number or letter after √ (with its scripts);
  `√{x^2+y^2}`, `√(…)`, `√⟨x,x⟩` put it over the whole group.
- **Fractions:** inline, write `a/b`. On a display line, `\frac{num}{den}` stacks them
  (`$$ q_1 = \frac{v_1}{‖v_1‖} $$`); spaces inside the braces are fine.
- **Sums and max/min with limits** (display lines): `\sum_{i=1}^{n}`, `\sum_{i≠j}`, `\max_{‖x‖=1}`,
  `\min_{z∈S}` put the limits below and above. Inline, write `Σ_{i=1}^{n}` with the limits beside it.
- **Accents:** a combining mark after the letter: `x̄` (U+0304, bar), `x̂` (U+0302, hat).
  Aligned multi-line equations, underbraces and wide hats aren't supported yet: write them as plain
  lines and tell the user.
- **Matrices** (display lines only): `\pmatrix{…}` for round brackets, `\bmatrix{…}` for square ones,
  `\vmatrix{…}` for determinant bars. Entries are separated by spaces and rows by `;`, so each entry
  must be one word with no spaces inside (`−1`, `a_{11}`, `x^2`):
  `$$ det\pmatrix{3 1 4; 2 α −1; 1 2 1} = 0. $$`. Writing a letter or word straight against the matrix
  (`a\pmatrix{…}`, `\pmatrix{…}.`) keeps it close, as a coefficient or a full stop. A line of several
  matrices is fine: it wraps before a `+` or `=` when it's too wide for the page. Braces on a line
  with a matrix (`{t\bmatrix{1; 2} : t ∈ ℝ}`) are drawn as tall as the matrix.
- **Boxed answers, blanks, wide gaps:** `$$ \boxed α = 19. $$` draws a box around a display line;
  `________` is an answer blank; `\quad` is the space between side-by-side equations.
- **No other markup:** drop bold, italics, links and code formatting; keep the words.
- Leave out everything that isn't the document's text: app or browser chrome, copy buttons, page
  numbers of the source viewer, watermarks.
- Never invent or "fix" text. If a word is unreadable, ask the user rather than guessing.
- **Continuous mode, several images of one document:** a paragraph cut between two images belongs in
  one transcript: finish it at the end of the earlier transcript and start the next transcript after
  it. Start a transcript with a blank line when its image begins a new section, so there's a gap
  before it; leave the blank line out when it carries straight on. (At the top of a page a leading
  blank line is ignored, so it's harmless in per-image mode.)

## 4. Render

```
$PY scripts/render.py --out ./handwritten_out/<ts> [--continuous] [--identity] [--pdf] [--clean] \
    [--ink …] [--seed N] ./handwritten_out/<ts>/transcripts/input_01.md ./handwritten_out/<ts>/transcripts/input_02.md …
```

The output lists each page, the inputs on it, the QA images to check, and the characters the profile
doesn't have:
- *drawn another way* — a stand-in (an enlarged lowercase letter, a ≠ built from = and /, or a
  fallback font redrawn at the user's pen width). Fine to deliver; mention them.
- *NOT DRAWN* — left as a gap. Tell the user which characters and where.

## 5. Check every page (QA)

Read both QA halves of every page (`qa/page_XX_top.jpg`, `qa/page_XX_bottom.jpg`; they overlap a
little). Compare them with the transcript, line by line:
- every paragraph and equation is there, in order, with nothing cut off at an edge;
- no letter is unreadable, wrongly placed (floating, sunk, overlapping a neighbour badly) or wrong;
- display equations are centred and superscripts sit above the line;
- with `--identity`: the first page has both rows of the block in its top-left corner, the line under
  it and the line down its right side, and the text starts below the block without touching it. No
  other page has the block.

If something is wrong, fix the cause and render again into the same folder:
- a transcript mistake → fix the transcript;
- one bad-looking glyph or overlap → re-render with a different `--seed` (variants are random);
- the same glyph wrong every time → note it for the user: it's a bad variant in the profile, which
  `/handwritten train` review can remove.

Do at most three re-renders; then deliver and describe what is still off.

## 6. Deliver

Show the final pages with SendUserFile (PNG files `page_*.png`, not the QA images; with `--pdf`, send
`handwritten.pdf` as well). Tell the user:
- how many pages, and which input went to which page (per-image: `page_03a.png` and `page_03b.png`
  mean input 3 needed two pages);
- the folder they are in;
- any stand-in or missing characters (from the render output), and that pages of their own
  writing containing those characters would fix them;
- the seed, in case they want the same look again.
