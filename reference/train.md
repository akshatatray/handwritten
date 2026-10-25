# Training procedure

Training turns photos of the user's handwriting into a profile. Claude does three jobs the scripts
can't: transcribing each page, labelling the pieces of ink the scripts couldn't label alone, and
reviewing the cut-out glyphs. Do every step for every new page; don't skip the reviews.

`PY=~/.handwritten/.venv/bin/python` and all scripts are in `scripts/`. Add `--profile NAME` to
every command when the user named a profile.

## 1. Get the photos

Take file paths or folders from the arguments. If there are none, ask for them. 5–8 pages of normal
writing is a good first profile; more pages are fine and can be added later with another
`/handwritten train`. Photos of filled-in template sheets (four black corner squares) are read by
`scripts/read_template.py` instead: run it on all the photos first (`reference/template.md`), and
carry on below only with the photos it lists as not template pages. Tips worth telling the user if the photos turn out poor: the whole page flat in
the frame with all four edges visible, even light, no shadow of the phone.

## 2. Preprocess

```
$PY scripts/preprocess.py <paths or folders>
```

Each photo becomes a sample `~/.handwritten/profiles/<name>/samples/<id>/`. Photos already in the
profile are skipped. Read the output:
- `page edges NOT found`: the scale is a guess. Fine to continue, but mention it.
- A warning about ink: open the preview and see whether the photo is usable.
- If the user writes in a notebook that isn't A4 (for example a B5 notebook, 176×250 mm), re-run those
  photos with `--page-mm 176x250 --redo`. A wrong page size makes the writing come out too big or too
  small.
- `page found (unknown)` or a large deskew (±5°) on a photo where the whole page is visible: detection
  picked up the desk too. Read the photo, find the page's four corners in pixels (crop the corners
  with a grid if needed) and re-run that photo alone with
  `--corners TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy`. It should then report `page found (A4)`.

## 3. Transcribe each new page

For each new sample, Read `preview_top.jpg` and `preview_bottom.jpg` (the two halves overlap by a
few lines), then Read `rows.png`: every row of writing the scripts found, in alternating colours,
numbered `R01`, `R02`… down the left. Write `transcript.txt` in the sample folder, one line per row
you transcribe, each starting with its row number:

```
R01 This satisfies the Markov property.
R02 Markov Property : Given the present state the next
R05 → This allow, # and #
R10 Q3. A factory operates.....
```

- **Only transcribe rows of ordinary writing.** Leave out rows that are maths layouts (stacked
  fractions, matrices, working spread over several heights), tables, crossed-out work, and rows
  where two lines of writing ran together. Skipping a row costs nothing; a wrong row teaches wrong
  letters. Notes pages with a lot of maths still give plenty of usable rows.
- The row numbers must be the ones in `rows.png`, each used once, in top-to-bottom order.
- Write each row exactly as written: keep the writer's spelling, capitals, punctuation and
  hyphenation. Words are separated by single spaces where the writer left a space. Punctuation stays
  attached to the word it touches (`dog.`, `soon,`).
- Write symbols as themselves (`→`, `⇒`, `∴`, `≃`, `+`, `=`, `"` for a ditto mark, `③` for a
  circled number) so each symbol is its own word. Use `#` only for a word you can't read, a crossed-out
  word or a maths expression; `#` can stand for something of any width, so a line made mostly of `#`
  can't be split into words reliably: leave such rows out instead.

## 4. Segment and check lines

```
$PY scripts/segment_freeform.py
```

- `PROBLEM`: the message says what's wrong with the transcript (usually a row number that isn't in
  `rows.png`). Fix it and run again.
- For every sample, Read `lines.png`: every word boxed, alternate words in red and blue, with the
  transcript line printed at the right. If a line's words are visibly misassigned, the transcript for
  that row is probably wrong (a missing or extra word): fix it and run again with `--force`.
- Lines reported as `words_touching` are skipped automatically; that's fine.

## 5. Label the sheets

Words the scripts couldn't label alone are on `sheets/sheet_NN.png`. Each row shows a word with each
piece of ink in its own colour and numbered, the transcript word, the number of pieces, and a guess.
`labels.txt` is pre-filled with those guesses, one row per word: `<id> <pieces> <split>`.

Read every sheet and check every row:
- A split has exactly `<pieces>` parts separated by `|`, and the parts joined spell the word. Each
  part is what that coloured piece shows: `th|e` means piece 1 is "th", piece 2 is "e".
- A part may be empty when a piece is only a stray fragment of the piece before it: `i||t`.
- A bar written in the word (`|A|`, a determinant) is `\|` in the split, since `|` separates the
  pieces: `\||A|\||,`.
- Use `?` when the row doesn't show the transcript word (a neighbouring word got pulled in, or part
  of the word is missing), when there's an underline or stray mark in it, or when you can't tell.
  Skipping is always safe.

Write all of a sample's corrections in one command (guesses you don't mention are kept):

```
$PY scripts/set_labels.py ~/.handwritten/profiles/<name>/samples/<id> w005=? w026=V|a|r|i|ab|l|es|. w053=T|o|mo|rrow
```

`set_labels.py` also accepts words that were labelled automatically, if you notice one is wrong.

## 6. Build and review

```
$PY scripts/build_profile.py
```

It prints the coverage report and writes `specimen_NN.png` sheets. Read **every** specimen sheet. In
each row, the least typical variants come first. Note the numbers of bad variants:
- the wrong letter, or two letters cut as one
- a fragment, or a letter with a stray mark or a piece of a neighbour
- sitting clearly too high or too low against the blue baseline

Then reject them in one go and rebuild:

```
$PY scripts/build_profile.py --reject 12,57,103
```

Numbers change on every build, so use only numbers from the sheets you just read. One review pass
is normally enough; do a second pass only if many were rejected.

Then Read every `suspects_NN.png`: variants whose shape looks more like a different character, which
the specimen sheets can miss (an `e` filed as `r` has the right size). Most are harmless lookalikes in
the writer's hand; reject only real mistakes, with their `s` numbers: `--reject s12,s40`.

**Tall brackets and braces.** If a page has brackets or braces drawn around matrices or column
vectors (several lines tall), capture them as tall variants, since ordinary training skips strokes that
tall:

```
$PY scripts/tall_glyphs.py <sample id>                  # numbered sheet: samples/<id>/tall_sheet.png
$PY scripts/tall_glyphs.py <sample id> 1={ 2=} 3=[ 4=]  # label by number; leave out box sides, arrows
$PY scripts/build_profile.py
```

Matrices and tall braces are then drawn with the writer's own strokes (`tall:` in the report).

## 7. Report to the user

Read `preview.png` so you can describe it. Then tell the user, in plain words:
- how many pages were used and roughly how well letters are covered (from the coverage report),
- which letters are missing or have few variants, and that more pages with those letters will fix
  it (a gap sheet for missing characters and math symbols comes in a later phase),
- the measured x-height. Typical handwriting is 2.5–4 mm; if it's far outside that, ask about the page
  size (step 2).

Give the full path to `preview.png` so they can open it.
