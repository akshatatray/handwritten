# Template procedure

Printable sheets with one box per symbol. The user hand-writes each box, photographs the pages, and
the scripts read every box back on their own: no transcribing, and labelling only for the few words
whose letters can't be matched to their pieces. Templates live in `templates/<name>.json`; `maths`
covers the symbols, scripts, roots, fractions, sums and operator names of typical linear-algebra
notes.

`PY=~/.handwritten/.venv/bin/python`; scripts are in `scripts/`. Add `--profile NAME` to every
command when the user named a profile.

## 1. Make the sheets (`/handwritten template maths`)

```
$PY scripts/template_sheets.py maths --out ./handwritten_templates
```

Give the user the PDF path and tell them:
- print at 100 % (actual size, not "fit to page"); colour or black and white both work;
- write each box once, the way they write maths in notes, in blue or black pen, at their normal size;
- sit letters on the faint dotted line, stay inside the boxes, and leave a box empty if unsure
  (empty boxes are simply skipped);
- for the √ boxes, draw the root round the dotted box with the bar just above it, writing nothing
  inside the box;
- photograph each page flat, with all four black corner squares in view.

## 2. Read the photos (`/handwritten train <photos>`)

When `train` gets photos, run the reader first; it takes the template pages and lists the rest:

```
$PY scripts/read_template.py <paths or folders>
```

- `OK … page N, a/b boxes written`: read. Read its `cells.png`: every box as it was cut out. Red
  names are empty boxes, orange ones ran over their outline and were not used. Check that the
  boxes show whole symbols that match their names; if a whole page looks shifted or garbled, ask
  for a sharper, flatter photo of that page.
- `OTHER … no template markers`: an ordinary page of writing. Train it with `reference/train.md`.
- `N words need labels`: label them exactly as in `reference/train.md` step 5 (sheets in the
  sample's `sheets/`, `set_labels.py`).
- A page photographed twice is read once; `--redo` reads it again after a better photo replaced it.

## 3. Build and review

```
$PY scripts/build_profile.py
```

The report gains maths lines: `scripts:` (the user's own small characters and how high and small
they write them), `roots:`, `accents:`, `fractions:`, `big sums:`. Review the specimen sheets as in
`reference/train.md` step 6; the template pieces have their own rows (`s 2` = a script 2, `√`,
`bigΣ`, `acc`) and are rejected by number like any glyph.

## 4. Report

Tell the user which template pages were read, how many boxes were used, and what the profile now
has (the report lines above). Pages of the template they haven't sent yet can come later: each
page is read on its own.
