# Identity block procedure

The identity block is the serial number and name the user writes in the top-left corner of the first
sheet, closed off by a line underneath (from the left edge of the page) and a line down its right
side (from the top edge). Training learns it from one photo; `render.py --identity` then writes it
again on the first page of the output only, with the user's own characters from that photo and the
small differences of writing it again.

`PY=~/.handwritten/.venv/bin/python`; scripts are in `scripts/`. Add `--profile NAME` to every
command when the user named a profile. A profile is not needed to train the block, but is needed to
use it.

## 1. Get the photo

One photo of a page with the block written on it, the whole page in the frame (all four edges
visible). A pasted image is fine when the chat shows where it was saved (`[Image: source: <path>]`).
The higher the photo's resolution, the crisper the characters.

## 2. Transcribe the block

Read the photo and write down each row of the block exactly as written: capitals, spaces, dots and
colons as they are (`SERIAL NO. : 12-34-56-78`, `NAME : JANE DOE.`). Spaces only
matter for reading: characters are matched in order, left to right.

## 3. Train

```
$PY scripts/identity.py train <photo> --line "<row 1>" --line "<row 2>"
```

- `PROBLEM: the page's edges weren't found`: look at the photo, estimate the page's corners in
  pixels and run again with `--corners x,y,x,y,x,y,x,y` (top-left, top-right, bottom-right,
  bottom-left); they are snapped to the paper's edges, so a few pixels off is fine.
- `PROBLEM: … rows` or `… characters found`: Read `sheet.png` to see what was found. Usually a row
  in the transcript is wrong (a missing dot or colon): fix it and run again.

## 4. Review

Read `identity/sheet.png`: the block with every character boxed and numbered, then each character on
its own with its label. Check every one: the label is the character in the box, and the box holds
one whole character with no piece of a neighbour. Retrain with `--reject 12,31` (numbers from the
sheet) to stop using badly cut ones; a rejected slot uses another copy of the same character from
the photo, or the profile's ordinary one if there is none.

Then Read `identity/preview.png`, the block as it will be written, and compare it with the photo:
same rows and spacing, the line underneath and the line down the right side in the same places.

## 5. Report to the user

Show `preview.png` with SendUserFile and tell the user what was learnt (the output of
`identity.py show`): the rows, the size of the capitals, and where the two lines run. Say that
`--identity` puts it on the first page of the output only, with `--pdf` too, and that training again
with a new photo replaces it.
