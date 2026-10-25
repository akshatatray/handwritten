"""Capture the writer's tall brackets and braces: the ones drawn around matrices and column vectors.

Ordinary training skips strokes several lines tall (they look like rules or box sides), so these
are collected separately. Two steps:

  tall_glyphs.py [--profile NAME] SAMPLE_ID
      finds tall strokes on the page and writes samples/<id>/tall_sheet.png, each numbered
  tall_glyphs.py [--profile NAME] SAMPLE_ID 1={ 2=} 3=[ 4=] ...
      labels them (by the numbers on the sheet); unlabelled ones are ignored. Then run
      build_profile.py, which adds them to the profile as tall variants.

Labels may be ( ) [ ] { } or | (a determinant bar).
"""

import argparse
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import read_json, samples_dir, ui_font, write_json
from segment_freeform import letter_height, load_page

TALL_CHARS = set("()[]{}|")


def find_tall(sdir):
    """Tall, thin strokes with any small pieces at their ends (a bracket's separate serifs)."""
    mask, ink, n, cc, stats, cents = load_page(sdir)
    lh = letter_height(stats)
    x, y, w, h = (stats[:, i] for i in range(4))
    tall = [c for c in range(1, n) if h[c] > 2.5 * lh and w[c] < 0.6 * h[c]]
    small = [c for c in range(1, n) if h[c] < 1.2 * lh and w[c] < 1.5 * lh and c not in tall]
    found = []
    for c in tall:
        comps = [int(c)]
        x0, y0, x1, y1 = x[c], y[c], x[c] + w[c], y[c] + h[c]
        for s in small:
            sx0, sy0, sx1, sy1 = x[s], y[s], x[s] + w[s], y[s] + h[s]
            overlap = min(x1, sx1) - max(x0, sx0)
            near_end = sy1 > y0 - 0.4 * lh and sy0 < y0 + 0.5 * lh or sy0 < y1 + 0.4 * lh and sy1 > y1 - 0.5 * lh
            if overlap > 0 and near_end:
                comps.append(int(s))
        bx0 = int(min(x[k] for k in comps))
        by0 = int(min(y[k] for k in comps))
        bx1 = int(max(x[k] + w[k] for k in comps))
        by1 = int(max(y[k] + h[k] for k in comps))
        found.append({"comps": comps, "bbox": [bx0, by0, bx1, by1]})
    found.sort(key=lambda f: (round(f["bbox"][1] / (4 * lh)), f["bbox"][0]))
    for i, f in enumerate(found, 1):
        f["n"] = i
    return found, n, ink, cc


def write_sheet(sdir, found, ink, cc):
    tiles = []
    f_num = ui_font(26)
    for f in found:
        x0, y0, x1, y1 = f["bbox"]
        pad = 10
        crop = ink[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].astype(np.float32) / 255
        own = np.isin(cc[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad], f["comps"])
        rgb = np.full(crop.shape + (3,), 255, np.float32)
        a = crop[..., None]
        rgb = np.where(own[..., None], rgb * (1 - a) + np.array([20, 60, 200], np.float32) * a,
                       rgb * (1 - 0.3 * a) + np.array([170, 170, 170], np.float32) * 0.3 * a)
        img = Image.fromarray(rgb.astype(np.uint8))
        img.thumbnail((140, 220))
        tile = Image.new("RGB", (170, 270), "white")
        tile.paste(img, ((170 - img.width) // 2, 40))
        ImageDraw.Draw(tile).text((8, 6), str(f["n"]), fill=(220, 30, 30), font=f_num)
        tiles.append(tile)
    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (170 * cols, 270 * max(rows, 1)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (170 * (i % cols), 270 * (i // cols)))
    sheet.save(sdir / "tall_sheet.png")
    return sdir / "tall_sheet.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample")
    ap.add_argument("labels", nargs="*", help="N=char, e.g. 1={")
    ap.add_argument("--profile", default="default")
    args = ap.parse_args()

    sdir = samples_dir(args.profile) / args.sample
    if not (sdir / "mask.png").exists():
        sys.exit(f"ERROR: no sample {args.sample}")
    if not args.labels:
        found, n, ink, cc = find_tall(sdir)
        write_json(sdir / "tall_candidates.json", {"n_components": int(n), "candidates": found})
        path = write_sheet(sdir, found, ink, cc)
        print(f"{len(found)} tall strokes found. Label them by number from: {path}")
        return
    cand = read_json(sdir / "tall_candidates.json")
    if not cand:
        sys.exit("ERROR: run without labels first to find the tall strokes")
    by_n = {str(f["n"]): f for f in cand["candidates"]}
    labels = {}
    for item in args.labels:
        num, _, ch = item.partition("=")
        if num not in by_n or ch not in TALL_CHARS:
            sys.exit(f"ERROR: '{item}': use a number from the sheet and one of {''.join(sorted(TALL_CHARS))}")
        labels[num] = ch
    write_json(sdir / "tall_labels.json", labels)
    print(f"{len(labels)} tall strokes labelled: " + " ".join(f"{k}={v}" for k, v in labels.items()))
    print("Now run build_profile.py to add them to the profile.")


if __name__ == "__main__":
    main()
