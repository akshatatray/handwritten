"""Printable template sheets: one box per symbol, read back automatically from a photo.

Each A4 page has four corner markers that say which template and page it is, so read_template.py
can find every box again in a phone photo without any transcribing or labelling. The boxes come
from a spec in templates/<name>.json; this file both draws the pages and gives the reader the
exact layout, so the two can never disagree.

Box kinds:
  glyph     one symbol (∈, λ, ℬ …) on a faint baseline
  word      a short word or abbreviation (span, rank, iff …)
  script    a letter with its superscript and/or subscript as you write them (x², A⁻¹, a_ij …)
  radical   a √ drawn round a dotted box: the box stands for what goes under the root
  tall      a bracket or bar as tall as a 2-, 3- or 4-row matrix
  fraction  a stacked fraction, numerator over denominator
  bigop     a large Σ or ∏ with its limits below and above
  accent    a letter with a bar, hat, tilde or dot over it

Usage: template_sheets.py maths [--out DIR]
  writes DIR/maths_template.pdf (print at 100 %) and a PNG per page
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import A4_MM, PX_PER_MM

SPEC_DIR = Path(__file__).resolve().parent.parent / "templates"
ARUCO_DICT = cv2.aruco.DICT_4X4_250
MARKER_MM = 11.0
MARKER_AT = [(7.0, 7.0), (192.0, 7.0), (192.0, 279.0), (7.0, 279.0)]  # top-left corner of each marker
CONTENT = (10.0, 24.0, 200.0, 275.0)  # x0, y0, x1, y1 of the area the boxes fill
GAP = 2.4  # between boxes: room for the baseline ticks outside each box
SECTION_H = 7.0
LABEL_H = 4.6  # strip at the top of a box for the printed label
LETTER_MM = 4.6  # room per handwritten character: wider than most hands write, so long words fit
TALL_LABEL_H = 9.5  # fractions and big sums show a small typeset example to copy
INK_GREY = 105  # printed labels
LINE_GREY = 150  # box outlines and ticks
DOT_GREY = 214  # faint guides inside the writing area (too light and small to be read as ink)

FONTS = [
    "/System/Library/Fonts/Supplemental/STIXTwoMath.otf",
    "/System/Library/Fonts/Supplemental/STIXGeneral.otf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
TEXT_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def px(mm):
    return int(round(mm * PX_PER_MM))


def load_spec(name):
    path = SPEC_DIR / f"{name}.json"
    if not path.exists():
        sys.exit(f"ERROR: no template '{name}' (looked for {path})")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- layout

def cell_geometry(kind, item):
    """Size of a box (w, h in mm) and where things go inside its writing area, relative to the
    writing area's top-left corner. Fixed sizes, so the reader always gets the same layout."""
    zone_w, zone_h, base = 11.0, 14.6, 7.2  # 7 mm above and below the line: tall loops and long tails
    extra = {}
    if kind == "glyph" or kind == "accent":
        zone_w = 15.0 if item.get("wide") or len(item.get("text", "x")) > 1 else 11.5
        if kind == "accent":
            zone_h, base = 14.5, 9.0
    elif kind == "word":
        zone_w = max(18.0, 7.0 + LETTER_MM * len(item["text"]))
    elif kind == "script":
        n = max(len(item.get("sup", "")), len(item.get("sub", ""))) + len(item["base"])
        zone_w = max(16.0, 8.0 + LETTER_MM * n)
        zone_h, base = 16.5, 9.0
    elif kind == "radical":
        w = float(item["width_mm"])
        zone_w, zone_h, base = w + 15.0, 15.0, 11.0
        # The dotted box: where the expression under the root would be.
        extra["placeholder"] = [9.0, base - 5.0, 9.0 + w, base + 0.7]
    elif kind == "tall":
        rows = int(item["rows"])
        zone_w, zone_h, base = 11.0, 6.5 * rows + 5.0, None
        extra["rows"] = rows
    elif kind == "fraction":
        n = max(len(plain(item["num"])), len(plain(item["den"])))
        zone_w, zone_h, base = max(22.0, 10.0 + LETTER_MM * n), 18.0, None
    elif kind == "bigop":
        zone_w, zone_h, base = 22.0, 22.0, None
    return zone_w, zone_h + label_height(kind), base, extra


def plain(s):
    """Characters written, without the ^ _ { } markup."""
    return "".join(c for c in s if c not in "^_{}")


def label_height(kind):
    return TALL_LABEL_H if kind in ("fraction", "bigop") else LABEL_H


def layout(spec):
    """Every page and box of a template: [{"page": n, "cells": [...], "sections": [...]}]."""
    x0, y0, x1, y1 = CONTENT
    pages, cells, sections = [], [], []
    page, y, row_h, x = 0, y0, 0.0, x0

    def new_page():
        nonlocal page, y, row_h, x, cells, sections
        pages.append({"page": page, "cells": cells, "sections": sections})
        page, y, row_h, x = page + 1, y0, 0.0, x0
        cells, sections = [], []

    for s_index, section in enumerate(spec["sections"]):
        kind = section["kind"]
        items = []
        for item in section["items"]:
            items += [item] * int(item.get("n", 1))
        first = True
        k = 0
        while k < len(items):
            if first:
                if y + SECTION_H + 20 > y1:
                    new_page()
                sections.append({"title": section["title"], "note": section.get("note", ""), "y": y,
                                 "continued": False})
                y += SECTION_H
                first = False
                x, row_h = x0, 0.0
            item = items[k]
            zw, zh, base, extra = cell_geometry(kind, item)
            w, h = zw, zh
            if x + w > x1 + 0.01:
                x, y, row_h = x0, y + row_h + GAP, 0.0
            if y + h > y1:
                new_page()
                sections.append({"title": section["title"] + " (continued)", "note": "", "y": y,
                                 "continued": True})
                y += SECTION_H
            cell = {"id": f"p{page + 1}c{len(cells) + 1:03d}", "kind": kind, "section": s_index,
                    "rect": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
                    "zone": [round(x, 2), round(y + label_height(kind), 2), round(x + w, 2), round(y + h, 2)]}
            cell.update({key: item[key] for key in ("text", "base", "sup", "sub", "op", "upper", "lower",
                                                    "num", "den", "accent", "width_mm", "rows", "label")
                         if key in item})
            if base is not None:
                cell["baseline"] = round(y + label_height(kind) + base, 2)
            if "placeholder" in extra:
                p = extra["placeholder"]
                cell["placeholder"] = [round(x + p[0], 2), round(y + label_height(kind) + p[1], 2),
                                       round(x + p[2], 2), round(y + label_height(kind) + p[3], 2)]
            cells.append(cell)
            x += w + GAP
            row_h = max(row_h, h)
            k += 1
        x, y, row_h = x0, y + row_h + GAP + 1.0, 0.0
    pages.append({"page": page, "cells": cells, "sections": sections})
    return [p for p in pages if p["cells"]]


def marker_ids(set_index, page):
    base = set_index * 40 + page * 4
    return [base, base + 1, base + 2, base + 3]


# ---------------------------------------------------------------- drawing

def font(paths, size_px):
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size_px)
    return ImageFont.load_default()


class Painter:
    def __init__(self):
        self.img = Image.new("L", (px(A4_MM[0]), px(A4_MM[1])), 255)
        self.d = ImageDraw.Draw(self.img)
        self.math = {}
        self.text = {}

    def mf(self, mm):
        s = px(mm)
        if s not in self.math:
            self.math[s] = font(FONTS, s)
        return self.math[s]

    def tf(self, mm):
        s = px(mm)
        if s not in self.text:
            self.text[s] = font(TEXT_FONTS, s)
        return self.text[s]

    def put(self, xy_mm, s, size_mm, grey=INK_GREY, math=True, anchor="ls"):
        f = self.mf(size_mm) if math else self.tf(size_mm)
        self.d.text((px(xy_mm[0]), px(xy_mm[1])), s, fill=grey, font=f, anchor=anchor)
        l, t, r, b = self.d.textbbox((px(xy_mm[0]), px(xy_mm[1])), s, font=f, anchor=anchor)
        return r / PX_PER_MM

    def width(self, s, size_mm):
        f = self.mf(size_mm)
        l, t, r, b = self.d.textbbox((0, 0), s, font=f, anchor="ls")
        return (r - l) / PX_PER_MM

    def put_math(self, xy_mm, s, size_mm, grey=INK_GREY, dry=False):
        """Text with ^{…} and _{…} (or ^c, _c) drawn as small raised and lowered runs; returns the
        width in mm. dry=True only measures."""
        x, y = xy_mm
        start = x
        for text, level in runs(s):
            size = size_mm * (0.66 if level else 1.0)
            dy = {1: -0.42 * size_mm, -1: 0.2 * size_mm, 0: 0.0}[level]
            if not dry:
                self.put((x, y + dy), text, size, grey=grey)
            x += self.width(text, size) + (0.1 if level else 0.0)
        return x - start

    def rect(self, r, grey=LINE_GREY, width_mm=0.22):
        self.d.rectangle([px(r[0]), px(r[1]), px(r[2]), px(r[3])], outline=grey, width=max(1, px(width_mm)))

    def dotted_h(self, x0, x1, y, grey=DOT_GREY, step_mm=1.2):
        r = max(1, px(0.08))
        for xx in np.arange(x0 + 0.6, x1 - 0.4, step_mm):
            self.d.ellipse([px(xx) - r, px(y) - r, px(xx) + r, px(y) + r], fill=grey)

    def dotted_rect(self, r, grey=170, step_mm=0.9):
        rad = max(1, px(0.11))
        x0, y0, x1, y1 = r
        pts = [(xx, y0) for xx in np.arange(x0, x1, step_mm)] + [(xx, y1) for xx in np.arange(x0, x1, step_mm)]
        pts += [(x0, yy) for yy in np.arange(y0, y1, step_mm)] + [(x1, yy) for yy in np.arange(y0, y1 + 0.01, step_mm)]
        for xx, yy in pts:
            self.d.ellipse([px(xx) - rad, px(yy) - rad, px(xx) + rad, px(yy) + rad], fill=grey)

    def tick(self, x, y, length=1.6):
        self.d.line([px(x), px(y), px(x + length), px(y)], fill=LINE_GREY, width=max(1, px(0.3)))

    def marker(self, marker_id, x_mm, y_mm):
        side = px(MARKER_MM)
        m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, side)
        self.img.paste(Image.fromarray(m), (px(x_mm), px(y_mm)))


def runs(s):
    """'x^{1/2}' -> [('x', 0), ('1/2', 1)]; '_' gives level -1."""
    out, buf, i = [], "", 0
    while i < len(s):
        c = s[i]
        if c in "^_" and i + 1 < len(s):
            if buf:
                out.append((buf, 0))
                buf = ""
            level = 1 if c == "^" else -1
            if s[i + 1] == "{":
                j = s.find("}", i + 2)
                j = len(s) if j == -1 else j
                out.append((s[i + 2:j], level))
                i = j + 1
            else:
                out.append((s[i + 1], level))
                i += 2
            continue
        buf += c
        i += 1
    if buf:
        out.append((buf, 0))
    return out


def label(p, cell):
    """The printed example at the top of a box: what to write in it."""
    x0, y0 = cell["rect"][0] + 1.2, cell["rect"][1] + 3.6
    kind = cell["kind"]
    if "label" in cell:
        p.put((x0, y0), cell["label"], 2.9)
        return
    if kind in ("glyph", "word"):
        p.put((x0, y0), cell["text"], 3.2)
    elif kind == "accent":
        p.put((x0, y0), cell["base"] + cell["accent"], 3.2)
    elif kind == "script":
        x = p.put((x0, y0), cell["base"], 3.2)
        if "sub" in cell and "sup" in cell:
            p.put((x + 0.2, y0 - 1.6), cell["sup"], 2.1)
            p.put((x + 0.2, y0 + 0.7), cell["sub"], 2.1)
        elif "sup" in cell:
            p.put((x + 0.2, y0 - 1.5), cell["sup"], 2.1)
        else:
            p.put((x + 0.2, y0 + 0.7), cell["sub"], 2.1)
    elif kind == "radical":
        p.put((x0, y0), "√", 3.2)
    elif kind == "tall":
        x = p.put((x0, y0), cell["text"], 3.2)
        p.put((x + 0.8, y0), f"{cell['rows']} rows", 2.2, math=False)
    elif kind == "fraction":
        # A small stacked example: numerator, bar, denominator.
        size = 2.4
        wn, wd = p.put_math((0, 0), cell["num"], size, dry=True), p.put_math((0, 0), cell["den"], size, dry=True)
        w = max(wn, wd) + 0.8
        cx = cell["rect"][0] + 2.0 + w / 2
        top = cell["rect"][1]
        p.put_math((cx - wn / 2, top + 3.6), cell["num"], size)
        p.d.line([px(cx - w / 2), px(top + 4.7), px(cx + w / 2), px(top + 4.7)], fill=INK_GREY, width=max(1, px(0.15)))
        p.put_math((cx - wd / 2, top + 7.6), cell["den"], size)
    elif kind == "bigop":
        top = cell["rect"][1]
        cx = cell["rect"][0] + 5.0
        wo = p.width(cell["op"], 4.2)
        p.put((cx - wo / 2, top + 6.9), cell["op"], 4.2)
        if cell.get("upper"):
            wu = p.put_math((0, 0), cell["upper"], 1.9, dry=True)
            p.put_math((cx - wu / 2, top + 2.3), cell["upper"], 1.9)
        if cell.get("lower"):
            wl = p.put_math((0, 0), cell["lower"], 1.9, dry=True)
            p.put_math((cx - wl / 2, top + 9.0), cell["lower"], 1.9)


def draw_page(spec, page, n_pages, set_index):
    p = Painter()
    for k, (mx, my) in enumerate(MARKER_AT):
        p.marker(marker_ids(set_index, page["page"])[k], mx, my)
    p.put((21.0, 11.5), f"Handwriting template: {spec['title']}", 4.0, grey=40, math=False)
    p.put((21.0, 16.5), f"Page {page['page'] + 1} of {n_pages}   ·   {spec['id']}", 2.6, grey=90, math=False)
    for k, line in enumerate(spec["instructions"]):
        p.put((95.0, 10.0 + 3.4 * k), line, 2.3, grey=70, math=False)
    for s in page["sections"]:
        p.put((CONTENT[0], s["y"] + 4.6), s["title"], 3.1, grey=50, math=False)
        if s.get("note"):
            p.put((CONTENT[0] + 2.0 + len(s["title"]) * 1.75, s["y"] + 4.6), s["note"], 2.3, grey=95, math=False)
    for cell in page["cells"]:
        r = cell["rect"]
        p.rect(r)
        label(p, cell)
        z = cell["zone"]
        if "baseline" in cell:
            p.dotted_h(z[0], z[2], cell["baseline"])
            p.tick(r[0] - 1.8, cell["baseline"])
            p.tick(r[2] + 0.2, cell["baseline"])
        if "placeholder" in cell:
            p.dotted_rect(cell["placeholder"])
        if cell["kind"] == "tall":
            # Faint row marks: how tall a matrix of that many rows is.
            rows = cell["rows"]
            top, bot = z[1] + 2.0, z[3] - 2.0
            for k in range(rows):
                yy = top + (k + 0.5) * (bot - top) / rows
                p.dotted_h(z[0] + 5.0, z[2] - 0.5, yy, step_mm=1.0)
    p.put((21.0, 286.0), spec.get("footer", ""), 2.2, grey=110, math=False)
    return p.img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="template name, e.g. maths")
    ap.add_argument("--out", default=".", help="folder for the PDF and page images")
    args = ap.parse_args()
    spec = load_spec(args.name)
    pages = layout(spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = [draw_page(spec, pg, len(pages), spec["set_index"]) for pg in pages]
    for k, img in enumerate(images):
        img.save(out / f"{args.name}_template_p{k + 1}.png")
    pdf = out / f"{args.name}_template.pdf"
    rgb = [im.convert("RGB") for im in images]
    rgb[0].save(pdf, save_all=True, append_images=rgb[1:], resolution=300.0)
    n = sum(len(pg["cells"]) for pg in pages)
    print(f"{len(pages)} pages, {n} boxes: {pdf}")
    for pg in pages:
        kinds = {}
        for c in pg["cells"]:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        print(f"  page {pg['page'] + 1}: " + ", ".join(f"{v} {k}" for k, v in kinds.items()))


if __name__ == "__main__":
    main()
