"""Read photos of filled-in template sheets (template_sheets.py) into training samples.

No transcribing and almost no labelling: the corner markers say which template page a photo is,
the page is flattened onto the template, every box is lined up with its printed outline, and
what is written in each box is already known. Each page becomes a sample in
~/.handwritten/profiles/<profile>/samples/<id>/ that build_profile.py reads like any other:

  ink.png, mask.png   the page's handwriting only (printed boxes and labels removed)
  segments.json       glyph and word boxes as lines of words, with their baselines
  labels.txt, sheets/ only for words whose pieces couldn't be matched to letters
  tall_*.json         tall brackets and braces
  template.json       scripts, roots, fractions, sums and accents, for the maths renderer
  cells.png           every box as it was read, for checking

Usage: read_template.py [--profile NAME] [--redo] PHOTO_OR_FOLDER...
Photos without template markers are listed and left for preprocess.py.
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_sauvola

from common import PX_PER_MM, file_sha, join_split, load_image_rgb, profile_dir, read_json, samples_dir, \
    ui_font, write_json
from preprocess import collect_images, flatten_lighting
from segment_freeform import build_pieces, suggest_split, write_label_template, write_sheets
import template_sheets as ts

PAGE_W, PAGE_H = ts.px(ts.A4_MM[0]), ts.px(ts.A4_MM[1])
SEARCH_MM = 2.0  # how far a box may sit from where the corner markers put it (paper curl, printer)
EDGE_MM = 0.4  # strip inside each box outline that is never read, so the outline itself isn't ink
MIN_INK_PX = 40  # less than this inside a box means it was left empty
COLOURED_INK = 12.0  # handwriting this much bluer (or redder) than grey: print can be told apart by colour
PRINT_TINT = 7.0  # a stroke whose colour is this close to grey is printed, on a page written in colour
SHIFT_TOLERANCE_MM = 0.6  # a box found further than this from its neighbours' alignment was mis-found


def find_markers(gray):
    """{marker id: 4x2 corners in photo pixels}."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ts.ARUCO_DICT),
                                  cv2.aruco.DetectorParameters())
    found = {}
    for scale in (1.0, 0.5, 0.25):
        img = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        corners, ids, _ = det.detectMarkers(img)
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                found.setdefault(int(i), c.reshape(4, 2) / scale)
        if len(found) >= 4:
            break
    return found


def identify(found, specs):
    """(spec, page index) of the template page the markers belong to, or None."""
    votes = {}
    for mid in found:
        set_index, page, corner = mid // 40, (mid % 40) // 4, mid % 4
        votes.setdefault((set_index, page), []).append(mid)
    if not votes:
        return None
    (set_index, page), mids = max(votes.items(), key=lambda kv: len(kv[1]))
    if len(mids) < 2:
        return None
    for spec in specs:
        if spec["set_index"] == set_index:
            return spec, page, mids
    return None


def page_homography(found, mids, page):
    """Photo -> template pixels, from the corners of every marker of this page that was seen."""
    src, dst = [], []
    side = ts.MARKER_MM
    for mid in mids:
        k = mid % 4
        mx, my = ts.MARKER_AT[k]
        tmpl = [(mx, my), (mx + side, my), (mx + side, my + side), (mx, my + side)]
        src += found[mid].tolist()
        dst += [(ts.px(x), ts.px(y)) for x, y in tmpl]
    src, dst = np.float32(src), np.float32(dst)
    if len(mids) >= 2:
        h, _ = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        return h
    return None


def ink_maps(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    norm = flatten_lighting(gray)
    ink = 1.0 - norm
    sauvola = threshold_sauvola(norm, window_size=51, k=0.2)
    mask = (norm < sauvola) & (ink > 0.18)
    return norm, ink, mask


def register(dark_rows, dark_cols, rect_px, search):
    """Shift (dx, dy) that best lays the printed box outline over the dark lines in the photo."""
    x0, y0, x1, y1 = rect_px
    best, best_score = (0, 0), -1.0
    H, W = dark_rows.shape
    for dy in range(-search, search + 1, 2):
        for dx in range(-search, search + 1, 2):
            a0, a1, b0, b1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
            if a0 < 1 or b0 < 1 or a1 >= W - 1 or b1 >= H - 1:
                continue
            score = (dark_rows[b0, a1] - dark_rows[b0, a0] + dark_rows[b1, a1] - dark_rows[b1, a0]
                     + dark_cols[b1, a0] - dark_cols[b0, a0] + dark_cols[b1, a1] - dark_cols[b0, a1])
            if score > best_score:
                best, best_score = (dx, dy), score
    # Refine to the pixel around the best coarse shift.
    cx, cy = best
    for dy in range(cy - 1, cy + 2):
        for dx in range(cx - 1, cx + 2):
            a0, a1, b0, b1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
            if a0 < 1 or b0 < 1 or a1 >= W - 1 or b1 >= H - 1:
                continue
            score = (dark_rows[b0, a1] - dark_rows[b0, a0] + dark_rows[b1, a1] - dark_rows[b1, a0]
                     + dark_cols[b1, a0] - dark_cols[b0, a0] + dark_cols[b1, a1] - dark_cols[b0, a1])
            if score > best_score:
                best, best_score = (dx, dy), score
    return best


def clean_cell(mask, zone, placeholder, xh, tint=None, printed=None):
    """The handwriting inside one box: outline remnants, printed labels, guide dots and specks
    removed. tint (a page written in colour) drops strokes that are grey like the print; printed
    (the template, aligned to this box) drops strokes that lie almost wholly on printed ink."""
    x0, y0, x1, y1 = zone
    m = mask[y0:y1, x0:x1].copy().astype(np.uint8)
    t = tint[y0:y1, x0:x1] if tint is not None else None
    pr = printed  # already cut to this zone
    if placeholder is not None:
        px0, py0, px1, py1 = (v - o for v, o in zip(placeholder, (x0, y0, x0, y0)))
        pad = ts.px(0.45)
        frame = np.zeros_like(m)
        cv2.rectangle(frame, (px0, py0), (px1, py1), 1, thickness=2 * pad)
        m[frame > 0] = 0
        m[max(0, py0 + pad):max(0, py1 - pad), max(0, px0 + pad):max(0, px1 - pad)] = 0
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    h, w = m.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < 15:
            continue
        # A thin line hugging an edge is the box outline caught by a slightly off alignment.
        thin_h = bh <= ts.px(0.7) and bw > ts.px(3) and (by <= 2 or by + bh >= h - 2)
        thin_v = bw <= ts.px(0.7) and bh > ts.px(3) and (bx <= 2 or bx + bw >= w - 2)
        if thin_h or thin_v:
            continue
        own = cc == i
        on_print = float(pr[own].mean()) if pr is not None and pr.shape == own.shape else 0.0
        # Print caught in the box lies on the template's printed lines and labels; on a page written
        # in colour it is also grey. Small dark marks of the writer's own (dots, primes) are kept.
        if on_print > 0.8 or (t is not None and on_print > 0.25 and float(np.median(t[own])) < PRINT_TINT):
            continue
        keep[i] = True
    return keep[cc]


def smooth_shifts(cells, shifts):
    """Box alignments vary smoothly over a page (paper curl, printer): one far from its neighbours'
    was locked onto the wrong line, so it gets theirs instead."""
    centres = np.array([[(c["rect"][0] + c["rect"][2]) / 2, (c["rect"][1] + c["rect"][3]) / 2] for c in cells])
    sh = np.array(shifts, float)
    out = []
    tol = ts.px(SHIFT_TOLERANCE_MM)
    for k in range(len(cells)):
        d = np.hypot(*(centres - centres[k]).T)
        near = np.argsort(d)[1:9]
        med = np.median(sh[near], axis=0) if len(near) else sh[k]
        out.append(tuple(int(round(v)) for v in (sh[k] if np.abs(sh[k] - med).max() <= tol else med)))
    return out


def runs_over(mask, printed, comp_box, zone, rect, shift):
    """True when a stroke that reaches the edge of the writing area carries on past the box outline
    (or up into the label strip): then the box only holds part of it. Printed lines and labels
    are ignored, so a stroke that merely touches the outline is kept."""
    x0, y0, x1, y1 = comp_box
    dx, dy = shift
    near, far = ts.px(0.35), ts.px(1.4)
    strips = []
    if x0 <= zone[0] + 1:
        strips.append((rect[0] - far, y0, rect[0] - near, y1))
    if x1 >= zone[2] - 1:
        strips.append((rect[2] + near, y0, rect[2] + far, y1))
    if y1 >= zone[3] - 1:
        strips.append((x0, rect[3] + near, x1, rect[3] + far))
    if y0 <= zone[1] + 1:
        strips.append((x0, zone[1] - ts.px(1.2), x1, zone[1] - ts.px(0.1)))
    H, W = mask.shape
    for a0, b0, a1, b1 in strips:
        a0, b0, a1, b1 = max(0, a0), max(0, b0), min(W, a1), min(H, b1)
        if a1 <= a0 or b1 <= b0:
            continue
        ink = mask[b0:b1, a0:a1]
        pr = printed[max(0, b0 - dy):max(0, b1 - dy), max(0, a0 - dx):max(0, a1 - dx)]
        if pr.shape == ink.shape:
            ink = ink & ~pr
        if ink.sum() > 12:
            return True
    return False


def comps_in(cc, zone):
    x0, y0, x1, y1 = zone
    ids = np.unique(cc[y0:y1, x0:x1])
    return [int(i) for i in ids if i]


def bbox_of(stats, comps):
    xs0 = [int(stats[c, 0]) for c in comps]
    ys0 = [int(stats[c, 1]) for c in comps]
    xs1 = [int(stats[c, 0] + stats[c, 2]) for c in comps]
    ys1 = [int(stats[c, 1] + stats[c, 3]) for c in comps]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def touching_pieces(comps, stats, xh):
    """Group a box's strokes into pieces: strokes that cross or touch are one piece, and dots join
    the stroke below them. Unlike word pieces, strokes that only line up (a subscript under a
    superscript, a numerator over its bar) stay apart."""
    def box(c):
        x, y, w, h = (int(v) for v in stats[c, :4])
        return [x, y, x + w, y + h]

    small = [c for c in comps if stats[c, 3] < 0.45 * xh and stats[c, 2] < 0.5 * xh]
    big = [c for c in comps if c not in small]
    parent = {c: c for c in big}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for i, a in enumerate(big):
        ba = box(a)
        for b in big[i + 1:]:
            bb = box(b)
            if ba[0] <= bb[2] + 1 and bb[0] <= ba[2] + 1 and ba[1] <= bb[3] + 1 and bb[1] <= ba[3] + 1:
                parent[find(a)] = find(b)
    groups = {}
    for c in big:
        groups.setdefault(find(c), []).append(c)
    pieces = list(groups.values())
    for c in small:
        bc = box(c)
        cx = (bc[0] + bc[2]) / 2
        best, best_d = None, None
        for p in pieces:
            pb = [min(box(k)[0] for k in p), min(box(k)[1] for k in p), max(box(k)[2] for k in p), max(box(k)[3] for k in p)]
            below = pb[1] >= bc[1] - 0.2 * xh
            d = 0 if pb[0] <= cx <= pb[2] else min(abs(cx - pb[0]), abs(cx - pb[2]))
            if below and d <= 0.35 * xh and pb[1] - bc[3] <= 0.8 * xh and (best_d is None or d < best_d):
                best, best_d = p, d
        if best is not None:
            best.append(c)
        else:
            pieces.append([c])
    out = []
    for p in pieces:
        out.append({"comps": sorted(int(c) for c in p), "bbox": bbox_of(stats, p)})
    return sorted(out, key=lambda p: p["bbox"][0])


def script_parts(pieces, baseline, xh):
    """Split a script box's pieces into the base, the superscript and the subscript, right to left:
    scripts are the trailing pieces that sit clearly above or below the base's line."""
    sup, sub = [], []
    rest = sorted(pieces, key=lambda p: p["bbox"][0])
    while rest:
        p = rest[-1]
        x0, y0, x1, y1 = p["bbox"]
        if y1 <= baseline - 0.45 * xh:
            sup.insert(0, p)
        elif y0 >= baseline - 0.55 * xh and y1 >= baseline + 0.15 * xh:
            sub.insert(0, p)
        else:
            break
        rest.pop()
    return rest, sup, sub


def split_script(text, pieces):
    """[{"text", "comps"}] for a script: one unit per character when the pieces match, else one
    unit for the whole script (kept for measuring, not as glyphs)."""
    if not pieces:
        return []
    if len(pieces) == len(text):
        return [{"text": ch, "comps": p["comps"], "bbox": p["bbox"]} for ch, p in zip(text, pieces)]
    comps = sorted(c for p in pieces for c in p["comps"])
    xs = [p["bbox"] for p in pieces]
    box = [min(b[0] for b in xs), min(b[1] for b in xs), max(b[2] for b in xs), max(b[3] for b in xs)]
    return [{"text": text, "comps": comps, "bbox": box, "whole": True}]


def read_photo(path, specs, out_root, xh, redo=False):
    sha = file_sha(path)
    sid = sha[:12]
    out = out_root / sid
    if (out / "template.json").exists() and not redo:
        return sid, read_json(out / "meta.json"), "skip"
    rgb = load_image_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found = find_markers(gray)
    ident = identify(found, specs)
    if ident is None:
        return sid, {"markers": len(found)}, "not_template"
    spec, page_index, mids = ident
    pages = ts.layout(spec)
    if page_index >= len(pages):
        return sid, {"markers": len(found)}, "not_template"
    page = pages[page_index]
    hmat = page_homography(found, mids, page_index)
    if hmat is None:
        return sid, {"markers": len(found)}, "not_template"
    warped = cv2.warpPerspective(rgb, hmat, (PAGE_W, PAGE_H), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    norm, ink, mask = ink_maps(warped)
    w32 = warped.astype(np.float32)
    tint = np.abs(w32[..., 2] - (w32[..., 0] + w32[..., 1]) / 2)  # blue (or red) ink stands out from grey print

    # Everything printed on the page, a little thickened, so it is never taken for handwriting.
    printed = np.array(ts.draw_page(spec, page, len(pages), spec["set_index"])) < 190  # labels and outlines, not the faint guide dots
    printed = cv2.dilate(printed.astype(np.uint8), np.ones((ts.px(0.5) * 2 + 1,) * 2, np.uint8)) > 0
    handwriting = mask & ~printed
    coloured = handwriting.any() and float(np.median(tint[handwriting])) > COLOURED_INK

    # Printed box outlines are darker than paper: find each box where it really is.
    dark = cv2.blur((norm < 0.82).astype(np.float32), (3, 3))
    dark_rows = np.cumsum(dark, axis=1)
    dark_cols = np.cumsum(dark, axis=0)
    search = ts.px(SEARCH_MM)
    keep = np.zeros(mask.shape, bool)
    placed = []
    edge = ts.px(EDGE_MM)
    rects = [[ts.px(v) for v in cell["rect"]] for cell in page["cells"]]
    shifts = smooth_shifts(page["cells"], [register(dark_rows, dark_cols, r, search) for r in rects])
    for cell, rect, (dx, dy) in zip(page["cells"], rects, shifts):
        zx0, zy0, zx1, zy1 = (ts.px(v) for v in cell["zone"])
        zone = [zx0 + dx + edge, zy0 + dy + int(edge / 2), zx1 + dx - edge, zy1 + dy - edge]
        ph = [ts.px(v) + o for v, o in zip(cell["placeholder"], (dx, dy, dx, dy))] if "placeholder" in cell else None
        a0, b0 = zone[0] - dx, zone[1] - dy
        pr_crop = printed[max(0, b0):b0 + zone[3] - zone[1], max(0, a0):a0 + zone[2] - zone[0]]
        cell_keep = clean_cell(mask, zone, ph, xh, tint if coloured else None, pr_crop)
        keep[zone[1]:zone[3], zone[0]:zone[2]] |= cell_keep
        base = ts.px(cell["baseline"]) + dy if "baseline" in cell else None
        placed.append({"cell": cell, "zone": zone, "baseline": base, "placeholder": ph, "shift": [dx, dy],
                       "rect": [rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy]})

    mask8 = keep.astype(np.uint8) * 255
    hi = float(np.percentile(ink[keep], 95)) if keep.any() else 1.0
    ink8 = (np.clip(ink / max(hi, 0.2), 0, 1) * 255 * cv2.dilate(mask8, np.ones((3, 3), np.uint8)).astype(bool)).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats((mask8 > 0).astype(np.uint8), connectivity=8)
    # Ink outside the boxes, for telling a stroke that crosses an outline from one that stops at it.
    outside = mask & (tint > PRINT_TINT) if coloured else mask

    if out.exists() and redo:
        for stale in ("segments.json", "labels.txt", "tall_labels.json", "tall_candidates.json", "template.json"):
            (out / stale).unlink(missing_ok=True)
        shutil.rmtree(out / "sheets", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out / f"original{path.suffix.lower()}")
    cv2.imwrite(str(out / "ink.png"), ink8)
    cv2.imwrite(str(out / "mask.png"), mask8)

    core_h = xh / 0.55
    result = {"sample": sid, "transcript_sha": None, "n_components": int(n), "lines": [], "words": {},
              "status": "ok", "template": True}
    extras = {"template": spec["id"], "page": page_index, "xh": xh, "scripts": [], "radicals": [],
              "fractions": [], "bigops": [], "accents": []}
    tall_cands, tall_labels = [], {}
    rows = {}
    empty, clipped, review = [], [], []
    wcount = 0
    for pc in placed:
        cell, zone, base = pc["cell"], pc["zone"], pc["baseline"]
        comps = comps_in(cc, zone)
        area = int(sum(stats[c, cv2.CC_STAT_AREA] for c in comps))
        if area < MIN_INK_PX:
            review.append((cell, zone, "empty"))
            empty.append(cell["id"])
            continue
        # Writing that runs over the box outline was cut by it: the part inside is not the whole symbol.
        over = [c for c in comps if runs_over(outside, printed, bbox_of(stats, [c]), zone, pc["rect"], pc["shift"])]
        if over and cell["kind"] != "tall":
            review.append((cell, zone, "edge"))
            clipped.append(cell["id"])
            continue
        review.append((cell, zone, "ok"))
        kind = cell["kind"]
        box = bbox_of(stats, comps)
        if kind in ("glyph", "word"):
            text = cell["text"]
            if kind == "glyph":
                pieces = [{"comps": sorted(comps), "bbox": box}]
            else:
                pieces = build_pieces(comps, stats, core_h, base - 0.5 * xh)
            wcount += 1
            wid = f"w{wcount:03d}"
            word = {"line": None, "index": 0, "text": text, "pieces": pieces, "bbox": box, "cell": cell["id"]}
            for k, p in enumerate(pieces):
                p["pid"] = f"{sid}-{wid}-p{k + 1}"
                p["rel_w"] = (p["bbox"][2] - p["bbox"][0]) / core_h
            if len(pieces) == 1:
                word["status"], word["split"] = "auto", text
            elif len(pieces) == len(text):
                word["status"], word["split"] = "auto", join_split(list(text))
            else:
                word["status"], word["split"] = "needs_label", None
                word["suggested"] = suggest_split(text, pieces, {})
            result["words"][wid] = word
            rows.setdefault(round(cell["baseline"], 1), []).append((wid, base, (zone[0] + zone[2]) / 2))
        elif kind == "tall":
            num = len(tall_cands) + 1
            tall_cands.append({"n": num, "bbox": box, "comps": sorted(comps)})
            tall_labels[str(num)] = cell["text"]
        elif kind == "script":
            pieces = touching_pieces(comps, stats, xh)
            rest, sup, sub = script_parts(pieces, base, xh)
            entry = {"cell": cell["id"], "base": cell["base"], "baseline": base,
                     "base_bbox": bbox_of(stats, [c for p in rest for c in p["comps"]]) if rest else None,
                     "sup": split_script(cell.get("sup", ""), sup) if cell.get("sup") else [],
                     "sub": split_script(cell.get("sub", ""), sub) if cell.get("sub") else []}
            entry["ok"] = bool(rest) and (not cell.get("sup") or bool(sup)) and (not cell.get("sub") or bool(sub))
            extras["scripts"].append(entry)
        elif kind == "radical":
            extras["radicals"].append({"cell": cell["id"], "comps": sorted(comps), "bbox": box,
                                       "placeholder": pc["placeholder"], "baseline": base})
        elif kind == "fraction":
            extras["fractions"].append({"cell": cell["id"], "num": cell["num"], "den": cell["den"],
                                        "pieces": touching_pieces(comps, stats, xh)})
        elif kind == "bigop":
            extras["bigops"].append({"cell": cell["id"], "op": cell["op"], "lower": cell.get("lower"),
                                     "upper": cell.get("upper"), "comps": sorted(comps), "bbox": box,
                                     "pieces": touching_pieces(comps, stats, xh)})
        elif kind == "accent":
            extras["accents"].append({"cell": cell["id"], "base": cell["base"], "accent": cell["accent"],
                                      "baseline": base, "pieces": touching_pieces(comps, stats, xh)})

    # Glyph and word boxes in one printed row share a baseline: one line per row.
    for li, (_, members) in enumerate(sorted(rows.items())):
        members.sort(key=lambda m: m[2])
        xs = np.array([m[2] for m in members], float)
        ys = np.array([m[1] for m in members], float)
        b = float(np.polyfit(xs, ys, 1)[0]) if len(members) >= 3 and np.ptp(xs) > 100 else 0.0
        a = float(np.median(ys - b * xs))
        line = {"index": li, "y": int(np.median(ys)), "text": " ".join(result["words"][m[0]]["text"] for m in members),
                "words": [m[0] for m in members], "status": "ok", "core_h": core_h,
                "baseline": [a, b], "xh": xh}
        for k, m in enumerate(members):
            result["words"][m[0]]["line"], result["words"][m[0]]["index"] = li, k
        result["lines"].append(line)

    write_json(out / "segments.json", result)
    write_json(out / "template.json", extras)
    if tall_cands:
        write_json(out / "tall_candidates.json", {"n_components": int(n), "candidates": tall_cands})
        write_json(out / "tall_labels.json", tall_labels)
    sheets, todo = write_sheets(out, result, cc)
    if todo:
        write_label_template(out, todo)
    else:
        shutil.rmtree(out / "sheets", ignore_errors=True)
    write_review(out, review, ink8)
    meta = {"id": sid, "sha256": sha, "source": str(path), "kind": "template", "template": spec["id"],
            "template_page": page_index + 1, "markers": len(mids), "scale_source": "template_markers",
            "size_px": [PAGE_W, PAGE_H], "photo_px": list(rgb.shape[1::-1]), "empty_cells": empty,
            "over_edge_cells": clipped,
            "needs_label": len(todo), "x_height_px": xh}
    write_json(out / "meta.json", meta)
    return sid, meta, "ok"


def write_review(out, review, ink8):
    """cells.png: every box as read (red = left empty, orange = ran over the box edge and was not
    used), for a quick look."""
    thumbs = []
    f = ts.font(ts.FONTS, 20)
    colours = {"ok": (0, 0, 0), "empty": (220, 30, 30), "edge": (235, 130, 0)}
    for cell, zone, state in review:
        x0, y0, x1, y1 = zone
        crop = 255 - ink8[y0:y1, x0:x1]
        scale = 110 / max(1, crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 110), interpolation=cv2.INTER_AREA)
        tile = Image.new("RGB", (max(crop.shape[1], 70) + 8, 140), "white")
        tile.paste(Image.fromarray(crop).convert("RGB"), (4, 4))
        d = ImageDraw.Draw(tile)
        name = cell.get("text") or cell.get("label") or cell.get("base") or cell.get("op") or cell.get("num") or "√"
        if cell["kind"] == "script":
            name = cell["base"] + ("^" + cell["sup"] if "sup" in cell else "") + ("_" + cell["sub"] if "sub" in cell else "")
        d.text((4, 116), name[:12], fill=colours[state], font=f)
        d.rectangle([0, 0, tile.width - 1, tile.height - 1], outline=(220, 220, 220))
        thumbs.append(tile)
    width, rows, row, w = 1800, [], [], 0
    for t in thumbs:
        if w + t.width > width:
            rows.append(row)
            row, w = [], 0
        row.append(t)
        w += t.width
    if row:
        rows.append(row)
    sheet = Image.new("RGB", (width, 140 * len(rows)), "white")
    for r, row in enumerate(rows):
        x = 0
        for t in row:
            sheet.paste(t, (x, r * 140))
            x += t.width
    sheet.save(out / "cells.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--redo", action="store_true", help="read photos again that were read before")
    args = ap.parse_args()
    specs = [read_json(p) for p in sorted(ts.SPEC_DIR.glob("*.json"))]
    prof = read_json(profile_dir(args.profile) / "profile.json", {})
    xh = (prof.get("style") or {}).get("x_height_px") or 2.6 * PX_PER_MM
    out_root = samples_dir(args.profile)
    out_root.mkdir(parents=True, exist_ok=True)
    others = []
    for path in collect_images(args.paths):
        sid, meta, status = read_photo(path, specs, out_root, xh, args.redo)
        if status == "not_template":
            others.append(path)
            print(f"OTHER   {path.name}: no template markers ({meta['markers']} found); use preprocess.py for it")
            continue
        if status == "skip":
            print(f"SKIP    {path.name} -> {sid}: already read (use --redo to read it again)")
            continue
        boxes_total = len(ts.layout(next(s for s in specs if s["id"] == meta["template"]))[meta["template_page"] - 1]["cells"])
        print(f"OK      {path.name} -> {sid}: {meta['template']} page {meta['template_page']}, "
              f"{boxes_total - len(meta['empty_cells'])}/{boxes_total} boxes written, "
              f"{meta['markers']}/4 corner markers")
        if meta["over_edge_cells"]:
            print(f"        {len(meta['over_edge_cells'])} boxes ran over their outline and were not used "
                  f"(orange in cells.png): {' '.join(meta['over_edge_cells'][:12])}")
        print(f"        check: {out_root / sid / 'cells.png'}")
        if meta["needs_label"]:
            print(f"        {meta['needs_label']} words need labels: {out_root / sid / 'sheets'}  (edit labels.txt)")
    if others:
        print(f"{len(others)} photo(s) are not template pages.")


if __name__ == "__main__":
    main()
