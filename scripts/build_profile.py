"""Build the handwriting profile from every labelled sample.

Reads each sample's segments.json and labels.txt, cuts out every labelled
character and ligature, aligns them on a measured baseline, drops implausible
cut-outs, measures the writing style, and writes:

  profile.json            glyphs, ligatures, style statistics, coverage
  glyphs/<hex>/<id>.png   one image per character variant (255 = ink, 300 DPI)
  ligatures/<hex_hex>/    joined runs such as "th"
  coverage.txt            the coverage report
  specimen_NN.png         every variant on its baseline, numbered, for review
  scripts/, radicals/ …   maths pieces from template pages (template_extras.py)
  preview.png             a test sentence written with the profile

The profile is rebuilt from scratch each time, so adding pages or rejecting
glyphs is always: change the inputs, run this again.

Usage: build_profile.py [--profile NAME] [--reject 12,57,...]
"""

import argparse
import math
import random
import shutil
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

from template_extras import collect_extras, report_lines
from common import (DESCENDERS, GREEK_DESCENDERS, DIGITS, LOWER, MATH, PUNCT, PX_PER_MM, UPPER, X_HEIGHT_ONLY, apply_settings, char_key,
                    mm, parse_split, profile_dir, read_json, read_settings, samples_dir, ui_font, write_json)

ASCENDER_TEXT = set("bdhklt")
NOT_ON_BASELINE = set("'\"-^*~`()[]{}/|")  # marks that float above or reach below the baseline
INK_BLUE = (25, 45, 140)
PANGRAMS = ["The quick brown fox jumps over the lazy dog.",
            "Sphinx of black quartz, judge my vow!",
            "Pack my box with five dozen liquor jugs: 0123456789"]


# ---------------------------------------------------------------- labels

def parse_labels(path):
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) >= 3:
            out[parts[0]] = parts[2]
        elif len(parts) == 2:  # "<id> ?" is accepted too
            out[parts[0]] = parts[1]
    return out


def units_for_word(word, split):
    """Turn a word and its split ("th|e", "i||t") into [(text, [component ids])]; None if invalid."""
    if split is None or split == "?":
        return None
    parts = parse_split(split)
    if len(parts) != len(word["pieces"]) or "".join(parts) != word["text"] or not any(parts):
        return None
    units = []
    pending = []  # comps of empty leading parts, given to the next real unit
    for part, piece in zip(parts, word["pieces"]):
        if part:
            units.append([part, pending + list(piece["comps"])])
            pending = []
        elif units:
            units[-1][1] += list(piece["comps"])
        else:
            pending += list(piece["comps"])
    return [(t, c) for t, c in units]


# ---------------------------------------------------------------- geometry

def fit_baseline(points, fallback_y):
    """Robust straight-line fit y = a + b*x through glyph bottoms. Returns (a, b)."""
    if not points:
        return float(fallback_y), 0.0
    xs = np.array([p[0] for p in points], np.float64)
    ys = np.array([p[1] for p in points], np.float64)
    if len(points) < 3 or np.ptp(xs) < 100:
        return float(np.median(ys)), 0.0
    keep = np.ones(len(xs), bool)
    a, b = float(np.median(ys)), 0.0
    for _ in range(3):
        b, a = np.polyfit(xs[keep], ys[keep], 1)
        res = np.abs(ys - (a + b * xs))
        thr = max(3.0, 2.5 * float(np.median(res[keep])))
        new_keep = res <= thr
        if new_keep.sum() < 3 or (new_keep == keep).all():
            break
        keep = new_keep
    return float(a), float(b)


def line_slant(mask, y0, y1, x0, x1):
    """Writing slant in degrees (positive = leaning right) from a shear search on one line."""
    crop = mask[max(0, y0):y1, max(0, x0):x1].astype(np.float32)
    if crop.size == 0 or crop.sum() < 50:
        return None
    crop = cv2.resize(crop, (max(1, crop.shape[1] // 2), max(1, crop.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    h, w = crop.shape
    pad = int(h * math.tan(math.radians(35))) + 2
    canvas = np.zeros((h, w + 2 * pad), np.float32)
    canvas[:, pad:pad + w] = crop
    best, best_score = 0.0, -1.0
    for deg in np.arange(-30, 30.1, 1.5):
        t = math.tan(math.radians(deg))
        m = np.float32([[1, t, -t * h / 2], [0, 1, 0]])
        sheared = cv2.warpAffine(canvas, m, (canvas.shape[1], h))
        score = float((sheared.sum(axis=0) ** 2).sum())
        if score > best_score:
            best, best_score = float(deg), score
    return best


def stroke_width_px(mask):
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 100:
        return None
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    skel = skeletonize(m > 0)
    vals = dt[skel]
    return float(2 * np.median(vals)) if vals.size else None


def stat(values):
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not v:
        return None
    return {"median": round(float(np.median(v)), 3), "mean": round(float(np.mean(v)), 3),
            "std": round(float(np.std(v)), 3), "n": len(v)}


# ---------------------------------------------------------------- extraction

def extract_sample(sdir, rejected, counters):
    seg = read_json(sdir / "segments.json")
    if not seg or seg.get("status") != "ok":
        counters["samples_not_ready"] += 1
        return [], {}
    labels = parse_labels(sdir / "labels.txt")
    mask = cv2.imread(str(sdir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    ink = cv2.imread(str(sdir / "ink.png"), cv2.IMREAD_GRAYSCALE)
    n, cc, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n != seg["n_components"]:
        print(f"WARNING {sdir.name}: page changed since segmentation; re-run segment_freeform.py --force")
        counters["samples_not_ready"] += 1
        return [], {}

    def bbox(comps):
        xs0 = [int(stats[c, 0]) for c in comps]
        ys0 = [int(stats[c, 1]) for c in comps]
        xs1 = [int(stats[c, 0] + stats[c, 2]) for c in comps]
        ys1 = [int(stats[c, 1] + stats[c, 3]) for c in comps]
        return min(xs0), min(ys0), max(xs1), max(ys1)

    meta = read_json(sdir / "meta.json", {})
    page_stats = {"id": seg["sample"], "lines": [], "letter_gaps": [], "word_gaps": [], "slants": [],
                  "stroke_px": stroke_width_px(mask), "width_px": mask.shape[1],
                  "trusted": meta.get("scale_source") in TRUSTED_SCALE,
                  # Template pages have boxes, not lines of writing: no spacing or margins to learn.
                  "template": meta.get("kind") == "template"}
    units = []
    for line in seg["lines"]:
        if line["status"] != "ok":
            continue
        line_units = []
        prev_word_x1 = None
        for wid in line["words"]:
            word = seg["words"][wid]
            # labels.txt wins, so Claude can also correct or skip words that were labelled automatically.
            split = labels.get(wid, word["split"] if word["status"] == "auto" else None)
            if word["status"] == "needs_label":
                if wid not in labels:
                    counters["words_unlabelled"] += 1
                elif split == "?":
                    counters["words_skipped"] += 1
            wu = units_for_word(word, split) if word["status"] != "skipped" else None
            if wid in labels and split != "?" and wu is None:
                counters["labels_invalid"] += 1
                print(f"WARNING {sdir.name} {wid}: label '{split}' does not fit "
                      f"\"{word['text']}\" with {len(word['pieces'])} pieces")
            if wu is None:
                prev_word_x1 = None
                continue
            counters["words_used"] += 1
            boxes = [bbox(c) for _, c in wu]
            for k, ((t, c), b) in enumerate(zip(wu, boxes)):
                pid = f"{seg['sample']}-{wid}-{k + 1}"
                line_units.append({"text": t, "comps": c, "bbox": b, "pid": pid, "word": wid})
            page_stats["letter_gaps"] += [boxes[i + 1][0] - boxes[i][2] for i in range(len(boxes) - 1)]
            wx0 = min(b[0] for b in boxes)
            if prev_word_x1 is not None:
                page_stats["word_gaps"].append(max(0, wx0 - prev_word_x1))
            prev_word_x1 = max(b[2] for b in boxes)
        if not line_units:
            continue

        core_h = line.get("core_h", 40.0)
        if "baseline" in line:  # known from a template's printed baseline
            a, b = line["baseline"]
        else:
            pts = [((u["bbox"][0] + u["bbox"][2]) / 2, u["bbox"][3]) for u in line_units
                   if not (set(u["text"]) & (DESCENDERS | NOT_ON_BASELINE))]
            a, b = fit_baseline(pts, line["y"] + 0.5 * core_h)
        tops = []
        for u in line_units:
            if set(u["text"]) <= X_HEIGHT_ONLY:
                cx = (u["bbox"][0] + u["bbox"][2]) / 2
                tops.append(a + b * cx - u["bbox"][1])
        xh = line.get("xh") or (float(np.median(tops)) if tops else 0.55 * core_h)
        x_min = min(u["bbox"][0] for u in line_units)
        x_max = max(u["bbox"][2] for u in line_units)
        y_top = int(a + b * (x_min + x_max) / 2 - 2.5 * xh)
        y_bot = int(a + b * (x_min + x_max) / 2 + 1.5 * xh)
        page_stats["slants"].append(line_slant(mask > 0, y_top, y_bot, x_min, x_max))
        page_stats["lines"].append({"a": a, "b": b, "xh": xh, "x_min": x_min, "x_max": x_max})
        for u in line_units:
            if u["pid"] in rejected:
                counters["rejected"] += 1
                continue
            x0, y0, x1, y1 = u["bbox"]
            cx = (x0 + x1) / 2
            base = a + b * cx
            u.update({"xh": xh, "above": (base - y0) / xh, "below": (y1 - base) / xh,
                      "w_rel": (x1 - x0) / xh, "baseline_abs": base, "sample": seg["sample"],
                      "_cc": cc, "_ink": ink})
            units.append(u)
    page_stats["xh"] = float(np.median([l["xh"] for l in page_stats["lines"]])) if page_stats["lines"] else None
    return units, page_stats


def settle_template_baselines(units, template_ids):
    """On template pages the baseline is the printed dotted line, but people write across it rather
    than on it. Letters that sit on the line (no tail) are put where their ink sits, and the
    other symbols on the page move by how far its letters sat from the dotted line."""
    sits = lambda t: len(t) == 1 and t.isalnum() and t not in DESCENDERS and t not in GREEK_DESCENDERS
    by_page = {}
    for u in units:
        if u["sample"] in template_ids and sits(u["text"]):
            by_page.setdefault(u["sample"], []).append(u["bbox"][3] - u["baseline_abs"])
    every = [o for v in by_page.values() for o in v]
    overall = float(np.median(every)) if len(every) >= 5 else 0.0
    for u in units:
        if u["sample"] not in template_ids or u["word"] is None:
            continue
        offs = by_page.get(u["sample"], [])
        if sits(u["text"]):
            base = float(u["bbox"][3])
        elif len(u["text"]) == 1:
            base = u["baseline_abs"] + (float(np.median(offs)) if len(offs) >= 5 else overall)
        else:
            continue  # joined runs from word boxes keep their line
        x0, y0, x1, y1 = u["bbox"]
        u["baseline_abs"] = base
        u["above"], u["below"] = (base - y0) / u["xh"], (y1 - base) / u["xh"]


def plausible(u):
    """Reject only cut-outs no handwriting could produce; writers differ a lot (long looped tails,
    wide letters), so anything merely unusual is left to the per-letter outlier check."""
    t = u["text"]
    letters = sum(ch.isalnum() for ch in t)
    if u["above"] > 4.5 or u["below"] > 4.0:
        return "too tall (probably two lines touching)"
    wide_limit = 2.0 * max(len(t), 1) + (1.2 if set(t) & set("mwMW") else 0) + 2.5
    if u["w_rel"] > wide_limit:
        return "too wide for its label"
    if letters and (u["above"] + u["below"]) < 0.45 and not set(t) & set(".,-'\"_"):
        return "too small (a fragment)"
    return None


def drop_outliers(variants):
    """Drop variants far from the others in height, depth or width (robust z > 4).

    Every kept variant gets an "oddness" score so the review sheet can show the least
    typical ones first; extra separate strokes (a stray dot or tick) count as odd too."""
    for v in variants:
        v["odd"] = 0.0
    if len(variants) < 4:
        return variants, []
    keep, dropped = [], []
    feats = np.array([[v["above"], v["below"], v["w_rel"]] for v in variants])
    med = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - med), axis=0) * 1.4826 + np.array([0.08, 0.08, 0.08])
    z = np.abs(feats - med) / mad
    usual_strokes = np.median([len(v["comps"]) for v in variants])
    for v, zz in zip(variants, z):
        v["odd"] = float(zz.max()) + 1.5 * max(0, len(v["comps"]) - usual_strokes)
        (dropped if zz.max() > 4 else keep).append(v)
    return keep, dropped


def collect_tall(sroot, pdir, rejected):
    """The writer's tall brackets and braces (labelled with tall_glyphs.py), as {char: [meta]}."""
    tall = {}
    for sdir in sorted(p for p in sroot.iterdir() if p.is_dir()):
        labels = read_json(sdir / "tall_labels.json")
        cand = read_json(sdir / "tall_candidates.json")
        if not labels or not cand:
            continue
        mask = cv2.imread(str(sdir / "mask.png"), cv2.IMREAD_GRAYSCALE)
        ink = cv2.imread(str(sdir / "ink.png"), cv2.IMREAD_GRAYSCALE)
        n, cc = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
        if n != cand["n_components"]:
            print(f"WARNING {sdir.name}: page changed since tall_glyphs.py; run it again")
            continue
        by_n = {str(f["n"]): f for f in cand["candidates"]}
        for num, ch in labels.items():
            gid = f"{sdir.name}-T{num}"
            if gid in rejected:
                continue
            f = by_n[num]
            x0, y0, x1, y1 = f["bbox"]
            own = np.isin(cc[y0:y1, x0:x1], f["comps"])
            own = cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            crop = np.where(own, ink[y0:y1, x0:x1], 0).astype(np.uint8)
            out = pdir / "tall" / char_key(ch)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{gid}.png"
            cv2.imwrite(str(path), crop)
            tall.setdefault(ch, []).append({"id": gid, "file": str(path.relative_to(pdir)),
                                            "w": int(crop.shape[1]), "h": int(crop.shape[0])})
    return tall


def save_unit(u, out_dir, scale=1.0):
    """Write the cut-out (only this unit's own strokes) and return its metadata.

    scale resizes cut-outs from pages whose physical size was only guessed."""
    pad = 6
    x0, y0, x1, y1 = u["bbox"]
    ink, cc = u["_ink"], u["_cc"]
    y0p, x0p = max(0, y0 - pad), max(0, x0 - pad)
    y1p, x1p = min(ink.shape[0], y1 + pad), min(ink.shape[1], x1 + pad)
    own = np.isin(cc[y0p:y1p, x0p:x1p], u["comps"])
    own = cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    crop = np.where(own, ink[y0p:y1p, x0p:x1p], 0).astype(np.uint8)
    baseline = float(u["baseline_abs"] - y0p)
    if abs(scale - 1) > 0.02:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        baseline *= scale
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{u['pid']}.png"
    cv2.imwrite(str(path), crop)
    return {"id": u["pid"], "file": str(path.relative_to(out_dir.parent.parent)), "w": int(crop.shape[1]),
            "h": int(crop.shape[0]), "baseline": round(baseline, 1),
            "x_height_px": round(float(u["xh"]) * scale, 1), "odd": round(u.get("odd", 0.0), 2)}


TRUSTED_SCALE = {"page_edges", "frame_is_page", "template_markers"}


def sample_scales(pages):
    """Per-sample resize factors: pages with a guessed scale are matched to the trusted pages' x-height."""
    trusted = [p["xh"] for p in pages if p["trusted"] and p["xh"]]
    if not trusted:
        return {p["id"]: 1.0 for p in pages}
    target = float(np.median(trusted))
    return {p["id"]: 1.0 if p["trusted"] or not p["xh"] else float(np.clip(target / p["xh"], 0.5, 2.0))
            for p in pages}


# ---------------------------------------------------------------- review images

def glyph_image(prof_dir, meta, scale):
    img = cv2.imread(str(prof_dir / meta["file"]), cv2.IMREAD_GRAYSCALE)
    if scale != 1:
        img = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    return img


def paste_ink(canvas, img, x, y, color):
    """Alpha-blend a 0..255 ink image onto an RGB float canvas at (x, y)."""
    h, w = img.shape
    H, W = canvas.shape[:2]
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(W, x + w), min(H, y + h)
    a = (img[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)].astype(np.float32) / 255)[..., None]
    region = canvas[dy0:dy1, dx0:dx1]
    canvas[dy0:dy1, dx0:dx1] = region * (1 - a) + np.array(color, np.float32) * a


def write_specimens(prof_dir, profile):
    for old in prof_dir.glob("specimen_*.png"):
        old.unlink()
    tracked = LOWER + UPPER + DIGITS + PUNCT + MATH
    rows = [(ch, profile["glyphs"][ch]) for ch in tracked if ch in profile["glyphs"]]
    rows += [(ch, v) for ch, v in sorted(profile["glyphs"].items()) if ch not in tracked]  # ∈, ⊆, ‖ and other symbols
    rows += [(t, v) for t, v in sorted(profile["ligatures"].items(), key=lambda kv: -len(kv[1]))]
    # Maths pieces from template pages, marked so they read apart from the glyphs.
    rows += [(f"s {ch}", v) for ch, v in sorted((profile.get("scripts") or {}).items())]
    rows += [(f"acc", v) for ch, v in sorted((profile.get("accents") or {}).items())]
    if profile.get("radicals"):
        rows.append(("√", profile["radicals"]))
    rows += [(f"big{ch}", v) for ch, v in sorted((profile.get("bigops") or {}).items())]
    xh = profile["style"]["x_height_px"]
    scale = 34.0 / max(xh, 1)  # draw every glyph with a 34 px x-height
    row_h = int(34 * 5.2)
    base_y = int(34 * 3.0)
    per_sheet, width, max_variants = 14, 1500, 12
    f_label, f_num = ui_font(30), ui_font(15)
    mapping, paths, number = {}, [], 0
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        canvas = np.full((50 + row_h * len(chunk), width, 3), 255, np.float32)
        nums = []
        for r, (label, variants) in enumerate(chunk):
            y = 50 + r * row_h
            canvas[y + base_y, 120:width - 20] = (205, 215, 235)
            x = 130
            for v in sorted(variants, key=lambda v: -v.get("odd", 0))[:max_variants]:
                img = glyph_image(prof_dir, v, scale)
                paste_ink(canvas, img, x, int(y + base_y - v["baseline"] * scale), (0, 0, 0))
                number += 1
                mapping[str(number)] = v["id"]
                nums.append((x + img.shape[1] // 2, y + row_h - 26, number))
                x += img.shape[1] + 18
                if x > width - 80:
                    break
        im = Image.fromarray(canvas.astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((20, 12), f"Specimen {s // per_sheet + 1}: least typical variants first, each on its measured "
                         f"baseline (blue line). Reject bad ones by number.", fill=(90, 90, 90), font=ui_font(18))
        for r, (label, variants) in enumerate(chunk):
            y = 50 + r * row_h
            d.line([(0, y), (width, y)], fill=(225, 225, 225), width=1)
            d.text((20, y + base_y - 34), label, fill=(200, 30, 30), font=f_label)
            if len(variants) > max_variants:
                d.text((width - 110, y + base_y - 10), f"+{len(variants) - max_variants} typical",
                       fill=(120, 120, 120), font=f_num)
        for cx, ny, num in nums:
            d.text((cx - 8, ny), str(num), fill=(120, 120, 120), font=f_num)
        path = prof_dir / f"specimen_{s // per_sheet + 1:02d}.png"
        im.save(path)
        paths.append(path)
    write_json(prof_dir / "specimen_map.json", mapping)
    return paths


def shape_descriptor(prof_dir, meta):
    """The glyph's ink in a fixed band around its baseline, shrunk to 20x28 and normalised."""
    img = cv2.imread(str(prof_dir / meta["file"]), cv2.IMREAD_GRAYSCALE)
    base, xh = meta["baseline"], meta["x_height_px"]
    top, bot = int(base - 2.2 * xh), int(base + 1.4 * xh)
    pad_t, pad_b = max(0, -top), max(0, bot - img.shape[0])
    img = cv2.copyMakeBorder(img, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=0)[top + pad_t:bot + pad_t]
    cols = np.nonzero(img.max(axis=0) > 40)[0]
    if len(cols) == 0:
        return None
    d = cv2.resize(img[:, cols[0]:cols[-1] + 1], (20, 28), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.GaussianBlur(d, (3, 3), 1).ravel()
    return d / (np.linalg.norm(d) + 1e-6)


def write_suspects(prof_dir, profile):
    """Variants that look more like a different character than like the others with their own label.

    Mislabelled cut-outs (an "e" filed as "r") look right in size and width, so the specimen sheets
    can miss them; this compares shapes instead. Most hits are harmless lookalikes (0/o, l/1)."""
    for old in prof_dir.glob("suspects_*.png"):
        old.unlink()
    descs = {k: [(m, shape_descriptor(prof_dir, m)) for m in v] for k, v in profile["glyphs"].items() if k.isalnum()}
    centroids = {}
    for k, v in descs.items():
        arr = np.array([d for _, d in v if d is not None])
        if len(arr) >= 3:
            c = np.median(arr, axis=0)
            centroids[k] = c / np.linalg.norm(c)
    lookalikes = {frozenset(p) for p in ("0o", "0O", "oO", "1l", "1I", "lI", "il", "i1", "iI", "sS", "cC", "xX", "vV",
                                         "wW", "uU", "zZ", "pP", "kK")}
    flagged = []
    for k, v in descs.items():
        if k not in centroids:
            continue
        for m, d in v:
            if d is None:
                continue
            sims = {o: float(d @ c) for o, c in centroids.items()}
            best = max(sims, key=sims.get)
            if best != k and sims[best] - sims[k] > 0.04 and frozenset((k, best)) not in lookalikes:
                flagged.append((k, best, m))
    tiles = []
    f = ui_font(18)
    for k, best, m in flagged:
        img = glyph_image(prof_dir, m, 1)
        tile = Image.new("RGB", (150, 170), "white")
        crop = Image.fromarray(255 - img[:130, :140])
        tile.paste(crop, (5, 32))
        ImageDraw.Draw(tile).text((5, 6), f"{len(tiles) + 1}: {k} (like {best})", fill=(200, 30, 30), font=f)
        tiles.append((tile, m["id"]))
    mapping, paths = {}, []
    for s in range(0, len(tiles), 40):
        chunk = tiles[s:s + 40]
        sheet = Image.new("RGB", (150 * 8, 170 * ((len(chunk) + 7) // 8)), "white")
        for i, (tile, gid) in enumerate(chunk):
            sheet.paste(tile, (150 * (i % 8), 170 * (i // 8)))
            mapping[f"s{s + i + 1}"] = gid
        path = prof_dir / f"suspects_{s // 40 + 1:02d}.png"
        sheet.save(path)
        paths.append(path)
    return paths, mapping


def write_preview(prof_dir, profile):
    """A plain test render: variants chosen at random, aligned on the baseline, measured spacing
    (or the user's settings), words wrapped to the width of an A4 text block."""
    style = apply_settings(profile["style"], read_settings(profile["name"]))
    rng = random.Random(7)
    xh = style["x_height_px"]
    gap = PX_PER_MM * (style["letter_gap_mm"] or {"median": 0.4})["median"]
    space = PX_PER_MM * (style["word_gap_mm"] or {"median": 3.0})["median"]
    pitch = PX_PER_MM * (style["line_pitch_mm"] or {"median": 9.0})["median"]
    margin = 0.5 * space
    width = int(170 * PX_PER_MM)

    def layout_word(word):
        """[(glyph meta or None for a missing character, x offset)], total width."""
        parts, x, i = [], 0.0, 0
        while i < len(word):
            for n in (4, 3, 2, 1):
                chunk = word[i:i + n]
                pool = profile["ligatures"].get(chunk) if n > 1 else profile["glyphs"].get(chunk)
                if pool:
                    v = rng.choice(pool)
                    parts.append((v, x))
                    x += v["w"] - 12 + gap
                    i += n
                    break
            else:
                parts.append((None, x))
                x += 0.7 * xh + gap
                i += 1
        return parts, x

    lines, current, x = [], [], margin
    for text in PANGRAMS:
        for word in text.split():
            parts, w = layout_word(word)
            if current and x + w > width - margin:
                lines.append(current)
                current, x = [], margin
            current.append((parts, x))
            x += w + space
        lines.append(current)
        current, x = [], margin

    canvas = np.full((int(pitch * (len(lines) + 1)), width, 3), 255, np.float32)
    missing_boxes = []
    for li, line in enumerate(lines):
        base = pitch * (li + 0.8)
        for parts, wx in line:
            for v, dx in parts:
                if v is None:
                    x0 = int(wx + dx)
                    missing_boxes.append((x0, int(base - xh), int(x0 + 0.7 * xh), int(base)))
                else:
                    paste_ink(canvas, glyph_image(prof_dir, v, 1), int(wx + dx), int(base - v["baseline"]), INK_BLUE)
    im = Image.fromarray(canvas.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for b in missing_boxes:
        d.rectangle(b, outline=(220, 30, 30), width=3)
    im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    im.save(prof_dir / "preview.png")


# ---------------------------------------------------------------- coverage

def coverage_report(name, profile, counters, n_samples):
    g, lig = profile["glyphs"], profile["ligatures"]
    lines = [f'Profile "{name}": {sum(len(v) for v in g.values())} glyphs, {len(lig)} ligatures '
             f"({sum(len(v) for v in lig.values())} variants), from {n_samples} pages"]

    def group(label, chars, needed=3):
        have = {ch: len(g.get(ch, [])) for ch in chars}
        missing = [ch for ch, k in have.items() if k == 0]
        thin = [ch for ch, k in have.items() if 0 < k < needed]
        present = [k for k in have.values() if k]
        mark = "✔" if not missing else "✘"
        rng = f"{min(present)}–{max(present)} variants" if present else "none"
        txt = f"  {label:<6}{mark} {len(chars) - len(missing)}/{len(chars)}, {rng}"
        if missing:
            txt += f"; missing: {' '.join(missing)}"
        if thin:
            txt += f"; fewer than {needed}: {' '.join(thin)}"
        lines.append(txt)
        return missing

    group("a–z", LOWER)
    group("A–Z", UPPER, needed=2)
    group("0–9", DIGITS, needed=2)
    group("punct", PUNCT, needed=1)
    math_missing = group("math", MATH, needed=1)
    if math_missing:
        lines.append("        (math symbols are collected with /handwritten gaps, coming in Phase 4)")
    only_joined = []
    for ch in LOWER:
        if not g.get(ch):
            runs = [t for t in lig if ch in t]
            if runs:
                only_joined.append(f"{ch} (in {', '.join(runs[:3])})")
    if only_joined:
        lines.append("  joined only: " + "; ".join(only_joined))
    tall = profile.get("tall") or {}
    if tall:
        lines.append("  tall:  " + " ".join(f"{ch}({len(v)})" for ch, v in sorted(tall.items()))
                     + "  (your own brackets around matrices)")
    lines += report_lines(profile)
    top = sorted(lig.items(), key=lambda kv: -len(kv[1]))[:20]
    if top:
        lines.append("  ligatures: " + " ".join(f"{t}({len(v)})" for t, v in top))
    st = profile["style"]
    lines.append(f"  style: x-height {st['x_height_mm']:.2f} mm, slant {st['slant_deg']:+.1f} deg, "
                 f"stroke {st['stroke_mm']:.2f} mm, line pitch "
                 f"{(st['line_pitch_mm'] or {'median': float('nan')})['median']:.1f} mm")
    if counters["words_unlabelled"]:
        lines.append(f"  NOTE: {counters['words_unlabelled']} words still need labels (labels.txt)")
    if counters["labels_invalid"]:
        lines.append(f"  NOTE: {counters['labels_invalid']} labels did not fit their word and were ignored")
    lines.append(f"  dropped as implausible: {counters['implausible']}, as outliers: {counters['outliers']}, "
                 f"rejected in review: {counters['rejected']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--reject", help="comma-separated numbers from the latest specimen sheets")
    args = ap.parse_args()

    pdir = profile_dir(args.profile)
    sroot = samples_dir(args.profile)
    if not sroot.exists():
        sys.exit(f"ERROR: profile '{args.profile}' has no samples yet. Run preprocess.py first.")

    rejected_path = pdir / "rejected.json"
    rejected = set(read_json(rejected_path, []))
    if args.reject:
        mapping = read_json(pdir / "specimen_map.json", {})
        for num in (x.strip() for x in args.reject.split(",") if x.strip()):
            if num in mapping:
                rejected.add(mapping[num])
            else:
                print(f"WARNING: no glyph number {num} on the latest specimen sheets")
        write_json(rejected_path, sorted(rejected))

    counters = {k: 0 for k in ("samples_not_ready", "words_unlabelled", "words_skipped", "labels_invalid",
                               "words_used", "implausible", "outliers", "rejected")}
    all_units, pages = [], []
    sdirs = sorted(p for p in sroot.iterdir() if p.is_dir())
    for sdir in sdirs:
        units, page = extract_sample(sdir, rejected, counters)
        if units:
            all_units += units
            pages.append(page)
    settle_template_baselines(all_units, {p["id"] for p in pages if p.get("template")})
    if not all_units:
        sys.exit("ERROR: no labelled glyphs yet. Finish transcripts, segmentation and labels first.")

    # Filter and group.
    by_text = {}
    for u in all_units:
        reason = plausible(u)
        if reason:
            counters["implausible"] += 1
            continue
        by_text.setdefault(u["text"], []).append(u)

    scales = sample_scales(pages)
    for sid, f in scales.items():
        if f != 1.0:
            print(f"NOTE    {sid}: page size was guessed; its glyphs are resized by {f:.2f} to match the other pages")
    for d in ("glyphs", "ligatures", "tall", "scripts", "radicals", "accents", "bigops"):
        shutil.rmtree(pdir / d, ignore_errors=True)
    profile = {"schema": 1, "name": args.profile, "dpi": 300,
               "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "samples": [], "glyphs": {}, "ligatures": {}}
    for text, variants in sorted(by_text.items()):
        if len(text) > 4:
            continue  # whole joined words are kept in the samples for later engines
        keep, dropped = drop_outliers(variants)
        counters["outliers"] += len(dropped)
        kind = "glyphs" if len(text) == 1 else "ligatures"
        out = pdir / kind / char_key(text)
        profile[kind][text] = [save_unit(u, out, scales[u["sample"]]) for u in keep]

    n_pages = len(pages)
    profile["tall"] = collect_tall(sroot, pdir, rejected)
    # Style statistics come from pages with a reliable scale when there are any.
    if any(p["trusted"] for p in pages):
        pages = [p for p in pages if p["trusted"]]
        all_units = [u for u in all_units if scales[u["sample"]] == 1.0]
    template_ids = {p["id"] for p in pages if p.get("template")}
    if template_ids and len(template_ids) < len(pages):
        pages = [p for p in pages if not p.get("template")]
        all_units = [u for u in all_units if u["sample"] not in template_ids]
    lines = [l for p in pages for l in p["lines"]]
    singles = [u for u in all_units if len(u["text"]) == 1]
    pitches, margins_l, margins_r, slopes = [], [], [], []
    for p in pages:
        ls = sorted(p["lines"], key=lambda l: l["a"])
        mid = p["width_px"] / 2
        ys = [l["a"] + l["b"] * mid for l in ls]
        pitches += [mm(y2 - y1) for y1, y2 in zip(ys, ys[1:])]
        margins_l += [mm(l["x_min"]) for l in ls]
        margins_r += [mm(p["width_px"] - l["x_max"]) for l in ls]
        slopes += [math.degrees(math.atan(l["b"])) for l in ls]
    pitch_stat = stat(pitches)
    if pitch_stat:  # ignore paragraph gaps when summarising the usual pitch
        med = pitch_stat["median"]
        pitch_stat = stat([x for x in pitches if x < 1.5 * med])
    xh_px = float(np.median([l["xh"] for l in lines]))
    style = {
        "x_height_px": round(xh_px, 1),
        "x_height_mm": round(mm(xh_px), 3),
        "ascender_mm": stat([mm(u["above"] * u["xh"]) for u in singles if u["text"] in "bdhkl"]),
        "cap_height_mm": stat([mm(u["above"] * u["xh"]) for u in singles if u["text"] in UPPER and u["text"] not in "JQ"]),
        "descender_mm": stat([mm(u["below"] * u["xh"]) for u in singles if u["text"] in "gjpqy"]),
        "stroke_mm": round(mm(float(np.median([p["stroke_px"] for p in pages if p["stroke_px"]]))), 3),
        "slant_deg": round(float(np.median([s for p in pages for s in p["slants"] if s is not None] or [0])), 1),
        "letter_gap_mm": stat([mm(g) for p in pages for g in p["letter_gaps"]]),
        "word_gap_mm": stat([mm(g) for p in pages for g in p["word_gaps"]]),
        "line_pitch_mm": pitch_stat,
        "left_margin_mm": stat(margins_l),
        "right_margin_mm": stat(margins_r),
        "baseline_slope_deg": stat(slopes),
    }
    profile["style"] = style
    profile.update(collect_extras(sroot, pdir, rejected, style))
    for sdir in sdirs:
        meta = read_json(sdir / "meta.json", {})
        if (sdir / "segments.json").exists():
            profile["samples"].append({"id": meta.get("id"), "source": meta.get("source"),
                                       "scale_source": meta.get("scale_source")})
    report = coverage_report(args.profile, profile, counters, n_pages)
    profile["coverage"] = report
    write_json(pdir / "profile.json", profile)
    (pdir / "coverage.txt").write_text(report + "\n", encoding="utf-8")
    specimens = write_specimens(pdir, profile)
    suspects, suspect_map = write_suspects(pdir, profile)
    mapping = read_json(pdir / "specimen_map.json", {})
    mapping.update(suspect_map)
    write_json(pdir / "specimen_map.json", mapping)
    write_preview(pdir, profile)

    print(report)
    print()
    for p in specimens:
        print(f"review:  {p}")
    for p in suspects:
        print(f"check:   {p}  (shapes that look like another character; reject real mistakes as s<N>)")
    print(f"preview: {pdir / 'preview.png'}")


if __name__ == "__main__":
    main()
