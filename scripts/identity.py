"""Train the identity block: the serial number and name written in the top-left corner of the first
page, closed off by a line underneath and a line down the right side, as on an exam answer sheet.

Training reads one photo of a page with the block on it. It finds the page and the two border
lines, cuts out every character, and measures where each one sits, so render.py --identity can
write the block again on the first page of the output: the same rows and spacing, every character
one of the writer's own from the photo, with the small differences of writing it again. Results go
to ~/.handwritten/profiles/<profile>/identity/:

  identity.json   the rows, where every character sits, the border lines (all in mm)
  glyphs/NN.png   every character cut from the photo (ink 0..255)
  sheet.png       the block with every character boxed and numbered, then the characters with
                  their labels: Claude checks these
  preview.png     the block written the way render.py --identity writes it

Usage: identity.py train PHOTO --line "SERIAL NO. : 12-34-56-78" --line "NAME : JANE DOE."
                         [--profile NAME] [--corners x,y,x,y,x,y,x,y] [--reject 3,17]
       identity.py show [--profile NAME]
"""

import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize

from common import (A4_MM, LETTER_MM, PX_PER_MM, file_sha, load_image_rgb, profile_dir, read_json, ui_font,
                    write_json)
from preprocess import find_page_quad, flatten_lighting, order_corners
from render import Glyph, jittered, lay_ink, pen_image, resized

EDGE_MM = 1.0  # the paper's own edge shows up as a dark line this close to the border: not ink
LINE_KERNEL_MM = (10.0, 6.0)  # runs of ink this long (across, down) are the border lines
FROM_EDGE_MM = 3.0  # a border line that stops this close to the page edge was drawn off the edge
SPECK_PX = 8  # ink pieces smaller than this are dust; full stops are bigger


# ---------------------------------------------------------------- finding the page

def rough_page(rgb):
    """The page's corners for photos where find_page_quad fails, such as a page on a notebook on a
    pale, textured desk.

    Scans outward from the middle of the photo, left, right, up and down, and stops where the paper
    gives way to something much darker for longer than any pen stroke is wide: the desk or the
    cover the page lies on. A line through each side's stopping points gives the page's edges."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0).astype(np.float32)
    h, w = gray.shape
    run = max(6, int(0.005 * max(h, w)))  # wider than a pen stroke in the photo
    sides = []
    for axis, sign in ((0, -1), (1, +1), (0, +1), (1, -1)):  # top, right, bottom, left
        pts = []
        across = w if axis == 0 else h
        for t in np.linspace(0.15, 0.85, 50):
            c = int(t * across)
            line = gray[:, c] if axis == 0 else gray[c, :]
            start = (h if axis == 0 else w) // 2
            idx = np.arange(start, -1, -1) if sign < 0 else np.arange(start, len(line))
            prof = line[idx]
            for i in range(25, len(prof) - run):
                ref = np.median(prof[i - 25:i - 3])
                if prof[i:i + run].max() < 0.65 * ref:
                    pos = idx[i]
                    pts.append((c, pos) if axis == 0 else (pos, c))
                    break
        if len(pts) < 15:
            return None
        pts = np.array(pts, np.float64)
        u, v = (pts[:, 0], pts[:, 1]) if axis == 0 else (pts[:, 1], pts[:, 0])
        keep = np.abs(v - np.median(v)) < 0.03 * max(h, w)  # stops on a shadow or a pen are off the line
        slope, pos = np.polyfit(u[keep], v[keep], 1)
        sides.append((axis, slope, pos))
    corners = []
    for k in range(4):
        (a1, s1, p1), (a2, s2, p2) = sides[k - 1], sides[k]
        # One side is y = s x + p (axis 0), the other x = s y + p (axis 1).
        (sh, ph), (sv, pv) = ((s1, p1), (s2, p2)) if a1 == 0 else ((s2, p2), (s1, p1))
        x = (sv * ph + pv) / (1 - sv * sh)
        corners.append((x, sh * x + ph))
    quad = np.array(corners, np.float32)  # tl, tr, br, bl
    tl, tr, br, bl = quad
    qw, qh = np.linalg.norm(tr - tl), np.linalg.norm(bl - tl)
    aspect = max(qw, qh) / max(1.0, min(qw, qh))
    if not any(abs(aspect - d[1] / d[0]) < 0.08 for d in (A4_MM, LETTER_MM)):
        return None
    return quad


def snap_to_edges(gray, quad):
    """Move each side of the quad onto the paper's edge nearby: where it turns from dark outside to
    bright inside. Corners are where the fitted sides meet."""
    quad = np.asarray(quad, np.float64)
    size = np.linalg.norm(quad[2] - quad[0])
    reach = 0.03 * size
    centre = quad.mean(axis=0)
    g = cv2.GaussianBlur(gray, (5, 5), 1.5).astype(np.float64)
    lines = []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        along = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-along[1], along[0]])
        if np.dot(centre - a, normal) < 0:
            normal = -normal  # point into the page
        steps = np.arange(-reach, reach, 0.5)
        pts = []
        for t in np.linspace(0.1, 0.9, 60):
            p = a + (b - a) * t
            xy = p[None, :] + steps[:, None] * normal[None, :]
            prof = map_coordinates(g, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
            d = np.diff(prof)
            if d.max() <= 4:  # per half pixel
                continue
            # The first strong rise coming in from outside: writing near the edge rises as sharply
            # (dark stroke to paper) but lies further in.
            strong = np.nonzero(d > 0.5 * d.max())[0]
            i = int(strong[0])
            while i + 1 < len(d) and d[i + 1] > d[i]:
                i += 1  # the steepest point of that rise
            pts.append(p + (steps[i] + 0.25) * normal)
        if len(pts) < 10:
            lines.append((a, along))
            continue
        pts = np.array(pts)
        for _ in range(2):
            vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
            if (d < 2.5).sum() >= 10:
                pts = pts[d < 2.5]
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    corners = []
    for k in range(4):
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        t = np.linalg.solve(np.column_stack([d1, -d2]), p2 - p1)
        corners.append(p1 + t[0] * d1)
    return np.array(corners, np.float32)


def flat_page(rgb, corners=None):
    """The photo as a flat A4 page at 300 DPI, and the corners used."""
    quad = np.float32(corners) if corners is not None else find_page_quad(rgb)
    if quad is None:
        quad = rough_page(rgb)
    if quad is None:
        return None, None
    quad = snap_to_edges(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), quad)
    w, h = int(round(A4_MM[0] * PX_PER_MM)), int(round(A4_MM[1] * PX_PER_MM))
    m = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    return cv2.warpPerspective(rgb, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), quad


def ink_maps(page):
    """Ink intensity (0..255) and the binary ink mask, as preprocess.py makes them."""
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    norm = flatten_lighting(gray)
    ink = 1.0 - norm
    win = max(25, int(min(gray.shape) / 60) | 1)
    mask = ((norm < threshold_sauvola(norm, window_size=win, k=0.2)) & (ink > 0.18)).astype(np.uint8) * 255
    e = int(EDGE_MM * PX_PER_MM)
    mask[:e], mask[-e:], mask[:, :e], mask[:, -e:] = 0, 0, 0, 0
    keep = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
    vals = ink[mask > 0]
    hi = float(np.percentile(vals, 95)) if vals.size else 1.0
    return (np.clip(ink / max(hi, 0.2), 0, 1) * 255 * keep).astype(np.uint8), mask


# ---------------------------------------------------------------- the border lines

def border_line(mask, horizontal):
    """The longest straight run of ink across (horizontal) or down the page in the top part of the page.

    Returns {"pos": y at x=0 (or x at y=0), "slope", "start", "end", "wobble"} in pixels, and the
    pixels of the line, or None."""
    k = int(LINE_KERNEL_MM[0 if horizontal else 1] * PX_PER_MM)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1) if horizontal else (1, k))
    runs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
    if n < 2:
        return None, None
    length = stats[1:, cv2.CC_STAT_WIDTH if horizontal else cv2.CC_STAT_HEIGHT]
    best = 1 + int(np.argmax(length))
    ys, xs = np.nonzero(cc == best)
    u, v = (xs, ys) if horizontal else (ys, xs)  # u along the line, v across it
    m = mask if horizontal else mask.T  # rows of m run along the line
    # Follow the line out from that run one column at a time, so a page that doesn't lie quite
    # flat (the line curving gently in the photo) doesn't lose its ends; stop at a break of over 1 mm.
    centre = {int(c): float(v[u == c].mean()) for c in np.unique(u)}
    half, gap = 0.5 * PX_PER_MM, PX_PER_MM
    for step in (-1, +1):
        c = min(centre) if step < 0 else max(centre)
        at, miss = centre[c], 0
        while 0 <= c + step < m.shape[1] and miss <= gap:
            c += step
            lo, hi = int(at - half), int(at + half) + 1
            rows = np.nonzero(m[max(0, lo):hi, c])[0]
            if len(rows):
                at = 0.7 * at + 0.3 * (max(0, lo) + rows.mean())
                centre[c], miss = at, 0
            else:
                miss += 1
    cols = np.array(sorted(centre))
    mids = np.array([centre[c] for c in cols])
    slope, pos = np.polyfit(cols, mids, 1)
    # How far the line in the photo strays from straight (the paper's bend counts too).
    wobble = float(np.std(mids - (slope * cols + pos)))
    vv, uu = np.nonzero(m)
    near = (np.abs(vv - np.interp(uu, cols, mids)) < half + 2) & (uu >= cols[0]) & (uu <= cols[-1])
    line_px = (vv[near], uu[near]) if horizontal else (uu[near], vv[near])
    width = float(np.median(np.bincount(uu[near])[np.unique(uu[near])]))  # ink pixels across the line
    return {"pos": float(pos), "slope": float(slope), "start": float(cols[0]), "end": float(cols[-1]),
            "wobble": wobble, "width": width}, line_px


def stroke_width(mask):
    """Typical pen width in pixels: twice the distance to the stroke edge along the stroke centres."""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    skel = skeletonize(mask > 0)
    vals = dist[skel]
    return float(2 * np.median(vals)) if vals.size else 0.5 * PX_PER_MM


# ---------------------------------------------------------------- characters

def text_rows(stats, ids):
    """Group ink pieces into rows of writing by their vertical centres."""
    centres = sorted(ids, key=lambda c: stats[c, cv2.CC_STAT_TOP] + stats[c, cv2.CC_STAT_HEIGHT] / 2)
    heights = [stats[c, cv2.CC_STAT_HEIGHT] for c in ids]
    split = 0.8 * float(np.percentile(heights, 90))
    rows, prev = [], None
    for c in centres:
        y = stats[c, cv2.CC_STAT_TOP] + stats[c, cv2.CC_STAT_HEIGHT] / 2
        if prev is None or y - prev > split:
            rows.append([])
        rows[-1].append(c)
        prev = y
    return rows


def group_characters(stats, row, count):
    """Join the pieces of a row into `count` characters, left to right.

    Pieces stacked above each other (the dots of a colon) or mostly overlapping (the strokes of a K)
    are one character. Then, while there are too many, the closest neighbours whose joined width
    still looks like one character are joined."""
    boxes = sorted(([stats[c, 0], stats[c, 1], stats[c, 0] + stats[c, 2], stats[c, 1] + stats[c, 3], [c]] for c in row),
                   key=lambda b: b[0])
    merged = []
    for b in boxes:
        if merged:
            m = merged[-1]
            overlap = min(m[2], b[2]) - max(m[0], b[0])
            if overlap > 0.5 * min(m[2] - m[0], b[2] - b[0]):
                merged[-1] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3]), m[4] + b[4]]
                continue
        merged.append(b)
    widths = [b[2] - b[0] for b in merged]
    widest = 1.4 * float(np.percentile(widths, 75)) if widths else 0
    while len(merged) > count:
        gaps = [(merged[k + 1][0] - merged[k][2], k) for k in range(len(merged) - 1)
                if max(merged[k][2], merged[k + 1][2]) - merged[k][0] <= widest]
        if not gaps:
            break
        _, k = min(gaps)
        a, b = merged[k], merged.pop(k + 1)
        merged[k] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
    return merged


def ink_gain(pdir, crops):
    """How much to darken the cut-out characters so the cores of their strokes (75th percentile of
    the ink) match the writing on the rest of the page, the profile's glyphs; within 1 to 1.5."""
    profile = read_json(pdir / "profile.json") or {}
    files = [m["file"] for pool in (profile.get("glyphs") or {}).values() for m in pool[:10]]
    ref = [cv2.imread(str(pdir / f), cv2.IMREAD_GRAYSCALE) for f in files]
    ref = np.concatenate([r.ravel() for r in ref if r is not None] or [np.zeros(1, np.uint8)])
    ref = ref[ref > 40]
    own = np.concatenate([c.ravel() for c in crops])
    own = own[own > 40]
    target = float(np.percentile(ref, 75)) if ref.size > 1000 else 230.0
    return float(np.clip(target / max(1.0, float(np.percentile(own, 75))), 1.0, 1.5)) if own.size else 1.0


def fit_baseline(chars, labels):
    """The row's baseline as y = slope * x + pos, through the bottoms of its letters and digits."""
    pts = np.array([((c[0] + c[2]) / 2, c[3]) for c, ch in zip(chars, labels) if ch.isalnum()], np.float64)
    if len(pts) < 2:
        y = float(np.median([c[3] for c in chars]))
        return 0.0, y
    slope, pos = np.polyfit(pts[:, 0], pts[:, 1], 1)
    res = pts[:, 1] - (slope * pts[:, 0] + pos)
    good = np.abs(res) < max(0.35 * PX_PER_MM, 2 * np.median(np.abs(res)))
    if good.sum() >= 2:
        slope, pos = np.polyfit(pts[good, 0], pts[good, 1], 1)
    return float(slope), float(pos)


# ---------------------------------------------------------------- writing the block again

class IdentityBlock:
    """The trained block, written again in the top-left corner of a page.

    Every character is one of the writer's own copies of it from the photo, placed with the spacing
    measured there. The rows and the two border lines move a little, as when the block is written
    out again on a new sheet. bottom is the lowest point of the line under the block (pixels)."""

    def __init__(self, pdir, data, rng, bank=None):
        idir = pdir / "identity"
        self.rng, self.bank = rng, bank
        self.cap_px = (data.get("cap_height_mm") or 3.4) * PX_PER_MM
        self.pen = max(2.0, data.get("stroke_mm", 0.5) * PX_PER_MM)
        self.glyphs = {g["n"]: Glyph(cv2.imread(str(idir / g["file"]), cv2.IMREAD_GRAYSCALE), g["baseline"])
                       for g in data["glyphs"]}
        self.copies = {}
        for g in data["glyphs"]:
            if not g["rejected"]:
                self.copies.setdefault(g["char"], []).append(g["n"])
        self.last = {}
        self.plan(data)

    def pick(self, slot):
        """One of the writer's copies of the slot's character, not the one used just before."""
        ch = slot["char"]
        copies = self.copies.get(ch)
        if not copies:
            # Every copy was rejected in review: use the writer's ordinary one, at the block's size.
            g = self.bank.single(ch, self.rng) if self.bank is not None else None
            if g is not None:
                return resized(g, self.cap_px / self.bank.cap_px)
            return self.glyphs[slot["glyph"]]
        n = self.rng.choice([c for c in copies if c != self.last.get(ch)] or copies)
        self.last[ch] = n
        return self.glyphs[n]

    def plan(self, data):
        rng, P = self.rng, PX_PER_MM
        mm = lambda sd: rng.gauss(0, sd) * P
        self.rows = []
        for row in data["rows"]:
            x0 = row["x_mm"] * P + mm(0.6)
            y0 = row["baseline_mm"] * P + mm(0.4)
            slope = row["slope"] + rng.gauss(0, 0.002)
            placed, right = [], x0
            for k, slot in enumerate(row["slots"]):
                g = jittered(self.pick(slot), rng)
                x = x0 if k == 0 else right + slot["gap_mm"] * P * rng.uniform(0.85, 1.15)
                placed.append([g, x])
                right = x + g.adv
            # The row comes out about as long as it was written: the gaps take up the difference,
            # word spaces more than the gaps inside numbers.
            want = row["width_mm"] * P * (1 + rng.gauss(0, 0.01))
            weights = np.array([0.0] + [max(s["gap_mm"], 0.3) for s in row["slots"][1:]])
            shift = np.cumsum(weights) / max(weights.sum(), 1e-6) * (want - (right - x0))
            for p, s in zip(placed, shift):
                p[1] += s
            self.rows.append({"placed": placed, "x0": x0, "y0": y0, "slope": slope, "end": x0 + want})

        b, r = data["border"]["bottom"], data["border"]["right"]
        # The line underneath keeps its distance below the last row, the right one its distance
        # after the longest row, so the block stays closed around the writing.
        last = self.rows[-1]
        by0 = last["y0"] + b["below_row_mm"] * P + mm(0.4)
        bslope = b["slope"] + rng.gauss(0, 0.002)
        bottom_y = lambda x: by0 + bslope * (x - last["x0"])
        widest = max(self.rows, key=lambda row: row["end"])
        rx0 = widest["end"] + r["after_row_mm"] * P + mm(0.8)
        rslope = r["slope"] + rng.gauss(0, 0.002)
        right_x = lambda y: rx0 + rslope * (y - widest["y0"])
        cx = rx0
        for _ in range(3):  # where the two lines cross
            cx = right_x(bottom_y(cx))
        cy = bottom_y(cx)
        x_from = -P if b["start_mm"] is None else b["start_mm"] * P + mm(0.5)
        x_to = cx + (b["past_right_mm"] + rng.gauss(0, 0.2)) * P
        y_from = -P if r["start_mm"] is None else r["start_mm"] * P + mm(0.5)
        y_to = cy + (r["past_bottom_mm"] + rng.gauss(0, 0.2)) * P
        self.lines = [self.ruled(x_from, x_to, bottom_y, True, b), self.ruled(y_from, y_to, right_x, False, r)]
        self.bottom = max(bottom_y(x_from), bottom_y(x_to)) + self.pen

    def ruled(self, a, b, across, horizontal, line):
        """Points, pen widths and darkness of a border line from a to b: drawn against a ruler, so
        straight but for a faint wave, and a little thinner where the pen lifts off at the corner."""
        rng, P = self.rng, PX_PER_MM
        amp = min(0.25, max(0.03, line["wobble_mm"])) * P
        pen = self.pen
        if line.get("pen_mm"):  # the line's own width in the photo, within reason
            pen = min(1.2 * self.pen, max(0.6 * self.pen, line["pen_mm"] * P))
        wave = rng.uniform(60, 140) * P
        phase = rng.uniform(0, 2 * math.pi)
        n = max(2, int(abs(b - a) / (3 * P)))
        pts = []
        for f in np.linspace(0, 1, n + 1):
            u = a + (b - a) * f
            v = across(u) + amp * math.sin(2 * math.pi * u / wave + phase)
            pts.append((u, v) if horizontal else (v, u))
        pen *= rng.uniform(0.95, 1.05)
        widths = [pen] * n + [0.8 * pen]
        strength = max(0.55, line.get("darkness", 1.0)) * rng.uniform(0.85, 0.95)
        return pts, widths, strength

    def draw(self, alpha):
        """Write the block onto a page's ink coverage (0..1)."""
        rng = self.rng
        for row in self.rows:
            for g, x in row["placed"]:
                base = row["y0"] + row["slope"] * (x - row["x0"]) + rng.gauss(0, 0.06) * PX_PER_MM
                img = g.img.astype(np.float32) / 255 * rng.uniform(0.85, 1.0)
                lay_ink(alpha, img, int(round(x - g.left)), int(round(base - g.baseline)))
        for pts, widths, strength in self.lines:
            img, x0, y0 = pen_image(pts, widths)
            lay_ink(alpha, img * strength, x0, y0)


# ---------------------------------------------------------------- review images

def write_sheet(path, page, chars, labels, numbers):
    """The block with every character boxed and numbered, and below it each character on its own."""
    x1 = max(c[2] for c in chars) + int(4 * PX_PER_MM)
    y1 = max(c[3] for c in chars) + int(3 * PX_PER_MM)
    block = Image.fromarray(page[:y1, :x1]).convert("RGB")
    d = ImageDraw.Draw(block)
    font = ui_font(22)
    for c, n in zip(chars, numbers):
        d.rectangle([c[0] - 2, c[1] - 2, c[2] + 2, c[3] + 2], outline=(220, 30, 30), width=2)
        d.text((c[0], c[1] - 28), str(n), fill=(220, 30, 30), font=font)
    cell, cols = 150, max(1, x1 // 150)
    rows = -(-len(chars) // cols)
    grid = Image.new("RGB", (x1, rows * cell), "white")
    gd = ImageDraw.Draw(grid)
    for k, (c, ch, n) in enumerate(zip(chars, labels, numbers)):
        crop = Image.fromarray(page[c[1]:c[3], c[0]:c[2]]).convert("RGB")
        crop.thumbnail((cell - 20, cell - 50))
        gx, gy = (k % cols) * cell, (k // cols) * cell
        grid.paste(crop, (gx + 10, gy + 10))
        gd.text((gx + 10, gy + cell - 36), f"{n}  {ch}", fill=(20, 90, 220), font=font)
        gd.rectangle([gx, gy, gx + cell - 1, gy + cell - 1], outline=(200, 200, 200))
    sheet = Image.new("RGB", (x1, block.height + grid.height + 20), "white")
    sheet.paste(block, (0, 0))
    sheet.paste(grid, (0, block.height + 20))
    sheet.save(path)


def write_preview(pdir, data, path, seed=11):
    """The block as render.py --identity writes it, on a clean page cut off below the block."""
    from render import GlyphBank

    profile = read_json(pdir / "profile.json")
    bank = GlyphBank(pdir, profile) if profile and profile.get("glyphs") else None
    w, h = int(round(A4_MM[0] * PX_PER_MM)), int(round(A4_MM[1] * PX_PER_MM))
    alpha = np.zeros((h, w), np.float32)
    block = IdentityBlock(pdir, data, random.Random(seed), bank)
    block.draw(alpha)
    bottom = min(h, int(block.bottom + 6 * PX_PER_MM))
    ink = np.array((25, 45, 140), np.float32)
    rgb = (255 * (1 - alpha[:bottom, :, None]) + ink * alpha[:bottom, :, None]).astype(np.uint8)
    cv2.rectangle(rgb, (0, 0), (w - 1, bottom - 1), (180, 180, 180), 2)  # the page's edge
    Image.fromarray(rgb).save(path)


# ---------------------------------------------------------------- training

def train(args):
    photo = Path(args.photo).expanduser()
    if not photo.is_file():
        sys.exit(f"ERROR: not found: {photo}")
    lines = [ln.strip() for ln in args.line if ln.strip()]
    if not lines:
        sys.exit('ERROR: give the block\'s text, one --line per row, e.g. --line "NAME : ..."')
    pdir = profile_dir(args.profile)
    idir = pdir / "identity"

    corners = None
    if args.corners:
        v = [float(x) for x in args.corners.split(",")]
        if len(v) != 8:
            sys.exit("ERROR: --corners takes 8 numbers: top-left top-right bottom-right bottom-left")
        corners = [v[0:2], v[2:4], v[4:6], v[6:8]]
    rgb = load_image_rgb(photo)
    page, quad = flat_page(rgb, corners)
    if page is None:
        sys.exit("PROBLEM: the page's edges weren't found. Give its corners with --corners "
                 "(x,y pixel pairs in the photo: top-left top-right bottom-right bottom-left).")
    ink, mask = ink_maps(page)

    # The block sits in the top part of the page: everything below it is ignored.
    top_part = mask.copy()
    top_part[int(100 * PX_PER_MM):] = 0
    bottom, bottom_px = border_line(top_part, horizontal=True)
    right, right_px = border_line(top_part, horizontal=False)
    if bottom is None or right is None or bottom["end"] - bottom["start"] < 50 * PX_PER_MM:
        sys.exit("PROBLEM: the line under the block and the line down its right side weren't both found.")
    text = top_part.copy()
    for ys, xs in (bottom_px, right_px):
        line = np.zeros_like(mask)
        line[ys, xs] = 255
        text[cv2.dilate(line, np.ones((5, 5), np.uint8)) > 0] = 0
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    inside = (yy < bottom["slope"] * xx + bottom["pos"] - 0.5 * PX_PER_MM) & \
             (xx < right["slope"] * yy + right["pos"] - 0.5 * PX_PER_MM)
    text[~inside] = 0
    del yy, xx

    n, cc, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
    ids = [c for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= SPECK_PX]
    rows = text_rows(stats, ids)
    if len(rows) != len(lines):
        sys.exit(f"PROBLEM: found {len(rows)} rows of writing inside the block but {len(lines)} --line were given.")

    reject = {int(x) for x in args.reject.split(",") if x.strip()} if args.reject else set()
    shutil.rmtree(idir / "glyphs", ignore_errors=True)
    (idir / "glyphs").mkdir(parents=True, exist_ok=True)
    data = {"schema": 1, "source": str(photo), "sha256": file_sha(photo),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "corners_px": np.round(quad, 1).tolist(),
            "lines": lines, "rows": [], "glyphs": [], "rejected": sorted(reject)}
    all_chars, all_labels, numbers, problems, crops = [], [], [], [], []
    for r, (row, line) in enumerate(zip(rows, lines)):
        labels = list(line.replace(" ", ""))
        chars = group_characters(stats, row, len(labels))
        if len(chars) != len(labels):
            problems.append(f"row {r + 1}: {len(chars)} characters found, {len(labels)} in the text")
            labels = (labels + ["?"] * len(chars))[:len(chars)]
        slope, pos = fit_baseline(chars, labels)
        slots = []
        for k, (c, ch) in enumerate(zip(chars, labels)):
            num = len(data["glyphs"]) + 1
            pad = int(0.6 * PX_PER_MM)
            x0, y0 = max(0, c[0] - pad), max(0, c[1] - pad)
            x1, y1 = c[2] + pad, c[3] + pad
            own = np.isin(cc[y0:y1, x0:x1], c[4]).astype(np.uint8)
            own = cv2.dilate(own, np.ones((5, 5), np.uint8))
            crop = (ink[y0:y1, x0:x1] * own).astype(np.uint8)
            name = f"glyphs/{num:02d}.png"
            crops.append((idir / name, crop))
            baseline = slope * (c[0] + c[2]) / 2 + pos
            data["glyphs"].append({"n": num, "char": ch, "file": name, "baseline": float(baseline - y0),
                                   "rejected": num in reject})
            slots.append({"char": ch, "glyph": num, "gap_mm": 0.0 if k == 0 else (c[0] - chars[k - 1][2]) / PX_PER_MM})
            all_chars.append(c)
            all_labels.append(ch)
            numbers.append(num)
        data["rows"].append({"text": line, "x_mm": chars[0][0] / PX_PER_MM,
                             "baseline_mm": (slope * chars[0][0] + pos) / PX_PER_MM, "slope": slope,
                             "width_mm": (chars[-1][2] - chars[0][0]) / PX_PER_MM, "slots": slots})

    # A photo at lower resolution than the training pages blurs the strokes, so the same pen comes
    # out paler: bring the ink up to the level of the writer's ordinary capitals and digits.
    gain = ink_gain(pdir, [c for _, c in crops])
    for path, crop in crops:
        cv2.imwrite(str(path), np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8))
    data["ink_gain"] = gain

    caps = [c[3] - c[1] for c, ch in zip(all_chars, all_labels) if ch.isupper() or ch.isdigit()]
    data["cap_height_mm"] = float(np.median(caps)) / PX_PER_MM if caps else None
    data["stroke_mm"] = stroke_width(text) / PX_PER_MM
    last = data["rows"][-1]
    widest = max(data["rows"], key=lambda r: r["x_mm"] + r["width_mm"])
    bx, by = bottom["start"], bottom["slope"] * bottom["start"] + bottom["pos"]
    # Where the lines cross: y = sb x + pb and x = sr y + pr.
    rx_at_bottom = (right["slope"] * bottom["pos"] + right["pos"]) / (1 - right["slope"] * bottom["slope"])
    by_at_right = bottom["slope"] * rx_at_bottom + bottom["pos"]
    widest_y = widest["baseline_mm"] * PX_PER_MM
    # How dark each line is next to the writing: a line drawn fast against a ruler often comes out lighter.
    text_ink = float(ink[text > 0].mean())
    darkness = [min(1.0, float(ink[ys, xs].mean()) / text_ink) for ys, xs in (bottom_px, right_px)]
    data["border"] = {
        "bottom": {
            # where the line sits, measured from the last row's baseline at that row's start
            "below_row_mm": (bottom["slope"] * last["x_mm"] * PX_PER_MM + bottom["pos"]) / PX_PER_MM - last["baseline_mm"],
            "slope": bottom["slope"],
            "start_mm": None if bx < FROM_EDGE_MM * PX_PER_MM else bx / PX_PER_MM,  # None: from the page edge
            "past_right_mm": (bottom["end"] - rx_at_bottom) / PX_PER_MM,  # how far it runs past the right line
            "wobble_mm": bottom["wobble"] / PX_PER_MM,
            "pen_mm": bottom["width"] / PX_PER_MM,
            "darkness": darkness[0],
            "y_mm": by / PX_PER_MM,
        },
        "right": {
            # where the line sits, measured from the end of the longest row
            "after_row_mm": (right["slope"] * widest_y + right["pos"]) / PX_PER_MM - (widest["x_mm"] + widest["width_mm"]),
            "slope": right["slope"],
            "start_mm": None if right["start"] < FROM_EDGE_MM * PX_PER_MM else right["start"] / PX_PER_MM,
            "past_bottom_mm": (right["end"] - by_at_right) / PX_PER_MM,  # how far it runs past the bottom line
            "wobble_mm": right["wobble"] / PX_PER_MM,
            "pen_mm": right["width"] / PX_PER_MM,
            "darkness": darkness[1],
            "x_mm": rx_at_bottom / PX_PER_MM,
        },
    }
    idir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(photo, idir / f"source{photo.suffix.lower()}")
    write_json(idir / "identity.json", data)
    write_sheet(idir / "sheet.png", page, all_chars, all_labels, numbers)
    write_preview(pdir, data, idir / "preview.png")

    if problems:
        print("PROBLEM: " + "; ".join(problems))
        print(f"  check: {idir / 'sheet.png'}")
        sys.exit(1)
    report(data, idir)


def report(data, idir):
    b, r = data["border"]["bottom"], data["border"]["right"]
    print(f"Identity block: {len(data['rows'])} rows, {len(data['glyphs'])} characters, "
          f"capitals {data['cap_height_mm']:.1f} mm tall, pen {data['stroke_mm']:.2f} mm")
    for row in data["rows"]:
        print(f"  {row['text']}   (starts {row['x_mm']:.1f} mm from the left, baseline {row['baseline_mm']:.1f} mm from the top)")
    start = "the page edge" if b["start_mm"] is None else f"{b['start_mm']:.1f} mm"
    print(f"  line underneath: {b['y_mm']:.1f} mm from the top, from {start}, slope {np.degrees(np.arctan(b['slope'])):+.2f} deg")
    top = "the page edge" if r["start_mm"] is None else f"{r['start_mm']:.1f} mm"
    print(f"  line down the right: {r['x_mm']:.1f} mm from the left, from {top}")
    if data["rejected"]:
        print(f"  rejected: {', '.join(map(str, data['rejected']))}")
    print(f"  check: {idir / 'sheet.png'}")
    print(f"         {idir / 'preview.png'}")


def show(args):
    idir = profile_dir(args.profile) / "identity"
    data = read_json(idir / "identity.json")
    if not data:
        print(f'NO IDENTITY: profile "{args.profile}" has no identity block. Train one with identity.py train.')
        sys.exit(2)
    report(data, idir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="learn the block from a photo")
    t.add_argument("photo")
    t.add_argument("--line", action="append", default=[], help="one row of the block's text, as written")
    t.add_argument("--corners", help="page corners in the photo when they aren't found: x,y pairs "
                                     "top-left top-right bottom-right bottom-left")
    t.add_argument("--reject", help="numbers of badly cut characters on sheet.png; another of the "
                                    "writer's copies of that character is used instead")
    s = sub.add_parser("show", help="describe the trained block")
    for p in (t, s):
        p.add_argument("--profile", default="default")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else show(args)


if __name__ == "__main__":
    main()
