"""Write transcripts out by hand: lay out text with the profile's glyphs on A4 pages.

Input is one transcript file per source image (format in reference/write.md, including matrices
written as \\pmatrix{a b; c d}, roots as √{…}, and on display lines \\frac{a}{b} and
\\sum_{i=1}^{n}). Output, in --out:

  page_01.png …          final pages (phone-photo look unless --clean)
  handwritten.pdf        with --pdf: every page in order, one exact A4 page each
  qa/page_01_top.jpg …   clean renders, split in halves, for checking the text
  report.json            pages per input, characters drawn from fallbacks, anything not drawn

Per-image mode (default) starts every input on a new page; an input that needs more than one page
gets page_03a.png, page_03b.png. --continuous flows all inputs into one document. --identity writes
the identity block trained with identity.py in the top-left corner of the first page only.

Usage: render.py [--profile NAME] --out DIR [--continuous] [--clean] [--seed N] [--ink blue]
                 [--identity] [--pdf] TRANSCRIPT...
"""

import argparse
import math
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize

from common import A4_MM, DESCENDERS, GREEK_DESCENDERS, PX_PER_MM, apply_settings, profile_dir, read_json, read_settings, write_json
from photo_effect import photo_look

INKS = {"blue": (25, 45, 140), "black": (28, 28, 32)}
LIGATURE_CHANCE = 0.8  # how often a joined run from the profile is used when one exists
SCRIPT_SCALE = 0.62  # superscripts and subscripts, until a template page measures the writer's own
SUP_RAISE_XH = 0.9  # a superscript's baseline sits this far above the line
SUB_DROP_XH = 0.45  # and a subscript's this far below it
SIZE_TOLERANCE = 0.18  # variants more than this much taller or shorter than usual are brought back toward it
MAX_WORD_GAP_XH = 1.7  # word gaps measured on maths-heavy pages run wide; cap them at this many x-heights
MATH_GAP = 0.6  # spacing next to = + − and in display equations, as a share of a word gap
BLANK_LINE = 0.7  # a blank transcript line skips this much of a line
DISPLAY_SPACE = 0.3  # extra room above and below a display equation, as a share of the line pitch
SYMBOL_WORDS = set("=+−-≠∈∉⊆⊂∪∩⊕⊗∘↦⇔→⇒≃<>≤≥×÷±∴:")
NO_LINE_START = set("=≠∈∉⊆⊂⇔")  # relations stay on the line of what they relate
BAND = (0.3, 0.95)  # rows between these heights (in x-heights above the baseline) set letter spacing
BAND_GAP_XH = 0.45  # typical closest distance between neighbouring letters in that band
QUAD_XH = 2.2  # width of \quad, the space between side-by-side equations
BLANK = re.compile(r"^(_{3,})([.,;:]?)$")  # an answer blank such as ______.
BOX_PAD_XH = 0.9  # space between a \boxed answer and the sides of its box
BOX_STYLES = (("two", 0.55), ("four", 0.3), ("one", 0.15))  # how a box gets drawn: two L-shaped strokes
                                                             # is the usual way, then four sides, then one go
MATRIX = re.compile(r"\\([pbv])matrix\{")  # \pmatrix{a b; c d}: ( ), [ ] or | | around the rows
MATRIX_ROW_PITCH = 0.8  # matrix rows sit closer than lines of text, as a share of the line pitch
MATRIX_COL_GAP_XH = 1.0  # space between matrix columns
MATH_CLEAR = (0.5, 1.0)  # a tall line starts this many pitches below the last baseline, and the next
                           # line starts this many pitches below its bottom
MATRIX_AXIS_XH = 0.55  # a matrix is centred this far above the baseline, level with = and +
GLUE_XH = 0.3  # gap between a matrix and a word touching it in the transcript, as in a\pmatrix{…}.
BRACKETS = set("()[]{}")
TRAILING = set(",.;:")  # punctuation after a letter: kept clear of the whole letter, not tucked under a t
SPACED = set("−-–+=±×÷<>≤≥≠")  # thin signs that must never run into the digit or letter beside them
DELIMS = BRACKETS
ANGLE = {"⟨": "<", "⟩": ">", "〈": "<", "〉": ">"}  # angle brackets are made from the writer's < and >
DOTS = {"⋯": "·", "…": "."}  # centred and low ellipses from the writer's own dots
MIRROR = {"∖": "/", "\\": "/"}  # set minus and backslash: the writer's / turned round
BRACKET_GAP = 1.8  # brackets curve around what they hold, so they get more room than letters
MARGINS_MM = {"top": 18.0, "bottom": 16.0, "right": 14.0}
HAND_FONTS = [  # for missing Latin letters, digits and punctuation: closest to handwriting
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Noteworthy.ttc",
]
SYMBOL_FONTS = [  # for Greek and maths: handwriting fonts often draw α like a, so use a plain font
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
BLACKBOARD = {"𝔽": "F", "ℝ": "R", "ℕ": "N", "ℤ": "Z", "ℚ": "Q", "ℂ": "C", "ℙ": "P"}
SUBSTITUTE = {"−": "-", "–": "-", "—": "-", "‐": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "×": "x",
              "∙": "·", "⋅": "·", "⩽": "≤", "⩾": "≥", "⊤": "T", "⟶": "→", "⟹": "⇒", "⟼": "↦", "⟺": "⇔"}
MATH_ITEM = re.compile(r"\\([pbv])matrix\{|\\frac\{|\\(sum|max|min)(?=[_^])")  # tall items of a display line
CAP_FROM_LOWER = set("CKOPSUVWXZ")  # capitals that are just bigger lowercase in most hands
LOWER_FROM_CAP = set("cksouvwxz")


# ---------------------------------------------------------------- transcripts

def parse_transcript(path):
    """Blocks: {"kind": "text" | "display" | "blank", "words": [...]}. One line = one paragraph."""
    blocks = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip().replace("**", "")
        if not line:
            # A leading blank line is kept: in continuous mode it separates this input from the last.
            if not blocks or blocks[-1]["kind"] != "blank":
                blocks.append({"kind": "blank", "words": []})
            continue
        kind = "text"
        if line.startswith("$$") and line.endswith("$$") and len(line) >= 4:
            line, kind = line[2:-2].strip(), "display"
        elif line.startswith("#") and line.lstrip("#").startswith(" "):
            line = line.lstrip("#").strip()
        if kind == "display" and MATH_ITEM.search(line):
            boxed = line.startswith("\\boxed ")
            items = tokenize_math(line[len("\\boxed "):] if boxed else line)
            blocks.append({"kind": "math", "items": items, "boxed": boxed, "words": []})
            continue
        blocks.append({"kind": kind, "words": line.split()})
    while blocks and blocks[-1]["kind"] == "blank":
        blocks.pop()
    return blocks


def tokenize_math(line):
    """Split a display line into words and tall items (matrices, stacked fractions, big operators),
    remembering which ones touch (no space between).

    Returns [{"kind": "word", "text": …} | {"kind": "matrix", "rows": [[entry…]…], "bracket": "p"|"b"|"v"}
             | {"kind": "frac", "num", "den"} | {"kind": "bigop", "op", "lower", "upper"}],
    each with "glue": True when it touches the item before it."""
    def group(i):
        """The {…} starting at line[i] (i at the "{"): (content, index after the "}")."""
        depth, j = 1, i + 1
        while j < len(line) and depth:
            depth += {"{": 1, "}": -1}.get(line[j], 0)
            j += 1
        return line[i + 1:j - 1], j

    def script(i):
        """The ^… or _… argument at line[i]: {…} or one character."""
        if i < len(line) and line[i] == "{":
            return group(i)
        return (line[i], i + 1) if i < len(line) else ("", i)

    items, i = [], 0
    while i < len(line):
        m = MATH_ITEM.search(line, i)
        end = m.start() if m else len(line)
        seg = line[i:end]
        words = seg.split()
        for k, w in enumerate(words):
            glue = k == 0 and bool(items) and not seg[:1].isspace()
            items.append({"kind": "word", "text": w, "glue": glue})
        if not m:
            break
        glue = bool(items) and bool(seg) and not seg[-1:].isspace() or (bool(items) and not seg)
        if m.group(1):
            body, j = group(m.end() - 1)
            rows = [r.split() for r in body.split(";")]
            items.append({"kind": "matrix", "rows": [r for r in rows if r], "bracket": m.group(1), "glue": glue})
        elif m.group(0).startswith("\\frac"):
            num, j = group(m.end() - 1)
            den, j = group(j) if j < len(line) and line[j] == "{" else ("", j)
            items.append({"kind": "frac", "num": num, "den": den, "glue": glue})
        else:
            op = {"sum": "Σ"}.get(m.group(2), m.group(2))
            lower = upper = ""
            j = m.end()
            while j < len(line) and line[j] in "^_":
                arg, k = script(j + 1)
                if line[j] == "_":
                    lower = arg
                else:
                    upper = arg
                j = k
            items.append({"kind": "bigop", "op": op, "lower": lower, "upper": upper, "glue": glue})
        i = j
    return items


def split_braces(items):
    """Peel { and } off the words of a maths line, so they can be drawn as tall as the matrices."""
    out = []
    for it in items:
        if it["kind"] != "word":
            out.append(it)
            continue
        lead, core, trail, punct = re.match(r"^(\{*)(.*?)(\}*)([.,;:]*)$", it["text"]).groups()
        if not lead and not trail:
            out.append(it)
            continue
        parts = [("delim", c) for c in lead] + ([("word", core)] if core else [])
        parts += [("delim", c) for c in trail] + ([("word", punct)] if punct else [])
        for k, (kind, text) in enumerate(parts):
            glue = it["glue"] if k == 0 else True
            out.append({"kind": kind, "char": text, "glue": glue} if kind == "delim"
                       else {"kind": "word", "text": text, "glue": glue})
    return out


def parse_word(word):
    """Split a word into runs [(text, level)]: level +1 for ^{...}, -1 for _{...}, 0 otherwise."""
    runs, buf, i = [], "", 0
    while i < len(word):
        c = word[i]
        if c in "^_" and i + 1 < len(word):
            if buf:
                runs.append((buf, 0))
                buf = ""
            level = 1 if c == "^" else -1
            if word[i + 1] == "{":
                j = word.find("}", i + 2)
                j = len(word) if j == -1 else j
                runs.append((word[i + 2:j], level))
                i = j + 1
            else:
                runs.append((word[i + 1], level))
                i += 2
            continue
        buf += c
        i += 1
    if buf:
        runs.append((buf, 0))
    return [(t, lvl) for t, lvl in runs if t]


# ---------------------------------------------------------------- glyphs

class Glyph:
    """An ink image (0..255) with its baseline row, where its ink starts, and how wide it is."""

    __slots__ = ("img", "baseline", "left", "adv", "kind")

    def __init__(self, img, baseline, left=None, adv=None, kind="profile"):
        self.img = img
        self.baseline = float(baseline)
        if left is None or adv is None:
            cols = np.nonzero(img.max(axis=0) > 30)[0]
            left, adv = (int(cols[0]), int(cols[-1] - cols[0] + 1)) if len(cols) else (0, img.shape[1])
        self.left, self.adv, self.kind = float(left), float(adv), kind


def resized(g, factor):
    h, w = g.img.shape
    img = cv2.resize(g.img, (max(1, round(w * factor)), max(1, round(h * factor))),
                     interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)
    return Glyph(img, g.baseline * factor, g.left * factor, g.adv * factor, g.kind)


def drop_specks(img, xh):
    """Remove small marks standing apart from the glyph's own strokes (dust, a neighbour's dot)."""
    img = np.where(img < 30, 0, img).astype(np.uint8)  # faint haze around the cut-out, not ink
    n, cc, stats, _ = cv2.connectedComponentsWithStats((img > 40).astype(np.uint8), connectivity=8)
    if n <= 2:
        return img
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = areas.sum()
    peak = np.array([img[cc == c].max() for c in range(1, n)])
    keep = np.zeros(n, bool)
    keep[1:] = (areas >= 0.04 * total) & (peak >= 0.6 * peak.max())
    main = np.isin(cc, np.nonzero(keep)[0])
    if not main.any():
        return img
    near = cv2.dilate(main.astype(np.uint8), np.ones((int(0.35 * xh) | 1, int(0.35 * xh) | 1), np.uint8)) > 0
    for c in np.nonzero(~keep[1:])[0] + 1:
        if not (near & (cc == c)).any():  # far from every real stroke: drop it
            img = np.where(cc == c, 0, img)
    return img.astype(np.uint8)


def ink_centre_rows(img):
    """Centre of mass and 5–95% height span of the ink, by rows."""
    mass = img.astype(np.float64).sum(axis=1)
    cum = np.cumsum(mass) / max(mass.sum(), 1)
    rows = np.arange(len(mass))
    return float((rows * mass).sum() / max(mass.sum(), 1)), float(np.searchsorted(cum, 0.95) - np.searchsorted(cum, 0.05) + 1)


def ink_centre_cols(img):
    mass = img.astype(np.float64).sum(axis=0)
    return float((np.arange(len(mass)) * mass).sum() / max(mass.sum(), 1))


def font_has(font, ch):
    """True when the font draws ch itself rather than its 'missing glyph' box."""
    def draw(c):
        im = Image.new("L", (160, 160), 0)
        ImageDraw.Draw(im).text((20, 120), c, font=font, fill=255, anchor="ls")
        return np.array(im)
    missing = draw("\U0010fffd")
    got = draw(ch)
    return got.any() and not np.array_equal(got, missing)


class GlyphBank:
    def __init__(self, pdir, profile):
        self.pdir = pdir
        self.glyphs, self.ligs = profile["glyphs"], profile["ligatures"]
        self.tall = profile.get("tall") or {}
        self.delim_axis = {}
        style = profile["style"]
        self.xh = style["x_height_px"]
        self.stroke_px = style["stroke_mm"] * PX_PER_MM
        cap = style.get("cap_height_mm") or {}
        self.cap_px = cap.get("median", 1.6 * style["x_height_mm"]) * PX_PER_MM
        self.images, self.last, self.fonts = {}, {}, {}
        # Maths pieces from template pages, and how this writer places scripts.
        self.scripts = profile.get("scripts") or {}
        self.radicals = profile.get("radicals") or []
        self.accents = profile.get("accents") or {}
        self.bigops = profile.get("bigops") or {}
        st = profile.get("script_style") or {}
        clamp = lambda v, lo, hi, default: default if v is None else min(hi, max(lo, v))
        self.sup_scale = clamp(st.get("sup_scale"), 0.45, 0.8, SCRIPT_SCALE)
        self.sub_scale = clamp(st.get("sub_scale"), 0.45, 0.8, SCRIPT_SCALE)
        self.sup_raise = clamp(st.get("sup_raise"), 0.55, 1.3, SUP_RAISE_XH)
        self.sup_raise_rel = clamp(st.get("sup_raise_rel"), 0.5, 1.2, None)
        self.sup_short = clamp(st.get("sup_scale_short"), 0.4, 0.85, self.sup_scale)
        self.sup_tall = clamp(st.get("sup_scale_tall"), 0.4, 0.85, self.sup_scale)
        self.sub_short = clamp(st.get("sub_scale_short"), 0.4, 0.85, self.sub_scale)
        self.sub_tall = clamp(st.get("sub_scale_tall"), 0.4, 0.85, self.sub_scale)
        self.heights = {}
        self.sub_drop = clamp(st.get("sub_drop"), 0.25, 0.8, SUB_DROP_XH)
        fs = profile.get("fraction_style") or {}
        self.frac_num_gap = clamp(fs.get("num_gap"), 0.12, 0.8, 0.35)
        self.frac_den_gap = clamp(fs.get("den_gap"), 0.15, 0.9, 0.45)
        self.frac_overhang = clamp(fs.get("overhang"), 0.05, 0.8, 0.25)
        bs = profile.get("bigop_style") or {}
        self.bigop_height = clamp(bs.get("height"), 1.6, 3.6, 2.5)
        self.limit_gap_up = clamp(bs.get("upper_gap"), 0.05, 0.9, 0.25)
        self.limit_gap_down = clamp(bs.get("lower_gap"), 0.05, 0.9, 0.25)
        self.fallback, self.missing = Counter(), Counter()
        self.fallback_how = {}

    def _load(self, meta, key):
        if meta["id"] not in self.images:
            img = drop_specks(cv2.imread(str(self.pdir / meta["file"]), cv2.IMREAD_GRAYSCALE), self.xh)
            baseline = meta["baseline"]
            if key in "acemnorsuvwxz":
                # Short letters never reach far below the line: anything there is a neighbour's tail
                # or an underline caught in the cut-out.
                img = img.copy()
                img[int(baseline + 0.3 * self.xh):] = 0
            if key.isalnum() and not set(key) & (DESCENDERS | GREEK_DESCENDERS):
                # Letters without tails sit on the line: the bottom of the ink is the baseline. This
                # corrects variants whose line was measured a little off on the training page.
                rows = np.nonzero(img.max(axis=1) > 60)[0]
                if len(rows) and abs(rows[-1] + 1 - baseline) < 0.5 * self.xh:
                    baseline = rows[-1] + 1
            if key in DELIMS:
                # Brackets sit at the writer's usual height for that bracket: some cut-outs (from lines
                # with no letters to measure against) had their line placed a little off.
                baseline = ink_centre_rows(img)[0] + self.axis_of(key) * self.xh
            g = Glyph(img, baseline)
            target = self.usual_height(key)
            rows = np.nonzero(img.max(axis=1) > 60)[0]
            if target and len(rows):
                height = baseline - rows[0]
                ratio = height / target
                if not 0.55 < ratio < 1.6:
                    # Far off: the cut-out caught part of another line. Don't use this variant.
                    self.images[meta["id"]] = None
                    return None
                if abs(ratio - 1) > SIZE_TOLERANCE:
                    # A capital or digit written much smaller or bigger than usual reads as a mistake.
                    want = target * (1 + SIZE_TOLERANCE * (1 if ratio > 1 else -1))
                    g = resized(g, want / max(1.0, height))
            self.images[meta["id"]] = g
        return self.images[meta["id"]]

    def plain(self, meta):
        """A template piece (script glyph, root, big Σ, accent) as written: no resizing to the usual height."""
        if meta["id"] not in self.images:
            img = drop_specks(cv2.imread(str(self.pdir / meta["file"]), cv2.IMREAD_GRAYSCALE), self.xh)
            self.images[meta["id"]] = Glyph(img, meta.get("baseline", img.shape[0]))
        return self.images[meta["id"]]

    def median_height(self, ch):
        """How tall ch usually is in the writer's normal writing (ink height in pixels), or None."""
        if ch not in self.heights:
            hs = []
            for m in (self.glyphs.get(ch) or [])[:12]:
                img = cv2.imread(str(self.pdir / m["file"]), cv2.IMREAD_GRAYSCALE)
                rows = np.nonzero(img.max(axis=1) > 60)[0]
                if len(rows):
                    hs.append(rows[-1] - rows[0] + 1)
            self.heights[ch] = float(np.median(hs)) if hs else None
        return self.heights[ch]

    def script_glyph(self, ch, rng):
        """One of the writer's own small script characters, or None."""
        pool = self.scripts.get(ch) or self.scripts.get(SUBSTITUTE.get(ch, ""))
        if not pool:
            return None
        meta = rng.choice([m for m in pool if m["id"] != self.last.get("s" + ch)] or pool)
        self.last["s" + ch] = meta["id"]
        return self.plain(meta)

    def thickened(self, g, scale):
        """A glyph drawn smaller keeps the pen's width: shrinking alone would make the strokes thin."""
        k = int(round(self.stroke_px * (1 - scale)))
        if scale >= 0.92 or k < 2:
            return g
        img = cv2.dilate(g.img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1)))
        return Glyph(img, g.baseline, kind=g.kind)

    def axis_of(self, key):
        """Median height of the ink centre of key's variants above their baseline, in x-heights."""
        if key not in self.delim_axis:
            rel = []
            for m in self.glyphs.get(key, []):
                img = cv2.imread(str(self.pdir / m["file"]), cv2.IMREAD_GRAYSCALE)
                rel.append((m["baseline"] - ink_centre_rows(img)[0]) / max(1.0, m["x_height_px"]))
            self.delim_axis[key] = float(np.median(rel)) if rel else 0.75
        return self.delim_axis[key]

    def usual_height(self, key):
        """Expected height above the baseline for single capitals, digits and short letters."""
        if len(key) != 1:
            return None
        if key.isupper() and key not in "JQ" or key.isdigit():
            return self.cap_px
        if key in "acemnorsuvwxz":
            return self.xh
        return None

    def _pick(self, key, pool, rng):
        usable = [m for m in pool if self.images.get(m["id"], True) is not None]
        while usable:
            choices = [m for m in usable if m["id"] != self.last.get(key)] or usable
            meta = rng.choice(choices)
            g = self._load(meta, key)
            if g is not None:
                self.last[key] = meta["id"]
                return g
            usable.remove(meta)
        return self._load(pool[0], key) or Glyph(np.zeros((4, 4), np.uint8), 2)

    def _note(self, ch, how):
        self.fallback[ch] += 1
        self.fallback_how[ch] = how

    def take(self, text, i, rng):
        """The glyph for text[i:], and how many characters it covers (None glyph = nothing to draw)."""
        for n in (4, 3, 2):
            chunk = text[i:i + n]
            if len(chunk) == n and chunk in self.ligs and rng.random() < LIGATURE_CHANCE:
                return self._pick(chunk, self.ligs[chunk], rng), n
        return self.single(text[i], rng), 1

    def single(self, ch, rng):
        c = ch if ch in self.glyphs else BLACKBOARD.get(ch, ch)  # the writer's own ℝ, if trained
        if c != ch:
            self._note(ch, f"written as {c}")
        if c in self.glyphs:
            return self._pick(c, self.glyphs[c], rng)
        alt = SUBSTITUTE.get(c)
        if alt and alt in self.glyphs:
            if c not in ("−", "–", "—", "‐"):  # a hyphen for a minus sign is what people write anyway
                self._note(c, f"written as {alt}")
            return self._pick(alt, self.glyphs[alt], rng)
        if c == "≠" and "=" in self.glyphs and "/" in self.glyphs:
            self._note(c, "built from = and /")
            return self._not_equal(rng)
        if c in CAP_FROM_LOWER and c.lower() in self.glyphs:
            self._note(c, f"your {c.lower()}, enlarged")
            g = self._pick(c.lower(), self.glyphs[c.lower()], rng)
            return resized(g, self.cap_px / self.xh)
        if c in LOWER_FROM_CAP and c.upper() in self.glyphs:
            self._note(c, f"your {c.upper()}, reduced")
            g = self._pick(c.upper(), self.glyphs[c.upper()], rng)
            return resized(g, self.xh / self.cap_px)
        if c in ANGLE and ANGLE[c] in self.glyphs:
            self._note(c, f"your {ANGLE[c]}, made taller and narrower")
            return self.angle_bracket(c, rng)
        if c in DOTS and DOTS[c] in self.glyphs:
            self._note(c, f"three of your {DOTS[c]}")
            return self.dot_row(DOTS[c], rng)
        if c in MIRROR and MIRROR[c] in self.glyphs:
            self._note(c, f"your {MIRROR[c]}, mirrored")
            return self.mirrored(c, rng)
        if c in "{}":
            self._note(c, "drawn as a pen stroke at your pen width")
            # As tall as the writer's parentheses: from above the capitals to below the line.
            return brace_glyph(c, self.cap_px + 1.0 * self.xh, self.xh, self.stroke_px, rng,
                               above=self.cap_px + 0.35 * self.xh)
        g = self._font_glyph(c)
        if g is not None:
            self._note(c, "fallback font, redrawn at your pen width")
            return g
        self.missing[c] += 1
        return None

    def angle_bracket(self, ch, rng):
        """⟨ or ⟩ from the writer's < or >: as tall as their parentheses, much narrower."""
        g = self._pick(ANGLE[ch], self.glyphs[ANGLE[ch]], rng)
        rows = np.nonzero(g.img.max(axis=1) > 40)[0]
        cols = np.nonzero(g.img.max(axis=0) > 40)[0]
        ink = g.img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        height = self.cap_px + 0.7 * self.xh
        width = max(4, int(round(0.65 * self.xh * rng.uniform(0.9, 1.1))))
        img = cv2.resize(ink, (width, int(round(height))), interpolation=cv2.INTER_CUBIC)
        pad = 4
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        # Centred where the writer's parentheses sit.
        return Glyph(img, pad + height / 2 + self.axis_of("(") * self.xh, kind="composite")

    def mirrored(self, ch, rng):
        """∖ or \\ from one of the writer's / strokes, flipped left to right, about as tall as a capital."""
        g = self._pick(MIRROR[ch], self.glyphs[MIRROR[ch]], rng)
        g = Glyph(cv2.flip(g.img, 1), g.baseline, kind="composite")
        rows = np.nonzero(g.img.max(axis=1) > 40)[0]
        span = rows[-1] - rows[0] + 1 if len(rows) else g.img.shape[0]
        return resized(g, self.cap_px / max(span, 1))

    def dot_row(self, ch, rng):
        """⋯ or … as three of the writer's dots, evenly spaced, sharing one baseline."""
        dots = [self._pick(ch, self.glyphs[ch], rng) for _ in range(3)]
        gap = 0.38 * self.xh
        above = max(d.baseline for d in dots)
        below = max(d.img.shape[0] - d.baseline for d in dots)
        width = int(sum(d.adv for d in dots) + 2 * gap) + 8
        canvas = np.zeros((int(np.ceil(above + below)) + 2, width), np.uint8)
        x = 4.0
        for d in dots:
            top, left = int(round(above - d.baseline)), int(round(x - d.left))
            h, w = d.img.shape
            region = canvas[top:top + h, max(0, left):left + w]
            np.maximum(region, d.img[:region.shape[0], max(0, -left):max(0, -left) + region.shape[1]], out=region)
            x += d.adv + gap
        return Glyph(canvas, above, kind="composite")

    def tall_delim(self, ch, height, rng):
        """One of the writer's own tall brackets or braces, resized to height (tight crop, uint8).

        Square brackets keep their hooks and stretch only the stem; curved ones are scaled whole,
        with the width growing more slowly than the height so a stretched bracket stays slender.
        None when the profile has no tall version of ch."""
        pool = self.tall.get(ch)
        if not pool:
            return None
        choices = [m for m in pool if m["id"] != self.last.get("tall" + ch)] or pool
        meta = rng.choice(choices)
        self.last["tall" + ch] = meta["id"]
        img = cv2.imread(str(self.pdir / meta["file"]), cv2.IMREAD_GRAYSCALE)
        rows = np.nonzero(img.max(axis=1) > 40)[0]
        cols = np.nonzero(img.max(axis=0) > 40)[0]
        ink = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        h, w = ink.shape
        height = max(4, int(round(height)))
        if ch in "[]" and height > h:
            cap = int(0.2 * h)
            stem = cv2.resize(ink[cap:h - cap], (w, height - 2 * cap), interpolation=cv2.INTER_LINEAR)
            return np.vstack([ink[:cap], stem, ink[h - cap:]])
        factor = height / h
        width = max(1, int(round(w * factor ** 0.5)))
        return cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)

    def tall_bracket(self, ch, height, rng):
        """One of the writer's own [ or ] made taller: the hooks at the top and bottom stay as
        written and only the stem between them is stretched. None if the profile has none."""
        if ch not in self.glyphs:
            return None
        g = self._pick(ch, self.glyphs[ch], rng)
        rows = np.nonzero(g.img.max(axis=1) > 40)[0]
        cols = np.nonzero(g.img.max(axis=0) > 40)[0]
        ink = g.img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        h, w = ink.shape
        height = int(round(height))
        if height <= h:
            return cv2.resize(ink, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)
        cap = int(0.3 * h)
        stem = cv2.resize(ink[cap:h - cap], (w, height - 2 * cap), interpolation=cv2.INTER_LINEAR)
        tall = np.vstack([ink[:cap], stem, ink[h - cap:]])
        # Serifs on a tall bracket are drawn no wider than on a short one, a little narrower if anything.
        squeeze = rng.uniform(0.8, 0.95)
        return cv2.resize(tall, (max(1, round(w * squeeze)), height), interpolation=cv2.INTER_AREA)

    def blank_line(self, n, rng):
        """An answer blank: one of the writer's hyphens stretched to n underscores, on the baseline."""
        if "-" not in self.glyphs:
            return None
        g = self._pick("-", self.glyphs["-"], rng)
        rows = np.nonzero(g.img.max(axis=1) > 30)[0]
        cols = np.nonzero(g.img.max(axis=0) > 30)[0]
        ink = g.img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        length = max(ink.shape[1], int(n * 0.9 * self.xh))
        line = cv2.resize(ink, (length, ink.shape[0]), interpolation=cv2.INTER_CUBIC)
        pad = 4
        img = cv2.copyMakeBorder(line, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        return Glyph(img, pad + line.shape[0] + 0.05 * self.xh, kind="composite")

    def _not_equal(self, rng):
        """≠ from one of the writer's = signs with one of their / strokes across its middle."""
        eq = self._pick("=", self.glyphs["="], rng)
        sl = self._pick("/", self.glyphs["/"], rng)
        e_mid, e_span = ink_centre_rows(eq.img)
        _, s_span = ink_centre_rows(sl.img)
        sl = resized(sl, min(1.7 * e_span, 1.2 * self.xh) / max(1.0, s_span))
        s_mid, _ = ink_centre_rows(sl.img)
        sh, sw = sl.img.shape
        eh, ew = eq.img.shape
        canvas = np.zeros((eh + 2 * sh, ew + 2 * sw), np.uint8)
        ey, ex = sh, sw
        canvas[ey:ey + eh, ex:ex + ew] = eq.img
        sy = int(round(ey + e_mid - s_mid))
        sx = int(round(ex + ink_centre_cols(eq.img) - ink_centre_cols(sl.img)))
        region = canvas[sy:sy + sh, sx:sx + sw]
        np.maximum(region, sl.img, out=region)
        return Glyph(canvas, ey + eq.baseline, kind="composite")

    def _font_glyph(self, ch):
        if ch in self.fonts:
            return self.fonts[ch]
        g = None
        fonts = HAND_FONTS + SYMBOL_FONTS if ch.isascii() else SYMBOL_FONTS + HAND_FONTS
        for path in fonts:
            if not Path(path).exists():
                continue
            probe = ImageFont.truetype(path, 200)
            if not font_has(probe, ch):
                continue
            xh_font = -probe.getbbox("x", anchor="ls")[1]
            ref = self.xh if not ch.isupper() else self.cap_px * xh_font / max(1, -probe.getbbox("H", anchor="ls")[1])
            font = ImageFont.truetype(path, max(8, int(200 * ref / max(1, xh_font))))
            l, t, r, b = font.getbbox(ch, anchor="ls")
            pad = int(self.stroke_px) + 4
            im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
            ImageDraw.Draw(im).text((pad - l, pad - t), ch, font=font, fill=255, anchor="ls")
            mask = np.array(im) > 127
            skel = skeletonize(mask).astype(np.uint8) * 255
            k = max(3, int(round(self.stroke_px)) | 1)
            pen = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            pen = cv2.GaussianBlur(pen, (3, 3), 0.8)
            g = Glyph(pen, pad - t, kind="font")
            break
        self.fonts[ch] = g
        return g


def ink_extent(placed):
    """(top, bottom, left, right) of the ink of built glyphs; top and bottom relative to the baseline."""
    tops, bottoms, lefts, rights = [], [], [], []
    for g, x, dy in placed:
        rows = np.nonzero(g.img.max(axis=1) > 40)[0]
        if not len(rows):
            continue
        tops.append(dy - g.baseline + rows[0])
        bottoms.append(dy - g.baseline + rows[-1])
        lefts.append(x)
        rights.append(x + g.adv)
    if not tops:
        return 0.0, 0.0, 0.0, 0.0
    return min(tops), max(bottoms), min(lefts), max(rights)


def split_radical(word):
    """(before, under the root, after) for the first √ in a word that has something under it:
    √{x+y}, √(…), √⟨…⟩, or the number, letter or −1 right after it with its scripts."""
    i = word.find("√")
    if i < 0 or i == len(word) - 1:
        return None
    j = i + 1
    close = {"{": "}", "(": ")", "⟨": "⟩"}
    if word[j] in close:
        depth, k = 0, j
        while k < len(word):
            if word[k] == word[j]:
                depth += 1
            elif word[k] == close[word[j]]:
                depth -= 1
                if depth == 0:
                    break
            k += 1
        inner = word[j + 1:k] if word[j] == "{" else word[j:k + 1]
        return word[:i], inner, word[k + 1:]
    k = j
    if word[k] in "−-":
        k += 1
    while k < len(word) and (word[k].isalnum() or word[k] in "."):
        k += 1
    while k < len(word) and word[k] in "^_":
        if k + 1 < len(word) and word[k + 1] == "{":
            k = word.find("}", k) + 1 or len(word)
        else:
            k += 2
    if k == j:
        return None
    return word[:i], word[j:k], word[k:]


def group_end(built, start):
    """Index of the built word where the bracketed group opened at built[start] closes."""
    depth = 0
    for k in range(start, len(built)):
        for ch in built[k][2]:
            if ch in "([{⟨":
                depth += 1
            elif ch in ")]}⟩" and depth:
                depth -= 1
        if depth == 0:
            return k
    return len(built) - 1


def lay_ink(alpha, a, x0, y0):
    """Lay ink coverage a (0..1) onto alpha with its top-left corner at (x0, y0)."""
    h, w = a.shape
    rx0, ry0, rx1, ry1 = max(0, x0), max(0, y0), min(alpha.shape[1], x0 + w), min(alpha.shape[0], y0 + h)
    if rx1 <= rx0 or ry1 <= ry0:
        return
    a = a[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0]
    region = alpha[ry0:ry1, rx0:rx1]
    region[:] = 1 - (1 - region) * (1 - a)


def pen_image(pts, widths):
    """A pen stroke through pts (x, y), with a width per point, as its own ink image (0..1).

    Returns (image, x of the image's left edge, y of its top edge)."""
    pts = np.asarray(pts, np.float64)
    pad = int(max(widths)) + 4
    x0, y0 = int(math.floor(pts[:, 0].min())) - pad, int(math.floor(pts[:, 1].min())) - pad
    x1, y1 = int(math.ceil(pts[:, 0].max())) + pad, int(math.ceil(pts[:, 1].max())) + pad
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    # Dense dots give a stroke whose width can change along its length (pens taper at the ends).
    for (ax, ay), (bx, by), wa, wb in zip(pts[:-1], pts[1:], widths[:-1], widths[1:]):
        n = max(1, int(math.hypot(bx - ax, by - ay)))
        for f in np.arange(n + 1) / n:
            x, y, w = ax + (bx - ax) * f, ay + (by - ay) * f, wa + (wb - wa) * f
            cv2.circle(canvas, (int(round((x - x0) * 4)), int(round((y - y0) * 4))), max(1, int(round(w * 2))),
                       255, -1, cv2.LINE_AA, shift=2)
    return cv2.GaussianBlur(canvas, (3, 3), 0.7).astype(np.float32) / 255, x0, y0


def cubic(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.array(p, np.float64) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3).tolist()


def brace_glyph(ch, height, xh, stroke_px, rng, above):
    """A curly brace written in one stroke: hook, stem, a point in the middle, stem, hook.

    height is its full height; above is how far it reaches above the baseline."""
    w = 0.5 * xh + 0.05 * height
    j = lambda k=0.05: rng.gauss(0, k * w)
    h, stem, mid = height, 0.5 * w, 0.5 * height + rng.gauss(0, 0.02 * height)
    pts = []
    pts += cubic((w + j(), 0), (0.55 * w + j(), 0), (stem + j(), 0.04 * h), (stem, 0.14 * h))
    pts += cubic((stem, 0.14 * h), (stem + j(), 0.25 * h), (stem + j(), 0.33 * h), (stem, mid - 0.09 * h))
    pts += cubic((stem, mid - 0.09 * h), (stem, mid - 0.03 * h), (0.25 * w, mid), (j(), mid))
    pts += cubic((0, mid), (0.25 * w, mid), (stem, mid + 0.03 * h), (stem, mid + 0.09 * h))
    pts += cubic((stem, mid + 0.09 * h), (stem + j(), 0.67 * h), (stem + j(), 0.75 * h), (stem, 0.86 * h))
    pts += cubic((stem, 0.86 * h), (stem + j(), 0.96 * h), (0.55 * w + j(), h), (w + j(), h))
    slant = math.tan(math.radians(rng.gauss(4, 2)))  # a hand-drawn brace leans a little
    pts = [((w - x if ch == "}" else x) + (0.5 * h - y) * slant, y) for x, y in pts]
    n = len(pts)
    t = stroke_px
    widths = [t * (0.45 + 0.55 * math.sin(math.pi * k / (n - 1))) * (0.85 if abs(k - n / 2) < n / 12 else 1.0)
              for k in range(n)]  # tapered ends, a little thinner at the point
    img, x0, y0 = pen_image(pts, widths)
    img = (img * 255).astype(np.uint8)
    return Glyph(img, above - y0, kind="drawn")


def jittered(g, rng, scale=1.0):
    """A slightly resized and rotated copy, turning about the middle of its baseline."""
    s = scale * (1 + min(0.06, max(-0.06, rng.gauss(0, 0.03))))
    angle = min(3.0, max(-3.0, rng.gauss(0, 1.2)))
    h, w = g.img.shape
    pad = int(0.12 * max(h, w) * max(1.0, s)) + 3
    big = cv2.copyMakeBorder(g.img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cx, cy = pad + g.left + g.adv / 2, pad + g.baseline
    m = cv2.getRotationMatrix2D((cx, cy), angle, s)
    out = cv2.warpAffine(big, m, (big.shape[1], big.shape[0]), flags=cv2.INTER_LINEAR)
    return Glyph(out, cy, cx - s * g.adv / 2, s * g.adv, g.kind)


# ---------------------------------------------------------------- layout

class Writer:
    """Lays words out line by line, page by page, with the writer's own spacing and wobble."""

    def __init__(self, bank, style, rng, page_mm=A4_MM):
        self.bank, self.rng = bank, rng
        self.w, self.h = int(round(page_mm[0] * PX_PER_MM)), int(round(page_mm[1] * PX_PER_MM))
        xh = bank.xh
        lm = (style.get("left_margin_mm") or {}).get("median", 20.0)
        self.left = min(26.0, max(14.0, lm)) * PX_PER_MM
        self.right = self.w - MARGINS_MM["right"] * PX_PER_MM
        self.top = MARGINS_MM["top"] * PX_PER_MM + bank.cap_px
        self.bottom = self.h - MARGINS_MM["bottom"] * PX_PER_MM
        self.pitch = (style.get("line_pitch_mm") or {}).get("median", 9.0) * PX_PER_MM
        gap = style.get("letter_gap_mm") or {"median": 0.4, "std": 0.4}
        # Gaps between separate pieces of one word; joined letters are already inside ligatures.
        self.gap_mu, self.gap_sd = 0.5 * gap["median"] * PX_PER_MM, 0.2 * gap["std"] * PX_PER_MM
        word = style.get("word_gap_mm") or {"median": 3.0, "std": 1.0}
        self.space_mu = min(word["median"] * PX_PER_MM, MAX_WORD_GAP_XH * xh)
        self.space_sd = min(0.3 * word["std"] * PX_PER_MM, 0.25 * self.space_mu)
        self.xh = xh
        self.slant = float(style.get("slant_deg") or 0.0)
        self.first_top = None  # first baseline of the first page, when something sits above the text
        self.pages = []  # [{"alpha": array, "lines": n, "label": str, "source": set, "top": first baseline}]
        self.page = None
        self.y = None

    # -- spacing samples
    def letter_gap(self):
        return min(0.8 * PX_PER_MM, max(-0.3 * PX_PER_MM, self.rng.gauss(self.gap_mu, self.gap_sd)))

    def band_gap(self):
        """Space between neighbouring letters where they come closest, in the middle band."""
        return max(0.08 * self.xh, self.rng.gauss(BAND_GAP_XH * self.xh, 0.08 * self.xh))

    def word_gap(self):
        return min(1.5 * self.space_mu, max(0.65 * self.space_mu, self.rng.gauss(self.space_mu, self.space_sd)))

    # -- words
    def band_edges(self, g, shift, full=False):
        """Left and right ink edge of each row in the middle band of the letters, keyed by the row's
        height above the word's baseline. Tails along the baseline and ascender tops are left out, so
        an exit stroke can run under the next letter the way it does in real joined writing.
        With full, every row counts: for brackets and punctuation, which must not overlap at all."""
        top = 0 if full else int(round(g.baseline - BAND[1] * self.xh))
        bottom = g.img.shape[0] if full else int(round(g.baseline - BAND[0] * self.xh))
        edges = {}
        for r in range(max(0, top), min(g.img.shape[0], bottom)):
            cols = np.nonzero(g.img[r] > 60)[0]
            if len(cols):
                edges[int(round(r - g.baseline + shift))] = (cols[0] - g.left, cols[-1] - g.left)
        return edges

    def build_word(self, word):
        """[(glyph, x of its ink start, baseline shift)], total width."""
        if word == "\\quad":
            return [], QUAD_XH * self.xh
        blank = BLANK.match(word)
        if blank:
            g = self.bank.blank_line(len(blank.group(1)), self.rng)
            if g is not None:
                placed, x = [(g, 0.0, 0.0)], g.adv
                for ch in blank.group(2):
                    p = self.bank.single(ch, self.rng)
                    if p is not None:
                        x += 0.15 * self.xh
                        placed.append((jittered(p, self.rng), x, 0.0))
                        x += p.adv
                return placed, x
        root = split_radical(word)
        if root:
            return self.join_parts([self.build_word(root[0]) if root[0] else None, self.build_radical(root[1]),
                                    self.build_word(root[2]) if root[2] else None])
        placed, prev = [], None  # prev: (x of ink start, glyph, band edges, text)
        right = 0.0
        base_h = None  # height of the last full-size glyph above the line: the letter scripts belong to
        last_level, script_x, before_stack = 0, None, None
        base_glyph = None  # (x, glyph, dy) of the last full-size glyph: scripts tuck in beside its outline
        for text, level in parse_word(word):
            # Scripts are sized and raised for the letter they belong to: smaller after x than after A.
            short = base_h is not None and base_h < 1.3 * self.xh
            bank = self.bank
            scale = {1: bank.sup_short if short else bank.sup_tall,
                     -1: bank.sub_short if short else bank.sub_tall, 0: 1.0}[level]
            raise_px = bank.sup_raise * self.xh
            if level == 1 and bank.sup_raise_rel and base_h:
                raise_px = min(1.8 * self.xh, max(0.7 * self.xh, bank.sup_raise_rel * base_h))
            shift = {1: -raise_px, -1: bank.sub_drop * self.xh, 0: 0.0}[level]
            if placed:
                if level and last_level and level != last_level and script_x is not None:
                    # A power over an index on one letter (u_i^T, σ_i^2): one above the other.
                    before_stack, right = right, script_x
                else:
                    right += 0.12 * self.xh  # a script starts after the letter it belongs to, not under it
                if level == 0 and text[:1] in ",.;:":
                    right += 0.25 * self.xh  # a comma after q₁ must not read as another subscript
                prev = None
            run_start = None
            tuck = level != 0 and last_level == 0 and base_glyph is not None
            i = 0
            while i < len(text):
                if unicodedata.combining(text[i]):  # a bar or hat over the letter just written
                    if placed:
                        placed.append(self.accent(text[i], placed[-1]))
                    i += 1
                    continue
                own = self.bank.script_glyph(text[i], self.rng) if level else None
                if own is not None:
                    g, n = own, 1
                else:
                    g, n = self.bank.take(text, i, self.rng) if not level else (self.bank.single(text[i], self.rng), 1)
                chars = text[i:i + n]
                i += n
                if g is None:  # nothing to draw: leave the room a letter would take
                    right += 0.7 * self.xh * scale + self.letter_gap()
                    prev = None
                    continue
                if own is not None:
                    # The writer's own small character, at the size they write it after this letter.
                    normal, rows = self.bank.median_height(chars), np.nonzero(g.img.max(axis=1) > 60)[0]
                    f = 1.0
                    if normal and len(rows):
                        f = min(1.6, max(0.6, scale * normal / (rows[-1] - rows[0] + 1)))
                    j = jittered(g, self.rng, f)
                else:
                    j = jittered(g, self.rng, scale)
                    if level:
                        j = self.bank.thickened(j, scale)
                dy = shift + self.rng.gauss(0, 0.04 * self.xh)
                edges = self.band_edges(j, dy)
                if prev is None and tuck:
                    # The first character of a power or index sits just clear of the letter's outline
                    # at its own height: beside the peak of A, under the arm of F, not after the whole letter.
                    bx, bg, bdy = base_glyph
                    fe, bfe = self.band_edges(j, dy, full=True), self.band_edges(bg, bdy, full=True)
                    rows = set(fe) & set(bfe)
                    x = right + self.letter_gap()
                    if rows:
                        x = min(x, bx + max(bfe[r][1] - fe[r][0] for r in rows) + 0.3 * self.xh)
                    # Only a power after a tall letter (A, T, ℝ) tucks in much; others start near its end.
                    tall = base_h is not None and base_h >= 1.3 * self.xh
                    x = max(x, bx + bg.adv - (0.35 if level == 1 and tall else 0.1) * self.xh)
                    tuck = False
                elif prev is None:
                    x = right + (self.letter_gap() if placed else 0.0)
                else:
                    px, pg, pedges, ptext, pdy = prev
                    common = set(edges) & set(pedges)
                    # Brackets are as tall as the line, punctuation sits low and − + = sit in a thin
                    # band: keep them clear of loops, tails and crossbars too, so f) can't read as ß,
                    # 1.5 as 15 or −3 as 3.
                    clear = (BRACKETS & set(ptext[-1:] + chars[:1]) or chars[:1] in TRAILING
                             or ptext[-1:] in TRAILING or (SPACED & set(ptext[-1:] + chars[:1])))
                    gap = self.band_gap() * scale * (BRACKET_GAP if BRACKETS & set(ptext[-1:] + chars[:1]) else 1)
                    if len(common) >= 3:
                        # Optical spacing: the closest approach in the band gets the writer's gap.
                        x = px + max(pedges[r][1] - edges[r][0] for r in common) + gap
                        x = max(x, px + 0.45 * pg.adv)
                    else:
                        x = right + self.letter_gap() * scale
                    if clear:
                        fe, pfe = self.band_edges(j, dy, full=True), self.band_edges(pg, pdy, full=True)
                        rows = set(fe) & set(pfe)
                        if rows:
                            x = max(x, px + max(pfe[r][1] - fe[r][0] for r in rows) + gap)
                placed.append((j, x, dy))
                right = max(right, x + j.adv)
                prev = (x, j, edges, chars, dy)
                if run_start is None:
                    run_start = x
                if level == 0:
                    rows = np.nonzero(j.img.max(axis=1) > 60)[0]
                    base_h = (j.baseline - rows[0] - dy) if len(rows) else base_h
                    base_glyph = (x, j, dy)
            if before_stack is not None:
                right, before_stack = max(right, before_stack), None
            script_x = run_start if level else None
            last_level = level
        return placed, right

    def build_phrase(self, text):
        """A few words inside a fraction, root or limit, at maths spacing."""
        words = text.split()
        if len(words) <= 1:
            return self.build_word(text.strip())
        placed, x = [], 0.0
        for k, w in enumerate(words):
            pl, ww = self.build_word(w)
            if k:
                x += self.word_gap() * MATH_GAP
            placed += [(g, x + dx, dy) for g, dx, dy in pl]
            x += ww
        return placed, x

    def join_parts(self, parts):
        """Several built pieces of one word side by side, at the writer's letter spacing."""
        placed, x = [], 0.0
        for part in parts:
            if not part:
                continue
            pl, w = part
            if placed:
                x += self.letter_gap() + 0.1 * self.xh
            placed += [(g, x + dx, dy) for g, dx, dy in pl]
            x += w
        return placed, x

    def accent(self, mark, base):
        """A bar, hat or dot over the glyph just placed: the writer's own from a template, or a pen stroke."""
        g, x, dy = base
        top, bottom, left, right = ink_extent([base])
        cx = (left + right) / 2
        pool = self.bank.accents.get(mark)
        if pool:
            meta = self.rng.choice(pool)
            a = self.bank.plain(meta)
            w = max(0.6 * self.xh, (right - left) * meta.get("rel_w", 1.0))
            f = w / max(1.0, a.adv)
            a = resized(a, min(1.4, max(0.6, f)))
            gap = meta.get("gap", 0.25) * self.xh
            ax = cx + meta.get("dx", 0.0) * self.xh - a.adv / 2
            ink_rows = np.nonzero(a.img.max(axis=1) > 40)[0]
            a_bottom = (ink_rows[-1] if len(ink_rows) else a.img.shape[0]) - a.baseline
            return (a, ax, top - gap - a_bottom)
        w = max(0.7 * self.xh, right - left + 0.1 * self.xh)
        y = -max(0.35 * self.xh, 0.0)
        t = 0.9 * self.bank.stroke_px
        if mark == "\u0302":  # hat
            pts = [(0, 0.35 * self.xh), (w / 2, 0), (w, 0.35 * self.xh)]
        elif mark == "\u0307":  # dot
            pts = [(w / 2 - 0.05 * self.xh, 0), (w / 2 + 0.05 * self.xh, 0.02 * self.xh)]
        else:  # bar (macron, overline) and anything else
            pts = [(0, 0.02 * self.xh * self.rng.uniform(-1, 1)), (w, 0.03 * self.xh * self.rng.uniform(-1, 1))]
        img, x0, y0 = pen_image(pts, [t] * len(pts))
        a = Glyph((img * 255).astype(np.uint8), -y0)
        return (a, cx - w / 2, top - 0.3 * self.xh - a.img.shape[0] + a.baseline + y)

    def build_radical(self, radicand):
        """√ over an expression: the writer's own root sign, its hook sized to the expression and its
        bar stretched over all of it."""
        rp, rw = self.build_phrase(radicand)
        top, bottom, _, _ = ink_extent(rp) if rp else (-self.xh, 0.0, 0.0, rw)
        bar_at = top - 0.28 * self.xh  # the bar: just above the expression
        foot = min(max(bottom, 0.0), 0.35 * self.xh) + 0.15 * self.xh  # the point: near the line, tails or not
        gap = 0.15 * self.xh
        if self.bank.radicals:
            meta = self.rng.choice(self.bank.radicals)
            img = self.bank.plain(meta).img
            split = int(meta["split_x"])
            hook, bar = img[:, :split + 2], img[:, split:]
            rows = np.nonzero(hook.max(axis=1) > 50)[0]
            brow = np.nonzero(bar.max(axis=1) > 50)[0]
            if len(rows) and len(brow):
                bar_y = float(meta["bar_y"])
                f = (foot - bar_at) / max(1.0, rows[-1] - bar_y)
                f = min(1.5, max(0.6, f))
                hook = cv2.resize(hook, (max(2, int(hook.shape[1] * f ** 0.75)), max(2, int(hook.shape[0] * f))),
                                  interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
                hook_g = Glyph(hook, bar_y * f - bar_at, kind="profile")
                # Keep the bar's own thickness; stretch only its length over the expression.
                strip = bar[max(0, brow[0] - 3):brow[-1] + 4]
                length = int(round(gap + rw + 0.3 * self.xh))
                strip = cv2.resize(strip, (max(4, length), strip.shape[0]), interpolation=cv2.INTER_LINEAR)
                bar_g = Glyph(strip, (bar_y - max(0, brow[0] - 3)) - bar_at, kind="profile")
                hw = hook_g.left + hook_g.adv
                placed = [(hook_g, 0.0, 0.0), (bar_g, hw - 3 - 0, 0.0)]
                placed += [(g, hw + gap + dx, dy) for g, dx, dy in rp]
                return placed, hw + gap + rw + 0.3 * self.xh
        # No root sign from a template: the writer's √, then a pen-stroke bar from its top.
        g = self.bank.single("√", self.rng)
        if g is None:
            return rp, rw
        rows = np.nonzero(g.img.max(axis=1) > 50)[0]
        h = (rows[-1] - rows[0]) if len(rows) else g.img.shape[0]
        f = min(1.8, max(0.6, (foot - bar_at) / max(1.0, h)))
        gs = resized(g, f)
        rows = np.nonzero(gs.img.max(axis=1) > 50)[0]
        gs = Glyph(gs.img, rows[0] - bar_at if len(rows) else gs.baseline, kind=gs.kind)
        hw = gs.left + gs.adv
        length = gap + rw + 0.3 * self.xh
        t = 0.9 * self.bank.stroke_px
        img, x0, y0 = pen_image([(0, 0), (length * 0.5, self.rng.gauss(0, 0.03 * self.xh)), (length, 0)], [t, t, t * 0.8])
        bar_g = Glyph((img * 255).astype(np.uint8), -y0 - bar_at)
        placed = [(gs, 0.0, 0.0), (bar_g, hw - 4, 0.0)] + [(gg, hw + gap + dx, dy) for gg, dx, dy in rp]
        return placed, hw + gap + rw + 0.3 * self.xh

    def build_frac(self, num, den):
        """A stacked fraction for a display line: numerator over a pen-stroke bar over denominator."""
        npl, nw = self.build_phrase(num)
        dpl, dw = self.build_phrase(den)
        ntop, nbot, _, _ = ink_extent(npl) if npl else (-self.xh, 0.0, 0, 0)
        dtop, dbot, _, _ = ink_extent(dpl) if dpl else (-self.xh, 0.0, 0, 0)
        axis = MATRIX_AXIS_XH * self.xh
        over = self.bank.frac_overhang * self.xh
        width = max(nw, dw) + 2 * over
        num_base = -axis - self.bank.frac_num_gap * self.xh - nbot  # relative to the line's baseline
        den_base = -axis + self.bank.frac_den_gap * self.xh - dtop
        above = -(num_base + ntop) + 0.1 * self.xh
        below = den_base + dbot + 0.1 * self.xh
        wob = [self.rng.gauss(0, 0.04 * self.xh) for _ in range(3)]

        def draw(x, baseline):
            y = baseline - axis
            t = 0.9 * self.bank.stroke_px
            self.stroke([(x, y + wob[0]), (x + width / 2, y + wob[1]), (x + width, y + wob[2])],
                        [t * 0.8, t, t * 0.75], self.rng.uniform(0.85, 0.95))
            for pl, w, b in ((npl, nw, num_base), (dpl, dw, den_base)):
                x0 = x + (width - w) / 2
                for g, dx, dy in pl:
                    self.paste(g, x0 + dx, baseline + b + dy)

        return {"width": width, "above": above, "below": below, "draw": draw}

    def build_bigop(self, op, lower, upper):
        """Σ (written large) or max/min, with limits below and above as the writer places them."""
        if op == "Σ":
            pool = self.bank.bigops.get("Σ")
            target = self.bank.bigop_height * self.xh
            if pool:
                g = self.bank.plain(self.rng.choice(pool))
                rows = np.nonzero(g.img.max(axis=1) > 50)[0]
                h = (rows[-1] - rows[0] + 1) if len(rows) else g.img.shape[0]
                g = resized(g, target / max(1.0, h))
            else:
                g = self.bank.single("Σ", self.rng)
                if g is None:
                    return None
                rows = np.nonzero(g.img.max(axis=1) > 50)[0]
                h = (rows[-1] - rows[0] + 1) if len(rows) else g.img.shape[0]
                g = self.bank.thickened(resized(g, target / max(1.0, h)), max(0.55, h / target))
            rows = np.nonzero(g.img.max(axis=1) > 50)[0]
            mid = (rows[0] + rows[-1]) / 2 if len(rows) else g.img.shape[0] / 2
            axis = MATRIX_AXIS_XH * self.xh
            g = Glyph(g.img, mid + axis, kind=g.kind)  # centred on the maths axis
            op_pl, op_w = [(g, 0.0, 0.0)], g.adv
        else:
            op_pl, op_w = self.build_word(op)
        otop, obot, _, _ = ink_extent(op_pl)
        parts = []
        for text, where in ((upper, "up"), (lower, "down")):
            if not text:
                continue
            pl, w = self.build_phrase(" ".join("^{" + t + "}" for t in text.split()))
            t, b, _, _ = ink_extent(pl)
            if where == "up":
                shift = otop - self.bank.limit_gap_up * self.xh - b
            else:
                shift = max(obot, 0.25 * self.xh) + self.bank.limit_gap_down * self.xh - t
            parts.append((pl, w, shift, t + shift, b + shift))
        width = max([op_w] + [w for _, w, _, _, _ in parts])
        above = -min([otop] + [t for *_, t, _ in parts]) + 0.1 * self.xh
        below = max([obot] + [b for *_, b in parts]) + 0.1 * self.xh

        def draw(x, baseline):
            x0 = x + (width - op_w) / 2
            for g, dx, dy in op_pl:
                self.paste(g, x0 + dx, baseline + dy)
            for pl, w, shift, _, _ in parts:
                xl = x + (width - w) / 2
                for g, dx, dy in pl:
                    self.paste(g, xl + dx, baseline + dy + shift)

        return {"width": width, "above": above, "below": below, "draw": draw}

    # -- pages and lines
    def new_page(self, label, source):
        top = self.top if self.pages or self.first_top is None else max(self.top, self.first_top)
        self.page = {"alpha": np.zeros((self.h, self.w), np.float32), "lines": 0, "label": label,
                     "source": {source}, "top": top}
        self.pages.append(self.page)
        self.y = top

    def new_line(self, extra=0):
        self.y += self.pitch * (1 + extra)

    def start_line(self, label, source):
        """Make sure the next line fits on the current page; returns the line's wobble."""
        if self.page is None or self.y > self.bottom:
            self.new_page(label, source)
        self.page["source"].add(source)
        self.page["lines"] += 1
        return self.wobble(self.y)

    def wobble(self, y0):
        """A line's baseline: a slight slope and a gentle wave, as when writing without ruled lines."""
        slope = math.tan(math.radians(min(0.8, max(-0.8, self.rng.gauss(0, 0.35)))))
        amp = self.rng.uniform(0.02, 0.07) * self.xh
        wave = self.rng.uniform(500, 1200)
        phase = self.rng.uniform(0, 2 * math.pi)
        return lambda x: y0 + slope * (x - self.left) + amp * math.sin(x / wave * 2 * math.pi + phase)

    def paste(self, g, x, base):
        img = g.img.astype(np.float32) / 255 * self.rng.uniform(0.85, 1.0)
        lay_ink(self.page["alpha"], img, int(round(x - g.left)), int(round(base - g.baseline)))

    def draw_box(self, x0, y0, x1, y1, slope=0.0):
        """A box drawn by hand around an answer: two to four pen strokes whose sides bow a little, tilt
        on their own and lean like the letters, cross or stop short where two strokes meet, turn a
        corner in a curve when one stroke goes round it, and taper where the pen lands and lifts."""
        r, xh = self.rng, self.xh
        j = lambda k: r.gauss(0, k * xh)
        h, w = y1 - y0, x1 - x0
        lean = lambda: math.tan(math.radians(self.slant * r.uniform(0.2, 0.9) + r.gauss(0, 1.2))) * h / 2
        left_lean, right_lean = lean(), lean()  # each side leans its own way, so the box is a little skewed
        # Top and bottom tilt on their own, but a long box can't tilt into the letters it holds.
        tilt = lambda: slope + max(-0.2 * xh, min(0.2 * xh, j(0.1))) / max(w, 1.0)
        top_tilt, bottom_tilt = tilt(), tilt()
        out = lambda k: abs(j(k))  # corners wander outwards, never into the text
        tl = (x0 + left_lean - out(0.15), y0 - out(0.1))
        tr = (x1 + right_lean + out(0.15), tl[1] + top_tilt * w - out(0.06))
        bl = (x0 - left_lean - out(0.15), y1 + out(0.1))
        br = (x1 - right_lean + out(0.15), bl[1] + bottom_tilt * w + out(0.06))
        near = lambda p: (p[0] + j(0.06), p[1] + j(0.06))  # separate strokes never meet exactly
        over = lambda lo, hi: r.uniform(lo, hi) * xh  # past a corner (+) or short of it (-)
        style = r.choices([k for k, _ in BOX_STYLES], [p for _, p in BOX_STYLES])[0]
        if style == "two":  # top and right in one go, then left and bottom
            strokes = [[(tl, tr, over(0.0, 0.35), 0), (tr, br, 0, over(0.05, 0.5))],
                       [(near(tl), bl, over(0.1, 0.45), 0), (bl, near(br), 0, over(0.05, 0.6))]]
        elif style == "four":
            strokes = [[(near(a), near(b), over(-0.15, 0.55), over(-0.15, 0.55))]
                       for a, b in ((tl, tr), (tr, br), (bl, br), (tl, bl))]
        else:  # all the way round, ending near where it started
            strokes = [[(tl, tr, over(0.05, 0.3), 0), (tr, br, 0, 0), (br, bl, 0, 0),
                        (bl, near(tl), 0, over(-0.25, 0.45))]]
        for sides in strokes:
            pts = np.vstack([self.box_side(a, b, e0, e1) for a, b, e0, e1 in sides])
            if len(sides) > 1:  # the pen turns a corner in a curve, not a point
                sigma = r.uniform(0.06, 0.22) * xh / 2
                pts = np.stack([gaussian_filter1d(pts[:, k], sigma, mode="nearest") for k in (0, 1)], 1)
            self.box_stroke(self.pen_ends(pts))

    def box_side(self, p, q, ext0, ext1):
        """Points about 2 px apart along one side of a box, from ext0 before p to ext1 past q: one gentle
        bow (more on short sides, drawn with the fingers), a slower drift and a faint tremor."""
        r, xh = self.rng, self.xh
        (ax, ay), (bx, by) = p, q
        length = max(1.0, math.hypot(bx - ax, by - ay))
        ux, uy = (bx - ax) / length, (by - ay) / length
        ax, ay, bx, by = ax - ux * ext0, ay - uy * ext0, bx + ux * ext1, by + uy * ext1
        length = max(1.0, math.hypot(bx - ax, by - ay))
        f = np.linspace(0, 1, max(8, int(length / 2)))
        bow = max(-0.15 * xh, min(0.15 * xh, r.gauss(0, 0.008 if length > 6 * xh else 0.035) * length))
        off = bow * np.sin(np.pi * f) + r.gauss(0, 0.002) * length * np.sin(2 * np.pi * f)
        wave = r.uniform(150, 500)
        off += r.gauss(0, 0.012 * xh) * np.sin(2 * np.pi * f * length / wave + r.uniform(0, 2 * np.pi))
        return np.stack([ax + (bx - ax) * f - uy * off, ay + (by - ay) * f + ux * off], 1)

    def pen_ends(self, pts):
        """Now and then a small hook where the pen lands and a flick where it lifts off."""
        r, xh = self.rng, self.xh
        out = [pts]
        for end in (0, 1):
            if r.random() > (0.45 if end == 0 else 0.35):
                continue
            a, b = (pts[0], pts[min(6, len(pts) - 1)]) if end == 0 else (pts[-1], pts[max(0, len(pts) - 7)])
            d = (a - b) / max(1e-6, np.hypot(*(a - b)))  # pointing outwards, past the end
            n = np.array([-d[1], d[0]]) * r.choice((-1, 1))
            size, bend = r.uniform(0.12, 0.3) * xh, r.uniform(0.06, 0.18) * xh
            u = np.linspace(0, 1, max(3, int(size / 2)))[1:, None]
            hook = a + d * size * u + n * bend * u ** 2
            out = [hook[::-1]] + out if end == 0 else out + [hook]
        return np.vstack(out)

    def box_stroke(self, pts):
        """One pen stroke of a box, as thin as the writing: thinner where the pen lands and lifts, a
        little uneven in width and ink along its length."""
        r, xh = self.rng, self.xh
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        total = max(s[-1], 1.0)
        t = self.bank.stroke_px * r.uniform(0.6, 0.75)
        land, lift = r.uniform(0.2, 0.5) * xh, r.uniform(0.3, 0.8) * xh
        wave, phase = r.uniform(150, 400), r.uniform(0, 2 * np.pi)
        widths = t * (0.45 + 0.55 * np.clip(s / land, 0, 1)) * (0.4 + 0.6 * np.clip((total - s) / lift, 0, 1))
        widths *= 1 + 0.12 * np.sin(2 * np.pi * s / wave + phase)
        img, x0, y0 = pen_image(pts, list(widths))
        noise = np.random.default_rng(r.getrandbits(32)).uniform(0.72, 1.0, (max(2, img.shape[0] // 40),
                                                                              max(2, img.shape[1] // 40)))
        ink = cv2.resize(noise.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        self.paste_alpha(img * np.clip(ink, 0.65, 1.0) * r.uniform(0.82, 0.95), x0, y0)

    def stroke(self, pts, widths, strength):
        """Draw a pen stroke along pts (x, y) with a width per point, onto the page."""
        img, x0, y0 = pen_image(pts, widths)
        self.paste_alpha(img * strength, x0, y0)

    def paste_alpha(self, a, x0, y0):
        """Lay ink coverage a (0..1) onto the page with its top-left corner at (x0, y0)."""
        lay_ink(self.page["alpha"], a, x0, y0)

    def draw_bracket(self, kind, side, x, top, bottom):
        """A tall hand-drawn bracket. x is its inner edge; side is -1 for the left bracket, +1 for the right."""
        t = 0.95 * self.bank.stroke_px
        h = bottom - top
        strength = self.rng.uniform(0.85, 0.95)
        drift = self.rng.gauss(0, 0.04 * self.xh)
        if kind == "p":
            bulge = 0.35 * self.xh + 0.05 * h
            n = 40
            pts, widths = [], []
            for k in range(n + 1):
                f = k / n
                wobble = drift * math.sin(math.pi * f * 2)
                pts.append((x + side * (bulge * math.sin(math.pi * f) + wobble), top + h * f))
                widths.append(t * (0.55 + 0.45 * math.sin(math.pi * f)))  # thinner where the pen lands and lifts
            self.stroke(pts, widths, strength)
        elif kind == "b":
            outer = x + side * (0.35 * self.xh)
            serif = 0.4 * self.xh
            j = lambda: self.rng.gauss(0, 0.03 * self.xh)
            pts = [(outer - side * serif + j(), top + j()), (outer + j(), top + j()),
                   (outer + drift, (top + bottom) / 2), (outer + j(), bottom + j()),
                   (outer - side * serif + j(), bottom + j())]
            self.stroke(pts, [t * 0.8, t, t, t, t * 0.8], strength)
        else:  # v: a plain bar, as in a determinant
            bar = x + side * (0.2 * self.xh)
            self.stroke([(bar, top), (bar + drift, (top + bottom) / 2), (bar, bottom)], [t * 0.7, t, t * 0.7],
                        strength)

    def build_matrix(self, rows, bracket):
        """Lay out a matrix: entries centred in aligned columns, brackets as tall as the rows.

        Returns a dict with its width, its extent above and below the baseline, and a draw(x, baseline)."""
        cells = [[self.build_word(e) for e in row] for row in rows]
        # Tails of g, j, p, q, y would otherwise poke into the row below: give such rows more room.
        deep = [any(set(e) & set("gjpqy,") for e in row) for row in rows]
        ncols = max(len(r) for r in cells)
        colw = [max(cells[r][c][1] for r in range(len(cells)) if c < len(cells[r])) for c in range(ncols)]
        col_gap = MATRIX_COL_GAP_XH * self.xh
        row_pitch = max(MATRIX_ROW_PITCH * self.pitch, 2.3 * self.xh)
        row_h = [row_pitch * (1.3 if d else 1.0) for d in deep]
        inner = sum(colw) + col_gap * (ncols - 1)
        height = sum(row_h)
        bracket_w = (0.35 * self.xh + 0.05 * height) if bracket == "p" else 0.45 * self.xh
        own = None
        pair = {"b": "[]", "p": "()", "v": "||"}[bracket]
        target = height + 0.15 * self.xh
        own = [self.bank.tall_delim(ch, target, self.rng) for ch in pair]
        if any(o is None for o in own) and bracket == "b":
            own = [self.bank.tall_bracket(ch, target, self.rng) for ch in pair]
        own = None if any(o is None for o in own) else own
        bw = [o.shape[1] - 0.15 * self.xh for o in own] if own else [bracket_w, bracket_w]
        pad = 0.35 * self.xh
        width = bw[0] + bw[1] + 2 * pad + inner
        axis = MATRIX_AXIS_XH * self.xh
        jitter = [[(self.rng.gauss(0, 0.06 * self.xh), self.rng.gauss(0, 0.05 * self.xh)) for _ in row] for row in cells]

        def draw(x, baseline):
            mid = baseline - axis
            top, bottom = mid - height / 2, mid + height / 2
            if own:
                strength = self.rng.uniform(0.85, 1.0)
                self.paste_alpha(own[0].astype(np.float32) / 255 * strength, int(round(x)), int(round(top - 0.1 * self.xh)))
            else:
                self.draw_bracket(bracket, -1, x + bracket_w, top - 0.1 * self.xh, bottom + 0.05 * self.xh)
            cx = x + bw[0] + pad
            y_row = top
            for r, row in enumerate(cells):
                row_base = y_row + 0.5 * row_pitch + 0.65 * self.xh
                y_row += row_h[r]
                xc = cx
                for c in range(ncols):
                    if c < len(row):
                        placed, w = row[c]
                        dx0 = xc + (colw[c] - w) / 2 + jitter[r][c][0]
                        for g, dx, dy in placed:
                            self.paste(g, dx0 + dx, row_base + dy + jitter[r][c][1])
                    xc += colw[c] + col_gap
            if own:
                strength = self.rng.uniform(0.85, 1.0)
                self.paste_alpha(own[1].astype(np.float32) / 255 * strength,
                                 int(round(x + width - own[1].shape[1])), int(round(top - 0.1 * self.xh)))
            else:
                self.draw_bracket(bracket, +1, x + width - bracket_w, top - 0.1 * self.xh, bottom + 0.05 * self.xh)

        return {"width": width, "above": height / 2 + axis + 0.1 * self.xh,
                "below": height / 2 - axis + 0.05 * self.xh, "draw": draw}

    def write_math(self, items, label, source, boxed=False):
        """Write a display line that holds matrices: centred, as tall as its tallest matrix, and
        wrapped before a + or = when it doesn't fit on one line."""
        built = []
        items = split_braces(items)
        mats = []
        for it in items:
            if it["kind"] == "matrix":
                mats.append(self.build_matrix(it["rows"], it["bracket"]))
            elif it["kind"] == "frac":
                mats.append(self.build_frac(it["num"], it["den"]))
            elif it["kind"] == "bigop":
                mats.append(self.build_bigop(it["op"], it["lower"], it["upper"]))
            else:
                mats.append(None)
        tall_above = max((m["above"] for m in mats if m), default=1.9 * self.xh) + 0.1 * self.xh
        tall_below = max((m["below"] for m in mats if m), default=0.9 * self.xh) + 0.1 * self.xh
        for it, m in zip(items, mats):
            if it["kind"] in ("matrix", "frac", "bigop") and m is not None:
                built.append({"text": None, "glue": it["glue"], **m})
            elif it["kind"] == "delim":
                own = self.bank.tall_delim(it["char"], tall_above + tall_below, self.rng)
                if own is not None:
                    g = Glyph(own, tall_above, kind="profile")
                else:
                    g = brace_glyph(it["char"], tall_above + tall_below, self.xh, self.bank.stroke_px, self.rng,
                                    above=tall_above)
                    self.bank._note(it["char"], "drawn as a pen stroke at your pen width")
                built.append({"text": it["char"], "glue": it["glue"], "width": g.adv, "placed": [(g, 0.0, 0.0)],
                              "above": tall_above, "below": tall_below})
            else:
                placed, w = self.build_word(it["text"])
                built.append({"text": it["text"], "glue": it["glue"], "width": w, "placed": placed,
                              "above": 1.9 * self.xh, "below": 0.9 * self.xh})
        for k, b in enumerate(built):
            if k == 0:
                b["gap"] = 0.0
            elif b["glue"]:
                b["gap"] = GLUE_XH * self.xh
            else:
                b["gap"] = self.word_gap() * MATH_GAP

        # Break into lines: fill each line, and when it overflows go back to the last + − or = .
        avail = self.right - self.left
        lines, cur, width = [], [], 0.0
        for b in built:
            add = b["width"] + (b["gap"] if cur else 0)
            if cur and width + add > avail:
                cut = max((k for k, c in enumerate(cur) if k and c["text"] in ("+", "−", "-", "=")), default=None)
                if cut is None:
                    lines.append(cur)
                    cur = []
                else:
                    lines.append(cur[:cut])
                    cur = cur[cut:]
                width = sum(c["width"] for c in cur) + sum(c["gap"] for c in cur[1:])
                add = b["width"] + (b["gap"] if cur else 0)
            cur.append(b)
            width += add
        if cur:
            lines.append(cur)

        if boxed and self.page is not None:
            self.y += 0.3 * self.pitch  # room for the box
        for line in lines:
            above = max(b["above"] for b in line) + (0.3 * self.xh if boxed else 0)
            below = max(b["below"] for b in line) + (0.3 * self.xh if boxed else 0)
            base = None if self.page is None else self.y + max(0.0, above - MATH_CLEAR[0] * self.pitch)
            if base is None or base + below > self.bottom + 0.3 * self.pitch:
                self.new_page(label, source)
                base = self.page["top"] + max(0.0, above - self.bank.cap_px)
            self.page["source"].add(source)
            self.page["lines"] += 1
            width = sum(b["width"] for b in line) + sum(b["gap"] for b in line[1:])
            start = (self.left + self.right) / 2 - width / 2 + self.rng.gauss(0, 2 * PX_PER_MM)
            line_y = self.wobble(base)
            x = start
            for k, b in enumerate(line):
                if k:
                    x += b["gap"]
                if b["text"] is None:
                    b["draw"](x, line_y(x + b["width"] / 2))
                else:
                    for g, dx, dy in b["placed"]:
                        self.paste(g, x + dx, line_y(x + dx) + dy)
                x += b["width"]
            if boxed and len(lines) == 1:
                pad = BOX_PAD_XH * self.xh
                self.draw_box(start - pad, base - above, start + width + pad, base + below,
                              slope=(line_y(start + width) - line_y(start)) / max(width, 1.0))
            self.y = base + max(self.pitch, below + MATH_CLEAR[1] * self.pitch)
        if boxed:
            self.y += 0.3 * self.pitch

    def write_line(self, built, label, source, centered=False):
        """Write as many of the built words as fit on one line; returns how many were written."""
        line_y = self.start_line(label, source)
        start = self.left + self.rng.gauss(0, 0.8 * PX_PER_MM)
        gaps = []
        for k, (_, _, text) in enumerate(built):
            g = self.word_gap()
            near_symbol = text in SYMBOL_WORDS or (k and built[k - 1][2] in SYMBOL_WORDS)
            if centered or near_symbol:
                g *= MATH_GAP
            gaps.append(g)
        count, width = 0, 0.0
        for k, (_, w, _) in enumerate(built):
            add = w + (gaps[k] if count else 0)
            if count and start + width + add > self.right + self.rng.gauss(0, 1.5 * PX_PER_MM):
                break
            width += add
            count += 1
        # Don't break inside a short bracketed group such as "(1, 0)": go back to where it opened.
        # Don't end a line with = + − ≠ …, or start the next one with "=": in maths the break goes
        # before a + or −. Each rule can undo the other, so apply both until neither moves the break.
        for _ in range(4):
            before = count
            if 0 < count < len(built):
                depth, opened = 0, None
                for k in range(count):
                    for ch in built[k][2]:
                        if ch in "([{⟨":
                            if depth == 0:
                                opened = k
                            depth += 1
                        elif ch in ")]}⟩" and depth:
                            depth -= 1
                if depth and opened and group_end(built, opened) - opened < 4:
                    while count > opened:
                        count -= 1
                        width -= built[count][1] + gaps[count]
            for _ in range(3):
                if not 1 < count < len(built):
                    break
                if built[count - 1][2] in SYMBOL_WORDS or built[count][2] in NO_LINE_START:
                    count -= 1
                    width -= built[count][1] + gaps[count]
                else:
                    break
            if count == before:
                break
        if centered and count == len(built):
            start = (self.left + self.right) / 2 - width / 2 + self.rng.gauss(0, 2 * PX_PER_MM)
        x = start
        for k in range(count):
            placed, w, _ = built[k]
            if k:
                x += gaps[k]
            for g, dx, dy in placed:
                self.paste(g, x + dx, line_y(x + dx) + dy)
            x += w
        ink = [ink_extent(built[k][0]) for k in range(count) if built[k][0]]
        self.line_extent = (self.page, start, start + width, line_y(start + width / 2),
                            (line_y(start + width) - line_y(start)) / max(width, 1.0),
                            min([0.0] + [e[0] for e in ink]), max([0.0] + [e[1] for e in ink]))
        self.new_line()
        return count

    def box_lines(self, extents):
        """One box around the lines of a \\boxed display (a box per page if it runs onto the next one)."""
        pad = BOX_PAD_XH * self.xh
        current = self.page
        for page in {id(e[0]): e[0] for e in extents}.values():
            ext = [e for e in extents if e[0] is page]
            self.page = page
            # Clear of the tallest ink on the first line (a root's bar, a power) and the lowest on the last.
            top = min(-2.1 * self.xh, ext[0][5] - 0.55 * self.xh)
            bottom = max(0.9 * self.xh, ext[-1][6] + 0.45 * self.xh)
            self.draw_box(min(e[1] for e in ext) - pad, ext[0][3] + top,
                          max(e[2] for e in ext) + pad + 0.3 * self.xh, ext[-1][3] + bottom,
                          slope=ext[0][4])
        self.page = current

    def write_blocks(self, blocks, label, source):
        for b in blocks:
            if b["kind"] == "blank":
                if self.page is not None and self.y > self.page["top"]:
                    self.y += BLANK_LINE * self.pitch
                continue
            if b["kind"] == "math":
                self.write_math(b["items"], label, source, boxed=b["boxed"])
                continue
            words = b["words"]
            boxed = b["kind"] == "display" and words[:1] == ["\\boxed"]
            if boxed:
                words = words[1:]
                if self.page is not None:
                    self.y += 0.3 * self.pitch  # room for the box
                    if any("√" in w or "^" in w for w in words):
                        self.y += 0.25 * self.pitch  # and for a root's bar or a power above the letters
            built = [self.build_word(w) + (w,) for w in words]
            centered = b["kind"] == "display"
            if centered and not boxed and self.page is not None and self.y > self.page["top"]:
                self.y += DISPLAY_SPACE * self.pitch  # display maths gets room, so tails don't collide
            extents = []
            while built:
                n = self.write_line(built, label, source, centered=centered)
                extents.append(self.line_extent)
                built = built[n:]
                if built and centered:
                    centered = False  # a display line too long for one line wraps like text
            if boxed:
                self.box_lines(extents)
                self.y += 0.3 * self.pitch
            elif b["kind"] == "display":
                self.y += DISPLAY_SPACE * self.pitch


# ---------------------------------------------------------------- output

def clean_rgb(alpha, ink):
    a = alpha[..., None]
    return (255 * (1 - a) + np.array(ink, np.float32) * a).astype(np.uint8)


def write_qa(rgb, qa_dir, stem):
    h = rgb.shape[0]
    overlap = int(0.05 * h)
    out = []
    for name, (y0, y1) in (("top", (0, h // 2 + overlap)), ("bottom", (h // 2 - overlap, h))):
        part = rgb[y0:y1]
        scale = 1500 / max(part.shape[:2])
        part = cv2.resize(part, (int(part.shape[1] * scale), int(part.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        path = qa_dir / f"{stem}_{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(part, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append(path)
    return out


def parse_ink(text):
    if text in INKS:
        return INKS[text]
    t = text.lstrip("#")
    if len(t) == 6:
        return tuple(int(t[i:i + 2], 16) for i in (0, 2, 4))
    sys.exit(f"ERROR: unknown ink '{text}' (use blue, black or #rrggbb)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcripts", nargs="+")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--continuous", action="store_true", help="flow all inputs into one document")
    ap.add_argument("--clean", action="store_true", help="flat white scan instead of the phone-photo look")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ink", default="blue")
    ap.add_argument("--line-pitch", type=float, help="line spacing in mm for this run only")
    ap.add_argument("--identity", action="store_true",
                    help="write the identity block (identity.py train) in the top-left corner of the first page")
    ap.add_argument("--pdf", action="store_true", help="also put every page, in order, into handwritten.pdf")
    args = ap.parse_args()

    pdir = profile_dir(args.profile)
    profile = read_json(pdir / "profile.json")
    if not profile or not profile.get("glyphs"):
        print(f'NO PROFILE: there is no handwriting profile "{args.profile}". Run /handwritten train first.')
        sys.exit(2)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    rng = random.Random(seed)
    ink = parse_ink(args.ink)
    out = Path(args.out).expanduser()
    qa_dir = out / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    bank = GlyphBank(pdir, profile)
    settings = read_settings(args.profile)
    if args.line_pitch:
        settings["line_pitch_mm"] = args.line_pitch
    writer = Writer(bank, apply_settings(profile["style"], settings), rng)
    block = None
    if args.identity:
        data = read_json(pdir / "identity" / "identity.json")
        if not data:
            print(f'NO IDENTITY: profile "{args.profile}" has no identity block. Train one with identity.py train first.')
            sys.exit(2)
        from identity import IdentityBlock
        # Its own random numbers, so a seed gives the same writing with or without the block.
        block = IdentityBlock(pdir, data, random.Random(seed + 1), bank)
        writer.first_top = block.bottom + writer.pitch  # the text starts a line below the block
    inputs = [Path(t) for t in args.transcripts]
    for n, path in enumerate(inputs, 1):
        blocks = parse_transcript(path)
        if not args.continuous:
            writer.page = None  # every input starts on a fresh page
        writer.write_blocks(blocks, label=f"{n:02d}", source=n)
    if block:
        if not writer.pages:
            writer.new_page("01", 1)
        block.draw(writer.pages[0]["alpha"])  # the first page of the output only

    # Name the pages.
    if args.continuous:
        names = [f"page_{i:02d}" for i in range(1, len(writer.pages) + 1)]
    else:
        per_label = Counter(p["label"] for p in writer.pages)
        seen = Counter()
        names = []
        for p in writer.pages:
            seen[p["label"]] += 1
            suffix = "" if per_label[p["label"]] == 1 else "abcdefghijklmnopqrstuvwxyz"[seen[p["label"]] - 1]
            names.append(f"page_{p['label']}{suffix}")

    report = {"seed": seed, "mode": "continuous" if args.continuous else "per-image", "look": "clean" if args.clean else "photo",
              "inputs": [str(p) for p in inputs], "pages": [],
              "fallback": {c: {"count": k, "how": bank.fallback_how[c]} for c, k in bank.fallback.most_common()},
              "not_drawn": dict(bank.missing), "identity": bool(block)}
    for i, (name, page) in enumerate(zip(names, writer.pages)):
        clean = clean_rgb(page["alpha"], ink)
        qa = write_qa(clean, qa_dir, name)
        final = clean if args.clean else photo_look(page["alpha"], ink, seed + 7919 * (i + 1))
        path = out / f"{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        report["pages"].append({"file": str(path), "inputs": sorted(page["source"]), "lines": page["lines"],
                                "qa": [str(q) for q in qa]})
    if args.pdf:
        # The finished pages exactly as saved, at 300 DPI, so every PDF page is exactly A4.
        pdf = out / "handwritten.pdf"
        pages = [Image.open(p["file"]).convert("RGB") for p in report["pages"]]
        pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=PX_PER_MM * 25.4, quality=92,
                      title="Handwritten pages", creator="/handwritten skill (generated, not handwritten)")
        report["pdf"] = str(pdf)
    write_json(out / "report.json", report)

    print(f"{len(writer.pages)} page(s) from {len(inputs)} input(s), {report['mode']}, {report['look']} look, seed {seed}")
    for p in report["pages"]:
        print(f"  {Path(p['file']).name}: inputs {','.join(map(str, p['inputs']))}, {p['lines']} lines")
        for q in p["qa"]:
            print(f"        check: {q}")
    if report["fallback"]:
        print("  not in your profile, drawn another way:")
        for c, f in report["fallback"].items():
            print(f"    {c}  x{f['count']}  ({f['how']})")
    if report["not_drawn"]:
        print("  NOT DRAWN (left as a gap): " + " ".join(f"{c} x{k}" for c, k in report["not_drawn"].items()))
    if block:
        print(f"  identity block: top-left of {Path(report['pages'][0]['file']).name} only")
    if args.pdf:
        print(f"  pdf: {report['pdf']} ({len(report['pages'])} A4 pages)")
    print(f"  report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
