"""Split preprocessed pages into lines, words and pieces of ink, guided by the transcript.

A piece is a connected run of ink after small marks (i-dots, t-crosses, broken
strokes) are merged into the letter they belong to. For semi-joined writing a
piece is either one character or several characters written without lifting
the pen (a ligature such as "th").

Labelling rules:
  * a word that is one piece is labelled as the whole word
  * a word with as many pieces as characters gets one character per piece
  * every other word goes on a contact sheet for Claude to label

Outputs per sample: segments.json, lines.png (line check), sheets/sheet_NN.png
and labels.txt (a template Claude edits: insert "|" between the pieces).

Usage: segment_freeform.py [--profile NAME] [--force] [SAMPLE_ID...]
"""

import argparse
import hashlib
import math
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from common import join_split, read_json, read_transcript, samples_dir, ui_font, write_json

PIECE_COLORS = [(220, 30, 30), (20, 90, 220), (0, 150, 60), (230, 120, 0), (150, 40, 190), (0, 150, 160)]
ROWS_PER_SHEET = 12
MIN_GAP_RATIO = 1.25  # word gaps must be this much wider than letter gaps to auto-label a line


BIG_STROKE = 4.0  # strokes this many letter-heights tall are rules, braces or box sides, not writing
SKIP_WEIGHT = 1.0  # cost of leaving a full-width row of ink out of the transcript
GAP_POSITION_WEIGHT = 10.0  # how much word-gap positions count when matching lines to rows
TAIL_WEIGHT = 3.0  # how much the positions of g/j/p/q/y tails count when matching lines to rows
TAIL_LETTERS = set("gjpqy")
PITCH_WEIGHT = 1.0  # how much neighbouring lines are expected to sit one line pitch apart


def letter_height(stats):
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    h = h[h >= 5]
    return float(np.median(h)) if len(h) else 20.0


def non_text_components(stats, letter_h):
    """Component ids that aren't writing: box sides, braces, table rules and underlines."""
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    # Joined words with a tall letter and a tail ("working") are tall too, but wide; box sides,
    # braces and table rules are tall and thin, or flat and long.
    tall_thin = (h > BIG_STROKE * letter_h) & (w < 0.5 * h)
    bad = tall_thin | (h > 6 * letter_h) | ((w > 6 * letter_h) & (h < 1.0 * letter_h))
    bad[0] = False
    return set(int(c) for c in np.nonzero(bad)[0])


def is_placeholder(words):
    """A transcript line "~" stands for a row of ink that isn't writing (a fraction row, a table rule)."""
    return words == ["~"]


def noise_components(stats, cc, ink, letter_h):
    """Tiny marks that are faint or far from any writing: show-through from the back of the page,
    dust, scanner specks. Full stops and dots on i/j are dark and close to letters, so they stay."""
    from scipy.ndimage import maximum

    n = len(stats)
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    tiny = (w < 0.35 * letter_h) & (h < 0.35 * letter_h)
    tiny[0] = False
    if not tiny.any():
        return set()
    peak = np.asarray(maximum(ink, cc, index=np.arange(n)), dtype=np.float64)
    big = np.nonzero(~tiny)[0][1:]
    x0, y0 = stats[big, 0], stats[big, 1]
    x1, y1 = x0 + stats[big, 2], y0 + stats[big, 3]
    out = set()
    for c in np.nonzero(tiny)[0]:
        cx, cy = stats[c, 0] + w[c] / 2, stats[c, 1] + h[c] / 2
        dx = np.maximum(0, np.maximum(x0 - cx, cx - x1))
        dy = np.maximum(0, np.maximum(y0 - cy, cy - y1))
        isolated = len(big) == 0 or float(np.sqrt(dx ** 2 + dy ** 2).min()) > 1.2 * letter_h
        if peak[c] < 140 or isolated:
            out.add(int(c))
    return out


def detect_rows(mask, cc, stats, cents, ink):
    """Find the rows of writing on a page: centres of the row profile after removing rules, box
    sides and specks. Every component of writing is assigned to its nearest row."""
    letter_h = letter_height(stats)
    noise = noise_components(stats, cc, ink, letter_h)
    excluded = non_text_components(stats, letter_h) | noise
    text_mask = mask.copy()
    if excluded:
        text_mask[np.isin(cc, sorted(excluded))] = False
    comp_ids = np.array([c for c in range(1, len(stats)) if c not in excluded], dtype=int)
    prof = gaussian_filter1d(text_mask.sum(axis=1).astype(np.float64), sigma=max(1.5, 0.35 * letter_h))
    peaks, _ = find_peaks(prof, distance=max(3, int(0.9 * letter_h)), prominence=max(prof.max(), 1) * 0.03)
    # Long tails (y, g) and loops pull a piece's centre towards the next row, so tall pieces are
    # placed by where their letter body starts instead of by their centre.
    top = stats[comp_ids, cv2.CC_STAT_TOP].astype(np.float64)
    h = stats[comp_ids, cv2.CC_STAT_HEIGHT].astype(np.float64)
    anchor = np.where(h > 1.5 * letter_h, top + 0.75 * letter_h, cents[comp_ids, 1])
    row_of = (np.abs(anchor[:, None] - peaks[None, :]).argmin(axis=1)
              if len(peaks) and len(comp_ids) else np.zeros(len(comp_ids), dtype=int))
    return {"letter_h": letter_h, "noise": noise, "excluded": excluded, "comp_ids": comp_ids,
            "peaks": peaks, "row_of": row_of}


ROW_TAG = re.compile(r"^R(\d+):?$")


def parse_row_tags(transcript):
    """Lines may start with a row number from rows.png ("R07 text…"), pinning them to that row.

    Returns [(row index, words)] when every line is tagged, None when none are."""
    tagged, plain = [], 0
    for words in transcript:
        if is_placeholder(words):
            continue
        m = ROW_TAG.match(words[0]) if words else None
        if m and len(words) > 1:
            tagged.append((int(m.group(1)) - 1, words[1:]))
        else:
            plain += 1
    if tagged and plain:
        raise ValueError("some transcript lines have row numbers and some don't; tag every line or none")
    return tagged or None


def load_page(sdir):
    mask = cv2.imread(str(sdir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    ink = cv2.imread(str(sdir / "ink.png"), cv2.IMREAD_GRAYSCALE)
    n, cc, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return mask, ink, n, cc, stats, cents


def write_rows_image(sdir):
    """rows.png: every detected row of writing in its own colour, numbered R01, R02… at the left,
    so transcript lines can be pinned to rows on pages with maths, tables or crossings-out."""
    mask, ink, n, cc, stats, cents = load_page(sdir)
    rows = detect_rows(mask, cc, stats, cents, ink)
    a = (ink.astype(np.float32) / 255)[..., None]
    rgb = np.full(ink.shape + (3,), 255, np.float32) * (1 - a) + np.array([190, 190, 190], np.float32) * a
    lut = np.full(n, -1, dtype=int)
    lut[rows["comp_ids"]] = rows["row_of"]
    row_map = lut[cc]
    for j in range(len(rows["peaks"])):
        sel = (row_map == j)[..., None]
        col = np.array(PIECE_COLORS[j % 2], np.float32)
        rgb = np.where(sel, 255 * (1 - a) + col * a, rgb)
    scale = 1600 / ink.shape[0]
    img = Image.fromarray(rgb.astype(np.uint8)).resize((int(ink.shape[1] * scale), 1600), Image.LANCZOS)
    canvas = Image.new("RGB", (img.width + 90, img.height), "white")
    canvas.paste(img, (90, 0))
    d = ImageDraw.Draw(canvas)
    f = ui_font(20)
    for j, y in enumerate(rows["peaks"]):
        d.text((6, y * scale - 11), f"R{j + 1:02d}", fill=PIECE_COLORS[j % 2], font=f)
        d.line([(62, y * scale), (88, y * scale)], fill=PIECE_COLORS[j % 2], width=2)
    canvas.save(sdir / "rows.png")
    write_json(sdir / "rows.json", {"peaks": [int(y) for y in rows["peaks"]]})
    return sdir / "rows.png", len(rows["peaks"])


def line_expected_width(words):
    """Expected width of a transcript line in letter heights; None when the line is only '#' or '~'."""
    if all(w in ("#", "~") for w in words):
        return None
    return sum(3.0 if w == "#" else sum(prior_width(ch) for ch in w) for w in words) + 0.6 * (len(words) - 1)


def word_break_positions(words):
    """Where the word breaks of a transcript line should fall, as fractions of the line's width."""
    widths = [3.0 if w == "#" else sum(prior_width(ch) for ch in w) for w in words]
    total = sum(widths) + 0.6 * (len(words) - 1)
    pos, x = [], 0.0
    for w in widths[:-1]:
        x += w + 0.6
        pos.append((x - 0.3) / total)
    return pos


def tail_positions(words):
    """Expected positions (fractions of line width) of letters with tails below the line."""
    total = sum(3.0 if w == "#" else sum(prior_width(ch) for ch in w) for w in words) + 0.6 * (len(words) - 1)
    pos, x = [], 0.0
    for wi, w in enumerate(words):
        if w == "#":
            x += 3.0
        else:
            for ch in w:
                cw = prior_width(ch)
                if ch in TAIL_LETTERS:
                    pos.append((x + cw / 2) / total)
                x += cw
        x += 0.6
    return pos


def row_tail_positions(comps, stats, letter_h):
    """Positions of components reaching well below the row's baseline (fractions of row width)."""
    if len(comps) == 0:
        return []
    x0 = int(stats[comps, 0].min())
    x1 = int((stats[comps, 0] + stats[comps, 2]).max())
    bottoms = stats[comps, 1] + stats[comps, 3]
    tall = stats[comps, 3] >= 0.5 * letter_h
    base = float(np.median(bottoms[tall])) if tall.any() else float(np.median(bottoms))
    deep = comps[bottoms > base + 0.35 * letter_h]
    return sorted(((stats[c, 0] + stats[c, 2] / 2) - x0) / max(x1 - x0, 1) for c in deep)


def tails_mismatch(want, got):
    """Distance between two sets of tail positions, with a penalty for different counts."""
    if not want and not got:
        return 0.0
    if not want or not got:
        return 0.5
    a, b = np.array(want), np.array(got)
    near = (np.abs(a[:, None] - b[None, :]).min(axis=1).mean() + np.abs(b[:, None] - a[None, :]).min(axis=1).mean()) / 2
    return float(near) + 0.3 * abs(len(a) - len(b)) / max(len(a), len(b))


def row_gap_positions(intervals, n):
    """Positions (fractions of the row's width) of the row's n widest gaps, left to right."""
    if n == 0:
        return []
    gaps = [(intervals[i + 1][0] - intervals[i][1], (intervals[i + 1][0] + intervals[i][1]) / 2)
            for i in range(len(intervals) - 1)]
    if len(gaps) < n:
        return None
    x0, x1 = intervals[0][0], intervals[-1][1]
    return sorted((c - x0) / max(x1 - x0, 1) for _, c in sorted(gaps, reverse=True)[:n])


def match_lines(rows, stats, transcript):
    """Match each transcript line to a row of ink, in order, skipping rows that aren't in the
    transcript (stacked fractions, tables, crossed-out work).

    Returns (row y per line, component ids per line, width mismatch per line, rows skipped),
    or None when the page has fewer rows than the transcript has lines."""
    # "~" lines mark where rows were left out: skipping rows there (and above the first line or
    # below the last) is free; skipping rows anywhere else costs, by how much ink they hold.
    free = [True]
    lines = []
    for words in transcript:
        if is_placeholder(words):
            free[-1] = True
        else:
            lines.append(words)
            free.append(False)
    free[-1] = True
    transcript = lines
    k = len(transcript)
    peaks, row_of, comp_ids, letter_h = rows["peaks"], rows["row_of"], rows["comp_ids"], rows["letter_h"]
    if len(peaks) < k or len(comp_ids) == 0:
        return None
    widths, row_intervals = [], []
    for j in range(len(peaks)):
        comps = comp_ids[row_of == j]
        ivs = merge_intervals([(int(stats[c, 0]), int(stats[c, 0] + stats[c, 2]), int(c)) for c in comps])
        row_intervals.append(ivs)
        widths.append(max(1.0, float(sum(iv[1] - iv[0] for iv in ivs))))
    widths = np.array(widths)
    expected = line_widths_full = [line_expected_width(words) for words in transcript]
    has_unknown = ["#" in words for words in transcript]
    # For lines with "#", the expected width counts only the known words (plus one letter per "#").
    expected = [None if e is None else (sum(sum(prior_width(ch) for ch in w) for w in words if w != "#")
                                        + words.count("#") + 0.6 * (len(words) - 1)) if unk else e
                for e, unk, words in zip(expected, has_unknown, transcript)]
    breaks = [word_break_positions(words) for words in transcript]
    tails_want = [tail_positions(words) for words in transcript]
    tails_got = [row_tail_positions(comp_ids[row_of == j], stats, letter_h) for j in range(len(peaks))]
    gap_cost = {}

    def gaps_mismatch(i, j):
        """How far the row's widest gaps are from where the line's word breaks should be."""
        if (i, j) not in gap_cost:
            want = breaks[i]
            if not want or not row_intervals[j]:
                gap_cost[i, j] = 0.0
            else:
                got = row_gap_positions(row_intervals[j], len(want))
                gap_cost[i, j] = 1.0 if got is None else float(np.mean(np.abs(np.array(got) - np.array(want))))
        return gap_cost[i, j]
    ref = float(np.median(np.sort(widths)[::-1][:k]))
    known = [e for e in line_widths_full if e]
    scale = float(np.sort(widths)[::-1][:k].sum() / (sum(known) + 3.0 * (k - len(known))))

    # Lines with no "~" between them are neighbours on the page, so their rows should be about one
    # line pitch apart; a big jump means rows were skipped that the transcript doesn't account for.
    diffs = np.diff(peaks)
    pitch = float(np.median(diffs)) if len(diffs) else 1.0

    def align(scale):
        p = len(peaks)
        inf = float("inf")
        skip = SKIP_WEIGHT * widths / ref
        cum = np.concatenate([[0.0], np.cumsum(skip)])  # cum[j] = cost of skipping rows 0..j-1

        def skipped(zone, a, b):  # cost of skipping rows a..b-1 inside zone
            return 0.0 if free[zone] or b <= a else float(cum[b] - cum[a])

        def match(i, j):
            e = expected[i]
            if e is None:
                return 0.3
            ratio = math.log(widths[j] / (scale * e))
            if has_unknown[i]:
                # A "#" can be any width, so the row only has to be at least as wide as the known words;
                # where its word breaks and tails fall can't be predicted either.
                return min(ratio, 0.0) ** 2
            return (ratio ** 2 + min(1.0, GAP_POSITION_WEIGHT * gaps_mismatch(i, j))
                    + min(1.0, TAIL_WEIGHT * tails_mismatch(tails_want[i], tails_got[j])))

        best = np.full((k, p), inf)
        back = np.full((k, p), -1, dtype=int)
        for j in range(p):
            best[0, j] = match(0, j) + skipped(0, 0, j)
        for i in range(1, k):
            for j in range(i, p):
                m = match(i, j)
                for jp in range(i - 1, j):
                    if best[i - 1, jp] == inf:
                        continue
                    c = best[i - 1, jp] + skipped(i, jp + 1, j)
                    if not free[i]:
                        c += PITCH_WEIGHT * math.log((peaks[j] - peaks[jp]) / pitch) ** 2
                    if c + m < best[i, j]:
                        best[i, j], back[i, j] = c + m, jp
        total = [best[k - 1, j] + skipped(k, j + 1, p) for j in range(p)]
        j = int(np.argmin(total))
        rows = [j]
        for i in range(k - 1, 0, -1):
            j = int(back[i, j])
            rows.append(j)
        return rows[::-1]

    rows = align(scale)
    ratios = [widths[r] / e for r, e, unk in zip(rows, expected, has_unknown) if e and not unk]
    if ratios:
        scale = float(np.median(ratios))
        rows = align(scale)
    mismatch = [None if e is None or unk else abs(math.log(widths[r] / (scale * e)))
                for r, e, unk in zip(rows, expected, has_unknown)]
    return ([int(peaks[r]) for r in rows], [comp_ids[row_of == r] for r in rows], mismatch,
            len(peaks) - k, transcript)


def merge_intervals(boxes):
    """boxes: list of (x0, x1, comp). Returns merged [[x0, x1, [comps]]] sorted by x."""
    out = []
    for x0, x1, c in sorted(boxes, key=lambda b: b[0]):
        if out and x0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], x1)
            out[-1][2].append(c)
        else:
            out.append([x0, x1, [c]])
    return out


def build_pieces(comps, stats, core_h, core_mid):
    """Group a word's components into pieces, left to right."""
    def box(c):
        x, y, w, h = (int(v) for v in stats[c, :4])
        return x, y, x + w, y + h

    small_upper, rest = [], []
    for c in comps:
        x0, y0, x1, y1 = box(c)
        if (y1 - y0) < 0.45 * core_h and (x1 - x0) < 0.7 * core_h and y1 < core_mid:
            small_upper.append(c)
        else:
            rest.append(c)
    pieces = [[c] for c in sorted(rest, key=lambda c: box(c)[0])]

    def pbox(p):
        bs = [box(c) for c in p]
        return min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)

    # Dots, crosses and accents join the piece below them.
    for c in small_upper:
        x0, _, x1, _ = box(c)
        cx = (x0 + x1) / 2
        best, best_score = None, None
        for p in pieces:
            px0, _, px1, _ = pbox(p)
            overlap = min(x1, px1) - max(x0, px0)
            dist = 0 if overlap > 0 else min(abs(cx - px0), abs(cx - px1))
            score = (-overlap, dist)
            if dist <= 0.6 * core_h and (best_score is None or score < best_score):
                best, best_score = p, score
        if best is not None:
            best.append(c)
        else:
            pieces.append([c])

    # Strokes of one letter that don't touch (k, x, E, 4, =, :) overlap horizontally.
    pieces.sort(key=lambda p: pbox(p)[0])
    merged = True
    while merged:
        merged = False
        for i in range(len(pieces) - 1):
            a, b = pbox(pieces[i]), pbox(pieces[i + 1])
            overlap = min(a[2], b[2]) - max(a[0], b[0])
            if overlap > 0.5 * min(a[2] - a[0], b[2] - b[0]):
                pieces[i] = pieces[i] + pieces.pop(i + 1)
                merged = True
                break
    return [{"comps": sorted(int(c) for c in p), "bbox": list(pbox(p))} for p in pieces]


GAP_WEIGHT = 1.0  # how strongly wide gaps attract word breaks, against matching expected word widths
UNKNOWN_WORD_WEIGHT = 0.25  # how strongly a "#" word leans towards two letters wide


def line_intervals(comps, stats, cc, core_h, core_mid):
    """Merge a line's components into horizontal runs; full stops and commas join the run before them.

    Widths are measured inside the line's middle band only, so tails of j, f, y or g that reach
    under a neighbouring word don't close the gap between words."""
    band0 = max(0, int(core_mid - 0.3 * core_h))
    band1 = int(core_mid + 0.3 * core_h) + 1
    marks, body = [], []
    for c in comps:
        x, y, w, h = (int(v) for v in stats[c, :4])
        low_small = h < 0.4 * core_h and w < 0.5 * core_h and (y + h / 2) > core_mid + 0.15 * core_h
        x0, x1 = x, x + w
        if not low_small:
            cols = np.nonzero((cc[band0:band1, x:x + w] == c).any(axis=0))[0]
            if len(cols):
                x0, x1 = x + int(cols[0]), x + int(cols[-1]) + 1
        (marks if low_small else body).append((x0, x1, int(c)))
    intervals = merge_intervals(body)
    for x0, x1, c in marks:
        left = [iv for iv in intervals if iv[1] <= x1]
        if left and x0 - left[-1][1] <= 1.0 * core_h:
            iv = left[-1]
            iv[1] = max(iv[1], x1)
            iv[2].append(c)
        else:
            intervals.append([x0, x1, [c]])
            intervals.sort(key=lambda iv: iv[0])
    return intervals


def drop_stray_marks(intervals, core_h):
    """Remove tiny marks standing alone: at the ends of a line, or with wide gaps on both sides.
    Their big gaps would otherwise attract word breaks and shift every word on the line."""
    def small(iv):
        return iv[1] - iv[0] < 0.5 * core_h

    ivs = list(intervals)
    changed = True
    while changed and len(ivs) > 1:
        changed = False
        if small(ivs[-1]) and ivs[-1][0] - ivs[-2][1] > 1.5 * core_h:
            ivs.pop()
            changed = True
        elif small(ivs[0]) and ivs[1][0] - ivs[0][1] > 1.5 * core_h:
            ivs.pop(0)
            changed = True
    keep = [ivs[0]] if ivs else []
    for i in range(1, len(ivs) - 1):
        lone = (ivs[i][1] - ivs[i][0] < 0.35 * core_h and ivs[i][0] - ivs[i - 1][1] > 2 * core_h
                and ivs[i + 1][0] - ivs[i][1] > 2 * core_h)
        if not lone:
            keep.append(ivs[i])
    if len(ivs) > 1:
        keep.append(ivs[-1])
    return keep


def split_words(intervals, words, core_h, scale=None):
    """Choose word breaks that both fall on wide gaps and give each word its expected width.

    Returns (groups of intervals, indexes of words that need a human check)."""
    n, w = len(intervals), len(words)
    gaps = [intervals[i + 1][0] - intervals[i][1] for i in range(n - 1)]
    g_med = max(float(np.median(gaps)), 1.0) if gaps else 1.0
    # A "#" word (an unclear word, an arrow, a piece of maths) can be any width, so it only leans
    # weakly towards a couple of letters; otherwise it could swallow its neighbours for free.
    expected = [None if word == "#" else core_h * sum(prior_width(ch) for ch in word) for word in words]
    if scale is None:  # fit this line's own letter width when the whole line is known
        span = intervals[-1][1] - intervals[0][0]
        scale = span / (sum(e if e else 3.0 * core_h for e in expected) + (w - 1) * 0.6 * core_h)

    def width_cost(a, b, k):
        width = intervals[b][1] - intervals[a][0]
        if expected[k] is None:
            return UNKNOWN_WORD_WEIGHT * math.log(max(width, 1) / max(scale * 2.0 * core_h, 1)) ** 2
        return math.log(max(width, 1) / max(scale * expected[k], 1)) ** 2

    inf = float("inf")
    dp = [[inf] * n for _ in range(w + 1)]
    back = [[-1] * n for _ in range(w + 1)]
    for b in range(n):
        dp[1][b] = width_cost(0, b, 0)
    for k in range(2, w + 1):
        for b in range(k - 1, n):
            for a in range(k - 1, b + 1):
                prev = dp[k - 1][a - 1]
                if prev == inf:
                    continue
                cost = prev - GAP_WEIGHT * math.log(max(gaps[a - 1], 1) / g_med) + width_cost(a, b, k - 1)
                if cost < dp[k][b]:
                    dp[k][b], back[k][b] = cost, a
    starts, b = [], n - 1
    for k in range(w, 1, -1):
        a = back[k][b]
        starts.append(a)
        b = a - 1
    starts = [0] + sorted(starts)
    groups = [intervals[starts[i]:(starts[i + 1] if i + 1 < len(starts) else n)] for i in range(w)]
    chosen = set(a - 1 for a in starts[1:])

    # A gap is unclear when it could be either a word gap or a letter gap; only the words
    # touching an unclear gap (or with an unexpected width) need Claude.
    word_of = {}
    for gi, a in enumerate(starts):
        end = starts[gi + 1] if gi + 1 < len(starts) else n
        for i in range(a, end):
            word_of[i] = gi
    word_gaps = [gaps[i] for i in chosen]
    letter_gaps = [g for i, g in enumerate(gaps) if i not in chosen]
    smallest_word_gap = min(word_gaps) if word_gaps else None
    largest_letter_gap = max(letter_gaps) if letter_gaps else None
    unclear = set()
    for i, g in enumerate(gaps):
        if i in chosen and largest_letter_gap is not None and g < MIN_GAP_RATIO * largest_letter_gap:
            unclear |= {word_of[i], word_of[i + 1]}
        if i not in chosen and smallest_word_gap is not None and g * MIN_GAP_RATIO > smallest_word_gap:
            unclear.add(word_of[i])
    for k, g in enumerate(groups):
        width = g[-1][1] - g[0][0]
        if expected[k] is not None and abs(math.log(max(width, 1) / max(scale * expected[k], 1))) > 0.45:
            unclear.add(k)
    return groups, unclear


def segment_sample(sdir):
    meta = read_json(sdir / "meta.json")
    transcript = read_transcript(sdir / "transcript.txt")
    tsha = hashlib.sha256((sdir / "transcript.txt").read_bytes()).hexdigest()
    mask = cv2.imread(str(sdir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    n, cc, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    result = {"sample": meta["id"], "transcript_sha": tsha, "n_components": int(n),
              "lines": [], "words": {}, "status": "ok"}
    ink = cv2.imread(str(sdir / "ink.png"), cv2.IMREAD_GRAYSCALE)
    rows = detect_rows(mask, cc, stats, cents, ink)
    result["non_text_strokes"] = len(rows["excluded"]) - len(rows["noise"])
    result["specks_removed"] = len(rows["noise"])
    try:
        tags = parse_row_tags(transcript)
    except ValueError as e:
        result["status"], result["reason"] = "bad_transcript", str(e)
        return result, cc, stats
    if tags:
        nrows = len(rows["peaks"])
        order = [r for r, _ in tags]
        if any(r < 0 or r >= nrows for r in order) or len(set(order)) != len(order):
            result["status"] = "bad_transcript"
            result["reason"] = f"row numbers must be different and between R01 and R{nrows:02d} (see rows.png)"
            return result, cc, stats
        peaks = [int(rows["peaks"][r]) for r in order]
        comps_per_line = [rows["comp_ids"][rows["row_of"] == r] for r in order]
        mismatch = [None] * len(tags)
        transcript = [w for _, w in tags]
        result["rows_skipped"] = nrows - len(tags)
        result["pinned_rows"] = True
    else:
        matched = match_lines(rows, stats, transcript)
        if matched is None:
            result["status"] = "line_mismatch"
            result["reason"] = (f"transcript has {len(transcript)} lines but fewer rows of writing "
                                f"were found on the page")
            return result, cc, stats
        peaks, comps_per_line, mismatch, result["rows_skipped"], transcript = matched

    # First pass: each line's runs of ink, and how wide this writer's letters are, measured on
    # lines that have no "#" (a "#" hides how much of the line is unknown).
    prepared, ratios = [], []
    for words, peak, comps in zip(transcript, peaks, comps_per_line):
        if len(comps) == 0:
            prepared.append(None)
            continue
        heights = stats[comps, cv2.CC_STAT_HEIGHT]
        core_h = float(np.median(heights[heights >= np.percentile(heights, 30)]))
        intervals = drop_stray_marks(line_intervals(comps, stats, cc, core_h, float(peak)), core_h)
        prepared.append((core_h, intervals))
        if "#" not in words and intervals:
            span = intervals[-1][1] - intervals[0][0]
            expected = sum(sum(prior_width(ch) for ch in w) for w in words) + 0.6 * (len(words) - 1)
            ratios.append(span / (core_h * expected))
    page_scale = float(np.median(ratios)) if ratios else None

    wid = 0
    for li, (words, peak, comps, miss) in enumerate(zip(transcript, peaks, comps_per_line, mismatch)):
        line = {"index": li, "y": int(peak), "text": " ".join(words), "words": [], "status": "ok",
                "width_mismatch": None if miss is None else round(miss, 2)}
        doubtful = miss is not None and miss > 0.5
        result["lines"].append(line)
        if prepared[li] is None:
            line["status"] = "empty"
            continue
        core_h, intervals = prepared[li]
        core_mid = float(peak)
        if len(intervals) < len(words):
            line["status"] = "words_touching"
            continue
        line["core_h"] = core_h
        groups, unclear_words = split_words(intervals, words, core_h,
                                            scale=page_scale if "#" in words else None)

        for wi, (text, group) in enumerate(zip(words, groups)):
            confident = wi not in unclear_words and not doubtful
            wid += 1
            word_id = f"w{wid:03d}"
            wcomps = [c for iv in group for c in iv[2]]
            pieces = build_pieces(wcomps, stats, core_h, core_mid)
            for pi, p in enumerate(pieces):
                p["pid"] = f"{meta['id']}-{word_id}-p{pi + 1}"
            word = {"line": li, "index": wi, "text": text, "pieces": pieces,
                    "bbox": [min(p["bbox"][0] for p in pieces), min(p["bbox"][1] for p in pieces),
                             max(p["bbox"][2] for p in pieces), max(p["bbox"][3] for p in pieces)]}
            m, chars = len(pieces), list(text)
            word["unclear_gap"] = not confident
            if text == "#":
                word["status"], word["split"] = "skipped", None
            elif confident and m == 1:
                word["status"], word["split"] = "auto", text
            elif confident and m == len(chars):
                word["status"], word["split"] = "auto", join_split(chars)
            else:
                word["status"], word["split"] = "needs_label", None
            for p in pieces:
                p["rel_w"] = (p["bbox"][2] - p["bbox"][0]) / core_h
            result["words"][word_id] = word
            line["words"].append(word_id)
    return result, cc, stats


# Rough character widths in units of the line's typical letter height, used until
# real widths have been measured from auto-labelled words.
def prior_width(ch):
    if ch in "il.,;:'!|":
        return 0.3
    if ch in "jft()[]-\"1rI":
        return 0.5
    if ch in "mwMW":
        return 1.3
    if ch.isupper() or ch in "@%&":
        return 1.0
    return 0.8


def learn_widths(results):
    """Median measured width of each character, from words labelled one piece per character."""
    seen = {}
    for res in results:
        for w in res["words"].values():
            if w["status"] == "auto" and len(w["pieces"]) == len(w["text"]) > 1:
                for ch, p in zip(w["text"], w["pieces"]):
                    seen.setdefault(ch, []).append(p["rel_w"])
    return {ch: float(np.median(v)) for ch, v in seen.items() if len(v) >= 2}


def suggest_split(text, pieces, widths):
    """Best split of text into len(pieces) groups, matching measured piece widths."""
    from itertools import combinations

    n, m = len(text), len(pieces)
    if m == 1:
        return text
    if m == n:
        return join_split(text)
    if m > n or n > 20:
        return None
    exp = [widths.get(ch, prior_width(ch)) for ch in text]
    best = None
    for cuts in combinations(range(1, n), m - 1):
        bounds = (0,) + cuts + (n,)
        cost = 0.0
        for k in range(m):
            want = sum(exp[bounds[k]:bounds[k + 1]])
            cost += np.log(max(pieces[k]["rel_w"], 0.05) / max(want, 0.05)) ** 2
        if best is None or cost < best[0]:
            best = (cost, bounds)
    bounds = best[1]
    return join_split(text[bounds[k]:bounds[k + 1]] for k in range(m))


def tinted_word(ink, cc, word, height=110, max_w=1050):
    x0, y0, x1, y1 = word["bbox"]
    pad = 8
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(ink.shape[1], x1 + pad), min(ink.shape[0], y1 + pad)
    crop_ink = ink[y0:y1, x0:x1].astype(np.float32) / 255
    crop_cc = cc[y0:y1, x0:x1]
    rgb = np.full(crop_ink.shape + (3,), 255, np.float32)
    centers = []
    for i, p in enumerate(word["pieces"]):
        color = np.array(PIECE_COLORS[i % len(PIECE_COLORS)], np.float32)
        sel = np.isin(crop_cc, p["comps"])
        a = np.where(sel, np.maximum(crop_ink, 0.6 * sel), 0)[..., None]
        rgb = rgb * (1 - a) + color * a
        centers.append(((p["bbox"][0] + p["bbox"][2]) / 2 - x0, color))
    scale = min(height / rgb.shape[0], max_w / rgb.shape[1])
    img = Image.fromarray(rgb.astype(np.uint8)).resize(
        (max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))), Image.LANCZOS)
    return img, [(cx * scale, tuple(int(v) for v in col)) for cx, col in centers]


def write_sheets(sdir, result, cc):
    ink = cv2.imread(str(sdir / "ink.png"), cv2.IMREAD_GRAYSCALE)
    todo = [(wid, w) for wid, w in result["words"].items() if w["status"] == "needs_label"]
    sheet_dir = sdir / "sheets"
    sheet_dir.mkdir(exist_ok=True)
    for old in sheet_dir.glob("sheet_*.png"):
        old.unlink()
    f_big, f_mid, f_small = ui_font(30), ui_font(24), ui_font(20)
    paths = []
    n_sheets = (len(todo) + ROWS_PER_SHEET - 1) // ROWS_PER_SHEET
    for s in range(n_sheets):
        rows = todo[s * ROWS_PER_SHEET:(s + 1) * ROWS_PER_SHEET]
        row_h, left, width = 165, 330, 1420
        sheet = Image.new("RGB", (width, 70 + row_h * len(rows)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((20, 18), f"Sample {result['sample']}  sheet {s + 1}/{n_sheets}  "
                         f"(in labels.txt put | between the coloured pieces)", fill=(90, 90, 90), font=f_small)
        for r, (wid, w) in enumerate(rows):
            y = 70 + r * row_h
            d.line([(0, y), (width, y)], fill=(215, 215, 215), width=2)
            d.text((20, y + 22), wid, fill=(0, 0, 0), font=f_big)
            d.text((20, y + 68), f"“{w['text']}”", fill=(0, 0, 0), font=f_mid)
            d.text((20, y + 104), f"{len(w['pieces'])} pieces", fill=(110, 110, 110), font=f_small)
            if w.get("suggested"):
                d.text((20, y + 130), f"guess: {w['suggested']}", fill=(110, 110, 110), font=f_small)
            img, centers = tinted_word(ink, cc, w)
            sheet.paste(img, (left, y + 44))
            for i, (cx, col) in enumerate(centers):
                d.text((left + cx - 7, y + 10), str(i + 1), fill=col, font=f_mid)
        path = sheet_dir / f"sheet_{s + 1:02d}.png"
        sheet.save(path)
        paths.append(path)
    return paths, todo


def write_label_template(sdir, todo):
    path = sdir / "labels.txt"
    lines = ["# One line per word: <id> <pieces> <split>.",
             "# Splits are pre-filled with a guess from letter widths. Check every row against the",
             "# sheet and fix wrong ones: exactly <pieces> parts separated by |, e.g.  w017 2 th|e",
             "# A part may be empty when a piece is a stray fragment of the previous one: w018 3 i||t",
             "# Write ? as the split to skip a word (wrong word, unclear, crossed out).", ""]
    lines += [f"{wid} {len(w['pieces'])} {w.get('suggested') or w['text']}" for wid, w in todo]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_line_check(sdir, result, cc):
    """Downscaled page with every word boxed, alternate words in alternate colours per line,
    and the transcript line beside it, so line and word assignment can be checked at a glance."""
    ink = cv2.imread(str(sdir / "ink.png"), cv2.IMREAD_GRAYSCALE)
    a = (ink.astype(np.float32) / 255)[..., None]
    rgb = np.full(ink.shape + (3,), 255, np.float32) * (1 - a) + np.array([170, 170, 170], np.float32) * a
    for line in result["lines"]:
        for k, wid in enumerate(line["words"]):
            w = result["words"][wid]
            comps = [c for p in w["pieces"] for c in p["comps"]]
            x0, y0, x1, y1 = w["bbox"]
            sel = np.isin(cc[y0:y1, x0:x1], comps)[..., None]
            col = np.array(PIECE_COLORS[(line["index"] + k) % 2], np.float32)
            region = rgb[y0:y1, x0:x1]
            rgb[y0:y1, x0:x1] = np.where(sel, 255 * (1 - a[y0:y1, x0:x1]) + col * a[y0:y1, x0:x1], region)
    scale = 1600 / ink.shape[0]
    img = Image.fromarray(rgb.astype(np.uint8)).resize((int(ink.shape[1] * scale), 1600), Image.LANCZOS)
    canvas = Image.new("RGB", (img.width + 700, img.height), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    f = ui_font(18)
    for line in result["lines"]:
        for wid in line["words"]:
            x0, y0, x1, y1 = (v * scale for v in result["words"][wid]["bbox"])
            d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=(200, 200, 200))
        d.text((img.width + 10, line["y"] * scale - 10), f"{line['index'] + 1:02d} {line['text'][:60]}",
               fill=(0, 0, 0) if line["status"] == "ok" else (220, 30, 30), font=f)
    canvas.save(sdir / "lines.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="*")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--force", action="store_true", help="re-segment even if the transcript is unchanged")
    args = ap.parse_args()

    root = samples_dir(args.profile)
    if not root.exists():
        sys.exit(f"ERROR: no samples in profile '{args.profile}'. Run preprocess.py first.")
    sdirs = [root / s for s in args.samples] if args.samples else sorted(p for p in root.iterdir() if p.is_dir())
    done, others = [], []
    for sdir in sdirs:
        if read_json(sdir / "meta.json", {}).get("kind") == "template":
            continue  # read by read_template.py: its boxes are already segmented
        if not (sdir / "transcript.txt").exists():
            print(f"WAIT    {sdir.name}: no transcript.txt yet")
            continue
        old = read_json(sdir / "segments.json")
        tsha = hashlib.sha256((sdir / "transcript.txt").read_bytes()).hexdigest()
        if old and old.get("transcript_sha") == tsha and not args.force:
            print(f"SKIP    {sdir.name}: already segmented")
            others.append(old)
            continue
        if old and (sdir / "labels.txt").exists():
            (sdir / "labels.txt").replace(sdir / "labels.old.txt")
        if not (sdir / "rows.png").exists():
            write_rows_image(sdir)
        result, cc, stats = segment_sample(sdir)
        done.append((sdir, result, cc))
    for p in root.iterdir():
        if p.is_dir() and p not in sdirs and (p / "segments.json").exists():
            others.append(read_json(p / "segments.json"))

    widths = learn_widths([r for _, r, _ in done] + others)
    for sdir, result, cc in done:
        for w in result["words"].values():
            if w["status"] == "needs_label":
                w["suggested"] = suggest_split(w["text"], w["pieces"], widths)
        write_json(sdir / "segments.json", result)
        if result["status"] != "ok":
            print(f"PROBLEM {sdir.name}: {result['reason']}. Check the transcript's line breaks.")
            continue
        write_line_check(sdir, result, cc)
        sheets, todo = write_sheets(sdir, result, cc)
        labels = write_label_template(sdir, todo) if todo else None
        words = result["words"].values()
        auto = sum(w["status"] == "auto" for w in words)
        bad_lines = [l for l in result["lines"] if l["status"] != "ok"]
        unclear = sum(1 for w in words if w.get("unclear_gap"))
        print(f"OK      {sdir.name}: {len(result['lines'])} lines, {len(result['words'])} words, "
              f"{auto} auto-labelled, {len(todo)} need labels; {result.get('rows_skipped', 0)} rows of ink "
              f"not in the transcript were skipped")
        doubtful = [l for l in result["lines"] if (l.get("width_mismatch") or 0) > 0.5]
        for l in doubtful:
            print(f"        line {l['index'] + 1} may be matched to the wrong row (\"{l['text'][:40]}\"): check lines.png")
        print(f"        check: {sdir / 'lines.png'}")
        for l in bad_lines:
            if l["status"] != "placeholder":
                print(f"        line {l['index'] + 1}: {l['status']} (\"{l['text'][:40]}\") - its words are skipped")
        if unclear:
            print(f"        {unclear} words sit next to an unclear word gap (sent to sheets)")
        for p in sheets:
            print(f"        sheet: {p}")
        if labels:
            print(f"        edit:  {labels}")


if __name__ == "__main__":
    sys.exit(main())
