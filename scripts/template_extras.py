"""Maths pieces from template pages (read_template.py) for the profile: script glyphs and how high
and small the writer puts them, square-root signs, accents, and how fractions and big sums are laid
out. build_profile.py calls collect_extras(); render.py uses the result.

Every measurement is in x-heights so it applies at any writing size.
"""

import cv2
import numpy as np

from common import DESCENDERS, X_HEIGHT_ONLY, char_key, read_json

FLOATING = set("−-+=×∗*†′'⊥∞·~^")  # script characters that don't sit on the script's own baseline
ASCENDERS = set("bdfhklt")
GREEK_SHORT = set("αεικνοπστυωγημρχ")  # Greek lowercase without an ascender: x-height letters
GREEK_TALL = set("βδζθλξφψ")
DOUBLE_STRUCK = set("ℝℂ𝔽ℕℤℚ𝕊𝔼")


def _page(sdir):
    mask = cv2.imread(str(sdir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    ink = cv2.imread(str(sdir / "ink.png"), cv2.IMREAD_GRAYSCALE)
    n, cc, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return ink, cc, stats, n


def _crop(ink, cc, comps, bbox, pad=6):
    x0, y0, x1, y1 = bbox
    y0p, x0p = max(0, y0 - pad), max(0, x0 - pad)
    y1p, x1p = min(ink.shape[0], y1 + pad), min(ink.shape[1], x1 + pad)
    own = np.isin(cc[y0p:y1p, x0p:x1p], comps)
    own = cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return np.where(own, ink[y0p:y1p, x0p:x1p], 0).astype(np.uint8), x0p, y0p


def _bbox(stats, comps):
    return [int(min(stats[c, 0] for c in comps)), int(min(stats[c, 1] for c in comps)),
            int(max(stats[c, 0] + stats[c, 2] for c in comps)), int(max(stats[c, 1] + stats[c, 3] for c in comps))]


def _save(img, out_dir, gid, pdir, **meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{gid}.png"
    cv2.imwrite(str(path), img)
    return {"id": gid, "file": str(path.relative_to(pdir)), "w": int(img.shape[1]), "h": int(img.shape[0]), **meta}


def _normal_height(ch, xh, style):
    """How tall ch is when written at full size, in pixels (None when it has no usual height)."""
    cap = ((style.get("cap_height_mm") or {}).get("median") or 1.45 * style["x_height_mm"]) / style["x_height_mm"] * xh
    asc = ((style.get("ascender_mm") or {}).get("median") or 1.75 * style["x_height_mm"]) / style["x_height_mm"] * xh
    if ch.isdigit() or ch.isupper() or ch in DOUBLE_STRUCK:
        return cap
    if ch in X_HEIGHT_ONLY or ch in GREEK_SHORT:
        return xh
    if ch in ASCENDERS or ch in GREEK_TALL:
        return asc
    return None


def _box_size(e, xh, style):
    """How much bigger than usual the letter carrying the scripts was written in its box, or None.
    People write bigger in boxes; scripts are measured against their own letter, not the page."""
    base = e["base"]
    normal = _normal_height(base, xh, style) if len(base) == 1 else None
    if not normal or not e.get("base_bbox"):
        return None
    f = (e["baseline"] - e["base_bbox"][1]) / normal
    return f if 0.6 < f < 2.2 else None


def _script_group(units, stats):
    """The script's own baseline: the bottom of its letters and digits that sit on a line."""
    anchors = [u for u in units if u["text"] not in FLOATING and not set(u["text"]) & DESCENDERS]
    if anchors:
        return float(np.median([_bbox(stats, u["comps"])[3] for u in anchors]))
    return float(max(_bbox(stats, u["comps"])[3] for u in units))


def collect_extras(sroot, pdir, rejected, style):
    out = {"scripts": {}, "script_style": None, "radicals": [], "accents": {}, "fraction_style": None,
           "bigops": {}, "bigop_style": None}
    sup_raise, sub_drop, sup_scale, sub_scale = [], [], [], []
    sup_raise_rel = []  # a power's baseline above the line, as a share of its letter's height
    by_class = {("sup", True): [], ("sup", False): [], ("sub", True): [], ("sub", False): []}
    frac = {"num_gap": [], "den_gap": [], "overhang": [], "num_scale": []}
    big = {"height": [], "upper_gap": [], "lower_gap": [], "limit_scale": []}
    for sdir in sorted(p for p in sroot.iterdir() if p.is_dir()):
        t = read_json(sdir / "template.json")
        seg = read_json(sdir / "segments.json")
        if not t or not seg:
            continue
        ink, cc, stats, n = _page(sdir)
        if n != seg["n_components"]:
            print(f"WARNING {sdir.name}: page changed since read_template.py; read it again with --redo")
            continue
        xh = float(t["xh"])
        sid = sdir.name

        sizes = [f for f in (_box_size(e, xh, style) for e in t["scripts"] if e.get("ok")) if f]
        page_size = float(np.median(sizes)) if sizes else 1.0
        for e in t["scripts"]:
            if not e.get("ok"):
                continue
            base_y = e["baseline"]
            measured = _box_size(e, xh, style)
            f = measured or page_size  # this box's letters relative to the writer's usual size
            short = len(e["base"]) == 1 and (e["base"] in X_HEIGHT_ONLY or e["base"] in GREEK_SHORT)
            for level in ("sup", "sub"):
                units = e[level]
                if not units:
                    continue
                y_line = _script_group(units, stats)
                if measured:
                    if level == "sup":
                        sup_raise.append((base_y - y_line) / (xh * f))
                        sup_raise_rel.append((base_y - y_line) / max(1.0, base_y - e["base_bbox"][1]))
                    else:
                        sub_drop.append((y_line - base_y) / (xh * f))
                for u in units:
                    if u.get("whole"):
                        continue
                    ch = u["text"]
                    gid = f"{sid}-{e['cell']}-{level}-{ch_index(units, u)}"
                    if gid in rejected:
                        continue
                    bb = _bbox(stats, u["comps"])
                    normal = _normal_height(ch, xh, style)
                    if normal and ch not in FLOATING and measured:
                        size = (y_line - bb[1]) / (normal * f)
                        (sup_scale if level == "sup" else sub_scale).append(size)
                        by_class[(level, short)].append(size)
                    img, x0p, y0p = _crop(ink, cc, u["comps"], bb)
                    # Stored at the writer's usual size: the box's own enlargement taken out.
                    if abs(f - 1) > 0.03:
                        img = cv2.resize(img, (max(1, round(img.shape[1] / f)), max(1, round(img.shape[0] / f))),
                                         interpolation=cv2.INTER_AREA if f > 1 else cv2.INTER_CUBIC)
                    meta = _save(img, pdir / "scripts" / char_key(ch), gid, pdir,
                                 baseline=round((y_line - y0p) / f, 1), x_height_px=round(xh, 1), level=level)
                    out["scripts"].setdefault(ch, []).append(meta)

        for e in t["radicals"]:
            gid = f"{sid}-{e['cell']}"
            if gid in rejected or not e["comps"]:
                continue
            img, x0p, y0p = _crop(ink, cc, e["comps"], e["bbox"])
            ph = [e["placeholder"][0] - x0p, e["placeholder"][1] - y0p, e["placeholder"][2] - x0p,
                  e["placeholder"][3] - y0p]
            split = max(1, min(img.shape[1] - 2, int(ph[0])))
            right = img[:, split:]
            tops = [int(np.nonzero(right[:, c] > 60)[0][0]) for c in range(right.shape[1]) if (right[:, c] > 60).any()]
            bar_y = float(np.median(tops)) if tops else float(ph[1])
            out["radicals"].append(_save(img, pdir / "radicals", gid, pdir, split_x=split, bar_y=round(bar_y, 1),
                                         placeholder=[round(v, 1) for v in ph],
                                         baseline=round(e["baseline"] - y0p, 1), x_height_px=round(xh, 1)))

        for e in t["accents"]:
            base_y = e["baseline"]
            pieces = e["pieces"]
            base = [p for p in pieces if p["bbox"][3] >= base_y - 0.3 * xh]
            top = min((p["bbox"][1] for p in base), default=base_y - xh)
            accent = [p for p in pieces if p not in base and p["bbox"][3] <= top + 0.15 * xh]
            gid = f"{sid}-{e['cell']}"
            if not accent or not base or gid in rejected:
                continue
            ac = [c for p in accent for c in p["comps"]]
            bc = [c for p in base for c in p["comps"]]
            ab, bb = _bbox(stats, ac), _bbox(stats, bc)
            img, x0p, y0p = _crop(ink, cc, ac, ab)
            meta = _save(img, pdir / "accents" / char_key(e["accent"]), gid, pdir, baseline=float(img.shape[0] - 6),
                         dx=round(((ab[0] + ab[2]) - (bb[0] + bb[2])) / 2 / xh, 2),
                         gap=round((bb[1] - ab[3]) / xh, 2),
                         rel_w=round((ab[2] - ab[0]) / max(1, bb[2] - bb[0]), 2), x_height_px=round(xh, 1))
            out["accents"].setdefault(e["accent"], []).append(meta)

        for e in t["fractions"]:
            pieces = e["pieces"]
            bars = [p for p in pieces if (p["bbox"][3] - p["bbox"][1]) < 0.6 * xh
                    and (p["bbox"][2] - p["bbox"][0]) > 1.2 * xh]
            if not bars:
                continue
            bar = max(bars, key=lambda p: p["bbox"][2] - p["bbox"][0])
            by = (bar["bbox"][1] + bar["bbox"][3]) / 2
            num = [p for p in pieces if p is not bar and p["bbox"][3] <= by]
            den = [p for p in pieces if p is not bar and p["bbox"][1] >= by]
            if not num or not den:
                continue
            nb = [min(p["bbox"][0] for p in num), min(p["bbox"][1] for p in num),
                  max(p["bbox"][2] for p in num), max(p["bbox"][3] for p in num)]
            db = [min(p["bbox"][0] for p in den), min(p["bbox"][1] for p in den),
                  max(p["bbox"][2] for p in den), max(p["bbox"][3] for p in den)]
            frac["num_gap"].append((by - nb[3]) / xh)
            frac["den_gap"].append((db[1] - by) / xh)
            bw = bar["bbox"][2] - bar["bbox"][0]
            frac["overhang"].append((bw - max(nb[2] - nb[0], db[2] - db[0])) / 2 / xh)
            if e["num"] in ("1", "2", "n", "a"):
                normal = _normal_height(e["num"], xh, style)
                if normal:
                    frac["num_scale"].append((nb[3] - nb[1]) / normal)

        for e in t["bigops"]:
            pieces = e["pieces"]
            if not pieces:
                continue
            tallest = max(pieces, key=lambda p: p["bbox"][3] - p["bbox"][1])
            ty0, ty1 = tallest["bbox"][1], tallest["bbox"][3]
            if e["op"] == "Σ":
                op = [tallest]
            else:  # max/min: every piece in the band of the tallest letter
                op = [p for p in pieces if ty0 - 0.3 * xh <= (p["bbox"][1] + p["bbox"][3]) / 2 <= ty1 + 0.3 * xh]
            ob = [min(p["bbox"][0] for p in op), min(p["bbox"][1] for p in op),
                  max(p["bbox"][2] for p in op), max(p["bbox"][3] for p in op)]
            upper = [p for p in pieces if p not in op and p["bbox"][3] <= ob[1] + 0.2 * xh]
            lower = [p for p in pieces if p not in op and p["bbox"][1] >= ob[3] - 0.2 * xh]
            if upper:
                big["upper_gap"].append((ob[1] - max(p["bbox"][3] for p in upper)) / xh)
            if lower:
                big["lower_gap"].append((min(p["bbox"][1] for p in lower) - ob[3]) / xh)
                lh = max(p["bbox"][3] for p in lower) - min(p["bbox"][1] for p in lower)
                big["limit_scale"].append(lh / (_normal_height("1", xh, style) or xh))
            if e["op"] == "Σ":
                big["height"].append((ob[3] - ob[1]) / xh)
                gid = f"{sid}-{e['cell']}"
                if gid in rejected:
                    continue
                comps = [c for p in op for c in p["comps"]]
                img, x0p, y0p = _crop(ink, cc, comps, ob)
                meta = _save(img, pdir / "bigops" / char_key("Σ"), gid, pdir, baseline=round(ob[3] - y0p, 1),
                             centre=round((ob[1] + ob[3]) / 2 - y0p, 1), x_height_px=round(xh, 1))
                out["bigops"].setdefault("Σ", []).append(meta)

    med = lambda v: round(float(np.median(v)), 3) if v else None
    if sup_raise or sub_drop:
        out["script_style"] = {"sup_raise": med(sup_raise), "sub_drop": med(sub_drop),
                               "sup_scale": med(sup_scale), "sub_scale": med(sub_scale),
                               "sup_raise_rel": med(sup_raise_rel),
                               # after a short letter (x, u, σ) and after a tall one (A, ℝ, Σ)
                               "sup_scale_short": med(by_class[("sup", True)]),
                               "sup_scale_tall": med(by_class[("sup", False)]),
                               "sub_scale_short": med(by_class[("sub", True)]),
                               "sub_scale_tall": med(by_class[("sub", False)]),
                               "n": len(sup_raise) + len(sub_drop)}
    if frac["num_gap"]:
        out["fraction_style"] = {k: med(v) for k, v in frac.items()}
    if big["height"] or big["lower_gap"]:
        out["bigop_style"] = {k: med(v) for k, v in big.items()}
    return out


def ch_index(units, u):
    return next(k for k, x in enumerate(units) if x is u) + 1


def report_lines(profile):
    lines = []
    sc = profile.get("scripts") or {}
    st = profile.get("script_style")
    if sc or st:
        txt = "  scripts: " + " ".join(f"{ch}({len(v)})" for ch, v in sorted(sc.items()))
        if st:
            txt += (f"  (powers {st['sup_scale']} of normal size, raised {st.get('sup_raise_rel')} of their "
                    f"letter's height; indices {st['sub_scale']} of normal, lowered {st['sub_drop']} x-heights)")
        lines.append(txt)
    if profile.get("radicals"):
        lines.append(f"  roots: {len(profile['radicals'])} √ signs of your own, bar stretched to fit")
    if profile.get("accents"):
        lines.append("  accents: " + " ".join(f"x{m}({len(v)})" for m, v in profile["accents"].items()))
    if profile.get("fraction_style"):
        lines.append("  fractions: your own spacing round the bar")
    if profile.get("bigops") or profile.get("bigop_style"):
        n = sum(len(v) for v in (profile.get("bigops") or {}).values())
        lines.append(f"  big sums: {n} large Σ of your own, limits placed your way")
    return lines
