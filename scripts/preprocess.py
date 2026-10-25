"""Turn phone photos of handwritten pages into clean, flat, 300 DPI ink maps.

For every photo this finds the page, removes perspective, evens out the lighting,
removes ruled lines, deskews, and works out the physical scale. Results go to
~/.handwritten/profiles/<profile>/samples/<id>/:

  original.<ext>   untouched copy of the photo
  ink.png          ink intensity, 0 = paper, 255 = full ink, 300 DPI
  mask.png         binary ink mask used for segmentation
  preview*.jpg     what Claude reads to write the transcript
  meta.json        page detection, scale, skew, ruled-line info

Usage: preprocess.py [--profile NAME] [--page-mm 210x297] PHOTO_OR_FOLDER...
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.filters import threshold_sauvola

from common import (A4_MM, IMAGE_EXTS, LETTER_MM, PX_PER_MM, file_sha, load_image_rgb,
                    read_json, samples_dir, write_json)

DEFAULT_RULED_PITCH_MM = 8.0  # common notebook ruling, used only when the page edges are not visible
DEFAULT_LINE_PITCH_MM = 9.0  # typical unruled line pitch, last-resort scale guess
FRAME_PAGE_ASPECT = (1.33, 1.56)  # an image this shape with no visible edges is a scan cropped to the page


def collect_images(args_paths):
    out = []
    for p in map(Path, args_paths):
        p = p.expanduser()
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            print(f"WARNING: not found: {p}", file=sys.stderr)
    return out


def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def find_page_quad(rgb):
    """Return the page's 4 corners in full-resolution coordinates, or None."""
    h, w = rgb.shape[:2]
    scale = 1200.0 / max(h, w)
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    sh, sw = gray.shape
    img_area = sh * sw

    candidates = []
    # White paper is almost colourless; desks, wood and cloth are tinted. Bright low-saturation
    # pixels give the page even when it is barely lighter than the desk.
    sh, sw = gray.shape
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_RGB2HSV)
    paper = ((hsv[..., 1] < 30) & (hsv[..., 2] > 120)).astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))  # fill the writing
    # Cut away other white things touching the page (a cable, a monitor stand): they are much
    # thinner than a sheet of paper, so a wide opening removes them and leaves the page.
    k = int(0.07 * max(sh, sw)) | 1
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    candidates.append(paper)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    candidates.append(otsu)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    candidates.append(edges)

    # Score every 4-cornered outline: paper-shaped outlines beat bigger ones, and an outline with a
    # corner on the photo's edge is a page merged with something pale beside it (or one that fills
    # the frame), so it is not used.
    margin = 0.01 * max(sh, sw)
    best = None
    for binimg in candidates:
        contours, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < 0.25 * img_area:
                break
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.03, 0.05, 0.07):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = order_corners(approx)
                    if any(x < margin or y < margin or x > sw - margin or y > sh - margin for x, y in quad):
                        break
                    tl, tr, br, bl = quad
                    qw = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
                    qh = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
                    aspect = max(qw, qh) / max(1.0, min(qw, qh))
                    paper_like = any(abs(aspect - d[1] / d[0]) < 0.06 for d in (A4_MM, LETTER_MM))
                    score = (paper_like, area)
                    if best is None or score > best[0]:
                        best = (score, quad)
                    break
    if best is None:
        return None
    return best[1] / scale


def classify_paper(quad, page_mm_override):
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    portrait = height >= width
    if page_mm_override:
        w_mm, h_mm = sorted(page_mm_override)
        return ("custom", (w_mm, h_mm) if portrait else (h_mm, w_mm), width, height)
    aspect = max(width, height) / min(width, height)
    for name, dims in (("A4", A4_MM), ("letter", LETTER_MM)):
        if abs(aspect - dims[1] / dims[0]) < 0.06:
            return (name, dims if portrait else dims[::-1], width, height)
    return ("unknown", None, width, height)


def flatten_lighting(gray):
    """Divide out the paper's lighting so the background becomes uniform white."""
    k = max(31, (min(gray.shape) // 40) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, 21)
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1), 0, 1)
    return norm


def remove_ruled_lines(mask):
    """Remove long horizontal and vertical rules; returns (mask, ruled_pitch_px or None)."""
    h, w = mask.shape
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 12, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 12)))
    lines = cv2.bitwise_or(horiz, vert)
    pitch = None
    rows = np.where(horiz.sum(axis=1) > 0.3 * w * 255)[0]
    if len(rows) > 3:
        starts = rows[np.insert(np.diff(rows) > 3, 0, True)]
        if len(starts) > 3:
            pitch = float(np.median(np.diff(starts)))
    if not lines.any():
        return mask, pitch
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    cleaned = cv2.bitwise_and(mask, cv2.bitwise_not(lines))
    # Reconnect strokes that crossed a rule.
    bridged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    cleaned = cv2.bitwise_or(cleaned, cv2.bitwise_and(bridged, lines))
    return cleaned, pitch


def estimate_skew(mask):
    small = cv2.resize(mask, (mask.shape[1] // 4, mask.shape[0] // 4), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    center = (w / 2, h / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-5, 5.01, 0.25):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def text_line_pitch(mask):
    """Median distance between neighbouring text lines (px), from peaks of the row profile."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    heights = heights[heights >= 5]
    if len(heights) < 10:
        return None
    letter_h = float(np.median(heights))
    prof = gaussian_filter1d(mask.sum(axis=1).astype(np.float64), sigma=max(1.0, 0.4 * letter_h))
    peaks, _ = find_peaks(prof, distance=max(3, int(1.2 * letter_h)), prominence=prof.max() * 0.05)
    if len(peaks) < 3:
        return None
    return float(np.median(np.diff(peaks)))


def write_previews(ink, out_dir):
    page = 255 - ink
    h, w = page.shape
    full = cv2.resize(page, (int(w * 1600 / h), 1600), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "preview.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 88])
    names = []
    overlap = int(0.06 * h)
    for i, (y0, y1) in enumerate(((0, h // 2 + overlap), (h // 2 - overlap, h))):
        part = page[y0:y1]
        scale = 1500 / max(part.shape)
        part = cv2.resize(part, (int(part.shape[1] * scale), int(part.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
        name = f"preview_{'top' if i == 0 else 'bottom'}.jpg"
        cv2.imwrite(str(out_dir / name), part, [cv2.IMWRITE_JPEG_QUALITY, 90])
        names.append(name)
    return ["preview.jpg"] + names


def process(path, out_root, page_mm_override, redo=False, corners=None):
    sha = file_sha(path)
    sid = sha[:12]
    out_dir = out_root / sid
    if (out_dir / "meta.json").exists():
        if not redo:
            return sid, read_json(out_dir / "meta.json"), True
        # The page geometry changes, so earlier segmentation and labels no longer apply.
        # The transcript is kept: the words on the page are the same.
        for stale in ("segments.json", "lines.png"):
            (out_dir / stale).unlink(missing_ok=True)
        if (out_dir / "labels.txt").exists():
            (out_dir / "labels.txt").replace(out_dir / "labels.old.txt")
        shutil.rmtree(out_dir / "sheets", ignore_errors=True)

    rgb = load_image_rgb(path)
    meta = {"id": sid, "sha256": sha, "source": str(path), "photo_px": list(rgb.shape[1::-1])}

    quad = np.array(corners, np.float32) if corners is not None else find_page_quad(rgb)
    if quad is not None:
        paper, dims_mm, qw, qh = classify_paper(quad, page_mm_override)
        if dims_mm:
            out_w, out_h = int(round(dims_mm[0] * PX_PER_MM)), int(round(dims_mm[1] * PX_PER_MM))
            scale_source = "page_edges"
        else:
            out_w, out_h = int(qw), int(qh)
            scale_source = None
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
        m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        rgb = cv2.warpPerspective(rgb, m, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        meta["page"] = {"found": True, "paper": paper, "mm": list(dims_mm) if dims_mm else None,
                        "corners_px": quad.round(1).tolist()}
    else:
        h, w = rgb.shape[:2]
        aspect = max(h, w) / min(h, w)
        if FRAME_PAGE_ASPECT[0] <= aspect <= FRAME_PAGE_ASPECT[1]:
            # Scanner apps crop the photo to the page, so the image's short side is the paper's.
            short_mm = min(page_mm_override) if page_mm_override else A4_MM[0]
            factor = short_mm * PX_PER_MM / min(h, w)
            rgb = cv2.resize(rgb, (int(round(w * factor)), int(round(h * factor))),
                             interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)
            scale_source = "frame_is_page"
            meta["page"] = {"found": False, "frame_is_page": True,
                            "paper": "custom" if page_mm_override else "A4", "short_side_mm": short_mm}
            meta["scale_factor"] = factor
        else:
            meta["page"] = {"found": False}
            scale_source = None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    norm = flatten_lighting(gray)
    ink = 1.0 - norm

    win = max(25, (int(min(gray.shape) / 60)) | 1)
    sauvola = threshold_sauvola(norm, window_size=win, k=0.2)
    mask = ((norm < sauvola) & (ink > 0.18)).astype(np.uint8) * 255

    # Page edges and shadows near the border are never writing. Scans cropped to the page can
    # have writing close to the edge, so they lose only a thin strip.
    b = int((0.005 if meta["page"].get("frame_is_page") else 0.015) * max(mask.shape))
    mask[:b] = 0
    mask[-b:] = 0
    mask[:, :b] = 0
    mask[:, -b:] = 0

    mask, ruled_pitch = remove_ruled_lines(mask)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < 15)[0]
    mask[np.isin(cc, small)] = 0

    angle = estimate_skew(mask)
    if abs(angle) > 0.1:
        h, w = mask.shape
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        mask = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_NEAREST)
        ink = cv2.warpAffine(ink, rot, (w, h), flags=cv2.INTER_LINEAR)
    meta["deskew_deg"] = angle
    meta["ruled"] = {"found": ruled_pitch is not None, "pitch_px": ruled_pitch}

    if scale_source is None:
        if ruled_pitch:
            factor = DEFAULT_RULED_PITCH_MM * PX_PER_MM / ruled_pitch
            scale_source = "ruled_lines_assumed_8mm"
        else:
            pitch = text_line_pitch(mask)
            factor = DEFAULT_LINE_PITCH_MM * PX_PER_MM / pitch if pitch else 1.0
            scale_source = "line_pitch_assumed_9mm" if pitch else "unknown"
        if abs(factor - 1) > 0.02:
            h, w = mask.shape
            size = (max(1, int(w * factor)), max(1, int(h * factor)))
            mask = (cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
            ink = cv2.resize(ink, size, interpolation=cv2.INTER_LINEAR)
        meta["scale_factor"] = factor
    meta["scale_source"] = scale_source

    # Keep soft stroke edges for the ink map, but nothing outside the strokes.
    keep = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
    ink_vals = ink[mask > 0]
    hi = float(np.percentile(ink_vals, 95)) if ink_vals.size else 1.0
    ink8 = (np.clip(ink / max(hi, 0.2), 0, 1) * 255 * keep).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out_dir / f"original{path.suffix.lower()}")
    cv2.imwrite(str(out_dir / "ink.png"), ink8)
    cv2.imwrite(str(out_dir / "mask.png"), mask)
    meta["size_px"] = [int(mask.shape[1]), int(mask.shape[0])]
    meta["previews"] = write_previews(ink8, out_dir)
    meta["ink_coverage"] = float((mask > 0).mean())
    write_json(out_dir / "meta.json", meta)
    from segment_freeform import write_rows_image
    _, meta["rows"] = write_rows_image(out_dir)
    write_json(out_dir / "meta.json", meta)
    return sid, meta, False


def parse_page_mm(text):
    if not text:
        return None
    w, h = text.lower().split("x")
    return float(w), float(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--page-mm", help="real paper size, e.g. 176x250 for a B5 notebook")
    ap.add_argument("--redo", action="store_true", help="re-process photos already in the profile")
    ap.add_argument("--corners", help="page corners in the photo when detection fails, as x,y pixel pairs: "
                                      "top-left top-right bottom-right bottom-left, e.g. 82,250,1045,250,1040,1650,75,1640")
    args = ap.parse_args()

    images = collect_images(args.paths)
    if not images:
        sys.exit("ERROR: no images found (supported: " + ", ".join(sorted(IMAGE_EXTS)) + ")")
    out_root = samples_dir(args.profile)
    out_root.mkdir(parents=True, exist_ok=True)

    for path in images:
        try:
            corners = None
            if args.corners:
                v = [float(x) for x in args.corners.split(",")]
                if len(v) != 8 or len(images) != 1:
                    sys.exit("ERROR: --corners takes 8 numbers and exactly one photo")
                corners = [v[0:2], v[2:4], v[4:6], v[6:8]]
            sid, meta, existed = process(path, out_root, parse_page_mm(args.page_mm), args.redo or bool(corners),
                                         corners)
        except Exception as e:  # keep going: one bad photo should not stop the batch
            print(f"FAILED  {path}: {e}")
            continue
        d = out_root / sid
        if existed:
            print(f"SKIP    {path.name} -> {sid} (already in profile)")
            continue
        page = meta["page"]
        if page["found"]:
            page_txt = f"page found ({page['paper']})"
        elif page.get("frame_is_page"):
            page_txt = f"image is the whole page, assumed {page['paper']} width"
        else:
            page_txt = "page edges NOT found"
        warn = ""
        if meta["ink_coverage"] < 0.003:
            warn = "  WARNING: almost no ink found"
        elif meta["ink_coverage"] > 0.2:
            warn = "  WARNING: too much ink, check lighting/shadows"
        print(f"OK      {path.name} -> {sid}: {page_txt}, scale from {meta['scale_source']}, "
              f"deskew {meta['deskew_deg']:+.2f} deg{warn}")
        print(f"        read:  {d / 'preview_top.jpg'}")
        print(f"               {d / 'preview_bottom.jpg'}")
        print(f"        rows:  {d / 'rows.png'} ({meta['rows']} rows found)")
        print(f"        write: {d / 'transcript.txt'}")


if __name__ == "__main__":
    main()
