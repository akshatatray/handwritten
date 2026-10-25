"""Turn a clean handwritten page into what looks like a phone photo of real paper.

The page still fills the frame at its own proportions (no desk around it). Every effect is drawn
from the seed, so each page of a set gets its own light, tilt and grain.
"""

import cv2
import numpy as np


def smooth_noise(shape, cell, rng):
    """Smooth random field in about -1..1 with features `cell` pixels across."""
    h, w = shape
    small = rng.normal(0, 1, (max(2, h // cell + 2), max(2, w // cell + 2))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def cover_zoom(dst, centre, w, h):
    """The smallest zoom of the tilted page outline dst that still covers the whole frame."""
    corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    lo, hi = 1.0, 1.1
    for _ in range(20):
        mid = (lo + hi) / 2
        quad = (centre + (dst - centre) * mid).reshape(-1, 1, 2)
        if all(cv2.pointPolygonTest(quad, (float(x), float(y)), False) >= 0 for x, y in corners):
            hi = mid
        else:
            lo = mid
    return hi


def photo_look(alpha, ink_rgb, seed):
    """alpha: HxW float32 ink coverage 0..1. Returns an HxWx3 uint8 RGB image."""
    rng = np.random.default_rng(seed)
    h, w = alpha.shape

    # Ballpoint ink: slightly soft edges, pressure varying along the strokes.
    a = cv2.GaussianBlur(alpha, (0, 0), 0.6)
    a *= 0.9 + 0.1 * np.clip(smooth_noise((h, w), 45, rng), -1, 1)
    ink = np.array(ink_rgb, np.float32) * (1 + 0.06 * rng.uniform(-1, 1))

    # Paper: a clean, barely warm white with fine grain and faint mottling.
    base = np.array([249, 247, 242], np.float32) + rng.normal(0, 1.2, 3).astype(np.float32)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8) * 2.2
    grain += smooth_noise((h, w), 90, rng) * 1.6
    img = (base[None, None, :] + grain[..., None]) * (1 - a[..., None]) + ink[None, None, :] * a[..., None]

    # Light: a gentle gradient across the page, a vignette, and a soft shadow from one edge.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = xx / w - 0.5, yy / h - 0.5
    theta = rng.uniform(0, 2 * np.pi)
    grad = nx * np.cos(theta) + ny * np.sin(theta)
    light = 1 - rng.uniform(0.025, 0.06) * (grad + 0.5)
    light *= 1 - rng.uniform(0.035, 0.07) * (nx ** 2 + ny ** 2) * 2
    edge = rng.integers(4)
    dist = [yy / h, 1 - yy / h, xx / w, 1 - xx / w][edge]
    width = rng.uniform(0.12, 0.3)
    fall = np.clip(1 - dist / width, 0, 1)
    light *= 1 - rng.uniform(0.025, 0.06) * fall * fall * (3 - 2 * fall)
    img *= light[..., None]
    del yy, xx, nx, ny, grad, dist, fall, light

    # Phone white balance: only a hint warm.
    img *= np.array([1.0, 0.99, 0.972], np.float32)

    # A slight tilt: move the corners a little, then zoom in just enough that no edge shows, so
    # writing near the top of the page (an identity block) stays in the frame.
    j = 0.012 * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, (4, 2)).astype(np.float32)
    centre = np.float32([w / 2, h / 2])
    dst = centre + (dst - centre) * (cover_zoom(dst, centre, w, h) + 0.004)
    m = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Camera: a little softness, sensor noise and JPEG compression.
    img = cv2.GaussianBlur(img, (0, 0), 0.5)
    img += rng.normal(0, 1.7, (h, w, 1)).astype(np.float32)
    img += rng.normal(0, 0.6, (h, w, 3)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 84])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
