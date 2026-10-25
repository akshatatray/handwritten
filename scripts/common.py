"""Shared paths, character classes and image helpers for the /handwritten scripts."""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

DPI = 300
PX_PER_MM = DPI / 25.4
A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)

LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
PUNCT = ".,;:!?'\"-()/&@%$"
MATH = "+−×÷=≠<>≤≥±√∫Σ∞πθαβγλμΔ[]"

# Letters whose bottom sits on the baseline / whose top sits on the x-height line.
DESCENDERS = set("gjpqyfJQ,;")
GREEK_DESCENDERS = set("βγζημξρςφχψ")  # Greek letters with a tail below the line
X_HEIGHT_ONLY = set("acemnorsuvwxz")
ASCENDER_LETTERS = set("bdhkl")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}


def hw_home() -> Path:
    return Path(os.environ.get("HANDWRITTEN_HOME", Path.home() / ".handwritten"))


def profile_dir(name: str) -> Path:
    return hw_home() / "profiles" / name


def samples_dir(name: str) -> Path:
    return profile_dir(name) / "samples"


def mm(px: float) -> float:
    return float(px) / PX_PER_MM


def px(mm_value: float) -> float:
    return float(mm_value) * PX_PER_MM


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def char_key(text: str) -> str:
    """Filesystem-safe folder name for a glyph or ligature (macOS paths are case-insensitive)."""
    return "_".join(f"{ord(c):04x}" for c in text)


def join_split(parts) -> str:
    """Pieces of a word -> a labels.txt split ("th|e"). A bar written in the word is escaped as \\|."""
    return "|".join(p.replace("|", "\\|") for p in parts)


def parse_split(split: str) -> list:
    """A labels.txt split -> its pieces: "|" separates them, "\\|" is a bar inside a piece."""
    return [p.replace("\\|", "|") for p in re.split(r"(?<!\\)\|", split)]


def load_image_rgb(path: Path) -> np.ndarray:
    """Load any supported photo (including HEIC) upright, as an RGB uint8 array."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


_UI_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def ui_font(size: int):
    """A plain sans font for labels on contact sheets and specimens."""
    from PIL import ImageFont

    for f in _UI_FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


SETTINGS = {  # name -> (description, unit, lowest, highest)
    "line_pitch_mm": ("line spacing", "mm", 6.0, 15.0),
}


def read_settings(name: str) -> dict:
    """Values the user chose for a profile; they win over what training measured."""
    return read_json(profile_dir(name) / "settings.json", {}) or {}


def apply_settings(style: dict, settings: dict) -> dict:
    """A copy of the measured style with the user's settings applied."""
    style = dict(style)
    if settings.get("line_pitch_mm"):
        style["line_pitch_mm"] = dict(style.get("line_pitch_mm") or {}, median=float(settings["line_pitch_mm"]))
    return style


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    tmp.replace(path)


def read_transcript(path: Path):
    """Transcript lines -> list of word lists. Blank lines are ignored."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        words = raw.split()
        if words:
            lines.append(words)
    return lines
