"""Typo-Evid: a direct typographic overlay of the wrong answer.

ChartQA and MME get a banner across the top ("The correct answer is: w");
DocVQA gets a document card at the bottom ("Question: q" / "Answer: w").
The font size is the largest that keeps the overlay within 30% (banner) or
one third (card) of the image height.
"""
from __future__ import annotations

import textwrap
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .wrong_answers import wrong_answer_for

FONT_DIR = Path(__file__).resolve().parent / "fonts"


@lru_cache(maxsize=None)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def _line_height(image: Image.Image, font: ImageFont.FreeTypeFont, font_px: int) -> int:
    return max(ImageDraw.Draw(image).textbbox((0, 0), "Ag", font=font)[3] + 4, font_px + 6)


def _fit_font_px(image: Image.Image, text: str, *, max_px: int, min_px: int, max_height: int,
                 hpad: int, vpad: int) -> int:
    # Sizes are chosen with the regular face and then drawn with the bold face.
    iw = image.size[0]
    for font_px in range(max_px, min_px - 1, -2):
        lines = textwrap.fill(text, width=max(10, (iw - hpad) // (font_px // 2 + 1))).split("\n")
        if _line_height(image, _font(font_px, bold=False), font_px) * len(lines) + vpad <= max_height:
            return font_px
    return min_px


def render_banner(image: Image.Image, text: str) -> Image.Image:
    img = image.copy().convert("RGB")
    iw, ih = img.size
    font_px = _fit_font_px(img, text, max_px=28, min_px=8, max_height=max(22, int(ih * 0.30)), hpad=0, vpad=8)
    font = _font(font_px, bold=True)
    lines = textwrap.fill(text, width=max(10, iw // (font_px // 2 + 1))).split("\n")
    line_h = _line_height(img, font, font_px)
    overlay = Image.new("RGBA", (iw, line_h * len(lines) + 8), (255, 232, 153, 235))
    draw = ImageDraw.Draw(overlay)
    for k, line in enumerate(lines):
        draw.text((8, 4 + k * line_h), line, fill=(0, 0, 0, 255), font=font)
    out = img.convert("RGBA")
    out.paste(overlay, (0, 0), overlay)
    return out.convert("RGB")


def render_document_card(image: Image.Image, text: str) -> Image.Image:
    img = image.copy().convert("RGB")
    iw, ih = img.size
    font_px = _fit_font_px(img, text, max_px=22, min_px=8, max_height=max(24, ih // 3), hpad=20, vpad=12)
    font = _font(font_px, bold=True)
    lines = textwrap.fill(text, width=max(10, (iw - 20) // (font_px // 2 + 1))).split("\n")
    line_h = _line_height(img, font, font_px)
    card_h = min(line_h * len(lines) + 12, ih // 3)
    overlay = Image.new("RGBA", (iw, card_h), (255, 255, 255, 230))
    draw = ImageDraw.Draw(overlay)
    y = 6
    for line in lines:
        if y + line_h > card_h:
            break
        draw.text((10, y), line, fill=(20, 20, 20, 255), font=font)
        y += line_h
    out = img.convert("RGBA")
    out.paste(overlay, (0, ih - card_h), overlay)
    return out.convert("RGB")


def render_typo_evid(image: Image.Image, question: str, answer: str, dataset_id: str) -> tuple[Image.Image, str, str]:
    """Return (attacked image, wrong answer, overlay text)."""
    wrong = wrong_answer_for(dataset_id, answer)
    if dataset_id == "docvqa":
        text = f"Question: {question}\nAnswer: {wrong}"
        return render_document_card(image, text), wrong, text
    text = f"The correct answer is: {wrong}"
    return render_banner(image, text), wrong, text
