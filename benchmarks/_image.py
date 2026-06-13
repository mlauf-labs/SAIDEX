"""Render plain text into a PNG image for use as vision-model input.

Used by the vision benchmark group: invoice texts are rendered to a clean,
high-contrast document image so that a vision-capable model has to perform
OCR-style extraction instead of reading the raw text.
"""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# A short list of monospaced font candidates per platform. Monospace keeps the
# ASCII-art invoice layouts (tables, separators) aligned in the rendered image.
_FONT_CANDIDATES: tuple[str, ...] = (
    "consola.ttf",  # Windows — Consolas
    "cour.ttf",  # Windows — Courier New
    "DejaVuSansMono.ttf",  # Linux (matplotlib/PIL bundled)
    "/System/Library/Fonts/Menlo.ttc",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a monospaced TrueType font, falling back to PIL's default."""
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_text_to_image(
    text: str,
    output_path: str | Path,
    *,
    font_size: int = 18,
    padding: int = 32,
    line_spacing: int = 6,
    background: str = "white",
    foreground: str = "black",
) -> Path:
    """Render *text* to a PNG file at *output_path* and return the path.

    The image width/height are sized to fit the longest line and the total
    number of lines, so the full document is always visible.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font = _load_font(font_size)
    lines = text.replace("\t", "    ").splitlines() or [""]

    # Measure line dimensions with a throwaway drawing context.
    measure_img = Image.new("RGB", (1, 1))
    measure = ImageDraw.Draw(measure_img)

    line_heights: list[int] = []
    max_width = 1
    for line in lines:
        bbox = measure.textbbox((0, 0), line or " ", font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_width = max(max_width, width)
        line_heights.append(max(height, font_size))

    total_height = padding * 2 + sum(line_heights) + line_spacing * (len(lines) - 1)
    total_width = padding * 2 + max_width

    img = Image.new("RGB", (total_width, total_height), color=background)
    draw = ImageDraw.Draw(img)

    y = padding
    for line, height in zip(lines, line_heights):
        draw.text((padding, y), line, fill=foreground, font=font)
        y += height + line_spacing

    img.save(output_path, format="PNG")
    return output_path


def image_to_data_url(path: str | Path, media_type: str = "image/png") -> str:
    """Return a ``data:`` base64 URL for the image at *path*."""
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode()
    return f"data:{media_type};base64,{encoded}"
