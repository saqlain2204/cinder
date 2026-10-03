"""Clay tablets marked in the fork tally.

A notch cut into the top of the tablet is five. A pin in the lower field is one.
The raw tally is ``5 * notches + pins``. A copper line across the middle means
the tablet was turned on the fork, and the batch number is ``9 - raw``.

The picture never contains a decimal digit. Two questions can be asked of the
same tablet: the batch number, or the pin count alone. The pin count ignores
notches and the copper line.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

TRAIN_SIZE = 64
VIEW_SIZE = 420


@dataclass(frozen=True)
class Tablet:
    """One marked tablet. ``value`` is the batch number, after the turn line."""

    value: int
    quench: bool
    notches: int
    pins: int
    seed: int

    @property
    def raw(self) -> int:
        return 5 * self.notches + self.pins


def make_tablet(rng: random.Random) -> Tablet:
    """Sample a batch number, then choose whether the turn line is present."""
    value = rng.randrange(10)
    quench = bool(rng.randrange(2))
    raw = (9 - value) if quench else value
    notches, pins = divmod(raw, 5)
    seed = rng.randrange(1_000_000_000)
    return Tablet(value=value, quench=quench, notches=notches, pins=pins, seed=seed)


def tablet_from_parts(value: int, quench: bool, seed: int = 1) -> Tablet:
    """Build the marks that encode ``value`` under the fork tally."""
    if not 0 <= value <= 9:
        raise ValueError("value must be from 0 to 9")
    raw = (9 - value) if quench else value
    notches, pins = divmod(raw, 5)
    return Tablet(value=value, quench=quench, notches=notches, pins=pins, seed=seed)


def render(tablet: Tablet, size: int = VIEW_SIZE) -> bytes:
    """Draw the tablet to a PNG. The same tablet and size always match."""
    if size < 32:
        raise ValueError("size must be at least 32")
    scale = 2 if size <= 96 else 2
    canvas = size * scale
    rng = np.random.default_rng(tablet.seed)
    clay = rng.integers(-10, 11, size=(canvas, canvas, 1), dtype=np.int16)
    tone = 8 + (tablet.seed % 7) - 3
    base = np.array([96 + tone, 62 + tone // 2, 46], dtype=np.int16)
    light = np.linspace(16, -12, canvas, dtype=np.float32)[:, None, None]
    pixels = np.clip(base + clay + light, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    draw = ImageDraw.Draw(image)

    margin = int(canvas * 0.08)
    slab = [margin, margin, canvas - margin - 1, canvas - margin - 1]
    body = (118 + tone, 78 + tone // 2, 56)
    draw.rounded_rectangle(slab, radius=int(canvas * 0.08), fill=body)

    hole_r = max(2, int(canvas * 0.028))
    hole_c = margin + int(canvas * 0.14)
    hole_y = margin + int(canvas * 0.10)
    draw.ellipse(
        [hole_c - hole_r, hole_y - hole_r, hole_c + hole_r, hole_y + hole_r],
        fill=(42, 28, 22),
    )

    _draw_notches(draw, canvas, margin, tablet.notches, rng)
    if tablet.quench:
        y = int(canvas * 0.50)
        thick = max(2, int(canvas * 0.028))
        draw.rectangle(
            [margin + int(canvas * 0.08), y, canvas - margin - int(canvas * 0.08), y + thick],
            fill=(214, 118, 62),
        )
    _draw_pins(draw, canvas, margin, tablet.pins, rng)
    _draw_freckles(draw, canvas, margin, rng)

    yy, xx = np.mgrid[0:canvas, 0:canvas]
    center = (canvas - 1) / 2
    radius = np.sqrt(((yy - center) / center) ** 2 + ((xx - center) / center) ** 2)
    vignette = np.clip(1.04 - 0.22 * radius, 0.72, 1.04).astype(np.float32)
    shaded = np.clip(np.asarray(image).astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)
    out = Image.fromarray(shaded, "RGB")
    if scale != 1:
        out = out.resize((size, size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()


def render_pair(left: Tablet, right: Tablet, size: int = VIEW_SIZE) -> bytes:
    """Place two tablets on one board. Left is the left half."""
    gap = max(8, size // 18)
    board = Image.new("RGB", (size * 2 + gap, size), (22, 16, 14))
    board.paste(Image.open(io.BytesIO(render(left, size))), (0, 0))
    board.paste(Image.open(io.BytesIO(render(right, size))), (size + gap, 0))
    buffer = io.BytesIO()
    board.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_notches(draw: ImageDraw.ImageDraw, canvas: int, margin: int, count: int, rng: np.random.Generator) -> None:
    if count <= 0:
        return
    width = int(canvas * 0.11)
    top = int(canvas * 0.24)
    height = int(canvas * 0.20)
    jitter = int(rng.integers(-canvas // 28, canvas // 28 + 1))
    cx = canvas // 2 + jitter
    left = max(margin + 4, cx - width // 2)
    draw.rounded_rectangle(
        [left, top, left + width, top + height],
        radius=max(2, width // 4),
        fill=(38, 24, 20),
    )
    draw.rectangle([left, top, left + max(2, width // 6), top + height], fill=(168, 124, 96))


def _draw_pins(draw: ImageDraw.ImageDraw, canvas: int, margin: int, count: int, rng: np.random.Generator) -> None:
    if count <= 0:
        return
    y = int(canvas * 0.72)
    inner_left = margin + int(canvas * 0.16)
    inner_right = canvas - margin - int(canvas * 0.16)
    span = inner_right - inner_left
    radius = max(3, int(canvas * 0.045))
    for index in range(count):
        center = inner_left + int((index + 1) * span / (count + 1))
        center += int(rng.integers(-canvas // 40, canvas // 40 + 1))
        draw.ellipse(
            [center - radius, y - radius, center + radius, y + radius],
            fill=(232, 196, 150),
        )
        hole = max(1, radius // 3)
        draw.ellipse(
            [center - hole, y - hole, center + hole, y + hole],
            fill=(62, 36, 28),
        )


def _draw_freckles(draw: ImageDraw.ImageDraw, canvas: int, margin: int, rng: np.random.Generator) -> None:
    """Side specks only. They stay out of the notch band and the pin row."""
    for _ in range(5):
        on_left = bool(rng.integers(0, 2))
        if on_left:
            x = int(rng.integers(margin + 4, margin + int(canvas * 0.08)))
        else:
            x = int(rng.integers(canvas - margin - int(canvas * 0.08), canvas - margin - 4))
        y = int(rng.integers(int(canvas * 0.30), int(canvas * 0.80)))
        r = max(1, int(canvas * 0.010))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(86, 56, 44))
