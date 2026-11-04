"""Draws the MosaiCxM app icon and writes every size the release needs.

Run from the repo root:  python3 scripts/make_icons.py   (needs Pillow)
Outputs (committed):
  crates/mosaic-gui/ui/icon.png          256 px, window icon
  crates/mosaic-gui/icons/icon-1024.png  master, used for the macOS .icns
  crates/mosaic-gui/icons/icon.ico       Windows .exe icon (16-256 px)
"""
from PIL import Image, ImageDraw

import math

S = 1024
RUST = "#e8590c"  # Rust-style orange for the gear
TEAL, CORAL, AMBER, VIOLET = "#0f9d8a", "#e2493d", "#f2b100", "#6c5ce7"
# 3x3 photo tiles inside the gear, colours sweep diagonally.
GRID = [
    [TEAL, TEAL, AMBER],
    [TEAL, VIOLET, CORAL],
    [AMBER, CORAL, VIOLET],
]


def gear(cx, cy, r_out, r_in, teeth):
    """Points of a gear outline with flat-topped teeth."""
    pts = []
    step = 2 * math.pi / teeth
    for i in range(teeth):
        a = i * step
        for frac, r in ((0.0, r_in), (0.12, r_out), (0.42, r_out), (0.54, r_in)):
            t = a + frac * step
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


def draw(size=S):
    k = 4  # supersample for smooth edges
    n = size * k
    c = n / 2
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = n * 0.04
    d.rounded_rectangle([pad, pad, n - pad, n - pad], radius=n * 0.21, fill="#1d2233")
    d.polygon(gear(c, c, n * 0.42, n * 0.355, 12), fill=RUST)
    hole = n * 0.29
    d.ellipse([c - hole, c - hole, c + hole, c + hole], fill="#1d2233")
    side = hole * 1.3
    gap = side * 0.06
    cell = (side - 2 * gap) / 3
    x0 = y0 = c - side / 2
    for r, row in enumerate(GRID):
        for col_i, col in enumerate(row):
            x = x0 + col_i * (cell + gap)
            y = y0 + r * (cell + gap)
            d.rounded_rectangle([x, y, x + cell, y + cell], radius=cell * 0.2, fill=col)
    return img.resize((size, size), Image.LANCZOS)


master = draw()
master.save("crates/mosaic-gui/icons/icon-1024.png")
master.resize((256, 256), Image.LANCZOS).save("crates/mosaic-gui/ui/icon.png")
master.save("crates/mosaic-gui/icons/icon.ico",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
