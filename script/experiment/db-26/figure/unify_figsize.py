#!/usr/bin/env python3
"""Pad the paper-row figures to one common canvas size.

The plot scripts save with bbox_inches="tight", which crops each figure
differently, so the five PDFs end up with slightly different aspect ratios
and render at different heights when LaTeX scales them to equal width.
This script centers every page on the max-width x max-height canvas of the
set (pure whitespace padding, content untouched), making all aspect ratios
identical. The matching PNG previews are padded the same way.

Run after regenerating any of these figures:
    python unify_figsize.py
"""

import os

from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = [
    "scaling_e2e",
    "scaling_memory",
    "ablation_e2e",
    "throughput_combined",
    "predict_per_horizon",
]

# ---- PDFs -------------------------------------------------------------------
sizes = {}
for n in NAMES:
    box = PdfReader(os.path.join(HERE, f"{n}.pdf")).pages[0].mediabox
    sizes[n] = (float(box.width), float(box.height))

W = max(w for w, _ in sizes.values())
H = max(h for _, h in sizes.values())
print(f"target canvas: {W:.1f} x {H:.1f} pt (ratio {W / H:.3f})")

for n in NAMES:
    w, h = sizes[n]
    path = os.path.join(HERE, f"{n}.pdf")
    reader = PdfReader(path)
    page = reader.pages[0]
    dx, dy = (W - w) / 2, (H - h) / 2
    page.add_transformation(Transformation().translate(dx, dy))
    for boxname in ("mediabox", "cropbox", "trimbox", "artbox", "bleedbox"):
        setattr(page, boxname, RectangleObject([0, 0, W, H]))
    writer = PdfWriter()
    writer.add_page(page)
    with open(path, "wb") as fh:
        writer.write(fh)
    print(f"  {n}.pdf: {w:.1f}x{h:.1f} -> {W:.1f}x{H:.1f}")

# ---- PNG previews: pad to the same aspect ratio -----------------------------
for n in NAMES:
    path = os.path.join(HERE, f"{n}.png")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    tw, th = w, h
    if w / h < W / H:
        tw = int(round(h * W / H))
    else:
        th = int(round(w * H / W))
    canvas = Image.new("RGB", (tw, th), "white")
    canvas.paste(img, ((tw - w) // 2, (th - h) // 2))
    canvas.save(path)
    print(f"  {n}.png: {w}x{h} -> {tw}x{th}")
