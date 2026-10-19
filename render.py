"""render.py renders the labelled page of every row of data/manifest.jsonl at DPI to
data/<split>/<id>.png, the page image every classifier and ocr.py reads, and records its pixel
`width` and `height` in the manifest row. Rendering is pypdfium2's, the renderer fetch.py used to
find the labelled page, with the page's own rotation applied.

A PNG already on disk is not rendered again, only measured, so an interrupted run resumes. split.py
rewrites the manifest without `width` and `height`; run this after it."""

import json
import os
import sys

import pypdfium2
from PIL import Image

MANIFEST = "data/manifest.jsonl"
PDF_DIR = "data/pdf"
DPI = 200


def render(pdf_path, page_index, out_path):
    """Render page `page_index` of `pdf_path` at DPI to the PNG `out_path`; return its size."""
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        img = pdf[page_index].render(scale=DPI / 72).to_pil()
    finally:
        pdf.close()
    img.save(out_path + ".part", format="PNG")
    os.replace(out_path + ".part", out_path)
    return img.size


def main():
    with open(MANIFEST) as f:
        rows = [json.loads(line) for line in f]
    rendered = 0
    for i, r in enumerate(rows, 1):
        out = os.path.join("data", r["split"], r["id"] + ".png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            with Image.open(out) as img:
                size = img.size
        else:
            size = render(os.path.join(PDF_DIR, r["pdf"]), r["page"], out)
            rendered += 1
        r["width"], r["height"] = size
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}", file=sys.stderr)
    with open(MANIFEST, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    widths = sorted(r["width"] for r in rows)
    heights = sorted(r["height"] for r in rows)
    print(f"{len(rows)} pages, {rendered} rendered this run; width {widths[0]}–{widths[-1]}, "
          f"median {widths[len(widths) // 2]}; height {heights[0]}–{heights[-1]}, "
          f"median {heights[len(heights) // 2]}")


if __name__ == "__main__":
    main()
