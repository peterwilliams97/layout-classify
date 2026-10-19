"""quality.py applies the corpus's quality gate to each PDF in data/pdf/manifest.jsonl that is not
an alias, on its labelled page, and writes one row per PDF to data/pdf/quality.jsonl. It drops
nothing; `kept` records the verdict for split.py to read.

The gate stands in for "native digital documents or good scans of them". A page passes if
pdfplumber extracts at least MIN_CHARS non-space characters from it (`has_text`), or else if its
largest embedded image, scaled to the page size, has an effective resolution of at least MIN_DPI
(`image_dpi`, the lower of the horizontal and vertical figures; null with no image). A text layer
passes a page whatever its image resolution, so `image_dpi` is recorded for text pages too, where
the largest image is often a logo; `image_cover`, the share of the page that image's box covers,
tells a scan from a logo.

The run ends with kept and dropped counts per label and the image_dpi of the dropped pages."""

import collections
import json
import sys

import pdfplumber

MANIFEST = "data/pdf/manifest.jsonl"
OUT = "data/pdf/quality.jsonl"
MIN_CHARS = 50
MIN_DPI = 150


def check(path, page_index):
    """Page count, extractable non-space characters, image_dpi and image_cover of page
    `page_index`."""
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[page_index]
        chars = sum(1 for c in page.extract_text() or "" if not c.isspace())
        dpi = cover = None
        images = [im for im in page.images if im.get("srcsize")]
        if images:
            im = max(images, key=lambda im: im["srcsize"][0] * im["srcsize"][1])
            w_px, h_px = im["srcsize"]
            dpi = min(w_px / (float(page.width) / 72), h_px / (float(page.height) / 72))
            cover = float(im["width"] * im["height"] / (page.width * page.height))
        return len(pdf.pages), chars, dpi, cover


def main():
    with open(MANIFEST) as f:
        rows = [json.loads(line) for line in f]
    out = []
    for r in rows:
        if r["alias_of"]:
            continue
        q = {"file": r["file"], "set": r["set"], "label": r["label"], "page": r["page"]}
        try:
            q["pages"], q["chars"], dpi, cover = check("data/pdf/" + r["file"], r["page"])
        except Exception as e:  # pdfminer raises many types on damaged files
            q.update(pages=r["pages"], chars=None, has_text=False, image_dpi=None, image_cover=None,
                     kept=False,
                     reason=f"unreadable: {type(e).__name__}: {e}")
            out.append(q)
            continue
        q["has_text"] = q["chars"] >= MIN_CHARS
        q["image_dpi"] = None if dpi is None else round(dpi, 1)
        q["image_cover"] = None if cover is None else round(cover, 3)
        if q["has_text"]:
            q["kept"], q["reason"] = True, f"text layer, {q['chars']} chars"
        elif dpi is None:
            q["kept"], q["reason"] = False, f"no text layer ({q['chars']} chars) and no image"
        elif dpi >= MIN_DPI:
            q["kept"], q["reason"] = True, f"image at {dpi:.0f} dpi"
        else:
            q["kept"], q["reason"] = False, f"no text layer ({q['chars']} chars), image at {dpi:.0f} dpi"
        out.append(q)
    with open(OUT, "w") as f:
        for q in out:
            f.write(json.dumps(q) + "\n")

    by_label = collections.defaultdict(collections.Counter)
    for q in out:
        by_label[q["label"]]["kept" if q["kept"] else "dropped"] += 1
        by_label[q["label"]]["text" if q.get("has_text") else "image"] += q["kept"]
    print(f"{'label':24} {'kept':>5} {'dropped':>7} {'by text':>7} {'by image':>8}")
    for label in sorted(by_label):
        c = by_label[label]
        print(f"{label:24} {c['kept']:5} {c['dropped']:7} {c['text']:7} {c['image']:8}")
    total = sum(by_label.values(), collections.Counter())
    print(f"{'total':24} {total['kept']:5} {total['dropped']:7} {total['text']:7} {total['image']:8}")

    dropped = [q for q in out if not q["kept"]]
    bins = collections.Counter()
    for q in dropped:
        d = q["image_dpi"]
        if d is None:
            bins["no image" if q["chars"] is not None else "unreadable"] += 1
        else:
            lo = int(d // 25) * 25
            bins[f"{lo}-{lo + 25}"] += 1
    print("\ndropped by image_dpi")
    for b in sorted(bins, key=lambda b: (not b[0].isdigit(), int(b.split("-")[0]) if b[0].isdigit() else 0)):
        print(f"  {b:12} {bins[b]}")
    print(f"{len(out)} PDFs checked, {len(rows) - len(out)} aliases skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
