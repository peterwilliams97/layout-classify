"""quality.py applies the corpus's quality gate to each PDF in data/pdf/manifest.jsonl that is not
an alias, on its labelled page, and writes one row per PDF to data/pdf/quality.jsonl. It drops
nothing; `kept` records the verdict for split.py to read.

The gate stands in for "native digital documents or good scans of them". `image_dpi` is the
effective resolution of the page's largest embedded image, scaled to its box on the page (the lower
of the horizontal and vertical figures; null with no image), and `image_cover` the share of the
page that box covers. A page whose largest image covers at least SCAN_COVER of it is a scan
(`scan`), and passes only with `image_dpi` at least MIN_DPI, whatever its text layer: an OCR layer
on a low-resolution scan is still a low-resolution scan. Any other page passes if pdfplumber
extracts at least MIN_CHARS non-space characters from it (`has_text`), or else with `image_dpi` at
least MIN_DPI; there the largest image is often a logo, and its resolution does not matter.
`image_dpi` is compared after rounding to 0.1, so a 150-dpi scan whose page size in points is not
exact still passes.

The run ends with kept and dropped counts per label and the image_dpi of the dropped pages."""

import collections
import json
import sys

import pdfplumber

MANIFEST = "data/pdf/manifest.jsonl"
OUT = "data/pdf/quality.jsonl"
MIN_CHARS = 50
MIN_DPI = 150
SCAN_COVER = 0.8


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
                     scan=False, kept=False,
                     reason=f"unreadable: {type(e).__name__}: {e}")
            out.append(q)
            continue
        q["has_text"] = q["chars"] >= MIN_CHARS
        q["image_dpi"] = None if dpi is None else round(dpi, 1)
        q["image_cover"] = None if cover is None else round(cover, 3)
        q["scan"] = cover is not None and cover >= SCAN_COVER
        dpi = q["image_dpi"]
        if q["scan"]:
            q["kept"] = dpi >= MIN_DPI
            q["reason"] = f"scan, image covers {cover:.0%}, at {dpi:.1f} dpi"
        elif q["has_text"]:
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
        by_label[q["label"]]["text" if q.get("has_text") and not q["scan"] else "image"] += q["kept"]
        by_label[q["label"]]["scan dropped"] += q["scan"] and not q["kept"]
    print(f"{'label':24} {'kept':>5} {'dropped':>7} {'by text':>7} {'by image':>8} {'scan dropped':>12}")
    for label in sorted(by_label):
        c = by_label[label]
        print(f"{label:24} {c['kept']:5} {c['dropped']:7} {c['text']:7} {c['image']:8} "
              f"{c['scan dropped']:12}")
    total = sum(by_label.values(), collections.Counter())
    print(f"{'total':24} {total['kept']:5} {total['dropped']:7} {total['text']:7} {total['image']:8} "
          f"{total['scan dropped']:12}")

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
