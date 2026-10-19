"""fetch.py downloads the corpus's source PDFs from the RVL-CDIP-N Drive folder and writes
data/pdf/manifest.jsonl, one row per source PDF, as docs/2026-10-19-corpus-plan.md lays out.

The sources are docs/rvl-cdip-ood/n-to-pdf.jsonl (all 1002 RVL-CDIP-N images, each with its PDF)
and the 254 rows of docs/rvl-cdip-ood/o-to-pdf.jsonl that name a PDF. A PDF is saved as
data/pdf/<folder>-<stem>.pdf, e.g. pdfs_4-documentcloud_00044_memo.pdf: six stems occur in two
Drive folders with different images, so the stem alone does not name one file. Downloads run one
at a time with a pause between them; a file already on disk is not fetched again, so an interrupted
run resumes. A download that does not begin `%PDF` is discarded and counted as failed.

For each N row the RVL-CDIP-N image is fetched too, only to find the labelled page: its perceptual
hash (imagehash phash) and pixel size go to data/pdf/n-image-phash.jsonl, a cache so a rerun does
not fetch the image again, and the image itself is not kept. Every page of the PDF is rendered at
the image's pixel size and the page with the smallest phash distance is the labelled page.

Manifest rows carry `file`, `set` ("N" or "O"), `label` (the RVL-CDIP-N type, or "ood"), the Drive
`pdf` path and `pdf_id`, the `image` and `image_id`, `sha256`, `bytes`, `pages`, and for N rows
`page` (0-based) and `page_distance`; O rows have `page` 0 and `page_distance` null. PDFs with the
same sha256 are one document: the first in manifest order (N before O, then by file name) is kept
and lists the others under `aliases`; each of those has `alias_of` set to the kept file."""

import hashlib
import io
import json
import os
import subprocess
import sys
import time

import imagehash
import pypdfium2
from PIL import Image

N_SOURCE = "docs/rvl-cdip-ood/n-to-pdf.jsonl"
O_SOURCE = "docs/rvl-cdip-ood/o-to-pdf.jsonl"
OUT_DIR = "data/pdf"
MANIFEST = "data/pdf/manifest.jsonl"
PHASH_CACHE = "data/pdf/n-image-phash.jsonl"
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
PAUSE = 0.5  # seconds between downloads
RETRIES = 2
MATCH_DISTANCE = 10  # a best page further than this from the image counts as no match
LABELS = ["budget", "email", "form", "handwritten", "invoice", "letter", "memo", "news_article",
          "questionnaire", "resume", "scientific_publication", "specification"]


def sources():
    """The 1256 source rows, N first, each with its local `file`, `set` and `label`."""
    rows = []
    for set_name, path in (("N", N_SOURCE), ("O", O_SOURCE)):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("pdf"):
                    continue
                folder, base = r["pdf"].split("/")[-2:]
                stem = base.removesuffix(".pdf")
                if set_name == "O":
                    label = "ood"
                else:
                    label = next(l for l in LABELS if stem.endswith("_" + l))
                rows.append({"file": f"{folder}-{stem}.pdf", "set": set_name, "label": label,
                             "pdf": r["pdf"], "pdf_id": r["pdf_id"],
                             "image": r["image"], "image_id": r["image_id"]})
    rows.sort(key=lambda r: (r["set"], r["file"]))
    return rows


def download(drive_id):
    """The bytes of the Drive file `drive_id`, fetched by curl, which fails a transfer shorter
    than its Content-Length; urllib through the sandbox proxy returns such short reads."""
    result = subprocess.run(
        ["curl", "-sSfL", "--retry", str(RETRIES), "--max-time", "300", URL.format(drive_id)],
        capture_output=True)
    if result.returncode != 0:
        raise OSError(f"curl exit {result.returncode}: {result.stderr.decode().strip()}")
    return result.stdout


def fetch_pdfs(rows):
    """Download every PDF not yet on disk; return the files that failed."""
    failed = []
    todo = [r for r in rows if not os.path.exists(os.path.join(OUT_DIR, r["file"]))]
    print(f"PDFs: {len(rows) - len(todo)} on disk, {len(todo)} to fetch", file=sys.stderr)
    for i, r in enumerate(todo, 1):
        path = os.path.join(OUT_DIR, r["file"])
        try:
            data = download(r["pdf_id"])
        except OSError as e:
            print(f"  {r['file']}: {e}", file=sys.stderr)
            failed.append(r["file"])
            continue
        if not data.startswith(b"%PDF"):
            print(f"  {r['file']}: begins {data[:16]!r}, not %PDF", file=sys.stderr)
            failed.append(r["file"])
            continue
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", file=sys.stderr)
        time.sleep(PAUSE)
    return failed


def fetch_image_hashes(rows):
    """Phash and pixel size of every N image, from PHASH_CACHE or fetched; return them by image
    name and the images that failed."""
    cache = {}
    if os.path.exists(PHASH_CACHE):
        with open(PHASH_CACHE) as f:
            for line in f:
                c = json.loads(line)
                cache[c["image"]] = c
    todo = [r for r in rows if r["set"] == "N" and r["image"] not in cache]
    print(f"N images: {len(cache)} hashed, {len(todo)} to fetch", file=sys.stderr)
    failed = []
    with open(PHASH_CACHE, "a") as out:
        for i, r in enumerate(todo, 1):
            try:
                img = Image.open(io.BytesIO(download(r["image_id"])))
                img.load()
            except (OSError, Image.UnidentifiedImageError) as e:
                print(f"  {r['image']}: {e}", file=sys.stderr)
                failed.append(r["image"])
                continue
            c = {"image": r["image"], "width": img.width, "height": img.height,
                 "phash": str(imagehash.phash(img.convert("L")))}
            cache[r["image"]] = c
            out.write(json.dumps(c) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", file=sys.stderr)
            time.sleep(PAUSE)
    return cache, failed


def best_page(path, width, height, target):
    """The page count of the PDF at `path`, and the 0-based page whose render at `width` x
    `height` has the smallest phash distance to `target`, with that distance."""
    pdf = pypdfium2.PdfDocument(path)
    try:
        best = (None, None)
        for i in range(len(pdf)):
            page = pdf[i]
            scale = height / page.get_height()
            img = page.render(scale=scale).to_pil().convert("L").resize((width, height))
            d = int(imagehash.phash(img) - target)
            if best[1] is None or d < best[1]:
                best = (i, d)
        return len(pdf), best[0], best[1]
    finally:
        pdf.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = sources()
    failed = fetch_pdfs(rows)
    hashes, failed_images = fetch_image_hashes(rows)
    if failed or failed_images:
        print(f"{len(failed)} PDFs and {len(failed_images)} images failed; rerun to retry",
              file=sys.stderr)
        sys.exit(1)

    kept_by_hash = {}
    for r in rows:
        path = os.path.join(OUT_DIR, r["file"])
        with open(path, "rb") as f:
            data = f.read()
        r["sha256"] = hashlib.sha256(data).hexdigest()
        r["bytes"] = len(data)
        r["alias_of"] = None
        r["aliases"] = []
        if r["sha256"] in kept_by_hash:
            kept = kept_by_hash[r["sha256"]]
            r["alias_of"] = kept["file"]
            kept["aliases"].append(r["file"])
        else:
            kept_by_hash[r["sha256"]] = r
        if r["set"] == "N":
            h = hashes[r["image"]]
            target = imagehash.hex_to_hash(h["phash"])
            r["pages"], r["page"], r["page_distance"] = best_page(path, h["width"], h["height"],
                                                                  target)
        else:
            pdf = pypdfium2.PdfDocument(path)
            r["pages"], r["page"], r["page_distance"] = len(pdf), 0, None
            pdf.close()

    with open(MANIFEST, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = [r for r in rows if r["set"] == "N"]
    not_first = sum(1 for r in n if r["page"] != 0)
    no_match = sum(1 for r in n if r["page_distance"] > MATCH_DISTANCE)
    aliases = sum(1 for r in rows if r["alias_of"])
    print(f"{len(rows)} PDFs ({len(n)} N, {len(rows) - len(n)} O), {aliases} aliases")
    print(f"N labelled page not page 1: {not_first}; best page over distance {MATCH_DISTANCE}: "
          f"{no_match}")


if __name__ == "__main__":
    main()
