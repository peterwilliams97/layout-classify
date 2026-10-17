"""ocr.py runs Tesseract, through pytesseract, once on every page in data/manifest.jsonl and caches
the result as ocr/<id>.json: {"text": str, "words": [{"text", "x0", "y0", "x1", "y1", "conf"}]}.
Word boxes are in pixels of data/<split>/<id>.png, origin top left; `conf` is Tesseract's word
confidence, 0 to 100. `text` is the same words, one line of the page per line, blocks separated by a
blank line. It is the only OCR in this repo: every classifier that reads text reads this cache.

A page whose ocr/<id>.json exists is skipped, so an interrupted run resumes. The seconds of the
pages run this time, total and per page, go to results/ocr-time.txt."""

import json
import os
import statistics
import sys
import time

import pytesseract

MANIFEST = "data/manifest.jsonl"
OUT_DIR = "ocr"
TIME_FILE = "results/ocr-time.txt"


def ocr_page(path):
    """Tesseract's words and their boxes for the image at `path`, and the page text built from
    them."""
    d = pytesseract.image_to_data(path, output_type=pytesseract.Output.DICT)
    words, blocks = [], {}
    for i, text in enumerate(d["text"]):
        if not text.strip():
            continue
        x0, y0 = d["left"][i], d["top"][i]
        words.append({"text": text, "x0": x0, "y0": y0, "x1": x0 + d["width"][i],
                      "y1": y0 + d["height"][i], "conf": float(d["conf"][i])})
        line = (d["par_num"][i], d["line_num"][i])
        blocks.setdefault(d["block_num"][i], {}).setdefault(line, []).append(text)
    page = "\n\n".join("\n".join(" ".join(ws) for ws in lines.values())
                       for lines in blocks.values())
    return {"text": page, "words": words}


def main():
    with open(MANIFEST) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [(r["id"], r["split"]) for r in rows]
    if not ids:
        sys.exit(f"{MANIFEST}: no pages")
    os.makedirs(OUT_DIR, exist_ok=True)

    seconds, skipped, empty = [], 0, []
    start = time.time()
    for n, (page_id, split) in enumerate(ids, 1):
        out = os.path.join(OUT_DIR, page_id + ".json")
        if os.path.exists(out):
            skipped += 1
            continue
        t = time.perf_counter()
        result = ocr_page(f"data/{split}/{page_id}.png")
        seconds.append(time.perf_counter() - t)
        if not result["words"]:
            empty.append(page_id)
        with open(out, "w") as f:
            json.dump(result, f)
        if n % 100 == 0:
            print(f"{n} of {len(ids)} pages, {time.time() - start:.0f} s", file=sys.stderr)
    wall = time.time() - start

    lines = [
        f"tesseract {pytesseract.get_tesseract_version()}, pytesseract defaults, one page per call",
        f"pages in manifest      {len(ids)}",
        f"pages run              {len(seconds)}",
        f"pages skipped          {skipped} (ocr/<id>.json already present)",
        f"pages with no words    {len(empty)}",
    ]
    if seconds:
        lines += [
            f"total seconds          {sum(seconds):.1f} (wall {wall:.1f})",
            f"seconds per page       mean {statistics.mean(seconds):.3f}"
            f"  median {statistics.median(seconds):.3f}  max {max(seconds):.3f}",
        ]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if seconds:
        os.makedirs(os.path.dirname(TIME_FILE), exist_ok=True)
        with open(TIME_FILE, "w") as f:
            f.write(text)
    else:
        print(f"no pages run; {TIME_FILE} left as it was", file=sys.stderr)


if __name__ == "__main__":
    main()
