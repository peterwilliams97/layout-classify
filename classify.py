"""classify.py is the repo's classifier put to use on a folder of documents: every .pdf, .png, .jpg
and .tif in it gets a type and a decision, `file` to use the type unread or `review` for a person.
The classifier is the row the README's coverage table puts first, d3-flash with the `definitions`
prompt, 150 state words and REJECT_LABEL as an option, and the cut-off is its `tau_cp` from the
pool. It runs in `.venv-d3`, like score.py --d3, and needs MPS.

A PDF gives its page 1, rendered by render.py's `render` at its DPI; an image is OCR'd as it is, a
multi-page TIFF its first frame. The OCR is ocr.py's `ocr_page`, cached in
<folder>/.layout-classify/ocr/<sha256 of the file>.json, so a second run reads it instead. The
question and state are d3.py's, built the way score.py --d3 builds them; d3's responses are not
cached, as d3.py's cache is keyed by manifest page id.

  .venv-d3/bin/python classify.py <folder> [--out results.csv] [--level 0.05|0.02]"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
# d3-flash is read from the Hugging Face cache CLAUDE.md names, and nothing is fetched into it.
os.environ.setdefault("HF_HOME", os.path.join(REPO, "data", "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    from PIL import Image
    import pypdfium2, pytesseract, torch, transformers  # noqa: F401 — imported here to fail early
except ModuleNotFoundError as e:
    sys.exit(f"classify.py: {e}; run it with .venv-d3/bin/python, built from requirements-d3.txt")

from classes import CLASSES, REJECT_LABEL
from d3 import EXAMPLE_WORDS, REVISIONS, d3_question, first_words, load
from ocr import ocr_page, write_page
from render import render

MODEL = "vllm-sr/d3-flash"
PROMPT = "definitions"
STATE_WORDS = 150
# CUTOFFS are the tau_cp lines of results/confidence/d3-flash-definitions-s150-reject-pool.txt at
# 43b454c, by alpha: on test, 0.05 filed 0.7205 of all pages with 0.0380 wrong, 0.02 filed 0.6110
# with none wrong.
CUTOFFS = {"0.05": 0.692011, "0.02": 0.903935}
EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CACHE_DIR = ".layout-classify"


def main():
    ap = argparse.ArgumentParser(description="Classify page 1 of every document in a folder.")
    ap.add_argument("folder")
    ap.add_argument("--out", default="results.csv", help="CSV written; default results.csv")
    ap.add_argument("--level", choices=list(CUTOFFS), default="0.05",
                    help="error level whose cut-off decides file or review; default 0.05")
    args = ap.parse_args()

    files = sorted(
        name for name in os.listdir(args.folder)
        if os.path.splitext(name)[1].lower() in EXTENSIONS
        and os.path.isfile(os.path.join(args.folder, name))
    )
    if not files:
        sys.exit(f"{args.folder}: no .pdf, .png, .jpg or .tif files")
    cutoff = CUTOFFS[args.level]
    ocr_dir = os.path.join(args.folder, CACHE_DIR, "ocr")
    os.makedirs(ocr_dir, exist_ok=True)
    names = list(CLASSES) + [REJECT_LABEL]
    question = d3_question(PROMPT, EXAMPLE_WORDS, True, "all")

    model, rows, failed = None, [], 0
    for i, name in enumerate(files, 1):
        path = os.path.join(args.folder, name)
        start = time.perf_counter()
        try:
            text = page_text(path, ocr_dir)
            if model is None:
                t = time.perf_counter()
                model = load(MODEL, REVISIONS[MODEL])
                start += time.perf_counter() - t
            state = first_words(text, STATE_WORDS)
            answer = model.system_one(state=state, questions={"type": question})["answers"]["type"]
            if "error" in answer:
                raise ValueError(f"{MODEL} answered {answer['error']!r}")
            if answer["choice"] not in names:
                raise ValueError(f"{MODEL} chose {answer['choice']!r}, not an option")
            label, score = answer["choice"], float(answer["probabilities"][answer["choice"]])
        except Exception as e:
            # One unreadable file goes to review rather than end the run.
            print(f"{name}: {e}", file=sys.stderr)
            failed += 1
            label, score = "", None
        seconds = time.perf_counter() - start
        decision = "file" if label not in ("", REJECT_LABEL) and score >= cutoff else "review"
        rows.append({"file": name, "page": 1, "type": label,
                     "score": "" if score is None else f"{score:.6f}",
                     "decision": decision, "seconds": f"{seconds:.3f}"})
        print(f"{i}/{len(files)} {name} {label} {decision} ({seconds:.2f} s)", file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "page", "type", "score", "decision", "seconds"])
        w.writeheader()
        w.writerows(rows)
    filed = sum(r["decision"] == "file" for r in rows)
    per_page = statistics.mean(float(r["seconds"]) for r in rows)
    print(f"files {len(rows)}, filed {filed}, review {len(rows) - filed}, "
          f"seconds per page {per_page:.2f} (level {args.level}, cut-off {cutoff}; "
          f"{failed} failed; {args.out})")


def page_text(path, ocr_dir):
    """page_text returns the OCR text of `path`'s page 1 from <ocr_dir>/<sha256>.json, OCRing it
    and writing that file first if it is missing. The PNG a PDF is rendered to is kept beside it,
    as ocr.py's word boxes are in its pixels."""
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    cached = os.path.join(ocr_dir, digest + ".json")
    if not os.path.exists(cached):
        if path.lower().endswith(".pdf"):
            image = os.path.join(ocr_dir, digest + ".png")
            render(path, 0, image)
            result = ocr_page(image)
        else:
            with Image.open(path) as im:
                result = ocr_page(im.copy() if getattr(im, "n_frames", 1) > 1 else path)
        write_page(cached + ".part", result["text"], result["words"])
        os.replace(cached + ".part", cached)
    with open(cached) as f:
        return json.load(f)["text"]


if __name__ == "__main__":
    main()
