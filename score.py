"""score.py scores a model saved by train.py on every page of one split, reject pages included,
and writes the report it prints to results/<split>/<name>.txt. The model predicts one of CLASSES;
a reject page is labelled REJECT_LABEL, which no model predicts, so it is always wrong. `--dump`
also writes each page's prediction and top probability to
results/confidence/dumps/<name>-<split>.jsonl, for confidence.py.

A text model's input is the OCR text in ocr/<id>.json; nothing here runs OCR, so the seconds are
the classifier's alone. ocr.py's time is in results/ocr-time.txt. A `dit-` model's input is the page
image, data/<split>/<id>.png, and its seconds are embedding and logistic regression, the image read
excluded. Adapted from synth-classify's score.py."""

import argparse
import json
import os
import statistics
import sys
import time

from sklearn.metrics import confusion_matrix

from classes import CLASSES, REJECT_LABEL
from train import manifest, page_image, page_text

SPLITS = ["dev", "test"]


def predict(model_dir, ids):
    """predict returns, for each page of `ids`, a label index in CLASSES, its probability, and the
    seconds taken. Each page is classified in a call of its own, so the seconds are what one page
    costs. The page's text or image is read before the clock starts."""
    import joblib

    preds, scores, seconds = [], [], []
    tfidf = os.path.join(model_dir, "tfidf.joblib")
    frozen = os.path.join(model_dir, "embed.joblib")
    dit = os.path.join(model_dir, "dit.joblib")
    read = page_text
    if os.path.exists(tfidf):
        clf = joblib.load(tfidf)
        features = lambda text: [text]
    elif os.path.exists(frozen):
        from train import embed, load_embedder

        clf = joblib.load(frozen)
        tok, model, device = load_embedder()
        features = lambda text: embed([text], tok, model, device)
    elif os.path.exists(dit):
        from train import embed_images, load_dit

        clf = joblib.load(dit)
        processor, model, device = load_dit()
        read = page_image
        features = lambda image: embed_images([image], processor, model, device)
    else:
        sys.exit(f"{model_dir}: no tfidf.joblib, embed.joblib or dit.joblib")

    for page_id in ids:
        page = read(page_id)
        start = time.perf_counter()
        row = clf.predict_proba(features(page))[0]
        seconds.append(time.perf_counter() - start)
        # `classes_` holds the training label indices, so a column is mapped through it rather
        # than taken as a position in CLASSES.
        best = row.argmax()
        preds.append(int(clf.classes_[best]))
        scores.append(float(row[best]))
    return preds, scores, seconds


def report(model, split, gold, preds, seconds, names):
    """report formats the scores. `gold` and `preds` index `names`, which is CLASSES then
    REJECT_LABEL."""
    ids = range(len(names))
    width = max(len(name) for name in names)
    reject = names.index(REJECT_LABEL)
    excluded = "image read" if os.path.basename(model).startswith("dit-") else "OCR"
    wrong = sum(g != p for g, p in zip(gold, preds))
    kept = [(g, p) for g, p in zip(gold, preds) if g != reject]
    lines = [
        f"model     {model}",
        f"{split + ' pages':<9} {len(gold)} ({len(gold) - len(kept)} reject pages, always wrong)",
        f"accuracy  {1 - wrong / len(gold):.4f}",
        f"wrong     {wrong}",
        f"accuracy  {sum(g == p for g, p in kept) / len(kept):.4f} without reject pages",
        f"seconds   {statistics.median(seconds):.3g} median, {statistics.mean(seconds):.3g} mean"
        f" per page, one page per call, {excluded} excluded",
        "confusion (row: true class, column: predicted class, numbered as the rows are)",
        f"  {'':<{width + 3}}" + "".join(f"{i:>4}" for i in ids),
    ]
    for i, (name, row) in enumerate(zip(names, confusion_matrix(gold, preds, labels=ids))):
        lines.append(f"  {i:>2} {name:<{width}}" + "".join(f"{n:>4}" for n in row))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="name of a directory under models/")
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--dump", action="store_true", help="write results/confidence/dumps/")
    args = ap.parse_args()

    names = list(CLASSES) + [REJECT_LABEL]
    rows = [r for r in manifest() if r["split"] == args.split]
    if not rows:
        sys.exit(f"no {args.split} pages in the manifest")
    unknown = sorted({r["label"] for r in rows} - set(names))
    if unknown:
        sys.exit(f"{args.split}: labels not in CLASSES or REJECT_LABEL: {unknown}")
    gold = [names.index(r["label"]) for r in rows]

    model_dir = os.path.join("models", args.model)
    preds, scores, seconds = predict(model_dir, [r["id"] for r in rows])

    text = report(model_dir, args.split, gold, preds, seconds, names)
    sys.stdout.write(text)
    out = os.path.join("results", args.split, args.model + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}", file=sys.stderr)
    if args.dump:
        dump = os.path.join("results", "confidence", "dumps", f"{args.model}-{args.split}.jsonl")
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        with open(dump, "w") as f:
            for r, pred, score in zip(rows, preds, scores):
                row = {"id": r["id"], "label": r["label"], "predicted": names[pred], "score": score}
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows)} rows to {dump}", file=sys.stderr)


if __name__ == "__main__":
    main()
