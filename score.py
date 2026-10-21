"""score.py scores a model saved by train.py on every page of one split, reject pages included,
and writes the report it prints to results/<split>/<name>.txt. The model predicts one of CLASSES;
a reject page is labelled REJECT_LABEL, which no model predicts, so it is always wrong. `--dump`
also writes each page's prediction and top probability to
results/confidence/dumps/<name>-<split>.jsonl, for confidence.py.

A text model's input is the OCR text in ocr/<id>.json; nothing here runs OCR, so the seconds are
the classifier's alone. ocr.py's time is in results/ocr-time.txt. A `dit-` model's input is the page
image, data/<split>/<id>.png, and its seconds are embedding and logistic regression, the image read
excluded. A `layoutlmv3-` model, from finetune.py, reads both the OCR words and boxes and the image;
its seconds are the forward pass alone, the page read and the processor excluded, and its score is
the softmax's top probability. Adapted from synth-classify's score.py.

`--llm` scores one of llm.py's `NAMES` instead of a saved model: gemma4:e4b on Ollama, given the
page image, with no hand labels. It gives no probability, so `--dump` is refused. The
`gemma4-image-reject` variant may answer REJECT_LABEL, which is right on a reject page, and its
report adds the share of reject pages it gave REJECT_LABEL and the share of in-class pages it
did.

`--d3 MODEL` scores a Decision 3.0 model run locally by d3.py, `vllm-sr/d3-flash`, on the OCR text,
with `--prompt definitions` (the default) or `fewshot`; the report name is <model basename>-<prompt>.
`--state-words N` and, with `fewshot`, `--example-words N` cut the page and each example to their
first N words; a value other than d3.py's STATE_WORDS or EXAMPLE_WORDS adds -s<N> or -e<N> to the
name. Its score is the chosen option's probability. It runs in `.venv-d3`. With `fewshot`, any
data/fewshot.txt page in the split is left out, so no page is scored with itself as an example.
`--reject-option` makes REJECT_LABEL a 13th option and adds -reject to the name; a page the model
labels REJECT_LABEL is right on a reject page and wrong on an in-class page, as for
`gemma4-image-reject`, and its report adds the same two reject shares. `--classes six` asks with
d3.py's CLASS_SETS["six"] as the options instead of all 12, scores only the pages labelled one of
them or REJECT_LABEL, and adds -c6 to the name; `--pages six` scores the same pages with all 12
options and adds -on6, so the two reports are on identical pages."""

import argparse
import json
import os
import statistics
import sys
import time

from sklearn.metrics import confusion_matrix

from classes import CLASSES, REJECT_LABEL
from d3 import CLASS_SETS, D3_PROMPTS, EXAMPLE_WORDS, STATE_WORDS, classify_d3, fewshot_ids
from llm import MODEL, NAMES, classify
from train import manifest, page_image, page_text

SPLITS = ["validation", "test"]


def predict(model_dir, ids):
    """predict returns, for each page of `ids`, a label index in CLASSES, its probability, and the
    seconds taken. Each page is classified in a call of its own, so the seconds are what one page
    costs. The page's text or image is read, and for a `layoutlmv3-` model encoded, before the clock
    starts."""
    import joblib

    preds, scores, seconds = [], [], []
    tfidf = os.path.join(model_dir, "tfidf.joblib")
    frozen = os.path.join(model_dir, "embed.joblib")
    dit = os.path.join(model_dir, "dit.joblib")
    read = page_text
    proba = lambda page: clf.predict_proba(features(page))[0]
    if os.path.exists(os.path.join(model_dir, "config.json")):
        import torch

        from finetune import encode, load_model, load_processor

        model, device = load_model(model_dir)
        model.eval()
        processor = load_processor(model_dir, None)
        read = lambda page_id: encode([page_id], processor, False).to(device)

        def proba(batch):
            with torch.no_grad():
                return model(**batch).logits.softmax(dim=1)[0].cpu().numpy()

        clf = None
    elif os.path.exists(tfidf):
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
        sys.exit(f"{model_dir}: no config.json, tfidf.joblib, embed.joblib or dit.joblib")
    # A joblib model's columns are its `classes_`, the training label indices; a `layoutlmv3-`
    # model's are CLASSES in order.
    classes = range(len(CLASSES)) if clf is None else clf.classes_

    for page_id in ids:
        page = read(page_id)
        start = time.perf_counter()
        row = proba(page)
        seconds.append(time.perf_counter() - start)
        best = row.argmax()
        preds.append(int(classes[best]))
        scores.append(float(row[best]))
    return preds, scores, seconds


def report(model, split, gold, preds, seconds, names, can_reject=False):
    """report formats the scores. `gold` and `preds` index `names`, which is CLASSES then
    REJECT_LABEL. `can_reject` is for a model that may predict REJECT_LABEL, and adds its two
    reject shares."""
    ids = range(len(names))
    width = max(len(name) for name in names)
    reject = names.index(REJECT_LABEL)
    excluded = {
        "dit": "image read",
        "gemma4": "image read",
        "layoutlmv3": "page read and processor",
    }.get(os.path.basename(model).split("-")[0], "OCR")
    wrong = sum(g != p for g, p in zip(gold, preds))
    kept = [(g, p) for g, p in zip(gold, preds) if g != reject]
    lines = [
        f"model     {model}",
        f"{split + ' pages':<9} {len(gold)} ({len(gold) - len(kept)} reject pages"
        + (")" if can_reject else ", always wrong)"),
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
    if can_reject:
        rejects = [p for g, p in zip(gold, preds) if g == reject]
        lines += [
            f"reject    {sum(p == reject for p in rejects) / len(rejects):.4f} of reject pages"
            f" labelled {REJECT_LABEL}",
            f"reject    {sum(p == reject for _, p in kept) / len(kept):.4f} of in-class pages"
            f" labelled {REJECT_LABEL}",
        ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", help="name of a directory under models/")
    which.add_argument("--llm", choices=list(NAMES), help="gemma4:e4b on the page image")
    which.add_argument("--d3", metavar="MODEL", help="Decision 3.0 model on the OCR text, d3.py")
    ap.add_argument("--prompt", choices=D3_PROMPTS, help="with --d3; default definitions")
    ap.add_argument("--state-words", type=int, help=f"with --d3; default {STATE_WORDS}")
    ap.add_argument("--example-words", type=int, help=f"with --prompt fewshot; default {EXAMPLE_WORDS}")
    ap.add_argument("--reject-option", action="store_true", help=f"with --d3; {REJECT_LABEL} is an option")
    ap.add_argument("--classes", choices=list(CLASS_SETS), help="with --d3; options and pages; default all")
    ap.add_argument("--pages", choices=["six"], help="with --d3; only pages labelled a CLASS_SETS['six'] type or none")
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--dump", action="store_true", help="write results/confidence/dumps/")
    args = ap.parse_args()
    if args.llm and args.dump:
        sys.exit("--llm gives no score, so there is nothing to --dump")
    if args.prompt and not args.d3:
        ap.error("--prompt goes with --d3")
    prompt = args.prompt or "definitions"
    if args.state_words is not None and not args.d3:
        ap.error("--state-words goes with --d3")
    if args.example_words is not None and prompt != "fewshot":
        ap.error("--example-words goes with --prompt fewshot")
    if args.reject_option and not args.d3:
        ap.error("--reject-option goes with --d3")
    if (args.classes or args.pages) and not args.d3:
        ap.error("--classes and --pages go with --d3")
    if args.classes and args.pages:
        ap.error("--classes already sets the pages")
    classes = args.classes or "all"
    state_words = args.state_words or STATE_WORDS
    example_words = args.example_words or EXAMPLE_WORDS

    names = list(CLASSES) + [REJECT_LABEL]
    rows = [r for r in manifest() if r["split"] == args.split]
    if not rows:
        sys.exit(f"no {args.split} pages in the manifest")
    unknown = sorted({r["label"] for r in rows} - set(names))
    if unknown:
        sys.exit(f"{args.split}: labels not in CLASSES or REJECT_LABEL: {unknown}")
    if classes != "all" or args.pages:
        kept = set(CLASS_SETS[classes if classes != "all" else args.pages]) | {REJECT_LABEL}
        rows = [r for r in rows if r["label"] in kept]
    left_out = 0
    if args.d3 and prompt == "fewshot":
        shots = {page_id for ids in fewshot_ids().values() for page_id in ids}
        left_out = sum(r["id"] in shots for r in rows)
        rows = [r for r in rows if r["id"] not in shots]
        print(f"{left_out} data/fewshot.txt pages left out of {args.split}", file=sys.stderr)
    gold = [names.index(r["label"]) for r in rows]

    if args.llm:
        preds, seconds = [], []
        for i, r in enumerate(rows):
            path = os.path.join("data", args.split, r["id"] + ".png")
            label, s = classify(path, NAMES[args.llm])
            preds.append(names.index(label))
            seconds.append(s)
            print(f"{i + 1}/{len(rows)} {r['id']}: {label} ({s:.2f} s)", file=sys.stderr)
        name, model = args.llm, f"{args.llm} ({MODEL} on Ollama)"
    elif args.d3:
        name = f"{os.path.basename(os.path.normpath(args.d3))}-{prompt}"
        model = f"{args.d3} prompt {prompt}, state {state_words} words"
        if state_words != STATE_WORDS:
            name += f"-s{state_words}"
        if prompt == "fewshot":
            model += f", examples {example_words} words"
        if example_words != EXAMPLE_WORDS:
            name += f"-e{example_words}"
        if prompt == "fewshot":
            model += f", {left_out} data/fewshot.txt pages left out"
        if args.reject_option:
            name += "-reject"
            model += f", {REJECT_LABEL} an option"
        if classes == "six":
            name += "-c6"
            model += f", options {len(CLASS_SETS['six'])} types, pages of those and {REJECT_LABEL}"
        if args.pages:
            name += "-on6"
            model += f", pages of the {len(CLASS_SETS['six'])} --classes six types and {REJECT_LABEL}"
        answers = classify_d3(
            [r["id"] for r in rows],
            prompt,
            args.d3,
            state_words,
            example_words,
            args.reject_option,
            classes,
        )
        options = CLASS_SETS[classes] + ([REJECT_LABEL] if args.reject_option else [])
        preds = [names.index(label) for label, _, _ in answers]
        scores = [v[options.index(label)] for label, v, _ in answers]
        seconds = [s for _, _, s in answers]
    else:
        name = args.model
        model = os.path.join("models", args.model)
        preds, scores, seconds = predict(model, [r["id"] for r in rows])

    can_reject = bool(args.llm and NAMES[args.llm]) or args.reject_option
    text = report(model, args.split, gold, preds, seconds, names, can_reject)
    sys.stdout.write(text)
    out = os.path.join("results", args.split, name + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}", file=sys.stderr)
    if args.dump:
        dump = os.path.join("results", "confidence", "dumps", f"{name}-{args.split}.jsonl")
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        with open(dump, "w") as f:
            for r, pred, score in zip(rows, preds, scores):
                row = {"id": r["id"], "label": r["label"], "predicted": names[pred], "score": score}
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows)} rows to {dump}", file=sys.stderr)


if __name__ == "__main__":
    main()
