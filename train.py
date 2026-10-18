"""train.py fits a classifier over CLASSES on the OCR text of train pages, read from the cache
ocr.py writes, and saves it to models/<name>/. `--tfidf` fits TF-IDF + logistic regression; `--embed`
fits logistic regression on frozen `EMBED_BASE` embeddings of the page text with `EMBED_PROMPT`
before it. `--train real` is every train page; `--train small` is `SMALL_PER_CLASS` of them per
class, drawn with SEED and recorded in data/selection.json under "small". Neither holds a reject
page: train has none.

Adapted from synth-classify's train.py. Regularisation, n-grams and the token limit are fixed, not
tuned, so score.py's numbers are not tuned on what they measure. Both classifiers weight classes
inversely to their train counts (`class_weight="balanced"`): train runs from 16 email pages to 76
letter pages, and without weights `tfidf-real` predicts no test page as memo
(results/unweighted/test/tfidf-real.txt)."""

import argparse
import json
import os
import random
import sys
import time

from classes import CLASSES

MANIFEST = "data/manifest.jsonl"
SELECTION = "data/selection.json"
SEED = 0
SMALL_PER_CLASS = 10

# The embedding model is pinned: its modeling code is fetched from the Hub and run
# (trust_remote_code), so a revision that moves would change what runs as well as the weights.
EMBED_BASE = "dmis-lab/UltraBERT-Embed-150M"
EMBED_REVISION = "367d5343b5a0164d57b810cedf5de4b8983d5984"
# The model card wants a task instruction on the query side. Like synth-classify's Banking77
# instruction, this one names the domain and the task and no class.
EMBED_PROMPT = "Instruct: Given the OCR text of a scanned document page, find its document type\nQuery: "
# A page runs to hundreds of words where a Banking77 message has a dozen, so the limit is 512
# tokens rather than synth-classify's 128. It is fixed, not chosen on dev.
EMBED_MAX_TOKENS = 512


def manifest():
    with open(MANIFEST) as f:
        return [json.loads(line) for line in f if line.strip()]


def page_text(page_id):
    """The OCR text of one page from ocr/<id>.json; "" for a page Tesseract found no words on."""
    with open(os.path.join("ocr", page_id + ".json")) as f:
        return json.load(f)["text"]


def small_ids(train_rows):
    """small_ids draws SMALL_PER_CLASS train pages per class, each class with its own seeded
    generator as in data.py's `main`, and records the ids in data/selection.json under "small".
    The draw is deterministic, so rewriting the entry after a data.py run restores it."""
    ids = []
    for label in CLASSES:
        pool = [r["id"] for r in train_rows if r["label"] == label]
        rng = random.Random(f"{SEED}/small/{label}")
        ids += rng.sample(pool, SMALL_PER_CLASS)
    with open(SELECTION) as f:
        selection = json.load(f)
    selection["small"] = {"seed": SEED, "per_class": SMALL_PER_CLASS, "ids": ids}
    with open(SELECTION, "w") as f:
        json.dump(selection, f, indent=1)
        f.write("\n")
    return set(ids)


def training_pages(which):
    """The (texts, label indices in CLASSES) of the `which` training set, in manifest order."""
    rows = [r for r in manifest() if r["split"] == "train"]
    if not rows:
        sys.exit(f"{MANIFEST}: no train pages")
    names = list(CLASSES)
    unknown = sorted({r["label"] for r in rows} - set(names))
    if unknown:
        sys.exit(f"{MANIFEST}: train labels not in CLASSES: {unknown}")
    if which == "small":
        keep = small_ids(rows)
        rows = [r for r in rows if r["id"] in keep]
    return [page_text(r["id"]) for r in rows], [names.index(r["label"]) for r in rows]


def write_seconds(name, which, pages, seconds):
    """write_seconds records how long a training run took, base model load included, in
    results/train/<name>.txt."""
    text = (
        f"model     models/{name}\n"
        f"train     {which} ({pages} pages)\n"
        f"pages     {pages}\n"
        f"seconds   {seconds:.2f} (base model load to save; OCR excluded)\n"
    )
    sys.stderr.write(text)
    path = os.path.join("results", "train", name + ".txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def train_tfidf(texts, labels, out):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(C=10, max_iter=1000, class_weight="balanced"),
    )
    model.fit(texts, labels)
    joblib.dump(model, os.path.join(out, "tfidf.joblib"))


def load_embedder():
    """load_embedder returns the tokenizer, model and device `embed` takes. SDPA attention in
    float32: FlashAttention does not run on MPS, and MPS has no float64."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(EMBED_BASE, revision=EMBED_REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBED_BASE,
        revision=EMBED_REVISION,
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=torch.float32,
    )
    return tok, model.to(device).eval(), device


def embed(texts, tok, model, device):
    """embed returns one row per text: the final hidden state at position 0, `EMBED_PROMPT`
    prepended, unnormalised — the model card's Sentence Transformers pipeline (CLS pooling, prompt
    included, no Normalize module) without the sentence-transformers dependency. score.py's
    `predict` calls it with one text at a time; train.py with a batch."""
    import torch

    batch = tok(
        [EMBED_PROMPT + t for t in texts],
        padding=True,
        truncation=True,
        max_length=EMBED_MAX_TOKENS,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        return model(**batch).last_hidden_state[:, 0].float().cpu().numpy()


def train_embed(texts, labels, out, batch_size):
    """train_embed fits logistic regression on frozen embeddings; the encoder is not updated. C is
    sklearn's default, as in MTEB's classification evaluator, and is not tuned. Class weights are
    balanced, as in `train_tfidf`."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    tok, model, device = load_embedder()
    starts = range(0, len(texts), batch_size)
    x = np.concatenate([embed(texts[i : i + batch_size], tok, model, device) for i in starts])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x, labels)
    joblib.dump(clf, os.path.join(out, "embed.joblib"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["real", "small"], required=True)
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--embed", action="store_true", help=f"frozen {EMBED_BASE} + logistic regression")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.tfidf == args.embed:
        ap.error("give one of --tfidf and --embed")

    name = f"{'tfidf' if args.tfidf else 'ultrabert-embed'}-{args.train}"
    out = os.path.join("models", name)
    texts, labels = training_pages(args.train)
    os.makedirs(out, exist_ok=True)
    start = time.perf_counter()
    if args.tfidf:
        train_tfidf(texts, labels, out)
    else:
        train_embed(texts, labels, out, args.batch_size)
    write_seconds(name, args.train, len(texts), time.perf_counter() - start)


if __name__ == "__main__":
    main()
