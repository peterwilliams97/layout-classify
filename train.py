"""train.py fits a classifier over CLASSES on the OCR text of train pages, read from the cache
ocr.py writes, and saves it to models/<name>/. `--tfidf` fits TF-IDF + logistic regression; `--embed`
fits logistic regression on frozen `EMBED_BASE` embeddings of the page text with `EMBED_PROMPT`
before it. `--dit` reads the page image instead of the text: logistic regression on frozen
`DIT_BASE` embeddings of data/<split>/<id>.png. `--train real` is every train page; `--train small` is `SMALL_PER_CLASS` of them per
class, drawn with SEED and recorded in data/selection.json under "small". Neither holds a reject
page: train has none.

Adapted from synth-classify's train.py. Regularisation, n-grams and the token limit are fixed, not
tuned, so score.py's numbers are not tuned on what they measure. Every classifier weights classes
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

# The image model: DiT's self-supervised checkpoint, not microsoft/dit-base-finetuned-rvlcdip,
# which was fine-tuned on RVL-CDIP test pages. Pinned so the weights cannot move under the results.
DIT_BASE = "microsoft/dit-base"
DIT_REVISION = "9bc15e2112d775c8e8f5a88f4bba876f50b77892"


def manifest():
    with open(MANIFEST) as f:
        return [json.loads(line) for line in f if line.strip()]


def page_text(page_id):
    """The OCR text of one page from ocr/<id>.json; "" for a page Tesseract found no words on."""
    with open(os.path.join("ocr", page_id + ".json")) as f:
        return json.load(f)["text"]


def page_image(page_id):
    """The page image data/<split>/<id>.png as RGB; an id starts with its split."""
    from PIL import Image

    split = page_id.split("-", 1)[0]
    with Image.open(os.path.join("data", split, page_id + ".png")) as im:
        return im.convert("RGB")


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
    """The (page ids, label indices in CLASSES) of the `which` training set, in manifest order."""
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
    return [r["id"] for r in rows], [names.index(r["label"]) for r in rows]


def write_seconds(name, which, pages, seconds, scope):
    """write_seconds records how long a training run took, base model load included, in
    results/train/<name>.txt. `scope` says what the seconds cover."""
    text = (
        f"model     models/{name}\n"
        f"train     {which} ({pages} pages)\n"
        f"pages     {pages}\n"
        f"seconds   {seconds:.2f} ({scope})\n"
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


def load_dit():
    """load_dit returns the image processor, model and device `embed_images` takes. The checkpoint
    is `BeitForMaskedImageModeling`; loading it as `BeitModel` keeps the encoder and drops the
    masked-image head. No pooling layer: `embed_images` pools itself."""
    import torch
    from transformers import BeitImageProcessor, BeitModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)
    processor = BeitImageProcessor.from_pretrained(DIT_BASE, revision=DIT_REVISION)
    model = BeitModel.from_pretrained(
        DIT_BASE, revision=DIT_REVISION, add_pooling_layer=False, dtype=torch.float32
    )
    return processor, model.to(device).eval(), device


def embed_images(images, processor, model, device):
    """embed_images returns one row of 768 per image: the final hidden state averaged over the
    patch positions, position 0 (CLS) left out. The processor resizes each page to 224 x 224
    without keeping its aspect ratio, as DiT was pretrained. score.py's `predict` calls it with one
    image at a time; train.py with a batch."""
    import torch

    batch = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**batch).last_hidden_state[:, 1:].mean(dim=1).float().cpu().numpy()


def train_dit(ids, labels, out, batch_size):
    """train_dit fits logistic regression on frozen DiT embeddings, as `train_embed` does on text
    embeddings: encoder not updated, sklearn's default C, balanced class weights. A StandardScaler
    fitted on the train embeddings comes first; results/unscaled/ holds the fit without it. Images are read one batch at a time, so the read is inside
    the training seconds."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    processor, model, device = load_dit()
    rows = []
    for i in range(0, len(ids), batch_size):
        images = [page_image(page_id) for page_id in ids[i : i + batch_size]]
        rows.append(embed_images(images, processor, model, device))
    x = np.concatenate(rows)
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    ).fit(x, labels)
    joblib.dump(clf, os.path.join(out, "dit.joblib"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["real", "small"], required=True)
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--embed", action="store_true", help=f"frozen {EMBED_BASE} + logistic regression")
    ap.add_argument("--dit", action="store_true", help=f"frozen {DIT_BASE} + logistic regression")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.tfidf + args.embed + args.dit != 1:
        ap.error("give one of --tfidf, --embed and --dit")

    kind = "tfidf" if args.tfidf else "ultrabert-embed" if args.embed else "dit"
    name = f"{kind}-{args.train}"
    out = os.path.join("models", name)
    ids, labels = training_pages(args.train)
    os.makedirs(out, exist_ok=True)
    if args.dit:
        start = time.perf_counter()
        train_dit(ids, labels, out, args.batch_size)
        scope = "base model load to save; page image reads included"
    else:
        texts = [page_text(page_id) for page_id in ids]
        start = time.perf_counter()
        if args.tfidf:
            train_tfidf(texts, labels, out)
        else:
            train_embed(texts, labels, out, args.batch_size)
        scope = "base model load to save; OCR excluded"
    write_seconds(name, args.train, len(ids), time.perf_counter() - start, scope)


if __name__ == "__main__":
    main()
