"""train.py fits a classifier over CLASSES on the OCR text of train pages, read from the cache
ocr.py writes, and saves it to models/<name>/. `--tfidf` fits TF-IDF + logistic regression; `--embed`
fits logistic regression on frozen `EMBED_BASE` embeddings of the page text with `EMBED_PROMPT`
before it. `--dit` reads the page image instead of the text: logistic regression on frozen
`DIT_BASE` embeddings of data/<split>/<id>.png. `--train real` is every train page; `--train
small` is `SMALL_PER_CLASS` of them per class, drawn with SEED and recorded in data/selection.json
under "small". Neither holds a reject page: train's are left out, as no model predicts
REJECT_LABEL. `--labels llm` fits on the labels
llm.py's `label_split` wrote to data/llm-labels/train.jsonl instead of the hand labels, and the
model name takes the suffix `-llm`; validation keeps its hand labels, so C is still chosen on them.

Adapted from synth-classify's train.py. For `--embed` and `--dit`, the logistic regression's C is
chosen from C_GRID on validation, never test: each C is fitted on train, and the one whose top
probability best ranks validation pages right over wrong, by AUROC, is kept (`choose_c`). Everything
else — TF-IDF's C, n-grams, the token limit — is fixed, not tuned. Every classifier weights classes
inversely to their train counts (`class_weight="balanced"`), so a class with few train pages is not
crowded out by one with many."""

import argparse
import json
import os
import random
import sys
import time

from classes import CLASSES, REJECT_LABEL

MANIFEST = "data/manifest.jsonl"
SELECTION = "data/selection.json"
SEED = 0
SMALL_PER_CLASS = 10
LLM_LABELS = "data/llm-labels/train.jsonl"
# The logistic regression C values `choose_c` tries for the frozen embeddings, smallest first.
C_GRID = [0.01, 0.1, 1, 10]

# The embedding model is pinned: its modeling code is fetched from the Hub and run
# (trust_remote_code), so a revision that moves would change what runs as well as the weights.
EMBED_BASE = "dmis-lab/UltraBERT-Embed-150M"
EMBED_REVISION = "367d5343b5a0164d57b810cedf5de4b8983d5984"
# The model card wants a task instruction on the query side. Like synth-classify's Banking77
# instruction, this one names the domain and the task and no class.
EMBED_PROMPT = "Instruct: Given the OCR text of a scanned document page, find its document type\nQuery: "
# A page runs to hundreds of words where a Banking77 message has a dozen, so the limit is 512
# tokens rather than synth-classify's 128. It is fixed, not chosen on validation.
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
    generator, and records the ids in data/selection.json under "small". A class with fewer train
    pages gives all of them: email has 9. The draw is deterministic, so rewriting the entry
    restores it."""
    ids = []
    for label in CLASSES:
        pool = [r["id"] for r in train_rows if r["label"] == label]
        rng = random.Random(f"{SEED}/small/{label}")
        ids += rng.sample(pool, min(SMALL_PER_CLASS, len(pool)))
    selection = {}
    if os.path.exists(SELECTION):
        with open(SELECTION) as f:
            selection = json.load(f)
    selection["small"] = {"seed": SEED, "per_class": SMALL_PER_CLASS, "ids": ids}
    with open(SELECTION, "w") as f:
        json.dump(selection, f, indent=1)
        f.write("\n")
    return set(ids)


def llm_labels(rows):
    """llm_labels returns gemma4:e4b's label for each of `rows` from LLM_LABELS, in order. Every
    page must have one, and every label must be in CLASSES."""
    with open(LLM_LABELS) as f:
        by_id = {r["id"]: r["label"] for r in map(json.loads, f) if r}
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        sys.exit(f"{LLM_LABELS}: no label for {len(missing)} train pages, first {missing[0]}")
    labels = [by_id[r["id"]] for r in rows]
    unknown = sorted(set(labels) - set(CLASSES))
    if unknown:
        sys.exit(f"{LLM_LABELS}: labels not in CLASSES: {unknown}")
    return labels


def agreement(hand, llm):
    """agreement returns the results/train/ lines comparing the `llm` labels a model was fitted on
    with the `hand` labels of the same pages, both label indices in CLASSES: the share equal and the
    confusion matrix, rows hand and columns llm."""
    from sklearn.metrics import confusion_matrix

    names = list(CLASSES)
    width = max(len(name) for name in names)
    same = sum(h == l for h, l in zip(hand, llm))
    lines = [
        f"agreement {same / len(hand):.4f} ({same} of {len(hand)} llm labels equal the hand label)",
        "confusion (row: hand label, column: llm label, numbered as the rows are)",
        f"  {'':<{width + 3}}" + "".join(f"{i:>4}" for i in range(len(names))),
    ]
    matrix = confusion_matrix(hand, llm, labels=range(len(names)))
    for i, (name, row) in enumerate(zip(names, matrix)):
        lines.append(f"  {i:>2} {name:<{width}}" + "".join(f"{n:>4}" for n in row))
    return lines


def training_pages(which, source):
    """The (page ids, label indices in CLASSES) of the `which` training set, in manifest order, and
    the "agreement" lines for results/train/. Train's REJECT_LABEL pages are left out. The labels
    are the hand labels, or with `source` "llm" those in LLM_LABELS, and the lines are then
    `agreement`'s; with "hand" there are none."""
    rows = [r for r in manifest() if r["split"] == "train" and r["label"] != REJECT_LABEL]
    if not rows:
        sys.exit(f"{MANIFEST}: no train pages")
    names = list(CLASSES)
    unknown = sorted({r["label"] for r in rows} - set(names))
    if unknown:
        sys.exit(f"{MANIFEST}: train labels not in CLASSES: {unknown}")
    if which == "small":
        keep = small_ids(rows)
        rows = [r for r in rows if r["id"] in keep]
    hand = [names.index(r["label"]) for r in rows]
    if source == "hand":
        return [r["id"] for r in rows], hand, []
    llm = [names.index(label) for label in llm_labels(rows)]
    return [r["id"] for r in rows], llm, agreement(hand, llm)


def validation_pages():
    """The (page ids, label indices) of every validation page, reject pages included, in manifest
    order. A reject page's index is -1, which no prediction equals."""
    rows = [r for r in manifest() if r["split"] == "validation"]
    if not rows:
        sys.exit(f"{MANIFEST}: no validation pages")
    names = list(CLASSES)
    unknown = sorted({r["label"] for r in rows} - set(names) - {REJECT_LABEL})
    if unknown:
        sys.exit(f"{MANIFEST}: validation labels not in CLASSES or REJECT_LABEL: {unknown}")
    gold = [names.index(r["label"]) if r["label"] in names else -1 for r in rows]
    return [r["id"] for r in rows], gold


def choose_c(make, x, labels, validation_x, validation_gold):
    """choose_c fits `make(C)` on the train rows `x` for each C in C_GRID and keeps the one with the
    highest validation AUROC of top probability against correctness — the ranking confidence.py's
    cut-off depends on. Every validation page counts, a reject page as wrong. A tie goes to the
    smaller C. Returns the chosen C, its fitted classifier and the "C … validation AUROC …" lines
    for results/train/."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    best, lines = None, []
    for c in C_GRID:
        clf = make(c).fit(x, labels)
        proba = clf.predict_proba(validation_x)
        correct = clf.classes_[proba.argmax(axis=1)] == np.array(validation_gold)
        auroc = roc_auc_score(correct, proba.max(axis=1))
        lines.append(f"C {c:<5} validation AUROC {auroc:.4f}")
        if best is None or auroc > best[2]:
            best = (c, clf, auroc)
    lines.append(f"chosen    C {best[0]} (highest validation AUROC; a tie goes to the smaller C)")
    return best[0], best[1], lines


def write_seconds(name, which, source, pages, seconds, scope, grid):
    """write_seconds records how long a training run took, base model load included, in
    results/train/<name>.txt, followed by `grid`: `choose_c`'s lines and any of `agreement`'s.
    `scope` says what the seconds cover."""
    labels = "" if source == "hand" else f", labels from {LLM_LABELS}"
    text = (
        f"model     models/{name}\n"
        f"train     {which} ({pages} pages{labels})\n"
        f"pages     {pages}\n"
        f"seconds   {seconds:.2f} ({scope})\n"
    ) + "".join(line + "\n" for line in grid)
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


def train_embed(texts, labels, validation_texts, validation_gold, out, batch_size):
    """train_embed fits logistic regression on frozen embeddings; the encoder is not updated. C is
    chosen on validation by `choose_c`. Class weights are balanced, as in `train_tfidf`. Returns
    `choose_c`'s lines."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    tok, model, device = load_embedder()
    rows = lambda ts: np.concatenate(
        [embed(ts[i : i + batch_size], tok, model, device) for i in range(0, len(ts), batch_size)]
    )
    make = lambda c: LogisticRegression(C=c, max_iter=1000, class_weight="balanced")
    _, clf, grid = choose_c(make, rows(texts), labels, rows(validation_texts), validation_gold)
    joblib.dump(clf, os.path.join(out, "embed.joblib"))
    return grid


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


def train_dit(ids, labels, validation_ids, validation_gold, out, batch_size):
    """train_dit fits logistic regression on frozen DiT embeddings, as `train_embed` does on text
    embeddings: encoder not updated, C chosen on validation by `choose_c`, balanced class weights. A
    StandardScaler fitted on the train embeddings comes first. Images are read one batch at a time,
    so the read is inside the training seconds. Returns `choose_c`'s lines."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    processor, model, device = load_dit()

    def rows(page_ids):
        out = []
        for i in range(0, len(page_ids), batch_size):
            images = [page_image(page_id) for page_id in page_ids[i : i + batch_size]]
            out.append(embed_images(images, processor, model, device))
        return np.concatenate(out)

    make = lambda c: make_pipeline(
        StandardScaler(), LogisticRegression(C=c, max_iter=1000, class_weight="balanced")
    )
    _, clf, grid = choose_c(make, rows(ids), labels, rows(validation_ids), validation_gold)
    joblib.dump(clf, os.path.join(out, "dit.joblib"))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["real", "small"], required=True)
    ap.add_argument("--labels", choices=["hand", "llm"], default="hand", help=f"llm: {LLM_LABELS}")
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--embed", action="store_true", help=f"frozen {EMBED_BASE} + logistic regression")
    ap.add_argument("--dit", action="store_true", help=f"frozen {DIT_BASE} + logistic regression")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.tfidf + args.embed + args.dit != 1:
        ap.error("give one of --tfidf, --embed and --dit")

    kind = "tfidf" if args.tfidf else "ultrabert-embed" if args.embed else "dit"
    name = f"{kind}-{args.train}" + ("-llm" if args.labels == "llm" else "")
    out = os.path.join("models", name)
    ids, labels, agree = training_pages(args.train, args.labels)
    os.makedirs(out, exist_ok=True)
    grid = []
    if args.dit:
        validation_ids, validation_gold = validation_pages()
        start = time.perf_counter()
        grid = train_dit(ids, labels, validation_ids, validation_gold, out, args.batch_size)
        scope = ("base model load to save; validation embedding, C grid and page image reads"
                 " included")
    else:
        texts = [page_text(page_id) for page_id in ids]
        if args.tfidf:
            start = time.perf_counter()
            train_tfidf(texts, labels, out)
            scope = "base model load to save; OCR excluded"
        else:
            validation_ids, validation_gold = validation_pages()
            validation_texts = [page_text(page_id) for page_id in validation_ids]
            start = time.perf_counter()
            grid = train_embed(texts, labels, validation_texts, validation_gold, out,
                               args.batch_size)
            scope = ("base model load to save; validation embedding and C grid included; OCR"
                     " excluded")
    seconds = time.perf_counter() - start
    write_seconds(name, args.train, args.labels, len(ids), seconds, scope, grid + agree)


if __name__ == "__main__":
    main()
