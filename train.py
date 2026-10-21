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
crowded out by one with many.

`--folds K --fold-seed S` trains on the pool instead: every train and validation page, the pages
confidence.py's `--calibration pool` sets its cut-offs on. `pool_folds` splits the pool's in-class
pages K ways within each class and its reject pages K ways apart from them. For each fold the
classifier is fitted on the other folds' in-class pages and scores the fold's in-class and reject
pages, and the pooled out-of-fold scores go to results/confidence/dumps/<name>-pool.jsonl. Each
fold's model is saved to models/<name>/fold<f>/, and those K models are the ones `score.py --folds`
scores on test, so the cut-off is set on the scores of the models it is applied to. No model is
fitted on the whole pool, and no reject page is trained on. With `--train small`, the
SMALL_PER_CLASS draw is made from each fold's training pages, recorded in data/selection.json under
"small-pool". For `--embed` and `--dit`, C is the C_GRID value whose pooled out-of-fold scores rank
right over wrong best, by AUROC, every pool page counted and a reject page wrong.
results/train/<name>-pool.txt records the folds, the C grid and the seconds."""

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


def draw_small(rows):
    """draw_small returns the ids of SMALL_PER_CLASS of `rows` per class, each class with its own
    generator seeded from SEED. A class with fewer rows gives all of them."""
    ids = []
    for label in CLASSES:
        pool = [r["id"] for r in rows if r["label"] == label]
        rng = random.Random(f"{SEED}/small/{label}")
        ids += rng.sample(pool, min(SMALL_PER_CLASS, len(pool)))
    return ids


def record_selection(key, entry):
    """record_selection writes `entry` to data/selection.json under `key`, keeping the other keys."""
    selection = {}
    if os.path.exists(SELECTION):
        with open(SELECTION) as f:
            selection = json.load(f)
    selection[key] = entry
    with open(SELECTION, "w") as f:
        json.dump(selection, f, indent=1)
        f.write("\n")


def small_ids(train_rows):
    """small_ids draws the small training set from `train_rows` with `draw_small` and records the
    ids in data/selection.json under "small". Email has 9 train pages and gives all 9. The draw is
    deterministic, so rewriting the entry restores it."""
    ids = draw_small(train_rows)
    record_selection("small", {"seed": SEED, "per_class": SMALL_PER_CLASS, "ids": ids})
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


def pool_rows():
    """pool_rows returns the manifest rows of every train and validation page, reject pages
    included, in manifest order. Exits if a label is not in CLASSES or REJECT_LABEL."""
    rows = [r for r in manifest() if r["split"] in ("train", "validation")]
    if not rows:
        sys.exit(f"{MANIFEST}: no train or validation pages")
    unknown = sorted({r["label"] for r in rows} - set(CLASSES) - {REJECT_LABEL})
    if unknown:
        sys.exit(f"{MANIFEST}: pool labels not in CLASSES or REJECT_LABEL: {unknown}")
    return rows


def pool_folds(rows, k, seed):
    """pool_folds returns each row's fold, 0 to `k` - 1. The rows of each class of CLASSES, and the
    REJECT_LABEL rows apart from them, are shuffled by a generator seeded from `seed` and the label,
    and dealt to the folds in turn, each label's deal starting at the fold after the last one
    dealt, so each fold holds a `k`th of each class, give or take one, and of the whole pool."""
    fold, dealt = [None] * len(rows), 0
    for label in list(CLASSES) + [REJECT_LABEL]:
        mine = [i for i, r in enumerate(rows) if r["label"] == label]
        random.Random(f"{seed}/folds/{label}").shuffle(mine)
        for i in mine:
            fold[i] = dealt % k
            dealt += 1
    return fold


def fold_training_sets(rows, fold, k, which):
    """fold_training_sets returns, as lists of row indices, each fold's training pages: the other
    folds' in-class rows. With `which` "small" each is `draw_small`'s draw from those rows, and the
    draws are recorded in data/selection.json under "small-pool"."""
    inclass = [i for i, r in enumerate(rows) if r["label"] != REJECT_LABEL]
    sets = [[i for i in inclass if fold[i] != f] for f in range(k)]
    if which == "small":
        drawn = [draw_small([rows[i] for i in s]) for s in sets]
        record_selection("small-pool", {"seed": SEED, "per_class": SMALL_PER_CLASS, "folds": drawn})
        index = {r["id"]: i for i, r in enumerate(rows)}
        sets = [sorted(index[page_id] for page_id in ids) for ids in drawn]
    return sets


def fold_lines(rows, fold, k, sets):
    """fold_lines returns the results/train/<name>-pool.txt lines describing the folds: per fold,
    its in-class and reject pages and its training pages."""
    lines = []
    for f in range(k):
        held = [r for i, r in enumerate(rows) if fold[i] == f]
        reject = sum(r["label"] == REJECT_LABEL for r in held)
        lines.append(
            f"fold {f}    held out {len(held) - reject} in-class and {reject} reject pages;"
            f" trained on {len(sets[f])}"
        )
    return lines


def write_pool_dump(name, rows, pred, score):
    """write_pool_dump writes the out-of-fold `pred` (indices in CLASSES) and `score` of every pool
    row to results/confidence/dumps/<name>-pool.jsonl, score.py's --dump format."""
    names = list(CLASSES)
    path = os.path.join("results", "confidence", "dumps", f"{name}-pool.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r, p, s in zip(rows, pred, score):
            row = {"id": r["id"], "label": r["label"], "predicted": names[p], "score": float(s)}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {path}", file=sys.stderr)


def out_of_fold(make, x, gold, fold, sets):
    """out_of_fold returns the prediction, an index in CLASSES, and top probability of every row of
    `x`, each from `make()` fitted on `sets[f]`, the training rows of the row's fold f, and the
    fitted classifiers in fold order. `x` and `gold` are numpy arrays over the pool rows, a reject
    row's gold -1."""
    import numpy as np

    pred, score, clfs = np.zeros(len(gold), dtype=int), np.zeros(len(gold)), []
    for f in range(len(sets)):
        clf = make().fit(x[sets[f]], gold[sets[f]])
        held = np.flatnonzero(np.array(fold) == f)
        proba = clf.predict_proba(x[held])
        pred[held], score[held] = clf.classes_[proba.argmax(axis=1)], proba.max(axis=1)
        clfs.append(clf)
    return pred, score, clfs


def train_folds(kind, name, which, k, seed, batch_size):
    """train_folds is `--folds`: the out-of-fold pool dump, and each fold's model at the chosen C
    saved to models/<name>/fold<f>/. A frozen encoder embeds every pool page once, before the folds.
    Writes results/train/<name>-pool.txt."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = pool_rows()
    fold = pool_folds(rows, k, seed)
    sets = fold_training_sets(rows, fold, k, which)
    names = list(CLASSES)
    gold = np.array([names.index(r["label"]) if r["label"] in names else -1 for r in rows])
    ids = [r["id"] for r in rows]
    texts = [page_text(page_id) for page_id in ids] if kind != "dit" else None

    start = time.perf_counter()
    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        x = np.array(texts, dtype=object)
        grid, file = [10], "tfidf.joblib"
        make = lambda c: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(C=c, max_iter=1000, class_weight="balanced"),
        )
        scope = "OCR read excluded"
    elif kind == "ultrabert-embed":
        tok, model, device = load_embedder()
        x = np.concatenate([
            embed(texts[i : i + batch_size], tok, model, device)
            for i in range(0, len(texts), batch_size)
        ])
        grid, file = C_GRID, "embed.joblib"
        make = lambda c: LogisticRegression(C=c, max_iter=1000, class_weight="balanced")
        scope = "base model load and pool embedding included; OCR read excluded"
    else:
        processor, model, device = load_dit()
        x = np.concatenate([
            embed_images([page_image(p) for p in ids[i : i + batch_size]], processor, model, device)
            for i in range(0, len(ids), batch_size)
        ])
        grid, file = C_GRID, "dit.joblib"
        make = lambda c: make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=1000, class_weight="balanced")
        )
        scope = "base model load, page image reads and pool embedding included"
    encoded = time.perf_counter() - start

    best, lines = None, []
    for c in grid:
        pred, score, clfs = out_of_fold(lambda: make(c), x, gold, fold, sets)
        auroc = roc_auc_score(pred == gold, score)
        lines.append(f"C {c:<5} out-of-fold AUROC {auroc:.4f}, all {len(rows)} pool pages")
        if best is None or auroc > best[1]:
            best = (c, auroc, pred, score, clfs)
    folded = time.perf_counter() - start - encoded
    if len(grid) > 1:
        lines.append(f"chosen    C {best[0]} (highest out-of-fold AUROC; a tie goes to the smaller C)")
    else:
        lines.append(f"fixed     C {best[0]}")
    write_pool_dump(name, rows, best[2], best[3])
    for f, clf in enumerate(best[4]):
        out = os.path.join("models", name, f"fold{f}")
        os.makedirs(out, exist_ok=True)
        joblib.dump(clf, os.path.join(out, file))

    text = "\n".join([
        f"model     models/{name}",
        f"train     {which}, {k} folds, fold seed {seed}; pool {len(rows)} pages, train and"
        f" validation, {int((gold < 0).sum())} reject",
        *fold_lines(rows, fold, k, sets),
        f"seconds   {encoded:.2f} encoding the pool ({scope})",
        f"seconds   {folded:.2f} fitting and scoring {k} folds for each of {len(grid)} C",
        *lines,
    ]) + "\n"
    sys.stderr.write(text)
    path = os.path.join("results", "train", name + "-pool.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


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
    ap.add_argument("--folds", type=int, metavar="K", help="train on the pool, K folds out of fold")
    ap.add_argument("--fold-seed", type=int, default=0, metavar="S", help="seed of --folds")
    args = ap.parse_args()
    if args.tfidf + args.embed + args.dit != 1:
        ap.error("give one of --tfidf, --embed and --dit")
    if args.folds is not None and (args.folds < 2 or args.labels != "hand"):
        ap.error("--folds needs K of at least 2 and hand labels")

    kind = "tfidf" if args.tfidf else "ultrabert-embed" if args.embed else "dit"
    name = f"{kind}-{args.train}" + ("-llm" if args.labels == "llm" else "")
    if args.folds is not None:
        train_folds(kind, name, args.train, args.folds, args.fold_seed, args.batch_size)
        return
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
