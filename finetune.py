"""finetune.py fine-tunes `LAYOUTLMV3_BASE` over CLASSES on train pages and saves it to
models/<name>/, for score.py. Each page's input is its OCR words and their boxes from the cache
ocr.py writes, the boxes scaled to 0–1000 of the page as LayoutLMv3 expects, and the page image
data/<split>/<id>.png; the first `MAX_TOKENS` tokens are kept. Every weight is trained, encoder
included, unlike train.py's frozen embeddings. `--train real` and `--train small` are train.py's
training sets (`training_pages`).

The ModernBERT analogue of synth-classify's train.py. The settings were fixed before any dev page
was read: `EPOCHS`, `LR`, `BATCH_SIZE`, seed `SEED`, AdamW with weight decay 0.01, a linear schedule
with a 10% warm-up, cross-entropy weighted inversely to train counts as train.py's
`class_weight="balanced"`. The one choice dev makes is the epoch: after each, every dev page is
scored, a reject page as wrong, and the model is saved when its dev accuracy is the highest so far,
a tie keeping the earlier epoch. So dev is not held out for confidence.py, as for train.py's C."""

import argparse
import os
import random
import sys
import time

# MPS lacks some ops; fall back to CPU for them rather than failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from classes import CLASSES
from train import SEED, dev_pages, page_image, training_pages

# CC BY-NC-SA 4.0. Pinned so the weights cannot move under the results.
LAYOUTLMV3_BASE = "microsoft/layoutlmv3-base"
LAYOUTLMV3_REVISION = "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4"
MAX_TOKENS = 512
EPOCHS = 10
LR = 2e-5
BATCH_SIZE = 8


def load_processor(path=LAYOUTLMV3_BASE, revision=LAYOUTLMV3_REVISION):
    """load_processor returns the processor `encode` takes. `apply_ocr=False`: the words and boxes
    come from ocr/, so no second OCR path runs."""
    from transformers import LayoutLMv3Processor

    return LayoutLMv3Processor.from_pretrained(path, revision=revision, apply_ocr=False)


def encode(page_ids, processor, padding):
    """encode returns the processor's tensors for `page_ids`, `padding` as the processor takes it:
    OCR words from ocr/<id>.json, each box divided by the page image's size and scaled to 0–1000,
    and the image itself. A page with no OCR words is the image and the special tokens alone."""
    import json

    images, words, boxes = [], [], []
    for page_id in page_ids:
        image = page_image(page_id)
        w, h = image.size
        with open(os.path.join("ocr", page_id + ".json")) as f:
            ocr = json.load(f)["words"]
        scale = lambda v, n: min(1000, max(0, int(1000 * v / n)))
        images.append(image)
        words.append([word["text"] for word in ocr])
        boxes.append(
            [[scale(b["x0"], w), scale(b["y0"], h), scale(b["x1"], w), scale(b["y1"], h)] for b in ocr]
        )
    return processor(
        images,
        words,
        boxes=boxes,
        truncation=True,
        padding=padding,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )


def load_model(path, revision=None):
    """load_model returns the classifier at `path` on MPS when there is one, and the device. float32:
    MPS has no float64. From `LAYOUTLMV3_BASE` the classification head is new, over CLASSES."""
    import torch
    from transformers import LayoutLMv3ForSequenceClassification

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)
    names = list(CLASSES)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        path,
        revision=revision,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={name: i for i, name in enumerate(names)},
        dtype=torch.float32,
    )
    return model.to(device), device


def finetune(ids, labels, dev_ids, dev_gold, out):
    """finetune trains for `EPOCHS` and saves the epoch with the best dev accuracy to `out`. Returns
    the best epoch, from 1, and each epoch's dev accuracy."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    processor = load_processor()
    model, device = load_model(LAYOUTLMV3_BASE, LAYOUTLMV3_REVISION)
    # Encoded once, every page padded to MAX_TOKENS so a batch is rows of one tensor. Padding is
    # masked out of attention, so it changes the time and not the result.
    pages = encode(ids, processor, "max_length")
    dev = encode(dev_ids, processor, "max_length")

    counts = [labels.count(c) for c in range(len(CLASSES))]
    weight = torch.tensor([len(labels) / (len(CLASSES) * n) for n in counts], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    steps = EPOCHS * -(-len(ids) // BATCH_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    rng = random.Random(SEED)
    order = list(range(len(ids)))
    best, accuracies = 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i : i + BATCH_SIZE]
            batch = {k: v[idx].to(device) for k, v in pages.items()}
            target = torch.tensor([labels[j] for j in idx], device=device)
            loss = loss_fn(model(**batch).logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item() * len(idx)

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(dev_ids), BATCH_SIZE):
                batch = {k: v[i : i + BATCH_SIZE].to(device) for k, v in dev.items()}
                preds += model(**batch).logits.argmax(dim=1).tolist()
        accuracy = sum(p == g for p, g in zip(preds, dev_gold)) / len(dev_gold)
        accuracies.append(accuracy)
        print(
            f"epoch {epoch}/{EPOCHS} loss {total / len(order):.4f} dev accuracy {accuracy:.4f}",
            file=sys.stderr,
        )
        if accuracy > max(accuracies[:-1], default=-1):
            best = epoch
            model.save_pretrained(out)
            processor.save_pretrained(out)
    return best, accuracies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["real", "small"], required=True)
    ap.add_argument("--layoutlmv3", action="store_true", help=f"fine-tune {LAYOUTLMV3_BASE}")
    args = ap.parse_args()
    if not args.layoutlmv3:
        ap.error("give --layoutlmv3")

    name = f"layoutlmv3-{args.train}"
    out = os.path.join("models", name)
    ids, labels, _ = training_pages(args.train, "hand")
    dev_ids, dev_gold = dev_pages()
    os.makedirs(out, exist_ok=True)
    start = time.perf_counter()
    best, accuracies = finetune(ids, labels, dev_ids, dev_gold, out)
    minutes = (time.perf_counter() - start) / 60

    text = (
        f"model     models/{name}\n"
        f"train     {args.train} ({len(ids)} pages)\n"
        f"pages     {len(ids)}\n"
        f"epochs    {EPOCHS}\n"
        f"minutes   {minutes:.1f} (base model load to last save; encoding and dev scoring included)\n"
        f"best      epoch {best} (highest dev accuracy, all {len(dev_gold)} dev pages, reject pages"
        f" wrong; a tie keeps the earlier epoch)\n"
    ) + "".join(f"epoch {i:<3} dev accuracy {a:.4f}\n" for i, a in enumerate(accuracies, 1))
    sys.stderr.write(text)
    path = os.path.join("results", "train", name + ".txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
