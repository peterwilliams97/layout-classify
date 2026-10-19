# layout-classify

[synth-classify](../synth-classify/README.md) measured local text classifiers on short messages.
This repo asks the same questions of document pages: native digital documents, or good scans of
them, rendered here at 200 dpi, plus pages that fit none of the types. It holds two classifiers on
the OCR text, one on the page image, one fine-tuned on both, and a local vision LLM given the page
image and no hand labels. It holds no corpus and no results yet.

## Aim

Four questions, measured on one laptop:

1. **How much work can it do without a person checking?** The classifier gives each page a type and
   a score. Above a cut-off, the type is used and nobody reads the page.
   *Coverage* is the share of test pages above the cut-off.
   *Error* is the share of those whose type is wrong.
   We report coverage at 5% error, with 10 hand-labelled pages per class and with every train page.
2. **Does the page image beat OCR text?** Same pages, same tables; one classifier reads the text,
   another reads the image.
3. **What do pages that fit no type cost?** Production must send these to a person.
   Coverage and error are reported with and without them.
4. **How many seconds per page?**

## Data

Not chosen yet. The corpus must be native digital documents or good scans of them, rendered at
200 dpi, split train / validation / test with no page in two splits. RVL-CDIP and RVL-CDIP-N page
images are not used, nor any model trained on them; the data, OCR cache, models and results built
from them have been removed. The scripts read `data/manifest.jsonl`, one row per page with `id`,
`split` and `label`, and `data/<split>/<id>.png`; the builder that writes them is still to come.

## Files

- `classes.py` — `CLASSES`, the 9 types with a one-sentence definition each; `REJECT_CLASSES`, the
  other 7 RVL-CDIP types; `REJECT_LABEL`, `"none of these"`. `test_classes.py` checks that `CLASSES`
  and `REJECT_CLASSES` together are the 16 label names of `aharley/rvl_cdip`.
- `ocr.py` — runs Tesseract once on every page in the manifest. It writes `ocr/<id>.json`:
  `{"text", "words": [{"text", "x0", "y0", "x1", "y1", "conf"}]}`, with boxes in pixels of the
  page image. It skips pages already cached. Each run that OCRs a page appends a block to
  `results/ocr-time.txt`, headed by the date and the splits that had pages run.
- `train.py` — fits a classifier on the OCR text of train pages and saves it to `models/<name>/`.
  `--tfidf` is TF-IDF + logistic regression; `--embed` is logistic regression on frozen
  `dmis-lab/UltraBERT-Embed-150M` embeddings, at synth-classify's pinned revision, with a task
  instruction before the page text and the first 512 tokens kept. `--train real` is every train
  page; `--train small` is 10 per class, drawn with seed 0, the ids in
  data/selection.json under `small`. Neither holds a reject page. Every classifier uses
  `class_weight="balanced"`. For `--embed` and `--dit`, C is the value in `C_GRID` (0.01, 0.1, 1,
  10), each fitted on train, with the highest dev AUROC of top probability against correctness
  over all dev pages, a tie going to the smaller C; `results/train/<name>.txt` gives each value's
  AUROC and the choice. `--tfidf` keeps its fixed C of 10 on its sparse features, with no grid.
  `--dit` reads the page image instead: logistic regression on frozen
  `microsoft/dit-base` embeddings, at a pinned revision, each the mean of the last hidden state over
  the 196 patches of the page resized to 224 × 224, standardised by a `StandardScaler` fitted on
  the train embeddings. Names are `tfidf-real`, `tfidf-small`,
  `ultrabert-embed-real`, `ultrabert-embed-small`, `dit-real` and `dit-small`. `--labels llm` fits
  on the labels in `data/llm-labels/train.jsonl` instead of the hand labels and adds `-llm` to the
  name, as in `tfidf-real-llm` and `ultrabert-embed-real-llm`; dev keeps its hand labels, so C is
  still chosen on them. Its `results/train/<name>.txt` adds the share of train pages whose llm
  label equals the hand label, and the confusion matrix, rows hand and columns llm.
- `finetune.py` — fine-tunes every weight of `microsoft/layoutlmv3-base`, at a pinned revision, with
  a new head over `CLASSES`, and saves it to `models/<name>/`. A page's input is its OCR words and
  boxes from `ocr/<id>.json`, the boxes scaled to 0–1000 of the page, and `data/<split>/<id>.png`;
  the first 512 tokens are kept. `--layoutlmv3 --train real` or `--train small` takes `train.py`'s
  training sets. The settings were fixed before any dev page was read: 10 epochs, learning rate
  2e-5, batch 8, seed 0, AdamW with weight decay 0.01, a linear schedule with 10% warm-up, and
  cross-entropy weighted as `class_weight="balanced"`. After each epoch every dev page is scored, a
  reject page as wrong, and the epoch with the highest dev accuracy is kept, a tie keeping the
  earlier; `results/train/<name>.txt` gives each epoch's accuracy and the minutes. Names are
  `layoutlmv3-real` and `layoutlmv3-small`.
- `score.py` — scores a model on every page of dev or test, reject pages included. A model predicts
  one of `CLASSES`, so a reject page is always wrong. A `dit-` model reads `data/<split>/<id>.png`;
  its seconds leave out the image read. A `layoutlmv3-` model reads the OCR words and boxes and the
  image; its seconds are the forward pass, the page read and the processor excluded. `--dump` writes
  each page's prediction and top probability for `confidence.py`.
- `llm.py` — `classify` sends one page image, `data/<split>/<id>.png`, to `gemma4:e4b` on Ollama
  with `PROMPT`: the 9 `CLASSES` names and their definitions. The reply is restricted by Ollama's
  structured output to an enum of those names, at temperature 0 with thinking off. With
  `allow_reject` the prompt is `REJECT_PROMPT`, which adds `none of these` for a page that fits no
  type, and the enum holds it too. `score.py --llm gemma4-image` or `--llm gemma4-image-reject`
  scores it; its seconds are the Ollama request, the image read excluded. It gives no probability,
  so there is no `--dump` and no cut-off. The report of `gemma4-image-reject` adds the share of
  reject pages it labelled `none of these` and the share of in-class pages it did. `label_split`
  runs `classify` without reject on every page of a split and writes `data/llm-labels/<split>.jsonl`,
  one row per page with `id`, `label` and `seconds`: `uv run python llm.py --split train`. That
  file is tracked.
- `confidence.py` — adapted from synth-classify's. It picks the cut-offs on every dev page, then
  reports on test, at each cut-off, coverage and error on in-class pages, the reject leak, and
  coverage and error for a stream with 10% reject pages and for all test pages.
- `data/` — the manifest and the selection are tracked. The page images are ignored by git.
- `ocr/` — the OCR cache, one JSON file per page.
- `models/` — trained classifiers, ignored by git and rebuilt by `train.py` and `finetune.py`.
- `results/` — every reported number, in synth-classify's layout without its class-set level:
  `results/<split>/<name>.txt` for score reports, `results/train/<name>.txt` for training time,
  `results/confidence/<name>.txt` for routing, with its inputs in `results/confidence/dumps/`, and
  `results/ocr-time.txt` for OCR time.

## Run

Tesseract must be installed (`brew install tesseract`).

```sh
uv sync
uv run pytest           # checks classes.py against aharley/rvl_cdip's label names
uv run python ocr.py

uv run python train.py --tfidf --train real          # also --train small; --embed for UltraBERT, --dit for DiT
uv run python finetune.py --layoutlmv3 --train real  # also --train small
for split in dev test; do
  uv run python score.py --model tfidf-real --split $split --dump
done
uv run python confidence.py --model tfidf-real
ollama serve &   # gemma4:e4b pulled
uv run python score.py --llm gemma4-image --split test   # also gemma4-image-reject; dev
uv run python llm.py --split train                       # gemma4:e4b labels train
uv run python train.py --tfidf --train real --labels llm # also --embed
```

`--embed`, `--dit` and `finetune.py` training and scoring fetch the model from Hugging Face on
first use.

## Licences

layout-classify is MIT; see `LICENSE`. `microsoft/dit-base` is MIT, per
its model card. `microsoft/layoutlmv3-base` is under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0), per its model card, so the
`layoutlmv3-` models are for non-commercial use only. `gemma4:e4b` is under the Apache License 2.0,
per the licence Ollama ships with it (`ollama show gemma4:e4b --license`).
