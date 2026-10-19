# layout-classify

[synth-classify](../synth-classify/README.md) measured local text classifiers on short messages.
This repo asks the same questions of document pages: native digital documents, or good scans of
them, rendered here at 200 dpi, plus pages that fit none of the types. It holds two classifiers on
the OCR text, one on the page image, one fine-tuned on both, and a local vision LLM given the page
image and no hand labels. It holds the corpus manifest, the page split and the OCR cache, and no
results yet.

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

**Source.** The corpus is the source PDFs of RVL-CDIP-N (Larson et al., *Evaluating
Out-of-Distribution Performance on Document Image Classifiers*, NeurIPS 2022 Datasets and
Benchmarks), from the Google Drive folder that `github.com/gxlarson/rvl-cdip-ood` links: 1002 PDFs
behind RVL-CDIP-N's 1002 images, in 12 types, and the 254 PDFs that RVL-CDIP-O's images name,
documents of none of the types. Most are DocumentCloud documents. The dataset is CC BY 3.0, per the
`license` field of `jordyvl/RVL-CDIP-N`'s dataset card; the Drive folder's `LICENSE.txt` was not
read. RVL-CDIP and RVL-CDIP-N page images are not used, nor any model trained on them: every page
is rendered here from its PDF at 200 dpi. `docs/2026-10-19-corpus-plan.md` has the Drive listing
and the match of images to PDFs. `fetch.py` downloads the PDFs to `data/pdf/` and writes
`data/pdf/manifest.jsonl`.

**Duplicates.** `fetch.py` hashes every PDF. 27 of the 1256 are byte copies of another and become
aliases of the first in manifest order, which drops them before the gate: 15 N PDFs copy an N PDF
of the same type (`pdfs_2-documentcloud_00259_news_article` has two copies), and 12 O PDFs copy an
N one. Six file names occur in two Drive folders with different contents, so a local file name is
`<folder>-<stem>.pdf`.

**In-class pages in O.** The 12 O copies, `pdfs_2-documentcloud_00589_ood` to `_00602_ood` less
`_00590` and `_00592`, are all the N email `pdfs_2-documentcloud_00590_email`, whose N copy is
`_00592_email`. So RVL-CDIP-O, the set of documents of none of the types, holds an email. As
aliases of an N PDF they are never drawn as reject pages.

**Labelled page.** RVL-CDIP-N does not record which page of a PDF its image shows; 612 of the 1002
N PDFs have more than one page. `fetch.py` renders every page at the image's pixel size and takes
the page with the smallest perceptual-hash (phash) distance to the image. For all 1002 it is page
1: 636 at distance 0 and 978 within 10 of 64 bits. The other 24, at 12 to 40, are RVL-CDIP-N images
distorted from their PDF page; page 1 is taken for them too, and 8 of them are in the split.

**Quality gate.** `quality.py` checks the labelled page of each of the 1229 PDFs that are not
aliases against "native digital documents or good scans of them". A page whose largest image covers
at least 80% of it is a scan and passes only at an image resolution of at least 150 dpi, whatever
its text layer. Any other page passes with at least 50 non-space characters of text layer, or else
with its largest image at 150 dpi or more. 1045 pass and 184 fail: 129 scans under 150 dpi, 53
pages with neither a text layer nor an image at 150 dpi, and 2 that pdfplumber cannot read
(`data/pdf/quality.jsonl`).

| Type                   | Kept | Dropped |
|------------------------|------|---------|
| budget                 | 50   | 8       |
| email                  | 30   | 2       |
| form                   | 64   | 4       |
| handwritten            | 131  | 40      |
| invoice                | 45   | 12      |
| letter                 | 123  | 26      |
| memo                   | 42   | 5       |
| news_article           | 62   | 22      |
| questionnaire          | 39   | 0       |
| resume                 | 131  | 51      |
| scientific_publication | 39   | 0       |
| specification          | 60   | 1       |
| O (reject source)      | 229  | 13      |
| **Total**              | 1045 | 184     |

**Splits.** `split.py` splits the 816 kept N PDFs by class with `random.Random(0)`: test 40%,
validation 30%, train 30%, each class rounded by largest remainder, a tie going to test, then
validation, then train. A PDF gives one page, so no document is in two splits, and the run asserts
that no sha256 is. It writes `data/manifest.jsonl` (`id`, `split`, `label`, `pdf`, `page`,
`sha256`, and from `render.py` `width` and `height`) and `data/fewshot.txt`, the first 3 train pages
of each class, 36 in all, a subset of train and not a fourth split. `render.py` writes each page to
`data/<split>/<id>.png`.

| Class                  | Train | Validation | Test | Total |
|------------------------|-------|------------|------|-------|
| budget                 | 15    | 15         | 20   | 50    |
| email                  | 9     | 9          | 12   | 30    |
| form                   | 19    | 19         | 26   | 64    |
| handwritten            | 39    | 39         | 53   | 131   |
| invoice                | 13    | 14         | 18   | 45    |
| letter                 | 37    | 37         | 49   | 123   |
| memo                   | 12    | 13         | 17   | 42    |
| news_article           | 18    | 19         | 25   | 62    |
| questionnaire          | 12    | 12         | 15   | 39    |
| resume                 | 39    | 39         | 53   | 131   |
| scientific_publication | 12    | 12         | 15   | 39    |
| specification          | 18    | 18         | 24   | 60    |
| **In-class**           | 243   | 246        | 327  | 816   |
| reject (RVL-CDIP-O)    | 27    | 27         | 36   | 90    |
| **Split total**        | 270   | 273        | 363  | 906   |

**Reject pages.** Each split gets round(in-class / 9) reject pages, a tenth of the split: page 1 of
kept O PDFs, drawn with the same `random.Random(0)` after the classes, labelled `none`. Train's 27
are in the manifest but no classifier trains on them.

**Limits.**

- Phone photos of documents pass the gate: a photo at 150 dpi or more is a scan by its rule.
- email, questionnaire and scientific_publication have 12 to 16 test pages each (12, 15 and 15), so
  their per-class figures are coarse.
- DocumentCloud documents keep their own rights; CC BY 3.0 covers the dataset, not them.
- DiT and LayoutLMv3 are pretrained on IIT-CDIP, which does not contain these documents.

## Files

- `classes.py` — `CLASSES`, the 12 RVL-CDIP-N types with a one-sentence definition each, saying
  what separates the type from its near pair; `REJECT_LABEL`, `"none"`. `test_classes.py` checks
  that `CLASSES` are the labels of the N rows of `data/pdf/manifest.jsonl`.
- `fetch.py` — downloads the source PDFs to `data/pdf/` and writes `data/pdf/manifest.jsonl`, one
  row per PDF with its sha256, aliases and labelled page; resumable.
- `quality.py` — the quality gate above, on each PDF's labelled page; writes
  `data/pdf/quality.jsonl`.
- `split.py` — the split above; writes `data/manifest.jsonl` and `data/fewshot.txt`.
- `render.py` — renders each manifest page at 200 dpi to `data/<split>/<id>.png` and records its
  pixel size in the manifest.
- `ocr.py` — runs Tesseract once on every page in the manifest. It writes `ocr/<id>.json`:
  `{"text", "words": [{"text", "x0", "y0", "x1", "y1", "conf"}]}`, with boxes in pixels of the
  page image. It skips pages already cached. Each run that OCRs a page appends a block to
  `results/ocr-time.txt`, headed by the date and the splits that had pages run.
- `train.py` — fits a classifier on the OCR text of train pages and saves it to `models/<name>/`.
  `--tfidf` is TF-IDF + logistic regression; `--embed` is logistic regression on frozen
  `dmis-lab/UltraBERT-Embed-150M` embeddings, at synth-classify's pinned revision, with a task
  instruction before the page text and the first 512 tokens kept. `--train real` is every train
  page; `--train small` is 10 per class, drawn with seed 0, the ids in data/selection.json under
  `small`; email has 9 train pages and gives all 9, so small is 119 pages. Neither holds a reject
  page: train's 27 are left out. Every classifier uses `class_weight="balanced"`. For
  `--embed` and `--dit`, C is the value in `C_GRID` (0.01, 0.1, 1, 10), each fitted on train, with
  the highest validation AUROC of top probability against correctness over all validation pages, a
  tie going to the smaller C; `results/train/<name>.txt` gives each value's AUROC and the choice.
  `--tfidf` keeps its fixed C of 10 on its sparse features, with no grid. `--dit` reads the page
  image instead: logistic regression on frozen `microsoft/dit-base` embeddings, at a pinned
  revision, each the mean of the last hidden state over the 196 patches of the page resized to 224 ×
  224, standardised by a `StandardScaler` fitted on the train embeddings. Names are `tfidf-real`,
  `tfidf-small`, `ultrabert-embed-real`, `ultrabert-embed-small`, `dit-real` and `dit-small`.
  `--labels llm` fits on the labels in `data/llm-labels/train.jsonl` instead of the hand labels and
  adds `-llm` to the name, as in `tfidf-real-llm` and `ultrabert-embed-real-llm`; validation keeps
  its hand labels, so C is still chosen on them. Its `results/train/<name>.txt` adds the share of
  train pages whose llm label equals the hand label, and the confusion matrix, rows hand and columns
  llm.
- `finetune.py` — fine-tunes every weight of `microsoft/layoutlmv3-base`, at a pinned revision, with
  a new head over `CLASSES`, and saves it to `models/<name>/`. A page's input is its OCR words and
  boxes from `ocr/<id>.json`, the boxes scaled to 0–1000 of the page, and `data/<split>/<id>.png`;
  the first 512 tokens are kept. `--layoutlmv3 --train real` or `--train small` takes `train.py`'s
  training sets. The settings were fixed before any validation page was read: 10 epochs, learning
  rate 2e-5, batch 8, seed 0, AdamW with weight decay 0.01, a linear schedule with 10% warm-up, and
  cross-entropy weighted as `class_weight="balanced"`. After each epoch every validation page is
  scored, a reject page as wrong, and the epoch with the highest validation accuracy is kept, a tie
  keeping the earlier; `results/train/<name>.txt` gives each epoch's accuracy and the minutes. Names
  are `layoutlmv3-real` and `layoutlmv3-small`.
- `score.py` — scores a model on every page of validation or test, reject pages included. A model
  predicts one of `CLASSES`, so a reject page is always wrong. A `dit-` model reads
  `data/<split>/<id>.png`; its seconds leave out the image read. A `layoutlmv3-` model reads the OCR
  words and boxes and the image; its seconds are the forward pass, the page read and the processor
  excluded. `--dump` writes each page's prediction and top probability for `confidence.py`.
- `llm.py` — `classify` sends one page image, `data/<split>/<id>.png`, to `gemma4:e4b` on Ollama
  with `PROMPT`: the 12 `CLASSES` names and their definitions. The reply is restricted by Ollama's
  structured output to an enum of those names, at temperature 0 with thinking off. With
  `allow_reject` the prompt is `REJECT_PROMPT`, which adds `none` for a page that fits no type, and
  the enum holds it too. `score.py --llm gemma4-image` or `--llm gemma4-image-reject` scores it; its
  seconds are the Ollama request, the image read excluded. It gives no probability, so there is no
  `--dump` and no cut-off. The report of `gemma4-image-reject` adds the share of reject pages it
  labelled `none` and the share of in-class pages it did. `label_split` runs `classify` without
  reject on every page of a split and writes `data/llm-labels/<split>.jsonl`, one row per page with
  `id`, `label` and `seconds`: `uv run python llm.py --split train`. That file is tracked.
- `d3.py` — `vllm-sr/d3-flash`, a Decision 3.0 model from vLLM Semantic Router, run locally on MPS
  in bfloat16 at a pinned revision and asked one `choice` question per page; it answers with a
  probability for every type. Copied from synth-classify's. The question's options are the 12
  `CLASSES` and its criteria their definitions; with `--prompt fewshot` each definition is followed
  by the OCR text of that type's 3 `data/fewshot.txt` pages, each cut to its first 300 words. The
  state is the page's OCR text cut to its first 1,500 words. Every answer is cached in
  `results/d3/cache.jsonl`, so a rerun of `score.py --d3` rebuilds its report without the model.
  `score.py --d3 vllm-sr/d3-flash` scores it; with `fewshot`, a `data/fewshot.txt` page in the split
  would be left out, and as those are train pages none is. Its score is the chosen type's
  probability, and its seconds are the `system_one` call. It needs transformers 5.17.0, so it runs
  in `.venv-d3`, built from `requirements-d3.txt`, with `HF_HOME` at `data/hf`, a symlink to
  synth-classify's `data/hf`, which holds the model's 16 GB. Names are `d3-flash-definitions` and
  `d3-flash-fewshot`.
- `confidence.py` — adapted from synth-classify's. For α 0.02, 0.05 and 0.10 it picks two cut-offs
  on every validation page: `tau`, the lowest score whose accepted validation error is at most α,
  and `tau_cp`, the same with that error's one-sided 90% Clopper-Pearson upper bound. A reject page
  scoring at or above a cut-off is an accepted error. It then reports on test, at each cut-off,
  coverage and error on in-class pages, the reject leak (the share of reject pages at or above it),
  and coverage and error for a stream with 10% reject pages and for all test pages. `--dev-size N
  --dev-seed S` fits on N validation pages, N / 13 from each class and from the reject pages, and
  writes `results/confidence/<name>-dev<N>-seed<S>.txt`. `test_confidence.py` checks it on a
  hand-made 12-row dump.
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
uv run python fetch.py      # source PDFs to data/pdf/; resumable
uv run python quality.py
uv run python split.py
uv run python render.py
uv run pytest               # classes.py against data/pdf/manifest.jsonl; confidence.py on a hand-made dump
uv run python ocr.py

uv run python train.py --tfidf --train real          # also --train small; --embed for UltraBERT, --dit for DiT
uv run python finetune.py --layoutlmv3 --train real  # also --train small
for split in validation test; do
  uv run python score.py --model tfidf-real --split $split --dump
done
uv run python confidence.py --model tfidf-real
ollama serve &   # gemma4:e4b pulled
uv run python score.py --llm gemma4-image --split test   # also gemma4-image-reject; validation
uv run python llm.py --split train                       # gemma4:e4b labels train
uv run python train.py --tfidf --train real --labels llm # also --embed

# d3-flash: its own venv, and synth-classify's Hugging Face cache, where the model already is
ln -s ../../synth-classify/data/hf data/hf
uv venv --python 3.13 .venv-d3
uv pip install --python .venv-d3/bin/python -r requirements-d3.txt
export HF_HOME=$PWD/data/hf HF_HUB_OFFLINE=1
for split in validation test; do      # also --prompt fewshot
  .venv-d3/bin/python score.py --d3 vllm-sr/d3-flash --prompt definitions --split $split --dump
done
.venv-d3/bin/python confidence.py --model d3-flash-definitions
```

`--embed`, `--dit` and `finetune.py` training and scoring fetch the model from Hugging Face on
first use.

## Licences

layout-classify is MIT; see `LICENSE`. `microsoft/dit-base` is MIT, per
its model card. `microsoft/layoutlmv3-base` is under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0), per its model card, so the
`layoutlmv3-` models are for non-commercial use only. `gemma4:e4b` is under the Apache License 2.0,
per the licence Ollama ships with it (`ollama show gemma4:e4b --license`).
